/***************************************************************************
 *
 * Author: "Sjors H.W. Scheres", "Takanori Nakane", "Jiang Lab"
 * MRC Laboratory of Molecular Biology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/align_map_to_map.h"
#include "src/fftw.h"
#include "src/transformations.h"

namespace
{

/* This search is full of exact ties, and which one of a tie is kept matters.
 *
 * With tilt == 0 the ZYZ angles rot and psi are both rotations about Z, so
 * (rot, 0, psi) and (rot + d, 0, psi - d) are the *same* transformation and
 * score exactly the same diff2.  A plain "diff2 < best_diff2" keeps whichever
 * one the loops happened to reach first, which is the most negative rot in the
 * search range.  For two identical maps that means returning (-3, 0, +3)
 * instead of (0, 0, 0) - and because each refinement level re-centres its own
 * search on that answer and again takes the first of the tie, every level walks
 * the result one more step away from zero.
 *
 * So among rotations that score equally well, keep the smallest one.  mag2 is
 * the squared magnitude of the angles being searched.
 *
 * Only the rotation searches need this: the degeneracy is specific to the ZYZ
 * angles, and dx/dy/dz are independent with no such equivalence.
 */
bool isBetterCandidate(double diff2, double best_diff2,
                       double mag2, double best_mag2)
{
	const double tol = 1e-9 * (fabs(diff2) + fabs(best_diff2));
	if (diff2 < best_diff2 - tol) return true;
	if (diff2 > best_diff2 + tol) return false;
	return mag2 < best_mag2;
}

} // anonymous namespace

/* Conventions in this file, which several bugs used to violate:
 *
 *  - The returned (best_rot, best_tilt, best_psi, best_dx, best_dy, best_dz) is
 *    the transformation that IS APPLIED to vol_align to bring it onto vol_ref,
 *    in the order "rotate, then translate".  vol_align is left transformed.
 *  - Rotations follow Projector::rotate3D(A), which rotates the object BY A.
 *    applyGeometry(..., A, inv = true) rotates by A-inverse, so the final
 *    application below must pass inv = false to rotate the same way.  Negating
 *    the Euler angles is not a substitute: the inverse of ZYZ (rot, tilt, psi)
 *    is (-psi, -tilt, -rot), which only reduces to a plain negation in the
 *    single-angle Cn case.
 *  - Translations are in Angstrom and are applied with selfTranslate() in both
 *    the search and the final application, so the stored value is the shift that
 *    matched, not its negation.
 *  - vol_work is handed to the Projector in the ordinary Xmipp-centred layout.
 *    It must NOT be CenterFFT-ed first: that puts the object at the array
 *    corners, and rotating it in Fourier space then rotates the accompanying
 *    phase ramp as well, which silently destroyed rotation recovery.
 */
void alignMapToMap(
    MultidimArray<RFLOAT> &vol_align,
    const MultidimArray<RFLOAT> &vol_ref,
    int nr_freedom,
    RFLOAT angpix,
    RFLOAT maxres,
    int search_range,
    RFLOAT search_step_rot,
    RFLOAT search_step_trans,
    RFLOAT &best_rot,
    RFLOAT &best_tilt,
    RFLOAT &best_psi,
    RFLOAT &best_dx,
    RFLOAT &best_dy,
    RFLOAT &best_dz)
{
    best_rot = best_tilt = best_psi = best_dx = best_dy = best_dz = 0.;
    if (nr_freedom == 0) return;

    int orig_size = XSIZE(vol_ref);
    if (orig_size != XSIZE(vol_align))
        REPORT_ERROR("alignMapToMap: vol_align and vol_ref must have the same size!");

    int work_size = 64;
    if (orig_size < work_size) work_size = orig_size;
    if (work_size % 2 != 0) work_size++;
    RFLOAT work_angpix = angpix * orig_size / work_size;

    MultidimArray<RFLOAT> vol_work, vol_ref_down;
    vol_work = vol_align;
    vol_ref_down = vol_ref;

    resizeMap(vol_work, work_size);
    resizeMap(vol_ref_down, work_size);

    int r_max = (maxres > 0.) ? CEIL(work_size * work_angpix / maxres) : work_size;
    if (r_max > work_size) r_max = work_size;

    MultidimArray<RFLOAT> dummy;
    // padding_factor 2 keeps the trilinear interpolation accurate enough that a
    // real rotation beats the identity.  At padding_factor 1 the identity is the
    // only trial that lands exactly on grid points, so every genuine rotation
    // paid an interpolation penalty that swamped the signal and the search
    // always returned zero.  Gridding correction stays off: it is applied in
    // real space before padding and would have to be undone to compare against
    // the uncorrected reference.
    Projector projector(work_size, TRILINEAR, 2, 10, 3);
    projector.computeFourierTransformMap(vol_work, dummy, 2 * r_max, 1, false);

    MultidimArray<RFLOAT> rotated;
    MultidimArray<Complex> rot_ft;
    FourierTransformer transformer;
    rotated.reshape(vol_work);
    transformer.setReal(rotated);
    transformer.getFourierAlias(rot_ft);

    // Rotate the working map by (rot, tilt, psi) into `rotated`
    auto rotateInto = [&](RFLOAT rot, RFLOAT tilt, RFLOAT psi)
    {
        Matrix2D<RFLOAT> A;
        Euler_rotation3DMatrix(rot, tilt, psi, A);
        rot_ft.initZeros();
        projector.rotate3D(rot_ft, A);
        CenterFFTbySign(rot_ft);
        transformer.inverseFourierTransform();
    };

    // Squared difference to the reference of `rotated` shifted by (dx, dy, dz) A
    auto diff2At = [&](RFLOAT dx, RFLOAT dy, RFLOAT dz) -> double
    {
        MultidimArray<RFLOAT> shifted;
        const bool do_shift = (fabs(dx) > 0. || fabs(dy) > 0. || fabs(dz) > 0.);
        if (do_shift)
        {
            shifted = rotated;
            Matrix1D<RFLOAT> shift(3);
            XX(shift) = dx / work_angpix;
            YY(shift) = dy / work_angpix;
            ZZ(shift) = dz / work_angpix;
            selfTranslate(shifted, shift, WRAP);
        }
        const MultidimArray<RFLOAT> &cmp = do_shift ? shifted : rotated;

        double d2 = 0.;
        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(cmp)
        {
            double d = DIRECT_MULTIDIM_ELEM(cmp, n)
                     - DIRECT_MULTIDIM_ELEM(vol_ref_down, n);
            d2 += d * d;
        }
        return d2;
    };

    double best_diff2 = 1E99;

    // ---------------- coarse search ----------------
    if (nr_freedom == 2)
    {
        // Cn / helical: rotation about Z and a shift along Z, searched jointly
        double best_mag2 = 1E99;
        for (int irot = -search_range; irot <= search_range; irot++)
        {
            RFLOAT rot = irot * search_step_rot;
            rotateInto(rot, 0., 0.);

            for (int idz = -search_range; idz <= search_range; idz++)
            {
                RFLOAT dz = idz * search_step_trans;
                double diff2 = diff2At(0., 0., dz);

                if (isBetterCandidate(diff2, best_diff2, rot*rot, best_mag2))
                {
                    best_diff2 = diff2;
                    best_mag2 = rot*rot;
                    best_rot = rot;
                    best_dz = dz;
                }
            }
        }
    }
    else if (nr_freedom == 6)
    {
        // Rotation first, with no shift applied ...
        double best_ang_mag2 = 1E99;
        for (int irot = -search_range; irot <= search_range; irot++)
        {
            RFLOAT rot = irot * search_step_rot;
            for (int itilt = -search_range; itilt <= search_range; itilt++)
            {
                RFLOAT tilt = itilt * search_step_rot;
                for (int ipsi = -search_range; ipsi <= search_range; ipsi++)
                {
                    RFLOAT psi = ipsi * search_step_rot;
                    rotateInto(rot, tilt, psi);
                    double diff2 = diff2At(0., 0., 0.);

                    if (isBetterCandidate(diff2, best_diff2,
                                          rot*rot + tilt*tilt + psi*psi, best_ang_mag2))
                    {
                        best_diff2 = diff2;
                        best_ang_mag2 = rot*rot + tilt*tilt + psi*psi;
                        best_rot = rot;
                        best_tilt = tilt;
                        best_psi = psi;
                    }
                }
            }
        }

        // ... then the shift, with that rotation applied
        rotateInto(best_rot, best_tilt, best_psi);
        for (int idx = -search_range; idx <= search_range; idx++)
        {
            RFLOAT dx = idx * search_step_trans;
            for (int idy = -search_range; idy <= search_range; idy++)
            {
                RFLOAT dy = idy * search_step_trans;
                for (int idz = -search_range; idz <= search_range; idz++)
                {
                    RFLOAT dz = idz * search_step_trans;
                    double diff2 = diff2At(dx, dy, dz);

                    if (diff2 < best_diff2)
                    {
                        best_diff2 = diff2;
                        best_dx = dx;
                        best_dy = dy;
                        best_dz = dz;
                    }
                }
            }
        }
    }

    // ---------------- refinement ----------------
    // Alternate rotation and translation, halving the step each level.  Every
    // candidate is scored with the *other* parameter set at its current best, so
    // the two searches see a consistent diff2 and can hand improvements back and
    // forth; the incumbent starts as the thing to beat, so a level can never
    // make the result worse.
    {
        RFLOAT ang_step = search_step_rot;
        RFLOAT trans_step = search_step_trans;
        const RFLOAT min_ang = 0.05;
        const RFLOAT min_trans = 0.05;
        const int max_levels = 8;
        const int fine_range = 1;

        for (int level = 0; level < max_levels; level++)
        {
            // --- rotation, with the current shift applied ---
            {
                RFLOAT lv_rot = best_rot, lv_tilt = best_tilt, lv_psi = best_psi;
                double lv_best = best_diff2;
                double lv_mag2 = best_rot*best_rot + best_tilt*best_tilt + best_psi*best_psi;

                const int tilt_range = (nr_freedom == 2) ? 0 : fine_range;
                for (int irot = -fine_range; irot <= fine_range; irot++)
                for (int itilt = -tilt_range; itilt <= tilt_range; itilt++)
                for (int ipsi = -tilt_range; ipsi <= tilt_range; ipsi++)
                {
                    RFLOAT rot  = best_rot  + irot  * ang_step;
                    RFLOAT tilt = best_tilt + itilt * ang_step;
                    RFLOAT psi  = best_psi  + ipsi  * ang_step;

                    rotateInto(rot, tilt, psi);
                    double diff2 = diff2At(best_dx, best_dy, best_dz);

                    if (isBetterCandidate(diff2, lv_best,
                                          rot*rot + tilt*tilt + psi*psi, lv_mag2))
                    {
                        lv_best = diff2;
                        lv_mag2 = rot*rot + tilt*tilt + psi*psi;
                        lv_rot = rot; lv_tilt = tilt; lv_psi = psi;
                    }
                }

                best_rot = lv_rot; best_tilt = lv_tilt; best_psi = lv_psi;
                if (lv_best < best_diff2) best_diff2 = lv_best;
            }

            // --- translation, with the current rotation applied ---
            {
                rotateInto(best_rot, best_tilt, best_psi);

                RFLOAT lv_dx = best_dx, lv_dy = best_dy, lv_dz = best_dz;
                double lv_best = best_diff2;

                const int xy_range = (nr_freedom == 2) ? 0 : fine_range;
                for (int idx = -xy_range; idx <= xy_range; idx++)
                for (int idy = -xy_range; idy <= xy_range; idy++)
                for (int idz = -fine_range; idz <= fine_range; idz++)
                {
                    RFLOAT dx = best_dx + idx * trans_step;
                    RFLOAT dy = best_dy + idy * trans_step;
                    RFLOAT dz = best_dz + idz * trans_step;

                    double diff2 = diff2At(dx, dy, dz);
                    if (diff2 < lv_best)
                    {
                        lv_best = diff2;
                        lv_dx = dx; lv_dy = dy; lv_dz = dz;
                    }
                }

                best_dx = lv_dx; best_dy = lv_dy; best_dz = lv_dz;
                if (lv_best < best_diff2) best_diff2 = lv_best;
            }

            if (ang_step < min_ang && trans_step < min_trans) break;
            ang_step *= 0.5;
            trans_step *= 0.5;
        }
    }

    // ---------------- apply to vol_align ----------------
    bool do_rot = (fabs(best_rot) > 1e-6 || fabs(best_tilt) > 1e-6 || fabs(best_psi) > 1e-6);
    bool do_trans = (fabs(best_dx) > 1e-6 || fabs(best_dy) > 1e-6 || fabs(best_dz) > 1e-6);

    if (do_rot)
    {
        Matrix2D<RFLOAT> R;
        Euler_rotation3DMatrix(best_rot, best_tilt, best_psi, R);
        MultidimArray<RFLOAT> vol_tmp = vol_align;
        // inv = false, so that this rotates the same way Projector::rotate3D did
        applyGeometry(vol_tmp, vol_align, R, false, false, 0.);
    }

    if (do_trans)
    {
        Matrix1D<RFLOAT> shift(3);
        XX(shift) = best_dx / angpix;
        YY(shift) = best_dy / angpix;
        ZZ(shift) = best_dz / angpix;
        selfTranslate(vol_align, shift, DONT_WRAP);
    }
}

void applyInverseOrientationAdjustment(
    int nr_freedom,
    RFLOAT drot, RFLOAT dtilt, RFLOAT dpsi,
    RFLOAT ddx, RFLOAT ddy, RFLOAT ddz,
    RFLOAT &p_rot, RFLOAT &p_tilt, RFLOAT &p_psi,
    RFLOAT &p_dx, RFLOAT &p_dy, RFLOAT &p_dz)
{
    if (nr_freedom == 2)
    {
        p_rot -= drot;
        p_dz -= ddz;
    }
    else if (nr_freedom == 6)
    {
        Matrix2D<RFLOAT> A_rot(3,3), L(3,3), R(3,3);
        Euler_angles2matrix(drot, dtilt, dpsi, A_rot);
        L.initIdentity();
        R = A_rot.transpose();
        Euler_apply_transf(L, R, p_rot, p_tilt, p_psi, p_rot, p_tilt, p_psi);

        p_dx -= ddx;
        p_dy -= ddy;
        p_dz -= ddz;
    }
}