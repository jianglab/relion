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
 * This is deliberately applied to the rotation searches only.  The degeneracy
 * being worked around is specific to the ZYZ angles; dx, dy and dz are
 * independent with no such equivalence, so tie-breaking them would change which
 * shift is reported without fixing anything.
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

/* TODO (unresolved, found 2026-09-19): the sign conventions in this file are
 * inconsistent between the coarse and the refinement stage, and the Cn rotation
 * search does not work.  None of this is currently covered by a test.  Details,
 * so that whoever next touches this code does not have to re-derive them:
 *
 * 1. Cn / helical (nr_freedom == 2) never recovers a rotation.  Feeding it a map
 *    rotated about Z by 1, 2 or 3 degrees returns best_rot = 0 every time,
 *    whereas the corresponding Z *shift* is recovered correctly (dz of 1, 2, 3
 *    all come back exactly).  So the dz half of that search works and the rot
 *    half does not.  This predates the tie-breaking added above - it reproduces
 *    identically on the parent commit - so it is a separate, older defect.
 *
 * 2. The two stages disagree about the sign of the Cn result.  The coarse loop
 *    stores the trial value as found (best_rot = rot, best_dz = dz); the
 *    refinement stores its negation (lv_rot = -rot, lv_dz = -dz).  The
 *    refinement always overwrites the coarse answer - lv_* is assigned
 *    unconditionally at the end of every level, and lv_best starts at 1E99 so
 *    some candidate always wins - so the negated convention is what callers
 *    actually see.  The C1 branch negates translations in both stages and does
 *    not negate rotations in either, so it is self-consistent; only Cn is not.
 *
 * 3. Because the coarse stage stores translations already negated, the
 *    refinement re-centres its local search on the wrong side of zero: it scans
 *    cur_dx + {-step, 0, +step} when the trial that actually matched was -cur_dx.
 *    The search therefore explores a neighbourhood that cannot contain the
 *    optimum.  It is masked today because those candidates all score worse than
 *    the incumbent, so nothing updates and the coarse answer survives by
 *    default - but it means the translation refinement contributes nothing, and
 *    anything that makes a tied candidate win there (as a magnitude tie-break
 *    on translations did, briefly) turns the masking off and produces a wrong
 *    shift whenever |cur_dx| happens to equal the step size.
 *
 * Fixing 2 and 3 means settling the intended sign convention first.  Note that
 * tests/unit/test_align_map_to_map.cpp documents the returned translation sign
 * as "empirically" what the code does rather than what it should be, so the
 * tests encode the current behaviour and cannot be used to confirm the
 * convention is right.  The caller
 * (MlOptimiser, aligning each 2D class to the largest class) feeds these values
 * both to applyGeometry on the class map and to
 * applyInverseOrientationAdjustment on every particle in the class, so a sign
 * change there is a real change in output and needs validating end to end.
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

    CenterFFT(vol_work, true);
    CenterFFT(vol_ref_down, true);

    int r_max = (maxres > 0.) ? CEIL(work_size * work_angpix / maxres) : work_size;
    if (r_max > work_size) r_max = work_size;

    MultidimArray<RFLOAT> dummy;
    Projector projector(work_size, TRILINEAR, 1, 10, 3);
    projector.computeFourierTransformMap(vol_work, dummy, 2 * r_max, 1, false);

    MultidimArray<RFLOAT> rotated;
    MultidimArray<Complex> rot_ft;
    FourierTransformer transformer;
    rotated.reshape(vol_work);
    transformer.setReal(rotated);
    transformer.getFourierAlias(rot_ft);

    double best_diff2 = 1E99;

    if (nr_freedom == 2)
    {
        for (int irot = -search_range; irot <= search_range; irot++)
        {
            RFLOAT rot = irot * search_step_rot;
            Matrix2D<RFLOAT> A;
            Euler_rotation3DMatrix(rot, 0., 0., A);

            rot_ft.initZeros();
            projector.rotate3D(rot_ft, A);
            CenterFFTbySign(rot_ft);
            transformer.inverseFourierTransform();

            for (int idz = -search_range; idz <= search_range; idz++)
            {
                RFLOAT dz = idz * search_step_trans;

                MultidimArray<RFLOAT> trial = rotated;
                if (fabs(dz) > 0.)
                {
                    Matrix1D<RFLOAT> shift(3);
                    XX(shift) = 0.; YY(shift) = 0.; ZZ(shift) = dz / work_angpix;
                    selfTranslate(trial, shift, WRAP);
                }

                double diff2 = 0;
                FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(trial)
                {
                    double d = DIRECT_MULTIDIM_ELEM(trial, n)
                             - DIRECT_MULTIDIM_ELEM(vol_ref_down, n);
                    diff2 += d * d;
                }

                if (diff2 < best_diff2)
                {
                    best_diff2 = diff2;
                    best_rot = rot;
                    best_tilt = 0.;
                    best_psi = 0.;
                    best_dx = 0.;
                    best_dy = 0.;
                    best_dz = dz;
                }
            }
        }
    }
    else if (nr_freedom == 6)
    {
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
                    Matrix2D<RFLOAT> A;
                    Euler_rotation3DMatrix(rot, tilt, psi, A);

                    rot_ft.initZeros();
                    projector.rotate3D(rot_ft, A);
                    CenterFFTbySign(rot_ft);
                    transformer.inverseFourierTransform();

                    double diff2 = 0;
                    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(rotated)
                    {
                        double d = DIRECT_MULTIDIM_ELEM(rotated, n)
                                 - DIRECT_MULTIDIM_ELEM(vol_ref_down, n);
                        diff2 += d * d;
                    }

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

        {
            Matrix2D<RFLOAT> A;
            Euler_rotation3DMatrix(best_rot, best_tilt, best_psi, A);
            rot_ft.initZeros();
            projector.rotate3D(rot_ft, A);
            CenterFFTbySign(rot_ft);
            transformer.inverseFourierTransform();
        }

        for (int idx = -search_range; idx <= search_range; idx++)
        {
            RFLOAT dx = idx * search_step_trans;
            for (int idy = -search_range; idy <= search_range; idy++)
            {
                RFLOAT dy = idy * search_step_trans;
                for (int idz = -search_range; idz <= search_range; idz++)
                {
                    RFLOAT dz = idz * search_step_trans;

                    MultidimArray<RFLOAT> trial = rotated;
                    if (fabs(dx) > 0. || fabs(dy) > 0. || fabs(dz) > 0.)
                    {
                        Matrix1D<RFLOAT> shift(3);
                        XX(shift) = dx / work_angpix;
                        YY(shift) = dy / work_angpix;
                        ZZ(shift) = dz / work_angpix;
                        selfTranslate(trial, shift, WRAP);
                    }

                    double diff2 = 0;
                    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(trial)
                    {
                        double d = DIRECT_MULTIDIM_ELEM(trial, n)
                                 - DIRECT_MULTIDIM_ELEM(vol_ref_down, n);
                        diff2 += d * d;
                    }

                    if (diff2 < best_diff2)
                    {
                        best_diff2 = diff2;
                        best_dx = -dx;
                        best_dy = -dy;
                        best_dz = -dz;
                    }
                }
            }
        }
    }

    // --- Multi-resolution refinement ---
    {
        RFLOAT cur_rot = best_rot, cur_tilt = best_tilt, cur_psi = best_psi;
        RFLOAT cur_dx = best_dx, cur_dy = best_dy, cur_dz = best_dz;
        RFLOAT ang_step = search_step_rot;
        RFLOAT trans_step = search_step_trans;
        const RFLOAT min_ang = 0.05;
        const RFLOAT min_trans = 0.05;
        const int max_levels = 8;
        const int fine_range = 1;

        for (int level = 0; level < max_levels; level++)
        {
            RFLOAT lv_rot = cur_rot, lv_tilt = cur_tilt, lv_psi = cur_psi;
            RFLOAT lv_dx = cur_dx, lv_dy = cur_dy, lv_dz = cur_dz;
            double lv_best = 1E99;

            if (nr_freedom == 2)
            {
                for (int irot = -fine_range; irot <= fine_range; irot++)
                {
                    RFLOAT rot = cur_rot + irot * ang_step;
                    Matrix2D<RFLOAT> A;
                    Euler_rotation3DMatrix(rot, 0., 0., A);
                    rot_ft.initZeros();
                    projector.rotate3D(rot_ft, A);
                    CenterFFTbySign(rot_ft);
                    transformer.inverseFourierTransform();

                    for (int idz = -fine_range; idz <= fine_range; idz++)
                    {
                        RFLOAT dz = cur_dz + idz * trans_step;
                        MultidimArray<RFLOAT> trial = rotated;
                        if (fabs(dz) > 0.)
                        {
                            Matrix1D<RFLOAT> shift(3);
                            XX(shift) = 0.; YY(shift) = 0.; ZZ(shift) = dz / work_angpix;
                            selfTranslate(trial, shift, WRAP);
                        }
                        double diff2 = 0;
                        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(trial)
                        {
                            double d = DIRECT_MULTIDIM_ELEM(trial, n) - DIRECT_MULTIDIM_ELEM(vol_ref_down, n);
                            diff2 += d * d;
                        }
                        if (diff2 < lv_best)
                        {
                            lv_best = diff2;
                            lv_rot = -rot; lv_dz = -dz;
                        }
                    }
                }
            }
            else
            {
                double lv_ang_mag2 = 1E99;
                for (int irot = -fine_range; irot <= fine_range; irot++)
                {
                    RFLOAT rot = cur_rot + irot * ang_step;
                    for (int itilt = -fine_range; itilt <= fine_range; itilt++)
                    {
                        RFLOAT tilt = cur_tilt + itilt * ang_step;
                        for (int ipsi = -fine_range; ipsi <= fine_range; ipsi++)
                        {
                            RFLOAT psi = cur_psi + ipsi * ang_step;
                            Matrix2D<RFLOAT> A;
                            Euler_rotation3DMatrix(rot, tilt, psi, A);
                            rot_ft.initZeros();
                            projector.rotate3D(rot_ft, A);
                            CenterFFTbySign(rot_ft);
                            transformer.inverseFourierTransform();
                            double diff2 = 0;
                            FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(rotated)
                            {
                                double d = DIRECT_MULTIDIM_ELEM(rotated, n) - DIRECT_MULTIDIM_ELEM(vol_ref_down, n);
                                diff2 += d * d;
                            }
                            if (isBetterCandidate(diff2, lv_best,
                                                 rot*rot + tilt*tilt + psi*psi, lv_ang_mag2))
                            {
                                lv_best = diff2;
                                lv_ang_mag2 = rot*rot + tilt*tilt + psi*psi;
                                lv_rot = rot; lv_tilt = tilt; lv_psi = psi;
                            }
                        }
                    }
                }
{
                    Matrix2D<RFLOAT> A;
                    Euler_rotation3DMatrix(lv_rot, lv_tilt, lv_psi, A);
                    rot_ft.initZeros();
                    projector.rotate3D(rot_ft, A);
                    CenterFFTbySign(rot_ft);
                    transformer.inverseFourierTransform();
                }
                for (int idx = -fine_range; idx <= fine_range; idx++)
                {
                    RFLOAT dx = cur_dx + idx * trans_step;
                    for (int idy = -fine_range; idy <= fine_range; idy++)
                    {
                        RFLOAT dy = cur_dy + idy * trans_step;
                        for (int idz = -fine_range; idz <= fine_range; idz++)
                        {
                            RFLOAT dz = cur_dz + idz * trans_step;
                            MultidimArray<RFLOAT> trial = rotated;
                            if (fabs(dx) > 0. || fabs(dy) > 0. || fabs(dz) > 0.)
                            {
                                Matrix1D<RFLOAT> shift(3);
                                XX(shift) = dx / work_angpix;
                                YY(shift) = dy / work_angpix;
                                ZZ(shift) = dz / work_angpix;
                                selfTranslate(trial, shift, WRAP);
                            }
                            double diff2 = 0;
                            FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(trial)
                            {
                                double d = DIRECT_MULTIDIM_ELEM(trial, n) - DIRECT_MULTIDIM_ELEM(vol_ref_down, n);
                                diff2 += d * d;
                            }
                            if (diff2 < lv_best)
                            {
                                lv_best = diff2;
                                lv_dx = -dx; lv_dy = -dy; lv_dz = -dz;
                            }
                        }
                    }
                }
            }

            // Convergence: only apply if diff2 actually improved
            bool improved = (lv_best < best_diff2 - 1e-10);

            cur_rot = lv_rot; cur_tilt = lv_tilt; cur_psi = lv_psi;
            cur_dx = lv_dx; cur_dy = lv_dy; cur_dz = lv_dz;

            best_rot = cur_rot; best_tilt = cur_tilt; best_psi = cur_psi;
            best_dx = cur_dx; best_dy = cur_dy; best_dz = cur_dz;

            if (improved)
                best_diff2 = lv_best;

            if (ang_step < min_ang && trans_step < min_trans)
                break;
            if (!improved)
                break;

            ang_step *= 0.5;
            trans_step *= 0.5;
        }
    }

    bool do_rot = (fabs(best_rot) > 1e-6 || fabs(best_tilt) > 1e-6 || fabs(best_psi) > 1e-6);
    bool do_trans = (fabs(best_dx) > 1e-6 || fabs(best_dy) > 1e-6 || fabs(best_dz) > 1e-6);

    if (do_rot)
    {
        Matrix2D<RFLOAT> R;
        Euler_rotation3DMatrix(best_rot, best_tilt, best_psi, R);
        MultidimArray<RFLOAT> vol_tmp = vol_align;
        applyGeometry(vol_tmp, vol_align, R, true, false, 0.);
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