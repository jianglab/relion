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

    // Working box size for the search
    int work_size = 64;
    if (orig_size < work_size) work_size = orig_size;
    if (work_size % 2 != 0) work_size++;
    RFLOAT work_angpix = angpix * orig_size / work_size;

    // Downsample both maps to the working size
    MultidimArray<RFLOAT> vol_work, vol_ref_down;
    vol_work = vol_align;
    vol_ref_down = vol_ref;
    resizeMap(vol_work, work_size);
    resizeMap(vol_ref_down, work_size);

    // Center for FFT
    CenterFFT(vol_work, true);
    CenterFFT(vol_ref_down, true);

    // Maximum Fourier radius to consider (from maxres)
    int r_max = (maxres > 0.) ? CEIL(work_size * work_angpix / maxres) : work_size;
    if (r_max > work_size) r_max = work_size;

    // Create Projector from the alignment map (source)
    MultidimArray<RFLOAT> dummy;
    Projector projector(work_size, TRILINEAR, 1, 10, 3);
    projector.computeFourierTransformMap(vol_work, dummy, 2 * r_max, 1, false);

    // Set up transformer for rotating the map via Fourier interpolation
    MultidimArray<RFLOAT> rotated;
    MultidimArray<Complex> rot_ft;
    FourierTransformer transformer;
    rotated.reshape(vol_work);
    transformer.setReal(rotated);
    transformer.getFourierAlias(rot_ft);

    double best_diff2 = 1E99;

    // Generate trial parameters and evaluate
    if (nr_freedom == 2) // Cn (n>=2) or helical: search rot + dz
    {
        for (int irot = -search_range; irot <= search_range; irot++)
        {
            RFLOAT rot = irot * search_step_rot;
            Matrix2D<RFLOAT> A;
            Euler_rotation3DMatrix(rot, 0., 0., A);

            rot_ft.initZeros();
            projector.rotate3D(rot_ft, A);
            transformer.inverseFourierTransform();

            for (int idz = -search_range; idz <= search_range; idz++)
            {
                RFLOAT dz = idz * search_step_trans;

                MultidimArray<RFLOAT> trial = rotated;
                if (fabs(dz) > 0.)
                {
                    Matrix1D<RFLOAT> shift(3);
                    XX(shift) = 0.; YY(shift) = 0.; ZZ(shift) = dz / work_angpix;
                    selfTranslate(trial, shift, DONT_WRAP);
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
    else if (nr_freedom == 6) // C1: search all 6 DOF
    {
        // Stage 1: rotation-only coarse search (125 evaluations)
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
                    transformer.inverseFourierTransform();

                    double diff2 = 0;
                    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(rotated)
                    {
                        double d = DIRECT_MULTIDIM_ELEM(rotated, n)
                                 - DIRECT_MULTIDIM_ELEM(vol_ref_down, n);
                        diff2 += d * d;
                    }

                    if (diff2 < best_diff2)
                    {
                        best_diff2 = diff2;
                        best_rot = rot;
                        best_tilt = tilt;
                        best_psi = psi;
                    }
                }
            }
        }

        // Apply the best rotation to the working map for translation search
        {
            Matrix2D<RFLOAT> A;
            Euler_rotation3DMatrix(best_rot, best_tilt, best_psi, A);
            rot_ft.initZeros();
            projector.rotate3D(rot_ft, A);
            transformer.inverseFourierTransform();
        }

        // Stage 2: translation-only search (125 evaluations)
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
                        selfTranslate(trial, shift, DONT_WRAP);
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
                        best_dx = dx;
                        best_dy = dy;
                        best_dz = dz;
                    }
                }
            }
        }
    }

    // Apply the best transformation to the original full-size vol_align
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
    if (nr_freedom == 2) // Cn/helical: only Z-rotation and Z-shift
    {
        p_rot -= drot;
        p_dz -= ddz;
    }
    else if (nr_freedom == 6) // C1: full inverse
    {
        // Compose the inverse of the reference rotation into the particle orientation
        // The reference was rotated by (drot, dtilt, dpsi) → new particle orientation
        // is old orientation composed with the inverse of that rotation.
        // Euler_new = Euler_old * R_inv → using Euler_apply_transf(L=I, R=R_inv)
        Matrix2D<RFLOAT> A_rot(3,3), L(3,3), R(3,3);
        Euler_angles2matrix(drot, dtilt, dpsi, A_rot);
        L.initIdentity();
        R = A_rot.transpose(); // inverse = transpose for rotation matrices
        Euler_apply_transf(L, R, p_rot, p_tilt, p_psi, p_rot, p_tilt, p_psi);

        // The map was shifted by (ddx, ddy, ddz). The particle offset needs
        // to be adjusted by the inverse shift.
        p_dx -= ddx;
        p_dy -= ddy;
        p_dz -= ddz;
    }
    // nr_freedom == 0: do nothing
}
