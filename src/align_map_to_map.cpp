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
#include "src/exact_map3d.h"
#include "src/fftw.h"
#include "src/gpu_reconstruct.h"
#include "src/transformations.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_thread_num() { return 0; }
#endif

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

MultidimArray<RFLOAT> extractCentralCube(const MultidimArray<RFLOAT> &in)
{
	const long int nx = XSIZE(in), ny = YSIZE(in), nz = ZSIZE(in);
	long int side = XMIPP_MIN(nx, XMIPP_MIN(ny, nz));
	side -= side % 2;
	MultidimArray<RFLOAT> out(side, side, side);
	const long int ox = nx / 2 - side / 2, oy = ny / 2 - side / 2, oz = nz / 2 - side / 2;
	for (long int k = 0; k < side; k++)
		for (long int i = 0; i < side; i++)
			memcpy(&DIRECT_A3D_ELEM(out, k, i, 0), &DIRECT_A3D_ELEM(in, k + oz, i + oy, ox), (size_t)side * sizeof(RFLOAT));
	return out;
}

void downsampleCube(const MultidimArray<RFLOAT> &src, MultidimArray<RFLOAT> &dst, int work_size)
{
	const int orig_size = (int)XSIZE(src);
	if (orig_size == work_size)
	{
		dst = src;
		return;
	}
	if (orig_size > work_size && orig_size >= 192 && gpurec::requested() && gpurec::gpuRequested())
	{
		dst.reshape(work_size, work_size, work_size);
		if (gpurec::downsampleVolume3D(MULTIDIM_ARRAY(src), orig_size, MULTIDIM_ARRAY(dst), work_size))
			return;
	}
	dst = src;
	resizeMap(dst, work_size);
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
int alignWorkSize(int orig_size)
{
    int work_size = 64;
    if (orig_size < work_size) work_size = orig_size;
    if (work_size % 2 != 0) work_size++;
    return work_size;
}

void reduceMapForAlignment(MultidimArray<RFLOAT> &vol)
{
    const int orig_size = (int)XSIZE(vol);
    const int work_size = alignWorkSize(orig_size);
    if (work_size != orig_size)
    {
        if (orig_size > work_size && orig_size >= 192 && gpurec::requested() && gpurec::gpuRequested())
        {
            MultidimArray<RFLOAT> out;
            out.reshape(work_size, work_size, work_size);
            if (gpurec::downsampleVolume3D(MULTIDIM_ARRAY(vol), orig_size, MULTIDIM_ARRAY(out), work_size))
            {
                vol = out;
                return;
            }
        }
        resizeMap(vol, work_size);
    }
}

AlignMapReference prepareAlignReference(
    const MultidimArray<RFLOAT> &vol_ref,
    RFLOAT angpix,
    RFLOAT maxres,
    int nr_threads,
    bool full_size)
{
    AlignMapReference ref;
    ref.orig_x = XSIZE(vol_ref);
    ref.orig_y = YSIZE(vol_ref);
    ref.orig_z = ZSIZE(vol_ref);
    ref.angpix = angpix;
    ref.maxres = maxres;
    ref.full_size = full_size;

    MultidimArray<RFLOAT> cube_storage;
    const MultidimArray<RFLOAT> *cube_ptr = &vol_ref;
    if (ref.orig_x != ref.orig_y || ref.orig_y != ref.orig_z)
    {
        cube_storage = extractCentralCube(vol_ref);
        cube_ptr = &cube_storage;
    }
    ref.cube_side = (int)XSIZE(*cube_ptr);
    ref.work_size = full_size ? ref.cube_side : alignWorkSize(ref.cube_side);
    ref.work_angpix = angpix * ref.cube_side / ref.work_size;

    MultidimArray<RFLOAT> vol_ref_down;
    relionFftwPlanThreads(nr_threads > 0 ? nr_threads : 1);
    downsampleCube(*cube_ptr, vol_ref_down, ref.work_size);

    ref.r_max = (maxres > 0.) ? CEIL(ref.work_size * ref.work_angpix / maxres) : ref.work_size;
    if (ref.r_max > ref.work_size) ref.r_max = ref.work_size;
    ref.fsc_rmax = XMIPP_MIN(ref.r_max, ref.work_size / 2 - 1);

    // See alignMapToMap() comment: rotate reference by ALIGN_REF_ROT so reference and candidates
    // undergo matching trilinear interpolation damping.
    const RFLOAT ALIGN_REF_ROT = 7.3;
    Euler_rotation3DMatrix(ALIGN_REF_ROT, 0., 0., ref.A0);
    ref.ref_power.assign(ref.fsc_rmax + 1, 0.);

    MultidimArray<RFLOAT> dummy;
    Projector ref_projector(ref.work_size, TRILINEAR, 2, 10, 3);
    ref_projector.computeFourierTransformMap(vol_ref_down, dummy, 2 * ref.r_max, 1, false);
    ref.ref_ft.initZeros(ref.work_size, ref.work_size, ref.work_size / 2 + 1);
    ref_projector.rotate3D(ref.ref_ft, ref.A0);
    CenterFFTbySign(ref.ref_ft);

    const Complex * __restrict ref_arr = MULTIDIM_ARRAY(ref.ref_ft);
    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(ref.ref_ft)
    {
        const int r = ROUND(sqrt((RFLOAT) (kp * kp + ip * ip + jp * jp)));
        if (r <= ref.fsc_rmax)
        {
            const long int idx = &FFTW_ELEM(ref.ref_ft, kp, ip, jp) - ref_arr;
            const Complex b = ref_arr[idx];
            ref.ref_power[r] += norm(b);
            ref.power_voxels.push_back({idx, r});
            if (r >= 1)
                ref.active_voxels.push_back({idx, r, (int)jp, (int)ip, (int)kp, (double)b.real, (double)b.imag});
        }
    }
    return ref;
}

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
    RFLOAT &best_dz,
    int nr_threads,
    bool apply_result,
    bool full_size,
    const AlignMapReference *prepared_ref)
{
    best_rot = best_tilt = best_psi = best_dx = best_dy = best_dz = 0.;
    if (nr_freedom == 0) return;

    // RELION_ALIGN_TIMING=1: seconds spent in each stage, on stderr
    const bool timing = getenv("RELION_ALIGN_TIMING") != NULL;
    auto t_last = std::chrono::steady_clock::now();
    auto lap = [&](const char *what)
    {
        if (!timing) return;
        auto now = std::chrono::steady_clock::now();
        fprintf(stderr, "ALIGN_TIMING %s %.3f s\n", what, std::chrono::duration<double>(now - t_last).count());
        t_last = now;
    };

    const int align_size = (int)XSIZE(vol_align);
    const bool have_prepared = (prepared_ref != nullptr && prepared_ref->valid() &&
                                prepared_ref->cube_side == align_size &&
                                prepared_ref->angpix == angpix &&
                                prepared_ref->maxres == maxres &&
                                prepared_ref->full_size == full_size);
    if (!have_prepared && (int)XSIZE(vol_ref) != align_size)
        REPORT_ERROR("alignMapToMap: vol_align and vol_ref must have the same size!");

    const int work_size = full_size ? align_size : alignWorkSize(align_size);
    const RFLOAT work_angpix = angpix * align_size / work_size;

    relionFftwPlanThreads(nr_threads > 0 ? nr_threads : 1);
    MultidimArray<RFLOAT> vol_work;
    downsampleCube(vol_align, vol_work, work_size);

    AlignMapReference local_ref;
    if (!have_prepared)
        local_ref = prepareAlignReference(vol_ref, angpix, maxres, nr_threads, full_size);
    const AlignMapReference &ref = have_prepared ? *prepared_ref : local_ref;
    lap("resize_to_work_size");

    const int r_max = ref.r_max;
    const int fsc_rmax = ref.fsc_rmax;
    const Matrix2D<RFLOAT> &A0 = ref.A0;
    const std::vector<double> &ref_power = ref.ref_power;

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
    lap("projector_setup");

    // One worker per thread: every candidate is scored independently, in parallel, into a
    // table; the winner is then picked serially in the original order, so the result does
    // not depend on the number of threads.
    const int lim = fsc_rmax + 1;
    struct Worker
    {
        MultidimArray<Complex> rot_ft; // the rotated map, in Fourier space
        std::vector<double> power, cross; // per shell, for the Fourier shell correlation score
        std::vector<double> cx, sx, cy, sy, cz, sz; // 1D phase trig tables for diff2At
    };
    std::vector<std::unique_ptr<Worker>> workers;
    {
        if (const char *e = getenv("RELION_ALIGN_THREADS")) nr_threads = atoi(e);
        if (nr_threads < 1) nr_threads = 1;
        const int max_candidates = (nr_freedom == 2) ? (2 * search_range + 1)
                                 : (2 * search_range + 1) * (2 * search_range + 1) * (2 * search_range + 1);
        nr_threads = std::min(nr_threads, std::max(max_candidates, 1));
        for (int t = 0; t < nr_threads; t++)
        {
            workers.emplace_back(new Worker);
            Worker &w = *workers.back();
            w.rot_ft.initZeros(work_size, work_size, work_size / 2 + 1);
            w.cx.resize(lim + 1);
            w.sx.resize(lim + 1);
            w.cy.resize(2 * lim + 1);
            w.sy.resize(2 * lim + 1);
            w.cz.resize(2 * lim + 1);
            w.sz.resize(2 * lim + 1);
        }
    }

    auto forEach = [&](int n, const std::function<void(int, Worker &)> &fn)
    {
        const int nt = std::min(nr_threads, n);
        #pragma omp parallel for schedule(dynamic, 1) num_threads(nt) if(nt > 1)
        for (int i = 0; i < n; i++)
            fn(i, *workers[omp_get_thread_num()]);
    };

    // Rotate the working map by (rot, tilt, psi), then by A0, into the worker's `rot_ft`
    auto rotateInto = [&](Worker &w, RFLOAT rot, RFLOAT tilt, RFLOAT psi)
    {
        Matrix2D<RFLOAT> A;
        Euler_rotation3DMatrix(rot, tilt, psi, A);
        w.rot_ft.initZeros();
        A = A0 * A;
        projector.rotate3D(w.rot_ft, A);
        CenterFFTbySign(w.rot_ft);
        // the shell power does not depend on the shift
        w.power.assign(fsc_rmax + 1, 0.);
        const Complex * __restrict rot_arr = MULTIDIM_ARRAY(w.rot_ft);
        for (const auto &pv : ref.power_voxels)
            w.power[pv.r] += norm(rot_arr[pv.idx]);
    };

    // Minus the weighted mean shell correlation of the worker's rotated
    // map, shifted by (dx, dy, dz) A, with the reference: smaller is better
    auto diff2At = [&](Worker &w, RFLOAT dx, RFLOAT dy, RFLOAT dz) -> double
    {
        // the shift, turned by A0 like the map
        const double f = -2. * PI / (work_angpix * work_size);
        const double fx = f * (A0(0, 0) * dx + A0(0, 1) * dy + A0(0, 2) * dz),
                     fy = f * (A0(1, 0) * dx + A0(1, 1) * dy + A0(1, 2) * dz),
                     fz = f * (A0(2, 0) * dx + A0(2, 1) * dy + A0(2, 2) * dz);
        for (int j = 0; j <= lim; j++)
        {
            const double a = fx * j;
            w.cx[j] = cos(a);
            w.sx[j] = sin(a);
        }
        for (int i = -lim; i <= lim; i++)
        {
            const double a = fy * i;
            w.cy[i + lim] = cos(a);
            w.sy[i + lim] = sin(a);
            const double b = fz * i;
            w.cz[i + lim] = cos(b);
            w.sz[i + lim] = sin(b);
        }
        w.cross.assign(fsc_rmax + 1, 0.);
        const Complex * __restrict rot_arr = MULTIDIM_ARRAY(w.rot_ft);
        const double * __restrict cxp = w.cx.data();
        const double * __restrict sxp = w.sx.data();
        const double * __restrict cyp = w.cy.data() + lim;
        const double * __restrict syp = w.sy.data() + lim;
        const double * __restrict czp = w.cz.data() + lim;
        const double * __restrict szp = w.sz.data() + lim;
        for (const auto &v : ref.active_voxels)
        {
            const double cxv = cxp[v.jp], sxv = sxp[v.jp];
            const double cyv = cyp[v.ip], syv = syp[v.ip];
            const double czv = czp[v.kp], szv = szp[v.kp];
            const double cxy = cxv * cyv - sxv * syv;
            const double sxy = sxv * cyv + cxv * syv;
            const double cph = cxy * czv - sxy * szv;
            const double sph = sxy * czv + cxy * szv;
            const Complex a = rot_arr[v.idx];
            // Re( a * exp(i phase) * conj(b) )
            w.cross[v.r] += (a.real * v.ref_re + a.imag * v.ref_im) * cph
                          - (a.imag * v.ref_re - a.real * v.ref_im) * sph;
        }
        double sum = 0., weight = 0.;
        for (int r = 1; r <= fsc_rmax; r++)
        {
            const double den = sqrt(w.power[r] * ref_power[r]);
            // Weight of a shell: the reference's power in it, so that a noisy or empty shell
            // counts for no more than it does in a plain correlation. The full-size axial
            // search (half maps, maxres at the current resolution) weights by shell size
            // instead: the axial register of a helix shows only in its weak outer shells.
            const double shell_weight = (full_size && nr_freedom == 2) ? r * (double) r : ref_power[r];
            if (den > 0.) sum += shell_weight * w.cross[r] / den;
            weight += shell_weight;
        }
        return -sum / weight;
    };
    auto rotateAll = [&](RFLOAT rot, RFLOAT tilt, RFLOAT psi)
    {
        forEach((int) workers.size(), [&](int, Worker &w) { rotateInto(w, rot, tilt, psi); });
    };

    // Shifts (dx, dy, dz) = best + (ix, iy, iz) * step for ix, iy in [-xy, xy] and iz in
    // [-z, z], z fastest, scored with the rotation currently applied by rotateAll
    auto scoreShifts = [&](RFLOAT cx, RFLOAT cy, RFLOAT cz, RFLOAT step, int xy, int z,
                           std::vector<double> &out)
    {
        const int nxy = 2 * xy + 1, nz = 2 * z + 1;
        out.assign((size_t) nxy * nxy * nz, 0.);
        forEach((int) out.size(), [&](int i, Worker &w)
        {
            const int iz = i % nz - z, iy = (i / nz) % nxy - xy, ix = i / (nz * nxy) - xy;
            out[i] = diff2At(w, cx + ix * step, cy + iy * step, cz + iz * step);
        });
    };

    double best_diff2 = 1E99;

    // ---------------- coarse search ----------------
    if (nr_freedom == 2)
    {
        // Cn / helical: rotation about Z and a shift along Z, searched jointly
        const int n = 2 * search_range + 1;
        std::vector<double> table((size_t) n * n);
        forEach(n, [&](int i, Worker &w)
        {
            rotateInto(w, (i - search_range) * search_step_rot, 0., 0.);
            for (int j = 0; j < n; j++)
                table[(size_t) i * n + j] = diff2At(w, 0., 0., (j - search_range) * search_step_trans);
        });

        double best_mag2 = 1E99;
        for (int irot = -search_range; irot <= search_range; irot++)
        {
            RFLOAT rot = irot * search_step_rot;
            for (int idz = -search_range; idz <= search_range; idz++)
            {
                RFLOAT dz = idz * search_step_trans;
                double diff2 = table[(size_t) (irot + search_range) * n + (idz + search_range)];

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
        const int n = 2 * search_range + 1;
        std::vector<double> table((size_t) n * n * n);
        forEach((int) table.size(), [&](int i, Worker &w)
        {
            const int ipsi = i % n - search_range, itilt = (i / n) % n - search_range, irot = i / (n * n) - search_range;
            rotateInto(w, irot * search_step_rot, itilt * search_step_rot, ipsi * search_step_rot);
            table[i] = diff2At(w, 0., 0., 0.);
        });

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
                    double diff2 = table[((size_t) (irot + search_range) * n + (itilt + search_range)) * n + (ipsi + search_range)];

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
        rotateAll(best_rot, best_tilt, best_psi);
        std::vector<double> shifts;
        scoreShifts(0., 0., 0., search_step_trans, search_range, search_range, shifts);
        for (int idx = -search_range; idx <= search_range; idx++)
            for (int idy = -search_range; idy <= search_range; idy++)
                for (int idz = -search_range; idz <= search_range; idz++)
                {
                    double diff2 = shifts[((size_t) (idx + search_range) * n + (idy + search_range)) * n + (idz + search_range)];
                    if (diff2 < best_diff2)
                    {
                        best_diff2 = diff2;
                        best_dx = idx * search_step_trans;
                        best_dy = idy * search_step_trans;
                        best_dz = idz * search_step_trans;
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
                const int nr = 2 * fine_range + 1, nt = 2 * tilt_range + 1;
                std::vector<double> table((size_t) nr * nt * nt);
                forEach((int) table.size(), [&](int i, Worker &w)
                {
                    const int ipsi = i % nt - tilt_range, itilt = (i / nt) % nt - tilt_range, irot = i / (nt * nt) - fine_range;
                    rotateInto(w, best_rot + irot * ang_step, best_tilt + itilt * ang_step, best_psi + ipsi * ang_step);
                    table[i] = diff2At(w, best_dx, best_dy, best_dz);
                });

                for (int irot = -fine_range; irot <= fine_range; irot++)
                for (int itilt = -tilt_range; itilt <= tilt_range; itilt++)
                for (int ipsi = -tilt_range; ipsi <= tilt_range; ipsi++)
                {
                    RFLOAT rot  = best_rot  + irot  * ang_step;
                    RFLOAT tilt = best_tilt + itilt * ang_step;
                    RFLOAT psi  = best_psi  + ipsi  * ang_step;

                    double diff2 = table[((size_t) (irot + fine_range) * nt + (itilt + tilt_range)) * nt + (ipsi + tilt_range)];

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
                rotateAll(best_rot, best_tilt, best_psi);

                RFLOAT lv_dx = best_dx, lv_dy = best_dy, lv_dz = best_dz;
                double lv_best = best_diff2;

                const int xy_range = (nr_freedom == 2) ? 0 : fine_range;
                const int nxy = 2 * xy_range + 1, nz = 2 * fine_range + 1;
                std::vector<double> table;
                scoreShifts(best_dx, best_dy, best_dz, trans_step, xy_range, fine_range, table);

                for (int idx = -xy_range; idx <= xy_range; idx++)
                for (int idy = -xy_range; idy <= xy_range; idy++)
                for (int idz = -fine_range; idz <= fine_range; idz++)
                {
                    double diff2 = table[((size_t) (idx + xy_range) * nxy + (idy + xy_range)) * nz + (idz + fine_range)];
                    if (diff2 < lv_best)
                    {
                        lv_best = diff2;
                        lv_dx = best_dx + idx * trans_step;
                        lv_dy = best_dy + idy * trans_step;
                        lv_dz = best_dz + idz * trans_step;
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

    lap("search");
    if (apply_result)
        applyMapTransformation(vol_align, best_rot, best_tilt, best_psi, best_dx, best_dy, best_dz, angpix);
    lap("apply_to_full_map");
}

void applyMapTransformation(
    MultidimArray<RFLOAT> &vol_align,
    RFLOAT best_rot, RFLOAT best_tilt, RFLOAT best_psi,
    RFLOAT best_dx, RFLOAT best_dy, RFLOAT best_dz,
    RFLOAT angpix)
{
    bool do_rot = (fabs(best_rot) > 1e-6 || fabs(best_tilt) > 1e-6 || fabs(best_psi) > 1e-6);
    bool do_trans = (fabs(best_dx) > 1e-6 || fabs(best_dy) > 1e-6 || fabs(best_dz) > 1e-6);

#ifdef RELION_USE_FINUFFT
    // RELION_MAP_INTERPOLATION=nufft: rotation and shift as one exact resampling (the same transform as below)
    if ((do_rot || do_trans) && ExactMap3D::use(vol_align, "class and half-map alignment"))
    {
        Matrix2D<RFLOAT> R, T;
        Euler_rotation3DMatrix(best_rot, best_tilt, best_psi, R);
        translation3DMatrix(best_dx / angpix, best_dy / angpix, best_dz / angpix, T);
        const ExactMap3D em(vol_align, omp_get_max_threads());
        em.applyGeometry(vol_align, T * R, false, omp_get_max_threads());
        return;
    }
#endif

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

void alignCuboidMapToMap(
    MultidimArray<RFLOAT> &vol_align,
    const MultidimArray<RFLOAT> &vol_ref,
    int nr_freedom, RFLOAT angpix, RFLOAT maxres,
    int search_range, RFLOAT search_step_rot, RFLOAT search_step_trans,
    RFLOAT &best_rot, RFLOAT &best_tilt, RFLOAT &best_psi,
    RFLOAT &best_dx, RFLOAT &best_dy, RFLOAT &best_dz,
    int nr_threads,
    bool apply_result,
    bool full_size,
    const AlignMapReference *prepared_ref)
{
    if (XSIZE(vol_align) != XSIZE(vol_ref) || YSIZE(vol_align) != YSIZE(vol_ref) || ZSIZE(vol_align) != ZSIZE(vol_ref))
        REPORT_ERROR("alignCuboidMapToMap: the two maps must have the same size.");

    const long int nx = XSIZE(vol_align), ny = YSIZE(vol_align), nz = ZSIZE(vol_align);
    if (nx == ny && ny == nz)
    {
        alignMapToMap(vol_align, vol_ref, nr_freedom, angpix, maxres, search_range, search_step_rot,
                      search_step_trans, best_rot, best_tilt, best_psi, best_dx, best_dy, best_dz, nr_threads,
                      apply_result, full_size, prepared_ref);
        return;
    }

    // The search needs a cube. Use the central cube whose side is the shortest axis
    // (even, so the centre of the cube is the centre of the cuboid), then apply the
    // transformation that was found to the whole cuboid.
    MultidimArray<RFLOAT> cube_align = extractCentralCube(vol_align);
    const bool use_prepared = (prepared_ref != nullptr && prepared_ref->matches(nx, ny, nz, angpix, maxres, full_size));
    if (use_prepared)
    {
        alignMapToMap(cube_align, cube_align, nr_freedom, angpix, maxres, search_range, search_step_rot,
                      search_step_trans, best_rot, best_tilt, best_psi, best_dx, best_dy, best_dz, nr_threads,
                      false, full_size, prepared_ref);
    }
    else
    {
        MultidimArray<RFLOAT> cube_ref = extractCentralCube(vol_ref);
        alignMapToMap(cube_align, cube_ref, nr_freedom, angpix, maxres, search_range, search_step_rot,
                      search_step_trans, best_rot, best_tilt, best_psi, best_dx, best_dy, best_dz, nr_threads,
                      false, full_size);
    }
    if (apply_result)
        applyMapTransformation(vol_align, best_rot, best_tilt, best_psi, best_dx, best_dy, best_dz, angpix);
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