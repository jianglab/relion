/***************************************************************************
 *
 * Author: "Sjors H.W. Scheres"
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
#include <omp.h>
#include "src/mask.h"
#include "src/auto_mask.h"

// https://stackoverflow.com/questions/48273190/undefined-symbol-error-for-stdstringempty-c-standard-method-linking-error/48273604#48273604
#if defined(__APPLE__)
// explicit instantiation of std::string needed, otherwise we get a linker error on osx
// thats a bug in libc++, because of interaction with __attribute__ ((__visibility__("hidden"), __always_inline__)) in std::string
template class std::basic_string<char>;
#endif

// Workaround for compiler versions before 2018 update 2
#ifdef __INTEL_COMPILER
# if (__INTEL_COMPILER<1800)
#  pragma optimize ("", off)
# endif
# if (__INTEL_COMPILER==1800)
#  if (__INTEL_COMPILER_UPDATE<2)
#   pragma optimize ("", off)
#  endif
# endif
#endif
// Mask out corners outside sphere (replace by average value)
// Apply a soft mask (raised cosine with cosine_width pixels width)
// The same with the BoxEllipse of a rectangular image (cuboid) in place of the sphere
static void softMaskOutsideBoxEllipse(MultidimArray<RFLOAT> &vol, RFLOAT radius, RFLOAT cosine_width, MultidimArray<RFLOAT> *Mnoise)
{
	const BoxEllipse ellipse(vol, radius, cosine_width);
	RFLOAT sum_bg = 0., sum = 0.;
	if (Mnoise == NULL)
	{
		// Calculate average background value (raised cosine weights across the edge)
		FOR_ALL_ELEMENTS_IN_ARRAY3D(vol)
		{
			const RFLOAT t = ellipse.edge(j, i, k);
			if (t <= 0.)
				continue;
			const RFLOAT raisedcos = (t >= 1.) ? 1. : 0.5 - 0.5 * cos(PI * t);
			sum += raisedcos;
			sum_bg += raisedcos * A3D_ELEM(vol, k, i, j);
		}
		sum_bg /= sum;
	}

	// Apply noisy or average background value
	FOR_ALL_ELEMENTS_IN_ARRAY3D(vol)
	{
		const RFLOAT t = ellipse.edge(j, i, k);
		if (t <= 0.)
			continue;
		const RFLOAT add = (Mnoise == NULL) ? sum_bg : A3D_ELEM(*Mnoise, k, i, j);
		if (t >= 1.)
			A3D_ELEM(vol, k, i, j) = add;
		else
		{
			const RFLOAT raisedcos = 0.5 - 0.5 * cos(PI * t);
			A3D_ELEM(vol, k, i, j) = (1 - raisedcos) * A3D_ELEM(vol, k, i, j) + raisedcos * add;
		}
	}
}

void softMaskOutsideMap(MultidimArray<RFLOAT> &vol, RFLOAT radius, RFLOAT cosine_width, MultidimArray<RFLOAT> *Mnoise, bool follow_box_shape, bool radius_on_long_side)
{

	vol.setXmippOrigin();
	if (follow_box_shape && radius_on_long_side)
	{
		if (radius < 0)
		{
			RFLOAT longest = (XSIZE(vol) > YSIZE(vol)) ? XSIZE(vol) : YSIZE(vol);
			if (ZSIZE(vol) > longest) longest = ZSIZE(vol);
			radius = longest / 2.;
		}
		softMaskOutsideBoxEllipse(vol, radius, cosine_width, Mnoise);
		return;
	}

	RFLOAT r, radius_p, raisedcos, sum_bg = 0., sum = 0.;
	if (radius < 0)
		radius = (RFLOAT)XSIZE(vol)/2.;
	radius_p = radius + cosine_width;
	RFLOAT sx = 1., sy = 1., sz = 1.;
	if (follow_box_shape)
		boxShapeScales(vol, sx, sy, sz);


	const RFLOAT radius2 = radius * radius;
	const RFLOAT radius_p2 = radius_p * radius_p;
	const long int xdim = XSIZE(vol), ydim = YSIZE(vol), zdim = ZSIZE(vol);
	const long int sz0 = STARTINGZ(vol), sy0 = STARTINGY(vol), sx0 = STARTINGX(vol);
	const int nt = relionBulkThreads(MULTIDIM_SIZE(vol));

	if (Mnoise == NULL)
	{
		// Calculate average background value
		#pragma omp parallel for reduction(+:sum, sum_bg) num_threads(nt) if(nt > 1)
		for (long int kz = 0; kz < zdim; kz++)
		{
			const long int k = sz0 + kz;
			const RFLOAT kz2 = follow_box_shape ? (k * sz) * (k * sz) : (RFLOAT)(k * k);
			for (long int ky = 0; ky < ydim; ky++)
			{
				const long int i = sy0 + ky;
				const RFLOAT ky2z2 = kz2 + (follow_box_shape ? (i * sy) * (i * sy) : (RFLOAT)(i * i));
				RFLOAT * __restrict vrow = &DIRECT_A3D_ELEM(vol, kz, ky, 0);
				for (long int kx = 0; kx < xdim; kx++)
				{
					const long int j = sx0 + kx;
					const RFLOAT r2 = ky2z2 + (follow_box_shape ? (j * sx) * (j * sx) : (RFLOAT)(j * j));
					if (r2 < radius2)
						continue;
					else if (r2 > radius_p2)
					{
						sum    += 1.;
						sum_bg += vrow[kx];
					}
					else
					{
						const RFLOAT r = sqrt(r2);
						const RFLOAT rc = 0.5 + 0.5 * cos(PI * (radius_p - r) / cosine_width);
						sum += rc;
						sum_bg += rc * vrow[kx];
					}
				}
			}
		}
		sum_bg /= sum;
	}

	// Apply noisy or average background value
	#pragma omp parallel for num_threads(nt) if(nt > 1)
	for (long int kz = 0; kz < zdim; kz++)
	{
		const long int k = sz0 + kz;
		const RFLOAT kz2 = follow_box_shape ? (k * sz) * (k * sz) : (RFLOAT)(k * k);
		for (long int ky = 0; ky < ydim; ky++)
		{
			const long int i = sy0 + ky;
			const RFLOAT ky2z2 = kz2 + (follow_box_shape ? (i * sy) * (i * sy) : (RFLOAT)(i * i));
			RFLOAT * __restrict vrow = &DIRECT_A3D_ELEM(vol, kz, ky, 0);
			const RFLOAT * __restrict nrow = (Mnoise != NULL) ? &DIRECT_A3D_ELEM(*Mnoise, kz, ky, 0) : NULL;
			for (long int kx = 0; kx < xdim; kx++)
			{
				const long int j = sx0 + kx;
				const RFLOAT r2 = ky2z2 + (follow_box_shape ? (j * sx) * (j * sx) : (RFLOAT)(j * j));
				if (r2 < radius2)
					continue;
				const RFLOAT add = (nrow == NULL) ? sum_bg : nrow[kx];
				if (r2 > radius_p2)
				{
					vrow[kx] = add;
				}
				else
				{
					const RFLOAT r = sqrt(r2);
					const RFLOAT rc = 0.5 + 0.5 * cos(PI * (radius_p - r) / cosine_width);
					vrow[kx] = (1 - rc) * vrow[kx] + rc * add;
				}
			}
		}
	}

}

// May27,2015 - Shaoda, Helical refinement
void helixMaskParameters2D(
		int xdim, int ydim,
		RFLOAT psi_deg,
		RFLOAT mask_sphere_radius_pix,
		RFLOAT mask_cyl_radius_pix,
		RFLOAT cosine_width,
		RFLOAT &a00, RFLOAT &a01,
		RFLOAT &a10, RFLOAT &a11,
		RFLOAT &R1, RFLOAT &R2, RFLOAT &D1, RFLOAT &D2)
{
	// As softMaskOutsideMapForHelix for dim == 2 (tilt does not apply)
	int boxsize = (xdim < ydim) ? xdim : ydim;
	boxsize = boxsize / 2 - ((boxsize + 1) % 2);
	if ( (cosine_width < 0.)
			|| (mask_sphere_radius_pix < 1.) || (mask_sphere_radius_pix > boxsize)
			|| (mask_cyl_radius_pix < 1.) || (mask_cyl_radius_pix > boxsize)
			|| (mask_sphere_radius_pix < mask_cyl_radius_pix) ) {
		std::cerr << " cosine_width= " << cosine_width << std::endl;
		std::cerr << " boxsize= " << boxsize << std::endl;
		std::cerr << " mask_sphere_radius_pix= " << mask_sphere_radius_pix << std::endl;
		std::cerr << " mask_cyl_radius_pix= " << mask_cyl_radius_pix << std::endl;
		REPORT_ERROR("mask.cpp::softMaskOutsideMapForHelix(): Invalid radii of spherical and cylindrical masks or soft cosine widths! Is boxsize > mask_sphere_radius > mask_cyl_radius?");
	}
	R1 = mask_sphere_radius_pix;
	R2 = R1 + cosine_width;
	D1 = mask_cyl_radius_pix;
	D2 = D1 + cosine_width;
	Matrix2D<RFLOAT> A;
	A.resize(3, 3);
	Euler_angles2matrix(0., 0., psi_deg, A, false);
	A = A.transpose();
	a00 = A(0, 0);
	a01 = A(0, 1);
	a10 = A(1, 0);
	a11 = A(1, 1);
}

void softMaskOutsideMapForHelix(
		MultidimArray<RFLOAT> &vol,
		RFLOAT psi_deg,
		RFLOAT tilt_deg,
		RFLOAT mask_sphere_radius_pix,
		RFLOAT mask_cyl_radius_pix,
		RFLOAT cosine_width,
		MultidimArray<RFLOAT> *Mnoise,
		bool follow_box_shape,
		bool cylinder)
{
	Matrix2D<RFLOAT> A;
	RFLOAT sum_bg, sum, R1, R2, D1, D2, r, d, noise_w, noise_w1, noise_w2, noise_val;
	int dim = vol.getDim();
	int boxsize = -1;

	if (follow_box_shape)
	{
		vol.setXmippOrigin();
		const bool is3D = (dim == 3);
		long int longest = (XSIZE(vol) > YSIZE(vol)) ? XSIZE(vol) : YSIZE(vol);
		long int shortest = (XSIZE(vol) < YSIZE(vol)) ? XSIZE(vol) : YSIZE(vol);
		if (is3D && ZSIZE(vol) > longest) longest = ZSIZE(vol);
		if (is3D && ZSIZE(vol) < shortest) shortest = ZSIZE(vol);
		if (longest == shortest)
			follow_box_shape = false; // a square box: the usual circle, bit for bit
		else
		{
			// The ellipse reaches the longest side, the tube must fit across its shortest semi-axis
			const long int half_long = longest / 2 - ((longest + 1) % 2);
			const long int half_short = shortest / 2 - ((shortest + 1) % 2);
			const RFLOAT across = BoxEllipse(vol, mask_sphere_radius_pix, cosine_width).shortestSemiAxis(is3D);
			if ( (cosine_width < 0.)
					|| (mask_sphere_radius_pix < 1.) || (mask_sphere_radius_pix > half_long)
					|| (mask_cyl_radius_pix < 1.) || (mask_cyl_radius_pix > half_short)
					|| (across < mask_cyl_radius_pix) )
			{
				std::cerr << " box= " << XSIZE(vol) << " x " << YSIZE(vol);
				if (is3D) std::cerr << " x " << ZSIZE(vol);
				std::cerr << " cosine_width= " << cosine_width << " mask_radius_pix (long semi-axis)= " << mask_sphere_radius_pix
				          << " short semi-axis= " << across << " mask_cyl_radius_pix= " << mask_cyl_radius_pix << std::endl;
				REPORT_ERROR("mask.cpp::softMaskOutsideMapForHelix(): for a rectangular image the mask is an ellipse whose long axis is the mask diameter and whose short axis shrinks in proportion to the box. The mask radius must not exceed half the long side, and the tube radius must not exceed half the short side nor the short semi-axis of the ellipse (mask radius * short side / long side): use a larger mask diameter or a smaller tube diameter.");
			}
		}
	}

	// Center the box
	vol.setXmippOrigin();
	// Dimension of a particle (box) should be 2 or 3
	if ( (dim != 2) && (dim != 3) )
	{
		REPORT_ERROR("mask.cpp::softMaskOutsideMapForHelix(): Dimension of particles should be 2 or 3!");
		return;
	}
	// Check the shape of Mnoise
	if ( (Mnoise != NULL) && ((*Mnoise).sameShape(vol) == false) )
	{
		REPORT_ERROR("mask.cpp::softMaskOutsideMapForHelix(): Input particle and Mnoise should have same shape!");
		return;
	}
	// Box size is the minimum value of the 2 or 3 dimensions
	boxsize = (XSIZE(vol) < YSIZE(vol)) ? XSIZE(vol) : YSIZE(vol);
	// If it is a 2D particle, tilt angle does not apply
	if (dim == 2)
		tilt_deg = 0.;
	else
		boxsize = (boxsize < ZSIZE(vol)) ? boxsize : ZSIZE(vol);
	boxsize = boxsize / 2 - ((boxsize + 1) % 2);

	// Diameter of the cylindrical mask around the helix should not exceed the box size, otherwise noise cannot be estimated
	// (checked above for the ellipse of a rectangular box)
	if ( !follow_box_shape && ( (cosine_width < 0.)
			|| (mask_sphere_radius_pix < 1.) || (mask_sphere_radius_pix > boxsize)
			|| (mask_cyl_radius_pix < 1.) || (mask_cyl_radius_pix > boxsize)
			|| (mask_sphere_radius_pix < mask_cyl_radius_pix) ) ) {
        std::cerr << " cosine_width= " << cosine_width << std::endl;
        std::cerr << " boxsize= " << boxsize << std::endl;
        std::cerr << " mask_sphere_radius_pix= " << mask_sphere_radius_pix << std::endl;
        std::cerr << " mask_cyl_radius_pix= " << mask_cyl_radius_pix << std::endl;
        REPORT_ERROR("mask.cpp::softMaskOutsideMapForHelix(): Invalid radii of spherical and cylindrical masks or soft cosine widths! Is boxsize > mask_sphere_radius > mask_cyl_radius?");
    }

	// Spherical mask: 0 < R1 < R2
	R1 = mask_sphere_radius_pix;
	R2 = R1 + cosine_width;
	// Cylindrical mask: 0 < D1 < D2
	D1 = mask_cyl_radius_pix;
	D2 = D1 + cosine_width;

	// Init rotational matrix A
	A.clear();
	A.resize(3, 3);

	// Rotate the particle (helical axes are X and Z for 2D and 3D segments respectively)
	Euler_angles2matrix(0., tilt_deg, psi_deg, A, false);
	// Don't put negative signs before tilt and psi values, use 'transpose' instead
	A = A.transpose();

	// Distance of a pixel to the helical axis (X for 2D, Z for 3D) after rotating it
	// by A. The products are summed in the order Matrix2D * Matrix1D uses, so the
	// result is bitwise the same, without allocating a vector for every pixel.
	const RFLOAT a00 = A(0, 0), a01 = A(0, 1), a02 = A(0, 2);
	const RFLOAT a10 = A(1, 0), a11 = A(1, 1), a12 = A(1, 2);
	const RFLOAT a20 = A(2, 0), a21 = A(2, 1), a22 = A(2, 2);
	auto axisDistance = [&](long int k, long int i, long int j) -> RFLOAT
	{
		const RFLOAT x = (RFLOAT)j, y = (RFLOAT)i, z = (dim == 3) ? (RFLOAT)k : 0.;
		const RFLOAT ry = ((0. + a10 * x) + a11 * y) + a12 * z;
		if (dim == 3)
		{
			const RFLOAT rx = ((0. + a00 * x) + a01 * y) + a02 * z;
			return sqrt(ry * ry + rx * rx);
		}
		return ABS(ry);
	};
	// Distance of a pixel from the centre along the helical axis
	auto axisPosition = [&](long int k, long int i, long int j) -> RFLOAT
	{
		const RFLOAT x = (RFLOAT)j, y = (RFLOAT)i, z = (dim == 3) ? (RFLOAT)k : 0.;
		if (dim == 3)
			return ABS(((0. + a20 * x) + a21 * y) + a22 * z);
		return ABS(((0. + a00 * x) + a01 * y) + a02 * z);
	};

	// Calculate noise weights for all voxels
	sum_bg = sum = 0.;
	if (Mnoise == NULL)
	{
		FOR_ALL_ELEMENTS_IN_ARRAY3D(vol)
		{
			d = axisDistance(k, i, j);
			if (d > D2) // Noise areas (get values for noise estimations)
			{
				sum_bg += A3D_ELEM(vol, k, i, j);
				sum += 1.;
			}
			else if (d > D1) // Edges of noise areas (get values and weights for noise estimations)
			{
				noise_w = 0.5 + 0.5 * cos(PI * (D2 - d) / cosine_width );
				sum_bg += noise_w * A3D_ELEM(vol, k, i, j);
				sum += noise_w;
			}
		}
		// Test (this should not happen)
		if (sum < 0.00001)
			REPORT_ERROR("mask.cpp::softMaskOutsideMapForHelix(): No background (noise) areas found in this particle!");
		sum_bg /= sum;
	}

	// Apply noisy or average background value
	noise_val = sum_bg;
	// Rectangular images: an ellipse instead of the sphere; t is where a pixel lies across its edge
	const BoxEllipse ellipse(vol, R1, cosine_width);
	RFLOAT t = 0.;
	FOR_ALL_ELEMENTS_IN_ARRAY3D(vol)
	{
		d = axisDistance(k, i, j);

		bool in_sphere, out_sphere;
		if (cylinder)
		{
			// Distance from the centre along the helical axis
			r = axisPosition(k, i, j);
			in_sphere = (r < R1);
			out_sphere = (r > R2);
		}
		else if (follow_box_shape)
		{
			t = ellipse.edge(j, i, (dim == 3) ? k : 0);
			in_sphere = (t <= 0.);
			out_sphere = (t >= 1.);
		}
		else
		{
			// Distance from the origin
			r = (RFLOAT)(i * i + j * j);
			if (dim == 3)
				r += (RFLOAT)(k * k);
			r = sqrt(r);
			in_sphere = (r < R1);
			out_sphere = (r > R2);
		}

		// Info areas
		if ( in_sphere && (d < D1) )
			continue;

		if (Mnoise != NULL)
			noise_val = A3D_ELEM(*Mnoise, k, i, j);

		if ( out_sphere || (d > D2) )  // Noise areas, fill in background values
			A3D_ELEM(vol, k, i, j) = noise_val;
		else // Edges of info areas
		{
			noise_w1 = noise_w2 = 0.;
			if (follow_box_shape && !cylinder)
			{
				if (t > 0.)
					noise_w1 = 0.5 - 0.5 * cos(PI * t);
			}
			else if (r > R1)
				noise_w1 = 0.5 + 0.5 * cos(PI * (R2 - r) / cosine_width );
			if (d > D1)
				noise_w2 = 0.5 + 0.5 * cos(PI * (D2 - d) / cosine_width );
			noise_w = (noise_w1 > noise_w2) ? (noise_w1) : (noise_w2);
			A3D_ELEM(vol, k, i, j) = (1. - noise_w) * A3D_ELEM(vol, k, i, j) + noise_w * noise_val;
		}
	}
	return;
}

// Workaround for compiler versions before 2018 update 2
#ifdef __INTEL_COMPILER
# if (__INTEL_COMPILER<1800)
#  pragma optimize ("", on)
# endif
# if (__INTEL_COMPILER==1800)
#  if (__INTEL_COMPILER_UPDATE<2)
#   pragma optimize ("", on)
#  endif
# endif
#endif

void softMaskOutsideMap(MultidimArray<RFLOAT> &vol, MultidimArray<RFLOAT> &msk, bool invert_mask)
{

	if (msk.computeMax() > 1. || msk.computeMin() < 0.)
	{
		std::cerr << " msk.computeMax()= " << msk.computeMax() << " msk.computeMin()= " << msk.computeMin() << std::endl;
		REPORT_ERROR("ERROR: Values in the solvent mask should be between zero and one.");
	}
	if (!(msk.sameShape(vol)))
		REPORT_ERROR("ERROR: Solvent mask does not have the same size as the reference vol.");

	// Replace solvent by the average value in the solvent region
	RFLOAT sum = 0.;
	RFLOAT sum_bg = 0.;
	RFLOAT solv;
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(msk)
	{
		solv = (invert_mask) ? DIRECT_A3D_ELEM(msk, k, i, j) : 1. - DIRECT_A3D_ELEM(msk, k, i, j);
		sum    += solv;
		sum_bg += solv * DIRECT_A3D_ELEM(vol, k, i, j);
	}
	sum_bg /= sum;

	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(msk)
	{
		solv = (invert_mask) ? DIRECT_A3D_ELEM(msk, k, i, j) : 1. - DIRECT_A3D_ELEM(msk, k, i, j);
		DIRECT_A3D_ELEM(vol, k, i, j) = ( 1. - solv) * DIRECT_A3D_ELEM(vol, k, i, j) + solv * sum_bg;
	}


}

void autoMask(MultidimArray<RFLOAT> &img_in, MultidimArray<RFLOAT> &msk_out,
		RFLOAT ini_mask_density_threshold, RFLOAT extend_ini_mask, RFLOAT width_soft_mask_edge, bool verb, int n_threads)

{
	MultidimArray<RFLOAT> msk_cp;

	// Resize output mask
	img_in.setXmippOrigin();
	msk_out.clear();
	msk_out.resize(img_in);

	// A. Calculate initial binary mask based on density threshold
	const long int nr_vox = MULTIDIM_SIZE(img_in);
	const RFLOAT * __restrict in_ptr = MULTIDIM_ARRAY(img_in);
	RFLOAT * __restrict out_ptr = MULTIDIM_ARRAY(msk_out);
	#pragma omp parallel for num_threads(n_threads) if(n_threads > 1)
	for (long int n = 0; n < nr_vox; n++)
	{
		out_ptr[n] = (in_ptr[n] >= ini_mask_density_threshold) ? 1. : 0.;
	}

	// B. extend/shrink initial binary mask. A voxel changes if a voxel of the other value lies closer than
	// extend_ini_mask; the exact squared distance to the nearest such voxel replaces the search of the
	// (2 * extend_size + 1)^3 neighbours, with identical results (the distances are integers either way).
	if (extend_ini_mask > 0. || extend_ini_mask < 0.)
	{
		if (verb)
		{
			if (extend_ini_mask > 0.)
				std::cout << "== Extending initial binary mask ..." << std::endl;
			else
				std::cout << "== Shrinking initial binary mask ..." << std::endl;
		}

		const bool extend = extend_ini_mask > 0.;
		RFLOAT extend_ini_mask2 = extend_ini_mask * extend_ini_mask;
		// Sources: voxels that are one (extending) or zero (shrinking)
		msk_cp.resize(msk_out);
		RFLOAT * __restrict cp_ptr = MULTIDIM_ARRAY(msk_cp);
		#pragma omp parallel for num_threads(n_threads) if(n_threads > 1)
		for (long int n = 0; n < nr_vox; n++)
		{
			const RFLOAT m = out_ptr[n];
			cp_ptr[n] = (extend ? m > 0.999 : m < 0.001) ? 1. : 0.;
		}
		MultidimArray<RFLOAT> d2;
		autoMaskSquaredDistance(msk_cp, d2, n_threads);
		const RFLOAT * __restrict d2_ptr = MULTIDIM_ARRAY(d2);
		#pragma omp parallel for num_threads(n_threads) if(n_threads > 1)
		for (long int n = 0; n < nr_vox; n++)
		{
			const RFLOAT m = out_ptr[n];
			if ((extend ? m < 0.001 : m > 0.999) && d2_ptr[n] < extend_ini_mask2)
				out_ptr[n] = extend ? 1. : 0.;
		}
	}

	if (width_soft_mask_edge > 0.)
	{
		if (verb)
			std::cout << "== Making a soft edge on the extended mask ..." << std::endl;
		// C. Make a soft edge to the mask: zero voxels closer than width_soft_mask_edge to a voxel that is one

		RFLOAT width_soft_mask_edge2 = width_soft_mask_edge * width_soft_mask_edge;
		msk_cp.resize(msk_out);
		RFLOAT * __restrict cp_ptr = MULTIDIM_ARRAY(msk_cp);
		#pragma omp parallel for num_threads(n_threads) if(n_threads > 1)
		for (long int n = 0; n < nr_vox; n++)
			cp_ptr[n] = (out_ptr[n] > 0.999) ? 1. : 0.;
		MultidimArray<RFLOAT> d2;
		autoMaskSquaredDistance(msk_cp, d2, n_threads);
		const RFLOAT * __restrict d2_ptr = MULTIDIM_ARRAY(d2);
		const long int max_d2 = (long int)floor(width_soft_mask_edge2);
		std::vector<RFLOAT> edge_lut;
		if (max_d2 >= 0 && max_d2 <= 1048576)
		{
			edge_lut.resize(max_d2 + 1, 0.);
			for (long int d2i = 0; d2i <= max_d2; d2i++)
			{
				if ((RFLOAT)d2i < width_soft_mask_edge2)
					edge_lut[d2i] = 0.5 + 0.5 * cos(PI * sqrt((RFLOAT)d2i) / width_soft_mask_edge);
			}
		}
		const bool use_lut = !edge_lut.empty();
		#pragma omp parallel for num_threads(n_threads) if(n_threads > 1)
		for (long int n = 0; n < nr_vox; n++)
		{
			const RFLOAT min_r2 = d2_ptr[n];
			if (out_ptr[n] < 0.001 && min_r2 < width_soft_mask_edge2)
			{
				if (use_lut)
					out_ptr[n] = edge_lut[(long int)(min_r2 + 0.5)];
				else
					out_ptr[n] = 0.5 + 0.5 * cos(PI * sqrt(min_r2) / width_soft_mask_edge);
			}
		}
	}

}

void raisedCosineMask(MultidimArray<RFLOAT> &mask, RFLOAT radius, RFLOAT radius_p, int x, int y, int z)
{
	mask.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY3D(mask)
	{
		// calculate distance from the origin
		RFLOAT d = sqrt((RFLOAT)((z-k)*(z-k) + (y-i)*(y-i) + (x-j)*(x-j)));
		if (d > radius_p)
			A3D_ELEM(mask, k, i, j) = 0.;
		else if (d < radius)
			A3D_ELEM(mask, k, i, j) = 1.;
		else
			A3D_ELEM(mask, k, i, j) = 0.5 - 0.5 * cos(PI * (radius_p - d) / (radius_p - radius));
	}
}

void raisedCrownMask(MultidimArray<RFLOAT> &mask, RFLOAT inner_radius, RFLOAT outer_radius, RFLOAT width, RFLOAT x, RFLOAT y, RFLOAT z)
{
	RFLOAT inner_border = inner_radius - width;
	RFLOAT outer_border = outer_radius + width;

	mask.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY3D(mask)
	{
		RFLOAT d = sqrt((RFLOAT)((z-k)*(z-k) + (y-i)*(y-i) + (x-j)*(x-j)));
		if (d < inner_border)
			A3D_ELEM(mask, k, i, j) = 0.;
		else if (d < inner_radius)
			A3D_ELEM(mask, k, i, j) = 0.5  - 0.5 * cos(PI * (d - inner_border) / width);
		else if (d < outer_radius) 
			A3D_ELEM(mask, k, i, j) = 1.; 
		else if (d < outer_border)
			A3D_ELEM(mask, k, i, j) = 0.5 - 0.5 * cos(PI * (outer_border - d) / width);
		else
			A3D_ELEM(mask, k, i, j) = 0.;
	}
}


