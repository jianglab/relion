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

#ifndef MASK_H_
#define MASK_H_

#include "src/multidim_array.h"
#include "src/fftw.h"
#include "src/time.h"
#include "src/euler.h"
#include "src/macros.h"

// Scale factors that turn the ellipse (ellipsoid) inscribed in a rectangular
// image (cuboid) into a circle (sphere) whose radius is half the shortest side.
// All are 1 for a square image or a cube.
inline void boxShapeScales(const MultidimArray<RFLOAT> &v, RFLOAT &sx, RFLOAT &sy, RFLOAT &sz)
{
	RFLOAT m = (XSIZE(v) < YSIZE(v)) ? XSIZE(v) : YSIZE(v);
	if (ZSIZE(v) > 1 && ZSIZE(v) < m) m = ZSIZE(v);
	sx = m / (RFLOAT)XSIZE(v);
	sy = m / (RFLOAT)YSIZE(v);
	sz = (ZSIZE(v) > 1) ? m / (RFLOAT)ZSIZE(v) : 1.;
}

// The mask of a rectangular image (cuboid map) in relion_refine: an ellipse (ellipsoid)
// with the proportions of the box. The mask radius is its semi-axis along the longest
// side (so the mask diameter is the long axis), and the semi-axes along the other sides
// shrink in proportion to the box. The soft edge is cosine_width pixels wide along
// every axis: the outer ellipse has each semi-axis cosine_width longer.
class BoxEllipse
{
public:
	// Semi-axes of the inner ellipse, in pixels
	RFLOAT ax, ay, az;

	BoxEllipse(const MultidimArray<RFLOAT> &v, RFLOAT radius, RFLOAT cosine_width)
	{
		const bool is3D = (ZSIZE(v) > 1);
		RFLOAT longest = (XSIZE(v) > YSIZE(v)) ? XSIZE(v) : YSIZE(v);
		if (is3D && ZSIZE(v) > longest) longest = ZSIZE(v);
		ax = radius * XSIZE(v) / longest;
		ay = radius * YSIZE(v) / longest;
		az = is3D ? radius * ZSIZE(v) / longest : 1.;
		inv_ax2 = 1. / (ax * ax);
		inv_ay2 = 1. / (ay * ay);
		inv_az2 = is3D ? 1. / (az * az) : 0.;
		inv_bx2 = 1. / ((ax + cosine_width) * (ax + cosine_width));
		inv_by2 = 1. / ((ay + cosine_width) * (ay + cosine_width));
		inv_bz2 = is3D ? 1. / ((az + cosine_width) * (az + cosine_width)) : 0.;
	}

	// The shortest semi-axis
	RFLOAT shortestSemiAxis(bool is3D) const
	{
		RFLOAT a = (ax < ay) ? ax : ay;
		return (is3D && az < a) ? az : a;
	}

	// Where the pixel at (x, y, z) from the centre lies across the soft edge, along the
	// ray from the centre: 0 inside the inner ellipse, 1 outside the outer one, and the
	// fraction of the way from the inner to the outer ellipse in between. For a circle of
	// radius R this is (r - R) / cosine_width.
	RFLOAT edge(RFLOAT x, RFLOAT y, RFLOAT z) const
	{
		const RFLOAT ri2 = x * x * inv_ax2 + y * y * inv_ay2 + z * z * inv_az2;
		if (ri2 <= 1.)
			return 0.;
		const RFLOAT ro2 = x * x * inv_bx2 + y * y * inv_by2 + z * z * inv_bz2;
		if (ro2 >= 1.)
			return 1.;
		const RFLOAT ri = sqrt(ri2), ro = sqrt(ro2);
		return (ri - 1.) * ro / (ri - ro);
	}

private:
	RFLOAT inv_ax2, inv_ay2, inv_az2, inv_bx2, inv_by2, inv_bz2;
};

// The largest in-plane angle (degrees, in 0.5 degree steps; -1 if even 0 does, 90 if none
// does) by which a masked reference with semi-axes `along` and `across` (and, if tube > 0,
// cut to a tube of that radius along its long axis) can turn inside a box of
// box_along x box_across pixels before it reaches beyond the box. Reference projections are
// sampled on the image grid, so beyond the box they wrap around to its other side.
inline RFLOAT inPlaneAngleBeforeWrapAround(RFLOAT along, RFLOAT across, RFLOAT tube, RFLOAT box_along, RFLOAT box_across)
{
	auto wraps = [&](RFLOAT angle_deg) -> bool
	{
		const RFLOAT s = sin(DEG2RAD(angle_deg)), c = cos(DEG2RAD(angle_deg));
		RFLOAT h_across = sqrt(along * along * s * s + across * across * c * c);
		RFLOAT h_along = sqrt(along * along * c * c + across * across * s * s);
		if (tube > 0.)
		{
			h_across = XMIPP_MIN(h_across, along * s + tube * c);
			h_along = XMIPP_MIN(h_along, along * c + tube * s);
		}
		return h_across > box_across / 2. || h_along > box_along / 2.;
	};
	if (wraps(0.))
		return -1.;
	RFLOAT angle = 0.;
	while (angle < 90. && !wraps(angle + 0.5))
		angle += 0.5;
	return angle;
}

// Mask out corners outside sphere (replace by average value)
// Apply a soft mask (raised cosine with cosine_width pixels width)
// With follow_box_shape, the mask is an ellipse (ellipsoid) with the proportions of a
// rectangular image (cuboid): by default the radius belongs to the shortest side; with
// radius_on_long_side it is the BoxEllipse, whose semi-axis along the longest side is the radius.
void softMaskOutsideMap(
		MultidimArray<RFLOAT> &vol,
		RFLOAT radius = -1.,
		RFLOAT cosine_width = 3,
		MultidimArray<RFLOAT> *Mnoise = NULL,
		bool follow_box_shape = false,
		bool radius_on_long_side = false);

// The parameters of softMaskOutsideMapForHelix for a 2D image of xdim x ydim, with the
// same checks (errors as there): the radii and the terms of the rotation that give
// a pixel's distance to the helical axis, |a10 * x + a11 * y|, and its position along
// the axis, a00 * x + a01 * y (for the GPU version)
void helixMaskParameters2D(
		int xdim, int ydim,
		RFLOAT psi_deg,
		RFLOAT mask_sphere_radius_pix,
		RFLOAT mask_cyl_radius_pix,
		RFLOAT cosine_width,
		RFLOAT &a00, RFLOAT &a01,
		RFLOAT &a10, RFLOAT &a11,
		RFLOAT &R1, RFLOAT &R2, RFLOAT &D1, RFLOAT &D2);

// May27,2015 - Shaoda, Helical refinement
// The mask is a tube around the helical axis, cut off by a sphere (circle). With
// follow_box_shape (rectangular images) the sphere becomes the BoxEllipse of the box,
// whose semi-axis along the longest side is mask_sphere_radius_pix; the tube must fit
// inside its shortest semi-axis.
// Without follow_box_shape: keeps what is within mask_cyl_radius_pix of the helical axis and within
// mask_sphere_radius_pix of the centre (a sphere, or a circle in 2D); with cylinder,
// within mask_sphere_radius_pix of the centre along the axis instead (a cylinder as long
// as the mask diameter, a rectangle in 2D). Soft edges of cosine_width in both directions.
void softMaskOutsideMapForHelix(
		MultidimArray<RFLOAT> &vol,
		RFLOAT psi_deg,
		RFLOAT tilt_deg,
		RFLOAT mask_sphere_radius_pix,
		RFLOAT mask_cyl_radius_pix,
		RFLOAT cosine_width = 3,
		MultidimArray<RFLOAT> *Mnoise = NULL,
		bool follow_box_shape = false,
		bool cylinder = false);   // a cylinder overrides the box shape

// Apply a soft mask and set density outside the mask at the average value of those pixels in the original map
void softMaskOutsideMap(MultidimArray<RFLOAT> &vol, MultidimArray<RFLOAT> &msk, bool invert_mask = false);

// Make an automated mask, based on:
// 1. initial binarization (based on ini_mask_density_threshold)
// 2. Growing extend_ini_mask in all directions
// 3. Putting a raised-cosine edge on the mask with width width_soft_mask_edge
// If verb, then output description of steps and progress bars
void autoMask(MultidimArray<RFLOAT> &img_in, MultidimArray<RFLOAT> &msk_out,
		RFLOAT  ini_mask_density_threshold, RFLOAT extend_ini_mask, RFLOAT width_soft_mask_edge, bool verb = false, int n_threads = 1);

// Fills mask with a soft-edge circular mask (soft-edge in between radius and radius_p), centred at (x, y, z)
void raisedCosineMask(MultidimArray<RFLOAT> &mask, RFLOAT radius, RFLOAT radius_p, int x, int y, int z = 0);

// Make a raised crown mask.
void raisedCrownMask(MultidimArray<RFLOAT> &mask, RFLOAT inner_radius, RFLOAT outer_radius, RFLOAT width, RFLOAT x, RFLOAT y, RFLOAT z);

#endif /* MASK_H_ */
