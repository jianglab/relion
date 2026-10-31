/***************************************************************************
 *
 * Automatic solvent masks for 3D maps, after the masks of trueFSC in helicon:
 * https://github.com/jianglab/helicon/blob/main/src/helicon/commands/trueFSC.py
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
 ***************************************************************************/

#ifndef AUTO_MASK_H_
#define AUTO_MASK_H_

#include "src/multidim_array.h"

// The mask is made in four steps:
//  1. low-pass filter the map with a Gaussian (sigma = lowpass / (3.81 * angpix) pixels)
//  2. threshold it: Otsu's threshold of its histogram, unless a threshold, a fraction of
//     its maximum or the mass inside the mask is given
//  3. keep the regions above the threshold (26-connected) that hold any of its brightest voxels
//  4. add a raised-cosine soft edge outside them. With two half-maps the width of the edge is
//     the one that best separates signal from noise: the FSC of the masked half-maps against
//     that of the masked phase-randomised half-maps. Otherwise it is 3 times the resolution.
// With two half-maps the mask is made from their average, and the low-pass filter is the
// resolution at which their FSC drops below 0.8 (which is also where the phases are randomised).

struct AutoMaskOptions
{
	// Low-pass filter (A); <= 0: from the FSC of the half-maps (no filter for a single map)
	RFLOAT lowpass = -1.;
	// The threshold, by the first of these that is positive (otherwise Otsu's threshold):
	RFLOAT threshold = -1.;          // a map value
	RFLOAT threshold_fraction = -1.; // a fraction of the maximum of the low-passed map
	RFLOAT mass = -1.;               // the mass (kDa, at 0.81 Da/A^3) inside the binary mask
	// Grow (> 0) or shrink (< 0) the binary mask by this many pixels
	RFLOAT extend = 0.;
	// Width of the soft edge (pixels); < 0: optimised (two half-maps) or 3 times the resolution
	RFLOAT soft_edge = -1.;
	// The regions that hold any of this many brightest voxels are kept
	int nr_seeds = 1000;
	// Fourier shells beyond this one carry no signal (e.g. beyond the current image size of a
	// refinement); -1: all shells up to Nyquist
	int max_shell = -1;
	bool verb = false;
};

struct AutoMaskReport
{
	RFLOAT lowpass = -1.;     // A, -1 if none
	RFLOAT threshold = 0.;
	RFLOAT mass = 0.;         // kDa inside the binary mask
	RFLOAT soft_edge = 0.;    // pixels
	bool soft_edge_optimised = false;
	RFLOAT resolution = -1.;  // A at FSC = 0.143 of the unmasked half-maps
};

// Fourier shell correlation of two maps (also cuboids), in shells of 1 / (XSIZE * angpix);
// 0 in shells without any power
void autoMaskFsc(const MultidimArray<RFLOAT> &m1, const MultidimArray<RFLOAT> &m2, MultidimArray<RFLOAT> &fsc);

// The resolution (A) at which the FSC first drops below threshold (from shell 1, interpolated
// between shells), for a box of XSIZE box pixels; -1 if it never does
RFLOAT autoMaskResolutionAtFsc(const MultidimArray<RFLOAT> &fsc, RFLOAT threshold, int box, RFLOAT angpix);

// Low-pass filter with a Gaussian of sigma = lowpass / (3.81 * angpix) pixels (in Fourier space)
void autoMaskGaussianLowPass(MultidimArray<RFLOAT> &map, RFLOAT lowpass, RFLOAT angpix);

// Otsu's threshold of the histogram of the map (256 bins between its minimum and maximum, voxels
// that are exactly 0 left out), as in EMAN2
RFLOAT autoMaskOtsuThreshold(const MultidimArray<RFLOAT> &map, int nr_bins = 256);

// Steps 1-3: binary mask (1/0) of the map
void autoMaskBinary(const MultidimArray<RFLOAT> &map, RFLOAT angpix, RFLOAT lowpass,
		const AutoMaskOptions &opt, MultidimArray<RFLOAT> &binary, AutoMaskReport &report);

// Euclidean distance (pixels) of every voxel to the nearest voxel of the binary mask (0 inside)
void autoMaskDistance(const MultidimArray<RFLOAT> &binary, MultidimArray<RFLOAT> &dist, int n_threads = 1);

// Raised-cosine soft edge of width pixels outside the voxels at distance 0
void autoMaskSoftEdge(const MultidimArray<RFLOAT> &dist, RFLOAT width, MultidimArray<RFLOAT> &mask);

// The mask of two half-maps
void autoMaskFromHalfMaps(const MultidimArray<RFLOAT> &half1, const MultidimArray<RFLOAT> &half2,
		RFLOAT angpix, const AutoMaskOptions &opt, MultidimArray<RFLOAT> &mask,
		AutoMaskReport *report = NULL, int n_threads = 1);

// The mask of a single map; resolution (A) sets the soft edge when opt.soft_edge < 0
void autoMaskFromMap(const MultidimArray<RFLOAT> &map, RFLOAT angpix, RFLOAT resolution,
		const AutoMaskOptions &opt, MultidimArray<RFLOAT> &mask,
		AutoMaskReport *report = NULL, int n_threads = 1);

#endif /* AUTO_MASK_H_ */
