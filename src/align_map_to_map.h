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

#ifndef ALIGN_MAP_TO_MAP_H_
#define ALIGN_MAP_TO_MAP_H_

#include "src/multidim_array.h"
#include "src/projector.h"
#include "src/euler.h"

/// Align vol_align (source) to vol_ref (target) by searching over
/// orientation and translation parameters. Modifies vol_align in-place and
/// returns the best transformation parameters.
///
/// @param vol_align   Map to be aligned (modified in-place, full-size)
/// @param vol_ref     Target reference map (unmodified)
/// @param nr_freedom  6 (C1: rot/tilt/psi + dx/dy/dz),
///                    2 (Cn>=2 or helical: rot + dz),
///                    0 (Dn/T/O/I: skip)
/// @param angpix      Pixel size in Angstrom
/// @param maxres      Maximum resolution (A) for comparison
/// @param search_range  Local search half-width in steps
/// @param search_step_rot  Angular step in degrees
/// @param search_step_trans Translational step in Angstrom
/// @param best_rot output
/// @param best_tilt output
/// @param best_psi output
/// @param best_dx output (Angstrom)
/// @param best_dy output (Angstrom)
/// @param best_dz output (Angstrom)
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
    RFLOAT &best_dz);

/// Apply the inverse of a found transformation to a per-particle orientation.
/// For C1 (nr_freedom==6): composes the inverse rotation into the old one,
/// subtracts the inverse translation.
/// For Cn/helical (nr_freedom==2): only adjusts rot (Z-rotation) and dz (Z-shift).
///
/// @param nr_freedom  6 (C1), 2 (Cn/helical)
/// @param drot  rotation (Z) that was applied to the reference
/// @param dtilt tilt (Y) rotation applied to the reference
/// @param dpsi  psi (X) rotation applied to the reference
/// @param ddx   X translation applied to the reference (Angstrom)
/// @param ddy   Y translation applied to the reference (Angstrom)
/// @param ddz   Z translation applied to the reference (Angstrom)
/// @param p_rot   particle rot (modified in-place)
/// @param p_tilt  particle tilt (modified in-place)
/// @param p_psi   particle psi (modified in-place)
/// @param p_dx    particle X offset (modified in-place)
/// @param p_dy    particle Y offset (modified in-place)
/// @param p_dz    particle Z offset (modified in-place)
void applyInverseOrientationAdjustment(
    int nr_freedom,
    RFLOAT drot, RFLOAT dtilt, RFLOAT dpsi,
    RFLOAT ddx, RFLOAT ddy, RFLOAT ddz,
    RFLOAT &p_rot, RFLOAT &p_tilt, RFLOAT &p_psi,
    RFLOAT &p_dx, RFLOAT &p_dy, RFLOAT &p_dz);

#endif /* ALIGN_MAP_TO_MAP_H_ */
