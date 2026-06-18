/***************************************************************************
 *
 * Author: "Wen Jiang"
 * Pennsylvania State University
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

#ifndef SRC_SPATIAL_FREQUENCY_GRID_H_
#define SRC_SPATIAL_FREQUENCY_GRID_H_

#include <vector>

#include "src/macros.h"

template <typename T> class Matrix2D;

struct SpatialFrequencyGrid2D
{
    int size;
    int half_size;
    std::vector<RFLOAT> sample_x;
    std::vector<RFLOAT> sample_y;
    std::vector<RFLOAT> sample_weight;
    std::vector<RFLOAT> finufft_target_x;
    std::vector<RFLOAT> finufft_target_y;
    // Pre-computed bilinear interpolation coefficients for FFTW half-plane access
    // bx0/bx1: column indices at (x, x+1); by0/by1: row indices at (y, y+1)
    // btx/bty: fractional weights within the cell
    std::vector<int> bx0, bx1, by0, by1;
    std::vector<RFLOAT> btx, bty;
};

// Pre-compute bilinear interpolation coefficients for all samples in the grid.
// Must be called after all samples (including hybrid additions) have been added.
void computeBilinearCoeffs(SpatialFrequencyGrid2D& grid);

SpatialFrequencyGrid2D makeSignedS2CartesianGrid2D(int size, const Matrix2D<RFLOAT>* magMatrix = NULL,
    RFLOAT s2_step = -1.0);

// Compute the s2_step needed to achieve at least min_samples_per_oscillation
// samples per CTF oscillation at the given max defocus.
// Returns the required s2_step (≤ half), or half if no upsampling is needed.
// If min_samples_per_oscillation <= 0, returns half (feature disabled).
RFLOAT computeS2StepForCtfOversampling(int half, RFLOAT angpix,
                                RFLOAT max_defocus, int min_samples_per_oscillation, RFLOAT voltage);

void printS2CtfOversamplingStats(int half, RFLOAT angpix,
                                 RFLOAT dmin, RFLOAT dmean, RFLOAT dmedian, RFLOAT dmax,
                                 int min_samples_per_oscillation, RFLOAT voltage);

bool isMagAwareSignedS2HybridCellWithinCrossResolution(int y_index,
													   int x_index,
													   int size,
													   RFLOAT angpix,
													   RFLOAT s2_hybrid_cross_resolution,
													   const Matrix2D<RFLOAT>* magMatrix = NULL);

// Build an adaptive hybrid grid: full Cartesian base with s2-density fill.
// For each Cartesian cell [x,x+1] x [y,y+1], compute how many sub-intervals
// are needed so that the s^2 gap across the cell is <= delta_s2_step.
// If ns > 1, insert an (ns x ns) interior sub-grid of fill points.
// All sample weights are 1.0.
// If s2_step <= 0 or >= half, no fill is needed (returns pure Cartesian grid).
// Sentinel s2_step == -2.0: uniform ns=2 fill (all half-grid positions) instead of adaptive.
SpatialFrequencyGrid2D makeAdaptiveS2HybridGrid2D(int size, const Matrix2D<RFLOAT>* magMatrix = NULL,
                                                  RFLOAT s2_step = -1.0);

// Write diagnostic text file comparing s-grid and s2-grid sample positions
// for each of the given defocus values. Also writes a Python script to
// generate the PDF plot. The output_prefix is used for file naming:
//   <output_prefix>.txt  — raw data
//   <output_prefix>.py   — plotting script (run separately for PDF)
void writeS2GridDiagnostic(int size, RFLOAT angpix, RFLOAT voltage,
                            RFLOAT dmin, RFLOAT dmean, RFLOAT dmedian, RFLOAT dmax,
                            int min_samples_per_oscillation,
                            const std::string& output_prefix);

#endif /* SRC_SPATIAL_FREQUENCY_GRID_H_ */