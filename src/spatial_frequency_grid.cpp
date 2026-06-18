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

#include "src/spatial_frequency_grid.h"

#include "src/filename.h"
#include "src/matrix2d.h"

#include <cmath>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>

#include "src/error.h"

namespace
{

inline RFLOAT signedS2AxisCoordinate(RFLOAT index, int half_size)
{
	return index * (RFLOAT)half_size;
}

inline void applyMagMatrixToFrequencyCoordinates(RFLOAT& x,
	                                             RFLOAT& y,
	                                             const Matrix2D<RFLOAT>* magMatrix)
{
	if (magMatrix == NULL)
	{
		return;
	}

	const RFLOAT x0 = x;
	const RFLOAT y0 = y;
	x = (*magMatrix)(0,0) * x0 + (*magMatrix)(0,1) * y0;
	y = (*magMatrix)(1,0) * x0 + (*magMatrix)(1,1) * y0;
}

}

SpatialFrequencyGrid2D makeSignedS2CartesianGrid2D(int size, const Matrix2D<RFLOAT>* magMatrix,
    RFLOAT s2_step)
{
	if (size < 2)
	{
		REPORT_ERROR("makeSignedS2CartesianGrid2D: size must be at least 2");
	}

	SpatialFrequencyGrid2D grid;
	grid.size = size;
	grid.half_size = size / 2;

	const int half = grid.half_size;
	const RFLOAT delta_s2_step = (s2_step > 0.0) ? s2_step : (RFLOAT)half;
	const RFLOAT ratio = (RFLOAT)half / delta_s2_step;
	const int n_coord = (ratio > 1.0) ? (int)ceil((RFLOAT)half * ratio) : half;
	const int effective_sh = n_coord + 1;
	const int effective_size = 2 * n_coord;
	const int threshold = n_coord + 1;

	Matrix2D<RFLOAT> invMagStorage;
	const Matrix2D<RFLOAT>* invMag = NULL;
	if (magMatrix != NULL)
	{
		invMagStorage = magMatrix->inv();
		invMag = &invMagStorage;
	}

	grid.sample_x.reserve(effective_size * effective_sh);
	grid.sample_y.reserve(effective_size * effective_sh);
	grid.sample_weight.reserve(effective_size * effective_sh);
	const RFLOAT mag_weight = (magMatrix != NULL && std::fabs(magMatrix->det()) > XMIPP_EQUAL_ACCURACY)
		? (RFLOAT)1.0 / std::fabs(magMatrix->det())
		: (RFLOAT)1.0;

	for (int i = 0; i < effective_size; i++)
	{
		const int ip = (i < threshold) ? i : i - effective_size;
		const int first_x = (i < threshold) ? 0 : 1;
		const RFLOAT sy2 = (RFLOAT)ip * delta_s2_step;

		for (int x = first_x; x <= n_coord; x++)
		{
			const RFLOAT sx2 = (RFLOAT)x * delta_s2_step;
			const RFLOAT s2_radius = hypot(sx2, sy2);

			if (s2_radius < XMIPP_EQUAL_ACCURACY)
			{
				grid.sample_x.push_back(0.0);
				grid.sample_y.push_back(0.0);
				grid.sample_weight.push_back(0.0);
				continue;
			}

			const RFLOAT inv_s = (RFLOAT)1.0 / sqrt(s2_radius);
			RFLOAT sx = sx2 * inv_s;
			RFLOAT sy = sy2 * inv_s;
			applyMagMatrixToFrequencyCoordinates(sx, sy, invMag);
			grid.sample_x.push_back(sx);
			grid.sample_y.push_back(sy);

			// Uniform cells in the signed s^2 plane map to nonuniform cells in
			// physical Fourier space with Jacobian dA_s / dA_s2 = 1 / (2 * s^2).
			grid.sample_weight.push_back(
			    ((delta_s2_step * delta_s2_step) / ((RFLOAT)2.0 * s2_radius)) * mag_weight);
		}
	}

	return grid;
}

RFLOAT computeS2StepForCtfOversampling(int half, RFLOAT angpix,
    RFLOAT max_defocus, int min_samples_per_oscillation, RFLOAT voltage)
{
    if (min_samples_per_oscillation <= 0 || max_defocus <= 0.0 || angpix <= 0.0 || half <= 0)
        return (RFLOAT)half;

    const RFLOAT kV = voltage * 1000.0;
    const RFLOAT wavelength = (RFLOAT)12.2643247 / sqrt(kV * (1.0 + kV * 0.978466e-6));

    // Current on-axis s² spacing in actual frequency² units (1/Å²):
    // Δ|s|² = delta_s2_step / (2 * half * angpix)²
    // For the default grid where delta_s2_step = half: Δ|s|² = 1 / (4 * half * angpix²)
    const RFLOAT current_s2_step_actual = (RFLOAT)1.0 / ((RFLOAT)4.0 * (RFLOAT)half * angpix * angpix);

    // CTF oscillation period in |s|² (1/Å²): P = 2 / (λ * Δf)
    const RFLOAT period = (RFLOAT)2.0 / (wavelength * max_defocus);

    // Required s² spacing for N samples per oscillation
    const RFLOAT required_s2_step_actual = period / (RFLOAT)min_samples_per_oscillation;

    if (required_s2_step_actual >= current_s2_step_actual)
        return (RFLOAT)half; // No upsampling needed

    // Upsample factor = current / required
    const RFLOAT ratio = current_s2_step_actual / required_s2_step_actual;
    const RFLOAT s2_step = (RFLOAT)half / ratio;

    return s2_step;
}

void printS2CtfOversamplingStats(int half, RFLOAT angpix,
                                 RFLOAT dmin, RFLOAT dmean, RFLOAT dmedian, RFLOAT dmax,
                                 int min_samples_per_oscillation, RFLOAT voltage)
{
  const RFLOAT kV = voltage * 1000.0;
  const RFLOAT wavelength = (RFLOAT)12.2643247 / sqrt(kV * (1.0 + kV * 0.978466e-6));
  const RFLOAT current_s2_step_actual = (RFLOAT)1.0 / ((RFLOAT)4.0 * (RFLOAT)half * angpix * angpix);

 auto samplesPerOsc = [&](RFLOAT defocus) -> RFLOAT {
 RFLOAT period = (RFLOAT)2.0 / (wavelength * defocus);
 return period / current_s2_step_actual;
 };

	if (min_samples_per_oscillation > 0)
	{
		std::cout << " + s2 CTF oversampling: target=" << min_samples_per_oscillation
		          << " samples/osc" << std::endl;
	}

	std::cout << " Defocus statistics:"
	          << " min=" << (long int)dmin << " (" << std::fixed << std::setprecision(1) << samplesPerOsc(dmin) << " samples/osc)"
	          << " mean=" << (long int)dmean << " (" << samplesPerOsc(dmean) << " samples/osc)"
	          << " median=" << (long int)dmedian << " (" << samplesPerOsc(dmedian) << " samples/osc)"
	          << " max=" << (long int)dmax << " (" << samplesPerOsc(dmax) << " samples/osc)"
	          << " Angstrom" << std::endl;

	if (min_samples_per_oscillation > 0)
	{
		auto s2StepForDefocus = [&](RFLOAT defocus) -> RFLOAT {
			RFLOAT period = (RFLOAT)2.0 / (wavelength * defocus);
			RFLOAT required = period / (RFLOAT)min_samples_per_oscillation;
			if (required >= current_s2_step_actual) return (RFLOAT)half;
			return (RFLOAT)half / (current_s2_step_actual / required);
		};

		RFLOAT steps_min = s2StepForDefocus(dmin);
		RFLOAT steps_max = s2StepForDefocus(dmax);
		RFLOAT steps_median = s2StepForDefocus(dmedian);
		RFLOAT steps_mean = s2StepForDefocus(dmean);
		auto upsampFactor = [&](RFLOAT step) -> RFLOAT {
			return (step < (RFLOAT)half) ? (RFLOAT)half / step : (RFLOAT)1.0;
		};
		std::cout << " Oversampled s2 samples (orig=" << half << "):"
		          << " min_defocus=" << steps_min << " (" << upsampFactor(steps_min) << "x)"
		          << " max_defocus=" << steps_max << " (" << upsampFactor(steps_max) << "x)"
		          << " median_defocus=" << steps_median << " (" << upsampFactor(steps_median) << "x)"
		          << " mean_defocus=" << steps_mean << " (" << upsampFactor(steps_mean) << "x)"
		          << std::endl;
	}

	std::cout << std::endl;
}

bool isMagAwareSignedS2HybridCellWithinCrossResolution(int y_index,
	                                                   int x_index,
	                                                   int size,
	                                                   RFLOAT angpix,
	                                                   RFLOAT s2_hybrid_cross_resolution,
	                                                   const Matrix2D<RFLOAT>* magMatrix)
{
	if (angpix <= 0.0)
	{
		return false;
	}

	const RFLOAT cross = (s2_hybrid_cross_resolution > 0.0)
		? s2_hybrid_cross_resolution
		: (RFLOAT)(2.0 * sqrt(2.0) * angpix); // half/√2 = 0.707 Nyquist
	const RFLOAT s_cross = (RFLOAT)1.0 / cross;

	const int sh = size / 2 + 1;
	const RFLOAT x = (RFLOAT)x_index / ((RFLOAT)size * angpix);
	const RFLOAT y = ((y_index < sh) ? (RFLOAT)y_index : (RFLOAT)(y_index - size)) / ((RFLOAT)size * angpix);

	RFLOAT mx = x;
	RFLOAT my = y;
	applyMagMatrixToFrequencyCoordinates(mx, my, magMatrix);

	return hypot(mx, my) <= s_cross;
}

SpatialFrequencyGrid2D makeAdaptiveS2HybridGrid2D(int size, const Matrix2D<RFLOAT>* magMatrix,
                                                  RFLOAT s2_step)
{
    if (size < 2)
    {
        REPORT_ERROR("makeAdaptiveS2HybridGrid2D: size must be at least 2");
    }

    SpatialFrequencyGrid2D grid;
    grid.size = size;
    grid.half_size = size / 2;

    const int half = grid.half_size;
    const int sh = half + 1;
    const bool uniform_fill = (s2_step == -2.0);
    const RFLOAT delta_s2_step = uniform_fill ? 1.0 : ((s2_step > 0.0) ? s2_step : (RFLOAT)half);
    const bool need_fill = uniform_fill || (delta_s2_step > 0.0 && delta_s2_step < (RFLOAT)half);

    Matrix2D<RFLOAT> invMagStorage;
    const Matrix2D<RFLOAT>* invMag = NULL;
    if (magMatrix != NULL)
    {
        invMagStorage = magMatrix->inv();
        invMag = &invMagStorage;
    }

    // Phase 1: Build the full Cartesian grid (FFTW half-plane convention).
    {
        long int n_cart = 0;
        for (int i = 0; i < size; i++)
        {
            const int y = (i < sh) ? i : i - size;
            const int first_x = (i < sh) ? 0 : 1;
            for (int x = first_x; x < sh; x++)
                n_cart++;
        }
        grid.sample_x.reserve(n_cart * (need_fill ? 4 : 1));
        grid.sample_y.reserve(n_cart * (need_fill ? 4 : 1));
        grid.sample_weight.reserve(n_cart * (need_fill ? 4 : 1));
    }

    for (int i = 0; i < size; i++)
    {
        const int y = (i < sh) ? i : i - size;
        const int first_x = (i < sh) ? 0 : 1;
        for (int x = first_x; x < sh; x++)
        {
            RFLOAT sx = (RFLOAT)x;
            RFLOAT sy = (RFLOAT)y;
            applyMagMatrixToFrequencyCoordinates(sx, sy, invMag);
            if (sx * sx + sy * sy > (RFLOAT)(half * half))
                continue;
            grid.sample_x.push_back(sx);
            grid.sample_y.push_back(sy);
            grid.sample_weight.push_back(1.0);
        }
    }

    // Phase 2: Insert s2-density fill points inside each Cartesian cell.
    //
    // For cell [x0, x0+1] x [y0, y0+1], the s^2 values at the four corners are
    // s^2 = x^2 + y^2 (in pixel-frequency^2 units).
    //
    // The maximum s^2 change across the cell determines the fill factor:
    // ns = ceil(gap / delta_s2_step), gap = max(s^2) - min(s^2) over corners.
    //
    // If ns > 1, we subdivide the cell into an (ns x ns) sub-grid and insert
    // fill points at (x0 + k/ns, y0 + j/ns) for k,j in [0..ns].
    //
    // Skip rules (non-overlapping partition of the global sub-grid):
    //   - j == ns: top edge belongs to the cell above (its j=0 row)
    //   - k == ns: right edge belongs to the cell to the right (its k=0 col)
    //   - j == 0 && k == 0: bottom-left corner is a Cartesian grid point
    // This ensures each fill point is added exactly once and no Cartesian
    // points are duplicated.
    //
    // Fill weight: each fill point gets weight 1/(ns^2) so that the total
    // fill weight per cell is approximately 1.0, matching the weight of a
    // single Cartesian grid point. Without this normalization, cells with
    // denser fill (larger ns) would be over-weighted in reconstruction.
    if (need_fill)
    {
        for (int y0 = -half; y0 < half; y0++)
        {
            const int y1 = y0 + 1;
            const int first_x = (y0 >= 0) ? 0 : 1;

            for (int x0 = first_x; x0 < sh - 1; x0++)
            {
                const int x1 = x0 + 1;

                const RFLOAT s2_00 = (RFLOAT)(x0 * x0) + (RFLOAT)(y0 * y0);
                const RFLOAT s2_10 = (RFLOAT)(x1 * x1) + (RFLOAT)(y0 * y0);
                const RFLOAT s2_01 = (RFLOAT)(x0 * x0) + (RFLOAT)(y1 * y1);
                const RFLOAT s2_11 = (RFLOAT)(x1 * x1) + (RFLOAT)(y1 * y1);

                const RFLOAT s2_min = std::min(s2_00, std::min(s2_10, std::min(s2_01, s2_11)));
                const RFLOAT s2_max = std::max(s2_00, std::max(s2_10, std::max(s2_01, s2_11)));
                const RFLOAT gap = s2_max - s2_min;

                if (gap <= delta_s2_step)
                    continue;

                const int ns = uniform_fill ? 2 : (int)ceil(gap / delta_s2_step);
                if (ns <= 1)
                    continue;

            const RFLOAT inv_ns = (RFLOAT)1.0 / (RFLOAT)ns;
            const RFLOAT fill_weight = inv_ns * inv_ns;
        for (int j = 0; j <= ns; j++)
        {
            const RFLOAT fy = (RFLOAT)j * inv_ns;
            for (int k = 0; k <= ns; k++)
            {
                if (j == ns || k == ns || (j == 0 && k == 0))
                    continue;

                    const RFLOAT fx = (RFLOAT)k * inv_ns;
                    RFLOAT sx = (RFLOAT)x0 + fx;
                    RFLOAT sy = (RFLOAT)y0 + fy;
                    applyMagMatrixToFrequencyCoordinates(sx, sy, invMag);
                    if (sx * sx + sy * sy > (RFLOAT)(half * half))
                        continue;
                    grid.sample_x.push_back(sx);
                    grid.sample_y.push_back(sy);
                        grid.sample_weight.push_back(fill_weight);
                    }
                }
            }
        }
    }

    return grid;
}

void computeBilinearCoeffs(SpatialFrequencyGrid2D& grid)
{
	const int size = grid.size;
	const int sh = size / 2 + 1;
	const int n = grid.sample_x.size();
	grid.bx0.resize(n);
	grid.bx1.resize(n);
	grid.by0.resize(n);
	grid.by1.resize(n);
	grid.btx.resize(n);
	grid.bty.resize(n);

	// Set FINUFFT type-2 target coordinates (angular frequency units).
	// sample_x/y are in pixel-frequency units; scale by 2*pi/size.
	grid.finufft_target_x.resize(n);
	grid.finufft_target_y.resize(n);
	const RFLOAT scale = (RFLOAT)(2.0 * PI) / size;

	for (int i = 0; i < n; i++)
	{
		RFLOAT xc = grid.sample_x[i];
		if (xc < 0.0) xc = 0.0;
		if (xc > (RFLOAT)(sh - 1)) xc = (RFLOAT)(sh - 1);

		RFLOAT yw = grid.sample_y[i];
		while (yw < 0.0) yw += (RFLOAT)size;
		while (yw >= (RFLOAT)size) yw -= (RFLOAT)size;

		const int x0 = (int)FLOOR(xc);
		const int x1 = (x0 + 1 < sh) ? x0 + 1 : x0;
		const int y0 = (int)FLOOR(yw);
		const int y1 = (y0 + 1 < size) ? y0 + 1 : 0;

		grid.bx0[i] = x0;
		grid.bx1[i] = x1;
		grid.by0[i] = y0;
		grid.by1[i] = y1;
		grid.btx[i] = xc - x0;
		grid.bty[i] = yw - y0;

		// FINUFFT target coordinates in radians (angular frequency)
        grid.finufft_target_x[i] = scale * grid.sample_x[i];
        grid.finufft_target_y[i] = scale * grid.sample_y[i];
    }
}

void writeS2GridDiagnostic(int size, RFLOAT angpix, RFLOAT voltage,
                            RFLOAT dmin, RFLOAT dmean, RFLOAT dmedian, RFLOAT dmax,
                            int min_samples_per_oscillation,
                            const std::string& output_prefix)
{
    const int half = size / 2;

    struct DefocusLevel { std::string label; RFLOAT defocus; };
    DefocusLevel levels[] = {
        {"min", dmin}, {"mean", dmean}, {"median", dmedian}, {"max", dmax}
    };

    std::string txt_path = output_prefix + "_radial.txt";
    std::string zoom_path = output_prefix + "_zoom.txt";
    std::string py_path = output_prefix + ".py";

    size_t slash_pos = output_prefix.rfind('/');
    if (slash_pos != std::string::npos)
    {
        FileName out_dir = output_prefix.substr(0, slash_pos);
        mktree(out_dir);
    }

    // --- Radial density data ---
    {
        std::ofstream out(txt_path);
        if (!out.is_open())
        {
            std::cerr << "WARNING: writeS2GridDiagnostic: could not open " << txt_path << std::endl;
            return;
        }

    out << "# S2 grid radial density diagnostic: box=" << size << " angpix=" << angpix
        << " voltage=" << voltage << " min_samples=" << min_samples_per_oscillation << "\n";
    out << "# Columns: defocus_label defocus s2_step n_cartesian n_hybrid\n";
    out << "# Then per-interval rows: x_interval_center n_cart_on_axis n_fill_on_axis\n";

        for (int li = 0; li < 4; li++)
        {
            RFLOAT defocus = levels[li].defocus;
            const std::string& label = levels[li].label;

    RFLOAT s2_step = computeS2StepForCtfOversampling(half, angpix, defocus,
                    min_samples_per_oscillation, voltage);
    SpatialFrequencyGrid2D grid_hybrid = makeAdaptiveS2HybridGrid2D(size, NULL, s2_step);
    SpatialFrequencyGrid2D grid_cart = makeAdaptiveS2HybridGrid2D(size, NULL, (RFLOAT)half);

    long int n_cart = grid_cart.sample_x.size();
    long int n_hybrid = grid_hybrid.sample_x.size();

    out << label << " " << defocus << " " << s2_step << " " << n_cart << " " << n_hybrid << "\n";

    std::vector<long int> cart_count(half, 0);
    std::vector<long int> fill_count(half, 0);

    for (long int i = 0; i < n_cart; i++)
    {
        RFLOAT sy = grid_cart.sample_y[i];
        if (sy != (RFLOAT)0.0) continue;
        int x_bin = (int)floor(grid_cart.sample_x[i]);
        if (x_bin < 0 || x_bin >= half) continue;
        cart_count[x_bin]++;
    }

    for (long int i = 0; i < n_hybrid; i++)
    {
        RFLOAT sy = grid_hybrid.sample_y[i];
        if (sy != (RFLOAT)0.0) continue;
        int x_bin = (int)floor(grid_hybrid.sample_x[i]);
        if (x_bin < 0 || x_bin >= half) continue;
        fill_count[x_bin]++;
    }

    for (int b = 0; b < half; b++)
        fill_count[b] = std::max(0L, fill_count[b] - cart_count[b]);

    for (int b = 0; b < half; b++)
    {
        out << (b + 0.5) << " " << cart_count[b] << " " << fill_count[b] << "\n";
    }

            out << "# ---\n";
        }
        out.close();
    }

    // --- 2D zoom region data ---
    {
        std::ofstream out(zoom_path);
        if (!out.is_open())
        {
            std::cerr << "WARNING: writeS2GridDiagnostic: could not open " << zoom_path << std::endl;
            return;
        }

        const int zoom_lo = std::max(half / 4, 1);
        const int zoom_hi = zoom_lo + 4;
        const int zoom_y_lo = 0;
        const int zoom_y_hi = zoom_hi - zoom_lo;

        out << "# S2 grid 2D zoom diagnostic: box=" << size << "\n";
        out << "# Zoom region: x=[" << zoom_lo << "," << zoom_hi << "] y=[" << zoom_y_lo << "," << zoom_y_hi << "]\n";
        out << "# Columns: defocus_label sample_x sample_y is_fill\n";

        for (int li = 0; li < 4; li++)
        {
            RFLOAT defocus = levels[li].defocus;
            const std::string& label = levels[li].label;

            RFLOAT s2_step = computeS2StepForCtfOversampling(half, angpix, defocus,
                                                              min_samples_per_oscillation, voltage);

            SpatialFrequencyGrid2D grid_cart = makeAdaptiveS2HybridGrid2D(size, NULL, (RFLOAT)half);
            SpatialFrequencyGrid2D grid_hybrid = makeAdaptiveS2HybridGrid2D(size, NULL, s2_step);

            long int n_cart = grid_cart.sample_x.size();

            // Write only points in the zoom region from the hybrid grid
            for (long int i = 0; i < (long int)grid_hybrid.sample_x.size(); i++)
            {
                RFLOAT sx = grid_hybrid.sample_x[i];
                RFLOAT sy = grid_hybrid.sample_y[i];
                if (sx >= zoom_lo && sx <= zoom_hi && sy >= zoom_y_lo && sy <= zoom_y_hi)
                {
                    int is_fill = (i >= n_cart) ? 1 : 0;
                    out << label << " " << std::fixed << std::setprecision(4)
                        << sx << " " << sy << " " << is_fill << "\n";
                }
            }
            out << "# ---\n";
        }
        out.close();
    }

    // --- Python plot script ---
    {
        std::ofstream py(py_path);
        if (!py.is_open())
        {
            std::cerr << "WARNING: writeS2GridDiagnostic: could not open " << py_path << std::endl;
            return;
        }

py << "#!/usr/bin/env python3\n"
"# Usage: python3 " << py_path << "\n"
"# Output: " << output_prefix << ".pdf\n\n"
"import numpy as np\n"
"import matplotlib\n"
"matplotlib.use('Agg')\n"
"import matplotlib.pyplot as plt\n"
"from matplotlib.backends.backend_pdf import PdfPages\n\n"
"radial_file = '" << txt_path << "'\n"
"zoom_file = '" << zoom_path << "'\n"
"outfile = '" << output_prefix << ".pdf'\n\n"
"# Parse radial data\n"
"defocus_labels = []\n"
"defocus_info = {}\n"
"shells = {}\n"
"current_label = None\n"
"with open(radial_file) as f:\n"
"    for line in f:\n"
"        if line.startswith('#'): continue\n"
"        parts = line.split()\n"
"        if len(parts) == 5: # header line: label defocus s2_step n_cart n_hybrid\n"
"            current_label = parts[0]\n"
"            if current_label not in defocus_labels:\n"
"                defocus_labels.append(current_label)\n"
"                defocus_info[current_label] = {\n"
"                    'defocus': float(parts[1]),\n"
"                    's2_step': float(parts[2]),\n"
"                    'n_cart': int(parts[3]),\n"
"                    'n_hybrid': int(parts[4]),\n"
"                }\n"
"                shells[current_label] = {'x_center': [], 'n_cart': [], 'n_fill': []}\n"
"        elif len(parts) == 3 and current_label is not None:\n"
"            shells[current_label]['x_center'].append(float(parts[0]))\n"
"            shells[current_label]['n_cart'].append(int(parts[1]))\n"
"            shells[current_label]['n_fill'].append(int(parts[2]))\n\n"
"# Parse zoom data\n"
"zoom_data = {}\n"
"with open(zoom_file) as f:\n"
"    for line in f:\n"
"        if line.startswith('#'): continue\n"
"        parts = line.split()\n"
"        if len(parts) == 4:\n"
"            label = parts[0]\n"
"            if label not in zoom_data:\n"
"                zoom_data[label] = {'base_x': [], 'base_y': [], 'fill_x': [], 'fill_y': []}\n"
"            if parts[3] == '0':\n"
"                zoom_data[label]['base_x'].append(float(parts[1]))\n"
"                zoom_data[label]['base_y'].append(float(parts[2]))\n"
"            else:\n"
"                zoom_data[label]['fill_x'].append(float(parts[1]))\n"
"                zoom_data[label]['fill_y'].append(float(parts[2]))\n\n"
"with PdfPages(outfile) as pdf:\n"
"    for dlabel in defocus_labels:\n"
"        info = defocus_info[dlabel]\n"
"        sh = shells[dlabel]\n"
"        x_arr = np.array(sh['x_center'])\n"
"        n_cart_arr = np.array(sh['n_cart'])\n"
"        n_fill_arr = np.array(sh['n_fill'])\n"
"        n_total_arr = n_cart_arr + n_fill_arr\n\n"
"        fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n"
"        fig.suptitle(f'Defocus {dlabel}={info[\"defocus\"]:.0f} A '\n"
"                     f's2_step={info[\"s2_step\"]:.1f} '\n"
"                     f'N_cart={info[\"n_cart\"]} N_hybrid={info[\"n_hybrid\"]}',\n"
"                     fontsize=12)\n\n"
"        # Panel 1: sample density per x-interval\n"
"        ax = axes[0]\n"
"        ax.bar(x_arr - 0.3, n_cart_arr, width=0.3, label='Cartesian base', color='blue', alpha=0.7)\n"
"        ax.bar(x_arr, n_fill_arr, width=0.3, label='s2 fill', color='red', alpha=0.7)\n"
"        ax.set_xlabel('x (pixel frequency interval center)')\n"
"        ax.set_ylabel('Number of samples')\n"
"        ax.set_title('Sample density per grid interval')\n"
"        ax.legend(fontsize=9)\n\n"
"        # Panel 2: 2D scatter of zoomed region\n"
"        ax = axes[1]\n"
"        zd = zoom_data.get(dlabel, None)\n"
"        if zd is not None:\n"
"            if zd['base_x']:\n"
"                ax.scatter(zd['base_x'], zd['base_y'],\n"
"                           c='blue', s=30, marker='s', label='Cartesian base', zorder=3)\n"
"            if zd['fill_x']:\n"
"                ax.scatter(zd['fill_x'], zd['fill_y'],\n"
"                           c='red', s=12, marker='.', label='s2 fill', zorder=2)\n"
"        ax.set_xlabel('sample_x (pixel freq)')\n"
"        ax.set_ylabel('sample_y (pixel freq)')\n"
"        ax.set_title('2D zoom (showing sub-grid structure)')\n"
"        ax.legend(fontsize=9)\n"
"        ax.set_aspect('equal')\n\n"
"        plt.tight_layout()\n"
"        pdf.savefig(fig)\n"
"        plt.close(fig)\n\n"
"print(f'Wrote {outfile}')\n";
        py.close();
    }

    std::cout << " + Wrote s2 grid diagnostic: " << txt_path << " + " << zoom_path << std::endl;
    std::cout << " + Wrote s2 grid plot script: " << py_path << std::endl;
    std::cout << " + To generate PDF: python3 " << py_path << std::endl;
}
