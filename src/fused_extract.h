/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef FUSED_EXTRACT_H
#define FUSED_EXTRACT_H

/* Fused extraction and Fourier transform of a particle.
 *
 * Instead of cutting a particle out of its micrograph, resampling it onto the
 * output pixel grid and Fourier transforming it, the micrograph pixels inside
 * the particle's box are taken as they are, as nonuniform samples, and their
 * Fourier transform at the frequencies of the output grid is evaluated directly
 * with a type-1 NUFFT. Nothing is interpolated, so the output pixel size and
 * the box size are free choices.
 *
 * The result is the array that FourierTransform() followed by CenterFFTbySign()
 * gives for the particle cut at the output pixel size: FFTW half-plane layout
 * (ny rows, nx/2+1 columns), the origin at pixel (nx/2, ny/2) of the box,
 * normalised by 1/(nx*ny).
 *
 * Geometry is that of resample_rotate.h: box pixel (x, y) sits at offset
 * (dx, dy) = (x - nx/2, y - ny/2) output pixels from the centre, and samples
 * the micrograph at centre + step * (dx * (cos a, sin a) + dy * (-sin a, cos a)).
 * A micrograph pixel belongs to the box if its offset lies in
 * [-nx/2 - 1/(2 step), nx/2 - 1/(2 step)) x [-ny/2 - 1/(2 step), ny/2 - 1/(2 step))
 * output pixels, i.e. [-nx/2 step - 1/2, nx/2 step - 1/2) micrograph pixels
 * along the box axes: for step 1 the box's own pixels, and for a coarser output
 * grid the micrograph pixels that a Fourier crop of the finer box would keep. Pixels outside the
 * micrograph take the value of the nearest pixel inside it, as in extraction.
 *
 * Normalisation is done on the samples, with the background defined in box
 * coordinates, like normalise() does on a cut particle.
 *
 * Differences from the stored particle, by design: no interpolation damping,
 * and the box edge cuts micrograph pixels instead of resampled ones. A pixel
 * size finer than the micrograph's adds no information: frequencies above the
 * micrograph's Nyquist are set to zero rather than left aliased.
 */

#include <cmath>
#include <vector>

#include "src/complex.h"
#include "src/macros.h"
#include "src/multidim_array.h"

namespace fusedextract {

struct Geometry {
	int nx, ny;         ///< output box (even), output pixels
	double step;        ///< output pixel size / micrograph pixel size
	double cx, cy;      ///< centre in micrograph pixels
	double angle;       ///< degrees, as in resample_rotate.h
};

struct Normalisation {
	bool normalise;
	bool ramp;
	bool invert;
	bool helical;
	double bg_radius;      ///< output pixels (circle, or ellipse for a rectangular box)
	double helical_radius; ///< output pixels
	double psi;            ///< tube orientation in the box (degrees); 0 for a box turned to horizontal

	Normalisation() : normalise(false), ramp(false), invert(false), helical(false),
	                  bg_radius(0.), helical_radius(0.), psi(0.) {}
};

/// Micrograph pixels inside the box, with their offsets from the centre in output pixels.
struct Samples {
	std::vector<double> dx, dy, value;
};

/// Collect the samples; `at(y, x)` reads a micrograph pixel (inside the micrograph).
template <class Reader>
void gather(const Reader& at, long mic_nx, long mic_ny, const Geometry& g, Samples& s)
{
	const double a = DEG2RAD(g.angle);
	const double ca = std::cos(a), sa = std::sin(a);
	const double hw = g.step * (std::fabs(ca) * (g.nx / 2 + 1) + std::fabs(sa) * (g.ny / 2 + 1));
	const double hh = g.step * (std::fabs(sa) * (g.nx / 2 + 1) + std::fabs(ca) * (g.ny / 2 + 1));
	const long x0 = (long)std::floor(g.cx - hw), x1 = (long)std::ceil(g.cx + hw);
	const long y0 = (long)std::floor(g.cy - hh), y1 = (long)std::ceil(g.cy + hh);
	const double xlo = -g.nx / 2 - 0.5 / g.step, xhi = g.nx / 2 - 0.5 / g.step;
	const double ylo = -g.ny / 2 - 0.5 / g.step, yhi = g.ny / 2 - 0.5 / g.step;

	s.dx.clear(); s.dy.clear(); s.value.clear();
	s.dx.reserve((size_t)(g.nx * 1.5) * (size_t)(g.ny * 1.5));
	for (long sy = y0; sy <= y1; sy++)
	{
		const long ry = std::min(std::max(sy, 0L), mic_ny - 1);
		for (long sx = x0; sx <= x1; sx++)
		{
			const double ux = (sx - g.cx) / g.step, uy = (sy - g.cy) / g.step;
			const double dx = ux * ca + uy * sa;
			const double dy = -ux * sa + uy * ca;
			if (dx < xlo || dx >= xhi || dy < ylo || dy >= yhi) continue;
			s.dx.push_back(dx);
			s.dy.push_back(dy);
			s.value.push_back((double)at(ry, std::min(std::max(sx, 0L), mic_nx - 1)));
		}
	}
}

/// Ramp removal, background mean and standard deviation, and contrast inversion, in place.
void normaliseSamples(Samples& s, int nx, int ny, const Normalisation& n);

/// Type-1 NUFFT of the samples onto the output grid (see the top of this file).
/// Needs a build with RELION_USE_FINUFFT; otherwise it is an error.
void fourierTransform(const Samples& s, const Geometry& g, MultidimArray<Complex>& F2D);

} // namespace fusedextract

#endif
