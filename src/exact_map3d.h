/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef EXACT_MAP3D_H
#define EXACT_MAP3D_H

/* Exact resampling of a real 3D map (FINUFFT builds only): the value of the map's band-limited interpolant (its
 * periodic Fourier series) at any position, where trilinear interpolation damps high frequencies by a different
 * amount at each fractional offset. Averaging copies of a map that were each resampled at different offsets
 * (helical or local symmetry) then loses signal near Nyquist with trilinear, not with this.
 *
 * A type-2 NUFFT with a persistent grid: the map's transform, divided by the kernel's, on a 2x grid (one inverse FFT),
 * then each value is a w x w x w weighted sum of that grid around the position (exponential-of-semicircle kernel).
 * Tolerance about 10^-(w-1). Memory: about 32 bytes per voxel of the map.
 *
 * RELION_MAP_INTERPOLATION=nufft asks for it where maps are rotated or shifted (helical and local symmetry, class
 * alignment and centring); see ExactMap3D::requested().
 */

#ifdef RELION_USE_FINUFFT

#include <string>
#include <vector>

#include "src/matrix2d.h"
#include "src/multidim_array.h"

namespace exactmap
{
	// Fourier transform of the kernel exp(beta (sqrt(1 - (2t/w)^2) - 1)), |t| <= w/2, at angular frequency kappa
	double esKernelFT(double kappa, int w, double beta);
	// smallest even size >= n without prime factors above 7
	int smoothEvenSize(int n);
}

class ExactMap3D
{
public:
	ExactMap3D(const MultidimArray<RFLOAT> &map, int threads, int w = 5);

	// The value at logical coordinates (x, y, z), as A3D_ELEM(map, z, y, x) at whole voxels; the map is periodic, so
	// positions outside the box wrap around. Const and thread safe.
	RFLOAT value(RFLOAT x, RFLOAT y, RFLOAT z) const;

	// As applyGeometry(map, out, A, inv, DONT_WRAP) in transformations.h (0 where the source is outside the box), but
	// every value exact. With where, only voxels where *where > 0 are computed (0 elsewhere).
	void applyGeometry(MultidimArray<RFLOAT> &out, const Matrix2D<RFLOAT> &A, bool inv, int threads,
	                   const MultidimArray<RFLOAT> *where = NULL) const;

	// Bytes the grid of a map this size takes, and whether that fits in half the memory still available
	static size_t bytes(const MultidimArray<RFLOAT> &map);
	static bool fits(const MultidimArray<RFLOAT> &map);

	// RELION_MAP_INTERPOLATION=nufft (else linear, the default)
	static bool requested();

	// Requested and fits: the choice for map, with a one-time note on stdout when it does not fit
	static bool use(const MultidimArray<RFLOAT> &map, const std::string &where);

private:
	long nx, ny, nz, x0, y0, z0;
	int n1x, n1y, n1z, w;
	size_t row;           // floats per grid row (padded for the in-place FFT)
	double sx, sy, sz;    // grid points per voxel
	std::vector<float> grid, tab;
	static const int L = 2048;   // kernel table points per unit
};

#endif // RELION_USE_FINUFFT
#endif
