/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef FFTW_RECT_H
#define FFTW_RECT_H

/* Fourier-space helpers for rectangular images and cuboid volumes.
 *
 * RELION's FFTW helpers assume nx == ny (== nz). The ones here take the sizes
 * of every axis. For a square or cubic input they give exactly the same result
 * as the originals (the unit tests check this bit for bit).
 *
 * Shell index ("shell-unit scheme"): the optimiser keeps one nominal size
 * L = max(nx, ny, nz) and counts resolution shells in units of 1/(L * pixel).
 * An axis of length n therefore has its Fourier index multiplied by L/n before
 * the radius is taken, so shells stay tied to physical frequency.
 */

#include "src/multidim_array.h"
#include "src/complex.h"
#include "src/macros.h"
#include "src/tabfuncs.h"

// Like FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM, but the sign of ky and kz is decided
// by the size of their own axis instead of by the x size.
#define FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM_RECT(V) \
	for (long int k = 0, kp = 0; k<ZSIZE(V); k++, kp = (k <= ZSIZE(V)/2) ? k : k - ZSIZE(V)) \
		for (long int i = 0, ip = 0 ; i<YSIZE(V); i++, ip = (i <= YSIZE(V)/2) ? i : i - YSIZE(V)) \
			for (long int j = 0, jp = 0; j<XSIZE(V); j++, jp = j)

/// Window (crop or zero-pad) an FFTW half transform to real-space size
/// new_nx x new_ny (x new_nz for a 3D transform). The input real-space x size
/// is taken to be even, i.e. 2*(XSIZE(in)-1).
/// When any axis grows, components outside the ellipse (ellipsoid) inscribed in the
/// input are left zero, as the square version does for its corners.
void windowFourierTransformRect(const MultidimArray<Complex> &in, MultidimArray<Complex> &out,
                                long int new_nx, long int new_ny, long int new_nz = 1);

/// Resample a 2D image to new_nx x new_ny pixels by cropping or padding its Fourier
/// transform (as resizeMap does for square images). Both sizes must be even.
void resizeMapRect(MultidimArray<RFLOAT> &img, long int new_nx, long int new_ny);

/// Shift through phase ramps; orix/oriy/oriz are the real-space sizes used to
/// normalise the shifts (the shifts are in the same unit).
void shiftImageInFourierTransformRect(const MultidimArray<Complex> &in, MultidimArray<Complex> &out,
                                      RFLOAT orix, RFLOAT oriy, RFLOAT oriz,
                                      RFLOAT xshift, RFLOAT yshift, RFLOAT zshift = 0.);

/// Shell index of Fourier pixel (kp, ip, jp) for an image of real-space size
/// nx x ny x nz and nominal size L. For nx == ny == nz == L it is exactly
/// ROUND(sqrt(kp^2 + ip^2 + jp^2)).
inline long int shellIndexRect(long int kp, long int ip, long int jp,
                               long int nx, long int ny, long int nz, long int L)
{
	if (nx == L && ny == L && nz == L)
		return ROUND(std::sqrt((double)(kp*kp + ip*ip + jp*jp)));
	const double sx = (double)L / nx, sy = (double)L / ny, sz = (double)L / nz;
	const double x = jp * sx, y = ip * sy, z = kp * sz;
	return ROUND(std::sqrt(x*x + y*y + z*z));
}

/// Per-axis size of a cropped Fourier image. s is the nominal cropped size
/// (in units of the nominal box L), n the real-space size of the axis.
/// A full-size axis stays whole; otherwise the size is kept even and not larger than n.
inline long int axisSize(long int s, long int n, long int L)
{
	if (n == L)
		return s;
	return XMIPP_MIN(n, 2 * (long int)CEIL(0.5 * s * (double)n / L));
}

/// As shiftImageInFourierTransformWithTabSincos for a 2D rectangular image.
/// The output has real-space size new_nx x new_ny, which must equal that of the input.
void shiftImageInFourierTransformWithTabSincosRect(const MultidimArray<Complex> &in, MultidimArray<Complex> &out,
                                                   RFLOAT orix, RFLOAT oriy, long int new_nx, long int new_ny,
                                                   TabSine &tabsin, TabCosine &tabcos,
                                                   RFLOAT xshift, RFLOAT yshift);

/// Radially averaged (power) spectrum with nominal size L; the result has L/2+1 entries.
/// The image must be real-space nx x ny (x nz).
void getSpectrumRect(MultidimArray<RFLOAT> &Min, MultidimArray<RFLOAT> &spectrum, long int L,
                     int spectrum_type = 0 /* POWER_SPECTRUM */);

#endif
