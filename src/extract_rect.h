/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef EXTRACT_RECT_H
#define EXTRACT_RECT_H

/* Cutting rectangular (and rotated rectangular) particles out of a micrograph.
 *
 * relion_preprocess and the virtual particle reader both call the functions
 * here, with different ways of reading micrograph pixels (a loaded image, or a
 * memory-mapped file), so that the two give identical pixels. The pixel reader
 * is a template argument: any callable `at(y, x)` returning a number.
 *
 * Pixels outside the micrograph take the value of the nearest pixel inside it,
 * as in square extraction.
 *
 * Rotation convention: a box turned by `angle` (degrees) has its horizontal
 * axis along the direction (cos angle, sin angle) in micrograph coordinates
 * (x to the right, y down); see resample_rotate.h. To lay a helical tube with
 * psi prior `psi` horizontal, turn the box by -psi.
 */

#include <algorithm>
#include <cmath>

#include "src/macros.h"
#include "src/multidim_array.h"
#include "src/resample_rotate.h"

namespace extractrect {

/// Copy the w x h window of the micrograph whose top-left pixel is (x0, y0),
/// repeating edge pixels outside the micrograph.
template <class Reader>
void cutClamped(const Reader& at, long mic_nx, long mic_ny,
                long x0, long y0, long w, long h, MultidimArray<RFLOAT>& out)
{
	out.resize(h, w);
	for (long i = 0; i < h; i++)
	{
		const long y = std::min(std::max(y0 + i, 0L), mic_ny - 1);
		for (long j = 0; j < w; j++)
		{
			const long x = std::min(std::max(x0 + j, 0L), mic_nx - 1);
			DIRECT_A2D_ELEM(out, i, j) = at(y, x);
		}
	}
}

/// The nx x ny box whose pixel (nx/2, ny/2) is micrograph pixel (xpos, ypos).
template <class Reader>
void cutBox(const Reader& at, long mic_nx, long mic_ny, long xpos, long ypos,
            int nx, int ny, MultidimArray<RFLOAT>& out)
{
	cutClamped(at, mic_nx, mic_ny, xpos + FIRST_XMIPP_INDEX(nx), ypos + FIRST_XMIPP_INDEX(ny), nx, ny, out);
}

/// Pixels needed around the box on each side for each resampling method (NUFFT
/// treats its source as periodic, so it gets room for the wrap-around to fade).
inline int resampleMargin(ResampleMethod method)
{
	return method == RESAMPLE_NUFFT ? 32 : 3;
}

/// The nx x ny box centred on the exact position (cx, cy), turned by `angle`.
template <class Reader>
void cutRotatedBox(const Reader& at, long mic_nx, long mic_ny, double cx, double cy,
                   int nx, int ny, double angle, ResampleMethod method, MultidimArray<RFLOAT>& out)
{
	const double a = DEG2RAD(angle);
	const double ca = std::fabs(std::cos(a)), sa = std::fabs(std::sin(a));
	const double hw = ca * nx / 2.0 + sa * ny / 2.0;
	const double hh = sa * nx / 2.0 + ca * ny / 2.0;
	const int m = resampleMargin(method);
	const long x0 = (long)std::floor(cx - hw) - m, x1 = (long)std::ceil(cx + hw) + m;
	const long y0 = (long)std::floor(cy - hh) - m, y1 = (long)std::ceil(cy + hh) + m;

	MultidimArray<RFLOAT> sub;
	cutClamped(at, mic_nx, mic_ny, x0, y0, x1 - x0 + 1, y1 - y0 + 1, sub);
	resampleRotatedRectangle(sub, out, nx, ny, cx - x0, cy - y0, angle, method);
}

/// Height of a rectangular box of size_x x size_y after it is rescaled to the width `scale`;
/// -1 unless that is an even whole number of pixels (the same factor must apply to both axes).
inline int rescaledHeight(int size_x, int size_y, int scale)
{
	const long long num = (long long)size_y * scale;
	if (num % size_x != 0) return -1;
	const long long h = num / size_x;
	return (h >= 2 && h % 2 == 0) ? (int)h : -1;
}

/// Defocus angle (degrees, in (-90, 90]) of the CTF as seen in a box turned by
/// `box_angle`: a direction at angle t in the micrograph is at t - box_angle in
/// the particle. The CTF repeats every 180 degrees.
inline double rotatedDefocusAngle(double defocus_angle, double box_angle)
{
	double a = std::fmod(defocus_angle - box_angle, 180.0);
	if (a <= -90.0) a += 180.0;
	if (a > 90.0) a -= 180.0;
	return a;
}

} // namespace extractrect

#endif
