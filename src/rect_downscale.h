/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef RECT_DOWNSCALE_H
#define RECT_DOWNSCALE_H

/* Downscaling an extracted box by a factor (relion_preprocess --downscale_factor
 * and the Extract job of the GUI).
 *
 * A box of size_x x size_y downscaled by `factor` becomes size_x/factor by
 * size_y/factor pixels. Both sides must be even whole numbers, otherwise the
 * two axes would get different pixel sizes. With g = gcd(size_x, size_y), the
 * boxes that keep the aspect ratio are 2m * size_x/g by 2m * size_y/g, so the
 * factors that work are g / (2m) for m = 1, 2, ...
 *
 * Header only and free of image code, so that the GUI can use it.
 */

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace extractrect {

/// A side divided by the factor may differ from a whole number by this much
/// (so that a typed factor such as 2.6667 for 768 -> 288 is accepted).
const double DOWNSCALE_PIXEL_TOLERANCE = 0.01;

/// The box of size_x x size_y downscaled by `factor` (2 halves each side), rounded
/// to whole pixels; true if both sides are then even whole numbers of at least 2.
inline bool downscaledBox(int size_x, int size_y, double factor, int &new_x, int &new_y)
{
	new_x = new_y = -1;
	if (!(factor > 0.) || size_x <= 0 || size_y <= 0)
		return false;
	const double fx = size_x / factor, fy = size_y / factor;
	new_x = (int)std::lround(fx);
	new_y = (int)std::lround(fy);
	return std::fabs(fx - new_x) <= DOWNSCALE_PIXEL_TOLERANCE && std::fabs(fy - new_y) <= DOWNSCALE_PIXEL_TOLERANCE
	       && new_x >= 2 && new_y >= 2 && new_x % 2 == 0 && new_y % 2 == 0;
}

inline long long greatestCommonDivisor(long long a, long long b)
{
	while (b != 0)
	{
		const long long t = a % b;
		a = b;
		b = t;
	}
	return a;
}

/// Empty if `factor` downscales the box to even whole numbers of pixels; otherwise
/// a sentence that says why not and which factors nearby would work.
inline std::string downscaleFactorProblem(int size_x, int size_y, double factor)
{
	std::ostringstream msg;
	if (!(factor > 0.))
	{
		msg << "the downscaling factor must be larger than zero (it is " << factor << ").";
		return msg.str();
	}
	int new_x, new_y;
	if (downscaledBox(size_x, size_y, factor, new_x, new_y))
		return "";

	msg << "a " << size_x << " x " << size_y << " box downscaled by " << factor << " would be "
	    << std::setprecision(6) << size_x / factor << " x " << size_y / factor
	    << " pixels, but both sides must be even whole numbers so that they keep the same pixel size.";

	// Factors g / (2m): the nearest with m at or below g / (2 factor) is the larger factor
	const long long g = greatestCommonDivisor(size_x, size_y);
	if (g < 2 || size_x % 2 != 0 || size_y % 2 != 0)
		return msg.str();
	const double m_exact = (double)g / (2. * factor);
	const long long m_lo = (long long)std::floor(m_exact), m_hi = (long long)std::ceil(m_exact);
	const long long px = size_x / g, py = size_y / g;
	std::ostringstream options;
	int nr_options = 0;
	const long long ms[2] = {m_hi, m_lo};
	for (int k = 0; k < 2; k++)
	{
		const long long m = ms[k];
		if (m < 1 || (k == 1 && m == m_hi))
			continue;
		const double f = (double)g / (2. * m);
		options << (nr_options > 0 ? " or " : "") << std::setprecision(6) << f
		        << " (" << 2 * m * px << " x " << 2 * m * py << ")";
		nr_options++;
	}
	if (nr_options > 0)
		msg << " The nearest factors that work are " << options.str() << ".";
	return msg.str();
}

} // namespace extractrect

#endif
