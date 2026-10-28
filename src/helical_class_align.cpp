/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/helical_class_align.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace helixalign
{

namespace
{

// Value of an image (origin at its centre) at (x, y) by bilinear interpolation, 0 outside
double sample(const MultidimArray<RFLOAT> &img, double x, double y)
{
	const int j0 = (int)std::floor(x), i0 = (int)std::floor(y);
	const double fx = x - j0, fy = y - i0;
	double v = 0.;
	for (int di = 0; di <= 1; di++)
		for (int dj = 0; dj <= 1; dj++)
		{
			const int i = i0 + di, j = j0 + dj;
			if (i < STARTINGY(img) || i > FINISHINGY(img) || j < STARTINGX(img) || j > FINISHINGX(img))
				continue;
			v += (di ? fy : 1. - fy) * (dj ? fx : 1. - fx) * A2D_ELEM(img, i, j);
		}
	return v;
}

}

double classAngle(std::vector<double> d, size_t min_segments)
{
	if (d.size() < min_segments || d.empty())
		return std::numeric_limits<double>::quiet_NaN();
	// Into [-90, 90): opposite polarity is the same axis
	for (double &x : d)
		x -= 180. * std::floor((x + 90.) / 180.);
	std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
	return d[d.size() / 2];
}

void apply(MultidimArray<RFLOAT> &img, double angle_deg, double offset)
{
	img.setXmippOrigin();
	MultidimArray<RFLOAT> in = img;
	const double t = angle_deg * PI / 180., c = cos(t), s = sin(t);
	// Output point (x, y) shows the input point at x along the axis and y + offset across it
	FOR_ALL_ELEMENTS_IN_ARRAY2D(img)
	{
		const double u = j, v = i + offset;
		A2D_ELEM(img, i, j) = sample(in, u * c - v * s, u * s + v * c);
	}
}

double offsetAcross(const MultidimArray<RFLOAT> &img, double angle_deg)
{
	MultidimArray<RFLOAT> a = img;
	a.setXmippOrigin();
	Matrix1D<RFLOAT> com;
	a.centerOfMass(com);
	const double t = angle_deg * PI / 180.;
	return -XX(com) * sin(t) + YY(com) * cos(t);
}

}
