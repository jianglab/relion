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

double classDiameter(const MultidimArray<RFLOAT> &img, double gap_px, double frac, double pad)
{
	const int ny = YSIZE(img), nx = XSIZE(img);
	const int x0 = (int)(0.15 * nx), x1 = std::max(x0 + 1, (int)(0.85 * nx));
	std::vector<double> raw(ny, 0.), p(ny, 0.), y(ny);
	for (int i = 0; i < ny; i++)
	{
		y[i] = i - ny / 2;
		for (int j = x0; j < x1; j++)
			raw[i] += DIRECT_A2D_ELEM(img, i, j);
		raw[i] /= x1 - x0;
	}
	// Gaussian of sigma 0.7 pixels, renormalised at the ends
	for (int i = 0; i < ny; i++)
	{
		double s = 0., w = 0.;
		for (int d = -3; d <= 3; d++)
			if (i + d >= 0 && i + d < ny)
			{
				const double g = exp(-0.5 * d * d / 0.49);
				s += g * raw[i + d], w += g;
			}
		p[i] = s / w;
	}
	// Straight background through the outer bands; points more than 3 MADs above it (other filaments) left out
	std::vector<char> bg(ny);
	for (int i = 0; i < ny; i++)
		bg[i] = std::fabs(y[i]) > 0.35 * ny;
	double a = 0., b = 0.;
	for (int it = 0; it < 3; it++)
	{
		double n = 0., sy = 0., sp = 0., syy = 0., syp = 0.;
		for (int i = 0; i < ny; i++)
			if (bg[i])
				n++, sy += y[i], sp += p[i], syy += y[i] * y[i], syp += y[i] * p[i];
		if (n < 4.)
			break;
		const double det = n * syy - sy * sy;
		b = (det > 0.) ? (n * syp - sy * sp) / det : 0.;
		a = (sp - b * sy) / n;
		std::vector<double> r;
		for (int i = 0; i < ny; i++)
			if (bg[i])
				r.push_back(p[i] - a - b * y[i]);
		std::vector<double> s = r;
		std::nth_element(s.begin(), s.begin() + s.size() / 2, s.end());
		const double med = s[s.size() / 2];
		for (double &x : s)
			x = std::fabs(x - med);
		std::nth_element(s.begin(), s.begin() + s.size() / 2, s.end());
		const double mad = s[s.size() / 2] + 1e-12;
		for (int i = 0; i < ny; i++)
			if (bg[i] && p[i] - a - b * y[i] >= med + 3. * mad)
				bg[i] = 0;
	}
	std::vector<double> q(ny);
	int pk = -1;
	for (int i = 0; i < ny; i++)
	{
		q[i] = p[i] - a - b * y[i];
		if (std::fabs(y[i]) < 0.3 * ny && (pk < 0 || q[i] > q[pk]))
			pk = i;
	}
	double pmax = 0.;
	for (double v : p)
		pmax = std::max(pmax, std::fabs(v));
	// rounding errors of the background are not a peak
	if (pk < 0 || !(q[pk] > 1e-6 * pmax))
		return ny;
	const double thr = frac * q[pk];
	const int gap = std::max(2, (int)std::lround(gap_px));
	auto walk = [&](int step)
	{
		int e = pk, below = 0;
		for (int i = pk + step; i >= 0 && i < ny && below < gap; i += step)
		{
			if (q[i] > thr)
				e = i, below = 0;
			else
				below++;
		}
		return e;
	};
	const double half = std::max(std::fabs(y[walk(-1)]), std::fabs(y[walk(+1)])) + 0.5;
	return 2. * half * (1. + 2. * pad);
}

}
