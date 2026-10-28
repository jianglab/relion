/*
 * tests/unit/test_helical_class_align.cpp
 *
 * Turning and centring 2D class averages of helical segments (src/helical_class_align.h),
 * on synthetic filaments with a known angle and offset.
 */

#include <catch2/catch.hpp>

#include "src/helical_class_align.h"

#include <cmath>

namespace
{
// A filament along direction angle_deg, offset across it: two strands 6 pixels either side of its
// axis, modulated along the axis with a period of 24 pixels (a twisted ribbon), faded out towards
// the edge of the box so that the box cuts no direction
MultidimArray<RFLOAT> filament(int n, double angle_deg, double offset)
{
	MultidimArray<RFLOAT> img(n, n);
	img.setXmippOrigin();
	const double t = angle_deg * PI / 180., c = cos(t), s = sin(t);
	FOR_ALL_ELEMENTS_IN_ARRAY2D(img)
	{
		const double u = j * c + i * s;          // along the axis
		const double v = -j * s + i * c - offset; // across it, from the axis
		const double strands = exp(-(v - 6) * (v - 6) / 8.) + exp(-(v + 6) * (v + 6) / 8.);
		const double r = sqrt((double)(i * i + j * j)), fade = r < 0.3 * n ? 1. : r > 0.45 * n ? 0. : 0.5 + 0.5 * cos(PI * (r - 0.3 * n) / (0.15 * n));
		A2D_ELEM(img, i, j) = fade * strands * (1. + 0.6 * cos(2. * PI * u / 24.));
	}
	return img;
}
}

TEST_CASE("helical class alignment: the angle of a class from its segments", "[helical_class_align]")
{
	// the median; opposite polarity (prior + 180, or - 180) counts as the same axis
	std::vector<double> d;
	for (int k = 0; k < 30; k++)
		d.push_back(1.5 + 0.1 * (k - 15) + (k % 3 == 0 ? 180. : k % 3 == 1 ? -180. : 0.));
	CHECK(helixalign::classAngle(d) == Approx(1.5).margin(1e-9));
	// a few wild segments do not move it
	d.push_back(60.); d.push_back(-70.);
	CHECK(helixalign::classAngle(d) == Approx(1.5).margin(0.11));
	// too few segments: no angle
	CHECK(std::isnan(helixalign::classAngle(std::vector<double>(5, 2.))));
}

TEST_CASE("helical class alignment: centre across the axis is the centre of mass", "[helical_class_align]")
{
	for (double angle : {0., 4.3, -7.9})
		for (double offset : {0., 3.6, -5.2})
		{
			INFO("angle " << angle << " offset " << offset);
			CHECK(helixalign::offsetAcross(filament(128, angle, offset), angle) == Approx(offset).margin(0.3));
		}
}

TEST_CASE("helical class alignment: apply moves the axis onto X through the centre", "[helical_class_align]")
{
	const MultidimArray<RFLOAT> centred = filament(128, 0., 0.);
	for (double angle : {0., 4.3, -7.9})
		for (double offset : {0., 4., -2.5})
		{
			INFO("angle " << angle << " offset " << offset);
			MultidimArray<RFLOAT> img = filament(128, angle, offset);
			helixalign::apply(img, angle, offset);
			double max_diff = 0.;
			for (int i = -30; i <= 30; i++)
				for (int j = -30; j <= 30; j++)
					max_diff = XMIPP_MAX(max_diff, ABS(A2D_ELEM(img, i, j) - A2D_ELEM(centred, i, j)));
			CHECK(max_diff < 0.06);  // bilinear interpolation of strands 2 pixels wide
		}
}
