/*
 * tests/unit/test_helical_class_align.cpp
 *
 * Turning and centring 2D class averages of helical segments (src/helical_class_align.h),
 * on synthetic filaments with a known angle and offset.
 */

#include <catch2/catch.hpp>

#include "src/helical_class_align.h"

#include <cmath>
#include <random>

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

TEST_CASE("helical class alignment: the diameter of a class across its axis", "[helical_class_align]")
{
	// A band along X of half-width h, with a dark stripe along its axis (2 pixels either side of it, as in two
	// protofilaments), on a sloped background, and another filament near the edge of the box
	auto band = [](int n, int h)
	{
		MultidimArray<RFLOAT> img(n, n);
		img.setXmippOrigin();
		FOR_ALL_ELEMENTS_IN_ARRAY2D(img)
		{
			const double inside = (ABS(i) <= h && ABS(i) > 1) ? 1. : 0.;
			const double other = (i >= 52 && i <= 58) ? 1. : 0.;
			A2D_ELEM(img, i, j) = 0.3 + 0.002 * i + inside + other;
		}
		return img;
	};
	// Edge: the last row above 10% of the peak after smoothing (h + 1), plus half a pixel; padded by 10% each side
	for (int h : {10, 30})
	{
		INFO("half-width " << h);
		CHECK(helixalign::classDiameter(band(128, h), 4.) == Approx(2. * (h + 1.5) * 1.2).margin(1.5));
	}
	// without the gap rule, the dark stripe would end the filament: a stripe narrower than the gap does not
	CHECK(helixalign::classDiameter(band(128, 10), 4.) > 20.);
	// nothing there: the box
	MultidimArray<RFLOAT> flat(128, 128);
	flat.initConstant(0.5);
	CHECK(helixalign::classDiameter(flat, 4.) == 128.);

	// coarse shape smoothing (smooth_sigma_px = 3.0) rejects pure Gaussian noise back to ny
	std::mt19937 rng(42);
	std::normal_distribution<double> noise(0.0, 0.5);
	MultidimArray<RFLOAT> pure_noise(128, 128);
	pure_noise.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY2D(pure_noise)
		A2D_ELEM(pure_noise, i, j) = noise(rng);
	CHECK(helixalign::classDiameter(pure_noise, 4., 0.1, 0.1, 3.0) == 128.);

	// noisy filament with negative CTF side-lobes + neighboring outer filament stays within reasonable width
	MultidimArray<RFLOAT> noisy_ctf(128, 128);
	noisy_ctf.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY2D(noisy_ctf)
	{
		const double core = (ABS(i) <= 8 && ABS(i) > 1) ? 2.0 : 0.0;
		const double sidelobe = (ABS(i) >= 11 && ABS(i) <= 15) ? -0.9 : 0.0;
		const double other = (i >= 52 && i <= 58) ? 1.5 : 0.0;
		A2D_ELEM(noisy_ctf, i, j) = 0.2 + core + sidelobe + other + 0.6 * noise(rng);
	}
	const double d_noisy = helixalign::classDiameter(noisy_ctf, 4., 0.1, 0.1, 3.0);
	CHECK(d_noisy >= 20.0);
	CHECK(d_noisy <= 48.0);
}
