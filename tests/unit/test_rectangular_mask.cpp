/*
 * tests/unit/test_rectangular_mask.cpp
 *
 * Rectangular particles: downscaling an extracted box by a factor
 * (src/rect_downscale.h), and the masks of relion_refine for rectangular
 * images, which are ellipses with the proportions of the box whose long axis
 * is the mask diameter (BoxEllipse in src/mask.h), for helical segments
 * combined with the tube mask.
 */

#include <catch2/catch.hpp>

#include "src/mask.h"
#include "src/rect_downscale.h"

#include <cmath>
#include <random>

using Catch::Matchers::Contains;

TEST_CASE("downscale factor: both sides must become even whole numbers", "[rectangular][rect_downscale]")
{
	int nx, ny;
	REQUIRE(extractrect::downscaledBox(768, 256, 2., nx, ny));
	REQUIRE(nx == 384);
	REQUIRE(ny == 128);

	REQUIRE(extractrect::downscaledBox(768, 256, 4., nx, ny));
	REQUIRE(nx == 192);
	REQUIRE(ny == 64);

	// A typed factor that is close enough to 768 / 288 is accepted
	REQUIRE(extractrect::downscaledBox(768, 256, 2.6667, nx, ny));
	REQUIRE(nx == 288);
	REQUIRE(ny == 96);

	// 256 / 3 is not a whole number
	REQUIRE_FALSE(extractrect::downscaledBox(768, 256, 3., nx, ny));
	REQUIRE_FALSE(extractrect::downscaledBox(768, 250, 2., nx, ny)); // 125 rows is odd
	REQUIRE_FALSE(extractrect::downscaledBox(768, 256, 0., nx, ny));
	REQUIRE_FALSE(extractrect::downscaledBox(768, 256, -2., nx, ny));

	// Upscaling works the same way
	REQUIRE(extractrect::downscaledBox(128, 64, 0.5, nx, ny));
	REQUIRE(nx == 256);
	REQUIRE(ny == 128);
}

TEST_CASE("downscale factor: the message names the nearest factors that work", "[rectangular][rect_downscale]")
{
	REQUIRE(extractrect::downscaleFactorProblem(768, 256, 2.) == "");
	REQUIRE(extractrect::downscaleFactorProblem(256, 256, 2.) == "");

	// 768 x 256 = 3 x 1 times 256: boxes are 6m x 2m, factors 128 / m
	const std::string msg = extractrect::downscaleFactorProblem(768, 256, 3.);
	REQUIRE_THAT(msg, Contains("768 x 256"));
	REQUIRE_THAT(msg, Contains("256 x 85.3333"));
	REQUIRE_THAT(msg, Contains("2.97674 (258 x 86)"));
	REQUIRE_THAT(msg, Contains("3.04762 (252 x 84)"));

	// Both suggestions give even whole boxes
	int nx, ny;
	REQUIRE(extractrect::downscaledBox(768, 256, 256. / 86., nx, ny));
	REQUIRE(extractrect::downscaledBox(768, 256, 256. / 84., nx, ny));

	// gcd(768, 250) = 2: only factor 1 keeps the aspect ratio with even sides
	REQUIRE_THAT(extractrect::downscaleFactorProblem(768, 250, 2.), Contains("are 1 (768 x 250)"));

	// A factor larger than the box allows: only the smallest box
	REQUIRE_THAT(extractrect::downscaleFactorProblem(768, 256, 500.), Contains("128 (6 x 2)"));

	REQUIRE_THAT(extractrect::downscaleFactorProblem(768, 256, 0.), Contains("larger than zero"));
}

TEST_CASE("BoxEllipse: the mask diameter is the long axis, the edge is cosine_width wide on every axis", "[rectangular][rect_mask]")
{
	// 64 (x) by 32 (y): radius 30 is the semi-axis along x, 15 along y
	MultidimArray<RFLOAT> img(32, 64);
	const BoxEllipse e(img, 30., 4.);
	REQUIRE(e.ax == Approx(30.));
	REQUIRE(e.ay == Approx(15.));
	REQUIRE(e.shortestSemiAxis(false) == Approx(15.));

	REQUIRE(e.edge(0., 0., 0.) == 0.);
	REQUIRE(e.edge(30., 0., 0.) == 0.);
	REQUIRE(e.edge(32., 0., 0.) == Approx(0.5));
	REQUIRE(e.edge(34., 0., 0.) == 1.);
	REQUIRE(e.edge(0., 15., 0.) == 0.);
	REQUIRE(e.edge(0., -17., 0.) == Approx(0.5));
	REQUIRE(e.edge(0., 19., 0.) == 1.);
	// Along a diagonal it grows monotonically from 0 to 1
	RFLOAT last = 0.;
	for (RFLOAT s = 0.; s < 2.; s += 0.01)
	{
		const RFLOAT t = e.edge(s * 22., s * 11., 0.);
		REQUIRE(t >= last);
		REQUIRE(t >= 0.);
		REQUIRE(t <= 1.);
		last = t;
	}
	REQUIRE(last == 1.);

	// A cuboid of 20 x 20 x 60 (z longest): radius 27 along z, 9 across
	MultidimArray<RFLOAT> vol(60, 20, 20);
	const BoxEllipse e3(vol, 27., 3.);
	REQUIRE(e3.az == Approx(27.));
	REQUIRE(e3.ax == Approx(9.));
	REQUIRE(e3.ay == Approx(9.));
	REQUIRE(e3.edge(0., 0., 28.5) == Approx(0.5));
	REQUIRE(e3.edge(10.5, 0., 0.) == Approx(0.5));

	// In a square it is (r - R) / width
	MultidimArray<RFLOAT> sq(32, 32);
	const BoxEllipse c(sq, 10., 3.);
	std::mt19937 rng(7);
	std::uniform_real_distribution<RFLOAT> u(-16., 16.);
	for (int n = 0; n < 1000; n++)
	{
		const RFLOAT x = u(rng), y = u(rng);
		const RFLOAT expected = std::min<RFLOAT>(1., std::max<RFLOAT>(0., (std::sqrt(x * x + y * y) - 10.) / 3.));
		REQUIRE(c.edge(x, y, 0.) == Approx(expected).margin(1e-12));
	}
}

static MultidimArray<RFLOAT> noiseImage(int ny, int nx, unsigned seed)
{
	MultidimArray<RFLOAT> img(ny, nx);
	std::mt19937 rng(seed);
	std::normal_distribution<RFLOAT> g(0., 1.);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img)
		DIRECT_MULTIDIM_ELEM(img, n) = g(rng);
	img.setXmippOrigin();
	return img;
}

TEST_CASE("softMaskOutsideMap: by default the radius of a rectangular mask is on the short side", "[rectangular][rect_mask]")
{
	// --rect_mask_length off: semi-axes 14 (y) and 28 (x), the radius on the short side
	const MultidimArray<RFLOAT> orig = noiseImage(32, 64, 1);
	MultidimArray<RFLOAT> img = orig;
	softMaskOutsideMap(img, 14., 3., NULL, true);
	REQUIRE(A2D_ELEM(img, 0, 26) == A2D_ELEM(orig, 0, 26));
	REQUIRE(A2D_ELEM(img, 13, 0) == A2D_ELEM(orig, 13, 0));
	REQUIRE(A2D_ELEM(img, 15, 31) != A2D_ELEM(orig, 15, 31));

	// The same ellipse as the long-side one with radius 28, apart from the width of its edge
	MultidimArray<RFLOAT> img_long = orig;
	softMaskOutsideMap(img_long, 28., 3., NULL, true, true);
	REQUIRE(A2D_ELEM(img_long, 0, 26) == A2D_ELEM(img, 0, 26));
	REQUIRE(A2D_ELEM(img_long, 13, 0) == A2D_ELEM(img, 13, 0));
}

TEST_CASE("softMaskOutsideMap: a rectangular image is masked by an ellipse along its long side", "[rectangular][rect_mask]")
{
	const MultidimArray<RFLOAT> orig = noiseImage(32, 64, 1);
	MultidimArray<RFLOAT> img = orig;
	softMaskOutsideMap(img, 28., 3., NULL, true, true); // semi-axes 28 (x) and 14 (y)
	const BoxEllipse e(img, 28., 3.);

	// Inside unchanged (also far along x, beyond half the short side), outside one constant
	REQUIRE(A2D_ELEM(img, 0, 26) == A2D_ELEM(orig, 0, 26));
	REQUIRE(A2D_ELEM(img, 0, -27) == A2D_ELEM(orig, 0, -27));
	REQUIRE(A2D_ELEM(img, 13, 0) == A2D_ELEM(orig, 13, 0));
	const RFLOAT bg = A2D_ELEM(img, 15, 31);
	FOR_ALL_ELEMENTS_IN_ARRAY2D(img)
	{
		const RFLOAT t = e.edge(j, i, 0.);
		if (t <= 0.)
			REQUIRE(A2D_ELEM(img, i, j) == A2D_ELEM(orig, i, j));
		else if (t >= 1.)
			REQUIRE(A2D_ELEM(img, i, j) == bg);
	}
	REQUIRE(A2D_ELEM(img, 15, 0) != A2D_ELEM(orig, 15, 0)); // in the edge across the short side

	// Without a radius the ellipse is inscribed in the box
	MultidimArray<RFLOAT> img2 = orig;
	softMaskOutsideMap(img2, -1., 3., NULL, true, true);
	REQUIRE(A2D_ELEM(img2, 0, 31) == A2D_ELEM(orig, 0, 31));
	REQUIRE(A2D_ELEM(img2, 15, 0) == A2D_ELEM(orig, 15, 0));
	REQUIRE(A2D_ELEM(img2, 15, 31) != A2D_ELEM(orig, 15, 31));

	// With a noise image, the background comes from it
	MultidimArray<RFLOAT> noise = noiseImage(32, 64, 2), img3 = orig;
	softMaskOutsideMap(img3, 28., 3., &noise, true, true);
	REQUIRE(A2D_ELEM(img3, 15, 31) == A2D_ELEM(noise, 15, 31));
	REQUIRE(A2D_ELEM(img3, 0, 26) == A2D_ELEM(orig, 0, 26));
}

TEST_CASE("softMaskOutsideMap: for a square image the ellipse is the usual circle", "[rectangular][rect_mask]")
{
	const MultidimArray<RFLOAT> orig = noiseImage(48, 48, 3);
	MultidimArray<RFLOAT> a = orig, b = orig;
	softMaskOutsideMap(a, 18., 5., NULL, false);
	softMaskOutsideMap(b, 18., 5., NULL, true, true);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a)
		REQUIRE(DIRECT_MULTIDIM_ELEM(b, n) == Approx(DIRECT_MULTIDIM_ELEM(a, n)).margin(1e-10));
}

TEST_CASE("softMaskOutsideMapForHelix: a rectangular segment keeps its full length", "[rectangular][rect_mask][helix_mask]")
{
	// 96 x 32, tube along x: ellipse semi-axes 44 (x) and 44/3 (y), tube radius 8
	const MultidimArray<RFLOAT> orig = noiseImage(32, 96, 4);
	MultidimArray<RFLOAT> img = orig;
	softMaskOutsideMapForHelix(img, 0., 0., 44., 8., 3., NULL, true);

	REQUIRE(A2D_ELEM(img, 0, 40) == A2D_ELEM(orig, 0, 40));   // far along the tube: kept
	REQUIRE(A2D_ELEM(img, 5, -38) == A2D_ELEM(orig, 5, -38)); // inside tube and ellipse
	REQUIRE(A2D_ELEM(img, 0, 47) != A2D_ELEM(orig, 0, 47));   // beyond the ellipse
	const RFLOAT bg = A2D_ELEM(img, 12, 0);                    // beyond the tube
	REQUIRE(bg != A2D_ELEM(orig, 12, 0));
	REQUIRE(A2D_ELEM(img, -12, 30) == bg);
	REQUIRE(A2D_ELEM(img, 0, 47) == bg);

	// The tilted case: with psi 90 the tube is vertical and cut by the short ellipse axis
	MultidimArray<RFLOAT> img2 = orig;
	softMaskOutsideMapForHelix(img2, 90., 0., 44., 8., 3., NULL, true);
	REQUIRE(A2D_ELEM(img2, 10, 0) == A2D_ELEM(orig, 10, 0));
	REQUIRE(A2D_ELEM(img2, 0, 20) != A2D_ELEM(orig, 0, 20));
}

TEST_CASE("softMaskOutsideMapForHelix: square images are unchanged by follow_box_shape", "[rectangular][rect_mask][helix_mask]")
{
	const MultidimArray<RFLOAT> orig = noiseImage(48, 48, 5);
	MultidimArray<RFLOAT> a = orig, b = orig;
	softMaskOutsideMapForHelix(a, 30., 0., 20., 9., 4., NULL, false);
	softMaskOutsideMapForHelix(b, 30., 0., 20., 9., 4., NULL, true);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a)
		REQUIRE(DIRECT_MULTIDIM_ELEM(b, n) == DIRECT_MULTIDIM_ELEM(a, n));
}

TEST_CASE("softMaskOutsideMapForHelix: the tube must fit across the ellipse of a rectangular image", "[rectangular][rect_mask][helix_mask]")
{
	MultidimArray<RFLOAT> img = noiseImage(32, 96, 6);
	// Radius 30 along x gives 10 across: a tube of radius 12 does not fit
	REQUIRE_THROWS(softMaskOutsideMapForHelix(img, 0., 0., 30., 12., 3., NULL, true));
	// A radius longer than half the long side does not fit either
	REQUIRE_THROWS(softMaskOutsideMapForHelix(img, 0., 0., 50., 8., 3., NULL, true));
	REQUIRE_NOTHROW(softMaskOutsideMapForHelix(img, 0., 0., 47., 8., 3., NULL, true));
}

TEST_CASE("in-plane angle before a long reference wraps around across the box", "[rectangular][rect_mask]")
{
	// Default helical masks: a circle that fits the width never reaches beyond the box
	REQUIRE(inPlaneAngleBeforeWrapAround(115., 115., 50., 768., 256.) == 90.);

	// Masks along the length of a 768 x 256 box: semi-axes 345 and 115, a 50 pixel tube.
	// Across the box the tube reaches 345 sin(a) + 50 cos(a), which is 128 at about 13.2 degrees
	const RFLOAT helical = inPlaneAngleBeforeWrapAround(345., 115., 50., 768., 256.);
	REQUIRE(helical >= 12.5);
	REQUIRE(helical <= 13.);
	// Without the tube the ellipse itself reaches across sooner (about 10 degrees)
	const RFLOAT ellipse = inPlaneAngleBeforeWrapAround(345., 115., -1., 768., 256.);
	REQUIRE(ellipse >= 9.5);
	REQUIRE(ellipse <= 10.);

	// A reference longer than the box does not fit even along it
	REQUIRE(inPlaneAngleBeforeWrapAround(200., 100., -1., 320., 224.) == -1.);
}
