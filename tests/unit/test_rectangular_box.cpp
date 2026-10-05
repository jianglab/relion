/*
 * tests/unit/test_rectangular_box.cpp
 *
 * Stage 1 of rectangular particle support: the Box type with physical
 * frequencies, its STAR size columns, and the shared rotated-rectangle
 * resampling (linear, cubic, NUFFT).
 */

#include <catch2/catch.hpp>

#include "src/image_box.h"
#include "src/resample_rotate.h"
#include "src/metadata_table.h"
#include "src/ctf.h"
#include "src/extract_rect.h"

#include <cmath>

TEST_CASE("Box: square box behaves like the old integer size", "[rectangular]")
{
	Box b = Box::fromSize(64, 2, 1.5);
	CHECK(b.isSquare());
	CHECK(b.size() == 64);
	CHECK(b.halfWidth() == 33);
	CHECK(b.dim() == 2);
	CHECK(Box::fromSize(32, 3).isSquare());
	CHECK(Box::fromSize(32, 3).dim() == 3);
	CHECK(b.radialFreq(5, 0) == Approx(5.0 / (64 * 1.5)));
	CHECK(b.limitX(0.2) == b.limitY(0.2));
}

TEST_CASE("Box: rectangular box uses physical frequencies per axis", "[rectangular]")
{
	Box b(256, 64, 1, 2.0);
	CHECK_FALSE(b.isSquare());
	CHECK_THROWS(b.size());
	CHECK(b.halfWidth() == 129);
	CHECK(b.freqX(10) == Approx(10.0 / 512.0));
	CHECK(b.freqY(10) == Approx(10.0 / 128.0));
	CHECK(b.radialFreq(3, 4) == Approx(std::sqrt(std::pow(3.0 / 512.0, 2) + std::pow(4.0 / 128.0, 2))));
	// The same physical resolution takes more Fourier pixels along the long axis.
	CHECK(b.limitX(0.1) == 52);
	CHECK(b.limitY(0.1) == 13);
	// Never beyond Nyquist.
	CHECK(b.limitY(10.0) == 32);
	CHECK(b.nyquist() == Approx(0.25));
}

TEST_CASE("Box: STAR columns", "[rectangular]")
{
	MetaDataTable sq, rect;
	sq.addObject();
	Box::fromSize(100).toTable(sq, sq.numberOfObjects() - 1);
	CHECK(sq.containsLabel(EMDL_IMAGE_SIZE));
	CHECK_FALSE(sq.containsLabel(EMDL_IMAGE_SIZE_X));

	rect.addObject();
	Box(300, 100, 1).toTable(rect, rect.numberOfObjects() - 1);
	CHECK_FALSE(rect.containsLabel(EMDL_IMAGE_SIZE));
	CHECK(rect.containsLabel(EMDL_IMAGE_SIZE_X));
	CHECK(rect.containsLabel(EMDL_IMAGE_SIZE_Y));

	Box got;
	REQUIRE(Box::fromTable(sq, sq.numberOfObjects() - 1, 2, got));
	CHECK(got.nx == 100);
	CHECK(got.ny == 100);
	REQUIRE(Box::fromTable(rect, rect.numberOfObjects() - 1, 2, got));
	CHECK(got.nx == 300);
	CHECK(got.ny == 100);

	MetaDataTable none;
	none.addObject();
	CHECK_FALSE(Box::fromTable(none, none.numberOfObjects() - 1, 2, got));
}

namespace
{
const int N = 96;

double smooth(double x, double y)
{
	const double t = 2.0 * PI / N;
	return std::cos(t * (3.0 * x + 2.0 * y) + 0.4) + 0.5 * std::sin(t * (5.0 * x - 4.0 * y) + 1.1);
}

MultidimArray<RFLOAT> makeSource()
{
	MultidimArray<RFLOAT> a(N, N);
	for (int y = 0; y < N; y++)
		for (int x = 0; x < N; x++)
			DIRECT_A2D_ELEM(a, y, x) = smooth(x, y);
	return a;
}

// Largest error against the analytic function over output pixels whose source position is well inside.
double maxError(const MultidimArray<RFLOAT> &src, ResampleMethod m, int nx, int ny, double cx, double cy, double ang)
{
	MultidimArray<RFLOAT> out;
	resampleRotatedRectangle(src, out, nx, ny, cx, cy, ang, m);
	const double a = DEG2RAD(ang);
	double worst = 0.0;
	for (int y = 0; y < ny; y++)
		for (int x = 0; x < nx; x++)
		{
			const double dx = x - nx / 2, dy = y - ny / 2;
			const double px = cx + dx * std::cos(a) - dy * std::sin(a);
			const double py = cy + dx * std::sin(a) + dy * std::cos(a);
			if (px < 2 || py < 2 || px > N - 3 || py > N - 3) continue;
			worst = std::max(worst, std::abs(DIRECT_A2D_ELEM(out, y, x) - smooth(px, py)));
		}
	return worst;
}
}

TEST_CASE("resample: zero angle and integer centre is an exact crop", "[rectangular]")
{
	MultidimArray<RFLOAT> src = makeSource();
	std::vector<ResampleMethod> methods = {RESAMPLE_LINEAR, RESAMPLE_CUBIC};
#ifdef RELION_USE_FINUFFT
	methods.push_back(RESAMPLE_NUFFT);
#endif
	for (ResampleMethod m : methods)
	{
		MultidimArray<RFLOAT> out;
		resampleRotatedRectangle(src, out, 40, 16, 48, 48, 0.0, m);
		REQUIRE(XSIZE(out) == 40);
		REQUIRE(YSIZE(out) == 16);
		double worst = 0.0;
		for (int y = 0; y < 16; y++)
			for (int x = 0; x < 40; x++)
				worst = std::max(worst, std::abs((double)DIRECT_A2D_ELEM(out, y, x) - DIRECT_A2D_ELEM(src, 48 + y - 8, 48 + x - 20)));
		INFO(resampleMethodName(m));
		CHECK(worst < 1e-9);
	}
}

TEST_CASE("resample: rotated rectangle recovers a smooth image, better with better methods", "[rectangular]")
{
	MultidimArray<RFLOAT> src = makeSource();
	const double lin = maxError(src, RESAMPLE_LINEAR, 60, 20, 47.3, 48.9, 37.0);
	const double cub = maxError(src, RESAMPLE_CUBIC, 60, 20, 47.3, 48.9, 37.0);
	CHECK(lin < 0.15);
	CHECK(cub < lin);
	CHECK(cub < 0.02);
#ifdef RELION_USE_FINUFFT
	const double nu = maxError(src, RESAMPLE_NUFFT, 60, 20, 47.3, 48.9, 37.0);
	CHECK(nu < 1e-6);
#endif
}

TEST_CASE("resample: parsing and unknown method", "[rectangular]")
{
	CHECK(parseResampleMethod("cubic") == RESAMPLE_CUBIC);
	CHECK(resampleMethodName(RESAMPLE_NUFFT) == "nufft");
	CHECK_THROWS(parseResampleMethod("lanczos"));
}

TEST_CASE("extraction: rotated defocus angle gives the same CTF in the rotated frame", "[rectangular]")
{
	// Particle frame = micrograph frame rotated by box angle a: a direction at
	// angle t in the micrograph appears at t - a in the particle.
	const double angles[] = {-170., -90., -37.5, 0., 12.25, 45., 90., 133.}; // box angle, degrees
	const double def_angles[] = {-80., 0., 33., 89.};
	for (double a : angles)
		for (double d : def_angles)
		{
			CTF before, after;
			before.setValues(21000., 17000., d, 300., 2.7, 0.1, 0.);
			const double d2 = extractrect::rotatedDefocusAngle(d, a);
			REQUIRE(d2 > -90. - 1e-9);
			REQUIRE(d2 <= 90. + 1e-9);
			after.setValues(21000., 17000., d2, 300., 2.7, 0.1, 0.);
			const double ar = a * M_PI / 180.;
			for (double t = 0.; t < 360.; t += 23.)
				for (double s : {0.05, 0.12, 0.2})
				{
					const double tr = t * M_PI / 180.;
					const double X = s * cos(tr), Y = s * sin(tr);
					const double Xr = s * cos(tr - ar), Yr = s * sin(tr - ar);
					REQUIRE(after.getCTF(Xr, Yr) == Approx(before.getCTF(X, Y)).margin(1e-9));
				}
		}
}
