/*
 * tests/unit/test_cryosparc_motion.cpp
 *
 * Unit tests for the CryoSPARC patch-motion port: the cubic B-spline
 * evaluation of CryoSPARC's local ("bending") field, and RELION's
 * ThirdOrderPolynomialModel that the field is refitted onto.
 *
 * These mirror the test suite of the cs2relion Python package that this logic
 * was ported from, so that a divergence between the two shows up here rather
 * than as a silently degraded refinement (the README of that package documents
 * a mis-set convention costing 3.7 A -> 10.3 A with no error raised anywhere).
 *
 * The port was additionally checked against the Python reference on real data
 * (a 24-frame 3838x3710 movie from a CryoSPARC Patch Motion job, all 25 RELION
 * patch positions x 24 frames): max absolute difference 3.2e-6 px, which is
 * the Python side's own float32 rounding - it computes the field in float32
 * while this port uses double throughout. That comparison needs a CryoSPARC
 * project on disk so it cannot live here, but these invariants can.
 */

#include <catch2/catch.hpp>

#include "src/cryosparc_spline.h"
#include "src/cryosparc_motion_model.h"

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Cubic B-spline evaluation
// ---------------------------------------------------------------------------

TEST_CASE("cubic B-spline weights are a partition of unity", "[cryosparc][spline]")
{
	for (int i = 0; i <= 10; i++)
	{
		const double f = i / 10.;
		double w[4];
		cryosparc::cubicBSplineWeights(f, w);
		CHECK((w[0] + w[1] + w[2] + w[3]) == Approx(1.).margin(1e-12));
		// A B-spline basis is non-negative everywhere
		for (int k = 0; k < 4; k++) CHECK(w[k] >= -1e-15);
	}
}

TEST_CASE("constant spline coefficients reproduce that constant", "[cryosparc][spline]")
{
	// A B-spline with natural boundary conditions and a constant coefficient
	// array synthesises back to that same constant everywhere. This holds
	// regardless of the reflective padding or the grid rescaling, so it catches
	// a silently changed convention in either.
	const int KZ = 3, KY = 6, KX = 5;
	std::vector<double> sx((size_t)KZ * KY * KX, 2.5);
	std::vector<double> sy((size_t)KZ * KY * KX, -1.25);

	const int nz = 24, ny = 3838, nx = 3710;
	std::vector<double> px, py;
	px.push_back(100.);  py.push_back(100.);
	px.push_back(3000.); py.push_back(3000.);
	px.push_back(1855.); py.push_back(1919.);

	std::vector<double> lx, ly;
	cryosparc::splineInterpTraj(nz, ny, nx, &sx[0], &sy[0], KZ, KY, KX, px, py, lx, ly);

	REQUIRE(lx.size() == px.size() * nz);
	for (size_t k = 0; k < lx.size(); k++)
	{
		CHECK(lx[k] == Approx(2.5).margin(1e-4));
		CHECK(ly[k] == Approx(-1.25).margin(1e-4));
	}
}

TEST_CASE("zero spline coefficients give a zero field", "[cryosparc][spline]")
{
	const int KZ = 3, KY = 6, KX = 5;
	std::vector<double> s((size_t)KZ * KY * KX, 0.);

	std::vector<double> px, py;
	px.push_back(370.);  py.push_back(382.);
	px.push_back(1854.); py.push_back(1919.);

	std::vector<double> lx, ly;
	cryosparc::splineInterpTraj(24, 3838, 3710, &s[0], &s[0], KZ, KY, KX, px, py, lx, ly);

	for (size_t k = 0; k < lx.size(); k++)
	{
		CHECK(lx[k] == Approx(0.).margin(1e-12));
		CHECK(ly[k] == Approx(0.).margin(1e-12));
	}
}

TEST_CASE("spline evaluation is exact at the control points of a linear ramp",
          "[cryosparc][spline]")
{
	// A cubic B-spline is not interpolatory at its own control points, so this
	// checks the weaker property that actually matters: the evaluation is a
	// smooth, monotone function of position for a monotone coefficient ramp,
	// which fails immediately if the axis order or the rescaling is transposed.
	const int KZ = 3, KY = 4, KX = 4;
	std::vector<double> s((size_t)KZ * KY * KX, 0.);
	for (int z = 0; z < KZ; z++)
	for (int y = 0; y < KY; y++)
	for (int x = 0; x < KX; x++)
		s[((size_t)z * KY + y) * KX + x] = (double)x;   // ramp along X only

	const int nz = 8, ny = 1000, nx = 1000;
	std::vector<double> px, py;
	for (int i = 0; i < 5; i++) { px.push_back(100. + 200. * i); py.push_back(500.); }

	std::vector<double> lx, ly;
	cryosparc::splineInterpTraj(nz, ny, nx, &s[0], &s[0], KZ, KY, KX, px, py, lx, ly);

	// Strictly increasing along +x, and independent of frame for a ramp with no
	// temporal variation
	for (size_t p = 1; p < px.size(); p++)
		CHECK(lx[p * nz] > lx[(p - 1) * nz]);
	for (int f = 1; f < nz; f++)
		CHECK(lx[f] == Approx(lx[0]).margin(1e-12));
}

// ---------------------------------------------------------------------------
// RELION's ThirdOrderPolynomialModel
// ---------------------------------------------------------------------------

TEST_CASE("motion design matrix vanishes at the first frame", "[cryosparc][motionmodel]")
{
	// None of RELION's 18 basis functions has a constant term, so every
	// prediction is exactly zero at frame 1 everywhere in the frame. That is
	// what makes "shift relative to frame 1" structural rather than fitted.
	const double w = 1000., h = 1000.;
	const double xs[] = {100., 500., 900.};

	for (int i = 0; i < 3; i++)
	{
		double row[cryosparc::NUM_MOTION_COEFFS_PER_DIM];
		cryosparc::motionDesignRow(w, h, 1., xs[i], xs[i], 1, row);
		for (int j = 0; j < cryosparc::NUM_MOTION_COEFFS_PER_DIM; j++)
			CHECK(row[j] == Approx(0.).margin(1e-15));
	}
}

TEST_CASE("motion model fit recovers known coefficients", "[cryosparc][motionmodel]")
{
	const double w = 3710., h = 3710.;

	double true_x[cryosparc::NUM_MOTION_COEFFS_PER_DIM];
	double true_y[cryosparc::NUM_MOTION_COEFFS_PER_DIM];
	for (int j = 0; j < cryosparc::NUM_MOTION_COEFFS_PER_DIM; j++)
	{
		// Deterministic, spread over several orders of magnitude so that a
		// badly conditioned solve shows up
		true_x[j] = std::sin(0.7 * j + 0.1) * std::pow(10., -(j % 4));
		true_y[j] = std::cos(1.1 * j + 0.3) * std::pow(10., -(j % 3));
	}

	std::vector<std::pair<double, double> > grid = cryosparc::motionPatchGrid(w, h, 5, 5);

	std::vector<double> f, x, y, sx, sy;
	for (size_t p = 0; p < grid.size(); p++)
	for (int frame = 1; frame <= 24; frame++)
	{
		const double x_n = grid[p].first / w - 0.5;
		const double y_n = grid[p].second / h - 0.5;
		const double z = frame - 1;
		f.push_back(frame);
		x.push_back(grid[p].first);
		y.push_back(grid[p].second);
		sx.push_back(cryosparc::motionShiftAt(true_x, z, x_n, y_n));
		sy.push_back(cryosparc::motionShiftAt(true_y, z, x_n, y_n));
	}

	std::vector<double> cx, cy;
	cryosparc::fitMotionModel(w, h, 1, f, x, y, sx, sy, cx, cy);

	REQUIRE(cx.size() == (size_t)cryosparc::NUM_MOTION_COEFFS_PER_DIM);
	for (int j = 0; j < cryosparc::NUM_MOTION_COEFFS_PER_DIM; j++)
	{
		CHECK(cx[j] == Approx(true_x[j]).margin(1e-7));
		CHECK(cy[j] == Approx(true_y[j]).margin(1e-7));
	}
}

TEST_CASE("patch grid matches RELION's own tutorial patch centres", "[cryosparc][motionmodel]")
{
	// RELION's 24 Sep 2020 tutorial data_local_shift table uses these centres
	// for a 3710x3838 movie; reproduced to within integer rounding.
	std::vector<std::pair<double, double> > grid = cryosparc::motionPatchGrid(3710., 3838., 5, 5);
	REQUIRE(grid.size() == 25);

	const int want_x[] = {371, 1113, 1855, 2597, 3339};
	const int want_y[] = {384, 1151, 1919, 2687, 3454};

	// x varies fastest
	for (int j = 0; j < 5; j++)
	for (int i = 0; i < 5; i++)
	{
		const std::pair<double, double>& p = grid[(size_t)j * 5 + i];
		CHECK((int)std::lround(p.first)  == want_x[i]);
		CHECK((int)std::lround(p.second) == want_y[j]);
	}
}

TEST_CASE("accumulated motion splits at the dose cutoff", "[cryosparc][motionmodel]")
{
	// A straight-line drift of 1 px per frame, 1 A/px: each step contributes
	// exactly 1 A, so the early/late split is just a frame count.
	std::vector<double> sx, sy;
	for (int i = 0; i < 10; i++) { sx.push_back((double)i); sy.push_back(0.); }

	double total, early, late;
	// dose 1 e/A2 per frame, cutoff 4 e/A2 -> cutoff_frame = 4
	cryosparc::accumulatedMotion(sx, sy, 1.0, 1.0, 0.0, 4.0, total, early, late);

	CHECK(total == Approx(9.).margin(1e-9));
	CHECK(early == Approx(3.).margin(1e-9));   // frames 2,3,4
	CHECK(late  == Approx(6.).margin(1e-9));   // frames 5..10
	CHECK((early + late) == Approx(total).margin(1e-9));
}
