/*
 * tests/unit/test_gp_motion_fit.cpp
 *
 * GpMotionFit (Polish's track fit): the gradient matches finite differences of the energy, and energy and
 * gradient are bit-identical at any thread count, so Polish gives the same tracks whatever --j is; correlation
 * windows give the same numbers as whole maps while the fit stays inside them, and report when it does not.
 */

#include <catch2/catch.hpp>

#include "src/jaz/single_particle/motion/gp_motion_fit.h"
#include "src/jaz/single_particle/interpolation.h"

#include <cmath>
#include <vector>

using gravis::d2Vector;

namespace {

struct Problem
{
	std::vector<std::vector<Image<double>>> cc;
	std::vector<d2Vector> positions, offsets;
};

// Smooth correlation peaks that move a little from frame to frame
Problem makeProblem(int pc, int fc, int s)
{
	Problem pr;
	pr.cc.resize(pc);

	for (int p = 0; p < pc; p++)
	{
		pr.positions.push_back(d2Vector(37 * p % 400, 91 * p % 400));
		pr.cc[p].resize(fc);

		for (int f = 0; f < fc; f++)
		{
			const double cx = 0.4 * std::sin(p + 0.7 * f), cy = 0.3 * std::cos(2.0 * p - f);
			pr.cc[p][f] = Image<double>(s, s);

			for (int y = 0; y < s; y++)
			for (int x = 0; x < s; x++)
			{
				const double xx = x < s/2? x : x - s, yy = y < s/2? y : y - s;
				const double r2 = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy);
				pr.cc[p][f](y, x) = std::exp(-r2 / 8.0) + 0.01 * std::sin(3.0 * x + 5.0 * y + p);
			}
		}
	}

	pr.offsets = std::vector<d2Vector>(fc, d2Vector(0.0, 0.0));
	return pr;
}

// Tracks with a little motion, as parameters, plus a small perturbation of every parameter
std::vector<double> startParams(const GpMotionFit& fit, int pc, int fc)
{
	std::vector<std::vector<d2Vector>> tracks(pc, std::vector<d2Vector>(fc));
	for (int p = 0; p < pc; p++)
	for (int f = 0; f < fc; f++)
		tracks[p][f] = d2Vector(0.1 * f * std::cos(p), -0.05 * f);

	std::vector<double> x;
	fit.posToParams(tracks, x);
	for (size_t i = 0; i < x.size(); i++) x[i] += 0.05 * std::sin(1.3 * i + 0.2);
	return x;
}

} // namespace

TEST_CASE("GpMotionFit gradient matches finite differences", "[gp_motion_fit]")
{
	const Problem pr = makeProblem(10, 6, 32);
	GpMotionFit fit(pr.cc, 1.0, 1.5, 200.0, 2.0, -1, pr.positions, pr.offsets, 3, true);

	const std::vector<double> x = startParams(fit, 10, 6);
	std::vector<double> g(x.size());
	fit.grad(x, g);

	const double h = 1e-5;

	for (size_t i = 0; i < x.size(); i++)
	{
		std::vector<double> xp = x, xm = x;
		xp[i] += h;
		xm[i] -= h;
		const double num = (fit.f(xp) - fit.f(xm)) / (2.0 * h);
		INFO("parameter " << i);
		REQUIRE(g[i] == Approx(num).epsilon(1e-4).margin(1e-6));
	}
}

TEST_CASE("GpMotionFit energy and gradient do not depend on the thread count", "[gp_motion_fit]")
{
	const Problem pr = makeProblem(23, 7, 32);

	std::vector<double> e(4);
	std::vector<std::vector<double>> g(4);
	const int threads[] = {1, 2, 3, 8};

	for (int k = 0; k < 4; k++)
	{
		GpMotionFit fit(pr.cc, 1.0, 1.5, 200.0, 2.0, -1, pr.positions, pr.offsets, threads[k], true);
		const std::vector<double> x = startParams(fit, 23, 7);

		void* ts = fit.allocateTempStorage();
		e[k] = fit.f(x, ts);
		g[k].resize(x.size());
		fit.grad(x, g[k], ts);

		// with and without temporary storage: the same numbers
		std::vector<double> g2(x.size());
		fit.grad(x, g2);
		REQUIRE(fit.f(x) == e[k]);
		REQUIRE(g2 == g[k]);

		fit.deallocateTempStorage(ts);
	}

	for (int k = 1; k < 4; k++)
	{
		INFO("threads " << threads[k]);
		REQUIRE(e[k] == e[0]);
		REQUIRE(g[k] == g[0]);
	}
}

TEST_CASE("GpMotionFit on correlation windows matches whole maps inside the window", "[gp_motion_fit]")
{
	const int pc = 9, fc = 5, s = 32, w = 12;
	const Problem pr = makeProblem(pc, fc, s);

	// Windows around shift (1, -2) for every particle and frame; shift k is kept in slot k mod w
	std::vector<std::vector<gravis::i2Vector>> centres(pc, std::vector<gravis::i2Vector>(fc, gravis::i2Vector(1, -2)));
	std::vector<std::vector<Image<double>>> win(pc, std::vector<Image<double>>(fc, Image<double>(w, w)));

	for (int p = 0; p < pc; p++)
	for (int f = 0; f < fc; f++)
	for (int ky = -2 - w/2; ky < -2 + w/2; ky++)
	for (int kx = 1 - w/2; kx < 1 + w/2; kx++)
	{
		win[p][f](INTERPOL_WRAP(ky, w), INTERPOL_WRAP(kx, w)) = pr.cc[p][f](INTERPOL_WRAP(ky, s), INTERPOL_WRAP(kx, s));
	}

	GpMotionFit full(pr.cc, 1.0, 1.5, 200.0, 2.0, -1, pr.positions, pr.offsets, 4, true);
	GpMotionFit cropped(win, 1.0, 1.5, 200.0, 2.0, -1, pr.positions, pr.offsets, 4, true);
	cropped.setWindow(&centres, w);

	const std::vector<double> x = startParams(full, pc, fc);
	std::vector<double> gf(x.size()), gc(x.size());
	full.grad(x, gf);
	cropped.grad(x, gc);

	REQUIRE(cropped.f(x) == full.f(x));
	REQUIRE(gc == gf);
	REQUIRE_FALSE(cropped.leftWindow());

	// a start position 5 pixels away needs pixels outside the +-6 pixel window
	std::vector<double> far = x;
	far[0] += 5.0;
	cropped.f(far);
	REQUIRE(cropped.leftWindow());
}
