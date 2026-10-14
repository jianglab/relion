/*
 * tests/unit/test_fftw_plans.cpp
 *
 * Measured FFTW plans (RELION_FFTW_PLANS=measure): measuring must not touch the caller's arrays,
 * the transforms must agree with the FFTW_ESTIMATE ones to rounding, and the wisdom file must be
 * written.
 */

#include <catch2/catch.hpp>

#include "src/fftw.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace
{
MultidimArray<RFLOAT> randomArray(int z, int y, int x, unsigned seed)
{
	MultidimArray<RFLOAT> a(z, y, x);
	std::mt19937 rng(seed);
	std::normal_distribution<RFLOAT> g(0., 1.);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a) DIRECT_MULTIDIM_ELEM(a, n) = g(rng);
	return a;
}

// Forward then inverse transform
void transformBoth(const MultidimArray<RFLOAT> &in, MultidimArray<Complex> &F, MultidimArray<RFLOAT> &back)
{
	MultidimArray<RFLOAT> work = in;
	FourierTransformer t;
	t.FourierTransform(work, F, true);
	// the input must be unchanged by planning and transforming
	bool same = true;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(in)
		if (DIRECT_MULTIDIM_ELEM(in, n) != DIRECT_MULTIDIM_ELEM(work, n)) same = false;
	CHECK(same);
	back = work;
	t.inverseFourierTransform(F, back);
}
}

TEST_CASE("measured FFTW plans: same transforms, input untouched, wisdom saved", "[fftw_plans]")
{
	const char *tmp = getenv("TMPDIR");
	const std::string dir = std::string(tmp != NULL && tmp[0] ? tmp : "/tmp") + "/relion_fftw_plans_test_" + std::to_string((long)getpid());
	mkdir(dir.c_str(), 0700);
	const std::string wisdom = dir + "/wisdom.txt";

	for (auto dims : {std::vector<int>{1, 50, 50}, std::vector<int>{24, 24, 24}, std::vector<int>{1, 36, 36}})
	{
		INFO("size " << dims[0] << "x" << dims[1] << "x" << dims[2]);
		const MultidimArray<RFLOAT> a = randomArray(dims[0], dims[1], dims[2], 3);
		MultidimArray<Complex> Fe, Fm;
		MultidimArray<RFLOAT> be, bm;
		relionFftwSetPlanMode(false, "");
		transformBoth(a, Fe, be);
		relionFftwSetPlanMode(true, wisdom, 5.);
		transformBoth(a, Fm, bm);
		relionFftwSetPlanMode(false, "");
		REQUIRE(Fe.sameShape(Fm));
		double maxdiff = 0., maxabs = 0.;
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fe)
		{
			maxdiff = std::max(maxdiff, (double)abs(DIRECT_MULTIDIM_ELEM(Fe, n) - DIRECT_MULTIDIM_ELEM(Fm, n)));
			maxabs = std::max(maxabs, (double)abs(DIRECT_MULTIDIM_ELEM(Fe, n)));
		}
		CHECK(maxdiff <= 1e-5 * maxabs);
		double backdiff = 0.;
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(be)
			backdiff = std::max(backdiff, (double)fabs(DIRECT_MULTIDIM_ELEM(be, n) - DIRECT_MULTIDIM_ELEM(bm, n)));
		CHECK(backdiff <= 1e-5);
	}
	struct stat st;
	REQUIRE(stat(wisdom.c_str(), &st) == 0);
	CHECK(st.st_size > 0);
	remove(wisdom.c_str());
	rmdir(dir.c_str());
}

#if defined(RELION_HAVE_FFTW_THREADS) && !defined(MKLFFT) && !defined(RELION_SINGLE_PRECISION)
// FourierTransformer::cleanup() calls fftw_cleanup(), which replaced FFTW's planner and so reset the
// number of threads to 1: every later transform of relion_refine ran on one core
TEST_CASE("FourierTransformer::cleanup keeps the FFTW thread count", "[fftw_plans]")
{
	relionFftwInitThreads();
	relionFftwPlanThreads(4);
	{
		MultidimArray<RFLOAT> v = randomArray(1, 32, 32, 2);
		MultidimArray<Complex> F;
		FourierTransformer t;
		t.FourierTransform(v, F, false);
		t.cleanup();
	}
	CHECK(fftw_planner_nthreads() == 4);

	// ... and FFTW's threaded algorithms: fftw_cleanup() alone left a planner without them, so plans
	// were mostly serial even with the thread count set
	auto planText = []()
	{
		const int n = 96;
		double *r = fftw_alloc_real((size_t)n * n * n);
		fftw_complex *c = fftw_alloc_complex((size_t)n * n * (n / 2 + 1));
		fftw_plan p = fftw_plan_dft_r2c_3d(n, n, n, r, c, FFTW_ESTIMATE);
		char *txt = fftw_sprint_plan(p);
		std::string out(txt);
		free(txt);
		fftw_destroy_plan(p);
		fftw_free(r);
		fftw_free(c);
		return out;
	};
	const std::string before = planText();
	{
		FourierTransformer t;
		MultidimArray<RFLOAT> v = randomArray(1, 16, 16, 3);
		MultidimArray<Complex> F;
		t.FourierTransform(v, F, false);
		t.cleanup();
	}
	const std::string after = planText();
	INFO("plan before cleanup: " << before.substr(0, 200));
	INFO("plan after cleanup: " << after.substr(0, 200));
	if (before.find("thr") != std::string::npos)
		CHECK(after.find("thr") != std::string::npos);
	CHECK(fftw_planner_nthreads() == 4);
	relionFftwPlanThreads(1);
}
#endif

// Timing only (hidden: run with [.fftw_timing]): RELION's FourierTransformer on a large 3D volume,
// with FFTW on 1 and on 8 threads
TEST_CASE("FourierTransformer timing with FFTW threads", "[.fftw_timing]")
{
	const char *ns = getenv("FFTW_TIMING_SIZE");
	const int n = ns ? atoi(ns) : 384;
	relionFftwInitThreads();
	for (int threads : {1, 8})
	{
		relionFftwPlanThreads(threads);
		MultidimArray<RFLOAT> v = randomArray(n, n, n, 1), back;
		MultidimArray<Complex> F;
		FourierTransformer t;
		auto t0 = std::chrono::steady_clock::now();
		t.FourierTransform(v, F, false);
		auto t1 = std::chrono::steady_clock::now();
		t.inverseFourierTransform();
		auto t2 = std::chrono::steady_clock::now();
		std::cout << n << "^3, " << threads << " threads: forward (incl. plan) "
		          << std::chrono::duration<double>(t1 - t0).count() << " s, inverse "
		          << std::chrono::duration<double>(t2 - t1).count() << " s" << std::endl;
	}
	relionFftwPlanThreads(1);
}
