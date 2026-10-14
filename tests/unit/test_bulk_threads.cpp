/*
 * tests/unit/test_bulk_threads.cpp
 *
 * Loops over large arrays (CenterFFTbySign, windowFourierTransform, Projector::decenter,
 * Projector::griddingCorrect) run over as many threads as FFTW plans with, outside the particle loop.
 * Every element is computed independently, so the result must be bitwise the same on 1 and on 4 threads.
 */

#include <catch2/catch.hpp>

#include "src/fftw.h"
#include "src/projector.h"

#include <random>

namespace
{
template <class T> bool sameBits(const MultidimArray<T> &a, const MultidimArray<T> &b)
{
	if (!a.sameShape(b)) return false;
	return memcmp(MULTIDIM_ARRAY(a), MULTIDIM_ARRAY(b), MULTIDIM_SIZE(a) * sizeof(T)) == 0;
}

MultidimArray<Complex> randomFourier(int n, unsigned seed)
{
	MultidimArray<Complex> F(n, n, n / 2 + 1);
	std::mt19937 rng(seed);
	std::normal_distribution<RFLOAT> g(0., 1.);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(F) DIRECT_MULTIDIM_ELEM(F, n) = Complex(g(rng), g(rng));
	return F;
}

// Run f on a copy of the input with FFTW (and so the loops) on the given number of threads
template <class T, class Fn> MultidimArray<T> withThreads(int threads, const MultidimArray<T> &in, Fn f)
{
	relionFftwPlanThreads(threads);
	MultidimArray<T> a = in;
	f(a);
	relionFftwPlanThreads(1);
	return a;
}
}

TEST_CASE("large-array loops give the same bits on 1 and 4 threads", "[bulk_threads]")
{
	relionFftwInitThreads();
	const int n = 136;   // 136 x 136 x 69 complex: above the 2^20-element threshold
	const MultidimArray<Complex> F = randomFourier(n, 7);

	auto center = [](MultidimArray<Complex> &a) { CenterFFTbySign(a); };
	CHECK(sameBits(withThreads(1, F, center), withThreads(4, F, center)));

	for (int newdim : {96, 200})
	{
		INFO("window to " << newdim);
		auto window = [newdim](MultidimArray<Complex> &a) { windowFourierTransform(a, newdim); };
		CHECK(sameBits(withThreads(1, F, window), withThreads(4, F, window)));
	}

	{
		Projector P(n / 2, TRILINEAR, 2, 10, 3);
		MultidimArray<Complex> Min(n, n, n / 2 + 1);
		Min = F;
		Min.setXmippOrigin();
		Min.xinit = 0;
		auto dec = [&](MultidimArray<Complex> &a) { P.decenter(Min, a, (n / 2 - 3) * (n / 2 - 3)); };
		MultidimArray<Complex> out(n, n, n / 2 + 1);
		CHECK(sameBits(withThreads(1, out, dec), withThreads(4, out, dec)));

		MultidimArray<RFLOAT> v(n, n, n);
		std::mt19937 rng(3);
		std::normal_distribution<RFLOAT> g(0., 1.);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(v) DIRECT_MULTIDIM_ELEM(v, n) = g(rng);
		auto grid = [&](MultidimArray<RFLOAT> &a) { P.griddingCorrect(a); };
		CHECK(sameBits(withThreads(1, v, grid), withThreads(4, v, grid)));
	}
}
