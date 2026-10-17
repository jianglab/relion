/*
 * tests/unit/test_async_serial.cpp
 *
 * RELION_ASYNC_SERIAL helpers: the background zero-fill, the deferred zeroing of a BackProjector
 * and the per-thread FFTW planner thread count.
 */

#include <catch2/catch.hpp>

#include "src/async_serial.h"
#include "src/backprojector.h"
#include "src/fftw.h"
#include "src/multidim_array.h"

#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

TEST_CASE("DeferredZero zeroes every span and joins", "[async_serial]")
{
	std::vector<unsigned char> a(1000, 0xAB), b((size_t) 40 << 20, 0xCD), c(7, 0xEF);
	std::vector<asyncserial::ZeroSpan> spans = {{a.data(), a.size()}, {b.data(), b.size()}, {c.data(), c.size()}};
	asyncserial::DeferredZero z;
	REQUIRE(z.start(spans, 4));
	CHECK(z.pending());
	CHECK(z.wait() >= 0.);
	CHECK_FALSE(z.pending());
	for (unsigned char v : a) REQUIRE(v == 0);
	for (size_t i = 0; i < b.size(); i += 4093) REQUIRE(b[i] == 0);
	CHECK(b.back() == 0);
	for (unsigned char v : c) REQUIRE(v == 0);
	CHECK(z.wait() == 0.);
}

TEST_CASE("DeferredZero: empty, null and repeated starts are safe", "[async_serial]")
{
	asyncserial::DeferredZero z;
	CHECK(z.start({}, 2));
	CHECK_FALSE(z.pending());
	std::vector<asyncserial::ZeroSpan> nulls = {{NULL, 100}, {NULL, 0}};
	REQUIRE(z.start(nulls, 2));
	z.wait();

	std::vector<unsigned char> a(4096, 1), b(4096, 2);
	REQUIRE(z.start({{a.data(), a.size()}}, 1));
	// a second start waits for the first
	REQUIRE(z.start({{b.data(), b.size()}}, 1));
	z.wait();
	CHECK(a[100] == 0);
	CHECK(b[100] == 0);
}

TEST_CASE("DeferredZero: the destructor joins a running fill", "[async_serial]")
{
	std::vector<unsigned char> a((size_t) 64 << 20, 9);
	{
		asyncserial::DeferredZero z;
		REQUIRE(z.start({{a.data(), a.size()}}, 2));
	}
	CHECK(a.front() == 0);
	CHECK(a.back() == 0);
}

TEST_CASE("asyncserial::helperThreads follows the job and the override", "[async_serial]")
{
	unsetenv("RELION_ASYNC_SERIAL_THREADS");
	CHECK(asyncserial::helperThreads(6) == 6);
	CHECK(asyncserial::helperThreads(0) == 1);
	setenv("RELION_ASYNC_SERIAL_THREADS", "3", 1);
	CHECK(asyncserial::helperThreads(6) == 3);
	setenv("RELION_ASYNC_SERIAL_THREADS", "junk", 1);
	CHECK(asyncserial::helperThreads(6) == 6);
	unsetenv("RELION_ASYNC_SERIAL_THREADS");
}

TEST_CASE("BackProjector::initZerosDeferred sizes now and zeroes later, like initZeros", "[async_serial]")
{
	BackProjector sync_bp(32, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
	BackProjector lazy_bp(32, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
	sync_bp.initZeros(-1);

	std::vector<asyncserial::ZeroSpan> spans;
	lazy_bp.initZerosDeferred(-1, spans);
	REQUIRE(spans.size() == 2);

	// the dimensions are already final: GPU setup reads them before the fill finishes
	CHECK(lazy_bp.data.nzyxdim == sync_bp.data.nzyxdim);
	CHECK(lazy_bp.weight.nzyxdim == sync_bp.weight.nzyxdim);
	CHECK(XSIZE(lazy_bp.data) == XSIZE(sync_bp.data));
	CHECK(ZSIZE(lazy_bp.data) == ZSIZE(sync_bp.data));

	// dirty the arrays, then let the background fill clean them
	for (long int n = 0; n < lazy_bp.data.nzyxdim; n++) DIRECT_MULTIDIM_ELEM(lazy_bp.data, n) = Complex(1., 2.);
	for (long int n = 0; n < lazy_bp.weight.nzyxdim; n++) DIRECT_MULTIDIM_ELEM(lazy_bp.weight, n) = 3.;
	asyncserial::DeferredZero z;
	REQUIRE(z.start(spans, 2));
	z.wait();
	for (long int n = 0; n < lazy_bp.data.nzyxdim; n++)
	{
		REQUIRE(DIRECT_MULTIDIM_ELEM(lazy_bp.data, n).real == DIRECT_MULTIDIM_ELEM(sync_bp.data, n).real);
		REQUIRE(DIRECT_MULTIDIM_ELEM(lazy_bp.data, n).imag == DIRECT_MULTIDIM_ELEM(sync_bp.data, n).imag);
	}
	for (long int n = 0; n < lazy_bp.weight.nzyxdim; n++)
		REQUIRE(DIRECT_MULTIDIM_ELEM(lazy_bp.weight, n) == DIRECT_MULTIDIM_ELEM(sync_bp.weight, n));
}

#if defined(RELION_HAVE_FFTW_THREADS) && !defined(MKLFFT) && !defined(RELION_SINGLE_PRECISION)
namespace
{
MultidimArray<RFLOAT> wave(int n)
{
	MultidimArray<RFLOAT> v(n, n, n);
	for (long int i = 0; i < v.nzyxdim; i++)
		DIRECT_MULTIDIM_ELEM(v, i) = std::sin(0.37 * i) + 0.01 * (i % 17);
	return v;
}
}

TEST_CASE("FFTW per-thread plan thread count: transforms equal and the global count is restored", "[async_serial]")
{
	relionFftwInitThreads();
	relionFftwPlanThreads(1);
	const int n = 48;
	MultidimArray<RFLOAT> v = wave(n);
	MultidimArray<Complex> ref;
	{
		FourierTransformer t;
		t.FourierTransform(v, ref, true);
	}

	MultidimArray<Complex> other;
	std::thread th([&]()
	{
		relionFftwSetThreadPlanThreads(4);
		FourierTransformer t;
		t.FourierTransform(v, other, true);
		relionFftwSetThreadPlanThreads(0);
	});
	th.join();

	// the helper's setting never leaks into the planner the main thread uses
	CHECK(fftw_planner_nthreads() == 1);
	REQUIRE(other.nzyxdim == ref.nzyxdim);
	for (long int i = 0; i < ref.nzyxdim; i++)
	{
		REQUIRE(DIRECT_MULTIDIM_ELEM(other, i).real == Approx(DIRECT_MULTIDIM_ELEM(ref, i).real).margin(1e-9));
		REQUIRE(DIRECT_MULTIDIM_ELEM(other, i).imag == Approx(DIRECT_MULTIDIM_ELEM(ref, i).imag).margin(1e-9));
	}

	// main thread changing the global count while a helper plans: both finish, no deadlock
	std::thread th2([&]()
	{
		relionFftwSetThreadPlanThreads(2);
		for (int k = 0; k < 20; k++)
		{
			FourierTransformer t;
			MultidimArray<Complex> F;
			t.FourierTransform(v, F, false);
		}
	});
	for (int k = 0; k < 200; k++) relionFftwPlanThreads(1 + (k % 3));
	th2.join();
	relionFftwPlanThreads(1);
	CHECK(fftw_planner_nthreads() == 1);
}
#endif
