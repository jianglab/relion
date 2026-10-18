/*
 * tests/unit/test_pinned_pool.cpp
 *
 * The pinned buffer pool hands the same buffers to the next thread instead of allocating again, and
 * the pool-off setting is plain allocate and free. Plain malloc stands in for cudaMallocHost.
 */

#include <catch2/catch.hpp>

#include "src/acc/pinned_pool.h"
#ifdef _CUDA_ENABLED
#include "src/acc/cuda/cuda_mem_utils.h"
#endif

#include <cstdlib>
#include <thread>
#include <vector>

namespace
{
int g_allocs = 0, g_frees = 0;
void *countingAlloc(size_t n) { g_allocs++; return malloc(n); }
void countingFree(void *p) { g_frees++; free(p); }
}

TEST_CASE("pinned pool sizes are powers of two from one MB", "[pinned_pool]")
{
	CHECK(pinnedpool::roundUp(1) == ((size_t) 1 << 20));
	CHECK(pinnedpool::roundUp((size_t) 1 << 20) == ((size_t) 1 << 20));
	CHECK(pinnedpool::roundUp(((size_t) 1 << 20) + 1) == ((size_t) 2 << 20));
	CHECK(pinnedpool::roundUp((size_t) 8 << 20) == ((size_t) 8 << 20));
}

TEST_CASE("pinned pool reuses released buffers", "[pinned_pool]")
{
	g_allocs = g_frees = 0;
	pinnedpool::Pool pool(countingAlloc, countingFree, true);
	size_t c1 = 0, c2 = 0, c3 = 0;
	void *a = pool.acquire(3 << 20, c1);
	CHECK(c1 == ((size_t) 4 << 20));
	pool.release(a, c1);
	CHECK(pool.freeBlocks() == 1);
	void *b = pool.acquire(2 << 20, c2); // fits in the 4 MB buffer
	CHECK(b == a);
	CHECK(c2 == c1);
	CHECK(g_allocs == 1);
	void *c = pool.acquire(16 << 20, c3); // too big for anything free: a new one
	CHECK(c != b);
	CHECK(g_allocs == 2);
	CHECK(pool.reuses() == 1);
	pool.release(b, c2);
	pool.release(c, c3);
	CHECK(pool.freeBytes() == c2 + c3);
	// the smallest free buffer that is big enough
	size_t c4 = 0;
	void *d = pool.acquire(1 << 20, c4);
	CHECK(d == b);
	pool.release(d, c4);
	// everything stays in the pool (nothing is freed until the program ends)
	CHECK(g_frees == 0);
}

TEST_CASE("pinned pool gives back what exceeds its limit", "[pinned_pool]")
{
	g_allocs = g_frees = 0;
	pinnedpool::Pool pool(countingAlloc, countingFree, true, (size_t) 4 << 20);
	size_t c1 = 0, c2 = 0;
	void *a = pool.acquire(4 << 20, c1);
	void *b = pool.acquire(4 << 20, c2);
	pool.release(a, c1);
	pool.release(b, c2); // would take the free bytes to 8 MB: freed
	CHECK(g_frees == 1);
	CHECK(pool.freeBytes() == ((size_t) 4 << 20));
}

TEST_CASE("pinned pool switched off allocates and frees every time", "[pinned_pool]")
{
	g_allocs = g_frees = 0;
	pinnedpool::Pool pool(countingAlloc, countingFree, false);
	size_t c = 0;
	void *a = pool.acquire(3 << 20, c);
	CHECK(c == ((size_t) 3 << 20)); // exact size, as the old code did
	pool.release(a, c);
	void *b = pool.acquire(3 << 20, c);
	pool.release(b, c);
	CHECK(g_allocs == 2);
	CHECK(g_frees == 2);
	CHECK(pool.freeBlocks() == 0);
}

TEST_CASE("pinned pool is safe for threads taking and returning buffers", "[pinned_pool]")
{
	g_allocs = g_frees = 0;
	pinnedpool::Pool pool(
		[](size_t n) -> void * { return malloc(n); }, [](void *p) { free(p); }, true);
	std::vector<std::thread> threads;
	for (int t = 0; t < 8; t++)
		threads.emplace_back([&pool]() {
			for (int i = 0; i < 200; i++)
			{
				size_t cap = 0;
				char *p = (char *) pool.acquire((size_t)((i % 5) + 1) << 20, cap);
				p[0] = 1; p[cap - 1] = 1; // the whole buffer is ours
				pool.release(p, cap);
			}
		});
	for (auto &t : threads) t.join();
	// a handful of buffers served 1600 requests
	CHECK(pool.allocations() <= 64);
	CHECK(pool.reuses() >= 1600 - 64);
}

#ifdef _CUDA_ENABLED
TEST_CASE("pinned big copies move the same bytes as cudaMemcpy", "[pinned_pool]")
{
	int ndev = 0;
	if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0)
	{
		cudaGetLastError();
		WARN("no GPU - skipping");
		return;
	}
	// sizes: below the 1 MB limit, not a multiple of the 8 MB chunk, exactly two chunks, and a bit over
	for (size_t bytes : {(size_t) 100 << 10, ((size_t) 1 << 20) + 12, ((size_t) 20 << 20) + 4, (size_t) 16 << 20, ((size_t) 17 << 20) + 8})
	{
		INFO("bytes " << bytes);
		std::vector<unsigned char> in(bytes), out(bytes, 0);
		for (size_t i = 0; i < bytes; i++) in[i] = (unsigned char)(i * 2654435761u >> 13);
		void *d = NULL;
		REQUIRE(cudaMalloc(&d, bytes) == cudaSuccess);
		REQUIRE(pinnedBigUpload(d, in.data(), bytes) == cudaSuccess);
		REQUIRE(pinnedBigDownload(out.data(), d, bytes) == cudaSuccess);
		CHECK(out == in);
		// the upload, read back by the plain copy
		std::fill(out.begin(), out.end(), 0);
		REQUIRE(cudaMemcpy(out.data(), d, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
		CHECK(out == in);
		cudaFree(d);
	}
}

TEST_CASE("asynchronous two-array download returns the bytes of both arrays", "[pinned_pool]")
{
	int ndev = 0;
	if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0)
	{
		cudaGetLastError();
		WARN("no GPU - skipping");
		return;
	}
	for (size_t n1 : {(size_t) 1, (size_t) 1000, (size_t) 300000})
	{
		const size_t n2 = n1 * 4 + 3;
		std::vector<float> a(n1), b(n2);
		for (size_t i = 0; i < n1; i++) a[i] = 0.5f * i;
		for (size_t i = 0; i < n2; i++) b[i] = -1.25f * i;
		float *da = NULL, *db = NULL;
		REQUIRE(cudaMalloc(&da, n1 * sizeof(float)) == cudaSuccess);
		REQUIRE(cudaMalloc(&db, n2 * sizeof(float)) == cudaSuccess);
		REQUIRE(cudaMemcpy(da, a.data(), n1 * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
		REQUIRE(cudaMemcpy(db, b.data(), n2 * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
		{
			AsyncDownload dl;
			REQUIRE(dl.start(da, n1 * sizeof(float), db, n2 * sizeof(float), cudaStreamPerThread));
			dl.wait();
			CHECK(memcmp(dl.first<float>(), a.data(), n1 * sizeof(float)) == 0);
			CHECK(memcmp(dl.second<float>(), b.data(), n2 * sizeof(float)) == 0);
		}
		{
			// destroyed with the copy still pending: waits, gives the buffer back, no crash
			AsyncDownload dl;
			REQUIRE(dl.start(da, n1 * sizeof(float), db, n2 * sizeof(float), cudaStreamPerThread));
		}
		cudaFree(da);
		cudaFree(db);
	}
}
#endif
