/*
 * tests/unit/test_gpu_pool_plan.cpp
 *
 * The sizing rules of the lean GPU memory pool (src/acc/gpu_pool_plan.h), which need no GPU, and the pool
 * itself (CudaCustomAllocator in lean mode), skipped when no GPU is visible.
 */

#include <catch2/catch.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>

#include "src/acc/gpu_pool_plan.h"
#include "src/tuning_cache.h"

using namespace gpupool;

namespace
{
const size_t MB = (size_t) 1 << 20;
const size_t GB = (size_t) 1 << 30;
}

TEST_CASE("lean pool margin grows with the change between peaks", "[gpu_pool]")
{
	CHECK(marginFor(0, 0) == Approx(1.2));
	CHECK(marginFor(0, 5 * GB) == Approx(1.2));            // one measurement
	CHECK(marginFor(5 * GB, 5 * GB) == Approx(1.15));      // steady: fragmentation only
	CHECK(marginFor(4 * GB, 5 * GB) == Approx(1.25));      // r = 20%: 2r is capped at +10%
	CHECK(marginFor(100 * MB, 102 * MB) > 1.15);
	CHECK(marginFor(100 * MB, 102 * MB) < 1.2);
	// symmetric in the direction of the change
	CHECK(marginFor(5 * GB, 4 * GB) == Approx(marginFor(4 * GB, 5 * GB)));
	for (size_t a : {1 * MB, 3 * GB})
		for (size_t b : {1 * MB, 7 * MB, 3 * GB, 40 * GB})
		{
			CHECK(marginFor(a, b) >= 1.15);
			CHECK(marginFor(a, b) <= 1.25);
		}
}

TEST_CASE("reference transform block: buffers plus 2% and 16 MB, far below the old 50% margin", "[gpu_pool]")
{
	const size_t GB = (size_t) 1 << 30;
	const size_t sum = 11 * GB;
	const size_t lean = gpupool::transformPoolBytes(sum);
	CHECK(lean > sum);
	CHECK(lean < sum + sum / 20);
	CHECK(lean < (size_t) (1.5 * (double) sum) - 4 * GB);
	CHECK(gpupool::transformPoolBytes(0) == ((size_t) 16 << 20));
}

TEST_CASE("lean pool plan: first chunk from the peak, never above the cap", "[gpu_pool]")
{
	const size_t cap = 20 * GB;

	SECTION("nothing measured: a small chunk, 1/64 of the cap but at least the minimum")
	{
		Plan p = plan(cap, 0, 0, 0);
		CHECK(p.first_chunk == cap / 64);
		CHECK(p.margin == 0.);
		CHECK(std::string(p.reason) == "no measurement yet");
		CHECK(plan(100 * MB, 0, 0, 0).first_chunk == kMinChunk);
		CHECK(plan(10 * MB, 0, 0, 0).first_chunk == 10 * MB); // cap below the minimum chunk
	}
	SECTION("previous iteration: peak times margin")
	{
		Plan p = plan(cap, 3 * GB, 3 * GB, 0);
		CHECK(p.margin == Approx(1.15));
		CHECK(p.first_chunk == (size_t) std::ceil(3. * GB * 1.15));
		CHECK(std::string(p.reason) == "peak of the previous iteration");
		Plan q = plan(cap, 3 * GB, 0, 0);
		CHECK(q.margin == Approx(1.2));
	}
	SECTION("remembered from an earlier run when there is no previous iteration")
	{
		Plan p = plan(cap, 0, 0, 2 * GB);
		CHECK(p.margin == Approx(1.2));
		CHECK(p.first_chunk == (size_t) std::ceil(2. * GB * 1.2));
		CHECK(std::string(p.reason) == "peak remembered from an earlier run");
		// the iteration before wins over a remembered value
		CHECK(plan(cap, 1 * GB, 1 * GB, 2 * GB).first_chunk == (size_t) std::ceil(1. * GB * 1.15));
	}
	SECTION("a peak close to the cap is limited by it")
	{
		CHECK(plan(cap, 19 * GB, 19 * GB, 0).first_chunk == cap);
		CHECK(plan(cap, 40 * GB, 0, 0).first_chunk == cap);
	}
	SECTION("a tiny peak still takes the minimum chunk")
	{
		CHECK(plan(cap, 1 * MB, 1 * MB, 0).first_chunk == kMinChunk);
	}
}

TEST_CASE("lean pool growth: geometric steps within the room", "[gpu_pool]")
{
	// the request does not fit in the room: cannot grow
	CHECK(nextChunk(1 * GB, 3 * GB, 2 * GB) == 0);
	CHECK(nextChunk(0, 1, 0) == 0);
	// at least the minimum, at least the request, half of what is held otherwise
	CHECK(nextChunk(0, 1 * MB, 10 * GB) == kMinChunk);
	CHECK(nextChunk(0, 300 * MB, 10 * GB) == 300 * MB);
	CHECK(nextChunk(4 * GB, 1 * MB, 10 * GB) == 2 * GB);
	// never above the room, but never below the request
	CHECK(nextChunk(4 * GB, 1 * MB, 1 * GB) == 1 * GB);
	CHECK(nextChunk(4 * GB, 1 * GB, 1 * GB) == 1 * GB);

	SECTION("growing to a need of N bytes takes a logarithmic number of chunks and respects the cap")
	{
		const size_t cap = 24 * GB;
		for (size_t need : {200 * MB, 1 * GB, 7 * GB, 20 * GB})
		{
			size_t held = 0, chunks = 0;
			while (held < need)
			{
				size_t c = nextChunk(held, 64 * MB, cap - held);
				REQUIRE(c > 0);
				held += c;
				chunks++;
				REQUIRE(held <= cap);
			}
			INFO("need " << need << " chunks " << chunks);
			CHECK(chunks <= 16); // steps of 1.5x from 64 MB
			// overshoot: at most half as much again, plus one minimum chunk
			CHECK(held <= need + need / 2 + kMinChunk);
		}
	}
}

TEST_CASE("lean pool keeps its chunks only when they fit the target", "[gpu_pool]")
{
	CHECK(keepChunks(1, 10 * GB, 10 * GB));
	CHECK(keepChunks(1, 12 * GB, 10 * GB));
	CHECK_FALSE(keepChunks(1, 13 * GB + 1, 10 * GB)); // holding 30% more than needed: give some back
	CHECK_FALSE(keepChunks(1, 9 * GB, 10 * GB));      // too small
	CHECK_FALSE(keepChunks(2, 10 * GB, 10 * GB));     // pieces: join them into one
	CHECK_FALSE(keepChunks(0, 0, 10 * GB));
}

TEST_CASE("lean pool problem key names the sizes and nothing else", "[gpu_pool]")
{
	Problem p;
	p.ori_size = 256; p.nr_classes = 4; p.ref_dim = 3; p.model_bytes = 123456789;
	p.threads_on_device = 8; p.ranks_on_device = 2; p.current_size = 120; p.coarse_size = 64;
	p.nr_orientations = 3686400; p.nr_pool = 8; p.cap_class = capClass(5 * GB);
	const std::string k = problemKey(p);
	CHECK(k == "box=256;K=4;dim=3;model=123456789;thr=8;ranks=2;cur=120;coarse=64;orient=3686400;pool=8;cap=49");
	CHECK(k.find(',') == std::string::npos); // it is a CSV field
	CHECK(k.find('/') == std::string::npos); // no path
	Problem q = p;
	q.current_size = 124;
	CHECK(problemKey(q) != k);
	q = p; q.ranks_on_device = 1;
	CHECK(problemKey(q) != k);
}

TEST_CASE("lean pool cap class: a peak is remembered per limit, not across limits", "[gpu_pool]")
{
	CHECK(capClass(0) == 0);
	CHECK(capClass(1 * MB) == 0);
	CHECK(capClass(2 * MB) == 4);
	CHECK(capClass(1024 * MB) == 40);
	CHECK(capClass(2048 * MB) == capClass(1024 * MB) + 4); // one octave is four classes
	// monotone, and another process taking or giving back a few per cent keeps the class
	CHECK(capClass(5 * GB) >= capClass(4 * GB));
	CHECK(capClass(5 * GB) == capClass((size_t) (4.9 * (double) GB)));
	CHECK(capClass(1081 * MB) != capClass(20 * GB));

	Problem big, small;
	big.ori_size = small.ori_size = 128; big.nr_classes = small.nr_classes = 3; big.ref_dim = small.ref_dim = 3;
	big.cap_class = capClass(20 * GB);
	small.cap_class = capClass(1081 * MB);
	CHECK(problemKey(big) != problemKey(small));
	big.cap_class = small.cap_class;
	CHECK(problemKey(big) == problemKey(small));

	// A peak taken at a large limit (1241 MB there) is not found under the small one, so the first chunk comes
	// from "no measurement yet" (small, grows as needed), not from that peak clamped up to the limit.
	const size_t cap = 1081 * MB;
	const Plan from_other_limit = plan(cap, 0, 0, 0);
	CHECK(std::string(from_other_limit.reason) == "no measurement yet");
	CHECK(from_other_limit.first_chunk < cap / 4);
	// the same peak found under its own limit still gives the old rule: 243 MB * 1.2 = 292 MB, not the limit
	const Plan own = plan(cap, 0, 0, 243 * MB);
	CHECK(own.first_chunk == (size_t) std::ceil(243. * (double) MB * 1.2));
	CHECK(own.first_chunk < cap / 3);
}

TEST_CASE("lean pool peak survives a tuning cache round trip, per GPU and problem", "[gpu_pool][tuning_cache]")
{
	const char *home = getenv("HOME"), *tmpd = getenv("TMPDIR");
	std::string base = home ? std::string(home) + "/temp" : (tmpd ? tmpd : ".");
	struct stat st;
	if (stat(base.c_str(), &st) != 0)
		base = tmpd ? tmpd : ".";
	std::string path = base + "/gpu_pool_peak_test_" + std::to_string(getpid()) + ".csv";
	std::remove(path.c_str());
	tuning::Context a; a.device = "Test GPU A"; a.sm_count = 48; a.driver = "12000"; a.runtime = "12040"; a.build = "5.1.0";
	tuning::Context b = a; b.device = "Test GPU B";
	Problem p; p.ori_size = 64; p.nr_classes = 4; p.ref_dim = 3; p.model_bytes = 1000; p.threads_on_device = 4;
	const std::string key = problemKey(p);
	{
		tuning::Cache c(a, path);
		tuning::Entry e;
		CHECK_FALSE(c.get("gpu_pool_peak", key, e));
		REQUIRE(c.put("gpu_pool_peak", key, std::to_string(1500 * MB)));
		REQUIRE(c.get("gpu_pool_peak", key, e));
		CHECK((size_t) strtoull(e.value.c_str(), NULL, 10) == 1500 * MB);
		// a later, larger peak replaces it
		REQUIRE(c.put("gpu_pool_peak", key, std::to_string(1800 * MB)));
		REQUIRE(c.get("gpu_pool_peak", key, e));
		CHECK((size_t) strtoull(e.value.c_str(), NULL, 10) == 1800 * MB);
		// another problem: no row
		Problem q = p; q.nr_classes = 5;
		CHECK_FALSE(c.get("gpu_pool_peak", problemKey(q), e));
		// another GPU: no row
		CHECK_FALSE(tuning::Cache(b, path).get("gpu_pool_peak", key, e));
	}
	std::remove(path.c_str());
	std::remove((path + ".lock").c_str());
}

// ---- the pool itself (needs a GPU) -----------------------------------------------------------------

#ifdef _CUDA_ENABLED
#include "src/acc/cuda/custom_allocator.cuh"

namespace
{
bool gpuVisible()
{
	int n = 0;
	return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

size_t freeGpuBytes()
{
	size_t free_b = 0, total_b = 0;
	return cudaMemGetInfo(&free_b, &total_b) == cudaSuccess ? free_b : 0;
}

// A pool test that finds the GPU out of memory because another process took it during the test (a shared
// GPU, several jobs at once) is not a failure of the pool: if less than `need` bytes are free now, say so and
// give up. With enough memory free, the error is a real one and is thrown again.
void skipIfGpuTaken(const RelionError &e, size_t need)
{
	if (freeGpuBytes() >= need)
		throw;
	WARN("GPU memory was taken by another process during the test (" << freeGpuBytes() / MB << " MB free, "
	     << need / MB << " MB needed): test skipped. Error: " << e.msg);
}
}

TEST_CASE("lean CudaCustomAllocator holds a small chunk, grows on demand within its cap", "[gpu_pool][gpu_allocator]")
{
	if (!gpuVisible())
	{
		WARN("no GPU visible: test skipped");
		return;
	}
	const size_t first_chunk = 128 * MB, block = 48 * MB;
	// The test holds 8 blocks at once (384 MB) and the chunks it grows are not filled to the last byte, so the
	// cap must be well above that: 1.5 times, 576 MB. A third of the free memory is left to other processes.
	const size_t need = 8 * block * 3 / 2;
	const size_t cap = std::min<size_t>(768 * MB, freeGpuBytes() / 3);
	if (cap < need)
	{
		WARN("too little free GPU memory (" << freeGpuBytes() / MB << " MB free, " << 3 * need / MB << " MB needed): test skipped");
		return;
	}
	try {

	CudaCustomAllocator pool(0, 512);
	pool.configureLean(cap, first_chunk);
	CHECK(pool.isLean());
	CHECK(pool.chunkSize() == first_chunk);
	CHECK(pool.capSize() == cap);
	CHECK(pool.numberOfChunks() == 1);
	CHECK(pool.getPotentialFreeSpace() == first_chunk + (cap - first_chunk));

	// 8 blocks of 48 MB do not fit in 128 MB: chunks are added, with no waiting (nothing is pending)
	std::vector<CudaCustomAllocator::Alloc*> blocks;
	for (int i = 0; i < 8; i++)
	{
		blocks.push_back(pool.alloc(block));
		REQUIRE(blocks.back()->getPtr() != NULL);
		// the memory is real: write and read back
		REQUIRE(cudaMemset(blocks.back()->getPtr(), i + 1, block) == cudaSuccess);
	}
	CHECK(pool.chunkSize() > first_chunk);
	CHECK(pool.chunkSize() <= cap);
	CHECK(pool.numberOfChunks() > 1);
	CHECK(pool.chunksGrown() == pool.numberOfChunks() - 1);
	CHECK(pool.peakUsedBytes() >= 8 * block);
	CHECK(pool.peakHeldBytes() == pool.chunkSize());
	CHECK(pool.getTotalUsedSpace() == pool.peakUsedBytes());

	// no block was overwritten by another
	for (int i = 0; i < 8; i++)
	{
		std::vector<unsigned char> h(4096);
		REQUIRE(cudaMemcpy(h.data(), blocks[i]->getPtr() + block - 4096, 4096, cudaMemcpyDeviceToHost) == cudaSuccess);
		for (unsigned char c : h)
			REQUIRE((int) c == i + 1);
	}

	for (auto *b : blocks)
		pool.free(b);
	CHECK(pool.getNumberOfAllocs() == 0);
	CHECK(pool.getTotalUsedSpace() == 0);
	CHECK(pool.getTotalFreeSpace() == pool.chunkSize());
	// free blocks do not merge across chunks: the largest piece is a chunk, not the whole
	CHECK(pool.getLargestContinuousFreeSpace() < pool.chunkSize());

	// next iteration: first chunk from the measured peak, in one piece, no growth needed
	const size_t peak = pool.peakUsedBytes();
	Plan p = plan(cap, peak, peak, 0);
	pool.configureLean(cap, p.first_chunk);
	CHECK(pool.numberOfChunks() == 1);
	CHECK(pool.chunkSize() == p.first_chunk);
	CHECK(pool.peakUsedBytes() == 0);
	for (int i = 0; i < 8; i++)
		blocks[i] = pool.alloc(block);
	CHECK(pool.chunksGrown() == 0);
	CHECK(pool.numberOfChunks() == 1);
	for (auto *b : blocks)
		pool.free(b);

	// same target again: the chunk is kept as it is
	cudaDeviceSynchronize();
	pool.configureLean(cap, p.first_chunk);
	CHECK(pool.numberOfChunks() == 1);
	CHECK(pool.chunkSize() == p.first_chunk);

	// a smaller peak gives memory back
	pool.configureLean(cap, kMinChunk);
	CHECK(pool.chunkSize() == kMinChunk);
	CHECK(pool.getNumberOfAllocs() == 0);
	} catch (const RelionError &e) { skipIfGpuTaken(e, need * 4 / 3); }
}

TEST_CASE("lean CudaCustomAllocator never holds more than its cap", "[gpu_pool][gpu_allocator]")
{
	if (!gpuVisible())
	{
		WARN("no GPU visible: test skipped");
		return;
	}
	// 15 blocks of 16 MB = 240 MB must fit under the cap, so the cap is the full 256 MB, and a third of the
	// free memory is left to other processes
	const size_t cap = std::min<size_t>(256 * MB, freeGpuBytes() / 3);
	if (cap < 256 * MB)
	{
		WARN("too little free GPU memory (" << freeGpuBytes() / MB << " MB free, " << 768 << " MB needed): test skipped");
		return;
	}
	try {
	CudaCustomAllocator pool(0, 512);
	pool.configureLean(cap, 64 * MB);
	std::vector<CudaCustomAllocator::Alloc*> blocks;
	// 15 blocks of 16 MB = 240 MB: fits under the cap of 256 MB, in several chunks
	for (int i = 0; i < 15; i++)
		blocks.push_back(pool.alloc(16 * MB));
	CHECK(pool.chunkSize() <= cap);
	CHECK(pool.getPotentialFreeSpace() == pool.getTotalFreeSpace() + (cap - pool.chunkSize()));
	for (auto *b : blocks)
		pool.free(b);
	} catch (const RelionError &e) { skipIfGpuTaken(e, 3 * cap / 2); }
}

TEST_CASE("the old pool is a single chunk and its cap is its size", "[gpu_pool][gpu_allocator]")
{
	if (!gpuVisible())
	{
		WARN("no GPU visible: test skipped");
		return;
	}
	if (freeGpuBytes() < 4 * 64 * MB)
	{
		WARN("too little free GPU memory (" << freeGpuBytes() / MB << " MB free, " << 4 * 64 << " MB needed): test skipped");
		return;
	}
	CudaCustomAllocator pool(0, 512);
	CHECK(pool.chunkSize() == 0);
	CHECK_FALSE(pool.isLean());
	pool.resize(64 * MB);
	CHECK_FALSE(pool.isLean());
	if (pool.chunkSize() < 64 * MB)
	{
		// The pool backs off to what is free when another process took the memory since the check above
		// (the designed behavior, it says so on the output): that is not what this test is about.
		WARN("GPU memory was taken by another process: the pool was reduced to " << pool.chunkSize() / MB << " MB, test skipped");
		return;
	}
	CHECK(pool.chunkSize() == 64 * MB);
	CHECK(pool.capSize() == 64 * MB);
	CHECK(pool.numberOfChunks() == 1);
	CHECK(pool.getPotentialFreeSpace() == 64 * MB);
	auto *a = pool.alloc(10 * MB);
	CHECK(pool.peakUsedBytes() == 10 * MB);
	pool.free(a);
	CHECK(pool.getTotalFreeSpace() == 64 * MB);
	CHECK(pool.getLargestContinuousFreeSpace() == 64 * MB);
	CHECK(pool.chunksGrown() == 0);
}

#include <cufft.h>

TEST_CASE("in-place real-to-complex 3D transform gives the same numbers as the out-of-place one", "[gpu_pool][gpu_inplace_fft]")
{
	if (!gpuVisible())
	{
		WARN("no GPU visible: test skipped");
		return;
	}
	for (int N : {48, 64, 100, 128})
	{
		const size_t nx2 = N / 2 + 1, nreal = (size_t) N * N * N, ncplx = (size_t) N * N * nx2;
		size_t free_b = 0, total_b = 0;
		REQUIRE(cudaMemGetInfo(&free_b, &total_b) == cudaSuccess);
		if (free_b < 6 * ncplx * 16 + ((size_t) 64 << 20))
		{
			WARN("too little free GPU memory: test skipped");
			return;
		}
		std::vector<double> h(nreal), hp(ncplx * 2, 0.);
		unsigned long long r = 12345;
		for (auto &x : h)
		{
			r = r * 6364136223846793005ULL + 1442695040888963407ULL;
			x = (double) (r >> 11) / 9007199254740992.0 - 0.5;
		}
		for (size_t z = 0; z < (size_t) N; z++)
			for (size_t y = 0; y < (size_t) N; y++)
				for (size_t x = 0; x < (size_t) N; x++)
					hp[(z * N + y) * 2 * nx2 + x] = h[(z * N + y) * N + x];

		double *dm = NULL, *dip = NULL;
		cufftDoubleComplex *dout = NULL;
		REQUIRE(cudaMalloc(&dm, nreal * 8) == cudaSuccess);
		REQUIRE(cudaMalloc(&dout, ncplx * 16) == cudaSuccess);
		REQUIRE(cudaMalloc(&dip, ncplx * 16) == cudaSuccess);
		REQUIRE(cudaMemcpy(dm, h.data(), nreal * 8, cudaMemcpyHostToDevice) == cudaSuccess);
		REQUIRE(cudaMemcpy(dip, hp.data(), ncplx * 16, cudaMemcpyHostToDevice) == cudaSuccess);

		int n[3] = {N, N, N};
		cufftHandle plan;
		size_t ws = 0;
		REQUIRE(cufftCreate(&plan) == CUFFT_SUCCESS);
		REQUIRE(cufftMakePlanMany(plan, 3, n, NULL, 0, 0, NULL, 0, 0, CUFFT_D2Z, 1, &ws) == CUFFT_SUCCESS);
		REQUIRE(cufftExecD2Z(plan, dm, dout) == CUFFT_SUCCESS);
		REQUIRE(cufftExecD2Z(plan, dip, (cufftDoubleComplex *) dip) == CUFFT_SUCCESS);
		std::vector<double> a(ncplx * 2), b(ncplx * 2);
		REQUIRE(cudaMemcpy(a.data(), dout, ncplx * 16, cudaMemcpyDeviceToHost) == cudaSuccess);
		REQUIRE(cudaMemcpy(b.data(), dip, ncplx * 16, cudaMemcpyDeviceToHost) == cudaSuccess);
		size_t differing = 0;
		for (size_t i = 0; i < a.size(); i++)
			if (a[i] != b[i])
				differing++;
		CHECK(differing == 0);
		cufftDestroy(plan);
		cudaFree(dm);
		cudaFree(dout);
		cudaFree(dip);
	}
}
#endif
