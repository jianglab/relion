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
	p.nr_orientations = 3686400; p.nr_pool = 8;
	const std::string k = problemKey(p);
	CHECK(k == "box=256;K=4;dim=3;model=123456789;thr=8;ranks=2;cur=120;coarse=64;orient=3686400;pool=8");
	CHECK(k.find(',') == std::string::npos); // it is a CSV field
	CHECK(k.find('/') == std::string::npos); // no path
	Problem q = p;
	q.current_size = 124;
	CHECK(problemKey(q) != k);
	q = p; q.ranks_on_device = 1;
	CHECK(problemKey(q) != k);
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
}

TEST_CASE("lean CudaCustomAllocator holds a small chunk, grows on demand within its cap", "[gpu_pool][gpu_allocator]")
{
	if (!gpuVisible())
	{
		WARN("no GPU visible: test skipped");
		return;
	}
	size_t free_b = 0, total_b = 0;
	REQUIRE(cudaMemGetInfo(&free_b, &total_b) == cudaSuccess);
	const size_t cap = std::min<size_t>(768 * MB, free_b / 4);
	if (cap < 256 * MB)
	{
		WARN("too little free GPU memory: test skipped");
		return;
	}
	const size_t first_chunk = 128 * MB, block = 48 * MB;

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
}

TEST_CASE("lean CudaCustomAllocator never holds more than its cap", "[gpu_pool][gpu_allocator]")
{
	if (!gpuVisible())
	{
		WARN("no GPU visible: test skipped");
		return;
	}
	size_t free_b = 0, total_b = 0;
	REQUIRE(cudaMemGetInfo(&free_b, &total_b) == cudaSuccess);
	const size_t cap = std::min<size_t>(256 * MB, free_b / 4);
	if (cap < 192 * MB)
	{
		WARN("too little free GPU memory: test skipped");
		return;
	}
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
}

TEST_CASE("the old pool is a single chunk and its cap is its size", "[gpu_pool][gpu_allocator]")
{
	if (!gpuVisible())
	{
		WARN("no GPU visible: test skipped");
		return;
	}
	CudaCustomAllocator pool(0, 512);
	CHECK(pool.chunkSize() == 0);
	CHECK_FALSE(pool.isLean());
	pool.resize(64 * MB);
	CHECK_FALSE(pool.isLean());
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
#endif
