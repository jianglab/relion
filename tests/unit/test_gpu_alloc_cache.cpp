/*
 * tests/unit/test_gpu_alloc_cache.cpp
 *
 * The per-thread block cache of the GPU pool (RELION_GPU_ALLOC_CACHE): reuse on the same stream and thread
 * only, a cap, give-back on a sync, and no request that fails because of it. Skipped when no GPU is visible.
 */

#include <catch2/catch.hpp>

#include "src/acc/cuda/cuda_alloc_cache_test.h"

TEST_CASE("gpu_alloc_cache: reuse rules, cap, give-back and exact fit", "[gpu_alloc_cache]")
{
	gpucache::Outcome o;
	if (!gpucache::run(o))
	{
		WARN("no GPU visible: test skipped");
		return;
	}
	CHECK(o.reusedSameStream);
	CHECK(o.hitsSameStream == 1);
	CHECK(o.dataOk);
	CHECK_FALSE(o.reusedOtherStream);
	CHECK_FALSE(o.reusedOtherThread);
	CHECK(o.cachedUnderCap <= o.cap);
	CHECK(o.cachedAfterSync == 0);
	CHECK(o.usedAfterSync == 0);
	CHECK(o.bigAfterSmallOk);
	CHECK(o.exactFitOk);
	CHECK(o.hitsCacheOff == 0);
	CHECK(o.hitsWhenFull == 0);
}
