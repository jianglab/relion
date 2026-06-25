/*
 * tests/unit/test_prefetch.cpp
 *
 * Tests for AsyncImagePrefetcher and MpiAsyncImagePrefetcher.
 *
 * The non-MPI AsyncImagePrefetcher is exercised directly.
 * The MPI variant (MpiAsyncImagePrefetcher) requires MPI and is
 * tested via the separate prefetch_mpi_test MPI test program.
 */

#include <catch2/catch.hpp>

#include "src/prefetch.h"
#include "src/exp_model.h"
#include "src/multidim_array.h"

TEST_CASE("AsyncImagePrefetcher: construction and destruction", "[prefetch]")
{
    Experiment exp;
    AsyncImagePrefetcher *p = nullptr;
    REQUIRE_NOTHROW(p = new AsyncImagePrefetcher(&exp));
    REQUIRE(p != nullptr);
    REQUIRE_NOTHROW(p->stop());
    REQUIRE_NOTHROW(delete p);
}

TEST_CASE("AsyncImagePrefetcher: idle without start does not crash", "[prefetch]")
{
    Experiment exp;
    AsyncImagePrefetcher prefetcher(&exp);
    std::vector<MultidimArray<RFLOAT>> target;

    // Calling waitAndSwap on an idle prefetcher should block forever,
    // so we don't call it without startPrefetch. Just verify stop works.
    REQUIRE_NOTHROW(prefetcher.stop());
}

TEST_CASE("AsyncImagePrefetcher: multiple stop calls are safe", "[prefetch]")
{
    Experiment exp;
    AsyncImagePrefetcher prefetcher(&exp);
    REQUIRE_NOTHROW(prefetcher.stop());
    REQUIRE_NOTHROW(prefetcher.stop());
}

TEST_CASE("AsyncImagePrefetcher: readTime accessors default to zero", "[prefetch]")
{
    Experiment exp;
    AsyncImagePrefetcher prefetcher(&exp);
    REQUIRE(prefetcher.readTimeOriginalUs() == 0);
    REQUIRE(prefetcher.readTimeScratchUs() == 0);
    REQUIRE(prefetcher.readTimeCacheUs() == 0);
}
