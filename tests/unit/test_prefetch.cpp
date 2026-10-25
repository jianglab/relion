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

#include "src/image.h"
#include <cstdlib>
#include <unistd.h>

namespace
{
// A stack of n 8x8 images, image k filled with k, and an Experiment whose particles are its images in order
struct StackFixture
{
    std::string dir, stack;
    Experiment exp;
    explicit StackFixture(int n)
    {
        const char *t = getenv("TMPDIR");
        std::string tmpl = std::string(t ? t : "/tmp") + "/relion_prefetch_XXXXXX";
        std::vector<char> buf(tmpl.begin(), tmpl.end());
        buf.push_back(0);
        dir = mkdtemp(buf.data());
        stack = dir + "/s.mrcs";
        Image<RFLOAT> s(8, 8, 1, n);
        for (int k = 0; k < n; k++)
            for (int i = 0; i < 64; i++)
                s().data[k * 64 + i] = k;
        s.write(stack);
        exp.cacheMode = true;   // names are read as they are
        for (int k = 0; k < n; k++)
        {
            ExpParticle p;
            p.name = integerToString(k + 1) + "@" + stack;
            exp.particles.push_back(p);
            exp.sorted_idx.push_back(k);
        }
    }
    ~StackFixture() { unlink(stack.c_str()); rmdir(dir.c_str()); }
};

bool holds(const std::vector<MultidimArray<RFLOAT>> &imgs, long int first, long int last)
{
    if ((long int) imgs.size() != last - first + 1) return false;
    for (long int i = 0; i < (long int) imgs.size(); i++)
        if (imgs[i].nzyxdim != 64 || imgs[i].data[0] != first + i || imgs[i].data[63] != first + i) return false;
    return true;
}
}

TEST_CASE("AsyncImagePrefetcher: pools read ahead arrive whole and in order", "[prefetch]")
{
    StackFixture f(22);
    setenv("RELION_PREFETCH_DEPTH", "3", 1);
    AsyncImagePrefetcher p(&f.exp, 4);
    unsetenv("RELION_PREFETCH_DEPTH");
    p.setReadAhead(21);
    std::vector<MultidimArray<RFLOAT>> imgs;
    p.startPrefetch(0, 3);
    for (long int first = 0; first <= 21; first += 4)
    {
        const long int last = std::min(21L, first + 3);
        REQUIRE(p.waitAndSwap(imgs, first, last));
        CHECK(holds(imgs, first, last));
        if (last < 21) p.startPrefetch(last + 1, std::min(21L, last + 4));
    }
    CHECK(p.nrPools() == 6);
}

TEST_CASE("AsyncImagePrefetcher: a wrong guess is dropped and the right pool still read", "[prefetch]")
{
    StackFixture f(16);
    AsyncImagePrefetcher p(&f.exp, 2);
    std::vector<MultidimArray<RFLOAT>> imgs;
    p.startPrefetch(0, 3);
    CHECK_FALSE(p.waitAndSwap(imgs, 8, 11, true));   // speculative: the guess was 0-3
    p.startPrefetch(8, 11);
    REQUIRE(p.waitAndSwap(imgs, 8, 11, true));
    CHECK(holds(imgs, 8, 11));
    // without read-ahead, pools may be asked for in any order
    p.startPrefetch(12, 15);
    REQUIRE(p.waitAndSwap(imgs, 12, 15));
    CHECK(holds(imgs, 12, 15));
    p.startPrefetch(4, 7);
    REQUIRE(p.waitAndSwap(imgs, 4, 7));
    CHECK(holds(imgs, 4, 7));
}
