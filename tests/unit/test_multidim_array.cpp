/*
 * tests/unit/test_multidim_array.cpp
 *
 * Unit tests for MultidimArray<RFLOAT> / MultidimArray<Complex>.
 * Ground-truth values are arithmetic identities.
 */

#include <catch2/catch.hpp>

#include "src/multidim_array.h"
#include "src/macros.h"

#include <cmath>

// ---------------------------------------------------------------------------
// 1D array
// ---------------------------------------------------------------------------

TEST_CASE("MDA: 1D resize sets correct dimensions", "[mda]")
{
    MultidimArray<RFLOAT> v(16);
    REQUIRE(XSIZE(v) == 16);
    REQUIRE(YSIZE(v) == 1);
    REQUIRE(ZSIZE(v) == 1);
    REQUIRE(NSIZE(v) == 1);
}

TEST_CASE("MDA: initZeros sets all elements to zero", "[mda]")
{
    MultidimArray<RFLOAT> v(32);
    v.initZeros();
    RFLOAT s = 0.0;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(v)
        s += DIRECT_MULTIDIM_ELEM(v, n);
    REQUIRE(s == Approx(0.0).margin(1e-12));
}

TEST_CASE("MDA: initConstant fills every element", "[mda]")
{
    const RFLOAT C = 3.14;
    MultidimArray<RFLOAT> v(16);
    v.initConstant(C);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(v)
        REQUIRE(DIRECT_MULTIDIM_ELEM(v, n) == Approx(C).epsilon(1e-10));
}

TEST_CASE("MDA: element write/read", "[mda]")
{
    MultidimArray<RFLOAT> v(8);
    v.initZeros();
    DIRECT_A1D_ELEM(v, 3) = 42.0;
    REQUIRE(DIRECT_A1D_ELEM(v, 3) == Approx(42.0).epsilon(1e-10));
    REQUIRE(DIRECT_A1D_ELEM(v, 0) == Approx(0.0).margin(1e-12));
}

TEST_CASE("MDA: computeStats returns correct mean and stddev for constant", "[mda]")
{
    const RFLOAT C = 5.0;
    MultidimArray<RFLOAT> v(64);
    v.initConstant(C);
    RFLOAT mn, mx, avg, std_dev;
    v.computeStats(avg, std_dev, mn, mx);
    REQUIRE(avg     == Approx(C).epsilon(1e-8));
    REQUIRE(std_dev == Approx(0.0).margin(1e-8));
    REQUIRE(mn      == Approx(C).epsilon(1e-8));
    REQUIRE(mx      == Approx(C).epsilon(1e-8));
}

TEST_CASE("MDA: computeStats correct mean for linear ramp", "[mda]")
{
    // Use descending order so that both min and max branches in computeStats
    // are exercised (the implementation uses if/else-if, so ascending data
    // would never update the minimum branch after the first element).
    const int N = 100;
    MultidimArray<RFLOAT> v(N);
    for (int i = 0; i < N; i++)
        DIRECT_A1D_ELEM(v, i) = (RFLOAT)(N - 1 - i);   // 99, 98, ..., 0
    RFLOAT mn, mx, avg, std_dev;
    v.computeStats(avg, std_dev, mn, mx);
    // mean of 0..99 = 49.5
    REQUIRE(avg == Approx(49.5).epsilon(1e-6));
    REQUIRE(mn  == Approx(0.0).margin(1e-10));
    REQUIRE(mx  == Approx(99.0).epsilon(1e-10));
}

// ---------------------------------------------------------------------------
// 2D array
// ---------------------------------------------------------------------------

TEST_CASE("MDA: 2D resize sets correct dimensions", "[mda]")
{
    MultidimArray<RFLOAT> img(32, 64);
    REQUIRE(YSIZE(img) == 32);
    REQUIRE(XSIZE(img) == 64);
    REQUIRE(ZSIZE(img) == 1);
    REQUIRE(NSIZE(img) == 1);
}

TEST_CASE("MDA: 2D element access", "[mda]")
{
    MultidimArray<RFLOAT> img(8, 8);
    img.initZeros();
    DIRECT_A2D_ELEM(img, 3, 5) = 99.0;
    REQUIRE(DIRECT_A2D_ELEM(img, 3, 5) == Approx(99.0).epsilon(1e-10));
    REQUIRE(DIRECT_A2D_ELEM(img, 0, 0) == Approx(0.0).margin(1e-12));
}

TEST_CASE("MDA: sum of all elements", "[mda]")
{
    const int N = 16;
    MultidimArray<RFLOAT> v(N);
    for (int i = 0; i < N; i++)
        DIRECT_A1D_ELEM(v, i) = 1.0;
    RFLOAT s = 0.0;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(v)
        s += DIRECT_MULTIDIM_ELEM(v, n);
    REQUIRE(s == Approx((RFLOAT)N).epsilon(1e-8));
}

// ---------------------------------------------------------------------------
// 3D array
// ---------------------------------------------------------------------------

TEST_CASE("MDA: 3D resize sets correct dimensions", "[mda]")
{
    MultidimArray<RFLOAT> vol(16, 32, 64);
    REQUIRE(ZSIZE(vol) == 16);
    REQUIRE(YSIZE(vol) == 32);
    REQUIRE(XSIZE(vol) == 64);
}

TEST_CASE("MDA: 3D element roundtrip", "[mda]")
{
    MultidimArray<RFLOAT> vol(8, 8, 8);
    vol.initZeros();
    DIRECT_A3D_ELEM(vol, 2, 3, 4) = 77.0;
    REQUIRE(DIRECT_A3D_ELEM(vol, 2, 3, 4) == Approx(77.0).epsilon(1e-10));
    REQUIRE(DIRECT_A3D_ELEM(vol, 0, 0, 0) == Approx(0.0).margin(1e-12));
    REQUIRE(DIRECT_A3D_ELEM(vol, 7, 7, 7) == Approx(0.0).margin(1e-12));
}

// ---------------------------------------------------------------------------
// Complex array
// ---------------------------------------------------------------------------

TEST_CASE("MDA: Complex initZeros", "[mda]")
{
    MultidimArray<Complex> vc(16);
    vc.initZeros();
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(vc)
    {
        const Complex& c = DIRECT_MULTIDIM_ELEM(vc, n);
        REQUIRE(c.real == Approx(0.0).margin(1e-12));
        REQUIRE(c.imag == Approx(0.0).margin(1e-12));
    }
}

TEST_CASE("MDA: nzyxdim equals total element count", "[mda]")
{
    MultidimArray<RFLOAT> v(4, 8, 16);
    REQUIRE((long long)NZYXSIZE(v) == 4LL * 8 * 16);
}
