/*
 * tests/unit/test_ctf.cpp
 *
 * Unit tests for the CTF class.
 * All tests use analytical ground truth values derived from the CTF definition:
 *
 *   gamma(s) = pi * lambda * Cs * lambda^2 * s^4 / 2  -  pi * lambda * delta_f * s^2  -  phi_s
 *   CTF(s)   = -sqrt(1-Q0^2) * sin(gamma) - Q0 * cos(gamma)          (with B-factor damping)
 *
 * where s = sqrt(X^2+Y^2), lambda = relativistic electron wavelength.
 */

#include <catch2/catch.hpp>

#include "src/ctf.h"
#include "src/spatial_frequency_grid.h"
#include "src/multidim_array.h"
#include "src/macros.h"

#include <cmath>

// Local implementation of the bilinear helper used by s2 remapping in
// production code (the original function has internal linkage).
static RFLOAT sampleRealFromFftwHalfBilinear(const MultidimArray<RFLOAT>& img,
                                             RFLOAT x, RFLOAT y_signed, int size)
{
    const int sh = size / 2 + 1;
    RFLOAT xc = x;
    if (xc < 0.0) xc = 0.0;
    if (xc > (RFLOAT)(sh - 1)) xc = (RFLOAT)(sh - 1);

    RFLOAT yw = y_signed;
    while (yw < 0.0) yw += (RFLOAT)size;
    while (yw >= (RFLOAT)size) yw -= (RFLOAT)size;

    const int x0 = (int)floor(xc);
    const int x1 = (x0 + 1 < sh) ? x0 + 1 : x0;
    const int y0 = (int)floor(yw);
    const int y1 = (y0 + 1 < size) ? y0 + 1 : 0;

    RFLOAT tx = xc - (RFLOAT)x0;
    RFLOAT ty = yw - (RFLOAT)y0;

    RFLOAT r00 = DIRECT_A2D_ELEM(img, y0, x0);
    RFLOAT r10 = DIRECT_A2D_ELEM(img, y1, x0);
    RFLOAT r01 = DIRECT_A2D_ELEM(img, y0, x1);
    RFLOAT r11 = DIRECT_A2D_ELEM(img, y1, x1);

    return (1.0 - tx) * ((1.0 - ty) * r00 + ty * r10)
         +       tx   * ((1.0 - ty) * r01 + ty * r11);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace
{

/// Build a typical 300 kV CTF with isotropic defocus (defU == defV).
CTF makeCTF(RFLOAT defocus_A = 10000.0,
            RFLOAT voltage   = 300.0,
            RFLOAT Cs        = 2.7,
            RFLOAT Q0        = 0.1,
            RFLOAT Bfac      = 0.0,
            RFLOAT scale     = 1.0)
{
    CTF ctf;
    ctf.setValues(defocus_A, defocus_A, 0.0, voltage, Cs, Q0, Bfac, scale, 0.0);
    return ctf;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

TEST_CASE("CTF: getCTF regression value at (10,10) Å-1/pix", "[ctf]")
{
    // Reproduce the original test already in tests/ctf.cpp.
    // defU=10000, defV=12000, defAng=90, V=300, Cs=2.7, Q0=0.1, Bfac=0
    CTF ctf;
    ctf.setValues(10000.0, 12000.0, 90.0, 300.0, 2.7, 0.1, 0.0, 1.0, 0.0);
    REQUIRE(ctf.getCTF(10.0, 10.0) == Approx(0.59154).epsilon(1e-4));
}

TEST_CASE("CTF: getCTF is bounded in [-1, 1] without B-factor", "[ctf]")
{
    CTF ctf = makeCTF(10000.0);
    // Sample many points; the CTF modulus must never exceed 1.0
    for (int i = 0; i <= 20; i++)
    {
        RFLOAT x = (RFLOAT)i * 0.02;   // 0 to 0.4 Å-1/pix
        for (int j = 0; j <= 20; j++)
        {
            RFLOAT y = (RFLOAT)j * 0.02;
            RFLOAT val = ctf.getCTF(x, y);
            REQUIRE(val >= Approx(-1.0).epsilon(1e-4));
            REQUIRE(val <= Approx( 1.0).epsilon(1e-4));
        }
    }
}

TEST_CASE("CTF: getCTF is exactly Q0 at zero spatial frequency", "[ctf]")
{
    // At s=0: gamma = -K3 = -atan(Q0/sqrt(1-Q0^2))
    //   CTF = -sin(gamma) = sin(atan(Q0/sqrt(1-Q0^2))) = Q0
    const RFLOAT Q0 = 0.1;
    CTF ctf;
    ctf.setValues(10000.0, 10000.0, 0.0, 300.0, 2.7, Q0, 0.0, 1.0, 0.0);
    RFLOAT val = ctf.getCTF(0.0, 0.0);
    REQUIRE(val == Approx(Q0).epsilon(1e-5));
}

TEST_CASE("CTF: isotropic CTF is centrosymmetric CTF(-x,-y)==CTF(x,y)", "[ctf]")
{
    CTF ctf = makeCTF(15000.0);
    for (int i = 1; i <= 10; i++)
    {
        RFLOAT x = (RFLOAT)i * 0.03;
        RFLOAT y = (RFLOAT)i * 0.01;
        RFLOAT val  = ctf.getCTF( x,  y);
        RFLOAT valm = ctf.getCTF(-x, -y);
        REQUIRE(val == Approx(valm).epsilon(1e-7));
    }
}

TEST_CASE("CTF: B-factor damping reduces magnitude monotonically with B", "[ctf]")
{
    const RFLOAT x = 0.1, y = 0.1;
    CTF ctf0 = makeCTF(10000.0, 300.0, 2.7, 0.1, 0.0);
    CTF ctfB = makeCTF(10000.0, 300.0, 2.7, 0.1, 100.0);  // B = 100 Å^2

    RFLOAT abs0 = std::fabs(ctf0.getCTF(x, y));
    RFLOAT absB = std::fabs(ctfB.getCTF(x, y));
    // Damping should reduce the amplitude
    REQUIRE(absB < abs0 + 1e-6);
}

TEST_CASE("CTF: abs mode returns non-negative values", "[ctf]")
{
    CTF ctf = makeCTF(10000.0);
    for (int i = 0; i <= 10; i++)
    {
        RFLOAT x = (RFLOAT)i * 0.03;
        RFLOAT y = 0.02;
        RFLOAT val = ctf.getCTF(x, y, /*do_abs=*/true);
        REQUIRE(val >= 0.0);
    }
}

TEST_CASE("CTF: scale factor multiplies the CTF value", "[ctf]")
{
    const RFLOAT x = 0.05, y = 0.05;
    CTF ctf1 = makeCTF(10000.0, 300.0, 2.7, 0.1, 0.0, 1.0);
    CTF ctf2 = makeCTF(10000.0, 300.0, 2.7, 0.1, 0.0, 2.0);
    RFLOAT v1 = ctf1.getCTF(x, y);
    RFLOAT v2 = ctf2.getCTF(x, y);
    REQUIRE(v2 == Approx(2.0 * v1).epsilon(1e-6));
}

TEST_CASE("CTF: astigmatic CTF breaks x-y equivalence when defAng=0", "[ctf]")
{
    // defU != defV, defAng = 0 deg.
    // The bilinear form has Axx = -defU, Ayy = -defV, Axy=0,
    // so CTF(a, 0) uses defU while CTF(0, a) uses defV.
    // With large astigmatism these should differ noticeably.
    CTF ctf;
    ctf.setValues(5000.0, 20000.0, 0.0, 300.0, 2.7, 0.1, 0.0, 1.0, 0.0);
    const RFLOAT a = 0.1;
    RFLOAT valX = ctf.getCTF(a, 0.0);
    RFLOAT valY = ctf.getCTF(0.0, a);
    // They must differ because defU != defV
    REQUIRE(std::fabs(valX - valY) > 0.01);
}

TEST_CASE("CTF: phase shift of PI flips the sign of the CTF", "[ctf]")
{
    // sin(gamma + PI) = -sin(gamma), so a phase shift of PI should flip the CTF.
    // Use a frequency known to give a large CTF value.
    const RFLOAT x = 0.15, y = 0.10;
    CTF ctf0, ctfPI;
    ctf0.setValues(10000.0, 10000.0, 0.0, 300.0, 2.7, 0.0, 0.0, 1.0, 0.0);
    ctfPI.setValues(10000.0, 10000.0, 0.0, 300.0, 2.7, 0.0, 0.0, 1.0, 180.0);
    RFLOAT v0  = ctf0.getCTF(x, y);
    RFLOAT vPI = ctfPI.getCTF(x, y);
    // Phase shift of PI flips the CTF sign: sin(gamma+PI) = -sin(gamma)
    // Also verify magnitude is non-trivial
    REQUIRE(std::fabs(v0) > 0.01);
    REQUIRE(vPI == Approx(-v0).epsilon(1e-5));
}

TEST_CASE("CTF: getLowOrderGamma is zero at origin", "[ctf]")
{
    CTF ctf = makeCTF(10000.0);
    // gamma(0,0) should be 0 (no K5/K3 shift when Q0=0 and phase_shift=0)
    CTF ctf0;
    ctf0.setValues(10000.0, 10000.0, 0.0, 300.0, 2.7, 0.0, 0.0, 1.0, 0.0);
    RFLOAT gamma = ctf0.getLowOrderGamma(0.0, 0.0);
    REQUIRE(gamma == Approx(0.0).margin(1e-8));
}

// ===========================================================================
// s-mode vs s2-mode CTF sampling
//
// In "s" mode (standard Cartesian), the CTF is evaluated directly at integer
// pixel positions (x_idx / size, y_signed / size) in cycles/pixel.
//
// In "s2" mode, the Cartesian CTF array is bilinearly resampled at the
// signed-s² grid positions (sample_x, sample_y) in pixel units.  The result
// at each grid index is:
//
//   CTF_s2[k] = sampleRealFromFftwHalfBilinear(Fctf_cart,
//                   grid.sample_x[k], grid.sample_y[k], size)
//
// which should closely match getCTF(sample_x[k]/size, sample_y[k]/size)
// (up to bilinear-interpolation error that vanishes at integer positions).
// ===========================================================================

namespace
{

/// Fill an FFTW half-plane MultidimArray with CTF values evaluated at each pixel.
/// Frequency of pixel (y_idx, x_idx) is (x_idx/size, y_signed/size) in cyc/pix.
MultidimArray<RFLOAT> buildCartesianCTFArray(const CTF& ctf, int size)
{
    const int sh = size / 2 + 1;
    MultidimArray<RFLOAT> arr(size, sh);
    for (int i = 0; i < size; i++)
    {
        int ip = (i < sh) ? i : i - size;
        RFLOAT fy = (RFLOAT)ip / size;
        for (int x = 0; x < sh; x++)
        {
            RFLOAT fx = (RFLOAT)x / size;
            DIRECT_A2D_ELEM(arr, i, x) = ctf.getCTF(fx, fy);
        }
    }
    return arr;
}

} // anonymous namespace

TEST_CASE("CTF s-mode: bilinear at integer pixel matches getCTF exactly", "[ctf][smode]")
{
    // In s mode, the CTF at grid cell (y_idx, x_idx) is getCTF(x/size, y/size).
    // Bilinear sampling at an exact integer position must reproduce that value.
    CTF ctf = makeCTF(10000.0);
    const int size = 32;
    MultidimArray<RFLOAT> Fctf = buildCartesianCTFArray(ctf, size);

    // Check several integer grid positions
    struct Pt { int x_idx, y_idx; };
    for (auto p : {Pt{1,0}, Pt{3,0}, Pt{5,2}, Pt{8,4}})
    {
        RFLOAT fx = (RFLOAT)p.x_idx / size;
        RFLOAT fy = (RFLOAT)p.y_idx / size;
        RFLOAT via_bilinear = sampleRealFromFftwHalfBilinear(
            Fctf, (RFLOAT)p.x_idx, (RFLOAT)p.y_idx, size);
        RFLOAT via_getCTF   = ctf.getCTF(fx, fy);
        REQUIRE(via_bilinear == Approx(via_getCTF).margin(1e-8));
    }
}

TEST_CASE("CTF s2-mode: bilinear-sampled CTF at s2 positions is in [-1,1]", "[ctf][s2grid]")
{
    // After s2 remapping every cell of the Cartesian CTF array is still a CTF
    // value (bilinear interpolation of values in [-1,1] stays in [-1,1]).
    CTF ctf = makeCTF(10000.0);
    const int size = 32;
    MultidimArray<RFLOAT> Fctf = buildCartesianCTFArray(ctf, size);
    SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(size);

    for (size_t k = 1; k < grid.sample_x.size(); k++)
    {
        RFLOAT val = sampleRealFromFftwHalfBilinear(
            Fctf, grid.sample_x[k], grid.sample_y[k], size);
        REQUIRE(val >= -1.0 - 1e-6);
        REQUIRE(val <=  1.0 + 1e-6);
    }
}

TEST_CASE("CTF s2-mode: Nyquist cell bilinear value matches getCTF at Nyquist", "[ctf][s2grid][smode]")
{
    // The on-axis Nyquist s2 sample (x_idx=half_size, y=0) maps to pixel
    // half_size exactly (integer position), so bilinear and getCTF must agree.
    CTF ctf = makeCTF(10000.0);
    const int size = 32;
    const int half_size = size / 2;
    MultidimArray<RFLOAT> Fctf = buildCartesianCTFArray(ctf, size);
    SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(size);

    // Grid index half_size = (y=0, x=half_size): s2 maps to pixel half_size exactly
    RFLOAT s2_val = sampleRealFromFftwHalfBilinear(
        Fctf, grid.sample_x[half_size], grid.sample_y[half_size], size);
    RFLOAT cart_val = ctf.getCTF(0.5, 0.0);   // Nyquist = 0.5 cyc/pix
    REQUIRE(s2_val == Approx(cart_val).margin(1e-8));
}

TEST_CASE("CTF s-mode vs s2-mode: inner-shell cells differ because s2 is remapped", "[ctf][smode][s2grid]")
{
    // In s mode the CTF at grid index (y=0, x=4) is getCTF(4/32, 0) = CTF at 4 pixels.
    // In s2 mode the same index maps to sqrt(4*16)=8 pixels, so CTF at 8 pixels.
    // These differ (non-trivially) because CTF is not constant.
    CTF ctf = makeCTF(10000.0);
    const int size = 32;
    const int x_idx = 4;    // choose a non-trivial inner-shell cell
    MultidimArray<RFLOAT> Fctf = buildCartesianCTFArray(ctf, size);
    SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(size);

    // s-mode: CTF evaluated at pixel x_idx exactly
    RFLOAT s_val  = ctf.getCTF((RFLOAT)x_idx / size, 0.0);

    // s2-mode: CTF bilinearly sampled at the s2-mapped position for grid index x_idx
    // (grid index x_idx lies in the y=0 row)
    RFLOAT s2_val = sampleRealFromFftwHalfBilinear(
        Fctf, grid.sample_x[x_idx], grid.sample_y[x_idx], size);

    // The two values correspond to different physical frequencies → they differ
    REQUIRE(std::fabs(s_val - s2_val) > 1e-3);
}

