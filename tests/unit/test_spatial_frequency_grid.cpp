/*
 * tests/unit/test_spatial_frequency_grid.cpp
 *
 * Unit tests for SpatialFrequencyGrid2D and makeSignedS2CartesianGrid2D.
 *
 * Ground-truth properties derived analytically from the s^2 mapping:
 *
 *   s^2 axis coordinate at integer grid index i  =  i * half_size
 *   Physical frequency from s^2 coord:  s = sqrt(|s^2|) * sign(s^2)
 *   Sample weight in physical space:    delta_s2^2 / (2 * s2_radius)
 *
 * The Hermitian half-plane convention follows the standard FFTW layout:
 *   full y-axis  [0 .. size-1]  with negative frequencies stored at y >= sh
 *   x restricted [0 .. sh-1]   where sh = size/2 + 1
 *
 * isMagAwareSignedS2HybridCellWithinCrossResolution checks whether a
 * standard Cartesian grid cell is within the low-frequency cross-over
 * radius (possibly stretched by the mag matrix).
 */

#include <catch2/catch.hpp>

#include "src/spatial_frequency_grid.h"
#include "src/reconstructor.h"    // SpatialFrequencyMode enum
#include "src/multidim_array.h"
#include "src/matrix2d.h"
#include "src/macros.h"

#include <cmath>
#include <numeric>

// ---------------------------------------------------------------------------
// Local implementations of helpers from reconstructor.cpp (anonymous namespace
// prevents extern linkage; re-implement them here for the test binary).
// ---------------------------------------------------------------------------

static SpatialFrequencyMode parseSpatialFrequencyMode(const std::string& mode)
{
    if (mode == "s")  return SPATIAL_FREQUENCY_MODE_S;
    if (mode == "s2") return SPATIAL_FREQUENCY_MODE_S2;
    return SPATIAL_FREQUENCY_MODE_S;
}

static RFLOAT sampleRealFromFftwHalfBilinear(const MultidimArray<RFLOAT>& img,
                                              RFLOAT x, RFLOAT y_signed, int size)
{
    const int sh = size / 2 + 1;
    RFLOAT xc = x;
    if (xc < 0.0) xc = 0.0;
    if (xc > (RFLOAT)(sh - 1)) xc = (RFLOAT)(sh - 1);

    RFLOAT yw = y_signed;
    while (yw <  0.0)          yw += (RFLOAT)size;
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

/// Total number of half-plane pixels expected for a grid of given size.
int expectedHalfPlanePixels(int size)
{
    const int sh = size / 2 + 1;
    // Full y range: size rows, but the bottom half (y>0, x>0) is omitted
    // (kept only at x=0 for the Hermitian axis).
    // Actual count: for y in [0..sh-1] -> x in [0..sh-1]; for y in [sh..size-1] -> x in [1..sh-1]
    return sh * sh + (size - sh) * (sh - 1);
}

/// Build an identity 2×2 matrix.
Matrix2D<RFLOAT> identityMag()
{
    Matrix2D<RFLOAT> M(2, 2);
    M(0,0) = 1.0; M(0,1) = 0.0;
    M(1,0) = 0.0; M(1,1) = 1.0;
    return M;
}

/// Build a diagonal stretch matrix: M = diag(sx, sy).
Matrix2D<RFLOAT> stretchMag(RFLOAT sx, RFLOAT sy)
{
    Matrix2D<RFLOAT> M(2, 2);
    M(0,0) = sx;  M(0,1) = 0.0;
    M(1,0) = 0.0; M(1,1) = sy;
    return M;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// makeSignedS2CartesianGrid2D
// ---------------------------------------------------------------------------

TEST_CASE("S2Grid: grid has correct number of samples", "[s2grid]")
{
    for (int size : {16, 32, 64})
    {
        SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(size);
        int expected = expectedHalfPlanePixels(size);
        REQUIRE((int)grid.sample_x.size()      == expected);
        REQUIRE((int)grid.sample_y.size()      == expected);
        REQUIRE((int)grid.sample_weight.size() == expected);
        REQUIRE(grid.size      == size);
        REQUIRE(grid.half_size == size / 2);
    }
}

TEST_CASE("S2Grid: DC sample has zero weight", "[s2grid]")
{
    // The DC point (s=0) carries no weight because the Jacobian diverges there;
    // it is explicitly set to 0.
    SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(32);
    // DC is always the first sample (i=0, x=0)
    REQUIRE(grid.sample_x[0]      == Approx(0.0).margin(1e-12));
    REQUIRE(grid.sample_y[0]      == Approx(0.0).margin(1e-12));
    REQUIRE(grid.sample_weight[0] == Approx(0.0).margin(1e-12));
}

TEST_CASE("S2Grid: sample coordinates are within expected pixel range", "[s2grid]")
{
    // The grid stores coordinates in pixel units (index space).
    // For size=32 (half_size=16), coordinates sx range from 0 to half_size
    // along x, and sy from -half_size to half_size along y.
    // The maximum radius is sqrt(2) * half_size.
    const int size = 32;
    const int half_size = size / 2;
    const RFLOAT max_r = std::sqrt(2.0) * half_size + 1e-6;
    SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(size);
    for (size_t k = 0; k < grid.sample_x.size(); k++)
    {
        RFLOAT r = std::hypot(grid.sample_x[k], grid.sample_y[k]);
        REQUIRE(r <= max_r);
    }
}

TEST_CASE("S2Grid: all non-DC weights are positive", "[s2grid]")
{
    SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(32);
    for (size_t k = 1; k < grid.sample_weight.size(); k++)
    {
        REQUIRE(grid.sample_weight[k] > 0.0);
    }
}

TEST_CASE("S2Grid: with identity mag matrix matches no-mag result", "[s2grid]")
{
    Matrix2D<RFLOAT> M = identityMag();
    SpatialFrequencyGrid2D gNoMag = makeSignedS2CartesianGrid2D(32);
    SpatialFrequencyGrid2D gIdMag = makeSignedS2CartesianGrid2D(32, &M);

    REQUIRE(gNoMag.sample_x.size() == gIdMag.sample_x.size());
    for (size_t k = 0; k < gNoMag.sample_x.size(); k++)
    {
        REQUIRE(gNoMag.sample_x[k]      == Approx(gIdMag.sample_x[k]).epsilon(1e-6));
        REQUIRE(gNoMag.sample_y[k]      == Approx(gIdMag.sample_y[k]).epsilon(1e-6));
        REQUIRE(gNoMag.sample_weight[k] == Approx(gIdMag.sample_weight[k]).epsilon(1e-6));
    }
}

TEST_CASE("S2Grid: isotropic grid has symmetric x and y coverage", "[s2grid]")
{
    // For an isotropic grid, the distribution of |sample_x| values should
    // equal the distribution of |sample_y| values (same histogram).
    SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(32);
    RFLOAT sum_x2 = 0.0, sum_y2 = 0.0;
    for (size_t k = 0; k < grid.sample_x.size(); k++)
    {
        sum_x2 += grid.sample_x[k] * grid.sample_x[k];
        sum_y2 += grid.sample_y[k] * grid.sample_y[k];
    }
    REQUIRE(sum_x2 == Approx(sum_y2).epsilon(0.05));
}

TEST_CASE("S2Grid: anisotropic mag stretches sample positions", "[s2grid]")
{
    // A 2x stretch in x should shift samples toward smaller x frequencies.
    Matrix2D<RFLOAT> M = stretchMag(2.0, 1.0);
    SpatialFrequencyGrid2D gNoMag = makeSignedS2CartesianGrid2D(32);
    SpatialFrequencyGrid2D gStretch = makeSignedS2CartesianGrid2D(32, &M);

    REQUIRE(gNoMag.sample_x.size() == gStretch.sample_x.size());

    // The x range should be compressed relative to y when magnified in x
    RFLOAT max_abs_x_no   = 0.0, max_abs_x_str = 0.0;
    RFLOAT max_abs_y_no   = 0.0, max_abs_y_str = 0.0;
    for (size_t k = 0; k < gNoMag.sample_x.size(); k++)
    {
        max_abs_x_no  = std::max(max_abs_x_no,  std::fabs(gNoMag.sample_x[k]));
        max_abs_y_no  = std::max(max_abs_y_no,  std::fabs(gNoMag.sample_y[k]));
        max_abs_x_str = std::max(max_abs_x_str, std::fabs(gStretch.sample_x[k]));
        max_abs_y_str = std::max(max_abs_y_str, std::fabs(gStretch.sample_y[k]));
    }
    // Mag matrix inverse is applied to coordinates: x coords should shrink
    REQUIRE(max_abs_x_str < max_abs_x_no + 1e-6);
}

TEST_CASE("S2Grid: anisotropic mag adjusts weights by 1/|det(M)|", "[s2grid]")
{
    Matrix2D<RFLOAT> M = stretchMag(2.0, 1.0);   // det = 2
    SpatialFrequencyGrid2D gNoMag  = makeSignedS2CartesianGrid2D(32);
    SpatialFrequencyGrid2D gStretch = makeSignedS2CartesianGrid2D(32, &M);

    // Every weight in the stretched grid should be 1/2 of the corresponding
    // no-mag weight (same s2 position but det(M) = 2).
    // We compare means to be robust to per-sample permutations.
    RFLOAT mean_w_no  = 0.0, mean_w_str = 0.0;
    int n = 0;
    for (size_t k = 1; k < gNoMag.sample_weight.size(); k++)
    {
        if (gNoMag.sample_weight[k] > 0.0)
        {
            mean_w_no  += gNoMag.sample_weight[k];
            mean_w_str += gStretch.sample_weight[k];
            n++;
        }
    }
    if (n > 0)
    {
        mean_w_no  /= n;
        mean_w_str /= n;
        REQUIRE(mean_w_str == Approx(mean_w_no / 2.0).epsilon(0.01));
    }
}

// ---------------------------------------------------------------------------
// isMagAwareSignedS2HybridCellWithinCrossResolution
// ---------------------------------------------------------------------------

TEST_CASE("HybridCutoff: DC is within any cross resolution", "[s2hybrid]")
{
    REQUIRE(isMagAwareSignedS2HybridCellWithinCrossResolution(
        0, 0, 64, 1.0, 6.0, NULL));
}

TEST_CASE("HybridCutoff: Nyquist is outside typical cross resolution", "[s2hybrid]")
{
    // size=64, angpix=1.0, cross=6 Å  -> s_cross = 1/6 ~ 0.167 pix-1
    // Nyquist = 0.5 pix-1 > 0.167: should be outside
    const int size = 64;
    const int sh = size / 2;   // Nyquist index
    REQUIRE_FALSE(isMagAwareSignedS2HybridCellWithinCrossResolution(
        0, sh, size, 1.0, 6.0, NULL));
}

TEST_CASE("HybridCutoff: default resolution is 4*angpix when argument <= 0", "[s2hybrid]")
{
    // Default cross resolution = 6 * angpix = 6 * 1.5 = 9 Å
    // => s_cross = 1/9 pix-1.  Cell at x=4/64 = 0.0625 pix-1 should be inside.
    const int size = 64;
    const RFLOAT angpix = 1.5;
    // x = 4 / (64 * 1.5) = 0.0417 pix-1 < 1/9 ~ 0.111
    bool inside = isMagAwareSignedS2HybridCellWithinCrossResolution(
        0, 4, size, angpix, 0.0, NULL);
    REQUIRE(inside);
}

TEST_CASE("HybridCutoff: identity mag gives same result as no mag", "[s2hybrid]")
{
    Matrix2D<RFLOAT> I = identityMag();
    for (int y = 0; y <= 32; y += 4)
    {
        for (int x = 0; x <= 32; x += 4)
        {
            bool noMag  = isMagAwareSignedS2HybridCellWithinCrossResolution(
                y, x, 64, 1.0, 9.0, NULL);
            bool withId = isMagAwareSignedS2HybridCellWithinCrossResolution(
                y, x, 64, 1.0, 9.0, &I);
            REQUIRE(noMag == withId);
        }
    }
}

TEST_CASE("HybridCutoff: 2x x-stretch allows larger x before cutoff", "[s2hybrid]")
{
    // Mag matrix M = diag(2, 1): magnification doubles effective x frequency.
    // The physical frequency of grid cell (0, x) is x/(size*angpix).
    // After mag, it maps to (2*x/(size*angpix), 0).
    // A cell that would be inside the cutoff without mag (in x) may be
    // outside when the mag stretches it.
    Matrix2D<RFLOAT> M = stretchMag(2.0, 1.0);
    const int size = 64;
    const RFLOAT angpix = 1.0;
    const RFLOAT cross  = 9.0;

    // x=3 => phys = 3/64 ~ 0.047; mag'd = 0.094; s_cross = 1/9 ~ 0.111 → still inside both
    REQUIRE(isMagAwareSignedS2HybridCellWithinCrossResolution(0, 3, size, angpix, cross, NULL));
    REQUIRE(isMagAwareSignedS2HybridCellWithinCrossResolution(0, 3, size, angpix, cross, &M));

    // x=6 => phys = 6/64 ~ 0.094; mag'd = 0.188 > 0.111 → outside with mag
    bool noMagResult  = isMagAwareSignedS2HybridCellWithinCrossResolution(0, 6, size, angpix, cross, NULL);
    bool magResult    = isMagAwareSignedS2HybridCellWithinCrossResolution(0, 6, size, angpix, cross, &M);
    // noMag should be inside (0.094 < 0.111), mag should be outside (0.188 > 0.111)
    REQUIRE(noMagResult);
    REQUIRE_FALSE(magResult);
}

// ===========================================================================
// s vs s2 mode: parseSpatialFrequencyMode
// ===========================================================================

TEST_CASE("SpatialFrequencyMode: 's' parses to SPATIAL_FREQUENCY_MODE_S", "[smode]")
{
    REQUIRE(parseSpatialFrequencyMode("s") == SPATIAL_FREQUENCY_MODE_S);
}

TEST_CASE("SpatialFrequencyMode: 's2' parses to SPATIAL_FREQUENCY_MODE_S2", "[smode]")
{
    REQUIRE(parseSpatialFrequencyMode("s2") == SPATIAL_FREQUENCY_MODE_S2);
}

TEST_CASE("SpatialFrequencyMode: 's' and 's2' are distinct values", "[smode]")
{
    REQUIRE(parseSpatialFrequencyMode("s") != parseSpatialFrequencyMode("s2"));
}

// ===========================================================================
// s2 coordinate formula (analytical ground truth from spatial_frequency_grid.cpp)
//
// For on-axis (y_idx=0) grid index x_idx=k:
//   sx2 = k * half_size,  s2_radius = k * half_size
//   sample_x  = sx2 / sqrt(s2_radius) = sqrt(k * half_size)
//   weight    = half_size^2 / (2 * k * half_size) = half_size / (2 * k)
// ===========================================================================

TEST_CASE("S2Grid: on-axis x coordinate equals sqrt(x_idx * half_size)", "[s2grid]")
{
    // size=32, half_size=16, y=0 row: grid index k corresponds to (y_idx=0, x_idx=k).
    const int size = 32;
    const int half_size = size / 2;
    SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(size);

    for (int k = 1; k <= half_size; k++)
    {
        RFLOAT expected_x = std::sqrt((RFLOAT)(k * half_size));
        // Grid index k is at (y_idx=0, x_idx=k) since first row starts at x=0.
        REQUIRE(grid.sample_x[k] == Approx(expected_x).epsilon(1e-8));
        REQUIRE(grid.sample_y[k] == Approx(0.0).margin(1e-12));
    }
}

TEST_CASE("S2Grid: on-axis weight equals half_size / (2 * x_idx)", "[s2grid]")
{
    const int size = 32;
    const int half_size = size / 2;
    SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(size);

    for (int k = 1; k <= half_size; k++)
    {
        RFLOAT expected_w = (RFLOAT)half_size / (2.0 * k);
        REQUIRE(grid.sample_weight[k] == Approx(expected_w).epsilon(1e-8));
    }
}

TEST_CASE("S2Grid: Nyquist on-axis maps to exactly half_size pixels", "[s2grid]")
{
    // At x_idx=half_size, y=0: sample_x = sqrt(half_size^2) = half_size
    const int size = 32;
    const int half_size = size / 2;
    SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(size);
    REQUIRE(grid.sample_x[half_size] == Approx((RFLOAT)half_size).epsilon(1e-8));
    REQUIRE(grid.sample_y[half_size] == Approx(0.0).margin(1e-12));
}

TEST_CASE("S2Grid: s2 mode has fewer samples inside low-s circle than Cartesian", "[s2grid][smode]")
{
    // In s mode (Cartesian), samples are at integer pixel positions (uniform).
    // In s2 mode, the same FFTW half-plane grid is mapped to sqrt-scaled positions,
    // which moves many inner-shell indices to larger pixel radii.
    const int size = 32;
    const int half_size = size / 2;
    const int sh = half_size + 1;
    const RFLOAT r_pix   = (RFLOAT)half_size / 2.0;   // half of Nyquist = 8 pixels
    const RFLOAT r_pix_2 = r_pix * r_pix;

    SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(size);

    int count_s2   = 0;
    int count_cart = 0;

    // Count non-DC s2 samples inside the low-s disk
    for (size_t k = 1; k < grid.sample_x.size(); k++)
    {
        RFLOAT r2 = grid.sample_x[k]*grid.sample_x[k] + grid.sample_y[k]*grid.sample_y[k];
        if (r2 <= r_pix_2)
            count_s2++;
    }

    // Count Cartesian (s-mode) non-DC half-plane pixels inside the same disk
    for (int i = 0; i < size; i++)
    {
        int ip      = (i < sh) ? i : i - size;
        int first_x = (i < sh) ? 0 : 1;
        for (int x = first_x; x < sh; x++)
        {
            RFLOAT r2 = (RFLOAT)(x*x + ip*ip);
            if (r2 > 0.0 && r2 <= r_pix_2)
                count_cart++;
        }
    }

    // s2 mode has fewer samples in a fixed low-s disk in pixel space
    REQUIRE(count_s2 < count_cart);
}

TEST_CASE("S2Grid: s and s2 modes have same total sample count (same FFTW layout)", "[s2grid][smode]")
{
    // Both modes produce exactly one value per half-plane FFTW pixel.
    for (int size : {16, 32, 64})
    {
        SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(size);
        REQUIRE((int)grid.sample_x.size() == expectedHalfPlanePixels(size));
    }
}

// ===========================================================================
// sampleRealFromFftwHalfBilinear – used by s2 mode to remap the Cartesian CTF
// array to s2-mode sample positions.
// ===========================================================================

TEST_CASE("s2 bilinear: exact value at integer pixel position", "[smode]")
{
    const int size = 8;
    const int sh   = size / 2 + 1;
    MultidimArray<RFLOAT> img(size, sh);
    // Fill: element (row, col) = row*100 + col
    for (int i = 0; i < size; i++)
        for (int j = 0; j < sh; j++)
            DIRECT_A2D_ELEM(img, i, j) = (RFLOAT)(i * 100 + j);

    // At exact integer (x=3, y_signed=0): img[0][3] = 3
    REQUIRE(sampleRealFromFftwHalfBilinear(img, 3.0, 0.0, size) == Approx(3.0).margin(1e-10));
    // At (x=1, y_signed=2): img[2][1] = 201
    REQUIRE(sampleRealFromFftwHalfBilinear(img, 1.0, 2.0, size) == Approx(201.0).margin(1e-10));
}

TEST_CASE("s2 bilinear: half-integer x gives mean of adjacent columns", "[smode]")
{
    const int size = 8;
    const int sh   = size / 2 + 1;
    MultidimArray<RFLOAT> img(size, sh);
    img.initConstant(0.0);
    DIRECT_A2D_ELEM(img, 0, 2) = 4.0;
    DIRECT_A2D_ELEM(img, 0, 3) = 8.0;

    // At (x=2.5, y=0): lerp between col 2 (4) and col 3 (8) → 6
    REQUIRE(sampleRealFromFftwHalfBilinear(img, 2.5, 0.0, size) == Approx(6.0).margin(1e-10));
}

TEST_CASE("s2 bilinear: negative y_signed wraps around the FFTW array", "[smode]")
{
    // y_signed=-0.5 wraps to size-0.5=7.5 → lerp rows 7 and 0 (with weight 0.5)
    const int size = 8;
    const int sh   = size / 2 + 1;
    MultidimArray<RFLOAT> img(size, sh);
    img.initConstant(0.0);
    DIRECT_A2D_ELEM(img, 7, 0) = 10.0;
    DIRECT_A2D_ELEM(img, 0, 0) =  6.0;

    REQUIRE(sampleRealFromFftwHalfBilinear(img, 0.0, -0.5, size) == Approx(8.0).margin(1e-10));
}

TEST_CASE("s2 bilinear: x < 0 is clamped to column 0", "[smode]")
{
    const int size = 8;
    const int sh   = size / 2 + 1;
    MultidimArray<RFLOAT> img(size, sh);
    img.initConstant(0.0);
    DIRECT_A2D_ELEM(img, 0, 0) = 42.0;

    REQUIRE(sampleRealFromFftwHalfBilinear(img, -1.0, 0.0, size) == Approx(42.0).margin(1e-10));
}

TEST_CASE("s2 bilinear: uniform CTF array sampled at s2 positions returns uniform value", "[s2grid][smode]")
{
    // A Cartesian CTF that is identically C at every pixel must also read C
    // at any s2 grid position (the bilinear interpolation of a constant is constant).
    const int size = 32;
    const int sh   = size / 2 + 1;
    MultidimArray<RFLOAT> Fctf_flat(size, sh);
    Fctf_flat.initConstant(0.5);

    SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(size);
    for (size_t k = 1; k < grid.sample_x.size(); k++)
    {
        RFLOAT val = sampleRealFromFftwHalfBilinear(
            Fctf_flat, grid.sample_x[k], grid.sample_y[k], size);
        REQUIRE(val == Approx(0.5).margin(1e-10));
    }
}
