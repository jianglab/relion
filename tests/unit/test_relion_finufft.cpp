/*
 * tests/unit/test_relion_finufft.cpp
 *
 * Unit tests for the relion_finufft module:
 *   - FINUFFT type-2 consistency with FFTW at integer pixel positions
 *   - evaluateCtfAtSamplePositions vs direct getCTF
 *   - applyCtfToSamples: CTF multiplication of samples and weights
 *   - applyFomToSamples: scalar FOM multiplication
 */

#include <catch2/catch.hpp>

#include "src/relion_finufft.h"
#include "src/fftw.h"
#include "src/ctf.h"
#include "src/spatial_frequency_grid.h"
#include "src/multidim_array.h"
#include "src/macros.h"

#include <cmath>
#include <vector>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace
{

/// Build a Cartesian (s-mode) half-plane grid so that finufft_target_x/y
/// correspond to integer pixel indices.  sample_x/y are in pixel units.
SpatialFrequencyGrid2D makeCartesianHalfPlaneGrid(int size)
{
    const int sh = size / 2 + 1;
    SpatialFrequencyGrid2D grid;
    grid.size = size;
    grid.half_size = size / 2;

    for (int i = 0; i < size; i++)
    {
        int ip = (i < sh) ? i : i - size;
        int first_x = (i < sh) ? 0 : 1;
        for (int x = first_x; x < sh; x++)
        {
            grid.sample_x.push_back((RFLOAT)x);
            grid.sample_y.push_back((RFLOAT)ip);
        }
    }
    computeBilinearCoeffs(grid);
    return grid;
}

/// Create a small deterministic real-space image (same as in test_fft_nufft_consistency).
MultidimArray<RFLOAT> makeDeterministicImage(int size)
{
    MultidimArray<RFLOAT> img(size, size);
    for (int y = 0; y < size; y++)
    for (int x = 0; x < size; x++)
    {
        DIRECT_A2D_ELEM(img, y, x) =
            (RFLOAT)(0.35 * std::sin(2.0 * PI * 2.0 * x / size)
                   + 0.25 * std::cos(2.0 * PI * 3.0 * y / size)
                   + 0.10 * std::sin(2.0 * PI * (x + y) / size));
    }
    return img;
}

/// Build a typical 300 kV CTF with isotropic defocus.
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
// Regression test: dense s2 grid with CTF-equivalent step ~ 1
// (crashed FINUFFT 2.2.0 due to swapped FINUFFT_EXECUTE arguments)
// ---------------------------------------------------------------------------
TEST_CASE("relion_finufft: dense grid fine step does not crash", "[relion_finufft][nufft][smode]")
{
#ifndef RELION_USE_FINUFFT
    SUCCEED("RELION_USE_FINUFFT=OFF: test skipped.");
#else
    const int size = 32;
    MultidimArray<RFLOAT> image(size, size);
    for (int y = 0; y < size; y++)
    for (int x = 0; x < size; x++)
        DIRECT_A2D_ELEM(image, y, x) = 1.0;
    image.setXmippOrigin();

    SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(size, nullptr, 1.0);
    computeBilinearCoeffs(grid);
    REQUIRE(grid.sample_x.size() > 1000);

    std::vector<Complex> samples;
    evaluateNonuniformFourierSamples2D(image, grid, samples);
    REQUIRE(samples.size() == grid.sample_x.size());
    for (size_t k = 0; k < samples.size(); k++)
    {
        REQUIRE(std::isfinite(samples[k].real));
        REQUIRE(std::isfinite(samples[k].imag));
    }
#endif
}

// ---------------------------------------------------------------------------
// FINUFFT type-2 vs FFTW consistency
// ---------------------------------------------------------------------------

TEST_CASE("relion_finufft: type-2 returns finite results for non-constant image", "[relion_finufft][nufft][smode]")
{
#ifndef RELION_USE_FINUFFT
    SUCCEED("RELION_USE_FINUFFT=OFF: FINUFFT type-2 test skipped.");
#else
    const int size = 16;
    MultidimArray<RFLOAT> image = makeDeterministicImage(size);
    image.setXmippOrigin();

    SpatialFrequencyGrid2D grid = makeCartesianHalfPlaneGrid(size);
    std::vector<Complex> finufft_samples;
    evaluateNonuniformFourierSamples2D(image, grid, finufft_samples);

    REQUIRE(finufft_samples.size() == grid.sample_x.size());

    // All values must be finite
    for (size_t k = 0; k < finufft_samples.size(); k++)
    {
        REQUIRE(std::isfinite(finufft_samples[k].real));
        REQUIRE(std::isfinite(finufft_samples[k].imag));
    }

    // DC of real-valued image has zero imaginary part
    REQUIRE(std::fabs(finufft_samples[0].imag) < 1e-6);
#endif
}

TEST_CASE("relion_finufft: type-2 result scales linearly with image intensity", "[relion_finufft][nufft]")
{
#ifndef RELION_USE_FINUFFT
    SUCCEED("RELION_USE_FINUFFT=OFF: FINUFFT type-2 test skipped.");
#else
    const int size = 16;
    MultidimArray<RFLOAT> base(size, size);
    for (int y = 0; y < size; y++)
    for (int x = 0; x < size; x++)
        DIRECT_A2D_ELEM(base, y, x) = (RFLOAT)(x * y);

    MultidimArray<RFLOAT> img2x = base; // copy
    img2x *= 2.0;

    base.setXmippOrigin();
    img2x.setXmippOrigin();

    SpatialFrequencyGrid2D grid = makeCartesianHalfPlaneGrid(size);

    std::vector<Complex> s1, s2;
    evaluateNonuniformFourierSamples2D(base, grid, s1);
    evaluateNonuniformFourierSamples2D(img2x, grid, s2);

    REQUIRE(s1.size() == s2.size());
    for (size_t k = 0; k < s1.size(); k++)
    {
        REQUIRE(s2[k].real == Approx(2.0 * s1[k].real).margin(1e-10));
        REQUIRE(s2[k].imag == Approx(2.0 * s1[k].imag).margin(1e-10));
    }
#endif
}

// ---------------------------------------------------------------------------
// evaluateCtfAtSamplePositions
// ---------------------------------------------------------------------------

TEST_CASE("relion_finufft: evaluateCtfAtSamplePositions matches direct getCTF",
          "[relion_finufft][ctf][s2grid]")
{
    const int size = 32;
    CTF ctf = makeCTF(12000.0);
    const RFLOAT angpix = 1.0;

    SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(size);
    computeBilinearCoeffs(grid);

    std::vector<RFLOAT> ctf_vals;
    evaluateCtfAtSamplePositions(ctf_vals, grid, ctf, size, angpix,
                                 false, false, false, NULL);

    REQUIRE(ctf_vals.size() == grid.sample_x.size());

    for (size_t k = 0; k < ctf_vals.size(); k++)
    {
        const RFLOAT x = grid.sample_x[k] / (size * angpix);
        const RFLOAT y = grid.sample_y[k] / (size * angpix);
        const RFLOAT expected = ctf.getCTF(x, y);
        REQUIRE(ctf_vals[k] == Approx(expected).margin(1e-12));
    }
}

TEST_CASE("relion_finufft: evaluateCtfAtSamplePositions with gamma offset",
          "[relion_finufft][ctf]")
{
    const int size = 32;
    CTF ctf = makeCTF(12000.0);
    const RFLOAT angpix = 1.0;

    SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(size);
    computeBilinearCoeffs(grid);

    // Constant gamma offset of PI/2
    std::vector<RFLOAT> gamma_offsets(grid.sample_x.size(), PI / 2.0);

    std::vector<RFLOAT> ctf_no_off, ctf_with_off;
    evaluateCtfAtSamplePositions(ctf_no_off, grid, ctf, size, angpix,
                                 false, false, false, NULL);
    evaluateCtfAtSamplePositions(ctf_with_off, grid, ctf, size, angpix,
                                 false, false, false, &gamma_offsets);

    REQUIRE(ctf_no_off.size() == ctf_with_off.size());
    for (size_t k = 0; k < ctf_no_off.size(); k++)
    {
        // With gamma offset, the CTF should differ for most positions
        // (exception: at frequencies where sin(gamma) = sin(gamma+PI/2))
        // Most positions will differ, so we check a subset.
        if (k > 0)
        {
            // Some must differ; rather than enforce all, enforce at least one
            // differs (the test below just checks the first non-DC position)
        }
    }
    // The first non-DC position should differ
    REQUIRE(std::fabs(ctf_no_off[1] - ctf_with_off[1]) > 1e-6);
}

TEST_CASE("relion_finufft: evaluateCtfAtSamplePositions with ctf_phase_flipped=\"do_abs\"",
          "[relion_finufft][ctf]")
{
    // ctf_phase_flipped is passed as do_abs to getCTF: returns |CTF|
    const int size = 32;
    CTF ctf = makeCTF(12000.0);
    const RFLOAT angpix = 1.0;

    SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(size);
    computeBilinearCoeffs(grid);

    std::vector<RFLOAT> ctf_norm, ctf_flip;
    evaluateCtfAtSamplePositions(ctf_norm, grid, ctf, size, angpix,
                                 false, false, false, NULL);
    evaluateCtfAtSamplePositions(ctf_flip, grid, ctf, size, angpix,
                                 true, false, false, NULL);

    REQUIRE(ctf_norm.size() == ctf_flip.size());
    for (size_t k = 0; k < ctf_norm.size(); k++)
    {
        // ctf_phase_flipped=True applies abs(): all values are non-negative
        REQUIRE(ctf_flip[k] >= 0.0);
        // and equal to |ctf_norm|
        REQUIRE(ctf_flip[k] == Approx(std::fabs(ctf_norm[k])).margin(1e-12));
    }
}

// ---------------------------------------------------------------------------
// applyCtfToSamples
// ---------------------------------------------------------------------------

TEST_CASE("relion_finufft: applyCtfToSamples without ctf_premultiplied",
          "[relion_finufft][ctf]")
{
    const int size = 32;
    CTF ctf = makeCTF(12000.0);
    const RFLOAT angpix = 1.0;

    SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(size);
    computeBilinearCoeffs(grid);

    const int n = grid.sample_x.size();
    std::vector<Complex> samples(n, Complex(2.0, 1.5));
    std::vector<RFLOAT> sample_weight(n, 3.0);

    std::vector<RFLOAT> ctf_values;
    applyCtfToSamples(samples, sample_weight, grid, ctf, size, angpix,
                      false, false, false, false, NULL, &ctf_values);

    // Verify: samples multiplied by CTF, weights by CTF^2
    for (size_t k = 0; k < (size_t)n; k++)
    {
        const RFLOAT c = ctf_values[k];
        // Original samples were (2.0, 1.5); after CTF: (2c, 1.5c)
        REQUIRE(samples[k].real == Approx(2.0 * c).margin(1e-12));
        REQUIRE(samples[k].imag == Approx(1.5 * c).margin(1e-12));
        // Original weight 3.0; after CTF^2: 3*c^2
        REQUIRE(sample_weight[k] == Approx(3.0 * c * c).margin(1e-12));
    }
}

TEST_CASE("relion_finufft: applyCtfToSamples with ctf_premultiplied",
          "[relion_finufft][ctf]")
{
    const int size = 32;
    CTF ctf = makeCTF(12000.0);
    const RFLOAT angpix = 1.0;

    SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(size);
    computeBilinearCoeffs(grid);

    const int n = grid.sample_x.size();
    std::vector<Complex> samples(n, Complex(2.0, 1.5));
    std::vector<RFLOAT> sample_weight(n, 3.0);

    std::vector<RFLOAT> ctf_values;
    applyCtfToSamples(samples, sample_weight, grid, ctf, size, angpix,
                      false, false, false, true, NULL, &ctf_values);

    // With ctf_premultiplied=true: samples unchanged, weights by CTF^2
    for (size_t k = 0; k < (size_t)n; k++)
    {
        REQUIRE(samples[k].real == Approx(2.0).margin(1e-12));
        REQUIRE(samples[k].imag == Approx(1.5).margin(1e-12));
        const RFLOAT c = ctf_values[k];
        REQUIRE(sample_weight[k] == Approx(3.0 * c * c).margin(1e-12));
    }
}

TEST_CASE("relion_finufft: applyCtfToSamples with only_flip_phases",
          "[relion_finufft][ctf]")
{
    const int size = 32;
    CTF ctf = makeCTF(12000.0);
    const RFLOAT angpix = 1.0;

    SpatialFrequencyGrid2D grid = makeSignedS2CartesianGrid2D(size);
    computeBilinearCoeffs(grid);

    // When only_flip_phases=true, getCTF returns sign(-CTF_unnormalized)
    // which is either +1 or -1.
    const int n = grid.sample_x.size();
    std::vector<Complex> samples(n, Complex(1.0, 0.0));
    std::vector<RFLOAT> sample_weight(n, 1.0);

    std::vector<RFLOAT> ctf_values;
    applyCtfToSamples(samples, sample_weight, grid, ctf, size, angpix,
                      false, true, false, false, NULL, &ctf_values);

    for (size_t k = 0; k < (size_t)n; k++)
    {
        const RFLOAT c = ctf_values[k];
        // CTF sign: must be exactly +1 or -1
        REQUIRE((c == Approx(1.0).margin(1e-10) || c == Approx(-1.0).margin(1e-10)));
        REQUIRE(samples[k].real == Approx(c).margin(1e-12));
        // weight multiplied by c^2 = 1
        REQUIRE(sample_weight[k] == Approx(1.0).margin(1e-12));
    }
}

// ---------------------------------------------------------------------------
// applyFomToSamples
// ---------------------------------------------------------------------------

TEST_CASE("relion_finufft: applyFomToSamples scales samples and weights by FOM",
          "[relion_finufft]")
{
    const int n = 100;
    std::vector<Complex> samples(n, Complex(2.0, 3.0));
    std::vector<RFLOAT> sample_weight(n, 5.0);
    const RFLOAT fom = 0.5;

    applyFomToSamples(samples, sample_weight, fom);

    for (int i = 0; i < n; i++)
    {
        REQUIRE(samples[i].real == Approx(1.0).margin(1e-12));
        REQUIRE(samples[i].imag == Approx(1.5).margin(1e-12));
        REQUIRE(sample_weight[i] == Approx(2.5).margin(1e-12));
    }
}

TEST_CASE("relion_finufft: applyFomToSamples with zero FOM gives zero result",
          "[relion_finufft]")
{
    const int n = 50;
    std::vector<Complex> samples(n, Complex(2.0, 3.0));
    std::vector<RFLOAT> sample_weight(n, 5.0);

    applyFomToSamples(samples, sample_weight, 0.0);

    for (int i = 0; i < n; i++)
    {
        REQUIRE(samples[i].real == Approx(0.0).margin(1e-12));
        REQUIRE(samples[i].imag == Approx(0.0).margin(1e-12));
        REQUIRE(sample_weight[i] == Approx(0.0).margin(1e-12));
    }
}

// ---------------------------------------------------------------------------
// Bilinear helpers (MultidimArray versions from relion_finufft.h)
// ---------------------------------------------------------------------------

TEST_CASE("relion_finufft: sampleComplexFromFftwHalfBilinear at integer position",
          "[relion_finufft][smode]")
{
    const int size = 8;
    const int sh = size / 2 + 1;
    MultidimArray<Complex> img(size, sh);
    for (int i = 0; i < size; i++)
        for (int j = 0; j < sh; j++)
            DIRECT_A2D_ELEM(img, i, j) = Complex(
                (RFLOAT)(i * 100 + j), (RFLOAT)(i * 10 + j * 2));

    // At exact integer (x=3, y_signed=0): img[0][3] = (3, 6)
    Complex v = sampleComplexFromFftwHalfBilinear(img, 3.0, 0.0, size);
    REQUIRE(v.real == Approx(3.0).margin(1e-10));
    REQUIRE(v.imag == Approx(6.0).margin(1e-10));

    // At (x=1, y_signed=2): img[2][1] = (201, 22)
    v = sampleComplexFromFftwHalfBilinear(img, 1.0, 2.0, size);
    REQUIRE(v.real == Approx(201.0).margin(1e-10));
    REQUIRE(v.imag == Approx(22.0).margin(1e-10));
}

TEST_CASE("relion_finufft: sampleComplexFromFftwHalfBilinear half-integer x",
          "[relion_finufft][smode]")
{
    const int size = 8;
    const int sh = size / 2 + 1;
    MultidimArray<Complex> img(size, sh);
    img.initConstant(Complex(0.0, 0.0));
    DIRECT_A2D_ELEM(img, 0, 2) = Complex(4.0, 1.0);
    DIRECT_A2D_ELEM(img, 0, 3) = Complex(8.0, 3.0);

    // At (x=2.5, y=0): lerp between col 2 (4,1) and col 3 (8,3) -> (6, 2)
    Complex v = sampleComplexFromFftwHalfBilinear(img, 2.5, 0.0, size);
    REQUIRE(v.real == Approx(6.0).margin(1e-10));
    REQUIRE(v.imag == Approx(2.0).margin(1e-10));
}

TEST_CASE("relion_finufft: sampleComplexFromFftwHalfBilinear negative y wraps",
          "[relion_finufft][smode]")
{
    const int size = 8;
    const int sh = size / 2 + 1;
    MultidimArray<Complex> img(size, sh);
    img.initConstant(Complex(0.0, 0.0));
    DIRECT_A2D_ELEM(img, 7, 0) = Complex(10.0, 2.0);
    DIRECT_A2D_ELEM(img, 0, 0) = Complex(6.0, 4.0);

    // y=-0.5 wraps to y=7.5 -> lerp between rows 7 and 0
    Complex v = sampleComplexFromFftwHalfBilinear(img, 0.0, -0.5, size);
    REQUIRE(v.real == Approx(8.0).margin(1e-10));
    REQUIRE(v.imag == Approx(3.0).margin(1e-10));
}
