// test_s2_vs_smode_consistency.cpp
//
// Compares FINUFFT type-2 evaluation + CenterFFTbySign at integer Cartesian
// positions against FFTW in the exact same convention used by
// Reconstructor::reconstructImage for both s-mode and s2-mode paths.
//
// Build & run:
//   cmake -S relion.git -B build -DRELION_TEST=ON -DBUILD_TESTS=ON
//   cmake --build build --target relion_unit_tests -- -j$(nproc)
//   build/bin/relion_unit_tests [s2smode]
//
// Or standalone:
//   mpicxx -O2 -I. -I/usr/include -I/path/to/eigen3 \
//     tests/unit/test_s2_vs_smode_consistency.cpp \
//     -Lbuild/lib -lrelion_lib -lfinufft -lfftw3 -lfftw3f -ltiff -lm \
//     -o test_s2_vs_smode && ./test_s2_vs_smode

#include <catch2/catch.hpp>

#include "src/relion_finufft.h"
#include "src/fftw.h"
#include "src/spatial_frequency_grid.h"
#include "src/multidim_array.h"
#include "src/macros.h"

#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace {

/// Deterministic test image (same style as test_fft_nufft_consistency)
MultidimArray<RFLOAT> makeTestImage(int size)
{
    MultidimArray<RFLOAT> img(size, size);
    for (int y = 0; y < size; y++)
    for (int x = 0; x < size; x++)
    {
        DIRECT_A2D_ELEM(img, y, x) =
            (RFLOAT)(0.50 * std::sin(2.0 * PI * 3.0 * x / size)
                   + 0.30 * std::cos(2.0 * PI * 4.0 * y / size)
                   + 0.15 * std::sin(2.0 * PI * 5.0 * (x + y) / size)
                   + 0.10);
    }
    return img;
}

} // anonymous namespace

TEST_CASE("s2_vs_smode: FINUFFT type-2 matches FFTW+CenterFFTbySign at integer Cartesian positions",
          "[s2smode][relion_finufft][fftw][nufft]")
{
#ifndef RELION_USE_FINUFFT
    SUCCEED("RELION_USE_FINUFFT=OFF: test skipped.");
#else
    // Test several box sizes to catch size-dependent issues
    const std::vector<int> sizes = {16, 32, 64};
    const double tolerance = 5.0e-4;  // FINUFFT eps=1e-6 gives ~1e-6*N² error

    for (int size : sizes)
    {
        SECTION("size = " + std::to_string(size))
        {
            MultidimArray<RFLOAT> img = makeTestImage(size);

            // --- s-mode path: FFTW + CenterFFTbySign ---
            img.setXmippOrigin();
            MultidimArray<Complex> F2D;
            FourierTransformer transformer;
            transformer.FourierTransform(img, F2D);  // normalizes by 1/N²
            CenterFFTbySign(F2D);

            // --- Build a signed Cartesian grid (all integer positions) ---
            SpatialFrequencyGrid2D grid;
            grid.size = size;
            grid.half_size = size / 2;
            const int half = size / 2;
            for (int y = -half; y <= half; y++)
            for (int x = -half; x <= half; x++)
            {
                grid.sample_x.push_back((RFLOAT)x);
                grid.sample_y.push_back((RFLOAT)y);
            }
            computeBilinearCoeffs(grid);

            // --- s2-mode path: FINUFFT type-2 ---
            std::vector<Complex> finufft_samples;
            evaluateNonuniformFourierSamples2D(img, grid, finufft_samples);
            REQUIRE(finufft_samples.size() == grid.sample_x.size());

            // --- Compare at each FFTW half-plane position ---
            const int sh = size / 2 + 1;
            int n_mismatch = 0;
            double max_dr = 0.0, max_di = 0.0, max_da = 0.0;

            for (int i = 0; i < size; i++)
            {
                const int ky = (i < sh) ? i : i - size;
                const int first_x = (i < sh) ? 0 : 1;
                for (int x = first_x; x < sh; x++)
                {
                    // FFTW value
                    const Complex fftw_val = DIRECT_A2D_ELEM(F2D, i, x);

                    // Map to signed Cartesian index:
                    //   grid index = (ky + half) * (size+1) + (x + half)
                    const int idx = (ky + half) * (size + 1) + (x + half);
                    REQUIRE(idx >= 0);
                    REQUIRE(idx < (int)finufft_samples.size());

                    const Complex finufft_val = finufft_samples[idx];

                    const double dr = std::fabs(fftw_val.real - finufft_val.real);
                    const double di = std::fabs(fftw_val.imag - finufft_val.imag);
                    const double da = std::sqrt(dr*dr + di*di);

                    if (dr > max_dr) max_dr = dr;
                    if (di > max_di) max_di = di;
                    if (da > max_da) max_da = da;

                    if (da > tolerance)
                    {
                        n_mismatch++;
                        if (n_mismatch <= 5)
                        {
                            std::cerr << "  MISMATCH size=" << size
                                      << " (kx=" << x << ", ky=" << ky
                                      << "): FFTW=(" << fftw_val.real
                                      << "," << fftw_val.imag
                                      << ") FINUFFT=(" << finufft_val.real
                                      << "," << finufft_val.imag
                                      << ") diff=" << da
                                      << " tol=" << tolerance
                                      << std::endl;
                        }
                    }
                }
            }

            // Print summary
            std::cerr << "  size=" << size
                      << "  compared=" << (int)(finufft_samples.size() / 2)
                      << "  mismatches=" << n_mismatch
                      << "  max_dr=" << max_dr
                      << "  max_di=" << max_di
                      << "  max_da=" << max_da
                      << std::endl;

            // Fail if any mismatch exceeds tolerance
            REQUIRE(n_mismatch == 0);
        }
    }
#endif
}

TEST_CASE("s2_vs_smode: comparison includes the Nyquist boundary correctly",
          "[s2smode][relion_finufft][fftw][nufft]")
{
#ifndef RELION_USE_FINUFFT
    SUCCEED("RELION_USE_FINUFFT=OFF: test skipped.");
#else
    const int size = 32;
    MultidimArray<RFLOAT> img = makeTestImage(size);

    // s-mode
    img.setXmippOrigin();
    MultidimArray<Complex> F2D;
    FourierTransformer transformer;
    transformer.FourierTransform(img, F2D);
    CenterFFTbySign(F2D);

    // Build the actual adaptive s2 hybrid grid used in production
    RFLOAT s2_step = 1.0;  // force fine sampling
    SpatialFrequencyGrid2D grid = makeAdaptiveS2HybridGrid2D(
        size, nullptr, s2_step);
    computeBilinearCoeffs(grid);

    std::vector<Complex> finufft_samples;
    evaluateNonuniformFourierSamples2D(img, grid, finufft_samples);
    REQUIRE(finufft_samples.size() == grid.sample_x.size());

    // Only compare Cartesian positions (where sample_x and sample_y are integers)
    const int half = size / 2;
    int n_mismatch = 0;
    double max_da = 0.0;

    for (size_t idx = 0; idx < grid.sample_x.size(); idx++)
    {
        const RFLOAT sx = grid.sample_x[idx];
        const RFLOAT sy = grid.sample_y[idx];

        // Only check integer Cartesian positions within the FFTW half-plane
        const int kx = (int)std::round(sx);
        const int ky = (int)std::round(sy);
        if (std::fabs(sx - kx) > 1e-10 || std::fabs(sy - ky) > 1e-10)
            continue;
        if (kx < 0 || kx > half)
            continue;
        if (ky < -half || ky > half)
            continue;

        // Map to FFTW index
        const int i = (ky >= 0) ? ky : ky + size;
        if (i < 0 || i >= size) continue;
        if (i >= half+1 && kx == 0) continue; // x=0 is stored only in positive half

        const Complex fftw_val = DIRECT_A2D_ELEM(F2D, i, kx);
        const Complex finufft_val = finufft_samples[idx];

        const double da = std::sqrt(
            std::pow(fftw_val.real - finufft_val.real, 2) +
            std::pow(fftw_val.imag - finufft_val.imag, 2));

        if (da > max_da) max_da = da;

        if (da > 0.01)
        {
            n_mismatch++;
            if (n_mismatch <= 5)
            {
                std::cerr << "  HYBRID MISMATCH at sample " << idx
                          << " (sx=" << sx << ", sy=" << sy
                          << " → kx=" << kx << ", ky=" << ky << ")"
                          << " FFTW=(" << fftw_val.real << "," << fftw_val.imag
                          << ") FINUFFT=(" << finufft_val.real << "," << finufft_val.imag
                          << ") diff=" << da
                          << std::endl;
            }
        }
    }

    std::cerr << "  hybrid grid: " << grid.sample_x.size() << " samples, "
              << n_mismatch << " mismatches, max_da=" << max_da
              << std::endl;

    // All Cartesian positions in the hybrid grid should match FFTW
    // The fill positions (non-integer) are expected to differ since FFTW
    // doesn't sample there — that's the whole point of the hybrid grid
    REQUIRE(n_mismatch == 0);
#endif
}
