/*
 * tests/unit/test_fft_nufft_consistency.cpp
 *
 * Consistency check between FFT (FFTW half-plane bins) and NUFFT (FINUFFT
 * type-3 evaluation) on the same Cartesian set of s samples.
 */

#include <catch2/catch.hpp>

#include "src/fftw.h"
#include "src/multidim_array.h"
#include "src/macros.h"

#include <complex>
#include <vector>

#ifdef RELION_USE_FINUFFT
#include <finufft.h>
#endif

namespace
{

struct HalfPlanePoint
{
    int row;
    int x;
    RFLOAT sx;
    RFLOAT sy;
};

std::vector<HalfPlanePoint> makeCartesianHalfPlaneSamples(int size)
{
    std::vector<HalfPlanePoint> pts;
    const int sh = size / 2 + 1;
    pts.reserve(sh * sh + (size - sh) * (sh - 1));

    for (int i = 0; i < size; i++)
    {
        const int y = (i < sh) ? i : i - size;
        const int first_x = (i < sh) ? 0 : 1;
        for (int x = first_x; x < sh; x++)
        {
            HalfPlanePoint p;
            p.row = i;
            p.x = x;
            p.sx = (RFLOAT)x;
            p.sy = (RFLOAT)y;
            pts.push_back(p);
        }
    }
    return pts;
}

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

#ifdef RELION_USE_FINUFFT
std::vector<Complex> evaluateNufftAtSamples(const MultidimArray<RFLOAT>& image,
                                            const std::vector<HalfPlanePoint>& pts)
{
    const int size = XSIZE(image);
    const int64_t nj = (int64_t)size * (int64_t)size;

    std::vector<RFLOAT> xj(nj), yj(nj);
    std::vector<std::complex<RFLOAT> > source(nj);

    int64_t idx = 0;
    for (int y = 0; y < size; y++)
    for (int x = 0; x < size; x++, idx++)
    {
        xj[idx] = (RFLOAT)(2.0 * PI * x / size);
        yj[idx] = (RFLOAT)(2.0 * PI * y / size);
        source[idx] = std::complex<RFLOAT>(DIRECT_A2D_ELEM(image, y, x), 0.0);
    }

    std::vector<RFLOAT> target_x(pts.size()), target_y(pts.size());
    for (size_t k = 0; k < pts.size(); k++)
    {
        target_x[k] = pts[k].sx;
        target_y[k] = pts[k].sy;
    }

    std::vector<std::complex<RFLOAT> > output(pts.size());
    finufft_opts opts;

#ifdef RELION_SINGLE_PRECISION
    finufftf_default_opts(&opts);
    opts.modeord = 0;
    const int status = finufftf2d3(nj, xj.data(), yj.data(), source.data(),
                                   -1, (float)1e-6,
                                   (int64_t)pts.size(), target_x.data(), target_y.data(),
                                   output.data(), &opts);
#else
    finufft_default_opts(&opts);
    opts.modeord = 0;
    const int status = finufft2d3(nj, xj.data(), yj.data(), source.data(),
                                  -1, 1e-12,
                                  (int64_t)pts.size(), target_x.data(), target_y.data(),
                                  output.data(), &opts);
#endif

    REQUIRE(status <= 1);

    const RFLOAT norm = (RFLOAT)1 / ((RFLOAT)size * (RFLOAT)size);
    std::vector<Complex> samples(pts.size());
    for (size_t k = 0; k < pts.size(); k++)
        samples[k] = Complex(output[k].real() * norm, output[k].imag() * norm);
    return samples;
}
#endif

} // namespace

TEST_CASE("FFT and NUFFT agree on identical Cartesian s-samples", "[fftw][nufft][smode]")
{
#ifndef RELION_USE_FINUFFT
    SUCCEED("RELION_USE_FINUFFT=OFF: NUFFT consistency test skipped.");
#else
    const int size = 16;
    MultidimArray<RFLOAT> image = makeDeterministicImage(size);

    MultidimArray<Complex> fft_half;
    FourierTransformer transformer;
    transformer.FourierTransform(image, fft_half, true);

    const std::vector<HalfPlanePoint> pts = makeCartesianHalfPlaneSamples(size);
    const std::vector<Complex> nufft_vals = evaluateNufftAtSamples(image, pts);

    for (size_t k = 0; k < pts.size(); k++)
    {
        const Complex fft_val = DIRECT_A2D_ELEM(fft_half, pts[k].row, pts[k].x);
        const Complex nufft_val = nufft_vals[k];

        REQUIRE(nufft_val.real == Approx(fft_val.real).margin(1e-5));
        REQUIRE(nufft_val.imag == Approx(fft_val.imag).margin(1e-5));
    }
#endif
}
