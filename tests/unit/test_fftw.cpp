/*
 * tests/unit/test_fftw.cpp
 *
 * Unit tests for FourierTransformer (wrapping FFTW).
 *
 * RELION normalises the FORWARD transform by dividing by size = Nx*Ny*Nz,
 * so the conventions are:
 *
 *  DC bin after FFT:     V[0] = mean(v)       (i.e. sum(v) / N)
 *  Round-trip identity:  IFFT(FFT(v)) == v    (no extra factor needed)
 *  Parseval:             N * sum|V|^2 == sum|v|^2  (because V is already 1/N scaled)
 */

#include <catch2/catch.hpp>

#include "src/fftw.h"
#include "src/multidim_array.h"
#include "src/macros.h"

#include <cmath>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace
{

/// Fill a 1D array with a pure sine wave of frequency k.
void fillSine1D(MultidimArray<RFLOAT>& v, int freq)
{
    const int N = XSIZE(v);
    for (int i = 0; i < N; i++)
        DIRECT_A1D_ELEM(v, i) = std::sin(2.0 * PI * freq * i / N);
}

/// Fill a 1D array with a constant value.
void fillConstant1D(MultidimArray<RFLOAT>& v, RFLOAT val)
{
    for (int i = 0; i < (int)XSIZE(v); i++)
        DIRECT_A1D_ELEM(v, i) = val;
}

/// Compute sum of |x|^2 over a real array.
RFLOAT sumSquares(const MultidimArray<RFLOAT>& v)
{
    RFLOAT s = 0.0;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(v)
        s += DIRECT_MULTIDIM_ELEM(v, n) * DIRECT_MULTIDIM_ELEM(v, n);
    return s;
}

/// Compute sum of |X|^2 over a half-spectrum Complex array, accounting for
/// the Hermitian symmetry (each non-DC, non-Nyquist bin counts twice).
RFLOAT parseval1D_halfSpectrum(const MultidimArray<Complex>& V, int N)
{
    const int sh = N / 2 + 1;
    RFLOAT s = 0.0;
    for (int k = 0; k < sh; k++)
    {
        const Complex& c = DIRECT_A1D_ELEM(V, k);
        RFLOAT mag2 = c.real * c.real + c.imag * c.imag;
        s += (k == 0 || k == N / 2) ? mag2 : 2.0 * mag2;
    }
    return s;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// 1D tests
// ---------------------------------------------------------------------------

TEST_CASE("FFT1D: round-trip IFFT(FFT(x)) recovers x exactly", "[fftw]")
{
    // RELION already normalises the forward FFT, so IFFT(FFT(x)) == x.
    const int N = 64;
    MultidimArray<RFLOAT> v(N);
    fillSine1D(v, 3);

    MultidimArray<Complex> V;
    MultidimArray<RFLOAT>  vback(N);

    FourierTransformer transformer;
    transformer.FourierTransform(v, V, true);
    transformer.inverseFourierTransform(V, vback);

    for (int i = 0; i < N; i++)
    {
        RFLOAT expected = DIRECT_A1D_ELEM(v, i);
        RFLOAT got      = DIRECT_A1D_ELEM(vback, i);
        REQUIRE(got == Approx(expected).epsilon(1e-5).margin(1e-10));
    }
}

TEST_CASE("FFT1D: DC bin equals mean of input", "[fftw]")
{
    // After normalisation: V[0] = sum(v) / N = mean(v) = C for constant input
    const int N = 64;
    const RFLOAT C = 3.7;
    MultidimArray<RFLOAT> v(N);
    fillConstant1D(v, C);

    MultidimArray<Complex> V;
    FourierTransformer transformer;
    transformer.FourierTransform(v, V, true);

    RFLOAT dc_real = DIRECT_A1D_ELEM(V, 0).real;
    RFLOAT dc_imag = DIRECT_A1D_ELEM(V, 0).imag;
    REQUIRE(dc_real == Approx(C).epsilon(1e-6));
    REQUIRE(dc_imag == Approx(0.0).margin(1e-8));
}

TEST_CASE("FFT1D: zero input gives zero output", "[fftw]")
{
    const int N = 32;
    MultidimArray<RFLOAT> v(N);
    v.initZeros();

    MultidimArray<Complex> V;
    FourierTransformer transformer;
    transformer.FourierTransform(v, V, true);

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V)
    {
        const Complex& c = DIRECT_MULTIDIM_ELEM(V, n);
        REQUIRE(c.real == Approx(0.0).margin(1e-10));
        REQUIRE(c.imag == Approx(0.0).margin(1e-10));
    }
}

TEST_CASE("FFT1D: Parseval's theorem (normalised): N*sum|V|^2 == sum|v|^2", "[fftw]")
{
    // With RELION's normalisation (V = FFT(v)/N):
    //   sum|v|^2 = N * sum_full|V|^2
    // Reconstructing sum_full from half-spectrum using Hermitian symmetry.
    const int N = 64;
    MultidimArray<RFLOAT> v(N);
    fillSine1D(v, 5);

    MultidimArray<Complex> V;
    FourierTransformer transformer;
    transformer.FourierTransform(v, V, true);

    RFLOAT parseval_fft = (RFLOAT)N * parseval1D_halfSpectrum(V, N);
    RFLOAT parseval_real = sumSquares(v);

    REQUIRE(parseval_fft == Approx(parseval_real).epsilon(1e-4));
}

TEST_CASE("FFT1D: single-frequency sine concentrates energy at that bin", "[fftw]")
{
    const int N = 64;
    const int freq = 7;
    MultidimArray<RFLOAT> v(N);
    fillSine1D(v, freq);

    MultidimArray<Complex> V;
    FourierTransformer transformer;
    transformer.FourierTransform(v, V, true);

    // After normalisation the peak at freq has magnitude ~0.5/N (amplitude/2)/N * N = 0.5
    // All other bins (except the Hermitian partner) should be negligible.
    const int sh = N / 2 + 1;
    for (int k = 0; k < sh; k++)
    {
        const Complex& c = DIRECT_A1D_ELEM(V, k);
        RFLOAT mag = std::hypot(c.real, c.imag);
        if (k == freq)
            REQUIRE(mag > 0.01);    // strong peak at signal frequency
        else
            REQUIRE(mag < 1e-5);    // all other bins negligible
    }
}

// ---------------------------------------------------------------------------
// 2D tests
// ---------------------------------------------------------------------------

TEST_CASE("FFT2D: round-trip recovers input image", "[fftw]")
{
    const int Nx = 32, Ny = 32;
    MultidimArray<RFLOAT> img(Ny, Nx);
    for (int y = 0; y < Ny; y++)
        for (int x = 0; x < Nx; x++)
            DIRECT_A2D_ELEM(img, y, x) = std::sin(2.0 * PI * 3 * x / Nx) *
                                          std::cos(2.0 * PI * 2 * y / Ny);

    MultidimArray<Complex> IMG;
    MultidimArray<RFLOAT>  imgback(Ny, Nx);

    FourierTransformer transformer;
    transformer.FourierTransform(img, IMG, true);
    transformer.inverseFourierTransform(IMG, imgback);

    for (int y = 0; y < Ny; y++)
        for (int x = 0; x < Nx; x++)
        {
            RFLOAT expected = DIRECT_A2D_ELEM(img, y, x);
            RFLOAT got      = DIRECT_A2D_ELEM(imgback, y, x);
            REQUIRE(got == Approx(expected).epsilon(1e-5).margin(1e-10));
        }
}

TEST_CASE("FFT2D: DC bin equals mean of input", "[fftw]")
{
    // After normalisation V[0,0] = sum(img) / (Nx*Ny) = C for constant image
    const int Nx = 32, Ny = 32;
    const RFLOAT C = 2.5;
    MultidimArray<RFLOAT> img(Ny, Nx);
    img.initConstant(C);

    MultidimArray<Complex> IMG;
    FourierTransformer transformer;
    transformer.FourierTransform(img, IMG, true);

    RFLOAT dc_real = DIRECT_A2D_ELEM(IMG, 0, 0).real;
    RFLOAT dc_imag = DIRECT_A2D_ELEM(IMG, 0, 0).imag;
    REQUIRE(dc_real == Approx(C).epsilon(1e-5));
    REQUIRE(dc_imag == Approx(0.0).margin(1e-8));
}

TEST_CASE("FFT2D: DC bin is real for real input", "[fftw]")
{
    const int Nx = 32, Ny = 32;
    MultidimArray<RFLOAT> img(Ny, Nx);
    for (int y = 0; y < Ny; y++)
        for (int x = 0; x < Nx; x++)
            DIRECT_A2D_ELEM(img, y, x) = (RFLOAT)(y * Nx + x + 1) / (RFLOAT)(Nx * Ny);

    MultidimArray<Complex> IMG;
    FourierTransformer transformer;
    transformer.FourierTransform(img, IMG, true);

    Complex& DC = DIRECT_A2D_ELEM(IMG, 0, 0);
    REQUIRE(std::fabs(DC.imag) < 1e-6);
}
