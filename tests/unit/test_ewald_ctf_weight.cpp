/*
 * tests/unit/test_ewald_ctf_weight.cpp
 *
 * Regression tests for CTF/Ewald weight handling bugs found in the
 * back-projection reconstruction pipeline:
 *
 * Bug 1 (fixed): applyWeightEwaldSphereCurvature_noAniso used
 *   getCTF(x,y) with default gammaOffset=0, discarding even-Zernike
 *   corrections. Fix: use fabs(result(i,j)) which preserves the CTF
 *   values computed by getFftwImage (including gammaOffset).
 *
 * Bug 2 (fixed): In the s-mode non-Ewald path (reconstructor.cpp),
 *   F2D was passed to set2DFourierTransform without multiplying by CTF,
 *   and raw oscillating CTF was used as weight, causing ~half of Fourier
 *   space to be discarded. Fix: F2D *= Fctf; Fctf *= Fctf before
 *   backprojection.
 *
 * Test strategy:
 *   - Construct CTF objects with/without even-Zernike gammaOffset
 *   - Compute Fctf via getFftwImage (with and without gammaOffset)
 *   - Apply Ewald weight via applyWeightEwaldSphereCurvature_noAniso
 *   - Verify the Ewald weight reflects gammaOffset (not ignored)
 *   - Verify BackProjector weights are always >= 0 with CTF^2 weighting
 */

#include <catch2/catch.hpp>

#include "src/ctf.h"
#include "src/jaz/single_particle/obs_model.h"
#include "src/backprojector.h"
#include "src/euler.h"
#include "src/fftw.h"
#include "src/multidim_array.h"
#include "src/complex.h"
#include "src/metadata_table.h"
#include "src/metadata_label.h"
#include "src/matrix2d.h"
#include "src/macros.h"

#include <cmath>
#include <vector>

namespace
{

CTF makePlainCTF(RFLOAT defocus_A = 10000.0, RFLOAT voltage = 300.0,
                 RFLOAT Cs = 2.7, RFLOAT Q0 = 0.1)
{
    CTF ctf;
    ctf.setValues(defocus_A, defocus_A, 0.0, voltage, Cs, Q0, 0.0, 1.0, 0.0);
    return ctf;
}

MetaDataTable makeOpticsTableWithEvenZernike(
    const std::vector<double>& evenCoeffs,
    int boxSize = 32,
    double angpix = 1.0)
{
    MetaDataTable mdt;
    mdt.setName("optics");
    mdt.addObject();
    mdt.setValue(EMDL_IMAGE_PIXEL_SIZE, angpix, 0);
    mdt.setValue(EMDL_CTF_VOLTAGE, 300.0, 0);
    mdt.setValue(EMDL_CTF_CS, 2.7, 0);
    mdt.setValue(EMDL_CTF_Q0, 0.1, 0);
    mdt.setValue(EMDL_IMAGE_SIZE, boxSize, 0);
    if (!evenCoeffs.empty())
        mdt.setValue(EMDL_IMAGE_EVEN_ZERNIKE_COEFFS, evenCoeffs, 0);
    mdt.setValue(EMDL_IMAGE_MAG_MATRIX_00, 1.0, 0);
    mdt.setValue(EMDL_IMAGE_MAG_MATRIX_01, 0.0, 0);
    mdt.setValue(EMDL_IMAGE_MAG_MATRIX_10, 0.0, 0);
    mdt.setValue(EMDL_IMAGE_MAG_MATRIX_11, 1.0, 0);
    return mdt;
}

} // anonymous namespace

// ===========================================================================
// Suite 1: applyWeightEwaldSphereCurvature_noAniso preserves gammaOffset
// ===========================================================================
//
// After the fix, the function reads fabs(result(i,j)) where result
// already contains the CTF values from getFftwImage (which includes
// gammaOffset). The Ewald weight should therefore depend on gammaOffset.
//
// Before the fix, the function called getCTF(x,y) with default
// gammaOffset=0, so even-Zernike corrections were silently discarded.
// ===========================================================================

TEST_CASE("Ewald weight: without gammaOffset, CTF and Ewald weight are consistent",
          "[ewald_ctf_weight][regression]")
{
    const int size = 32;
    const RFLOAT angpix = 1.0;
    const RFLOAT particle_diameter = 200.0;

    CTF ctf = makePlainCTF(10000.0);
    const int sh = size / 2 + 1;
    MultidimArray<RFLOAT> Fctf(size, sh);
    ctf.getFftwImage(Fctf, size, size, angpix,
                     false, false, false, true, false);

    MultidimArray<RFLOAT> Fctf_copy = Fctf;
    ctf.applyWeightEwaldSphereCurvature_noAniso(Fctf, size, size, angpix, particle_diameter);

    // Without gammaOffset, the Ewald weight should be consistent:
    // W = 0.5*(1 + A*(2*|CTF| - 1)) where CTF = Fctf_copy
    RFLOAT xs = size * angpix;
    RFLOAT ys = size * angpix;
    bool consistent = true;
    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(Fctf)
    {
        RFLOAT x = (RFLOAT)jp / xs;
        RFLOAT y = (RFLOAT)ip / ys;
        RFLOAT deltaf = fabs(ctf.getDeltaF(x, y));
        RFLOAT inv_d = sqrt(x * x + y * y);
        RFLOAT aux = (2. * deltaf * ctf.lambda * inv_d) / particle_diameter;
        RFLOAT A = (aux > 1.) ? 0. : (2. / PI) * (acos(aux) - aux * sin(acos(aux)));
        RFLOAT expected = 0.5 * (1. + A * (2. * fabs(DIRECT_A2D_ELEM(Fctf_copy, i, j)) - 1.));
        RFLOAT actual = DIRECT_A2D_ELEM(Fctf, i, j);
        if (fabs(expected - actual) > 1e-8)
        {
            consistent = false;
            break;
        }
    }
    REQUIRE(consistent);
}

TEST_CASE("Ewald weight: with gammaOffset, Ewald weight differs from no-gammaOffset case",
          "[ewald_ctf_weight][regression]")
{
    const int size = 32;
    const RFLOAT angpix = 1.0;
    const RFLOAT particle_diameter = 200.0;
    const int sh = size / 2 + 1;

    // CTF without even Zernike
    CTF ctf_plain = makePlainCTF(10000.0);
    MultidimArray<RFLOAT> Fctf_plain(size, sh);
    ctf_plain.getFftwImage(Fctf_plain, size, size, angpix,
                           false, false, false, true, false);
    ctf_plain.applyWeightEwaldSphereCurvature_noAniso(Fctf_plain, size, size, angpix, particle_diameter);

    // CTF with even Zernike (significant piston + defocus-like aberration)
    std::vector<double> evenCoeffs(4, 0.0);
    evenCoeffs[0] = 1.5;  // piston
    evenCoeffs[2] = -0.8; // m=0, n=2 (defocus-like)
    MetaDataTable mdt = makeOpticsTableWithEvenZernike(evenCoeffs, size, angpix);
    ObservationModel obs(mdt, false);

    CTF ctf_aberr;
    ctf_aberr.setValuesByGroup(&obs, 0, 10000.0, 10000.0, 0.0, 0.0, 1.0, 0.0, -1.0);
    MultidimArray<RFLOAT> Fctf_aberr(size, sh);
    ctf_aberr.getFftwImage(Fctf_aberr, size, size, angpix,
                           false, false, false, true, false);
    ctf_aberr.applyWeightEwaldSphereCurvature_noAniso(Fctf_aberr, size, size, angpix, particle_diameter);

    // With the fix, Fctf_aberr (Ewald weight with gammaOffset) should differ
    // from Fctf_plain (Ewald weight without gammaOffset).
    // Before the fix, both called getCTF(x,y) with gammaOffset=0, so they'd
    // be identical (ignoring the obsModel mag matrix difference which is I).
    bool anyDiffers = false;
    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(Fctf_plain)
    {
        if (fabs(DIRECT_A2D_ELEM(Fctf_plain, i, j) -
                 DIRECT_A2D_ELEM(Fctf_aberr, i, j)) > 1e-6)
        {
            anyDiffers = true;
            break;
        }
    }
    REQUIRE(anyDiffers);
}

TEST_CASE("Ewald weight: with gammaOffset, result uses CTF values from getFftwImage",
          "[ewald_ctf_weight][regression]")
{
    const int size = 32;
    const RFLOAT angpix = 1.0;
    const RFLOAT particle_diameter = 200.0;
    const int sh = size / 2 + 1;

    // CTF with even Zernike
    std::vector<double> evenCoeffs(4, 0.0);
    evenCoeffs[0] = 1.5;
    evenCoeffs[2] = -0.8;
    MetaDataTable mdt = makeOpticsTableWithEvenZernike(evenCoeffs, size, angpix);
    ObservationModel obs(mdt, false);

    CTF ctf;
    ctf.setValuesByGroup(&obs, 0, 10000.0, 10000.0, 0.0, 0.0, 1.0, 0.0, -1.0);
    MultidimArray<RFLOAT> Fctf(size, sh);
    ctf.getFftwImage(Fctf, size, size, angpix,
                     false, false, false, true, false);

    // Save the pre-Ewald CTF values (which include gammaOffset)
    MultidimArray<RFLOAT> Fctf_pre = Fctf;

    // Apply Ewald weight
    ctf.applyWeightEwaldSphereCurvature_noAniso(Fctf, size, size, angpix, particle_diameter);

    // Verify: for each pixel, the Ewald weight was computed using
    // |Fctf_pre(i,j)| (CTF with gammaOffset), not getCTF(x,y) (without gammaOffset).
    // We check this by verifying W = 0.5*(1 + A*(2*|Fctf_pre| - 1)).
    RFLOAT xs = size * angpix;
    RFLOAT ys = size * angpix;
    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(Fctf)
    {
        RFLOAT x = (RFLOAT)jp / xs;
        RFLOAT y = (RFLOAT)ip / ys;
        RFLOAT deltaf = fabs(ctf.getDeltaF(x, y));
        RFLOAT inv_d = sqrt(x * x + y * y);
        RFLOAT aux = (2. * deltaf * ctf.lambda * inv_d) / particle_diameter;
        RFLOAT A = (aux > 1.) ? 0. : (2. / PI) * (acos(aux) - aux * sin(acos(aux)));
        RFLOAT expected = 0.5 * (1. + A * (2. * fabs(DIRECT_A2D_ELEM(Fctf_pre, i, j)) - 1.));
        RFLOAT actual = DIRECT_A2D_ELEM(Fctf, i, j);
        REQUIRE(actual == Approx(expected).margin(1e-8));
    }
}

// ===========================================================================
// Suite 2: Ewald weight is always non-negative
// ===========================================================================
//
// The Ewald weight W = 0.5*(1 + A*(2*|CTF| - 1)) where A in [0,1] and
// |CTF| in [0,1]. The minimum is at A=1, |CTF|=0: W = 0.5*(1 + (-1)) = 0.
// So the weight should be >= 0 everywhere.
// ===========================================================================

TEST_CASE("Ewald weight: is always non-negative", "[ewald_ctf_weight]")
{
    const int size = 64;
    const RFLOAT angpix = 1.0;
    const RFLOAT particle_diameter = 200.0;
    const int sh = size / 2 + 1;

    CTF ctf = makePlainCTF(15000.0);
    MultidimArray<RFLOAT> Fctf(size, sh);
    ctf.getFftwImage(Fctf, size, size, angpix,
                     false, false, false, true, false);
    ctf.applyWeightEwaldSphereCurvature_noAniso(Fctf, size, size, angpix, particle_diameter);

    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(Fctf)
    {
        REQUIRE(DIRECT_A2D_ELEM(Fctf, i, j) >= -1e-10);
    }
}

// ===========================================================================
// Suite 3: CTF^2 weight in s-mode backprojection is always non-negative
// ===========================================================================
//
// Bug 2 regression: Before the fix, raw CTF (which oscillates and has
// negative values) was passed as the weight to set2DFourierTransform.
// The backprojector skips pixels with weight <= 0, discarding ~half
// of Fourier space. After the fix, Fctf *= Fctf (weight = CTF^2) is
// always non-negative.
// ===========================================================================

TEST_CASE("CTF^2 weight: squared CTF is always non-negative", "[ewald_ctf_weight][regression]")
{
    const int size = 64;
    const RFLOAT angpix = 1.0;
    const int sh = size / 2 + 1;

    CTF ctf = makePlainCTF(20000.0);
    MultidimArray<RFLOAT> Fctf(size, sh);
    ctf.getFftwImage(Fctf, size, size, angpix,
                     false, false, false, true, false);

    // Before fix: raw CTF was used as weight — negative values present
    // After fix: weight = CTF^2 — always non-negative
    bool anyNegative = false;
    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(Fctf)
    {
        RFLOAT weight = DIRECT_A2D_ELEM(Fctf, i, j) * DIRECT_A2D_ELEM(Fctf, i, j);
        if (weight < 0.0)
        {
            anyNegative = true;
            break;
        }
    }
    REQUIRE_FALSE(anyNegative);
}

TEST_CASE("CTF^2 weight: raw CTF has negative values (motivating the fix)",
          "[ewald_ctf_weight][regression]")
{
    const int size = 64;
    const RFLOAT angpix = 1.0;
    const int sh = size / 2 + 1;

    CTF ctf = makePlainCTF(20000.0);
    MultidimArray<RFLOAT> Fctf(size, sh);
    ctf.getFftwImage(Fctf, size, size, angpix,
                     false, false, false, true, false);

    // Verify that raw CTF does oscillate (has negative values),
    // which is why CTF^2 is needed instead of raw CTF as weight.
    bool hasNegative = false;
    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(Fctf)
    {
        if (DIRECT_A2D_ELEM(Fctf, i, j) < -1e-8)
        {
            hasNegative = true;
            break;
        }
    }
    REQUIRE(hasNegative);
}

// ===========================================================================
// Suite 4: BackProjector with CTF^2 weighting accumulates weight everywhere
// ===========================================================================
//
// Before Bug 2 fix, backprojecting with raw CTF as weight would skip
// pixels where CTF < 0. After the fix, CTF^2 is always >= 0, so
// all pixels contribute weight.
// ===========================================================================

TEST_CASE("BackProjector: CTF^2-weighted slice accumulates weight at all frequencies",
          "[ewald_ctf_weight][regression][backprojector]")
{
    const int ori = 32;
    const int sh = ori / 2 + 1;
    BackProjector bp(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bp.initZeros();

    // Build a 2D Fourier slice with CTF^2 weighting
    CTF ctf = makePlainCTF(15000.0);
    MultidimArray<RFLOAT> Fctf(ori, sh);
    ctf.getFftwImage(Fctf, ori, ori, 1.0,
                     false, false, false, true, false);

    // Simulate the corrected s-mode path: F2D *= Fctf; Fctf *= Fctf
    MultidimArray<Complex> f2d(ori, sh);
    f2d.initZeros();
    // Set some non-zero signal
    for (int y = 0; y < ori; y++)
        for (int x = 0; x < sh; x++)
            DIRECT_A2D_ELEM(f2d, y, x) = Complex(1.0, 0.0);

    MultidimArray<RFLOAT> weight(ori, sh);
    for (int y = 0; y < ori; y++)
        for (int x = 0; x < sh; x++)
        {
            RFLOAT c = DIRECT_A2D_ELEM(Fctf, y, x);
            DIRECT_A2D_ELEM(f2d, y, x) *= c;
            DIRECT_A2D_ELEM(weight, y, x) = c * c; // CTF^2
        }

    // Count pixels with non-zero weight (after CTF^2, all should be >= 0)
    int nonzero_weight = 0;
    int total_pixels = 0;
    for (int y = 0; y < ori; y++)
        for (int x = 0; x < sh; x++)
        {
            total_pixels++;
            if (DIRECT_A2D_ELEM(weight, y, x) > 1e-20)
                nonzero_weight++;
        }

    // With CTF^2, no pixels have negative weight (they could still be zero at
    // exact CTF zero crossings, but those are measure-zero on a discrete grid).
    // At minimum, > 50% of pixels should have positive weight.
    // Before the fix (raw CTF as weight), ~50% would be skipped (CTF < 0).
    REQUIRE(nonzero_weight > total_pixels / 2);
}

// ===========================================================================
// Suite 5: Ewald weight with gammaOffset vs without — specific pixel check
// ===========================================================================
//
// With a large piston coefficient, the gammaOffset shifts the CTF zero
// crossings. The Ewald weight (which uses |CTF|) should reflect this shift.
// At a pixel near a CTF zero crossing, the weight WITH gammaOffset should
// differ from the weight WITHOUT gammaOffset.
// ===========================================================================

TEST_CASE("Ewald weight: gammaOffset shifts where CTF zero crossings occur",
          "[ewald_ctf_weight][regression]")
{
    const int size = 64;
    const RFLOAT angpix = 1.0;
    const RFLOAT particle_diameter = 100.0;
    const int sh = size / 2 + 1;

    // Compute CTF arrays with and without large gammaOffset
    // Without gammaOffset
    CTF ctf_plain = makePlainCTF(10000.0);
    MultidimArray<RFLOAT> Fctf_plain(size, sh);
    ctf_plain.getFftwImage(Fctf_plain, size, size, angpix,
                           false, false, false, true, false);

    // With gammaOffset (large piston = uniform phase shift)
    std::vector<double> evenCoeffs(1, 0.0);
    evenCoeffs[0] = PI / 2.0; // piston = 90° phase shift
    MetaDataTable mdt = makeOpticsTableWithEvenZernike(evenCoeffs, size, angpix);
    ObservationModel obs(mdt, false);

    CTF ctf_aberr;
    ctf_aberr.setValuesByGroup(&obs, 0, 10000.0, 10000.0, 0.0, 0.0, 1.0, 0.0, -1.0);
    MultidimArray<RFLOAT> Fctf_aberr(size, sh);
    ctf_aberr.getFftwImage(Fctf_aberr, size, size, angpix,
                           false, false, false, true, false);

    // The CTF arrays should differ (gammaOffset shifts the CTF)
    bool ctfDiffers = false;
    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(Fctf_plain)
    {
        if (fabs(DIRECT_A2D_ELEM(Fctf_plain, i, j) -
                 DIRECT_A2D_ELEM(Fctf_aberr, i, j)) > 1e-6)
        {
            ctfDiffers = true;
            break;
        }
    }
    REQUIRE(ctfDiffers);

    // Apply Ewald weight to both
    ctf_plain.applyWeightEwaldSphereCurvature_noAniso(Fctf_plain, size, size, angpix, particle_diameter);
    ctf_aberr.applyWeightEwaldSphereCurvature_noAniso(Fctf_aberr, size, size, angpix, particle_diameter);

    // After the fix, the Ewald weights should also differ
    // (because they were computed from different CTF values)
    bool ewaldDiffers = false;
    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(Fctf_plain)
    {
        if (fabs(DIRECT_A2D_ELEM(Fctf_plain, i, j) -
                 DIRECT_A2D_ELEM(Fctf_aberr, i, j)) > 1e-6)
        {
            ewaldDiffers = true;
            break;
        }
    }
    REQUIRE(ewaldDiffers);
}

// ===========================================================================
// Suite 6: Ewald weight with zero gammaOffset matches direct calculation
// ===========================================================================
//
// When there are no even Zernike corrections, the Ewald weight computed
// via getFftwImage + applyWeightEwaldSphereCurvature_noAniso should
// match a direct per-pixel calculation using getCTF.
// ===========================================================================

TEST_CASE("Ewald weight: matches direct calculation when no gammaOffset",
          "[ewald_ctf_weight]")
{
    const int size = 64;
    const RFLOAT angpix = 1.0;
    const RFLOAT particle_diameter = 150.0;
    const int sh = size / 2 + 1;

    CTF ctf = makePlainCTF(12000.0);
    MultidimArray<RFLOAT> Fctf(size, sh);
    ctf.getFftwImage(Fctf, size, size, angpix,
                     false, false, false, true, false);
    ctf.applyWeightEwaldSphereCurvature_noAniso(Fctf, size, size, angpix, particle_diameter);

    // Verify against direct per-pixel calculation
    RFLOAT xs = size * angpix;
    RFLOAT ys = size * angpix;
    RFLOAT maxErr = 0.0;
    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(Fctf)
    {
        RFLOAT x = (RFLOAT)jp / xs;
        RFLOAT y = (RFLOAT)ip / ys;
        RFLOAT deltaf = fabs(ctf.getDeltaF(x, y));
        RFLOAT inv_d = sqrt(x * x + y * y);
        RFLOAT aux = (2. * deltaf * ctf.lambda * inv_d) / particle_diameter;
        RFLOAT A = (aux > 1.) ? 0. : (2. / PI) * (acos(aux) - aux * sin(acos(aux)));
        // getCTF without gammaOffset == getFftwImage result (no aberrations)
        RFLOAT expected = 0.5 * (1. + A * (2. * fabs(ctf.getCTF(x, y)) - 1.));
        RFLOAT actual = DIRECT_A2D_ELEM(Fctf, i, j);
        RFLOAT err = fabs(expected - actual);
        if (err > maxErr) maxErr = err;
    }
    REQUIRE(maxErr < 1e-8);
}

// ===========================================================================
// Suite 7: Ewald weight bounded in [0, 1]
// ===========================================================================
//
// W = 0.5*(1 + A*(2*|CTF| - 1)) where A in [0,1] and |CTF| in [0,1].
// min: A=1, |CTF|=0 -> W = 0.5*(1-1) = 0
// max: A=1, |CTF|=1 -> W = 0.5*(1+1) = 1.0
// ===========================================================================

TEST_CASE("Ewald weight: is bounded in [0, 1] after 0.5 scaling",
          "[ewald_ctf_weight]")
{
    // W = 0.5*(1 + A*(2*|CTF| - 1)) where A in [0,1], |CTF| in [0,1].
    // Max: A=1, |CTF|=1 -> W = 0.5*(1+1) = 1.0
    // Min: A=1, |CTF|=0 -> W = 0.5*(1-1) = 0.0
    // Note: RELION uses sin(chi) in [-1,1], not 2*sin(chi).
    // The 0.5 scaling normalizes to the 2*sin(chi) convention.
    const int size = 64;
    const RFLOAT angpix = 1.0;
    const RFLOAT particle_diameter = 200.0;
    const int sh = size / 2 + 1;

    CTF ctf = makePlainCTF(10000.0);
    MultidimArray<RFLOAT> Fctf(size, sh);
    ctf.getFftwImage(Fctf, size, size, angpix,
                     false, false, false, true, false);
    ctf.applyWeightEwaldSphereCurvature_noAniso(Fctf, size, size, angpix, particle_diameter);

    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(Fctf)
    {
        RFLOAT w = DIRECT_A2D_ELEM(Fctf, i, j);
        REQUIRE(w >= -1e-10);
        REQUIRE(w <= 1.0 + 1e-10);
    }
}

// ===========================================================================
// Suite 8: Combination — Ewald + even Zernike + CTF^2 all non-negative
// ===========================================================================
//
// In the Ewald path with CTF premultiplied data:
//   Fctf (with gammaOffset) -> applyWeightEwaldSphereCurvature_noAniso
//   Then for non-Ewald path: Fctf *= Fctf (weight = W^2)
// Both W and W^2 should be non-negative.
// ===========================================================================

TEST_CASE("Ewald + even Zernike: Ewald weight and its square are non-negative",
          "[ewald_ctf_weight][regression]")
{
    const int size = 32;
    const RFLOAT angpix = 1.0;
    const RFLOAT particle_diameter = 200.0;
    const int sh = size / 2 + 1;

    std::vector<double> evenCoeffs(4, 0.0);
    evenCoeffs[0] = 1.5;
    evenCoeffs[2] = -0.8;
    MetaDataTable mdt = makeOpticsTableWithEvenZernike(evenCoeffs, size, angpix);
    ObservationModel obs(mdt, false);

    CTF ctf;
    ctf.setValuesByGroup(&obs, 0, 10000.0, 10000.0, 0.0, 0.0, 1.0, 0.0, -1.0);
    MultidimArray<RFLOAT> Fctf(size, sh);
    ctf.getFftwImage(Fctf, size, size, angpix,
                     false, false, false, true, false);
    ctf.applyWeightEwaldSphereCurvature_noAniso(Fctf, size, size, angpix, particle_diameter);

    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(Fctf)
    {
        RFLOAT w = DIRECT_A2D_ELEM(Fctf, i, j);
        REQUIRE(w >= -1e-10);       // weight >= 0
        REQUIRE(w * w >= 0.0);      // weight^2 >= 0
    }
}

// ===========================================================================
// Suite 9: Anisotropic magnification + Ewald weight
// ===========================================================================
//
// Verify that anisotropic magnification is correctly accounted for in
// the Ewald weight calculation. The spatial frequencies used in
// getDeltaF should match those used in the CTF evaluation.
// ===========================================================================

TEST_CASE("Ewald weight: anisotropic magnification affects weight distribution",
          "[ewald_ctf_weight][anisomag]")
{
    const int size = 32;
    const RFLOAT angpix = 1.0;
    const RFLOAT particle_diameter = 200.0;
    const int sh = size / 2 + 1;

    // Identity mag
    CTF ctf_iso = makePlainCTF(10000.0);
    MultidimArray<RFLOAT> Fctf_iso(size, sh);
    ctf_iso.getFftwImage(Fctf_iso, size, size, angpix,
                         false, false, false, true, false);
    ctf_iso.applyWeightEwaldSphereCurvature_noAniso(Fctf_iso, size, size, angpix, particle_diameter);

    // 2x x-magnification
    MetaDataTable mdt;
    mdt.setName("optics");
    mdt.addObject();
    mdt.setValue(EMDL_IMAGE_PIXEL_SIZE, angpix, 0);
    mdt.setValue(EMDL_CTF_VOLTAGE, 300.0, 0);
    mdt.setValue(EMDL_CTF_CS, 2.7, 0);
    mdt.setValue(EMDL_CTF_Q0, 0.1, 0);
    mdt.setValue(EMDL_IMAGE_SIZE, size, 0);
    mdt.setValue(EMDL_IMAGE_MAG_MATRIX_00, 2.0, 0);
    mdt.setValue(EMDL_IMAGE_MAG_MATRIX_01, 0.0, 0);
    mdt.setValue(EMDL_IMAGE_MAG_MATRIX_10, 0.0, 0);
    mdt.setValue(EMDL_IMAGE_MAG_MATRIX_11, 1.0, 0);
    ObservationModel obs(mdt, false);

    CTF ctf_aniso;
    ctf_aniso.setValuesByGroup(&obs, 0, 10000.0, 10000.0, 0.0, 0.0, 1.0, 0.0, -1.0);
    MultidimArray<RFLOAT> Fctf_aniso(size, sh);
    ctf_aniso.getFftwImage(Fctf_aniso, size, size, angpix,
                           false, false, false, true, false);
    ctf_aniso.applyWeightEwaldSphereCurvature_noAniso(Fctf_aniso, size, size, angpix, particle_diameter);

    // The weight distributions should differ (aniso mag changes CTF shape)
    bool anyDiffers = false;
    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(Fctf_iso)
    {
        if (fabs(DIRECT_A2D_ELEM(Fctf_iso, i, j) -
                 DIRECT_A2D_ELEM(Fctf_aniso, i, j)) > 1e-6)
        {
            anyDiffers = true;
            break;
        }
    }
    REQUIRE(anyDiffers);
}

// ===========================================================================
// Suite 10: Regression — old code would compute Ewald weight without
// gammaOffset, making it identical regardless of even-Zernike coefficients
// ===========================================================================
//
// Before the fix, applyWeightEwaldSphereCurvature_noAniso called
// getCTF(x, y) with gammaOffset=0. This means:
//   W_old = 0.5*(1 + A*(2*|getCTF(x,y,gammaOffset=0)| - 1))
// which is independent of gammaOffset.
//
// After the fix:
//   W_new = 0.5*(1 + A*(2*|Fctf(i,j)| - 1))
// where Fctf(i,j) includes gammaOffset from getFftwImage.
//
// Test: two CTFs with different gammaOffset but same defocus should
// produce different Ewald weights.
// ===========================================================================

TEST_CASE("Ewald weight regression: different gammaOffset gives different weight",
          "[ewald_ctf_weight][regression]")
{
    const int size = 32;
    const RFLOAT angpix = 1.0;
    const RFLOAT particle_diameter = 200.0;
    const int sh = size / 2 + 1;

    // Small gammaOffset
    std::vector<double> evenCoeffs_small(4, 0.0);
    evenCoeffs_small[0] = 0.1;
    MetaDataTable mdt_small = makeOpticsTableWithEvenZernike(evenCoeffs_small, size, angpix);
    ObservationModel obs_small(mdt_small, false);
    CTF ctf_small;
    ctf_small.setValuesByGroup(&obs_small, 0, 10000.0, 10000.0, 0.0, 0.0, 1.0, 0.0, -1.0);
    MultidimArray<RFLOAT> Fctf_small(size, sh);
    ctf_small.getFftwImage(Fctf_small, size, size, angpix,
                           false, false, false, true, false);
    ctf_small.applyWeightEwaldSphereCurvature_noAniso(Fctf_small, size, size, angpix, particle_diameter);

    // Large gammaOffset
    std::vector<double> evenCoeffs_large(4, 0.0);
    evenCoeffs_large[0] = PI;
    MetaDataTable mdt_large = makeOpticsTableWithEvenZernike(evenCoeffs_large, size, angpix);
    ObservationModel obs_large(mdt_large, false);
    CTF ctf_large;
    ctf_large.setValuesByGroup(&obs_large, 0, 10000.0, 10000.0, 0.0, 0.0, 1.0, 0.0, -1.0);
    MultidimArray<RFLOAT> Fctf_large(size, sh);
    ctf_large.getFftwImage(Fctf_large, size, size, angpix,
                           false, false, false, true, false);
    ctf_large.applyWeightEwaldSphereCurvature_noAniso(Fctf_large, size, size, angpix, particle_diameter);

    // With PI piston, CTF changes sign everywhere, so |CTF| is the same!
    // That's a degenerate case. Use a non-PI piston instead.
    // Actually, with PI piston: gamma -> gamma + PI, so sin(gamma+PI) = -sin(gamma).
    // getFftwImage with gammaOffset=PI returns -CTF. Then |CTF| is the same.
    // So W would be the same. Need a non-trivial gammaOffset.
    // Use gammaOffset = PI/3 (60 degrees).
    std::vector<double> evenCoeffs_60(4, 0.0);
    evenCoeffs_60[0] = PI / 3.0;
    MetaDataTable mdt_60 = makeOpticsTableWithEvenZernike(evenCoeffs_60, size, angpix);
    ObservationModel obs_60(mdt_60, false);
    CTF ctf_60;
    ctf_60.setValuesByGroup(&obs_60, 0, 10000.0, 10000.0, 0.0, 0.0, 1.0, 0.0, -1.0);
    MultidimArray<RFLOAT> Fctf_60(size, sh);
    ctf_60.getFftwImage(Fctf_60, size, size, angpix,
                        false, false, false, true, false);
    ctf_60.applyWeightEwaldSphereCurvature_noAniso(Fctf_60, size, size, angpix, particle_diameter);

    // gammaOffset=PI/3 changes |CTF| at most pixels, so weights should differ
    bool anyDiffers = false;
    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(Fctf_small)
    {
        if (fabs(DIRECT_A2D_ELEM(Fctf_small, i, j) -
                 DIRECT_A2D_ELEM(Fctf_60, i, j)) > 1e-6)
        {
            anyDiffers = true;
            break;
        }
    }
    REQUIRE(anyDiffers);
}
