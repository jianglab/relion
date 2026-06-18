/*
 * tests/unit/test_backprojector_nonuniform.cpp
 *
 * Unit tests for BackProjector::backprojectNonuniform2Dto3D.
 *
 * Test strategy:
 *  1. Construction and initZeros
 *  2. Single DC sample: weight accumulates at DC
 *  3. Uniform-weight backprojection of multiple samples
 *  4. Hermitian conjugate symmetry: backprojecting conjugate pairs
 *  5. Ewald sphere curvature changes the data array
 *  6. Anisotropic magnification changes weight distribution
 */

#include <catch2/catch.hpp>

#include "src/backprojector.h"
#include "src/euler.h"
#include "src/multidim_array.h"
#include "src/complex.h"
#include "src/macros.h"
#include "src/matrix2d.h"

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace
{

/// Build a 3×3 rotation matrix from Euler angles (in degrees).
Matrix2D<RFLOAT> eulerMatrix(RFLOAT rot, RFLOAT tilt, RFLOAT psi)
{
    Matrix2D<RFLOAT> A(3, 3);
    Euler_angles2matrix(rot, tilt, psi, A);
    return A;
}

/// Sum the weight array.
RFLOAT sumWeight(const MultidimArray<RFLOAT>& w)
{
    RFLOAT s = 0.0;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(w)
        s += DIRECT_MULTIDIM_ELEM(w, n);
    return s;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// 1. Construction and initZeros
// ---------------------------------------------------------------------------

TEST_CASE("backprojectNonuniform: construction does not throw", "[backprojector][unon]")
{
    REQUIRE_NOTHROW(BackProjector(32, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false));
}

TEST_CASE("backprojectNonuniform: initZeros produces zero weight", "[backprojector][unon]")
{
    BackProjector bp(32, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bp.initZeros();

    RFLOAT total = sumWeight(bp.weight);
    REQUIRE(total == Approx(0.0).margin(1e-12));
}

TEST_CASE("backprojectNonuniform: data and weight have expected padded size", "[backprojector][unon]")
{
    const int ori = 32;
    const int pad = 2;
    BackProjector bp(ori, 3, "c1", TRILINEAR, pad, 10, 0, 1.9, 15, 2, false);
    bp.initZeros();

    const int r_max_val = ori / 2;
    const int psize = 2 * (int(pad * r_max_val + 0.5) + 1) + 1;
    REQUIRE((int)XSIZE(bp.data) == psize / 2 + 1);
    REQUIRE((int)YSIZE(bp.data) == psize);
    REQUIRE((int)ZSIZE(bp.data) == psize);
}

// ---------------------------------------------------------------------------
// 2. Single DC sample accumulates positive weight at DC
// ---------------------------------------------------------------------------

TEST_CASE("backprojectNonuniform: single DC sample gives positive weight at DC",
          "[backprojector][unon]")
{
    const int ori = 32;
    BackProjector bp(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bp.initZeros();

    // A single sample at DC with non-zero value and unit weight
    std::vector<Complex> samples(1, Complex(1.0, 0.0));
    std::vector<RFLOAT> sample_x(1, 0.0);
    std::vector<RFLOAT> sample_y(1, 0.0);
    std::vector<RFLOAT> sample_weight(1, 1.0);
    Matrix2D<RFLOAT> A = eulerMatrix(0.0, 0.0, 0.0);

    // With sample_weight pointer (= data-dependent weighting) and
    // r_ewald_sphere<=0 (flat plane, no Ewald correction)
    bp.backprojectNonuniform2Dto3D(samples, sample_x, sample_y, A,
                                   &sample_weight, -1.0);

    RFLOAT wDC = A3D_ELEM(bp.weight, 0, 0, 0);
    REQUIRE(wDC > 0.0);
}

TEST_CASE("backprojectNonuniform: single DC sample without weight vector",
          "[backprojector][unon]")
{
    const int ori = 32;
    BackProjector bp(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bp.initZeros();

    std::vector<Complex> samples(1, Complex(1.0, 0.0));
    std::vector<RFLOAT> sample_x(1, 0.0);
    std::vector<RFLOAT> sample_y(1, 0.0);
    Matrix2D<RFLOAT> A = eulerMatrix(0.0, 0.0, 0.0);

    // NULL sample_weight -> weight defaults to 1.0 per sample
    bp.backprojectNonuniform2Dto3D(samples, sample_x, sample_y, A,
                                   NULL, -1.0);

    RFLOAT wDC = A3D_ELEM(bp.weight, 0, 0, 0);
    REQUIRE(wDC > 0.0);
}

// ---------------------------------------------------------------------------
// 3. Multiple samples: backprojecting N samples at DC accumulates Nx weight
// ---------------------------------------------------------------------------

TEST_CASE("backprojectNonuniform: two DC samples double the DC weight",
          "[backprojector][unon]")
{
    const int ori = 32;

    // Single DC sample
    BackProjector bp1(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bp1.initZeros();
    {
        std::vector<Complex> s(1, Complex(1.0, 0.0));
        std::vector<RFLOAT> sx(1, 0.0), sy(1, 0.0);
        std::vector<RFLOAT> sw(1, 1.0);
        Matrix2D<RFLOAT> A = eulerMatrix(0.0, 0.0, 0.0);
        bp1.backprojectNonuniform2Dto3D(s, sx, sy, A, &sw, -1.0);
    }

    // Two DC samples
    BackProjector bp2(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bp2.initZeros();
    {
        std::vector<Complex> s(2, Complex(1.0, 0.0));
        std::vector<RFLOAT> sx(2, 0.0), sy(2, 0.0);
        std::vector<RFLOAT> sw(2, 1.0);
        Matrix2D<RFLOAT> A = eulerMatrix(0.0, 0.0, 0.0);
        bp2.backprojectNonuniform2Dto3D(s, sx, sy, A, &sw, -1.0);
    }

    RFLOAT w1 = sumWeight(bp1.weight);
    RFLOAT w2 = sumWeight(bp2.weight);
    // Two samples give approximately double the total weight
    REQUIRE(w2 == Approx(2.0 * w1).epsilon(0.1));
}

// ---------------------------------------------------------------------------
// 4. Off-axis single sample: backproject at known position
// ---------------------------------------------------------------------------

TEST_CASE("backprojectNonuniform: off-axis sample populates non-zero voxel",
          "[backprojector][unon]")
{
    const int ori = 32;
    BackProjector bp(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bp.initZeros();

    // A sample at (8, 0) in pixel units with identity rotation
    std::vector<Complex> samples(1, Complex(1.0, 0.0));
    std::vector<RFLOAT> sample_x(1, 8.0);
    std::vector<RFLOAT> sample_y(1, 0.0);
    std::vector<RFLOAT> sample_weight(1, 1.0);
    Matrix2D<RFLOAT> A = eulerMatrix(0.0, 0.0, 0.0);

    bp.backprojectNonuniform2Dto3D(samples, sample_x, sample_y, A,
                                   &sample_weight, -1.0);

    // Backprojected weight must be non-zero somewhere
    RFLOAT total = sumWeight(bp.weight);
    REQUIRE(total > 0.0);

    // The weight is at the padded voxel corresponding to (8,0,0)
    // (approximately, after padding-factor scaling)
    REQUIRE(A3D_ELEM(bp.weight, 0, 0, 0) == 0.0); // DC unaffected
}

// ---------------------------------------------------------------------------
// 5. Ewald sphere: r_ewald > 0 changes data and weight distributions
// ---------------------------------------------------------------------------

TEST_CASE("backprojectNonuniform: Ewald correction changes backprojected data",
          "[backprojector][unon]")
{
    const int ori = 32;
    std::vector<Complex> samples(1, Complex(1.0, 0.0));
    std::vector<RFLOAT> sx(1, 8.0), sy(1, 4.0);
    std::vector<RFLOAT> sw(1, 1.0);
    Matrix2D<RFLOAT> A = eulerMatrix(0.0, 0.0, 0.0);

    // No Ewald
    BackProjector bpFlat(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bpFlat.initZeros();
    bpFlat.backprojectNonuniform2Dto3D(samples, sx, sy, A, &sw, -1.0);

    // With Ewald (r_ewald=50)
    BackProjector bpEwald(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bpEwald.initZeros();
    bpEwald.backprojectNonuniform2Dto3D(samples, sx, sy, A, &sw, 50.0);

    // Data arrays should differ
    bool identical = true;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bpFlat.data)
    {
        if (std::fabs(DIRECT_MULTIDIM_ELEM(bpFlat.data, n).real -
                      DIRECT_MULTIDIM_ELEM(bpEwald.data, n).real) > 1e-6)
        {
            identical = false;
            break;
        }
    }
    REQUIRE_FALSE(identical);
}

TEST_CASE("backprojectNonuniform: positive and negative Ewald differ in data",
          "[backprojector][unon]")
{
    const int ori = 32;
    const RFLOAT rEwald = 50.0;
    std::vector<Complex> samples(1, Complex(1.0, 0.0));
    std::vector<RFLOAT> sx(1, 8.0), sy(1, 4.0);
    std::vector<RFLOAT> sw(1, 1.0);
    Matrix2D<RFLOAT> A = eulerMatrix(0.0, 0.0, 0.0);

    BackProjector bpPos(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bpPos.initZeros();
    bpPos.backprojectNonuniform2Dto3D(samples, sx, sy, A, &sw, rEwald, true);

    BackProjector bpNeg(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bpNeg.initZeros();
    bpNeg.backprojectNonuniform2Dto3D(samples, sx, sy, A, &sw, rEwald, false);

    // Weights should be identical (same sample, same z-range)
    RFLOAT wPos = sumWeight(bpPos.weight);
    RFLOAT wNeg = sumWeight(bpNeg.weight);
    REQUIRE(wPos == Approx(wNeg).epsilon(1e-6));

    // Data should differ (z positions swapped)
    bool dataDiffers = false;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bpPos.data)
    {
        RFLOAT diff = std::fabs(DIRECT_MULTIDIM_ELEM(bpPos.data, n).real -
                                DIRECT_MULTIDIM_ELEM(bpNeg.data, n).real);
        if (diff > 1e-6) { dataDiffers = true; break; }
    }
    REQUIRE(dataDiffers);
}

// ---------------------------------------------------------------------------
// 6. Anisotropic magnification changes weight distribution
// ---------------------------------------------------------------------------

TEST_CASE("backprojectNonuniform: x-stretch mag changes backprojected data",
          "[backprojector][unon]")
{
    const int ori = 32;
    std::vector<Complex> samples(1, Complex(1.0, 0.0));
    std::vector<RFLOAT> sx(1, 8.0), sy(1, 4.0);
    std::vector<RFLOAT> sw(1, 1.0);
    Matrix2D<RFLOAT> A = eulerMatrix(0.0, 0.0, 0.0);

    // No mag
    BackProjector bpNoMag(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bpNoMag.initZeros();
    bpNoMag.backprojectNonuniform2Dto3D(samples, sx, sy, A, &sw, -1.0, true, NULL);

    // 2x x-magnification
    Matrix2D<RFLOAT> magM(2, 2);
    magM(0,0) = 2.0; magM(0,1) = 0.0;
    magM(1,0) = 0.0; magM(1,1) = 1.0;

    BackProjector bpMag(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bpMag.initZeros();
    bpMag.backprojectNonuniform2Dto3D(samples, sx, sy, A, &sw, -1.0, true, &magM);

    // Total weight should differ (sample mapped to different 3D position)
    RFLOAT wNoMag = sumWeight(bpNoMag.weight);
    RFLOAT wMag   = sumWeight(bpMag.weight);
    REQUIRE(wMag != Approx(wNoMag).epsilon(1e-3));
}

// ---------------------------------------------------------------------------
// 7. Inconsistent array sizes throw errors
// ---------------------------------------------------------------------------

TEST_CASE("backprojectNonuniform: mismatched sample and coordinate sizes throw",
          "[backprojector][unon]")
{
    BackProjector bp(32, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bp.initZeros();

    std::vector<Complex> samples(1, Complex(1.0, 0.0));
    std::vector<RFLOAT> sample_x(2, 0.0);  // size mismatch
    std::vector<RFLOAT> sample_y(1, 0.0);
    Matrix2D<RFLOAT> A = eulerMatrix(0.0, 0.0, 0.0);

    REQUIRE_THROWS(bp.backprojectNonuniform2Dto3D(
        samples, sample_x, sample_y, A, NULL, -1.0));
}

TEST_CASE("backprojectNonuniform: mismatched weight and sample sizes throw",
          "[backprojector][unon]")
{
    BackProjector bp(32, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bp.initZeros();

    std::vector<Complex> samples(1, Complex(1.0, 0.0));
    std::vector<RFLOAT> sample_x(1, 0.0);
    std::vector<RFLOAT> sample_y(1, 0.0);
    std::vector<RFLOAT> sample_weight(2, 1.0); // size mismatch
    Matrix2D<RFLOAT> A = eulerMatrix(0.0, 0.0, 0.0);

    REQUIRE_THROWS(bp.backprojectNonuniform2Dto3D(
        samples, sample_x, sample_y, A, &sample_weight, -1.0));
}

// ---------------------------------------------------------------------------
// 8. Zero-weight samples are skipped
// ---------------------------------------------------------------------------

TEST_CASE("backprojectNonuniform: zero-weight samples are skipped",
          "[backprojector][unon]")
{
    const int ori = 32;
    BackProjector bp(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bp.initZeros();

    std::vector<Complex> samples(2, Complex(1.0, 0.0));
    std::vector<RFLOAT> sample_x(2, 0.0);
    std::vector<RFLOAT> sample_y(2, 0.0);
    std::vector<RFLOAT> sample_weight(2, 0.0); // all zero - skip everything
    Matrix2D<RFLOAT> A = eulerMatrix(0.0, 0.0, 0.0);

    bp.backprojectNonuniform2Dto3D(samples, sample_x, sample_y, A,
                                   &sample_weight, -1.0);

    RFLOAT total = sumWeight(bp.weight);
    REQUIRE(total == Approx(0.0).margin(1e-12));
}

TEST_CASE("backprojectNonuniform: mixing zero and positive weights works",
          "[backprojector][unon]")
{
    const int ori = 32;
    BackProjector bp(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bp.initZeros();

    std::vector<Complex> samples(2, Complex(1.0, 0.0));
    std::vector<RFLOAT> sample_x(2, 0.0);
    std::vector<RFLOAT> sample_y(2, 0.0);
    std::vector<RFLOAT> sample_weight = {0.0, 1.0}; // skip first, use second
    Matrix2D<RFLOAT> A = eulerMatrix(0.0, 0.0, 0.0);

    bp.backprojectNonuniform2Dto3D(samples, sample_x, sample_y, A,
                                   &sample_weight, -1.0);

    // One non-zero-weight DC sample
    RFLOAT wDC = A3D_ELEM(bp.weight, 0, 0, 0);
    REQUIRE(wDC > 0.0);
}
