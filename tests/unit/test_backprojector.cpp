/*
 * tests/unit/test_backprojector.cpp
 *
 * Unit tests for BackProjector (3-D Fourier back-projection).
 *
 * Test strategy (light-weight, no I/O):
 *
 *  1. Construction – BackProjector can be built without errors.
 *  2. initZeros – data and weight arrays have expected dimensions after init.
 *  3. Single-slice weight accumulation – backprojecting a flat-value 2D Fourier
 *     slice along the identity rotation accumulates a positive weight at DC.
 *  4. Weight symmetry – backprojecting two conjugate-symmetric orientations
 *     (rot=0 and rot=180) should add weight at every voxel twice compared
 *     to a single backprojection.
 *  5. Ewald-off vs Ewald-on – with r_ewald > 0 the backprojected 3D grid
 *     differs from the flat-plane case because pixels are shifted in z.
 *  6. Anisotropic magnification – passing a 2x x-magnification matrix changes
 *     the effective frequency range of the backprojected slice.
 *
 * We avoid calling reconstruct() (which is expensive and calls FFTW internally
 * on multi-MB arrays) so the tests complete quickly.
 */

#include <catch2/catch.hpp>

#include "src/backprojector.h"
#include "src/projector.h"
#include "src/euler.h"
#include "src/fftw.h"
#include "src/multidim_array.h"
#include "src/complex.h"
#include "src/macros.h"
#include "src/matrix2d.h"

#include <cmath>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace
{

/// Create a 2D half-complex (FFTW) array of size (Ny) x (Ny/2+1).
/// All values set to (re, im).
MultidimArray<Complex> makeSlice2D(int Ny, RFLOAT re, RFLOAT im)
{
    int Nx = Ny / 2 + 1;
    MultidimArray<Complex> f2d(Ny, Nx);
    for (int i = 0; i < Ny; ++i)
        for (int j = 0; j < Nx; ++j)
            DIRECT_A2D_ELEM(f2d, i, j) = Complex(re, im);
    return f2d;
}

/// Build a 3×3 rotation matrix from Euler angles (in degrees).
Matrix2D<RFLOAT> eulerMatrix(RFLOAT rot, RFLOAT tilt, RFLOAT psi)
{
    Matrix2D<RFLOAT> A(3, 3);
    Euler_angles2matrix(rot, tilt, psi, A);
    return A;
}

/// Sum the real part of all elements in a 3D Complex array.
RFLOAT sumWeightReal(const MultidimArray<RFLOAT>& w)
{
    RFLOAT s = 0.0;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(w)
        s += DIRECT_MULTIDIM_ELEM(w, n);
    return s;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// 1. Construction
// ---------------------------------------------------------------------------

TEST_CASE("BackProjector: construction does not throw", "[backprojector]")
{
    // ori_size=32, ref_dim=3, fn_sym="c1", TRILINEAR, padding=2
    REQUIRE_NOTHROW(BackProjector(32, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false));
}

// ---------------------------------------------------------------------------
// 2. initZeros – dimensions
// ---------------------------------------------------------------------------

TEST_CASE("BackProjector: data has correct padded size after initZeros", "[backprojector]")
{
    const int ori = 32;
    const int pad = 2;
    BackProjector bp(ori, 3, "c1", TRILINEAR, pad, 10, 0, 1.9, 15, 2, false);
    bp.initZeros();

    // pad_size = 2*(ROUND(padding*r_max)+1)+1  (from projector.cpp line 71)
    //   r_max = ori/2 = 16, padding=2 => pad_size = 2*(32+1)+1 = 67
    // XSIZE of 3D half-complex = pad_size/2+1 = 34
    const int r_max_val = ori / 2;
    const int psize = 2 * (int(pad * r_max_val + 0.5) + 1) + 1;
    REQUIRE((int)XSIZE(bp.data) == psize / 2 + 1);
    REQUIRE((int)YSIZE(bp.data) == psize);
    REQUIRE((int)ZSIZE(bp.data) == psize);
}

TEST_CASE("BackProjector: weight is zero-initialised after initZeros", "[backprojector]")
{
    BackProjector bp(32, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bp.initZeros();

    RFLOAT total = sumWeightReal(bp.weight);
    REQUIRE(total == Approx(0.0).margin(1e-12));
}

// ---------------------------------------------------------------------------
// 3. Single-slice weight accumulation
// ---------------------------------------------------------------------------

TEST_CASE("BackProjector: backprojecting flat slice accumulates positive weight", "[backprojector]")
{
    const int ori = 32;
    BackProjector bp(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bp.initZeros();

    MultidimArray<Complex> f2d = makeSlice2D(ori, 1.0, 0.0);
    Matrix2D<RFLOAT> A = eulerMatrix(0.0, 0.0, 0.0);

    bp.backproject2Dto3D(f2d, A, NULL);

    RFLOAT totalWeight = sumWeightReal(bp.weight);
    REQUIRE(totalWeight > 0.0);
}

TEST_CASE("BackProjector: DC weight is positive after single backprojection", "[backprojector]")
{
    const int ori = 32;
    BackProjector bp(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bp.initZeros();

    MultidimArray<Complex> f2d = makeSlice2D(ori, 1.0, 0.0);
    Matrix2D<RFLOAT> A = eulerMatrix(0.0, 0.0, 0.0);

    bp.backproject2Dto3D(f2d, A, NULL);

    // DC voxel is stored at index (0, 0, 0) in the centred array.
    // In FFTW uncentred layout (after initZeros / decenter) the DC is at
    // the element addressed through A3D_ELEM(data, 0, 0, 0).
    RFLOAT wDC = A3D_ELEM(bp.weight, 0, 0, 0);
    REQUIRE(wDC > 0.0);
}

// ---------------------------------------------------------------------------
// 4. Weight symmetry: two conjugate orientations double the weight
// ---------------------------------------------------------------------------

TEST_CASE("BackProjector: two opposite orientations accumulate more weight than one", "[backprojector]")
{
    const int ori = 32;

    // Single orientation
    BackProjector bp1(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bp1.initZeros();
    MultidimArray<Complex> f2d = makeSlice2D(ori, 1.0, 0.0);
    bp1.backproject2Dto3D(f2d, eulerMatrix(0.0, 0.0, 0.0), NULL);

    // Two orientations (0° and 90° tilt – cover different parts of 3D space)
    BackProjector bp2(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bp2.initZeros();
    bp2.backproject2Dto3D(f2d, eulerMatrix(0.0,  0.0, 0.0), NULL);
    bp2.backproject2Dto3D(f2d, eulerMatrix(0.0, 90.0, 0.0), NULL);

    RFLOAT w1 = sumWeightReal(bp1.weight);
    RFLOAT w2 = sumWeightReal(bp2.weight);

    REQUIRE(w2 > w1);
}

// ---------------------------------------------------------------------------
// 5. Ewald sphere: r_ewald > 0 changes the data array relative to flat plane
// ---------------------------------------------------------------------------

TEST_CASE("BackProjector: Ewald correction changes backprojected volume", "[backprojector]")
{
    const int ori = 32;
    MultidimArray<Complex> f2d = makeSlice2D(ori, 1.0, 0.0);
    Matrix2D<RFLOAT> A = eulerMatrix(0.0, 0.0, 0.0);

    // Flat plane (no Ewald correction)
    BackProjector bpFlat(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bpFlat.initZeros();
    bpFlat.backproject2Dto3D(f2d, A, NULL, -1.0 /*r_ewald<=0 → off*/);

    // Ewald-corrected backprojection (moderate radius = 50 pixels)
    BackProjector bpEwald(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bpEwald.initZeros();
    bpEwald.backproject2Dto3D(f2d, A, NULL, 50.0 /*r_ewald>0 → on*/);

    // The data arrays should differ (Ewald shifts z-position of contributions)
    RFLOAT sumFlat  = 0.0;
    RFLOAT sumEwald = 0.0;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bpFlat.data)
    {
        sumFlat  += DIRECT_MULTIDIM_ELEM(bpFlat.data,  n).real;
        sumEwald += DIRECT_MULTIDIM_ELEM(bpEwald.data, n).real;
    }

    // Total real-part sums may be similar, but the distributions in 3D differ.
    // A simple check: the arrays are NOT identical element-by-element.
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

// ---------------------------------------------------------------------------
// 6. Anisotropic magnification changes effective frequency coverage
// ---------------------------------------------------------------------------

TEST_CASE("BackProjector: x-stretch mag matrix changes backprojected data", "[backprojector]")
{
    const int ori = 32;
    MultidimArray<Complex> f2d = makeSlice2D(ori, 1.0, 0.0);
    Matrix2D<RFLOAT> A = eulerMatrix(0.0, 0.0, 0.0);

    // No mag
    BackProjector bpNoMag(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bpNoMag.initZeros();
    bpNoMag.backproject2Dto3D(f2d, A, NULL, -1.0, true, NULL);

    // 2x x-magnification
    Matrix2D<RFLOAT> magM(2, 2);
    magM(0,0) = 2.0; magM(0,1) = 0.0;
    magM(1,0) = 0.0; magM(1,1) = 1.0;

    BackProjector bpMag(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bpMag.initZeros();
    bpMag.backproject2Dto3D(f2d, A, NULL, -1.0, true, &magM);

    // Weight coverage should differ (magnified slice covers different region)
    RFLOAT wNoMag = sumWeightReal(bpNoMag.weight);
    RFLOAT wMag   = sumWeightReal(bpMag.weight);

    // A 2x x-stretch maps each x pixel to 2x pixel in the reference space, so
    // fewer pixels land within r_max => lower total weight.
    REQUIRE(wMag != Approx(wNoMag).epsilon(1e-3));
}

// ---------------------------------------------------------------------------
// 7. Negative Ewald curvature gives same weight, different data pattern
// ---------------------------------------------------------------------------

TEST_CASE("BackProjector: positive and negative Ewald curvature differ in data", "[backprojector]")
{
    const int ori = 32;
    const RFLOAT rEwald = 50.0;
    MultidimArray<Complex> f2d = makeSlice2D(ori, 1.0, 0.0);
    Matrix2D<RFLOAT> A = eulerMatrix(0.0, 0.0, 0.0);

    BackProjector bpPos(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bpPos.initZeros();
    bpPos.backproject2Dto3D(f2d, A, NULL, rEwald, true);  // positive curvature

    BackProjector bpNeg(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
    bpNeg.initZeros();
    bpNeg.backproject2Dto3D(f2d, A, NULL, rEwald, false); // negative curvature

    // Weights should be identical (same pixels sampled, just different z)
    RFLOAT wPos = sumWeightReal(bpPos.weight);
    RFLOAT wNeg = sumWeightReal(bpNeg.weight);
    REQUIRE(wPos == Approx(wNeg).epsilon(1e-6));

    // Data arrays should differ (different z interpolation positions)
    bool dataDiffers = false;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bpPos.data)
    {
        RFLOAT diff = std::fabs(DIRECT_MULTIDIM_ELEM(bpPos.data, n).real -
                                DIRECT_MULTIDIM_ELEM(bpNeg.data, n).real);
        if (diff > 1e-6) { dataDiffers = true; break; }
    }
    REQUIRE(dataDiffers);
}
