/*
 * tests/unit/test_anisomag_ctf.cpp
 *
 * Tests for anisotropic magnification (aniso-mag) effects on the CTF.
 *
 * Strategy: The CTF::getCTF() function applies the mag matrix M to the
 * frequency coordinates (X, Y) before computing the phase.  Therefore:
 *
 *   CTF_with_mag(x, y; M)  ==  CTF_no_mag(M*[x,y])
 *
 * for any invertible magnification matrix M.  We verify this identity for
 * both isotropic and anisotropic (non-diagonal) M by constructing a minimal
 * ObservationModel from scratch via a synthetic optics MetaDataTable.
 *
 * We also test that:
 *  - identity M gives the same result as no obsModel at all
 *  - x-stretch shrinks effective x-frequency range (aniso-CTF varies along x
 *    faster than along y)
 */

#include <catch2/catch.hpp>

#include "src/ctf.h"
#include "src/jaz/single_particle/obs_model.h"
#include "src/metadata_table.h"
#include "src/metadata_label.h"
#include "src/matrix2d.h"
#include "src/macros.h"

#include <cmath>

// ---------------------------------------------------------------------------
// Helper: build a minimal one-group ObservationModel with a given 2x2 mag M
// ---------------------------------------------------------------------------

namespace
{

/// Build a minimal one-optics-group ObservationModel whose mag matrix is M.
ObservationModel makeObsModelWithMag(const Matrix2D<RFLOAT>& M)
{
    MetaDataTable opticsMdt;
    opticsMdt.setName("optics");
    opticsMdt.addObject();
    opticsMdt.setValue(EMDL_IMAGE_PIXEL_SIZE,   (RFLOAT)1.0, 0);
    opticsMdt.setValue(EMDL_CTF_VOLTAGE,        (RFLOAT)300.0, 0);
    opticsMdt.setValue(EMDL_CTF_CS,             (RFLOAT)2.7,  0);
    opticsMdt.setValue(EMDL_CTF_Q0,             (RFLOAT)0.1,  0);
    // Mag matrix columns
    opticsMdt.setValue(EMDL_IMAGE_MAG_MATRIX_00, M(0,0), 0);
    opticsMdt.setValue(EMDL_IMAGE_MAG_MATRIX_01, M(0,1), 0);
    opticsMdt.setValue(EMDL_IMAGE_MAG_MATRIX_10, M(1,0), 0);
    opticsMdt.setValue(EMDL_IMAGE_MAG_MATRIX_11, M(1,1), 0);
    return ObservationModel(opticsMdt, /*do_die_upon_error=*/false);
}

/// Build a CTF whose mag matrix comes from ObservationModel obs.
/// Uses setValuesByGroup so obsModel and opticsGroup are set via the public API.
CTF makeCTFWithObs(ObservationModel& obs, RFLOAT defU, RFLOAT defV = -1.0)
{
    if (defV < 0) defV = defU;
    CTF ctf;
    ctf.setValuesByGroup(&obs, 0, defU, defV, 0.0, 0.0, 1.0, 0.0, -1.0);
    return ctf;
}

/// Build a standard CTF for a given (defU, defV) without any obsModel.
CTF makePlainCTF(RFLOAT defU, RFLOAT defV = -1.0)
{
    if (defV < 0) defV = defU;
    CTF ctf;
    ctf.setValues(defU, defV, 0.0, 300.0, 2.7, 0.1, 0.0, 1.0, 0.0);
    return ctf;
}

/// 2x2 identity matrix
Matrix2D<RFLOAT> identityM()
{
    Matrix2D<RFLOAT> M(2,2);
    M.initIdentity();
    return M;
}

/// Diagonal mag matrix diag(sx, sy)
Matrix2D<RFLOAT> diagM(RFLOAT sx, RFLOAT sy)
{
    Matrix2D<RFLOAT> M(2,2);
    M.initZeros();
    M(0,0) = sx; M(1,1) = sy;
    return M;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Core identity: CTF_with_mag(x,y;M) == CTF_no_mag(M[x,y])
// ---------------------------------------------------------------------------

TEST_CASE("AnisoMag: identity mag gives same CTF as no mag", "[anisomag]")
{
    ObservationModel obs = makeObsModelWithMag(identityM());
    CTF ctfMag   = makeCTFWithObs(obs, 10000.0);
    CTF ctfPlain = makePlainCTF(10000.0);  // no obsModel

    for (double x : {0.05, 0.1, 0.15})
        for (double y : {0.0, 0.05, 0.08})
            REQUIRE(ctfMag.getCTF((RFLOAT)x, (RFLOAT)y) ==
                    Approx(ctfPlain.getCTF((RFLOAT)x, (RFLOAT)y)).epsilon(1e-6));
}

TEST_CASE("AnisoMag: x-stretch-2 CTF(x,y;M) == CTF(2x,y)", "[anisomag]")
{
    // M = diag(2,1): x-frequency is stretched by 2x
    // CTF_with_mag(x, y) should equal CTF_no_mag(2x, y)
    const RFLOAT sx = 2.0;
    ObservationModel obs = makeObsModelWithMag(diagM(sx, 1.0));
    CTF ctfMag   = makeCTFWithObs(obs, 10000.0);
    CTF ctfPlain = makePlainCTF(10000.0);

    for (double x : {0.02, 0.05, 0.08})
        for (double y : {0.0, 0.03, 0.07})
        {
            RFLOAT valMag   = ctfMag.getCTF((RFLOAT)x, (RFLOAT)y);
            RFLOAT valPlain = ctfPlain.getCTF((RFLOAT)(sx * x), (RFLOAT)y);
            REQUIRE(valMag == Approx(valPlain).epsilon(1e-6));
        }
}

TEST_CASE("AnisoMag: y-stretch-3 CTF(x,y;M) == CTF(x,3y)", "[anisomag]")
{
    const RFLOAT sy = 3.0;
    ObservationModel obs = makeObsModelWithMag(diagM(1.0, sy));
    CTF ctfMag   = makeCTFWithObs(obs, 12000.0);
    CTF ctfPlain = makePlainCTF(12000.0);

    for (double x : {0.0, 0.02})
        for (double y : {0.01, 0.04, 0.06})
        {
            RFLOAT valMag   = ctfMag.getCTF((RFLOAT)x, (RFLOAT)y);
            RFLOAT valPlain = ctfPlain.getCTF((RFLOAT)x, (RFLOAT)(sy * y));
            REQUIRE(valMag == Approx(valPlain).epsilon(1e-6));
        }
}

TEST_CASE("AnisoMag: off-diagonal shear M CTF(x,y;M) == CTF(x+a*y, y)", "[anisomag]")
{
    // M = [[1, a], [0, 1]]: shear in x
    const RFLOAT a = 0.5;
    Matrix2D<RFLOAT> M(2, 2);
    M(0,0) = 1.0; M(0,1) = a;
    M(1,0) = 0.0; M(1,1) = 1.0;

    ObservationModel obs = makeObsModelWithMag(M);
    CTF ctfMag   = makeCTFWithObs(obs, 9000.0);
    CTF ctfPlain = makePlainCTF(9000.0);

    for (double x : {0.02, 0.06})
        for (double y : {0.01, 0.04})
        {
            RFLOAT valMag   = ctfMag.getCTF((RFLOAT)x, (RFLOAT)y);
            RFLOAT valPlain = ctfPlain.getCTF((RFLOAT)(x + a * y), (RFLOAT)y);
            REQUIRE(valMag == Approx(valPlain).epsilon(1e-6));
        }
}

// ---------------------------------------------------------------------------
// Physics: aniso mag breaks isotropic symmetry of the CTF
// ---------------------------------------------------------------------------

TEST_CASE("AnisoMag: 2x x-stretch breaks CTF(x,y)==CTF(y,x) symmetry", "[anisomag]")
{
    // Without mag, isotropic CTF satisfies CTF(x,y) == CTF(y,x)
    // With 2x x-stretch, this symmetry is broken.
    ObservationModel obs = makeObsModelWithMag(diagM(2.0, 1.0));
    CTF ctf = makeCTFWithObs(obs, 10000.0);

    const RFLOAT a = 0.05, b = 0.08;
    RFLOAT v1 = ctf.getCTF(a, b);
    RFLOAT v2 = ctf.getCTF(b, a);
    REQUIRE(std::fabs(v1 - v2) > 0.01);
}

TEST_CASE("AnisoMag: identity M preserves isotropic CTF symmetry", "[anisomag]")
{
    ObservationModel obs = makeObsModelWithMag(identityM());
    CTF ctf = makeCTFWithObs(obs, 10000.0);

    // Centrosymmetry: CTF(x,y) == CTF(-x,-y)  (the CTF is even)
    const RFLOAT a = 0.05, b = 0.08;
    REQUIRE(ctf.getCTF(a, b) == Approx(ctf.getCTF(-a, -b)).epsilon(1e-6));
}
