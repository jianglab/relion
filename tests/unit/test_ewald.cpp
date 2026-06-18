/*
 * tests/unit/test_ewald.cpp
 *
 * Tests for the Ewald-sphere geometry used in backprojection.
 *
 * The key formula (from backprojector.cpp, lines ~213):
 *
 *   z_on_ewald = inv_diam_ewald * (xu^2 + yu^2)
 *              = (xu^2 + yu^2) / (2 * r_ewald_sphere)
 *
 * This is the paraxial approximation of the 3D position on the Ewald sphere
 * at radius r_ewald for a 2D detector pixel (xu, yu):
 *
 *   Exact:    z = r - sqrt(r^2 - xu^2 - yu^2)  ≈  (xu^2+yu^2)/(2r)  for xu,yu << r
 *
 * We test:
 *  1. The paraxial z-offset formula against the exact expression
 *  2. Curvature sign flip (positive vs negative curvature)
 *  3. Zero curvature when r_ewald <= 0
 *  4. Monotonically increasing z with increasing radius on the detector
 *  5. The wavelength->Ewald radius relationship: r_ewald = 1/lambda
 *
 * We also test a helper that calculates the Ewald radius from physical units:
 *   r_ewald [pixels] = angpix / lambda   (since lambda is in Angstroms)
 */

#include <catch2/catch.hpp>

#include "src/macros.h"
#include "src/ctf.h"

#include <cmath>

// ---------------------------------------------------------------------------
// Pure geometry helpers (implement the formulas from backprojector.cpp inline)
// ---------------------------------------------------------------------------

namespace
{

/// Paraxial z-offset of point (xu, yu) on an Ewald sphere of radius r.
/// When r <= 0, curvature is disabled and z = 0.
/// is_positive_curvature = false flips the sign.
double ewaldZOffset(double xu, double yu,
                    double r_ewald_sphere,
                    bool   is_positive_curvature = true)
{
    if (r_ewald_sphere <= 0.0) return 0.0;
    double inv_diam = 1.0 / (2.0 * r_ewald_sphere);
    if (!is_positive_curvature) inv_diam = -inv_diam;
    return inv_diam * (xu * xu + yu * yu);
}

/// Exact z on Ewald sphere (for comparison against paraxial approximation).
double ewaldZExact(double xu, double yu, double r_ewald_sphere)
{
    // r - sqrt(r^2 - xu^2 - yu^2), valid only when xu^2+yu^2 < r^2
    double r2 = r_ewald_sphere * r_ewald_sphere;
    double xy2 = xu * xu + yu * yu;
    if (xy2 >= r2) return r_ewald_sphere;  // at the equator
    return r_ewald_sphere - std::sqrt(r2 - xy2);
}

/// Ewald sphere radius in pixels given angpix and lambda_angstroms.
/// r_ewald [pixels] = angpix / lambda_angstroms
double ewaldRadiusPix(double angpix, double lambda_angstroms)
{
    return angpix / lambda_angstroms;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Z-offset formula
// ---------------------------------------------------------------------------

TEST_CASE("Ewald: z-offset is zero at origin (xu=0, yu=0)", "[ewald]")
{
    double r = 100.0;
    REQUIRE(ewaldZOffset(0.0, 0.0, r) == Approx(0.0).margin(1e-12));
}

TEST_CASE("Ewald: paraxial z-offset is (xu^2+yu^2)/(2r)", "[ewald]")
{
    const double r = 500.0;
    for (double xu : {1.0, 5.0, 10.0, 20.0})
        for (double yu : {0.0, 3.0, 7.0})
        {
            double expected = (xu*xu + yu*yu) / (2.0 * r);
            REQUIRE(ewaldZOffset(xu, yu, r) == Approx(expected).epsilon(1e-10));
        }
}

TEST_CASE("Ewald: negative curvature flips z-offset sign", "[ewald]")
{
    const double r = 200.0;
    const double xu = 8.0, yu = 6.0;
    double zPos = ewaldZOffset(xu, yu, r, true);
    double zNeg = ewaldZOffset(xu, yu, r, false);
    REQUIRE(zPos > 0.0);
    REQUIRE(zNeg < 0.0);
    REQUIRE(zPos == Approx(-zNeg).epsilon(1e-10));
}

TEST_CASE("Ewald: zero curvature when r_ewald <= 0", "[ewald]")
{
    REQUIRE(ewaldZOffset(10.0, 5.0, 0.0)  == Approx(0.0).margin(1e-12));
    REQUIRE(ewaldZOffset(10.0, 5.0, -1.0) == Approx(0.0).margin(1e-12));
}

TEST_CASE("Ewald: z-offset increases monotonically with detector radius", "[ewald]")
{
    const double r = 300.0;
    const double radii[] = {2.0, 5.0, 10.0, 20.0, 40.0};
    double prevZ = -1e10;
    for (double s : radii)
    {
        double z = ewaldZOffset(s, 0.0, r);
        REQUIRE(z > prevZ);
        prevZ = z;
    }
}

// ---------------------------------------------------------------------------
// Paraxial vs exact comparison
// ---------------------------------------------------------------------------

TEST_CASE("Ewald: paraxial agrees with exact to < 1% for s << r", "[ewald]")
{
    // For xu/r << 1, the paraxial approximation should be accurate to ~(xu/r)^2
    const double r = 1000.0;
    for (double s : {5.0, 10.0, 20.0})
    {
        double zParaxial = ewaldZOffset(s, 0.0, r);
        double zExact    = ewaldZExact(s, 0.0, r);
        // Relative error ~ (s/r)^2 / 4  which for s=20, r=1000 is 1e-4
        double relErr = std::fabs(zParaxial - zExact) / std::max(zExact, 1e-15);
        REQUIRE(relErr < 0.01);
    }
}

// ---------------------------------------------------------------------------
// Ewald radius from physical units
// ---------------------------------------------------------------------------

TEST_CASE("Ewald: radius in pixels scales with angpix/lambda", "[ewald]")
{
    // For 300 kV electrons, lambda ≈ 1.969e-3 Angstroms
    // CTF::setValues triggers initialise() which computes lambda
    const double angpix = 1.0;   // 1 Angstrom/pixel
    // From CTF initialise(): lambda = 12.2643247 / sqrt(V*(1+V*0.978466e-6))
    // V = 300e3 eV
    double V = 300e3;
    double lambda = 12.2643247 / std::sqrt(V * (1.0 + V * 0.978466e-6));

    double r_pix = ewaldRadiusPix(angpix, lambda);

    // For 300kV, lambda = 12.2643247/sqrt(V*(1+V*0.978466e-6)) ≈ 0.01969 Angstrom
    // => r_pix = angpix/lambda ≈ 50.8 pixels (at angpix=1 A/pix)
    REQUIRE(r_pix > 40.0);
    REQUIRE(r_pix < 100.0);
}

TEST_CASE("Ewald: higher voltage gives smaller lambda, hence larger Ewald radius", "[ewald]")
{
    const double angpix = 1.0;
    // 120 kV
    double V1 = 120e3;
    double lambda1 = 12.2643247 / std::sqrt(V1 * (1.0 + V1 * 0.978466e-6));
    // 300 kV
    double V2 = 300e3;
    double lambda2 = 12.2643247 / std::sqrt(V2 * (1.0 + V2 * 0.978466e-6));

    REQUIRE(lambda2 < lambda1);   // higher voltage => shorter wavelength
    REQUIRE(ewaldRadiusPix(angpix, lambda2) > ewaldRadiusPix(angpix, lambda1));
}

// ---------------------------------------------------------------------------
// Ewald correction breaks centrosymmetry of the backprojection
// ---------------------------------------------------------------------------

TEST_CASE("Ewald: positive and negative curvature give opposite z offsets", "[ewald]")
{
    // This models the two half-datasets split by sign of curvature.
    // Their reconstructions should differ.
    const double r = 300.0;
    const double xu = 15.0, yu = 10.0;
    double zPos = ewaldZOffset(xu, yu, r, true);
    double zNeg = ewaldZOffset(xu, yu, r, false);

    // Opposite signs — these map to different 3D positions
    REQUIRE(zPos + zNeg == Approx(0.0).margin(1e-10));
    REQUIRE(zPos != Approx(0.0).margin(1e-8));
}
