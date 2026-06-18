/*
 * tests/unit/test_zernike.cpp
 *
 * Unit tests for Zernike polynomial evaluation (src/jaz/math/Zernike.h).
 *
 * Ground-truth values come from the standard analytical Zernike definitions:
 *
 *   Z(m, n, rho, phi)  - polar-coordinate form
 *   Z_cart(m, n, x, y) - Cartesian form (same value, different coordinates)
 *   R(m, n, rho)       - radial polynomial
 *
 * Key identities used here:
 *   R(0, 0, rho) = 1
 *   R(1, 1, rho) = rho
 *   R(0, 2, rho) = 2*rho^2 - 1
 *   R(2, 2, rho) = rho^2
 *
 *   Z(0,  0, rho, phi) = R(0,0,rho)*cos(0) = 1           (piston)
 *   Z(1,  1, rho, phi) = R(1,1,rho)*cos(phi) = rho*cos(phi) = x  (x-tilt)
 *   Z(-1, 1, rho, phi) = R(1,1,rho)*sin(phi) = rho*sin(phi) = y  (y-tilt)
 *   Z(2,  2, rho, phi) = R(2,2,rho)*cos(2*phi) = rho^2*(cos^2-sin^2) = x^2-y^2
 *   Z(0,  2, rho, phi) = R(0,2,rho)*cos(0) = 2*rho^2 - 1 (defocus-like)
 */

#include <catch2/catch.hpp>

#include "src/jaz/math/Zernike.h"
#include "src/macros.h"

#include <cmath>

// ---------------------------------------------------------------------------
// Radial polynomial R(m, n, rho)
// ---------------------------------------------------------------------------

TEST_CASE("Zernike R(0,0,rho) = 1 for any rho", "[zernike]")
{
    for (double rho : {0.0, 0.3, 0.7, 1.0})
        REQUIRE(Zernike::R(0, 0, rho) == Approx(1.0).epsilon(1e-10));
}

TEST_CASE("Zernike R(1,1,rho) = rho", "[zernike]")
{
    for (double rho : {0.0, 0.25, 0.5, 0.75, 1.0})
        REQUIRE(Zernike::R(1, 1, rho) == Approx(rho).margin(1e-10));
}

TEST_CASE("Zernike R(0,2,rho) = 2*rho^2 - 1", "[zernike]")
{
    for (double rho : {0.0, 0.5, 1.0 / std::sqrt(2.0), 1.0})
    {
        double expected = 2.0 * rho * rho - 1.0;
        REQUIRE(Zernike::R(0, 2, rho) == Approx(expected).margin(1e-12));
    }
}

TEST_CASE("Zernike R(2,2,rho) = rho^2", "[zernike]")
{
    for (double rho : {0.0, 0.5, 0.8, 1.0})
        REQUIRE(Zernike::R(2, 2, rho) == Approx(rho * rho).margin(1e-10));
}

TEST_CASE("Zernike R(m,n,0) = 0 when n > 0", "[zernike]")
{
    // R(m, n, 0) = coefficient of rho^n term only when n > 0, at rho=0 all powers vanish
    REQUIRE(Zernike::R(1, 1, 0.0) == Approx(0.0).margin(1e-12));
    REQUIRE(Zernike::R(2, 2, 0.0) == Approx(0.0).margin(1e-12));
    REQUIRE(Zernike::R(1, 3, 0.0) == Approx(0.0).margin(1e-12));
}

TEST_CASE("Zernike R(m,n,rho) = 0 when (n-m) is odd", "[zernike]")
{
    // Standard property: R(m,n,rho) = 0 whenever (n-m) is odd
    REQUIRE(Zernike::R(0, 1, 0.5) == Approx(0.0).margin(1e-12));
    REQUIRE(Zernike::R(1, 2, 0.5) == Approx(0.0).margin(1e-12));
    REQUIRE(Zernike::R(2, 3, 0.5) == Approx(0.0).margin(1e-12));
}

// ---------------------------------------------------------------------------
// Full polynomial Z(m, n, rho, phi)
// ---------------------------------------------------------------------------

TEST_CASE("Zernike Z(0,0) = 1 everywhere (piston)", "[zernike]")
{
    for (double rho : {0.0, 0.5, 1.0})
        for (double phi : {0.0, 1.0, 2.5})
            REQUIRE(Zernike::Z(0, 0, rho, phi) == Approx(1.0).epsilon(1e-10));
}

TEST_CASE("Zernike Z(1,1) = rho*cos(phi) = x (x-tilt)", "[zernike]")
{
    // Exact at unit circle (rho=1)
    double phi = 1.234;
    REQUIRE(Zernike::Z(1, 1, 1.0, phi) == Approx(std::cos(phi)).epsilon(1e-10));
    // And at arbitrary rho
    double rho = 0.6;
    REQUIRE(Zernike::Z(1, 1, rho, phi) == Approx(rho * std::cos(phi)).epsilon(1e-10));
}

TEST_CASE("Zernike Z(-1,1) = rho*sin(phi) = y (y-tilt)", "[zernike]")
{
    double phi = 0.789;
    double rho = 0.7;
    REQUIRE(Zernike::Z(-1, 1, rho, phi) == Approx(rho * std::sin(phi)).epsilon(1e-10));
}

TEST_CASE("Zernike Z(2,2) = rho^2*cos(2*phi) = x^2 - y^2 (astigmatism)", "[zernike]")
{
    double rho = 0.8;
    double phi = 0.5;
    double expected = rho * rho * std::cos(2.0 * phi);
    REQUIRE(Zernike::Z(2, 2, rho, phi) == Approx(expected).epsilon(1e-10));
}

TEST_CASE("Zernike Z(0,2) = 2*rho^2 - 1 (defocus)", "[zernike]")
{
    for (double rho : {0.0, 0.5, 1.0})
    {
        double expected = 2.0 * rho * rho - 1.0;
        REQUIRE(Zernike::Z(0, 2, rho, 0.0) == Approx(expected).epsilon(1e-10));
    }
}

// ---------------------------------------------------------------------------
// Cartesian form Z_cart vs polar Z should be consistent
// ---------------------------------------------------------------------------

TEST_CASE("Zernike Z_cart(m,n,x,y) == Z(m,n,rho,phi) for x-tilt", "[zernike]")
{
    const double x = 0.4, y = 0.3;
    const double rho = std::hypot(x, y);
    const double phi = std::atan2(y, x);
    REQUIRE(Zernike::Z_cart(1, 1, x, y) == Approx(Zernike::Z(1, 1, rho, phi)).epsilon(1e-10));
}

TEST_CASE("Zernike Z_cart(m,n,x,y) == Z(m,n,rho,phi) for astigmatism", "[zernike]")
{
    const double x = 0.6, y = -0.2;
    const double rho = std::hypot(x, y);
    const double phi = std::atan2(y, x);
    REQUIRE(Zernike::Z_cart(2, 2, x, y) == Approx(Zernike::Z(2, 2, rho, phi)).epsilon(1e-10));
}

TEST_CASE("Zernike Z_cart at origin returns correct piston value", "[zernike]")
{
    // Z(0,0) = 1 everywhere including origin
    REQUIRE(Zernike::Z_cart(0, 0, 0.0, 0.0) == Approx(1.0).epsilon(1e-10));
}

// ---------------------------------------------------------------------------
// Index <-> (m, n) mappings
// ---------------------------------------------------------------------------

TEST_CASE("Zernike evenIndexToMN: index 0 maps to (m=0, n=0)", "[zernike]")
{
    int m, n;
    Zernike::evenIndexToMN(0, m, n);
    REQUIRE(m == 0);
    REQUIRE(n == 0);
}

TEST_CASE("Zernike evenIndexToMN: first few even coefficients", "[zernike]")
{
    // Even basis: (m,n) pairs with n even, grouped by shell
    // Index 0: n=0, m=0  (piston)
    // Index 1: n=2, m=-2  Index 2: n=2, m=0  Index 3: n=2, m=2
    // Index 4: n=4, m=-4  ... etc.
    struct Expected { int i, m, n; };
    Expected cases[] = { {0, 0, 0}, {2, 0, 2} };
    for (auto& c : cases)
    {
        int m, n;
        Zernike::evenIndexToMN(c.i, m, n);
        REQUIRE(m == c.m);
        REQUIRE(n == c.n);
    }
}

TEST_CASE("Zernike numberOfEvenCoeffs is correct for small n_max", "[zernike]")
{
    // n_max=0: just the piston => 1 coeff
    REQUIRE(Zernike::numberOfEvenCoeffs(0) == 1);
    // n_max=2: piston + 3 quadratic => 4 coeffs
    REQUIRE(Zernike::numberOfEvenCoeffs(2) == 4);
    // n_max=4: 4 + 5 => 9 coeffs
    REQUIRE(Zernike::numberOfEvenCoeffs(4) == 9);
}

TEST_CASE("Zernike numberOfOddCoeffs is correct for small n_max", "[zernike]")
{
    // n_max=1: 2 coeffs (x-tilt, y-tilt)
    REQUIRE(Zernike::numberOfOddCoeffs(1) == 2);
    // n_max=3: 2 + 4 => 6 coeffs
    REQUIRE(Zernike::numberOfOddCoeffs(3) == 6);
}

TEST_CASE("Zernike oddIndexToMN: x-tilt and y-tilt indices", "[zernike]")
{
    // n=1 (odd), m = -1 (y-tilt), m=+1 (x-tilt)
    int m0, n0, m1, n1;
    Zernike::oddIndexToMN(0, m0, n0);
    Zernike::oddIndexToMN(1, m1, n1);
    REQUIRE(n0 == 1);
    REQUIRE(n1 == 1);
    // m values should be -1 and +1 in some order
    REQUIRE(m0 * m1 == -1);   // one is -1, the other is +1
}
