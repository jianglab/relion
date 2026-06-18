/*
 * tests/unit/test_matrix2d.cpp
 *
 * Unit tests for Matrix2D<RFLOAT>: basic arithmetic, determinant,
 * inverse correctness (M * inv(M) == I), and transpose.
 */

#include <catch2/catch.hpp>

#include "src/matrix2d.h"
#include "src/macros.h"

#include <cmath>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace
{

/// Build a 2×2 matrix from four explicit values.
Matrix2D<RFLOAT> mat2x2(RFLOAT a00, RFLOAT a01,
                         RFLOAT a10, RFLOAT a11)
{
    Matrix2D<RFLOAT> M(2, 2);
    M(0,0) = a00; M(0,1) = a01;
    M(1,0) = a10; M(1,1) = a11;
    return M;
}

/// Check whether M is the 2×2 identity matrix within a tolerance.
bool isIdentity2x2(const Matrix2D<RFLOAT>& M, RFLOAT tol = 1e-8)
{
    if (M.Xdim() != 2 || M.Ydim() != 2) return false;
    return std::fabs(M(0,0) - 1.0) < tol &&
           std::fabs(M(1,1) - 1.0) < tol &&
           std::fabs(M(0,1))       < tol &&
           std::fabs(M(1,0))       < tol;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Basic properties
// ---------------------------------------------------------------------------

TEST_CASE("Mat2D: identity matrix has det 1", "[matrix2d]")
{
    Matrix2D<RFLOAT> I = mat2x2(1,0, 0,1);
    REQUIRE(I.det() == Approx(1.0).epsilon(1e-10));
}

TEST_CASE("Mat2D: zero matrix has det 0", "[matrix2d]")
{
    Matrix2D<RFLOAT> Z(2, 2);
    Z.initZeros();
    REQUIRE(Z.det() == Approx(0.0).margin(1e-12));
}

TEST_CASE("Mat2D: known 2x2 determinant ad-bc", "[matrix2d]")
{
    // det([[3,7],[1,2]]) = 3*2 - 7*1 = -1
    Matrix2D<RFLOAT> M = mat2x2(3,7, 1,2);
    REQUIRE(M.det() == Approx(-1.0).epsilon(1e-10));
}

TEST_CASE("Mat2D: M * inv(M) == identity for non-singular matrix", "[matrix2d]")
{
    Matrix2D<RFLOAT> M = mat2x2(3,7, 1,2);
    Matrix2D<RFLOAT> Minv = M.inv();
    Matrix2D<RFLOAT> prod(2, 2);
    // Manual 2x2 multiplication
    prod(0,0) = M(0,0)*Minv(0,0) + M(0,1)*Minv(1,0);
    prod(0,1) = M(0,0)*Minv(0,1) + M(0,1)*Minv(1,1);
    prod(1,0) = M(1,0)*Minv(0,0) + M(1,1)*Minv(1,0);
    prod(1,1) = M(1,0)*Minv(0,1) + M(1,1)*Minv(1,1);
    REQUIRE(isIdentity2x2(prod));
}

TEST_CASE("Mat2D: inv(identity) == identity", "[matrix2d]")
{
    Matrix2D<RFLOAT> I = mat2x2(1,0, 0,1);
    Matrix2D<RFLOAT> Iinv = I.inv();
    REQUIRE(isIdentity2x2(Iinv));
}

TEST_CASE("Mat2D: diagonal matrix inv has reciprocal diagonal", "[matrix2d]")
{
    // diag(2, 5) => inv = diag(0.5, 0.2)
    Matrix2D<RFLOAT> D = mat2x2(2,0, 0,5);
    Matrix2D<RFLOAT> Dinv = D.inv();
    REQUIRE(Dinv(0,0) == Approx(0.5).epsilon(1e-8));
    REQUIRE(Dinv(1,1) == Approx(0.2).epsilon(1e-8));
    REQUIRE(std::fabs(Dinv(0,1)) < 1e-10);
    REQUIRE(std::fabs(Dinv(1,0)) < 1e-10);
}

TEST_CASE("Mat2D: transpose swaps off-diagonal elements", "[matrix2d]")
{
    Matrix2D<RFLOAT> M = mat2x2(1, 2, 3, 4);
    Matrix2D<RFLOAT> Mt = M.transpose();
    REQUIRE(Mt(0,0) == Approx(M(0,0)).epsilon(1e-10));
    REQUIRE(Mt(1,1) == Approx(M(1,1)).epsilon(1e-10));
    REQUIRE(Mt(0,1) == Approx(M(1,0)).epsilon(1e-10));
    REQUIRE(Mt(1,0) == Approx(M(0,1)).epsilon(1e-10));
}

TEST_CASE("Mat2D: det(M^-1) == 1/det(M)", "[matrix2d]")
{
    Matrix2D<RFLOAT> M = mat2x2(4, 7, 2, 6);
    RFLOAT detM    = M.det();
    RFLOAT detMinv = M.inv().det();
    REQUIRE(detMinv == Approx(1.0 / detM).epsilon(1e-8));
}

TEST_CASE("Mat2D: det of product equals product of dets", "[matrix2d]")
{
    // det(A*B) = det(A)*det(B)
    Matrix2D<RFLOAT> A = mat2x2(2, 1, 0, 3);
    Matrix2D<RFLOAT> B = mat2x2(1, 4, 2, 0);
    Matrix2D<RFLOAT> AB(2, 2);
    AB(0,0) = A(0,0)*B(0,0) + A(0,1)*B(1,0);
    AB(0,1) = A(0,0)*B(0,1) + A(0,1)*B(1,1);
    AB(1,0) = A(1,0)*B(0,0) + A(1,1)*B(1,0);
    AB(1,1) = A(1,0)*B(0,1) + A(1,1)*B(1,1);
    REQUIRE(AB.det() == Approx(A.det() * B.det()).epsilon(1e-8));
}

// ---------------------------------------------------------------------------
// 3×3 matrix tests
// ---------------------------------------------------------------------------

TEST_CASE("Mat3D: known 3x3 determinant", "[matrix2d]")
{
    // [[1,2,3],[4,5,6],[7,8,10]] det = 1*(50-48) - 2*(40-42) + 3*(32-35)
    //                                = 2 + 4 - 9 = -3
    Matrix2D<RFLOAT> M(3, 3);
    M(0,0)=1; M(0,1)=2; M(0,2)=3;
    M(1,0)=4; M(1,1)=5; M(1,2)=6;
    M(2,0)=7; M(2,1)=8; M(2,2)=10;
    REQUIRE(M.det() == Approx(-3.0).epsilon(1e-8));
}

TEST_CASE("Mat3D: M * inv(M) == identity for 3x3 non-singular", "[matrix2d]")
{
    Matrix2D<RFLOAT> M(3, 3);
    M(0,0)=1; M(0,1)=2; M(0,2)=3;
    M(1,0)=4; M(1,1)=5; M(1,2)=6;
    M(2,0)=7; M(2,1)=8; M(2,2)=10;
    Matrix2D<RFLOAT> Minv = M.inv();
    // Compute M * Minv and check it is identity
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            RFLOAT sum = 0.0;
            for (int k = 0; k < 3; k++)
                sum += M(i,k) * Minv(k,j);
            RFLOAT expected = (i == j) ? 1.0 : 0.0;
            REQUIRE(sum == Approx(expected).margin(1e-8));
        }
    }
}
