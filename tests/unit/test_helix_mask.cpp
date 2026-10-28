/*
 * tests/unit/test_helix_mask.cpp
 *
 * softMaskOutsideMapForHelix() rotates each pixel with plain arithmetic instead
 * of a Matrix2D * Matrix1D product (which allocated a vector per pixel); the
 * masked images must stay bitwise identical to the original implementation,
 * kept here as the reference.
 */

#include <catch2/catch.hpp>

#include "src/mask.h"
#include "src/matrix2d.h"
#include "src/euler.h"

#include <random>

namespace
{
// The implementation before the change, verbatim except for its name
void referenceMask(MultidimArray<RFLOAT> &vol, RFLOAT psi_deg, RFLOAT tilt_deg, RFLOAT mask_sphere_radius_pix,
                   RFLOAT mask_cyl_radius_pix, RFLOAT cosine_width, MultidimArray<RFLOAT> *Mnoise)
{
	Matrix1D<RFLOAT> coords;
	Matrix2D<RFLOAT> A;
	RFLOAT sum_bg, sum, R1, R2, D1, D2, r, d, noise_w, noise_w1, noise_w2, noise_val;
	int dim = vol.getDim();
	vol.setXmippOrigin();
	if (dim == 2) tilt_deg = 0.;
	R1 = mask_sphere_radius_pix; R2 = R1 + cosine_width;
	D1 = mask_cyl_radius_pix; D2 = D1 + cosine_width;
	coords.resize(3); coords.initZeros();
	A.resize(3, 3);
	Euler_angles2matrix(0., tilt_deg, psi_deg, A, false);
	A = A.transpose();
	sum_bg = sum = 0.;
	if (Mnoise == NULL)
	{
		FOR_ALL_ELEMENTS_IN_ARRAY3D(vol)
		{
			ZZ(coords) = (dim == 3) ? (RFLOAT)k : 0.;
			YY(coords) = (RFLOAT)i;
			XX(coords) = (RFLOAT)j;
			coords = A * coords;
			d = (dim == 3) ? sqrt(YY(coords) * YY(coords) + XX(coords) * XX(coords)) : ABS(YY(coords));
			if (d > D2) { sum_bg += A3D_ELEM(vol, k, i, j); sum += 1.; }
			else if (d > D1)
			{
				noise_w = 0.5 + 0.5 * cos(PI * (D2 - d) / cosine_width);
				sum_bg += noise_w * A3D_ELEM(vol, k, i, j);
				sum += noise_w;
			}
		}
		sum_bg /= sum;
	}
	noise_val = sum_bg;
	FOR_ALL_ELEMENTS_IN_ARRAY3D(vol)
	{
		ZZ(coords) = (dim == 3) ? (RFLOAT)k : 0.;
		YY(coords) = (RFLOAT)i;
		XX(coords) = (RFLOAT)j;
		coords = A * coords;
		d = (dim == 3) ? sqrt(YY(coords) * YY(coords) + XX(coords) * XX(coords)) : ABS(YY(coords));
		r = (RFLOAT)(i * i + j * j);
		if (dim == 3) r += (RFLOAT)(k * k);
		r = sqrt(r);
		if ((r < R1) && (d < D1)) continue;
		if (Mnoise != NULL) noise_val = A3D_ELEM(*Mnoise, k, i, j);
		if ((r > R2) || (d > D2)) A3D_ELEM(vol, k, i, j) = noise_val;
		else
		{
			noise_w1 = noise_w2 = 0.;
			if (r > R1) noise_w1 = 0.5 + 0.5 * cos(PI * (R2 - r) / cosine_width);
			if (d > D1) noise_w2 = 0.5 + 0.5 * cos(PI * (D2 - d) / cosine_width);
			noise_w = (noise_w1 > noise_w2) ? noise_w1 : noise_w2;
			A3D_ELEM(vol, k, i, j) = (1. - noise_w) * A3D_ELEM(vol, k, i, j) + noise_w * noise_val;
		}
	}
}

MultidimArray<RFLOAT> randomArray(int nz, int ny, int nx, unsigned seed)
{
	std::mt19937 rng(seed);
	std::normal_distribution<double> g(0., 1.);
	MultidimArray<RFLOAT> a;
	if (nz > 1) a.resize(nz, ny, nx); else a.resize(ny, nx);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a) DIRECT_MULTIDIM_ELEM(a, n) = g(rng);
	return a;
}

bool identical(const MultidimArray<RFLOAT> &a, const MultidimArray<RFLOAT> &b)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a)
		if (DIRECT_MULTIDIM_ELEM(a, n) != DIRECT_MULTIDIM_ELEM(b, n)) return false;
	return true;
}
}

TEST_CASE("the helical mask is bitwise unchanged in 2D and 3D, with and without noise", "[helix_mask]")
{
	for (int dim : {2, 3})
		for (double psi : {0., 17.3, 90., 133.7, 271.})
			for (bool noise : {false, true})
			{
				const int n = (dim == 3) ? 40 : 64;
				MultidimArray<RFLOAT> a = randomArray(dim == 3 ? n : 1, n, n, 7), b = a;
				MultidimArray<RFLOAT> mn = randomArray(dim == 3 ? n : 1, n, n, 11);
				mn.setXmippOrigin();
				softMaskOutsideMapForHelix(a, psi, 63.1, n / 2 - 4, 9., 3., noise ? &mn : NULL);
				referenceMask(b, psi, 63.1, n / 2 - 4, 9., 3., noise ? &mn : NULL);
				INFO("dim " << dim << " psi " << psi << " noise " << noise);
				CHECK(identical(a, b));
			}
}

namespace
{
// A tube of ones (|y| <= tube_half) along x on a background of zeros, n x n (2D) or n x n x n
// with the tube along z (3D)
MultidimArray<RFLOAT> tube(int dim, int n, int tube_half)
{
	MultidimArray<RFLOAT> a;
	if (dim == 2) a.resize(n, n); else a.resize(n, n, n);
	a.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY3D(a)
	{
		const long int across = (dim == 2) ? i : std::max(std::abs(i), std::abs(j));
		A3D_ELEM(a, k, i, j) = (std::abs(across) <= tube_half) ? 1. : 0.;
	}
	return a;
}
}

TEST_CASE("the cylindrical helical mask is as wide as the tube and as long as the mask diameter", "[helix_mask]")
{
	const int n = 64;
	const RFLOAT R = 16., D = 9., w = 3.;   // mask radius, tube radius, soft edge
	// the last two arguments: follow the box shape (no: square boxes), cylinder

	SECTION("2D, helical axis along x (psi 0)")
	{
		MultidimArray<RFLOAT> sphere = tube(2, n, 6), cyl = sphere;
		softMaskOutsideMapForHelix(sphere, 0., 0., R, D, w, NULL, false, false);
		softMaskOutsideMapForHelix(cyl, 0., 0., R, D, w, NULL, false, true);
		// inside the tube, 15 along the axis and 6 across: outside the circle (r = 16.2 > 16)
		CHECK(A2D_ELEM(cyl, 6, 15) == 1.);                 // kept by the cylinder
		CHECK(A2D_ELEM(sphere, 6, 15) < 1.);               // softened by the circle
		CHECK(A2D_ELEM(cyl, 0, 15) == 1.);
		// along the axis, beyond the soft edge: background
		CHECK(A2D_ELEM(cyl, 0, 20) == Approx(0.).margin(1e-12));
		CHECK(A2D_ELEM(cyl, 0, -20) == Approx(0.).margin(1e-12));
		// in the soft edge along the axis: the raised cosine, as across the tube
		CHECK(A2D_ELEM(cyl, 0, 17) == Approx(1. - (0.5 + 0.5 * cos(PI * (R + w - 17.) / w))));
	}

	SECTION("2D, the mask turns with psi")
	{
		MultidimArray<RFLOAT> along_x = tube(2, n, 6), along_y;
		along_y.resize(n, n);
		along_y.setXmippOrigin();
		FOR_ALL_ELEMENTS_IN_ARRAY2D(along_y)
			A2D_ELEM(along_y, i, j) = A2D_ELEM(along_x, j, i);
		softMaskOutsideMapForHelix(along_x, 0., 0., R, D, w, NULL, false, true);
		softMaskOutsideMapForHelix(along_y, 90., 0., R, D, w, NULL, false, true);
		FOR_ALL_ELEMENTS_IN_ARRAY2D(along_y)
			if (i > STARTINGY(along_y) && j > STARTINGX(along_y))
				REQUIRE(A2D_ELEM(along_y, i, j) == Approx(A2D_ELEM(along_x, j, i)).margin(1e-9));
	}

	SECTION("3D, helical axis along z (tilt 0, psi 0)")
	{
		MultidimArray<RFLOAT> sphere = tube(3, 40, 6), cyl = sphere;
		softMaskOutsideMapForHelix(sphere, 0., 0., R, D, w, NULL, false, false);
		softMaskOutsideMapForHelix(cyl, 0., 0., R, D, w, NULL, false, true);
		CHECK(A3D_ELEM(cyl, 15, 6, 0) == 1.);
		CHECK(A3D_ELEM(sphere, 15, 6, 0) < 1.);
		CHECK(A3D_ELEM(cyl, 19, 0, 0) == Approx(0.).margin(1e-12));
	}
}
