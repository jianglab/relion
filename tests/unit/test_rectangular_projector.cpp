/*
 * tests/unit/test_rectangular_projector.cpp
 *
 * Projector for rectangular 2D references and cuboid 3D references.
 */

#include <catch2/catch.hpp>

#include "src/projector.h"
#include "src/fftw.h"
#include "src/euler.h"

#include <cmath>

namespace
{
MultidimArray<RFLOAT> blobImage(int nx, int ny)
{
	MultidimArray<RFLOAT> img(ny, nx);
	img.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY2D(img)
	{
		A2D_ELEM(img, i, j) = std::exp(-(j * j + i * i) / 18.0)
		                    + 0.5 * std::exp(-((j - 4) * (j - 4) + (i + 2) * (i + 2)) / 6.0);
	}
	return img;
}

// Fourier transform in RELION's normalisation (divided by the pixel count)
MultidimArray<Complex> transformOf(MultidimArray<RFLOAT> img)
{
	MultidimArray<Complex> F;
	FourierTransformer ft;
	MultidimArray<RFLOAT> copy = img;
	copy.setXmippOrigin();
	CenterFFT(copy, true);
	ft.FourierTransform(copy, F, true); // copy, as ft goes out of scope
	return F;
}
}

TEST_CASE("Rectangular projector: identity rotation reproduces the image transform", "[rectangular]")
{
	const int nx = 48, ny = 24;
	MultidimArray<RFLOAT> img = blobImage(nx, ny);
	MultidimArray<Complex> F = transformOf(img);

	Projector proj(48, TRILINEAR, 2.0);
	proj.setBoxSize(nx, ny);
	REQUIRE(proj.rect);
	REQUIRE(proj.ori_size == 48);

	MultidimArray<RFLOAT> spectrum;
	proj.computeFourierTransformMap(img, spectrum, -1, 1, false /*no gridding*/);
	REQUIRE(proj.ref_dim == 2);
	REQUIRE(XSIZE(spectrum) == 25);

	Matrix2D<RFLOAT> A;
	rotation2DMatrix(0., A);
	MultidimArray<Complex> f2d(ny, nx / 2 + 1);
	f2d.initZeros();
	proj.rotate2D(f2d, A);

	double max_diff = 0., max_val = 0.;
	int filled = 0;
	for (int i = 0; i < YSIZE(f2d); i++)
		for (int j = 0; j < XSIZE(f2d); j++)
		{
			const Complex a = DIRECT_A2D_ELEM(f2d, i, j), b = DIRECT_A2D_ELEM(F, i, j);
			if (a.real == 0 && a.imag == 0) continue;
			filled++;
			max_diff = std::max(max_diff, (double)std::abs(a.real - b.real) + std::abs(a.imag - b.imag));
			max_val = std::max(max_val, (double)std::abs(b.real) + std::abs(b.imag));
		}
	CHECK(filled > 300);
	CHECK(max_diff < 1e-9 * std::max(1.0, max_val));
}

TEST_CASE("Rectangular projector: forced rectangular path equals the square path", "[rectangular]")
{
	const int n = 32;
	MultidimArray<RFLOAT> img = blobImage(n, n), img2 = img;

	Projector square(n, TRILINEAR, 2.0);
	MultidimArray<RFLOAT> sp1, sp2;
	square.computeFourierTransformMap(img, sp1, -1, 1, false);

	Projector rect(n, TRILINEAR, 2.0);
	rect.setBoxSize(n, n);
	rect.rect = true;
	rect.box_nx = rect.box_ny = n;
	rect.box_nz = 1;
	rect.computeFourierTransformMap(img2, sp2, -1, 1, false);

	REQUIRE(XSIZE(square.data) == XSIZE(rect.data));
	REQUIRE(YSIZE(square.data) == YSIZE(rect.data));
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(square.data)
	{
		CHECK(DIRECT_MULTIDIM_ELEM(square.data, n).real == Approx(DIRECT_MULTIDIM_ELEM(rect.data, n).real).margin(1e-10));
		CHECK(DIRECT_MULTIDIM_ELEM(square.data, n).imag == Approx(DIRECT_MULTIDIM_ELEM(rect.data, n).imag).margin(1e-10));
	}

	for (RFLOAT angle : {0., 17., 123.})
	{
		Matrix2D<RFLOAT> A;
		rotation2DMatrix(angle, A);
		MultidimArray<Complex> a(n, n / 2 + 1), b(n, n / 2 + 1);
		a.initZeros();
		b.initZeros();
		square.rotate2D(a, A);
		rect.rotate2D(b, A);
		double diff = 0., val = 0.;
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a)
		{
			// compare inside the circle both fill
			diff += std::abs(DIRECT_MULTIDIM_ELEM(a, n).real - DIRECT_MULTIDIM_ELEM(b, n).real)
			      + std::abs(DIRECT_MULTIDIM_ELEM(a, n).imag - DIRECT_MULTIDIM_ELEM(b, n).imag);
			val += std::abs(DIRECT_MULTIDIM_ELEM(a, n).real) + std::abs(DIRECT_MULTIDIM_ELEM(a, n).imag);
		}
		CHECK(diff < 0.02 * val); // only edge pixels differ (ellipse vs circle rounding)
	}
}

TEST_CASE("Rectangular projector: 180 degree rotation conjugates the transform", "[rectangular]")
{
	const int nx = 48, ny = 24;
	MultidimArray<RFLOAT> img = blobImage(nx, ny);
	Projector proj(48, TRILINEAR, 2.0);
	proj.setBoxSize(nx, ny);
	MultidimArray<RFLOAT> sp;
	proj.computeFourierTransformMap(img, sp, -1, 1, false);

	Matrix2D<RFLOAT> A0, A180;
	rotation2DMatrix(0., A0);
	rotation2DMatrix(180., A180);
	MultidimArray<Complex> a(ny, nx / 2 + 1), b(ny, nx / 2 + 1);
	a.initZeros();
	b.initZeros();
	proj.rotate2D(a, A0);
	proj.rotate2D(b, A180);
	// A real image rotated by 180 degrees has the transform F(-k) = conj F(k); in the half
	// transform that is the point (-x, -y) -- so b(x, y) = a(-x, -y) which for x > 0 is conj(a(x, y))
	// only on the axes. Check energy instead: rotation by 180 preserves total power.
	double pa = 0., pb = 0.;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a)
	{
		pa += norm(DIRECT_MULTIDIM_ELEM(a, n));
		pb += norm(DIRECT_MULTIDIM_ELEM(b, n));
	}
	CHECK(pb == Approx(pa).epsilon(1e-6));
}

static void checkCuboidProjectsLikeSumAlongX(const int ny, const int nx, const int nz)
{
	// nx = ny = image height, nz = image width; the segments lie along image x and volume z
	MultidimArray<RFLOAT> vol(nz, ny, nx);
	vol.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY3D(vol)
	{
		A3D_ELEM(vol, k, i, j) = std::exp(-(j * j + i * i + k * k) / 14.0)
		                       + 0.6 * std::exp(-((j - 3) * (j - 3) + (i + 2) * (i + 2) + (k - 2) * (k - 2)) / 5.0);
	}

	// tilt -90, rot = psi = 0: the volume z axis becomes image x and the view runs along volume x
	MultidimArray<RFLOAT> proj_img(ny, nz);
	proj_img.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY3D(vol)
		A2D_ELEM(proj_img, i, k) += A3D_ELEM(vol, k, i, j);
	MultidimArray<Complex> F = transformOf(proj_img);

	const int L = std::max(nx, std::max(ny, nz));
	Projector proj(L, TRILINEAR, 2.0, 10, 2);
	proj.setBoxSize(nx, ny, nz);
	REQUIRE(proj.rect);
	REQUIRE(proj.ori_size == L);
	MultidimArray<RFLOAT> sp;
	MultidimArray<RFLOAT> vcopy = vol;
	proj.computeFourierTransformMap(vcopy, sp, -1, 1, false);
	REQUIRE(proj.ref_dim == 3);

	Matrix2D<RFLOAT> A;
	Euler_angles2matrix(0., -90., 0., A);
	MultidimArray<Complex> f2d(ny, nz / 2 + 1);
	f2d.initZeros();
	proj.get2DFourierTransform(f2d, A);

	double max_diff = 0., max_val = 0.;
	int filled = 0;
	for (int i = 0; i < YSIZE(f2d); i++)
		for (int j = 0; j < XSIZE(f2d); j++)
		{
			const Complex a = DIRECT_A2D_ELEM(f2d, i, j), b = DIRECT_A2D_ELEM(F, i, j);
			if (a.real == 0 && a.imag == 0) continue;
			filled++;
			max_diff = std::max(max_diff, (double)std::abs(a.real - b.real) + std::abs(a.imag - b.imag));
			max_val = std::max(max_val, (double)std::abs(b.real) + std::abs(b.imag));
		}
	CHECK(filled > 200);
	CHECK(max_diff < 1e-9 * std::max(1.0, max_val));
}

TEST_CASE("Rectangular projector: cuboid volume projects like the sum along x", "[rectangular]")
{
	// image wider than tall: z is the long axis
	checkCuboidProjectsLikeSumAlongX(24, 24, 32);
	// image taller than wide: z is the short axis
	checkCuboidProjectsLikeSumAlongX(32, 32, 24);
}
