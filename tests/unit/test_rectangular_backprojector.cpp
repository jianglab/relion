/*
 * tests/unit/test_rectangular_backprojector.cpp
 *
 * BackProjector for rectangular 2D references and cuboid 3D references.
 */

#include <catch2/catch.hpp>

#include "src/backprojector.h"
#include "src/projector.h"
#include "src/fftw.h"
#include "src/euler.h"

#include <cmath>
#include <vector>

namespace
{
// Smooth analytic image, so that any rotation of it can be sampled exactly
double blobAt(double x, double y)
{
	return std::exp(-(x * x + y * y) / 18.0)
	     + 0.5 * std::exp(-((x - 4) * (x - 4) + (y + 2) * (y + 2)) / 6.0);
}

MultidimArray<RFLOAT> blobImage(int nx, int ny, bool flipped = false)
{
	MultidimArray<RFLOAT> img(ny, nx);
	img.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY2D(img)
		A2D_ELEM(img, i, j) = flipped ? blobAt(-j, -i) : blobAt(j, i);
	return img;
}

MultidimArray<Complex> transformOf(MultidimArray<RFLOAT> img)
{
	MultidimArray<Complex> F;
	FourierTransformer ft;
	MultidimArray<RFLOAT> copy = img;
	copy.setXmippOrigin();
	CenterFFT(copy, true);
	ft.FourierTransform(copy, F, true);
	return F;
}

double correlation(const MultidimArray<RFLOAT> &a, const MultidimArray<RFLOAT> &b, double *ratio = NULL)
{
	double sab = 0., saa = 0., sbb = 0.;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a)
	{
		const double x = DIRECT_MULTIDIM_ELEM(a, n), y = DIRECT_MULTIDIM_ELEM(b, n);
		sab += x * y;
		saa += x * x;
		sbb += y * y;
	}
	if (ratio != NULL) *ratio = sab / saa;
	return sab / std::sqrt(saa * sbb);
}

MultidimArray<RFLOAT> blobVolume(int nx, int ny, int nz)
{
	MultidimArray<RFLOAT> vol(nz, ny, nx);
	vol.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY3D(vol)
	{
		A3D_ELEM(vol, k, i, j) = std::exp(-(j * j + i * i + k * k) / 14.0)
		                       + 0.6 * std::exp(-((j - 3) * (j - 3) + (i + 2) * (i + 2) + (k - 2) * (k - 2)) / 5.0);
	}
	return vol;
}

// Directions spread over the sphere
std::vector<std::vector<double> > viewAngles(int n)
{
	std::vector<std::vector<double> > v;
	const double golden = PI * (3. - std::sqrt(5.));
	for (int m = 0; m < n; m++)
	{
		const double z = 1. - 2. * (m + 0.5) / n;
		std::vector<double> a(3);
		a[0] = RAD2DEG(golden * m);
		a[1] = RAD2DEG(std::acos(z));
		a[2] = 37. * m;
		v.push_back(a);
	}
	return v;
}

// Back-project projections of a cuboid (nx = ny = image height, nz = image width), using the rectangular Projector as forward model
void fillFromProjections(BackProjector &bp, const MultidimArray<RFLOAT> &vol, int nx, int ny, int nz, int nviews)
{
	Projector proj(bp.ori_size, TRILINEAR, 2.0, 10, 2);
	proj.setBoxSize(nx, ny, nz);
	MultidimArray<RFLOAT> vcopy = vol, sp;
	proj.computeFourierTransformMap(vcopy, sp, -1, 1, true);

	std::vector<std::vector<double> > views = viewAngles(nviews);
	for (size_t m = 0; m < views.size(); m++)
	{
		Matrix2D<RFLOAT> A;
		Euler_angles2matrix(views[m][0], views[m][1], views[m][2], A);
		MultidimArray<Complex> f2d(nx, nz / 2 + 1);
		f2d.initZeros();
		proj.get2DFourierTransform(f2d, A);
		bp.set2DFourierTransform(f2d, A);
	}
}
}

TEST_CASE("Rectangular backprojector: forced rectangular path equals the square path", "[rectangular]")
{
	const int n = 32;
	std::vector<RFLOAT> angles = {0., 41., 97., 160., 233.};

	BackProjector square(n, 2, "c1", TRILINEAR, 2.0);
	BackProjector rect(n, 2, "c1", TRILINEAR, 2.0);
	rect.rect = true;
	rect.box_nx = rect.box_ny = n;
	rect.box_nz = 1;
	square.initZeros(-1);
	rect.initZeros(-1);
	REQUIRE(XSIZE(square.data) == XSIZE(rect.data));
	REQUIRE(YSIZE(square.data) == YSIZE(rect.data));
	REQUIRE(YSIZE(square.weight) == YSIZE(rect.weight));

	MultidimArray<Complex> F = transformOf(blobImage(n, n));
	for (RFLOAT a : angles)
	{
		Matrix2D<RFLOAT> A;
		rotation2DMatrix(a, A);
		square.set2DFourierTransform(F, A);
		rect.set2DFourierTransform(F, A);
	}

	double dd = 0., vv = 0., dw = 0., ww = 0.;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(square.data)
	{
		dd += std::abs(DIRECT_MULTIDIM_ELEM(square.data, n).real - DIRECT_MULTIDIM_ELEM(rect.data, n).real)
		    + std::abs(DIRECT_MULTIDIM_ELEM(square.data, n).imag - DIRECT_MULTIDIM_ELEM(rect.data, n).imag);
		vv += std::abs(DIRECT_MULTIDIM_ELEM(square.data, n).real) + std::abs(DIRECT_MULTIDIM_ELEM(square.data, n).imag);
		dw += std::abs(DIRECT_MULTIDIM_ELEM(square.weight, n) - DIRECT_MULTIDIM_ELEM(rect.weight, n));
		ww += DIRECT_MULTIDIM_ELEM(square.weight, n);
	}
	CHECK(vv > 0.);
	CHECK(dd <= 1e-9 * vv);
	CHECK(dw <= 1e-9 * ww);

	MultidimArray<RFLOAT> tau2, vs, vr;
	tau2.initZeros(n / 2 + 1);
	square.reconstruct(vs, 10, false, tau2, 1., 1., -1, false, NULL);
	rect.reconstruct(vr, 10, false, tau2, 1., 1., -1, false, NULL);
	REQUIRE(XSIZE(vs) == XSIZE(vr));
	REQUIRE(YSIZE(vs) == YSIZE(vr));
	double ratio;
	CHECK(correlation(vs, vr, &ratio) > 0.99999);
	CHECK(ratio == Approx(1.).epsilon(1e-6));
}

TEST_CASE("Rectangular backprojector: 48 x 24 image is recovered from two views", "[rectangular]")
{
	const int nx = 48, ny = 24;
	MultidimArray<RFLOAT> img = blobImage(nx, ny);
	MultidimArray<Complex> F0 = transformOf(img);
	MultidimArray<Complex> F180 = transformOf(blobImage(nx, ny, true)); // the image turned by 180 degrees

	BackProjector bp(48, 2, "c1", TRILINEAR, 2.0);
	bp.setBoxSize(nx, ny);
	REQUIRE(bp.rect);
	REQUIRE(bp.ori_size == 48);
	bp.initZeros(-1);
	REQUIRE(bp.ref_dim == 2);
	REQUIRE(XSIZE(bp.data) > 1);

	Matrix2D<RFLOAT> A0, A180;
	rotation2DMatrix(0., A0);
	rotation2DMatrix(180., A180);
	bp.set2DFourierTransform(F0, A0);
	bp.set2DFourierTransform(F180, A180);

	MultidimArray<RFLOAT> tau2, vol;
	tau2.initZeros(48 / 2 + 1);
	bp.reconstruct(vol, 10, false, tau2, 1., 1., -1, false, NULL);

	REQUIRE(XSIZE(vol) == nx);
	REQUIRE(YSIZE(vol) == ny);
	double ratio;
	vol.setXmippOrigin();
	img.setXmippOrigin();
	CHECK(correlation(vol, img, &ratio) > 0.99);
	CHECK(ratio == Approx(1.).epsilon(0.1));
}

TEST_CASE("Rectangular backprojector: 24 x 24 x 32 cuboid from projections", "[rectangular]")
{
	const int nx = 24, ny = 24, nz = 32;
	MultidimArray<RFLOAT> vol = blobVolume(nx, ny, nz);

	BackProjector bp(32, 3, "c1", TRILINEAR, 2.0, 10, 0, 1.9, 15, 2);
	bp.setBoxSize(nx, ny, nz);
	REQUIRE(bp.rect);
	REQUIRE(bp.ori_size == 32);
	bp.initZeros(-1);
	REQUIRE(bp.ref_dim == 3);
	fillFromProjections(bp, vol, nx, ny, nz, 80);

	MultidimArray<RFLOAT> tau2, rec;
	tau2.initZeros(32 / 2 + 1);
	bp.reconstruct(rec, 10, false, tau2, 1., 1., -1, false, NULL);

	REQUIRE(XSIZE(rec) == nx);
	REQUIRE(YSIZE(rec) == ny);
	REQUIRE(ZSIZE(rec) == nz);
	double ratio;
	CHECK(correlation(rec, vol, &ratio) > 0.99);
	CHECK(ratio == Approx(1.).epsilon(0.15));
}

TEST_CASE("Rectangular backprojector: 32 x 32 x 24 cuboid (short z) from projections", "[rectangular]")
{
	const int nx = 32, ny = 32, nz = 24;
	MultidimArray<RFLOAT> vol = blobVolume(nx, ny, nz);

	BackProjector bp(32, 3, "c1", TRILINEAR, 2.0, 10, 0, 1.9, 15, 2);
	bp.setBoxSize(nx, ny, nz);
	REQUIRE(bp.rect);
	REQUIRE(bp.ori_size == 32);
	bp.initZeros(-1);
	REQUIRE(bp.ref_dim == 3);
	fillFromProjections(bp, vol, nx, ny, nz, 80);

	MultidimArray<RFLOAT> tau2, rec;
	tau2.initZeros(32 / 2 + 1);
	bp.reconstruct(rec, 10, false, tau2, 1., 1., -1, false, NULL);

	REQUIRE(XSIZE(rec) == nx);
	REQUIRE(YSIZE(rec) == ny);
	REQUIRE(ZSIZE(rec) == nz);
	double ratio;
	CHECK(correlation(rec, vol, &ratio) > 0.99);
	CHECK(ratio == Approx(1.).epsilon(0.15));
}

TEST_CASE("Rectangular backprojector: Hermitian symmetry, low-res exchange and FSC", "[rectangular]")
{
	const int nx = 24, ny = 24, nz = 32;
	MultidimArray<RFLOAT> vol = blobVolume(nx, ny, nz);

	BackProjector bp(32, 3, "c1", TRILINEAR, 2.0, 10, 0, 1.9, 15, 2);
	bp.setBoxSize(nx, ny, nz);
	bp.initZeros(-1);
	fillFromProjections(bp, vol, nx, ny, nz, 20);

	// Hermitian symmetry on the x = 0 plane after enforcing it
	bp.enforceHermitianSymmetry();
	double worst = 0.;
	for (int k = STARTINGZ(bp.data) + 1; k <= FINISHINGZ(bp.data); k++)
		for (int i = STARTINGY(bp.data) + 1; i <= FINISHINGY(bp.data); i++)
		{
			if (k == 0 && i == 0) continue; // the origin is its own partner and is left alone, as in the square code
			const Complex a = A3D_ELEM(bp.data, k, i, 0), b = A3D_ELEM(bp.data, -k, -i, 0);
			worst = std::max(worst, (double)(std::abs(a.real - b.real) + std::abs(a.imag + b.imag)));
			worst = std::max(worst, (double)std::abs(A3D_ELEM(bp.weight, k, i, 0) - A3D_ELEM(bp.weight, -k, -i, 0)));
		}
	CHECK(worst < 1e-9);

	// Low-resolution data and weight can be taken out and put into an empty twin
	BackProjector twin(32, 3, "c1", TRILINEAR, 2.0, 10, 0, 1.9, 15, 2);
	twin.setBoxSize(nx, ny, nz);
	twin.initZeros(-1);
	MultidimArray<Complex> lowd;
	MultidimArray<RFLOAT> loww;
	bp.getLowResDataAndWeight(lowd, loww, 6);
	twin.setLowResDataAndWeight(lowd, loww, 6);
	double sw_bp = 0., sw_twin = 0.;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(twin.weight)
		sw_twin += DIRECT_MULTIDIM_ELEM(twin.weight, n);
	REQUIRE(sw_twin > 0.);
	const double rr = 12. * 12.;
	FOR_ALL_ELEMENTS_IN_ARRAY3D(bp.weight)
	{
		const double x = j * bp.shellScaleX(), y = i * bp.shellScaleY(), z = k * bp.shellScaleZ();
		if (x * x + y * y + z * z <= rr)
		{
			CHECK(A3D_ELEM(twin.weight, k, i, j) == Approx(A3D_ELEM(bp.weight, k, i, j)).margin(1e-12));
			sw_bp += A3D_ELEM(bp.weight, k, i, j);
		}
	}
	CHECK(sw_bp == Approx(sw_twin).epsilon(1e-9));

	// FSC of a half set with itself is one wherever there is data
	MultidimArray<Complex> avg;
	bp.getDownsampledAverage(avg, false);
	MultidimArray<RFLOAT> fsc;
	bp.calculateDownSampledFourierShellCorrelation(avg, avg, fsc);
	REQUIRE(XSIZE(fsc) == 32 / 2 + 1);
	for (int i = 0; i <= 8; i++)
		CHECK(fsc(i) == Approx(1.).epsilon(1e-9));

	// SSNR arrays have length L/2+1
	MultidimArray<RFLOAT> tau2, sigma2, dvp, cov, avgctf2;
	tau2.initZeros(32 / 2 + 1);
	avgctf2.initZeros(32 / 2 + 1);
	bp.updateSSNRarrays(1., tau2, sigma2, dvp, cov, fsc, avgctf2, false, false, false);
	CHECK(XSIZE(sigma2) == 32 / 2 + 1);
	CHECK(XSIZE(dvp) == 32 / 2 + 1);
	CHECK(XSIZE(cov) == 32 / 2 + 1);
}

TEST_CASE("Rectangular backprojector: C2 symmetrisation of a cuboid", "[rectangular]")
{
	const int nx = 24, ny = 24, nz = 32;
	// a volume with two-fold symmetry about z
	MultidimArray<RFLOAT> vol(nz, ny, nx);
	vol.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY3D(vol)
	{
		A3D_ELEM(vol, k, i, j) = std::exp(-(j * j + i * i + k * k) / 14.0)
		                       + 0.6 * std::exp(-((j - 4) * (j - 4) + (i - 3) * (i - 3) + (k - 2) * (k - 2)) / 5.0)
		                       + 0.6 * std::exp(-((j + 4) * (j + 4) + (i + 3) * (i + 3) + (k - 2) * (k - 2)) / 5.0);
	}

	BackProjector bp(32, 3, "c2", TRILINEAR, 2.0, 10, 0, 1.9, 15, 2);
	bp.setBoxSize(nx, ny, nz);
	bp.initZeros(-1);
	fillFromProjections(bp, vol, nx, ny, nz, 12);
	bp.symmetrise(1, 0., 0., 1);

	MultidimArray<RFLOAT> tau2, rec;
	tau2.initZeros(32 / 2 + 1);
	bp.reconstruct(rec, 10, false, tau2, 1., 1., -1, false, NULL);
	rec.setXmippOrigin();

	double diff = 0., val = 0.;
	for (int k = -nz / 2 + 1; k < nz / 2; k++)
		for (int i = -ny / 2 + 1; i < ny / 2; i++)
			for (int j = -nx / 2 + 1; j < nx / 2; j++)
			{
				diff += std::abs(A3D_ELEM(rec, k, i, j) - A3D_ELEM(rec, k, -i, -j));
				val += std::abs(A3D_ELEM(rec, k, i, j));
			}
	CHECK(val > 0.);
	CHECK(diff < 0.03 * val);
	double ratio;
	CHECK(correlation(rec, vol, &ratio) > 0.97);
}

TEST_CASE("Rectangular backprojector: unsupported operations give errors", "[rectangular]")
{
	BackProjector bp(32, 3, "c1", TRILINEAR, 2.0, 10, 0, 1.9, 15, 2);
	bp.setBoxSize(24, 24, 32);
	bp.initZeros(-1);
	MultidimArray<RFLOAT> tau2, vol;
	tau2.initZeros(17);
	CHECK_THROWS(bp.reconstructGrad(vol, tau2, 1., 1., 1., false, false));
}
