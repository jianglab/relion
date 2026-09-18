/*
 * tests/unit/test_finufft_projector.cpp
 *
 * Unit tests for the FINUFFT central-slice projector (interpolator == FINUFFT).
 *
 * Test strategy:
 *
 *  1. evaluateNonuniformFourierSamplesFromFourierVolume3D reproduces a
 *     brute-force direct summation of the same mode array, in 3D and in 2D.
 *  2. Projector::project() with FINUFFT reproduces `data` exactly at integer
 *     query positions (the identity orientation puts every query point on a
 *     grid node, where the band-limited interpolant must be interpolatory).
 *  3. Projector::project() with FINUFFT is much closer to the exact central
 *     slice (a direct DFT of the input volume) than the trilinear projector.
 *  4. Projector::rotate2D() with FINUFFT is exact for the identity rotation of
 *     a 2D reference.
 *  5. get2DFourierTransformMany() gives bit-comparable results to projecting
 *     each orientation separately.
 *
 * All test volumes are small and band-limited, so the spherical band-limit that
 * computeFourierTransformMap() applies to `data` is not a confounder.
 */

#include <catch2/catch.hpp>

#include "src/projector.h"
#include "src/finufft_central_slice.h"
#include "src/euler.h"
#include "src/multidim_array.h"
#include "src/complex.h"
#include "src/macros.h"
#include "src/matrix2d.h"
#include "src/transformations.h"

#include <cmath>
#include <vector>

namespace
{

// A deterministic pseudo-random sequence, so the tests do not depend on the
// platform's rand().
struct Lcg
{
	unsigned long long state;
	explicit Lcg(unsigned long long seed) : state(seed) {}
	double next()   // uniform in [-1, 1)
	{
		state = state * 6364136223846793005ULL + 1442695040888963407ULL;
		return ((double)((state >> 11) & 0x1FFFFFFFFFFFFFULL) / (double)0x20000000000000ULL) * 2.0 - 1.0;
	}
};

/// Brute-force evaluation of  (1/P^dim) * sum_r modes[r] * exp(-2 pi i q.r / P)
Complex directSum(const FinufftProjectorModes& m, RFLOAT qx, RFLOAT qy, RFLOAT qz)
{
	const int M = m.mode_size;
	const double ang = -2.0 * PI / (double)m.padded_size;

	double re = 0.0, im = 0.0;
	if (m.dim == 3)
	{
		for (int t3 = 0; t3 < M; t3++)
		for (int t2 = 0; t2 < M; t2++)
		for (int t1 = 0; t1 < M; t1++)
		{
			const double phase = ang * ((double)(t1 - M/2) * qx +
			                            (double)(t2 - M/2) * qy +
			                            (double)(t3 - M/2) * qz);
			const std::complex<RFLOAT>& v = m.modes[((size_t)t3 * M + t2) * M + t1];
			re += (double)v.real() * cos(phase) - (double)v.imag() * sin(phase);
			im += (double)v.real() * sin(phase) + (double)v.imag() * cos(phase);
		}
		const double norm = 1.0 / ((double)m.padded_size * m.padded_size * m.padded_size);
		return Complex(re * norm, im * norm);
	}

	for (int t2 = 0; t2 < M; t2++)
	for (int t1 = 0; t1 < M; t1++)
	{
		const double phase = ang * ((double)(t1 - M/2) * qx + (double)(t2 - M/2) * qy);
		const std::complex<RFLOAT>& v = m.modes[(size_t)t2 * M + t1];
		re += (double)v.real() * cos(phase) - (double)v.imag() * sin(phase);
		im += (double)v.real() * sin(phase) + (double)v.imag() * cos(phase);
	}
	const double norm = 1.0 / ((double)m.padded_size * m.padded_size);
	return Complex(re * norm, im * norm);
}

/// A band-limited 3D test object: a handful of Gaussian blobs inside the box.
MultidimArray<RFLOAT> makeBlobVolume(int size, RFLOAT sigma = 1.6)
{
	struct Blob { RFLOAT x, y, z, a; };
	const Blob blobs[] = {
		{  0.0,  0.0,  0.0,  1.0 },
		{  4.3, -2.1,  1.7,  0.7 },
		{ -3.5,  3.9, -2.4, -0.5 },
		{  1.2,  5.1,  4.6,  0.4 },
	};
	const int nb = sizeof(blobs) / sizeof(blobs[0]);

	MultidimArray<RFLOAT> vol(size, size, size);
	vol.setXmippOrigin();

	const RFLOAT inv2s2 = 1.0 / (2.0 * sigma * sigma);
	FOR_ALL_ELEMENTS_IN_ARRAY3D(vol)
	{
		RFLOAT v = 0.0;
		for (int b = 0; b < nb; b++)
		{
			const RFLOAT dx = j - blobs[b].x;
			const RFLOAT dy = i - blobs[b].y;
			const RFLOAT dz = k - blobs[b].z;
			v += blobs[b].a * exp(-(dx*dx + dy*dy + dz*dz) * inv2s2);
		}
		A3D_ELEM(vol, k, i, j) = v;
	}
	return vol;
}

/// The exact central slice of `vol` at orientation A, in RELION's normalisation:
///   F(q) = (1/ori_size^2) * sum_r vol[r] exp(-2 pi i q.r / ori_size)
/// with q = Ainv * (x, y) in *unpadded* frequency units.
MultidimArray<Complex> exactCentralSlice(const MultidimArray<RFLOAT>& vol,
		Matrix2D<RFLOAT>& A, int out_size)
{
	const int ori_size = XSIZE(vol);
	Matrix2D<RFLOAT> Ainv = A.inv();

	MultidimArray<Complex> f2d(out_size, out_size/2 + 1);
	f2d.initZeros();

	const int r_max_out = XSIZE(f2d) - 1;
	const int r_max_out_2 = r_max_out * r_max_out;
	const double ang = -2.0 * PI / (double)ori_size;
	const double norm = 1.0 / ((double)ori_size * (double)ori_size);

	for (int i = 0; i < YSIZE(f2d); i++)
	{
		const int y = (i <= r_max_out) ? i : i - YSIZE(f2d);
		for (int x = 0; x <= r_max_out; x++)
		{
			if (x*x + y*y > r_max_out_2) continue;

			const double qx = Ainv(0,0) * x + Ainv(0,1) * y;
			const double qy = Ainv(1,0) * x + Ainv(1,1) * y;
			const double qz = Ainv(2,0) * x + Ainv(2,1) * y;

			double re = 0.0, im = 0.0;
			FOR_ALL_ELEMENTS_IN_ARRAY3D(vol)
			{
				const double phase = ang * (qx * j + qy * i + qz * k);
				const double v = A3D_ELEM(vol, k, i, j);
				re += v * cos(phase);
				im += v * sin(phase);
			}
			DIRECT_A2D_ELEM(f2d, i, x) = Complex(re * norm, im * norm);
		}
	}
	return f2d;
}

/// Relative L2 difference over the pixels inside r_max_out
double relativeError(const MultidimArray<Complex>& a, const MultidimArray<Complex>& ref)
{
	const int r_max_out = XSIZE(ref) - 1;
	const int r_max_out_2 = r_max_out * r_max_out;

	double num = 0.0, den = 0.0;
	for (int i = 0; i < YSIZE(ref); i++)
	{
		const int y = (i <= r_max_out) ? i : i - YSIZE(ref);
		for (int x = 0; x <= r_max_out; x++)
		{
			if (x*x + y*y > r_max_out_2) continue;
			const Complex d = DIRECT_A2D_ELEM(a, i, x) - DIRECT_A2D_ELEM(ref, i, x);
			num += d.real*d.real + d.imag*d.imag;
			const Complex r = DIRECT_A2D_ELEM(ref, i, x);
			den += r.real*r.real + r.imag*r.imag;
		}
	}
	return (den > 0.0) ? sqrt(num / den) : 0.0;
}

/// RAII guard so one test's mode-crop setting does not leak into the next.
struct FinufftConfigGuard
{
	float saved_crop;
	double saved_tol;
	double saved_ups;
	FinufftConfigGuard()
		: saved_crop(Projector::finufft_mode_crop),
		  saved_tol(Projector::finufft_tol),
		  saved_ups(Projector::finufft_upsampfac) {}
	~FinufftConfigGuard()
	{
		Projector::finufft_mode_crop = saved_crop;
		Projector::finufft_tol = saved_tol;
		Projector::finufft_upsampfac = saved_ups;
	}
};

} // anonymous namespace

// ---------------------------------------------------------------------------

TEST_CASE("FINUFFT volume sampler matches a direct sum in 3D", "[finufft_projector]")
{
	if (!haveFinufftSupport())
	{
		WARN("built without FINUFFT support - skipping");
		return;
	}

	const int M = 10;
	const int P = 24;

	FinufftProjectorModes modes;
	modes.dim = 3;
	modes.mode_size = M;
	modes.padded_size = P;
	modes.modes.resize((size_t)M * M * M);

	Lcg rng(12345);
	for (size_t n = 0; n < modes.modes.size(); n++)
		modes.modes[n] = std::complex<RFLOAT>((RFLOAT)rng.next(), (RFLOAT)0);

	const int n_points = 25;
	std::vector<RFLOAT> qx(n_points), qy(n_points), qz(n_points);
	for (int n = 0; n < n_points; n++)
	{
		qx[n] = (RFLOAT)(rng.next() * (P / 2));
		qy[n] = (RFLOAT)(rng.next() * (P / 2));
		qz[n] = (RFLOAT)(rng.next() * (P / 2));
	}

	std::vector<Complex> got(n_points);
	evaluateNonuniformFourierSamplesFromFourierVolume3D(
		modes, qx.data(), qy.data(), qz.data(), n_points, got.data(), 1e-12, 2.0);

	for (int n = 0; n < n_points; n++)
	{
		const Complex want = directSum(modes, qx[n], qy[n], qz[n]);
		CHECK(got[n].real == Approx(want.real).margin(1e-9));
		CHECK(got[n].imag == Approx(want.imag).margin(1e-9));
	}
}

TEST_CASE("FINUFFT volume sampler matches a direct sum in 2D", "[finufft_projector]")
{
	if (!haveFinufftSupport())
	{
		WARN("built without FINUFFT support - skipping");
		return;
	}

	const int M = 16;
	const int P = 32;

	FinufftProjectorModes modes;
	modes.dim = 2;
	modes.mode_size = M;
	modes.padded_size = P;
	modes.modes.resize((size_t)M * M);

	Lcg rng(987654321);
	for (size_t n = 0; n < modes.modes.size(); n++)
		modes.modes[n] = std::complex<RFLOAT>((RFLOAT)rng.next(), (RFLOAT)0);

	const int n_points = 30;
	std::vector<RFLOAT> qx(n_points), qy(n_points);
	for (int n = 0; n < n_points; n++)
	{
		qx[n] = (RFLOAT)(rng.next() * (P / 2));
		qy[n] = (RFLOAT)(rng.next() * (P / 2));
	}

	std::vector<Complex> got(n_points);
	evaluateNonuniformFourierSamplesFromFourierVolume3D(
		modes, qx.data(), qy.data(), NULL, n_points, got.data(), 1e-12, 2.0);

	for (int n = 0; n < n_points; n++)
	{
		const Complex want = directSum(modes, qx[n], qy[n], 0.0);
		CHECK(got[n].real == Approx(want.real).margin(1e-9));
		CHECK(got[n].imag == Approx(want.imag).margin(1e-9));
	}
}

TEST_CASE("FINUFFT projector is interpolatory at grid nodes", "[finufft_projector]")
{
	if (!haveFinufftSupport())
	{
		WARN("built without FINUFFT support - skipping");
		return;
	}

	FinufftConfigGuard guard;
	// Keeping the whole padded box makes the interpolation of `data` exact
	Projector::finufft_mode_crop = 2.0f;
	Projector::finufft_tol = 1e-12;
	Projector::finufft_upsampfac = 2.0;

	const int ori_size = 32;
	MultidimArray<RFLOAT> vol = makeBlobVolume(ori_size);
	MultidimArray<RFLOAT> dummy;

	Projector proj(ori_size, FINUFFT, 2.0);
	proj.computeFourierTransformMap(vol, dummy, -1, 1);

	REQUIRE(proj.finufft_modes.isPrepared());

	// The identity orientation samples `data` at integer positions (pf*x, pf*y, 0)
	Matrix2D<RFLOAT> A;
	Euler_angles2matrix(0., 0., 0., A, false);

	MultidimArray<Complex> f2d(ori_size, ori_size/2 + 1);
	f2d.initZeros();
	proj.project(f2d, A);

	const int pf = ROUND(proj.padding_factor);
	const int r_max_out = XSIZE(f2d) - 1;

	double max_abs_diff = 0.0, max_abs_ref = 0.0;
	for (int i = 0; i < YSIZE(f2d); i++)
	{
		const int y = (i <= r_max_out) ? i : i - YSIZE(f2d);
		for (int x = 0; x <= r_max_out; x++)
		{
			if (x*x + y*y > r_max_out*r_max_out) continue;
			if ((pf*x)*(pf*x) + (pf*y)*(pf*y) > (proj.r_max*pf)*(proj.r_max*pf)) continue;

			const Complex want = A3D_ELEM(proj.data, 0, pf*y, pf*x);
			const Complex got  = DIRECT_A2D_ELEM(f2d, i, x);
			max_abs_diff = XMIPP_MAX(max_abs_diff, sqrt(norm(got - want)));
			max_abs_ref  = XMIPP_MAX(max_abs_ref, sqrt(norm(want)));
		}
	}

	REQUIRE(max_abs_ref > 0.0);
	CHECK(max_abs_diff / max_abs_ref < 1e-8);
}

TEST_CASE("FINUFFT projector beats trilinear against the exact central slice", "[finufft_projector]")
{
	if (!haveFinufftSupport())
	{
		WARN("built without FINUFFT support - skipping");
		return;
	}

	FinufftConfigGuard guard;
	Projector::finufft_mode_crop = 1.0f;   // the default: keep only the unpadded box
	Projector::finufft_tol = 1e-8;

	const int ori_size = 32;
	const MultidimArray<RFLOAT> vol = makeBlobVolume(ori_size);

	// computeFourierTransformMap() modifies its input (gridding correction), so
	// each projector gets its own copy and the ground truth uses the original.
	MultidimArray<RFLOAT> vol_tri = vol;
	MultidimArray<RFLOAT> vol_nuf = vol;
	MultidimArray<RFLOAT> dummy;

	Projector proj_tri(ori_size, TRILINEAR, 2.0);
	proj_tri.computeFourierTransformMap(vol_tri, dummy, -1, 1);

	Projector proj_nuf(ori_size, FINUFFT, 2.0);
	proj_nuf.computeFourierTransformMap(vol_nuf, dummy, -1, 1);

	// An orientation that puts essentially every query point off-grid
	Matrix2D<RFLOAT> A;
	Euler_angles2matrix(37.0, 61.0, 23.0, A, false);

	MultidimArray<Complex> f_tri(ori_size, ori_size/2 + 1);
	MultidimArray<Complex> f_nuf(ori_size, ori_size/2 + 1);
	f_tri.initZeros();
	f_nuf.initZeros();

	proj_tri.project(f_tri, A);
	proj_nuf.project(f_nuf, A);

	MultidimArray<RFLOAT> vol_gt = vol;
	vol_gt.setXmippOrigin();
	const MultidimArray<Complex> f_gt = exactCentralSlice(vol_gt, A, ori_size);

	const double err_tri = relativeError(f_tri, f_gt);
	const double err_nuf = relativeError(f_nuf, f_gt);

	WARN("relative error vs exact central slice: trilinear = " << err_tri
	     << ", FINUFFT = " << err_nuf);

	CHECK(err_nuf < 1e-4);
	CHECK(err_nuf < 0.1 * err_tri);
}

TEST_CASE("FINUFFT rotate2D is interpolatory at grid nodes", "[finufft_projector]")
{
	if (!haveFinufftSupport())
	{
		WARN("built without FINUFFT support - skipping");
		return;
	}

	FinufftConfigGuard guard;
	Projector::finufft_mode_crop = 2.0f;
	Projector::finufft_tol = 1e-12;
	Projector::finufft_upsampfac = 2.0;

	const int ori_size = 32;
	MultidimArray<RFLOAT> img(ori_size, ori_size);
	img.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY2D(img)
	{
		const RFLOAT r2 = (RFLOAT)(i*i + j*j);
		A2D_ELEM(img, i, j) = exp(-r2 / (2.0 * 2.0 * 2.0)) + 0.4 * exp(-((j-5)*(j-5) + (i+3)*(i+3)) / 8.0);
	}

	MultidimArray<RFLOAT> dummy;
	Projector proj(ori_size, FINUFFT, 2.0);
	proj.computeFourierTransformMap(img, dummy, -1, 1);
	REQUIRE(proj.ref_dim == 2);
	REQUIRE(proj.finufft_modes.isPrepared());
	REQUIRE(proj.finufft_modes.dim == 2);

	Matrix2D<RFLOAT> A;
	rotation2DMatrix(0., A);

	MultidimArray<Complex> f2d(ori_size, ori_size/2 + 1);
	f2d.initZeros();
	proj.rotate2D(f2d, A);

	const int pf = ROUND(proj.padding_factor);
	const int r_max_out = XSIZE(f2d) - 1;

	double max_abs_diff = 0.0, max_abs_ref = 0.0;
	for (int i = 0; i < YSIZE(f2d); i++)
	{
		const int y = (i <= r_max_out) ? i : i - YSIZE(f2d);
		for (int x = 0; x <= r_max_out; x++)
		{
			if (x*x + y*y > r_max_out*r_max_out) continue;
			if ((pf*x)*(pf*x) + (pf*y)*(pf*y) > (proj.r_max*pf)*(proj.r_max*pf)) continue;

			const Complex want = A2D_ELEM(proj.data, pf*y, pf*x);
			const Complex got  = DIRECT_A2D_ELEM(f2d, i, x);
			max_abs_diff = XMIPP_MAX(max_abs_diff, sqrt(norm(got - want)));
			max_abs_ref  = XMIPP_MAX(max_abs_ref, sqrt(norm(want)));
		}
	}

	REQUIRE(max_abs_ref > 0.0);
	CHECK(max_abs_diff / max_abs_ref < 1e-8);
}

TEST_CASE("get2DFourierTransformMany agrees with one call per orientation", "[finufft_projector]")
{
	if (!haveFinufftSupport())
	{
		WARN("built without FINUFFT support - skipping");
		return;
	}

	FinufftConfigGuard guard;
	Projector::finufft_mode_crop = 1.0f;
	Projector::finufft_tol = 1e-9;

	const int ori_size = 24;
	MultidimArray<RFLOAT> vol = makeBlobVolume(ori_size, 1.4);
	MultidimArray<RFLOAT> dummy;

	Projector proj(ori_size, FINUFFT, 2.0);
	proj.computeFourierTransformMap(vol, dummy, -1, 1);

	const RFLOAT angles[][3] = {
		{  0.0,   0.0,  0.0 },
		{ 31.0,  47.0, 11.0 },
		{ 95.0, 123.0, 77.0 },
		{200.0,  17.0,  5.0 },
	};
	const int n_orient = sizeof(angles) / sizeof(angles[0]);

	std::vector<Matrix2D<RFLOAT> > As(n_orient);
	std::vector<MultidimArray<Complex> > batched(n_orient);
	std::vector<MultidimArray<Complex> > single(n_orient);

	for (int n = 0; n < n_orient; n++)
	{
		Euler_angles2matrix(angles[n][0], angles[n][1], angles[n][2], As[n], false);
		batched[n].resize(ori_size, ori_size/2 + 1);
		batched[n].initZeros();
		single[n].resize(ori_size, ori_size/2 + 1);
		single[n].initZeros();
	}

	proj.get2DFourierTransformMany(batched, As);
	for (int n = 0; n < n_orient; n++)
		proj.get2DFourierTransform(single[n], As[n]);

	for (int io = 0; io < n_orient; io++)
	{
		double max_diff = 0.0, max_ref = 0.0;
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(single[io])
		{
			const Complex d = DIRECT_MULTIDIM_ELEM(batched[io], n)
			                - DIRECT_MULTIDIM_ELEM(single[io], n);
			max_diff = XMIPP_MAX(max_diff, sqrt(norm(d)));
			max_ref  = XMIPP_MAX(max_ref, sqrt(norm(DIRECT_MULTIDIM_ELEM(single[io], n))));
		}
		REQUIRE(max_ref > 0.0);
		CHECK(max_diff / max_ref < 1e-9);
	}
}
