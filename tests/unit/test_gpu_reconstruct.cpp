/*
 * tests/unit/test_gpu_reconstruct.cpp
 *
 * BackProjector::reconstruct() with skip_gridding on the GPU (RELION_GPU_RECONSTRUCT=1) must give the
 * CPU's map to rounding (cuFFT versus FFTW, and the per-shell weight sums added in another order).
 * Skipped when no GPU is visible.
 */

#include <catch2/catch.hpp>

#include "src/backprojector.h"
#include "src/fftw.h"
#include "src/gpu_locres.h"
#include "src/gpu_reconstruct.h"
#include "src/helix.h"

#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

TEST_CASE("GPU reconstruction (skip_gridding) equals the CPU's to rounding", "[gpu_reconstruct]")
{
	if (!gpurec::deviceAvailable())
	{
		WARN("no GPU visible: test skipped");
		return;
	}
	for (int current_size : {64, 40})   // full resolution, and a lower one (Fourier window grows)
	{
		INFO("current_size " << current_size);
		const int ori = 64;
		BackProjector bp(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, true);
		bp.initZeros(current_size);
		std::mt19937 rng(11);
		std::normal_distribution<RFLOAT> g(0., 1.);
		std::uniform_real_distribution<RFLOAT> u(0.5, 2.);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bp.data) DIRECT_MULTIDIM_ELEM(bp.data, n) = Complex(g(rng), g(rng));
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bp.weight) DIRECT_MULTIDIM_ELEM(bp.weight, n) = u(rng);
		// as in BackProjector::symmetrise() before every reconstruction: an inverse real FFT of a
		// non-Hermitian x = 0 plane depends on the library's algorithm
		bp.enforceHermitianSymmetry();
		// ... which leaves the zero frequency alone: real for any real map (the sum of the zero
		// frequencies of real images), but random here
		A3D_ELEM(bp.data, 0, 0, 0).imag = 0.;
		// Likewise the points on the Nyquist planes of the padded box, which the sphere reaches at full
		// resolution: libraries treat their (invalid) imaginary parts differently. In real data these are
		// single coefficients at the highest frequency, of no consequence.
		const int nyq = ROUND(bp.r_max * bp.padding_factor);
		FOR_ALL_ELEMENTS_IN_ARRAY3D(bp.data)
			if (ABS(k) == nyq || ABS(i) == nyq || j == nyq)
				A3D_ELEM(bp.data, k, i, j).imag = 0.;
		MultidimArray<RFLOAT> tau2(ori / 2 + 1);
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(tau2) DIRECT_A1D_ELEM(tau2, i) = (i % 7 == 3) ? 0. : u(rng);

		MultidimArray<RFLOAT> cpu;
		{
			BackProjector b = bp;
			gpurec::setGpuRequested(false);
			b.reconstruct(cpu, 1, true, tau2, 2., 1., 2);
		}
		// double precision: to rounding; single precision: to float accuracy
		{
			// Diagnostic: the CPU on input rounded to float, i.e. how sensitive the map is to the inputs
			BackProjector b = bp;
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(b.data)
				DIRECT_MULTIDIM_ELEM(b.data, n) = Complex((float)DIRECT_MULTIDIM_ELEM(b.data, n).real, (float)DIRECT_MULTIDIM_ELEM(b.data, n).imag);
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(b.weight)
				DIRECT_MULTIDIM_ELEM(b.weight, n) = (float)DIRECT_MULTIDIM_ELEM(b.weight, n);
			MultidimArray<RFLOAT> cf;
			gpurec::setGpuRequested(false);
			b.reconstruct(cf, 1, true, tau2, 2., 1., 2);
			double ss_diff = 0., ss = 0.;
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(cpu)
			{
				const double dd = DIRECT_MULTIDIM_ELEM(cpu, n) - DIRECT_MULTIDIM_ELEM(cf, n);
				ss_diff += dd * dd;
				ss += DIRECT_MULTIDIM_ELEM(cpu, n) * DIRECT_MULTIDIM_ELEM(cpu, n);
			}
			std::cout << " current_size " << current_size << ": CPU on float-rounded input, relative RMS difference "
			          << sqrt(ss_diff / ss) << std::endl;
		}
		// (single: RMS of the difference relative to the RMS of the map, and the largest difference
		// relative to the largest value)
		struct Prec { const char *name; double rms, max; };
		for (const Prec prec : {Prec{"double", 1e-12, 1e-9}, Prec{"single", 1e-5, 1e-3}})
		{
			INFO("precision " << prec.name);
			BackProjector b = bp;
			MultidimArray<RFLOAT> gpu;
			setenv("RELION_GPU_RECONSTRUCT", prec.name, 1);
			gpurec::setGpuRequested(true);
			b.reconstruct(gpu, 1, true, tau2, 2., 1., 2);
			unsetenv("RELION_GPU_RECONSTRUCT");
			gpurec::setGpuRequested(false);

			REQUIRE(cpu.sameShape(gpu));
			double maxdiff = 0., maxabs = 0., ss_diff = 0., ss = 0.;
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(cpu)
			{
				const double a = DIRECT_MULTIDIM_ELEM(cpu, n), dd = a - DIRECT_MULTIDIM_ELEM(gpu, n);
				maxdiff = std::max(maxdiff, fabs(dd));
				maxabs = std::max(maxabs, fabs(a));
				ss_diff += dd * dd;
				ss += a * a;
			}
			const double rel_rms = sqrt(ss_diff / ss);
			INFO("max |cpu - gpu| = " << maxdiff << ", max |cpu| = " << maxabs << ", relative RMS difference " << rel_rms);
			REQUIRE(maxabs > 0.);
			CHECK(rel_rms <= prec.rms);
			CHECK(maxdiff <= prec.max * maxabs);
		}
	}
}

TEST_CASE("GPU real-space helical symmetry equals the CPU's to rounding", "[gpu_reconstruct]")
{
	if (!gpurec::deviceAvailable())
	{
		WARN("no GPU visible: test skipped");
		return;
	}
	const int n = 48;
	const RFLOAT angpix = 2.;
	struct Case { RFLOAT sphere_A, inner_A, outer_A, z_frac, rise_A, twist; };
	for (const bool cyl : {false, true})   // a sphere, or a cylinder along Z (--helical_cylindrical_mask)
	for (const Case &c : {Case{44., -1., 30., 0.3, 4.75, -1.2}, Case{40., 8., 26., 0.3, 4., 30.},
	                      Case{46., -1., 20., 0.5, 2.38, 179.4}})
	{
		INFO("twist " << c.twist << " rise " << c.rise_A << " cylinder " << cyl);
		MultidimArray<RFLOAT> a(n, n, n);
		std::mt19937 rng(11);
		std::normal_distribution<RFLOAT> g(0., 1.);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a) DIRECT_MULTIDIM_ELEM(a, n) = g(rng);
		MultidimArray<RFLOAT> b = a;
		gpurec::setGpuRequested(false);
		imposeHelicalSymmetryInRealSpace(a, angpix, c.sphere_A, c.inner_A, c.outer_A, c.z_frac, c.rise_A, c.twist, 3., false, 4, cyl);
		setenv("RELION_GPU_RECONSTRUCT", "1", 1);
		gpurec::setGpuRequested(true);
		imposeHelicalSymmetryInRealSpace(b, angpix, c.sphere_A, c.inner_A, c.outer_A, c.z_frac, c.rise_A, c.twist, 3., false, 4, cyl);
		unsetenv("RELION_GPU_RECONSTRUCT");
		gpurec::setGpuRequested(false);
		REQUIRE(a.sameShape(b));
		double maxdiff = 0., maxabs = 0.;
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a)
		{
			maxdiff = std::max(maxdiff, (double)fabs(DIRECT_MULTIDIM_ELEM(a, n) - DIRECT_MULTIDIM_ELEM(b, n)));
			maxabs = std::max(maxabs, (double)fabs(DIRECT_MULTIDIM_ELEM(a, n)));
		}
		INFO("max |cpu - gpu| = " << maxdiff << ", max |cpu| = " << maxabs);
		REQUIRE(maxabs > 0.);
		CHECK(maxdiff <= 1e-12 * maxabs);
	}
}

TEST_CASE("GPU downsampleVolume3D equals CPU resizeMap to rounding", "[gpu_reconstruct]")
{
	if (!gpurec::deviceAvailable())
	{
		WARN("no GPU visible: test skipped");
		return;
	}
	const int orig = 64, target = 32;
	MultidimArray<RFLOAT> src(orig, orig, orig);
	std::mt19937 rng(23);
	std::normal_distribution<RFLOAT> g(0., 1.);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(src) DIRECT_MULTIDIM_ELEM(src, n) = g(rng);

	MultidimArray<RFLOAT> cpu = src;
	resizeMap(cpu, target);

	gpurec::ScopedDownsampleCache ds_cache(true);
	struct Prec { const char *name; double rms, max; };
	for (const Prec prec : {Prec{"double", 1e-12, 1e-10}, Prec{"single", 1e-5, 1e-3}})
	{
		INFO("precision " << prec.name);
		setenv("RELION_GPU_RECONSTRUCT", prec.name, 1);
		for (int pass = 0; pass < 2; pass++) // verify cached plans/buffers on pass 1
		{
			MultidimArray<RFLOAT> gpu(target, target, target);
			REQUIRE(gpurec::downsampleVolume3D(MULTIDIM_ARRAY(src), orig, MULTIDIM_ARRAY(gpu), target));
			double maxdiff = 0., maxabs = 0., ss_diff = 0., ss = 0.;
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(cpu)
			{
				const double a = DIRECT_MULTIDIM_ELEM(cpu, n);
				const double dd = a - DIRECT_MULTIDIM_ELEM(gpu, n);
				maxdiff = std::max(maxdiff, fabs(dd));
				maxabs = std::max(maxabs, fabs(a));
				ss_diff += dd * dd;
				ss += a * a;
			}
			const double rel_rms = sqrt(ss_diff / ss);
			CHECK(rel_rms <= prec.rms);
			CHECK(maxdiff <= prec.max * maxabs);
		}
		unsetenv("RELION_GPU_RECONSTRUCT");
	}
}

TEST_CASE("GPU LocalRes session matches CPU pruned FSC and filtered accumulation", "[gpu_reconstruct]")
{
	if (!gpurec::deviceAvailable())
	{
		WARN("no GPU visible: test skipped");
		return;
	}
	const long int N = 32, nx = N, ny = N, nz = N, nfx = N / 2 + 1;
	const long int R = 5, S = 2 * R + 1;
	const RFLOAT radius = 3.5, radius_p = 5.5;
	const long int nfx2 = nfx * nfx;

	auto fillRand = [&](MultidimArray<RFLOAT> &v, std::mt19937 &rng) {
		v.initZeros(nz, ny, nx);
		v.setXmippOrigin();
		std::normal_distribution<RFLOAT> g(0., 1.);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(v) DIRECT_MULTIDIM_ELEM(v, n) = g(rng);
	};

	std::mt19937 rng(37);
	MultidimArray<RFLOAT> I1, I2, I1p, I2p;
	fillRand(I1, rng);
	fillRand(I2, rng);
	fillRand(I1p, rng);
	fillRand(I2p, rng);

	MultidimArray<RFLOAT> Isum(nz, ny, nx);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Isum)
		DIRECT_MULTIDIM_ELEM(Isum, n) = (DIRECT_MULTIDIM_ELEM(I1, n) + DIRECT_MULTIDIM_ELEM(I2, n)) * (RFLOAT)0.5;
	FourierTransformer ft_sum;
	MultidimArray<Complex> FTsum;
	ft_sum.FourierTransform(Isum, FTsum, true);
	std::vector<Complex> FTsum_yxz((size_t)ny * nfx * nz);
	for (long int ky = 0; ky < ny; ky++)
		for (long int fx = 0; fx < nfx; fx++)
			for (long int kz = 0; kz < nz; kz++)
				FTsum_yxz[((size_t)ky * nfx + fx) * nz + kz] = DIRECT_A3D_ELEM(FTsum, kz, ky, fx);

	std::vector<RFLOAT> mask_tmpl((size_t)S * S * S, 0.);
	for (long int dk = -R; dk <= R; dk++)
	for (long int di = -R; di <= R; di++)
	for (long int dj = -R; dj <= R; dj++)
	{
		const RFLOAT d = sqrt((RFLOAT)(dk * dk + di * di + dj * dj));
		RFLOAT val = 0.;
		if (d <= radius) val = 1.;
		else if (d <= radius_p) val = 0.5 - 0.5 * cos(PI * (radius_p - d) / (radius_p - radius));
		mask_tmpl[((size_t)(dk + R) * S + (di + R)) * S + (dj + R)] = val;
	}

	std::vector<int> r2_to_shell((size_t)nfx2);
	for (long int d2 = 0; d2 < nfx2; d2++)
		r2_to_shell[d2] = ROUND(sqrt((RFLOAT)d2));

	const long int r2_bound = 144, n_fx_inv = 13;
	std::vector<double> inv_w((size_t)r2_bound + 1, 0.);
	for (long int d2 = 0; d2 <= r2_bound; d2++)
		inv_w[d2] = std::max(0.0, 1.0 - (double)d2 / (double)(r2_bound + 1));

	struct Pt { long int x, y, z; double inv_res; };
	const Pt pts[2] = { {0, 0, 0, 1.0 / 4.2}, {13, -12, 13, 1.0 / 5.1} }; // interior + boundary-clipped

	// CPU reference using full 3D FourierTransformer
	const long int sx0 = STARTINGX(I1), sy0 = STARTINGY(I1), sz0 = STARTINGZ(I1);
	std::vector<std::vector<double> > cpu_fscs(2, std::vector<double>((size_t)2 * nfx, 0.));
	MultidimArray<RFLOAT> cpu_Ifil(nz, ny, nx), cpu_Ilocres(nz, ny, nx), cpu_Isumw(nz, ny, nx);
	cpu_Ifil.initZeros(); cpu_Ilocres.initZeros(); cpu_Isumw.initZeros();
	for (int ip = 0; ip < 2; ip++)
	{
		const long int x = pts[ip].x, y = pts[ip].y, z = pts[ip].z;
		const long int z0 = std::max(sz0, z - R), z1 = std::min(FINISHINGZ(I1), z + R);
		const long int y0 = std::max(sy0, y - R), y1 = std::min(FINISHINGY(I1), y + R);
		const long int x0 = std::max(sx0, x - R), x1 = std::min(FINISHINGX(I1), x + R);

		auto evalFscPair = [&](const MultidimArray<RFLOAT> &A, const MultidimArray<RFLOAT> &B, double *fsc_out) {
			MultidimArray<RFLOAT> Am(nz, ny, nx), Bm(nz, ny, nx);
			Am.initZeros(); Bm.initZeros(); Am.setXmippOrigin(); Bm.setXmippOrigin();
			for (long int k = z0; k <= z1; k++)
			for (long int i = y0; i <= y1; i++)
			for (long int j = x0; j <= x1; j++)
			{
				const RFLOAT m = mask_tmpl[((size_t)(k - z + R) * S + (i - y + R)) * S + (j - x + R)];
				A3D_ELEM(Am, k, i, j) = A3D_ELEM(A, k, i, j) * m;
				A3D_ELEM(Bm, k, i, j) = A3D_ELEM(B, k, i, j) * m;
			}
			FourierTransformer tf1, tf2;
			MultidimArray<Complex> FA, FB;
			tf1.FourierTransform(Am, FA, true);
			tf2.FourierTransform(Bm, FB, true);
			std::vector<double> num(nfx, 0.), d1(nfx, 0.), d2v(nfx, 0.);
			for (long int kz = 0; kz < nz; kz++)
			{
				const long int kp = (kz < nfx) ? kz : (kz - nz);
				for (long int ky = 0; ky < ny; ky++)
				{
					const long int ipy = (ky < nfx) ? ky : (ky - ny);
					for (long int fx = 0; fx < nfx; fx++)
					{
						const long int d2 = kp * kp + ipy * ipy + fx * fx;
						if (d2 >= nfx2) continue;
						const int sh = r2_to_shell[d2];
						if (sh >= nfx) continue;
						const Complex a = DIRECT_A3D_ELEM(FA, kz, ky, fx);
						const Complex b = DIRECT_A3D_ELEM(FB, kz, ky, fx);
						num[sh] += a.real * b.real + a.imag * b.imag;
						d1[sh]  += a.real * a.real + a.imag * a.imag;
						d2v[sh] += b.real * b.real + b.imag * b.imag;
					}
				}
			}
			for (long int i = 0; i < nfx; i++)
				fsc_out[i] = num[i] / sqrt(d1[i] * d2v[i]);
		};
		evalFscPair(I1, I2, cpu_fscs[ip].data());
		evalFscPair(I1p, I2p, cpu_fscs[ip].data() + nfx);

		MultidimArray<Complex> Ffilt = FTsum;
		for (long int kz = 0; kz < nz; kz++)
		{
			const long int kp = (kz < nfx) ? kz : (kz - nz);
			for (long int ky = 0; ky < ny; ky++)
			{
				const long int ipy = (ky < nfx) ? ky : (ky - ny);
				for (long int fx = 0; fx < nfx; fx++)
				{
					const long int d2 = kp * kp + ipy * ipy + fx * fx;
					const double wt = (d2 <= r2_bound) ? inv_w[d2] : 0.0;
					DIRECT_A3D_ELEM(Ffilt, kz, ky, fx) *= (RFLOAT)wt;
				}
			}
		}
		MultidimArray<RFLOAT> Ifilt(nz, ny, nx);
		FourierTransformer tf_inv;
		tf_inv.inverseFourierTransform(Ffilt, Ifilt);
		Ifilt.setXmippOrigin();
		cpu_Ifil.setXmippOrigin(); cpu_Ilocres.setXmippOrigin(); cpu_Isumw.setXmippOrigin();
		for (long int k = z0; k <= z1; k++)
		for (long int i = y0; i <= y1; i++)
		for (long int j = x0; j <= x1; j++)
		{
			const RFLOAT m = mask_tmpl[((size_t)(k - z + R) * S + (i - y + R)) * S + (j - x + R)];
			A3D_ELEM(cpu_Ifil, k, i, j) += m * A3D_ELEM(Ifilt, k, i, j);
			A3D_ELEM(cpu_Ilocres, k, i, j) += m * (RFLOAT)pts[ip].inv_res;
			A3D_ELEM(cpu_Isumw, k, i, j) += m;
		}
	}

	struct Prec { bool fp64; double tol_fsc, tol_vol; };
	for (const Prec prec : {Prec{true, 1e-12, 1e-12}, Prec{false, 1e-5, 1e-5}})
	{
		INFO("double_precision=" << prec.fp64);
		gpulocres::LocresInitData in;
		in.I1 = MULTIDIM_ARRAY(I1);
		in.I2 = MULTIDIM_ARRAY(I2);
		in.I1p = MULTIDIM_ARRAY(I1p);
		in.I2p = MULTIDIM_ARRAY(I2p);
		in.FTsum_yxz = (const RFLOAT*)FTsum_yxz.data();
		in.mask_tmpl = mask_tmpl.data();
		in.r2_to_shell = r2_to_shell.data();
		in.nx = nx; in.ny = ny; in.nz = nz; in.nfx = nfx; in.S = S;
		in.double_precision = prec.fp64;

		gpulocres::LocresGpuSession sess;
		std::string reason;
		REQUIRE(sess.init(0, in, reason));
		std::vector<double> stats((size_t)6 * nfx);
		for (int ip = 0; ip < 2; ip++)
		{
			const long int x = pts[ip].x, y = pts[ip].y, z = pts[ip].z;
			const long int z0 = std::max(sz0, z - R), z1 = std::min(FINISHINGZ(I1), z + R);
			const long int y0 = std::max(sy0, y - R), y1 = std::min(FINISHINGY(I1), y + R);
			const long int x0 = std::max(sx0, x - R), x1 = std::min(FINISHINGX(I1), x + R);
			gpulocres::LocresPointBox box;
			box.kz0 = z0 - sz0; box.cz = z1 - z0 + 1;
			box.ky0 = y0 - sy0; box.cy = y1 - y0 + 1;
			box.kx0 = x0 - sx0; box.cx = x1 - x0 + 1;
			box.dm_z0 = z0 - z + R; box.dm_y0 = y0 - y + R; box.dm_x0 = x0 - x + R;

			REQUIRE(sess.computePointFsc(box, stats.data(), reason));
			for (long int i = 0; i < nfx; i++)
			{
				const double fsc_m = stats[i] / sqrt(stats[nfx + i] * stats[2 * nfx + i]);
				const double fsc_r = stats[3 * nfx + i] / sqrt(stats[4 * nfx + i] * stats[5 * nfx + i]);
				CHECK(fabs(fsc_m - cpu_fscs[ip][i]) <= prec.tol_fsc);
				CHECK(fabs(fsc_r - cpu_fscs[ip][nfx + i]) <= prec.tol_fsc);
			}
			REQUIRE(sess.accumFilteredPoint(box, r2_bound, n_fx_inv, inv_w.data(), pts[ip].inv_res, reason));
		}

		MultidimArray<RFLOAT> gpu_Ifil(nz, ny, nx), gpu_Ilocres(nz, ny, nx), gpu_Isumw(nz, ny, nx);
		REQUIRE(sess.finishVolumes(MULTIDIM_ARRAY(gpu_Ifil), MULTIDIM_ARRAY(gpu_Ilocres), MULTIDIM_ARRAY(gpu_Isumw), reason));

		auto checkRelL2 = [&](const MultidimArray<RFLOAT> &a, const MultidimArray<RFLOAT> &b) {
			double sd = 0., sb = 0.;
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a)
			{
				const double d = DIRECT_MULTIDIM_ELEM(a, n) - DIRECT_MULTIDIM_ELEM(b, n);
				sd += d * d;
				sb += DIRECT_MULTIDIM_ELEM(b, n) * DIRECT_MULTIDIM_ELEM(b, n);
			}
			REQUIRE(sb > 0.);
			CHECK(sqrt(sd / sb) <= prec.tol_vol);
		};
		checkRelL2(gpu_Ifil, cpu_Ifil);
		checkRelL2(gpu_Ilocres, cpu_Ilocres);
		checkRelL2(gpu_Isumw, cpu_Isumw);
	}
}
