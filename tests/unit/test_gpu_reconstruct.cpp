/*
 * tests/unit/test_gpu_reconstruct.cpp
 *
 * BackProjector::reconstruct() with skip_gridding on the GPU (RELION_GPU_RECONSTRUCT=1) must give the
 * CPU's map to rounding (cuFFT versus FFTW, and the per-shell weight sums added in another order).
 * Skipped when no GPU is visible.
 */

#include <catch2/catch.hpp>

#include "src/backprojector.h"
#include "src/gpu_reconstruct.h"
#include "src/helix.h"

#include <cstdlib>
#include <random>

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
	for (const Case &c : {Case{44., -1., 30., 0.3, 4.75, -1.2}, Case{40., 8., 26., 0.3, 4., 30.},
	                      Case{46., -1., 20., 0.5, 2.38, 179.4}})
	{
		INFO("twist " << c.twist << " rise " << c.rise_A);
		MultidimArray<RFLOAT> a(n, n, n);
		std::mt19937 rng(11);
		std::normal_distribution<RFLOAT> g(0., 1.);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a) DIRECT_MULTIDIM_ELEM(a, n) = g(rng);
		MultidimArray<RFLOAT> b = a;
		gpurec::setGpuRequested(false);
		imposeHelicalSymmetryInRealSpace(a, angpix, c.sphere_A, c.inner_A, c.outer_A, c.z_frac, c.rise_A, c.twist, 3., 4);
		setenv("RELION_GPU_RECONSTRUCT", "1", 1);
		gpurec::setGpuRequested(true);
		imposeHelicalSymmetryInRealSpace(b, angpix, c.sphere_A, c.inner_A, c.outer_A, c.z_frac, c.rise_A, c.twist, 3., 4);
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
