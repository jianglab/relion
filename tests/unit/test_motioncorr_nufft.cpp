/*
 * tests/unit/test_motioncorr_nufft.cpp
 *
 * MotioncorrRunner::nufftLocalShifts (RELION_MOTIONCORR_INTERP=nufft): every pixel of the output is the frame's
 * Fourier series evaluated where the local motion model moves it, checked against a direct evaluation of that series;
 * and the sum recipe that records it (version 2, refused by builds that only know version 1).
 */

#include <catch2/catch.hpp>

#ifdef RELION_USE_FINUFFT

#include <cmath>
#include <random>
#include <sstream>

#include "src/motioncorr_runner.h"
#include <src/jaz/single_particle/new_ft.h>

namespace
{

// The real Fourier series of a frame at (x, y), from its forward-normalised half spectrum; a Nyquist bin counts as the
// average of +N/2 and -N/2 (cos), which keeps the series real
double seriesAt(const MultidimArray<fComplex> &F, int nx, int ny, double x, double y)
{
	double s = 0;
	for (int ky = -ny / 2; ky < ny / 2; ky++)
		for (int kx = -nx / 2; kx < nx / 2; kx++)
		{
			const bool neg = kx < 0;
			const fComplex v = DIRECT_A2D_ELEM(F, ((neg ? -ky : ky) + ny) % ny, neg ? -kx : kx);
			const double re = v.real, im = neg ? -v.imag : v.imag;
			// e^{2 pi i k x / n}, or cos at Nyquist
			const double ax = 2 * PI * kx * x / nx, ay = 2 * PI * ky * y / ny;
			const double cxr = std::cos(ax), cxi = (2 * kx == -nx) ? 0. : std::sin(ax);
			const double cyr = std::cos(ay), cyi = (2 * ky == -ny) ? 0. : std::sin(ay);
			const double er = cxr * cyr - cxi * cyi, ei = cxr * cyi + cxi * cyr;
			s += re * er - im * ei;
		}
	return s;
}

}

TEST_CASE("Exact local shifts match the Fourier series at the shifted positions", "[motioncorr_nufft]")
{
	const int nx = 40, ny = 36, n_frames = 3;
	std::mt19937 rng(7);
	std::normal_distribution<float> gauss;
	MultidimArray<float> img(ny, nx);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img) DIRECT_MULTIDIM_ELEM(img, n) = gauss(rng);
	MultidimArray<fComplex> F;
	NewFFT::FourierTransform(img, F);
	std::vector<MultidimArray<fComplex> > Fframes(n_frames, F);

	// a shift that varies over the frame and with z (frame 0 is not moved)
	ThirdOrderPolynomialModel model;
	model.coeffX.resize(18); model.coeffX.initZeros();
	model.coeffY.resize(18); model.coeffY.initZeros();
	model.coeffX(0) = 0.37; model.coeffX(3) = 0.8; model.coeffX(16) = 0.3;
	model.coeffY(0) = -0.21; model.coeffY(9) = 0.5; model.coeffY(7) = -0.4;

	MotioncorrRunner R;
	R.n_threads = 2;
	std::ostringstream quiet;
	std::vector<Image<float> > aligned(n_frames);
	R.nufftLocalShifts(Fframes, nx, ny, model, 5, NULL, &aligned, quiet);

	double rms = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img) rms += DIRECT_MULTIDIM_ELEM(img, n) * DIRECT_MULTIDIM_ELEM(img, n);
	rms = std::sqrt(rms / NZYXSIZE(img));

	double max_err = 0, max_shift = 0;
	int checked = 0;
	for (int f = 0; f < n_frames; f++)
		for (int iy = 0; iy < ny; iy++)
			for (int ix = 0; ix < nx; ix++)
			{
				RFLOAT sx, sy;
				model.getShiftAt(f, (RFLOAT)ix / nx - 0.5, (RFLOAT)iy / ny - 0.5, sx, sy);
				const double x = ix - sx, y = iy - sy;
				if (x < 0 || x > nx - 1 || y < 0 || y > ny - 1) continue; // the edge is clamped
				max_shift = std::max(max_shift, std::hypot(sx, sy));
				max_err = std::max(max_err, std::fabs(DIRECT_A2D_ELEM(aligned[f](), iy, ix) - seriesAt(F, nx, ny, x, y)));
				checked++;
			}
	INFO("max error " << max_err << " (image rms " << rms << "), " << checked << " pixels, shifts up to " << max_shift);
	CHECK(checked > 3000);
	CHECK(max_shift > 1.0);
	CHECK(max_err < 1e-3 * rms);

	// frame 0 has no shift: the frame itself
	double err0 = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img)
		err0 = std::max(err0, (double)std::fabs(DIRECT_MULTIDIM_ELEM(aligned[0](), n) - DIRECT_MULTIDIM_ELEM(img, n)));
	CHECK(err0 < 1e-3 * rms);

	// the sum is the aligned frames added in order, and the same bits with another thread count
	Image<float> sum1(nx, ny), sum4(nx, ny);
	sum1().initZeros(); sum4().initZeros();
	R.n_threads = 1;
	R.nufftLocalShifts(Fframes, nx, ny, model, 5, &sum1, NULL, quiet);
	R.n_threads = 4;
	R.nufftLocalShifts(Fframes, nx, ny, model, 5, &sum4, NULL, quiet);
	bool same = true, added = true;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(sum1())
	{
		float s = 0;
		for (int f = 0; f < n_frames; f++) s += DIRECT_MULTIDIM_ELEM(aligned[f](), n);
		same = same && DIRECT_MULTIDIM_ELEM(sum1(), n) == DIRECT_MULTIDIM_ELEM(sum4(), n);
		added = added && DIRECT_MULTIDIM_ELEM(sum1(), n) == s;
	}
	CHECK(same);
	CHECK(added);
}

TEST_CASE("The sum recipe records exact local shifts as version 2", "[motioncorr_nufft]")
{
	MotioncorrRunner::SumRecipe r, back;
	r.nx = 4096; r.ny = 4096;
	CHECK(r.str() == "version=1 dose_weighted=1 early_binning=0 fix_defects=1 float16=0 nx=4096 ny=4096");
	REQUIRE(MotioncorrRunner::SumRecipe::parse(r.str(), back));
	CHECK(back.local_nufft == 0);

	r.local_nufft = 5;
	CHECK(r.str() == "version=2 dose_weighted=1 early_binning=0 fix_defects=1 float16=0 nx=4096 ny=4096 local_nufft=5");
	REQUIRE(MotioncorrRunner::SumRecipe::parse(r.str(), back));
	CHECK(back.version == 2);
	CHECK(back.local_nufft == 5);

	// inconsistent records are refused
	CHECK_FALSE(MotioncorrRunner::SumRecipe::parse("version=2 dose_weighted=1 early_binning=0 fix_defects=1 float16=0 nx=8 ny=8", back));
	CHECK_FALSE(MotioncorrRunner::SumRecipe::parse("version=1 dose_weighted=1 early_binning=0 fix_defects=1 float16=0 nx=8 ny=8 local_nufft=5", back));
}

#endif // RELION_USE_FINUFFT
