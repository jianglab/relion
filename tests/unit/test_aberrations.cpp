/*
 * tests/unit/test_aberrations.cpp
 *
 * Unit tests for high-order even and odd Zernike aberration support in the
 * single-particle CTF refinement / correction pipeline.
 *
 * Architecture being tested:
 *   ESTIMATION
 *     Even (symmetric):  AberrationEstimator -> TiltHelper::fitEvenZernike /
 *                         optimiseEvenZernike -> EMDL_IMAGE_EVEN_ZERNIKE_COEFFS
 *     Odd  (antisymm):   TiltEstimator       -> TiltHelper::fitOddZernike /
 *                         optimiseOddZernike  -> EMDL_IMAGE_ODD_ZERNIKE_COEFFS
 *                         (beam-tilt only for --odd_aberr_max_n < 3)
 *
 *   CORRECTION
 *     Even: ObservationModel::getGammaOffset()   -> CTF::getFftwImage() adds
 *           the per-pixel Zernike phase to gamma before computing -sin(gamma).
 *     Odd:  ObservationModel::getPhaseCorrection() -> multiplicative complex
 *           phase e^{i*phi} applied in predictObservation*().
 *     Odd (inverse, on observations):
 *           ObservationModel::demodulatePhase() multiplies by e^{-i*phi}.
 *
 * Known bugs fixed (tested by regression cases below):
 *   1. CTF::getFftwImage with even Zernike + CTF padding crashed because the
 *      box-size check compared native box size vs padded size.
 *   2. getGammaOffset / getPhaseCorrection used boxSizes[og] (native) instead
 *      of s (requested) for spatial-frequency normalisation, giving wrong
 *      Zernike values at padded sizes.
 */

#include <catch2/catch.hpp>

#include "src/jaz/single_particle/obs_model.h"
#include "src/jaz/optics/aberrations_cache.h"
#include "src/jaz/math/Zernike.h"
#include "src/ctf.h"
#include "src/metadata_table.h"
#include "src/metadata_label.h"
#include "src/matrix2d.h"
#include "src/macros.h"

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

namespace
{

// Build a minimal single-group optics MetaDataTable suitable for constructing
// an ObservationModel.
MetaDataTable makeOpticsTable(
		const std::vector<double>& evenCoeffs,
		const std::vector<double>& oddCoeffs,
		int boxSize   = 16,
		double angpix = 1.0)
{
	MetaDataTable mdt;
	mdt.setName("optics");
	mdt.addObject();

	mdt.setValue(EMDL_IMAGE_PIXEL_SIZE, angpix,   0);
	mdt.setValue(EMDL_CTF_VOLTAGE,     300.0,     0);
	mdt.setValue(EMDL_CTF_CS,          2.7,       0);
	mdt.setValue(EMDL_CTF_Q0,          0.1,       0);
	mdt.setValue(EMDL_IMAGE_SIZE,      boxSize,   0);

	if (!evenCoeffs.empty())
		mdt.setValue(EMDL_IMAGE_EVEN_ZERNIKE_COEFFS, evenCoeffs, 0);
	if (!oddCoeffs.empty())
		mdt.setValue(EMDL_IMAGE_ODD_ZERNIKE_COEFFS,  oddCoeffs,  0);

	// Identity mag matrix
	mdt.setValue(EMDL_IMAGE_MAG_MATRIX_00, 1.0, 0);
	mdt.setValue(EMDL_IMAGE_MAG_MATRIX_01, 0.0, 0);
	mdt.setValue(EMDL_IMAGE_MAG_MATRIX_10, 0.0, 0);
	mdt.setValue(EMDL_IMAGE_MAG_MATRIX_11, 1.0, 0);

	return mdt;
}

// Expected gammaOffset at pixel (px, py) in a box of size s with pixel size angpix.
// Matches the implementation in ObservationModel::getGammaOffset.
double expectedGammaOffset(const std::vector<double>& coeffs,
                            int px, int py, int s, double angpix)
{
	const int sh = s / 2 + 1;
	const double as = angpix * s;                     // corrected normalization
	const double xx = px / as;
	const double yy = (py < sh - 1) ? py / as : (py - s) / as;

	double phase = 0.0;
	for (int i = 0; i < (int)coeffs.size(); i++)
	{
		int m, n;
		Zernike::evenIndexToMN(i, m, n);
		phase += coeffs[i] * Zernike::Z_cart(m, n, xx, yy);
	}
	return phase;
}

// Expected phaseCorrection.real at pixel (px, py).
double expectedPhaseReal(const std::vector<double>& coeffs,
                          int px, int py, int s, double angpix)
{
	const int sh = s / 2 + 1;
	const double as = angpix * s;
	const double xx = px / as;
	const double yy = (py < sh - 1) ? py / as : (py - s) / as;

	double phase = 0.0;
	for (int i = 0; i < (int)coeffs.size(); i++)
	{
		int m, n;
		Zernike::oddIndexToMN(i, m, n);
		phase += coeffs[i] * Zernike::Z_cart(m, n, xx, yy);
	}
	return std::cos(phase);
}

double expectedPhaseImag(const std::vector<double>& coeffs,
                          int px, int py, int s, double angpix)
{
	const int sh = s / 2 + 1;
	const double as = angpix * s;
	const double xx = px / as;
	const double yy = (py < sh - 1) ? py / as : (py - s) / as;

	double phase = 0.0;
	for (int i = 0; i < (int)coeffs.size(); i++)
	{
		int m, n;
		Zernike::oddIndexToMN(i, m, n);
		phase += coeffs[i] * Zernike::Z_cart(m, n, xx, yy);
	}
	return std::sin(phase);
}

} // anonymous namespace

// ===========================================================================
// Suite 1: Zernike index / count helper functions
// ===========================================================================

TEST_CASE("Zernike index helpers", "[aberrations]")
{
	SECTION("numberOfEvenCoeffs matches index range for n_max=2,4,6")
	{
		// n_max=2: n=0,2 → orders 0 and 2 → 4 coefficients (l=(2/2)=1, count=1+2+1=4)
		CHECK(Zernike::numberOfEvenCoeffs(2) == 4);
		// n_max=4: count = 9
		CHECK(Zernike::numberOfEvenCoeffs(4) == 9);
		// n_max=6: count = 16
		CHECK(Zernike::numberOfEvenCoeffs(6) == 16);
	}

	SECTION("numberOfOddCoeffs matches index range for n_max=1,3,5")
	{
		// n_max=1: only n=1 terms → 2 coefficients
		CHECK(Zernike::numberOfOddCoeffs(1) == 2);
		// n_max=3: n=1 and n=3 → 6 coefficients
		CHECK(Zernike::numberOfOddCoeffs(3) == 6);
		// n_max=5: 12 coefficients
		CHECK(Zernike::numberOfOddCoeffs(5) == 12);
	}

	SECTION("evenIndexToMN: index 0 → (m=0, n=0) piston")
	{
		int m, n;
		Zernike::evenIndexToMN(0, m, n);
		CHECK(m == 0);
		CHECK(n == 0);
	}

	SECTION("evenIndexToMN: index 3 → (m=2, n=2)")
	{
		// k=1: m=2*(3-1-1)=2, n=2
		int m, n;
		Zernike::evenIndexToMN(3, m, n);
		CHECK(n == 2);
		CHECK(m == 2);
	}

	SECTION("oddIndexToMN: index 0 → (m=-1, n=1) tilt-x")
	{
		int m, n;
		Zernike::oddIndexToMN(0, m, n);
		CHECK(n == 1);
		CHECK(m == -1);
	}

	SECTION("oddIndexToMN: index 1 → (m=1, n=1) tilt-y")
	{
		int m, n;
		Zernike::oddIndexToMN(1, m, n);
		CHECK(n == 1);
		CHECK(m == 1);
	}
}

// ===========================================================================
// Suite 2: ObservationModel loads even and odd coefficients from optics table
// ===========================================================================

TEST_CASE("ObservationModel loads Zernike coefficients from optics metadata", "[aberrations]")
{
	const int s       = 16;
	const double apix = 1.5;

	SECTION("hasEvenZernike is true when EVEN_ZERNIKE_COEFFS label present")
	{
		std::vector<double> even(4, 0.0);
		even[0] = 0.5;
		MetaDataTable mdt = makeOpticsTable(even, {}, s, apix);
		ObservationModel obs(mdt, false);
		CHECK(obs.hasEvenZernike == true);
	}

	SECTION("hasOddZernike is true when ODD_ZERNIKE_COEFFS label present")
	{
		std::vector<double> odd(2, 0.0);
		odd[1] = -0.3;
		MetaDataTable mdt = makeOpticsTable({}, odd, s, apix);
		ObservationModel obs(mdt, false);
		CHECK(obs.hasOddZernike == true);
	}

	SECTION("hasEvenZernike false when label absent")
	{
		MetaDataTable mdt = makeOpticsTable({}, {}, s, apix);
		ObservationModel obs(mdt, false);
		CHECK(obs.hasEvenZernike == false);
	}

	SECTION("hasOddZernike false when label absent (and no beam-tilt)")
	{
		MetaDataTable mdt = makeOpticsTable({}, {}, s, apix);
		ObservationModel obs(mdt, false);
		CHECK(obs.hasOddZernike == false);
	}
}

// ===========================================================================
// Suite 3: getGammaOffset — even (symmetric) Zernike per-pixel phase offset
// ===========================================================================

TEST_CASE("ObservationModel::getGammaOffset — even Zernike correction", "[aberrations]")
{
	const int s       = 16;
	const double apix = 1.0;

	SECTION("All-zero even coefficients → gammaOffset is identically zero")
	{
		std::vector<double> even(4, 0.0);
		ObservationModel obs(makeOpticsTable(even, {}, s, apix), false);
		const BufferedImage<RFLOAT>& g = obs.getGammaOffset(0, s);

		bool allZero = true;
		for (int y = 0; y < s; y++)
		for (int x = 0; x < s/2+1; x++)
			if (std::fabs(g(x, y)) > 1e-15) { allZero = false; break; }
		CHECK(allZero);
	}

	SECTION("Single piston coefficient (index 0) → uniform non-zero gammaOffset")
	{
		// Zernike Z_00 (piston) is 1 everywhere: gammaOffset = coeff[0]
		std::vector<double> even(1, 0.0);
		even[0] = 2.0;
		ObservationModel obs(makeOpticsTable(even, {}, s, apix), false);
		const BufferedImage<RFLOAT>& g = obs.getGammaOffset(0, s);

		int m0, n0;
		Zernike::evenIndexToMN(0, m0, n0);
		// Piston Z(0,0) = 1 at (0,0) → offset == coeff
		CHECK(g(0, 0) == Approx(even[0] * Zernike::Z_cart(m0, n0, 0.0, 0.0)).epsilon(1e-12));
	}

	SECTION("Non-zero coefficients: specific pixel matches analytic Zernike sum")
	{
		// Use 4 even Zernike terms (n_max=2) with known values
		std::vector<double> even(4, 0.0);
		even[1] = 0.5;   // index 1 → m=-2, n=2
		even[3] = -0.3;  // index 3 → m=+2, n=2

		ObservationModel obs(makeOpticsTable(even, {}, s, apix), false);
		const BufferedImage<RFLOAT>& g = obs.getGammaOffset(0, s);

		// Check several interior pixels
		for (int px : {1, 3, 5})
		for (int py : {0, 2, 7})
		{
			const double expected = expectedGammaOffset(even, px, py, s, apix);
			CHECK(g(px, py) == Approx(expected).epsilon(1e-12));
		}
	}

	SECTION("gammaOffset at padded size (2x) uses correct spatial frequency normalisation")
	{
		// Bug fix regression: before the fix getGammaOffset used boxSizes instead
		// of s, giving 2× the correct spatial frequency for padded boxes.
		std::vector<double> even(4, 0.0);
		even[2] = 1.0;  // index 2 → m=0, n=2 (defocus-like)

		ObservationModel obs(makeOpticsTable(even, {}, s, apix), false);

		const int s_pad = 2 * s;
		const BufferedImage<RFLOAT>& g_native = obs.getGammaOffset(0, s);
		const BufferedImage<RFLOAT>& g_padded = obs.getGammaOffset(0, s_pad);

		// At pixel (1, 0) in the padded box, the physical spatial frequency is
		// half that of pixel (1, 0) in the native box. Verify independently.
		for (int px : {1, 2})
		{
			const double val_native = expectedGammaOffset(even, px, 0, s,     apix);
			const double val_padded = expectedGammaOffset(even, px, 0, s_pad, apix);

			CHECK(g_native(px, 0) == Approx(val_native).epsilon(1e-11));
			CHECK(g_padded(px, 0) == Approx(val_padded).epsilon(1e-11));

			// They should differ because the spatial frequency differs
			// (same pixel index but different physical frequency)
			CHECK_FALSE(std::fabs(val_native - val_padded) < 1e-15);
		}
	}
}

// ===========================================================================
// Suite 4: getPhaseCorrection — odd (antisymmetric) Zernike per-pixel e^{iφ}
// ===========================================================================

TEST_CASE("ObservationModel::getPhaseCorrection — odd Zernike correction", "[aberrations]")
{
	const int s       = 16;
	const double apix = 1.0;

	SECTION("All-zero odd coefficients → phaseCorrection is identically (1,0)")
	{
		std::vector<double> odd(6, 0.0);
		ObservationModel obs(makeOpticsTable({}, odd, s, apix), false);
		const BufferedImage<Complex>& c = obs.getPhaseCorrection(0, s);

		bool allUnit = true;
		for (int y = 0; y < s; y++)
		for (int x = 0; x < s/2+1; x++)
		{
			if (std::fabs(c(x,y).real - 1.0) > 1e-14 ||
			    std::fabs(c(x,y).imag)        > 1e-14)
			{ allUnit = false; break; }
		}
		CHECK(allUnit);
	}

	SECTION("phaseCorrection has unit modulus everywhere (it is a pure phase)")
	{
		std::vector<double> odd = {0.0, 0.4, -0.2, 0.1, 0.0, 0.3};
		ObservationModel obs(makeOpticsTable({}, odd, s, apix), false);
		const BufferedImage<Complex>& c = obs.getPhaseCorrection(0, s);

		for (int y = 0; y < s; y++)
		for (int x = 0; x < s/2+1; x++)
		{
			const double mod2 = c(x,y).real * c(x,y).real + c(x,y).imag * c(x,y).imag;
			CHECK(mod2 == Approx(1.0).epsilon(1e-12));
		}
	}

	SECTION("phaseCorrection.real and .imag match expected Zernike cos/sin")
	{
		std::vector<double> odd(6, 0.0);
		odd[0] =  0.8;  // tilt-x  (m=-1, n=1)
		odd[1] = -0.6;  // tilt-y  (m=+1, n=1)
		odd[4] =  0.2;  // 3rd order odd (coma etc.)

		ObservationModel obs(makeOpticsTable({}, odd, s, apix), false);
		const BufferedImage<Complex>& c = obs.getPhaseCorrection(0, s);

		for (int px : {0, 1, 3})
		for (int py : {0, 2, 9})
		{
			CHECK(c(px,py).real == Approx(expectedPhaseReal(odd, px, py, s, apix)).epsilon(1e-12));
			CHECK(c(px,py).imag == Approx(expectedPhaseImag(odd, px, py, s, apix)).epsilon(1e-12));
		}
	}
}

// ===========================================================================
// Suite 5: demodulatePhase — inverse odd Zernike correction on observations
// ===========================================================================

TEST_CASE("ObservationModel::demodulatePhase — inverse phase correction", "[aberrations]")
{
	const int s       = 8;
	const double apix = 1.0;
	const int sh      = s / 2 + 1;

	std::vector<double> odd(2, 0.0);
	odd[0] =  0.5;  // tilt-x
	odd[1] = -0.3;  // tilt-y

	ObservationModel obs(makeOpticsTable({}, odd, s, apix), false);

	// Build an arbitrary complex image
	MultidimArray<Complex> img(s, sh);
	for (int y = 0; y < s;  y++)
	for (int x = 0; x < sh; x++)
		img(y, x) = Complex(1.0, 0.0);

	// Apply forward phase correction (modulate) then demodulate — should recover original
	obs.demodulatePhase(0, img, /*do_modulate_instead=*/true);   // applies e^{+i*phi}
	obs.demodulatePhase(0, img, /*do_modulate_instead=*/false);  // applies e^{-i*phi}

	for (int y = 0; y < s;  y++)
	for (int x = 0; x < sh; x++)
	{
		CHECK(img(y, x).real == Approx(1.0).epsilon(1e-12));
		CHECK(img(y, x).imag == Approx(0.0).margin(1e-12));
	}
}

// ===========================================================================
// Suite 6: CTF::getFftwImage with even Zernike — native and padded sizes
// ===========================================================================

TEST_CASE("CTF::getFftwImage applies even Zernike gammaOffset correctly", "[aberrations]")
{
	const int s       = 16;
	const double apix = 1.0;

	// Even Zernike: piston (index 0) and defocus-like (index 2)
	std::vector<double> even(4, 0.0);
	even[0] = 0.1;  // piston
	even[2] = 0.3;  // m=0, n=2

	MetaDataTable mdt = makeOpticsTable(even, {}, s, apix);
	ObservationModel obs(mdt, false);

	// Construct a CTF using setValuesByGroup
	CTF ctf;
	ctf.setValuesByGroup(&obs, 0,
	                     /*defU=*/1.5e4, /*defV=*/1.5e4, /*defAng=*/0.0,
	                     /*Bfac=*/0.0,  /*scale=*/1.0,  /*phase_shift=*/0.0,
	                     /*dose=*/0.0);

	SECTION("getFftwImage at native size does not throw")
	{
		MultidimArray<RFLOAT> result(s, s/2+1);
		CHECK_NOTHROW(
			ctf.getFftwImage(result, s, s, apix,
			                 false, false, false, true, false)
		);
	}

	SECTION("getFftwImage at 2x padded size does not throw (regression for padding bug)")
	{
		const int s2 = 2 * s;
		MultidimArray<RFLOAT> result(s2, s2/2+1);
		CHECK_NOTHROW(
			ctf.getFftwImage(result, s2, s2, apix,
			                 false, false, false, true, false)
		);
	}

	SECTION("gammaOffset shifts CTF values relative to no-aberration baseline")
	{
		// CTF without aberrations (no obsModel)
		CTF ctf_ref;
		ctf_ref.setValues(1.5e4, 1.5e4, 0.0, 300.0, 2.7, 0.0, 1.0, 0.1, 0.0, 0.0);

		MultidimArray<RFLOAT> res_ref(s, s/2+1);
		ctf_ref.getFftwImage(res_ref, s, s, apix, false, false, false, true, false);

		MultidimArray<RFLOAT> res_aberr(s, s/2+1);
		ctf.getFftwImage(res_aberr, s, s, apix, false, false, false, true, false);

		// At pixel (0,0) the Zernike piston shifts gamma: values should differ
		bool anyDiffers = false;
		for (int y = 0; y < s; y++)
		for (int x = 0; x < s/2+1; x++)
		{
			if (std::fabs(DIRECT_A2D_ELEM(res_ref, y, x) -
			              DIRECT_A2D_ELEM(res_aberr, y, x)) > 1e-10)
			{
				anyDiffers = true;
				break;
			}
		}
		CHECK(anyDiffers);
	}
}

// ===========================================================================
// Suite 7: AberrationsCache (tomography) loads both even and odd coefficients
// ===========================================================================

TEST_CASE("AberrationsCache loads even (symmetrical) and odd (phaseShift) Zernike coefficients", "[aberrations]")
{
	const int s       = 16;
	const double apix = 1.0;

	std::vector<double> even(4, 0.0);
	even[1] =  0.7;  // m=-2, n=2

	std::vector<double> odd(6, 0.0);
	odd[2] = -0.4;   // 3rd-order odd

	MetaDataTable mdt = makeOpticsTable(even, odd, s, apix);
	AberrationsCache cache(mdt, s, apix);

	SECTION("symmetrical (even Zernike) cache is non-empty and contains correct values")
	{
		REQUIRE(cache.symmetrical.size() > 0);
		const BufferedImage<RFLOAT>& symm = cache.symmetrical[0];

		for (int px : {1, 3})
		for (int py : {0, 5})
		{
			const double expected = expectedGammaOffset(even, px, py, s, apix);
			CHECK(symm(px, py) == Approx(expected).epsilon(1e-11));
		}
	}

	SECTION("phaseShift (odd Zernike) cache is non-empty and has unit modulus")
	{
		REQUIRE(cache.phaseShift.size() > 0);
		const BufferedImage<fComplex>& ps = cache.phaseShift[0];

		for (int y = 0; y < s; y++)
		for (int x = 0; x < s/2+1; x++)
		{
			const double mod2 = (double)ps(x,y).real * ps(x,y).real + (double)ps(x,y).imag * ps(x,y).imag;
			// fComplex is float-precision; cos²+sin² is accurate to ~1e-7
			CHECK(mod2 == Approx(1.0).epsilon(1e-6));
		}
	}

	SECTION("phaseShift cache matches ObservationModel::getPhaseCorrection")
	{
		ObservationModel obs(mdt, false);
		const BufferedImage<Complex>& fromObs   = obs.getPhaseCorrection(0, s);
		const BufferedImage<fComplex>& fromCache = cache.phaseShift[0];

		for (int y = 0; y < s; y++)
		for (int x = 0; x < s/2+1; x++)
		{
			// fComplex is float, so use float-level tolerance
			CHECK(fromCache(x,y).real == Approx(fromObs(x,y).real).epsilon(1e-6));
			CHECK(fromCache(x,y).imag == Approx(fromObs(x,y).imag).epsilon(1e-6));
		}
	}
}
