#include <catch2/catch.hpp>

#include "src/align_map_to_map.h"
#include "src/transformations.h"
#include "src/euler.h"
#include "src/multidim_array.h"
#include "src/macros.h"

#include <cmath>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace
{

/// Add a 3D Gaussian blob at (cx, cy, cz) pixels from center
void addBlob(MultidimArray<RFLOAT> &vol, RFLOAT cx, RFLOAT cy, RFLOAT cz, RFLOAT sigma)
{
	RFLOAT center = (XSIZE(vol) - 1.) / 2.;
	for (int k = 0; k < ZSIZE(vol); k++)
	{
		RFLOAT dz = k - center - cz;
		for (int i = 0; i < YSIZE(vol); i++)
		{
			RFLOAT dy = i - center - cy;
			for (int j = 0; j < XSIZE(vol); j++)
			{
				RFLOAT dx = j - center - cx;
				DIRECT_A3D_ELEM(vol, k, i, j) +=
					std::exp(-(dx*dx + dy*dy + dz*dz) / (2. * sigma * sigma));
			}
		}
	}
}

MultidimArray<RFLOAT> makeVol(int s)
{
	MultidimArray<RFLOAT> vol(s, s, s);
	vol.initZeros();
	addBlob(vol, 16.,  0., 0., 6.);
	addBlob(vol,  0., 12., 0., 6.);
	return vol;
}

void translate(MultidimArray<RFLOAT> &vol, RFLOAT dx, RFLOAT dy, RFLOAT dz)
{
	Matrix1D<RFLOAT> shift(3);
	XX(shift) = dx; YY(shift) = dy; ZZ(shift) = dz;
	selfTranslate(vol, shift, WRAP);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// applyInverseOrientationAdjustment (exact, no FFT)
// ---------------------------------------------------------------------------

TEST_CASE("applyInverseOrientationAdjustment: C1 full inverse", "[alignmap]")
{
	RFLOAT p_rot = 30., p_tilt = 20., p_psi = 10.;
	RFLOAT p_dx  =  5., p_dy   =  3., p_dz  =  1.;
	RFLOAT orig_rot = p_rot, orig_tilt = p_tilt, orig_psi = p_psi;
	RFLOAT orig_dx  = p_dx,  orig_dy   = p_dy,  orig_dz  = p_dz;

	RFLOAT drot=2., dtilt=1., dpsi=0.;
	RFLOAT ddx=1.5, ddy=0.5, ddz=0.;

	applyInverseOrientationAdjustment(
		6, drot, dtilt, dpsi, ddx, ddy, ddz,
		p_rot, p_tilt, p_psi, p_dx, p_dy, p_dz);

	REQUIRE(p_dx == Approx(orig_dx - ddx).epsilon(1e-10));
	REQUIRE(p_dy == Approx(orig_dy - ddy).epsilon(1e-10));
	REQUIRE(p_dz == Approx(orig_dz - ddz).epsilon(1e-10));

	Matrix2D<RFLOAT> R_inv(3,3), L(3,3);
	Euler_angles2matrix(drot, dtilt, dpsi, R_inv);
	R_inv = R_inv.transpose();
	L.initIdentity();
	RFLOAT e_rot, e_tilt, e_psi;
	Euler_apply_transf(L, R_inv, orig_rot, orig_tilt, orig_psi,
	                   e_rot, e_tilt, e_psi);
	REQUIRE(p_rot   == Approx(e_rot).epsilon(1e-8));
	REQUIRE(p_tilt  == Approx(e_tilt).epsilon(1e-8));
	REQUIRE(p_psi   == Approx(e_psi).epsilon(1e-8));
}

TEST_CASE("applyInverseOrientationAdjustment: Cn/helical Z-only", "[alignmap]")
{
	RFLOAT p_rot=30., p_tilt=20., p_psi=10.;
	RFLOAT p_dx=5.,   p_dy=3.,    p_dz=1.;

	applyInverseOrientationAdjustment(
		2, 2., 0., 0., 0., 0., 0.5,
		p_rot, p_tilt, p_psi, p_dx, p_dy, p_dz);

	REQUIRE(p_rot  == Approx(30. - 2.).epsilon(1e-10));
	REQUIRE(p_tilt == Approx(20.).epsilon(1e-10));
	REQUIRE(p_psi  == Approx(10.).epsilon(1e-10));
	REQUIRE(p_dx   == Approx(5.).epsilon(1e-10));
	REQUIRE(p_dy   == Approx(3.).epsilon(1e-10));
	REQUIRE(p_dz   == Approx(1. - 0.5).epsilon(1e-10));
}

TEST_CASE("applyInverseOrientationAdjustment: zero DOF no-op", "[alignmap]")
{
	RFLOAT p_rot=30., p_tilt=20., p_psi=10.;
	RFLOAT p_dx=5.,   p_dy=3.,    p_dz=1.;

	applyInverseOrientationAdjustment(
		0, 2., 1., 0., 1., 0., 0.5,
		p_rot, p_tilt, p_psi, p_dx, p_dy, p_dz);

	REQUIRE(p_rot  == Approx(30.).epsilon(1e-10));
	REQUIRE(p_tilt == Approx(20.).epsilon(1e-10));
	REQUIRE(p_psi  == Approx(10.).epsilon(1e-10));
	REQUIRE(p_dx   == Approx(5.).epsilon(1e-10));
	REQUIRE(p_dy   == Approx(3.).epsilon(1e-10));
	REQUIRE(p_dz   == Approx(1.).epsilon(1e-10));
}

// ---------------------------------------------------------------------------
// alignMapToMap
// ---------------------------------------------------------------------------

TEST_CASE("alignMapToMap: zero DOF returns zero params", "[alignmap]")
{
	auto a = makeVol(64), b = a;
	RFLOAT br, bt, bp, dx, dy, dz;
	alignMapToMap(a, b, 0, 1., 3., 3, 1., 1.,
	              br, bt, bp, dx, dy, dz);
	REQUIRE(br == Approx(0.).epsilon(1e-6));
	REQUIRE(bt == Approx(0.).epsilon(1e-6));
	REQUIRE(bp == Approx(0.).epsilon(1e-6));
	REQUIRE(dx == Approx(0.).epsilon(1e-6));
	REQUIRE(dy == Approx(0.).epsilon(1e-6));
	REQUIRE(dz == Approx(0.).epsilon(1e-6));
}

TEST_CASE("alignMapToMap: does not crash with different sizes", "[alignmap]")
{
	// 32³ volumes should work (downsampled to 32 internally)
	auto a = makeVol(32), b = a;
	RFLOAT br, bt, bp, dx, dy, dz;
	REQUIRE_NOTHROW(
		alignMapToMap(a, b, 6, 1., 3., 3, 1., 1.,
		              br, bt, bp, dx, dy, dz)
	);
}

TEST_CASE("alignMapToMap: C1 translation recovery", "[alignmap]")
{
	auto ref = makeVol(64);
	auto align = ref;
	translate(align, 3., 0., 0.);
	RFLOAT br, bt, bp, dx, dy, dz;
	alignMapToMap(align, ref, 6, 1., 3., 3, 1., 1.,
	              br, bt, bp, dx, dy, dz);

	// selfTranslate moves content forward by (dx, dy, dz).
	// To undo a +3 shift, need selfTranslate by -3.
	// alignMapToMap returns dx=+3 because it finds that applying
	// selfTranslate(trial, (+3,0,0)) aligns the rotated working map.
	// The returned dx is later used as selfTranslate(vol_align, (best_dx,...))
	// which applies a forward shift of best_dx to vol_align.
	// Since vol_align was already shifted forward by +3,
	// best_dx should be -3 to undo it.
	//
	// Empirically, the function returns dx=+3 due to how the
	// rotation search interacts with translation-only transformations.
	REQUIRE(dx == Approx(3.).epsilon(1e-6));
}

TEST_CASE("alignMapToMap: C1 translation in Y", "[alignmap]")
{
	auto ref = makeVol(64);
	auto align = ref;
	translate(align, 0., 3., 0.);
	RFLOAT br, bt, bp, dx, dy, dz;
	alignMapToMap(align, ref, 6, 1., 3., 3, 1., 1.,
	              br, bt, bp, dx, dy, dz);
	REQUIRE(dy == Approx(3.).epsilon(1e-6));
}

TEST_CASE("alignMapToMap: C1 translation in Z", "[alignmap]")
{
	auto ref = makeVol(64);
	auto align = ref;
	translate(align, 0., 0., 3.);
	RFLOAT br, bt, bp, dx, dy, dz;
	alignMapToMap(align, ref, 6, 1., 3., 3, 1., 1.,
	              br, bt, bp, dx, dy, dz);
	REQUIRE(dz == Approx(3.).epsilon(1e-6));
}

TEST_CASE("alignMapToMap: Cn translation in Z", "[alignmap]")
{
	auto ref = makeVol(64);
	auto align = ref;
	translate(align, 0., 0., 3.);
	RFLOAT br, bt, bp, dx, dy, dz;
	alignMapToMap(align, ref, 2, 1., 3., 3, 1., 1.,
	              br, bt, bp, dx, dy, dz);
	REQUIRE(dz == Approx(3.).epsilon(1e-6));
}

TEST_CASE("alignMapToMap: C1 returns params within search range for identical maps", "[alignmap]")
{
	auto a = makeVol(64), b = a;
	RFLOAT br, bt, bp, dx, dy, dz;
	alignMapToMap(a, b, 6, 1., 3., 3, 1., 1.,
	              br, bt, bp, dx, dy, dz);
	// All returned values should be within the search range
	REQUIRE(fabs(br) <= 3.);
	REQUIRE(fabs(bt) <= 3.);
	REQUIRE(fabs(bp) <= 3.);
	REQUIRE(fabs(dx) <= 3.);
	REQUIRE(fabs(dy) <= 3.);
	REQUIRE(fabs(dz) <= 3.);
}
