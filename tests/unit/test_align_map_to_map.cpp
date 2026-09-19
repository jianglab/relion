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

/// Rotate about Z by `deg`, in the same sense as the angles alignMapToMap
/// returns (i.e. rotate the object BY the matrix, hence inv = false).
void rotateZ(MultidimArray<RFLOAT> &vol, RFLOAT deg)
{
	Matrix2D<RFLOAT> R;
	Euler_rotation3DMatrix(deg, 0., 0., R);
	MultidimArray<RFLOAT> tmp = vol;
	applyGeometry(tmp, vol, R, false, false, 0.);
}

/// Pearson correlation, used to check that a map really did end up aligned -
/// which is the actual contract, independent of how the parameters are signed.
double corr(const MultidimArray<RFLOAT> &a, const MultidimArray<RFLOAT> &b)
{
	double sa = 0., sb = 0.;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a) { sa += DIRECT_MULTIDIM_ELEM(a, n); }
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(b) { sb += DIRECT_MULTIDIM_ELEM(b, n); }
	const double ma = sa / MULTIDIM_SIZE(a), mb = sb / MULTIDIM_SIZE(b);

	double num = 0., da = 0., db = 0.;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a)
	{
		const double x = DIRECT_MULTIDIM_ELEM(a, n) - ma;
		const double y = DIRECT_MULTIDIM_ELEM(b, n) - mb;
		num += x * y; da += x * x; db += y * y;
	}
	return num / std::sqrt(da * db);
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
	const double before = corr(align, ref);

	RFLOAT br, bt, bp, dx, dy, dz;
	alignMapToMap(align, ref, 6, 1., 3., 3, 1., 1.,
	              br, bt, bp, dx, dy, dz);

	// The returned parameters are the transformation that IS APPLIED to the
	// map, so undoing a +3 shift means returning -3 - and alignMapToMap has
	// already applied it, which is what the correlation check confirms.
	REQUIRE(dx == Approx(-3.).margin(0.5));
	REQUIRE(corr(align, ref) > 0.99);
	REQUIRE(corr(align, ref) > before);
}

TEST_CASE("alignMapToMap: C1 translation in Y", "[alignmap]")
{
	auto ref = makeVol(64);
	auto align = ref;
	translate(align, 0., 3., 0.);
	const double before = corr(align, ref);

	RFLOAT br, bt, bp, dx, dy, dz;
	alignMapToMap(align, ref, 6, 1., 3., 3, 1., 1.,
	              br, bt, bp, dx, dy, dz);
	REQUIRE(dy == Approx(-3.).margin(0.5));
	REQUIRE(corr(align, ref) > 0.99);
	REQUIRE(corr(align, ref) > before);
}

TEST_CASE("alignMapToMap: C1 translation in Z", "[alignmap]")
{
	auto ref = makeVol(64);
	auto align = ref;
	translate(align, 0., 0., 3.);
	const double before = corr(align, ref);

	RFLOAT br, bt, bp, dx, dy, dz;
	alignMapToMap(align, ref, 6, 1., 3., 3, 1., 1.,
	              br, bt, bp, dx, dy, dz);
	REQUIRE(dz == Approx(-3.).margin(0.5));
	REQUIRE(corr(align, ref) > 0.99);
	REQUIRE(corr(align, ref) > before);
}

TEST_CASE("alignMapToMap: Cn translation in Z", "[alignmap]")
{
	auto ref = makeVol(64);
	auto align = ref;
	translate(align, 0., 0., 3.);
	const double before = corr(align, ref);

	RFLOAT br, bt, bp, dx, dy, dz;
	alignMapToMap(align, ref, 2, 1., 3., 3, 1., 1.,
	              br, bt, bp, dx, dy, dz);
	REQUIRE(dz == Approx(-3.).margin(0.5));
	REQUIRE(corr(align, ref) > 0.99);
	REQUIRE(corr(align, ref) > before);
}

// The absence of these two is why a completely non-functional rotation search
// went unnoticed: every existing test used a translation only.
TEST_CASE("alignMapToMap: Cn rotation recovery", "[alignmap]")
{
	auto ref = makeVol(64);
	auto align = ref;
	rotateZ(align, 2.);
	const double before = corr(align, ref);

	RFLOAT br, bt, bp, dx, dy, dz;
	alignMapToMap(align, ref, 2, 1., 3., 3, 1., 1.,
	              br, bt, bp, dx, dy, dz);

	REQUIRE(br == Approx(-2.).margin(0.6));
	REQUIRE(corr(align, ref) > 0.99);
	REQUIRE(corr(align, ref) > before);
}

TEST_CASE("alignMapToMap: C1 rotation recovery", "[alignmap]")
{
	auto ref = makeVol(64);
	auto align = ref;
	rotateZ(align, 2.);
	const double before = corr(align, ref);

	RFLOAT br, bt, bp, dx, dy, dz;
	alignMapToMap(align, ref, 6, 1., 3., 3, 1., 1.,
	              br, bt, bp, dx, dy, dz);

	// With tilt == 0 the ZYZ rot and psi are both rotations about Z, so only
	// their sum is determined; check the net rotation, not the split.
	REQUIRE(bt == Approx(0.).margin(0.6));
	REQUIRE(br + bp == Approx(-2.).margin(0.7));
	REQUIRE(corr(align, ref) > 0.99);
	REQUIRE(corr(align, ref) > before);
}

TEST_CASE("alignMapToMap: combined rotation and shift is undone", "[alignmap]")
{
	auto ref = makeVol(64);
	auto align = ref;
	rotateZ(align, 2.);
	translate(align, 2., 0., 0.);
	const double before = corr(align, ref);

	RFLOAT br, bt, bp, dx, dy, dz;
	alignMapToMap(align, ref, 6, 1., 3., 3, 1., 1.,
	              br, bt, bp, dx, dy, dz);

	REQUIRE(corr(align, ref) > 0.99);
	REQUIRE(corr(align, ref) > before);
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
