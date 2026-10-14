/*
 * tests/unit/test_rectangular_helix_mask.cpp
 *
 * Helical references of rectangular particles are cuboids, long along the
 * helical axis z. Their mask is an ellipsoid whose long axis is the mask
 * diameter (BoxEllipse in src/mask.h), so the helical symmetry keeps the whole
 * length of the cuboid instead of a sphere that fits the short side.
 */

#include <catch2/catch.hpp>

#include "src/helix.h"
#include "src/mask.h"

static bool checkCuboid(int box_z, int box_xy, RFLOAT particle_diameter, RFLOAT tube_diameter, RFLOAT z_percentage = 0.3)
{
	return checkParametersFor3DHelicalReconstruction(false, false, 1, 4.75, 4.75, 4.75, 1., 1., 1.,
	                                                 box_z, 1., z_percentage, particle_diameter, -1., tube_diameter,
	                                                 false, box_xy);
}

TEST_CASE("helical parameters: a cuboid has an ellipsoidal mask", "[rectangular][helix_mask]")
{
	// 32 across, 96 along z: a particle diameter of 88 is the long axis, 88 / 3 across
	REQUIRE(checkCuboid(96, 32, 88., 16.));
	// The tube does not fit across the ellipsoid (29.3) or the box
	REQUIRE_THROWS(checkCuboid(96, 32, 88., 30.));
	REQUIRE_THROWS(checkCuboid(96, 32, 60., 24.)); // 20 across
	// The particle diameter is limited by the long side, not the short one
	REQUIRE_THROWS(checkCuboid(96, 32, 100., 16.));
	// A cube behaves as before: the diameter must fit the box
	REQUIRE_THROWS(checkParametersFor3DHelicalReconstruction(false, false, 1, 4.75, 4.75, 4.75, 1., 1., 1.,
	                                                         32, 1., 0.3, 88., -1., 16.));
	// z_percentage may reach the height of the ellipsoid at the tube radius:
	// 2 * 44 * sqrt(1 - (8 / 14.67)^2) / 96 = 0.77
	REQUIRE(checkCuboid(96, 32, 88., 16., 0.75));
	REQUIRE_THROWS(checkCuboid(96, 32, 88., 16., 0.8));
}

TEST_CASE("imposeHelicalSymmetryInRealSpace: a cuboid keeps its length", "[rectangular][helix_mask]")
{
	// A filled tube of radius 5 in a 32 x 32 x 96 cuboid, no twist
	MultidimArray<RFLOAT> v(96, 32, 32);
	v.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY3D(v)
		A3D_ELEM(v, k, i, j) = (i * i + j * j <= 25) ? 1. : 0.;

	MultidimArray<RFLOAT> v_default = v;
	imposeHelicalSymmetryInRealSpace(v, 1., 44., -1., 8., 0.3, 4.75, 0., 3., true);

	// Along the axis the density reaches the ends of the ellipsoid (44 along z)...
	REQUIRE(A3D_ELEM(v, 40, 0, 0) == Approx(1.));
	REQUIRE(A3D_ELEM(v, -40, 2, 1) == Approx(1.));
	// ...goes through its soft edge, and is zero beyond it
	REQUIRE(A3D_ELEM(v, 45, 0, 0) > 0.);
	REQUIRE(A3D_ELEM(v, 45, 0, 0) < 1.);
	REQUIRE(A3D_ELEM(v, 47, 0, 0) == 0.);
	// Across, the tube mask (8 + 3) cuts it
	REQUIRE(A3D_ELEM(v, 0, 0, 12) == 0.);

	// By default the mask is a sphere that must fit the short side, as before
	MultidimArray<RFLOAT> w = v_default;
	REQUIRE_THROWS(imposeHelicalSymmetryInRealSpace(w, 1., 44., -1., 8., 0.3, 4.75, 0., 3.));
	imposeHelicalSymmetryInRealSpace(v_default, 1., 14., -1., 8., 0.3, 4.75, 0., 1.);
	REQUIRE(A3D_ELEM(v_default, 10, 0, 0) == Approx(1.));
	REQUIRE(A3D_ELEM(v_default, 20, 0, 0) == 0.);
}

TEST_CASE("createCylindricalReference: a cuboid as deep as the reference", "[rectangular][helix_mask]")
{
	MultidimArray<RFLOAT> v;
	createCylindricalReference(v, 32, 0., 16., 3., 96);
	REQUIRE(ZSIZE(v) == 96);
	REQUIRE(YSIZE(v) == 32);
	REQUIRE(XSIZE(v) == 32);
	REQUIRE(A3D_ELEM(v, 47, 0, 3) == 1.);
	REQUIRE(A3D_ELEM(v, -48, 3, 0) == 1.);
	REQUIRE(A3D_ELEM(v, -48, 0, 15) == 0.);

	createCylindricalReference(v, 32, 0., 16., 3.);
	REQUIRE(ZSIZE(v) == 32);
}
