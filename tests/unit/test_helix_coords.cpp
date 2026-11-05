/*
 * tests/unit/test_helix_coords.cpp
 *
 * HelicalCoordsRotation (one rotation built per particle, applied to many shifts) must give
 * exactly the numbers of transformCartesianAndHelicalCoords().
 */

#include <catch2/catch.hpp>

#include "src/helix.h"

#include <cstring>
#include <random>

TEST_CASE("HelicalCoordsRotation equals transformCartesianAndHelicalCoords bit for bit",
          "[helix_coords]")
{
	std::mt19937 gen(7);
	std::uniform_real_distribution<double> ang(-180., 180.), pos(-30., 30.);
	for (int dim = 2; dim <= 3; dim++)
	for (int dir = 0; dir < 2; dir++)
	for (int trial = 0; trial < 50; trial++)
	{
		const double rot = ang(gen), tilt = ang(gen) / 2., psi = ang(gen);
		const bool direction = dir ? CART_TO_HELICAL_COORDS : HELICAL_TO_CART_COORDS;
		HelicalCoordsRotation r(rot, tilt, psi, dim, direction);
		for (int k = 0; k < 20; k++)
		{
			const double x = pos(gen), y = pos(gen), z = pos(gen);
			double ox = 1., oy = 2., oz = 3., nx = 1., ny = 2., nz = 3.;
			transformCartesianAndHelicalCoords(x, y, z, ox, oy, oz, rot, tilt, psi, dim, direction);
			r.apply(x, y, z, nx, ny, nz);
			REQUIRE(std::memcmp(&ox, &nx, sizeof(double)) == 0);
			REQUIRE(std::memcmp(&oy, &ny, sizeof(double)) == 0);
			REQUIRE(std::memcmp(&oz, &nz, sizeof(double)) == 0);
		}
	}
}

// One tube along direction_deg in the micrograph, its segments every 30 pixels
static void addTube(MetaDataTable &MD, const std::string &mic, int tube, RFLOAT direction_deg, int nr_segments,
                    RFLOAT prior, bool with_angle, RFLOAT extraction_angle)
{
	const RFLOAT c = cos(DEG2RAD(direction_deg)), s = sin(DEG2RAD(direction_deg));
	for (int i = 0; i < nr_segments; i++)
	{
		MD.addObject();
		MD.setValue(EMDL_MICROGRAPH_NAME, mic);
		MD.setValue(EMDL_PARTICLE_HELICAL_TUBE_ID, tube);
		MD.setValue(EMDL_IMAGE_COORD_X, (RFLOAT)(500. + 30. * i * c));
		MD.setValue(EMDL_IMAGE_COORD_Y, (RFLOAT)(500. + 30. * i * s));
		MD.setValue(EMDL_ORIENT_PSI_PRIOR, prior);
		if (with_angle)
			MD.setValue(EMDL_PARTICLE_EXTRACTION_ANGLE, extraction_angle);
	}
}

TEST_CASE("checkPsiPriorsAgainstTubeDirections: priors that follow the tubes pass", "[helix_coords]")
{
	MetaDataTable MD;
	addTube(MD, "m1.mrc", 1, 40., 5, -40., false, 0.);
	addTube(MD, "m1.mrc", 2, 40., 4, 140., false, 0.);  // the other polarity
	addTube(MD, "m2.mrc", 1, -75., 6, 77., false, 0.);  // 2 degrees off, and tube 1 again on another micrograph
	addTube(MD, "m2.mrc", 3, 10., 1, 90., false, 0.);   // alone in its tube: not checked
	PsiPriorTubeCheck check = checkPsiPriorsAgainstTubeDirections(MD);
	REQUIRE(check.nr_checked == 15);
	REQUIRE(check.nr_bad == 0);
}

TEST_CASE("checkPsiPriorsAgainstTubeDirections: turned boxes, with and without their extraction angle", "[helix_coords]")
{
	// A turned extraction of a tube with prior -40 turns its box by 40 and leaves a prior of 0
	MetaDataTable turned, lost;
	addTube(turned, "m1.mrc", 1, 40., 5, 0., true, 40.);
	addTube(turned, "m1.mrc", 2, 0., 3, 0., true, 0.);
	PsiPriorTubeCheck ok = checkPsiPriorsAgainstTubeDirections(turned);
	REQUIRE(ok.nr_checked == 8);
	REQUIRE(ok.nr_bad == 0);

	// The same particles once rlnParticleExtractionAngle has been dropped: the tube at 40 degrees is caught,
	// the horizontal one (prior 0 is right for it anyway) is not
	addTube(lost, "m1.mrc", 1, 40., 5, 0., false, 0.);
	addTube(lost, "m1.mrc", 2, 0., 3, 0., false, 0.);
	PsiPriorTubeCheck bad = checkPsiPriorsAgainstTubeDirections(lost);
	REQUIRE(bad.nr_checked == 8);
	REQUIRE(bad.nr_bad == 5);
	REQUIRE(bad.nr_bad_zero == 5);
}

TEST_CASE("checkPsiPriorsAgainstTubeDirections: no tube IDs, nothing checked", "[helix_coords]")
{
	MetaDataTable MD;
	MD.addObject();
	MD.setValue(EMDL_IMAGE_COORD_X, (RFLOAT)1.);
	MD.setValue(EMDL_IMAGE_COORD_Y, (RFLOAT)1.);
	MD.setValue(EMDL_ORIENT_PSI_PRIOR, (RFLOAT)0.);
	REQUIRE(checkPsiPriorsAgainstTubeDirections(MD).nr_checked == 0);
}
