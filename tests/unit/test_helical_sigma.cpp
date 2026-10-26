/*
 * tests/unit/test_helical_sigma.cpp
 *
 * The distance over which helical segments are averaged along their filament is a length in
 * Angstroms (the persistence length of the filament): it must not depend on the box size or the
 * pixel size.
 */

#include <catch2/catch.hpp>

#include <cmath>
#include <vector>

#include "src/helix.h"
#include "src/ml_optimiser.h"

// 41 segments 5 A apart (0..200 A) on one filament, tilt 90 up to 97.5 A and 100 from 100 A on
static std::vector<RFLOAT> tiltPriorsOfStep(RFLOAT sigma_A)
{
	std::vector<HelicalSegmentPriorInfoEntry> list(41);
	for (int i = 0; i < 41; i++)
	{
		HelicalSegmentPriorInfoEntry &s = list[i];
		s.helical_tube_name = "tube";
		s.MDobjectID = i;
		s.track_pos_A = 5. * i;
		s.tilt_deg = s.tilt_prior_deg = (s.track_pos_A < 99.) ? 90. : 100.;
		s.psi_deg = s.psi_prior_deg = 0.;
		s.rot_deg = s.rot_prior_deg = 0.;
		s.classID = 1;
		s.psi_flip_ratio = UNIMODAL_PSI_PRIOR_FLIP_RATIO;
	}
	int nr_wrong = 0;
	bool reverse = false;
	updatePriorsForOneHelicalTube(list, 0, 40, nr_wrong, reverse, sigma_A,
			std::vector<RFLOAT>(1, 4.75), std::vector<RFLOAT>(1, 0.), 1,
			false, false, 0., 0., 0., 0.);
	std::vector<RFLOAT> out;
	for (const auto &s : list)
		out.push_back(s.tilt_prior_deg);
	return out;
}

TEST_CASE("Helical averaging: sigma is a length along the filament in Angstroms",
          "[helical][helical_sigma]")
{
	// sigma 30 A: neighbours up to 90 A away count. The segment at 60 A (40 A below the step) is
	// pulled up a little, the one at 95 A (5 A below) about halfway, the one at 0 A not at all.
	std::vector<RFLOAT> t30 = tiltPriorsOfStep(30.);
	REQUIRE(t30[0] == Approx(90.).margin(1e-6));        // the step is 100 A away: beyond 3 sigma
	REQUIRE(t30[12] > 90.1);
	REQUIRE(t30[12] < 92.);
	REQUIRE(t30[19] > 93.5);
	REQUIRE(t30[19] < 96.5);                            // 5 A below the step
	REQUIRE(t30[20] > 93.5);                            // 5 A above the step
	REQUIRE(t30[20] < 96.5);
	REQUIRE(t30[40] == Approx(100.).margin(1e-6));

	// sigma 10 A: neighbours up to 30 A away. 40 A below the step nothing changes, 5 A below it does.
	std::vector<RFLOAT> t10 = tiltPriorsOfStep(10.);
	REQUIRE(t10[12] == Approx(90.).margin(1e-6));
	REQUIRE(t10[19] > 92.);
	REQUIRE(t10[19] < 98.);
	// a wider sigma pulls the segment 40 A below the step up more than a narrower one
	REQUIRE(t30[12] > t10[12]);
}

TEST_CASE("Helical averaging: the older fraction-of-box option converts with box and pixel size",
          "[helical][helical_sigma]")
{
	MlOptimiser o;
	o.mymodel.ori_size = 200;
	o.mymodel.pixel_size = 1.5;

	REQUIRE(o.helicalSigmaDistanceAngst() < 0.);        // off by default
	o.helical_sigma_distance = 0.1;                      // 10% of a 300 A box
	REQUIRE(o.helicalSigmaDistanceAngst() == Approx(30.));
	o.helical_sigma_distance_A = 45.;                    // the length in Angstroms wins and ignores the box
	REQUIRE(o.helicalSigmaDistanceAngst() == Approx(45.));
	o.mymodel.pixel_size = 3.;
	REQUIRE(o.helicalSigmaDistanceAngst() == Approx(45.));
}
