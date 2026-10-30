/*
 * tests/unit/test_filament_groups.cpp
 *
 * The shared coarse pass for neighbouring helical segments is used only while the groups' orientation
 * sets overlap enough (own / union >= 2): global searches share, local searches around different priors
 * do not. One group in ShareProbe still runs shared to keep the ratio current.
 */

#include <catch2/catch.hpp>

#include "src/filament_groups.h"

TEST_CASE("shared group pass follows the overlap of the orientation sets", "[filament_groups]")
{
	filgroup::shareRatio().store(-1.);
	CHECK(filgroup::sharePays()); // nothing measured yet: share

	for (int i = 0; i < 40; i++)
		filgroup::shareRatioUpdate(1.2);
	CHECK(filgroup::shareRatio().load() < filgroup::ShareMin);
	int shared = 0;
	for (int i = 0; i < 2 * filgroup::ShareProbe; i++)
		shared += filgroup::sharePays() ? 1 : 0;
	CHECK(shared >= 1);
	CHECK(shared <= 3); // only the probes

	for (int i = 0; i < 40; i++)
		filgroup::shareRatioUpdate(8.);
	CHECK(filgroup::shareRatio().load() >= filgroup::ShareMin);
	for (int i = 0; i < 10; i++)
		CHECK(filgroup::sharePays());
}
