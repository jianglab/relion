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
