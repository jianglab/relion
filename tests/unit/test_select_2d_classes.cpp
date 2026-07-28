/*
 * tests/unit/test_select_2d_classes.cpp
 *
 * Unit tests for Select 2D class ordering, assignment validation, and
 * filament voting. All inputs are synthetic.
 */

#include <catch2/catch.hpp>

#include "src/select_2d_classes_utils.h"

#include <string>
#include <vector>

TEST_CASE("Select2D: removed type IDs become junk", "[select2d]")
{
	REQUIRE(Select2DClasses::DEFAULT_NR_TYPES == 1);
	REQUIRE(Select2DClasses::MAX_NR_TYPES == 20);
	REQUIRE(Select2DClasses::validTypeOrJunk(1, 1) == 1);
	REQUIRE(Select2DClasses::validTypeOrJunk(2, 1) == 0);
	REQUIRE(Select2DClasses::validTypeOrJunk(0, 1) == 0);
	REQUIRE(Select2DClasses::validTypeOrJunk(-1, 3) == 0);
}

TEST_CASE("Select2D: similarity order is deterministic", "[select2d]")
{
	const std::vector<std::vector<RFLOAT> > features = {
		{1., 0.},
		{0.9, 0.1},
		{-1., 0.}
	};
	const std::vector<RFLOAT> populations = {1., 5., 2.};
	const std::vector<int> assignments = {2, 0, 1};

	const std::vector<int> order =
			Select2DClasses::similarityOrder(features, populations);
	const std::vector<int> expected = {1, 0, 2};
	REQUIRE(order == expected);
	REQUIRE(assignments == std::vector<int>({2, 0, 1}));
}

TEST_CASE("Select2D: junk and lower type win exact ties", "[select2d]")
{
	REQUIRE(Select2DClasses::winningType({4, 4, 0}) == 0);
	REQUIRE(Select2DClasses::winningType({0, 3, 3}) == 1);
	REQUIRE(Select2DClasses::winningType({0, 2, 4}) == 2);
	REQUIRE_THROWS(Select2DClasses::winningType({}));
}

TEST_CASE("Select2D: voting keeps micrographs and tube IDs distinct", "[select2d]")
{
	const std::vector<std::string> micrographs = {
		"micA.mrc", "micA.mrc", "micA.mrc",
		"micB.mrc", "micB.mrc",
		"micA.mrc", "micA.mrc"
	};
	const std::vector<int> tube_ids = {1, 1, 1, 1, 1, 2, 2};
	const std::vector<int> assigned_types = {1, 1, 0, 2, 2, 1, 0};

	const Select2DClasses::FilamentVoteResult result =
			Select2DClasses::voteFilaments(
					micrographs, tube_ids, assigned_types, 2);

	const std::vector<int> expected_particle_types = {1, 1, 1, 2, 2, 0, 0};
	const std::vector<long int> expected_filament_counts = {1, 1, 1};
	REQUIRE(result.particle_types == expected_particle_types);
	REQUIRE(result.filament_counts == expected_filament_counts);
}

TEST_CASE("Select2D: voting rejects inconsistent particle inputs", "[select2d]")
{
	REQUIRE_THROWS(Select2DClasses::voteFilaments(
			{"micA.mrc"}, {}, {1}, 1));
	REQUIRE_THROWS(Select2DClasses::voteFilaments(
			{}, {}, {}, 0));
}
