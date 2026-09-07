#include <catch2/catch.hpp>

#include "src/class2d_consensus.h"

#include <algorithm>
#include <vector>

TEST_CASE("Class2D consensus is invariant to replica label permutations", "[consensus]")
{
	std::vector<std::vector<int> > runs;
	runs.push_back(std::vector<int>{0, 0, 0, 1, 1, 1, 2, 2, 2});
	runs.push_back(std::vector<int>{2, 2, 2, 0, 0, 0, 1, 1, 1});
	runs.push_back(std::vector<int>{1, 1, 1, 2, 2, 2, 0, 0, 0});

	Class2DConsensusResult result = Class2DConsensus::fit(runs, 3);
	REQUIRE(result.assignment == runs[0]);
	for (size_t i = 0; i < result.assignment.size(); ++i)
	{
		REQUIRE(result.probability[i] > .95);
		REQUIRE(result.entropy[i] < .15);
		REQUIRE(result.agreement[i] == Approx(1.));
	}
}

TEST_CASE("Class2D consensus ignores an uninformative replica", "[consensus]")
{
	std::vector<int> truth{0,0,0,0,1,1,1,1,2,2,2,2};
	std::vector<std::vector<int> > runs;
	runs.push_back(truth);
	runs.push_back(std::vector<int>{1,1,1,1,2,2,2,2,0,0,0,0});
	runs.push_back(truth);
	runs.push_back(std::vector<int>{0,1,2,0,1,2,0,1,2,0,1,2});

	Class2DConsensusResult result = Class2DConsensus::fit(runs, 3);
	REQUIRE(result.assignment == truth);
	REQUIRE(result.run_adjusted_rand[3] < .1);
	REQUIRE(result.run_mapped_agreement[3] < .6);
}

TEST_CASE("Class2D consensus validates dimensions and classes", "[consensus]")
{
	REQUIRE_THROWS(Class2DConsensus::fit(std::vector<std::vector<int> >{{0,1}}, 2));
	REQUIRE_THROWS(Class2DConsensus::fit(std::vector<std::vector<int> >{{0,1},{0}}, 2));
	REQUIRE_THROWS(Class2DConsensus::fit(std::vector<std::vector<int> >{{0,2},{0,1}}, 2));
}
