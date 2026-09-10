#include <catch2/catch.hpp>

#include "src/class2d_consensus.h"

#include <algorithm>
#include <cmath>
#include <numeric>
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

TEST_CASE("Class2D consensus permits unoccupied class slots", "[consensus]")
{
	const int nr_classes = 4;
	std::vector<std::vector<int> > runs;
	SECTION("Replicas have different empty labels, including gaps")
	{
		runs = {{0,0,0,0,2,2,2,2}, {3,3,3,3,1,1,1,1}, {2,2,2,2,0,0,0,0}};
	}
	SECTION("Only one class is occupied")
	{
		runs = {{2,2,2,2,2,2}, {0,0,0,0,0,0}, {3,3,3,3,3,3}};
	}
	SECTION("There are fewer particles than class slots")
	{
		runs = {{2}, {0}, {3}};
	}

	const Class2DConsensusResult result = Class2DConsensus::fit(runs, nr_classes);
	REQUIRE(result.assignment == runs[result.anchor_run]);
	REQUIRE(result.probability.size() == runs[0].size());
	REQUIRE(result.entropy.size() == runs[0].size());
	REQUIRE(result.agreement.size() == runs[0].size());
	REQUIRE(std::isfinite(result.log_likelihood));
	REQUIRE(result.class_posterior_mass.size() == nr_classes);
	REQUIRE(std::accumulate(result.class_posterior_mass.begin(), result.class_posterior_mass.end(), 0.)
		== Approx((double)runs[0].size()));
	std::vector<int> counts(nr_classes, 0);
	for (size_t i = 0; i < result.assignment.size(); ++i)
	{
		counts[result.assignment[i]]++;
		REQUIRE(std::isfinite(result.probability[i]));
		REQUIRE(result.probability[i] > 0.);
		REQUIRE(result.probability[i] <= 1.);
		REQUIRE(std::isfinite(result.entropy[i]));
		REQUIRE(result.entropy[i] >= 0.);
		REQUIRE(result.entropy[i] <= 1.);
		REQUIRE(result.agreement[i] == Approx(1.));
	}
	REQUIRE(std::count(counts.begin(), counts.end(), 0) > 0);
	// Zero hard occupancy does not imply zero posterior mass under smoothing.
	for (int k = 0; k < nr_classes; ++k)
		REQUIRE(result.class_posterior_mass[k] > 0.);
	REQUIRE(result.run_adjusted_rand.size() == runs.size());
	REQUIRE(result.run_mapped_agreement.size() == runs.size());
	REQUIRE(result.confusion.size() == runs.size() * nr_classes * nr_classes);
	for (size_t run = 0; run < runs.size(); ++run)
	{
		REQUIRE(result.run_adjusted_rand[run] == Approx(1.));
		REQUIRE(result.run_mapped_agreement[run] == Approx(1.));
		for (int k = 0; k < nr_classes; ++k)
		{
			double sum = 0.;
			for (int observed = 0; observed < nr_classes; ++observed)
			{
				const double value = result.confusion[(run * nr_classes + k) * nr_classes + observed];
				REQUIRE(std::isfinite(value));
				REQUIRE(value > 0.);
				sum += value;
			}
			REQUIRE(sum == Approx(1.));
		}
	}
}

TEST_CASE("Class2D consensus can leave a class empty even when every input class is occupied", "[consensus]")
{
	const std::vector<std::vector<int> > runs{{0,1,2,2}, {2,0,1,2}, {1,0,2,1}};
	const Class2DConsensusResult result = Class2DConsensus::fit(runs, 3);
	REQUIRE(result.assignment.size() == 4);
	std::vector<int> counts(3, 0);
	for (size_t i = 0; i < result.assignment.size(); ++i)
	{
		REQUIRE(result.assignment[i] >= 0);
		REQUIRE(result.assignment[i] < 3);
		counts[result.assignment[i]]++;
	}
	REQUIRE(std::count(counts.begin(), counts.end(), 0) > 0);
	REQUIRE(std::isfinite(result.log_likelihood));
}
