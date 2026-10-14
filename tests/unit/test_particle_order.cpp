/*
 * tests/unit/test_particle_order.cpp
 *
 * RELION_PARTICLE_ORDER=micrograph (Experiment::shuffleMicrographs): whole micrographs are shuffled,
 * each one's particles stay together and in file order.
 */

#include <catch2/catch.hpp>

#include "src/exp_model.h"

#include <map>
#include <random>
#include <set>
#include <string>

// 5 micrographs x 6 particles, listed in a scattered order (as after a particle-level shuffle)
static void makeExperiment(Experiment &exp)
{
	exp.clear();
	const int nmic = 5, nper = 6;
	std::vector<std::pair<int, int> > order;   // (micrograph, image number)
	for (int m = 0; m < nmic; m++)
		for (int k = 1; k <= nper; k++)
			order.push_back(std::make_pair(m, k));
	std::mt19937 scatter(7);
	std::shuffle(order.begin(), order.end(), scatter);
	for (const auto &o : order)
	{
		ExpParticle p;
		p.name = std::to_string(o.second) + "@Extract/mic" + std::to_string(o.first) + ".mrcs";
		exp.particles.push_back(p);
		exp.MDimg.addObject();
		exp.MDimg.setValue(EMDL_MICROGRAPH_NAME, std::string("MotionCorr/mic") + std::to_string(o.first) + ".mrc");
	}
	for (long int i = 0; i < (long int) exp.particles.size(); i++)
		exp.sorted_idx.push_back(i);
}

static std::string micrographOf(Experiment &exp, long int i)
{
	return exp.getMicrographName(i);
}

TEST_CASE("micrograph order: each micrograph's particles together, in file order", "[particle_order]")
{
	Experiment exp;
	makeExperiment(exp);
	std::mt19937 rng(42);
	exp.shuffleMicrographs(exp.sorted_idx.begin(), exp.sorted_idx.end(), rng);

	// every particle exactly once
	std::set<long int> seen(exp.sorted_idx.begin(), exp.sorted_idx.end());
	REQUIRE(seen.size() == exp.particles.size());

	// each micrograph in one stretch, image numbers rising within it
	std::set<std::string> finished;
	std::string current;
	long int last_img = 0;
	for (long int idx : exp.sorted_idx)
	{
		const std::string mic = micrographOf(exp, idx);
		long int img;
		FileName stack;
		exp.particles[idx].name.decompose(img, stack);
		if (mic != current)
		{
			REQUIRE(finished.count(mic) == 0);
			if (!current.empty()) finished.insert(current);
			current = mic;
			last_img = 0;
		}
		REQUIRE(img > last_img);
		last_img = img;
	}
}

TEST_CASE("micrograph order: the micrograph order follows the seed", "[particle_order]")
{
	auto micOrder = [](int seed)
	{
		Experiment exp;
		makeExperiment(exp);
		std::mt19937 rng(seed);
		exp.shuffleMicrographs(exp.sorted_idx.begin(), exp.sorted_idx.end(), rng);
		std::vector<std::string> mics;
		for (long int idx : exp.sorted_idx)
		{
			const std::string m = exp.getMicrographName(idx);
			if (mics.empty() || mics.back() != m) mics.push_back(m);
		}
		return mics;
	};
	CHECK(micOrder(1) == micOrder(1));    // reproducible
	bool any_different = false;
	for (int seed = 2; seed < 8; seed++)
		any_different = any_different || micOrder(seed) != micOrder(1);
	CHECK(any_different);
	CHECK(micOrder(1).size() == 5);
}

TEST_CASE("micrograph order: without micrograph names, by stack file", "[particle_order]")
{
	Experiment exp;
	exp.clear();
	for (int i = 0; i < 12; i++)
	{
		ExpParticle p;
		p.name = std::to_string(12 - i) + "@stack" + std::to_string(i % 3) + ".mrcs";
		exp.particles.push_back(p);
		exp.MDimg.addObject();
		exp.sorted_idx.push_back(i);
	}
	std::mt19937 rng(3);
	exp.shuffleMicrographs(exp.sorted_idx.begin(), exp.sorted_idx.end(), rng);
	std::vector<std::string> stacks;
	for (long int idx : exp.sorted_idx)
	{
		long int img;
		FileName stack;
		exp.particles[idx].name.decompose(img, stack);
		if (stacks.empty() || stacks.back() != stack) stacks.push_back(stack);
	}
	CHECK(stacks.size() == 3);
}
