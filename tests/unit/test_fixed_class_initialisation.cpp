#include <catch2/catch.hpp>
#include "src/fixed_class_initialisation.h"
#include <algorithm>
#include <set>

TEST_CASE("Fresh fixed-class seeds cover rare classes and retain empty slots", "[consensus]")
{
	FixedClassSeedSelection selection(4, 1);
	for (int i = 0; i < 500; ++i) selection.add("common_" + std::to_string(i), i, 1);
	selection.add("rare", 500, 3);
	auto chosen = selection.takeIndices();
	REQUIRE(chosen.size() == 101);
	REQUIRE(std::count(chosen.begin(), chosen.end(), 500) == 1);
	REQUIRE(std::set<long int>(chosen.begin(), chosen.end()).size() == chosen.size());
}

TEST_CASE("Fresh fixed-class seeds use stable identities and a reproducible seed", "[consensus]")
{
	FixedClassSeedSelection forward(2, 11), reverse(2, 11), changed(2, 12);
	for (int i = 0; i < 400; ++i)
	{
		forward.add("image_" + std::to_string(i), i, i % 2 + 1);
		changed.add("image_" + std::to_string(i), i, i % 2 + 1);
	}
	for (int i = 399; i >= 0; --i) reverse.add("image_" + std::to_string(i), i, i % 2 + 1);
	auto a = forward.takeIndices(), b = reverse.takeIndices(), c = changed.takeIndices();
	REQUIRE(a == b);
	REQUIRE(a != c);
	for (int i = 0; i < 400; ++i)
	{
		const auto name = "image_" + std::to_string(i);
		const double psi = fixedClassInitialPsi(name, 11);
		REQUIRE(psi >= 0.);
		REQUIRE(psi < 360.);
		REQUIRE(psi == fixedClassInitialPsi(name, 11));
		REQUIRE(psi != fixedClassInitialPsi(name, 12));
	}
}

TEST_CASE("Fresh fixed-class selection validates classes and seed", "[consensus]")
{
	REQUIRE_THROWS(FixedClassSeedSelection(1, 1));
	REQUIRE_THROWS(FixedClassSeedSelection(2, 0));
	FixedClassSeedSelection selection(2, 1);
	REQUIRE_THROWS(selection.add("image", 0, 0));
	REQUIRE_THROWS(selection.add("image", 0, 3));
	REQUIRE_THROWS(selection.add("image", -1, 1));
}
