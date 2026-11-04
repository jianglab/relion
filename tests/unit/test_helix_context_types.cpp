/*
 * tests/unit/test_helix_context_types.cpp
 *
 * Choosing the number of helical types (HelicalContext::selectNrTypes with agree_min > 0):
 * synthetic filaments of a known number of types, each type with its own classes.
 */

#include <catch2/catch.hpp>

#include "src/helix_context.h"

#include <cmath>
#include <cstdlib>
#include <random>

namespace
{
// nfil filaments of nseg segments; filament f has type f % ntypes, and each segment's best class is one
// of its type's two classes (or, with probability mix, any class)
void makeTypes(HelicalContext &hc, int ntypes, int K, int nfil, int nseg, double mix, unsigned seed)
{
	std::mt19937 rng(seed);
	std::uniform_real_distribution<double> u(0., 1.);
	hc.verb = getenv("HCT_VERB") ? 1 : 0;
	hc.K = K;
	hc.N = (long int)nfil * nseg;
	hc.filament_name.clear();
	hc.filament_of_chain.clear();
	hc.seg_chain.assign(hc.N, -1);
	hc.loglik.assign(hc.N * K, -10.f);
	for (int f = 0; f < nfil; f++)
	{
		hc.filament_name.push_back("mic:" + std::to_string(f));
		hc.filament_of_chain.push_back(f);
		const int t = f % ntypes;
		for (int s = 0; s < nseg; s++)
		{
			const long int i = (long int)f * nseg + s;
			const int k = (u(rng) < mix) ? (int)(u(rng) * K) % K : (2 * t + (u(rng) < 0.5 ? 0 : 1)) % K;
			hc.seg_chain[i] = f;
			hc.loglik[i * K + k] = 0.f;
		}
	}
}
}

TEST_CASE("helical types: the number of types is found from the filaments' classes", "[helix_context_types]")
{
	for (int ntypes : {2, 4, 6})
	{
		INFO("true types " << ntypes);
		HelicalContext hc;
		makeTypes(hc, ntypes, 2 * ntypes + 2, 240, 30, 0.3, 7u + ntypes);
		CHECK(hc.selectNrTypes(2, 10, 3, 0.01, 0.75, false) == ntypes);
		CHECK(hc.nr_types == ntypes);
	}
}

TEST_CASE("helical types: replicas with their own class numbers find the same types", "[helix_context_types]")
{
	// what relion_class2d_consensus --helical_types does: per replica a fit to its own class counts, the number of
	// types from the agreement between the replicas
	const int ntypes = 4, K = 10, nfil = 240, nseg = 30;
	std::vector<helixtypes::FilamentCounts> X(2);
	for (int r = 0; r < 2; r++)
	{
		std::mt19937 rng(100u + r);
		std::uniform_real_distribution<double> u(0., 1.);
		std::vector<int> perm(K);
		for (int k = 0; k < K; k++) perm[k] = (k * (3 + 4 * r) + r) % K;  // 3 and 7 are coprime to 10
		X[r].C = K;
		for (int f = 0; f < nfil; f++)
		{
			std::vector<double> c(K, 0.);
			for (int s = 0; s < nseg; s++)
				c[perm[(u(rng) < 0.3) ? (int)(u(rng) * K) % K : 2 * (f % ntypes) + (u(rng) < 0.5 ? 0 : 1)]] += 1.;
			X[r].rows.emplace_back();
			for (int k = 0; k < K; k++)
				if (c[k] > 0.) X[r].rows.back().push_back({k, c[k]});
			X[r].total.push_back(nseg);
		}
	}
	std::vector<double> score(9, 0.), between(9, 0.);
	for (int T = 2; T <= 8; T++)
	{
		std::vector<double> s, a;
		for (int r = 0; r < 2; r++)
		{
			helixtypes::scan(X[r], T, T, 3, 2, s, a);
			score[T] += s[T] / 2;
		}
		between[T] = helixtypes::adjustedRand(helixtypes::fitTypes(X[0], T, 1u), helixtypes::fitTypes(X[1], T, 1u));
	}
	CHECK(between[ntypes] > 0.9);
	std::string why;
	CHECK(helixtypes::choose(score, between, 2, 8, 0.01, 0.75, 0, why) == ntypes);
}

TEST_CASE("helical types: junk classes are poorly resolved or wider than the box", "[helix_context_types]")
{
	// median 10 A: 16 A is junk at 1.5 x, NaN (never resolved) too; 300 A wide in a 360 A box too
	const std::vector<RFLOAT> res = {10., 9., 11., 16., NAN, 10.};
	const std::vector<char> junk = helixtypes::junkClasses(res, {100., 100., 100., 100., 100., 300.}, 360., 1.5);
	CHECK(junk == std::vector<char>({0, 0, 0, 1, 1, 1}));
	CHECK(helixtypes::junkClasses(res, {}, 360., 1.5) == std::vector<char>({0, 0, 0, 1, 1, 0}));
	// and classes that do not continue the helix
	CHECK(helixtypes::junkClasses(res, {}, 360., 1.5, {0.9, 0.2, 0.6, 0.9, 0.9, 0.9}) == std::vector<char>({0, 1, 0, 1, 1, 0}));
}

TEST_CASE("helical types: classes of a helix continue each other along the filaments, a blob does not", "[helix_context_types]")
{
	// 8 classes: the phases of a helix with a period of 40 pixels along x; class 8: a blob, given to 10% of the
	// segments at random. Segments 2 pixels apart; every third filament has the opposite polarity (its class
	// sequence runs backwards along its track).
	const int n = 64, P = 40;
	const RFLOAT apix = 5.;
	std::vector<MultidimArray<RFLOAT> > refs(9, MultidimArray<RFLOAT>(n, n));
	for (int k = 0; k < 9; k++)
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++)
			{
				const double y = i - n / 2, x = j - n / 2;
				DIRECT_A2D_ELEM(refs[k], i, j) = (k < 8) ? exp(-y * y / 30.) * (1. + 0.6 * cos(2. * PI * (x + k * P / 8.) / P))
				                                         : exp(-((x - 10) * (x - 10) + (y + 8) * (y + 8)) / 20.);
			}
	std::vector<long int> fil;
	std::vector<RFLOAT> track;
	std::vector<int> cls;
	std::vector<char> flipped;
	std::mt19937 rng(5u);
	for (int f = 0; f < 40; f++)
		for (int s = 0; s < 50; s++)
		{
			const double phase = (f % 3 == 2 ? -1. : 1.) * 2. * s + 7. * f;  // pixels
			fil.push_back(f);
			track.push_back(s * 2. * apix);
			cls.push_back(rng() % 10 == 0 ? 8 : (int)floor(fmod(fmod(phase, P) + P, P) / (P / 8.)) % 8);
			flipped.push_back(f % 3 == 2);
		}
	std::string note;
	const std::vector<RFLOAT> cont = helixtypes::classContinuity(refs, apix, fil, track, cls, flipped, note);
	INFO(note);
	for (int k = 0; k < 8; k++)
		CHECK(cont[k] > 0.8);
	CHECK(cont[8] < 0.6);
	CHECK(helixtypes::junkClasses(std::vector<RFLOAT>(9, 10.), {}, n * apix, 1.5, cont) ==
	      std::vector<char>({0, 0, 0, 0, 0, 0, 0, 0, 1}));
}

TEST_CASE("helical types: per-type Jaccard finds the types a replica splits differently", "[helix_context_types]")
{
	// replica 0: types 0, 1, 2 of 100 filaments each; replica 1 the same but with type 2 split at random into 2 and 0
	std::vector<std::vector<int> > lab(2);
	std::mt19937 rng(3u);
	for (int f = 0; f < 300; f++)
	{
		lab[0].push_back(f / 100);
		lab[1].push_back(f < 200 ? f / 100 : (rng() % 2 ? 2 : 0));
	}
	lab[1][5] = -1;  // left out
	std::vector<double> j = helixtypes::typeJaccard(lab, 3);
	REQUIRE(j.size() == 6);
	CHECK(j[1] == Approx(1.));                  // replica 0, type 1: found again exactly
	CHECK(j[2] == Approx(0.5).margin(0.1));     // replica 0, type 2: half of it
	CHECK(*std::min_element(j.begin(), j.end()) < 0.6);
}

TEST_CASE("helical types: junk classes are left out", "[helix_context_types]")
{
	// 3 types; class 7, from no type, is in every filament (carbon edges): leaving it out changes nothing
	HelicalContext hc;
	makeTypes(hc, 3, 8, 180, 30, 0.2, 11u);
	for (long int i = 0; i < hc.N; i += 3)
	{
		std::fill(hc.loglik.begin() + i * hc.K, hc.loglik.begin() + (i + 1) * hc.K, -10.f);
		hc.loglik[i * hc.K + 7] = 0.f;
	}
	hc.junk_class.assign(hc.K, 0);
	hc.junk_class[7] = 1;
	CHECK(hc.selectNrTypes(2, 8, 3, 0.01, 0.75, false) == 3);
}
