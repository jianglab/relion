/*
 * tests/unit/test_pair_list.cpp
 *
 * The fast fine-pass pair-list builder (src/acc/pair_list.h) must give exactly the lists and jobs
 * of the plain loop, for every mix of significant translations, chunk size and oversampling.
 */

#include <catch2/catch.hpp>

#include <memory>
#include <random>
#include <vector>

#include "src/acc/pair_list.h"

namespace
{
struct Case
{
	size_t orientation_num, nt, nov, nor, nr_trans, rot_mod, chunk, n_oc;
	double p_sig;
	unsigned seed;
};

void runCase(const Case &c)
{
	std::mt19937 rng(c.seed);
	std::vector<size_t> oc(c.orientation_num), over(c.orientation_num);
	for (size_t i = 0; i < c.orientation_num; i++)
	{
		oc[i] = rng() % c.n_oc;
		over[i] = rng() % c.nor;
	}
	std::vector<char> sigc(c.n_oc * c.nr_trans);
	std::bernoulli_distribution b(c.p_sig);
	for (auto &s : sigc)
		s = b(rng);
	std::vector<unsigned char> sigv(sigc.begin(), sigc.end());
	std::unique_ptr<bool[]> sig(new bool[sigv.size()]);
	for (size_t i = 0; i < sigv.size(); i++)
		sig[i] = sigv[i] != 0;

	pairlist::Input in;
	in.orientation_num = c.orientation_num;
	in.nt = c.nt;
	in.nov = c.nov;
	in.nor = c.nor;
	in.nr_trans = c.nr_trans;
	in.rot_mod = c.rot_mod;
	in.chunk = c.chunk;
	in.iorientclasses = oc.data();
	in.iover_rots = over.data();
	in.sig = sig.get();

	const size_t maxpairs = c.orientation_num * c.nt * c.nov;
	struct Buf
	{
		std::vector<size_t> a, b, t, h, jo, je;
		explicit Buf(size_t n) : a(n + 1, 777), b(n + 1, 777), t(n + 1, 777), h(n + 1, 777), jo(n + 2, 777), je(n + 2, 777) {}
		pairlist::Output out()
		{
			pairlist::Output o;
			o.rot_id = a.data(); o.rot_idx = b.data(); o.trans_idx = t.data(); o.ihidden_overs = h.data();
			o.job_origin = jo.data(); o.job_extent = je.data();
			o.pairs = o.jobs = 0;
			return o;
		}
	} ref(maxpairs), fast(maxpairs);
	pairlist::Output ro = ref.out(), fo = fast.out();
	pairlist::buildReference(in, ro);
	pairlist::build(in, fo);

	REQUIRE(fo.pairs == ro.pairs);
	REQUIRE(fo.jobs == ro.jobs);
	for (size_t w = 0; w < ro.pairs; w++)
	{
		REQUIRE(fast.a[w] == ref.a[w]);
		REQUIRE(fast.b[w] == ref.b[w]);
		REQUIRE(fast.t[w] == ref.t[w]);
		REQUIRE(fast.h[w] == ref.h[w]);
	}
	size_t covered = 0;
	for (size_t k = 0; k < ro.jobs; k++)
	{
		REQUIRE(fast.jo[k] == ref.jo[k]);
		REQUIRE(fast.je[k] == ref.je[k]);
		REQUIRE(fast.je[k] > 0);
		REQUIRE(fast.je[k] <= c.chunk);
		covered += fast.je[k];
	}
	REQUIRE(covered == ro.pairs);
	// nothing past the end of the lists was touched
	for (size_t w = ro.pairs; w <= maxpairs; w++)
		REQUIRE(fast.a[w] == 777);
}
}

TEST_CASE("fast pair list equals the plain loop", "[pair_list]")
{
	const double ps[] = {0.0, 0.03, 0.3, 0.7, 1.0};
	const size_t chunks[] = {1, 4, 7};
	const size_t novs[] = {1, 2, 4, 8};
	unsigned seed = 1;
	for (double p : ps)
		for (size_t chunk : chunks)
			for (size_t nov : novs)
			{
				Case c{57, 13, nov, 8, 20, 5, chunk, 3, p, seed++};
				runCase(c);
			}
}

TEST_CASE("fast pair list: one orientation, one translation", "[pair_list]")
{
	runCase(Case{1, 1, 1, 1, 1, 1, 4, 1, 1.0, 5});
	runCase(Case{1, 1, 1, 1, 1, 1, 4, 1, 0.0, 6});
}

TEST_CASE("fast pair list: a row split across chunks continues in the next row only as a new job", "[pair_list]")
{
	runCase(Case{40, 9, 4, 2, 9, 1, 4, 1, 0.9, 11});
	runCase(Case{40, 9, 4, 2, 9, 1, 7, 2, 0.9, 12});
}
