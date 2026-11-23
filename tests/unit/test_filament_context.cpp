/*
 * tests/unit/test_filament_context.cpp
 *
 * Class priors for helical segments from the other segments of their filament
 * (src/filament_context.h): the prior must never use the segment's own data,
 * the chain must learn the order in which classes follow each other, and the
 * filament mode must recover a filament's family of classes.
 */

#include <catch2/catch.hpp>

#include "src/filament_context.h"
#include "src/healpix_sampling.h"
#include "src/metadata_table.h"

#include <cmath>
#include <random>

namespace
{
// Filaments whose true class cycles 0,1,...,K-1 with `dwell` segments per class;
// log-likelihoods are `signal` for the true class plus Gaussian noise.
void cyclicData(int K, int nfil, int len, int dwell, double signal, double noise, unsigned seed,
                std::vector<std::vector<long int> > &fil, std::vector<float> &ll, std::vector<int> &truth)
{
	std::mt19937 rng(seed);
	std::normal_distribution<double> g(0., noise);
	fil.clear(); ll.clear(); truth.clear();
	long int id = 0;
	for (int f = 0; f < nfil; f++)
	{
		std::vector<long int> one;
		int start = f % (K * dwell);
		for (int t = 0; t < len; t++, id++)
		{
			int c = ((start + t) / dwell) % K;
			truth.push_back(c);
			for (int k = 0; k < K; k++)
				ll.push_back((float)((k == c ? signal : 0.) + g(rng)));
			one.push_back(id);
		}
		fil.push_back(one);
	}
}

int argmax(const float *x, int K)
{
	int b = 0;
	for (int k = 1; k < K; k++) if (x[k] > x[b]) b = k;
	return b;
}
}

TEST_CASE("a segment's prior does not depend on its own likelihood", "[filament_context]")
{
	const int K = 4;
	std::vector<std::vector<long int> > fil;
	std::vector<float> ll;
	std::vector<int> truth;
	cyclicData(K, 20, 30, 5, 2., 1., 1, fil, ll, truth);
	std::vector<double> pi(K, 1. / K);

	for (filctx::Mode m : {filctx::CHAIN, filctx::FILAMENT})
	{
		filctx::Options o;
		o.mode = m;
		o.bw_iters = 0; // keep T fixed so only the cavity is compared
		std::vector<double> T1, T2;
		std::vector<float> p1, p2;
		filctx::computePriors(fil, ll, K, pi, o, 1., T1, p1);
		std::vector<float> ll2 = ll;
		const long int s = fil[3][10];
		for (int k = 0; k < K; k++) ll2[s * K + k] = (k == 0) ? 50.f : -50.f;
		filctx::computePriors(fil, ll2, K, pi, o, 1., T2, p2);
		for (int k = 0; k < K; k++)
			CHECK(p1[s * K + k] == Approx(p2[s * K + k]).margin(1e-6));
	}
}

TEST_CASE("the chain learns the cyclic order of the classes and predicts them", "[filament_context]")
{
	const int K = 5;
	std::vector<std::vector<long int> > fil;
	std::vector<float> ll;
	std::vector<int> truth;
	cyclicData(K, 60, 40, 4, 1.5, 1., 7, fil, ll, truth);
	std::vector<double> pi(K, 1. / K);

	filctx::Options o;
	o.mode = filctx::CHAIN;
	o.bw_iters = 10;
	o.floor = 0.;
	std::vector<double> T;
	std::vector<float> lp;
	filctx::computePriors(fil, ll, K, pi, o, 1., T, lp);

	// class k is followed by itself or by a neighbour in the cycle (either direction)
	for (int j = 0; j < K; j++)
	{
		double near = T[j * K + j] + T[j * K + (j + 1) % K] + T[j * K + (j + K - 1) % K];
		CHECK(near > 0.9);
	}

	// the prior alone predicts the class far better than chance, and prior x own
	// likelihood beats the own likelihood alone
	long int n = truth.size(), ok_prior = 0, ok_own = 0, ok_both = 0;
	for (long int i = 0; i < n; i++)
	{
		std::vector<float> both(K);
		for (int k = 0; k < K; k++) both[k] = lp[i * K + k] + ll[i * K + k];
		ok_prior += argmax(&lp[i * K], K) == truth[i];
		ok_own += argmax(&ll[i * K], K) == truth[i];
		ok_both += argmax(&both[0], K) == truth[i];
	}
	INFO("prior " << ok_prior << " own " << ok_own << " both " << ok_both << " of " << n);
	CHECK(ok_prior > 0.6 * n);
	CHECK(ok_both > ok_own + 0.1 * n);
}

TEST_CASE("the filament mode recovers each filament's family of classes", "[filament_context]")
{
	// two families: classes {0,1} and {2,3}; a filament visits only its own family
	const int K = 4;
	std::mt19937 rng(3);
	std::normal_distribution<double> g(0., 1.);
	std::vector<std::vector<long int> > fil;
	std::vector<float> ll;
	std::vector<int> fam;
	long int id = 0;
	for (int f = 0; f < 40; f++)
	{
		std::vector<long int> one;
		for (int t = 0; t < 30; t++, id++)
		{
			int c = 2 * (f % 2) + (t / 5) % 2;
			fam.push_back(f % 2);
			for (int k = 0; k < K; k++) ll.push_back((float)((k == c ? 1. : 0.) + g(rng)));
			one.push_back(id);
		}
		fil.push_back(one);
	}
	std::vector<double> pi(K, 0.25), T;
	std::vector<float> lp;
	filctx::Options o;
	o.mode = filctx::FILAMENT;
	filctx::computePriors(fil, ll, K, pi, o, 1., T, lp);
	long int ok = 0;
	for (long int i = 0; i < id; i++)
	{
		double in = std::exp(lp[i * K + 2 * fam[i]]) + std::exp(lp[i * K + 2 * fam[i] + 1]);
		ok += in > 0.5;
	}
	CHECK(ok > 0.95 * id);
}

TEST_CASE("segments are grouped per filament and in track order", "[filament_context]")
{
	MetaDataTable MD;
	const char *mics[] = {"a.mrc", "a.mrc", "b.mrc", "a.mrc", "a.mrc"};
	int tubes[] = {1, 2, 1, 1, 2};
	double track[] = {30., 0., 0., 15., 15.};
	for (int i = 0; i < 5; i++)
	{
		MD.addObject();
		MD.setValue(EMDL_MICROGRAPH_NAME, std::string(mics[i]));
		MD.setValue(EMDL_PARTICLE_HELICAL_TUBE_ID, tubes[i]);
		MD.setValue(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM, track[i]);
	}
	double step;
	std::vector<std::vector<long int> > f = filctx::groupFilaments(MD, step);
	REQUIRE(f.size() == 3);
	CHECK(step == Approx(15.));
	CHECK(f[0] == std::vector<long int>({3, 0}));  // a.mrc tube 1
	CHECK(f[1] == std::vector<long int>({1, 4}));  // a.mrc tube 2
	CHECK(f[2] == std::vector<long int>({2}));
}

TEST_CASE("3D helical bimodal psi selection resets is_psi_flipped per psi sample",
          "[filament_context][healpix]")
{
	HealpixSampling hs;
	hs.healpix_order = 1;
	hs.psi_step = 5.;
	hs.offset_step = 1.;
	hs.offset_range = 5.;
	hs.limit_tilt = -91.;
	hs.fn_sym = "C1";
	hs.initialise(3, true, false, false, false, true, 4.75, -2.6);

	std::vector<int> ptr_dir, ptr_psi;
	std::vector<RFLOAT> dir_prior, psi_prior;
	hs.selectOrientationsWithNonZeroPriorProbabilityFor3DHelicalReconstruction(
		0., 90., 15., 5., 5., 5., ptr_dir, dir_prior, ptr_psi, psi_prior, false, 0.25, 0.5);

	REQUIRE(ptr_psi.size() == psi_prior.size());
	REQUIRE_FALSE(psi_prior.empty());
	double sum_unflipped = 0., sum_flipped = 0., total = 0.;
	for (size_t i = 0; i < ptr_psi.size(); i++)
	{
		RFLOAT psi = 0.;
		hs.getPsiAngle(ptr_psi[i], psi);
		const double w = psi_prior[i];
		total += w;
		if (std::fabs(std::remainder(psi - 15., 360.)) < 20.)
			sum_unflipped += w;
		else if (std::fabs(std::remainder(psi + 165., 360.)) < 20.)
			sum_flipped += w;
	}
	CHECK(sum_unflipped > 0.);
	CHECK(sum_flipped > 0.);
	CHECK(total == Approx(1.0).margin(1e-6));
	CHECK((sum_unflipped / sum_flipped) == Approx(3.0).margin(0.25));
}

TEST_CASE("Huber contrast clipping prevents single extreme outlier from corrupting leave-one-out prior",
          "[filament_context]")
{
	const int K = 2;
	std::vector<std::vector<long int> > f = {{0, 1, 2}};
	std::vector<std::vector<double> > tracks = {{0., 15., 30.}};
	std::vector<float> ll = {
		0.5f, 0.f,
		0.5f, 0.f,
		0.f, 100.f
	};

	std::vector<double> pi = {0.5, 0.5};
	filctx::Options opt;
	opt.mode = filctx::CHAIN;
	opt.stay = 0.99;
	opt.bw_iters = 0;
	opt.floor = 0.;
	std::vector<double> T;
	std::vector<float> lp;

	opt.clip = -1.;
	filctx::computePriors(f, tracks, 15., ll, K, pi, opt, 1.0, T, lp);
	REQUIRE(std::exp(lp[0 * 2 + 0]) < 0.03);

	opt.clip = 1.0;
	filctx::computePriors(f, tracks, 15., ll, K, pi, opt, 1.0, T, lp);
	REQUIRE(std::exp(lp[0 * 2 + 0]) > 0.30);

	// Also check moderate Huber bound (clip = 6.0) caps an extreme segment relative to unclipped
	opt.mode = filctx::FILAMENT;
	opt.alpha = 0.;
	f = {{0, 1}};
	tracks = {{0., 15.}};
	ll = {6.f, 0.f, 0.f, 20.f};
	opt.clip = -1.;
	filctx::computePriors(f, tracks, 15., ll, K, pi, opt, 1.0, T, lp);
	const float p_unclipped = std::exp(lp[0 * 2 + 0]);
	opt.clip = 6.0;
	filctx::computePriors(f, tracks, 15., ll, K, pi, opt, 1.0, T, lp);
	const float p_clipped = std::exp(lp[0 * 2 + 0]);
	REQUIRE(p_clipped > p_unclipped * 1e5f);
}

TEST_CASE("Physical track exclusion leaves out nearby overlapping segments across irregular track spacing",
          "[filament_context]")
{
	const int K = 2;
	std::vector<std::vector<long int> > f = {{0, 1, 2, 3}};
	std::vector<std::vector<double> > tracks = {{0., 10., 20., 200.}};
	std::vector<float> ll = {
		10.f, 0.f,
		10.f, 0.f,
		10.f, 0.f,
		0.f, 10.f
	};
	std::vector<double> pi = {0.5, 0.5};
	filctx::Options opt;
	opt.mode = filctx::FILAMENT;
	opt.gap = 1;
	opt.exclude_angst = 50.;
	opt.floor = 0.;
	opt.alpha = 0.01;
	std::vector<double> T;
	std::vector<float> lp;

	filctx::computePriors(f, tracks, 10., ll, K, pi, opt, 1.0, T, lp);
	CHECK(std::exp(lp[1 * 2 + 1]) > 0.9);
	CHECK(std::exp(lp[3 * 2 + 0]) > 0.9);
}

TEST_CASE("3D helical phase coherence bonus favors azimuthally consistent rot progression",
          "[filament_context]")
{
	const long int N = 9;
	const int K = 2;
	std::vector<std::vector<long int> > f = {{0, 1, 2, 3, 4, 5, 6, 7, 8}};
	std::vector<std::vector<double> > tracks = {{0., 19., 38., 57., 76., 95., 114., 133., 152.}};
	std::vector<filctx::ClassHelixParams> hp(K);
	for (int k = 0; k < K; k++)
	{
		hp[k].twist_deg = -2.61;
		hp[k].rise_angst = 4.75;
		hp[k].sym_order = 2;
	}
	const double scrambled[9] = {0., 45., 90., 135., 20., 110., 5., 80., 150.};
	std::vector<double> rot((size_t)N * K, 0.), psi((size_t)N * K, 0.),
	                    dx((size_t)N * K, 0.), dy((size_t)N * K, 0.), psi_prior(N, 0.);
	std::vector<char> valid((size_t)N * K, 1);
	for (long int p = 0; p < N; p++)
	{
		rot[(size_t)p * K + 0] = -2.61 * (tracks[0][p] / 4.75) + 30.;
		rot[(size_t)p * K + 1] = scrambled[p];
	}
	std::vector<float> bonus;
	filctx::computeHelicalPhaseBonus(f, tracks, N, K, rot, psi, dx, dy, valid, psi_prior,
	                                 hp, 1, 20., 1.0, bonus);
	REQUIRE(bonus.size() == (size_t)N * K);
	CHECK(bonus[4 * 2 + 0] > bonus[4 * 2 + 1] + 0.5f);

	std::vector<float> ll((size_t)N * K, 0.f);
	std::vector<double> pi = {0.5, 0.5}, T;
	std::vector<float> lp;
	filctx::Options opt;
	opt.mode = filctx::FILAMENT;
	opt.gap = 1;
	opt.exclude_angst = 20.;
	opt.floor = 0.;
	opt.alpha = 0.;
	filctx::computePriors(f, tracks, 19., ll, K, pi, opt, 1.0, T, lp, &bonus);
	CHECK(std::exp(lp[4 * 2 + 0]) > 0.60);
}
