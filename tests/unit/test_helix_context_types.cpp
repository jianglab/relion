/*
 * tests/unit/test_helix_context_types.cpp
 *
 * Choosing the number of helical types (HelicalContext::selectNrTypes with agree_min > 0):
 * synthetic filaments of a known number of types, each type with its own classes.
 */

#include <catch2/catch.hpp>

#include "src/async_file_writer.h"
#include "src/auto_mask.h"
#include "src/helix_context.h"
#include "src/image.h"
#include "src/jaz/single_particle/obs_model.h"
#include "src/metadata_table.h"
#include "src/ml_optimiser.h"
#include "src/virtual_particles.h"

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <limits.h>
#include <random>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

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

TEST_CASE("helical types: FilamentChains distinguishes low-twist types sharing a crossover", "[helix_context_types]")
{
	// 3 types across 210 filaments x 40 segments: Types 0 and 1 share states 0 & 1 for ~45% of their cycle
	// (an indistinguishable crossover), then diverge along ordered transitions 2->3->4 vs 5->6.
	// Type 2 stays mostly in 7<->8. Chain mixture recovers T=3 and ARI > 0.95.
	helixtypes::FilamentChains FC;
	FC.C = 9;
	FC.kappa = 0.5;
	std::mt19937 rng(123u);
	std::uniform_real_distribution<double> u(0., 1.);
	std::vector<int> truth;
	for (int f = 0; f < 210; f++)
	{
		const int t = f % 3;
		truth.push_back(t);
		FC.mics.push_back("mic_" + std::to_string(f % 30));
		std::vector<int> seq;
		int cur = (t == 0) ? 2 : (t == 1) ? 5 : 7;
		for (int s = 0; s < 40; s++)
		{
			if (s > 0)
			{
				if (t == 0)
				{
					// 0->1->0 crossover (~45%), 0->2->3->4->0
					if (cur == 0) cur = (u(rng) < 0.65) ? 1 : 2;
					else if (cur == 1) cur = (u(rng) < 0.70) ? 0 : 2;
					else if (cur == 2) cur = 3;
					else if (cur == 3) cur = 4;
					else cur = 0;
				}
				else if (t == 1)
				{
					// 0->1->0 crossover (~45%), 1->5->6->0
					if (cur == 0) cur = (u(rng) < 0.70) ? 1 : 5;
					else if (cur == 1) cur = (u(rng) < 0.65) ? 0 : 5;
					else if (cur == 5) cur = 6;
					else cur = 0;
				}
				else
				{
					cur = (cur == 7) ? ((u(rng) < 0.90) ? 8 : 0) : 7;
				}
			}
			// 5% emission noise
			int obs = (u(rng) < 0.05) ? (int)(u(rng) * FC.C) % FC.C : cur;
			seq.push_back(obs);
		}
		FC.seqs.push_back(seq);
	}
	std::vector<double> score, agree, min_jacc;
	helixtypes::scanChains(FC, 1, 5, 3, 2, score, agree, &min_jacc);
	INFO("agree[3]=" << agree[3] << " min_jacc[3]=" << min_jacc[3]);
	CHECK(agree[3] > 0.90);
	std::string why;
	const int T = helixtypes::choose(score, agree, 1, 5, 0.01, 0.75, 0, why);
	INFO("choose=" << T << " why=" << why);
	CHECK(T == 3);
	const std::vector<int> pred = helixtypes::fitChainTypes(FC, 3, 1u);
	CHECK(helixtypes::adjustedRand(pred, truth) > 0.95);
}

TEST_CASE("helical context: IRLS cubic spline track smoothing rejects outliers and recovers tangent", "[helix_context_types]")
{
	// 40-segment curved filament along tilted axis with 12 A outlier bumps in observed shifts
	const int L = 40;
	const double apix = 1.5;
	const double step_A = 14.25;
	const double th = 25.0 * PI / 180.0;
	const double ct = cos(th), st = sin(th);
	MetaDataTable MD;
	std::vector<long int> seg_rows(L);
	std::vector<double> x_true(L), y_true(L), px_pick(L), py_pick(L), psi_true(L);
	for (int j = 0; j < L; j++)
	{
		const double s = j * step_A;
		const double bend = 6.0 * sin(2.0 * PI * j / (double)L);
		const double d_bend = 6.0 * (2.0 * PI / (L * step_A)) * cos(2.0 * PI * j / (double)L);
		x_true[j] = 500.0 + s * ct - bend * st;
		y_true[j] = 500.0 + s * st + bend * ct;
		// Pick coordinates slightly jittered across axis by +-2 A
		const double pick_jit = (j % 2 == 0 ? 1.5 : -1.5);
		px_pick[j] = x_true[j] - pick_jit * st;
		py_pick[j] = y_true[j] + pick_jit * ct;
		// Observed center from noisy E-step has 12 A outliers on j=10, 20, 30 and small +/-0.5 A jitter
		const double outlier = (j == 10 || j == 20 || j == 30) ? (j == 20 ? -12.0 : 12.0) : ((j % 3 == 0) ? 0.4 : -0.4);
		const double x_obs = x_true[j] - outlier * st;
		const double y_obs = y_true[j] + outlier * ct;
		// Local tangent angle in micrograph coordinates
		const double tx = ct - d_bend * st, ty = st + d_bend * ct;
		psi_true[j] = -RAD2DEG(atan2(ty, tx));

		MD.addObject();
		seg_rows[j] = j;
		MD.setValue(EMDL_MICROGRAPH_NAME, std::string("mic_001.mrc"), j);
		MD.setValue(EMDL_PARTICLE_HELICAL_TUBE_ID, 1, j);
		MD.setValue(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM, (RFLOAT)s, j);
		MD.setValue(EMDL_IMAGE_COORD_X, (RFLOAT)(px_pick[j] / apix), j);
		MD.setValue(EMDL_IMAGE_COORD_Y, (RFLOAT)(py_pick[j] / apix), j);
		MD.setValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, (RFLOAT)(px_pick[j] - x_obs), j);
		MD.setValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, (RFLOAT)(py_pick[j] - y_obs), j);
		MD.setValue(EMDL_ORIENT_PSI_PRIOR, (RFLOAT)(psi_true[j] + (j % 2 == 0 ? 3.5 : -3.5)), j);
	}
	HelicalContext hc;
	hc.verb = 0;
	hc.initialise(MD, seg_rows, 4, 180.0);
	hc.smoothFilamentTracks(MD, seg_rows, apix, 25.0, 2);

	double sq_pos = 0.0, sq_psi = 0.0;
	int cnt = 0;
	for (int j = 3; j < L - 3; j++)
	{
		RFLOAT pox = 0., poy = 0., ppsi = 0.;
		MD.getValue(EMDL_ORIENT_ORIGIN_X_PRIOR_ANGSTROM, pox, j);
		MD.getValue(EMDL_ORIENT_ORIGIN_Y_PRIOR_ANGSTROM, poy, j);
		MD.getValue(EMDL_ORIENT_PSI_PRIOR, ppsi, j);
		const double xs = px_pick[j] - pox, ys = py_pick[j] - poy;
		const double dx = xs - x_true[j], dy = ys - y_true[j];
		sq_pos += dx * dx + dy * dy;
		const double dpsi = std::remainder(ppsi - psi_true[j], 360.0);
		sq_psi += dpsi * dpsi;
		cnt++;
	}
	const double pos_rms = sqrt(sq_pos / cnt);
	const double psi_rms = sqrt(sq_psi / cnt);
	INFO("pos_rms=" << pos_rms << " A, psi_rms=" << psi_rms << " deg");
	CHECK(pos_rms < 1.5);
	CHECK(psi_rms < 1.5);
}

TEST_CASE("helical types: FilamentChains selects T=1 on homogeneous filaments without over-splitting", "[helix_context_types]")
{
	// Single polymorph cycling stochastically through states 0..5 across 180 filaments on 30 micrographs;
	// with tmin=1, tmax=5, scanChains + choose (and selectNrTypes) must stay at T=1.
	helixtypes::FilamentChains FC;
	FC.C = 6;
	FC.kappa = 0.25;
	std::mt19937 rng(777u);
	std::uniform_real_distribution<double> u(0., 1.);
	for (int f = 0; f < 180; f++)
	{
		FC.mics.push_back("mic_" + std::to_string(f % 30));
		std::vector<int> seq;
		int cur = (int)(rng() % FC.C);
		for (int s = 0; s < 45; s++)
		{
			if (s > 0)
			{
				const double r = u(rng);
				if (r < 0.75) cur = (cur + 1) % FC.C;
				else if (r < 0.90) cur = cur;
				else cur = (int)(rng() % FC.C);
			}
			seq.push_back((u(rng) < 0.05) ? (int)(rng() % FC.C) : cur);
		}
		FC.seqs.push_back(seq);
	}
	std::vector<double> score, agree, min_jacc;
	helixtypes::scanChains(FC, 1, 5, 3, 4, score, agree, &min_jacc);
	std::vector<double> eff = agree;
	for (int T = 2; T <= 5; T++)
		if (min_jacc[T] < 0.40) eff[T] = std::min(agree[T], min_jacc[T]);
	std::string why;
	int T = helixtypes::choose(score, eff, 1, 5, 0.01, 0.75, 0, why);
	if (T >= 2 && (score[T] <= score[1] + 0.01 || eff[T] < 0.55)) T = 1;
	INFO("homogeneous scores: " << score[1] << " " << score[2] << " " << score[3]
	     << " agree: " << agree[2] << " " << agree[3] << " min_jacc: " << min_jacc[2] << " " << min_jacc[3]);
	CHECK(T == 1);

	// Also verify HelicalContext::selectNrTypes(1, 5, ...) on homogeneous data returns 1
	HelicalContext hc;
	makeTypes(hc, 1, 6, 180, 40, 0.15, 91u);
	CHECK(hc.selectNrTypes(1, 5, 3, 0.01, 0.75, false) == 1);

	// And verify HelicalContext::selectNrTypes(1, 5, ...) on 3 distinct types returns 3 (not 1!)
	HelicalContext hc3;
	makeTypes(hc3, 3, 9, 240, 40, 0.10, 92u);
	CHECK(hc3.selectNrTypes(1, 5, 3, 0.01, 0.75, false) == 3);
}

TEST_CASE("helical types: constant labels give zero ARI and homogeneous filaments guard tmin>=2", "[helix_context_types]")
{
	std::vector<int> all0(100, 0), all1(100, 1);
	CHECK(helixtypes::adjustedRand(all0, all0) == Approx(0.0));
	CHECK(helixtypes::adjustedRand(all0, all1) == Approx(0.0));

	// Even when tmin=2, a truly homogeneous dataset must fall back to T=1 instead of collapsing into empty clusters
	HelicalContext hc;
	makeTypes(hc, 1, 8, 120, 40, 0.20, 105u);
	CHECK(hc.selectNrTypes(2, 6, 3, 0.01, 0.75, false) == 1);
	CHECK(hc.nr_types == 1);
}

TEST_CASE("helical context: gapped chain fragments of a filament share one unified type", "[helix_context_types]")
{
	// 2 polymorphs across 80 filaments x 28 segments broken by large (> max_gap_steps) gaps into 4 short fragments
	// of 7 segments each; classes 0..1 are shared crossovers (~40%), while {2,3} belong to type 0 and {4,5} to type 1.
	const int F = 80, K = 6, segs_per_fil = 28;
	MetaDataTable MD;
	std::vector<long int> rows;
	std::vector<int> truth_fil(F), seg_class;
	std::mt19937 rng(42u);
	std::uniform_real_distribution<double> u(0., 1.);
	long int row = 0;
	for (int f = 0; f < F; f++)
	{
		const int t = f % 2;
		truth_fil[f] = t;
		double track = 0.;
		for (int frag = 0; frag < 4; frag++)
		{
			if (frag > 0) track += 100.0; // > max_gap_steps * 15 A = 45 A gap splits into 4 chain fragments
			// Only one or two fragments show type-specific views {2,3} or {4,5}; others show mostly shared crossovers {0,1}
			const bool shared_only_frag = ((frag + f) % 2 == 1);
			for (int s = 0; s < segs_per_fil / 4; s++)
			{
				MD.addObject();
				MD.setValue(EMDL_MICROGRAPH_NAME, std::string("mic_") + std::to_string(f / 4) + ".mrc", row);
				MD.setValue(EMDL_PARTICLE_HELICAL_TUBE_ID, (f % 4) + 1, row);
				MD.setValue(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM, (RFLOAT)track, row);
				rows.push_back(row);
				int c;
				if (shared_only_frag && u(rng) < 0.85)
					c = (u(rng) < 0.5) ? 0 : 1;
				else if (u(rng) < 0.30)
					c = (u(rng) < 0.5) ? 0 : 1;
				else
					c = (t == 0) ? ((u(rng) < 0.5) ? 2 : 3) : ((u(rng) < 0.5) ? 4 : 5);
				seg_class.push_back(c);
				track += 15.0;
				row++;
			}
		}
	}

	HelicalContext hc;
	hc.verb = 0;
	hc.initialise(MD, rows, K, 150.0);
	REQUIRE(hc.chains.size() == (size_t)(F * 4));
	for (long int i = 0; i < hc.N; i++)
	{
		std::vector<RFLOAT> ll(K, -3.5);
		ll[seg_class[i]] = 0.0;
		hc.storeLogLik(i, ll);
	}
	CHECK(hc.selectNrTypes(1, 4, 3, 0.01, 0.75, false) == 2);
	hc.nr_types = 2;
	hc.initialised_model = false;
	hc.update();

	// Every filament's fragments must be assigned one coherent filament type matching truth
	int mixed = 0;
	std::vector<int> pred_fil(F, -1);
	for (int f = 0; f < F; f++)
	{
		int cnt[2] = {0, 0};
		for (long int i = (long int)f * segs_per_fil; i < (long int)(f + 1) * segs_per_fil; i++)
			if (hc.seg_type[i] == 0 || hc.seg_type[i] == 1)
				cnt[hc.seg_type[i]]++;
		if (cnt[0] > 0 && cnt[1] > 0) mixed++;
		pred_fil[f] = (cnt[1] > cnt[0]) ? 1 : 0;
	}
	INFO("mixed filaments=" << mixed << ", ARI=" << helixtypes::adjustedRand(pred_fil, truth_fil));
	CHECK(mixed == 0);
	CHECK(helixtypes::adjustedRand(pred_fil, truth_fil) > 0.95);
}

TEST_CASE("helical context: binned micrograph coordinates use original pixel size in smoothFilamentTracks", "[helix_context_types]")
{
	const RFLOAT orig_apix = 0.965;
	const RFLOAT img_apix = 7.72; // 8x binned extraction
	const int L = 24;
	const double step_A = 14.25;

	MetaDataTable MDopt, MD;
	MDopt.addObject();
	MDopt.setValue(EMDL_IMAGE_OPTICS_GROUP, 1, 0);
	MDopt.setValue(EMDL_MICROGRAPH_ORIGINAL_PIXEL_SIZE, orig_apix, 0);
	MDopt.setValue(EMDL_IMAGE_PIXEL_SIZE, img_apix, 0);

	std::vector<long int> rows(L);
	const double psi_deg = 25.0;
	const double th = -psi_deg * M_PI / 180.0;
	const double tx = cos(th), ty = sin(th);
	for (int j = 0; j < L; j++)
	{
		MD.addObject();
		rows[j] = j;
		const double x_A = 1000.0 + j * step_A * tx;
		const double y_A = 1500.0 + j * step_A * ty;
		MD.setValue(EMDL_MICROGRAPH_NAME, std::string("mic1.mrc"), j);
		MD.setValue(EMDL_PARTICLE_HELICAL_TUBE_ID, 1, j);
		MD.setValue(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM, (RFLOAT)(j * step_A), j);
		MD.setValue(EMDL_IMAGE_OPTICS_GROUP, 1, j);
		MD.setValue(EMDL_IMAGE_COORD_X, (RFLOAT)(x_A / orig_apix), j);
		MD.setValue(EMDL_IMAGE_COORD_Y, (RFLOAT)(y_A / orig_apix), j);
		MD.setValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, (RFLOAT)(j % 2 == 0 ? 1.5 : -1.5), j);
		MD.setValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, (RFLOAT)(j % 2 == 0 ? -1.0 : 1.0), j);
		MD.setValue(EMDL_ORIENT_PSI_PRIOR, (RFLOAT)psi_deg, j);
	}

	HelicalContext hc;
	hc.verb = 0;
	hc.initialise(MD, rows, 4, 250.0);
	hc.smoothFilamentTracks(MD, rows, img_apix, 25.0, 2, &MDopt);

	for (int j = 0; j < L; j++)
	{
		RFLOAT oxp = 0., oyp = 0.;
		MD.getValue(EMDL_ORIENT_ORIGIN_X_PRIOR_ANGSTROM, oxp, j);
		MD.getValue(EMDL_ORIENT_ORIGIN_Y_PRIOR_ANGSTROM, oyp, j);
		CHECK(std::fabs(oxp) < 20.0);
		CHECK(std::fabs(oyp) < 20.0);
	}
}

TEST_CASE("helical types: large-C sparse scanChains recovers 3 types identically across OpenMP threads", "[helix_context_types]")
{
	const int C = 180, F = 1200, len = 45, Ttrue = 3;
	helixtypes::FilamentChains X;
	X.C = C;
	X.kappa = 0.20;
	X.seqs.resize(F);
	X.mics.resize(F);
	std::vector<int> truth(F);
	std::mt19937 rng(77u);
	std::uniform_real_distribution<double> u(0., 1.);
	const int per_t = 30;
	for (int f = 0; f < F; f++)
	{
		const int t = f % Ttrue;
		truth[f] = t;
		X.mics[f] = "m" + std::to_string(f % 30);
		X.seqs[f].resize(len);
		for (int s = 0; s < len; s++)
		{
			if (u(rng) < 0.25)
				X.seqs[f][s] = (s + (int)(u(rng) * 3)) % 15; // shared crossover states 0..14
			else
				X.seqs[f][s] = 15 + t * per_t + (s % per_t);
		}
	}

	std::vector<double> sc1, ag1, mj1, sc4, ag4, mj4;
#ifdef _OPENMP
	const int prev_thr = omp_get_max_threads();
	omp_set_num_threads(1);
#endif
	helixtypes::scanChains(X, 1, 5, 3, 2, sc1, ag1, &mj1);
#ifdef _OPENMP
	omp_set_num_threads(4);
#endif
	helixtypes::scanChains(X, 1, 5, 3, 2, sc4, ag4, &mj4);
#ifdef _OPENMP
	omp_set_num_threads(prev_thr);
#endif

	for (int t = 1; t <= 5; t++)
	{
		CHECK(sc1[t] == sc4[t]);
		CHECK(ag1[t] == ag4[t]);
		CHECK(mj1[t] == mj4[t]);
	}

	std::string why;
	const int best = helixtypes::choose(sc4, ag4, 1, 5, 0.0025, 0.75, 0, why);
	CHECK(best == 3);
	const std::vector<int> pred = helixtypes::fitChainTypes(X, 3, 1u);
	CHECK(helixtypes::adjustedRand(pred, truth) > 0.98);
}

TEST_CASE("HelicalContext: phase ring prior and CV-gated twist-rate split", "[helix_context_types]")
{
	const int K = 4;
	const int F = 48, S = 44;
	const RFLOAT step = 15.0;

	// Build shared-view synthetic filaments with bimodal crossover periods (P=450 A vs P=900 A)
	auto buildContext = [&](bool bimodal, std::vector<int> &truth_type, std::vector<RFLOAT> &psi,
	                        std::vector<RFLOAT> &psip, std::vector<int> &true_cls, HelicalContext &ctx)
	{
		const long int N = (long int)F * S;
		truth_type.assign(F, 0);
		psi.assign(N, 0.);
		psip.assign(N, 0.);
		true_cls.assign(N, 0);

		MetaDataTable MD;
		MD.addLabel(EMDL_MICROGRAPH_NAME);
		MD.addLabel(EMDL_PARTICLE_HELICAL_TUBE_ID);
		MD.addLabel(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM);
		std::vector<long int> rows(N);

		for (int f = 0; f < F; f++)
		{
			const int t = bimodal ? (f % 2) : 0;
			truth_type[f] = t;
			const double P = bimodal ? ((t == 0) ? 450.0 : 900.0) : 700.0;
			for (int s = 0; s < S; s++)
			{
				const long int i = (long int)f * S + s;
				rows[i] = i;
				MD.addObject();
				MD.setValue(EMDL_MICROGRAPH_NAME, std::string("mic1.mrc"), i);
				MD.setValue(EMDL_PARTICLE_HELICAL_TUBE_ID, f + 1, i);
				MD.setValue(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM, (RFLOAT)(s * step), i);
				const int kc = ((int)std::floor(std::fmod(s * step, P) / (P / K))) % K;
				true_cls[i] = kc;
			}
		}

		ctx.verb = 0;
		ctx.nr_types = 1;
		ctx.use_phase_ring = true;
		ctx.use_twist_rate_split = true;
		ctx.initialise(MD, rows, K, 200.0);

		std::vector<RFLOAT> ll(K);
		for (long int i = 0; i < N; i++)
		{
			for (int k = 0; k < K; k++)
				ll[k] = std::log(0.08 / (K - 1));
			ll[true_cls[i]] = std::log(0.92);
			ctx.storeLogLik(i, ll);
		}
		ctx.update();
		ctx.fitPeriods(psi, psip);
	};

	std::vector<int> truth_bi, true_cls_bi;
	std::vector<RFLOAT> psi_bi, psip_bi;
	HelicalContext ctx_bi;
	buildContext(true, truth_bi, psi_bi, psip_bi, true_cls_bi, ctx_bi);

	INFO("type_period[0]=" << ctx_bi.type_period[0] << " type_period_R[0]=" << ctx_bi.type_period_R[0]
	     << " fil0=" << ctx_bi.fil_period[0] << " (R=" << ctx_bi.fil_period_R[0] << ")"
	     << " fil1=" << ctx_bi.fil_period[1] << " (R=" << ctx_bi.fil_period_R[1] << ")");

	REQUIRE(ctx_bi.nr_types == 1);
	// Phase-ring prior favors the true periodic class over the opposite phase class
	ctx_bi.applyPhaseRingPriors(psi_bi, psip_bi);
	long int boosted = 0, total_checked = 0;
	for (long int i = 0; i < ctx_bi.N; i++)
	{
		const int kt = true_cls_bi[i];
		const int kop = (kt + 2) % K;
		if (ctx_bi.prior[(size_t)i * K + kt] > ctx_bi.prior[(size_t)i * K + kop])
			boosted++;
		total_checked++;
	}
	CHECK(boosted > (total_checked * 85) / 100);

	// Cross-validated 2-Gaussian mixture on filament periods splits into 2 types cleanly
	const int nt_bi = ctx_bi.splitTypesByTwistRate(4);
	REQUIRE(nt_bi == 2);
	std::vector<int> pred_fil_bi(F, 0);
	for (int f = 0; f < F; f++)
	{
		const long int i = (long int)f * S + S / 2;
		pred_fil_bi[f] = ctx_bi.seg_type[i];
	}
	CHECK(helixtypes::adjustedRand(pred_fil_bi, truth_bi) > 0.90);

	// Unimodal control must remain 1 type
	std::vector<int> truth_uni, true_cls_uni;
	std::vector<RFLOAT> psi_uni, psip_uni;
	HelicalContext ctx_uni;
	buildContext(false, truth_uni, psi_uni, psip_uni, true_cls_uni, ctx_uni);
	CHECK(ctx_uni.splitTypesByTwistRate(4) == 1);
}

TEST_CASE("autoMaskFromMap: generates 2D filament contour mask in [0, 1] with ZSIZE == 1", "[helix_context_types][auto_mask]")
{
	MultidimArray<RFLOAT> ref(1, 64, 64);
	ref.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY2D(ref)
	{
		const RFLOAT ay = std::fabs((RFLOAT)i);
		const RFLOAT ax = std::fabs((RFLOAT)j);
		if (ay <= 5.0 && ax <= 24.0)
			A2D_ELEM(ref, i, j) = 10.0;
		else if (ay <= 10.0 && ax <= 24.0)
			A2D_ELEM(ref, i, j) = -3.0; // negative CTF side-lobe
		else
			A2D_ELEM(ref, i, j) = 0.0;
	}
	MultidimArray<RFLOAT> pos_ref = ref;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(pos_ref)
		if (DIRECT_MULTIDIM_ELEM(pos_ref, n) < 0.)
			DIRECT_MULTIDIM_ELEM(pos_ref, n) = 0.;

	AutoMaskOptions opt;
	opt.lowpass = 10.0;
	opt.soft_edge = 2.0;
	opt.nr_seeds = 16;
	opt.verb = false;
	MultidimArray<RFLOAT> mask;
	AutoMaskReport report;
	autoMaskFromMap(pos_ref, 4.0, 10.0, opt, mask, &report, 1);

	REQUIRE(mask.sameShape(pos_ref, true));
	CHECK(ZSIZE(mask) == 1);
	CHECK(mask.computeMin() >= 0.0 - 1e-6);
	CHECK(mask.computeMax() <= 1.0 + 1e-6);
	CHECK(A2D_ELEM(mask, 0, 0) > 0.9);
	CHECK(A2D_ELEM(mask, -28, -28) < 0.05);
}

TEST_CASE("relion_helix_subclassify: --prepare_only writes readable re-centered .vstack descriptors", "[helix_context_types][subclassify]")
{
	char exe_buf[PATH_MAX];
	const ssize_t exe_len = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
	REQUIRE(exe_len > 0);
	exe_buf[exe_len] = '\0';
	std::string bin_path(exe_buf);
	const size_t slash = bin_path.find_last_of('/');
	REQUIRE(slash != std::string::npos);
	bin_path = bin_path.substr(0, slash + 1) + "relion_helix_subclassify";
	REQUIRE(access(bin_path.c_str(), X_OK) == 0);

	const std::string tmpdir = "tmp_helix_subclassify_" + integerToString((int)std::clock());
	vcache::removeTree(tmpdir);
	vcache::makeDirs(tmpdir);

	// 1. Write synthetic 256x256 micrograph with horizontal filament signal near y = 124
	Image<RFLOAT> Imic(256, 256);
	Imic.setSamplingRateInHeader(2.0);
	FOR_ALL_ELEMENTS_IN_ARRAY2D(Imic())
	{
		const double dy = (double)i - 124.0;
		const double ridge = 5.0 * std::exp(-0.5 * (dy / 4.0) * (dy / 4.0)) * (1.0 + 0.3 * std::cos(0.2 * j));
		const double bg = 0.05 * std::sin(0.11 * i + 0.07 * j);
		A2D_ELEM(Imic(), i, j) = (RFLOAT)(ridge + bg);
	}
	Imic.write(tmpdir + "/mic1.mrc");

	// 2. Write Stage-1 STAR with flipped polarity (psi=180, psi_prior=0) and +8 A Y-offset (= 4 px at 2 A/px)
	MetaDataTable MDopt, MD;
	MDopt.setName("optics");
	MDopt.addObject();
	MDopt.setValue(EMDL_IMAGE_OPTICS_GROUP, 1, 0);
	MDopt.setValue(EMDL_IMAGE_OPTICS_GROUP_NAME, std::string("opticsGroup1"), 0);
	MDopt.setValue(EMDL_MICROGRAPH_ORIGINAL_PIXEL_SIZE, 2.0, 0);
	MDopt.setValue(EMDL_IMAGE_PIXEL_SIZE, 4.0, 0);
	MDopt.setValue(EMDL_IMAGE_SIZE, 32, 0);
	MDopt.setValue(EMDL_IMAGE_DIMENSIONALITY, 2, 0);
	MDopt.setValue(EMDL_CTF_VOLTAGE, 300.0, 0);
	MDopt.setValue(EMDL_CTF_CS, 2.7, 0);
	MDopt.setValue(EMDL_CTF_Q0, 0.1, 0);
	MDopt.setValue(EMDL_OPTIMISER_DATA_ARE_CTF_PREMULTIPLIED, false, 0);

	MD.setName("particles");
	for (int s = 0; s < 8; s++)
	{
		MD.addObject();
		MD.setValue(EMDL_MICROGRAPH_NAME, std::string("mic1.mrc"), s);
		MD.setValue(EMDL_IMAGE_NAME, integerToString(s + 1, 6) + "@" + tmpdir + "/dummy.mrcs", s);
		MD.setValue(EMDL_IMAGE_OPTICS_GROUP, 1, s);
		MD.setValue(EMDL_PARTICLE_HELICAL_TUBE_ID, 1, s);
		MD.setValue(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM, (RFLOAT)(s * 20.0), s);
		MD.setValue(EMDL_PARTICLE_HELICAL_TYPE, 1, s);
		MD.setValue(EMDL_IMAGE_COORD_X, (RFLOAT)(70.0 + s * 10.0), s);
		MD.setValue(EMDL_IMAGE_COORD_Y, (RFLOAT)128.0, s);
		MD.setValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, (RFLOAT)4.0, s);
		MD.setValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, (RFLOAT)8.0, s);
		MD.setValue(EMDL_ORIENT_TILT_PRIOR, (RFLOAT)90.0, s);
		MD.setValue(EMDL_ORIENT_PSI_PRIOR, (RFLOAT)0.0, s);
		MD.setValue(EMDL_ORIENT_PSI, (RFLOAT)180.0, s);
		MD.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)15000.0, s);
		MD.setValue(EMDL_CTF_DEFOCUSV, (RFLOAT)14500.0, s);
		MD.setValue(EMDL_CTF_DEFOCUS_ANGLE, (RFLOAT)20.0, s);
		MD.setValue(EMDL_CTF_BFACTOR, (RFLOAT)0.0, s);
		MD.setValue(EMDL_CTF_SCALEFACTOR, (RFLOAT)1.0, s);
		MD.setValue(EMDL_CTF_PHASESHIFT, (RFLOAT)0.0, s);
	}
	const std::string in_star = tmpdir + "/stage1_data.star";
	{
		std::ofstream fh(in_star.c_str());
		MDopt.write(fh);
		MD.write(fh);
	}

	const std::string out_dir = tmpdir + "/out";
	const std::string cmd = bin_path + " --i " + in_star + " --o " + out_dir + " --project_dir " + tmpdir
	                      + " --extract_size 64 --scale 32 --K 4 --min_segs 4 --dont_invert_contrast --prepare_only"
	                      + " > " + tmpdir + "/sub.out 2> " + tmpdir + "/sub.err";
	REQUIRE(std::system(cmd.c_str()) == 0);

	MetaDataTable MDopt_out, MD_out;
	MDopt_out.read(out_dir + "/type_1/particles.star", "optics");
	MD_out.read(out_dir + "/type_1/particles.star", "particles");
	REQUIRE(MD_out.numberOfObjects() == 8);
	bool prem = false;
	CHECK(MDopt_out.getValue(EMDL_OPTIMISER_DATA_ARE_CTF_PREMULTIPLIED, prem, 0));
	CHECK(prem);

	RFLOAT ny_coord = 0., res_oy = 999., new_psip = 0.;
	FileName vname;
	MD_out.getValue(EMDL_IMAGE_COORD_Y, ny_coord, 0);
	MD_out.getValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, res_oy, 0);
	MD_out.getValue(EMDL_ORIENT_PSI_PRIOR, new_psip, 0);
	MD_out.getValue(EMDL_IMAGE_NAME, vname, 0);
	CHECK(ny_coord == Approx(124.0).margin(1e-4));
	CHECK(res_oy == Approx(0.0).margin(1e-4));
	CHECK(std::fabs(std::remainder(new_psip - 180.0, 360.0)) < 1e-3);
	CHECK(vname.contains(".vstack"));

	Image<RFLOAT> Ipart;
	Ipart.read(vname);
	CHECK(XSIZE(Ipart()) == 32);
	CHECK(YSIZE(Ipart()) == 32);
	CHECK(Ipart().computeStddev() > 1e-5);

	// Verify RELION_OUTPUT_NODES.star written in --prepare_only mode
	REQUIRE(exists(out_dir + "/RELION_OUTPUT_NODES.star"));
	{
		MetaDataTable MDnodes;
		MDnodes.read(out_dir + "/RELION_OUTPUT_NODES.star", "output_nodes");
		REQUIRE(MDnodes.numberOfObjects() == 1);
		FileName nname;
		std::string ntype;
		MDnodes.getValue(EMDL_PIPELINE_NODE_NAME, nname, 0);
		MDnodes.getValue(EMDL_PIPELINE_NODE_TYPE_LABEL, ntype, 0);
		CHECK(nname == out_dir + "/type_1/particles.star");
		CHECK(ntype == "ParticleGroupMetadata.star.relion.helicalsegments");
	}

	// Verify full Stage-2 execution via _optimiser.star input writes merged_particles.star and RELION_OUTPUT_NODES.star
	const std::string opt_star = tmpdir + "/stage1_optimiser.star";
	{
		MetaDataTable MDopt_gen;
		MDopt_gen.setIsList(true);
		MDopt_gen.setName("optimiser_general");
		MDopt_gen.addObject();
		MDopt_gen.setValue(EMDL_OPTIMISER_DATA_STARFILE, FileName("stage1_data.star"), 0);
		MDopt_gen.write(opt_star);
	}
	const std::string full_out = tmpdir + "/out_full";
	const std::string full_cmd = bin_path + " --i " + opt_star + " --o " + full_out + " --project_dir " + tmpdir
	                           + " --extract_size 64 --scale 32 --outer_diameter 60 --K 2 --iter 1 --min_segs 4 --dont_invert_contrast"
	                           + " --auto_premultiply_ctf --dont_auto_mask --dont_keep_all_iterations --gpu none"
	                           + " --j 1 > " + tmpdir + "/full.out 2> " + tmpdir + "/full.err";
	REQUIRE(std::system(full_cmd.c_str()) == 0);
	REQUIRE(exists(full_out + "/merged_particles.star"));
	REQUIRE(exists(full_out + "/RELION_OUTPUT_NODES.star"));
	{
		MetaDataTable MDnodes;
		MDnodes.read(full_out + "/RELION_OUTPUT_NODES.star", "output_nodes");
		REQUIRE(MDnodes.numberOfObjects() == 3);
		bool has_merged = false, has_opt = false, has_data = false;
		for (long int i = 0; i < (long int)MDnodes.numberOfObjects(); i++)
		{
			FileName nname;
			std::string ntype;
			MDnodes.getValue(EMDL_PIPELINE_NODE_NAME, nname, i);
			MDnodes.getValue(EMDL_PIPELINE_NODE_TYPE_LABEL, ntype, i);
			if (nname == full_out + "/merged_particles.star" &&
			    ntype == "ParticleGroupMetadata.star.relion.class2d.helicalsegments")
				has_merged = true;
			if (nname == full_out + "/type_1/run_it001_optimiser.star" &&
			    ntype == "OptimiserData.star.relion.class2d")
				has_opt = true;
			if (nname == full_out + "/type_1/run_it001_data.star" &&
			    ntype == "ParticleGroupMetadata.star.relion.class2d.helicalsegments")
				has_data = true;
		}
		CHECK(has_merged);
		CHECK(has_opt);
		CHECK(has_data);
	}

	vcache::removeTree(tmpdir);
}

TEST_CASE("helical types: 8-type uneven mixture (with 1.5-2% rare types) and T=1 large-C negative control", "[helix_context_types]")
{
	// 1. 8-type uneven mixture over C=200 states with shared crossover states (0..19)
	//    and rare 1.5-2% minority types (types 6 and 7).
	{
		const int Ttrue = 8, C = 200;
		const int counts[8] = {280, 220, 170, 120, 80, 75, 20, 15}; // total F = 980; rare types 2.0% and 1.5%
		const int F = 980;
		helixtypes::FilamentChains X;
		X.C = C;
		X.kappa = 0.25;
		X.seqs.reserve(F);
		X.mics.reserve(F);
		std::vector<int> truth;
		truth.reserve(F);
		std::mt19937 rng(2026u);
		std::uniform_real_distribution<double> u(0., 1.);
		const int n_shared = 20;
		const int per_t = (C - n_shared) / Ttrue; // 22 states per type
		for (int t = 0; t < Ttrue; t++)
		{
			for (int k = 0; k < counts[t]; k++)
			{
				const int f = (int)truth.size();
				truth.push_back(t);
				X.mics.push_back("m" + std::to_string(f % 45));
				const int len = 36 + (f % 15);
				std::vector<int> seq(len);
				for (int s = 0; s < len; s++)
				{
					if (u(rng) < 0.25)
						seq[s] = (s + (int)(u(rng) * 3)) % n_shared;
					else
						seq[s] = n_shared + t * per_t + ((s + (u(rng) < 0.1 ? 1 : 0)) % per_t);
				}
				X.seqs.push_back(std::move(seq));
			}
		}

		std::vector<double> score, agree, min_jacc, w_jacc, sec_jacc;
		helixtypes::scanChains(X, 1, 10, 3, 3, score, agree, &min_jacc, &w_jacc, &sec_jacc);
		INFO("8-type uneven: agree[8]=" << agree[8] << " min_jacc[8]=" << min_jacc[8]
		     << " w_jacc[8]=" << w_jacc[8] << " sec_jacc[8]=" << sec_jacc[8]);
		CHECK(agree[8] > 0.90);
		CHECK(w_jacc[8] > 0.90);

		HelicalContext hc;
		hc.verb = 0;
		hc.K = C;
		hc.N = 0;
		for (const auto &s : X.seqs) hc.N += (long int)s.size();
		hc.step = 14.25;
		hc.diameter = 200.0;
		hc.temper = 0.25;
		hc.filament_name.resize(F);
		hc.filament_of_chain.resize(F);
		hc.chains.resize(F);
		hc.seg_chain.resize(hc.N);
		hc.seg_pos.resize(hc.N);
		hc.loglik.assign((size_t)hc.N * C, -8.f);
		long int idx = 0;
		for (int f = 0; f < F; f++)
		{
			hc.filament_name[f] = X.mics[f] + ":" + std::to_string(f + 1);
			hc.filament_of_chain[f] = f;
			hc.chains[f].resize(X.seqs[f].size());
			for (size_t s = 0; s < X.seqs[f].size(); s++)
			{
				hc.chains[f][s] = idx;
				hc.seg_chain[idx] = f;
				hc.seg_pos[idx] = (long int)s;
				hc.loglik[(size_t)idx * C + X.seqs[f][s]] = 0.f;
				idx++;
			}
		}
		const int chosen = hc.selectNrTypes(2, 10, 3, 0.01, 0.75, false);
		CHECK(chosen == 8);

		const std::vector<int> pred = helixtypes::fitChainTypes(X, chosen, 1u);
		const double ari = helixtypes::adjustedRand(pred, truth);
		INFO("8-type uneven fitChainTypes ARI=" << ari);
		CHECK(ari > 0.95);
	}

	// 2. Homogeneous T=1 negative control over large C=120 state space
	//    (12 crossover views x 10 random defocus/noise sub-states per view)
	{
		const int F = 450, C = 120, n_views = 12, n_vars = 10;
		HelicalContext hc;
		hc.verb = 0;
		hc.K = C;
		hc.step = 14.25;
		hc.diameter = 220.0;
		hc.temper = 0.12;
		std::mt19937 rng(909u);
		std::uniform_real_distribution<double> u(0., 1.);
		std::vector<long int> rows;
		std::vector<int> obs_cls;
		MetaDataTable MD;
		long int row = 0;
		for (int f = 0; f < F; f++)
		{
			const int len = 36 + (f % 12);
			int view = (int)(rng() % n_views);
			for (int s = 0; s < len; s++)
			{
				MD.addObject();
				MD.setValue(EMDL_MICROGRAPH_NAME, std::string("mic_") + std::to_string(f % 45) + ".mrc", row);
				MD.setValue(EMDL_PARTICLE_HELICAL_TUBE_ID, (f / 45) + 1, row);
				MD.setValue(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM, (RFLOAT)(s * 14.25), row);
				rows.push_back(row);
				if (s > 0)
				{
					const double r = u(rng);
					if (r < 0.70) view = (view + 1) % n_views;
					else if (r >= 0.85) view = (int)(rng() % n_views);
				}
				const int obs = (u(rng) < 0.10) ? (int)(rng() % C) : (view * n_vars + (int)(rng() % n_vars));
				obs_cls.push_back(obs);
				row++;
			}
		}
		hc.initialise(MD, rows, C, 220.0);
		for (long int i = 0; i < hc.N; i++)
		{
			std::vector<RFLOAT> ll(C, -6.0);
			ll[obs_cls[i]] = 0.0;
			hc.storeLogLik(i, ll);
		}
		CHECK(hc.selectNrTypes(2, 10, 3, 0.01, 0.75, false) == 1);
		std::vector<RFLOAT> psi(hc.N, 0.), psip(hc.N, 0.);
		CHECK(hc.runHierarchicalTyping(psi, psip, 2, 10, 16, 20, 3, 0.01, 0.75, true) == 1);
	}
}

TEST_CASE("HelicalContext: --hierarchical tree splitting, rlnHelicalTypePosterior, centerline smoothing, and coherent polarity", "[helix_context_types][helix_subclassify]")
{
	// 5 hierarchical leaves across F=300 filaments (K=18 classes):
	//   Supergroup A (180 fils): shares dominant crossover states 0..3 (45%), diverges at sub-types:
	//     Leaf A0 (90 fils): states 4..5
	//     Leaf A1 (60 fils): states 6..7
	//     Leaf A2 (30 fils): states 8..9 (rare sub-type)
	//   Supergroup B (120 fils): cyclic classes 10..17, splits by twist rate into:
	//     Leaf B0 (60 fils): period P = 420 A
	//     Leaf B1 (60 fils): period P = 840 A
	const int F = 300, K = 18, S = 44;
	const RFLOAT step = 15.0, apix = 2.0;
	MetaDataTable MDopt, MD;
	MDopt.setName("optics");
	MDopt.addObject();
	MDopt.setValue(EMDL_IMAGE_OPTICS_GROUP, 1, 0);
	MDopt.setValue(EMDL_MICROGRAPH_ORIGINAL_PIXEL_SIZE, apix, 0);
	MDopt.setValue(EMDL_IMAGE_PIXEL_SIZE, (RFLOAT)6.0, 0);

	std::mt19937 rng(4242u);
	std::uniform_real_distribution<double> u(0., 1.);
	std::vector<long int> rows;
	std::vector<int> truth_fil(F, 0), seg_cls;
	std::vector<RFLOAT> psi_vec, psip_vec;
	long int row = 0;

	for (int f = 0; f < F; f++)
	{
		int tf = 0;
		if (f < 90) tf = 0;
		else if (f < 150) tf = 1;
		else if (f < 180) tf = 2;
		else if (f < 240) tf = 3;
		else tf = 4;
		truth_fil[f] = tf;

		const bool flip_pol = (f % 2 == 1);
		const double psip_deg = 0.0;
		const double psi_deg = flip_pol ? 180.0 : 0.0;
		const std::string mic = "mic_" + std::to_string(f % 30) + ".mrc";
		const int tube = (f / 30) + 1;

		for (int s = 0; s < S; s++)
		{
			MD.addObject();
			rows.push_back(row);
			const double x_true_A = 400.0 + s * step;
			const double y_true_A = 600.0;
			// Pick offset +4 A, plus a large +18 A outlier spike at s==15 and s==30
			const double spike = (s == 15 || s == 30) ? 18.0 : 0.0;
			MD.setValue(EMDL_MICROGRAPH_NAME, mic, row);
			MD.setValue(EMDL_PARTICLE_HELICAL_TUBE_ID, tube, row);
			MD.setValue(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM, (RFLOAT)(s * step), row);
			MD.setValue(EMDL_IMAGE_OPTICS_GROUP, 1, row);
			MD.setValue(EMDL_IMAGE_COORD_X, (RFLOAT)(x_true_A / apix), row);
			MD.setValue(EMDL_IMAGE_COORD_Y, (RFLOAT)((y_true_A + 4.0) / apix), row);
			MD.setValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, (RFLOAT)0.0, row);
			MD.setValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, (RFLOAT)(4.0 + spike), row);
			MD.setValue(EMDL_ORIENT_PSI_PRIOR, (RFLOAT)psip_deg, row);
			MD.setValue(EMDL_ORIENT_PSI, (RFLOAT)psi_deg, row);
			MD.setValue(EMDL_PARTICLE_PMAX, (RFLOAT)0.9, row);
			psi_vec.push_back((RFLOAT)psi_deg);
			psip_vec.push_back((RFLOAT)psip_deg);

			int kc = 0;
			if (tf <= 2)
			{
				if (u(rng) < 0.45) kc = s % 4;
				else kc = 4 + tf * 2 + (s % 2);
			}
			else
			{
				const double P = (tf == 3) ? 420.0 : 840.0;
				const int ncyc = 8;
				kc = 10 + (((int)std::floor(std::fmod(s * step, P) / (P / ncyc))) % ncyc);
			}
			seg_cls.push_back(kc);
			MD.setValue(EMDL_PARTICLE_CLASS, kc + 1, row);
			row++;
		}
	}

	HelicalContext ctx;
	ctx.verb = 0;
	ctx.nr_types = 2;
	ctx.initialise(MD, rows, K, 200.0);
	for (long int i = 0; i < ctx.N; i++)
	{
		std::vector<RFLOAT> ll(K, std::log(0.05 / (K - 1)));
		ll[seg_cls[i]] = std::log(0.95);
		ctx.storeLogLik(i, ll);
	}

	const int L = ctx.runHierarchicalTyping(psi_vec, psip_vec, 2, 8, 12, 20, 3, 0.01, 0.75, true);
	INFO("hierarchical leaf count=" << L);
	CHECK(L == 5);

	std::vector<int> pred_fil(F, 0);
	for (int f = 0; f < F; f++)
		pred_fil[f] = ctx.seg_type[(long int)f * S + S / 2];
	const double ari = helixtypes::adjustedRand(pred_fil, truth_fil);
	INFO("hierarchical 5-leaf ARI=" << ari);
	CHECK(ari > 0.90);

	ctx.smoothAndRecenterCoordinates(MD, rows, &seg_cls, (RFLOAT)6.0, (RFLOAT)40.0, &MDopt, true, false, true);
	ctx.writeLabels(MD, rows, &seg_cls);

	REQUIRE(MD.containsLabel(EMDL_PARTICLE_HELICAL_TYPE_POSTERIOR));
	for (int f = 0; f < F; f++)
	{
		const long int mid_row = (long int)f * S + 15; // outlier spike row
		RFLOAT cy = 0., oy = 999., psip = 999., fpost = 0.;
		MD.getValue(EMDL_IMAGE_COORD_Y, cy, mid_row);
		MD.getValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, oy, mid_row);
		MD.getValue(EMDL_ORIENT_PSI_PRIOR, psip, mid_row);
		MD.getValue(EMDL_PARTICLE_HELICAL_TYPE_POSTERIOR, fpost, mid_row);
		CHECK(cy == Approx(600.0 / apix).margin(1.0)); // outlier rejected, smoothed onto y_true
		CHECK(oy == Approx(0.0).margin(1e-6));
		const double exp_psi = (f % 2 == 1) ? 180.0 : 0.0;
		CHECK(std::fabs(std::remainder(psip - exp_psi, 360.0)) < 2.0);
		CHECK(fpost > 0.80);
	}

	// calibrated posteriors do not decay with depth in the tree: the median over filaments of each leaf stays high
	std::vector<std::vector<double> > leaf_post(L);
	for (int f = 0; f < F; f++)
		leaf_post[pred_fil[f]].push_back((double)ctx.fil_type_posterior[f]);
	for (int l = 0; l < L; l++)
	{
		REQUIRE(!leaf_post[l].empty());
		std::sort(leaf_post[l].begin(), leaf_post[l].end());
		const double med = leaf_post[l][leaf_post[l].size() / 2];
		INFO("leaf " << l << " median filament posterior " << med);
		CHECK(med > 0.75);
	}
}

TEST_CASE("autoMask2DClass: coarse shape low-pass first, rim artifact fallback, and hysteresis", "[helix_context_types][auto_mask]")
{
	std::mt19937 rng(2026);
	std::normal_distribution<double> gaus(0.0, 0.3);

	// 1. Unclipped 2D filament with dark CTF side-lobes + Gaussian background noise
	MultidimArray<RFLOAT> ref(1, 64, 64);
	ref.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY2D(ref)
	{
		const RFLOAT ay = std::fabs((RFLOAT)i);
		const RFLOAT ax = std::fabs((RFLOAT)j);
		double val = gaus(rng);
		if (ay <= 3.0 && ax <= 24.0)
			val += 8.0;
		else if (ay <= 6.0 && ax <= 24.0)
			val -= 3.0;
		A2D_ELEM(ref, i, j) = val;
	}

	AutoMask2DOptions opt;
	opt.lowpass = 10.0;       // fine lowpass should be raised to min_lowpass = 35 A
	opt.min_lowpass = 35.0;
	opt.extend_angst = 18.0;  // 4.5 px envelope extension @ 4 A/px
	opt.soft_edge = 3.0;
	opt.helix_tube_diam = 100.0;
	MultidimArray<RFLOAT> mask1;
	AutoMaskReport rep;
	REQUIRE(autoMask2DClass(ref, 4.0, 10.0, opt, mask1, NULL, &rep, 1));
	CHECK(rep.lowpass >= 35.0 - 1e-4);
	CHECK(A2D_ELEM(mask1, 0, 0) > 0.99);
	CHECK(A2D_ELEM(mask1, 5, 0) > 0.50);    // 18 A dilation covers beyond +/-3 px core
	CHECK(A2D_ELEM(mask1, -28, -28) < 0.05);

	// 2. Small off-axis corner rim artifact patch inside helical mode is rejected back to all-ones
	MultidimArray<RFLOAT> rim(1, 64, 64);
	rim.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY2D(rim)
	{
		const bool corner = (i <= -23 && i >= -27 && j <= -23 && j >= -27);
		A2D_ELEM(rim, i, j) = corner ? 10.0 : 0.0;
	}
	MultidimArray<RFLOAT> mask_rim;
	CHECK_FALSE(autoMask2DClass(rim, 4.0, 10.0, opt, mask_rim, NULL, NULL, 1));
	CHECK(mask_rim.computeMin() == Approx(1.0).margin(1e-6));
	CHECK(mask_rim.computeMax() == Approx(1.0).margin(1e-6));

	// 3. Asymmetric hysteresis: narrow gradually, widen immediately
	MultidimArray<RFLOAT> wide(1, 64, 64), narrow(1, 64, 64);
	wide.setXmippOrigin();
	narrow.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY2D(wide)
	{
		const RFLOAT ay = std::fabs((RFLOAT)i);
		const RFLOAT ax = std::fabs((RFLOAT)j);
		A2D_ELEM(wide, i, j) = (ay <= 9.0 && ax <= 24.0) ? 10.0 : 0.0;
		A2D_ELEM(narrow, i, j) = (ay <= 2.0 && ax <= 24.0) ? 10.0 : 0.0;
	}
	AutoMask2DOptions opt_nohyst = opt;
	opt_nohyst.helix_tube_diam = 120.0;
	opt_nohyst.extend_angst = 8.0;
	opt_nohyst.soft_edge = 2.0;
	MultidimArray<RFLOAT> m_wide, m_narrow_raw, m_narrow_hyst, m_widen_back;
	REQUIRE(autoMask2DClass(wide, 4.0, 10.0, opt_nohyst, m_wide, NULL, NULL, 1));
	REQUIRE(autoMask2DClass(narrow, 4.0, 10.0, opt_nohyst, m_narrow_raw, NULL, NULL, 1));
	REQUIRE(autoMask2DClass(narrow, 4.0, 10.0, opt_nohyst, m_narrow_hyst, &m_wide, NULL, 1));
	// Where m_wide == 1 and m_narrow_raw < 0.1 (e.g., i = 9, j = 0), hysteresis keeps >= 0.5
	CHECK(A2D_ELEM(m_wide, 9, 0) > 0.9);
	CHECK(A2D_ELEM(m_narrow_raw, 9, 0) < 0.2);
	CHECK(A2D_ELEM(m_narrow_hyst, 9, 0) >= 0.5 * A2D_ELEM(m_wide, 9, 0) - 1e-6);
	// Widening from m_narrow_raw back to wide happens immediately without attenuation
	REQUIRE(autoMask2DClass(wide, 4.0, 10.0, opt_nohyst, m_widen_back, &m_narrow_raw, NULL, 1));
	CHECK(A2D_ELEM(m_widen_back, 9, 0) == Approx(A2D_ELEM(m_wide, 9, 0)).margin(1e-6));
}

TEST_CASE("autoMask2DClass: twin-strand preservation, out-of-tube isolation, edge-peak fallback, and non-helical mode", "[helix_context_types][auto_mask]")
{
	AutoMask2DOptions opt;
	opt.lowpass = 35.0;
	opt.min_lowpass = 35.0;
	opt.extend_angst = 18.0;
	opt.soft_edge = 3.0;
	opt.helix_tube_diam = 250.0; // +/-125 A => +/-19.2 px at 6.5 A/px

	// 1. Twin-strand helical class with a dimmer second strand separated by a deep dark central groove
	MultidimArray<RFLOAT> twin(1, 128, 128);
	twin.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY2D(twin)
	{
		const RFLOAT ax = std::fabs((RFLOAT)j);
		double val = 0.0;
		if (ax <= 56.0)
		{
			if (i >= 4 && i <= 8) val = 10.0;        // upper strand holds all top seeds
			else if (i <= -4 && i >= -8) val = 8.5;  // dimmer opposing strand inside tube (52 A vs 125 A)
			else if (std::fabs((RFLOAT)i) <= 2.0) val = -5.0; // deep dark central groove
		}
		A2D_ELEM(twin, i, j) = val;
	}
	MultidimArray<RFLOAT> m_twin;
	AutoMaskReport rep_twin;
	REQUIRE(autoMask2DClass(twin, 6.5, 14.0, opt, m_twin, NULL, &rep_twin, 1));
	CHECK(A2D_ELEM(m_twin, 6, 0) > 0.95);
	CHECK(A2D_ELEM(m_twin, -6, 0) > 0.95);
	CHECK(A2D_ELEM(m_twin, 50, 0) < 0.05);
	CHECK(rep_twin.mass > 0.0);

	// 2. Neighboring parallel density outside the 250 A helical tube (|y| = 28 px = 182 A > 125 A) is NOT merged
	MultidimArray<RFLOAT> nbr(1, 128, 128);
	nbr.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY2D(nbr)
	{
		const RFLOAT ax = std::fabs((RFLOAT)j);
		double val = 0.0;
		if (ax <= 56.0)
		{
			if (std::fabs((RFLOAT)i) <= 3.0) val = 10.0;
			else if (i >= 26 && i <= 30) val = 8.5; // outside 250 A tube
		}
		A2D_ELEM(nbr, i, j) = val;
	}
	MultidimArray<RFLOAT> m_nbr;
	REQUIRE(autoMask2DClass(nbr, 6.5, 14.0, opt, m_nbr, NULL, NULL, 1));
	CHECK(A2D_ELEM(m_nbr, 0, 0) > 0.95);
	CHECK(A2D_ELEM(m_nbr, 28, 0) < 0.05);

	// 3. Edge-peak hijacked helical class falls back to all-ones instead of erasing the central axis
	MultidimArray<RFLOAT> edge(1, 128, 128);
	edge.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY2D(edge)
	{
		double val = (std::fabs((RFLOAT)i) <= 4.0 && std::fabs((RFLOAT)j) <= 56.0) ? 2.0 : 0.0;
		if (std::fabs((RFLOAT)i) <= 8.0 && j >= 36 && j <= 56)
			val = 10.0;
		A2D_ELEM(edge, i, j) = val;
	}
	MultidimArray<RFLOAT> m_edge;
	CHECK_FALSE(autoMask2DClass(edge, 6.5, 14.0, opt, m_edge, NULL, NULL, 1));
	CHECK(m_edge.computeMin() == Approx(1.0).margin(1e-6));
	CHECK(m_edge.computeMax() == Approx(1.0).margin(1e-6));

	// 4. Non-helical mode (helix_tube_diam <= 0) does NOT apply helical axial-coverage rejection to compact particles
	MultidimArray<RFLOAT> glob(1, 64, 64);
	glob.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY2D(glob)
		A2D_ELEM(glob, i, j) = (i * i + j * j <= 144) ? 10.0 : 0.0;
	AutoMask2DOptions opt_nonhelix = opt;
	opt_nonhelix.helix_tube_diam = -1.0;
	opt_nonhelix.extend_angst = 8.0;
	opt_nonhelix.soft_edge = 2.0;
	MultidimArray<RFLOAT> m_glob;
	REQUIRE(autoMask2DClass(glob, 4.0, 10.0, opt_nonhelix, m_glob, NULL, NULL, 1));
	CHECK(A2D_ELEM(m_glob, 0, 0) > 0.95);
	CHECK(A2D_ELEM(m_glob, 28, 28) < 0.05);
}

TEST_CASE("MetaDataTable and junkClasses: inf/nan STAR resolution parses cleanly and is marked junk", "[helix_context_types]")
{
	const std::string fn = "tmp_star_inf_" + integerToString((int)std::clock()) + ".star";
	{
		std::ofstream ofs(fn.c_str());
		ofs << "\ndata_model_classes\n\nloop_\n_rlnEstimatedResolution #1\n_rlnClassDistribution #2\n";
		ofs << "6.600000 0.40\n";
		ofs << "8.000000 0.40\n";
		ofs << "     inf 0.001\n";
		ofs << "    -nan 0.001\n";
		ofs << "0.000000 0.001\n";
	}
	MetaDataTable MD;
	MD.read(fn, "model_classes");
	std::remove(fn.c_str());
	REQUIRE(MD.numberOfObjects() == 5);
	std::vector<RFLOAT> res(5, 0.), pdf(5, 0.);
	for (int k = 0; k < 5; k++)
	{
		MD.getValue(EMDL_MLMODEL_ESTIM_RESOL_REF, res[k], k);
		MD.getValue(EMDL_MLMODEL_PDF_CLASS, pdf[k], k);
	}
	CHECK(res[0] == Approx(6.6));
	CHECK(res[1] == Approx(8.0));
	CHECK(std::isinf(res[2]));
	CHECK(std::isnan(res[3]));
	CHECK(res[4] == 0.0);

	const std::vector<char> junk = helixtypes::junkClasses(res, std::vector<RFLOAT>(), 200.0, 1.5, std::vector<RFLOAT>(), pdf);
	REQUIRE(junk.size() == 5);
	CHECK(junk[0] == 0);
	CHECK(junk[1] == 0);
	CHECK(junk[2] != 0);
	CHECK(junk[3] != 0);
	CHECK(junk[4] != 0);
}

TEST_CASE("HelicalContext::selectNrTypesByMixture: falls back below tmin instead of collapsing to 1", "[helix_context_types]")
{
	HelicalContext hc;
	makeTypes(hc, 4, 8, 120, 20, 0.02, 1234);
	// Even when caller asks tmin = 8 > true 4 types, mixture selection should keep 4 rather than collapsing to 1
	const int chosen = hc.selectNrTypes(8, 12, 3, 0.01, 0.75, false);
	CHECK(chosen == 4);
}

TEST_CASE("MlOptimiser optimiser.star round-trip preserves helical options across --continue", "[helix_context_types][ml_optimiser]")
{
	const std::string tmpdir = "tmp_opt_continue_" + integerToString((int)std::clock());
	vcache::removeTree(tmpdir);
	vcache::makeDirs(tmpdir);

	// Minimal optics + 2-particle STAR file + dummy 2-slice stack (not preread during read())
	Image<RFLOAT> dummy(16, 16, 1, 2);
	dummy().initZeros();
	dummy.write(tmpdir + "/dummy.mrcs");
	const std::string data_star = tmpdir + "/init_data.star";
	{
		MetaDataTable MDopt, MD;
		MDopt.setName("optics");
		MDopt.addObject();
		MDopt.setValue(EMDL_IMAGE_OPTICS_GROUP, 1, 0);
		MDopt.setValue(EMDL_IMAGE_OPTICS_GROUP_NAME, std::string("opticsGroup1"), 0);
		MDopt.setValue(EMDL_MICROGRAPH_ORIGINAL_PIXEL_SIZE, 2.0, 0);
		MDopt.setValue(EMDL_IMAGE_PIXEL_SIZE, 2.0, 0);
		MDopt.setValue(EMDL_IMAGE_SIZE, 16, 0);
		MDopt.setValue(EMDL_IMAGE_DIMENSIONALITY, 2, 0);
		MDopt.setValue(EMDL_CTF_VOLTAGE, 300.0, 0);
		MDopt.setValue(EMDL_CTF_CS, 2.7, 0);
		MDopt.setValue(EMDL_CTF_Q0, 0.1, 0);
		MDopt.setValue(EMDL_OPTIMISER_DATA_ARE_CTF_PREMULTIPLIED, false, 0);

		MD.setName("particles");
		for (int s = 0; s < 2; s++)
		{
			MD.addObject();
			MD.setValue(EMDL_MICROGRAPH_NAME, std::string("mic1.mrc"), s);
			MD.setValue(EMDL_IMAGE_NAME, integerToString(s + 1, 6) + "@" + tmpdir + "/dummy.mrcs", s);
			MD.setValue(EMDL_IMAGE_OPTICS_GROUP, 1, s);
			MD.setValue(EMDL_PARTICLE_HELICAL_TUBE_ID, 1, s);
			MD.setValue(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM, (RFLOAT)(s * 20.0), s);
			MD.setValue(EMDL_ORIENT_TILT_PRIOR, (RFLOAT)90.0, s);
			MD.setValue(EMDL_ORIENT_PSI_PRIOR, (RFLOAT)0.0, s);
		}
		std::ofstream fh(data_star.c_str());
		MDopt.write(fh);
		MD.write(fh);
	}

	MlOptimiser opt;
	opt.fn_out = tmpdir + "/run";
	opt.fn_mask = opt.fn_mask2 = opt.fn_body_masks = opt.fn_tau = opt.fn_local_symmetry = opt.fn_fourier_mask = "None";
	opt.iter = 8;
	opt.nr_iter = 20;
	opt.grad_ini_frac = 0.3;
	opt.grad_fin_frac = 0.2;
	opt.particle_diameter = 200.;
	opt.helical_tube_outer_diameter = 160.;
	opt.do_center_classes = true;
	opt.do_bimodal_psi = true;
	opt.do_sort_classes = true;
	opt.sort_classes_junk = 2.0;
	opt.do_helical_context = true;
	opt.helical_context_start_iter = 9;
	opt.helical_context.nr_types = 8;
	opt.helical_context_types_min = 6;
	opt.helical_context_types_max = 25;
	opt.helical_context_types_selected = true;
	opt.helical_context_types_last_iter = 8;
	opt.helical_context.temper = 0.42;
	opt.helical_context.exclude_angst = 120.0;
	opt.do_helical_polarity = true;
	opt.helical_polarity_start_iter = 9;
	opt.do_helical_track_smooth = false;
	opt.helical_tilt_weight_sigma = 1.8;
	opt.do_fix_gpu_bimodal_psi = true;
	opt.helical_context_types_tol = 0.045;
	opt.helical_context_types_agree = 0.62;
	opt.helical_context_types_every = 9;
	opt.helical_context_warm = true;
	opt.helical_track_sigma_A = 2.2;

	opt.mymodel.ori_size = 16;
	opt.mymodel.pixel_size = 2.0;
	opt.mymodel.nr_classes = 2;
	opt.mymodel.nr_bodies = 1;
	opt.mymodel.ref_dim = 2;
	opt.mymodel.data_dim = 2;
	opt.mymodel.nr_groups = 1;
	opt.mymodel.nr_optics_groups = 1;
	opt.mymodel.current_size = 16;
	opt.mymodel.current_resolution = 1. / 10.;
	opt.mymodel.tau2_fudge_factor = 1.;
	opt.mymodel.avg_norm_correction = 1.;
	opt.mymodel.sigma2_offset = 1.;
	opt.mymodel.sigma2_rot = opt.mymodel.sigma2_tilt = opt.mymodel.sigma2_psi = 1.;
	opt.mymodel.padding_factor = 2.;
	opt.mymodel.interpolator = TRILINEAR;
	opt.mymodel.r_min_nn = 0;
	opt.mymodel.initialise();
	opt.mymodel.group_names[0] = "group1";
	opt.mymodel.nr_particles_per_group[0] = 2;
	opt.mymodel.sigma2_noise[0].initConstant(1.0);
	for (int k = 0; k < 2; k++)
	{
		opt.mymodel.Iref[k].initZeros(16, 16);
		opt.mymodel.pdf_direction[k].resize(1);
		opt.mymodel.pdf_direction[k].initConstant(1.0);
	}
	opt.mymodel.initialiseDataVersusPrior(false);

	opt.mydata.read(data_star, "", "", false, false, false, true, false);
	opt.sampling.offset_range = 6.;
	opt.sampling.offset_step = 2.;
	opt.sampling.initialise(2, false);

	opt.write(true, true, true, true, 0);
	AsyncFileWriter::instance().waitOrThrow();
	const std::string opt_star = tmpdir + "/run_it008_optimiser.star";
	REQUIRE(exists(opt_star));

	// 1. Bare --continue preserves every saved option
	MlOptimiser cont_bare;
	const char *argv_bare[] = {"relion_refine", "--continue", opt_star.c_str()};
	cont_bare.read(3, const_cast<char **>(argv_bare), 0);
	CHECK(cont_bare.do_center_classes);
	CHECK(cont_bare.do_bimodal_psi);
	CHECK(cont_bare.do_sort_classes);
	CHECK(cont_bare.sort_classes_junk == Approx(2.0));
	CHECK(cont_bare.do_helical_context);
	CHECK(cont_bare.helical_context_start_iter == 9);
	CHECK(cont_bare.helical_context.nr_types == 8);
	CHECK(cont_bare.helical_context_types_min == 6);
	CHECK(cont_bare.helical_context_types_max == 25);
	CHECK(cont_bare.helical_context_types_selected);
	CHECK(cont_bare.helical_context_types_last_iter == 8);
	CHECK(cont_bare.helical_context.temper == Approx(0.42));
	CHECK(cont_bare.helical_context.exclude_angst == Approx(120.0));
	CHECK(cont_bare.do_helical_polarity);
	CHECK(cont_bare.helical_polarity_start_iter == 9);
	CHECK_FALSE(cont_bare.do_helical_track_smooth);
	CHECK(cont_bare.helical_tilt_weight_sigma == Approx(1.8));
	CHECK(cont_bare.do_fix_gpu_bimodal_psi);
	CHECK(cont_bare.helical_context_types_tol == Approx(0.045));
	CHECK(cont_bare.helical_context_types_agree == Approx(0.62));
	CHECK(cont_bare.helical_context_types_every == 9);
	CHECK(cont_bare.helical_context_warm);
	CHECK(cont_bare.helical_track_sigma_A == Approx(2.2));

	// 2. Explicit negative flags override persisted options on --continue
	MlOptimiser cont_off;
	const char *argv_off[] = {
		"relion_refine", "--continue", opt_star.c_str(),
		"--dont_center_classes", "--dont_bimodal_psi", "--dont_sort_classes",
		"--dont_helical_context", "--dont_helical_polarity"
	};
	cont_off.read(8, const_cast<char **>(argv_off), 0);
	CHECK_FALSE(cont_off.do_center_classes);
	CHECK_FALSE(cont_off.do_bimodal_psi);
	CHECK_FALSE(cont_off.do_sort_classes);
	CHECK_FALSE(cont_off.do_helical_context);
	CHECK_FALSE(cont_off.do_helical_polarity);

	// 3. Re-passing unchanged min/max preserves selected nr_types; --no_init_blobs is accepted on continue
	MlOptimiser cont_same_range;
	const char *argv_same[] = {
		"relion_refine", "--continue", opt_star.c_str(),
		"--helical_context_types", "6",
		"--helical_context_types_max", "25",
		"--no_init_blobs"
	};
	cont_same_range.read(8, const_cast<char **>(argv_same), 0);
	CHECK(cont_same_range.helical_context.nr_types == 8);
	CHECK(cont_same_range.helical_context_types_min == 6);
	CHECK(cont_same_range.helical_context_types_max == 25);
	CHECK(cont_same_range.helical_context_types_selected);
	CHECK(cont_same_range.do_fix_gpu_bimodal_psi);
	CHECK(cont_same_range.helical_context_types_tol == Approx(0.045));
	CHECK(cont_same_range.helical_context_types_agree == Approx(0.62));
	CHECK(cont_same_range.helical_context_types_every == 9);
	CHECK(cont_same_range.helical_context_warm);
	CHECK(cont_same_range.helical_track_sigma_A == Approx(2.2));

	// 4. Changing range resets type selection; explicit negative secondary flags override
	MlOptimiser cont_changed;
	const char *argv_chg[] = {
		"relion_refine", "--continue", opt_star.c_str(),
		"--helical_context_types", "4",
		"--helical_context_types_max", "12",
		"--dont_fix_gpu_bimodal_psi",
		"--dont_helical_context_warm",
		"--init_blobs"
	};
	cont_changed.read(10, const_cast<char **>(argv_chg), 0);
	CHECK(cont_changed.helical_context.nr_types == 4);
	CHECK(cont_changed.helical_context_types_min == 4);
	CHECK(cont_changed.helical_context_types_max == 12);
	CHECK_FALSE(cont_changed.helical_context_types_selected);
	CHECK_FALSE(cont_changed.do_fix_gpu_bimodal_psi);
	CHECK_FALSE(cont_changed.helical_context_warm);

	vcache::removeTree(tmpdir);
}

TEST_CASE("HelicalContext: rare active leaf types survive hierarchical typing and write non-empty type classes", "[helix_context_types]")
{
	const std::string tmpdir = "tmp_rare_type_" + integerToString((int)std::clock());
	vcache::removeTree(tmpdir);
	vcache::makeDirs(tmpdir);

	// Part 1: hierarchical typing with a rare transition-defined subpopulation (<12% share in shared classes)
	const int K = 4, S = 4;
	const int F0 = 90, F1 = 90, F2 = 10; // F2 share in cls 0 is 20 / (180 + 20) = 10% (< 12% class_allowed threshold)
	const int F = F0 + F1 + F2;
	MetaDataTable MD;
	std::vector<long int> rows;
	std::vector<int> seg_cls;
	std::vector<RFLOAT> psi_vec, psip_vec;
	long int row = 0;
	for (int f = 0; f < F; f++)
	{
		const std::string mic = "mic_" + integerToString((f / 30) + 1) + ".mrc";
		const int tube = (f % 30) + 1;
		for (int s = 0; s < S; s++)
		{
			MD.addObject();
			rows.push_back(row);
			MD.setValue(EMDL_MICROGRAPH_NAME, mic, row);
			MD.setValue(EMDL_PARTICLE_HELICAL_TUBE_ID, tube, row);
			MD.setValue(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM, (RFLOAT)(s * 20.0), row);
			MD.setValue(EMDL_IMAGE_OPTICS_GROUP, 1, row);
			MD.setValue(EMDL_ORIENT_PSI_PRIOR, (RFLOAT)0.0, row);
			MD.setValue(EMDL_ORIENT_PSI, (RFLOAT)0.0, row);
			psi_vec.push_back(0.0);
			psip_vec.push_back(0.0);
			int kc = 0;
			if (f < F0) kc = s % 2;
			else if (f < F0 + F1) kc = 2 + (s % 2);
			else kc = (s % 2 == 0) ? 0 : 2;
			seg_cls.push_back(kc);
			MD.setValue(EMDL_PARTICLE_CLASS, kc + 1, row);
			row++;
		}
	}
	HelicalContext ctx;
	ctx.verb = 0;
	ctx.nr_types = 2;
	ctx.initialise(MD, rows, K, 200.0);
	for (long int i = 0; i < ctx.N; i++)
	{
		std::vector<RFLOAT> ll(K, std::log(0.05 / (K - 1)));
		ll[seg_cls[i]] = std::log(0.95);
		ctx.storeLogLik(i, ll);
	}
	const int L = ctx.runHierarchicalTyping(psi_vec, psip_vec, 2, 6, 12, 8, 3, 0.01, 0.5, false);
	REQUIRE(L >= 2);
	ctx.writeLabels(MD, rows);

	MetaDataTable MDopt;
	MDopt.addObject();
	MDopt.setValue(EMDL_IMAGE_OPTICS_GROUP, 1, 0);
	MDopt.setValue(EMDL_IMAGE_PIXEL_SIZE, (RFLOAT)2.0, 0);
	Image<RFLOAT> cls_stack(8, 8, 1, K);
	for (int k = 0; k < K; k++)
		for (int y = 0; y < 8; y++)
			for (int x = 0; x < 8; x++)
				DIRECT_NZYX_ELEM(cls_stack(), k, 0, y, x) = (RFLOAT)(k + 1);
	const FileName fn_classes = tmpdir + "/classes.mrcs";
	cls_stack.write(fn_classes);
	HelicalContext::writeTypeFiles(tmpdir + "/hier", MDopt, MD, L, fn_classes, 0.10, 20, 0);

	std::set<int> pop_types;
	for (long int r = 0; r < (long int)MD.numberOfObjects(); r++)
	{
		int ht = 0;
		MD.getValue(EMDL_PARTICLE_HELICAL_TYPE, ht, r);
		if (ht >= 1) pop_types.insert(ht);
	}
	for (int t : pop_types)
	{
		MetaDataTable MDc;
		MDc.read(tmpdir + "/hier_type" + integerToString(t) + "_classes.star", "model_classes");
		INFO("hier type=" << t << " classes=" << MDc.numberOfObjects());
		REQUIRE(MDc.numberOfObjects() >= 1);
	}

	// Part 2: explicit writeTypeFiles fallback when a rare type fails both min_count and min_frac
	MetaDataTable MDw;
	long int wr = 0;
	auto addSegs = [&](int type_id, int cls_id, int cnt) {
		for (int i = 0; i < cnt; i++)
		{
			MDw.addObject();
			MDw.setValue(EMDL_MICROGRAPH_NAME, std::string("mic_w.mrc"), wr);
			MDw.setValue(EMDL_PARTICLE_HELICAL_TUBE_ID, type_id, wr);
			MDw.setValue(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM, (RFLOAT)(i * 10.0), wr);
			MDw.setValue(EMDL_PARTICLE_HELICAL_TYPE, type_id, wr);
			MDw.setValue(EMDL_PARTICLE_CLASS, cls_id, wr);
			wr++;
		}
	};
	addSegs(1, 1, 100);
	addSegs(1, 2, 100);
	addSegs(2, 3, 100);
	addSegs(2, 4, 100);
	// Rare type 3: 4 segments in class 1, 2 in class 2, 1 in class 3 -> all < min_count(20) & < min_frac(0.10)
	addSegs(3, 1, 4);
	addSegs(3, 2, 2);
	addSegs(3, 3, 1);

	HelicalContext::writeTypeFiles(tmpdir + "/rare", MDopt, MDw, 3, fn_classes, 0.10, 20, 0);
	for (int t = 1; t <= 3; t++)
	{
		REQUIRE(exists(tmpdir + "/rare_type" + integerToString(t) + "_classes.star"));
		REQUIRE(exists(tmpdir + "/rare_type" + integerToString(t) + "_classes.mrcs"));
		MetaDataTable MDc;
		MDc.read(tmpdir + "/rare_type" + integerToString(t) + "_classes.star", "model_classes");
		REQUIRE(MDc.numberOfObjects() >= 1);
	}
	MetaDataTable MDc3;
	MDc3.read(tmpdir + "/rare_type3_classes.star", "model_classes");
	int top_cls = -1;
	MDc3.getValue(EMDL_PARTICLE_CLASS, top_cls, 0);
	CHECK(top_cls == 1);

	vcache::removeTree(tmpdir);
}

TEST_CASE("ObservationModel: tracks rlnCtfDataAreCtfPremultipliedIntactFirstPeak", "[helix_context_types]")
{
	const std::string tmpdir = "tmp_obs_intact_" + integerToString((int)std::clock());
	vcache::removeTree(tmpdir);
	vcache::makeDirs(tmpdir);

	MetaDataTable MDopt, MDpart;
	MDopt.setName("optics");
	MDopt.addObject();
	MDopt.setValue(EMDL_IMAGE_OPTICS_GROUP, 1, 0);
	MDopt.setValue(EMDL_IMAGE_OPTICS_GROUP_NAME, std::string("opticsGroup1"), 0);
	MDopt.setValue(EMDL_IMAGE_PIXEL_SIZE, (RFLOAT)2.0, 0);
	MDopt.setValue(EMDL_CTF_VOLTAGE, (RFLOAT)300.0, 0);
	MDopt.setValue(EMDL_CTF_CS, (RFLOAT)2.7, 0);
	MDopt.setValue(EMDL_CTF_Q0, (RFLOAT)0.1, 0);
	MDopt.setValue(EMDL_OPTIMISER_DATA_ARE_CTF_PREMULTIPLIED, true, 0);

	MDpart.setName("particles");
	MDpart.addObject();
	MDpart.setValue(EMDL_IMAGE_OPTICS_GROUP, 1, 0);
	MDpart.setValue(EMDL_IMAGE_NAME, std::string("1@dummy.mrcs"), 0);

	ObservationModel obs0(MDopt, false);
	CHECK(obs0.getCtfPremultiplied(0));
	CHECK_FALSE(obs0.hasCtfPremultipliedIntactFirstPeak());
	CHECK_FALSE(obs0.getCtfPremultipliedIntactFirstPeak(0));

	obs0.setCtfPremultipliedIntactFirstPeak(0, true);
	CHECK(obs0.hasCtfPremultipliedIntactFirstPeak());
	CHECK(obs0.getCtfPremultipliedIntactFirstPeak(0));

	const std::string fn_star = tmpdir + "/particles_true.star";
	obs0.save(MDpart, fn_star);
	ObservationModel obs_true;
	MetaDataTable MDpart_out;
	ObservationModel::loadSafely(fn_star, obs_true, MDpart_out, "particles", 0, false);
	CHECK(obs_true.getCtfPremultiplied(0));
	CHECK(obs_true.hasCtfPremultipliedIntactFirstPeak());
	CHECK(obs_true.getCtfPremultipliedIntactFirstPeak(0));

	obs_true.setCtfPremultipliedIntactFirstPeak(0, false);
	const std::string fn_false = tmpdir + "/particles_false.star";
	obs_true.save(MDpart_out, fn_false);
	ObservationModel obs_false;
	MetaDataTable MDpart_false;
	ObservationModel::loadSafely(fn_false, obs_false, MDpart_false, "particles", 0, false);
	CHECK(obs_false.hasCtfPremultipliedIntactFirstPeak());
	CHECK_FALSE(obs_false.getCtfPremultipliedIntactFirstPeak(0));

	vcache::removeTree(tmpdir);
}
