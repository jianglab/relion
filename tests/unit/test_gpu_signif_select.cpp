/*
 * tests/unit/test_gpu_signif_select.cpp
 *
 * The radix selection of the significance threshold (RELION_GPU_SIGNIF_SELECT) must give what the
 * sort-and-scan significance gives: the same weights, the same threshold weight, the same mask, the
 * same count of significant samples. Skipped when no GPU is visible.
 */

#include <catch2/catch.hpp>

#include "src/acc/cuda/cuda_signif_select_test.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

namespace
{

enum Shape { LOGNORMAL, MANY_ZEROS, TIES, FLAT };

std::vector<float> makeLogWeights(size_t n, Shape shape, unsigned seed)
{
	std::mt19937 rng(seed);
	std::normal_distribution<float> g(0.f, 1.f);
	std::vector<float> v(n);
	for (size_t i = 0; i < n; i++)
	{
		switch (shape)
		{
		case LOGNORMAL: v[i] = -5.f * std::fabs(g(rng)) * std::fabs(g(rng)); break;
		// most samples excluded: log-weights so small that the exponentiated weight is exactly zero
		case MANY_ZEROS: v[i] = (rng() % 20 == 0) ? -3.f * std::fabs(g(rng)) : -1.0e30f; break;
		// a handful of distinct values, so that the threshold falls inside a group of equal weights
		case TIES: v[i] = -0.5f * (float) (rng() % 6); break;
		case FLAT: v[i] = -1.f; break;
		}
	}
	return v;
}

}

TEST_CASE("radix selection of the significant weight equals the sort-and-scan one", "[gpu_signif_select]")
{
	gpusel::Outcome probe;
	if (!gpusel::runFusedWeightPass(std::vector<float>(1 << 17, -1.f), 0.999, 0, false, probe))
	{
		WARN("no GPU visible: test skipped");
		return;
	}
	const struct { size_t n; Shape shape; double af; long cap; } cases[] = {
		{ 100000, LOGNORMAL, 0.999, 0 },
		{ 300000, LOGNORMAL, 0.99, 0 },
		{ 300000, LOGNORMAL, 0.999, 2000 },
		{ 1000003, LOGNORMAL, 0.999, 0 },
		{ 262144, MANY_ZEROS, 0.999, 0 },
		{ 262144, MANY_ZEROS, 0.99, 500 },
		{ 200000, TIES, 0.999, 0 },
		{ 200000, TIES, 0.9, 77 },
		{ 70000, FLAT, 0.99, 0 },
		{ 70000, FLAT, 0.99, 10 },
	};
	unsigned seed = 3;
	for (const auto &c : cases)
	{
		INFO("n " << c.n << " shape " << c.shape << " af " << c.af << " cap " << c.cap);
		const std::vector<float> lw = makeLogWeights(c.n, c.shape, seed++);
		gpusel::Outcome sorted, radix;
		REQUIRE(gpusel::runFusedWeightPass(lw, c.af, c.cap, false, sorted));
		REQUIRE(gpusel::runFusedWeightPass(lw, c.af, c.cap, true, radix));

		// same exponentiation, same maximum
		CHECK(radix.max_key == sorted.max_key);
		CHECK(radix.max_value == sorted.max_value);
		REQUIRE(radix.weights == sorted.weights);

		// the total weight is added up in another order: equal to rounding
		CHECK(radix.sum_weight == Approx(sorted.sum_weight).epsilon(1e-5));
		// the weight at the threshold is one of the weights themselves, and exactly the same one
		CHECK(radix.significant_weight == sorted.significant_weight);
		CHECK(radix.idx_capped == sorted.idx_capped);
		// the cumulative sums differ by rounding, so the index of the threshold may differ by a few
		// positions inside a run of equal weights, never in a way that changes the mask
		size_t diff = 0;
		for (size_t i = 0; i < c.n; i++)
			diff += (radix.mask[i] != sorted.mask[i]);
		CHECK(diff == 0);

		// against the exact threshold (double sums): the weight picked must sit where the sum of the
		// smaller weights crosses (1 - af) of the total, to the rounding of float sums
		if (c.cap == 0)
		{
			double total = 0, below = 0, upto = 0;
			for (float x : radix.weights) total += x;
			for (float x : radix.weights)
			{
				if (x < radix.significant_weight) below += x;
				if (x <= radix.significant_weight) upto += x;
			}
			const double thr = (1 - c.af) * total;
			CHECK(below <= thr * (1 + 1e-3));
			CHECK(upto >= thr * (1 - 1e-3));
		}
	}
}

// Run with the fused selection on, for example:
//   RELION_GPU_SELECT_FUSED=2 relion_unit_tests "[gpu_signif_select_stress]"
// STRESS_SECONDS sets the length (default 3), STRESS_THREADS the number of host threads (default 8).
// The cooperative launch needs all its blocks resident at once: several threads (and processes) launching
// at the same time, and other kernels on the card, must neither deadlock nor change the result.
TEST_CASE("a refused cooperative launch falls back to the separate launches with the same result", "[gpu_signif_select]")
{
	std::vector<float> w(300000);
	unsigned long long state = 12345;
	for (auto &x : w)
	{
		state = state * 6364136223846793005ULL + 1442695040888963407ULL;
		x = std::exp(-6.f * (float) ((state >> 33) % 100000) / 100000.f);
	}
	gpusel::OnceOutcome normal, refused;
	if (!gpusel::radixSelectOnce(w, false, normal) || !gpusel::radixSelectOnce(w, true, refused))
	{
		WARN("no usable GPU: skipped");
		return;
	}
	if (normal.fused_launches == 0)
	{
		WARN("the fused selection is off or not supported here: skipped");
		return;
	}
	CHECK(refused.fused_launches == 0);
	CHECK(refused.fallbacks == 1); // the launch was tried and refused
	CHECK(refused.idx == normal.idx);
	CHECK(refused.idx_capped == normal.idx_capped);
	CHECK(refused.significant_weight == normal.significant_weight);
	CHECK(refused.sum_weight == Approx(normal.sum_weight).epsilon(1e-6));
}

TEST_CASE("the radix selection does not deadlock or change its result under concurrent launches", "[gpu_signif_select_stress][.]")
{
	const char *sec = getenv("STRESS_SECONDS");
	const char *thr = getenv("STRESS_THREADS");
	const double seconds = sec ? atof(sec) : 3.0;
	const int threads = thr ? atoi(thr) : 8;
	const char *only = getenv("STRESS_COMPETE_ONLY");
	for (int compete = (only && atoi(only)) ? 1 : 0; compete < 2; compete++)
	{
		INFO("threads " << threads << " competing kernels " << compete);
		gpusel::StressOutcome out;
		if (!gpusel::stressRadixSelect(threads, seconds, 300000, compete != 0, 30.0, out))
		{
			WARN("no GPU visible: test skipped");
			return;
		}
		INFO("selections done " << out.launches << " cooperative launch supported " << out.cooperative);
		std::cout << "  stress: threads " << threads << " competing " << compete << ": " << out.launches << " selections, " << out.fused_launches
				<< " cooperative launches, " << out.fallbacks << " fallbacks" << std::endl;
		REQUIRE_FALSE(out.stalled);
		CHECK(out.mismatches == 0);
		CHECK(out.launches > 0);
	}
}
