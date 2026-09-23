#include <catch2/catch.hpp>
#include "src/class2d_consensus.h"
#include "src/class2d_consensus_nmf.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace
{
double denseObjective(const std::vector<int> &labels, int k,
                     const std::vector<double> &w, const std::vector<double> &h)
{
	double residual = 0.;
	const size_t d = labels.size() * k;
	for (size_t j = 0; j < d; ++j)
	{
		double value = (int)(j % k) == labels[j / k] ? 1. : 0.;
		for (size_t a = 0; a < w.size(); ++a) value -= w[a] * h[a * d + j];
		residual += value * value;
	}
	return residual * .5;
}

void checkNmf(const Class2DConsensusNmfResult &result, int runs, size_t particles)
{
	const int c = result.consensus_classes, k = result.source_classes;
	REQUIRE(result.assignment.size() == particles);
	REQUIRE(result.components.size() == (size_t)runs * c * k);
	REQUIRE(std::isfinite(result.objective));
	REQUIRE(result.objective >= 0.);
	REQUIRE(std::accumulate(result.class_membership_mass.begin(), result.class_membership_mass.end(), 0.) == Approx((double)particles));
	for (size_t p = 0; p < particles; ++p)
	{
		REQUIRE(result.assignment[p] >= 0);
		REQUIRE(result.assignment[p] < c);
		REQUIRE(result.membership[p] >= 1. / c - 1.e-12);
		REQUIRE(result.membership[p] <= 1. + 1.e-12);
		REQUIRE(result.entropy[p] >= 0.);
		REQUIRE(result.entropy[p] <= 1.);
		REQUIRE(result.agreement[p] >= 0.);
		REQUIRE(result.agreement[p] <= 1.);
	}
	for (int r = 0; r < runs; ++r)
		for (int a = 0; a < c; ++a)
		{
			double sum = 0.;
			for (int j = 0; j < k; ++j)
			{
				const double value = result.components[((size_t)r * c + a) * k + j];
				REQUIRE(std::isfinite(value)); REQUIRE(value >= 0.);
				sum += value;
			}
			REQUIRE(sum == Approx(1.));
		}
	for (const auto &start : result.starts)
	{
		REQUIRE_FALSE(start.objective_history.empty());
		for (size_t i = 1; i < start.objective_history.size(); ++i)
			REQUIRE(start.objective_history[i] <= start.objective_history[i - 1] + 1.e-10);
		REQUIRE(result.objective <= start.objective_history.back() + 1.e-12);
	}
}
}

TEST_CASE("Sparse NMF implicit residual and gradients match dense finite differences", "[consensus][nmf]")
{
	const std::vector<int> labels{0, 2};
	const int k = 3, c = 2, d = 6;
	std::vector<double> w{.3, .7}, h{.2,.3,.5, .6,.1,.3, .8,.1,.1, .2,.5,.3};
	std::vector<double> gram, gradient;
	class2d_nmf_detail::gramMatrix(h, c, d, gram);
	const double value = class2d_nmf_detail::patternObjectiveGradient(labels, k, h, gram, w.data(), c, gradient);
	REQUIRE(value == Approx(denseObjective(labels, k, w, h)));
	const double delta = 1.e-6;
	for (int a = 0; a < c; ++a)
	{
		auto plus = w, minus = w;
		plus[a] += delta; minus[a] -= delta;
		REQUIRE(gradient[a] == Approx((denseObjective(labels, k, plus, h) - denseObjective(labels, k, minus, h)) / (2. * delta)).margin(1.e-9));
	}
	std::vector<double> s(c * c), t(c * d, 0.);
	for (int a = 0; a < c; ++a)
	{
		for (int b = 0; b < c; ++b) s[a * c + b] = w[a] * w[b];
		for (size_t r = 0; r < labels.size(); ++r) t[a * d + r * k + labels[r]] = w[a];
		for (int j = 0; j < d; ++j)
		{
			auto plus = h, minus = h;
			plus[a * d + j] += delta; minus[a * d + j] -= delta;
			REQUIRE(class2d_nmf_detail::profileGradient(h, s, t, c, d, a, j) ==
				Approx((denseObjective(labels, k, w, plus) - denseObjective(labels, k, w, minus)) / (2. * delta)).margin(1.e-9));
		}
	}
}

TEST_CASE("Sparse NMF simplex projection handles boundaries and preserves feasible rows", "[consensus][nmf]")
{
	std::vector<double> values{-1., 2., 0.}, scratch;
	class2d_nmf_detail::projectSimplex(values, scratch);
	REQUIRE(values == std::vector<double>{0.,1.,0.});
	values = {.1,.3,.6};
	class2d_nmf_detail::projectSimplex(values, scratch);
	REQUIRE(values[0] == Approx(.1)); REQUIRE(values[1] == Approx(.3)); REQUIRE(values[2] == Approx(.6));
	values = {-2.,-2.,-2.};
	class2d_nmf_detail::projectSimplex(values, scratch);
	for (double v : values) REQUIRE(v == Approx(1./3.));
}

TEST_CASE("Sparse NMF recovers agreeing replicas including a rare class", "[consensus][nmf]")
{
	std::vector<std::vector<int> > runs(3);
	std::vector<int> truth;
	for (int p = 0; p < 101; ++p)
	{
		const int label = p < 60 ? 0 : p < 100 ? 1 : 2;
		truth.push_back(label);
		for (int r = 0; r < 3; ++r) runs[r].push_back((label + r) % 3);
	}
	const auto result = Class2DConsensus::fitSparseNmf(runs, 3, 3);
	checkNmf(result, 3, truth.size());
	REQUIRE(result.assignment == truth);
	REQUIRE(result.objective < 1.e-9);
	REQUIRE(result.converged);
	for (double probability : result.membership) REQUIRE(probability > .999);
	for (double agreement : result.agreement) REQUIRE(agreement == Approx(1.));
}

TEST_CASE("Sparse NMF handles independent class counts and preserves identical patterns", "[consensus][nmf]")
{
	int k = 4, c = 2;
	std::vector<std::vector<int> > patterns{{0,1,2,3}, {0,0,2,2}, {1,1,3,3}};
	std::vector<int> truth{0,0,1,1};
	SECTION("fewer") {}
	SECTION("more") { k = 2; c = 4; patterns = {{0,0,1,1},{0,1,0,1},{0,0,1,1}}; truth = {0,1,2,3}; }
	SECTION("more slots than patterns") { k = 2; c = 7; patterns = {{0,0,1,1},{0,1,0,1},{0,0,1,1}}; truth = {0,1,2,3}; }
	std::vector<std::vector<int> > runs(3);
	std::vector<int> expected;
	for (size_t p = 0; p < truth.size(); ++p)
		for (int repeat = 0; repeat < 10; ++repeat)
		{
			for (int r = 0; r < 3; ++r) runs[r].push_back(patterns[r][p]);
			expected.push_back(truth[p]);
		}
	const auto result = Class2DConsensus::fitSparseNmf(runs, k, c);
	checkNmf(result, 3, expected.size());
	REQUIRE(result.assignment == expected);
	REQUIRE(result.nr_patterns == 4);
	REQUIRE(result.fitted_components == std::min(c, 4));
	for (int a = 4; a < c; ++a) REQUIRE(result.class_membership_mass[a] == 0.);
	const auto repeated = Class2DConsensus::fitSparseNmf(runs, k, c);
	REQUIRE(repeated.assignment == result.assignment);
	REQUIRE(repeated.membership == result.membership);
	for (int r = 0; r < 3; ++r) for (int &label : runs[r]) label = (k - 1 - label + r) % k;
	const auto permuted = Class2DConsensus::fitSparseNmf(runs, k, c, Class2DConsensusNmfOptions(), 2);
	REQUIRE(permuted.assignment == result.assignment);
	for (size_t p = 0; p < expected.size(); ++p) REQUIRE(permuted.membership[p] == Approx(result.membership[p]).margin(1.e-8));
}

TEST_CASE("Sparse NMF handles a single pattern, empty source slots and iteration limits", "[consensus][nmf]")
{
	const std::vector<std::vector<int> > runs{{2,2,2},{0,0,0},{3,3,3}};
	Class2DConsensusNmfOptions options;
	size_t reported = 0;
	options.report_workspace_bytes = [&](size_t bytes) { reported = bytes; };
	options.maximum_iterations = 1;
	const auto result = Class2DConsensus::fitSparseNmf(runs, 4, 6, options);
	checkNmf(result, 3, 3);
	REQUIRE(result.fitted_components == 1);
	REQUIRE(result.starts.size() == 1);
	REQUIRE(result.assignment == std::vector<int>{0,0,0});
	REQUIRE(result.membership == std::vector<double>{1.,1.,1.});
	REQUIRE(result.objective == Approx(0.));
	REQUIRE_FALSE(result.converged);
	REQUIRE(result.starts[0].termination == "iteration_limit");
	REQUIRE(result.iterations == 1);
	REQUIRE(reported == result.workspace_bytes);
	REQUIRE(reported >= 3 * sizeof(double));
}

TEST_CASE("Sparse NMF keeps noisy ambiguous particles and weights repeated patterns", "[consensus][nmf]")
{
	std::vector<std::vector<int> > runs{{0,0,0,0,1,1,1,1,0,1}, {1,1,1,1,0,0,0,0,0,1}, {0,1,0,1,0,1,0,1,1,0}};
	const auto first = Class2DConsensus::fitSparseNmf(runs, 2, 2);
	checkNmf(first, 3, 10);
	for (auto &run : runs) { auto copy = run; run.insert(run.end(), copy.begin(), copy.end()); }
	const auto duplicated = Class2DConsensus::fitSparseNmf(runs, 2, 2);
	checkNmf(duplicated, 3, 20);
	REQUIRE(first.objective == Approx(duplicated.objective));
	for (size_t p = 0; p < 10; ++p)
	{
		REQUIRE(duplicated.assignment[p] == first.assignment[p]);
		REQUIRE(duplicated.assignment[p + 10] == first.assignment[p]);
		REQUIRE(duplicated.membership[p] == Approx(first.membership[p]));
	}
}

TEST_CASE("Sparse NMF rejects invalid dimensions, labels and solver controls", "[consensus][nmf]")
{
	Class2DConsensusNmfOptions options;
	const std::vector<std::vector<int> > runs{{0,1},{1,0}};
	REQUIRE_THROWS(Class2DConsensus::fitSparseNmf({{0,1}}, 2, 2));
	REQUIRE_THROWS(Class2DConsensus::fitSparseNmf({{},{}}, 2, 2));
	REQUIRE_THROWS(Class2DConsensus::fitSparseNmf({{0,1},{0}}, 2, 2));
	REQUIRE_THROWS(Class2DConsensus::fitSparseNmf({{0,-1},{0,1}}, 2, 2));
	REQUIRE_THROWS(Class2DConsensus::fitSparseNmf(runs, 1, 2));
	REQUIRE_THROWS(Class2DConsensus::fitSparseNmf(runs, 2, 1));
	REQUIRE_THROWS(Class2DConsensus::fitSparseNmf(runs, std::numeric_limits<int>::max(), 2));
	REQUIRE_THROWS(Class2DConsensus::fitSparseNmf(runs, 2, 2, options, 0));
	SECTION("iterations") { options.maximum_iterations = 0; }
	SECTION("starts") { options.starts = 0; }
	SECTION("tolerance") { options.tolerance = 0.; }
	SECTION("nonfinite tolerance") { options.tolerance = std::numeric_limits<double>::quiet_NaN(); }
	REQUIRE_THROWS(Class2DConsensus::fitSparseNmf(runs, 2, 2, options));
}
