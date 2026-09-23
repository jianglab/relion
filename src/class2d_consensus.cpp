#include "src/class2d_consensus.h"
#include "src/class2d_consensus_nmf.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <new>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

struct Pattern
{
	std::vector<int> labels;
	long int count;
};

struct FitState
{
	std::vector<double> prior;
	std::vector<double> confusion;
	int iterations;
	double log_likelihood;
};

struct LabelPatternHash
{
	size_t operator()(const std::vector<int> &labels) const
	{
		size_t hash = 1469598103934665603ULL;
		for (size_t i = 0; i < labels.size(); ++i)
		{
			hash ^= (size_t)labels[i] + 1;
			hash *= 1099511628211ULL;
		}
		return hash;
	}
};

void compressPatterns(const std::vector<std::vector<int> > &runs,
	std::vector<Pattern> &patterns, std::vector<size_t> &particle_pattern)
{
	const size_t nr_particles = runs[0].size();
	std::unordered_map<std::vector<int>, size_t, LabelPatternHash> pattern_ids;
	pattern_ids.reserve(nr_particles);
	particle_pattern.resize(nr_particles);
	for (size_t particle = 0; particle < nr_particles; ++particle)
	{
		std::vector<int> labels(runs.size());
		for (size_t run = 0; run < runs.size(); ++run) labels[run] = runs[run][particle];
		std::pair<std::unordered_map<std::vector<int>, size_t, LabelPatternHash>::iterator, bool> inserted =
			pattern_ids.insert(std::make_pair(labels, patterns.size()));
		if (inserted.second)
			patterns.push_back(Pattern{labels, 0});
		particle_pattern[particle] = inserted.first->second;
		patterns[inserted.first->second].count += 1;
	}
}

double choose2(double value)
{
	return value * (value - 1.) / 2.;
}

std::vector<int> maximumAssignment(const std::vector<double> &weights, int size)
{
	if (size < 1 || weights.size() != (size_t)size * size)
		throw std::invalid_argument("maximumAssignment: invalid matrix size");

	double maximum = 0.;
	for (size_t i = 0; i < weights.size(); ++i)
		maximum = std::max(maximum, weights[i]);

	std::vector<double> u(size + 1, 0.), v(size + 1, 0.);
	std::vector<int> p(size + 1, 0), way(size + 1, 0);
	for (int row = 1; row <= size; ++row)
	{
		p[0] = row;
		int column0 = 0;
		std::vector<double> minv(size + 1, std::numeric_limits<double>::infinity());
		std::vector<char> used(size + 1, false);
		do
		{
			used[column0] = true;
			const int row0 = p[column0];
			double delta = std::numeric_limits<double>::infinity();
			int column1 = 0;
			for (int column = 1; column <= size; ++column)
			{
				if (used[column]) continue;
				const double cost = maximum - weights[(row0 - 1) * size + column - 1];
				const double current = cost - u[row0] - v[column];
				if (current < minv[column])
				{
					minv[column] = current;
					way[column] = column0;
				}
				if (minv[column] < delta)
				{
					delta = minv[column];
					column1 = column;
				}
			}
			for (int column = 0; column <= size; ++column)
			{
				if (used[column])
				{
					u[p[column]] += delta;
					v[column] -= delta;
				}
				else minv[column] -= delta;
			}
			column0 = column1;
		}
		while (p[column0] != 0);

		do
		{
			const int column1 = way[column0];
			p[column0] = p[column1];
			column0 = column1;
		}
		while (column0 != 0);
	}

	std::vector<int> assignment(size, -1);
	for (int column = 1; column <= size; ++column)
		assignment[p[column] - 1] = column - 1;
	return assignment;
}

template <class Result>
void assignmentDiagnostics(const std::vector<std::vector<int> > &runs,
	int source_classes, int nr_classes, Result &result)
{
	const size_t nr_particles = runs[0].size();
	result.run_adjusted_rand.resize(runs.size());
	result.run_mapped_agreement.resize(runs.size());
	result.agreement.assign(nr_particles, 0.);
	for (size_t run = 0; run < runs.size(); ++run)
	{
		result.run_adjusted_rand[run] = Class2DConsensus::adjustedRandIndex(runs[run], result.assignment, source_classes, nr_classes);
		const bool source_is_finer = source_classes >= nr_classes;
		const int rows = std::max(source_classes, nr_classes), columns = std::min(source_classes, nr_classes);
		const std::vector<int> &fine = source_is_finer ? runs[run] : result.assignment;
		const std::vector<int> &coarse = source_is_finer ? result.assignment : runs[run];
		std::vector<double> overlap((size_t)rows * columns, 0.);
		for (size_t p = 0; p < nr_particles; ++p) overlap[(size_t)fine[p] * columns + coarse[p]] += 1.;
		std::vector<int> mapping(rows);
		if (rows == columns) mapping = maximumAssignment(overlap, rows);
		else
		{
			std::vector<size_t> first_particle(columns, nr_particles);
			for (size_t p = 0; p < nr_particles; ++p)
				first_particle[coarse[p]] = std::min(first_particle[coarse[p]], p);
			for (int row = 0; row < rows; ++row)
			{
				const size_t offset = (size_t)row * columns;
				for (int column = 1; column < columns; ++column)
				{
					const int best = mapping[row];
					if (overlap[offset + column] > overlap[offset + best] ||
						(overlap[offset + column] == overlap[offset + best] && first_particle[column] < first_particle[best]))
						mapping[row] = column;
				}
			}
		}
		long int matches = 0;
		for (size_t p = 0; p < nr_particles; ++p)
			if (mapping[fine[p]] == coarse[p]) { ++matches; result.agreement[p] += 1.; }
		result.run_mapped_agreement[run] = (double)matches / nr_particles;
	}
	for (double &agreement : result.agreement) agreement /= runs.size();
}

size_t confusionIndex(int run, int consensus_class, int observed_class, int nr_classes, int source_classes)
{
	return ((size_t)run * nr_classes + consensus_class) * source_classes + observed_class;
}

void posteriorForPattern(
	const Pattern &pattern,
	const std::vector<double> &prior,
	const std::vector<double> &confusion,
	int nr_classes, int source_classes,
	std::vector<double> &posterior,
	double &log_normaliser)
{
	const int nr_runs = pattern.labels.size();
	double maximum = -std::numeric_limits<double>::infinity();
	for (int consensus_class = 0; consensus_class < nr_classes; ++consensus_class)
	{
		double value = std::log(prior[consensus_class]);
		for (int run = 0; run < nr_runs; ++run)
			value += std::log(confusion[confusionIndex(run, consensus_class, pattern.labels[run], nr_classes, source_classes)]);
		posterior[consensus_class] = value;
		maximum = std::max(maximum, value);
	}

	double sum = 0.;
	for (int consensus_class = 0; consensus_class < nr_classes; ++consensus_class)
	{
		posterior[consensus_class] = std::exp(posterior[consensus_class] - maximum);
		sum += posterior[consensus_class];
	}
	for (int consensus_class = 0; consensus_class < nr_classes; ++consensus_class)
		posterior[consensus_class] /= sum;
	log_normaliser = maximum + std::log(sum);
}

FitState fitFromAssignments(
	const std::vector<Pattern> &patterns,
	const std::vector<std::vector<int> > &runs,
	const std::vector<int> &initial_assignment,
	int nr_classes, int source_classes,
	int maximum_iterations,
	double relative_tolerance,
	double pseudocount,
	int nr_threads)
{
	const int nr_runs = runs.size();
	const long int nr_particles = runs[0].size();
	FitState state;
	state.prior.assign(nr_classes, pseudocount);
	state.confusion.assign((size_t)nr_runs * nr_classes * source_classes, pseudocount);

	for (long int particle = 0; particle < nr_particles; ++particle)
	{
		const int consensus_class = initial_assignment[particle];
		state.prior[consensus_class] += 1.;
		for (int run = 0; run < nr_runs; ++run)
			state.confusion[confusionIndex(run, consensus_class, runs[run][particle], nr_classes, source_classes)] += 1.;
	}
	for (int consensus_class = 0; consensus_class < nr_classes; ++consensus_class)
	{
		state.prior[consensus_class] /= nr_particles + pseudocount * nr_classes;
		for (int run = 0; run < nr_runs; ++run)
		{
			double total = 0.;
			for (int observed = 0; observed < source_classes; ++observed)
				total += state.confusion[confusionIndex(run, consensus_class, observed, nr_classes, source_classes)];
			for (int observed = 0; observed < source_classes; ++observed)
				state.confusion[confusionIndex(run, consensus_class, observed, nr_classes, source_classes)] /= total;
		}
	}

	double previous_log_likelihood = -std::numeric_limits<double>::infinity();
	for (int iteration = 1; iteration <= maximum_iterations; ++iteration)
	{
		int thread_count = 1;
#ifdef _OPENMP
		thread_count = std::max(1, nr_threads);
#endif
		std::vector<std::vector<double> > thread_class_counts(thread_count, std::vector<double>(nr_classes, 0.));
		std::vector<std::vector<double> > thread_observation_counts(thread_count,
			std::vector<double>((size_t)nr_runs * nr_classes * source_classes, 0.));
		std::vector<double> thread_log_likelihood(thread_count, 0.);
#ifdef _OPENMP
#pragma omp parallel num_threads(thread_count)
#endif
		{
			int thread = 0;
#ifdef _OPENMP
			thread = omp_get_thread_num();
#endif
			std::vector<double> posterior(nr_classes);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
			for (long int p = 0; p < (long int)patterns.size(); ++p)
			{
				double log_normaliser;
				posteriorForPattern(patterns[p], state.prior, state.confusion, nr_classes, source_classes, posterior, log_normaliser);
				thread_log_likelihood[thread] += patterns[p].count * log_normaliser;
				for (int consensus_class = 0; consensus_class < nr_classes; ++consensus_class)
				{
					const double mass = patterns[p].count * posterior[consensus_class];
					thread_class_counts[thread][consensus_class] += mass;
					for (int run = 0; run < nr_runs; ++run)
						thread_observation_counts[thread][confusionIndex(run, consensus_class, patterns[p].labels[run], nr_classes, source_classes)] += mass;
				}
			}
		}
		std::vector<double> class_counts(nr_classes, pseudocount);
		std::vector<double> observation_counts((size_t)nr_runs * nr_classes * source_classes, pseudocount);
		double log_likelihood = 0.;
		for (int thread = 0; thread < thread_count; ++thread)
		{
			log_likelihood += thread_log_likelihood[thread];
			for (int k = 0; k < nr_classes; ++k) class_counts[k] += thread_class_counts[thread][k];
			for (size_t i = 0; i < observation_counts.size(); ++i) observation_counts[i] += thread_observation_counts[thread][i];
		}

		const double prior_total = nr_particles + pseudocount * nr_classes;
		for (int consensus_class = 0; consensus_class < nr_classes; ++consensus_class)
		{
			state.prior[consensus_class] = class_counts[consensus_class] / prior_total;
			for (int run = 0; run < nr_runs; ++run)
			{
				double total = 0.;
				for (int observed = 0; observed < source_classes; ++observed)
					total += observation_counts[confusionIndex(run, consensus_class, observed, nr_classes, source_classes)];
				for (int observed = 0; observed < source_classes; ++observed)
					state.confusion[confusionIndex(run, consensus_class, observed, nr_classes, source_classes)] =
						observation_counts[confusionIndex(run, consensus_class, observed, nr_classes, source_classes)] / total;
			}
		}

		state.iterations = iteration;
		state.log_likelihood = log_likelihood;
		if (std::isfinite(previous_log_likelihood))
		{
			const double denominator = std::max(1., std::fabs(previous_log_likelihood));
			if (std::fabs(log_likelihood - previous_log_likelihood) / denominator < relative_tolerance)
				break;
		}
		previous_log_likelihood = log_likelihood;
	}
	state.log_likelihood = 0.;
	std::vector<double> final_posterior(nr_classes);
	for (size_t pattern = 0; pattern < patterns.size(); ++pattern)
	{
		double log_normaliser;
		posteriorForPattern(patterns[pattern], state.prior, state.confusion,
			nr_classes, source_classes, final_posterior, log_normaliser);
		state.log_likelihood += patterns[pattern].count * log_normaliser;
	}
	return state;
}

}

double Class2DConsensus::adjustedRandIndex(
	const std::vector<int> &lhs, const std::vector<int> &rhs, int nr_classes)
{
	return adjustedRandIndex(lhs, rhs, nr_classes, nr_classes);
}

double Class2DConsensus::adjustedRandIndex(
	const std::vector<int> &lhs,
	const std::vector<int> &rhs,
	int lhs_classes, int rhs_classes)
{
	if (lhs.size() != rhs.size() || lhs.empty() || lhs_classes < 1 || rhs_classes < 1)
		throw std::invalid_argument("adjustedRandIndex: invalid assignments");
	std::vector<double> rows(lhs_classes, 0.), columns(rhs_classes, 0.);
	std::vector<double> cells((size_t)lhs_classes * rhs_classes, 0.);
	for (size_t i = 0; i < lhs.size(); ++i)
	{
		if (lhs[i] < 0 || lhs[i] >= lhs_classes || rhs[i] < 0 || rhs[i] >= rhs_classes)
			throw std::invalid_argument("adjustedRandIndex: class outside valid range");
		rows[lhs[i]] += 1.;
		columns[rhs[i]] += 1.;
		cells[(size_t)lhs[i] * rhs_classes + rhs[i]] += 1.;
	}
	double cell_pairs = 0., row_pairs = 0., column_pairs = 0.;
	for (size_t i = 0; i < cells.size(); ++i) cell_pairs += choose2(cells[i]);
	for (double count : rows) row_pairs += choose2(count);
	for (double count : columns) column_pairs += choose2(count);
	const double all_pairs = choose2(lhs.size());
	if (all_pairs == 0.) return 1.;
	const double expected = row_pairs * column_pairs / all_pairs;
	const double denominator = .5 * (row_pairs + column_pairs) - expected;
	if (std::fabs(denominator) < 1.e-15)
		return std::fabs(cell_pairs - expected) < 1.e-15 ? 1. : 0.;
	return (cell_pairs - expected) / denominator;
}

Class2DConsensusResult Class2DConsensus::fitWithClassCount(
	const std::vector<std::vector<int> > &runs,
	int source_classes, int nr_classes,
	int maximum_iterations,
	double relative_tolerance,
	double pseudocount,
	int nr_threads)
{
	if (runs.size() < 2 || runs[0].empty() || source_classes < 2 || nr_classes < 2 ||
		maximum_iterations < 1 || !std::isfinite(relative_tolerance) || relative_tolerance <= 0. ||
		!std::isfinite(pseudocount) || pseudocount <= 0. || nr_threads < 1)
		throw std::invalid_argument("Class2DConsensus::fit: invalid parameters");
	const size_t nr_particles = runs[0].size();
	for (size_t run = 0; run < runs.size(); ++run)
	{
		if (runs[run].size() != nr_particles)
			throw std::invalid_argument("Class2DConsensus::fit: replicas have different particle counts");
		for (size_t particle = 0; particle < nr_particles; ++particle)
			if (runs[run][particle] < 0 || runs[run][particle] >= source_classes)
				throw std::invalid_argument("Class2DConsensus::fit: class outside valid range");
	}

	std::vector<double> mean_ari(runs.size(), 0.);
	for (size_t lhs = 0; lhs < runs.size(); ++lhs)
		for (size_t rhs = lhs + 1; rhs < runs.size(); ++rhs)
		{
			const double ari = adjustedRandIndex(runs[lhs], runs[rhs], source_classes);
			mean_ari[lhs] += ari;
			mean_ari[rhs] += ari;
		}
	for (size_t run = 0; run < runs.size(); ++run)
		mean_ari[run] /= runs.size() - 1;

	std::vector<int> candidates(runs.size());
	std::iota(candidates.begin(), candidates.end(), 0);
	std::stable_sort(candidates.begin(), candidates.end(), [&](int lhs, int rhs)
	{
		return mean_ari[lhs] > mean_ari[rhs];
	});

	std::vector<Pattern> patterns;
	std::vector<size_t> particle_pattern;
	compressPatterns(runs, patterns, particle_pattern);

	FitState best;
	best.log_likelihood = -std::numeric_limits<double>::infinity();
	int best_initial_run = candidates[0];
	std::vector<size_t> frequent(patterns.size());
	std::iota(frequent.begin(), frequent.end(), 0);
	std::stable_sort(frequent.begin(), frequent.end(), [&](size_t a, size_t b)
	{
		return patterns[a].count > patterns[b].count;
	});
	const int nr_starts = std::min(size_t(3), source_classes == nr_classes ? candidates.size() : patterns.size());
	for (int start = 0; start < nr_starts; ++start)
	{
		std::vector<int> initial_assignment;
		if (source_classes == nr_classes) initial_assignment = runs[candidates[start]];
		else
		{
			// Patterns are ordered by their first particle, so ties do not depend
			// on arbitrary numeric class labels. Select distinct farthest seeds.
			std::vector<int> distance(patterns.size(), (int)runs.size() + 1);
			std::vector<int> nearest(patterns.size(), 0);
			size_t seed = frequent[start];
			const size_t seed_count = std::min((size_t)nr_classes, patterns.size());
			for (size_t k = 0; k < seed_count; ++k)
			{
				for (size_t p = 0; p < patterns.size(); ++p)
				{
					int d = 0;
					for (size_t run = 0; run < runs.size(); ++run)
						d += patterns[p].labels[run] != patterns[seed].labels[run];
					if (d < distance[p]) { distance[p] = d; nearest[p] = (int)k; }
				}
				double best_score = -1.;
				for (size_t p = 0; p < patterns.size(); ++p)
				{
					const double score = (double)patterns[p].count * distance[p];
					if (score > best_score) { best_score = score; seed = p; }
				}
			}
			initial_assignment.resize(nr_particles);
			for (size_t p = 0; p < nr_particles; ++p) initial_assignment[p] = nearest[particle_pattern[p]];
		}
		FitState current = fitFromAssignments(patterns, runs, initial_assignment, nr_classes, source_classes,
			maximum_iterations, relative_tolerance, pseudocount, nr_threads);
		if (current.log_likelihood > best.log_likelihood)
		{
			best = current;
			best_initial_run = candidates[source_classes == nr_classes ? start : 0];
		}
	}

	std::vector<int> pattern_assignment(patterns.size());
	std::vector<double> pattern_probability(patterns.size());
	std::vector<double> pattern_entropy(patterns.size());
	std::vector<double> old_class_posterior_mass(nr_classes, 0.);
	std::vector<double> posterior(nr_classes);
	for (size_t pattern = 0; pattern < patterns.size(); ++pattern)
	{
		double ignored;
		posteriorForPattern(patterns[pattern], best.prior, best.confusion, nr_classes, source_classes, posterior, ignored);
		const int assigned = std::max_element(posterior.begin(), posterior.end()) - posterior.begin();
		pattern_assignment[pattern] = assigned;
		pattern_probability[pattern] = posterior[assigned];
		double entropy = 0.;
		for (int k = 0; k < nr_classes; ++k)
		{
			old_class_posterior_mass[k] += patterns[pattern].count * posterior[k];
			if (posterior[k] > 0.) entropy -= posterior[k] * std::log(posterior[k]);
		}
		pattern_entropy[pattern] = entropy / std::log((double)nr_classes);
	}

	std::vector<int> canonical(nr_classes, -1);
	if (source_classes == nr_classes)
	{
		std::vector<double> overlap((size_t)nr_classes * nr_classes, 0.);
		for (size_t particle = 0; particle < nr_particles; ++particle)
			overlap[(size_t)pattern_assignment[particle_pattern[particle]] * nr_classes + runs[best_initial_run][particle]] += 1.;
		canonical = maximumAssignment(overlap, nr_classes);
	}
	else
	{
		int next = 0;
		for (size_t particle = 0; particle < nr_particles; ++particle)
		{
			int &label = canonical[pattern_assignment[particle_pattern[particle]]];
			if (label < 0) label = next++;
		}
		for (int &label : canonical) if (label < 0) label = next++;
	}

	Class2DConsensusResult result;
	result.source_classes = source_classes;
	result.consensus_classes = nr_classes;
	result.nr_patterns = patterns.size();
	result.assignment.resize(nr_particles);
	result.probability.resize(nr_particles);
	result.entropy.resize(nr_particles);
	result.class_posterior_mass.assign(nr_classes, 0.);
	result.anchor_run = best_initial_run;
	result.iterations = best.iterations;
	result.log_likelihood = best.log_likelihood;
	result.confusion.assign(best.confusion.size(), 0.);
	for (size_t run = 0; run < runs.size(); ++run)
		for (int old_class = 0; old_class < nr_classes; ++old_class)
			for (int observed = 0; observed < source_classes; ++observed)
				result.confusion[confusionIndex(run, canonical[old_class], observed, nr_classes, source_classes)] =
					best.confusion[confusionIndex(run, old_class, observed, nr_classes, source_classes)];

	for (int k = 0; k < nr_classes; ++k)
		result.class_posterior_mass[canonical[k]] = old_class_posterior_mass[k];
	for (size_t particle = 0; particle < nr_particles; ++particle)
	{
		const size_t pattern = particle_pattern[particle];
		result.assignment[particle] = canonical[pattern_assignment[pattern]];
		result.probability[particle] = pattern_probability[pattern];
		result.entropy[particle] = pattern_entropy[pattern];
	}

	// Keep empty slots: smoothed posterior mass need not be zero when MAP
	// occupancy is zero. For unequal counts, agreement measures coarsening.
	assignmentDiagnostics(runs, source_classes, nr_classes, result);
	return result;
}

Class2DConsensusResult Class2DConsensus::fit(
	const std::vector<std::vector<int> > &runs, int nr_classes,
	int maximum_iterations, double relative_tolerance, double pseudocount, int nr_threads)
{
	return fitWithClassCount(runs, nr_classes, nr_classes, maximum_iterations,
		relative_tolerance, pseudocount, nr_threads);
}

namespace class2d_nmf_detail
{
double profileGradient(const std::vector<double> &h, const std::vector<double> &s,
    const std::vector<double> &t, int components, int features, int component, int feature)
{
	double gradient = -t[(size_t)component * features + feature];
	for (int b = 0; b < components; ++b)
		gradient += s[(size_t)component * components + b] * h[(size_t)b * features + feature];
	return gradient;
}

void projectSimplex(std::vector<double> &values, std::vector<double> &scratch)
{
	scratch = values;
	std::sort(scratch.begin(), scratch.end(), std::greater<double>());
	double sum = 0., threshold = 0.;
	for (size_t i = 0; i < scratch.size(); ++i)
	{
		sum += scratch[i];
		const double candidate = (sum - 1.) / (i + 1);
		if (scratch[i] > candidate) threshold = candidate;
	}
	for (double &value : values) value = std::max(0., value - threshold);
}

void gramMatrix(const std::vector<double> &h, int components, int features,
                std::vector<double> &gram)
{
	gram.assign((size_t)components * components, 0.);
	for (int a = 0; a < components; ++a)
		for (int b = 0; b <= a; ++b)
		{
			double dot = 0.;
			for (int j = 0; j < features; ++j)
				dot += h[(size_t)a * features + j] * h[(size_t)b * features + j];
			gram[(size_t)a * components + b] = gram[(size_t)b * components + a] = dot;
		}
}

double patternObjectiveGradient(const std::vector<int> &labels, int source_classes,
    const std::vector<double> &h, const std::vector<double> &gram,
    const double *membership, int components, std::vector<double> &gradient)
{
	const size_t features = labels.size() * source_classes;
	double residual = (double)labels.size();
	gradient.resize(components);
	for (int a = 0; a < components; ++a)
	{
		double cross = 0., quadratic = 0.;
		for (size_t r = 0; r < labels.size(); ++r)
			cross += h[(size_t)a * features + r * source_classes + labels[r]];
		for (int b = 0; b < components; ++b)
			quadratic += gram[(size_t)a * components + b] * membership[b];
		gradient[a] = quadratic - cross;
		residual += membership[a] * (quadratic - 2. * cross);
	}
	return .5 * residual;
}
}

namespace
{
size_t nmfProduct(size_t a, size_t b)
{
	if (b && a > std::numeric_limits<size_t>::max() / b)
		throw std::invalid_argument("Sparse NMF dimensions exceed the addressable size");
	return a * b;
}

void nmfAdd(size_t &total, size_t count)
{
	if (total > std::numeric_limits<size_t>::max() - count)
		throw std::invalid_argument("Sparse NMF workspace exceeds the addressable size");
	total += count;
}

double nmfLipschitz(const std::vector<double> &gram, int c)
{
	double bound = 0.;
	for (int a = 0; a < c; ++a)
	{
		double sum = 0.;
		for (int b = 0; b < c; ++b) sum += std::fabs(gram[(size_t)a * c + b]);
		bound = std::max(bound, sum);
	}
	if (!std::isfinite(bound) || bound <= 0.)
		throw std::runtime_error("Sparse NMF encountered an invalid gradient step");
	return bound;
}

double nmfObjective(const std::vector<Pattern> &patterns, const std::vector<double> &w,
    const std::vector<double> &h, int k, int c, double particles, int threads)
{
	std::vector<double> gram;
	class2d_nmf_detail::gramMatrix(h, c, (int)patterns[0].labels.size() * k, gram);
	std::vector<double> sums(threads, 0.);
	std::vector<std::vector<double> > gradients(threads, std::vector<double>(c));
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
	{
		int thread = 0;
#ifdef _OPENMP
		thread = omp_get_thread_num();
#endif
		std::vector<double> &gradient = gradients[thread];
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
		for (long long p = 0; p < (long long)patterns.size(); ++p)
			sums[thread] += (patterns[p].count / particles) *
				class2d_nmf_detail::patternObjectiveGradient(patterns[p].labels, k, h, gram,
					&w[(size_t)p * c], c, gradient);
	}
	double value = std::accumulate(sums.begin(), sums.end(), 0.) / patterns[0].labels.size();
	if (!std::isfinite(value) || value < -1.e-10)
		throw std::runtime_error("Sparse NMF encountered a non-finite or negative objective");
	return std::max(0., value); // Cancellation near an exact reconstruction.
}

Class2DConsensusNmfResult fitSparseNmfImpl(const std::vector<std::vector<int> > &runs,
    int k, int requested, const Class2DConsensusNmfOptions &options, int threads)
{
	if (runs.size() < 2 || runs[0].empty() || k < 2 || requested < 2 || threads < 1 ||
		options.maximum_iterations < 1 || options.starts < 1 ||
		!std::isfinite(options.tolerance) || options.tolerance <= 0.)
		throw std::invalid_argument("Sparse NMF: invalid parameters");
	const size_t n = runs[0].size();
	const size_t feature_count = nmfProduct(runs.size(), (size_t)k);
	if (feature_count > (size_t)std::numeric_limits<int>::max() ||
		n > (size_t)std::numeric_limits<long int>::max())
		throw std::invalid_argument("Sparse NMF: dimensions exceed supported index range");
	const int d = (int)feature_count;
	for (const auto &run : runs)
	{
		if (run.size() != n) throw std::invalid_argument("Sparse NMF: replicas have different particle counts");
		for (int label : run)
			if (label < 0 || label >= k) throw std::invalid_argument("Sparse NMF: class outside valid range");
	}
#ifndef _OPENMP
	threads = 1;
#endif
	std::vector<Pattern> patterns;
	std::vector<size_t> particle_pattern;
	compressPatterns(runs, patterns, particle_pattern);
	const int c = (int)std::min((size_t)requested, patterns.size());
	threads = (int)std::min((size_t)threads, patterns.size());
	const int starts = (int)std::min((size_t)options.starts, patterns.size());
	const size_t pc = nmfProduct(patterns.size(), c), cd = nmfProduct(c, d), cc = nmfProduct(c, c);
	// Conservative numeric-buffer bound, including summaries/output, but not
	// input STAR tables, input labels, or the pattern hash table.
	size_t doubles = pc;
	nmfAdd(doubles, nmfProduct((size_t)threads + 4, cd));
	nmfAdd(doubles, nmfProduct((size_t)threads + 4, cc));
	nmfAdd(doubles, nmfProduct(3, nmfProduct(requested, d)));
	nmfAdd(doubles, nmfProduct(k, k)); // Pairwise source ARI contingency table.
	nmfAdd(doubles, nmfProduct(12, n));
	nmfAdd(doubles, nmfProduct((size_t)threads + 4, 4 * (size_t)c + 2 * (size_t)k));
	nmfAdd(doubles, nmfProduct(starts, (size_t)options.maximum_iterations + 1));
	const size_t bytes = nmfProduct(doubles, sizeof(double));
	if (options.report_workspace_bytes) options.report_workspace_bytes(bytes);

	Class2DConsensusNmfResult result;
	result.source_classes = k;
	result.consensus_classes = requested;
	result.fitted_components = c;
	result.nr_patterns = patterns.size();
	result.workspace_bytes = bytes;
	result.objective = std::numeric_limits<double>::infinity();
	std::vector<double> mean_ari(runs.size(), 0.);
	for (size_t a = 0; a < runs.size(); ++a)
		for (size_t b = a + 1; b < runs.size(); ++b)
		{
			const double ari = Class2DConsensus::adjustedRandIndex(runs[a], runs[b], k);
			mean_ari[a] += ari; mean_ari[b] += ari;
		}
	result.anchor_run = std::max_element(mean_ari.begin(), mean_ari.end()) - mean_ari.begin();
	std::vector<size_t> frequent(patterns.size());
	std::iota(frequent.begin(), frequent.end(), 0);
	std::stable_sort(frequent.begin(), frequent.end(), [&](size_t a, size_t b) { return patterns[a].count > patterns[b].count; });
	std::vector<int> best_assignment;
	std::vector<double> best_membership, best_entropy;
	for (int start = 0; start < starts; ++start)
	{
		Class2DConsensusNmfStart status;
		try
		{
			std::vector<double> w(pc, .05 / c), h(cd, 0.);
			std::vector<int> distance(patterns.size(), (int)runs.size() + 1), nearest(patterns.size(), 0);
			size_t seed = frequent[start];
			for (int a = 0; a < c; ++a)
			{
				for (size_t r = 0; r < runs.size(); ++r) h[(size_t)a * d + r * k + patterns[seed].labels[r]] = 1.;
				for (size_t p = 0; p < patterns.size(); ++p)
				{
					int dist = 0;
					for (size_t r = 0; r < runs.size(); ++r) dist += patterns[p].labels[r] != patterns[seed].labels[r];
					if (dist < distance[p]) { distance[p] = dist; nearest[p] = a; }
				}
				double score = -1.;
				for (size_t p = 0; p < patterns.size(); ++p)
					if ((double)patterns[p].count * distance[p] > score)
					{ score = (double)patterns[p].count * distance[p]; seed = p; }
			}
			for (size_t p = 0; p < patterns.size(); ++p) w[p * c + nearest[p]] += .95;
			status.objective_history.push_back(nmfObjective(patterns, w, h, k, c, (double)n, threads));
			int stable = 0;
			std::vector<double> gram, s(cc), t(cd), next_h(cd);
			std::vector<std::vector<double> > thread_s(threads, std::vector<double>(cc)), thread_t(threads, std::vector<double>(cd));
			std::vector<std::vector<double> > gradients(threads, std::vector<double>(c)), rows(threads, std::vector<double>(c)), scratches(threads, std::vector<double>(c));
			for (int iteration = 0; iteration < options.maximum_iterations; ++iteration)
			{
				class2d_nmf_detail::gramMatrix(h, c, d, gram);
				const double step_w = 1. / nmfLipschitz(gram, c);
				std::vector<double> changes(threads, 0.);
				// Clear all buffers even when OpenMP dynamically uses fewer threads.
				for (int thread = 0; thread < threads; ++thread)
				{
					std::fill(thread_s[thread].begin(), thread_s[thread].end(), 0.);
					std::fill(thread_t[thread].begin(), thread_t[thread].end(), 0.);
				}
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
				{
					int thread = 0;
#ifdef _OPENMP
					thread = omp_get_thread_num();
#endif
					std::vector<double> &gradient = gradients[thread], &row = rows[thread], &scratch = scratches[thread];
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
					for (long long p = 0; p < (long long)patterns.size(); ++p)
					{
						double *wp = &w[(size_t)p * c];
						class2d_nmf_detail::patternObjectiveGradient(patterns[p].labels, k, h, gram, wp, c, gradient);
						for (int a = 0; a < c; ++a) row[a] = wp[a] - step_w * gradient[a];
						class2d_nmf_detail::projectSimplex(row, scratch);
						for (int a = 0; a < c; ++a)
						{ changes[thread] = std::max(changes[thread], std::fabs(row[a] - wp[a])); wp[a] = row[a]; }
						const double weight = (double)patterns[p].count / n;
						for (int a = 0; a < c; ++a)
						{
							const double mass = weight * wp[a];
							for (int b = 0; b < c; ++b) thread_s[thread][(size_t)a * c + b] += mass * wp[b];
							for (size_t r = 0; r < runs.size(); ++r) thread_t[thread][(size_t)a * d + r * k + patterns[p].labels[r]] += mass;
						}
					}
				}
				std::fill(s.begin(), s.end(), 0.); std::fill(t.begin(), t.end(), 0.);
				for (int thread = 0; thread < threads; ++thread)
				{
					for (size_t i = 0; i < cc; ++i) s[i] += thread_s[thread][i];
					for (size_t i = 0; i < cd; ++i) t[i] += thread_t[thread][i];
				}
				const double step_h = 1. / nmfLipschitz(s, c);
				double change = *std::max_element(changes.begin(), changes.end());
				std::vector<double> block(k), scratch(k);
				for (int a = 0; a < c; ++a)
					for (size_t r = 0; r < runs.size(); ++r)
					{
						for (int j = 0; j < k; ++j)
						{
							const size_t index = (size_t)a * d + r * k + j;
							const double gradient = class2d_nmf_detail::profileGradient(h, s, t, c, d, a, (int)r * k + j);
							block[j] = h[index] - step_h * gradient;
						}
						class2d_nmf_detail::projectSimplex(block, scratch);
						for (int j = 0; j < k; ++j)
						{
							const size_t index = (size_t)a * d + r * k + j;
							change = std::max(change, std::fabs(block[j] - h[index])); next_h[index] = block[j];
						}
					}
				h.swap(next_h);
				const double objective = nmfObjective(patterns, w, h, k, c, (double)n, threads);
				const double previous = status.objective_history.back();
				if (objective > previous + 1.e-10 * std::max(1., previous))
					throw std::runtime_error("Sparse NMF objective increased beyond roundoff tolerance");
				status.objective_history.push_back(objective);
				stable = (std::fabs(previous - objective) / std::max(1., previous) < options.tolerance && change < options.tolerance) ? stable + 1 : 0;
				if (stable >= 3) { status.converged = true; status.termination = "converged"; break; }
			}
			const double objective = status.objective_history.back();
			if (objective < result.objective - 1.e-12)
			{
				result.objective = objective; result.selected_start = start;
				result.iterations = (int)status.objective_history.size() - 1; result.converged = status.converged;
				best_assignment.resize(patterns.size()); best_membership.resize(patterns.size()); best_entropy.resize(patterns.size());
				std::vector<int> canonical(requested, -1);
				int next = 0;
				for (size_t p = 0; p < patterns.size(); ++p)
				{
					int assigned = 0;
					for (int a = 1; a < c; ++a) if (w[p * c + a] > w[p * c + assigned] + 1.e-12) assigned = a;
					if (canonical[assigned] < 0) canonical[assigned] = next++;
					best_assignment[p] = assigned; best_membership[p] = w[p * c + assigned];
					double entropy = 0.;
					for (int a = 0; a < c; ++a) if (w[p * c + a] > 0.) entropy -= w[p * c + a] * std::log(w[p * c + a]);
					best_entropy[p] = std::max(0., std::min(1., entropy / std::log((double)requested)));
				}
				for (int &label : canonical) if (label < 0) label = next++;
				result.class_membership_mass.assign(requested, 0.);
				for (size_t p = 0; p < patterns.size(); ++p)
				{
					best_assignment[p] = canonical[best_assignment[p]];
					for (int a = 0; a < c; ++a) result.class_membership_mass[canonical[a]] += patterns[p].count * w[p * c + a];
				}
				result.components.assign(nmfProduct(requested, d), 1. / k);
				for (int a = 0; a < c; ++a)
					for (size_t r = 0; r < runs.size(); ++r)
						for (int j = 0; j < k; ++j)
							result.components[(r * requested + canonical[a]) * k + j] = h[(size_t)a * d + r * k + j];
			}
		}
		catch (const std::runtime_error &error) { status.converged = false; status.termination = error.what(); }
		result.starts.push_back(std::move(status));
	}
	if (!std::isfinite(result.objective)) throw std::runtime_error("Sparse NMF: all starts failed numerically");
	result.assignment.resize(n); result.membership.resize(n); result.entropy.resize(n);
	for (size_t p = 0; p < n; ++p)
	{
		result.assignment[p] = best_assignment[particle_pattern[p]];
		result.membership[p] = best_membership[particle_pattern[p]];
		result.entropy[p] = best_entropy[particle_pattern[p]];
	}
	assignmentDiagnostics(runs, k, requested, result);
	return result;
}
}

Class2DConsensusNmfResult Class2DConsensus::fitSparseNmf(
    const std::vector<std::vector<int> > &runs, int source_classes, int consensus_classes,
    const Class2DConsensusNmfOptions &options, int nr_threads)
{
	try { return fitSparseNmfImpl(runs, source_classes, consensus_classes, options, nr_threads); }
	catch (const std::bad_alloc &) { throw std::runtime_error("Sparse NMF allocation failed; reduce the consensus class count or use categorical EM"); }
	catch (const std::length_error &) { throw std::runtime_error("Sparse NMF dimensions exceed the maximum allocatable container size"); }
}
