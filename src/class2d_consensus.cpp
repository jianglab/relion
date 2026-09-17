#include "src/class2d_consensus.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
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

	std::unordered_map<std::vector<int>, size_t, LabelPatternHash> pattern_ids;
	pattern_ids.reserve(nr_particles);
	std::vector<Pattern> patterns;
	std::vector<size_t> particle_pattern(nr_particles);
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
	result.run_adjusted_rand.resize(runs.size());
	result.run_mapped_agreement.resize(runs.size());
	result.agreement.assign(nr_particles, 0.);
	for (size_t run = 0; run < runs.size(); ++run)
	{
		result.run_adjusted_rand[run] = adjustedRandIndex(runs[run], result.assignment, source_classes, nr_classes);
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
	return result;
}

Class2DConsensusResult Class2DConsensus::fit(
	const std::vector<std::vector<int> > &runs, int nr_classes,
	int maximum_iterations, double relative_tolerance, double pseudocount, int nr_threads)
{
	return fitWithClassCount(runs, nr_classes, nr_classes, maximum_iterations,
		relative_tolerance, pseudocount, nr_threads);
}
