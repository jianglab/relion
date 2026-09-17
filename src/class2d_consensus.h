#ifndef SRC_CLASS2D_CONSENSUS_H_
#define SRC_CLASS2D_CONSENSUS_H_

#include <vector>
#include <cstddef>

struct Class2DConsensusResult
{
	std::vector<int> assignment;
	std::vector<double> probability;
	std::vector<double> entropy;
	std::vector<double> agreement;
	std::vector<double> run_adjusted_rand;
	std::vector<double> run_mapped_agreement;
	std::vector<double> class_posterior_mass;
	std::vector<double> confusion;
	int source_classes;
	int consensus_classes;
	std::size_t nr_patterns;
	int anchor_run;
	int iterations;
	double log_likelihood;
};

class Class2DConsensus
{
public:
	// nr_classes is the number of class slots, including unoccupied classes.
	// Returned assignments retain this label space and may leave slots empty.
	static Class2DConsensusResult fit(
		const std::vector<std::vector<int> > &run_assignments,
		int nr_classes,
		int maximum_iterations = 200,
		double relative_tolerance = 1.e-6,
		double pseudocount = 1.0,
		int nr_threads = 1);

	// Particle order defines stable tie breaks. The CLI orders unequal-count
	// inputs by rlnImageName before calling this entry point.
	static Class2DConsensusResult fitWithClassCount(
		const std::vector<std::vector<int> > &run_assignments,
		int source_classes,
		int consensus_classes,
		int maximum_iterations = 200,
		double relative_tolerance = 1.e-6,
		double pseudocount = 1.0,
		int nr_threads = 1);

	static double adjustedRandIndex(
		const std::vector<int> &lhs,
		const std::vector<int> &rhs,
		int nr_classes);

	static double adjustedRandIndex(
		const std::vector<int> &lhs,
		const std::vector<int> &rhs,
		int lhs_classes, int rhs_classes);
};

#endif
