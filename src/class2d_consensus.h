#ifndef SRC_CLASS2D_CONSENSUS_H_
#define SRC_CLASS2D_CONSENSUS_H_

#include <vector>
#include <cstddef>
#include <string>
#include <functional>

struct Class2DConsensusNmfOptions
{
	int maximum_iterations = 500;
	double tolerance = 1.e-6;
	int starts = 3;
	// Called after pattern compression, before allocating factors/workspace.
	std::function<void(std::size_t)> report_workspace_bytes;
};

struct Class2DConsensusNmfStart
{
	std::vector<double> objective_history; // Includes the initial objective.
	bool converged = false;
	std::string termination = "iteration_limit";
};

struct Class2DConsensusNmfResult
{
	std::vector<int> assignment;
	std::vector<double> membership, entropy, agreement;
	std::vector<double> run_adjusted_rand, run_mapped_agreement;
	std::vector<double> class_membership_mass;
	// Layout: [run][consensus class][source class], including inactive slots.
	std::vector<double> components;
	std::vector<Class2DConsensusNmfStart> starts;
	int source_classes = 0, consensus_classes = 0, fitted_components = 0;
	std::size_t nr_patterns = 0, workspace_bytes = 0;
	int anchor_run = 0, selected_start = 0, iterations = 0;
	double objective = 0.;
	bool converged = false;
};

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
	// Input particle order defines identity-based ties. The CLI sorts by image
	// name for NMF, independently of the requested class count.
	static Class2DConsensusNmfResult fitSparseNmf(
		const std::vector<std::vector<int> > &run_assignments,
		int source_classes, int consensus_classes,
		const Class2DConsensusNmfOptions &options = Class2DConsensusNmfOptions(),
		int nr_threads = 1);

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
