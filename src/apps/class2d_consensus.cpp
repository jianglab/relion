/***************************************************************************
 * Class2D consensus assignment from parallel RELION replicas.
 ***************************************************************************/

#include "src/args.h"
#include "src/class2d_consensus.h"
#include "src/class2d_consensus_metadata.h"
#include "src/filename.h"
#include "src/jaz/single_particle/obs_model.h"
#include "src/macros.h"
#include "src/metadata_table.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <climits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

class Class2DConsensusProgram
{
public:
	IOParser parser;
	FileName input_optimiser, output_root;
	int nr_runs, nr_threads, verb, requested_classes;

	void read(int argc, char **argv)
	{
		parser.setCommandLine(argc, argv);
		const int section = parser.addSection("Class2D consensus options");
		(void)section;
		input_optimiser = parser.getOption("--i", "Any runNNN_itXXX_optimiser.star from a parallel Class2D job");
		output_root = parser.getOption("--o", "Output root name", "consensus");
		nr_runs = textToInteger(parser.getOption("--nr_runs", "Number of parallel Class2D replicas"));
		nr_threads = textToInteger(parser.getOption("--j", "Number of threads reserved for consensus preparation", "1"));
		const std::string count = parser.getOption("--K", "Consensus classes (0 inherits the source count)", "0");
		if (count.empty() || count.find_first_not_of("0123456789") != std::string::npos)
			REPORT_ERROR("Consensus class count must be 0 or an integer at least 2");
		try
		{
			const long long value = std::stoll(count);
			if (value == 1 || value > INT_MAX) REPORT_ERROR("Invalid consensus class count");
			requested_classes = (int)value;
		}
		catch (const std::exception &) { REPORT_ERROR("Invalid consensus class count"); }
		if (parser.checkOption("--legacy_keep_alignments", "Report an ignored legacy keep-alignments setting"))
			std::cout << "Legacy keep-alignments setting ignored: fresh initialization always resets fitted poses." << std::endl;
		// Accepted for scripts from older versions. Fresh preparation always resets poses.
		parser.checkOption("--reset_alignments", "Legacy option: fresh initialization always resets fitted poses");
		verb = textToInteger(parser.getOption("--verb", "Verbosity", "1"));
		if (parser.checkForErrors(verb))
			REPORT_ERROR("Errors encountered on the command line");
		if (nr_runs < 2) REPORT_ERROR("Class2D consensus requires at least two replicas");
		if (nr_threads < 1) REPORT_ERROR("Number of threads must be positive");
	}

	static std::string directoryOf(const std::string &path)
	{
		const size_t slash = path.find_last_of("/\\");
		return slash == std::string::npos ? "" : path.substr(0, slash + 1);
	}

	static std::string basenameOf(const std::string &path)
	{
		const size_t slash = path.find_last_of("/\\");
		return slash == std::string::npos ? path : path.substr(slash + 1);
	}

	static void optimiserFiles(const FileName &selected, int count, std::vector<FileName> &files)
	{
		const std::string basename = basenameOf(selected);
		std::smatch match;
		const std::regex expression("^run[0-9]{3}(_it[0-9]+_optimiser\\.star)$");
		if (!std::regex_match(basename, match, expression))
			REPORT_ERROR("Input optimiser must be named runNNN_itXXX_optimiser.star");
		const std::string directory = directoryOf(selected);
		files.resize(count);
		for (int run = 0; run < count; ++run)
		{
			files[run] = directory + "run" + integerToString(run + 1, 3) + match[1].str();
			if (!exists(files[run])) REPORT_ERROR("Missing Class2D replica optimiser: " + files[run]);
		}
	}

	static void readOptimiser(const FileName &filename, FileName &data, FileName &model)
	{
		MetaDataTable table;
		table.read(filename, "optimiser_general");
		if (!table.getValue(EMDL_OPTIMISER_DATA_STARFILE, data) ||
			!table.getValue(EMDL_OPTIMISER_MODEL_STARFILE, model))
			REPORT_ERROR("Cannot obtain data/model STAR paths from " + filename);
		if (!exists(data)) REPORT_ERROR("Missing replica data STAR: " + data);
		if (!exists(model)) REPORT_ERROR("Missing replica model STAR: " + model);
	}

	static int readClassCount(const FileName &model)
	{
		MetaDataTable table;
		table.read(model, "model_general");
		int count = 0, dimension = 0;
		if (!table.getValue(EMDL_MLMODEL_NR_CLASSES, count) ||
			!table.getValue(EMDL_MLMODEL_DIMENSIONALITY, dimension))
			REPORT_ERROR("Cannot read class count/dimensionality from " + model);
		if (dimension != 2) REPORT_ERROR("Class2D consensus only supports 2D reference models");
		return count;
	}

	void run()
	{
		std::vector<FileName> optimiser_files;
		optimiserFiles(input_optimiser, nr_runs, optimiser_files);
		std::vector<FileName> data_files(nr_runs), model_files(nr_runs);
		for (int run = 0; run < nr_runs; ++run)
			readOptimiser(optimiser_files[run], data_files[run], model_files[run]);

		const int source_classes = readClassCount(model_files[0]);
		const int nr_classes = requested_classes == 0 ? source_classes : requested_classes;
		if (source_classes < 2) REPORT_ERROR("Class2D consensus requires at least two classes");
		for (int run = 1; run < nr_runs; ++run)
			if (readClassCount(model_files[run]) != source_classes)
				REPORT_ERROR("All Class2D replicas must have the same number of classes");

		std::vector<std::vector<int> > assignments(nr_runs);
		std::unordered_map<std::string, long int> canonical_index;
		std::vector<double> canonical_pixel_sizes;
		std::vector<int> canonical_box_sizes;
		long int nr_particles = -1;
		for (int run = 0; run < nr_runs; ++run)
		{
			ObservationModel observation;
			MetaDataTable particles;
			ObservationModel::loadSafely(data_files[run], observation, particles, "particles", 0);
			if (!particles.containsLabel(EMDL_IMAGE_NAME) || !particles.containsLabel(EMDL_PARTICLE_CLASS))
				REPORT_ERROR("Replica data STAR must contain rlnImageName and rlnClassNumber: " + data_files[run]);
			if (run == 0)
			{
				nr_particles = particles.numberOfObjects();
				canonical_index.reserve(nr_particles);
				canonical_pixel_sizes = observation.getPixelSizes();
				std::vector<int> ignored;
				observation.getBoxSizes(canonical_box_sizes, ignored);
				assignments[run].resize(nr_particles);
				for (long int particle = 0; particle < nr_particles; ++particle)
				{
					FileName image_name;
					int class_number;
					particles.getValue(EMDL_IMAGE_NAME, image_name, particle);
					particles.getValue(EMDL_PARTICLE_CLASS, class_number, particle);
					if (!canonical_index.insert(std::make_pair((std::string)image_name, particle)).second)
						REPORT_ERROR("Duplicate rlnImageName in " + data_files[run] + ": " + image_name);
					if (class_number < 1 || class_number > source_classes)
						REPORT_ERROR("Invalid rlnClassNumber in " + data_files[run]);
					assignments[run][particle] = class_number - 1;
				}
			}
			else
			{
				if ((long int)particles.numberOfObjects() != nr_particles)
					REPORT_ERROR("Class2D replicas have different particle counts");
				if (observation.getPixelSizes() != canonical_pixel_sizes)
					REPORT_ERROR("Class2D replicas have incompatible optics pixel sizes");
				std::vector<int> box_sizes, ignored;
				observation.getBoxSizes(box_sizes, ignored);
				if (box_sizes != canonical_box_sizes)
					REPORT_ERROR("Class2D replicas have incompatible optics box sizes");
				assignments[run].assign(nr_particles, -1);
				std::vector<char> seen(nr_particles, false);
				for (long int row = 0; row < nr_particles; ++row)
				{
					FileName image_name;
					int class_number;
					particles.getValue(EMDL_IMAGE_NAME, image_name, row);
					particles.getValue(EMDL_PARTICLE_CLASS, class_number, row);
					std::unordered_map<std::string, long int>::const_iterator found = canonical_index.find(image_name);
					if (found == canonical_index.end()) REPORT_ERROR("Particle sets differ between Class2D replicas");
					if (seen[found->second]) REPORT_ERROR("Duplicate rlnImageName in " + data_files[run] + ": " + image_name);
					if (class_number < 1 || class_number > source_classes) REPORT_ERROR("Invalid rlnClassNumber in " + data_files[run]);
					seen[found->second] = true;
					assignments[run][found->second] = class_number - 1;
				}
			}
		}

		if (verb > 0)
		{
			for (int run = 0; run < nr_runs; ++run)
			{
				std::vector<long int> counts(source_classes, 0);
				for (size_t particle = 0; particle < assignments[run].size(); ++particle)
					counts[assignments[run][particle]]++;
				std::cout << "Replica run" << integerToString(run + 1, 3) << ": "
				          << std::count(counts.begin(), counts.end(), 0)
				          << " of " << source_classes << " source classes have no assigned particles." << std::endl;
			}
			std::cout << "Fitting categorical consensus for " << nr_particles << " particles, "
			          << nr_runs << " replicas, " << source_classes << " source classes and " << nr_classes << " consensus classes..." << std::endl;
		}
		// Stable identities, rather than STAR row order, break unequal-count ties.
		if (nr_classes != source_classes)
		{
			std::vector<std::pair<std::string, long int> > identities(canonical_index.begin(), canonical_index.end());
			std::sort(identities.begin(), identities.end());
			for (int run = 0; run < nr_runs; ++run)
			{
				std::vector<int> ordered(nr_particles);
				for (long int i = 0; i < nr_particles; ++i) ordered[i] = assignments[run][identities[i].second];
				assignments[run].swap(ordered);
			}
			for (long int i = 0; i < nr_particles; ++i) canonical_index[identities[i].first] = i;
		}
		Class2DConsensusResult result;
		try
		{
			result = Class2DConsensus::fitWithClassCount(assignments, source_classes, nr_classes, 200, 1.e-6, 1.0, nr_threads);
		}
		catch (const std::exception &error)
		{
			REPORT_ERROR(error.what());
		}
		if ((size_t)nr_classes > result.nr_patterns && verb > 0)
			std::cout << "WARNING: requested classes exceed the " << result.nr_patterns
			          << " distinct label patterns; identical patterns cannot be subdivided." << std::endl;
		std::vector<int> class_counts(nr_classes, 0);
		for (size_t particle = 0; particle < result.assignment.size(); ++particle)
			class_counts[result.assignment[particle]]++;
		if (verb > 0)
			std::cout << "Consensus: " << std::count(class_counts.begin(), class_counts.end(), 0)
			          << " of " << nr_classes << " classes have no assigned particles; retaining all class slots for refinement."
			          << std::endl;

		ObservationModel anchor_observation;
		MetaDataTable anchor_particles;
		ObservationModel::loadSafely(data_files[result.anchor_run], anchor_observation, anchor_particles, "particles", 0);
		for (long int row = 0; row < anchor_particles.numberOfObjects(); ++row)
		{
			FileName image_name;
			anchor_particles.getValue(EMDL_IMAGE_NAME, image_name, row);
			const long int particle = canonical_index.find(image_name)->second;
			anchor_particles.setValue(EMDL_PARTICLE_CLASS, result.assignment[particle] + 1, row);
			anchor_particles.setValue(EMDL_PARTICLE_CLASS2D_CONSENSUS_PROBABILITY, result.probability[particle], row);
			anchor_particles.setValue(EMDL_PARTICLE_CLASS2D_CONSENSUS_ENTROPY, result.entropy[particle], row);
			anchor_particles.setValue(EMDL_PARTICLE_CLASS2D_CONSENSUS_AGREEMENT, result.agreement[particle], row);
		}
		resetClass2DConsensusAlignments(anchor_particles);
		anchor_observation.save(anchor_particles, output_root + "_data.star", "particles");

		std::ofstream diagnostics((output_root + "_diagnostics.star").c_str());
		if (!diagnostics) REPORT_ERROR("Cannot write consensus diagnostics STAR");
		MetaDataTable general;
		general.setIsList(true);
		general.setName("consensus_general");
		general.addObject();
		general.setValue(EMDL_MLMODEL_NR_CLASSES, nr_classes);
		general.setValue(EMDL_CLASS2D_CONSENSUS_SOURCE_CLASSES, source_classes);
		general.setValue(EMDL_CLASS2D_CONSENSUS_OCCUPIED_CLASSES, nr_classes - (int)std::count(class_counts.begin(), class_counts.end(), 0));
		general.setValue(EMDL_CLASS2D_CONSENSUS_PATTERNS, (long int)result.nr_patterns);
		general.setValue(EMDL_PARTICLE_NUMBER, (int)nr_particles);
		general.setValue(EMDL_CLASS2D_CONSENSUS_ANCHOR_RUN, result.anchor_run + 1);
		general.setValue(EMDL_CLASS2D_CONSENSUS_LOG_LIKELIHOOD, result.log_likelihood);
		general.setValue(EMDL_CLASS2D_CONSENSUS_ITERATIONS, result.iterations);
		general.write(diagnostics);

		MetaDataTable run_table;
		run_table.setName("consensus_runs");
		for (int run = 0; run < nr_runs; ++run)
		{
			run_table.addObject();
			run_table.setValue(EMDL_CLASS2D_CONSENSUS_RUN_NUMBER, run + 1);
			run_table.setValue(EMDL_CLASS2D_CONSENSUS_MAPPING_MODE,
				std::string(source_classes == nr_classes ? "one_to_one" :
				(source_classes > nr_classes ? "source_to_consensus" : "consensus_to_source")));
			run_table.setValue(EMDL_CLASS2D_CONSENSUS_SOURCE_OPTIMISER, optimiser_files[run]);
			run_table.setValue(EMDL_CLASS2D_CONSENSUS_ADJUSTED_RAND, result.run_adjusted_rand[run]);
			run_table.setValue(EMDL_CLASS2D_CONSENSUS_MAPPED_AGREEMENT, result.run_mapped_agreement[run]);
		}
		run_table.write(diagnostics);

		MetaDataTable class_table;
		class_table.setName("consensus_classes");
		for (int k = 0; k < nr_classes; ++k)
		{
			class_table.addObject();
			class_table.setValue(EMDL_PARTICLE_CLASS, k + 1);
			class_table.setValue(EMDL_PARTICLE_NUMBER, class_counts[k]);
			class_table.setValue(EMDL_MLMODEL_PDF_CLASS, result.class_posterior_mass[k] / nr_particles);
		}
		class_table.write(diagnostics);

		MetaDataTable confusion_table;
		confusion_table.setName("consensus_confusion");
		for (int run = 0; run < nr_runs; ++run)
			for (int consensus_class = 0; consensus_class < nr_classes; ++consensus_class)
				for (int source_class = 0; source_class < source_classes; ++source_class)
				{
					confusion_table.addObject();
					confusion_table.setValue(EMDL_CLASS2D_CONSENSUS_RUN_NUMBER, run + 1);
					confusion_table.setValue(EMDL_PARTICLE_CLASS, consensus_class + 1);
					confusion_table.setValue(EMDL_CLASS2D_CONSENSUS_SOURCE_CLASS, source_class + 1);
					const size_t index = ((size_t)run * nr_classes + consensus_class) * source_classes + source_class;
					confusion_table.setValue(EMDL_CLASS2D_CONSENSUS_CONDITIONAL_PROBABILITY, result.confusion[index]);
				}
		confusion_table.write(diagnostics);
		diagnostics.close();

		if (verb > 0)
		{
			std::cout << "Consensus anchor: run" << integerToString(result.anchor_run + 1, 3) << std::endl;
			std::cout << "Written " << output_root << "_data.star, " << output_root
			          << "_diagnostics.star; refinement initializes fresh images from consensus members" << std::endl;
		}
	}
};

int main(int argc, char **argv)
{
	Class2DConsensusProgram program;
	try
	{
		program.read(argc, argv);
		program.run();
	}
	catch (RelionError error)
	{
		std::cerr << error;
		return RELION_EXIT_FAILURE;
	}
	return RELION_EXIT_SUCCESS;
}
