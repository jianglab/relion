/***************************************************************************
 *
 * Filament context for a finished helical 2D classification: learns how the
 * classes follow each other along the filaments, finds the helical types and
 * their periods, and labels every segment with its type and filament section.
 * The emissions are the assigned classes (rlnClassNumber) with
 * rlnMaxValueProbDistribution as their probability. See src/helix_context.h.
 *
 ***************************************************************************/

#include "src/args.h"
#include "src/metadata_table.h"
#include "src/helix_context.h"
#include <fstream>
#include <iomanip>

class helix_filament_context_parameters
{
public:
	IOParser parser;
	FileName fn_in, fn_out;
	int K, nr_iter, types_max, nr_folds;
	RFLOAT diameter, max_prob, types_tol;
	FileName fn_classes;
	RFLOAT class_min_frac;
	long int class_min_segs;
	HelicalContext ctx;

	void read(int argc, char **argv)
	{
		parser.setCommandLine(argc, argv);
		parser.addSection("Options");
		fn_in = parser.getOption("--i", "Input _data.star file of a helical 2D classification");
		fn_out = parser.getOption("--o", "Output rootname", "filament_context");
		K = textToInteger(parser.getOption("--K", "Number of classes (default: the largest rlnClassNumber)", "-1"));
		diameter = textToFloat(parser.getOption("--diameter", "Particle (mask) diameter in A (default: box size)", "-1"));
		ctx.nr_types = textToInteger(parser.getOption("--types", "Number of helical types (one more if there are junk filaments)", "4"));
		types_max = textToInteger(parser.getOption("--types_max", "Choose the number of types between --types and this by held-out log-likelihood (cross-validation over filaments; <= 0: no)", "-1"));
		types_tol = textToFloat(parser.getOption("--types_tol", "With --types_max: use the fewest types whose held-out log-likelihood per segment is within this of the best", "0.01"));
		nr_folds = textToInteger(parser.getOption("--folds", "Number of cross-validation folds for --types_max", "3"));
		fn_classes = parser.getOption("--classes", "Class averages (default: the _classes.mrcs next to the input _data.star)", "");
		class_min_frac = textToFloat(parser.getOption("--class_min_frac", "Files per type: include a class if at least this fraction of its segments is of the type", "0.1"));
		class_min_segs = textToInteger(parser.getOption("--class_min_segs", "Files per type: ... and at least this many of its segments", "20"));
		ctx.temper = textToFloat(parser.getOption("--temper", "Factor on the log-likelihoods of the emissions", "1"));
		ctx.exclude_angst = textToFloat(parser.getOption("--exclude", "Prior of a segment from segments at least this far away (A, default: diameter)", "-1"));
		ctx.verb = textToInteger(parser.getOption("--verb", "Verbosity", "1"));
		ctx.max_gap_steps = textToFloat(parser.getOption("--max_gap", "A gap of more than this many segment steps breaks a filament", "3"));
		nr_iter = textToInteger(parser.getOption("--iter", "Number of Baum-Welch iterations", "3"));
		max_prob = textToFloat(parser.getOption("--max_prob", "Upper limit of the probability of the assigned class", "0.99"));
		if (parser.checkForErrors())
			REPORT_ERROR("Errors encountered on the command line, exiting...");
	}

	void run()
	{
		MetaDataTable MDopt, MD;
		MDopt.read(fn_in, "optics");
		MD.read(fn_in, "particles");
		if (MD.numberOfObjects() == 0)
			REPORT_ERROR("No particles in " + fn_in);
		std::vector<long int> rows(MD.numberOfObjects());
		std::vector<int> cls(rows.size());
		std::vector<RFLOAT> q(rows.size(), 0.9), psi(rows.size(), 0.), psip(rows.size(), 0.);
		int maxK = 0;
		for (long int i = 0; i < (long int)rows.size(); i++)
		{
			rows[i] = i;
			MD.getValue(EMDL_PARTICLE_CLASS, cls[i], i);
			cls[i]--;
			maxK = std::max(maxK, cls[i] + 1);
			if (MD.containsLabel(EMDL_PARTICLE_PMAX))
				MD.getValue(EMDL_PARTICLE_PMAX, q[i], i);
			if (MD.containsLabel(EMDL_ORIENT_PSI))
				MD.getValue(EMDL_ORIENT_PSI, psi[i], i);
			if (MD.containsLabel(EMDL_ORIENT_PSI_PRIOR))
				MD.getValue(EMDL_ORIENT_PSI_PRIOR, psip[i], i);
			else
				psip[i] = psi[i];
		}
		if (K < maxK) K = maxK;
		if (diameter <= 0.)
		{
			int box;
			RFLOAT angpix;
			MDopt.getValue(EMDL_IMAGE_SIZE, box, 0);
			MDopt.getValue(EMDL_IMAGE_PIXEL_SIZE, angpix, 0);
			diameter = box * angpix;
		}

		ctx.nr_bw_iter = nr_iter;
		ctx.initialise(MD, rows, K, diameter);
		std::vector<RFLOAT> ll(K);
		for (long int i = 0; i < (long int)rows.size(); i++)
		{
			RFLOAT p = std::min(max_prob, std::max(1. / K, q[i]));
			for (int k = 0; k < K; k++)
				ll[k] = log((1. - p) / (K - 1));
			ll[cls[i]] = log(p);
			ctx.storeLogLik(i, ll);
		}
		if (types_max > ctx.nr_types)
			ctx.selectNrTypes(ctx.nr_types, types_max, nr_folds, types_tol);
		ctx.update();
		{
			std::ofstream fa((fn_out + "_q.txt").c_str());
			fa << std::setprecision(8);
			for (int tt = 0; tt < ctx.nr_types; tt++)
			{
				for (int k = 0; k < K; k++)
					fa << ctx.q[tt * K + k] << " ";
				fa << "\n";
			}
		}
		ctx.fitPeriods(psi, psip);

		// how well the filament alone (segments at least --exclude away) predicts the class
		long int n = 0, hit_c = 0;
		for (long int i = 0; i < (long int)rows.size(); i++)
		{
			if (ctx.seg_chain[i] < 0) continue;
			int best = 0;
			for (int k = 0; k < K; k++)
				if (ctx.prior[i * K + k] > ctx.prior[i * K + best]) best = k;
			n++;
			hit_c += (best == cls[i]);
		}
		std::cout << " Log-likelihood " << ctx.loglik_total << "; type switch every " << ctx.step / ctx.rho
		          << " A; context-only prediction of the class (from segments >= " << ctx.exclude_steps * ctx.step
		          << " A away): " << (double)hit_c / n << std::endl;
		std::cout << " Types:" << std::endl;
		for (int t = 0; t < ctx.nr_types; t++)
			std::cout << "  " << t + 1 << ": " << ROUND(ctx.type_mass[t]) << " segments, " << ctx.type_nr_filaments[t]
			          << " filaments, " << ctx.type_nr_sections[t] << " sections (mean " << ctx.type_section_angst[t]
			          << " A), period " << ctx.type_period[t] << " A (score " << ctx.type_period_R[t] << ")" << std::endl;

		ctx.writeLabels(MD, rows, &cls);
		FileName fn_data = fn_out + "_data.star";
		std::ofstream fh(fn_data.c_str());
		MDopt.setName("optics");
		MDopt.write(fh);
		MD.setName("particles");
		MD.write(fh);
		fh.close();
		ctx.writeStar(fn_out + "_filaments.star");
		if (fn_classes == "" && fn_in.contains("_data.star"))
			fn_classes = fn_in.beforeLastOf("_data.star") + "_classes.mrcs";
		HelicalContext::writeTypeFiles(fn_out, MDopt, MD, ctx.nr_types, fn_classes, class_min_frac, class_min_segs, ctx.verb);
		std::cout << " Written " << fn_data << " and " << fn_out + "_filaments.star" << std::endl;
	}
};

int main(int argc, char *argv[])
{
	helix_filament_context_parameters prm;
	try
	{
		prm.read(argc, argv);
		prm.run();
	}
	catch (RelionError XE)
	{
		std::cerr << XE;
		return RELION_EXIT_FAILURE;
	}
	return RELION_EXIT_SUCCESS;
}
