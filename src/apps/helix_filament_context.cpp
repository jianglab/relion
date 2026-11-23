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
#include "src/image.h"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

class helix_filament_context_parameters
{
public:
	IOParser parser;
	FileName fn_in, fn_out, fn_project_dir, fn_mtf;
	int K, nr_iter, types_max, nr_folds, nr_threads, max_types_total, min_type_filaments;
	RFLOAT diameter, outer_diameter, sort_classes_junk, max_prob, types_tol, types_agree;
	FileName fn_classes, fn_model;
	RFLOAT class_min_frac;
	long int class_min_segs;
	bool dont_smooth_flicker, do_hierarchical, dont_smooth_tracks, dont_resolve_polarity;
	HelicalContext ctx;

	void read(int argc, char **argv)
	{
		parser.setCommandLine(argc, argv);
		parser.addSection("Options");
		fn_in = parser.getOption("--i", "Input _data.star file of a helical 2D classification");
		fn_out = parser.getOption("--o", "Output rootname", "filament_context");
		fn_project_dir = parser.getOption("--project_dir", "Optional RELION project directory to resolve relative MTF paths", "");
		fn_mtf = parser.getOption("--mtf", "Optional override for _rlnMtfFileName in output STAR files", "");
		K = textToInteger(parser.getOption("--K", "Number of classes (default: the largest rlnClassNumber)", "-1"));
		diameter = textToFloat(parser.getOption("--diameter", "Particle (mask) diameter in A (default: box size)", "-1"));
		outer_diameter = textToFloat(parser.getOption("--outer_diameter", "Helical outer tube diameter in A (0: none)", "0"));
		sort_classes_junk = textToFloat(parser.getOption("--sort_classes_junk", "Mark classes with resolution worse than this factor x median as junk", "1.5"));
		ctx.nr_types = textToInteger(parser.getOption("--types", "Number of helical types (one more if there are junk filaments)", "4"));
		types_max = textToInteger(parser.getOption("--types_max", "Choose the number of types between --types and this by held-out log-likelihood (cross-validation over filaments; <= 0: no)", "-1"));
		types_tol = textToFloat(parser.getOption("--types_tol", "With --types_max: use the fewest types whose held-out log-likelihood per segment is within this of the best", "0.01"));
		types_agree = textToFloat(parser.getOption("--types_agree", "With --types_max: use the most types whose segment types, learnt from separate folds of the filaments, agree to at least this (adjusted Rand index); 0: by log-likelihood (--types_tol)", "0.75"));
		nr_folds = textToInteger(parser.getOption("--folds", "Number of cross-validation folds for --types_max", "3"));
		do_hierarchical = parser.checkOption("--hierarchical", "Enable recursive hierarchical tree splitting (selectNrTypes + splitTypesByTwistRate per node)");
		max_types_total = textToInteger(parser.getOption("--max_types_total", "With --hierarchical: maximum total leaf types across the tree", "24"));
		min_type_filaments = textToInteger(parser.getOption("--min_type_filaments", "With --hierarchical: minimum filaments in a node to attempt further splitting", "20"));
		fn_classes = parser.getOption("--classes", "Class averages (default: the _classes.mrcs next to the input _data.star)", "");
		fn_model = parser.getOption("--model", "Model STAR file (default: the _model.star next to the input _data.star)", "");
		class_min_frac = textToFloat(parser.getOption("--class_min_frac", "Files per type: include a class if at least this fraction of its segments is of the type", "0.1"));
		class_min_segs = textToInteger(parser.getOption("--class_min_segs", "Files per type: ... and at least this many of its segments", "20"));
		ctx.temper = textToFloat(parser.getOption("--temper", "Factor on the log-likelihoods of the emissions (<= 0: automatic from step and diameter)", "-1"));
		ctx.exclude_angst = textToFloat(parser.getOption("--exclude", "Prior of a segment from segments at least this far away (A, default: diameter)", "-1"));
		ctx.verb = textToInteger(parser.getOption("--verb", "Verbosity", "1"));
		ctx.max_gap_steps = textToFloat(parser.getOption("--max_gap", "A gap of more than this many segment steps breaks a filament", "3"));
		nr_iter = textToInteger(parser.getOption("--iter", "Number of Baum-Welch iterations", "3"));
		max_prob = textToFloat(parser.getOption("--max_prob", "Upper limit of the probability of the assigned class", "0.99"));
		dont_smooth_flicker = parser.checkOption("--dont_smooth_flicker", "Do not run the final mismatch/flicker class smoothing on rlnClassNumber");
		dont_smooth_tracks = parser.checkOption("--dont_smooth_tracks", "Do not smooth centerlines into rlnCoordinateX/Y");
		dont_resolve_polarity = parser.checkOption("--dont_resolve_polarity", "Do not overwrite rlnAnglePsiPrior with coherent filament polarity");
		nr_threads = textToInteger(parser.getOption("--j", "Number of OpenMP threads", "1"));
		if (parser.checkForErrors())
			REPORT_ERROR("Errors encountered on the command line, exiting...");
	}

	void run()
	{
#ifdef _OPENMP
		if (nr_threads > 0)
			omp_set_num_threads(nr_threads);
#endif
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

		int box = 0;
		RFLOAT angpix = 1.0;
		if (MDopt.numberOfObjects() > 0)
		{
			MDopt.getValue(EMDL_IMAGE_SIZE, box, 0);
			MDopt.getValue(EMDL_IMAGE_PIXEL_SIZE, angpix, 0);
		}
		if (diameter <= 0.)
		{
			diameter = box * angpix;
			if (outer_diameter > 0.)
				diameter = std::min(diameter, outer_diameter);
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

		if (fn_classes == "" && fn_in.contains("_data.star"))
			fn_classes = fn_in.beforeLastOf("_data.star") + "_classes.mrcs";
		if (fn_model == "" && fn_in.contains("_data.star"))
			fn_model = fn_in.beforeLastOf("_data.star") + "_model.star";

		std::vector<long int> cls_cnt(K, 0);
		for (size_t i = 0; i < cls.size(); i++)
			if (cls[i] >= 0 && cls[i] < K) cls_cnt[cls[i]]++;

		std::vector<RFLOAT> res(K, 0.), pdf(K, 0.);
		bool have_model = false;
		if (fn_model != "" && exists(fn_model))
		{
			MetaDataTable MDmod;
			MDmod.read(fn_model, "model_classes");
			if ((int)MDmod.numberOfObjects() >= K)
			{
				have_model = true;
				for (int k = 0; k < K; k++)
				{
					if (!MDmod.getValue(EMDL_MLMODEL_ESTIM_RESOL_REF, res[k], k)) res[k] = 0.;
					if (!MDmod.getValue(EMDL_MLMODEL_PDF_CLASS, pdf[k], k))
						pdf[k] = (RFLOAT)cls_cnt[k] / std::max((long int)1, (long int)rows.size());
				}
			}
		}
		if (!have_model)
		{
			res.clear();
			for (int k = 0; k < K; k++)
				pdf[k] = (RFLOAT)cls_cnt[k] / std::max((long int)1, (long int)rows.size());
		}

		std::vector<MultidimArray<RFLOAT> > refs;
		if (fn_classes != "" && exists(fn_classes))
		{
			Image<RFLOAT> Istk;
			Istk.read(fn_classes);
			if (NSIZE(Istk()) >= K)
			{
				refs.resize(K);
				for (int k = 0; k < K; k++)
				{
					Istk().getImage(k, refs[k]);
					refs[k].setXmippOrigin();
				}
			}
		}

		std::vector<RFLOAT> cont;
		std::string cont_note = "no class stack";
		if (!refs.empty())
		{
			std::vector<RFLOAT> track(rows.size(), 0.);
			std::vector<long int> chain(rows.size(), -1);
			std::vector<char> flipped(rows.size(), 0);
			for (long int i = 0; i < (long int)rows.size(); i++)
			{
				chain[i] = ctx.seg_chain[i];
				track[i] = ctx.seg_pos[i] * ctx.step;
				flipped[i] = (fabs(remainder(psi[i] - psip[i], 360.)) > 90.);
			}
			cont = helixtypes::classContinuity(refs, angpix, chain, track, cls, flipped, cont_note, outer_diameter);
		}

		const RFLOAT box_angst = (box > 0) ? (box * angpix) : diameter;
		ctx.junk_class = helixtypes::junkClasses(res, std::vector<RFLOAT>(), box_angst,
		                                         sort_classes_junk > 0. ? sort_classes_junk : 1.5, cont, pdf);
		const long int min_cls_segs = ((long int)rows.size() >= 2000) ? 5 : 1;
		for (int k = 0; k < K; k++)
			if (cls_cnt[k] < min_cls_segs) ctx.junk_class[k] = 1;
		if (ctx.verb > 0)
		{
			int nj = 0, nc = 0;
			for (int k = 0; k < K; k++)
			{
				nj += (ctx.junk_class[k] != 0);
				if (k < (int)cont.size() && std::isfinite(cont[k]) && cont[k] < 0.6) nc++;
			}
			std::cout << " Filament context: " << nj << " junk classes, " << nc << " below continuity 0.6 ("
			          << cont_note << ")" << std::endl;
		}
		if (!refs.empty())
			ctx.computeClassEquivalence(refs, angpix, outer_diameter);

		auto runFinalSmooth = [&]()
		{
			const int T = ctx.nr_types;
			const std::vector<int> orig_cls = cls;
			auto is_junk_k = [&](int k) { return k < 0 || k >= K || cls_cnt[k] < min_cls_segs || (!ctx.junk_class.empty() && ctx.junk_class[k]); };
			auto is_allowed_tk = [&](int k, int t) {
				return !is_junk_k(k) && ((int)ctx.class_allowed.size() != K * T || ctx.class_allowed[(size_t)k * T + t]);
			};
			std::vector<int> fil_best_t(rows.size(), 0);
			for (long int i = 0; i < (long int)rows.size(); i++)
			{
				int bt = 0;
				if ((long int)ctx.tpost.size() == (long int)rows.size() * (long int)T)
					for (int t = 1; t < T; t++)
						if (ctx.tpost[(size_t)i * T + t] > ctx.tpost[(size_t)i * T + bt]) bt = t;
				fil_best_t[i] = bt;
			}
			long int nr_smoothed = 0;
			for (size_t c = 0; c < ctx.chains.size(); c++)
			{
				const std::vector<long int> &grid = ctx.chains[c];
				const size_t G = grid.size();
				for (size_t p = 0; p < G; p++)
				{
					const long int i = grid[p];
					if (i < 0 || is_junk_k(orig_cls[i])) continue;
					const int bt = fil_best_t[i];
					int new_k = orig_cls[i];
					if (!is_allowed_tk(orig_cls[i], bt))
					{
						int best_k = -1;
						float best_p = 0.f;
						for (int k = 0; k < K; k++)
						{
							if (!is_allowed_tk(k, bt)) continue;
							const float pv = ctx.post[(size_t)i * K + k];
							if (pv > best_p) { best_p = pv; best_k = k; }
						}
						if (best_k >= 0) new_k = best_k;
					}
					else if (p > 0 && p + 1 < G && grid[p - 1] >= 0 && grid[p + 1] >= 0 &&
					         orig_cls[grid[p - 1]] == orig_cls[grid[p + 1]] &&
					         orig_cls[grid[p - 1]] != orig_cls[i] && is_allowed_tk(orig_cls[grid[p - 1]], bt))
					{
						new_k = orig_cls[grid[p - 1]];
					}
					else if (p > 0 && p + 2 < G && grid[p - 1] >= 0 && grid[p + 1] >= 0 && grid[p + 2] >= 0 &&
					         orig_cls[grid[p + 1]] == orig_cls[i] &&
					         orig_cls[grid[p - 1]] == orig_cls[grid[p + 2]] &&
					         orig_cls[grid[p - 1]] != orig_cls[i] && is_allowed_tk(orig_cls[grid[p - 1]], bt))
					{
						new_k = orig_cls[grid[p - 1]];
					}
					else if (p >= 2 && p + 1 < G && grid[p - 2] >= 0 && grid[p - 1] >= 0 && grid[p + 1] >= 0 &&
					         orig_cls[grid[p - 1]] == orig_cls[i] &&
					         orig_cls[grid[p - 2]] == orig_cls[grid[p + 1]] &&
					         orig_cls[grid[p - 2]] != orig_cls[i] && is_allowed_tk(orig_cls[grid[p - 2]], bt))
					{
						new_k = orig_cls[grid[p - 2]];
					}
					if (new_k != cls[i])
					{
						cls[i] = new_k;
						MD.setValue(EMDL_PARTICLE_CLASS, new_k + 1, i);
						nr_smoothed++;
					}
				}
			}
			if (ctx.verb > 0 && nr_smoothed > 0)
				std::cout << " Filament context: surgical final filter updated " << nr_smoothed << " segments" << std::endl;
		};

		if (do_hierarchical)
		{
			const int tmax_node = (types_max > 1) ? types_max : std::min(12, std::max(2, max_types_total));
			ctx.runHierarchicalTyping(psi, psip, ctx.nr_types, tmax_node, max_types_total, min_type_filaments,
			                          nr_folds, types_tol, types_agree, true);
			if (!dont_smooth_flicker && (long int)ctx.post.size() == (long int)rows.size() * (long int)K)
				runFinalSmooth();
			ctx.fitPeriods(psi, psip);
		}
		else
		{
			if (types_max > ctx.nr_types)
				ctx.selectNrTypes(ctx.nr_types, types_max, nr_folds, types_tol, types_agree);
			ctx.update();
			if (!dont_smooth_flicker && (long int)ctx.post.size() == (long int)rows.size() * (long int)K)
				runFinalSmooth();

			ctx.fitPeriods(psi, psip);
			if (types_max > ctx.nr_types)
			{
				const int prev_nt = ctx.nr_types;
				if (ctx.splitTypesByTwistRate(types_max) > prev_nt)
				{
					if (!dont_smooth_flicker && (long int)ctx.post.size() == (long int)rows.size() * (long int)K)
						runFinalSmooth();
					ctx.fitPeriods(psi, psip);
				}
			}
		}
		ctx.applyPhaseRingPriors(psi, psip);

		// Post-hoc centerline smoothing (rlnCoordinateX/Y) and coherent filament polarity (rlnAnglePsiPrior)
		if (!dont_smooth_tracks || !dont_resolve_polarity)
			ctx.smoothAndRecenterCoordinates(MD, rows, &cls, angpix, 0.45 * diameter, &MDopt,
			                                 !dont_resolve_polarity, false, !dont_smooth_tracks);

		// Resolve or override _rlnMtfFileName in output optics table if requested
		if (MDopt.containsLabel(EMDL_IMAGE_MTF_FILENAME))
		{
			for (long int oi = 0; oi < (long int)MDopt.numberOfObjects(); oi++)
			{
				if (!fn_mtf.empty())
				{
					MDopt.setValue(EMDL_IMAGE_MTF_FILENAME, fn_mtf, oi);
				}
				else if (!fn_project_dir.empty())
				{
					FileName mtf;
					if (MDopt.getValue(EMDL_IMAGE_MTF_FILENAME, mtf, oi) && !mtf.empty() && mtf[0] != '/' && !exists(mtf))
					{
						std::string pdir = fn_project_dir;
						while (pdir.size() > 1 && pdir.back() == '/') pdir.pop_back();
						const std::string cand = pdir + "/" + std::string(mtf);
						if (exists(cand))
						{
							char buf[PATH_MAX];
							if (realpath(cand.c_str(), buf) != NULL)
								MDopt.setValue(EMDL_IMAGE_MTF_FILENAME, FileName(buf), oi);
							else
								MDopt.setValue(EMDL_IMAGE_MTF_FILENAME, FileName(cand), oi);
						}
					}
				}
			}
		}
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

		// how well the filament alone (segments at least --exclude away) predicts the class
		long int n = 0, hit_c = 0;
		for (long int i = 0; i < (long int)rows.size(); i++)
		{
			if (ctx.seg_chain[i] < 0 || cls[i] < 0 || cls[i] >= K) continue;
			if (!ctx.junk_class.empty() && ctx.junk_class[cls[i]]) continue;
			int best = -1;
			for (int k = 0; k < K; k++)
			{
				if (!ctx.junk_class.empty() && ctx.junk_class[k]) continue;
				if (best < 0 || ctx.prior[i * K + k] > ctx.prior[i * K + best]) best = k;
			}
			if (best < 0) continue;
			n++;
			hit_c += (best == cls[i]);
		}
		std::cout << " Log-likelihood " << ctx.loglik_total << "; type switch every " << ctx.step / ctx.rho
		          << " A; context-only prediction of the class (from segments >= " << ctx.exclude_steps * ctx.step
		          << " A away): " << ((n > 0) ? (double)hit_c / n : 0.) << std::endl;
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
