/***************************************************************************
 *
 * Coarse-to-unbinned two-stage fused helical sub-classification:
 *   Stage 1 (binned global classification + HelicalContext HMM) separates
 *   polymorphs and estimates segment centerlines & polarities.
 *   Stage 2 (this tool) re-centers each polymorph onto its Stage-1 filament
 *   centerline and resolved polarity, emits zero-copy re-centered virtual
 *   particle stacks (.vstack) pointing at the original micrographs, and runs
 *   focused unbinned or 2x-binned 2D sub-classification per polymorph.
 *
 ***************************************************************************/

#include "src/args.h"
#include "src/ctf.h"
#include "src/funcs.h"
#include "src/helix_context.h"
#include "src/image.h"
#include "src/jaz/single_particle/obs_model.h"
#include "src/metadata_table.h"
#include "src/pipeline_control.h"
#include "src/virtual_particles.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits.h>
#include <map>
#include <sstream>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static bool isEnvFlagOff(const char *name)
{
	const char *env = getenv(name);
	if (env == NULL || env[0] == '\0')
		return false;
	std::string v(env);
	for (size_t i = 0; i < v.size(); i++) v[i] = tolower(v[i]);
	return (v == "0" || v == "no" || v == "off" || v == "false");
}

namespace {

std::string trimTrailingSlashes(std::string s)
{
	while (s.size() > 1 && s.back() == '/')
		s.pop_back();
	return s;
}

std::string sanitizeStem(const std::string &path)
{
	std::string base = FileName(path).beforeLastOf(".");
	const size_t slash = base.find_last_of('/');
	if (slash != std::string::npos)
		base = base.substr(slash + 1);
	for (char &c : base)
	{
		if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
		      (c >= '0' && c <= '9') || c == '_' || c == '-'))
			c = '_';
	}
	return base.empty() ? std::string("micrograph") : base;
}

std::string resolveExistingPath(const std::string &raw, const std::string &project_dir, const std::string &star_dir)
{
	auto toReal = [](const std::string &p) -> std::string {
		char buf[PATH_MAX];
		if (realpath(p.c_str(), buf) != NULL)
			return std::string(buf);
		return p;
	};

	if (exists(raw))
		return toReal(raw);
	if (!raw.empty() && raw[0] != '/')
	{
		if (!project_dir.empty())
		{
			const std::string cand = trimTrailingSlashes(project_dir) + "/" + raw;
			if (exists(cand))
				return toReal(cand);
		}
		if (!star_dir.empty())
		{
			const std::string cand = trimTrailingSlashes(star_dir) + "/" + raw;
			if (exists(cand))
				return toReal(cand);
		}
	}
	REPORT_ERROR("helix_subclassify ERROR: cannot find micrograph file '" + raw +
	             (project_dir.empty() ? "' (pass --project_dir if paths are relative to the RELION project)."
	                                  : "' under CWD or --project_dir '" + project_dir + "'."));
	return raw;
}

std::string findCompanionBinary(const char *argv0, const std::string &name)
{
	char exe_buf[PATH_MAX];
	const ssize_t len = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
	if (len > 0)
	{
		exe_buf[len] = '\0';
		std::string exe(exe_buf);
		const size_t slash = exe.find_last_of('/');
		if (slash != std::string::npos)
		{
			const std::string cand = exe.substr(0, slash + 1) + name;
			if (access(cand.c_str(), X_OK) == 0)
				return cand;
		}
	}
	if (argv0 != NULL)
	{
		std::string a0(argv0);
		const size_t slash = a0.find_last_of('/');
		if (slash != std::string::npos)
		{
			const std::string cand = a0.substr(0, slash + 1) + name;
			if (access(cand.c_str(), X_OK) == 0)
				return cand;
		}
	}
	const char *env_bin = getenv("RELION_BIN_DIR");
	if (env_bin != NULL && env_bin[0] != '\0')
	{
		const std::string cand = trimTrailingSlashes(env_bin) + "/" + name;
		if (access(cand.c_str(), X_OK) == 0)
			return cand;
	}
	return name;
}

std::string shellQuote(const std::string &s)
{
	std::string out = "'";
	for (char c : s)
	{
		if (c == '\'')
			out += "'\\''";
		else
			out += c;
	}
	out += "'";
	return out;
}

void smoothAndRecenterFilaments(MetaDataTable &MD, const std::vector<double> &row_mic_angpix, double max_shift_A)
{
	const long int N = MD.numberOfObjects();
	if (N <= 0 || (long int)row_mic_angpix.size() != N)
		return;

	const bool has_tube = MD.containsLabel(EMDL_PARTICLE_HELICAL_TUBE_ID);
	const bool has_track = MD.containsLabel(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM);
	const bool has_psi = MD.containsLabel(EMDL_ORIENT_PSI);
	const bool has_psip = MD.containsLabel(EMDL_ORIENT_PSI_PRIOR);
	const bool has_ox = MD.containsLabel(EMDL_ORIENT_ORIGIN_X_ANGSTROM);
	const bool has_oy = MD.containsLabel(EMDL_ORIENT_ORIGIN_Y_ANGSTROM);

	if (!has_psip && !MD.containsLabel(EMDL_ORIENT_PSI_PRIOR))
		MD.addLabel(EMDL_ORIENT_PSI_PRIOR);
	if (!has_psi)
		MD.addLabel(EMDL_ORIENT_PSI);
	if (!has_ox)
		MD.addLabel(EMDL_ORIENT_ORIGIN_X_ANGSTROM);
	if (!has_oy)
		MD.addLabel(EMDL_ORIENT_ORIGIN_Y_ANGSTROM);

	std::map<std::string, std::vector<long int> > filaments;
	std::vector<double> cx(N, 0.), cy(N, 0.), ox(N, 0.), oy(N, 0.), psi(N, 0.), psip(N, 0.), track(N, 0.);
	for (long int i = 0; i < N; i++)
	{
		RFLOAT v = 0.;
		if (MD.getValue(EMDL_IMAGE_COORD_X, v, i)) cx[i] = v;
		if (MD.getValue(EMDL_IMAGE_COORD_Y, v, i)) cy[i] = v;
		if (has_ox && MD.getValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, v, i)) ox[i] = v;
		if (has_oy && MD.getValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, v, i)) oy[i] = v;
		if (has_psip && MD.getValue(EMDL_ORIENT_PSI_PRIOR, v, i)) psip[i] = v;
		if (has_psi && MD.getValue(EMDL_ORIENT_PSI, v, i)) psi[i] = v;
		else psi[i] = psip[i];
		if (!has_psip) psip[i] = psi[i];
		if (has_track && MD.getValue(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM, v, i))
			track[i] = v;
		else
			track[i] = (double)i;

		FileName mic = "";
		MD.getValue(EMDL_MICROGRAPH_NAME, mic, i);
		int tid = 1;
		if (has_tube) MD.getValue(EMDL_PARTICLE_HELICAL_TUBE_ID, tid, i);
		std::string key = std::string(mic) + "\n" + integerToString(tid);
		filaments[key].push_back(i);
	}

	for (auto &kv : filaments)
	{
		std::vector<long int> &m = kv.second;
		if (has_track)
		{
			std::sort(m.begin(), m.end(), [&](long int a, long int b) {
				if (track[a] != track[b]) return track[a] < track[b];
				return a < b;
			});
		}
		const int L = (int)m.size();

		// 1. Resolve coherent polarity along the filament from (psi - psi_prior)
		int n_flip = 0;
		for (int j = 0; j < L; j++)
		{
			const long int idx = m[j];
			if (std::fabs(std::remainder(psi[idx] - psip[idx], 360.0)) > 90.0)
				n_flip++;
		}
		const double flip_shift = (n_flip * 2 > L) ? 180.0 : 0.0;
		std::vector<double> aligned_psi(L);
		for (int j = 0; j < L; j++)
			aligned_psi[j] = std::remainder(psip[m[j]] + flip_shift, 360.0);

		// 2. Smooth psi_prior circularly along the filament ([0.25, 0.5, 0.25])
		std::vector<double> smooth_psi(L);
		for (int j = 0; j < L; j++)
		{
			double sx = 0., sy = 0.;
			const int j0 = std::max(0, j - 1), j1 = std::min(L - 1, j + 1);
			for (int q = j0; q <= j1; q++)
			{
				const double w = (q == j) ? 0.5 : 0.25;
				const double rad = DEG2RAD(aligned_psi[q]);
				sx += w * std::cos(rad);
				sy += w * std::sin(rad);
			}
			smooth_psi[j] = (sx * sx + sy * sy > 1e-12) ? RAD2DEG(std::atan2(sy, sx)) : aligned_psi[j];
			MD.setValue(EMDL_ORIENT_PSI_PRIOR, (RFLOAT)smooth_psi[j], m[j]);
			MD.setValue(EMDL_ORIENT_PSI, (RFLOAT)smooth_psi[j], m[j]);
		}

		// 3. Smooth transverse Stage-1 centerline offsets and re-center integer micrograph coordinates
		double tx = 1., ty = 0., nx_v = 0., ny_v = 1.;
		bool have_chord = false;
		if (L >= 2)
		{
			const int qcnt = std::max(1, L / 4);
			double x0 = 0., y0 = 0., x1 = 0., y1 = 0.;
			for (int j = 0; j < qcnt; j++) { x0 += cx[m[j]]; y0 += cy[m[j]]; }
			for (int j = L - qcnt; j < L; j++) { x1 += cx[m[j]]; y1 += cy[m[j]]; }
			double dx = x1 / qcnt - x0 / qcnt, dy = y1 / qcnt - y0 / qcnt;
			double len = std::sqrt(dx * dx + dy * dy);
			if (len < 1e-3)
			{
				dx = cx[m[L - 1]] - cx[m[0]];
				dy = cy[m[L - 1]] - cy[m[0]];
				len = std::sqrt(dx * dx + dy * dy);
			}
			if (len >= 1e-3)
			{
				tx = dx / len;
				ty = dy / len;
				nx_v = -ty;
				ny_v = tx;
				have_chord = true;
			}
		}

		std::vector<double> ox_eff(L), oy_eff(L);
		if (have_chord && L >= 3)
		{
			std::vector<double> ot(L), on(L), on_s(L);
			for (int j = 0; j < L; j++)
			{
				const long int idx = m[j];
				ot[j] = ox[idx] * tx + oy[idx] * ty;
				on[j] = ox[idx] * nx_v + oy[idx] * ny_v;
			}
			if (L >= 4)
			{
				double step_A = 14.25;
				if (has_track)
				{
					std::vector<double> diffs;
					for (int j = 1; j < L; j++)
						if (track[m[j]] > track[m[j - 1]])
							diffs.push_back(track[m[j]] - track[m[j - 1]]);
					if (!diffs.empty())
					{
						std::nth_element(diffs.begin(), diffs.begin() + diffs.size() / 2, diffs.end());
						step_A = diffs[diffs.size() / 2];
					}
				}
				const std::vector<double> init_w(L, 1.0), y_zero(L, 0.0);
				const double fil_angpix = (row_mic_angpix[m[0]] > 0.) ? row_mic_angpix[m[0]] : 1.0;
				const double huber_delta = std::max(4.0, 2.0 * fil_angpix);
				HelicalContext::smoothCenterline1D(on, init_w, step_A, huber_delta, 0.0, y_zero, on_s);
			}
			else
			{
				for (int j = 0; j < L; j++)
				{
					double s = 0., wsum = 0.;
					for (int q = std::max(0, j - 1); q <= std::min(L - 1, j + 1); q++)
					{
						const double w = (q == j) ? 0.5 : 0.25;
						s += w * on[q];
						wsum += w;
					}
					on_s[j] = s / std::max(1e-12, wsum);
				}
			}
			for (int j = 0; j < L; j++)
			{
				ox_eff[j] = ot[j] * tx + on_s[j] * nx_v;
				oy_eff[j] = ot[j] * ty + on_s[j] * ny_v;
			}
		}
		else
		{
			for (int j = 0; j < L; j++)
			{
				ox_eff[j] = ox[m[j]];
				oy_eff[j] = oy[m[j]];
			}
		}

		for (int j = 0; j < L; j++)
		{
			const long int idx = m[j];
			const double px_idx = (row_mic_angpix[idx] > 0.) ? row_mic_angpix[idx] : 1.0;
			const double clamp_px = (max_shift_A > 0.) ? (max_shift_A / px_idx) : 1e9;
			double xoff_mic = ox_eff[j] / px_idx;
			double yoff_mic = oy_eff[j] / px_idx;
			if (clamp_px > 0.)
			{
				const double mag = std::sqrt(xoff_mic * xoff_mic + yoff_mic * yoff_mic);
				if (mag > clamp_px)
				{
					xoff_mic *= clamp_px / mag;
					yoff_mic *= clamp_px / mag;
				}
			}
			const long dx = ROUND(xoff_mic);
			const long dy = ROUND(yoff_mic);
			const long cx_round = (long)std::floor(cx[idx] + 0.5);
			const long cy_round = (long)std::floor(cy[idx] + 0.5);
			const long new_cx = cx_round - dx;
			const long new_cy = cy_round - dy;
			const double res_ox = (xoff_mic - (double)dx) * px_idx;
			const double res_oy = (yoff_mic - (double)dy) * px_idx;
			MD.setValue(EMDL_IMAGE_COORD_X, (RFLOAT)new_cx, idx);
			MD.setValue(EMDL_IMAGE_COORD_Y, (RFLOAT)new_cy, idx);
			MD.setValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, (RFLOAT)res_ox, idx);
			MD.setValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, (RFLOAT)res_oy, idx);
		}
	}
}

} // namespace

class helix_subclassify_parameters
{
public:
	IOParser parser;
	FileName fn_in, fn_out, fn_project_dir, fn_model, fn_classes;
	int types_min, types_max, max_types_total, min_type_filaments, nr_folds, K, nr_iter, nr_mpi, nr_threads, nr_pool, random_seed;
	int extract_size, scale_size;
	long int min_segs;
	RFLOAT angpix_override, particle_diameter, outer_diameter, helical_rise;
	RFLOAT psi_step, sigma_psi, offset_range, offset_step, tau2_fudge, highres_limit, helical_context_weight;
	RFLOAT auto_mask_start_res, auto_mask_soft, auto_mask_mass;
	RFLOAT sort_classes_junk, types_tol, types_agree;
	bool do_retype, do_hierarchical, do_dont_hierarchical, do_premult, do_auto_premult, dont_premult, ctf_intact_first_peak;
	bool do_invert_contrast, do_float16, do_auto_mask, do_dont_auto_mask;
	bool do_keep_all_iter, do_dont_keep_all_iter, do_bimodal_psi, do_prepare_only;
	std::string gpu_ids;
	int verb;

	void read(int argc, char **argv)
	{
		parser.setCommandLine(argc, argv);
		parser.addSection("I/O and Stage-1 typing options");
		fn_in = parser.getOption("--i", "Input Stage-1 _data.star file");
		fn_out = parser.getOption("--o", "Output directory / root name", "helix_subclassify");
		fn_project_dir = parser.getOption("--project_dir", "RELION project directory for resolving relative micrograph paths", "");
		fn_model = parser.getOption("--model", "Stage-1 _model.star (default: inferred from --i)", "");
		fn_classes = parser.getOption("--classes", "Stage-1 _classes.mrcs (default: inferred from --i)", "");
		types_min = textToInteger(parser.getOption("--types", "Minimum number of helical types when running typing", "2"));
		types_max = textToInteger(parser.getOption("--types_max", "Maximum number of helical types for automatic HMM + twist selection (<= --types: fixed)", "10"));
		types_tol = textToFloat(parser.getOption("--types_tol", "Cross-validation tolerance for --types_max", "0.01"));
		types_agree = textToFloat(parser.getOption("--types_agree", "Minimum fold ARI agreement for --types_max", "0.75"));
		nr_folds = textToInteger(parser.getOption("--folds", "Cross-validation folds for --types_max", "3"));
		sort_classes_junk = textToFloat(parser.getOption("--sort_classes_junk", "Mark classes with resolution worse than this factor x median as junk", "1.5"));
		do_retype = parser.checkOption("--retype", "Re-run HelicalContext HMM typing even if rlnHelicalType is already present");
		const bool opt_hier = parser.checkOption("--hierarchical", "Enable recursive hierarchical tree splitting when running Stage-1 typing (default on; disable with --dont_hierarchical or RELION_HELICAL_HIERARCHICAL=0)");
		do_dont_hierarchical = parser.checkOption("--dont_hierarchical", "Disable recursive hierarchical tree splitting in Stage-1 typing");
		if (opt_hier && do_dont_hierarchical)
			REPORT_ERROR("Specify at most one of --hierarchical and --dont_hierarchical.");
		do_hierarchical = opt_hier ? true : (do_dont_hierarchical ? false : !isEnvFlagOff("RELION_HELICAL_HIERARCHICAL"));
		max_types_total = textToInteger(parser.getOption("--max_types_total", "With --hierarchical: maximum total leaf types across the tree", "24"));
		min_type_filaments = textToInteger(parser.getOption("--min_type_filaments", "With --hierarchical: minimum filaments in a node to attempt further splitting", "20"));
		min_segs = textToInteger(parser.getOption("--min_segs", "Minimum non-junk segments in a type to run Stage 2", "20"));

		parser.addSection("Zero-copy Stage-2 re-extraction options");
		extract_size = textToInteger(parser.getOption("--extract_size", "Extraction box size in original micrograph pixels (-1: infer from Stage-1 FOV)", "-1"));
		scale_size = textToInteger(parser.getOption("--scale", "Output box size in pixels (-1: unbinned --extract_size)", "-1"));
		angpix_override = textToFloat(parser.getOption("--angpix", "Original micrograph pixel size in A (-1: read from optics/header)", "-1"));
		do_premult = parser.checkOption("--premultiply_ctf", "Force CTF premultiplication in virtual stacks (default: auto if CTF present)");
		do_auto_premult = parser.checkOption("--auto_premultiply_ctf", "Automatically premultiply virtual stacks by CTF when CTF metadata are present (default)");
		dont_premult = parser.checkOption("--dont_premultiply_ctf", "Do not premultiply virtual stacks by CTF");
		ctf_intact_first_peak = parser.checkOption("--ctf_intact_first_peak", "Ignore CTF until first peak during CTF premultiplication and Stage 2");
		do_invert_contrast = !parser.checkOption("--dont_invert_contrast", "Do not invert contrast in virtual extraction");
		do_float16 = !parser.checkOption("--dont_float16", "Store/round virtual particle stacks as float32 instead of float16");

		parser.addSection("Stage-2 classification options");
		K = textToInteger(parser.getOption("--K", "Number of Stage-2 sub-classes per polymorph type", "12"));
		nr_iter = textToInteger(parser.getOption("--iter", "Number of Stage-2 EM iterations", "25"));
		particle_diameter = textToFloat(parser.getOption("--particle_diameter", "Mask diameter in A (-1: 90% of output box)", "-1"));
		outer_diameter = textToFloat(parser.getOption("--outer_diameter", "Helical outer tube diameter in A", "220"));
		helical_rise = textToFloat(parser.getOption("--rise", "Helical rise in A for longitudinal offset restriction (<= 0: off)", "4.75"));
		psi_step = textToFloat(parser.getOption("--psi_step", "Stage-2 relion_refine psi_step before oversampling=1 (degrees)", "2"));
		sigma_psi = textToFloat(parser.getOption("--sigma_psi", "Local psi prior sigma around smoothed Stage-1 polarity (degrees)", "4"));
		do_bimodal_psi = parser.checkOption("--bimodal_psi", "Keep bimodal psi search in Stage 2 instead of resolved Stage-1 polarity");
		offset_range = textToFloat(parser.getOption("--offset_range", "Stage-2 translational search range in output pixels", "3"));
		offset_step = textToFloat(parser.getOption("--offset_step", "Stage-2 translational search step before oversampling=1 (pixels)", "1"));
		tau2_fudge = textToFloat(parser.getOption("--tau2_fudge", "Regularisation parameter T for Stage 2", "2"));
		highres_limit = textToFloat(parser.getOption("--highres_limit", "High-resolution limit for Stage-2 alignment in A (-1: none)", "-1"));
		do_auto_mask = parser.checkOption("--auto_mask", "Enable 2D contour reference auto-masking in Stage 2 (default on; disable with --dont_auto_mask)");
		do_dont_auto_mask = parser.checkOption("--dont_auto_mask", "Disable 2D contour reference auto-masking in Stage 2");
		auto_mask_start_res = textToFloat(parser.getOption("--auto_mask_start_res", "Start resolution (A) for Stage-2 2D auto-masking", "15"));
		auto_mask_soft = textToFloat(parser.getOption("--auto_mask_soft", "Soft edge width (A) for Stage-2 auto-masking (-1: 3x resolution)", "-1"));
		auto_mask_mass = textToFloat(parser.getOption("--auto_mask_mass", "Enclosed mass (kDa) for Stage-2 auto-masking (-1: Otsu threshold)", "-1"));
		helical_context_weight = textToFloat(parser.getOption("--helical_context_weight", "Local helical context weight in Stage 2 (0: disable)", "0.5"));

		parser.addSection("Execution options");
		nr_mpi = textToInteger(parser.getOption("--mpi", "Number of MPI ranks per Stage-2 run (1: single process relion_refine)", "1"));
		nr_threads = textToInteger(parser.getOption("--j", "Number of OpenMP threads per rank", "4"));
		nr_pool = textToInteger(parser.getOption("--pool", "Number of pooled particles per thread in Stage 2", "30"));
		do_keep_all_iter = parser.checkOption("--keep_all_iterations", "Keep all Stage-2 iteration files");
		do_dont_keep_all_iter = parser.checkOption("--dont_keep_all_iterations", "Prune old Stage-2 iteration files (default)");
		gpu_ids = parser.getOption("--gpu", "GPU device IDs for Stage 2 ('none': CPU only, 'auto': GPU autoselect, or e.g. '0')", "none");
		random_seed = textToInteger(parser.getOption("--random_seed", "Base random seed for Stage 2", "1"));
		do_prepare_only = parser.checkOption("--prepare_only", "Write typed particles, re-centered .vstacks, and stage2_commands.sh without running relion_refine");
		verb = textToInteger(parser.getOption("--verb", "Verbosity", "1"));
		if (parser.checkForErrors())
			REPORT_ERROR("Errors encountered on the command line, exiting...");
		if ((do_premult || do_auto_premult) && dont_premult)
			REPORT_ERROR("Specify at most one of --premultiply_ctf/--auto_premultiply_ctf and --dont_premultiply_ctf.");
		if (do_auto_mask && do_dont_auto_mask)
			REPORT_ERROR("Specify at most one of --auto_mask and --dont_auto_mask.");
		do_auto_mask = !do_dont_auto_mask;
		if (do_keep_all_iter && do_dont_keep_all_iter)
			REPORT_ERROR("Specify at most one of --keep_all_iterations and --dont_keep_all_iterations.");
		if (nr_pool < 1)
			REPORT_ERROR("--pool must be at least 1.");
	}

	static void appendOutputNode(MetaDataTable &MD_nodes, const std::string &name, const std::string &type_label)
	{
		MD_nodes.addObject();
		const long int r = MD_nodes.numberOfObjects() - 1;
		MD_nodes.setValue(EMDL_PIPELINE_NODE_NAME, FileName(name), r);
		MD_nodes.setValue(EMDL_PIPELINE_NODE_TYPE_LABEL, type_label, r);
	}

	void run(const char *argv0)
	{
		const std::string out_dir = trimTrailingSlashes(fn_out);
		vcache::makeDirs(out_dir);

		// Keep virtual particle cache local to the output directory unless explicitly overridden
		const std::string local_cache = out_dir + "/vcache";
		vcache::makeDirs(local_cache);
		setenv("RELION_VPARTICLE_CACHE", local_cache.c_str(), 0);

		if (fn_in.contains("_optimiser.star") &&
		    fn_in.size() >= std::string("_optimiser.star").size() &&
		    fn_in.compare(fn_in.size() - std::string("_optimiser.star").size(),
		                  std::string("_optimiser.star").size(), "_optimiser.star") == 0 &&
		    exists(fn_in))
		{
			const std::string opt_prefix = fn_in.beforeLastOf("_optimiser.star");
			MetaDataTable MDopt_in;
			MDopt_in.read(fn_in, "optimiser_general");
			if (fn_model.empty())
			{
				FileName opt_model;
				if (MDopt_in.getValue(EMDL_OPTIMISER_MODEL_STARFILE, opt_model) && !opt_model.empty())
				{
					if (exists(opt_model))
						fn_model = opt_model;
					else if (!fn_project_dir.empty() && exists(trimTrailingSlashes(fn_project_dir) + "/" + opt_model))
						fn_model = trimTrailingSlashes(fn_project_dir) + "/" + opt_model;
				}
			}
			if (fn_classes.empty() && exists(opt_prefix + "_classes.mrcs"))
				fn_classes = opt_prefix + "_classes.mrcs";

			FileName opt_data;
			bool resolved = false;
			if (MDopt_in.getValue(EMDL_OPTIMISER_DATA_STARFILE, opt_data) && !opt_data.empty())
			{
				if (exists(opt_data))
				{
					fn_in = opt_data;
					resolved = true;
				}
				else if (!fn_project_dir.empty() && exists(trimTrailingSlashes(fn_project_dir) + "/" + opt_data))
				{
					fn_in = trimTrailingSlashes(fn_project_dir) + "/" + opt_data;
					resolved = true;
				}
			}
			if (!resolved && exists(opt_prefix + "_data.star"))
				fn_in = opt_prefix + "_data.star";
		}

		const size_t in_slash = fn_in.find_last_of('/');
		const std::string star_dir = (in_slash != std::string::npos) ? fn_in.substr(0, in_slash) : std::string("");

		ObservationModel obsModel;
		MetaDataTable MD;
		ObservationModel::loadSafely(fn_in, obsModel, MD, "particles", verb);
		const long int N = MD.numberOfObjects();
		if (N <= 0)
			REPORT_ERROR("helix_subclassify ERROR: no particles found in " + fn_in);
		if (!MD.containsLabel(EMDL_IMAGE_COORD_X) || !MD.containsLabel(EMDL_IMAGE_COORD_Y) ||
		    !MD.containsLabel(EMDL_MICROGRAPH_NAME))
			REPORT_ERROR("helix_subclassify ERROR: input STAR must contain rlnCoordinateX, rlnCoordinateY, and rlnMicrographName.");

		// Resolve all unique micrographs early so headers / paths are verified once
		std::map<std::string, std::string> mic_resolved;
		std::vector<std::string> row_mic(N);
		for (long int i = 0; i < N; i++)
		{
			FileName mfn;
			MD.getValue(EMDL_MICROGRAPH_NAME, mfn, i);
			auto it = mic_resolved.find(mfn);
			if (it == mic_resolved.end())
				it = mic_resolved.emplace(mfn, resolveExistingPath(mfn, fn_project_dir, star_dir)).first;
			row_mic[i] = it->second;
		}

		// Determine per-optics Stage-1 box/angpix and original micrograph pixel size
		const int n_optics_raw = (int)obsModel.opticsMdt.numberOfObjects();
		const int n_og = std::max(1, n_optics_raw);
		std::vector<int> row_og(N, 0);
		std::vector<std::string> og_first_mic(n_og);
		for (long int i = 0; i < N; i++)
		{
			int og = (n_optics_raw > 0) ? obsModel.getOpticsGroup(MD, i) : 0;
			if (og < 0 || og >= n_og) og = 0;
			row_og[i] = og;
			if (og_first_mic[og].empty())
				og_first_mic[og] = row_mic[i];
		}

		std::vector<int> og_stage1_box(n_og, 0), og_extract_size(n_og, 0), og_output_size(n_og, 0);
		std::vector<double> og_stage1_angpix(n_og, 1.0), og_mic_angpix(n_og, -1.0), og_output_angpix(n_og, 1.0);
		for (int og = 0; og < n_og; og++)
		{
			double m_angpix = angpix_override;
			if (og < n_optics_raw)
			{
				if (obsModel.opticsMdt.containsLabel(EMDL_IMAGE_SIZE))
					obsModel.opticsMdt.getValue(EMDL_IMAGE_SIZE, og_stage1_box[og], og);
				if (obsModel.opticsMdt.containsLabel(EMDL_IMAGE_PIXEL_SIZE))
					obsModel.opticsMdt.getValue(EMDL_IMAGE_PIXEL_SIZE, og_stage1_angpix[og], og);
				if (m_angpix <= 0. && obsModel.opticsMdt.containsLabel(EMDL_MICROGRAPH_ORIGINAL_PIXEL_SIZE))
					obsModel.opticsMdt.getValue(EMDL_MICROGRAPH_ORIGINAL_PIXEL_SIZE, m_angpix, og);
				if (m_angpix <= 0. && obsModel.opticsMdt.containsLabel(EMDL_MICROGRAPH_PIXEL_SIZE))
					obsModel.opticsMdt.getValue(EMDL_MICROGRAPH_PIXEL_SIZE, m_angpix, og);
			}
			if (m_angpix <= 0.)
			{
				const std::string &mic_ref = !og_first_mic[og].empty() ? og_first_mic[og] : row_mic[0];
				Image<RFLOAT> Imic_head;
				Imic_head.read(mic_ref, false);
				m_angpix = Imic_head.samplingRateX();
			}
			if (m_angpix <= 0.)
				m_angpix = og_stage1_angpix[og];
			og_mic_angpix[og] = m_angpix;

			int ext_sz = extract_size;
			if (ext_sz <= 0)
			{
				if (og_stage1_box[og] <= 0 || og_stage1_angpix[og] <= 0.)
					REPORT_ERROR("helix_subclassify ERROR: cannot infer --extract_size without Stage-1 optics box/pixel size.");
				ext_sz = ROUND((double)og_stage1_box[og] * og_stage1_angpix[og] / m_angpix);
				if (ext_sz % 2 != 0) ext_sz++;
				ext_sz = std::max(16, ext_sz);
			}
			og_extract_size[og] = ext_sz;
			og_output_size[og] = (scale_size > 0 && scale_size != ext_sz) ? scale_size : ext_sz;
			og_output_angpix[og] = m_angpix * (double)ext_sz / (double)og_output_size[og];
		}

		std::vector<double> row_mic_angpix(N, 1.0);
		for (long int i = 0; i < N; i++)
			row_mic_angpix[i] = og_mic_angpix[row_og[i]];

		const int stage1_box = og_stage1_box[0];
		const double stage1_angpix = og_stage1_angpix[0];
		const int output_size = og_output_size[0];
		const double output_angpix = og_output_angpix[0];

		// Stage-1 typing (run HMM + phase ring + twist-rate split when absent or --retype)
		std::vector<int> seg_type(N, 1);
		int nr_types = 1;
		const bool need_typing = do_retype || !MD.containsLabel(EMDL_PARTICLE_HELICAL_TYPE);
		if (need_typing)
		{
			std::vector<long int> rows(N);
			std::vector<int> cls(N, 0);
			std::vector<RFLOAT> q(N, 0.9), psi(N, 0.), psip(N, 0.);
			int stage1_K = 0;
			const bool has_cls = MD.containsLabel(EMDL_PARTICLE_CLASS);
			for (long int i = 0; i < N; i++)
			{
				rows[i] = i;
				if (has_cls)
				{
					MD.getValue(EMDL_PARTICLE_CLASS, cls[i], i);
					cls[i]--;
				}
				stage1_K = std::max(stage1_K, cls[i] + 1);
				if (MD.containsLabel(EMDL_PARTICLE_PMAX)) MD.getValue(EMDL_PARTICLE_PMAX, q[i], i);
				if (MD.containsLabel(EMDL_ORIENT_PSI)) MD.getValue(EMDL_ORIENT_PSI, psi[i], i);
				if (MD.containsLabel(EMDL_ORIENT_PSI_PRIOR)) MD.getValue(EMDL_ORIENT_PSI_PRIOR, psip[i], i);
				else psip[i] = psi[i];
			}
			stage1_K = std::max(1, stage1_K);
			if (stage1_K > 1)
			{
				if (stage1_K >= 2)
				{
					types_min = std::max(1, std::min(types_min, stage1_K));
					if (types_max > 0)
						types_max = std::max(1, std::min(types_max, stage1_K));
				}
				HelicalContext ctx;
				ctx.verb = verb;
				ctx.nr_types = std::max(1, types_min);
				RFLOAT diam = (particle_diameter > 0.) ? particle_diameter : ((stage1_box > 0) ? (stage1_box * stage1_angpix) : (output_size * output_angpix));
				if (outer_diameter > 0.) diam = std::min(diam, outer_diameter);
				ctx.initialise(MD, rows, stage1_K, diam);
				std::vector<RFLOAT> ll(stage1_K);
				for (long int i = 0; i < N; i++)
				{
					const RFLOAT p = std::min((RFLOAT)0.99, std::max((RFLOAT)(1. / stage1_K), q[i]));
					for (int k = 0; k < stage1_K; k++)
						ll[k] = std::log((1. - p) / (stage1_K - 1));
					if (cls[i] >= 0 && cls[i] < stage1_K)
						ll[cls[i]] = std::log(p);
					ctx.storeLogLik(i, ll);
				}

				if (fn_classes.empty() && fn_in.contains("_data.star"))
					fn_classes = fn_in.beforeLastOf("_data.star") + "_classes.mrcs";
				if (fn_model.empty() && fn_in.contains("_data.star"))
					fn_model = fn_in.beforeLastOf("_data.star") + "_model.star";

				std::vector<long int> cls_cnt(stage1_K, 0);
				for (long int i = 0; i < N; i++)
					if (cls[i] >= 0 && cls[i] < stage1_K) cls_cnt[cls[i]]++;
				std::vector<RFLOAT> res(stage1_K, 0.), pdf(stage1_K, 0.);
				bool have_model = false;
				if (!fn_model.empty() && exists(fn_model))
				{
					MetaDataTable MDmod;
					MDmod.read(fn_model, "model_classes");
					if ((int)MDmod.numberOfObjects() >= stage1_K)
					{
						have_model = true;
						for (int k = 0; k < stage1_K; k++)
						{
							if (!MDmod.getValue(EMDL_MLMODEL_ESTIM_RESOL_REF, res[k], k)) res[k] = 0.;
							if (!MDmod.getValue(EMDL_MLMODEL_PDF_CLASS, pdf[k], k))
								pdf[k] = (RFLOAT)cls_cnt[k] / std::max(1L, N);
						}
					}
				}
				if (!have_model)
				{
					res.clear();
					for (int k = 0; k < stage1_K; k++)
						pdf[k] = (RFLOAT)cls_cnt[k] / std::max(1L, N);
				}
				std::vector<MultidimArray<RFLOAT> > refs;
				if (!fn_classes.empty() && exists(fn_classes))
				{
					Image<RFLOAT> Istk;
					Istk.read(fn_classes);
					if (NSIZE(Istk()) >= stage1_K)
					{
						refs.resize(stage1_K);
						for (int k = 0; k < stage1_K; k++)
						{
							Istk().getImage(k, refs[k]);
							refs[k].setXmippOrigin();
						}
					}
				}
				std::vector<RFLOAT> cont;
				if (!refs.empty())
				{
					std::vector<RFLOAT> track(N, 0.);
					std::vector<long int> chain(N, -1);
					std::vector<char> flipped(N, 0);
					for (long int i = 0; i < N; i++)
					{
						chain[i] = ctx.seg_chain[i];
						track[i] = ctx.seg_pos[i] * ctx.step;
						flipped[i] = (std::fabs(std::remainder(psi[i] - psip[i], 360.)) > 90.);
					}
					std::string cont_note;
					cont = helixtypes::classContinuity(refs, stage1_angpix, chain, track, cls, flipped, cont_note, outer_diameter);
				}
				const RFLOAT box_A = (stage1_box > 0) ? (stage1_box * stage1_angpix) : diam;
				ctx.junk_class = helixtypes::junkClasses(res, std::vector<RFLOAT>(), box_A, sort_classes_junk, cont, pdf);
				const long int min_cls_segs = (N >= 2000) ? 5 : 1;
				for (int k = 0; k < stage1_K; k++)
					if (cls_cnt[k] < min_cls_segs) ctx.junk_class[k] = 1;
				if (!refs.empty())
					ctx.computeClassEquivalence(refs, stage1_angpix, outer_diameter);
				if (do_hierarchical && types_max > ctx.nr_types && (int)ctx.filament_name.size() >= min_type_filaments)
				{
					const int tmax_node = (types_max > 1) ? types_max : std::min(12, std::max(2, max_types_total));
					ctx.runHierarchicalTyping(psi, psip, ctx.nr_types, tmax_node, max_types_total, min_type_filaments,
					                          nr_folds, types_tol, types_agree, true);
					ctx.fitPeriods(psi, psip);
				}
				else
				{
					if (types_max > ctx.nr_types)
						ctx.selectNrTypes(ctx.nr_types, types_max, nr_folds, types_tol, types_agree);
					ctx.update();
					ctx.fitPeriods(psi, psip);
					if (types_max > ctx.nr_types)
						ctx.splitTypesByTwistRate(types_max);
				}
				ctx.applyPhaseRingPriors(psi, psip);
				ctx.writeLabels(MD, rows, &cls);
				ctx.writeStar(out_dir + "/stage1_filaments.star");
				nr_types = std::max(1, ctx.nr_types);
				for (long int i = 0; i < N; i++)
					seg_type[i] = (ctx.seg_type[i] >= 0) ? (ctx.seg_type[i] + 1) : 0;
			}
		}
		else
		{
			for (long int i = 0; i < N; i++)
			{
				int t = 0;
				MD.getValue(EMDL_PARTICLE_HELICAL_TYPE, t, i);
				seg_type[i] = t;
				nr_types = std::max(nr_types, t);
			}
		}

		// Coherently resolve polarity & smooth centerlines before re-centering
		double max_fov_A = 0.;
		for (int og = 0; og < n_og; og++)
			max_fov_A = std::max(max_fov_A, (double)og_extract_size[og] * og_mic_angpix[og]);
		const RFLOAT max_rec_shift = (particle_diameter > 0.) ? (0.45 * particle_diameter) : (0.35 * max_fov_A);
		smoothAndRecenterFilaments(MD, row_mic_angpix, max_rec_shift);

		// Determine CTF premultiplication policy
		const bool has_ctf = MD.containsLabel(EMDL_CTF_DEFOCUSU) &&
		                       (obsModel.opticsMdt.containsLabel(EMDL_CTF_VOLTAGE) || MD.containsLabel(EMDL_CTF_VOLTAGE));
		const bool use_premult = has_ctf && (!dont_premult);
		if (do_premult && !has_ctf)
			REPORT_ERROR("helix_subclassify ERROR: --premultiply_ctf requested, but input STAR has no CTF metadata.");

		auto makeRecipeForOptics = [&](int og) -> vparticles::Recipe {
			vparticles::Recipe r;
			r.extract_size = og_extract_size[og];
			r.scale = (og_output_size[og] != og_extract_size[og]) ? og_output_size[og] : -1;
			r.window = -1;
			r.normalise = true;
			r.bg_radius = std::max(1, (int)std::floor(0.75 * 0.5 * (double)og_output_size[og]));
			r.ramp = true;
			r.white_dust = -1.;
			r.black_dust = -1.;
			r.invert_contrast = do_invert_contrast;
			r.float16 = do_float16;
			r.angpix = og_output_angpix[og];
			r.helical = true;
			const double hrad_px = (outer_diameter > 0.) ? (0.5 * outer_diameter / og_output_angpix[og]) : (double)r.bg_radius;
			r.helical_radius = std::min((double)r.bg_radius, hrad_px);
			if (use_premult)
			{
				r.premultiply_size = vparticles::Recipe::PREMULTIPLY_AUTO;
				r.ctf_intact_first_peak = ctf_intact_first_peak;
				r.ctf_angpix = og_mic_angpix[og];
			}
			return r;
		};

		// Partition segments into active types
		std::vector<std::vector<long int> > type_rows(nr_types + 1);
		for (long int i = 0; i < N; i++)
		{
			if (seg_type[i] >= 1 && seg_type[i] <= nr_types)
				type_rows[seg_type[i]].push_back(i);
		}

		struct Stage2Job {
			int type_id;
			long int count;
			std::string type_dir;
			std::string star_path;
			std::string run_root;
			std::string command;
		};
		std::vector<Stage2Job> jobs;
		const std::string refine_bin = findCompanionBinary(argv0, (nr_mpi > 1) ? "relion_refine_mpi" : "relion_refine");
		RFLOAT min_max_helix_diam = 1e30;
		double min_box_A = 1e30, max_out_apix = 0.;
		for (int og = 0; og < n_og; og++)
		{
			const int half_box_pix = og_output_size[og] / 2 - ((og_output_size[og] + 1) % 2);
			const RFLOAT og_max_diam = std::max((RFLOAT)(2.0 * og_output_angpix[og]), (RFLOAT)(2.0 * (double)half_box_pix * og_output_angpix[og]));
			min_max_helix_diam = std::min(min_max_helix_diam, og_max_diam);
			min_box_A = std::min(min_box_A, (double)og_output_size[og] * og_output_angpix[og]);
			max_out_apix = std::max(max_out_apix, og_output_angpix[og]);
		}
		const RFLOAT eff_diam = (particle_diameter > 0.)
			? particle_diameter
			: std::min(min_max_helix_diam, (RFLOAT)std::floor(0.9 * min_box_A));

		for (int t = 1; t <= nr_types; t++)
		{
			if ((long int)type_rows[t].size() < min_segs)
			{
				if (verb > 0 && !type_rows[t].empty())
					std::cout << " Type " << t << ": skipping Stage 2 (" << type_rows[t].size()
					          << " segments < --min_segs " << min_segs << ")" << std::endl;
				continue;
			}
			const std::string type_dir = out_dir + "/type_" + integerToString(t);
			const std::string stacks_dir = type_dir + "/stacks";
			vcache::makeDirs(stacks_dir);

			// Group selected rows by (resolved micrograph path, optics group) to emit one .vstack per pair
			std::map<std::pair<std::string, int>, std::vector<long int> > by_mic;
			for (long int r : type_rows[t])
				by_mic[std::make_pair(row_mic[r], row_og[r])].push_back(r);

			MetaDataTable MD_t;
			MD_t.addMissingLabels(&MD);
			if (!MD_t.containsLabel(EMDL_PARTICLE_HELICAL_TYPE))
				MD_t.addLabel(EMDL_PARTICLE_HELICAL_TYPE);

			long int mic_idx = 0;
			for (const auto &mkv : by_mic)
			{
				const std::string &abs_mic = mkv.first.first;
				const int group_og = mkv.first.second;
				const std::vector<long int> &mrows = mkv.second;
				const vparticles::Recipe recipe = makeRecipeForOptics(group_og);
				const std::string fn_vstack = stacks_dir + "/mic_" + integerToString(mic_idx, 5) + "_" + sanitizeStem(abs_mic) + ".vstack";
				mic_idx++;

				std::vector<std::pair<long, long> > centres;
				std::vector<double> virtual_psi;
				std::vector<vparticles::ParticleCtf> virtual_ctf;
				centres.reserve(mrows.size());
				virtual_psi.reserve(mrows.size());
				if (use_premult)
					virtual_ctf.reserve(mrows.size());

				for (size_t k = 0; k < mrows.size(); k++)
				{
					const long int orig_r = mrows[k];
					RFLOAT cx = 0., cy = 0., psip = 0.;
					MD.getValue(EMDL_IMAGE_COORD_X, cx, orig_r);
					MD.getValue(EMDL_IMAGE_COORD_Y, cy, orig_r);
					MD.getValue(EMDL_ORIENT_PSI_PRIOR, psip, orig_r);
					const long x_round = (long)std::floor(cx + 0.5);
					const long y_round = (long)std::floor(cy + 0.5);
					centres.push_back(std::make_pair(x_round, y_round));
					virtual_psi.push_back((double)psip);
					if (use_premult)
					{
						CTF ctf;
						ctf.readByGroup(MD, &obsModel, orig_r);
						const int og = (n_optics_raw > 0) ? obsModel.getOpticsGroup(MD, orig_r) : 0;
						virtual_ctf.push_back(vparticles::ParticleCtf{
							ctf.DeltafU, ctf.DeltafV, ctf.azimuthal_angle, ctf.kV, ctf.Cs, ctf.Q0,
							ctf.Bfac, ctf.scale, ctf.phase_shift, vparticles::opticsAberrations(obsModel, og)
						});
					}

					MD_t.addObject(MD.getObject(orig_r));
					const long int out_r = MD_t.numberOfObjects() - 1;
					const std::string img_name = integerToString((long int)(k + 1), 6) + "@" + fn_vstack;
					MD_t.setValue(EMDL_IMAGE_NAME, img_name, out_r);
					MD_t.setValue(EMDL_MICROGRAPH_NAME, abs_mic, out_r);
					MD_t.setValue(EMDL_PARTICLE_HELICAL_TYPE, t, out_r);
				}

				vparticles::writeDescriptor(fn_vstack, abs_mic, recipe, centres, virtual_psi,
				                            std::vector<double>(), std::vector<double>(),
				                            std::vector<double>(), virtual_ctf);
			}

			MetaDataTable MDopt_t = obsModel.opticsMdt;
			if (!MDopt_t.containsLabel(EMDL_IMAGE_SIZE)) MDopt_t.addLabel(EMDL_IMAGE_SIZE);
			if (!MDopt_t.containsLabel(EMDL_IMAGE_PIXEL_SIZE)) MDopt_t.addLabel(EMDL_IMAGE_PIXEL_SIZE);
			if (!MDopt_t.containsLabel(EMDL_OPTIMISER_DATA_ARE_CTF_PREMULTIPLIED))
				MDopt_t.addLabel(EMDL_OPTIMISER_DATA_ARE_CTF_PREMULTIPLIED);
			if (!MDopt_t.containsLabel(EMDL_OPTIMISER_DATA_ARE_CTF_PREMULTIPLIED_INTACT_FIRST_PEAK))
				MDopt_t.addLabel(EMDL_OPTIMISER_DATA_ARE_CTF_PREMULTIPLIED_INTACT_FIRST_PEAK);
			if (!MDopt_t.containsLabel(EMDL_MICROGRAPH_ORIGINAL_PIXEL_SIZE))
				MDopt_t.addLabel(EMDL_MICROGRAPH_ORIGINAL_PIXEL_SIZE);
			for (long int oi = 0; oi < (long int)MDopt_t.numberOfObjects(); oi++)
			{
				const int og = (oi < n_og) ? (int)oi : 0;
				MDopt_t.setValue(EMDL_IMAGE_SIZE, og_output_size[og], oi);
				MDopt_t.setValue(EMDL_IMAGE_PIXEL_SIZE, (RFLOAT)og_output_angpix[og], oi);
				MDopt_t.setValue(EMDL_OPTIMISER_DATA_ARE_CTF_PREMULTIPLIED, use_premult, oi);
				MDopt_t.setValue(EMDL_OPTIMISER_DATA_ARE_CTF_PREMULTIPLIED_INTACT_FIRST_PEAK, use_premult && ctf_intact_first_peak, oi);
				MDopt_t.setValue(EMDL_MICROGRAPH_ORIGINAL_PIXEL_SIZE, (RFLOAT)og_mic_angpix[og], oi);
				if (MDopt_t.containsLabel(EMDL_IMAGE_MTF_FILENAME) && !fn_project_dir.empty())
				{
					FileName mtf;
					if (MDopt_t.getValue(EMDL_IMAGE_MTF_FILENAME, mtf, oi) && !mtf.empty() && mtf[0] != '/' && !exists(mtf))
					{
						const std::string cand = trimTrailingSlashes(fn_project_dir) + "/" + std::string(mtf);
						if (exists(cand))
						{
							char buf[PATH_MAX];
							if (realpath(cand.c_str(), buf) != NULL)
								MDopt_t.setValue(EMDL_IMAGE_MTF_FILENAME, FileName(buf), oi);
							else
								MDopt_t.setValue(EMDL_IMAGE_MTF_FILENAME, FileName(cand), oi);
						}
					}
				}
			}

			const std::string fn_type_star = type_dir + "/particles.star";
			MetaDataTable MDgen_empty;
			ObservationModel::saveNew(MD_t, MDopt_t, MDgen_empty, fn_type_star, "particles");

			const std::string run_root = type_dir + "/run";
			std::ostringstream cmd;
			if (nr_mpi > 1)
			{
				const char *env_mpirun = getenv("RELION_MPIRUN");
				const std::string mpirun_bin = (env_mpirun != NULL && env_mpirun[0] != '\0') ? std::string(env_mpirun) : std::string("mpirun");
				// The Slurm script asks for one task fewer than the ranks (the leader only coordinates),
				// so mpirun must be told to oversubscribe. The flag, unlike the Open MPI 4 environment
				// variable, is also understood by Open MPI 5 (PRRTE).
				cmd << "OMPI_MCA_rmaps_base_oversubscribe=1 " << mpirun_bin << " ";
				if (mpirun_bin.find("srun") == std::string::npos)
					cmd << "--oversubscribe --bind-to none ";
				cmd << "-n " << nr_mpi << " ";
			}
			cmd << shellQuote(refine_bin)
			    << " --i " << shellQuote(fn_type_star)
			    << " --o " << shellQuote(run_root)
			    << " --K " << K
			    << " --iter " << nr_iter
			    << " --tau2_fudge " << tau2_fudge
			    << " --flatten_solvent --zero_mask --norm --scale --pad 2 --oversampling 1"
			    << " --particle_diameter " << eff_diam
			    << " --psi_step " << psi_step
			    << " --sigma_psi " << sigma_psi
			    << " --offset_range " << offset_range
			    << " --offset_step " << offset_step
			    << " --center_classes --sort_classes "
			    << (do_keep_all_iter ? "--keep_all_iterations" : "--dont_keep_all_iterations")
			    << " --no_init_blobs --dont_combine_weights_via_disc --pool " << nr_pool
			    << " --j " << nr_threads
			    << " --random_seed " << (random_seed + t);
			if (has_ctf)
			{
				cmd << " --ctf";
				if (ctf_intact_first_peak) cmd << " --ctf_intact_first_peak";
			}
			const RFLOAT eff_outer = (outer_diameter > 0.)
				? std::min(outer_diameter, std::floor(0.85 * eff_diam))
				: -1.;
			const bool has_valid_tube = (eff_outer >= 2.0 * max_out_apix && eff_outer < eff_diam);
			if (has_valid_tube)
				cmd << " --helical_outer_diameter " << eff_outer
				    << " --helical_cylindrical_mask --helical_auto_diameter";
			if (do_bimodal_psi)
				cmd << " --bimodal_psi";
			if (helical_rise > 0. && has_valid_tube)
				cmd << " --helix --helical_rise_initial " << helical_rise;
			if (highres_limit > 0.)
				cmd << " --strict_highres_exp " << highres_limit;
			if (do_auto_mask)
				cmd << " --auto_mask --auto_mask_start_res " << auto_mask_start_res
				    << " --auto_mask_soft " << auto_mask_soft
				    << " --auto_mask_mass " << auto_mask_mass;
			else
				cmd << " --dont_auto_mask";
			if (helical_context_weight > 0.)
				cmd << " --helical_context --helical_context_types 1 --helical_context_types_max 1 --helical_context_temper "
				    << helical_context_weight;
			if (gpu_ids != "none")
			{
				if (gpu_ids == "auto")
					cmd << " --gpu \"\"";
				else
					cmd << " --gpu " << shellQuote(gpu_ids);
			}
			jobs.push_back({t, (long int)type_rows[t].size(), type_dir, fn_type_star, run_root, cmd.str()});
			if (verb > 0)
			{
				std::cout << " Prepared Type " << t << ": " << type_rows[t].size() << " re-centered virtual segments ("
				          << og_output_size[0] << " px @ " << og_output_angpix[0] << " A/px, extract " << og_extract_size[0] << " px @ "
				          << og_mic_angpix[0] << " A/px";
				if (n_og > 1)
					std::cout << ", " << n_og << " optics groups";
				std::cout << (use_premult ? ", CTF-premultiplied" : "") << ")" << std::endl;
			}
		}

		// Write reproducibility script
		{
			const std::string sh_path = out_dir + "/stage2_commands.sh";
			std::ofstream sh(sh_path.c_str());
			sh << "#!/bin/bash\nset -euo pipefail\n";
			sh << "export RELION_VPARTICLE_CACHE=" << shellQuote(local_cache) << "\n";
			for (const Stage2Job &jb : jobs)
				sh << jb.command << " > " << shellQuote(jb.type_dir + "/run.out")
				   << " 2> " << shellQuote(jb.type_dir + "/run.err") << "\n";
			sh.close();
			chmod(sh_path.c_str(), 0755);
		}

		MetaDataTable MD_summary;
		MD_summary.setName("subclassification_summary");
		MD_summary.addLabel(EMDL_PARTICLE_HELICAL_TYPE);
		MD_summary.addLabel(EMDL_PARTICLE_CLASS);
		MD_summary.addLabel(EMDL_IMAGE_SIZE);
		MD_summary.addLabel(EMDL_IMAGE_PIXEL_SIZE);
		MD_summary.addLabel(EMDL_MLMODEL_PDF_CLASS);
		MD_summary.addLabel(EMDL_MLMODEL_ESTIM_RESOL_REF);
		MD_summary.addLabel(EMDL_IMAGE_NAME);

		MetaDataTable MD_nodes;
		MD_nodes.setName("output_nodes");

		if (do_prepare_only)
		{
			for (const Stage2Job &jb : jobs)
			{
				MD_summary.addObject();
				const long int r = MD_summary.numberOfObjects() - 1;
				MD_summary.setValue(EMDL_PARTICLE_HELICAL_TYPE, jb.type_id, r);
				MD_summary.setValue(EMDL_PARTICLE_CLASS, 0, r);
				MD_summary.setValue(EMDL_IMAGE_SIZE, output_size, r);
				MD_summary.setValue(EMDL_IMAGE_PIXEL_SIZE, (RFLOAT)output_angpix, r);
				MD_summary.setValue(EMDL_MLMODEL_PDF_CLASS, (RFLOAT)((double)jb.count / std::max(1L, N)), r);
				MD_summary.setValue(EMDL_MLMODEL_ESTIM_RESOL_REF, (RFLOAT)0., r);
				MD_summary.setValue(EMDL_IMAGE_NAME, jb.star_path, r);
				appendOutputNode(MD_nodes, jb.star_path, "ParticleGroupMetadata.star.relion.helicalsegments");
			}
			MD_summary.write(out_dir + "/subclassification_summary.star");
			MD_nodes.write(out_dir + "/RELION_OUTPUT_NODES.star");
			return;
		}

		MetaDataTable MD_merged;
		MetaDataTable MDopt_merged;
		bool have_merged_init = false;
		const std::string it_str = integerToString(nr_iter, 3);
		int class_stride = 100;
		while (class_stride < std::max(1, K))
			class_stride *= 10;
		for (const Stage2Job &jb : jobs)
		{
			if (pipeline_control_check_abort_job())
				exit(RELION_EXIT_ABORTED);
			if (verb > 0)
				std::cout << " Running Stage 2 for Type " << jb.type_id << " (" << jb.count << " segments) ..." << std::endl;
			const std::string full_cmd = jb.command + " > " + shellQuote(jb.type_dir + "/run.out")
			                           + " 2> " + shellQuote(jb.type_dir + "/run.err");
			const int rc = std::system(full_cmd.c_str());
			if (rc != 0)
				REPORT_ERROR("helix_subclassify ERROR: Stage-2 relion_refine failed for type " +
				             integerToString(jb.type_id) + " (see " + jb.type_dir + "/run.err).");

			const std::string fn_stage2_opt = jb.run_root + "_it" + it_str + "_optimiser.star";
			const std::string fn_stage2_data = jb.run_root + "_it" + it_str + "_data.star";
			const std::string fn_stage2_model = jb.run_root + "_it" + it_str + "_model.star";
			if (exists(fn_stage2_opt))
				appendOutputNode(MD_nodes, fn_stage2_opt, "OptimiserData.star.relion.class2d");
			if (exists(fn_stage2_data))
				appendOutputNode(MD_nodes, fn_stage2_data, "ParticleGroupMetadata.star.relion.class2d.helicalsegments");
			ObservationModel obs_s2;
			MetaDataTable MD_s2;
			ObservationModel::loadSafely(fn_stage2_data, obs_s2, MD_s2, "particles", 0);
			if (!have_merged_init)
			{
				MDopt_merged = obs_s2.opticsMdt;
				MD_merged.addMissingLabels(&MD_s2);
				if (!MD_merged.containsLabel(EMDL_PARTICLE_HELICAL_TYPE))
					MD_merged.addLabel(EMDL_PARTICLE_HELICAL_TYPE);
				have_merged_init = true;
			}
			for (long int r = 0; r < (long int)MD_s2.numberOfObjects(); r++)
			{
				int sub_cls = 1;
				MD_s2.getValue(EMDL_PARTICLE_CLASS, sub_cls, r);
				MD_merged.addObject(MD_s2.getObject(r));
				const long int mr = MD_merged.numberOfObjects() - 1;
				MD_merged.setValue(EMDL_PARTICLE_HELICAL_TYPE, jb.type_id, mr);
				MD_merged.setValue(EMDL_PARTICLE_CLASS, jb.type_id * class_stride + sub_cls, mr);
			}

			if (exists(fn_stage2_model))
			{
				MetaDataTable MDmod;
				MDmod.read(fn_stage2_model, "model_classes");
				for (long int k = 0; k < (long int)MDmod.numberOfObjects(); k++)
				{
					RFLOAT pdf = 0., res = 999.;
					FileName ref_name = "";
					MDmod.getValue(EMDL_MLMODEL_PDF_CLASS, pdf, k);
					MDmod.getValue(EMDL_MLMODEL_ESTIM_RESOL_REF, res, k);
					MDmod.getValue(EMDL_MLMODEL_REF_IMAGE, ref_name, k);
					MD_summary.addObject();
					const long int sr = MD_summary.numberOfObjects() - 1;
					MD_summary.setValue(EMDL_PARTICLE_HELICAL_TYPE, jb.type_id, sr);
					MD_summary.setValue(EMDL_PARTICLE_CLASS, jb.type_id * class_stride + (int)(k + 1), sr);
					MD_summary.setValue(EMDL_IMAGE_SIZE, output_size, sr);
					MD_summary.setValue(EMDL_IMAGE_PIXEL_SIZE, (RFLOAT)output_angpix, sr);
					MD_summary.setValue(EMDL_MLMODEL_PDF_CLASS, pdf, sr);
					MD_summary.setValue(EMDL_MLMODEL_ESTIM_RESOL_REF, res, sr);
					MD_summary.setValue(EMDL_IMAGE_NAME, ref_name, sr);
				}
			}
		}

		if (pipeline_control_check_abort_job())
			exit(RELION_EXIT_ABORTED);

		if (have_merged_init)
		{
			MetaDataTable MDgen_empty;
			const std::string merged_path = out_dir + "/merged_particles.star";
			ObservationModel::saveNew(MD_merged, MDopt_merged, MDgen_empty, merged_path, "particles");
			appendOutputNode(MD_nodes, merged_path, "ParticleGroupMetadata.star.relion.class2d.helicalsegments");
		}
		MD_summary.write(out_dir + "/subclassification_summary.star");
		MD_nodes.write(out_dir + "/RELION_OUTPUT_NODES.star");
		if (verb > 0)
			std::cout << " Written " << out_dir << "/merged_particles.star and " << out_dir << "/subclassification_summary.star" << std::endl;
	}
};

int main(int argc, char *argv[])
{
	helix_subclassify_parameters prm;
	try
	{
		prm.read(argc, argv);
		prm.run(argv[0]);
	}
	catch (RelionError XE)
	{
		std::cerr << XE;
		return RELION_EXIT_FAILURE;
	}
	return RELION_EXIT_SUCCESS;
}
