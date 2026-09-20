/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * Import a whole CryoSPARC project as a RELION project, reusing its
 * micrographs, motion correction, CTF fits, picks, extractions and
 * refinements instead of recomputing them.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include <src/args.h>
#include <src/cryosparc_project.h>
#include <src/cryosparc_motioncorr.h>
#include <src/cryosparc_jobs.h>
#include <src/cryosparc_pipeline.h>
#include <dirent.h>
#include <src/strings.h>
#include <src/macros.h>
#include <src/filename.h>
#include <src/pipeline_control.h>

#include <sys/stat.h>
#include <iostream>
#include <iomanip>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <algorithm>

/* How each CryoSPARC job type maps onto a RELION job directory. Job types that
 * do not appear here have no RELION equivalent worth importing (interactive
 * inspection jobs, utility jobs that only reshuffle particle sets, and so on);
 * they are reported as skipped rather than silently ignored, so that a project
 * using an unfamiliar job type is visible to the user.
 */
struct JobMapping {
	const char* cs_type;      // CryoSPARC job_type
	const char* relion_dir;   // RELION job directory prefix
	const char* description;
};

static const JobMapping job_mappings[] = {
	// Import
	{"import_movies",                 "Import",       "movies"},
	{"import_micrographs",            "Import",       "micrographs"},
	{"import_particles",              "Import",       "particles"},
	{"import_volumes",                "Import",       "volume"},

	// Motion correction
	{"patch_motion_correction_multi", "MotionCorr",   "patch motion correction"},
	{"motion_correction_multi",       "MotionCorr",   "full-frame motion correction"},
	{"motion_correction",             "MotionCorr",   "full-frame motion correction"},
	{"rigid_motion_correction",       "MotionCorr",   "rigid motion correction"},

	// CTF estimation
	{"patch_ctf_estimation_multi",    "CtfFind",      "patch CTF estimation"},
	{"ctf_estimation",                "CtfFind",      "CTF estimation"},

	// Picking
	{"blob_picker_gpu",               "AutoPick",     "blob picking"},
	{"template_picker_gpu",           "AutoPick",     "template picking"},
	{"filament_tracer_gpu",           "AutoPick",     "filament tracing"},
	{"manual_picker",                 "ManualPick",   "manual picking"},
	{"inspect_picks_v2",              "AutoPick",     "inspected picks"},

	// Extraction
	{"extract_micrographs_multi",     "Extract",      "particle extraction"},
	{"extract_micrographs",           "Extract",      "particle extraction"},

	// 2D
	{"class_2D",                      "Class2D",      "2D classification"},
	{"class_2D_new",                  "Class2D",      "2D classification"},
	{"select_2D",                     "Select",       "2D class selection"},

	// 3D
	{"homo_abinit",                   "InitialModel", "ab-initio reconstruction"},
	{"class_3D",                      "Class3D",      "3D classification"},
	{"hetero_refine",                 "Class3D",      "heterogeneous refinement"},
	{"homo_refine",                   "Refine3D",     "homogeneous refinement"},
	{"homo_refine_new",               "Refine3D",     "homogeneous refinement"},
	{"nonuniform_refine_new",         "Refine3D",     "non-uniform refinement"},
	{"helix_refine",                  "Refine3D",     "helical refinement"},
	{"new_local_refine",              "Refine3D",     "local refinement"},

	// CTF refinement
	{"ctf_refine_global",             "CtfRefine",    "global CTF refinement"},
	{"ctf_refine_local",              "CtfRefine",    "local CTF refinement"},

	{NULL, NULL, NULL}
};

static const JobMapping* findMapping(const std::string& cs_type)
{
	for (int i = 0; job_mappings[i].cs_type != NULL; i++)
		if (cs_type == job_mappings[i].cs_type) return &job_mappings[i];
	return NULL;
}

class ImportCryosparcProject
{
public:
	IOParser parser;

	FileName fn_in, fn_out, only_job;
	bool do_dry_run;
	int verb;
	int patches_x, patches_y;
	RFLOAT dose_per_frame, pre_exposure, amplitude_contrast;

	void read(int argc, char** argv)
	{
		parser.setCommandLine(argc, argv);

		int gen = parser.addSection("General options");
		fn_in  = parser.getOption("--i", "CryoSPARC project directory (the one holding J1, J2, ...)", "");
		fn_out = parser.getOption("--o", "RELION project directory to create the imported jobs in", "");
		only_job = parser.getOption("--job", "Import only this CryoSPARC job and the jobs it depends on (e.g. J48); default is every completed job", "");
		do_dry_run = parser.checkOption("--dry_run", "Report what would be imported, without writing anything");

		int mc = parser.addSection("Motion correction options");
		patches_x = textToInteger(parser.getOption("--patches_x", "Patches along X at which to sample CryoSPARC's local motion field", "5"));
		patches_y = textToInteger(parser.getOption("--patches_y", "Patches along Y at which to sample CryoSPARC's local motion field", "5"));
		dose_per_frame = textToFloat(parser.getOption("--dose_per_frame", "Dose per frame (e/A2); default is the import job's total dose divided by the frame count", "-1"));
		pre_exposure = textToFloat(parser.getOption("--pre_exposure", "Pre-exposure (e/A2)", "0"));
		amplitude_contrast = textToFloat(parser.getOption("--amplitude_contrast", "Amplitude contrast (CryoSPARC import jobs do not record one)", "0.1"));
		verb = textToInteger(parser.getOption("--verb", "Verbosity (1=normal, 0=silent)", "1"));

		if (parser.checkForErrors())
			REPORT_ERROR("Errors encountered on the command line (see above), exiting...");

		if (fn_in == "")
			REPORT_ERROR("ERROR: please provide a CryoSPARC project directory with --i");
		if (fn_out == "" && !do_dry_run)
			REPORT_ERROR("ERROR: please provide an output RELION project directory with --o (or use --dry_run)");
	}

	void run()
	{
		cryosparc::Project project = cryosparc::Project::read(fn_in, verb);

		if (project.jobs().empty())
			REPORT_ERROR("ERROR: no CryoSPARC jobs (J1, J2, ...) found in " + fn_in);

		std::vector<std::string> order;
		if (only_job != "")
		{
			if (project.job(only_job) == NULL)
				REPORT_ERROR("ERROR: no job " + only_job + " in " + fn_in);
			order = project.ancestryOf(only_job);
			if (verb > 0)
				std::cout << " Importing " << only_job << " and the jobs it depends on." << std::endl;
		}
		else
		{
			order = project.topologicalOrder();
			if (verb > 0)
				std::cout << " Importing every completed job." << std::endl;
		}

		const cryosparc::Project::Acquisition acq = project.acquisition();
		if (verb > 0)
		{
			std::cout << " CryoSPARC project : " << project.dir() << std::endl;
			std::cout << " Jobs in project   : " << project.jobs().size() << std::endl;
			if (acq.found)
			{
				std::cout << " Acquisition       : " << acq.voltage << " kV, "
				          << acq.pixel_size_A << " A/pix, Cs " << acq.spherical_aberration << " mm";
				if (acq.total_dose_e_per_A2 > 0.)
					std::cout << ", total dose " << acq.total_dose_e_per_A2 << " e/A2";
				std::cout << std::endl;
				if (!acq.gainref_path.empty())
					std::cout << " Gain reference    : " << acq.gainref_path << std::endl;
			}
			else
			{
				std::cout << " Acquisition       : not found (no import job with voltage and pixel size)"
				          << std::endl;
			}
			std::cout << std::endl;
		}

		reportPlan(project, order);

		if (!do_dry_run) execute(project, order, acq);
	}

private:
	void execute(const cryosparc::Project& project,
	             const std::vector<std::string>& order,
	             const cryosparc::Project::Acquisition& acq)
	{
		makeDir(fn_out);
		markProjectDir(fn_out);

		// RELION numbers jobs across the whole project, not within each job
		// type, so job001 appears once no matter which directory it is in.
		int next_job = 0;
		int n_done = 0, n_failed = 0;
		std::vector<cryosparc::PipelineJob> pipeline;

		if (verb > 0) std::cout << std::endl << " Writing jobs into " << fn_out << " ..." << std::endl;

		for (size_t i = 0; i < order.size(); i++)
		{
			const cryosparc::Job* j = project.job(order[i]);
			if (j == NULL) continue;

			const JobMapping* m = findMapping(j->type);
			if (m == NULL) continue;

			const int num = ++next_job;
			char rel[64];
			snprintf(rel, sizeof(rel), "%s/job%03d", m->relion_dir, num);
			const std::string job_dir = std::string(fn_out) + "/" + rel;

			try
			{
				if (cryosparc::isMotionCorrectionJob(j->type))
				{
					makeDir(job_dir);
					cryosparc::MotionCorrOptions opts;
					opts.patches_x = patches_x;
					opts.patches_y = patches_y;
					opts.pre_exposure_A2 = pre_exposure;
					opts.amplitude_contrast = amplitude_contrast;
					opts.dose_per_frame_A2 = dose_per_frame;
					opts.verb = verb;

					const long n = cryosparc::writeMotionCorrJob(
						project, *j, fn_out, job_dir, acq, opts);

					addPipelineJob(pipeline, *j, m->relion_dir, rel, job_dir,
					               std::string(rel) + "/corrected_micrographs.star");
					touch(FileName(job_dir + "/" + RELION_JOB_EXIT_SUCCESS));

					if (verb > 0)
						std::cout << "  " << j->uid << " -> " << rel
						          << "  (" << n << " micrographs)" << std::endl;
					n_done++;
				}
				else
				{
					makeDir(job_dir);
					cryosparc::JobWriteOptions o;
					o.amplitude_contrast = amplitude_contrast;
					o.verb = verb;

					const cryosparc::JobWriteResult r = cryosparc::writeGenericJob(
						project, *j, m->relion_dir, fn_out, job_dir, acq, o);

					addPipelineJob(pipeline, *j, m->relion_dir, rel, job_dir, r.main_star);
					touch(FileName(job_dir + "/" + RELION_JOB_EXIT_SUCCESS));

					if (verb > 0)
					{
						std::cout << "  " << j->uid << " -> " << rel
						          << "  (" << r.n_rows << " rows, " << r.n_links << " files linked)";
						if (!r.note.empty()) std::cout << "  [" << r.note << "]";
						std::cout << std::endl;
					}
					n_done++;
				}
			}
			catch (const std::exception& e)
			{
				std::cerr << "  " << j->uid << " -> " << rel << "  FAILED: " << e.what() << std::endl;
				n_failed++;
			}
		}

		const long n_star = cryosparc::writeJobStars(fn_out, pipeline);
		const long n_proc = cryosparc::writePipeline(fn_out, pipeline, next_job + 1);

		if (verb > 0)
		{
			std::cout << std::endl << " Wrote " << n_done << " job(s)";
			if (n_failed > 0) std::cout << ", " << n_failed << " failed";
			std::cout << "." << std::endl;
			std::cout << " Wrote default_pipeline.star with " << n_proc
			          << " process(es) and " << n_star << " job.star file(s);"
			          << " open the project in the RELION GUI to browse them." << std::endl;
		}
	}

	/* Record a written job for the pipeline. Its nodes are the files actually
	 * present in the job directory, so that whatever each writer produced is
	 * what the GUI is told about. */
	static void addPipelineJob(std::vector<cryosparc::PipelineJob>& out,
	                           const cryosparc::Job& j, const std::string& kind,
	                           const std::string& rel, const std::string& job_dir,
	                           const std::string& primary)
	{
		cryosparc::PipelineJob pj;
		pj.process = rel + "/";
		pj.type_label = cryosparc::processTypeLabel(kind, j.type);
		pj.cs_uid = j.uid;
		pj.parent_uids = j.parents;
		pj.primary_node = primary;

		DIR* d = opendir(job_dir.c_str());
		if (d != NULL)
		{
			struct dirent* ent;
			while ((ent = readdir(d)) != NULL)
			{
				const std::string name = ent->d_name;
				if (name == "." || name == "..") continue;

				// Only advertise the job's own products, not the many per
				// micrograph particle stacks it links in
				const bool is_star = name.size() > 5 && name.compare(name.size() - 5, 5, ".star") == 0;
				const bool is_run_mrc = name.compare(0, 4, "run_") == 0 &&
				                        name.size() > 4 &&
				                        (name.compare(name.size() - 4, 4, ".mrc") == 0 ||
				                         name.compare(name.size() - 5, 5, ".mrcs") == 0);
				if (!is_star && !is_run_mrc) continue;
				if (is_star && name.compare(0, 4, "run_") != 0 &&
				    name != "corrected_micrographs.star" && name != "micrographs.star" &&
				    name != "movies.star" && name != "particles.star" &&
				    name != "micrographs_ctf.star" && name != "particles_ctf_refine.star")
					continue;

				pj.nodes.push_back(std::make_pair(rel + "/" + name,
				                                  cryosparc::nodeTypeLabel(kind, name)));
			}
			closedir(d);
		}

		std::sort(pj.nodes.begin(), pj.nodes.end());
		out.push_back(pj);
	}

	/* Make the output directory recognisable as a RELION project.
	 *
	 * The GUI decides whether a directory is a project by the presence of the
	 * .gui_projectdir sentinel, and refuses to open it otherwise; .TMP_runfiles
	 * is where it stages job.star files when jobs are created. Only the GUI
	 * normally writes these, so an import run from the command line has to.
	 */
	static void markProjectDir(const std::string& path)
	{
		touch(FileName(path + "/.gui_projectdir"));
		makeDir(path + "/.TMP_runfiles");
	}

	static void makeDir(const std::string& path)
	{
		std::string cur;
		for (size_t i = 0; i < path.size(); i++)
		{
			cur += path[i];
			if (path[i] == '/' || i + 1 == path.size()) mkdir(cur.c_str(), 0755);
		}
	}

	void reportPlan(const cryosparc::Project& project, const std::vector<std::string>& order)
	{
		int next_job = 0;                     // project-wide, as RELION numbers jobs
		std::vector<std::string> unsupported;

		if (verb > 0)
		{
			std::cout << std::left
			          << std::setw(6)  << "job"
			          << std::setw(30) << "cryosparc type"
			          << std::setw(22) << "relion job"
			          << "final outputs" << std::endl;
			std::cout << std::string(100, '-') << std::endl;
		}

		int n_import = 0;
		for (size_t i = 0; i < order.size(); i++)
		{
			const cryosparc::Job* j = project.job(order[i]);
			if (j == NULL) continue;

			const JobMapping* m = findMapping(j->type);
			if (m == NULL)
			{
				unsupported.push_back(j->uid + " (" + j->type + ")");
				continue;
			}

			const int num = ++next_job;
			char rel[64];
			snprintf(rel, sizeof(rel), "%s/job%03d", m->relion_dir, num);

			// What would actually be read for this job
			std::string outputs;
			for (size_t g = 0; g < j->groups.size(); g++)
			{
				const std::string primary = j->groups[g].primaryMetafile();
				if (primary.empty()) continue;
				if (!outputs.empty()) outputs += ", ";
				outputs += j->groups[g].name + "=" + primary.substr(primary.find_last_of('/') + 1);
			}
			if (outputs.empty()) outputs = "(no metadata)";

			if (verb > 0)
			{
				std::cout << std::left
				          << std::setw(6)  << j->uid
				          << std::setw(30) << j->type
				          << std::setw(22) << rel
				          << outputs << std::endl;
			}
			n_import++;
		}

		if (verb > 0)
		{
			std::cout << std::endl << " " << n_import << " job(s) to import." << std::endl;

			if (!unsupported.empty())
			{
				std::cout << " " << unsupported.size()
				          << " completed job(s) have no RELION equivalent and will be skipped:" << std::endl;
				for (size_t i = 0; i < unsupported.size(); i++)
					std::cout << "   " << unsupported[i] << std::endl;
			}

			if (do_dry_run)
				std::cout << std::endl << " --dry_run: nothing was written." << std::endl;
		}
	}
};

int main(int argc, char** argv)
{
	ImportCryosparcProject app;

	try
	{
		app.read(argc, argv);
		app.run();
	}
	catch (RelionError XE)
	{
		std::cerr << XE;
		return RELION_EXIT_FAILURE;
	}

	return RELION_EXIT_SUCCESS;
}
