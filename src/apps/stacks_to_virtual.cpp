/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * Convert the particle stacks of existing Extract jobs into virtual particle
 * stacks, reclaiming their disk space.
 *
 * Each .mrcs written by relion_preprocess is replaced, under the same name, by
 * a small descriptor from which the same particles are computed on demand (see
 * documentation/virtual_particles.md). Because the name does not change, no
 * STAR file anywhere in the project has to be touched.
 *
 * Nothing is replaced on trust. For every stack the particles are recomputed
 * from the micrograph with the recipe recorded in the job's note.txt and
 * compared, pixel by pixel, with the stack on disk; only a stack reproduced
 * exactly is replaced. Without --convert nothing is written at all.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include <src/args.h>
#include <src/image.h>
#include <src/metadata_table.h>
#include <src/virtual_particles.h>
#include <src/float16.h>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string humanBytes(double b)
{
	const char* u[] = {"B", "KB", "MB", "GB", "TB"};
	int k = 0;
	while (b >= 1024. && k < 4) { b /= 1024.; k++; }
	std::ostringstream s;
	s << std::fixed << std::setprecision(k == 0 ? 0 : 1) << b << " " << u[k];
	return s.str();
}

long long fileSize(const std::string& path)
{
	struct stat st;
	return (stat(path.c_str(), &st) == 0) ? (long long)st.st_size : -1;
}

/// Every file named *_extract.star below dir (Extract writes one per micrograph).
void findExtractStars(const std::string& dir, std::vector<std::string>& out, int depth = 0)
{
	if (depth > 6) return;
	DIR* d = opendir(dir.c_str());
	if (d == NULL) return;
	struct dirent* e;
	while ((e = readdir(d)) != NULL)
	{
		const std::string name = e->d_name;
		if (name == "." || name == "..") continue;
		const std::string full = dir + "/" + name;
		struct stat st;
		if (lstat(full.c_str(), &st) != 0) continue;
		if (S_ISDIR(st.st_mode)) findExtractStars(full, out, depth + 1);
		else if (name.size() > 13 && name.compare(name.size() - 13, 13, "_extract.star") == 0)
			out.push_back(full);
	}
	closedir(d);
	std::sort(out.begin(), out.end());
}

/* The recipe an Extract job used, from the last relion_preprocess command in
 * its note.txt. Returns false, with a reason, for jobs the virtual reader
 * cannot reproduce. */
/// What a job's command line says beyond the per-particle recipe.
struct JobInfo {
	std::string mics_star;           ///< --i: the micrographs STAR (for pixel sizes)
	double helical_diameter = -1.;   ///< --helical_outer_diameter, in Angstrom
};

bool recipeFromNote(const std::string& job, vparticles::Recipe& r, JobInfo& info, std::string& why)
{
	std::ifstream f((job + "/note.txt").c_str());
	if (!f) { why = "no note.txt"; return false; }
	std::string line, cmd;
	while (std::getline(f, line))
		if (line.find("relion_preprocess") != std::string::npos) cmd = line;
	if (cmd.empty()) { why = "no relion_preprocess command in note.txt"; return false; }

	std::vector<std::string> t;
	{
		std::istringstream is(cmd);
		std::string w;
		while (is >> w) t.push_back(w);
	}
	std::map<std::string, std::string> opt;
	for (size_t i = 0; i < t.size(); i++)
	{
		if (t[i].compare(0, 2, "--") != 0) continue;
		const bool has_value = (i + 1 < t.size()) &&
			!(t[i + 1].compare(0, 2, "--") == 0 && !isdigit((unsigned char)t[i + 1][2]));
		opt[t[i]] = has_value ? t[i + 1] : "";
	}

	if (!opt.count("--extract")) { why = "not an extraction"; return false; }
	for (const char* bad : {"--phase_flip", "--premultiply_ctf", "--project3d"})
		if (opt.count(bad)) { why = std::string("uses ") + bad + ", which virtual particles do not support"; return false; }
	if (opt.count("--virtual")) { why = "already extracted as virtual particles"; return false; }

	r = vparticles::Recipe();
	r.extract_size = opt.count("--extract_size") ? textToInteger(opt["--extract_size"]) : -1;
	r.scale = opt.count("--scale") ? textToInteger(opt["--scale"]) : -1;
	r.window = opt.count("--window") ? textToInteger(opt["--window"]) : -1;
	// relion_preprocess rounds odd output sizes up to even
	if (r.scale > 0 && r.scale % 2) r.scale++;
	if (r.window > 0 && r.window % 2) r.window++;
	r.normalise = opt.count("--norm") > 0;
	r.bg_radius = opt.count("--bg_radius") ? textToInteger(opt["--bg_radius"]) : -1;
	r.ramp = opt.count("--no_ramp") == 0;
	r.white_dust = opt.count("--white_dust") ? textToFloat(opt["--white_dust"]) : -1;
	r.black_dust = opt.count("--black_dust") ? textToFloat(opt["--black_dust"]) : -1;
	r.invert_contrast = opt.count("--invert_contrast") > 0;
	r.float16 = opt.count("--float16") > 0;
	r.helical = opt.count("--helix") > 0;
	info.mics_star = opt.count("--i") ? opt["--i"] : "";
	info.helical_diameter = opt.count("--helical_outer_diameter") ? textToFloat(opt["--helical_outer_diameter"]) : -1.;
	if (r.extract_size <= 0) { why = "no --extract_size in note.txt"; return false; }
	if (r.helical && info.mics_star.empty()) { why = "helical job without --i in note.txt"; return false; }
	if (r.white_dust > 0 || r.black_dust > 0)
	{
		why = "uses dust removal, which fills dust pixels with random values that cannot be reproduced";
		return false;
	}
	return true;
}

std::string describe(const vparticles::Recipe& r)
{
	std::ostringstream s;
	s << "box " << r.extract_size;
	if (r.scale > 0) s << " -> " << r.scale;
	if (r.window > 0) s << ", window " << r.window;
	if (r.normalise) s << ", norm r=" << r.bg_radius << (r.ramp ? "" : " (no ramp)");
	if (r.white_dust > 0 || r.black_dust > 0) s << ", dust " << r.white_dust << "/" << r.black_dust;
	if (r.invert_contrast) s << ", invert";
	if (r.helical) s << ", helical";
	s << (r.float16 ? ", float16" : ", float32");
	return s.str();
}

/// One stack to consider, gathered serially before the parallel work.
struct StackTask {
	std::string stack, mic;
	std::vector<std::pair<long, long> > centres;   // in stack order
	std::vector<double> psi;                        // helical: per segment, in stack order
	double helical_radius = -1.;
	std::string skip;                               // non-empty: why it is left alone
};

struct StackResult {
	enum Status { IDENTICAL, FLOAT16_FIX, DIFFERENT, SKIPPED } status;
	std::string detail;
	long long bytes_before = 0, bytes_after = 0;
	long n = 0;
	long doubled = 0;
	double max_diff = 0;
	bool replaced = false;
};

/* A pixel the pre-5.0 float16 writer corrupted: the correctly rounded value is
 * a power of two and the stack holds exactly twice it (the rounding carry was
 * applied twice). Fixed upstream in 1b8adc2e (RELION 5.0.0). */
inline bool isDoubledPixel(RFLOAT stack_value, RFLOAT virtual_value)
{
	if (virtual_value == 0 || stack_value != 2 * virtual_value) return false;
	int e;
	return std::frexp(std::fabs(virtual_value), &e) == 0.5;
}

class StacksToVirtual {
public:
	void read(int argc, char** argv)
	{
		parser.setCommandLine(argc, argv);
		parser.addSection("Options");
		fn_project = parser.getOption("--project", "RELION project directory", ".");
		jobs_arg = parser.getOption("--job", "Extract job(s) to convert, comma-separated (default: every Extract job in the project)", "");
		do_convert = parser.checkOption("--convert", "Replace verified stacks (otherwise only verify and report; nothing is written)");
		accept_fix = parser.checkOption("--accept_float16_fix", "Also convert float16 stacks whose only differences are pixels corrupted by the pre-5.0 float16 writer");
		nr_threads = textToInteger(parser.getOption("--j", "Number of threads", "1"));
		max_stacks = textToInteger(parser.getOption("--max_stacks", "Consider at most this many stacks per job (for a trial run; -1: all)", "-1"));
		verb = textToInteger(parser.getOption("--verb", "Verbosity", "1"));
		if (parser.checkForErrors())
			REPORT_ERROR("Errors encountered on the command line (see above), exiting...");
	}

	void run()
	{
		if (chdir(fn_project.c_str()) != 0)
			REPORT_ERROR("Cannot enter project directory " + fn_project);

		// Converting must not fill a cache: verification computes particles directly
		setenv("RELION_VPARTICLE_CACHE", "off", 1);

		if (!do_convert)
		{
			const char* base = getenv("TMPDIR");
			std::string tmpl = std::string((base && base[0]) ? base : "/tmp") + "/relion_stacks_to_virtual.XXXXXX";
			std::vector<char> buf(tmpl.begin(), tmpl.end());
			buf.push_back('\0');
			if (mkdtemp(buf.data()) == NULL) REPORT_ERROR("Cannot create a temporary directory in " + tmpl);
			private_tmp = buf.data();
		}

		std::vector<std::string> jobs;
		if (!jobs_arg.empty())
		{
			std::stringstream ss(jobs_arg);
			std::string j;
			while (std::getline(ss, j, ',')) if (!j.empty()) jobs.push_back(j);
		}
		else
		{
			DIR* d = opendir("Extract");
			if (d != NULL)
			{
				struct dirent* e;
				while ((e = readdir(d)) != NULL)
				{
					const std::string n = e->d_name;
					if (n.compare(0, 3, "job") == 0) jobs.push_back("Extract/" + n);
				}
				closedir(d);
			}
			std::sort(jobs.begin(), jobs.end());
		}
		if (jobs.empty())
			REPORT_ERROR("No Extract jobs found in " + fn_project);

		std::cout << (do_convert ? " Converting" : " Checking (dry run; nothing will be written)")
		          << " " << jobs.size() << " Extract job(s) in " << fn_project << std::endl;

		long long total_before = 0, total_after = 0, total_saved = 0;
		for (size_t j = 0; j < jobs.size(); j++)
		{
			std::string job = jobs[j];
			while (job.size() > 1 && job[job.size() - 1] == '/') job.erase(job.size() - 1);
			long long before = 0, after = 0;
			processJob(job, before, after);
			total_before += before;
			total_after += after;
		}
		total_saved = total_before - total_after;

		std::cout << std::endl << " Total: " << humanBytes(total_before) << " of stacks "
		          << (do_convert ? "replaced by " : "can be replaced by ") << humanBytes(total_after)
		          << " of descriptors, " << (do_convert ? "saving " : "which would save ")
		          << humanBytes(total_saved) << "." << std::endl;
		if (!do_convert && total_saved > 0)
			std::cout << " Run again with --convert to replace the verified stacks." << std::endl;
		if (!private_tmp.empty())
			if (system(("rm -rf '" + private_tmp + "'").c_str()) != 0) { /* best effort */ }
	}

private:
	std::string private_tmp;   // dry run: descriptors are written here, never in the project

	/* Where a stack's candidate descriptor goes. Converting needs it beside the
	 * stack, so the final rename is atomic; a dry run must not write into the
	 * project at all (it may not even be ours to write to). */
	std::string tmpNameFor(const std::string& stack) const
	{
		if (do_convert) return stack + ".virtual.tmp";
		std::string flat = stack;
		for (size_t i = 0; i < flat.size(); i++) if (flat[i] == '/') flat[i] = '_';
		return private_tmp + "/" + flat;
	}

	IOParser parser;
	std::string fn_project, jobs_arg;
	bool do_convert, accept_fix;
	int nr_threads, max_stacks, verb;

	// Per job, for helical recipes
	std::map<std::string, double> mic_angpix;
	double helical_diameter = -1.;

	/// Micrograph pixel sizes as relion_preprocess saw them (optics group of each micrograph).
	bool readMicrographPixelSizes(const std::string& fn, std::string& why)
	{
		mic_angpix.clear();
		if (!exists(fn)) { why = "input micrographs STAR " + fn + " not found"; return false; }
		MetaDataTable optics, mics;
		optics.read(fn, "optics");
		mics.read(fn, "micrographs");
		std::map<int, double> by_group;
		FOR_ALL_OBJECTS_IN_METADATA_TABLE(optics)
		{
			int g; RFLOAT a;
			if (optics.getValue(EMDL_IMAGE_OPTICS_GROUP, g) && optics.getValue(EMDL_MICROGRAPH_PIXEL_SIZE, a))
				by_group[g] = a;
		}
		FOR_ALL_OBJECTS_IN_METADATA_TABLE(mics)
		{
			FileName name; int g = 1;
			mics.getValue(EMDL_MICROGRAPH_NAME, name);
			mics.getValue(EMDL_IMAGE_OPTICS_GROUP, g);
			if (by_group.count(g)) mic_angpix[name] = by_group[g];
		}
		if (mic_angpix.empty()) { why = "no micrograph pixel sizes in " + fn; return false; }
		return true;
	}

	void processJob(const std::string& job, long long& saved_before, long long& saved_after)
	{
		std::cout << std::endl << " " << job << "/" << std::endl;
		vparticles::Recipe recipe;
		JobInfo info;
		std::string why;
		if (!recipeFromNote(job, recipe, info, why) ||
		    (recipe.helical && !readMicrographPixelSizes(info.mics_star, why)))
		{
			std::cout << "   skipped: " << why << std::endl;
			return;
		}
		helical_diameter = info.helical_diameter;
		std::cout << "   recipe: " << describe(recipe) << std::endl;

		// Each stack's particle list: from the per-micrograph *_extract.star
		// files when they survive, else from the job's particles.star (many
		// projects have the former cleaned away)
		std::map<std::string, std::vector<Row> > by_stack;
		std::vector<std::string> stars;
		findExtractStars(job, stars);
		std::string source;
		if (!stars.empty())
		{
			source = "*_extract.star";
			for (size_t i = 0; i < stars.size(); i++)
			{
				MetaDataTable md;
				md.read(stars[i]);
				collectRows(md, recipe, by_stack);
			}
		}
		else if (exists(job + "/particles.star"))
		{
			source = "particles.star";
			MetaDataTable md;
			md.read(job + "/particles.star", "particles");
			collectRows(md, recipe, by_stack);
		}
		else
		{
			std::cout << "   skipped: neither *_extract.star files nor particles.star" << std::endl;
			return;
		}
		if (verb > 0) std::cout << "   particle lists from " << source << std::endl;
		std::vector<std::string> names;
		for (std::map<std::string, std::vector<Row> >::const_iterator it = by_stack.begin(); it != by_stack.end(); ++it)
			names.push_back(it->first);
		if (max_stacks > 0 && (long)names.size() > max_stacks) names.resize(max_stacks);

		// Gather serially: STAR parsing and the checks that need no pixels
		std::vector<StackTask> tasks(names.size());
		for (size_t i = 0; i < names.size(); i++) gather(names[i], by_stack[names[i]], recipe, tasks[i]);

		std::vector<StackResult> results(tasks.size());
		#pragma omp parallel for schedule(dynamic) num_threads(nr_threads)
		for (long i = 0; i < (long)tasks.size(); i++)
			verifyAndConvert(tasks[i], recipe, results[i]);

		// Report
		long n_ident = 0, n_fix = 0, n_diff = 0, n_skip = 0, n_part = 0;
		std::map<std::string, long> skip_reasons;
		std::ofstream log;
		if (do_convert) log.open((job + "/virtual_conversion.log").c_str(), std::ios::app);
		for (size_t i = 0; i < results.size(); i++)
		{
			const StackResult& r = results[i];
			switch (r.status)
			{
			case StackResult::IDENTICAL:   n_ident++; break;
			case StackResult::FLOAT16_FIX: n_fix++;   break;
			case StackResult::DIFFERENT:   n_diff++;  break;
			case StackResult::SKIPPED:     n_skip++; skip_reasons[r.detail]++; break;
			}
			n_part += r.n;
			const bool counts = (r.status == StackResult::IDENTICAL) ||
			                    (r.status == StackResult::FLOAT16_FIX && accept_fix);
			if (counts && (!do_convert || r.replaced))
			{
				saved_before += r.bytes_before;
				saved_after += r.bytes_after;
			}
			if (verb > 1 || r.status == StackResult::DIFFERENT)
				std::cout << "     " << tasks[i].stack << ": " << r.detail << std::endl;
			if (log.is_open())
				log << tasks[i].stack << "\t" << (r.replaced ? "replaced" : "kept") << "\t" << r.detail << "\n";
		}

		std::cout << "   stacks " << results.size() << " (" << n_part << " particles): "
		          << n_ident << " identical";
		if (n_fix) std::cout << ", " << n_fix << " identical but for pre-5.0 float16 pixels"
		                     << (accept_fix ? "" : " (use --accept_float16_fix to convert these)");
		if (n_diff) std::cout << ", " << n_diff << " DIFFERENT (kept)";
		if (n_skip) std::cout << ", " << n_skip << " skipped";
		std::cout << std::endl;
		for (std::map<std::string, long>::const_iterator it = skip_reasons.begin(); it != skip_reasons.end(); ++it)
			std::cout << "     skipped " << it->second << ": " << it->first << std::endl;
		std::cout << "   " << humanBytes(saved_before) << " of stacks -> " << humanBytes(saved_after)
		          << " of descriptors" << (do_convert ? " (replaced)" : " (if converted)") << std::endl;
	}

	/// One particle as the extraction saw it.
	struct Row {
		long idx;
		long x, y;          // centre the window was cut around (truncated, as relion_preprocess does)
		double psi;
		bool has_psi;
		std::string mic;
	};

	void collectRows(MetaDataTable& md, const vparticles::Recipe& recipe,
	                 std::map<std::string, std::vector<Row> >& by_stack)
	{
		FOR_ALL_OBJECTS_IN_METADATA_TABLE(md)
		{
			FileName name, stack, mic;
			RFLOAT x, y, psi = 0.;
			md.getValue(EMDL_IMAGE_NAME, name);
			md.getValue(EMDL_MICROGRAPH_NAME, mic);
			md.getValue(EMDL_IMAGE_COORD_X, x);
			md.getValue(EMDL_IMAGE_COORD_Y, y);
			Row r;
			r.has_psi = !recipe.helical || md.getValue(EMDL_ORIENT_PSI_PRIOR, psi);
			name.decompose(r.idx, stack);
			r.x = (long)x;
			r.y = (long)y;
			r.psi = psi;
			r.mic = mic;
			by_stack[stack].push_back(r);
		}
	}

	void gather(const std::string& stack, std::vector<Row>& rows, const vparticles::Recipe& recipe, StackTask& t)
	{
		t.stack = stack;
		if (rows.empty()) { t.skip = "no particles listed"; return; }
		std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.idx < b.idx; });
		t.mic = rows[0].mic;
		for (size_t i = 0; i < rows.size(); i++)
		{
			if (rows[i].mic != t.mic) { t.skip = "particles from several micrographs in one stack"; return; }
			if (!rows[i].has_psi) { t.skip = "helical segments without rlnAnglePsiPrior"; return; }
			if (rows[i].idx != (long)i + 1) { t.skip = "stack numbering is not 1..n"; return; }
			t.centres.push_back(std::make_pair(rows[i].x, rows[i].y));
			if (recipe.helical) t.psi.push_back(rows[i].psi);
		}
		if (recipe.helical)
		{
			std::map<std::string, double>::const_iterator a = mic_angpix.find(t.mic);
			if (a == mic_angpix.end()) { t.skip = "micrograph not in the job's input STAR file"; return; }
			// Preprocessing::helicalBackgroundRadius, types and all (int / int)
			RFLOAT radius = (helical_diameter * 0.5) / a->second;
			if (recipe.scale > 0) radius *= recipe.scale / recipe.extract_size;
			t.helical_radius = radius;
		}

		if (!exists(t.stack)) { t.skip = "stack file missing"; return; }
		if (vparticles::isVirtualStackFile(t.stack)) { t.skip = "already virtual"; return; }
		if (!exists(t.mic)) { t.skip = "micrograph missing"; return; }

		Image<RFLOAT> head;
		head.read(t.stack, false);
		if (NSIZE(head()) != (long)t.centres.size()) { t.skip = "stack holds a different number of particles than listed"; return; }
		if (XSIZE(head()) != recipe.outputSize() || YSIZE(head()) != recipe.outputSize())
			{ t.skip = "stack box size does not match the recorded recipe"; return; }
	}

	void verifyAndConvert(const StackTask& t, const vparticles::Recipe& recipe_in, StackResult& r)
	{
		r.n = t.centres.size();
		if (!t.skip.empty()) { r.status = StackResult::SKIPPED; r.detail = t.skip; return; }
		r.bytes_before = fileSize(t.stack);

		try
		{
			Image<RFLOAT> real;
			real.read(t.stack);   // whole stack: one sequential read

			vparticles::Recipe recipe = recipe_in;
			recipe.angpix = real.samplingRateX();
			recipe.helical_radius = t.helical_radius;

			const std::string tmp = tmpNameFor(t.stack);
			#pragma omp critical(stacks_to_virtual_write)
			vparticles::writeDescriptor(tmp, t.mic, recipe, t.centres, t.psi);

			std::vector<MultidimArray<RFLOAT> > virt;
			vparticles::computeParticles(tmp, virt);
			vparticles::forget(tmp);

			const size_t slice = (size_t)XSIZE(real()) * YSIZE(real());
			long differing = 0;
			for (size_t k = 0; k < virt.size(); k++)
			{
				for (size_t i = 0; i < slice; i++)
				{
					const RFLOAT s = DIRECT_MULTIDIM_ELEM(real(), k * slice + i);
					const RFLOAT v = DIRECT_MULTIDIM_ELEM(virt[k], i);
					if (s == v) continue;
					if (recipe.float16 && isDoubledPixel(s, v)) { r.doubled++; continue; }
					differing++;
					r.max_diff = std::max(r.max_diff, (double)std::fabs(s - v));
				}
			}

			std::ostringstream d;
			if (differing > 0)
			{
				r.status = StackResult::DIFFERENT;
				d << differing << " pixels differ (max |diff| " << r.max_diff << ")";
			}
			else if (r.doubled > 0)
			{
				r.status = StackResult::FLOAT16_FIX;
				d << "identical except " << r.doubled << " pixels doubled by the pre-5.0 float16 writer";
			}
			else
			{
				r.status = StackResult::IDENTICAL;
				d << "identical";
			}
			r.detail = d.str();
			r.bytes_after = fileSize(tmp);

			const bool ok = (r.status == StackResult::IDENTICAL) ||
			                (r.status == StackResult::FLOAT16_FIX && accept_fix);
			if (do_convert && ok)
			{
				// Atomic: a reader sees either the old stack or the descriptor.
				// Processes holding the old stack open keep reading it until they close it.
				if (rename(tmp.c_str(), t.stack.c_str()) == 0) r.replaced = true;
				else { unlink(tmp.c_str()); r.detail += "; could not replace the stack"; }
			}
			else
				unlink(tmp.c_str());
		}
		catch (RelionError& e)
		{
			unlink(tmpNameFor(t.stack).c_str());
			r.status = StackResult::SKIPPED;
			r.detail = "error: " + e.msg.substr(0, e.msg.find('\n'));
		}
	}
};

} // namespace

int main(int argc, char* argv[])
{
	StacksToVirtual prm;
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
