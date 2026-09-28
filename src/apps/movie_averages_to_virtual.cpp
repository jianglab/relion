/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

/* relion_movie_averages_to_virtual: replace the micrographs of a project's
 * MotionCorr jobs by virtual movie average descriptors, in place.
 *
 * Only micrographs whose motion record carries a sum recipe can be replaced
 * (made with RELION_VIRTUAL_MOVIE_AVERAGES=real): each one is regenerated from
 * its movie and compared pixel for pixel with the file, and replaced (atomic
 * rename) only if they are identical. Legacy micrographs cannot be regenerated
 * exactly and are left alone. Dry run unless --convert.
 *
 * The descriptor records the identity of the file it replaced, so virtual
 * particles extracted from the real micrograph keep verifying.
 */

#include <src/args.h>
#include <src/image.h>
#include <src/motioncorr_runner.h>
#include <src/micrograph_model.h>
#include <src/virtual_movie_averages.h>
#include <src/virtual_particles.h>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
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

bool endsWith(const std::string& s, const std::string& t)
{
	return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
}

/// The micrographs below a MotionCorr job: every x.mrc with an x.star record
/// next to it (power spectra, non-dose-weighted and odd/even sums excluded).
void findMicrographs(const std::string& dir, std::vector<std::string>& out, int depth = 0)
{
	if (depth > 8) return;
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
		if (S_ISDIR(st.st_mode)) { findMicrographs(full, out, depth + 1); continue; }
		if (!S_ISREG(st.st_mode) || !endsWith(name, ".mrc")) continue;
		if (endsWith(name, "_PS.mrc") || endsWith(name, "_noDW.mrc") ||
		    endsWith(name, "_ODD.mrc") || endsWith(name, "_EVN.mrc")) continue;
		if (fileSize(full.substr(0, full.size() - 4) + ".star") < 0) continue;
		out.push_back(full);
	}
	closedir(d);
}

class MovieAveragesToVirtual {
public:
	void read(int argc, char** argv)
	{
		parser.setCommandLine(argc, argv);
		parser.addSection("Options");
		fn_project = parser.getOption("--project", "RELION project directory", ".");
		jobs_arg = parser.getOption("--job", "MotionCorr job(s) to convert, comma-separated (default: every MotionCorr job in the project)", "");
		do_convert = parser.checkOption("--convert", "Replace verified micrographs (otherwise only verify and report; nothing is written)");
		nr_threads = textToInteger(parser.getOption("--j", "Number of threads for regenerating each micrograph", "1"));
		max_mics = textToInteger(parser.getOption("--max_micrographs", "Consider at most this many micrographs per job (for a trial run; -1: all)", "-1"));
		verb = textToInteger(parser.getOption("--verb", "Verbosity", "1"));
		if (parser.checkForErrors())
			REPORT_ERROR("Errors encountered on the command line (see above), exiting...");
	}

	void run()
	{
		if (chdir(fn_project.c_str()) != 0)
			REPORT_ERROR("Cannot enter project directory " + fn_project);

		std::vector<std::string> jobs;
		if (!jobs_arg.empty())
		{
			std::stringstream ss(jobs_arg);
			std::string j;
			while (std::getline(ss, j, ',')) if (!j.empty()) jobs.push_back(j);
		}
		else
		{
			DIR* d = opendir("MotionCorr");
			if (d != NULL)
			{
				struct dirent* e;
				while ((e = readdir(d)) != NULL)
				{
					const std::string n = e->d_name;
					if (n.compare(0, 3, "job") == 0) jobs.push_back("MotionCorr/" + n);
				}
				closedir(d);
			}
			std::sort(jobs.begin(), jobs.end());
		}
		if (jobs.empty())
			REPORT_ERROR("No MotionCorr jobs found in " + fn_project);

		std::cout << (do_convert ? " Converting" : " Checking (dry run; nothing will be written)")
		          << " " << jobs.size() << " MotionCorr job(s) in " << fn_project << std::endl;

		long long total_before = 0, total_after = 0, total_legacy = 0;
		for (size_t j = 0; j < jobs.size(); j++)
		{
			std::string job = jobs[j];
			while (job.size() > 1 && job[job.size() - 1] == '/') job.erase(job.size() - 1);
			processJob(job, total_before, total_after, total_legacy);
		}

		std::cout << std::endl << " Total: " << humanBytes(total_before) << " of micrographs "
		          << (do_convert ? "replaced by " : "can be replaced by ") << humanBytes(total_after)
		          << " of descriptors, " << (do_convert ? "saving " : "which would save ")
		          << humanBytes(total_before - total_after) << "." << std::endl;
		if (total_legacy > 0)
			std::cout << " " << humanBytes(total_legacy) << " of micrographs were made without a sum recipe (legacy);"
			          << " they cannot be regenerated exactly and are kept." << std::endl;
		if (!do_convert && total_before > total_after)
			std::cout << " Run again with --convert to replace the verified micrographs." << std::endl;
		if (total_before > total_after)
			std::cout << " Afterwards the movies are the only copy of these micrographs' data: keep them safe." << std::endl;
	}

private:
	IOParser parser;
	std::string fn_project, jobs_arg;
	bool do_convert;
	int nr_threads, max_mics, verb;

	void processJob(const std::string& job, long long& total_before, long long& total_after, long long& total_legacy)
	{
		std::cout << std::endl << " " << job << "/" << std::endl;
		std::vector<std::string> mics;
		findMicrographs(job, mics);
		std::sort(mics.begin(), mics.end());
		if (max_mics >= 0 && (int)mics.size() > max_mics) mics.resize(max_mics);

		std::ofstream log;
		if (do_convert) log.open((job + "/virtual_conversion.log").c_str(), std::ios::app);

		int n_same = 0, n_legacy = 0, n_virtual = 0, n_diff = 0, n_error = 0;
		long long before = 0, after = 0, legacy_bytes = 0;
		std::map<std::string, int> errors;
		for (size_t i = 0; i < mics.size(); i++)
		{
			const std::string& mic = mics[i];
			const std::string record = mic.substr(0, mic.size() - 4) + ".star";
			const long long size = fileSize(mic);
			if (vmovies::isVirtualMovieAverageFile(mic)) { n_virtual++; continue; }

			try
			{
				MotioncorrRunner::SumRecipe recipe;
				if (!MotioncorrRunner::readRecipe(record, recipe))
				{
					n_legacy++;
					legacy_bytes += size;
					continue;
				}

				// Regenerate without the cache (converting must not fill it) and compare
				Image<float> regenerated, stored;
				MotioncorrRunner::regenerateMicrograph(record, regenerated, nr_threads);
				stored.read(mic);
				bool same = XSIZE(stored()) == XSIZE(regenerated()) && YSIZE(stored()) == YSIZE(regenerated())
				            && NSIZE(stored()) == 1 && ZSIZE(stored()) == 1;
				if (same)
					FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(stored())
						if (DIRECT_MULTIDIM_ELEM(stored(), n) != DIRECT_MULTIDIM_ELEM(regenerated(), n)) { same = false; break; }
				if (!same)
				{
					n_diff++;
					if (verb > 0) std::cout << "     " << mic << ": DIFFERENT from its regeneration (kept)" << std::endl;
					continue;
				}

				n_same++;
				before += size;
				if (do_convert)
				{
					// The identity of the file this descriptor stands for, so that
					// virtual particles extracted from it still verify
					const std::string identity = vparticles::micrographChecksum(mic);
					vmovies::writeDescriptor(mic, record, identity);
					after += fileSize(mic);
					log << mic << " replaced (" << humanBytes(size) << " -> " << humanBytes(fileSize(mic)) << ")" << std::endl;
				}
				else
				{
					after += fileSize(record) + 128;   // the descriptor is the record plus two lines
				}
			}
			catch (RelionError& e)
			{
				n_error++;
				errors[e.msg.substr(0, e.msg.find('\n'))]++;
			}
		}

		std::cout << "   micrographs " << mics.size() << ": " << n_same
		          << (do_convert ? " replaced" : " regenerate bit for bit");
		if (n_virtual) std::cout << ", " << n_virtual << " already virtual";
		if (n_legacy) std::cout << ", " << n_legacy << " legacy (no sum recipe; kept)";
		if (n_diff) std::cout << ", " << n_diff << " DIFFERENT (kept)";
		if (n_error) std::cout << ", " << n_error << " failed (kept)";
		std::cout << std::endl;
		for (std::map<std::string, int>::const_iterator it = errors.begin(); it != errors.end(); ++it)
			std::cout << "     failed " << it->second << ": " << it->first << std::endl;
		if (n_same)
			std::cout << "   " << humanBytes(before) << " of micrographs -> " << humanBytes(after) << " of descriptors" << std::endl;

		total_before += before;
		total_after += after;
		total_legacy += legacy_bytes;
	}
};

} // namespace

int main(int argc, char* argv[])
{
	MovieAveragesToVirtual prm;
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
