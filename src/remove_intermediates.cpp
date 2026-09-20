#include "src/remove_intermediates.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <algorithm>
#include <map>
#include <sstream>

namespace relion_cleanup {

namespace {

bool isJobDirName(const std::string& name)
{
	if (name.size() != 6 || name.compare(0, 3, "job") != 0) return false;
	for (size_t i = 3; i < name.size(); i++)
		if (!isdigit((unsigned char)name[i])) return false;
	return true;
}

/* Split "run_ct7_it012_optimiser.star" into base "run_ct7" and iteration 12.
 *
 * The iteration number is taken from the last "_it" followed by digits, so a
 * job whose own name contains "_it" does not confuse it. Returns false for
 * anything that is not an iteration file. */
bool parseIterationFile(const std::string& name, std::string& base, long& iteration)
{
	if (name.compare(0, 3, "run") != 0) return false;

	size_t pos = std::string::npos;
	for (size_t at = name.find("_it"); at != std::string::npos; at = name.find("_it", at + 1))
	{
		size_t d = at + 3;
		if (d < name.size() && isdigit((unsigned char)name[d])) pos = at;
	}
	if (pos == std::string::npos) return false;

	size_t d = pos + 3, end = d;
	while (end < name.size() && isdigit((unsigned char)name[end])) end++;

	// There has to be something after the number: run_it025_data.star, not run_it025
	if (end >= name.size() || name[end] != '_') return false;

	base = name.substr(0, pos);
	iteration = strtol(name.substr(d, end - d).c_str(), NULL, 10);
	return true;
}

long long fileSize(const std::string& path)
{
	struct stat st;
	if (lstat(path.c_str(), &st) != 0) return 0;
	return (long long)st.st_size;
}

/// The project directory plus every jobNNN directory below it.
void collectJobDirs(const std::string& dir, std::vector<std::string>& out, int depth)
{
	if (depth > 8) return;   // a project is shallow; this only guards against link loops

	DIR* d = opendir(dir.c_str());
	if (d == NULL) return;

	struct dirent* ent;
	while ((ent = readdir(d)) != NULL)
	{
		const std::string name = ent->d_name;
		if (name == "." || name == "..") continue;

		const std::string full = dir + "/" + name;
		struct stat st;
		// lstat, so that a symlinked directory is not descended into
		if (lstat(full.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) continue;

		if (isJobDirName(name)) out.push_back(full);
		collectJobDirs(full, out, depth + 1);
	}
	closedir(d);
}

void scanOneDir(const std::string& dir, Plan& plan)
{
	DIR* d = opendir(dir.c_str());
	if (d == NULL) return;

	// base name -> iteration -> the files of that iteration
	std::map<std::string, std::map<long, std::vector<std::string> > > groups;

	struct dirent* ent;
	while ((ent = readdir(d)) != NULL)
	{
		const std::string name = ent->d_name;
		std::string base;
		long iteration = 0;
		if (!parseIterationFile(name, base, iteration)) continue;

		const std::string full = dir + "/" + name;
		struct stat st;
		if (lstat(full.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;

		groups[base][iteration].push_back(full);
	}
	closedir(d);

	for (std::map<std::string, std::map<long, std::vector<std::string> > >::const_iterator g =
	         groups.begin(); g != groups.end(); ++g)
	{
		const std::map<long, std::vector<std::string> >& by_iter = g->second;
		if (by_iter.empty()) continue;

		const long first = by_iter.begin()->first;
		const long last = by_iter.rbegin()->first;

		char kept[32];
		snprintf(kept, sizeof(kept), "_it%03ld", first);
		plan.kept.push_back(dir + "/" + g->first + kept);
		if (last != first)
		{
			snprintf(kept, sizeof(kept), "_it%03ld", last);
			plan.kept.push_back(dir + "/" + g->first + kept);
		}

		for (std::map<long, std::vector<std::string> >::const_iterator it = by_iter.begin();
		     it != by_iter.end(); ++it)
		{
			if (it->first == first || it->first == last) continue;
			for (size_t f = 0; f < it->second.size(); f++)
			{
				Candidate c;
				c.path = it->second[f];
				c.size_bytes = fileSize(c.path);
				plan.total_bytes += c.size_bytes;
				plan.remove.push_back(c);
			}
		}
	}
}

} // anonymous namespace

Plan planIntermediateRemoval(const std::string& project_dir)
{
	Plan plan;

	std::vector<std::string> dirs;
	dirs.push_back(project_dir);
	collectJobDirs(project_dir, dirs, 0);
	std::sort(dirs.begin(), dirs.end());
	dirs.erase(std::unique(dirs.begin(), dirs.end()), dirs.end());

	for (size_t i = 0; i < dirs.size(); i++)
		scanOneDir(dirs[i], plan);

	std::sort(plan.kept.begin(), plan.kept.end());
	return plan;
}

long applyRemoval(const Plan& plan, long long& freed_bytes,
                  std::vector<std::string>& errors,
                  ProgressFn progress, void* user_data)
{
	freed_bytes = 0;
	long n = 0;

	// Often enough to look alive, rarely enough not to cost more than the
	// unlinks themselves on a project with tens of thousands of files
	const size_t report_every = 32;

	for (size_t i = 0; i < plan.remove.size(); i++)
	{
		if (unlink(plan.remove[i].path.c_str()) == 0)
		{
			freed_bytes += plan.remove[i].size_bytes;
			n++;
		}
		else
		{
			errors.push_back(plan.remove[i].path);
		}

		if (progress != NULL && (i + 1) % report_every == 0)
			progress(i + 1, plan.remove.size(), user_data);
	}

	if (progress != NULL)
		progress(plan.remove.size(), plan.remove.size(), user_data);

	return n;
}

std::string humanSize(long long bytes)
{
	static const char* units[] = {"B", "KB", "MB", "GB", "TB"};
	double v = (double)bytes;
	int u = 0;
	while (v >= 1024.0 && u < 4) { v /= 1024.0; u++; }

	char buf[64];
	snprintf(buf, sizeof(buf), (u == 0) ? "%.0f %s" : "%.1f %s", v, units[u]);
	return buf;
}

} // namespace relion_cleanup
