#include "src/cryosparc_project.h"
#include "src/cryosparc_json.h"

#include <dirent.h>
#include <sys/stat.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <set>
#include <iostream>

namespace cryosparc {

bool isJobDirName(const std::string& name)
{
	if (name.size() < 2 || name[0] != 'J') return false;
	for (size_t i = 1; i < name.size(); i++)
		if (!isdigit((unsigned char)name[i])) return false;
	return true;
}

long jobNumber(const std::string& uid)
{
	if (!isJobDirName(uid)) return -1;
	return atol(uid.c_str() + 1);
}

// ---------------------------------------------------------------------------
// OutputGroup
// ---------------------------------------------------------------------------

std::string OutputGroup::primaryMetafile() const
{
	// A group's columns are split over several results that normally all point
	// at the same primary .cs file; pick the most frequently referenced
	// non-passthrough one so that an unusual extra output cannot win.
	std::map<std::string, int> tally;
	for (size_t i = 0; i < results.size(); i++)
	{
		if (results[i].passthrough) continue;
		if (results[i].metafile.empty()) continue;
		tally[results[i].metafile]++;
	}

	std::string best;
	int best_count = 0;
	for (std::map<std::string, int>::const_iterator it = tally.begin(); it != tally.end(); ++it)
	{
		if (it->second > best_count) { best = it->first; best_count = it->second; }
	}
	return best;
}

std::vector<std::string> OutputGroup::passthroughMetafiles() const
{
	std::vector<std::string> out;
	std::set<std::string> seen;
	for (size_t i = 0; i < results.size(); i++)
	{
		if (!results[i].passthrough) continue;
		if (results[i].metafile.empty()) continue;
		if (seen.insert(results[i].metafile).second)
			out.push_back(results[i].metafile);
	}
	return out;
}

const OutputResult* OutputGroup::result(const std::string& name) const
{
	for (size_t i = 0; i < results.size(); i++)
		if (results[i].name == name) return &results[i];
	return NULL;
}

bool OutputGroup::hasResultType(const std::string& type) const
{
	for (size_t i = 0; i < results.size(); i++)
		if (results[i].type == type) return true;
	return false;
}

// ---------------------------------------------------------------------------
// Job
// ---------------------------------------------------------------------------

const OutputGroup* Job::group(const std::string& name) const
{
	for (size_t i = 0; i < groups.size(); i++)
		if (groups[i].name == name) return &groups[i];
	return NULL;
}

const OutputGroup* Job::groupOfType(const std::string& type) const
{
	for (size_t i = 0; i < groups.size(); i++)
		if (groups[i].type == type) return &groups[i];
	return NULL;
}

// ---------------------------------------------------------------------------
// Project
// ---------------------------------------------------------------------------

const Job* Project::job(const std::string& uid) const
{
	std::map<std::string, Job>::const_iterator it = jobs_.find(uid);
	return (it == jobs_.end()) ? NULL : &it->second;
}

std::string Project::resolve(const std::string& relative) const
{
	if (!relative.empty() && relative[0] == '/') return relative;
	std::string out = dir_;
	if (!out.empty() && out[out.size() - 1] != '/') out += "/";
	out += relative;
	return out;
}

void Project::addJobFromFile(const std::string& job_dir, const std::string& uid, int verb)
{
	const std::string fn = job_dir + "/job.json";

	struct stat st;
	if (stat(fn.c_str(), &st) != 0) return;   // not a job directory

	JsonPtr root;
	try { root = jsonParseFile(fn); }
	catch (const std::exception& e)
	{
		// One damaged job should not make a whole project unimportable.
		if (verb > 0)
			std::cerr << " WARNING: skipping " << uid << ": " << e.what() << std::endl;
		return;
	}

	Job j;
	j.uid    = root->get("uid")->asString(uid);
	j.type   = root->get("job_type")->asString(root->get("type")->asString());
	j.status = root->get("status")->asString();
	j.title  = root->get("title")->asString();
	j.parents = root->get("parents")->asStringVector();

	// Group metadata first, so that results can be attached to a group that
	// already knows its own CryoSPARC type.
	std::map<std::string, size_t> group_index;
	JsonPtr groups = root->get("output_result_groups");
	for (size_t i = 0; i < groups->size(); i++)
	{
		JsonPtr g = groups->at(i);
		OutputGroup og;
		og.name  = g->get("name")->asString();
		og.type  = g->get("type")->asString();
		og.title = g->get("title")->asString();
		og.num_items = g->get("num_items")->asLong();
		if (og.name.empty()) continue;
		group_index[og.name] = j.groups.size();
		j.groups.push_back(og);
	}

	JsonPtr results = root->get("output_results");
	for (size_t i = 0; i < results->size(); i++)
	{
		JsonPtr r = results->at(i);

		OutputResult res;
		res.group_name  = r->get("group_name")->asString();
		res.name        = r->get("name")->asString();
		res.type        = r->get("type")->asString();
		res.passthrough = r->get("passthrough")->asBool();

		// An iterative job lists one metafile per round, with a parallel
		// `versions` array; the final round is the one with the highest version.
		// Falling back to the last entry keeps older jobs working, which wrote
		// metafiles without versions.
		const std::vector<std::string> metafiles = r->get("metafiles")->asStringVector();
		if (metafiles.empty()) continue;

		JsonPtr versions = r->get("versions");
		size_t pick = metafiles.size() - 1;
		if (versions->isArray() && versions->size() == metafiles.size())
		{
			long best = versions->at(0)->asLong();
			pick = 0;
			for (size_t v = 1; v < versions->size(); v++)
			{
				const long cur = versions->at(v)->asLong();
				if (cur >= best) { best = cur; pick = v; }
			}
		}
		res.metafile = metafiles[pick];

		// num_items is per-version too when the job is iterative
		JsonPtr ni = r->get("num_items");
		if (ni->isArray()) res.num_items = ni->at(pick < ni->size() ? pick : 0)->asLong();
		else               res.num_items = ni->asLong();

		std::map<std::string, size_t>::const_iterator gi = group_index.find(res.group_name);
		if (gi != group_index.end()) j.groups[gi->second].results.push_back(res);
		else
		{
			// A result naming a group that was not declared: keep it rather than
			// dropping data, in a group synthesised from what we know.
			OutputGroup og;
			og.name = res.group_name;
			group_index[og.name] = j.groups.size();
			j.groups.push_back(og);
			j.groups.back().results.push_back(res);
		}
	}

	jobs_[j.uid] = j;
}

Project Project::read(const std::string& dir, int verb)
{
	Project p;
	p.dir_ = dir;
	while (p.dir_.size() > 1 && p.dir_[p.dir_.size() - 1] == '/')
		p.dir_.erase(p.dir_.size() - 1);

	DIR* d = opendir(p.dir_.c_str());
	if (d == NULL)
		throw std::runtime_error("cannot open CryoSPARC project directory: " + p.dir_);

	std::vector<std::string> uids;
	struct dirent* ent;
	while ((ent = readdir(d)) != NULL)
	{
		const std::string name = ent->d_name;
		if (isJobDirName(name)) uids.push_back(name);
	}
	closedir(d);

	std::sort(uids.begin(), uids.end());
	for (size_t i = 0; i < uids.size(); i++)
		p.addJobFromFile(p.dir_ + "/" + uids[i], uids[i], verb);

	return p;
}

std::vector<std::string> Project::topologicalOrder() const
{
	// Kahn's algorithm over completed jobs, with ties broken by job number so
	// that the result is stable and reads in the order a user created the jobs.
	std::map<std::string, int> indeg;
	std::map<std::string, std::vector<std::string> > children;

	for (std::map<std::string, Job>::const_iterator it = jobs_.begin(); it != jobs_.end(); ++it)
	{
		if (!it->second.isCompleted()) continue;
		if (indeg.find(it->first) == indeg.end()) indeg[it->first] = 0;

		for (size_t i = 0; i < it->second.parents.size(); i++)
		{
			const std::string& par = it->second.parents[i];
			std::map<std::string, Job>::const_iterator pit = jobs_.find(par);
			// Only completed parents constrain the order; a parent that was
			// deleted or never finished cannot be imported anyway.
			if (pit == jobs_.end() || !pit->second.isCompleted()) continue;
			children[par].push_back(it->first);
			indeg[it->first]++;
		}
	}

	std::vector<std::string> ready;
	for (std::map<std::string, int>::const_iterator it = indeg.begin(); it != indeg.end(); ++it)
		if (it->second == 0) ready.push_back(it->first);

	struct ByNumber {
		bool operator()(const std::string& a, const std::string& b) const
		{ return jobNumber(a) < jobNumber(b); }
	};
	std::sort(ready.begin(), ready.end(), ByNumber());

	std::vector<std::string> out;
	while (!ready.empty())
	{
		const std::string cur = ready.front();
		ready.erase(ready.begin());
		out.push_back(cur);

		std::vector<std::string>& kids = children[cur];
		std::sort(kids.begin(), kids.end(), ByNumber());
		for (size_t i = 0; i < kids.size(); i++)
		{
			if (--indeg[kids[i]] == 0)
			{
				ready.push_back(kids[i]);
				std::sort(ready.begin(), ready.end(), ByNumber());
			}
		}
	}

	// A cycle would leave jobs unemitted. CryoSPARC cannot produce one, but
	// rather than silently dropping work, append whatever is left in job order.
	if (out.size() != indeg.size())
	{
		std::set<std::string> emitted(out.begin(), out.end());
		std::vector<std::string> rest;
		for (std::map<std::string, int>::const_iterator it = indeg.begin(); it != indeg.end(); ++it)
			if (!emitted.count(it->first)) rest.push_back(it->first);
		std::sort(rest.begin(), rest.end(), ByNumber());
		out.insert(out.end(), rest.begin(), rest.end());
	}

	return out;
}

std::vector<std::string> Project::ancestryOf(const std::string& uid) const
{
	std::set<std::string> want;
	std::vector<std::string> stack;
	stack.push_back(uid);

	while (!stack.empty())
	{
		const std::string cur = stack.back();
		stack.pop_back();
		if (!want.insert(cur).second) continue;

		const Job* j = job(cur);
		if (j == NULL) continue;
		for (size_t i = 0; i < j->parents.size(); i++)
			stack.push_back(j->parents[i]);
	}

	// Return them in dependency order, restricted to what is importable
	std::vector<std::string> topo = topologicalOrder();
	std::vector<std::string> out;
	for (size_t i = 0; i < topo.size(); i++)
		if (want.count(topo[i])) out.push_back(topo[i]);
	return out;
}

Project::Acquisition Project::acquisition() const
{
	Acquisition acq;

	// Prefer an import_movies job (it carries the dose), then import_micrographs.
	const char* wanted[] = {"import_movies", "import_micrographs"};
	for (int pass = 0; pass < 2 && !acq.found; pass++)
	{
		std::vector<std::string> uids;
		for (std::map<std::string, Job>::const_iterator it = jobs_.begin(); it != jobs_.end(); ++it)
			if (it->second.type == wanted[pass]) uids.push_back(it->first);

		struct ByNumber {
			bool operator()(const std::string& a, const std::string& b) const
			{ return jobNumber(a) < jobNumber(b); }
		};
		std::sort(uids.begin(), uids.end(), ByNumber());

		for (size_t i = 0; i < uids.size(); i++)
		{
			const std::string fn = dir_ + "/" + uids[i] + "/job.json";
			JsonPtr root;
			try { root = jsonParseFile(fn); }
			catch (const std::exception&) { continue; }

			JsonPtr p = root->get("params_spec");
			if (!p->isObject()) continue;

			// Each parameter is {"value": ...}
			const double kv   = p->get("accel_kv")->get("value")->asDouble(0.);
			const double ps   = p->get("psize_A")->get("value")->asDouble(0.);
			const double cs   = p->get("cs_mm")->get("value")->asDouble(0.);
			const double dose = p->get("total_dose_e_per_A2")->get("value")->asDouble(0.);
			const std::string gain = p->get("gainref_path")->get("value")->asString();

			if (kv > 0.)   acq.voltage = kv;
			if (ps > 0.)   acq.pixel_size_A = ps;
			if (cs > 0.)   acq.spherical_aberration = cs;
			if (dose > 0.) acq.total_dose_e_per_A2 = dose;
			if (!gain.empty()) acq.gainref_path = gain;

			if (acq.voltage > 0. && acq.pixel_size_A > 0.)
			{
				acq.found = true;
				break;
			}
		}
	}

	return acq;
}

} // namespace cryosparc
