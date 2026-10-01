/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/import_project.h"
#include "src/metadata_table.h"

#include <dirent.h>
#include <limits.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <set>

namespace relion_import {

namespace {

bool pathExists(const std::string &p)
{
	struct stat st;
	return ::stat(p.c_str(), &st) == 0;
}

bool isDirectory(const std::string &p)
{
	struct stat st;
	return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool isJobName(const std::string &name)
{
	if (name.size() <= 3 || name.compare(0, 3, "job") != 0) return false;
	for (size_t i = 3; i < name.size(); i++)
		if (!std::isdigit((unsigned char)name[i])) return false;
	return true;
}

std::string stripSlashes(std::string s)
{
	while (s.size() > 1 && s[s.size() - 1] == '/') s.erase(s.size() - 1);
	return s;
}

std::string parentOf(const std::string &p)
{
	const size_t k = p.rfind('/');
	if (k == std::string::npos) return ".";
	if (k == 0) return "/";
	return p.substr(0, k);
}

std::string baseOf(const std::string &p)
{
	const size_t k = p.rfind('/');
	return (k == std::string::npos) ? p : p.substr(k + 1);
}

std::string resolve(const std::string &p)
{
	char buf[PATH_MAX];
	if (::realpath(p.c_str(), buf) != NULL) return stripSlashes(buf);
	return stripSlashes(p);
}

std::vector<std::string> jobFoldersIn(const std::string &dir)
{
	std::vector<std::string> names;
	DIR *d = opendir(dir.c_str());
	if (d == NULL) return names;
	struct dirent *e;
	while ((e = readdir(d)) != NULL)
	{
		const std::string n = e->d_name;
		if (isJobName(n) && isDirectory(dir + "/" + n)) names.push_back(n);
	}
	closedir(d);
	std::sort(names.begin(), names.end());
	return names;
}

std::string withSlash(const std::string &s)
{
	return (!s.empty() && s[s.size() - 1] == '/') ? s : s + "/";
}

bool makeDir(const std::string &p)
{
	return isDirectory(p) || ::mkdir(p.c_str(), 0755) == 0;
}

struct SourcePipeline
{
	MetaDataTable general, nodes, processes, inputs, outputs;
};

} // namespace

bool isProject(const std::string &dir)
{
	return pathExists(dir + "/.gui_projectdir") || pathExists(dir + "/default_pipeline.star");
}

Selection classify(const std::string &dir)
{
	Selection sel;
	if (dir.empty() || !isDirectory(dir)) return sel;

	const std::string abs = resolve(dir);
	sel.chosen = abs;

	const std::string name = baseOf(abs);
	const std::string parent = parentOf(abs);

	// A job folder holds a copy of default_pipeline.star, so it must be
	// recognised before the project test, which would otherwise accept it.
	if (isJobName(name))
	{
		if (isProject(parent))                       // flat layout: project/jobNNN
		{
			sel.scope = SINGLE_JOB;
			sel.project = parent;
			sel.jobs.push_back(name + "/");
			sel.what = "job " + name + " and the jobs it depends on";
			return sel;
		}
		if (isProject(parentOf(parent)))             // project/Type/jobNNN
		{
			sel.scope = SINGLE_JOB;
			sel.project = parentOf(parent);
			sel.jobs.push_back(baseOf(parent) + "/" + name + "/");
			sel.what = "job " + baseOf(parent) + "/" + name + " and the jobs it depends on";
			return sel;
		}
	}

	if (isProject(abs))
	{
		sel.scope = WHOLE_PROJECT;
		sel.project = abs;
		sel.what = "the whole project";
		return sel;
	}

	if (!name.empty() && name[0] != '.' && name != "Trash" && isProject(parent))
	{
		const std::vector<std::string> found = jobFoldersIn(abs);
		if (!found.empty())
		{
			sel.scope = JOB_TYPE_FOLDER;
			sel.project = parent;
			for (size_t i = 0; i < found.size(); i++)
				sel.jobs.push_back(name + "/" + found[i] + "/");
			sel.what = "all jobs in " + name + " and the jobs they depend on";
		}
	}
	return sel;
}

Result importJobs(const Selection &sel, const std::string &destDir)
{
	Result res;
	if (sel.scope != SINGLE_JOB && sel.scope != JOB_TYPE_FOLDER)
	{
		res.error = "Only a job or a job type folder can be imported by importJobs().";
		return res;
	}
	if (!isDirectory(destDir))
	{
		res.error = "The destination folder does not exist: " + destDir;
		return res;
	}

	const std::string pipelineFile = sel.project + "/default_pipeline.star";
	std::ifstream in(pipelineFile.c_str(), std::ios_base::in);
	if (in.fail())
	{
		res.error = "Cannot read " + pipelineFile + ", which says how the jobs depend on each other.";
		return res;
	}

	SourcePipeline sp;
	if (!sp.general.readStar(in, "pipeline_general"))
	{
		res.error = "No pipeline_general table in " + pipelineFile;
		return res;
	}
	sp.nodes.readStar(in, "pipeline_nodes");
	sp.processes.readStar(in, "pipeline_processes");
	sp.inputs.readStar(in, "pipeline_input_edges");
	sp.outputs.readStar(in, "pipeline_output_edges");
	in.close();

	// Which jobs exist, which jobs read which nodes, which job wrote each node
	std::set<std::string> known;
	for (size_t i = 0; i < sp.processes.numberOfObjects(); i++)
	{
		std::string n;
		if (sp.processes.getValue(EMDL_PIPELINE_PROCESS_NAME, n, i)) known.insert(withSlash(n));
	}

	std::map<std::string, std::vector<std::string> > readsOf;   // job -> nodes it reads
	for (size_t i = 0; i < sp.inputs.numberOfObjects(); i++)
	{
		std::string proc, node;
		if (sp.inputs.getValue(EMDL_PIPELINE_EDGE_PROCESS, proc, i) &&
		    sp.inputs.getValue(EMDL_PIPELINE_EDGE_FROM, node, i))
			readsOf[withSlash(proc)].push_back(node);
	}
	std::map<std::string, std::string> writerOf;                 // node -> job that wrote it
	for (size_t i = 0; i < sp.outputs.numberOfObjects(); i++)
	{
		std::string proc, node;
		if (sp.outputs.getValue(EMDL_PIPELINE_EDGE_PROCESS, proc, i) &&
		    sp.outputs.getValue(EMDL_PIPELINE_EDGE_TO, node, i))
			writerOf[node] = withSlash(proc);
	}

	// Asked-for jobs and, repeatedly, the jobs that wrote what they read
	std::set<std::string> keep;
	std::vector<std::string> todo;
	for (size_t i = 0; i < sel.jobs.size(); i++)
	{
		const std::string j = withSlash(sel.jobs[i]);
		if (known.count(j) == 0) { res.not_in_pipeline.push_back(j); continue; }
		if (keep.insert(j).second) todo.push_back(j);
	}
	while (!todo.empty())
	{
		const std::string j = todo.back();
		todo.pop_back();
		const std::vector<std::string> &nodes = readsOf[j];
		for (size_t k = 0; k < nodes.size(); k++)
		{
			std::map<std::string, std::string>::const_iterator w = writerOf.find(nodes[k]);
			if (w != writerOf.end() && known.count(w->second) && keep.insert(w->second).second)
				todo.push_back(w->second);
		}
	}

	if (keep.empty())
	{
		res.error = "None of the selected jobs is listed in " + pipelineFile + ".";
		return res;
	}

	// The nodes any kept job reads or wrote
	std::set<std::string> keepNodes;
	for (size_t i = 0; i < sp.inputs.numberOfObjects(); i++)
	{
		std::string proc, node;
		if (sp.inputs.getValue(EMDL_PIPELINE_EDGE_PROCESS, proc, i) &&
		    sp.inputs.getValue(EMDL_PIPELINE_EDGE_FROM, node, i) && keep.count(withSlash(proc)))
			keepNodes.insert(node);
	}
	for (size_t i = 0; i < sp.outputs.numberOfObjects(); i++)
	{
		std::string proc, node;
		if (sp.outputs.getValue(EMDL_PIPELINE_EDGE_PROCESS, proc, i) &&
		    sp.outputs.getValue(EMDL_PIPELINE_EDGE_TO, node, i) && keep.count(withSlash(proc)))
			keepNodes.insert(node);
	}

	// Write the reduced pipeline, copying each kept row unchanged. The job
	// counter stays as it was, so new jobs never reuse a number.
	MetaDataTable outNodes, outProcs, outIn, outOut;
	outNodes.setName("pipeline_nodes");
	outProcs.setName("pipeline_processes");
	outIn.setName("pipeline_input_edges");
	outOut.setName("pipeline_output_edges");
	outNodes.setVersion(sp.nodes.getVersion());
	outProcs.setVersion(sp.processes.getVersion());
	outIn.setVersion(sp.inputs.getVersion());
	outOut.setVersion(sp.outputs.getVersion());

	for (size_t i = 0; i < sp.nodes.numberOfObjects(); i++)
	{
		std::string n;
		if (sp.nodes.getValue(EMDL_PIPELINE_NODE_NAME, n, i) && keepNodes.count(n))
			outNodes.addObject(sp.nodes.getObject(i));
	}
	for (size_t i = 0; i < sp.processes.numberOfObjects(); i++)
	{
		std::string n;
		if (sp.processes.getValue(EMDL_PIPELINE_PROCESS_NAME, n, i) && keep.count(withSlash(n)))
			outProcs.addObject(sp.processes.getObject(i));
	}
	for (size_t i = 0; i < sp.inputs.numberOfObjects(); i++)
	{
		std::string n;
		if (sp.inputs.getValue(EMDL_PIPELINE_EDGE_PROCESS, n, i) && keep.count(withSlash(n)))
			outIn.addObject(sp.inputs.getObject(i));
	}
	for (size_t i = 0; i < sp.outputs.numberOfObjects(); i++)
	{
		std::string n;
		if (sp.outputs.getValue(EMDL_PIPELINE_EDGE_PROCESS, n, i) && keep.count(withSlash(n)))
			outOut.addObject(sp.outputs.getObject(i));
	}

	const std::string destPipeline = destDir + "/default_pipeline.star";
	{
		std::ofstream fh(destPipeline.c_str(), std::ios::out);
		if (fh.fail())
		{
			res.error = "Cannot write " + destPipeline;
			return res;
		}
		sp.general.write(fh);
		outNodes.write(fh);
		outProcs.write(fh);
		outIn.write(fh);
		outOut.write(fh);
		if (fh.fail())
		{
			res.error = "Writing " + destPipeline + " failed.";
			return res;
		}
	}

	// Link the job folders
	for (std::set<std::string>::const_iterator it = keep.begin(); it != keep.end(); ++it)
	{
		const std::string job = stripSlashes(*it);          // e.g. Class2D/job008
		res.imported.push_back(*it);
		const std::string source = sel.project + "/" + job;
		if (!isDirectory(source)) { res.missing++; continue; }

		const std::string link = destDir + "/" + job;
		const size_t slash = job.rfind('/');
		if (slash != std::string::npos && !makeDir(destDir + "/" + job.substr(0, slash)))
		{
			res.error = "Cannot create folder " + destDir + "/" + job.substr(0, slash);
			return res;
		}
		struct stat lst;
		if (::lstat(link.c_str(), &lst) == 0) { res.existing++; continue; }
		if (::symlink(source.c_str(), link.c_str()) != 0)
		{
			res.error = "Cannot link " + link + " to " + source;
			return res;
		}
		res.linked++;
	}

	res.ok = true;
	return res;
}

} // namespace relion_import
