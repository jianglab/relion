/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/slurm_script.h"
#include "src/parallel_hint.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace slurmscript
{

namespace
{

const long FALLBACK_MB_PER_CORE = 4096;   // when Slurm's node list is not available

std::string trim(const std::string &s)
{
	const size_t a = s.find_first_not_of(" \t\r\n");
	if (a == std::string::npos) return "";
	return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}

bool safeName(const std::string &s)
{
	return !s.empty() && std::all_of(s.begin(), s.end(),
		[](char c) { return std::isalnum((unsigned char)c) || c == '_' || c == '-' || c == '.' || c == ','; });
}

std::string where(const Request &r)
{
	return trim(r.partition).empty() ? "the cluster" : "queue " + trim(r.partition);
}

// Memory the job asks for in MB, or -1 if it is not a number of GB
double givenMB(const Request &r, bool &is_auto)
{
	const std::string mem = trim(r.mem);
	is_auto = mem.empty() || mem == "auto";
	if (is_auto) return 0;
	char *end = NULL;
	const double gb = strtod(mem.c_str(), &end);
	if (end == mem.c_str() || gb <= 0 || trim(end) != "") return -1;
	return gb * 1024;
}

}

std::vector<NodeType> parseSinfo(const std::string &sinfo_output)
{
	std::vector<NodeType> nodes;
	std::istringstream in(sinfo_output);
	std::string line;
	while (std::getline(in, line))
	{
		std::istringstream ls(line);
		std::string count, cpus, mem, gres;
		if (!(ls >> count >> cpus >> mem >> gres)) continue;
		NodeType n;
		n.count = std::atoi(count.c_str());
		n.cpus = std::atoi(cpus.c_str());   // "20+" (mixed) reads as the smallest
		n.mem_mb = std::atol(mem.c_str());
		n.gpus = parhint::gpusInGres(gres);
		if (n.count > 0 && n.cpus > 0 && n.mem_mb > 0)
			nodes.push_back(n);
	}
	return nodes;
}

std::vector<NodeType> partitionNodes(const std::string &partition)
{
	std::string cmd = "timeout 10 sinfo -h -o '%D %c %m %G'";
	if (!partition.empty())
	{
		if (!safeName(partition)) return {};
		cmd += " -p '" + partition + "'";
	}
	cmd += " 2>/dev/null";
	std::string out;
	if (FILE *p = popen(cmd.c_str(), "r"))
	{
		char buf[512];
		while (fgets(buf, sizeof buf, p)) out += buf;
		pclose(p);
	}
	return parseSinfo(out);
}

Layout plan(const Request &r, const std::vector<NodeType> &nodes)
{
	Layout l;
	const int mpi = std::max(1, r.mpi), threads = std::max(1, r.threads);
	bool is_auto;
	const double given = givenMB(r, is_auto);
	if (given < 0)
	{
		l.error = "Memory must be a number of GB, or auto (it was \"" + trim(r.mem) + "\").";
		return l;
	}
	l.nodes_known = !nodes.empty();

	// Round down, so the job stays within its share: whole GB for the job, 100 MB per core
	auto done = [&l]() -> Layout &
	{
		if (l.mem_per_cpu)
			l.mem_mb = std::max(100L, l.mem_mb / 100 * 100);
		else
			l.mem_mb = std::max(1024L, l.mem_mb / 1024 * 1024);
		l.ok = true;
		return l;
	};

	// Slurm places every process on whichever nodes have room; memory per core
	auto spread = [&]()
	{
		l.one_node = false;
		l.tasks = mpi;
		l.mem_per_cpu = true;
		long capacity = 0, per_core = 0;
		for (const NodeType &n : nodes)
		{
			if (n.cpus < threads) continue;
			capacity += (long)n.count * (n.cpus / threads) * threads;
			const long mb = n.mem_mb / n.cpus;
			per_core = (per_core == 0) ? mb : std::min(per_core, mb);
		}
		if (l.nodes_known && capacity < (long)mpi * threads)
		{
			l.error = "Not enough CPU cores in " + where(r) + " for " + std::to_string(mpi) + " process" + (mpi == 1 ? "" : "es") +
			          " of " + std::to_string(threads) + " threads. Ask for fewer CPUs, or use another queue.";
			return l;
		}
		if (is_auto)
			l.mem_mb = l.nodes_known ? per_core : FALLBACK_MB_PER_CORE;
		else
			l.mem_mb = std::max(1L, (long)(given / ((double)mpi * threads)));
		return done();
	};

	// CPU-only programs whose processes work on their own (MotionCorr, CtfFind, Extract, ...):
	// one node or several makes no difference to them
	if (r.gpus == 0 && !r.leader_idle)
		return spread();

	// GPU jobs and refinements on one node: the leader of a refinement runs on its workers' cores
	l.one_node = true;
	l.tasks = (r.leader_idle && mpi > 1) ? mpi - 1 : mpi;
	const int cores = l.tasks * threads;
	if (!l.nodes_known)
	{
		l.mem_mb = is_auto ? FALLBACK_MB_PER_CORE * cores : (long)given;
		return done();
	}
	long best = 0;
	for (const NodeType &n : nodes)
	{
		if (cores > n.cpus || r.gpus > n.gpus) continue;
		double share = (double)cores / n.cpus;
		if (r.gpus > 0) share = std::max(share, (double)r.gpus / n.gpus);
		const long mb = (long)(n.mem_mb * std::min(1.0, share));
		best = (best == 0) ? mb : std::min(best, mb);
	}
	if (best > 0)
	{
		l.mem_mb = is_auto ? best : (long)given;
		return done();
	}
	if (r.gpus > 0)
	{
		l.error = "No node of " + where(r) + " has " + std::to_string(cores) + " CPU cores and " + std::to_string(r.gpus) +
		          " GPUs for this job (RELION's own Slurm script runs GPU jobs on one node). Ask for fewer CPUs or GPUs, or use another queue.";
		return l;
	}
	// a CPU refinement too large for any node: spread, the leader with a task of its own
	return spread();
}

std::string preamble()
{
	const char *e = getenv("RELION_SLURM_PREAMBLE");
	if (e == NULL || *e == '\0') return "";
	std::ifstream f(e);
	if (f)
	{
		std::stringstream ss;
		ss << f.rdbuf();
		return ss.str();
	}
	return e;
}

std::string templateText(const Request &r, const Layout &l, const std::string &pre)
{
	std::ostringstream t;
	t << "#!/bin/bash\n"
	  << "# Written by RELION (built-in Slurm script: no template was given)\n"
	  << "#SBATCH --job-name=XXXnameXXX\n";
	if (!trim(r.partition).empty())
		t << "#SBATCH --partition=XXXqueueXXX\n";
	t << "#SBATCH --output=XXXoutfileXXX\n"
	  << "#SBATCH --error=XXXerrfileXXX\n";
	if (l.one_node)
		t << "#SBATCH --nodes=1\n";
	t << "#SBATCH --ntasks=" << l.tasks << "\n";
	if (r.launch)
		t << "#SBATCH --cpus-per-task=" << std::max(1, r.threads) << "\n";
	else
		t << "#SBATCH --cpus-per-task=XXXthreadsXXX\n";
	if (r.gpus > 0)
		t << "#SBATCH --gpus=" << r.gpus << "\n";
	if (l.mem_mb > 0)
		t << "#SBATCH " << (l.mem_per_cpu ? "--mem-per-cpu=" + std::to_string(l.mem_mb) + "M"
		                                  : "--mem=" + std::to_string(l.mem_mb / 1024) + "G") << "\n";
	const std::string time = trim(r.time);
	if (!time.empty() && time != "0")
		t << "#SBATCH --time=" << time << "\n";
	const std::string args = trim(r.args);
	if (!args.empty())
		t << "#SBATCH " << args << "\n";
	if (!pre.empty())
	{
		t << pre;
		if (pre.back() != '\n') t << "\n";
	}
	for (const std::string &e : r.exports)
		t << "export " << e << "\n";
	if (r.launch)
		t << "relion_launch XXXcommandXXX\n";
	else if (r.mpi > 1)
	{
		const char *mpirun = getenv("RELION_MPIRUN");
		t << ((mpirun && *mpirun) ? mpirun : "mpirun")
		  << (l.tasks < r.mpi ? " --oversubscribe" : "")   // the leader on its workers' cores
		  << " --bind-to none -n XXXmpinodesXXX XXXcommandXXX\n";
	}
	else
		t << "XXXcommandXXX\n";
	return t.str();
}

std::string memoryHint(const Request &r)
{
	const Layout l = plan(r, partitionNodes(trim(r.partition)));
	if (!l.ok) return l.error;
	std::ostringstream t;
	bool is_auto;
	givenMB(r, is_auto);
	const int threads = std::max(1, r.threads);
	t << (is_auto ? "auto: " : "");
	if (l.mem_per_cpu)
		t << l.mem_mb << " MB per CPU core, " << l.mem_mb * l.tasks * threads / 1024 << " GB in all";
	else
		t << l.mem_mb / 1024 << " GB";
	if (is_auto && l.nodes_known)
		t << (l.one_node ? ", the job's share of the smallest node of\n" + where(r) + " that fits it (its share of the node's cores or GPUs,\nwhichever is larger)."
		                 : ", from the node type of " + where(r) + "\nwith the least memory per core.");
	else if (is_auto)
		t << " (4 GB per CPU core: Slurm's list of nodes is not available here).";
	else
		t << ".";
	t << "\nSlurm reserves " << l.tasks << " task" << (l.tasks == 1 ? "" : "s") << " x " << threads << " CPU cores"
	  << (l.one_node ? " on one node" : " on as many nodes as needed")
	  << (l.tasks < r.mpi ? "; the MPI leader runs on the same cores." : ".");
	return t.str();
}

bool build(const Request &r, std::string &text, std::string &error_message)
{
	const Layout l = plan(r, partitionNodes(trim(r.partition)));
	if (!l.ok)
	{
		error_message = l.error;
		return false;
	}
	text = templateText(r, l, preamble());
	return true;
}

}
