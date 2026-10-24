/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

// relion_launch [--dry_run] relion_refine[_mpi] <arguments>: start a GPU refinement with the
// MPI processes and threads chosen for the computer it runs on (see src/rank_launcher.h).

#include "src/rank_launcher.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>

int main(int argc, char **argv)
{
	int first = 1;
	bool dry_run = false;
	if (first < argc && strcmp(argv[first], "--dry_run") == 0)
	{
		dry_run = true;
		first++;
	}
	if (first >= argc)
	{
		std::cerr << "Usage: relion_launch [--dry_run] relion_refine[_mpi] <arguments>\n"
		          << "  Runs a GPU refinement with the MPI processes and threads chosen for the GPUs, cores and memory\n"
		          << "  this job got (RELION_RANKS_PER_GPU sets the workers per GPU). Any other command is run as given.\n";
		return 1;
	}
	std::vector<std::string> cmd(argv + first, argv + argc);

	// GPUs named with --gpu (e.g. "Which GPUs to use" of a local run): restrict the job to them and let
	// the workers be spread over them, since the per-process mapping cannot fit a chosen process count
	for (size_t i = 0; i + 1 < cmd.size(); i++)
		if (cmd[i] == "--gpu")
		{
			const std::string vis = rlaunch::visibleForGpuIds(cmd[i + 1], getenv("CUDA_VISIBLE_DEVICES"));
			if (!vis.empty())
			{
				setenv("CUDA_VISIBLE_DEVICES", vis.c_str(), 1);
				cmd[i + 1] = "";
			}
		}

	rlaunch::Command c;
	std::string why_not;
	const rlaunch::Node node = rlaunch::probeNode();
	long particles = 0;
	bool continued = false;
	for (size_t i = 0; i + 1 < cmd.size(); i++)
	{
		if (cmd[i] == "--continue") continued = true;
		if (cmd[i] == "--i") particles = parhint::countParticlesInStar(cmd[i + 1]);
	}
	if (continued) particles = 0;
	rlaunch::Record learned;
	if (!rlaunch::load(node.gpu_name, rlaunch::keyOf(cmd), learned) &&
	    rlaunch::load(node.gpu_name, rlaunch::kindKeyOf(cmd, node.cores / std::max(1, node.gpus)), learned))
		learned.from_other_jobs = true;

	if (rlaunch::build(cmd, node, particles, learned, c, why_not))
	{
		rlaunch::applyMps(c, node, cmd, learned);
		std::cout << c.report << std::endl;
		for (const std::string &e : c.env)
			putenv(strdup(e.c_str()));
	}
	else
	{
		c.argv = cmd;
		// without mpirun relion_refine_mpi would be one process with the --j of a many-process split:
		// one process with a thread per core instead
		const std::string &prog = cmd[0];
		const size_t at = prog.rfind("relion_refine_mpi");
		if (at != std::string::npos && at + 17 == prog.size() && node.cores > 0)
		{
			c.argv[0] = prog.substr(0, at) + "relion_refine";
			bool has_j = false;
			for (size_t i = 1; i + 1 < c.argv.size(); i++)
				if (c.argv[i] == "--j") { c.argv[i + 1] = std::to_string(node.cores); has_j = true; }
			if (!has_j) { c.argv.push_back("--j"); c.argv.push_back(std::to_string(node.cores)); }
			std::cout << " relion_launch: " << why_not << "; running one process with " << node.cores << " threads." << std::endl;
		}
		else
			std::cout << " relion_launch: running the command as given (" << why_not << ")." << std::endl;
	}
	if (dry_run)
	{
		for (size_t i = 0; i < c.argv.size(); i++)
		{
			const std::string &a = c.argv[i];
			const bool quote = a.empty() || a.find_first_of(" \t\"'$`\\;&|<>()*?") != std::string::npos;
			std::string q;
			for (char ch : a) q += (ch == '\'') ? std::string("'\\''") : std::string(1, ch);
			std::cout << (i ? " " : "") << (quote ? "'" + q + "'" : a);
		}
		std::cout << std::endl;
		return 0;
	}
	std::cout.flush();
	std::vector<char *> a;
	for (std::string &s : c.argv) a.push_back(&s[0]);
	a.push_back(NULL);
	execvp(a[0], a.data());
	perror(("relion_launch: cannot start " + c.argv[0]).c_str());
	return 127;
}
