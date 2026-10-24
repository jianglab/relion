/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef SLURM_SCRIPT_H_
#define SLURM_SCRIPT_H_

/*
 * The built-in Slurm submission script, used for queue jobs when the GUI's
 * "Submission script template (optional)" is empty. RELION writes it from the job itself,
 * so a site needs no template.
 *
 * Layout, from the nodes of the queue (sinfo). Each MPI process is a Slurm task
 * with a CPU core per thread, on one node:
 *  - CPU-only programs whose processes work on their own (MotionCorr, CtfFind,
 *    Extract, ...): Slurm places the tasks on one node or several, as it finds
 *    room; memory is given per core.
 *  - GPU jobs, and refinements (relion_refine_mpi) that fit on one node: one
 *    node. The leader of a refinement only coordinates: it is started on top of
 *    the workers (mpirun --oversubscribe), so no cores are reserved for it.
 *  - A CPU refinement too large for any node is spread like the first case,
 *    its leader with a task of its own.
 *  - --gpus only for jobs that use GPUs.
 *  - Memory is always given (without it some sites hand a job a whole node's
 *    memory): a number of GB for the job, or "auto": on one node, the job's
 *    share of the memory of the smallest node of the queue that fits it, by its
 *    share of that node's cores or of its GPUs, whichever is larger; on several
 *    nodes, the memory per core of the node type with the least.
 *  - A job that the queue cannot hold is refused before submission.
 *
 * RELION_SLURM_PREAMBLE adds site lines (module loads, environment) before the
 * command: the name of a file whose contents are copied, or the lines
 * themselves. RELION_MPIRUN names the MPI launcher (default mpirun).
 *
 * The text has the same XXX...XXX placeholders as a template, so it is
 * completed the same way (RelionJob::saveJobSubmissionScript).
 */

#include <string>
#include <vector>

namespace slurmscript
{

struct Request
{
	std::string partition;
	int mpi = 1;               // processes, the leader included
	int threads = 1;           // per process
	bool leader_idle = false;  // the leader only coordinates (relion_refine_mpi)
	int gpus = 0;
	std::string mem = "auto";  // GB for the job, or "auto"
	std::string time;          // empty: the queue's default
	std::string args;          // more sbatch options, as typed
	std::vector<std::string> exports;  // NAME=VALUE, exported before the command
	// relion_launch chooses the processes and threads on the node the job lands on (rank_launcher.h):
	// one task with all `threads` cores, and the command is run through relion_launch
	bool launch = false;
};

/// A kind of node of the queue, from `sinfo -h -o "%D %c %m %G"`.
struct NodeType
{
	int count = 0;
	int cpus = 0;
	long mem_mb = 0;
	int gpus = 0;
};

std::vector<NodeType> parseSinfo(const std::string &sinfo_output);

/// The nodes of a Slurm partition ("" for all), asking sinfo; empty if it cannot be asked.
std::vector<NodeType> partitionNodes(const std::string &partition);

struct Layout
{
	bool ok = false;
	std::string error;
	bool one_node = true;
	int tasks = 1;
	long mem_mb = 0;           // for the job (one node) or per core (several)
	bool mem_per_cpu = false;
	bool nodes_known = false;  // false: sinfo gave nothing, so nothing was checked
};

/// Where the job runs and how much memory it asks for. `nodes` empty: unknown, so one
/// node and 4 GB per core for "auto".
Layout plan(const Request &r, const std::vector<NodeType> &nodes);

/// Template text with placeholders for the request and its layout.
std::string templateText(const Request &r, const Layout &l, const std::string &preamble);

/// The preamble from RELION_SLURM_PREAMBLE ("" if not set).
std::string preamble();

/// Tooltip for the memory field: the layout and memory, or why the queue cannot hold the job.
std::string memoryHint(const Request &r);

/// Plan the job on its queue and write the template text; false with a message when the
/// queue cannot hold it or the memory is not a number.
bool build(const Request &r, std::string &text, std::string &error_message);

}

#endif /* SLURM_SCRIPT_H_ */
