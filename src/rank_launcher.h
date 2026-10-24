/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef RANK_LAUNCHER_H_
#define RANK_LAUNCHER_H_

/*
 * relion_launch: start a GPU refinement with the MPI processes and threads chosen on the
 * computer it landed on (see documentation/ranks_per_gpu.md).
 *
 * A queue job asks for CPUs, memory and GPUs, but which GPU model it gets is only known
 * when it starts. relion_launch looks at what the job got (GPUs and their free memory, the
 * cores and memory it may use), takes the kind of work and the particles from the command,
 * and asks parhint::adviseRanks() for the worker processes per GPU and the threads; then it
 * starts mpirun (or the single-process program) itself.
 *
 * Learning: a run that relion_launch started records, at its end, its E-step seconds per
 * particle (learnFromRun, called with the resource report). Those times compare only between
 * runs of the same job, so the record is kept per GPU model and job (keyOf: kind of work plus
 * a hash of the data and the size of the search) in the tuning cache (tunable "ranks_per_gpu").
 * With cores to spare, the next run of that job starts one more worker per GPU, and goes back
 * (and stays) if that was not faster. Other jobs start from the measured rule.
 *
 * NVIDIA MPS (started by relion_refine_mpi, src/mps_auto.h) is chosen the same way, per job and
 * workers per GPU (tunable "mps", see chooseMps).
 *
 * Overrides: RELION_RANKS_PER_GPU (workers per GPU); RELION_AUTO_MPS (on/off, nothing learned);
 * RELION_MPIRUN (the MPI launcher).
 */

#include <string>
#include <vector>
#include "src/parallel_hint.h"
#include "src/resource_report.h"

namespace rlaunch
{

/// What this job may use on this computer.
struct Node
{
	int gpus = 0;
	std::string gpu_name;    // model of the first GPU
	double gpu_bytes = 0;    // free memory of the GPU with the least free memory
	int cores = 0;
	double host_bytes = 0;   // memory the job may use
};

/// GPUs from `nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv,noheader,nounits`,
/// restricted to CUDA_VISIBLE_DEVICES (indices or UUIDs; NULL: all).
void parseGpus(const std::string &smi, const char *visible, Node &node);
Node probeNode();

/// The GPUs named by a --gpu value such as "0:1" or "0,1:2" as a CUDA_VISIBLE_DEVICES list (numbers
/// refer to `visible`, the current CUDA_VISIBLE_DEVICES, when set); "" when the value names no device
/// or is not a plain list of numbers. The launcher spreads its workers over exactly these GPUs.
std::string visibleForGpuIds(const std::string &ids, const char *visible);

/// Kind of work of a relion_refine command line; `vdam` is set for --grad (one process only).
/// For --continue, the kind is read from the optimiser STAR file.
parhint::Work workOfArgs(const std::vector<std::string> &args, bool &vdam);
const char *workName(parhint::Work w);

/// What earlier runs of one job (same data and search, see keyOf) on one GPU model found.
struct Record
{
	int next = 0;            // workers per GPU for the next run (0: none recorded)
	int prev = 0;            // the count before the last step up or down (0: none)
	double prev_time = 0;    // E-step seconds per particle at `prev`
	bool settled = false;    // the neighbouring counts were not faster: keep `next`
	double gpu_gb = 0;       // GPU memory per worker in the last run: the card's peak / its workers (0: unknown)
	bool from_other_jobs = false; // read from the record of the kind of work (not stored)
};
std::string encode(const Record &r);
bool decode(const std::string &s, Record &r);

/// The key of a job in the tuning cache: the kind of work and a hash of what makes E-step times of
/// two runs comparable (the input data and the size of the search).
std::string keyOf(const std::vector<std::string> &args);
/// The record shared by all jobs of one kind of work with about as many CPU cores per GPU (1-2, 3-4,
/// 5-8, 9-16, more): where a job without a record of its own starts.
std::string kindKeyOf(const std::vector<std::string> &args, int cores_per_gpu);

/// The record after a run with `used` workers per GPU whose E-step took `time` seconds per particle,
/// with `cores_busy` (0..1) of its cores busy and `threads` threads per worker. A count measured for
/// the first time tries a neighbour: one more worker when cores were to spare (below 75% busy) and
/// every worker has 2 threads or more, otherwise one fewer. A step that made the E-step at least 3%
/// faster is followed by another the same way; one that did not goes back, settled. GPU busy % is not
/// used: it reads 93-98% as soon as two processes share a GPU without MPS, faster or not.
Record next(const Record &r, int used, double time, double cores_busy, int threads);

bool load(const std::string &gpu_name, const std::string &key, Record &r);
bool store(const std::string &gpu_name, const std::string &key, const Record &r);

/// E-step seconds per particle of one job at one worker count with and without MPS (0: not measured).
struct MpsRecord
{
	double on = 0, off = 0;
};
std::string encodeMps(const MpsRecord &m);
bool decodeMps(const std::string &s, MpsRecord &m);
bool loadMps(const std::string &gpu_name, const std::string &key, MpsRecord &m);
bool storeMps(const std::string &gpu_name, const std::string &key, const MpsRecord &m);

/// Whether the next run uses MPS, and why. Measured: with 8 single-thread workers on one GPU MPS was
/// 7x faster (small launches from many processes); with 2 workers of 3 threads, each filling the
/// GPU with long coarse kernels, it was 6% slower. So the rule starts it at 4 or more workers per GPU.
/// Once the workers per GPU are settled, the setting not yet measured is tried once; when both are
/// measured, the faster is kept. `kind` (the job's kind of work, both measured) stands in for a job
/// without a record of its own.
bool chooseMps(const MpsRecord &job, const MpsRecord &kind, int workers_per_gpu, bool workers_settled, std::string &why);

/// Called at the end of a run that relion_launch started (it exports RELION_LAUNCH_*), with the
/// resource report of the run (the MPI leader first when `mpi`), its particles and E-steps: records
/// the run and returns one line for the output ("" when not started by relion_launch, shorter than
/// a minute, or on several computers).
std::string learnFromRun(const std::vector<resrep::ProcessStats> &procs, bool mpi, long particles, int steps);

struct Command
{
	std::vector<std::string> argv;
	std::vector<std::string> env;    // NAME=VALUE for the run
	parhint::RankAdvice advice;
	std::string report;
};

/// The command to run for `cmd` (relion_refine or relion_refine_mpi and its arguments) on `node`.
/// false (and why) when the command is not a GPU refinement: it is then run as given.
bool build(const std::vector<std::string> &cmd, const Node &node, long particles, const Record &learned,
		Command &out, std::string &why_not);

/// Adds to `c` (built for the run) the MPS choice: RELION_AUTO_MPS for the run and the keys under which
/// learnFromRun records its time. A RELION_AUTO_MPS already set is obeyed, and nothing is learned.
void applyMps(Command &c, const Node &node, const std::vector<std::string> &cmd, const Record &learned);

} // namespace rlaunch

#endif
