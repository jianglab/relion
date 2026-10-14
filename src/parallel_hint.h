/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef PARALLEL_HINT_H_
#define PARALLEL_HINT_H_

/*
 * Suggested numbers of MPI processes and threads for a refinement job, shown by
 * the GUI when the mouse rests on those fields.
 *
 * The suggestion follows from the number of GPUs the job asks for and the CPU
 * cores per GPU of the machines it will run on:
 *  - EM classification (Class2D without VDAM, Class3D): one process with a
 *    thread per core feeds one GPU as well as several processes do, so one
 *    process for one GPU; for several GPUs a leader (which only coordinates)
 *    plus one worker per GPU, each with the cores of its GPU.
 *  - VDAM (InitialModel, Class2D with VDAM): MPI is not supported; one process
 *    whose threads are spread over all GPUs.
 *  - Auto-refine (Refine3D, MultiBody): an odd number of processes, at least 3
 *    (a leader and the same number of workers for each half-set); the cores of
 *    all GPUs are shared among the workers.
 *
 * With "Choose MPI and threads automatically?" the user gives only the number of
 * CPUs, and the job splits them the same way (split(), autoSplit()).
 *
 * The GPU count is read from the queue option whose label mentions GPUs (a
 * RELION_QSUB_EXTRAi such as "Number of GPUs"), else from "Which GPUs to use",
 * else from the GPUs of this computer. Cores per GPU: RELION_CPUS_PER_GPU if
 * set; otherwise at most THREADS_PER_GPU_CAP, and no more than the nodes have: for
 * queue jobs the partition's nodes as Slurm describes them (sinfo), for local jobs
 * this computer.
 */

#include <functional>
#include <string>

namespace parhint
{

enum class Kind { Other, Classify, Vdam, AutoRefine };  // (not "None": X11 defines that)

// Threads that keep one GPU busy: on an RTX 2080 Ti the E-step time is the same with 6 to 16
// threads per GPU, because the threads of a process issue their GPU commands one at a time.
// Suggestions take at most this many cores per GPU, so a job leaves the rest of a shared node
// to other jobs. RELION_CPUS_PER_GPU overrides it.
const int THREADS_PER_GPU_CAP = 8;

// CPU-only jobs: threads per worker process when RELION splits the CPUs
const int THREADS_PER_CPU_WORKER = 8;

struct Suggestion
{
	bool valid = false;
	int mpi = 1;
	int threads = 1;
};

/// The suggestion for `gpus` GPUs (>= 1) with `cores_per_gpu` cores each (0: unknown,
/// then threads is 0 too).
Suggestion suggest(Kind kind, int gpus, int cores_per_gpu);

/// Kind of a GUI job: its type, and whether VDAM is chosen (Class2D).
Kind kindOf(int job_type, bool vdam);

/// What the GUI knows about a job right now. `value(key)` returns the current text of
/// a job option ("" if the job has none), `label(key)` its label.
struct JobView
{
	int job_type = -1;
	std::function<std::string(const std::string &)> value;
	std::function<std::string(const std::string &)> label;
};

/// Split `cpus` CPU cores into MPI processes and threads for a job of this kind with `gpus`
/// GPUs (0: a CPU-only job).
Suggestion split(Kind kind, int gpus, int cpus);

/// When the job chooses MPI processes and threads itself ("auto_parallel" is Yes), the split of
/// its "nr_cpus"; false otherwise.
bool autoSplit(const JobView &job, Suggestion &split);

enum class Field { Cpus, MpiOrThreads };

/// The tooltip text for the "Number of CPUs" field or the MPI and threads fields of a job
/// ("" for jobs without suggestions).
std::string tooltip(const JobView &job, Field field);

/// Number of distinct GPU ids in a "Which GPUs to use" string such as "0,1:2,3" (0 if empty).
int countGpuIds(const std::string &ids);

/// GPUs in a Slurm GRES string such as "gpu:a100:4(S:0-1)" (0 for "(null)").
int gpusInGres(const std::string &gres);

/// Cores per GPU from `sinfo -o "%c %G"` output: the smallest ratio over node types with
/// GPUs (0 if none has GPUs).
int coresPerGpuFromSinfo(const std::string &sinfo_output);

}

#endif /* PARALLEL_HINT_H_ */
