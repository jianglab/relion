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

// GPU memory one extra host thread costs (its workspaces and pool share): measured 0.34 GB on
// the tutorial Class3D case; 0.5 GB leaves margin. Only 60% of the free GPU memory is
// assumed usable by threads, the rest holds the maps and the pool's base.
const double GPU_BYTES_PER_THREAD = 0.5e9;
const double GPU_FREE_FRACTION_FOR_THREADS = 0.6;
// Fewer particles than this per thread and the fixed costs of an iteration dominate:
// a 300-particle Class2D case got slower with more than 2 threads per worker.
const long MIN_PARTICLES_PER_THREAD = 128;

/// Threads per process for a job started without --j. Three limits, the smallest wins:
///  1. cores: the cores each process may use (`cores` / `sharing`, where `sharing` is the
///     number of worker processes that share those cores);
///  2. GPU jobs (`gpus` > 0, the GPUs visible to the job; `per_gpu_cap` <= 0 switches this
///     limit off): at most `per_gpu_cap` per GPU the process feeds, because more threads do
///     not shorten the E-step (measured on an RTX 3090: no gain beyond 6 to 8 per process);
///  3. GPU memory (`free_gpu_bytes` > 0: free memory of all visible GPUs): the threads of
///     all `sharing` workers together must fit in GPU_FREE_FRACTION_FOR_THREADS of it at
///     GPU_BYTES_PER_THREAD each (a card of 8 GB ran out of memory with 8 threads in each of
///     two workers);
///  4. work (`particles_per_worker` > 0): at least MIN_PARTICLES_PER_THREAD per thread.
/// Always >= 1.
int autoThreads(int cores, int sharing, int gpus, int per_gpu_cap = THREADS_PER_GPU_CAP,
                double free_gpu_bytes = 0, long particles_per_worker = 0);

/// Number of particles in a particle STAR file (the data_particles block, or the unnamed block
/// of old files), counted without parsing the values; 0 if the file has none or cannot be read.
long countParticlesInStar(const std::string &fn);

/// The cap from RELION_AUTO_THREADS_CAP ("off" or 0: none; a number: threads per GPU);
/// THREADS_PER_GPU_CAP when unset or not a number.
int autoThreadsCapFromEnv();


/*
 * Ranks (MPI worker processes) per GPU.
 *
 * One process issues about 120 CUDA calls per particle, most of them to kernels of 10 to 20
 * microseconds, so it keeps a GPU only about a third busy, and more threads in the same
 * process do not help (the threads of a process submit their work one at a time). More worker
 * processes on the same GPU overlap their submissions. Measured on an RTX 3090, E-step time with
 * 12 worker threads in all, split as 1, 2, 3, 4, 6 or 12 workers: see the table in
 * documentation/ranks_per_gpu.md. The gain levels off at the workers per GPU given by
 * WORKERS_PER_GPU below, which depend on how much GPU work a particle needs.
 */

/// What a job asks of the GPU.
enum class Work
{
	Class2D,     // EM Class2D: tiny kernels, the host is the limit
	Class3D,     // Class3D (or any 3D classification) with alignment
	SkipAlign,   // 3D classification without alignment
	AutoRefine   // Refine3D, MultiBody: the workers are split in two half-sets
};

/// Workers per GPU at which the E-step stops getting faster, by kind of work (measured).
int workersPerGpuTarget(Work work);

/// GPU memory one worker process takes (bytes, measured with the lean pool, small boxes;
/// RELION_GPU_GB_PER_RANK overrides it).
double gpuBytesPerWorker(Work work);

struct RankQuery
{
	Work work = Work::Class3D;
	int gpus = 1;                // GPUs of the job (>= 1)
	int cores = 0;               // CPU cores for the job (0: unknown, no limit)
	double gpu_bytes = 0;        // memory of one GPU (0: unknown, no limit)
	double host_bytes = 0;       // memory of the computer for the job (0: unknown, no limit)
	long particles = 0;          // particles of the data set (0: unknown, no limit)
	int learned = 0;             // workers per GPU that earlier runs on this GPU model chose (0: none;
	                             // the limits of memory, cores and particles still apply)
	bool learned_settled = false; // false: earlier runs are still trying neighbouring counts
	bool learned_from_other_jobs = false; // `learned` comes from other jobs of the same kind (none of this one)
	double gpu_bytes_per_worker = 0; // measured in an earlier run of this job (0: the measured rule's)
};

struct RankAdvice
{
	int workers_per_gpu = 1;
	int mpi = 1;                 // MPI processes to start: workers (+ the leader if more than 1)
	int threads = 1;             // threads per worker
	std::string reason;          // one short paragraph; names every limit that applied
};

// Share of the GPU memory that workers may use; the rest is for other users of the card and
// for growth (the last iteration of an auto-refine needs more than the earlier ones).
const double GPU_FREE_FRACTION_FOR_RANKS = 0.75;
// CPU threads that feed one GPU when the number of cores is unknown (as measured).
const int TOTAL_THREADS_PER_GPU = 12;
const double HOST_FREE_FRACTION_FOR_RANKS = 0.6;

// A worker gets at least this many particles per GPU thread slot; fewer and the fixed costs
// of an iteration dominate.
const long MIN_PARTICLES_PER_WORKER = 300;

/// The recommended ranks per GPU for a GPU job. RELION_RANKS_PER_GPU (a positive number) overrides
/// the workers per GPU; the limits of memory, cores and particles are then not applied, but
/// the even number of workers of auto-refine still is. Never returns less than one worker.
RankAdvice adviseRanks(const RankQuery &q);

/// adviseRanks() for a GUI job (its kind of work, the particles of its input STAR file, the memory of the
/// GPUs and the computer when they are known: this computer's for local jobs; for queue jobs only the GPU
/// memory given in RELION_GPU_MEMORY_GB). `gpu_memory_note` says when the GPU memory is unknown.
RankAdvice adviseRanksForJob(const JobView &job, int gpus, int cores_per_gpu, bool queue, std::string &gpu_memory_note);

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
