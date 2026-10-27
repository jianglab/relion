/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/parallel_hint.h"
#include "src/pipeline_jobs.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>

namespace parhint
{

int autoThreads(int cores, int sharing, int gpus, int per_gpu_cap, double free_gpu_bytes, long particles_per_worker)
{
	const int share = std::max(1, sharing);
	int n = std::max(1, cores) / share;
	if (gpus > 0 && per_gpu_cap > 0)
	{
		// GPUs this process feeds: the GPUs of the node divided among its sharing workers
		const int gpus_each = std::max(1, (gpus + share - 1) / share);
		n = std::min(n, per_gpu_cap * gpus_each);
	}
	if (gpus > 0 && free_gpu_bytes > 0)
	{
		const double per_worker = free_gpu_bytes * GPU_FREE_FRACTION_FOR_THREADS / (GPU_BYTES_PER_THREAD * share);
		n = std::min<double>(n, std::max(1.0, per_worker));
	}
	if (particles_per_worker > 0)
		n = std::min<long>(n, std::max<long>(1, particles_per_worker / MIN_PARTICLES_PER_THREAD));
	return std::max(1, n);
}

long countParticlesInStar(const std::string &fn)
{
	std::ifstream in(fn);
	if (!in) return 0;
	std::string line;
	bool in_block = false, in_rows = false;
	long n = 0;
	while (std::getline(in, line))
	{
		if (line.compare(0, 5, "data_") == 0)
		{
			if (in_block && in_rows) break;
			in_block = (line.compare(0, 14, "data_particles") == 0 || line == "data_" || line.compare(0, 6, "data_ ") == 0);
			in_rows = false;
		}
		else if (!in_block) continue;
		else if (line.compare(0, 5, "loop_") == 0) in_rows = false;
		else if (!line.empty() && line[0] == '_') in_rows = true;
		else if (in_rows)
		{
			const size_t f = line.find_first_not_of(" \t\r");
			if (f == std::string::npos) break;           // blank line ends the table
			if (line[f] != '#') n++;
		}
	}
	return n;
}

int autoThreadsCapFromEnv()
{
	const char *e = getenv("RELION_AUTO_THREADS_CAP");
	if (e == nullptr || *e == 0) return THREADS_PER_GPU_CAP;
	const std::string v(e);
	if (v == "off" || v == "0") return 0;
	char *end = nullptr;
	const long n = strtol(e, &end, 10);
	return (end != e && *end == 0 && n > 0) ? (int)n : THREADS_PER_GPU_CAP;
}


namespace
{

std::string lower(std::string s)
{
	for (char &c : s) c = std::tolower((unsigned char)c);
	return s;
}

int leadingInt(const std::string &s)
{
	size_t i = s.find_first_of("0123456789");
	return (i == std::string::npos) ? 0 : std::atoi(s.c_str() + i);
}

std::string runCommand(const std::string &cmd)
{
	std::string out;
	FILE *p = popen(cmd.c_str(), "r");
	if (p == NULL) return out;
	char buf[512];
	while (fgets(buf, sizeof buf, p)) out += buf;
	pclose(p);
	return out;
}

int localGpus()
{
	static int n = -1;
	if (n < 0)
	{
		std::istringstream in(runCommand("timeout 5 nvidia-smi -L 2>/dev/null"));
		std::string line;
		n = 0;
		while (std::getline(in, line))
			if (line.compare(0, 4, "GPU ") == 0) n++;
	}
	return n;
}

// Cores per GPU on the nodes of a Slurm partition (0 if unknown); asked once per partition
int partitionCoresPerGpu(const std::string &queue)
{
	static std::map<std::string, int> cache;
	auto it = cache.find(queue);
	if (it != cache.end()) return it->second;
	int r = 0;
	bool safe = !queue.empty() && std::all_of(queue.begin(), queue.end(),
		[](char c) { return std::isalnum((unsigned char)c) || c == '_' || c == '-' || c == '.' || c == ','; });
	if (safe)
		r = coresPerGpuFromSinfo(runCommand("timeout 5 sinfo -h -p '" + queue + "' -o '%c %G' 2>/dev/null"));
	cache[queue] = r;
	return r;
}

std::string plural(int n, const std::string &word)
{
	return std::to_string(n) + " " + word + (n == 1 ? "" : "s");
}

}

Suggestion suggest(Kind kind, int gpus, int cores_per_gpu)
{
	Suggestion s;
	if (kind == Kind::Other || gpus < 1) return s;
	s.valid = true;
	const int c = std::max(0, cores_per_gpu);
	switch (kind)
	{
	case Kind::Classify:
		s.mpi = (gpus == 1) ? 1 : gpus + 1;
		s.threads = c;
		break;
	case Kind::Vdam:
		s.mpi = 1;
		s.threads = gpus * c;
		break;
	case Kind::AutoRefine:
	{
		const int workers = std::max(2, gpus + gpus % 2);
		s.mpi = workers + 1;
		s.threads = (c == 0) ? 0 : std::max(1, gpus * c / workers);
		break;
	}
	default:
		break;
	}
	return s;
}

Kind kindOf(int job_type, bool vdam)
{
	switch (job_type)
	{
	case PROC_2DCLASS:   return vdam ? Kind::Vdam : Kind::Classify;
	case PROC_3DCLASS:   return Kind::Classify;
	case PROC_INIMODEL:  return Kind::Vdam;
	case PROC_3DAUTO:
	case PROC_MULTIBODY: return Kind::AutoRefine;
	default:             return Kind::Other;
	}
}

int countGpuIds(const std::string &ids)
{
	std::set<std::string> distinct;
	std::string cur;
	for (char ch : ids + ":")
	{
		if (ch == ':' || ch == ',' || ch == ';' || std::isspace((unsigned char)ch))
		{
			if (!cur.empty()) distinct.insert(cur);
			cur.clear();
		}
		else
			cur += ch;
	}
	return (int)distinct.size();
}

int gpusInGres(const std::string &gres)
{
	// such as "gpu:4", "gpu:a100:4(S:0-1)", "gpu:2,mps:200", "(null)"
	int ngpu = 0;
	std::istringstream gs(gres);
	std::string item;
	while (std::getline(gs, item, ','))
	{
		if (item.compare(0, 4, "gpu:") != 0) continue;
		std::string count = item.substr(0, item.find('('));
		count = count.substr(count.rfind(':') + 1);
		ngpu += leadingInt(count);
	}
	return ngpu;
}

int coresPerGpuFromSinfo(const std::string &sinfo_output)
{
	std::istringstream in(sinfo_output);
	std::string line;
	int best = 0;
	while (std::getline(in, line))
	{
		std::istringstream ls(line);
		std::string cpus, gres;
		if (!(ls >> cpus >> gres)) continue;
		const int ncpu = leadingInt(cpus);
		const int ngpu = gpusInGres(gres);
		if (ncpu <= 0 || ngpu <= 0) continue;
		const int ratio = std::max(1, ncpu / ngpu);
		best = (best == 0) ? ratio : std::min(best, ratio);
	}
	return best;
}

namespace
{

struct Resources
{
	bool queue = false;
	int gpus = 0;               // 0: the job does not use GPUs
	std::string gpu_source;
	int cores_per_gpu = 0;      // 0: unknown
	std::string core_source;
};

Resources resourcesOf(const JobView &job)
{
	Resources r;
	r.queue = job.value("do_queue") == "Yes";
	if (job.value("use_gpu") != "Yes") return r;

	if (r.queue && (r.gpus = leadingInt(job.value("slurm_gpus"))) > 0)
		r.gpu_source = "\"Number of GPUs\" below";
	if (r.queue)
		for (int i = 1; i <= 32 && r.gpus == 0; i++)
		{
			const std::string key = "qsub_extra" + std::to_string(i);
			const std::string label = job.label(key);
			if (lower(label).find("gpu") == std::string::npos) continue;
			r.gpus = leadingInt(job.value(key));
			r.gpu_source = "\"" + label + "\" below";
		}
	// Class2D replicas give one mapping per replica, separated by ';': count those of one
	// replica, since the numbers of processes and threads are per replica
	const std::string ids = job.value("gpu_ids");
	if (r.gpus == 0 && (r.gpus = countGpuIds(ids.substr(0, ids.find(';')))) > 0)
		r.gpu_source = "\"Which GPUs to use\" on the Compute tab";
	if (r.gpus == 0 && !r.queue && (r.gpus = localGpus()) > 0)
		r.gpu_source = "the GPUs of this computer";
	if (r.gpus == 0)
	{
		r.gpus = 1;
		r.gpu_source = "none given, 1 assumed";
	}

	if (const char *e = getenv("RELION_CPUS_PER_GPU"))
		if ((r.cores_per_gpu = std::atoi(e)) > 0)
		{
			r.core_source = "RELION_CPUS_PER_GPU";
			return r;
		}
	int share = 0;
	if (r.queue)
	{
		const std::string q = job.value("queuename");
		if ((share = partitionCoresPerGpu(q)) > 0)
			r.core_source = "the nodes of queue " + q + " (Slurm)";
	}
	else
	{
		const int n = (int)std::thread::hardware_concurrency();
		if (n > 0)
		{
			share = std::max(1, n / std::max(1, localGpus()));
			r.core_source = "this computer";
		}
	}
	if (share > 0)
	{
		r.cores_per_gpu = std::min(share, THREADS_PER_GPU_CAP);
		if (share > THREADS_PER_GPU_CAP)
			r.core_source += ", at most " + std::to_string(THREADS_PER_GPU_CAP) +
				" (more threads do not make one GPU faster)";
	}
	return r;
}

void explain(std::ostringstream &t, Kind kind, const Suggestion &s, int gpus)
{
	switch (kind)
	{
	case Kind::Classify:
		if (s.mpi == 1)
			t << "One process with a thread per CPU core keeps one GPU as busy as\n"
			     "several processes do.";
		else if (gpus > 0)
			t << "A leader process, which only coordinates, plus one worker\n"
			     "process per GPU, each with the CPU cores of its GPU.";
		else
			t << "A leader process, which only coordinates, plus worker processes\n"
			     "that share the CPU cores.";
		break;
	case Kind::Vdam:
		t << "VDAM does not support MPI; all threads are in one process" << (gpus > 1 ? ",\nspread over the GPUs." : ".");
		break;
	case Kind::AutoRefine:
		t << "Auto-refine needs an odd number of MPI processes, at least 3: a\n"
		     "leader, which only coordinates, and the same number of workers\n"
		     "for each half-set. The CPU cores are shared among the workers.";
		if (gpus > 0 && s.mpi - 1 > gpus)
			t << "\nWorkers that share a GPU run on it at the same time (RELION\n"
			     "starts NVIDIA MPS for them).";
		break;
	default:
		break;
	}
}

void sources(std::ostringstream &t, const Resources &r)
{
	if (r.gpus == 0)
		t << "\n\nThis job does not use GPUs (see the Compute tab).";
	else
	{
		t << "\n\nGPUs: " << r.gpu_source << ".";
		if (r.cores_per_gpu > 0) t << "\nCPU cores per GPU: " << r.core_source << ".";
		else t << "\nCPU cores per GPU unknown (set RELION_CPUS_PER_GPU).";
	}
	if (r.queue)
		t << "\nThe queue reserves MPI processes x threads CPU cores.";
}

}

Suggestion split(Kind kind, int gpus, int cpus)
{
	Suggestion s;
	if (kind == Kind::Other || cpus < 1) return s;
	s.valid = true;
	int workers = 1;
	switch (kind)
	{
	case Kind::Classify:
		workers = (gpus > 0) ? gpus : std::max(1, cpus / THREADS_PER_CPU_WORKER);
		s.mpi = (workers == 1) ? 1 : workers + 1;
		break;
	case Kind::Vdam:
		s.mpi = 1;
		break;
	case Kind::AutoRefine:
		workers = (gpus > 0) ? std::max(2, gpus) : std::max(2, cpus / THREADS_PER_CPU_WORKER);
		workers += workers % 2;
		s.mpi = workers + 1;
		break;
	default:
		break;
	}
	s.threads = std::max(1, cpus / workers);
	return s;
}

bool autoSplit(const JobView &job, Suggestion &out)
{
	if (job.value("auto_parallel") != "Yes") return false;
	const Kind kind = kindOf(job.job_type, job.value("do_grad") == "Yes");
	const int cpus = leadingInt(job.value("nr_cpus"));
	if (kind == Kind::Other || cpus < 1) return false;
	const Resources r = resourcesOf(job);
	out = split(kind, r.gpus, cpus);
	if (out.valid && r.gpus > 0 && (kind == Kind::Classify || kind == Kind::AutoRefine))
	{
		// the measured rule, as in the hover hint and in relion_launch
		std::string note;
		const RankAdvice a = adviseRanksForJob(job, r.gpus, std::max(1, cpus / r.gpus), r.queue, note);
		out.mpi = a.mpi;
		out.threads = a.threads;
	}
	return out.valid;
}

namespace
{

// Measured on an RTX 3090 (see documentation/ranks_per_gpu.md): where the E-step stops getting
// faster with more worker processes on one GPU, and what a worker costs on the GPU and the host.
// min_threads: below it a worker is slower than fewer workers with more threads (all six cluster GPU models).
struct WorkProfile { int workers; double gpu_gb; double host_gb; const char *name; int min_threads; };
std::string gbText(double bytes)
{
	char buf[32];
	snprintf(buf, sizeof buf, "%.1f GB", bytes / 1e9);
	return buf;
}

WorkProfile profileOf(Work w)
{
	switch (w)
	{
	case Work::Class2D:    return {4, 0.4, 0.4, "EM Class2D", 1};
	case Work::Class3D:    return {2, 2.5, 0.7, "3D classification with alignment", 1};
	// 4 was best on a 4.5k-particle set; on 70k segments 4 workers were 8% slower than 1 or 2 (A4000)
	case Work::SkipAlign:  return {2, 2.5, 1.8, "3D classification without alignment", 2};
	case Work::AutoRefine: return {2, 7.8, 14.0, "auto-refine", 1};
	}
	return {1, 1.0, 1.0, "", 1};
}

}

int workersPerGpuTarget(Work work) { return profileOf(work).workers; }

double gpuBytesPerWorker(Work work)
{
	if (const char *e = getenv("RELION_GPU_GB_PER_RANK"))
		if (atof(e) > 0) return atof(e) * 1e9;
	return profileOf(work).gpu_gb * 1e9;
}

RankAdvice adviseRanks(const RankQuery &q)
{
	const WorkProfile p = profileOf(q.work);
	const int gpus = std::max(1, q.gpus);
	const bool refine = q.work == Work::AutoRefine;
	RankAdvice a;
	std::ostringstream why;
	int w = p.workers;                       // workers per GPU
	why << p.name << ": the E-step stops getting faster at " << w << " worker" << (w == 1 ? "" : "s") << " per GPU.";

	const char *env = getenv("RELION_RANKS_PER_GPU");
	const long forced = env ? atol(env) : 0;
	if (forced > 0)
	{
		w = (int)forced;
		why.str("");
		why << "RELION_RANKS_PER_GPU sets " << w << " worker" << (w == 1 ? "" : "s") << " per GPU.";
	}
	else
	{
		if (q.learned > 0)
		{
			w = q.learned;
			if (q.learned_from_other_jobs)
				why << " Other jobs of this kind on this GPU model, with as many cores per GPU, settled on " << w << ".";
			else
				why << " Earlier runs of this job on this GPU model " << (q.learned_settled ? "settled on " : "are trying ") << w << ".";
		}
		auto limit = [&](int cap, const std::string &what)
		{
			cap = std::max(1, cap);
			if (cap < w) { w = cap; why << " Limited to " << cap << " by " << what << "."; }
		};
		// a measurement of this job beats the rule's figure, but not an expert's RELION_GPU_GB_PER_RANK
		const bool measured = q.gpu_bytes_per_worker > 0 && getenv("RELION_GPU_GB_PER_RANK") == NULL;
		const double per_worker = measured ? q.gpu_bytes_per_worker : gpuBytesPerWorker(q.work);
		if (q.gpu_bytes > 0)
			limit((int)(q.gpu_bytes * GPU_FREE_FRACTION_FOR_RANKS / per_worker),
			      "GPU memory (about " + gbText(per_worker) + " per worker" + (measured ? ", as measured in an earlier run of this job" : "") + ")");
		if (q.host_bytes > 0)
			limit((int)(q.host_bytes * HOST_FREE_FRACTION_FOR_RANKS / (p.host_gb * 1e9)) / gpus,
			      "the computer's memory (about " + gbText(p.host_gb * 1e9) + " per worker)");
		if (q.cores > 0)
			limit(q.cores / (gpus * p.min_threads), p.min_threads == 1 ? std::string("the CPU cores (one per worker at least)")
			      : "the CPU cores (" + std::to_string(p.min_threads) + " per worker at least)");
		if (q.particles > 0)
			limit((int)std::min<long>(1000, q.particles / (gpus * MIN_PARTICLES_PER_WORKER)), "the number of particles");
	}
	int workers = w * gpus;
	if (refine && workers % 2)               // auto-refine needs the same number of workers for each half-set
	{
		workers = (workers > 1 && forced <= 0) ? workers - 1 : workers + 1;
		why << " Auto-refine needs an even number of workers: " << workers << ".";
	}
	a.workers_per_gpu = (workers + gpus - 1) / gpus;
	a.mpi = (workers == 1 && !refine) ? 1 : workers + 1;
	const int cores_per_worker = (q.cores > 0) ? q.cores / workers : TOTAL_THREADS_PER_GPU * gpus / workers;
	a.threads = std::max(1, std::min(THREADS_PER_GPU_CAP, cores_per_worker));
	if (a.mpi > 1) why << " The extra process is the leader, which only coordinates.";
	a.reason = why.str();
	return a;
}

namespace
{

// The work of a GUI job on the GPU
Work workOf(const JobView &job, Kind kind)
{
	if (kind == Kind::AutoRefine) return Work::AutoRefine;
	if (job.job_type == PROC_2DCLASS) return Work::Class2D;
	return job.value("dont_skip_align") == "No" ? Work::SkipAlign : Work::Class3D;
}

// Memory of the smallest GPU of this computer in bytes (0: unknown); RELION_GPU_MEMORY_GB overrides
double gpuBytesOfThisComputer()
{
	if (const char *e = getenv("RELION_GPU_MEMORY_GB"))
		if (atof(e) > 0) return atof(e) * 1e9;
	static double bytes = -1;
	if (bytes < 0)
	{
		std::istringstream in(runCommand("timeout 5 nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null"));
		double mb, best = 0;
		while (in >> mb) best = (best == 0) ? mb : std::min(best, mb);
		bytes = best * 1048576.0;
	}
	return bytes;
}

// Available memory of this computer in bytes (0: unknown); RELION_HOST_MEMORY_GB overrides
double hostBytesOfThisComputer()
{
	if (const char *e = getenv("RELION_HOST_MEMORY_GB"))
		if (atof(e) > 0) return atof(e) * 1e9;
	std::ifstream in("/proc/meminfo");
	std::string key, unit;
	double kb;
	while (in >> key >> kb >> unit)
		if (key == "MemAvailable:") return kb * 1024.0;
	return 0;
}

// Particles of the input STAR file, counted once per file and size
long particlesOf(const std::string &fn)
{
	static std::map<std::string, std::pair<long, long>> cache;   // fn -> (file size, particles)
	std::ifstream f(fn, std::ios::binary | std::ios::ate);
	if (!f) return 0;
	const long size = (long)f.tellg();
	auto it = cache.find(fn);
	if (it != cache.end() && it->second.first == size) return it->second.second;
	const long n = countParticlesInStar(fn);
	cache[fn] = {size, n};
	return n;
}

}

RankAdvice adviseRanksForJob(const JobView &job, int gpus, int cores_per_gpu, bool queue, std::string &gpu_memory_note)
{
	RankQuery q;
	q.work = workOf(job, kindOf(job.job_type, job.value("do_grad") == "Yes"));
	q.gpus = gpus;
	q.cores = cores_per_gpu * gpus;
	q.gpu_bytes = (queue && getenv("RELION_GPU_MEMORY_GB") == nullptr) ? 0 : gpuBytesOfThisComputer();
	q.host_bytes = queue ? 0 : hostBytesOfThisComputer();
	q.particles = particlesOf(job.value("fn_img"));
	gpu_memory_note = (q.gpu_bytes > 0) ? "" : "GPU memory unknown (set RELION_GPU_MEMORY_GB)";
	return adviseRanks(q);
}

std::string tooltip(const JobView &job, Field field)
{
	const Kind kind = kindOf(job.job_type, job.value("do_grad") == "Yes");
	if (kind == Kind::Other) return "";
	const Resources r = resourcesOf(job);
	const bool automatic = job.value("auto_parallel") == "Yes";
	std::ostringstream t;

	if (automatic || field == Field::Cpus)
	{
		Suggestion s;
		const int cpus = leadingInt(job.value("nr_cpus"));
		if (automatic && autoSplit(job, s))
		{
			t << "RELION splits the " << plural(cpus, "CPU") << " into\n"
			  << "    MPI processes: " << s.mpi << "    Threads: " << s.threads << "\n\n";
			if (r.gpus > 0 && (kind == Kind::Classify || kind == Kind::AutoRefine))
			{
				std::string note;
				t << adviseRanksForJob(job, r.gpus, std::max(1, cpus / r.gpus), r.queue, note).reason
				  << "\nrelion_launch makes the final choice on the computer the job runs on, from its\n"
				     "GPUs, cores and memory and from the speed of earlier runs of the same job.";
			}
			else
				explain(t, kind, s, r.gpus);
		}
		else if (!automatic)
			t << "Used only when MPI processes and threads are chosen automatically.";
		if (r.gpus > 0 && r.cores_per_gpu > 0)
		{
			t << "\n\nSuggested number of CPUs for " << plural(r.gpus, "GPU") << ": " << r.gpus * r.cores_per_gpu << ".";
		}
		sources(t, r);
		return t.str();
	}

	// Manual choice: suggest numbers
	if (r.gpus == 0)
	{
		t << "This job does not use GPUs (see the Compute tab); it takes\n"
		     "MPI processes x threads CPU cores.";
		if (kind == Kind::Vdam)
			t << "\nVDAM does not support MPI: use 1 MPI process and a thread per core.";
		if (kind == Kind::AutoRefine)
			t << "\nAuto-refine needs an odd number of MPI processes, at least 3.";
		return t.str();
	}
	if (kind == Kind::Classify || kind == Kind::AutoRefine)
	{
		std::string note;
		const RankAdvice a = adviseRanksForJob(job, r.gpus, r.cores_per_gpu, r.queue, note);
		t << "Suggested for " << plural(r.gpus, "GPU");
		if (r.cores_per_gpu > 0) t << " with " << r.cores_per_gpu << " CPU cores per GPU";
		t << ":\n    MPI processes: " << a.mpi << "    Threads: " << a.threads << "\n\n" << a.reason
		  << "\nWorkers that share a GPU run on it at the same time; each takes about "
		  << (int)(gpuBytesPerWorker(workOf(job, kind)) / 1e8 + 0.5) / 10.0 << " GB of its memory.";
		if (!note.empty()) t << "\n(" << note << ".)";
		sources(t, r);
		return t.str();
	}
	const Suggestion s = suggest(kind, r.gpus, r.cores_per_gpu);
	t << "Suggested for " << plural(r.gpus, "GPU");
	if (r.cores_per_gpu > 0) t << " with " << r.cores_per_gpu << " CPU cores per GPU";
	t << ":\n    MPI processes: " << s.mpi << "    Threads: ";
	if (s.threads > 0) t << s.threads;
	else t << "the CPU cores per GPU";
	t << "\n\n";
	explain(t, kind, s, r.gpus);
	sources(t, r);
	return t.str();
}

}
