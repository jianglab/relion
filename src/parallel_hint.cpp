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
#include <map>
#include <set>
#include <sstream>
#include <thread>

namespace parhint
{

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
	out = split(kind, resourcesOf(job).gpus, cpus);
	return out.valid;
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
