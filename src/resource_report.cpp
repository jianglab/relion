/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/resource_report.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <set>
#include <sstream>
#include <unistd.h>
#include <sched.h>
#include <sys/resource.h>

namespace resrep
{

namespace
{

// The few NVML entry points we need, looked up at run time
struct Nvml
{
	typedef int (*InitFn)();
	typedef int (*HandleFn)(const char *, void **);
	struct Util { unsigned int gpu, memory; };
	typedef int (*UtilFn)(void *, Util *);
	InitFn init = NULL;
	HandleFn by_pci = NULL;
	UtilFn util = NULL;
	bool ok = false;

	Nvml()
	{
		void *lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
		if (lib == NULL) lib = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_LOCAL);
		if (lib == NULL) return;
		init = (InitFn)dlsym(lib, "nvmlInit_v2");
		by_pci = (HandleFn)dlsym(lib, "nvmlDeviceGetHandleByPciBusId_v2");
		util = (UtilFn)dlsym(lib, "nvmlDeviceGetUtilizationRates");
		ok = init && by_pci && util && init() == 0;
	}
};

Nvml &nvml()
{
	static Nvml n;
	return n;
}

double now()
{
	return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double cpuSeconds()
{
	struct rusage r;
	getrusage(RUSAGE_SELF, &r);
	return r.ru_utime.tv_sec + 1e-6 * r.ru_utime.tv_usec + r.ru_stime.tv_sec + 1e-6 * r.ru_stime.tv_usec;
}

std::string pct(double f)
{
	std::ostringstream o;
	o << (int)std::lround(100. * f) << "%";
	return o.str();
}

}

bool enabled()
{
	const char *s = getenv("RELION_RESOURCE_REPORT");
	return !(s != NULL && (std::string(s) == "off" || std::string(s) == "0"));
}

bool mpsRunning()
{
	const char *d = getenv("CUDA_MPS_PIPE_DIRECTORY");
	const std::string pipe = std::string((d && *d) ? d : "/tmp/nvidia-mps") + "/control";
	return access(pipe.c_str(), F_OK) == 0;
}

std::vector<int> allowedCores()
{
	std::vector<int> out;
	cpu_set_t set;
	CPU_ZERO(&set);
	if (sched_getaffinity(0, sizeof(set), &set) == 0)
	{
		for (int c = 0; c < CPU_SETSIZE; c++)
			if (CPU_ISSET(c, &set)) out.push_back(c);
	}
	else
	{
		long n = sysconf(_SC_NPROCESSORS_ONLN);
		for (int c = 0; c < n; c++) out.push_back(c);
	}
	return out;
}

Monitor::~Monitor()
{
	running_ = false;
	if (thread_.joinable()) thread_.join();
}

void Monitor::start(const std::vector<std::string> &gpu_pci_ids)
{
	pci_ = gpu_pci_ids;
	handles_.assign(pci_.size(), NULL);
	sum_.assign(pci_.size(), 0.);
	n_.assign(pci_.size(), 0);
	if (!pci_.empty() && nvml().ok)
		for (size_t i = 0; i < pci_.size(); i++)
			if (nvml().by_pci(pci_[i].c_str(), &handles_[i]) != 0) handles_[i] = NULL;
	t0_ = now();
	cpu0_ = cpuSeconds();
	bool any = false;
	for (void *h : handles_) any = any || h != NULL;
	if (any)
	{
		running_ = true;
		thread_ = std::thread(&Monitor::sample, this);
	}
}

void Monitor::sample()
{
	while (running_)
	{
		for (size_t i = 0; i < handles_.size(); i++)
		{
			Nvml::Util u;
			if (handles_[i] != NULL && nvml().util(handles_[i], &u) == 0)
			{
				sum_[i] += u.gpu / 100.;
				n_[i]++;
			}
		}
		for (int k = 0; k < 5 && running_; k++)
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
}

ProcessStats Monitor::stop(int threads)
{
	running_ = false;
	if (thread_.joinable()) thread_.join();
	ProcessStats s;
	char host[256] = "";
	gethostname(host, sizeof(host) - 1);
	s.host = host;
	s.threads = threads;
	s.wall = now() - t0_;
	s.cpu = cpuSeconds() - cpu0_;
	s.cores = allowedCores();
	s.uses_gpu = !pci_.empty();
	s.gpus = pci_;
	s.mps = s.uses_gpu && mpsRunning();
	for (size_t i = 0; i < pci_.size(); i++)
		s.gpu_busy.push_back(n_[i] > 0 ? sum_[i] / n_[i] : -1.);
	return s;
}

std::string serialise(const ProcessStats &s)
{
	std::ostringstream o;
	o.precision(17);
	o << s.host << ' ' << s.threads << ' ' << s.uses_gpu << ' ' << s.mps << ' ' << s.wall << ' ' << s.cpu << ' ' << s.cores.size();
	for (int c : s.cores) o << ' ' << c;
	o << ' ' << s.gpus.size();
	for (size_t i = 0; i < s.gpus.size(); i++) o << ' ' << s.gpus[i] << ' ' << s.gpu_busy[i];
	return o.str();
}

ProcessStats deserialise(const std::string &str)
{
	std::istringstream in(str);
	ProcessStats s;
	size_t n;
	in >> s.host >> s.threads >> s.uses_gpu >> s.mps >> s.wall >> s.cpu >> n;
	s.cores.resize(n);
	for (size_t i = 0; i < n; i++) in >> s.cores[i];
	in >> n;
	s.gpus.resize(n);
	s.gpu_busy.resize(n);
	for (size_t i = 0; i < n; i++) in >> s.gpus[i] >> s.gpu_busy[i];
	return s;
}

void accumulate(ProcessStats &total, const ProcessStats &s)
{
	if (total.wall <= 0.)
	{
		total = s;
		for (double &b : total.gpu_busy) b = (b >= 0.) ? b * s.wall : -1.;
		return;
	}
	for (size_t i = 0; i < total.gpu_busy.size() && i < s.gpu_busy.size(); i++)
		total.gpu_busy[i] = (total.gpu_busy[i] >= 0. && s.gpu_busy[i] >= 0.) ? total.gpu_busy[i] + s.gpu_busy[i] * s.wall : -1.;
	total.wall += s.wall;
	total.cpu += s.cpu;
}

void finish(ProcessStats &total)
{
	for (double &b : total.gpu_busy)
		if (b >= 0. && total.wall > 0.) b /= total.wall;
}

std::vector<std::string> advise(const std::vector<ProcessStats> &procs, bool mpi)
{
	std::vector<std::string> out;
	std::map<std::string, std::vector<size_t> > by_host;
	for (size_t i = 0; i < procs.size(); i++) by_host[procs[i].host].push_back(i);
	const bool many_hosts = by_host.size() > 1;

	for (const auto &h : by_host)
	{
		const std::string where = many_hosts ? " on " + h.first : "";
		std::set<int> cores;
		int threads = 0, workers = 0, worker_threads = 0;
		double cpu = 0., wall = 0.;
		std::map<std::string, double> gpu_busy;
		for (size_t i : h.second)
		{
			const ProcessStats &p = procs[i];
			cores.insert(p.cores.begin(), p.cores.end());
			// the MPI leader only coordinates: count it as one thread
			const bool leader = mpi && i == 0;
			threads += leader ? 1 : p.threads;
			cpu += p.cpu;
			wall = std::max(wall, p.wall);
			if (!leader)
			{
				workers++;
				worker_threads += p.threads;
			}
			for (size_t g = 0; g < p.gpus.size(); g++)
				gpu_busy[p.gpus[g]] = std::max(gpu_busy.count(p.gpus[g]) ? gpu_busy[p.gpus[g]] : -1., p.gpu_busy[g]);
		}
		if (workers == 0 || wall <= 0.) continue;
		const int ncores = (int)cores.size();
		const double used = cpu / wall;                 // cores busy on average
		const bool cores_idle = used < 0.6 * ncores;
		const int per_worker = std::max(1, worker_threads / workers);
		std::ostringstream use;
		use.precision(2);
		use << "on average only " << (used < 10. ? std::round(used * 10.) / 10. : std::round(used)) << " of the " << ncores
		    << " allocated cores" << where << " were busy";
		const int need = std::max(1, (int)std::ceil(used * 1.2));

		if (!gpu_busy.empty())
		{
			double sum = 0.;
			int known = 0;
			for (const auto &g : gpu_busy)
				if (g.second >= 0.) { sum += g.second; known++; }
			const int ngpu = (int)gpu_busy.size();
			if (known == 0)
			{
				out.push_back("GPU use" + where + " could not be measured (NVML not found), so there is no advice on processes and threads.");
				continue;
			}
			const double busy = sum / known;
			std::ostringstream o;
			const std::string gpus = ngpu > 1 ? "GPUs were" : "GPU was";
			if (busy < 0.6)
			{
				// Measured on helical Class2D: raising --j inside one process made the E-step
				// slower (threads contend), and several processes sharing one GPU only helped
				// once NVIDIA MPS let their kernels run at the same time (2.7x faster).
				bool mps = false;
				for (size_t i : h.second) mps = mps || procs[i].mps;
				o << "The " << gpus << " busy only " << pct(busy) << " of the time" << where
				  << ": the work per particle is small and the GPU waits for the CPU side. ";
				const int suggest = std::max(2, ncores - 1);
				if (!mps)
					o << "Run several MPI processes per GPU (about " << suggest << ", each with --j 1) under NVIDIA MPS, "
					  << "which lets their GPU work run at the same time; without MPS they take turns and gain little. "
					  << "Start it inside the job before mpirun: export CUDA_MPS_PIPE_DIRECTORY=$TMPDIR/mps CUDA_MPS_LOG_DIRECTORY=$TMPDIR/mps; "
					  << "mkdir -p $TMPDIR/mps; nvidia-cuda-mps-control -d (and stop it afterwards with: echo quit | nvidia-cuda-mps-control).";
				else
					o << "MPS is running: more MPI processes per GPU (about " << suggest << ", each with --j 1) should feed it better.";
				out.push_back(o.str());
				if (threads >= ncores && cores_idle)
					out.push_back("Many threads were waiting rather than computing (" + use.str() + "); if images are read from a slow disk, --preread_images or --scratch_dir may help.");
			}
			else
			{
				if (busy > 0.9)
				{
					o << "The " << gpus << " busy " << pct(busy) << " of the time" << where
					  << ", so more CPU processes or threads are unlikely to make this faster.";
					out.push_back(o.str());
				}
				if (cores_idle)
					out.push_back("The " + gpus + " kept busy, but " + use.str() + ": request about " + std::to_string(need)
					              + " cores (for example --cpus-per-task with Slurm) to avoid wasting the rest.");
			}
		}
		else if (cores_idle)
		{
			std::ostringstream o;
			o << use.str() << ". ";
			if (threads < ncores)
				o << "Raise --j from " << per_worker << " to " << per_worker + (ncores - threads) / workers << ", or request "
				  << threads << " cores, to avoid wasting them.";
			else
				o << "The threads were mostly waiting, probably for images from disk: try --preread_images or --scratch_dir.";
			out.push_back(o.str());
		}
	}
	return out;
}

std::vector<std::string> summary(const std::vector<ProcessStats> &procs, bool mpi)
{
	std::vector<std::string> out;
	std::map<std::string, std::vector<size_t> > by_host;
	for (size_t i = 0; i < procs.size(); i++) by_host[procs[i].host].push_back(i);
	for (const auto &h : by_host)
	{
		std::set<int> cores;
		double cpu = 0., wall = 0.;
		int workers = 0;
		bool mps = false;
		std::map<std::string, double> gpu_busy;
		for (size_t i : h.second)
		{
			const ProcessStats &p = procs[i];
			cores.insert(p.cores.begin(), p.cores.end());
			cpu += p.cpu;
			wall = std::max(wall, p.wall);
			if (!(mpi && i == 0)) workers++;
			mps = mps || p.mps;
			for (size_t g = 0; g < p.gpus.size(); g++)
				gpu_busy[p.gpus[g]] = std::max(gpu_busy.count(p.gpus[g]) ? gpu_busy[p.gpus[g]] : -1., p.gpu_busy[g]);
		}
		std::ostringstream o;
		o.precision(2);
		if (by_host.size() > 1) o << h.first << ": ";
		o << workers << " worker process" << (workers > 1 ? "es" : "");
		if (!gpu_busy.empty())
		{
			o << ", GPU busy";
			for (const auto &g : gpu_busy) o << " " << (g.second >= 0. ? pct(g.second) : std::string("unknown"));
			o << (mps ? " (MPS on)" : "");
		}
		if (wall > 0.)
			o << ", on average " << std::round(10. * cpu / wall) / 10. << " of " << cores.size() << " cores busy";
		out.push_back(o.str());
	}
	return out;
}

}
