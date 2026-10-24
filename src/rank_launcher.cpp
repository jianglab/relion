/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/rank_launcher.h"
#include "src/resource_report.h"
#include "src/tuning_cache.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>

namespace rlaunch
{

namespace
{

std::string trim(const std::string &s)
{
	const size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
	return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

std::vector<std::string> split(const std::string &s, char sep)
{
	std::vector<std::string> out;
	std::stringstream in(s);
	std::string item;
	while (std::getline(in, item, sep)) out.push_back(trim(item));
	return out;
}

std::string run(const std::string &cmd)
{
	std::string out;
	if (FILE *p = popen(cmd.c_str(), "r"))
	{
		char buf[512];
		while (fgets(buf, sizeof buf, p)) out += buf;
		pclose(p);
	}
	return out;
}

// Memory this job may use: Slurm's limit, else the cgroup's, else what is available
double hostBytes()
{
	if (const char *e = getenv("SLURM_MEM_PER_NODE"))
		if (atof(e) > 0) return atof(e) * 1048576.0;
	std::ifstream cg("/sys/fs/cgroup/memory.max");
	std::string v;
	if (cg >> v && v != "max" && atof(v.c_str()) > 0) return atof(v.c_str());
	std::ifstream mi("/proc/meminfo");
	std::string key;
	double kb;
	while (mi >> key >> kb)
	{
		if (key == "MemAvailable:") return kb * 1024.0;
		mi.ignore(256, '\n');
	}
	return 0;
}

std::string valueOf(const std::vector<std::string> &args, const std::string &opt)
{
	for (size_t i = 0; i + 1 < args.size(); i++)
		if (args[i] == opt) return args[i + 1];
	return "";
}

bool has(const std::vector<std::string> &args, const std::string &opt)
{
	return std::find(args.begin(), args.end(), opt) != args.end();
}

// A boolean of the optimiser STAR file of a --continue run, such as "_rlnDoAutoRefine"
bool optimiserFlag(const std::string &fn, const std::string &label)
{
	std::ifstream in(fn);
	std::string line;
	while (std::getline(in, line))
	{
		std::istringstream ls(line);
		std::string key, value;
		if (ls >> key >> value && key == label) return value == "1";
	}
	return false;
}

tuning::Context contextOf(const std::string &gpu_name)
{
	tuning::Context c;
	c.device = gpu_name;  // sm_count stays 0: relion_launch does not open the GPU
	return c;
}

} // namespace

void parseGpus(const std::string &smi, const char *visible, Node &node)
{
	std::vector<std::string> want;
	if (visible) want = split(visible, ',');
	node.gpus = 0;
	node.gpu_bytes = 0;
	node.gpu_name.clear();
	std::istringstream in(smi);
	std::string line;
	while (std::getline(in, line))
	{
		const std::vector<std::string> f = split(line, ',');
		if (f.size() < 5) continue;
		if (visible && std::find(want.begin(), want.end(), f[0]) == want.end() &&
		    std::find(want.begin(), want.end(), f[1]) == want.end())
			continue;
		const double free_bytes = (atof(f[3].c_str()) - atof(f[4].c_str())) * 1048576.0;
		if (node.gpus == 0) node.gpu_name = f[2];
		node.gpu_bytes = (node.gpus == 0) ? free_bytes : std::min(node.gpu_bytes, free_bytes);
		node.gpus++;
	}
}

std::string visibleForGpuIds(const std::string &ids, const char *visible)
{
	std::vector<std::string> vis;
	if (visible != NULL && *visible)
	{
		std::stringstream ss(visible);
		for (std::string v; std::getline(ss, v, ',');) vis.push_back(v);
	}
	std::vector<std::string> out;
	std::string num;
	for (size_t i = 0; i <= ids.size(); i++)
	{
		const char ch = i < ids.size() ? ids[i] : ':';
		if (isdigit((unsigned char)ch)) { num += ch; continue; }
		if (ch != ':' && ch != ',' && ch != ' ') return "";   // not a plain list of device numbers
		if (num.empty()) continue;
		const size_t k = atoi(num.c_str());
		num.clear();
		if (!vis.empty() && k >= vis.size()) return "";      // not one of the devices this job may use
		const std::string dev = vis.empty() ? std::to_string(k) : vis[k];
		if (std::find(out.begin(), out.end(), dev) == out.end()) out.push_back(dev);
	}
	std::string s;
	for (const std::string &d : out) s += (s.empty() ? "" : ",") + d;
	return s;
}

Node probeNode()
{
	Node n;
	parseGpus(run("timeout 10 nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null"),
	          getenv("CUDA_VISIBLE_DEVICES"), n);
	n.cores = (int)resrep::allowedCores().size();
	n.host_bytes = hostBytes();
	return n;
}

parhint::Work workOfArgs(const std::vector<std::string> &args, bool &vdam)
{
	const std::string cont = valueOf(args, "--continue");
	if (!cont.empty())
	{
		vdam = optimiserFlag(cont, "_rlnDoGradientRefine");
		if (optimiserFlag(cont, "_rlnDoAutoRefine")) return parhint::Work::AutoRefine;
		if (optimiserFlag(cont, "_rlnDoSkipAlign")) return parhint::Work::SkipAlign;
		// shortcut: a continued classification is taken as 3D, which only matters for the starting count
		return parhint::Work::Class3D;
	}
	vdam = has(args, "--grad");
	if (has(args, "--auto_refine")) return parhint::Work::AutoRefine;
	if (has(args, "--skip_align")) return parhint::Work::SkipAlign;
	if (!has(args, "--ref") && !has(args, "--denovo_3dref")) return parhint::Work::Class2D;
	return parhint::Work::Class3D;
}

const char *workName(parhint::Work w)
{
	switch (w)
	{
	case parhint::Work::Class2D:    return "class2d";
	case parhint::Work::Class3D:    return "class3d";
	case parhint::Work::SkipAlign:  return "skip_align";
	case parhint::Work::AutoRefine: return "auto_refine";
	}
	return "other";
}

std::string signatureOf(const std::vector<std::string> &args)
{
	// what makes E-step times of two runs comparable: the data and the size of the search
	std::string sig;
	for (const char *opt : {"--i", "--continue", "--K", "--healpix_order", "--auto_local_healpix_order", "--offset_range",
	                        "--offset_step", "--particle_diameter", "--sym"})
		sig += std::string(opt) + "=" + valueOf(args, opt) + ";";
	char hex[17];
	snprintf(hex, sizeof hex, "%016llx", (unsigned long long)std::hash<std::string>()(sig));
	return hex;
}

std::string encode(const Record &r)
{
	std::ostringstream o;
	o << r.next << ':' << r.prev << ':' << r.prev_time << ':' << (r.settled ? 1 : 0) << ':' << r.gpu_gb;
	return o.str();
}

bool decode(const std::string &s, Record &r)
{
	const std::vector<std::string> f = split(s, ':');
	if ((f.size() != 4 && f.size() != 5) || atoi(f[0].c_str()) < 1) return false;
	r.next = atoi(f[0].c_str());
	r.prev = atoi(f[1].c_str());
	r.prev_time = atof(f[2].c_str());
	r.settled = f[3] == "1";
	r.gpu_gb = (f.size() == 5) ? atof(f[4].c_str()) : 0.;
	return true;
}

Record next(const Record &r, int used, double time, double cores_busy, int threads)
{
	const int MAX_WORKERS_PER_GPU = 8;
	Record n = r;
	n.next = used;
	if (time <= 0) return n;                       // nothing measured: keep what ran
	const bool up_ok = cores_busy < 0.75 && threads >= 2;
	auto step_to = [&](int to)
	{
		n.next = to;
		n.prev = used;
		n.prev_time = time;
		n.settled = false;
	};
	auto settle = [&](int at)
	{
		n.next = at;
		n.prev = 0;
		n.prev_time = 0;
		n.settled = true;
	};
	if (r.prev > 0 && used != r.prev && used == r.next)
	{
		// this run tried one step from `prev`
		if (time > 0.97 * r.prev_time)
			settle(r.prev);                        // not 3% faster: go back, and stay
		else
		{
			const int further = used + (used - r.prev);   // faster: one more step the same way
			if (further > used ? (up_ok && further <= MAX_WORKERS_PER_GPU) : further >= 1)
				step_to(further);
			else
				settle(used);
		}
		return n;
	}
	if (r.settled && used == r.next)
		return n;                                  // shortcut: a settled count is not probed again
	// a count measured for the first time: try a neighbour
	if (up_ok && used < MAX_WORKERS_PER_GPU)
		step_to(used + 1);                         // cores to spare: one more worker
	else if (used > 1)
		step_to(used - 1);                         // the cores are the limit: fewer workers, more threads each
	else
		settle(used);
	return n;
}

bool load(const std::string &gpu_name, const std::string &key, Record &r)
{
	if (gpu_name.empty() || key.empty()) return false;
	tuning::Cache cache(contextOf(gpu_name));
	tuning::Entry e;
	return cache.get("ranks_per_gpu", key, e) && decode(e.value, r);
}

bool store(const std::string &gpu_name, const std::string &key, const Record &r)
{
	if (gpu_name.empty() || key.empty()) return false;
	tuning::Cache cache(contextOf(gpu_name));
	return cache.put("ranks_per_gpu", key, encode(r));
}

std::string encodeMps(const MpsRecord &m)
{
	std::ostringstream o;
	o << m.on << ':' << m.off;
	return o.str();
}

bool decodeMps(const std::string &s, MpsRecord &m)
{
	const std::vector<std::string> f = split(s, ':');
	if (f.size() != 2) return false;
	m.on = atof(f[0].c_str());
	m.off = atof(f[1].c_str());
	return true;
}

bool loadMps(const std::string &gpu_name, const std::string &key, MpsRecord &m)
{
	if (gpu_name.empty() || key.empty()) return false;
	tuning::Cache cache(contextOf(gpu_name));
	tuning::Entry e;
	return cache.get("mps", key, e) && decodeMps(e.value, m);
}

bool storeMps(const std::string &gpu_name, const std::string &key, const MpsRecord &m)
{
	if (gpu_name.empty() || key.empty()) return false;
	tuning::Cache cache(contextOf(gpu_name));
	return cache.put("mps", key, encodeMps(m));
}

bool chooseMps(const MpsRecord &job, const MpsRecord &kind, int workers_per_gpu, bool workers_settled, std::string &why)
{
	std::ostringstream o;
	o.precision(3);
	bool on;
	if (workers_per_gpu < 2)
	{
		on = false;
		o << "one worker per GPU";
	}
	else if (job.on > 0 && job.off > 0)
	{
		on = job.on < job.off;
		o << "faster for this job (" << job.on * 1e3 << " ms of E-step per particle with it, " << job.off * 1e3 << " ms without)";
	}
	else if (workers_settled && (job.on > 0) != (job.off > 0))
	{
		on = job.off > 0;
		o << "trying it " << (on ? "on" : "off") << " once, since this job was measured only " << (on ? "without" : "with") << " it";
	}
	else if (kind.on > 0 && kind.off > 0)
	{
		on = kind.on < kind.off;
		o << "faster for other jobs of this kind with " << workers_per_gpu << " workers per GPU";
	}
	else
	{
		on = workers_per_gpu >= 4;
		o << (on ? "4 or more" : "fewer than 4") << " workers per GPU";
	}
	why = o.str();
	return on;
}

std::string learnFromRun(const std::vector<resrep::ProcessStats> &procs, bool mpi, long particles, int steps)
{
	const char *gpu = getenv("RELION_LAUNCH_GPU_NAME"), *key = getenv("RELION_LAUNCH_KEY"),
	           *used_s = getenv("RELION_LAUNCH_WORKERS_PER_GPU"), *kind_key = getenv("RELION_LAUNCH_KIND_KEY");
	if (!gpu || !key || !used_s || atoi(used_s) < 1 || particles < 1 || steps < 1) return "";
	const int used = atoi(used_s);
	double wall = 0., cpu = 0.;
	int threads = 1;
	std::set<int> cores;
	std::set<std::string> hosts;
	double mem_peak = 0.;   // the fullest card (every worker on a card sees the same device total)
	bool mps = false;       // what the workers ran with, not what was asked for (MPS may have failed to start)
	for (size_t i = (mpi && procs.size() > 1) ? 1 : 0; i < procs.size(); i++)
	{
		const resrep::ProcessStats &p = procs[i];
		mps = mps || p.mps;
		wall = std::max(wall, p.wall);
		cpu += p.cpu;
		threads = p.threads;
		cores.insert(p.cores.begin(), p.cores.end());
		hosts.insert(p.host);
		for (double m : p.gpu_mem_peak) mem_peak = std::max(mem_peak, m);
	}
	if (wall < 60. || hosts.size() > 1) return "";
	const double time = wall / ((double)particles * steps);          // E-step seconds per particle
	const double cores_busy = cores.empty() ? 1. : cpu / (wall * cores.size());
	Record r;
	load(gpu, key, r);
	Record n = next(r, used, time, cores_busy, threads);
	if (mem_peak > 0) n.gpu_gb = mem_peak / 1e9 / used;
	if (!store(gpu, key, n)) return "";
	if (n.settled && kind_key && *kind_key)
	{
		// shortcut: the latest settled job of a kind speaks for all of them; a median over jobs if they disagree a lot
		Record k;
		k.next = n.next;
		k.settled = true;   // GPU memory is not shared: it depends on the box size and the classes
		store(gpu, kind_key, k);
	}
	std::ostringstream o;
	o.precision(3);
	o << " relion_launch: " << used << " worker" << (used == 1 ? "" : "s") << " per GPU took " << time * 1e3
	  << " ms of E-step per particle with " << (int)(cores_busy * 100 + 0.5) << "% of the cores busy";
	if (r.prev > 0 && used != r.prev) o << " (" << r.prev << " per GPU took " << r.prev_time * 1e3 << " ms)";
	o << "; the next run of this job on " << gpu << " starts " << n.next << " per GPU.";
	const char *mps_key = getenv("RELION_LAUNCH_MPS_KEY"), *mps_kind_key = getenv("RELION_LAUNCH_MPS_KIND_KEY");
	if (mps_key && *mps_key)
	{
		MpsRecord m;
		loadMps(gpu, mps_key, m);
		(mps ? m.on : m.off) = time;
		storeMps(gpu, mps_key, m);
		if (m.on > 0 && m.off > 0 && mps_kind_key && *mps_kind_key)
			storeMps(gpu, mps_kind_key, m);   // shortcut: the latest job measured both ways speaks for its kind
		o << " This run was " << (mps ? "with" : "without") << " MPS.";
	}
	return o.str();
}

std::string keyOf(const std::vector<std::string> &args)
{
	bool vdam = false;
	return std::string(workName(workOfArgs(args, vdam))) + ":" + signatureOf(args);
}

std::string kindKeyOf(const std::vector<std::string> &args, int cores_per_gpu)
{
	bool vdam = false;
	int bucket = 2;
	while (bucket < 32 && cores_per_gpu > bucket) bucket *= 2;
	return std::string(workName(workOfArgs(args, vdam))) + ":any:cores" + std::to_string(bucket);
}

bool build(const std::vector<std::string> &cmd, const Node &node, long particles, const Record &learned,
		Command &out, std::string &why_not)
{
	if (cmd.empty())
	{
		why_not = "no command";
		return false;
	}
	const std::string prog = cmd[0];
	const size_t slash = prog.find_last_of('/');
	const std::string dir = (slash == std::string::npos) ? "" : prog.substr(0, slash + 1);
	const std::string base = prog.substr(slash == std::string::npos ? 0 : slash + 1);
	if (base != "relion_refine" && base != "relion_refine_mpi")
	{
		why_not = base + " is not relion_refine";
		return false;
	}
	std::vector<std::string> args;
	for (size_t i = 1; i < cmd.size(); i++)
	{
		if (cmd[i] == "--j" && i + 1 < cmd.size()) { i++; continue; }  // chosen here
		args.push_back(cmd[i]);
	}
	if (!has(args, "--gpu"))
	{
		why_not = "the command does not use the GPU (--gpu)";
		return false;
	}
	if (node.gpus < 1)
	{
		why_not = "no GPU is visible to this job";
		return false;
	}

	bool vdam = false;
	parhint::RankQuery q;
	q.work = workOfArgs(args, vdam);
	q.gpus = node.gpus;
	q.cores = node.cores;
	q.gpu_bytes = node.gpu_bytes;
	q.host_bytes = node.host_bytes;
	q.particles = particles;
	q.learned = learned.next;
	q.learned_settled = learned.settled;
	q.learned_from_other_jobs = learned.from_other_jobs;
	q.gpu_bytes_per_worker = learned.from_other_jobs ? 0. : learned.gpu_gb * 1e9;
	parhint::RankAdvice a = parhint::adviseRanks(q);
	if (vdam)
	{
		// the gradient (VDAM) algorithm runs as one process
		a.workers_per_gpu = 1;
		a.mpi = 1;
		a.threads = std::max(1, std::min(parhint::THREADS_PER_GPU_CAP, node.cores));
		a.reason = "The gradient (VDAM) algorithm runs as one process.";
	}
	out.advice = a;

	out.argv.clear();
	if (a.mpi > 1)
	{
		const char *mpirun = getenv("RELION_MPIRUN");
		out.argv = {(mpirun && *mpirun) ? mpirun : "mpirun", "--oversubscribe", "--bind-to", "none", "-n", std::to_string(a.mpi),
		            dir + "relion_refine_mpi"};
	}
	else
		out.argv = {dir + "relion_refine"};
	out.argv.insert(out.argv.end(), args.begin(), args.end());
	out.argv.push_back("--j");
	out.argv.push_back(std::to_string(a.threads));

	out.env = {"RELION_LAUNCH_GPU_NAME=" + node.gpu_name, "RELION_LAUNCH_KEY=" + keyOf(args),
	           "RELION_LAUNCH_KIND_KEY=" + kindKeyOf(args, node.cores / std::max(1, node.gpus)),
	           "RELION_LAUNCH_WORKERS_PER_GPU=" + std::to_string(a.workers_per_gpu)};

	std::ostringstream r;
	r << " relion_launch: " << node.gpus << " x " << node.gpu_name << " (" << (int)(node.gpu_bytes / 1e9 + 0.5)
	  << " GB free on the fullest), " << node.cores << " cores, " << (int)(node.host_bytes / 1e9 + 0.5) << " GB memory";
	if (particles > 0) r << ", " << particles << " particles";
	r << ": " << a.mpi << " process" << (a.mpi == 1 ? "" : "es") << " (" << a.workers_per_gpu << " worker"
	  << (a.workers_per_gpu == 1 ? "" : "s") << " per GPU) x " << a.threads << " thread" << (a.threads == 1 ? "" : "s")
	  << ". " << a.reason;
	out.report = r.str();
	return true;
}

void applyMps(Command &c, const Node &node, const std::vector<std::string> &cmd, const Record &learned)
{
	const char *set = getenv("RELION_AUTO_MPS");
	if (set && *set)
	{
		c.report += std::string(" MPS: as set by RELION_AUTO_MPS=") + set + ".";
		return;
	}
	if (c.advice.mpi < 2) return;   // relion_refine (one process) does not start MPS
	const int w = c.advice.workers_per_gpu;
	const std::string wk = ":w" + std::to_string(w);
	const std::string key = keyOf(cmd) + wk, kind_key = kindKeyOf(cmd, node.cores / std::max(1, node.gpus)) + wk;
	MpsRecord job, kind;
	loadMps(node.gpu_name, key, job);
	loadMps(node.gpu_name, kind_key, kind);
	std::string why;
	const bool on = chooseMps(job, kind, w, learned.settled && !learned.from_other_jobs && learned.next == w, why);
	c.env.push_back(std::string("RELION_AUTO_MPS=") + (on ? "on" : "off"));
	c.env.push_back("RELION_LAUNCH_MPS_KEY=" + key);
	c.env.push_back("RELION_LAUNCH_MPS_KIND_KEY=" + kind_key);
	c.report += std::string(" MPS ") + (on ? "on" : "off") + ": " + why + " (RELION_AUTO_MPS=on/off to choose).";
}

} // namespace rlaunch
