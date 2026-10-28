/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/virtual_movie_averages.h"
#include "src/virtual_particles.h"   // vcache helpers
#include "src/motioncorr_runner.h"
#include "src/micrograph_model.h"
#include "src/image.h"
#include "src/fftw.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <omp.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <ctime>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

namespace vmovies {

namespace {

/* First line of every descriptor. The rest is the motion record, a STAR file,
 * whose readers skip '#' lines; an MRC file starts with its binary dimensions,
 * so the two can never be confused. */
const char DESCRIPTOR_MAGIC[] = "# RELION virtual movie average\n";
const size_t MAGIC_LEN = sizeof(DESCRIPTOR_MAGIC) - 1;

std::mutex g_mutex;
std::map<std::string, std::pair<std::string, Header> > g_headers;   // path -> (file stamp, header)
std::map<std::string, std::shared_ptr<std::mutex> > g_building;     // cache key -> builder lock
std::set<std::string> g_touched;
std::map<std::string, time_t> g_unwritable;   // cache directory whose write failed -> when to try again; warned once
std::atomic<long long> g_written_since_scan(-1);

uint64_t fnv1a(const unsigned char* p, size_t n, uint64_t h = 1469598103934665603ULL)
{
	for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
	return h;
}

std::string fileStamp(const std::string& path)
{
	struct stat st;
	if (stat(path.c_str(), &st) != 0) return "";
	std::ostringstream s;
	s << (long long)st.st_size << ":" << (long long)st.st_mtim.tv_sec << "." << st.st_mtim.tv_nsec;
	return s.str();
}

std::string readWhole(const std::string& path)
{
	std::ifstream in(path.c_str(), std::ios::binary);
	if (!in) REPORT_ERROR("Cannot read " + path + ": " + std::string(strerror(errno)));
	std::ostringstream s;
	s << in.rdbuf();
	return s.str();
}

/// Cache key: everything that determines the pixels is in the descriptor; the
/// project directory is added because movie paths in it are project-relative,
/// and a cache set with RELION_VMOVIE_AVERAGE_CACHE may be shared by projects.
std::string cacheKey(const std::string& fn)
{
	const std::string text = readWhole(fn);
	char cwd[4096];
	const std::string project = (getcwd(cwd, sizeof(cwd)) != NULL) ? cwd : "";
	uint64_t h = fnv1a((const unsigned char*)text.data(), text.size());
	h = fnv1a((const unsigned char*)project.data(), project.size(), h);
	char buf[17];
	snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
	return buf;
}

long long cacheLimitBytes()
{
	const char* v = getenv("RELION_VMOVIE_AVERAGE_CACHE_MAX_GB");
	const double gb = (v != NULL && v[0] != '\0') ? atof(v) : 0.;   // default: no limit
	return (gb <= 0) ? 0 : (long long)(gb * 1024. * 1024. * 1024.);
}

/// As for virtual particles: scan only on the first write, then after every
/// ~5% of the limit written.
void enforceCacheLimit(const std::string& dir, long long just_written)
{
	const long long limit = cacheLimitBytes();
	if (limit <= 0) return;
	const long long before = g_written_since_scan.fetch_add(just_written);
	if (before >= 0 && before + just_written < limit / 20) return;
	g_written_since_scan = 0;
	vcache::prune(dir, limit, ".mrc");
}

// A failed write (a full or briefly failing file system) pauses writing for a minute, not for the whole run
const time_t CACHE_RETRY_SECONDS = 60;

void warnUnwritable(const std::string& dir, const std::string& why)
{
	std::lock_guard<std::mutex> guard(g_mutex);
	const bool first = g_unwritable.count(dir) == 0;
	g_unwritable[dir] = time(NULL) + CACHE_RETRY_SECONDS;
	if (first)
		std::cerr << " WARNING: cannot write the virtual movie average cache in " << dir << " (" << why
		          << "); micrographs not yet cached are computed from their movies, which is slower,"
		          << " and writing is tried again after a minute."
		          << " Set RELION_VMOVIE_AVERAGE_CACHE to a writable directory, or to 'off' to silence this."
		          << std::endl;
}

// Thread count for regenerate() on this thread when it is not the program's own (the read-ahead thread); 0: the program's
thread_local int t_regen_threads = 0;

void regenerate(const std::string& fn, Image<float>& I)
{
	// Readers inside a parallel loop (e.g. CtfRefine) must not fan out again
	const int threads = t_regen_threads > 0 ? t_regen_threads : omp_in_parallel() ? 1 : std::max(1, omp_get_max_threads());
	MotioncorrRunner::regenerateMicrograph(fn, I, threads);
}

} // anonymous namespace

Mode mode()
{
	const char* v = getenv("RELION_VIRTUAL_MOVIE_AVERAGES");
	if (v == NULL) return OFF;
	std::string s(v);
	for (size_t i = 0; i < s.size(); i++) s[i] = tolower(s[i]);
	if (s == "real") return REAL;
	return vcache::envTruthy(v) ? VIRTUAL : OFF;
}

bool isVirtualMovieAverageFile(int fd)
{
	char buf[64];
	return fd >= 0 && pread(fd, buf, MAGIC_LEN, 0) == (ssize_t)MAGIC_LEN
	       && memcmp(buf, DESCRIPTOR_MAGIC, MAGIC_LEN) == 0;
}

bool isVirtualMovieAverageFile(const std::string& path)
{
	const int fd = open(path.c_str(), O_RDONLY);
	if (fd < 0) return false;
	const bool yes = isVirtualMovieAverageFile(fd);
	close(fd);
	return yes;
}

const char REPLACES_TAG[] = "# replaces ";

void writeDescriptor(const std::string& fn_mic, const std::string& fn_record, const std::string& replaces)
{
	if (replaces.find('\n') != std::string::npos) REPORT_ERROR("writeDescriptor: bad checksum");
	const std::string record = readWhole(fn_record);
	std::ostringstream tmp;
	tmp << fn_mic << ".tmp." << getpid();
	{
		std::ofstream out(tmp.str().c_str(), std::ios::binary | std::ios::trunc);
		if (!out) REPORT_ERROR("Cannot write " + tmp.str() + ": " + std::string(strerror(errno)));
		out << DESCRIPTOR_MAGIC;
		if (!replaces.empty()) out << REPLACES_TAG << replaces << "\n";
		out << record;
		if (!out) REPORT_ERROR("Cannot write " + tmp.str());
	}
	if (rename(tmp.str().c_str(), fn_mic.c_str()) != 0)
		REPORT_ERROR("Cannot rename " + tmp.str() + " to " + fn_mic + ": " + std::string(strerror(errno)));
}

std::string replacedChecksum(const std::string& fn)
{
	std::ifstream in(fn.c_str());
	std::string magic, line;
	if (!std::getline(in, magic) || magic + "\n" != DESCRIPTOR_MAGIC) return "";
	if (!std::getline(in, line) || line.compare(0, strlen(REPLACES_TAG), REPLACES_TAG) != 0) return "";
	return line.substr(strlen(REPLACES_TAG));
}

Header readHeader(const std::string& fn)
{
	const std::string stamp = fileStamp(fn);
	{
		std::lock_guard<std::mutex> guard(g_mutex);
		std::map<std::string, std::pair<std::string, Header> >::iterator it = g_headers.find(fn);
		if (it != g_headers.end() && it->second.first == stamp) return it->second.second;
	}
	MotioncorrRunner::SumRecipe r;
	Micrograph mic;
	if (!MotioncorrRunner::readRecipe(fn, r, &mic))
		REPORT_ERROR(fn + " is a virtual movie average without a sum recipe; it cannot be read.");
	Header h;
	h.nx = r.nx;
	h.ny = r.ny;
	h.angpix = mic.angpix * mic.getBinningFactor();
	h.float16 = r.float16;
	std::lock_guard<std::mutex> guard(g_mutex);
	g_headers[fn] = std::make_pair(stamp, h);
	return h;
}

std::string cacheDirectory()
{
	const char* v = getenv("RELION_VMOVIE_AVERAGE_CACHE");
	if (v != NULL && v[0] != '\0')
		return vcache::envTruthy(v) ? std::string(v) : std::string("");
	return PROJECT_CACHE_DIR;
}

namespace {
std::string materialiseImpl(const std::string& fn, bool* built);
}

std::string materialise(const std::string& fn)
{
	return materialiseImpl(fn, NULL);
}

namespace {
std::string materialiseImpl(const std::string& fn, bool* built)
{
	const std::string dir = cacheDirectory();
	if (dir.empty()) return "";
	const std::string key = cacheKey(fn);
	const std::string shard = dir + "/" + key.substr(0, 2);
	const std::string entry = shard + "/" + key + ".mrc";

	std::shared_ptr<std::mutex> lock;
	{
		std::lock_guard<std::mutex> guard(g_mutex);
		std::shared_ptr<std::mutex>& l = g_building[key];
		if (!l) l.reset(new std::mutex());
		lock = l;
	}
	std::lock_guard<std::mutex> building(*lock);

	struct stat st;
	if (stat(entry.c_str(), &st) == 0 && st.st_size > 1024)
	{
		// Recency for the size limit, once per process per entry
		bool first;
		{
			std::lock_guard<std::mutex> guard(g_mutex);
			first = g_touched.insert(entry).second;
		}
		if (first) utimensat(AT_FDCWD, entry.c_str(), NULL, 0);
		return entry;
	}

	{
		// writing paused after a failure: the caller computes this one
		std::lock_guard<std::mutex> guard(g_mutex);
		const auto it = g_unwritable.find(dir);
		if (it != g_unwritable.end() && time(NULL) < it->second) return "";
	}
	Image<float> I;
	regenerate(fn, I);
	if (built != NULL) *built = true;

	vcache::makeDirs(shard);
	std::ostringstream tmp;
	tmp << shard << "/" << key << ".tmp." << getpid() << "." << std::hash<std::thread::id>()(std::this_thread::get_id()) << ".mrc";
	try
	{
		I.write(tmp.str(), -1, false, WRITE_OVERWRITE, readHeader(fn).float16 ? Float16 : Float);
	}
	catch (RelionError& e)
	{
		unlink(tmp.str().c_str());
		warnUnwritable(dir, "writing failed");
		return "";
	}
	if (rename(tmp.str().c_str(), entry.c_str()) != 0)
	{
		unlink(tmp.str().c_str());
		warnUnwritable(dir, strerror(errno));
		return "";
	}
	if (stat(entry.c_str(), &st) == 0) enforceCacheLimit(dir, st.st_size);
	return entry;
}
} // anonymous namespace

void readMicrograph(const std::string& fn, MultidimArray<RFLOAT>& out)
{
	const Header h = readHeader(fn);
	const std::string entry = materialise(fn);
	if (!entry.empty())
	{
		Image<RFLOAT> I;
		I.read(entry);
		out = I();
	}
	else
	{
		Image<float> I;
		regenerate(fn, I);
		out.resize(YSIZE(I()), XSIZE(I()));
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(out)
			DIRECT_MULTIDIM_ELEM(out, n) = DIRECT_MULTIDIM_ELEM(I(), n);
	}
	if (XSIZE(out) != h.nx || YSIZE(out) != h.ny)
		REPORT_ERROR("Virtual movie average " + fn + " came out " + integerToString(XSIZE(out)) + "x"
		             + integerToString(YSIZE(out)) + " instead of " + integerToString(h.nx) + "x" + integerToString(h.ny));
}

// ---------------------------------------------------------------------------
// ReadAhead

namespace {

bool cacheEntryExists(const std::string& fn)
{
	const std::string dir = cacheDirectory();
	if (dir.empty()) return false;
	const std::string key = cacheKey(fn);
	struct stat st;
	return stat((dir + "/" + key.substr(0, 2) + "/" + key + ".mrc").c_str(), &st) == 0 && st.st_size > 1024;
}

bool writingPaused(const std::string& dir)
{
	std::lock_guard<std::mutex> guard(g_mutex);
	const std::map<std::string, time_t>::const_iterator it = g_unwritable.find(dir);
	return it != g_unwritable.end() && time(NULL) < it->second;
}

/// "Name:   1234 kB" of /proc/<file>, in bytes, or -1
long long procKb(const char* file, const char* name)
{
	std::ifstream in(file);
	std::string line;
	const size_t n = strlen(name);
	while (std::getline(in, line))
		if (line.compare(0, n, name) == 0)
			return atoll(line.c_str() + n) * 1024LL;
	return -1;
}

/// Bytes left under the smallest memory.max - memory.current of the cgroup this process is in and its parents (cgroup v2), or -1
long long cgroupLeftBytes()
{
	std::ifstream in("/proc/self/cgroup");
	std::string line, path;
	while (std::getline(in, line))
		if (line.compare(0, 3, "0::") == 0) path = line.substr(3);
	if (path.empty()) path = "/";
	long long left = -1;
	for (;;)
	{
		const std::string dir = "/sys/fs/cgroup" + (path == "/" ? std::string("") : path);
		std::ifstream max_f((dir + "/memory.max").c_str()), cur_f((dir + "/memory.current").c_str());
		std::string max_s;
		long long cur = 0;
		if (max_f && cur_f && (max_f >> max_s) && (cur_f >> cur) && max_s != "max")
		{
			const long long l = std::max(0LL, atoll(max_s.c_str()) - cur);
			if (left < 0 || l < left) left = l;
		}
		if (path == "/" || path.empty()) break;
		const size_t slash = path.find_last_of('/');
		path = (slash == 0 || slash == std::string::npos) ? "/" : path.substr(0, slash);
	}
	return left;
}

/// The Slurm job's memory (SLURM_MEM_PER_NODE in MB, or SLURM_MEM_PER_CPU x SLURM_CPUS_ON_NODE) less what this process holds, or -1.
/// Some clusters set no cgroup limit, so --mem is not enforced and only these variables say what the job asked for.
long long slurmLeftBytes()
{
	long long mb = -1;
	const char* node = getenv("SLURM_MEM_PER_NODE");
	const char* cpu = getenv("SLURM_MEM_PER_CPU");
	const char* cpus = getenv("SLURM_CPUS_ON_NODE");
	if (node != NULL && atoll(node) > 0) mb = atoll(node);
	else if (cpu != NULL && cpus != NULL && atoll(cpu) > 0 && atoll(cpus) > 0) mb = atoll(cpu) * atoll(cpus);
	if (mb < 0) return -1;
	const long long rss = procKb("/proc/self/status", "VmRSS:");
	return std::max(0LL, mb * 1048576LL - std::max(0LL, rss));
}

/// Memory a new process could use: MemAvailable, or less if a cgroup limit or the Slurm job's memory is nearer
long long memoryAvailableBytes()
{
	long long avail = procKb("/proc/meminfo", "MemAvailable:");
	const long long cg = cgroupLeftBytes();
	if (cg >= 0 && (avail < 0 || cg < avail)) avail = cg;
	const long long sl = slurmLeftBytes();
	if (sl >= 0 && (avail < 0 || sl < avail)) avail = sl;
	return avail;
}

typedef std::chrono::steady_clock Clock;

} // anonymous namespace

struct ReadAhead::Impl
{
	enum Phase { PROBE, TRIAL_ONE, TRIAL_TWO, FINAL };

	int depth_env;            // -1 automatic
	int builders_env;         // 0 automatic
	int threads_env;          // 0 automatic
	int cores;
	std::mutex m;
	std::condition_variable cv;
	std::deque<std::string> queue;
	std::deque<std::string> backlog;   // batch(): built when the queue is empty, not checked yet
	int checking;                      // taken from the backlog and being checked (not inflight yet)
	std::set<std::string> inflight;
	std::set<std::string> settled;   // not descriptors, or cached
	std::vector<std::thread> workers;
	bool stop, disabled, started;
	int k_active, t_cur;
	int k_limit;              // 0: none (the manual picker passes its --vmovie_builders)
	Phase phase;
	long long per_build;      // bytes, measured; 0 until the first build is done
	long phase_done, phase_target;
	Clock::time_point phase_t0;
	double thr_one;           // builds per second with t = 1
	int k_one, k_two, t_two;
	double thr_two;
	long expected;            // micrographs the program will go through (0: unknown); trials need enough of them
	long long base_rss, peak_est;   // sampled: process RSS with no build running; the largest (RSS - base) / builds seen
	std::thread sampler;
	std::atomic<long> built, requested;

	Impl() : depth_env(-1), builders_env(0), threads_env(0), cores(1), checking(0), stop(false), disabled(false), started(false),
	         k_active(1), t_cur(1), k_limit(0), phase(PROBE), per_build(0), phase_done(0), phase_target(0), thr_one(0.), k_one(1), k_two(1), t_two(2), thr_two(0.), expected(0), base_rss(0), peak_est(0),
	         built(0), requested(0) {}

	int depthNow() const
	{
		if (depth_env >= 0) return depth_env;
		return per_build > 0 || builders_env > 0 ? 2 * k_active : 2;
	}

	// Builders the memory allows for builds of `per` bytes each (m held)
	int memoryBuilders(long long per, int k_cap) const
	{
		const long long avail = memoryAvailableBytes();
		if (per <= 0 || avail <= 0) return k_cap;
		return (int) std::max<long long>(1, (long long) (0.6 * (double) avail / (double) per));
	}

	int clampBuilders(int k_cap) const
	{
		int k = std::max(1, std::min(std::min(k_cap, memoryBuilders(per_build, k_cap)), 64));
		if (k_limit > 0) k = std::min(k, k_limit);
		return k;
	}

	// Builders that memory holds below the cores leave cores spare: give each build the threads (a build scales to about 4)
	void useSpareThreads()
	{
		if (threads_env == 0 && builders_env == 0 && k_limit == 0 && k_active > 0)
			t_cur = std::max(t_cur, std::min(4, cores / k_active));
	}

	// Called with m held, when the first build has been measured (or at once with an explicit K)
	void chooseBuilders()
	{
		int t = threads_env > 0 ? threads_env : 1;
		const int k_cap = builders_env > 0 ? builders_env : std::max(1, cores / t);
		const int k = clampBuilders(k_cap);
		if (builders_env > 0 && threads_env == 0) t = std::max(1, cores / k);
		const bool by_memory = k < std::min(k_cap, 64) && (k_limit == 0 || k < k_limit);
		k_active = k;
		t_cur = t;
		k_one = k;
		spawn(k);
		// The first K builds together measure the memory of a build (the probe's estimate counts the program's own work as well);
		// a second trial, of threads per build, needs ~3K builds more, so only a list of 4K or more gets it
		if (builders_env == 0 && threads_env == 0 && k_limit == 0 && k >= 2)
		{
			phase = TRIAL_ONE;
			phase_done = 0;
			phase_target = std::max(4, k);
			phase_t0 = Clock::now();
			peak_est = 0;
		}
		else
		{
			phase = FINAL;
			useSpareThreads();
		}
		report("building", k_cap, by_memory, false, 0., 0.);
	}

	void report(const char* what, int k_cap, bool by_memory, bool compared, double a, double b)
	{
		static std::atomic<int> lines(0);
		if (lines++ >= 4) return;
		std::ostringstream s;
		s << " virtual movie averages: " << what << " " << k_active << " at a time with " << t_cur << " thread"
		  << (t_cur == 1 ? "" : "s") << " each (" << cores << " cores";
		if (per_build > 0)
			s << ", ~" << (double) per_build / 1073741824. << " GB per build, " << (double) memoryAvailableBytes() / 1073741824. << " GB available";
		s << (by_memory ? "; limited by memory" : (k_cap > 0 ? "; limited by the cores" : "")) << ")";
		if (compared)
			s << "; " << k_one << " x 1 thread: " << a * 60. << " micrographs/min, " << k_two << " x " << t_two << " threads: " << b * 60. << " micrographs/min";
		std::cout << s.str() << std::endl;
	}

	// m held. Moves from one trial to the next once enough builds have finished
	void advance()
	{
		if (phase != TRIAL_ONE && phase != TRIAL_TWO) return;
		if (phase_done < phase_target) return;
		const double secs = std::max(1e-3, std::chrono::duration<double>(Clock::now() - phase_t0).count());
		const double thr = (double) phase_done / secs;
		// The memory of K builds running together, as the process's RSS grew over them (an upper bound: the program's own growth counts)
		if (peak_est > 0) per_build = (phase == TRIAL_ONE) ? std::max<long long>(64LL << 20, peak_est) : std::max(per_build, peak_est);
		if (phase == TRIAL_ONE)
		{
			thr_one = thr;
			if (k_one * 2 <= cores && expected >= 4L * k_one)
			{
				// spare cores: more threads per build
				k_two = k_one;
				t_two = std::min(4, cores / k_one);
				k_active = k_two;
				t_cur = t_two;
				phase = TRIAL_TWO;
				phase_done = 0;
				phase_target = std::max(4, 2 * k_two);
				phase_t0 = Clock::now();
				peak_est = 0;
				return;
			}
			// no spare cores or a short list: 1 thread each, with as many builders as the measured memory allows
			t_cur = 1;
			k_active = clampBuilders(std::max(1, cores));
			spawn(k_active);
			phase = FINAL;
			useSpareThreads();
			report("kept (memory measured with the first builders)", cores, k_active < std::min(cores, 64), false, 0., 0.);
			return;
		}
		const bool two_wins = thr > 1.1 * thr_one;
		t_cur = two_wins ? t_two : 1;
		k_active = clampBuilders(std::max(1, cores / t_cur));
		spawn(k_active);
		phase = FINAL;
		useSpareThreads();
		report("kept", cores / t_cur, k_active < std::min(cores / t_cur, 64), true, thr_one, thr);
	}

	void spawn(int k)
	{
		if (stop) return;
		if (!sampler.joinable()) sampler = std::thread(&Impl::sample, this);
		while ((int) workers.size() < k)
		{
			const int id = (int) workers.size();
			workers.push_back(std::thread(&Impl::run, this, id));
		}
	}

	// Samples the process's RSS to see what the builds take
	void sample()
	{
		for (;;)
		{
			{
				std::unique_lock<std::mutex> lock(m);
				cv.wait_for(lock, std::chrono::milliseconds(100), [this] { return stop; });
				if (stop) return;
			}
			const long long rss = procKb("/proc/self/status", "VmRSS:");
			std::lock_guard<std::mutex> lock(m);
			const long long n = (long long) inflight.size();
			if (n == 0)
				base_rss = rss;
			else if (base_rss > 0 && rss > base_rss)
				peak_est = std::max(peak_est, (rss - base_rss) / n);
		}
	}

	// m not held; called by the program's thread. False when the cache cannot be written: then nothing is built
	bool start(const std::string& dir)
	{
		if (started) return true;
		// The cache must be writable, or the first build would be wasted
		vcache::makeDirs(dir);
		if (access(dir.c_str(), W_OK) != 0)
		{
			disabled = true;
			return false;
		}
		std::lock_guard<std::mutex> lock(m);
		started = true;
		if (builders_env > 0)
			chooseBuilders();   // the user's choice: no probe
		else
			spawn(1);           // the first build is the probe: alone, its memory is measured
		return true;
	}

	void run(int id)
	{
		for (;;)
		{
			std::string fn;
			int t;
			long long need = 0;
			bool from_backlog = false;
			{
				std::unique_lock<std::mutex> lock(m);
				cv.wait(lock, [this, id] { return stop || ((!queue.empty() || !backlog.empty()) && id < k_active); });
				if (stop) return;
				from_backlog = queue.empty();
				std::deque<std::string>& from = from_backlog ? backlog : queue;
				fn = from.front();
				from.pop_front();
				if (from_backlog)
				{
					if (settled.count(fn) || inflight.count(fn))
					{
						cv.notify_all();   // waitIdle() may be waiting for the backlog to empty
						continue;
					}
					checking++;
				}
			}
			if (from_backlog)
			{
				// upcoming() checks its micrographs before queueing them; a batch is checked here, off the caller's thread
				bool needed = false;
				try
				{
					needed = isVirtualMovieAverageFile(fn) && !cacheEntryExists(fn);
				}
				catch (RelionError&) {}
				if (!needed)
				{
					std::lock_guard<std::mutex> lock(m);
					checking--;
					settled.insert(fn);
					cv.notify_all();
					continue;
				}
			}
			{
				std::unique_lock<std::mutex> lock(m);
				if (from_backlog) checking--;
				if (stop) return;
				if (inflight.count(fn))
				{
					cv.notify_all();
					continue;
				}
				if (from_backlog) requested++;
				if (inflight.empty()) base_rss = procKb("/proc/self/status", "VmRSS:");
				inflight.insert(fn);
				t = t_cur;
				need = per_build;
			}
			if (need > 0)
			{
				// back off while the memory of another build is not there
				for (;;)
				{
					const long long avail = memoryAvailableBytes();
					if (avail < 0 || avail >= need + need / 2) break;
					std::unique_lock<std::mutex> lock(m);
					if (stop) break;
					cv.wait_for(lock, std::chrono::seconds(1));
				}
			}
			bool made = false;
			const long long rss_before = procKb("/proc/self/status", "VmRSS:");
			t_regen_threads = t;
			try
			{
				// FFTW's cleanup (FourierTransformer::cleanup, run by the main thread) invalidates every plan: it waits for this
				// build, which makes and destroys plans, and the build starts only when no cleanup is waiting
				FftwBackgroundScope fftw_in_use;
				const std::string entry = materialiseImpl(fn, &made);
				if (!(made && !entry.empty())) made = false;
			}
			catch (RelionError&) { made = false; }   // the foreground read reports it, if it is real
			catch (std::exception&) { made = false; }
			const long long hwm = procKb("/proc/self/status", "VmHWM:");
			{
				std::lock_guard<std::mutex> lock(m);
				inflight.erase(fn);
				if (made)
				{
					built++;
					if (phase == PROBE && per_build == 0)
					{
						// alone: the sampled growth of the RSS, at least half the growth of the peak (which also counts what the program freed again)
						const long long grew = (hwm >= 0 && rss_before >= 0) ? hwm - rss_before : 0;
						per_build = std::max<long long>(64LL << 20, std::max(peak_est, grew / 2));
						chooseBuilders();
					}
					else
					{
						phase_done++;
						advance();
					}
				}
			}
			cv.notify_all();
		}
	}
};

ReadAhead::ReadAhead(int depth) : impl_(new Impl)
{
	if (depth < 0)
	{
		const char* v = getenv("RELION_VMOVIE_READ_AHEAD");
		depth = (v != NULL && v[0] != '\0') ? atoi(v) : -1;
		if (depth < 0) depth = -1;
	}
	impl_->depth_env = depth;
	const char* b = getenv("RELION_VMOVIE_BUILDERS");
	impl_->builders_env = (b != NULL && b[0] != '\0') ? std::max(0, atoi(b)) : 0;
	const char* t = getenv("RELION_VMOVIE_BUILD_THREADS");
	impl_->threads_env = (t != NULL && t[0] != '\0') ? std::max(0, atoi(t)) : 0;
	impl_->cores = std::max(1, omp_get_max_threads());
	impl_->t_cur = std::min(4, impl_->cores);   // the probe: a build scales to about 4 threads, so it is quick
}

ReadAhead::~ReadAhead()
{
	{
		std::lock_guard<std::mutex> lock(impl_->m);
		impl_->stop = true;
		impl_->queue.clear();
		impl_->backlog.clear();
	}
	impl_->cv.notify_all();
	for (size_t i = 0; i < impl_->workers.size(); i++)
		if (impl_->workers[i].joinable()) impl_->workers[i].join();
	if (impl_->sampler.joinable()) impl_->sampler.join();
	const char* s = getenv("RELION_VMOVIE_READ_AHEAD_STATS");
	if (s != NULL && vcache::envTruthy(s) && impl_->requested > 0)
		std::cerr << " virtual movie read-ahead: " << impl_->built << " micrographs built ahead of " << impl_->requested
		          << " asked for, " << impl_->k_active << " at a time with " << impl_->t_cur << " thread(s) each" << std::endl;
}

void ReadAhead::expectMicrographs(long n)
{
	std::lock_guard<std::mutex> lock(impl_->m);
	impl_->expected = std::max(0L, n);
}

void ReadAhead::limitBuilders(int k)
{
	std::lock_guard<std::mutex> lock(impl_->m);
	impl_->k_limit = std::max(0, k);
}

void ReadAhead::setBuildThreads(int t)
{
	std::lock_guard<std::mutex> lock(impl_->m);
	const char* env = getenv("RELION_VMOVIE_BUILD_THREADS");
	if (env == NULL || env[0] == '\0')
		impl_->threads_env = std::max(0, t);
}

bool ReadAhead::enabled() const { return impl_->depth_env != 0; }
int ReadAhead::depth() const
{
	std::lock_guard<std::mutex> lock(impl_->m);
	return impl_->depthNow();
}
int ReadAhead::builders() const { std::lock_guard<std::mutex> lock(impl_->m); return impl_->k_active; }
int ReadAhead::buildThreads() const { std::lock_guard<std::mutex> lock(impl_->m); return impl_->t_cur; }
long ReadAhead::built() const { return impl_->built; }
long ReadAhead::requested() const { return impl_->requested; }

void ReadAhead::upcoming(const std::vector<std::string>& next)
{
	if (impl_->depth_env == 0 || impl_->disabled || next.empty()) return;
	const std::string dir = cacheDirectory();
	if (dir.empty() || writingPaused(dir)) return;

	int depth;
	{
		std::lock_guard<std::mutex> lock(impl_->m);
		depth = impl_->depthNow();
	}
	std::vector<std::string> want;
	for (size_t i = 0; i < next.size() && i < (size_t) WINDOW && (int) want.size() < depth; i++)
	{
		{
			std::lock_guard<std::mutex> lock(impl_->m);
			if (impl_->settled.count(next[i])) continue;
		}
		bool needed = false;
		try
		{
			needed = isVirtualMovieAverageFile(next[i]) && !cacheEntryExists(next[i]);
		}
		catch (RelionError&) {}
		if (needed)
			want.push_back(next[i]);
		else
		{
			std::lock_guard<std::mutex> lock(impl_->m);
			impl_->settled.insert(next[i]);
		}
	}
	if (want.empty() || !impl_->start(dir)) return;
	{
		std::lock_guard<std::mutex> lock(impl_->m);
		impl_->queue.clear();
		for (size_t i = 0; i < want.size(); i++)
			if (!impl_->inflight.count(want[i]))
			{
				impl_->queue.push_back(want[i]);
				impl_->requested++;
			}
	}
	impl_->cv.notify_all();
}

void ReadAhead::waitIdle()
{
	std::unique_lock<std::mutex> lock(impl_->m);
	impl_->cv.wait(lock, [this] { return impl_->queue.empty() && impl_->backlog.empty() && impl_->checking == 0 && impl_->inflight.empty(); });
}

void ReadAhead::batch(const std::vector<std::string>& list)
{
	if (impl_->depth_env == 0 || impl_->disabled || list.empty()) return;
	const std::string dir = cacheDirectory();
	if (dir.empty() || writingPaused(dir) || !impl_->start(dir)) return;
	{
		std::lock_guard<std::mutex> lock(impl_->m);
		impl_->backlog.insert(impl_->backlog.end(), list.begin(), list.end());
	}
	impl_->cv.notify_all();
}

bool removeProjectCache(const std::string& project_dir)
{
	const std::string dir = project_dir + "/" + PROJECT_CACHE_DIR;
	struct stat st;
	if (lstat(dir.c_str(), &st) != 0) return false;
	const bool ok = vcache::removeTree(dir);
	rmdir((project_dir + "/Cache").c_str());   // only if nothing else lives there
	return ok;
}

long long memoryAvailable() { return memoryAvailableBytes(); }

void clearProcessState()
{
	std::lock_guard<std::mutex> guard(g_mutex);
	g_headers.clear();
	g_building.clear();
	g_touched.clear();
	g_unwritable.clear();
	g_written_since_scan = -1;
}

} // namespace vmovies
