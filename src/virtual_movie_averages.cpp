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

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <omp.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
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
std::set<std::string> g_touched, g_unwritable;
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
	const double gb = (v != NULL && v[0] != '\0') ? atof(v) : 100.;
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

void warnUnwritable(const std::string& dir, const std::string& why)
{
	std::lock_guard<std::mutex> guard(g_mutex);
	if (g_unwritable.insert(dir).second)
		std::cerr << " WARNING: cannot write the virtual movie average cache in " << dir << " (" << why
		          << "); computing micrographs from their movies on every read instead, which is slower."
		          << " Set RELION_VMOVIE_AVERAGE_CACHE to a writable directory, or to 'off' to silence this."
		          << std::endl;
}

void regenerate(const std::string& fn, Image<float>& I)
{
	// Readers inside a parallel loop (e.g. CtfRefine) must not fan out again
	const int threads = omp_in_parallel() ? 1 : std::max(1, omp_get_max_threads());
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

std::string materialise(const std::string& fn)
{
	const std::string dir = cacheDirectory();
	if (dir.empty()) return "";
	{
		std::lock_guard<std::mutex> guard(g_mutex);
		if (g_unwritable.count(dir)) return "";
	}
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

	Image<float> I;
	regenerate(fn, I);

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

bool removeProjectCache(const std::string& project_dir)
{
	const std::string dir = project_dir + "/" + PROJECT_CACHE_DIR;
	struct stat st;
	if (lstat(dir.c_str(), &st) != 0) return false;
	const bool ok = vcache::removeTree(dir);
	rmdir((project_dir + "/Cache").c_str());   // only if nothing else lives there
	return ok;
}

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
