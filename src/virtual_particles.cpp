/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/virtual_particles.h"

#include "src/image.h"
#include "src/metadata_table.h"
#include "src/float16.h"
#include "src/cache_manager.h"

#include <dirent.h>
#include <fcntl.h>
#include <ctime>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <unordered_map>

namespace vparticles {

namespace {

const int DESCRIPTOR_VERSION = 1;

/* First line of every descriptor. A converted stack keeps its ".mrcs" name so
 * that no STAR file downstream has to change; the reader tells it apart from a
 * real MRC stack by this line (an MRC file starts with its binary dimensions). */
const char DESCRIPTOR_MAGIC[] = "# RELION virtual particle stack\n";

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

uint64_t fnv1a(const unsigned char* p, size_t n, uint64_t h = 1469598103934665603ULL)
{
	for (size_t i = 0; i < n; i++)
	{
		h ^= p[i];
		h *= 1099511628211ULL;
	}
	return h;
}

std::string hex16(uint64_t h)
{
	char buf[17];
	snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
	return buf;
}

/// A double as text that reads back to the same bits.
std::string exactString(double v)
{
	char buf[40];
	snprintf(buf, sizeof(buf), "%.17g", v);
	return buf;
}

bool envTruthy(const char* v)
{
	if (v == NULL) return false;
	std::string s(v);
	for (size_t i = 0; i < s.size(); i++) s[i] = tolower(s[i]);
	return !(s.empty() || s == "0" || s == "no" || s == "off" || s == "false" || s == "none");
}

void makeDirs(const std::string& path)
{
	std::string cur;
	for (size_t i = 0; i < path.size(); i++)
	{
		cur += path[i];
		if (path[i] == '/' || i + 1 == path.size()) mkdir(cur.c_str(), 0775);
	}
}

// ---------------------------------------------------------------------------
// Descriptors
// ---------------------------------------------------------------------------

struct VStack {
	std::string path;
	std::string mic;
	std::string mic_checksum;
	Recipe recipe;
	std::vector<long> x, y;      // integer centres, as extraction truncated them
	std::vector<double> psi;     // helical: each segment's in-plane angle
	std::string key;             // cache key: everything that determines the pixels

	std::mutex build;            // one thread builds the cache entry
	std::atomic<bool> entry_ok;  // the cache entry was seen complete

	VStack() : entry_ok(false) {}
};

std::mutex g_mutex;
std::unordered_map<std::string, std::shared_ptr<VStack> > g_vstacks;
std::set<std::string> g_verified;       // "mic|checksum" pairs checked in this process
std::set<std::string> g_unwritable;     // cache directories that failed; warned once

std::string recipeKeyString(const VStack& v)
{
	const Recipe& r = v.recipe;
	std::ostringstream s;
	s.precision(17);
	s << "v" << DESCRIPTOR_VERSION << "|" << v.mic_checksum
	  << "|" << r.extract_size << "|" << r.scale << "|" << r.window
	  << "|" << r.normalise << "|" << r.bg_radius << "|" << r.ramp
	  << "|" << r.white_dust << "|" << r.black_dust
	  << "|" << r.invert_contrast << "|" << r.float16 << "|";
	if (r.helical) s << "helical " << exactString(r.helical_radius) << "|";
	for (size_t i = 0; i < v.x.size(); i++)
	{
		s << v.x[i] << "," << v.y[i];
		if (r.helical) s << "," << exactString(v.psi[i]);
		s << ";";
	}
	return s.str();
}

std::shared_ptr<VStack> getVStack(const std::string& path)
{
	std::lock_guard<std::mutex> guard(g_mutex);
	std::unordered_map<std::string, std::shared_ptr<VStack> >::iterator it = g_vstacks.find(path);
	if (it != g_vstacks.end()) return it->second;

	if (!exists(path))
		REPORT_ERROR("Virtual particle stack " + path + " does not exist.");

	MetaDataTable head, parts;
	head.read(path, "vstack");
	if (head.numberOfObjects() != 1)
		REPORT_ERROR("Virtual particle stack " + path + " has no data_vstack block.");

	std::shared_ptr<VStack> v(new VStack());
	v->path = path;
	int version = 0;
	head.getValue(EMDL_VSTACK_VERSION, version, 0);
	if (version != DESCRIPTOR_VERSION)
		REPORT_ERROR("Virtual particle stack " + path + " has version " + integerToString(version)
		             + "; this RELION reads version " + integerToString(DESCRIPTOR_VERSION) + ".");

	Recipe& r = v->recipe;
	head.getValue(EMDL_MICROGRAPH_NAME, v->mic, 0);
	head.getValue(EMDL_VSTACK_MICROGRAPH_CHECKSUM, v->mic_checksum, 0);
	head.getValue(EMDL_VSTACK_EXTRACT_SIZE, r.extract_size, 0);
	head.getValue(EMDL_VSTACK_RESCALE_SIZE, r.scale, 0);
	head.getValue(EMDL_VSTACK_WINDOW_SIZE, r.window, 0);
	head.getValue(EMDL_VSTACK_NORMALISE, r.normalise, 0);
	head.getValue(EMDL_VSTACK_BG_RADIUS, r.bg_radius, 0);
	head.getValue(EMDL_VSTACK_RAMP, r.ramp, 0);
	head.getValue(EMDL_VSTACK_WHITE_DUST, r.white_dust, 0);
	head.getValue(EMDL_VSTACK_BLACK_DUST, r.black_dust, 0);
	head.getValue(EMDL_VSTACK_INVERT_CONTRAST, r.invert_contrast, 0);
	head.getValue(EMDL_VSTACK_FLOAT16, r.float16, 0);
	head.getValue(EMDL_IMAGE_PIXEL_SIZE, r.angpix, 0);
	if (head.getValue(EMDL_VSTACK_HELICAL, r.helical, 0) && r.helical)
	{
		std::string radius;
		if (!head.getValue(EMDL_VSTACK_HELICAL_RADIUS, radius, 0))
			REPORT_ERROR("Virtual particle stack " + path + " is helical but has no rlnVirtualHelicalRadius.");
		r.helical_radius = strtod(radius.c_str(), NULL);
	}
	if (r.extract_size <= 0)
		REPORT_ERROR("Virtual particle stack " + path + " has no valid rlnVirtualExtractSize.");
	if (r.white_dust > 0 || r.black_dust > 0)
		REPORT_ERROR("Virtual particle stack " + path + " asks for dust removal, which replaces pixels with "
		             "random values and so cannot give the same particle twice.");

	parts.read(path, "particles");
	v->x.reserve(parts.numberOfObjects());
	v->y.reserve(parts.numberOfObjects());
	FOR_ALL_OBJECTS_IN_METADATA_TABLE(parts)
	{
		RFLOAT cx, cy;
		parts.getValue(EMDL_IMAGE_COORD_X, cx);
		parts.getValue(EMDL_IMAGE_COORD_Y, cy);
		// Stored as whole numbers; round rather than truncate so that text
		// round-off in the STAR file cannot move a particle by a pixel
		v->x.push_back((long)std::floor(cx + 0.5));
		v->y.push_back((long)std::floor(cy + 0.5));
		if (r.helical)
		{
			std::string psi;
			if (!parts.getValue(EMDL_VSTACK_PSI, psi))
				REPORT_ERROR("Virtual particle stack " + path + " is helical but lacks rlnVirtualPsi.");
			v->psi.push_back(strtod(psi.c_str(), NULL));
		}
	}

	v->key = CacheManager::hashString(recipeKeyString(*v));
	g_vstacks[path] = v;
	return v;
}

// ---------------------------------------------------------------------------
// Micrographs
// ---------------------------------------------------------------------------

/* Pixel access to a micrograph, memory-mapped when it is an MRC file this code
 * can decode exactly as RELION's reader does (native byte order, modes 1, 2, 6
 * and 12); anything else is read whole through Image<RFLOAT>, which is what
 * relion_preprocess does, so the values are the same either way. */
struct Micrograph {
	std::string fn;
	long nx, ny;

	int fd;
	size_t len;
	const unsigned char* base;
	const unsigned char* data;
	int mode, bytes;

	std::unique_ptr<Image<RFLOAT> > img;

	Micrograph() : nx(0), ny(0), fd(-1), len(0), base(NULL), data(NULL), mode(0), bytes(0) {}
	~Micrograph()
	{
		if (base != NULL) munmap((void*)base, len);
		if (fd >= 0) close(fd);
	}

	inline RFLOAT at(long y, long x) const
	{
		if (img) return DIRECT_A2D_ELEM((*img)(), y, x);
		const unsigned char* q = data + ((size_t)y * nx + x) * bytes;
		switch (mode)
		{
		case 12: { float16 h; memcpy(&h, q, 2); return half2float(h); }
		case 2:  { float f; memcpy(&f, q, 4); return f; }
		case 1:  { int16_t s; memcpy(&s, q, 2); return s; }
		default: { uint16_t u; memcpy(&u, q, 2); return u; }   // mode 6
		}
	}
};

std::list<std::string> g_mic_order;
std::unordered_map<std::string, std::shared_ptr<Micrograph> > g_mics;
const size_t MAX_OPEN_MICROGRAPHS = 8;

std::shared_ptr<Micrograph> openMicrographUncached(const std::string& fn)
{
	std::shared_ptr<Micrograph> m(new Micrograph());
	m->fn = fn;

	int fd = open(fn.c_str(), O_RDONLY);
	if (fd < 0)
		REPORT_ERROR("Cannot open micrograph " + fn + " to read virtual particles from it: "
		             + std::string(strerror(errno)) + ". Virtual particles need their micrographs.");

	int32_t h[256];
	struct stat st;
	bool mappable = (pread(fd, h, 1024, 0) == 1024) && (fstat(fd, &st) == 0);
	const FileName ext = FileName(fn).getExtension();
	if (mappable)
	{
		const int32_t nx = h[0], ny = h[1], nz = h[2], mode = h[3], nsymbt = h[23];
		const int bytes = (mode == 2) ? 4 : (mode == 1 || mode == 6 || mode == 12) ? 2 : 0;
		const size_t header = 1024 + (size_t)std::max(0, nsymbt);
		mappable = (ext == "mrc" || ext == "mrcs") && bytes > 0 && nz == 1
		        && nx > 0 && ny > 0 && nx < 65536 && ny < 65536   // native byte order
		        && nsymbt >= 0
		        && header + (size_t)nx * ny * bytes <= (size_t)st.st_size;
		if (mappable)
		{
			void* p = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
			if (p == MAP_FAILED) mappable = false;
			else
			{
				m->fd = fd;
				m->len = st.st_size;
				m->base = (const unsigned char*)p;
				m->data = m->base + header;
				m->nx = nx; m->ny = ny; m->mode = mode; m->bytes = bytes;
			}
		}
	}

	if (!mappable)
	{
		close(fd);
		m->img.reset(new Image<RFLOAT>());
		m->img->read(fn);
		m->nx = XSIZE((*m->img)());
		m->ny = YSIZE((*m->img)());
	}
	return m;
}

std::shared_ptr<Micrograph> openMicrograph(const std::string& fn)
{
	{
		std::lock_guard<std::mutex> guard(g_mutex);
		std::unordered_map<std::string, std::shared_ptr<Micrograph> >::iterator it = g_mics.find(fn);
		if (it != g_mics.end())
		{
			g_mic_order.remove(fn);
			g_mic_order.push_front(fn);
			return it->second;
		}
	}

	std::shared_ptr<Micrograph> m = openMicrographUncached(fn);

	std::lock_guard<std::mutex> guard(g_mutex);
	if (g_mics.find(fn) == g_mics.end())
	{
		g_mics[fn] = m;
		g_mic_order.push_front(fn);
		// Callers hold shared_ptrs, so dropping an entry never unmaps one in use
		while (g_mics.size() > MAX_OPEN_MICROGRAPHS)
		{
			g_mics.erase(g_mic_order.back());
			g_mic_order.pop_back();
		}
	}
	return m;
}

/// The micrograph must be the one the particles were extracted from.
void verifyMicrograph(const VStack& v)
{
	const std::string tag = v.mic + "|" + v.mic_checksum;
	{
		std::lock_guard<std::mutex> guard(g_mutex);
		if (g_verified.count(tag)) return;
	}
	const std::string now = micrographChecksum(v.mic);
	if (now != v.mic_checksum)
		REPORT_ERROR("Micrograph " + v.mic + " has changed since the virtual particles in " + v.path
		             + " were extracted from it (checksum " + now + ", expected " + v.mic_checksum
		             + "). Restore the original micrograph, or re-extract the particles.");
	std::lock_guard<std::mutex> guard(g_mutex);
	g_verified.insert(tag);
}

// ---------------------------------------------------------------------------
// Extraction: the plain 2D path of relion_preprocess, bit for bit
// ---------------------------------------------------------------------------

/* Preprocessing::extractParticlesFromOneMicrograph windows the box out of the
 * micrograph filling out-of-range pixels with the micrograph mean, then
 * overwrites every one of those with the nearest edge pixel (x first, then y);
 * the mean never survives, so clamping both coordinates is the same box.
 * performPerImageOperations then rescales, re-windows, normalises and inverts,
 * and the stack writer rounds to float or half. */
void extractOne(const Micrograph& m, const Recipe& r, long xpos, long ypos, double psi,
                MultidimArray<RFLOAT>& out)
{
	Image<RFLOAT> I(r.extract_size, r.extract_size);
	const long first = FIRST_XMIPP_INDEX(r.extract_size);
	for (long i = 0; i < r.extract_size; i++)
	{
		const long y = std::min(std::max(ypos + first + i, 0L), m.ny - 1);
		for (long j = 0; j < r.extract_size; j++)
		{
			const long x = std::min(std::max(xpos + first + j, 0L), m.nx - 1);
			DIRECT_A2D_ELEM(I(), i, j) = m.at(y, x);
		}
	}

	I().setXmippOrigin();
	if (r.scale > 0) rescale(I, r.scale);
	if (r.window > 0) rewindow(I, r.window);
	I().setXmippOrigin();
	if (r.normalise)
		normalise(I, r.bg_radius, r.white_dust, r.black_dust, r.ramp,
		          r.helical, r.helical_radius, 0., psi);   // tilt is ignored for 2D segments
	if (r.invert_contrast) invert_contrast(I);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I())
	{
		RFLOAT& v = DIRECT_MULTIDIM_ELEM(I(), n);
		v = r.float16 ? (RFLOAT)half2float(float2half((float)v)) : (RFLOAT)(float)v;
	}
	out = I();
	out.setXmippOrigin();
}

inline double psiOf(const VStack& v, size_t i) { return v.psi.empty() ? 0. : v.psi[i]; }

// ---------------------------------------------------------------------------
// Cache entries
// ---------------------------------------------------------------------------

struct EntryHeader {
	char     magic[8];     // "RVPCACHE"
	uint32_t version;
	uint32_t dtype;        // 12 = float16, 2 = float32 (MRC mode numbers)
	uint64_t n;
	uint32_t box;
	uint32_t reserved;
	char     key[32];
};

const char ENTRY_MAGIC[8] = {'R', 'V', 'P', 'C', 'A', 'C', 'H', 'E'};

std::string entryPath(const std::string& dir, const std::string& key)
{
	return dir + "/" + key.substr(0, 2) + "/" + key + ".vpc";
}

size_t bytesPerValue(const Recipe& r) { return r.float16 ? 2 : 4; }

void encode(const MultidimArray<RFLOAT>& img, const Recipe& r, std::vector<unsigned char>& buf)
{
	const size_t n = MULTIDIM_SIZE(img);
	buf.resize(n * bytesPerValue(r));
	for (size_t i = 0; i < n; i++)
	{
		if (r.float16)
		{
			float16 h = float2half((float)DIRECT_MULTIDIM_ELEM(img, i));
			memcpy(&buf[2 * i], &h, 2);
		}
		else
		{
			float f = (float)DIRECT_MULTIDIM_ELEM(img, i);
			memcpy(&buf[4 * i], &f, 4);
		}
	}
}

void decode(const unsigned char* p, const Recipe& r, int box, MultidimArray<RFLOAT>& out)
{
	out.resize(box, box);
	const size_t n = (size_t)box * box;
	for (size_t i = 0; i < n; i++)
	{
		if (r.float16)
		{
			float16 h; memcpy(&h, p + 2 * i, 2);
			DIRECT_MULTIDIM_ELEM(out, i) = half2float(h);
		}
		else
		{
			float f; memcpy(&f, p + 4 * i, 4);
			DIRECT_MULTIDIM_ELEM(out, i) = f;
		}
	}
	out.setXmippOrigin();
}

/// Serve particle `idx` from a complete cache entry; false if there is none.
bool readFromEntry(const std::string& path, VStack& v, long idx, MultidimArray<RFLOAT>& out)
{
	int fd = open(path.c_str(), O_RDONLY);
	if (fd < 0)
	{
		v.entry_ok = false;
		return false;
	}

	const int box = v.recipe.outputSize();
	const size_t one = (size_t)box * box * bytesPerValue(v.recipe);

	if (!v.entry_ok)
	{
		EntryHeader h;
		struct stat st;
		const bool ok = pread(fd, &h, sizeof(h), 0) == (ssize_t)sizeof(h)
		             && fstat(fd, &st) == 0
		             && memcmp(h.magic, ENTRY_MAGIC, 8) == 0
		             && h.version == 1
		             && h.dtype == (v.recipe.float16 ? 12u : 2u)
		             && h.n == (uint64_t)v.x.size()
		             && h.box == (uint32_t)box
		             && strncmp(h.key, v.key.c_str(), sizeof(h.key)) == 0
		             && (size_t)st.st_size == sizeof(h) + one * v.x.size();
		if (!ok)
		{
			close(fd);
			return false;   // incomplete or foreign: rebuild over it
		}
		v.entry_ok = true;
		// Recency for eviction: once per process, not on every particle read
		futimens(fd, NULL);
	}

	std::vector<unsigned char> buf(one);
	const ssize_t got = pread(fd, buf.data(), one, sizeof(EntryHeader) + one * idx);
	close(fd);
	if (got != (ssize_t)one)
	{
		v.entry_ok = false;
		return false;
	}
	decode(buf.data(), v.recipe, box, out);
	return true;
}

/* Cache capacity. Without a limit a converted project regrows all of its
 * particles in the cache on first use, and the disk space it was converted to
 * save comes straight back. RELION_VPARTICLE_CACHE_MAX_GB (default 100; 0 = no
 * limit) caps it; the least recently used entries are removed first, and a
 * removed entry is simply rebuilt from its micrograph when next needed. */
long long cacheLimitBytes()
{
	const char* v = getenv("RELION_VPARTICLE_CACHE_MAX_GB");
	const double gb = (v != NULL && v[0] != '\0') ? atof(v) : 100.;
	return (gb <= 0) ? 0 : (long long)(gb * 1024. * 1024. * 1024.);
}

std::atomic<long long> g_written_since_scan(-1);   // -1: not scanned yet in this process

/// Remove the oldest entries until the cache is back under 90% of its limit.
/// Scans only now and then (on the first write, then after every ~5% of the
/// limit written), since walking a large cache is not free.
void enforceCacheLimit(const std::string& dir, long long just_written)
{
	const long long limit = cacheLimitBytes();
	if (limit <= 0) return;

	const long long before = g_written_since_scan.fetch_add(just_written);
	if (before >= 0 && before + just_written < limit / 20) return;
	g_written_since_scan = 0;

	struct Entry { time_t mtime; long long size; std::string path; };
	std::vector<Entry> entries;
	long long total = 0;
	const time_t now = time(NULL);

	DIR* top = opendir(dir.c_str());
	if (top == NULL) return;
	struct dirent* d;
	while ((d = readdir(top)) != NULL)
	{
		if (d->d_name[0] == '.') continue;
		const std::string sub = dir + "/" + d->d_name;
		DIR* sd = opendir(sub.c_str());
		if (sd == NULL) continue;
		struct dirent* e;
		while ((e = readdir(sd)) != NULL)
		{
			const std::string name = e->d_name;
			const std::string path = sub + "/" + name;
			struct stat st;
			if (name[0] == '.' || stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
			if (name.find(".tmp.") != std::string::npos)
			{
				// A writer that died a day ago is not coming back for it
				if (now - st.st_mtime > 86400) unlink(path.c_str());
				continue;
			}
			if (name.size() < 4 || name.compare(name.size() - 4, 4, ".vpc") != 0) continue;
			entries.push_back(Entry{st.st_mtime, (long long)st.st_size, path});
			total += st.st_size;
		}
		closedir(sd);
	}
	closedir(top);

	if (total <= limit) return;
	std::sort(entries.begin(), entries.end(),
	          [](const Entry& a, const Entry& b) { return a.mtime < b.mtime; });
	const long long target = limit / 10 * 9;
	for (size_t i = 0; i < entries.size() && total > target; i++)
	{
		// Another process may be removing the same file; either way it is gone
		if (unlink(entries[i].path.c_str()) == 0 || errno == ENOENT) total -= entries[i].size;
	}
}

void warnUnwritable(const std::string& dir, const std::string& why)
{
	std::lock_guard<std::mutex> guard(g_mutex);
	if (g_unwritable.insert(dir).second)
		std::cerr << " WARNING: cannot write the virtual particle cache in " << dir << " (" << why
		          << "); reading particles straight from the micrographs instead, which is slower."
		          << " Set RELION_VPARTICLE_CACHE to a writable directory, or to 'off' to silence this."
		          << std::endl;
}

bool cacheUnwritable(const std::string& dir)
{
	std::lock_guard<std::mutex> guard(g_mutex);
	return g_unwritable.count(dir) > 0;
}

/* Extract every particle of the descriptor into a new cache entry, visiting
 * them sorted by (y, x) so the micrograph's pages are read in file order, and
 * keep particle `idx` for the caller. Returns false if the entry could not be
 * written; `out` is filled either way. */
bool buildEntry(const std::string& dir, VStack& v, long idx, MultidimArray<RFLOAT>& out)
{
	verifyMicrograph(v);
	std::shared_ptr<Micrograph> m = openMicrograph(v.mic);

	const std::string path = entryPath(dir, v.key);
	makeDirs(path.substr(0, path.find_last_of('/')));

	static std::atomic<long> counter(0);
	char host[256] = "host";
	gethostname(host, sizeof(host) - 1);
	std::ostringstream tmp;
	tmp << path << ".tmp." << host << "." << getpid() << "." << counter++;

	int fd = open(tmp.str().c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0664);
	if (fd < 0)
	{
		warnUnwritable(dir, strerror(errno));
		extractOne(*m, v.recipe, v.x[idx], v.y[idx], psiOf(v, idx), out);
		return false;
	}

	const int box = v.recipe.outputSize();
	const size_t one = (size_t)box * box * bytesPerValue(v.recipe);

	EntryHeader h;
	memset(&h, 0, sizeof(h));
	memcpy(h.magic, ENTRY_MAGIC, 8);
	h.version = 1;
	h.dtype = v.recipe.float16 ? 12 : 2;
	h.n = v.x.size();
	h.box = box;
	strncpy(h.key, v.key.c_str(), sizeof(h.key) - 1);
	bool ok = pwrite(fd, &h, sizeof(h), 0) == (ssize_t)sizeof(h);

	std::vector<size_t> order(v.x.size());
	for (size_t i = 0; i < order.size(); i++) order[i] = i;
	std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
		return (v.y[a] != v.y[b]) ? v.y[a] < v.y[b] : v.x[a] < v.x[b];
	});

	bool have_idx = false;
	MultidimArray<RFLOAT> img;
	std::vector<unsigned char> buf;
	for (size_t k = 0; k < order.size() && ok; k++)
	{
		const size_t i = order[k];
		extractOne(*m, v.recipe, v.x[i], v.y[i], psiOf(v, i), img);
		encode(img, v.recipe, buf);
		ok = pwrite(fd, buf.data(), one, sizeof(h) + one * i) == (ssize_t)one;
		if ((long)i == idx) { out = img; have_idx = true; }
	}
	const int saved_errno = errno;
	ok = (close(fd) == 0) && ok;

	if (ok && rename(tmp.str().c_str(), path.c_str()) == 0)
	{
		v.entry_ok = true;
		enforceCacheLimit(dir, (long long)(sizeof(h) + one * v.x.size()));
		return true;
	}

	unlink(tmp.str().c_str());
	warnUnwritable(dir, strerror(saved_errno ? saved_errno : errno));
	if (!have_idx) extractOne(*m, v.recipe, v.x[idx], v.y[idx], psiOf(v, idx), out);
	return false;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

Recipe::Recipe()
	: extract_size(-1), scale(-1), window(-1), normalise(false), bg_radius(-1), ramp(true),
	  white_dust(-1), black_dust(-1), invert_contrast(false), float16(false), angpix(1.),
	  helical(false), helical_radius(-1.)
{}

int Recipe::outputSize() const
{
	if (window > 0) return window;
	if (scale > 0) return scale;
	return extract_size;
}

bool isVirtualStackFormat(const std::string& ext)
{
	return ext == "vstack";
}

bool isVirtualStackFile(int fd)
{
	const size_t n = sizeof(DESCRIPTOR_MAGIC) - 1;
	char buf[sizeof(DESCRIPTOR_MAGIC)];
	return fd >= 0 && pread(fd, buf, n, 0) == (ssize_t)n && memcmp(buf, DESCRIPTOR_MAGIC, n) == 0;
}

bool isVirtualStackFile(const std::string& path)
{
	const int fd = open(path.c_str(), O_RDONLY);
	if (fd < 0) return false;
	const bool yes = isVirtualStackFile(fd);
	close(fd);
	return yes;
}

void computeParticles(const std::string& fn_vstack, std::vector<MultidimArray<RFLOAT> >& out)
{
	std::shared_ptr<VStack> v = getVStack(fn_vstack);
	verifyMicrograph(*v);
	std::shared_ptr<Micrograph> m = openMicrograph(v->mic);

	std::vector<size_t> order(v->x.size());
	for (size_t i = 0; i < order.size(); i++) order[i] = i;
	std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
		return (v->y[a] != v->y[b]) ? v->y[a] < v->y[b] : v->x[a] < v->x[b];
	});

	out.assign(v->x.size(), MultidimArray<RFLOAT>());
	for (size_t k = 0; k < order.size(); k++)
		extractOne(*m, v->recipe, v->x[order[k]], v->y[order[k]], psiOf(*v, order[k]), out[order[k]]);
}

void forget(const std::string& fn_vstack)
{
	std::lock_guard<std::mutex> guard(g_mutex);
	g_vstacks.erase(fn_vstack);
}

std::string micrographChecksum(const std::string& fn_mic)
{
	int fd = open(fn_mic.c_str(), O_RDONLY);
	if (fd < 0)
		REPORT_ERROR("Cannot open micrograph " + fn_mic + ": " + std::string(strerror(errno)));
	struct stat st;
	if (fstat(fd, &st) != 0)
	{
		close(fd);
		REPORT_ERROR("Cannot stat micrograph " + fn_mic);
	}

	// Header, middle and end: enough to tell a different micrograph (or a
	// re-run motion correction) apart, cheap enough to do once per process
	const off_t size = st.st_size;
	uint64_t h = 1469598103934665603ULL;
	std::vector<unsigned char> buf(65536);
	const off_t starts[3] = {0, std::max<off_t>(0, size / 2 - 32768), std::max<off_t>(0, size - 65536)};
	for (int k = 0; k < 3; k++)
	{
		const ssize_t got = pread(fd, buf.data(), buf.size(), starts[k]);
		if (got > 0) h = fnv1a(buf.data(), got, h);
	}
	close(fd);

	std::ostringstream s;
	s << (long long)size << ":" << hex16(h);
	return s.str();
}

void writeDescriptor(const std::string& fn_vstack, const std::string& fn_mic,
                     const Recipe& r, const std::vector<std::pair<long, long> >& coords,
                     const std::vector<double>& psi)
{
	if (r.helical && psi.size() != coords.size())
		REPORT_ERROR("writeDescriptor: a helical virtual stack needs one psi angle per segment.");
	if (r.white_dust > 0 || r.black_dust > 0)
		REPORT_ERROR("writeDescriptor: virtual particles cannot use dust removal.");
	MetaDataTable head;
	head.setIsList(true);
	head.setName("vstack");
	head.addObject();
	head.setValue(EMDL_VSTACK_VERSION, DESCRIPTOR_VERSION);
	head.setValue(EMDL_MICROGRAPH_NAME, fn_mic);
	head.setValue(EMDL_VSTACK_MICROGRAPH_CHECKSUM, micrographChecksum(fn_mic));
	head.setValue(EMDL_VSTACK_EXTRACT_SIZE, r.extract_size);
	head.setValue(EMDL_VSTACK_RESCALE_SIZE, r.scale);
	head.setValue(EMDL_VSTACK_WINDOW_SIZE, r.window);
	head.setValue(EMDL_IMAGE_SIZE, r.outputSize());
	head.setValue(EMDL_IMAGE_PIXEL_SIZE, r.angpix);
	head.setValue(EMDL_VSTACK_NORMALISE, r.normalise);
	head.setValue(EMDL_VSTACK_BG_RADIUS, r.bg_radius);
	head.setValue(EMDL_VSTACK_RAMP, r.ramp);
	head.setValue(EMDL_VSTACK_WHITE_DUST, r.white_dust);
	head.setValue(EMDL_VSTACK_BLACK_DUST, r.black_dust);
	head.setValue(EMDL_VSTACK_INVERT_CONTRAST, r.invert_contrast);
	head.setValue(EMDL_VSTACK_FLOAT16, r.float16);
	if (r.helical)
	{
		head.setValue(EMDL_VSTACK_HELICAL, true);
		head.setValue(EMDL_VSTACK_HELICAL_RADIUS, exactString(r.helical_radius));
	}

	MetaDataTable parts;
	parts.setName("particles");
	for (size_t i = 0; i < coords.size(); i++)
	{
		parts.addObject();
		parts.setValue(EMDL_IMAGE_COORD_X, (RFLOAT)coords[i].first);
		parts.setValue(EMDL_IMAGE_COORD_Y, (RFLOAT)coords[i].second);
		if (r.helical) parts.setValue(EMDL_VSTACK_PSI, exactString(psi[i]));
	}

	// Written aside and renamed, so a reader never sees half a descriptor
	const std::string tmp = fn_vstack + ".tmp." + integerToString(getpid());
	{
		std::ofstream fh(tmp.c_str());
		if (!fh) REPORT_ERROR("Cannot write virtual particle stack " + fn_vstack);
		fh << DESCRIPTOR_MAGIC;
		head.write(fh);
		parts.write(fh);
	}
	if (rename(tmp.c_str(), fn_vstack.c_str()) != 0)
		REPORT_ERROR("Cannot write virtual particle stack " + fn_vstack + ": " + std::string(strerror(errno)));

	// A rewritten descriptor must not be served from this process's memo
	std::lock_guard<std::mutex> guard(g_mutex);
	g_vstacks.erase(fn_vstack);
}

Header readHeader(const std::string& fn_vstack)
{
	std::shared_ptr<VStack> v = getVStack(fn_vstack);
	Header h;
	h.n = v->x.size();
	h.box = v->recipe.outputSize();
	h.angpix = v->recipe.angpix;
	h.float16 = v->recipe.float16;
	return h;
}

void readParticle(const std::string& fn_vstack, long index, MultidimArray<RFLOAT>& out)
{
	std::shared_ptr<VStack> v = getVStack(fn_vstack);
	if (index < 0 || index >= (long)v->x.size())
		REPORT_ERROR("Particle " + integerToString(index + 1) + " requested from virtual stack "
		             + fn_vstack + ", which holds " + integerToString(v->x.size()) + ".");

	const std::string dir = cacheDirectory();
	if (!dir.empty() && !cacheUnwritable(dir))
	{
		const std::string path = entryPath(dir, v->key);
		if (readFromEntry(path, *v, index, out)) return;

		// Build it once: other threads wanting this micrograph wait, then read
		std::lock_guard<std::mutex> guard(v->build);
		if (readFromEntry(path, *v, index, out)) return;
		buildEntry(dir, *v, index, out);
		return;
	}

	// No cache: cut just this particle; only the pages under its box are read
	verifyMicrograph(*v);
	std::shared_ptr<Micrograph> m = openMicrograph(v->mic);
	extractOne(*m, v->recipe, v->x[index], v->y[index], psiOf(*v, index), out);
}

std::string cacheDirectory()
{
	const char* explicit_dir = getenv("RELION_VPARTICLE_CACHE");
	if (explicit_dir != NULL && explicit_dir[0] != '\0')
		return envTruthy(explicit_dir) ? std::string(explicit_dir) : std::string("");

	// In the project, where users can see it and where every node running a
	// job of the project shares it (RELION programs run from the project root)
	return PROJECT_CACHE_DIR;
}

// Delete a directory tree without a shell, so no path can be misparsed; never
// follows symlinks out of the tree. True if everything was removed.
static bool removeTree(const std::string& path)
{
	struct stat st;
	if (lstat(path.c_str(), &st) != 0) return errno == ENOENT;
	if (!S_ISDIR(st.st_mode)) return unlink(path.c_str()) == 0 || errno == ENOENT;
	bool ok = true;
	DIR* d = opendir(path.c_str());
	if (d == NULL) return false;
	while (struct dirent* e = readdir(d))
	{
		if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0) continue;
		ok = removeTree(path + "/" + e->d_name) && ok;
	}
	closedir(d);
	return (rmdir(path.c_str()) == 0 || errno == ENOENT) && ok;
}

bool removeProjectCache(const std::string& project_dir)
{
	const std::string dir = project_dir + "/" + PROJECT_CACHE_DIR;
	struct stat st;
	if (lstat(dir.c_str(), &st) != 0) return false;
	const bool ok = removeTree(dir);
	// Leave Cache/ itself only if something else lives there
	rmdir((project_dir + "/Cache").c_str());
	return ok;
}

void clearProcessState()
{
	std::lock_guard<std::mutex> guard(g_mutex);
	g_vstacks.clear();
	g_verified.clear();
	g_unwritable.clear();
	g_mics.clear();
	g_mic_order.clear();
	g_written_since_scan = -1;
}

} // namespace vparticles
