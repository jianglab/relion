/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/virtual_particles.h"
#include "src/virtual_movie_averages.h"

#include "src/image.h"
#include "src/metadata_table.h"
#include "src/float16.h"
#include "src/extract_rect.h"
#include "src/fftw_rect.h"
#include "src/fftw.h"
#include "src/cache_manager.h"
#include "src/fused_extract.h"
#include "src/resource_report.h"
#include "src/ctf.h"
#include "src/jaz/single_particle/obs_model.h"
#include "src/jaz/single_particle/micrograph_handler.h"
#include "src/micrograph_model.h"
#include "src/async_file_writer.h"
#include "src/jaz/single_particle/damage_helper.h"
#include "src/jaz/single_particle/motion/frame_recombiner.h"

#include <dirent.h>
#include <fcntl.h>
#include <ctime>
#include <map>
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
#include <cmath>
#include <fstream>
#include <future>
#include <chrono>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <thread>
#include <set>
#include <sstream>
#include <unordered_map>


namespace vcache {

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

/// Remove the least recently used entries (files ending in `suffix`, one level
/// of shard directories down) until the cache is back under 90% of `limit`;
/// dead writers' temporary files older than a day go too.
void prune(const std::string& dir, long long limit, const std::string& suffix)
{
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
			if (name.size() < suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
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

// Delete a directory tree without a shell, so no path can be misparsed; never
// follows symlinks out of the tree. True if everything was removed.
bool removeTree(const std::string& path)
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

} // namespace vcache

namespace vparticles {

namespace {

// Version 1 is square, upright boxes; version 2 adds rectangular boxes and turned boxes;
// version 3 CTF-premultiplied particles; version 4 polished particles (recombined from movie frames). A descriptor is written with the lowest version
// it needs, so that older builds keep reading everything they always could (and refuse,
// rather than misread, premultiplied particles).
const int DESCRIPTOR_VERSION = 2;
const int DESCRIPTOR_VERSION_SQUARE = 1;
const int DESCRIPTOR_VERSION_PREMULT = 3;
const int DESCRIPTOR_VERSION_POLISH = 4;
const int DESCRIPTOR_VERSION_POLISH_WINDOWS = 6; // polished, each particle combined in a window of its own
const int DESCRIPTOR_VERSION_POLISH_SUM16 = 5;   // polished, the frame sums rounded to 16 bits before the CTF (so that they can be cached)

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

/// The numbers of a space-separated list; an error if anything else is in it.
std::vector<double> parseNumbers(const std::string& text)
{
	std::vector<double> out;
	const char* q = text.c_str();
	for (;;)
	{
		char* end;
		const double x = strtod(q, &end);
		if (end == q) break;
		out.push_back(x);
		q = end;
	}
	while (*q == ' ') q++;
	if (*q != 0) REPORT_ERROR("Not a list of numbers: " + text);
	return out;
}

using vcache::envTruthy;
using vcache::makeDirs;

// ---------------------------------------------------------------------------
// Descriptors
// ---------------------------------------------------------------------------

struct Micrograph;
struct Polished;

/// Set when the background builders are to stop (the program is ending): entries still being written are given up
std::atomic<bool> g_builds_stopping(false);

/// A cache entry being written. The thread that starts it and every thread that needs one of its particles
/// meanwhile take particles from the same list and cut them, so the work is shared instead of waited for.
struct EntryBuild {
	std::mutex m;
	std::condition_variable cv;
	bool prepared = false, finished = false;
	std::atomic<bool> failed{false};
	std::shared_ptr<Micrograph> mic;
	std::shared_ptr<Polished> pol;   // polished stacks: the particles being written, held until the entry is done
	int fd = -1;
	std::string path, tmp;
	std::vector<size_t> order;   // particles sorted by (y, x), so the micrograph's pages are read in file order
	size_t one = 0;              // bytes per particle
	std::atomic<size_t> next{0}, done{0};
};

struct VStack {
	std::string path;
	std::string mic;
	std::string mic_checksum;
	Recipe recipe;
	std::vector<long> x, y;      // integer centres, as extraction truncated them
	std::vector<double> psi;     // helical: each segment's in-plane angle
	std::vector<double> cx, cy, angle;   // rotated: exact centre and box angle of each particle
	std::vector<ParticleCtf> ctf;        // premultiplied: the CTF in effect for each particle
	// polished: how the frames are recombined, the movie and each particle's track (also as written, for the key)
	std::shared_ptr<PolishRecipe> polish;
	std::string movie, meta, polish_text, weights_text;
	std::vector<std::string> track_text;
	std::vector<std::vector<gravis::d2Vector> > tracks;
	std::vector<int> win;        // polished: the box each particle's frames are combined in (polish->s unless it has its own)
	int version;                 // descriptor version as read
	std::string key;             // cache key: everything that determines the pixels
	std::string sum_key;         // polished: cache key of the frame sums (everything but the CTF and the output)

	std::mutex build;            // guards `building`
	std::shared_ptr<EntryBuild> building;   // the cache entry being written, if any
	std::atomic<bool> entry_ok;  // the cache entry was seen complete

	VStack() : version(DESCRIPTOR_VERSION_SQUARE), entry_ok(false) {}
};

std::mutex g_mutex;
std::unordered_map<std::string, std::shared_ptr<VStack> > g_vstacks;
std::set<std::string> g_verified;       // "mic|checksum" pairs checked in this process
std::map<std::string, time_t> g_unwritable;   // cache directory whose write failed -> when to try again; warned once
// Premultiplied particles: the CTF their particle STAR file gives now, by stack and index (defU NaN: not given)
// shortcut: ~72 bytes per premultiplied particle held for the whole run; key by micrograph if that ever matters
std::unordered_map<std::string, std::vector<ParticleCtf> > g_current_ctfs;
bool g_read_unpremultiplied = false;

std::string ctfString(const ParticleCtf& c)
{
	const double v[9] = {c.defU, c.defV, c.angle, c.kV, c.Cs, c.Q0, c.bfac, c.scale, c.phase};
	std::string s;
	for (int k = 0; k < 9; k++) s += (k ? " " : "") + exactString(v[k]);
	for (double a : c.aberr) s += " " + exactString(a);
	return s;
}

// A value read back from a particle STAR file differs from the recorded one by text round-off
// (6 decimals, or 6 significant digits); a real change, e.g. by CTF refinement, is far larger
bool sameCtf(const ParticleCtf& a, const ParticleCtf& b)
{
	const double x[9] = {a.defU, a.defV, a.angle, a.kV, a.Cs, a.Q0, a.bfac, a.scale, a.phase};
	const double y[9] = {b.defU, b.defV, b.angle, b.kV, b.Cs, b.Q0, b.bfac, b.scale, b.phase};
	for (int k = 0; k < 9; k++)
		if (!(std::fabs(x[k] - y[k]) <= 1e-5 + 1e-6 * std::fabs(y[k]))) return false;
	if (a.aberr.size() != b.aberr.size()) return false;
	for (size_t k = 0; k < a.aberr.size(); k++)
		if (!(std::fabs(a.aberr[k] - b.aberr[k]) <= 1e-5 + 1e-6 * std::fabs(b.aberr[k]))) return false;
	return true;
}

std::string recipeKeyString(const VStack& v)
{
	const Recipe& r = v.recipe;
	std::ostringstream s;
	s.precision(17);
	s << "v" << v.version << "|" << v.mic_checksum
	  << "|" << r.extract_size << "|" << r.scale << "|" << r.window
	  << "|" << r.normalise << "|" << r.bg_radius << "|" << r.ramp
	  << "|" << r.white_dust << "|" << r.black_dust
	  << "|" << r.invert_contrast << "|" << r.float16 << "|";
	if (r.helical) s << "helical " << exactString(r.helical_radius) << "|";
	if (r.rectangular()) s << "rect " << r.extract_size_y << "|";
	if (r.rotated) s << "rotated " << resampleMethodName(r.interpolation) << "|";
	if (r.premultiplied())
		s << "premultiply " << r.premultiply_size << " " << r.ctf_intact_first_peak << " " << exactString(r.ctf_angpix) << "|";
	if (v.polish) s << "polish " << v.polish_text << "|" << v.weights_text << "|";
	for (size_t i = 0; i < v.x.size(); i++)
	{
		s << v.x[i] << "," << v.y[i];
		if (v.polish) s << "," << exactString(v.cx[i]) << "," << exactString(v.cy[i]) << "," << v.track_text[i];
		if (v.polish && v.win[i] != v.polish->s) s << ",w" << v.win[i];
		if (r.helical) s << "," << exactString(v.psi[i]);
		if (r.rotated) s << "," << exactString(v.cx[i]) << "," << exactString(v.cy[i]) << "," << exactString(v.angle[i]);
		if (r.premultiplied()) s << "," << ctfString(v.ctf[i]);
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
	if (version < DESCRIPTOR_VERSION_SQUARE || version > DESCRIPTOR_VERSION_POLISH_WINDOWS)
		REPORT_ERROR("Virtual particle stack " + path + " has version " + integerToString(version)
		             + "; this RELION reads versions " + integerToString(DESCRIPTOR_VERSION_SQUARE) + " to "
		             + integerToString(DESCRIPTOR_VERSION_POLISH_WINDOWS) + ".");
	v->version = version;

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
	head.getValue(EMDL_VSTACK_EXTRACT_SIZE_Y, r.extract_size_y, 0);
	head.getValue(EMDL_VSTACK_ROTATED, r.rotated, 0);
	if (r.rotated)
	{
		std::string method;
		head.getValue(EMDL_VSTACK_INTERPOLATION, method, 0);
		r.interpolation = parseResampleMethod(method);
	}
	head.getValue(EMDL_VSTACK_PREMULTIPLY_SIZE, r.premultiply_size, 0);
	if (r.premultiplied())
	{
		std::string a;
		head.getValue(EMDL_VSTACK_CTF_INTACT_FIRST_PEAK, r.ctf_intact_first_peak, 0);
		if (!head.getValue(EMDL_VSTACK_CTF_ANGPIX, a, 0))
			REPORT_ERROR("Virtual particle stack " + path + " is premultiplied but has no rlnVirtualCtfPixelSize.");
		r.ctf_angpix = strtod(a.c_str(), NULL);
	}
	if (version >= DESCRIPTOR_VERSION_POLISH)
	{
		PolishRecipe* pr = new PolishRecipe();
		v->polish.reset(pr);
		if (!head.getValue(EMDL_MICROGRAPH_MOVIE_NAME, v->movie, 0) || !head.getValue(EMDL_MICROGRAPH_METADATA_NAME, v->meta, 0)
		    || !head.getValue(EMDL_VSTACK_POLISH, v->polish_text, 0)
		    || !head.getValue(EMDL_VSTACK_FRAME_WEIGHTS, v->weights_text, 0))
			REPORT_ERROR("Virtual particle stack " + path + " is polished but lacks rlnMicrographMovieName, rlnMicrographMetadata, rlnVirtualPolish or rlnVirtualFrameWeights.");
		pr->meta_fn = v->meta;
		std::istringstream in(v->polish_text);
		in >> pr->corr_mic >> pr->first_frame >> pr->last_frame >> pr->eer_upsampling >> pr->eer_grouping
		   >> pr->movie_angpix >> pr->coords_angpix >> pr->data_angpix >> pr->hot_cutoff >> pr->s >> pr->angpix >> pr->out_size;
		if (!in || pr->s <= 0 || pr->out_size <= 0 || pr->out_size > pr->s || pr->last_frame < pr->first_frame)
			REPORT_ERROR("Virtual particle stack " + path + " has a malformed rlnVirtualPolish: " + v->polish_text);
		const std::vector<double> w = parseNumbers(v->weights_text);
		if (w.size() != 2 * (size_t)(pr->last_frame - pr->first_frame + 1))
			REPORT_ERROR("Virtual particle stack " + path + " needs a sigma and a scale per frame in rlnVirtualFrameWeights.");
		for (size_t f = 0; f < w.size(); f += 2) pr->bk.push_back(gravis::d2Vector(w[f], w[f + 1]));
		pr->float16 = r.float16;
		r.extract_size = pr->out_size;
		r.angpix = pr->angpix;
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
		if (r.rotated)
		{
			std::string sx, sy, sa;
			if (!parts.getValue(EMDL_VSTACK_CENTRE_X, sx) || !parts.getValue(EMDL_VSTACK_CENTRE_Y, sy)
			    || !parts.getValue(EMDL_VSTACK_ANGLE, sa))
				REPORT_ERROR("Virtual particle stack " + path + " is rotated but lacks rlnVirtualCentreX/Y or rlnVirtualAngle.");
			v->cx.push_back(strtod(sx.c_str(), NULL));
			v->cy.push_back(strtod(sy.c_str(), NULL));
			v->angle.push_back(strtod(sa.c_str(), NULL));
		}
		if (v->polish)
		{
			std::string sx, sy, st;
			if (!parts.getValue(EMDL_VSTACK_CENTRE_X, sx) || !parts.getValue(EMDL_VSTACK_CENTRE_Y, sy)
			    || !parts.getValue(EMDL_VSTACK_TRACK, st))
				REPORT_ERROR("Virtual particle stack " + path + " is polished but lacks rlnVirtualCentreX/Y or rlnVirtualTrack.");
			v->cx.push_back(strtod(sx.c_str(), NULL));
			v->cy.push_back(strtod(sy.c_str(), NULL));
			const std::vector<double> t = parseNumbers(st);
			if (t.size() != 2 * v->polish->bk.size())
				REPORT_ERROR("Virtual particle stack " + path + " needs an x and y shift per frame in rlnVirtualTrack.");
			std::vector<gravis::d2Vector> track;
			for (size_t f = 0; f < t.size(); f += 2) track.push_back(gravis::d2Vector(t[f], t[f + 1]));
			v->tracks.push_back(track);
			v->track_text.push_back(st);
			int w = v->polish->s;
			parts.getValue(EMDL_VSTACK_PREMULTIPLY_SIZE, w);
			if (w % 2 != 0 || w < v->polish->out_size || w > v->polish->s)
				REPORT_ERROR("Virtual particle stack " + path + " has a particle window (rlnVirtualPremultiplySize) that is odd or not between the output box and the window of the stack.");
			v->win.push_back(w);
		}
		if (r.premultiplied())
		{
			std::string sc;
			if (!parts.getValue(EMDL_VSTACK_CTF, sc))
				REPORT_ERROR("Virtual particle stack " + path + " is premultiplied but lacks rlnVirtualCtf.");
			const std::vector<double> c = parseNumbers(sc);
			if (c.size() < 9 || (c.size() > 9 && c.size() < 14))
				REPORT_ERROR("Virtual particle stack " + path + " has a malformed rlnVirtualCtf: " + sc);
			v->ctf.push_back(ParticleCtf{c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8],
			                             std::vector<double>(c.begin() + 9, c.end())});
		}
	}

	if (g_read_unpremultiplied) r.premultiply_size = -1;

	// Premultiplied: the CTF the particle STAR file gives now replaces the recorded one where it changed
	if (r.premultiplied())
	{
		std::unordered_map<std::string, std::vector<ParticleCtf> >::const_iterator c = g_current_ctfs.find(path);
		if (c != g_current_ctfs.end())
			for (size_t i = 0; i < v->ctf.size() && i < c->second.size(); i++)
				if (!std::isnan(c->second[i].defU) && !sameCtf(c->second[i], v->ctf[i]))
					v->ctf[i] = c->second[i];
	}

	v->key = CacheManager::hashString(recipeKeyString(*v));
	if (v->polish)
	{
		std::string s = "sum|v" + integerToString(v->version) + "|" + v->mic_checksum + "|" + v->polish_text + "|" + v->weights_text + "|";
		for (size_t i = 0; i < v->cx.size(); i++)
			s += exactString(v->cx[i]) + "," + exactString(v->cy[i]) + "," + v->track_text[i]
			   + (v->win[i] != v->polish->s ? ",w" + integerToString(v->win[i]) : std::string()) + ";";
		v->sum_key = CacheManager::hashString(s);
	}
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

	mutable std::once_flag mean_once;
	mutable RFLOAT mean_value;

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

	/// The micrograph's average, summed in the order MultidimArray::computeAvg sums it
	RFLOAT mean() const
	{
		std::call_once(mean_once, [this]() {
			if (img) { mean_value = (*img)().computeAvg(); return; }
			RFLOAT sum = 0;
			for (long y = 0; y < ny; y++)
				for (long x = 0; x < nx; x++)
					sum += at(y, x);
			mean_value = sum / (nx * ny);
		});
		return mean_value;
	}
};

std::list<std::string> g_mic_order;
std::unordered_map<std::string, std::shared_ptr<Micrograph> > g_mics;
const size_t MAX_OPEN_MICROGRAPHS = 8;

std::shared_ptr<Micrograph> openMicrographUncached(const std::string& fn)
{
	std::shared_ptr<Micrograph> m(new Micrograph());
	m->fn = fn;

	// A virtual movie average is read from its cached real file (built from the
	// movie if needed), or through Image when the cache is off
	std::string path = fn;
	if (vmovies::isVirtualMovieAverageFile(fn))
	{
		path = vmovies::materialise(fn);
		if (path.empty())
		{
			m->img.reset(new Image<RFLOAT>());
			m->img->read(fn);
			m->nx = XSIZE((*m->img)());
			m->ny = YSIZE((*m->img)());
			return m;
		}
	}

	int fd = open(path.c_str(), O_RDONLY);
	if (fd < 0)
		REPORT_ERROR("Cannot open micrograph " + path + " to read virtual particles from it: "
		             + std::string(strerror(errno)) + ". Virtual particles need their micrographs.");

	int32_t h[256];
	struct stat st;
	bool mappable = (pread(fd, h, 1024, 0) == 1024) && (fstat(fd, &st) == 0);
	const FileName ext = FileName(path).getExtension();
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
	// polished particles come from the movie
	const std::string& source = v.polish ? v.movie : v.mic;
	const std::string tag = source + "|" + v.mic_checksum;
	{
		std::lock_guard<std::mutex> guard(g_mutex);
		if (g_verified.count(tag)) return;
	}
	const std::string now = v.polish ? polishSourceChecksum(v.movie, v.meta) : micrographChecksum(source);
	if (now != v.mic_checksum)
		REPORT_ERROR((v.polish ? "Movie " : "Micrograph ") + source + " has changed since the virtual particles in " + v.path
		             + " were made from it (checksum " + now + ", expected " + v.mic_checksum
		             + "). Restore the original file, or make the particles again.");
	std::lock_guard<std::mutex> guard(g_mutex);
	g_verified.insert(tag);
}

// ---------------------------------------------------------------------------
// Extraction: the plain 2D path of relion_preprocess, bit for bit
// ---------------------------------------------------------------------------

// The CTF with magnification and symmetric aberrations, read exactly as relion_preprocess reads it:
// CTF::readByGroup through an optics group with that pixel size, voltage, Cs, Q0, box P and aberrations
// (one ObservationModel per optics and box, kept for the run: it also keeps the Zernike phases)
void readAberratedCtf(const ParticleCtf& p, long P, CTF& c)
{
	// never destroyed: background cache builders may still read particles while the process exits
	static std::mutex& mutex = *new std::mutex;
	static std::map<std::string, std::unique_ptr<ObservationModel> >& models = *new std::map<std::string, std::unique_ptr<ObservationModel> >;
	const std::vector<double>& a = p.aberr;
	std::ostringstream key;
	key.precision(17);
	key << P << " " << p.kV << " " << p.Cs << " " << p.Q0;
	for (double x : a) key << " " << x;
	ObservationModel* om;
	{
		std::lock_guard<std::mutex> guard(mutex);
		std::unique_ptr<ObservationModel>& slot = models[key.str()];
		if (!slot)
		{
			MetaDataTable optics;
			optics.setName("optics");
			optics.addObject();
			optics.setValue(EMDL_IMAGE_OPTICS_GROUP_NAME, std::string("opticsGroup1"));
			optics.setValue(EMDL_IMAGE_OPTICS_GROUP, 1);
			optics.setValue(EMDL_IMAGE_PIXEL_SIZE, a[0]);
			optics.setValue(EMDL_IMAGE_SIZE, (int)P);
			optics.setValue(EMDL_IMAGE_DIMENSIONALITY, 2);
			optics.setValue(EMDL_CTF_VOLTAGE, p.kV);
			optics.setValue(EMDL_CTF_CS, p.Cs);
			optics.setValue(EMDL_CTF_Q0, p.Q0);
			optics.setValue(EMDL_IMAGE_MAG_MATRIX_00, a[1]);
			optics.setValue(EMDL_IMAGE_MAG_MATRIX_01, a[2]);
			optics.setValue(EMDL_IMAGE_MAG_MATRIX_10, a[3]);
			optics.setValue(EMDL_IMAGE_MAG_MATRIX_11, a[4]);
			if (a.size() > 5)
				optics.setValue(EMDL_IMAGE_EVEN_ZERNIKE_COEFFS, std::vector<double>(a.begin() + 5, a.end()));
			slot.reset(new ObservationModel(optics));
		}
		om = slot.get();
	}
	MetaDataTable part;
	part.addObject();
	part.setValue(EMDL_IMAGE_OPTICS_GROUP, 1);
	part.setValue(EMDL_CTF_DEFOCUSU, p.defU);
	part.setValue(EMDL_CTF_DEFOCUSV, p.defV);
	part.setValue(EMDL_CTF_DEFOCUS_ANGLE, p.angle);
	part.setValue(EMDL_CTF_BFACTOR, p.bfac);
	part.setValue(EMDL_CTF_SCALEFACTOR, p.scale);
	part.setValue(EMDL_CTF_PHASESHIFT, p.phase);
	c.readByGroup(part, om, 0);
}

/* Preprocessing::extractParticlesFromOneMicrograph windows the box out of the
 * micrograph filling out-of-range pixels with the micrograph mean, then
 * overwrites every one of those with the nearest edge pixel (x first, then y);
 * the mean never survives, so clamping both coordinates is the same box.
 * performPerImageOperations then rescales, re-windows, normalises and inverts,
 * and the stack writer rounds to float or half. */
void extractOne(const Micrograph& m, const Recipe& r, long xpos, long ypos, double psi,
                double cx, double cy, double angle, const ParticleCtf* ctf, MultidimArray<RFLOAT>& out)
{
	Image<RFLOAT> I;
	if (r.rectangular() || r.rotated)
	{
		// The same functions relion_preprocess calls, so the pixels are the same
		const int ny = r.rectangular() ? r.extract_size_y : r.extract_size;
		const Micrograph* mp = &m;
		auto at = [mp](long y, long x) { return mp->at(y, x); };
		if (r.rotated)
			extractrect::cutRotatedBox(at, m.nx, m.ny, cx, cy, r.extract_size, ny, angle, r.interpolation, I());
		else
			extractrect::cutBox(at, m.nx, m.ny, xpos, ypos, r.extract_size, ny, I());
	}
	else if (r.premultiplied())
	{
		/* relion_preprocess --premultiply_ctf: the premultiply_size box, with the
		 * micrograph average off the micrograph, times the CTF, cut to extract_size;
		 * then pixels off the micrograph take the nearest edge pixel (x first, then y). */
		const long e = r.extract_size;
		const long P = r.premultiply_size == Recipe::PREMULTIPLY_AUTO
		             ? autoPremultiplySize(*ctf, r.ctf_angpix, r.extract_size, r.scale > 0 ? r.scale : r.extract_size)
		             : r.premultiply_size;
		const long first = FIRST_XMIPP_INDEX(P);
		const bool off = xpos + first < 0 || xpos + LAST_XMIPP_INDEX(P) >= m.nx
		              || ypos + first < 0 || ypos + LAST_XMIPP_INDEX(P) >= m.ny;
		const RFLOAT avg = off ? m.mean() : 0.;
		I = Image<RFLOAT>(P, P);
		for (long i = 0; i < P; i++)
		{
			const long y = ypos + first + i;
			for (long j = 0; j < P; j++)
			{
				const long x = xpos + first + j;
				DIRECT_A2D_ELEM(I(), i, j) = (y < 0 || y >= m.ny || x < 0 || x >= m.nx) ? avg : m.at(y, x);
			}
		}
		I().setXmippOrigin();

		FourierTransformer transformer;
		MultidimArray<Complex> FT;
		transformer.FourierTransform(I(), FT, false);
		MultidimArray<RFLOAT> Fctf(YSIZE(FT), XSIZE(FT));
		CTF c;
		if (ctf->aberr.empty())
			c.setValues(ctf->defU, ctf->defV, ctf->angle, ctf->kV, ctf->Cs, ctf->Q0, ctf->bfac, ctf->scale, ctf->phase);
		else
			readAberratedCtf(*ctf, P, c);
		c.getFftwImage(Fctf, P, P, r.ctf_angpix, false, false, r.ctf_intact_first_peak, true, false);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FT)
			DIRECT_MULTIDIM_ELEM(FT, n) *= DIRECT_MULTIDIM_ELEM(Fctf, n);
		transformer.inverseFourierTransform(FT, I());
		if (e != P)
			I().window(FIRST_XMIPP_INDEX(e), FIRST_XMIPP_INDEX(e), LAST_XMIPP_INDEX(e), LAST_XMIPP_INDEX(e));

		if (off)
		{
			MultidimArray<RFLOAT>& A = I();
			FOR_ALL_ELEMENTS_IN_ARRAY2D(A)
			{
				if (j + xpos < 0) A2D_ELEM(A, i, j) = A2D_ELEM(A, i, -xpos);
				else if (j + xpos >= m.nx) A2D_ELEM(A, i, j) = A2D_ELEM(A, i, m.nx - xpos - 1);
			}
			FOR_ALL_ELEMENTS_IN_ARRAY2D(A)
			{
				if (i + ypos < 0) A2D_ELEM(A, i, j) = A2D_ELEM(A, -ypos, j);
				else if (i + ypos >= m.ny) A2D_ELEM(A, i, j) = A2D_ELEM(A, m.ny - ypos - 1, j);
			}
		}
	}
	else
	{
		I = Image<RFLOAT>(r.extract_size, r.extract_size);
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
	}

	I().setXmippOrigin();
	if (r.scale > 0)
	{
		if (r.rectangular()) resizeMapRect(I(), r.scale, r.outputSizeY());
		else rescale(I, r.scale);
	}
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
// Polished particles: recombined from the movie frames, by the function Polish uses
// ---------------------------------------------------------------------------

/// Cores this process may use, shared with the other ranks on its node.
int localCores()
{
	// Ranks of one node share its cores, and each sees them all unless the launcher binds them
	int local_ranks = 1;
	for (const char* var : {"OMPI_COMM_WORLD_LOCAL_SIZE", "SLURM_NTASKS_PER_NODE", "MPI_LOCALNRANKS"})
	{
		const char* r = getenv(var);
		if (r != NULL && atoi(r) > 0) { local_ranks = atoi(r); break; }
	}
	return std::max(1, (int)resrep::allowedCores().size() / local_ranks);
}

/// Polished: the frame sums of a micrograph's particles (Fourier space), those with the same window in one stack
struct Sums {
	std::vector<BufferedImage<Complex> > stack;      // one per window size, smallest first
	std::vector<std::vector<int> > members;          // the particles of each stack
	std::vector<std::pair<int, int> > at;            // particle -> (stack, slice)

	explicit Sums(const VStack& v) : at(v.win.size())
	{
		std::map<int, int> index;
		for (int w : v.win) index[w] = 0;
		int n = 0;
		for (auto& i : index) i.second = n++;
		members.resize(n);
		stack.resize(n);
		for (size_t p = 0; p < v.win.size(); p++)
		{
			const int g = index[v.win[p]];
			at[p] = std::make_pair(g, (int)members[g].size());
			members[g].push_back(p);
		}
	}
	Complex* of(int p) { return &stack[at[p].first](0, 0, at[p].second); }
};

bool readSum(const VStack& v, Sums& sums);
void storeSum(const VStack& v, Sums& sums, int threads);

/// Cached damage weights (avoid recomputing and re-allocating across micrographs/particles)
static std::shared_ptr<const std::vector<Image<RFLOAT> > > cachedDamageWeights(
		const std::vector<gravis::d2Vector>& bk, int kc)
{
	struct Key
	{
		int kc;
		std::vector<double> bx, by;
		bool operator<(const Key& o) const
		{
			if (kc != o.kc) return kc < o.kc;
			if (bx != o.bx) return bx < o.bx;
			return by < o.by;
		}
	};
	Key key;
	key.kc = kc;
	key.bx.reserve(bk.size());
	key.by.reserve(bk.size());
	for (const auto& b : bk)
	{
		key.bx.push_back(b.x);
		key.by.push_back(b.y);
	}

	static std::mutex& wmutex = *new std::mutex;
	static std::map<Key, std::shared_ptr<const std::vector<Image<RFLOAT> > > >& wcache =
		*new std::map<Key, std::shared_ptr<const std::vector<Image<RFLOAT> > > >;
	{
		std::lock_guard<std::mutex> lock(wmutex);
		auto it = wcache.find(key);
		if (it != wcache.end()) return it->second;
	}
	std::shared_ptr<const std::vector<Image<RFLOAT> > > w(
		new std::vector<Image<RFLOAT> >(DamageHelper::computeWeights(bk, kc)));
	{
		std::lock_guard<std::mutex> lock(wmutex);
		if (wcache.size() >= 64) wcache.clear();
		return wcache.emplace(std::move(key), w).first->second;
	}
}

/// The frame sums of `v` from its movie, read by `mh`: one pass over the particles of each window size
void makeSums(const VStack& v, MicrographHandler& mh, Sums& sums, int threads)
{
	const PolishRecipe& r = *v.polish;
	for (size_t g = 0; g < sums.stack.size(); g++)
	{
		const int s = v.win[sums.members[g][0]];
		MetaDataTable mdt;
		std::vector<std::vector<gravis::d2Vector> > tracks;
		for (int p : sums.members[g])
		{
			mdt.addObject();
			mdt.setValue(EMDL_MICROGRAPH_NAME, v.mic);
			mdt.setValue(EMDL_IMAGE_COORD_X, (RFLOAT)v.cx[p]);
			mdt.setValue(EMDL_IMAGE_COORD_Y, (RFLOAT)v.cy[p]);
			tracks.push_back(v.tracks[p]);
		}
		// The dose weights fall off over the same resolution in every window: sigma is in pixels of box r.s
		std::vector<gravis::d2Vector> bk = r.bk;
		if (s != r.s) for (gravis::d2Vector& b : bk) b.x = b.x * s / r.s;
		std::shared_ptr<const std::vector<Image<RFLOAT> > > weights = cachedDamageWeights(bk, s / 2 + 1);
		sums.stack[g] = FrameRecombiner::sumMicrograph(mdt, mh, tracks, s, r.angpix, r.data_angpix, *weights, threads);
	}
}

/// Every particle of a polished stack, rounded as the stack would be, one after the other.
void computePolished(const VStack& v, std::vector<float>& pixels)
{
	const PolishRecipe& r = *v.polish;

	// A MicrographHandler is not thread-safe: hold its lock only when reading/decoding a movie on a cache miss
	// (never destroyed: background cache builders may still read particles while the process exits)
	static std::mutex& mutex = *new std::mutex;
	static std::map<std::string, std::unique_ptr<MicrographHandler> >& handlers = *new std::map<std::string, std::unique_ptr<MicrographHandler> >;

	const int threads = localCores();
	// From version 5 the sums are rounded to 16 bits, read from the cache or made now: the same particles either way
	const bool rounded = v.version >= DESCRIPTOR_VERSION_POLISH_SUM16;
	Sums sums(v);
	const bool have_sum = rounded && readSum(v, sums);
	if (!have_sum)
	{
		{
			std::lock_guard<std::mutex> guard(mutex);
			std::ostringstream key;
			key.precision(17);
			key << r.corr_mic << " " << r.first_frame << " " << r.last_frame << " " << r.eer_upsampling << " " << r.eer_grouping
			    << " " << r.movie_angpix << " " << r.coords_angpix << " " << r.hot_cutoff;
			std::unique_ptr<MicrographHandler>& mh = handlers[key.str()];
			if (!mh)
			{
				MetaDataTable mdt;
				mdt.addObject();
				mdt.setValue(EMDL_MICROGRAPH_NAME, v.mic);
				MicrographHandler* h = new MicrographHandler();
				h->corrMicFn = r.corr_mic;
				h->firstFrame = r.first_frame;
				h->lastFrame = r.last_frame;
				h->eer_upsampling = r.eer_upsampling;
				h->eer_grouping = r.eer_grouping;
				h->movie_angpix = r.movie_angpix;
				h->coords_angpix = r.coords_angpix;
				h->hotCutoff = r.hot_cutoff;
				int fc;
				double dose;
				std::string metaFn;
				h->init(std::vector<MetaDataTable>(1, mdt), false, threads, fc, dose, metaFn);
				mh.reset(h);
			}

			// Several window sizes: the movie is decoded once and kept for the passes after the first, if it fits
			mh->keepForCombination = sums.stack.size() > 1;
			makeSums(v, *mh, sums, threads);
			mh->releaseKept();
		}
		if (rounded) storeSum(v, sums, threads);
	}
	const size_t pc = v.cx.size();
	const size_t slice_sz = (size_t)r.out_size * (size_t)r.out_size;
	pixels.resize(pc * slice_sz);
	const bool has_ctf = v.recipe.premultiplied();

	// Group particles by window size so each thread stays on one cached FFTW plan slot
	std::vector<int> order;
	order.reserve(pc);
	for (const auto& grp : sums.members)
		order.insert(order.end(), grp.begin(), grp.end());

	#pragma omp parallel num_threads(threads)
	{
		FrameRecombiner::FinishWorkspace ws;
		Image<RFLOAT> slice(r.out_size, r.out_size);
		RawImage<RFLOAT> out(slice);
		#pragma omp for schedule(static)
		for (long idx = 0; idx < (long)pc; idx++)
		{
			const int p = order[idx];
			BufferedImage<Complex> sum = sums.stack[sums.at[p].first].getSliceRef(sums.at[p].second);
			if (has_ctf) ctfImage(v.ctf[p], v.win[p], r.angpix, ws.ctfImg);
			FrameRecombiner::finishParticle(sum, has_ctf ? &ws.ctfImg : NULL, out, ws);
			float* dst = &pixels[(size_t)p * slice_sz];
			if (r.float16)
			{
				for (size_t n = 0; n < slice_sz; n++)
					dst[n] = half2float(float2half((float)DIRECT_MULTIDIM_ELEM(slice(), n)));
			}
			else
			{
				for (size_t n = 0; n < slice_sz; n++)
					dst[n] = (float)DIRECT_MULTIDIM_ELEM(slice(), n);
			}
		}
	}
}

/// A polished stack's particles, computed once for all of them (the movie has to be read whole)
struct Polished {
	std::string key;
	std::mutex m;
	bool done = false;
	std::vector<float> pixels;
};

/// How many bytes of recombined micrographs are kept: RELION_VPOLISH_MEMORY_GB, else a tenth of the memory
/// that is free now, but room for two micrographs at least (a reader may alternate between two). Said once.
size_t polishedBudget(size_t one_stack_bytes)
{
	const char* e = getenv("RELION_VPOLISH_MEMORY_GB");
	long long budget;
	std::string why;
	if (e != NULL && e[0] != '\0')
	{
		budget = (long long)(atof(e) * 1073741824.);
		why = "RELION_VPOLISH_MEMORY_GB";
	}
	else
	{
		const long long avail = asyncwrite::memoryAvailableBytes();
		budget = avail >= 0 ? avail / 10 : 1LL << 30;
		why = "a tenth of the free memory";
	}
	const long long least = 2 * (long long)one_stack_bytes;
	static std::once_flag said;
	std::call_once(said, [&]()
	{
		std::cout << " Virtual polished particles: up to " << std::max(budget, least) / 1048576
		          << " MB of recombined micrographs kept in memory (" << why << ", at least two micrographs; "
		          << "RELION_VPOLISH_MEMORY_GB sets it)" << std::endl;
	});
	return (size_t)std::max(budget, least);
}

/// The recombined particles of `v`'s micrograph, from memory or computed now; the most recently used are
/// kept (see polishedBudget). The caller holds the result while it copies from it.
std::shared_ptr<Polished> polishedPixels(const VStack& v)
{
	// never destroyed: background cache builders may still read particles while the process exits
	static std::mutex& mutex = *new std::mutex;
	static std::list<std::shared_ptr<Polished> >& recent = *new std::list<std::shared_ptr<Polished> >;
	std::shared_ptr<Polished> p;
	{
		std::lock_guard<std::mutex> guard(mutex);
		for (std::list<std::shared_ptr<Polished> >::iterator it = recent.begin(); it != recent.end(); ++it)
			if ((*it)->key == v.key) { p = *it; recent.erase(it); break; }
		if (!p) { p.reset(new Polished()); p->key = v.key; }
		recent.push_front(p);
		// whole micrographs at a time; one being computed has no pixels yet and counts as one stack
		const size_t one = sizeof(float) * (size_t)v.polish->out_size * v.polish->out_size * v.x.size();
		const size_t budget = polishedBudget(one);
		size_t used = 0;
		for (std::list<std::shared_ptr<Polished> >::iterator it = recent.begin(); it != recent.end();)
		{
			used += std::max(one, (*it)->pixels.size() * sizeof(float));
			if (used > budget && it != recent.begin() && it != ++recent.begin()) it = recent.erase(it);
			else ++it;
		}
	}
	std::lock_guard<std::mutex> guard(p->m);
	if (!p->done)
	{
		computePolished(v, p->pixels);
		p->done = true;
	}
	return p;
}

void copyPolished(const VStack& v, const Polished& p, size_t i, MultidimArray<RFLOAT>& out)
{
	const int n = v.polish->out_size;
	out.resize(n, n);
	for (size_t k = 0; k < (size_t)n * n; k++)
		DIRECT_MULTIDIM_ELEM(out, k) = p.pixels[(size_t)n * n * i + k];
	out.setXmippOrigin();
}

void extractIndex(const Micrograph* m, const VStack& v, size_t i, MultidimArray<RFLOAT>& out, const Polished* pinned = NULL)
{
	if (v.polish)
	{
		if (pinned != NULL) copyPolished(v, *pinned, i, out);
		else copyPolished(v, *polishedPixels(v), i, out);
		return;
	}
	const bool rot = v.recipe.rotated;
	extractOne(*m, v.recipe, v.x[i], v.y[i], psiOf(v, i),
	           rot ? v.cx[i] : 0., rot ? v.cy[i] : 0., rot ? v.angle[i] : 0.,
	           v.ctf.empty() ? NULL : &v.ctf[i], out);
}

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

void decode(const unsigned char* p, const Recipe& r, int box, int box_y, MultidimArray<RFLOAT>& out)
{
	out.resize(box_y, box);
	const size_t n = (size_t)box * box_y;
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
	const int box_y = v.recipe.outputSizeY();
	const size_t one = (size_t)box * box_y * bytesPerValue(v.recipe);

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
		             && h.reserved == (v.recipe.rectangular() ? (uint32_t)box_y : 0u)
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
	decode(buf.data(), v.recipe, box, box_y, out);
	return true;
}

/* Cache capacity. Without a limit a converted project regrows all of its
 * particles in the cache on first use, and the disk space it was converted to
 * save comes straight back. RELION_VPARTICLE_CACHE_MAX_GB caps it (default 0:
 * no limit, so a data set larger than a fixed cap is not evicted and rebuilt
 * every iteration); the least recently used entries are removed first, and a
 * removed entry is simply rebuilt from its micrograph when next needed. */
long long cacheLimitBytes()
{
	const char* v = getenv("RELION_VPARTICLE_CACHE_MAX_GB");
	const double gb = (v != NULL && v[0] != '\0') ? atof(v) : 0.;
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

	vcache::prune(dir, limit, ".vpc");
}

// A failed write (a full or briefly failing file system) pauses writing for a minute, not for the whole run
const time_t CACHE_RETRY_SECONDS = 60;

void warnUnwritable(const std::string& dir, const std::string& why)
{
	std::lock_guard<std::mutex> guard(g_mutex);
	const bool first = g_unwritable.count(dir) == 0;
	g_unwritable[dir] = time(NULL) + CACHE_RETRY_SECONDS;
	if (first)
		std::cerr << " WARNING: cannot write the virtual particle cache in " << dir << " (" << why
		          << "); particles not yet cached are read straight from the micrographs, which is slower,"
		          << " and writing is tried again after a minute."
		          << " Set RELION_VPARTICLE_CACHE to a writable directory, or to 'off' to silence this."
		          << std::endl;
}

bool cacheUnwritable(const std::string& dir)
{
	std::lock_guard<std::mutex> guard(g_mutex);
	const auto it = g_unwritable.find(dir);
	return it != g_unwritable.end() && time(NULL) < it->second;
}

// ---------------------------------------------------------------------------
// Polished particles: the frame sums, kept so that a new CTF does not need the movie again
// ---------------------------------------------------------------------------

const char SUM_MAGIC[8] = {'R', 'V', 'P', 'C', 'S', 'U', 'M', 'S'};

/// False with RELION_VPOLISH_SUMS=0: the sums are then made from the movie whenever they are needed.
bool sumsKept()
{
	const char* e = getenv("RELION_VPOLISH_SUMS");
	return !(e != NULL && e[0] == '0');
}

/// The frame sums of `v` from the cache, as storeSum left them; false if they are not there.
/// File: the header, each particle's window (uint32), then each particle's sum (re, im as float16).
/// `max_bytes` > 0: not if the sums are larger. Reads only (called from a background thread too).
bool loadSumFile(const VStack& v, std::vector<float16>& buf, double max_bytes = 0.)
{
	const std::string dir = cacheDirectory();
	if (dir.empty() || !sumsKept()) return false;
	const size_t pc = v.win.size();
	std::vector<size_t> start(pc + 1, 0);   // in float16 values
	for (size_t p = 0; p < pc; p++) start[p + 1] = start[p] + (size_t)(v.win[p] / 2 + 1) * v.win[p] * 2;
	if (max_bytes > 0. && start[pc] * sizeof(float16) > max_bytes) return false;
	const size_t table = pc * sizeof(uint32_t);
	int fd = open(entryPath(dir, v.sum_key).c_str(), O_RDONLY);
	if (fd < 0) return false;
	EntryHeader h;
	struct stat st;
	std::vector<uint32_t> win(pc);
	bool ok = pread(fd, &h, sizeof(h), 0) == (ssize_t)sizeof(h)
	       && fstat(fd, &st) == 0
	       && memcmp(h.magic, SUM_MAGIC, 8) == 0
	       && h.version == 2 && h.dtype == 12u && h.n == (uint64_t)pc && h.box == (uint32_t)v.polish->s
	       && strncmp(h.key, v.sum_key.c_str(), sizeof(h.key)) == 0
	       && (size_t)st.st_size == sizeof(h) + table + start[pc] * sizeof(float16)
	       && pread(fd, win.data(), table, sizeof(h)) == (ssize_t)table;
	for (size_t p = 0; ok && p < pc; p++) ok = win[p] == (uint32_t)v.win[p];
	if (ok)
	{
		buf.resize(start[pc]);
		size_t got = 0;
		const size_t want = buf.size() * sizeof(float16);
		while (got < want)
		{
			const ssize_t n = pread(fd, (char*)buf.data() + got, want - got, sizeof(h) + table + got);
			if (n <= 0) break;
			got += n;
		}
		ok = got == want;
		if (ok) futimens(fd, NULL);   // recency for eviction
	}
	close(fd);
	return ok;
}

// The sum files a reader will want next, being read in the background, by key: up to sumAheadDepth()
// started while the current one (started earlier) is still waiting to be taken.
std::mutex g_sum_ahead_mutex;
std::map<std::string, std::shared_future<std::shared_ptr<std::vector<float16> > > > g_sum_ahead;
// What the readers waited for sum files (seconds x 1e6) and how many they read, for the report
std::atomic<long long> g_sum_wait_us(0), g_sum_reads(0), g_sum_reads_ahead(0);

// Read-ahead depth, chosen from the measured wait unless RELION_VPOLISH_SUM_READ_AHEAD sets it: start at one
// file, try one more after each window of SUM_WINDOW reads that still waited, keep it only if the wait per
// file fell by a fifth (on a slow disc more files in flight only share the same bandwidth), else go back and stop.
const int SUM_WINDOW = 8;
std::atomic<int> g_sum_depth(-1);
bool g_sum_depth_settled = false;
long long g_win_us = 0, g_win_reads = 0;
double g_prev_per_file = -1.;   // seconds per file at the depth before, -1: none yet

/// How many sum files are read ahead (0 = off).
int sumAheadDepth()
{
	int d = g_sum_depth;
	if (d < 0)
	{
		const char* e = getenv("RELION_VPOLISH_SUM_READ_AHEAD");
		const bool set = e != NULL && e[0] != '\0';
		d = set ? std::max(0, atoi(e)) : 1;
		std::lock_guard<std::mutex> guard(g_sum_ahead_mutex);
		if (g_sum_depth < 0) { g_sum_depth_settled = set; g_sum_depth = d; }
		d = g_sum_depth;
	}
	return d;
}

/// One read waited `us`; at the end of a window, try a deeper read-ahead or settle. `bytes`: one file's size.
void adaptSumDepth(long long us, size_t bytes)
{
	sumAheadDepth();
	std::lock_guard<std::mutex> guard(g_sum_ahead_mutex);
	if (g_sum_depth_settled) return;
	g_win_us += us;
	if (++g_win_reads < SUM_WINDOW) return;
	const double per_file = g_win_us / 1e6 / g_win_reads;
	g_win_us = g_win_reads = 0;
	const long long avail = asyncwrite::memoryAvailableBytes();
	const int cap = std::min(8, (int)std::max(1., (avail < 0 ? 1e9 : 0.25 * avail) / std::max<size_t>(bytes, 1)));
	if (g_prev_per_file >= 0. && per_file > 0.8 * g_prev_per_file)
	{
		g_sum_depth = g_sum_depth - 1;   // the deeper read-ahead did not help
		g_sum_depth_settled = true;
		return;
	}
	g_prev_per_file = per_file;
	if (per_file > 0.02 && g_sum_depth < cap) g_sum_depth = g_sum_depth + 1;
	else g_sum_depth_settled = true;
}

/// Start reading the frame sums of `v` in the background, for a readSum that follows: the file read then
/// costs the reader no time. Not if the files read ahead would take more than a quarter of the free memory.
void sumAhead(const std::shared_ptr<VStack>& v)
{
	const int depth = sumAheadDepth();
	if (!v->polish || v->version < DESCRIPTOR_VERSION_POLISH_SUM16 || depth == 0) return;
	const long long avail = asyncwrite::memoryAvailableBytes();   // -1: unknown
	const double max_bytes = (avail < 0 ? 1e9 : 0.25 * avail) / depth;
	std::lock_guard<std::mutex> guard(g_sum_ahead_mutex);
	if (g_sum_ahead.count(v->sum_key)) return;
	if ((int)g_sum_ahead.size() > depth) g_sum_ahead.erase(g_sum_ahead.begin());   // one nobody took
	g_sum_ahead[v->sum_key] = std::async(std::launch::async, [v, max_bytes]
	{
		std::shared_ptr<std::vector<float16> > buf(new std::vector<float16>());
		if (!loadSumFile(*v, *buf, max_bytes)) buf.reset();
		return buf;
	});
}

bool readSum(const VStack& v, Sums& sums)
{
	const size_t pc = v.win.size();
	std::vector<size_t> start(pc + 1, 0);   // in float16 values
	for (size_t p = 0; p < pc; p++) start[p + 1] = start[p] + (size_t)(v.win[p] / 2 + 1) * v.win[p] * 2;
	std::shared_ptr<std::vector<float16> > ahead;
	const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	std::shared_future<std::shared_ptr<std::vector<float16> > > fut;
	bool have_fut = false;
	{
		std::lock_guard<std::mutex> guard(g_sum_ahead_mutex);
		std::map<std::string, std::shared_future<std::shared_ptr<std::vector<float16> > > >::iterator
			it = g_sum_ahead.find(v.sum_key);
		if (it != g_sum_ahead.end()) { fut = it->second; g_sum_ahead.erase(it); have_fut = true; }
	}
	if (have_fut) ahead = fut.get();
	std::vector<float16> own;
	if (!ahead && !loadSumFile(v, own)) return false;
	const long long waited = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
	g_sum_wait_us += waited;
	g_sum_reads++;
	if (ahead) g_sum_reads_ahead++;
	const std::vector<float16>& buf = ahead ? *ahead : own;
	adaptSumDepth(waited, buf.size() * sizeof(float16));

	for (size_t g = 0; g < sums.stack.size(); g++)
	{
		const int s = v.win[sums.members[g][0]];
		sums.stack[g] = BufferedImage<Complex>(s / 2 + 1, s, sums.members[g].size());
	}
	#pragma omp parallel for num_threads(localCores())
	for (long p = 0; p < (long)pc; p++)
	{
		const float16* b = &buf[start[p]];
		Complex* d = sums.of(p);
		for (size_t k = 0; k < (start[p + 1] - start[p]) / 2; k++)
		{
			d[k].real = half2float(b[2 * k]);
			d[k].imag = half2float(b[2 * k + 1]);
		}
	}
	return true;
}

/* Round the frame sums to 16 bits and write them to the cache (best effort, in the background). Each particle
 * is first scaled by a power of two that brings its largest value to 16384..32768: the particle is normalised
 * by its own power afterwards, so the scale drops out exactly, and no sum overflows or underflows 16 bits. */
void storeSum(const VStack& v, Sums& sums, int threads)
{
	const size_t pc = v.win.size();
	std::vector<size_t> start(pc + 1, 0);
	for (size_t p = 0; p < pc; p++) start[p + 1] = start[p] + (size_t)(v.win[p] / 2 + 1) * v.win[p] * 2;
	std::shared_ptr<std::vector<float16> > buf(new std::vector<float16>(start[pc]));
	#pragma omp parallel for num_threads(threads)
	for (long p = 0; p < (long)pc; p++)
	{
		Complex* d = sums.of(p);
		const size_t n = (start[p + 1] - start[p]) / 2;
		double top = 0.;
		for (size_t k = 0; k < n; k++) top = std::max(top, std::max(std::fabs(d[k].real), std::fabs(d[k].imag)));
		const double scale = (top > 0. && std::isfinite(top)) ? std::ldexp(1., 14 - std::ilogb(top)) : 1.;
		float16* b = &(*buf)[start[p]];
		for (size_t k = 0; k < n; k++)
		{
			b[2 * k] = float2half((float)(d[k].real * scale));
			b[2 * k + 1] = float2half((float)(d[k].imag * scale));
			d[k].real = half2float(b[2 * k]);
			d[k].imag = half2float(b[2 * k + 1]);
		}
	}

	const std::string dir = cacheDirectory();
	if (dir.empty() || !sumsKept() || cacheUnwritable(dir)) return;
	const std::string path = entryPath(dir, v.sum_key);
	makeDirs(path.substr(0, path.find_last_of('/')));
	static std::atomic<long> counter(0);
	char host[256] = "host";
	gethostname(host, sizeof(host) - 1);
	std::ostringstream tmps;
	tmps << path << ".tmp." << host << "." << getpid() << "." << counter++;
	const std::string tmp = tmps.str();

	EntryHeader h;
	memset(&h, 0, sizeof(h));
	memcpy(h.magic, SUM_MAGIC, 8);
	h.version = 2;
	h.dtype = 12;
	h.n = pc;
	h.box = v.polish->s;
	strncpy(h.key, v.sum_key.c_str(), sizeof(h.key) - 1);
	const std::vector<uint32_t> win(v.win.begin(), v.win.end());

	// On the background writer (which waits when too much is queued), so that the caller goes on computing
	const size_t want = buf->size() * sizeof(float16);
	AsyncFileWriter::instance().submit([=]
	{
		const int fd = open(tmp.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0664);
		bool ok = fd >= 0 && write(fd, &h, sizeof(h)) == (ssize_t)sizeof(h)
		       && write(fd, win.data(), win.size() * sizeof(uint32_t)) == (ssize_t)(win.size() * sizeof(uint32_t));
		for (size_t done = 0; ok && done < want;)
		{
			const ssize_t n = write(fd, (const char*)buf->data() + done, want - done);
			if (n <= 0) ok = false; else done += n;
		}
		const int e = errno;
		if (fd >= 0 && close(fd) != 0) ok = false;
		if (ok && rename(tmp.c_str(), path.c_str()) == 0)
			enforceCacheLimit(dir, (long long)(sizeof(h) + want));
		else
		{
			unlink(tmp.c_str());
			warnUnwritable(dir, strerror(e));
		}
	}, want);
}

/* Set up the cache entry of `v`: the micrograph, the temporary file and its header. False if the file cannot
 * be written. */
bool prepareBuild(const std::string& dir, VStack& v, EntryBuild& b)
{
	verifyMicrograph(v);
	if (v.polish) b.pol = polishedPixels(v);   // held by the entry, so that no other micrograph's reads push it out
	else b.mic = openMicrograph(v.mic);
	b.path = entryPath(dir, v.key);
	makeDirs(b.path.substr(0, b.path.find_last_of('/')));

	static std::atomic<long> counter(0);
	char host[256] = "host";
	gethostname(host, sizeof(host) - 1);
	std::ostringstream tmp;
	tmp << b.path << ".tmp." << host << "." << getpid() << "." << counter++;
	b.tmp = tmp.str();

	b.fd = open(b.tmp.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0664);
	if (b.fd < 0)
	{
		warnUnwritable(dir, strerror(errno));
		return false;
	}
	const int box = v.recipe.outputSize();
	const int box_y = v.recipe.outputSizeY();
	b.one = (size_t)box * box_y * bytesPerValue(v.recipe);

	EntryHeader h;
	memset(&h, 0, sizeof(h));
	memcpy(h.magic, ENTRY_MAGIC, 8);
	h.version = 1;
	h.dtype = v.recipe.float16 ? 12 : 2;
	h.n = v.x.size();
	h.box = box;
	h.reserved = v.recipe.rectangular() ? (uint32_t)box_y : 0u;
	strncpy(h.key, v.key.c_str(), sizeof(h.key) - 1);
	if (pwrite(b.fd, &h, sizeof(h), 0) != (ssize_t)sizeof(h))
	{
		const int e = errno;
		close(b.fd);
		b.fd = -1;
		unlink(b.tmp.c_str());
		warnUnwritable(dir, strerror(e));
		return false;
	}

	b.order.resize(v.x.size());
	for (size_t i = 0; i < b.order.size(); i++) b.order[i] = i;
	std::stable_sort(b.order.begin(), b.order.end(), [&](size_t p, size_t q) {
		return (v.y[p] != v.y[q]) ? v.y[p] < v.y[q] : v.x[p] < v.x[q];
	});
	return true;
}

/* Cut particles of the entry until none are left; particle `idx` is kept in `out`. */
bool helpBuild(VStack& v, EntryBuild& b, long idx, MultidimArray<RFLOAT>& out)
{
	bool have_idx = false;
	MultidimArray<RFLOAT> img;
	std::vector<unsigned char> buf;
	size_t k;
	auto account = [&b]() { if (++b.done == b.order.size()) { std::lock_guard<std::mutex> l(b.m); b.cv.notify_all(); } };
	while ((k = b.next++) < b.order.size())
	{
		if (g_builds_stopping) { b.failed = true; account(); continue; }   // ending: give the entry up
		const size_t i = b.order[k];
		try
		{
			extractIndex(b.mic.get(), v, i, img, b.pol.get());
			encode(img, v.recipe, buf);
			if (pwrite(b.fd, buf.data(), b.one, sizeof(EntryHeader) + b.one * i) != (ssize_t)b.one)
				b.failed = true;
			if ((long)i == idx) { out = img; have_idx = true; }
		}
		catch (...)
		{
			// The entry is lost, and so is whatever this thread would have cut: count the particle and every one still
			// unclaimed as done, or the thread that finishes the entry would wait for them for ever
			b.failed = true;
			account();
			while (b.next++ < b.order.size()) account();
			throw;
		}
		account();
	}
	return have_idx;
}

/* Starter of an entry: once every particle is written, close and publish it (or drop it), and let the
 * threads waiting for it go. */
void finishBuild(const std::string& dir, VStack& v, EntryBuild& b)
{
	{
		std::unique_lock<std::mutex> l(b.m);
		b.cv.wait(l, [&]() { return b.done >= b.order.size(); });
	}
	if (b.fd >= 0)
	{
		const int saved_errno = errno;
		const bool closed = close(b.fd) == 0;
		b.fd = -1;
		if (!b.failed && closed && rename(b.tmp.c_str(), b.path.c_str()) == 0)
		{
			v.entry_ok = true;
			enforceCacheLimit(dir, (long long)(sizeof(EntryHeader) + b.one * v.x.size()));
		}
		else
		{
			unlink(b.tmp.c_str());
			warnUnwritable(dir, strerror(saved_errno ? saved_errno : errno));
			b.failed = true;
		}
	}
	{
		std::lock_guard<std::mutex> g(v.build);
		v.building.reset();
	}
	b.pol.reset();
	std::lock_guard<std::mutex> l(b.m);
	b.finished = true;
	b.cv.notify_all();
}

/* Read particle `idx` through the cache entry of `v`, building the entry if needed: the first thread to need
 * it starts it, and threads that need it meanwhile help cut its particles. False if the entry could not be
 * written; `out` is then not filled. */
bool readThroughEntry(const std::string& dir, VStack& v, long idx, MultidimArray<RFLOAT>& out)
{
	const std::string path = entryPath(dir, v.key);
	std::shared_ptr<EntryBuild> b;
	bool starter = false;
	{
		std::lock_guard<std::mutex> g(v.build);
		if (v.building)
			b = v.building;
		else
		{
			if (readFromEntry(path, v, idx, out)) return true;   // written meanwhile
			b = std::make_shared<EntryBuild>();
			v.building = b;
			starter = true;
		}
	}
	if (starter)
	{
		bool ok = false;
		try { ok = prepareBuild(dir, v, *b); }
		catch (...)
		{
			b->failed = true;
			{ std::lock_guard<std::mutex> l(b->m); b->prepared = true; }
			finishBuild(dir, v, *b);
			throw;
		}
		if (!ok) b->failed = true;
		{
			std::lock_guard<std::mutex> l(b->m);
			b->prepared = true;
			b->cv.notify_all();
		}
	}
	else
	{
		std::unique_lock<std::mutex> l(b->m);
		b->cv.wait(l, [&]() { return b->prepared; });
	}

	bool have = false;
	if (!b->failed)
	{
		try { have = helpBuild(v, *b, idx, out); }
		catch (...)
		{
			if (starter) finishBuild(dir, v, *b);
			throw;
		}
	}
	if (starter)
		finishBuild(dir, v, *b);
	else
	{
		std::unique_lock<std::mutex> l(b->m);
		b->cv.wait(l, [&]() { return b->finished; });
	}
	if (b->failed) return false;
	return have || readFromEntry(path, v, idx, out);
}

/* Cache entries built by background threads, so that a reader cuts only the particle it needs now (as without
 * a cache) instead of waiting while the whole micrograph's entry is written. */
class BackgroundBuilds
{
public:
	~BackgroundBuilds() { shutdown(); }

	/// Stop the threads: what is queued is dropped, the entry each is writing is given up (its temporary file removed,
	/// never left half-written), and they are joined. Safe to call again. Registered with atexit() at the first use, so
	/// it runs before the destructors of the statics that were built before then (the builders use them).
	void shutdown()
	{
		std::vector<std::thread> threads;
		{
			std::lock_guard<std::mutex> l(m_);
			stop_ = true;
			g_builds_stopping = true;
			queue_.clear();
			threads.swap(threads_);
		}
		cv_.notify_all();
		for (std::thread& t : threads) t.join();
	}

	static BackgroundBuilds*& self() { static BackgroundBuilds* p = NULL; return p; }

	/// Number of builder threads: RELION_VPARTICLE_BUILDERS, else half the cores this process may use.
	/// 0 means none: readers build entries themselves.
	int builders()
	{
		std::lock_guard<std::mutex> l(m_);
		if (n_ < 0)
		{
			// Ranks of one node share its cores, and each sees them all unless the launcher binds them
			int local_ranks = 1;
			for (const char* var : {"OMPI_COMM_WORLD_LOCAL_SIZE", "SLURM_NTASKS_PER_NODE", "MPI_LOCALNRANKS"})
			{
				const char* r = getenv(var);
				if (r != NULL && atoi(r) > 0) { local_ranks = atoi(r); break; }
			}
			const int cores = std::max<int>(1, resrep::allowedCores().size());
			const char* e = getenv("RELION_VPARTICLE_BUILDERS");
			n_ = (e != NULL && e[0] != '\0') ? std::max(0, atoi(e)) : std::max(1, cores / (2 * local_ranks));
			if (n_ > 0 && !atexit_registered_)
			{
				atexit_registered_ = true;
				self() = this;
				std::atexit([]() { if (self() != NULL) self()->shutdown(); });
			}
			if (n_ > 0)
				std::cout << " Virtual particles: " << n_ << " background thread(s) write the cache entries ("
				          << ((e != NULL && e[0] != '\0') ? std::string("RELION_VPARTICLE_BUILDERS")
				              : "half of the " + integerToString(cores) + " cores"
				                + (local_ranks > 1 ? " shared by " + integerToString(local_ranks) + " processes on this node" : std::string()))
				          << "; RELION_VPARTICLE_BUILDERS=0 makes readers write them)" << std::endl;
		}
		return n_;
	}

	/// Queue the entry of `v` unless it is queued or being built already.
	void add(const std::string& dir, const std::shared_ptr<VStack>& v)
	{
		{
			std::lock_guard<std::mutex> l(m_);
			if (stop_ || !queued_.insert(v->key).second) return;
			queue_.push_back(std::make_pair(dir, v));
			while ((int)threads_.size() < n_) threads_.emplace_back(&BackgroundBuilds::run, this);
		}
		cv_.notify_one();
	}

	/// Wait until every queued entry is written (or given up).
	void wait()
	{
		std::unique_lock<std::mutex> l(m_);
		cv_idle_.wait(l, [this]() { return queue_.empty() && busy_ == 0; });
	}

	/// Wait, then read RELION_VPARTICLE_BUILDERS again at the next miss.
	void reset()
	{
		wait();
		std::lock_guard<std::mutex> l(m_);
		n_ = -1;
	}

private:
	void run()
	{
		MultidimArray<RFLOAT> scratch;
		for (;;)
		{
			std::pair<std::string, std::shared_ptr<VStack> > job;
			{
				std::unique_lock<std::mutex> l(m_);
				cv_.wait(l, [this]() { return stop_ || !queue_.empty(); });
				if (stop_) return;
				job = queue_.front();
				queue_.pop_front();
				busy_++;
			}
			try
			{
				VStack& v = *job.second;
				FftwBackgroundScope fftw_in_use;   // the main thread's FFTW cleanup waits for this entry
				if (!cacheUnwritable(job.first) && !readFromEntry(entryPath(job.first, v.key), v, 0, scratch))
					readThroughEntry(job.first, v, 0, scratch);
			}
			catch (...) {}   // a reader that needs these particles cuts them itself and reports real errors
			{
				std::lock_guard<std::mutex> l(m_);
				queued_.erase(job.second->key);   // a later miss (a deleted or damaged entry) queues it again
				busy_--;
				if (queue_.empty() && busy_ == 0) cv_idle_.notify_all();
			}
		}
	}

	std::mutex m_;
	std::condition_variable cv_, cv_idle_;
	std::deque<std::pair<std::string, std::shared_ptr<VStack> > > queue_;
	std::set<std::string> queued_;
	std::vector<std::thread> threads_;
	int n_ = -1, busy_ = 0;
	bool stop_ = false, atexit_registered_ = false;
};

BackgroundBuilds g_builds;

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

Recipe::Recipe()
	: extract_size(-1), scale(-1), window(-1), normalise(false), bg_radius(-1), ramp(true),
	  white_dust(-1), black_dust(-1), invert_contrast(false), float16(false), angpix(1.),
	  helical(false), helical_radius(-1.), extract_size_y(-1), rotated(false), interpolation(RESAMPLE_LINEAR),
	  premultiply_size(-1), ctf_intact_first_peak(false), ctf_angpix(-1.)
{}

int Recipe::outputSize() const
{
	if (window > 0) return window;
	if (scale > 0) return scale;
	return extract_size;
}

// shortcut: the delocalisation ignores magnification and symmetric aberrations (c.aberr); a 4th-order Zernike term large enough to matter would need its own term
int autoPremultiplySize(const ParticleCtf& c, double angpix, int extract_size, int output_size)
{
	const double V = c.kV * 1e3;
	const double lambda = 12.2643247 / sqrt(V * (1. + V * 0.978466e-6));
	const double Cs = c.Cs * 1e7;
	const double smax = 0.5 / angpix * std::min(1., (double)output_size / extract_size);
	double deloc = 0.;
	for (double df : {c.defU, c.defV})
	{
		auto d = [&](double s) { return std::abs(lambda * df * s - Cs * lambda * lambda * lambda * s * s * s); };
		deloc = std::max(deloc, d(smax));
		if (Cs > 0. && df > 0.)
		{
			const double s_turn = sqrt(df / (3. * Cs * lambda * lambda));
			if (s_turn < smax) deloc = std::max(deloc, d(s_turn));
		}
	}
	const int cap = 4 * extract_size;
	int P = extract_size + 2 * (int)std::ceil(deloc / angpix);
	if (P >= cap) return cap;
	for (P += P % 2; ; P += 2)
	{
		int n = P;
		for (int f : {2, 3, 5}) while (n % f == 0) n /= f;
		if (n == 1) return std::min(P, cap);
	}
}

int Recipe::outputSizeY() const
{
	if (window > 0) return window;
	if (scale > 0) return rectangular() ? extractrect::rescaledHeight(extract_size, extract_size_y, scale) : scale;
	return rectangular() ? extract_size_y : extract_size;
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
	std::shared_ptr<Micrograph> m = v->polish ? std::shared_ptr<Micrograph>() : openMicrograph(v->mic);

	std::vector<size_t> order(v->x.size());
	for (size_t i = 0; i < order.size(); i++) order[i] = i;
	std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
		return (v->y[a] != v->y[b]) ? v->y[a] < v->y[b] : v->x[a] < v->x[b];
	});

	out.assign(v->x.size(), MultidimArray<RFLOAT>());
	for (size_t k = 0; k < order.size(); k++)
		extractIndex(m.get(), *v, order[k], out[order[k]]);
}

void forget(const std::string& fn_vstack)
{
	std::lock_guard<std::mutex> guard(g_mutex);
	g_vstacks.erase(fn_vstack);
}

void ctfImage(const ParticleCtf& c, int s, double angpix, MultidimArray<RFLOAT>& out)
{
	CTF ctf;
	if (c.aberr.empty())
		ctf.setValues(c.defU, c.defV, c.angle, c.kV, c.Cs, c.Q0, c.bfac, c.scale, c.phase);
	else
		readAberratedCtf(c, s, ctf);
	out.resize(s, s / 2 + 1);
	ctf.getFftwImage(out, s, s, angpix,
		false,  // do_abs
		false,  // do_only_flip_phases
		false,  // do_intact_until_first_peak
		true,   // do_damping
		false); // do_ctf_padding
}

std::string polishSourceChecksum(const std::string& movie, const std::string& meta_fn)
{
	std::string out = micrographChecksum(movie);

	// the whole motion record: small, and a re-run motion correction changes it
	std::ifstream in(meta_fn.c_str(), std::ios::binary);
	if (!in) REPORT_ERROR("Cannot read the motion record " + meta_fn + " of virtual polished particles.");
	uint64_t h = 1469598103934665603ULL;
	char buf[65536];
	while (in.read(buf, sizeof(buf)) || in.gcount() > 0)
		h = fnv1a((const unsigned char*)buf, (size_t)in.gcount(), h);
	out += ":" + hex16(h);

	// the gain reference and defect file it names: missing ones are part of the identity too
	::Micrograph record(meta_fn);   // (the motion record class; vparticles has its own Micrograph)
	for (const std::string& f : {std::string(record.getGainFilename()), std::string(record.fnDefect)})
		out += ":" + (f.empty() ? std::string("-") : exists(f) ? micrographChecksum(f) : std::string("missing"));
	return out;
}

void writePolishedDescriptor(const std::string& fn_vstack, const std::string& fn_movie, const PolishRecipe& r,
                             const MetaDataTable& particles,
                             const std::vector<std::vector<gravis::d2Vector> >& tracks,
                             const std::vector<ParticleCtf>& ctf,
                             const std::vector<int>& windows)
{
	const long n = particles.numberOfObjects();
	if (!windows.empty() && (long)windows.size() != n)
		REPORT_ERROR("writePolishedDescriptor: a window per particle (or none) is needed.");
	const size_t fc = r.bk.size();
	if ((long)tracks.size() != n || (!ctf.empty() && (long)ctf.size() != n) || fc != (size_t)(r.last_frame - r.first_frame + 1))
		REPORT_ERROR("writePolishedDescriptor: a track per particle, a CTF per particle (or none) and a dose weight per frame are needed.");

	std::ostringstream ps;
	ps << r.corr_mic << " " << r.first_frame << " " << r.last_frame << " " << r.eer_upsampling << " " << r.eer_grouping
	   << " " << exactString(r.movie_angpix) << " " << exactString(r.coords_angpix) << " " << exactString(r.data_angpix)
	   << " " << exactString(r.hot_cutoff) << " " << r.s << " " << exactString(r.angpix) << " " << r.out_size;
	std::string weights;
	for (size_t f = 0; f < fc; f++)
		weights += (f ? " " : "") + exactString(r.bk[f].x) + " " + exactString(r.bk[f].y);

	std::string mic;
	particles.getValue(EMDL_MICROGRAPH_NAME, mic, 0);

	MetaDataTable head;
	head.setIsList(true);
	head.setName("vstack");
	head.addObject();
	head.setValue(EMDL_VSTACK_VERSION, windows.empty() ? DESCRIPTOR_VERSION_POLISH_SUM16 : DESCRIPTOR_VERSION_POLISH_WINDOWS);
	head.setValue(EMDL_MICROGRAPH_NAME, mic);
	head.setValue(EMDL_MICROGRAPH_MOVIE_NAME, fn_movie);
	head.setValue(EMDL_MICROGRAPH_METADATA_NAME, r.meta_fn);
	head.setValue(EMDL_VSTACK_MICROGRAPH_CHECKSUM, polishSourceChecksum(fn_movie, r.meta_fn));
	head.setValue(EMDL_IMAGE_SIZE, r.out_size);
	head.setValue(EMDL_IMAGE_PIXEL_SIZE, r.angpix);
	head.setValue(EMDL_VSTACK_FLOAT16, r.float16);
	head.setValue(EMDL_VSTACK_POLISH, ps.str());
	head.setValue(EMDL_VSTACK_FRAME_WEIGHTS, weights);
	if (!ctf.empty())
	{
		head.setValue(EMDL_VSTACK_PREMULTIPLY_SIZE, r.s);
		head.setValue(EMDL_VSTACK_CTF_ANGPIX, exactString(r.angpix));
	}

	MetaDataTable parts;
	parts.setName("particles");
	for (long i = 0; i < n; i++)
	{
		RFLOAT x, y;
		particles.getValue(EMDL_IMAGE_COORD_X, x, i);
		particles.getValue(EMDL_IMAGE_COORD_Y, y, i);
		std::string track;
		for (size_t f = 0; f < fc; f++)
			track += (f ? " " : "") + exactString(tracks[i][f].x) + " " + exactString(tracks[i][f].y);
		parts.addObject();
		parts.setValue(EMDL_IMAGE_COORD_X, (RFLOAT)std::floor(x + 0.5));
		parts.setValue(EMDL_IMAGE_COORD_Y, (RFLOAT)std::floor(y + 0.5));
		parts.setValue(EMDL_VSTACK_CENTRE_X, exactString(x));
		parts.setValue(EMDL_VSTACK_CENTRE_Y, exactString(y));
		parts.setValue(EMDL_VSTACK_TRACK, track);
		if (!ctf.empty()) parts.setValue(EMDL_VSTACK_CTF, ctfString(ctf[i]));
		if (!windows.empty()) parts.setValue(EMDL_VSTACK_PREMULTIPLY_SIZE, windows[i]);
	}

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

	std::lock_guard<std::mutex> guard(g_mutex);
	g_vstacks.erase(fn_vstack);
}

std::vector<double> opticsAberrations(const ObservationModel& obs, int og)
{
	if (!obs.hasMagMatrices && !obs.hasEvenZernike) return std::vector<double>();
	const Matrix2D<RFLOAT>& M = obs.getMagMatrix(og);
	std::vector<double> a = {obs.getPixelSize(og), M(0,0), M(0,1), M(1,0), M(1,1)};
	std::vector<double> z;
	if (obs.hasEvenZernike && obs.opticsMdt.getValue(EMDL_IMAGE_EVEN_ZERNIKE_COEFFS, z, og))
		a.insert(a.end(), z.begin(), z.end());
	return a;
}

void setCurrentCtfs(const MetaDataTable& particles, ObservationModel& obs)
{
	if (!particles.containsLabel(EMDL_IMAGE_NAME) || !obs.opticsMdt.containsLabel(EMDL_OPTIMISER_DATA_ARE_CTF_PREMULTIPLIED))
		return;
	std::unordered_map<std::string, std::vector<ParticleCtf> > found;
	for (long i = 0; i < (long)particles.numberOfObjects(); i++)
	{
		const int og = obs.getOpticsGroup(particles, i);
		if (!obs.getCtfPremultiplied(og)) continue;
		FileName fn_img, fn_stack;
		long int no;
		particles.getValue(EMDL_IMAGE_NAME, fn_img, i);
		fn_img.decompose(no, fn_stack);
		if (no < 1) continue;
		CTF c;
		c.readByGroup(particles, &obs, i);
		std::vector<ParticleCtf>& list = found[fn_stack];
		if ((long)list.size() < no)
			list.resize(no, ParticleCtf{NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN});
		list[no - 1] = ParticleCtf{c.DeltafU, c.DeltafV, c.azimuthal_angle, c.kV, c.Cs, c.Q0, c.Bfac, c.scale, c.phase_shift,
		                           opticsAberrations(obs, og)};
	}

	std::lock_guard<std::mutex> guard(g_mutex);
	for (std::unordered_map<std::string, std::vector<ParticleCtf> >::iterator it = found.begin(); it != found.end(); ++it)
	{
		g_current_ctfs[it->first].swap(it->second);
		g_vstacks.erase(it->first);   // the next read takes the new CTF (and with it the new cache key)
	}
}

void prebuildEntries(const MetaDataTable& particles, int rank, int nr_ranks)
{
	const char* off = getenv("RELION_VPARTICLE_PREBUILD");
	if (nr_ranks < 2 || (off != NULL && off[0] == '0') || !particles.containsLabel(EMDL_IMAGE_NAME)) return;
	const std::string dir = cacheDirectory();
	if (dir.empty() || cacheUnwritable(dir)) return;

	// Every rank sees the same table, so every rank makes the same list and the same shares
	std::map<std::string, long> count;
	for (long i = 0; i < (long)particles.numberOfObjects(); i++)
	{
		FileName fn_img, fn_stack;
		long int no;
		particles.getValue(EMDL_IMAGE_NAME, fn_img, i);
		fn_img.decompose(no, fn_stack);
		if (no >= 1 && isVirtualStackFormat(fn_stack.getExtension())) count[fn_stack]++;
	}

	std::vector<std::pair<long, std::string> > todo;   // (particles, stack) of the entries that are missing
	for (std::map<std::string, long>::const_iterator it = count.begin(); it != count.end(); ++it)
	{
		const std::shared_ptr<VStack> v = getVStack(it->first);
		struct stat st;
		if (v->polish && stat(entryPath(dir, v->key).c_str(), &st) != 0)
			todo.push_back(std::make_pair(-it->second, it->first));
	}
	if (todo.empty()) return;

	// Largest first, each to the rank with the least so far: the work is about proportional to the particles
	std::sort(todo.begin(), todo.end());
	std::vector<long> load(nr_ranks, 0);
	std::vector<std::string> mine;
	for (size_t k = 0; k < todo.size(); k++)
	{
		const int r = (int)(std::min_element(load.begin(), load.end()) - load.begin());
		load[r] += 1 - todo[k].first;   // +1: reading a movie costs something for any number of particles
		if (r == rank) mine.push_back(todo[k].second);
	}
	if (rank == 0)
		std::cout << " Virtual polished particles: " << todo.size() << " micrographs are recombined once, shared by "
		          << nr_ranks << " processes, before the first iteration (RELION_VPARTICLE_PREBUILD=0 lets the readers do it)"
		          << std::endl;

	MultidimArray<RFLOAT> scratch;
	for (size_t k = 0; k < mine.size(); k++)
	{
		for (size_t d = 1; d <= (size_t)sumAheadDepth() && k + d < mine.size(); d++) sumAhead(getVStack(mine[k + d]));
		try { readThroughEntry(dir, *getVStack(mine[k]), 0, scratch); }
		catch (...) {}   // a reader that needs these particles cuts them itself and reports real errors
	}
	if (rank == 1 && g_sum_reads > 0)
		std::cout << " Virtual polished particles: process 1 read the frame sums of " << g_sum_reads << " micrographs ("
		          << g_sum_reads_ahead << " ahead, in the background, " << sumAheadDepth() << " at a time, "
		          << (getenv("RELION_VPOLISH_SUM_READ_AHEAD") ? "as RELION_VPOLISH_SUM_READ_AHEAD sets it"
		                                                     : "chosen from the measured wait; RELION_VPOLISH_SUM_READ_AHEAD sets it, 0 = off")
		          << ") and waited " << g_sum_wait_us / 1e6
		          << " s for them" << std::endl;
}

std::string micrographChecksum(const std::string& fn_mic)
{
	// A virtual movie average that replaced a real micrograph, after a bitwise
	// check, stands for that file: particles extracted from it still verify
	if (vmovies::isVirtualMovieAverageFile(fn_mic))
	{
		const std::string replaced = vmovies::replacedChecksum(fn_mic);
		if (!replaced.empty()) return replaced;
	}

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
                     const std::vector<double>& psi,
                     const std::vector<double>& cx, const std::vector<double>& cy,
                     const std::vector<double>& angle, const std::vector<ParticleCtf>& ctf)
{
	if (r.premultiplied() && (ctf.size() != coords.size() || (r.premultiply_size != Recipe::PREMULTIPLY_AUTO && r.premultiply_size < r.extract_size) || r.ctf_angpix <= 0.))
		REPORT_ERROR("writeDescriptor: a premultiplied virtual stack needs a CTF per particle, a premultiply box at least the extraction box and a pixel size.");
	if (r.premultiplied() && (r.rectangular() || r.rotated))
		REPORT_ERROR("writeDescriptor: rectangular or rotated virtual particles cannot be CTF-premultiplied.");
	if (r.rotated && (cx.size() != coords.size() || cy.size() != coords.size() || angle.size() != coords.size()))
		REPORT_ERROR("writeDescriptor: a rotated virtual stack needs an exact centre and a box angle per particle.");
	if ((r.rectangular() || r.rotated) && r.window > 0)
		REPORT_ERROR("writeDescriptor: rectangular or rotated virtual particles do not support re-windowing.");
	if (r.rectangular() && r.scale > 0 && r.outputSizeY() < 0)
		REPORT_ERROR("writeDescriptor: the rescaled height of a rectangular box must be an even whole number of pixels.");
	if (r.helical && psi.size() != coords.size())
		REPORT_ERROR("writeDescriptor: a helical virtual stack needs one psi angle per segment.");
	if (r.white_dust > 0 || r.black_dust > 0)
		REPORT_ERROR("writeDescriptor: virtual particles cannot use dust removal.");
	MetaDataTable head;
	head.setIsList(true);
	head.setName("vstack");
	head.addObject();
	const bool v2 = r.rectangular() || r.rotated;
	head.setValue(EMDL_VSTACK_VERSION, r.premultiplied() ? DESCRIPTOR_VERSION_PREMULT : v2 ? DESCRIPTOR_VERSION : DESCRIPTOR_VERSION_SQUARE);
	head.setValue(EMDL_MICROGRAPH_NAME, fn_mic);
	head.setValue(EMDL_VSTACK_MICROGRAPH_CHECKSUM, micrographChecksum(fn_mic));
	head.setValue(EMDL_VSTACK_EXTRACT_SIZE, r.extract_size);
	head.setValue(EMDL_VSTACK_RESCALE_SIZE, r.scale);
	head.setValue(EMDL_VSTACK_WINDOW_SIZE, r.window);
	if (r.rectangular())
	{
		head.setValue(EMDL_VSTACK_EXTRACT_SIZE_Y, r.extract_size_y);
		head.setValue(EMDL_IMAGE_SIZE_X, r.outputSize());
		head.setValue(EMDL_IMAGE_SIZE_Y, r.outputSizeY());
	}
	else
		head.setValue(EMDL_IMAGE_SIZE, r.outputSize());
	if (r.rotated)
	{
		head.setValue(EMDL_VSTACK_ROTATED, true);
		head.setValue(EMDL_VSTACK_INTERPOLATION, resampleMethodName(r.interpolation));
	}
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
	if (r.premultiplied())
	{
		head.setValue(EMDL_VSTACK_PREMULTIPLY_SIZE, r.premultiply_size);
		head.setValue(EMDL_VSTACK_CTF_INTACT_FIRST_PEAK, r.ctf_intact_first_peak);
		head.setValue(EMDL_VSTACK_CTF_ANGPIX, exactString(r.ctf_angpix));
	}

	MetaDataTable parts;
	parts.setName("particles");
	for (size_t i = 0; i < coords.size(); i++)
	{
		parts.addObject();
		parts.setValue(EMDL_IMAGE_COORD_X, (RFLOAT)coords[i].first);
		parts.setValue(EMDL_IMAGE_COORD_Y, (RFLOAT)coords[i].second);
		if (r.helical) parts.setValue(EMDL_VSTACK_PSI, exactString(psi[i]));
		if (r.rotated)
		{
			parts.setValue(EMDL_VSTACK_CENTRE_X, exactString(cx[i]));
			parts.setValue(EMDL_VSTACK_CENTRE_Y, exactString(cy[i]));
			parts.setValue(EMDL_VSTACK_ANGLE, exactString(angle[i]));
		}
		if (r.premultiplied()) parts.setValue(EMDL_VSTACK_CTF, ctfString(ctf[i]));
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
	h.box_y = v->recipe.outputSizeY();
	h.angpix = v->recipe.angpix;
	h.float16 = v->recipe.float16;
	return h;
}

void orderByMicrograph(const std::vector<std::string>& group, const std::vector<double>& y,
                       const std::vector<double>& x, int nr_ranks,
                       std::vector<long>& order, std::vector<long>& cut)
{
	const long n = (long)group.size();
	order.resize(n);
	for (long i = 0; i < n; i++)
		order[i] = i;
	std::stable_sort(order.begin(), order.end(), [&](long a, long b)
	{
		int c = group[a].compare(group[b]);
		if (c != 0) return c < 0;
		if (y[a] != y[b]) return y[a] < y[b];
		return x[a] < x[b];
	});

	std::vector<long> bounds; // positions in `order` where a new micrograph starts
	for (long i = 1; i < n; i++)
		if (group[order[i]] != group[order[i - 1]])
			bounds.push_back(i);

	if (nr_ranks < 1) nr_ranks = 1;
	cut.assign(nr_ranks + 1, n);
	cut[0] = 0;
	const double share = (double)n / nr_ranks;
	for (int r = 1; r < nr_ranks; r++)
	{
		long target = (long)(share * r + 0.5);
		long best = target;
		long best_dist = (long)(share * 0.25);
		std::vector<long>::const_iterator it = std::lower_bound(bounds.begin(), bounds.end(), target);
		for (int k = 0; k < 2; k++)
		{
			std::vector<long>::const_iterator c = (k == 0) ? it : (it == bounds.begin() ? bounds.end() : it - 1);
			if (c == bounds.end()) continue;
			long d = std::labs(*c - target);
			if (d <= best_dist) { best = *c; best_dist = d; }
		}
		cut[r] = std::max(best, cut[r - 1]);
	}
}

void readParticle(const std::string& fn_vstack, long index, MultidimArray<RFLOAT>& out)
{
	std::shared_ptr<VStack> v = getVStack(fn_vstack);
	if (index < 0 || index >= (long)v->x.size())
		REPORT_ERROR("Particle " + integerToString(index + 1) + " requested from virtual stack "
		             + fn_vstack + ", which holds " + integerToString(v->x.size()) + ".");

	const std::string dir = cacheDirectory();
	if (!dir.empty())
	{
		// Entries already written are read even while writing is paused after a failure
		const std::string path = entryPath(dir, v->key);
		if (readFromEntry(path, *v, index, out)) return;
		if (!cacheUnwritable(dir))
		{
			if (g_builds.builders() == 0)
			{
				if (readThroughEntry(dir, *v, index, out)) return;
			}
			else
				g_builds.add(dir, v);   // and cut this particle now, below
		}
	}

	// No cache: cut just this particle; only the pages under its box are read
	verifyMicrograph(*v);
	std::shared_ptr<Micrograph> m = v->polish ? std::shared_ptr<Micrograph>() : openMicrograph(v->mic);
	extractIndex(m.get(), *v, index, out);
}

double micrographPixelSize(const std::string& fn_vstack)
{
	std::shared_ptr<VStack> v = getVStack(fn_vstack);
	if (v->polish)
		REPORT_ERROR("Virtual particle stack " + fn_vstack + " holds polished particles, which come from movie frames: "
		             "fused extraction from the micrograph cannot make them.");
	const Recipe& r = v->recipe;
	return (r.scale > 0) ? r.angpix * r.scale / r.extract_size : r.angpix;
}

void sameExtentBox(const std::string& fn_vstack, double angpix, int& nx, int& ny)
{
	std::shared_ptr<VStack> v = getVStack(fn_vstack);
	if (v->polish)
		REPORT_ERROR("Virtual particle stack " + fn_vstack + " holds polished particles, which come from movie frames: "
		             "fused extraction from the micrograph cannot make them.");
	const Recipe& r = v->recipe;
	const double f = r.angpix / angpix;
	nx = 2 * (int)std::ceil(r.outputSize() * f / 2. - 1e-9);
	ny = 2 * (int)std::ceil(r.outputSizeY() * f / 2. - 1e-9);
}

namespace {
void fuseFromMicrograph(const Micrograph& m, const fusedextract::Geometry& g, const fusedextract::Normalisation& nrm,
                        int nx, int ny, MultidimArray<Complex>& F2D)
{
	const Micrograph* mp = &m;
	auto at = [mp](long y, long x) { return mp->at(y, x); };
	fusedextract::Samples samples;
	fusedextract::gather(at, m.nx, m.ny, g, samples);
	fusedextract::normaliseSamples(samples, nx, ny, nrm);
	fusedextract::fourierTransform(samples, g, F2D);
}
} // namespace

void readParticleFourier(const std::string& fn_vstack, long index, double angpix, int nx, int ny,
                         MultidimArray<Complex>& F2D)
{
	std::shared_ptr<VStack> v = getVStack(fn_vstack);
	if (index < 0 || index >= (long)v->x.size())
		REPORT_ERROR("Particle " + integerToString(index + 1) + " requested from virtual stack "
		             + fn_vstack + ", which holds " + integerToString(v->x.size()) + ".");
	const Recipe& r = v->recipe;
	if (r.white_dust > 0. || r.black_dust > 0.)
		REPORT_ERROR("Fused extraction does not support dust removal (" + fn_vstack + ").");
	if (r.premultiplied())
		REPORT_ERROR("Fused extraction does not support CTF-premultiplied particles (" + fn_vstack + ").");

	verifyMicrograph(*v);
	std::shared_ptr<Micrograph> m = openMicrograph(v->mic);

	const double mic_angpix = micrographPixelSize(fn_vstack);
	const double to_new = r.angpix / angpix;   // recipe pixels -> output pixels

	fusedextract::Geometry g;
	g.nx = nx; g.ny = ny;
	g.step = angpix / mic_angpix;
	const bool rot = r.rotated;
	g.cx = rot ? v->cx[index] : (double)v->x[index];
	g.cy = rot ? v->cy[index] : (double)v->y[index];
	g.angle = rot ? v->angle[index] : 0.;

	fusedextract::Normalisation nrm;
	nrm.normalise = r.normalise;
	nrm.ramp = r.ramp;
	nrm.invert = r.invert_contrast;
	nrm.helical = r.helical;
	nrm.bg_radius = r.bg_radius * to_new;
	nrm.helical_radius = r.helical_radius * to_new;
	nrm.psi = psiOf(*v, index);

	fuseFromMicrograph(*m, g, nrm, nx, ny, F2D);
}

void readParticleFourierDirect(const std::string& fn_mic, double cx, double cy, double box_angle, double psi_prior,
                               const DirectRecipe& d, double angpix, int nx, int ny,
                               MultidimArray<Complex>& F2D)
{
	if (d.mic_angpix <= 0. || angpix <= 0.)
		REPORT_ERROR("Fused extraction from coordinates needs the micrograph and the output pixel sizes.");
	if (d.helical && !d.normalise)
		REPORT_ERROR("Fused extraction: a helical background only matters with normalisation.");

	std::shared_ptr<Micrograph> m = openMicrograph(fn_mic);

	fusedextract::Geometry g;
	g.nx = nx; g.ny = ny;
	g.step = angpix / d.mic_angpix;
	if (d.rotate)
	{
		// As relion_preprocess --rotate_to_horizontal: the box is turned by box_angle around the exact coordinate
		g.cx = cx; g.cy = cy; g.angle = box_angle;
	}
	else
	{
		// Unturned boxes are cut around whole pixels; relion_preprocess rounds the coordinate to the nearest
		g.cx = std::floor(cx + 0.5); g.cy = std::floor(cy + 0.5); g.angle = 0.;
	}

	fusedextract::Normalisation nrm;
	nrm.normalise = d.normalise;
	nrm.ramp = d.ramp;
	nrm.invert = d.invert_contrast;
	nrm.helical = d.helical;
	nrm.bg_radius = d.bg_radius;
	nrm.helical_radius = d.helical_diameter * 0.5 / angpix;
	nrm.psi = d.rotate ? 0. : psi_prior;

	fuseFromMicrograph(*m, g, nrm, nx, ny, F2D);
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

bool removeProjectCache(const std::string& project_dir)
{
	const std::string dir = project_dir + "/" + PROJECT_CACHE_DIR;
	struct stat st;
	if (lstat(dir.c_str(), &st) != 0) return false;
	const bool ok = vcache::removeTree(dir);
	// Leave Cache/ itself only if something else lives there
	rmdir((project_dir + "/Cache").c_str());
	return ok;
}

bool polishedSumsWanted()
{
	return sumsKept() && !cacheDirectory().empty();
}

void storePolishedSum(const std::string& fn_vstack, MicrographHandler& mh, int threads)
{
	const std::shared_ptr<VStack> v = getVStack(fn_vstack);
	Sums sums(*v);
	makeSums(*v, mh, sums, threads);
	storeSum(*v, sums, threads);
}

void readUnpremultiplied()
{
	std::lock_guard<std::mutex> guard(g_mutex);
	g_read_unpremultiplied = true;
}

void finishCacheBuilds()
{
	g_builds.wait();
}

void clearProcessState()
{
	g_builds.reset();
	std::lock_guard<std::mutex> guard(g_mutex);
	g_vstacks.clear();
	g_verified.clear();
	g_unwritable.clear();
	g_mics.clear();
	g_mic_order.clear();
	g_written_since_scan = -1;
}

} // namespace vparticles
