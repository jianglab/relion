/*
 * tests/unit/test_virtual_particles.cpp
 *
 * Virtual particle stacks (.vstack): particles computed from their micrograph
 * when read, through a disposable cache. The bitwise match with what
 * relion_preprocess writes is tested end to end in
 * tests/integration/test_virtual_particles.py; these pin down the reader and
 * the cache - that every way of obtaining a particle gives the same pixels, and
 * that the failure modes (a changed micrograph, a damaged or deleted cache, an
 * attempt to write to a descriptor) fail safe.
 */

#include <catch2/catch.hpp>

#include "src/virtual_particles.h"
#include "src/image.h"
#include "src/error.h"
#include "src/float16.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

/// A temporary directory that removes itself.
class TempDir {
public:
	TempDir()
	{
		char tmpl[] = "/tmp/relion_vparticles_XXXXXX";
		const char* p = mkdtemp(tmpl);
		REQUIRE(p != NULL);
		path_ = p;
	}
	~TempDir()
	{
		// Undo any permission change a test made before removing
		const std::string cmd = "chmod -R u+rwx '" + path_ + "' 2>/dev/null; rm -rf '" + path_ + "'";
		if (system(cmd.c_str()) != 0) { /* best effort */ }
	}
	const std::string& path() const { return path_; }
private:
	std::string path_;
};

/// Point the particle cache somewhere for the duration of a test.
class CacheSetting {
public:
	explicit CacheSetting(const std::string& value)
	{
		const char* old = getenv("RELION_VPARTICLE_CACHE");
		had_ = (old != NULL);
		if (had_) old_ = old;
		setenv("RELION_VPARTICLE_CACHE", value.c_str(), 1);
		vparticles::clearProcessState();
	}
	~CacheSetting()
	{
		if (had_) setenv("RELION_VPARTICLE_CACHE", old_.c_str(), 1);
		else unsetenv("RELION_VPARTICLE_CACHE");
		vparticles::clearProcessState();
	}
private:
	bool had_;
	std::string old_;
};

const int NX = 300, NY = 260;

/// A micrograph with structure and noise, deterministic.
void writeMicrograph(const std::string& fn, DataType type, unsigned seed = 1)
{
	Image<RFLOAT> mic(NX, NY);
	unsigned s = seed;
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(mic())
	{
		s = s * 1103515245u + 12345u;
		const double noise = ((s >> 8) & 0xffff) / 65536.0 - 0.5;
		DIRECT_A2D_ELEM(mic(), i, j) = 3.0 * sin(0.07 * i) * cos(0.05 * j) + noise + 0.01 * i;
	}
	mic.write(fn, -1, false, WRITE_OVERWRITE, type);
}

vparticles::Recipe binnedRecipe()
{
	vparticles::Recipe r;
	r.extract_size = 64;
	r.scale = 48;
	r.normalise = true;
	r.bg_radius = 18;
	r.ramp = true;
	r.invert_contrast = true;
	r.float16 = true;
	r.angpix = 1.5 * 64 / 48;
	return r;
}

/// Centres inside, and hanging off every edge of, the micrograph.
std::vector<std::pair<long, long> > centres()
{
	std::vector<std::pair<long, long> > c;
	c.push_back(std::make_pair(150L, 130L));
	c.push_back(std::make_pair(40L, 50L));
	c.push_back(std::make_pair(5L, 7L));             // off the top-left corner
	c.push_back(std::make_pair(NX - 3L, 130L));      // off the right edge
	c.push_back(std::make_pair(150L, NY - 10L));     // off the bottom edge
	c.push_back(std::make_pair(220L, 90L));
	return c;
}

std::vector<MultidimArray<RFLOAT> > readAll(const std::string& fn_vstack, size_t n)
{
	std::vector<MultidimArray<RFLOAT> > out(n);
	for (size_t i = 0; i < n; i++) vparticles::readParticle(fn_vstack, i, out[i]);
	return out;
}

/// Same size and bitwise the same values. Origins are not compared: an image
/// read from disk starts at 0, a particle straight from the reader is centred.
bool identical(const MultidimArray<RFLOAT>& a, const MultidimArray<RFLOAT>& b)
{
	if (NSIZE(a) != NSIZE(b) || ZSIZE(a) != ZSIZE(b) || YSIZE(a) != YSIZE(b) || XSIZE(a) != XSIZE(b))
		return false;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a)
		if (DIRECT_MULTIDIM_ELEM(a, n) != DIRECT_MULTIDIM_ELEM(b, n)) return false;
	return true;
}

bool identical(const std::vector<MultidimArray<RFLOAT> >& a, const std::vector<MultidimArray<RFLOAT> >& b)
{
	if (a.size() != b.size()) return false;
	for (size_t i = 0; i < a.size(); i++) if (!identical(a[i], b[i])) return false;
	return true;
}

std::string findEntry(const std::string& dir)
{
	const std::string cmd = "find '" + dir + "' -name '*.vpc' | head -1";
	FILE* p = popen(cmd.c_str(), "r");
	char buf[4096] = {0};
	if (p) { if (!fgets(buf, sizeof(buf), p)) buf[0] = 0; pclose(p); }
	std::string s(buf);
	while (!s.empty() && (s.back() == '\n')) s.pop_back();
	return s;
}

} // namespace

TEST_CASE("direct extraction, cache miss and cache hit give identical particles", "[vparticles]")
{
	TempDir tmp;
	const std::string mic = tmp.path() + "/mic.mrc", vs = tmp.path() + "/mic.vstack";
	writeMicrograph(mic, Float);
	vparticles::writeDescriptor(vs, mic, binnedRecipe(), centres());

	std::vector<MultidimArray<RFLOAT> > direct, miss, hit;
	{
		CacheSetting off("off");
		direct = readAll(vs, centres().size());
	}
	{
		CacheSetting on(tmp.path() + "/cache");
		miss = readAll(vs, centres().size());
		REQUIRE(findEntry(tmp.path() + "/cache") != "");
		hit = readAll(vs, centres().size());
	}
	REQUIRE(direct[0].xdim == 48);
	CHECK(identical(direct, miss));
	CHECK(identical(direct, hit));
}

TEST_CASE("particles carry the rounding of the stack they replace", "[vparticles]")
{
	TempDir tmp;
	CacheSetting off("off");
	const std::string mic = tmp.path() + "/mic.mrc";
	writeMicrograph(mic, Float);

	vparticles::Recipe half = binnedRecipe();
	vparticles::Recipe single = binnedRecipe();
	single.float16 = false;
	vparticles::writeDescriptor(tmp.path() + "/h.vstack", mic, half, centres());
	vparticles::writeDescriptor(tmp.path() + "/s.vstack", mic, single, centres());

	MultidimArray<RFLOAT> h, f;
	vparticles::readParticle(tmp.path() + "/h.vstack", 0, h);
	vparticles::readParticle(tmp.path() + "/s.vstack", 0, f);
	bool all_half = true, all_single = true;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(h)
	{
		const RFLOAT v = DIRECT_MULTIDIM_ELEM(h, n);
		all_half &= ((RFLOAT)half2float(float2half((float)v)) == v);
		const RFLOAT w = DIRECT_MULTIDIM_ELEM(f, n);
		all_single &= ((RFLOAT)(float)w == w);
	}
	CHECK(all_half);
	CHECK(all_single);
	CHECK_FALSE(identical(h, f));   // the recipes really do differ
}

TEST_CASE("deleting the cache between reads changes nothing", "[vparticles]")
{
	TempDir tmp;
	const std::string mic = tmp.path() + "/mic.mrc", vs = tmp.path() + "/mic.vstack";
	writeMicrograph(mic, Float16);
	vparticles::writeDescriptor(vs, mic, binnedRecipe(), centres());

	CacheSetting on(tmp.path() + "/cache");
	const std::vector<MultidimArray<RFLOAT> > before = readAll(vs, centres().size());
	REQUIRE(system(("rm -rf '" + tmp.path() + "/cache'").c_str()) == 0);
	// Same process, so the reader has already seen the entry once
	const std::vector<MultidimArray<RFLOAT> > after = readAll(vs, centres().size());
	CHECK(identical(before, after));
	CHECK(findEntry(tmp.path() + "/cache") != "");   // and it was rebuilt
}

TEST_CASE("a damaged cache entry is rebuilt, not trusted", "[vparticles]")
{
	TempDir tmp;
	const std::string mic = tmp.path() + "/mic.mrc", vs = tmp.path() + "/mic.vstack";
	writeMicrograph(mic, Float);
	vparticles::writeDescriptor(vs, mic, binnedRecipe(), centres());

	std::vector<MultidimArray<RFLOAT> > reference;
	std::string entry;
	struct stat full;
	{
		CacheSetting on(tmp.path() + "/cache");
		reference = readAll(vs, centres().size());
		entry = findEntry(tmp.path() + "/cache");
		REQUIRE(entry != "");
		REQUIRE(stat(entry.c_str(), &full) == 0);
	}

	// Cut it short, as a killed writer on another node might have left it
	REQUIRE(truncate(entry.c_str(), full.st_size / 2) == 0);

	CacheSetting on(tmp.path() + "/cache");   // a fresh process, in effect
	CHECK(identical(reference, readAll(vs, centres().size())));
	struct stat now;
	REQUIRE(stat(entry.c_str(), &now) == 0);
	CHECK(now.st_size == full.st_size);
}

TEST_CASE("a micrograph that changed after extraction is refused", "[vparticles]")
{
	TempDir tmp;
	const std::string mic = tmp.path() + "/mic.mrc", vs = tmp.path() + "/mic.vstack";
	writeMicrograph(mic, Float, 1);
	vparticles::writeDescriptor(vs, mic, binnedRecipe(), centres());
	writeMicrograph(mic, Float, 2);   // e.g. motion correction re-run in place

	CacheSetting off("off");
	MultidimArray<RFLOAT> p;
	bool refused = false;
	try { vparticles::readParticle(vs, 0, p); }
	catch (const RelionError& e)
	{
		refused = true;
		CHECK(e.msg.find("has changed") != std::string::npos);
	}
	CHECK(refused);
}

TEST_CASE("header-only reads of a virtual stack need no micrograph", "[vparticles]")
{
	TempDir tmp;
	CacheSetting off("off");
	const std::string mic = tmp.path() + "/mic.mrc", vs = tmp.path() + "/mic.vstack";
	writeMicrograph(mic, Float);
	const vparticles::Recipe r = binnedRecipe();
	vparticles::writeDescriptor(vs, mic, r, centres());

	// Whole stack, through the ordinary image reader
	Image<RFLOAT> all;
	all.read(vs);
	CHECK(XSIZE(all()) == 48);
	CHECK(NSIZE(all()) == (long)centres().size());

	// One particle by name, as every program addresses them
	Image<RFLOAT> third;
	third.read("3@" + vs);
	MultidimArray<RFLOAT> direct;
	vparticles::readParticle(vs, 2, direct);
	MultidimArray<RFLOAT> slice;
	all().getImage(2, slice);
	CHECK(identical(third(), direct));
	CHECK(identical(slice, direct));

	// Header only, with the micrograph gone
	REQUIRE(unlink(mic.c_str()) == 0);
	vparticles::clearProcessState();
	Image<RFLOAT> head;
	head.read("3@" + vs, false);
	CHECK(XSIZE(head()) == 48);
	CHECK(YSIZE(head()) == 48);
	CHECK(head.samplingRateX() == Approx(r.angpix));
}

TEST_CASE("writing an image to a .vstack is refused and leaves it intact", "[vparticles]")
{
	TempDir tmp;
	const std::string mic = tmp.path() + "/mic.mrc", vs = tmp.path() + "/mic.vstack";
	writeMicrograph(mic, Float);
	vparticles::writeDescriptor(vs, mic, binnedRecipe(), centres());
	struct stat before;
	REQUIRE(stat(vs.c_str(), &before) == 0);

	Image<RFLOAT> img(48, 48);
	REQUIRE_THROWS_AS(img.write(vs), RelionError);
	REQUIRE_THROWS_AS(img.write("2@" + vs, -1, false, WRITE_REPLACE), RelionError);

	struct stat after;
	REQUIRE(stat(vs.c_str(), &after) == 0);
	CHECK(after.st_size == before.st_size);
	vparticles::clearProcessState();
	CHECK(vparticles::readHeader(vs).n == (long)centres().size());
}

TEST_CASE("asking for a particle that is not there is an error", "[vparticles]")
{
	TempDir tmp;
	CacheSetting off("off");
	const std::string mic = tmp.path() + "/mic.mrc", vs = tmp.path() + "/mic.vstack";
	writeMicrograph(mic, Float);
	vparticles::writeDescriptor(vs, mic, binnedRecipe(), centres());

	MultidimArray<RFLOAT> p;
	CHECK_THROWS_AS(vparticles::readParticle(vs, centres().size(), p), RelionError);
	CHECK_THROWS_AS(vparticles::readParticle(vs, -1, p), RelionError);
	Image<RFLOAT> img;
	CHECK_THROWS_AS(img.read(integerToString(centres().size() + 1) + "@" + vs), RelionError);
}

TEST_CASE("memory-mapped and conventionally read micrographs agree", "[vparticles]")
{
	TempDir tmp;
	CacheSetting off("off");
	// The same pixels as MRC (mapped) and SPIDER (read whole by Image)
	writeMicrograph(tmp.path() + "/mic.mrc", Float);
	writeMicrograph(tmp.path() + "/mic.spi", Float);
	vparticles::writeDescriptor(tmp.path() + "/m.vstack", tmp.path() + "/mic.mrc", binnedRecipe(), centres());
	vparticles::writeDescriptor(tmp.path() + "/s.vstack", tmp.path() + "/mic.spi", binnedRecipe(), centres());

	CHECK(identical(readAll(tmp.path() + "/m.vstack", centres().size()),
	                readAll(tmp.path() + "/s.vstack", centres().size())));
}

TEST_CASE("parallel reads match serial reads", "[vparticles]")
{
	TempDir tmp;
	const std::string mic = tmp.path() + "/mic.mrc", vs = tmp.path() + "/mic.vstack";
	writeMicrograph(mic, Float16);
	std::vector<std::pair<long, long> > many;
	for (long k = 0; k < 40; k++) many.push_back(std::make_pair(20 + (k * 37) % 260, 15 + (k * 53) % 230));
	vparticles::writeDescriptor(vs, mic, binnedRecipe(), many);

	std::vector<MultidimArray<RFLOAT> > serial;
	{
		CacheSetting off("off");
		serial = readAll(vs, many.size());
	}

	// Many threads racing to build the same entry, as CtfRefine's loader does
	CacheSetting on(tmp.path() + "/cache");
	std::vector<MultidimArray<RFLOAT> > parallel(many.size());
	#pragma omp parallel for num_threads(8)
	for (long i = 0; i < (long)many.size(); i++)
		vparticles::readParticle(vs, i, parallel[i]);
	CHECK(identical(serial, parallel));
}

TEST_CASE("an unwritable cache falls back to direct extraction", "[vparticles]")
{
	if (geteuid() == 0) return;   // permissions do not stop root

	TempDir tmp;
	const std::string mic = tmp.path() + "/mic.mrc", vs = tmp.path() + "/mic.vstack";
	writeMicrograph(mic, Float);
	vparticles::writeDescriptor(vs, mic, binnedRecipe(), centres());

	std::vector<MultidimArray<RFLOAT> > direct;
	{
		CacheSetting off("off");
		direct = readAll(vs, centres().size());
	}

	const std::string locked = tmp.path() + "/locked";
	REQUIRE(mkdir(locked.c_str(), 0555) == 0);
	CacheSetting on(locked + "/cache");
	CHECK(identical(direct, readAll(vs, centres().size())));
	CHECK(findEntry(locked) == "");
}

TEST_CASE("a descriptor under a .mrcs name is read as a virtual stack", "[vparticles]")
{
	// What relion_stacks_to_virtual leaves behind: the stack's own name, so no
	// STAR file that refers to it has to change
	TempDir tmp;
	CacheSetting off("off");
	const std::string mic = tmp.path() + "/mic.mrc", stack = tmp.path() + "/mic.mrcs";
	writeMicrograph(mic, Float);
	vparticles::writeDescriptor(stack, mic, binnedRecipe(), centres());
	REQUIRE(vparticles::isVirtualStackFile(stack));

	Image<RFLOAT> second;
	second.read("2@" + stack);
	MultidimArray<RFLOAT> direct;
	vparticles::readParticle(stack, 1, direct);
	CHECK(identical(second(), direct));

	Image<RFLOAT> head;
	head.read(stack, false);
	CHECK(NSIZE(head()) == (long)centres().size());
	CHECK(XSIZE(head()) == 48);

	// A real stack of the same name is still an ordinary MRC stack
	const std::string real = tmp.path() + "/real.mrcs";
	Image<RFLOAT> out(48, 48, 1, 3);
	out().initConstant(1.25);
	out.write(real);
	CHECK_FALSE(vparticles::isVirtualStackFile(real));
	Image<RFLOAT> back;
	back.read("3@" + real);
	CHECK(DIRECT_A2D_ELEM(back(), 10, 10) == 1.25);
}

TEST_CASE("the cache stays within its size limit", "[vparticles]")
{
	TempDir tmp;
	const std::string mic = tmp.path() + "/mic.mrc";
	writeMicrograph(mic, Float);

	// Ten descriptors, each a cache entry of ~28 KB; room for about four
	const char* old = getenv("RELION_VPARTICLE_CACHE_MAX_GB");
	const std::string old_value = old ? old : "";
	setenv("RELION_VPARTICLE_CACHE_MAX_GB", "0.0001", 1);   // ~107 KB
	std::vector<std::string> stacks;
	for (int k = 0; k < 10; k++)
	{
		std::vector<std::pair<long, long> > c = centres();
		for (size_t i = 0; i < c.size(); i++) c[i].first += k;   // a different entry each time
		stacks.push_back(tmp.path() + "/s" + integerToString(k) + ".vstack");
		vparticles::writeDescriptor(stacks.back(), mic, binnedRecipe(), c);
	}

	std::vector<std::vector<MultidimArray<RFLOAT> > > first;
	{
		CacheSetting on(tmp.path() + "/cache");
		for (size_t k = 0; k < stacks.size(); k++) first.push_back(readAll(stacks[k], centres().size()));

		long long total = 0;
		FILE* p = popen(("find '" + tmp.path() + "/cache' -name '*.vpc' -printf '%s\\n'").c_str(), "r");
		char line[64];
		while (p && fgets(line, sizeof(line), p)) total += atoll(line);
		if (p) pclose(p);
		CHECK(total > 0);
		CHECK(total <= (long long)(0.0001 * 1024 * 1024 * 1024));

		// Evicted entries come back from the micrographs, unchanged
		for (size_t k = 0; k < stacks.size(); k++)
			CHECK(identical(first[k], readAll(stacks[k], centres().size())));
	}
	if (old) setenv("RELION_VPARTICLE_CACHE_MAX_GB", old_value.c_str(), 1);
	else unsetenv("RELION_VPARTICLE_CACHE_MAX_GB");
}

TEST_CASE("removing a project cache needs no shell and stays inside the cache", "[vparticles]")
{
	TempDir tmp;
	// A quote in the project path would have broken the old `rm -rf '...'`
	const std::string project = tmp.path() + "/it's a project";
	const std::string outside = tmp.path() + "/outside";
	REQUIRE(mkdir(project.c_str(), 0755) == 0);
	REQUIRE(mkdir(outside.c_str(), 0755) == 0);
	REQUIRE(mkdir((project + "/Cache").c_str(), 0755) == 0);
	const std::string cache = project + "/Cache/virtual_particles";
	REQUIRE(mkdir(cache.c_str(), 0755) == 0);
	REQUIRE(mkdir((cache + "/sub").c_str(), 0755) == 0);
	FILE* f = fopen((cache + "/sub/entry.vpc").c_str(), "w"); REQUIRE(f); fclose(f);
	f = fopen((outside + "/keep").c_str(), "w"); REQUIRE(f); fclose(f);
	REQUIRE(symlink(outside.c_str(), (cache + "/link").c_str()) == 0);

	CHECK(vparticles::removeProjectCache(project));
	struct stat st;
	CHECK(lstat(cache.c_str(), &st) != 0);
	CHECK(lstat((project + "/Cache").c_str(), &st) != 0);     // empty, so removed too
	CHECK(lstat((outside + "/keep").c_str(), &st) == 0);      // the link was not followed
	CHECK_FALSE(vparticles::removeProjectCache(project));     // nothing left to remove
}
