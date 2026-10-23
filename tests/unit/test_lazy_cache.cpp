/*
 * tests/unit/test_lazy_cache.cpp
 *
 * The background cache fill (RELION_CACHE_LAZY): files are copied completely and with the source's
 * size and time, reading a cache path that is not there yet falls back to the original, stale cache
 * files are replaced, and a stack locked by another process is left to it.
 */

#include <catch2/catch.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <vector>
#include <sstream>
#include <string>
#include <thread>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utime.h>

#include "src/lazy_cache.h"
#include "src/image.h"

namespace
{

struct TempDir
{
	std::string path;
	TempDir()
	{
		const char *base = getenv("TMPDIR");
		std::string tmpl = std::string(base ? base : "/tmp") + "/relion_lazycache_XXXXXX";
		std::vector<char> buf(tmpl.begin(), tmpl.end());
		buf.push_back('\0');
		REQUIRE(mkdtemp(buf.data()) != NULL);
		path = buf.data();
	}
	~TempDir()
	{
		std::string cmd = "rm -rf '" + path + "'";
		if (system(cmd.c_str()) != 0) {}
	}
};

std::string readAll(const std::string &p)
{
	std::ifstream f(p, std::ios::binary);
	std::stringstream ss;
	ss << f.rdbuf();
	return ss.str();
}

void writeFile(const std::string &p, const std::string &content, time_t mtime)
{
	{
		std::ofstream f(p, std::ios::binary);
		f << content;
	}
	struct utimbuf t = { mtime, mtime };
	utime(p.c_str(), &t);
}

bool waitUntil(const std::function<bool()> &ok, double seconds = 20.)
{
	auto t0 = std::chrono::steady_clock::now();
	while (!ok())
	{
		if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > seconds)
			return false;
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	}
	return true;
}

struct LazyOn
{
	LazyOn() { setenv("RELION_CACHE_LAZY", "1", 1); }
	~LazyOn() { lazycache::stop(); unsetenv("RELION_CACHE_LAZY"); }
};

}

TEST_CASE("lazy cache: on unless RELION_CACHE_LAZY is 0", "[lazy_cache]")
{
	unsetenv("RELION_CACHE_LAZY");
	REQUIRE(lazycache::enabled());
	setenv("RELION_CACHE_LAZY", "0", 1);
	REQUIRE(lazycache::resolve("/does/not/exist.mrcs") == "");
	REQUIRE_FALSE(lazycache::enabled());
	setenv("RELION_CACHE_LAZY", "1", 1);
	REQUIRE(lazycache::enabled());
	unsetenv("RELION_CACHE_LAZY");
}

TEST_CASE("lazy cache: stacks are copied whole, with size and time of the source", "[lazy_cache]")
{
	LazyOn on;
	TempDir d;
	std::string src_dir = d.path + "/src", cache = d.path + "/cache/relion_cache/";
	REQUIRE(system(("mkdir -p '" + src_dir + "' '" + cache + "'").c_str()) == 0);

	std::string big(9 * 1000 * 1000 + 17, 'x'); // more than two copy chunks
	for (size_t i = 0; i < big.size(); i += 7)
		big[i] = char('a' + i % 23);
	const int n = 5;
	for (int i = 0; i < n; i++)
	{
		std::string c = (i == 0) ? big : "stack " + std::to_string(i);
		writeFile(src_dir + "/s" + std::to_string(i) + ".mrcs", c, 1700000000 + 100 * i);
		lazycache::add(cache + "key/s" + std::to_string(i) + ".mrcs", src_dir + "/s" + std::to_string(i) + ".mrcs");
	}
	lazycache::start(4, 0);

	for (int i = 0; i < n; i++)
	{
		std::string c = cache + "key/s" + std::to_string(i) + ".mrcs", s = src_dir + "/s" + std::to_string(i) + ".mrcs";
		REQUIRE(waitUntil([&]() { return access(c.c_str(), F_OK) == 0; }));
		struct stat a, b;
		REQUIRE(stat(c.c_str(), &a) == 0);
		REQUIRE(stat(s.c_str(), &b) == 0);
		CHECK(a.st_size == b.st_size);
		CHECK(a.st_mtime == b.st_mtime);
		CHECK(readAll(c) == readAll(s));
	}
	lazycache::stop();
	// no temporary or lock files left behind
	for (int i = 0; i < n; i++)
	{
		std::string c = cache + "key/s" + std::to_string(i) + ".mrcs";
		CHECK(access((c + ".lazytmp").c_str(), F_OK) != 0);
		CHECK(access((c + ".lazylock").c_str(), F_OK) != 0);
	}
}

TEST_CASE("lazy cache: a stack that is not copied yet is read from the original", "[lazy_cache]")
{
	LazyOn on;
	TempDir d;
	std::string cache = d.path + "/cache/relion_cache/";
	REQUIRE(system(("mkdir -p '" + d.path + "/src' '" + cache + "k'").c_str()) == 0);
	std::string src = d.path + "/src/a.mrcs", c = cache + "k/a.mrcs";
	writeFile(src, "original", 1700000000);

	SECTION("queued stack")
	{
		// queued but the copier not started: reading must go to the source
		lazycache::add(c, src);
		CHECK(lazycache::resolve(c) == src);
		CHECK(lazycache::resolve(cache + "k/other.mrcs") == "");
	}

	SECTION("stack this process did not queue, found from the cache path")
	{
		lazycache::setRoot(cache);
		// the cache path under the root is <key>/<original path without leading slash>
		std::string rel = src.substr(1);
		std::string cp = cache + "key1/" + rel;
		CHECK(lazycache::resolve(cp) == src);
		CHECK(lazycache::resolve(cache + "key1/no/such/file.mrcs") == "");
		lazycache::setRoot("");
	}
}

TEST_CASE("lazy cache: fImageHandler opens the original while the cache copy is missing", "[lazy_cache]")
{
	LazyOn on;
	TempDir d;
	std::string cache = d.path + "/cache/relion_cache/";
	REQUIRE(system(("mkdir -p '" + d.path + "/src' '" + cache + "k'").c_str()) == 0);
	std::string src = d.path + "/src/img.mrcs", c = cache + "k/img.mrcs";

	Image<RFLOAT> out(8, 8, 1, 3);
	for (long int i = 0; i < 8 * 8 * 3; i++)
		out().data[i] = RFLOAT(i % 11);
	out.write(src, -1, true);

	lazycache::add(c, src);
	Image<RFLOAT> in;
	in.read(c); // not in the cache yet: reads the original
	REQUIRE(NSIZE(in()) == 3);
	for (long int i = 0; i < 8 * 8 * 3; i++)
		REQUIRE(in().data[i] == out().data[i]);

	lazycache::start(2, 0);
	REQUIRE(waitUntil([&]() { return access(c.c_str(), F_OK) == 0; }));
	Image<RFLOAT> in2;
	in2.read(c); // now from the cache
	for (long int i = 0; i < 8 * 8 * 3; i++)
		REQUIRE(in2().data[i] == out().data[i]);
}

TEST_CASE("lazy cache: a stale cache file is replaced, not read", "[lazy_cache]")
{
	LazyOn on;
	TempDir d;
	std::string cache = d.path + "/cache/relion_cache/";
	REQUIRE(system(("mkdir -p '" + d.path + "/src' '" + cache + "k'").c_str()) == 0);
	std::string src = d.path + "/src/a.mrcs", c = cache + "k/a.mrcs";
	writeFile(src, "new content", 1700000500);
	writeFile(c, "old content", 1700000000); // same kind of file, other size and time

	lazycache::add(c, src);
	CHECK(access(c.c_str(), F_OK) != 0); // removed at once, so no reader gets the old data
	CHECK(lazycache::resolve(c) == src);
	lazycache::start(1, 0);
	REQUIRE(waitUntil([&]() { return access(c.c_str(), F_OK) == 0; }));
	CHECK(readAll(c) == "new content");
}

TEST_CASE("lazy cache: an up-to-date cache file is left alone", "[lazy_cache]")
{
	LazyOn on;
	TempDir d;
	std::string cache = d.path + "/cache/relion_cache/";
	REQUIRE(system(("mkdir -p '" + d.path + "/src' '" + cache + "k'").c_str()) == 0);
	std::string src = d.path + "/src/a.mrcs", c = cache + "k/a.mrcs";
	writeFile(src, "same", 1700000000);
	writeFile(c, "same", 1700000000);
	struct stat before;
	REQUIRE(stat(c.c_str(), &before) == 0);

	lazycache::add(c, src);
	CHECK(lazycache::resolve(c) == ""); // exists, nothing queued
	struct stat after;
	REQUIRE(stat(c.c_str(), &after) == 0);
	CHECK(after.st_ino == before.st_ino);
}

TEST_CASE("lazy cache: a stack locked by another process is not copied here", "[lazy_cache]")
{
	LazyOn on;
	TempDir d;
	std::string cache = d.path + "/cache/relion_cache/";
	REQUIRE(system(("mkdir -p '" + d.path + "/src' '" + cache + "k'").c_str()) == 0);
	std::string src = d.path + "/src/a.mrcs", c = cache + "k/a.mrcs";
	writeFile(src, "content", 1700000000);

	int lfd = open((c + ".lazylock").c_str(), O_CREAT | O_RDWR, 0666); // the other process
	REQUIRE(lfd >= 0);
	REQUIRE(flock(lfd, LOCK_EX | LOCK_NB) == 0);

	lazycache::add(c, src);
	lazycache::start(1, 0);
	std::this_thread::sleep_for(std::chrono::milliseconds(500));
	CHECK(access(c.c_str(), F_OK) != 0);
	CHECK(access((c + ".lazytmp").c_str(), F_OK) != 0);
	lazycache::stop();
	close(lfd);
}
