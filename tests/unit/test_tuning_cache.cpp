/*
 * tests/unit/test_tuning_cache.cpp
 *
 * The per-user tuning cache of src/tuning_cache.h: round trip, newest row wins, rows of other devices or
 * problems are not used, silent fallback, the environment switches, and concurrent appends from threads
 * and from processes do not corrupt the file.
 */

#include <catch2/catch.hpp>

#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "src/tuning_cache.h"

namespace
{
// a scratch directory under $HOME/temp (not /tmp), removed again
struct TempDir
{
	std::string dir;
	TempDir()
	{
		const char *home = getenv("HOME"), *tmp = getenv("TMPDIR");
		std::string base = home ? std::string(home) + "/temp" : (tmp ? tmp : ".");
		struct stat st;
		if (stat(base.c_str(), &st) != 0) base = tmp ? tmp : ".";
		std::string t = base + "/tuning_cache_test_XXXXXX";
		std::vector<char> b(t.begin(), t.end());
		b.push_back(0);
		if (mkdtemp(b.data())) dir = b.data();
	}
	~TempDir() { if (!dir.empty()) { std::string c = "rm -rf '" + dir + "'"; if (system(c.c_str())) {} } }
	std::string file(const std::string &n = "tuning.csv") const { return dir + "/" + n; }
};

tuning::Context ctxA() { tuning::Context c; c.device = "Test GPU A"; c.sm_count = 48; c.driver = "12000"; c.runtime = "12040"; c.build = "5.1.0"; return c; }

std::vector<std::string> readLines(const std::string &path)
{
	std::ifstream f(path);
	std::vector<std::string> v;
	std::string l;
	while (std::getline(f, l)) v.push_back(l);
	return v;
}

size_t columns(const std::string &l) { size_t n = 1; for (char c : l) if (c == ',') n++; return n; }
}

TEST_CASE("tuning_cache stores and returns a choice", "[tuning_cache]")
{
	TempDir td;
	REQUIRE(!td.dir.empty());
	tuning::Cache cache(ctxA(), td.file());
	REQUIRE(cache.enabled());
	tuning::Entry e;
	CHECK_FALSE(cache.get("bp_tile", "x=65;y=128", e));   // no file yet
	REQUIRE(cache.put("bp_tile", "x=65;y=128", "16x2", 0.91));
	REQUIRE(cache.get("bp_tile", "x=65;y=128", e));
	CHECK(e.value == "16x2");
	CHECK(e.rel_time == Approx(0.91));
	CHECK(e.date.size() == 20);   // 2026-10-13T22:00:00Z

	// the file is a CSV with a header and the documented columns
	const auto lines = readLines(td.file());
	REQUIRE(lines.size() == 2);
	CHECK(lines[0] == "tunable,device,sm_count,driver,runtime,build,key,value,rel_time,date");
	CHECK(columns(lines[1]) == 10);

	// the newest row wins
	REQUIRE(cache.put("bp_tile", "x=65;y=128", "8x4", 0.95));
	REQUIRE(cache.get("bp_tile", "x=65;y=128", e));
	CHECK(e.value == "8x4");
	CHECK(readLines(td.file()).size() == 3);

	// another problem, tunable, device or SM count does not match
	CHECK_FALSE(cache.get("bp_tile", "x=33;y=64", e));
	CHECK_FALSE(cache.get("other", "x=65;y=128", e));
	tuning::Context b = ctxA(); b.device = "Test GPU B";
	CHECK_FALSE(tuning::Cache(b, td.file()).get("bp_tile", "x=65;y=128", e));
	b = ctxA(); b.sm_count = 80;
	CHECK_FALSE(tuning::Cache(b, td.file()).get("bp_tile", "x=65;y=128", e));
	// a new driver or build still finds it (they are recorded, the cache is only a starting point)
	b = ctxA(); b.driver = "12100"; b.build = "5.2.0";
	CHECK(tuning::Cache(b, td.file()).get("bp_tile", "x=65;y=128", e));
}

TEST_CASE("tuning_cache keeps the CSV well formed and survives bad files", "[tuning_cache]")
{
	TempDir td;
	REQUIRE(!td.dir.empty());
	tuning::Context c = ctxA();
	c.device = "GPU, with comma\nand newline";
	tuning::Cache cache(c, td.file());
	REQUIRE(cache.put("t", "k,1", "v,1", 1.0));
	for (const auto &l : readLines(td.file())) CHECK(columns(l) == 10);
	tuning::Entry e;
	REQUIRE(cache.get("t", "k,1", e));
	CHECK(e.value == "v;1");

	// garbage lines, a short line and a half written last line are skipped
	{
		std::ofstream f(td.file(), std::ios::app);
		f << "garbage\n1,2,3\n\n";
		f << "t,half,written";   // no newline
	}
	REQUIRE(cache.get("t", "k,1", e));
	CHECK(e.value == "v;1");
	// and an empty file is fine
	{ std::ofstream f(td.file(), std::ios::trunc); }
	CHECK_FALSE(cache.get("t", "k,1", e));
	REQUIRE(cache.put("t", "k,1", "x", 0.));
	CHECK(readLines(td.file())[0].rfind("tunable,", 0) == 0);   // the header is written again
	REQUIRE(cache.get("t", "k,1", e));
	CHECK(e.value == "x");
}

TEST_CASE("tuning_cache falls back silently", "[tuning_cache]")
{
	tuning::Entry e;
	tuning::Cache bad(ctxA(), "/proc/nonexistent_dir/tuning.csv");
	CHECK(bad.enabled());
	CHECK_FALSE(bad.put("t", "k", "v"));
	CHECK_FALSE(bad.get("t", "k", e));
	TempDir td;
	REQUIRE(!td.dir.empty());
	// a directory where the file should be
	mkdir(td.file().c_str(), 0700);
	tuning::Cache dir(ctxA(), td.file());
	CHECK_FALSE(dir.put("t", "k", "v"));
	CHECK_FALSE(dir.get("t", "k", e));
}

TEST_CASE("tuning_cache environment switches", "[tuning_cache]")
{
	TempDir td;
	REQUIRE(!td.dir.empty());
	const char *old_home = getenv("HOME"), *old_file = getenv("RELION_TUNING_FILE"), *old_off = getenv("RELION_TUNING_CACHE");
	const std::string sh = old_home ? old_home : "", sf = old_file ? old_file : "", so = old_off ? old_off : "";
	unsetenv("RELION_TUNING_FILE"); unsetenv("RELION_TUNING_CACHE");

	setenv("HOME", td.dir.c_str(), 1);
	CHECK(tuning::defaultPath() == td.dir + "/.relion/tuning.csv");
	{
		tuning::Cache cache(ctxA());   // the default path; creates the directory
		REQUIRE(cache.put("t", "k", "v"));
		tuning::Entry e;
		REQUIRE(cache.get("t", "k", e));
		struct stat st;
		CHECK(stat((td.dir + "/.relion/tuning.csv").c_str(), &st) == 0);
	}
	setenv("RELION_TUNING_FILE", td.file("other.csv").c_str(), 1);
	CHECK(tuning::defaultPath() == td.file("other.csv"));
	setenv("RELION_TUNING_CACHE", "off", 1);
	CHECK(tuning::defaultPath() == "");
	{
		tuning::Cache cache(ctxA());
		tuning::Entry e;
		CHECK_FALSE(cache.enabled());
		CHECK_FALSE(cache.put("t", "k", "v"));
		CHECK_FALSE(cache.get("t", "k", e));
	}
	unsetenv("RELION_TUNING_CACHE"); unsetenv("RELION_TUNING_FILE");
	unsetenv("HOME");
	CHECK(tuning::defaultPath() == "");   // no home directory: no cache

	if (old_home) setenv("HOME", sh.c_str(), 1);
	if (old_file) setenv("RELION_TUNING_FILE", sf.c_str(), 1);
	if (old_off) setenv("RELION_TUNING_CACHE", so.c_str(), 1);
}

namespace
{
void checkAllRows(const std::string &path, int writers, int per_writer)
{
	const auto lines = readLines(path);
	REQUIRE(lines.size() == (size_t) writers * per_writer + 1);
	CHECK(lines[0].rfind("tunable,", 0) == 0);
	std::set<std::string> keys;
	for (size_t i = 1; i < lines.size(); i++)
	{
		CHECK(columns(lines[i]) == 10);
		std::stringstream ss(lines[i]);
		std::string f, key;
		for (int k = 0; k < 7 && std::getline(ss, f, ','); k++) key = f;
		keys.insert(key);
	}
	CHECK(keys.size() == (size_t) writers * per_writer);
	tuning::Cache cache(ctxA(), path);
	for (int w = 0; w < writers; w++)
		for (int i = 0; i < per_writer; i += 17)
		{
			tuning::Entry e;
			REQUIRE(cache.get("t", "w" + std::to_string(w) + "_" + std::to_string(i), e));
			CHECK(e.value == "v" + std::to_string(w * 1000 + i));
		}
}
}

TEST_CASE("tuning_cache concurrent appends from threads", "[tuning_cache]")
{
	TempDir td;
	REQUIRE(!td.dir.empty());
	const int writers = 8, per = 100;
	std::vector<std::thread> th;
	for (int w = 0; w < writers; w++)
		th.emplace_back([&, w] {
			tuning::Cache cache(ctxA(), td.file());   // each thread its own object, as each job would
			for (int i = 0; i < per; i++)
				cache.put("t", "w" + std::to_string(w) + "_" + std::to_string(i), "v" + std::to_string(w * 1000 + i), 0.5);
		});
	for (auto &t : th) t.join();
	checkAllRows(td.file(), writers, per);
}

TEST_CASE("tuning_cache concurrent appends from processes", "[tuning_cache]")
{
	TempDir td;
	REQUIRE(!td.dir.empty());
	const int writers = 6, per = 100;
	const std::string path = td.file();
	std::vector<pid_t> pids;
	for (int w = 0; w < writers; w++)
	{
		const pid_t pid = fork();
		REQUIRE(pid >= 0);
		if (pid == 0)
		{
			tuning::Cache cache(ctxA(), path);
			bool ok = true;
			for (int i = 0; i < per; i++)
				ok = cache.put("t", "w" + std::to_string(w) + "_" + std::to_string(i), "v" + std::to_string(w * 1000 + i), 0.5) && ok;
			_exit(ok ? 0 : 1);
		}
		pids.push_back(pid);
	}
	for (pid_t pid : pids)
	{
		int status = 0;
		REQUIRE(waitpid(pid, &status, 0) == pid);
		CHECK(WIFEXITED(status));
		CHECK(WEXITSTATUS(status) == 0);
	}
	checkAllRows(path, writers, per);
}

TEST_CASE("tuning_cache compacts a long file", "[tuning_cache]")
{
	TempDir td;
	REQUIRE(!td.dir.empty());
	tuning::Context c = ctxA();
	c.build = std::string(80, 'b');   // long rows, so that the file passes the size limit soon
	tuning::Cache cache(c, td.file());
	const int total = (int) tuning::Cache::kCompactRows + 600;
	for (int i = 0; i < total; i++)
		REQUIRE(cache.put("t", "k" + std::to_string(i % 5), "v" + std::to_string(i), 0.5));
	CHECK(readLines(td.file()).size() < (size_t) total / 2);
	for (int k = 0; k < 5; k++)
	{
		tuning::Entry e;
		REQUIRE(cache.get("t", "k" + std::to_string(k), e));
		int last = k;
		while (last + 5 < total) last += 5;
		CHECK(e.value == "v" + std::to_string(last));
	}
	// no stray temporary files
	CHECK(system(("ls '" + td.dir + "' | grep -q '\\.tmp\\.'").c_str()) != 0);
}
