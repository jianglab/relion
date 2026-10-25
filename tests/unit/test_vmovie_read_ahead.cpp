/*
 * tests/unit/test_vmovie_read_ahead.cpp
 *
 * vmovies::ReadAhead (building the cache entries of the next virtual movie averages in the
 * background): the depth comes from RELION_VMOVIE_READ_AHEAD, it asks for nothing when it is off,
 * when the cache is off or for files that are not virtual movie averages, and a descriptor that
 * cannot be built is left to the foreground read without a crash or a hang. That the entries made
 * are the pixels of the real micrograph is checked with synthetic movies in
 * tests/integration/test_virtual_movie_averages.py.
 */

#include <catch2/catch.hpp>

#include <algorithm>
#include <cstdlib>
#include <omp.h>
#include <fstream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "src/virtual_movie_averages.h"

namespace
{

struct Env
{
	std::string name;
	Env(const char* n, const char* v) : name(n)
	{
		if (v != NULL) setenv(n, v, 1); else unsetenv(n);
	}
	~Env() { unsetenv(name.c_str()); }
};

std::string makeTempDir()
{
	char tmpl[] = "/tmp/relion_vmra_XXXXXX";
	const char* d = mkdtemp(tmpl);
	REQUIRE(d != NULL);
	return d;
}

void writeFile(const std::string& path, const std::string& text)
{
	std::ofstream(path.c_str(), std::ios::binary) << text;
}

const char* MAGIC = "# RELION virtual movie average\n";

} // namespace

TEST_CASE("vmovies::ReadAhead depth", "[unit][vmovies]")
{
	{ Env e("RELION_VMOVIE_READ_AHEAD", NULL); vmovies::ReadAhead ra; CHECK(ra.enabled()); CHECK(ra.depth() == 2); }   // automatic: probe + one queued
	{ Env e("RELION_VMOVIE_READ_AHEAD", "3"); vmovies::ReadAhead ra; CHECK(ra.enabled()); CHECK(ra.depth() == 3); }
	{ Env e("RELION_VMOVIE_READ_AHEAD", "0"); vmovies::ReadAhead ra; CHECK(!ra.enabled()); CHECK(ra.depth() == 0); }
	{ Env e("RELION_VMOVIE_READ_AHEAD", "-2"); vmovies::ReadAhead ra; CHECK(ra.enabled()); CHECK(ra.depth() == 2); }
	{ Env e("RELION_VMOVIE_READ_AHEAD", "3"); CHECK(!vmovies::ReadAhead(0).enabled()); }   // an explicit depth wins
}

TEST_CASE("vmovies::ReadAhead builders and threads: automatic, overridden and limited", "[unit][vmovies]")
{
	const std::string dir = makeTempDir();
	writeFile(dir + "/desc.mrc", std::string(MAGIC) + "garbage\n");
	Env c("RELION_VMOVIE_AVERAGE_CACHE", (dir + "/cache").c_str());
	const std::vector<std::string> one(1, dir + "/desc.mrc");
	const int cores = std::max(1, omp_get_max_threads());

	SECTION("automatic: one builder (4 threads at most) until the first build has been measured")
	{
		Env b("RELION_VMOVIE_BUILDERS", NULL);
		vmovies::ReadAhead ra;
		ra.upcoming(one);
		ra.waitIdle();
		CHECK(ra.builders() == 1);
		CHECK(ra.buildThreads() == std::min(4, cores));
	}
	SECTION("RELION_VMOVIE_BUILDERS and RELION_VMOVIE_BUILD_THREADS")
	{
		Env b("RELION_VMOVIE_BUILDERS", "3");
		vmovies::ReadAhead ra;
		ra.upcoming(one);
		ra.waitIdle();
		CHECK(ra.builders() == 3);
		CHECK(ra.buildThreads() == std::max(1, cores / 3));
		CHECK(ra.depth() == 6);
		Env t("RELION_VMOVIE_BUILD_THREADS", "2");
		vmovies::ReadAhead rb;
		rb.upcoming(one);
		rb.waitIdle();
		CHECK(rb.builders() == 3);
		CHECK(rb.buildThreads() == 2);
	}
	SECTION("limitBuilders")
	{
		Env b("RELION_VMOVIE_BUILDERS", "8");
		vmovies::ReadAhead ra;
		ra.limitBuilders(2);
		ra.upcoming(one);
		ra.waitIdle();
		CHECK(ra.builders() == 2);
	}
	system(("rm -rf " + dir).c_str());
}

TEST_CASE("vmovies::ReadAhead asks for nothing it cannot use", "[unit][vmovies]")
{
	const std::string dir = makeTempDir();
	const std::string cache = dir + "/cache";
	writeFile(dir + "/plain.mrc", "not a descriptor, whatever its name\n");
	writeFile(dir + "/desc.mrc", std::string(MAGIC) + "garbage that is no motion record\n");
	const std::vector<std::string> next = {dir + "/plain.mrc", dir + "/missing.mrc"};

	SECTION("off")
	{
		Env c("RELION_VMOVIE_AVERAGE_CACHE", cache.c_str());
		vmovies::ReadAhead ra(0);
		ra.upcoming(std::vector<std::string>(1, dir + "/desc.mrc"));
		ra.waitIdle();
		CHECK(ra.requested() == 0);
	}
	SECTION("cache off")
	{
		Env c("RELION_VMOVIE_AVERAGE_CACHE", "off");
		vmovies::ReadAhead ra(1);
		ra.upcoming(std::vector<std::string>(1, dir + "/desc.mrc"));
		ra.waitIdle();
		CHECK(ra.requested() == 0);
		struct stat st;
		CHECK(stat((dir + "/Cache").c_str(), &st) != 0);
	}
	SECTION("files that are not virtual movie averages")
	{
		Env c("RELION_VMOVIE_AVERAGE_CACHE", cache.c_str());
		vmovies::ReadAhead ra(2);
		ra.upcoming(next);
		ra.waitIdle();
		CHECK(ra.requested() == 0);
		CHECK(ra.built() == 0);
	}
	SECTION("a descriptor that cannot be built is left to the foreground read")
	{
		Env c("RELION_VMOVIE_AVERAGE_CACHE", cache.c_str());
		{
			vmovies::ReadAhead ra(1);
			ra.upcoming(std::vector<std::string>(1, dir + "/desc.mrc"));
			ra.waitIdle();
			CHECK(ra.requested() == 1);
			CHECK(ra.built() == 0);
			ra.upcoming(std::vector<std::string>(1, dir + "/desc.mrc"));   // asked again: still no crash
			ra.waitIdle();
		}   // the destructor returns
	}
	SECTION("only depth micrographs of those offered are queued")
	{
		Env c("RELION_VMOVIE_AVERAGE_CACHE", cache.c_str());
		writeFile(dir + "/desc2.mrc", std::string(MAGIC) + "other garbage\n");
		writeFile(dir + "/desc3.mrc", std::string(MAGIC) + "more garbage\n");
		vmovies::ReadAhead ra(2);
		ra.upcoming({dir + "/plain.mrc", dir + "/desc.mrc", dir + "/desc2.mrc", dir + "/desc3.mrc"});
		ra.waitIdle();
		CHECK(ra.requested() == 2);
	}

	unlink((dir + "/plain.mrc").c_str());
	unlink((dir + "/desc.mrc").c_str());
	unlink((dir + "/desc2.mrc").c_str());
	unlink((dir + "/desc3.mrc").c_str());
	system(("rm -rf " + dir).c_str());
}

TEST_CASE("vmovies::memoryAvailable honours the Slurm job's memory", "[unit][vmovies]")
{
	Env none_node("SLURM_MEM_PER_NODE", NULL), none_cpu("SLURM_MEM_PER_CPU", NULL), none_cpus("SLURM_CPUS_ON_NODE", NULL);
	const long long free_mem = vmovies::memoryAvailable();
	CHECK(free_mem != 0);

	SECTION("per node, in MB, less what this process holds")
	{
		Env node("SLURM_MEM_PER_NODE", "1");
		CHECK(vmovies::memoryAvailable() <= 1048576LL);
	}
	SECTION("per CPU times the CPUs of the node")
	{
		Env cpu("SLURM_MEM_PER_CPU", "1"), cpus("SLURM_CPUS_ON_NODE", "2");
		CHECK(vmovies::memoryAvailable() <= 2 * 1048576LL);
	}
	SECTION("a limit far above the machine changes nothing")
	{
		Env node("SLURM_MEM_PER_NODE", "100000000");
		const long long a = vmovies::memoryAvailable();
		CHECK(a > 0);
		if (free_mem > 0) CHECK(a <= free_mem + (64LL << 20));
	}
}
