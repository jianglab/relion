/*
 * tests/unit/test_remove_intermediates.cpp
 *
 * Unit tests for the scan that decides which per-iteration refinement files
 * can be deleted. The scan is the dangerous half of the feature - once the
 * user agrees, the files are gone - so these pin down exactly which files it
 * proposes to delete and, more importantly, which ones it must never touch.
 */

#include <catch2/catch.hpp>

#include "src/remove_intermediates.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {

/// A temporary directory that removes itself, so a failing test leaves nothing.
class TempDir {
public:
	TempDir()
	{
		char tmpl[] = "/tmp/relion_cleanup_XXXXXX";
		const char* p = mkdtemp(tmpl);
		REQUIRE(p != NULL);
		path_ = p;
	}
	~TempDir()
	{
		const std::string cmd = "rm -rf '" + path_ + "'";
		if (system(cmd.c_str()) != 0) { /* best effort */ }
	}
	const std::string& path() const { return path_; }

private:
	std::string path_;
};

void makeDirs(const std::string& path)
{
	std::string cur;
	for (size_t i = 0; i < path.size(); i++)
	{
		cur += path[i];
		if (path[i] == '/' || i + 1 == path.size()) mkdir(cur.c_str(), 0755);
	}
}

void writeFile(const std::string& path, size_t bytes = 8)
{
	std::ofstream f(path.c_str(), std::ios::binary);
	REQUIRE(f.good());
	f << std::string(bytes, 'x');
}

std::set<std::string> removedNames(const relion_cleanup::Plan& plan)
{
	std::set<std::string> names;
	for (size_t i = 0; i < plan.remove.size(); i++)
	{
		const std::string& p = plan.remove[i].path;
		names.insert(p.substr(p.find_last_of('/') + 1));
	}
	return names;
}

} // namespace

TEST_CASE("intermediate scan keeps the first and last iteration", "[cleanup]")
{
	TempDir tmp;
	const std::string job = tmp.path() + "/Class3D/job005";
	makeDirs(job);

	for (int it = 0; it <= 3; it++)
	{
		char name[64];
		snprintf(name, sizeof(name), "/run_it%03d_data.star", it);
		writeFile(job + name);
		snprintf(name, sizeof(name), "/run_it%03d_optimiser.star", it);
		writeFile(job + name);
	}

	const relion_cleanup::Plan plan = relion_cleanup::planIntermediateRemoval(tmp.path());
	const std::set<std::string> gone = removedNames(plan);

	CHECK(gone.size() == 4);
	CHECK(gone.count("run_it001_data.star") == 1);
	CHECK(gone.count("run_it001_optimiser.star") == 1);
	CHECK(gone.count("run_it002_data.star") == 1);
	CHECK(gone.count("run_it002_optimiser.star") == 1);

	// The two rounds worth keeping
	CHECK(gone.count("run_it000_data.star") == 0);
	CHECK(gone.count("run_it003_data.star") == 0);
}

TEST_CASE("files that are not iteration output are never touched", "[cleanup]")
{
	TempDir tmp;
	const std::string job = tmp.path() + "/Refine3D/job012";
	makeDirs(job);

	writeFile(job + "/run_it000_data.star");
	writeFile(job + "/run_it001_data.star");
	writeFile(job + "/run_it002_data.star");

	// Final results and anything else a job directory holds
	writeFile(job + "/run_data.star");
	writeFile(job + "/run_class001.mrc");
	writeFile(job + "/run_half1_class001_unfil.mrc");
	writeFile(job + "/note.txt");
	writeFile(job + "/job.star");
	writeFile(job + "/RELION_JOB_EXIT_SUCCESS");
	// A name that ends at the iteration number is not an iteration file
	writeFile(job + "/run_it007");

	const relion_cleanup::Plan plan = relion_cleanup::planIntermediateRemoval(tmp.path());

	REQUIRE(plan.remove.size() == 1);
	CHECK(plan.remove[0].path == job + "/run_it001_data.star");
}

TEST_CASE("a continuation is its own group of iterations", "[cleanup]")
{
	TempDir tmp;
	const std::string job = tmp.path() + "/Class2D/job003";
	makeDirs(job);

	for (int it = 0; it <= 2; it++)
	{
		char name[64];
		snprintf(name, sizeof(name), "/run_it%03d_data.star", it);
		writeFile(job + name);
	}
	// Continued from iteration 2, so run_ct2_it003..005
	for (int it = 3; it <= 5; it++)
	{
		char name[64];
		snprintf(name, sizeof(name), "/run_ct2_it%03d_data.star", it);
		writeFile(job + name);
	}

	const relion_cleanup::Plan plan = relion_cleanup::planIntermediateRemoval(tmp.path());
	const std::set<std::string> gone = removedNames(plan);

	// One middle round from each group, not one across the two of them
	CHECK(gone.size() == 2);
	CHECK(gone.count("run_it001_data.star") == 1);
	CHECK(gone.count("run_ct2_it004_data.star") == 1);
	CHECK(gone.count("run_it002_data.star") == 0);
	CHECK(gone.count("run_ct2_it003_data.star") == 0);
}

TEST_CASE("a single iteration is kept", "[cleanup]")
{
	TempDir tmp;
	const std::string job = tmp.path() + "/Class3D/job001";
	makeDirs(job);
	writeFile(job + "/run_it000_data.star");

	const relion_cleanup::Plan plan = relion_cleanup::planIntermediateRemoval(tmp.path());
	CHECK(plan.empty());
	CHECK(plan.total_bytes == 0);
}

TEST_CASE("the reported size is the size of the files to be removed", "[cleanup]")
{
	TempDir tmp;
	const std::string job = tmp.path() + "/Class3D/job002";
	makeDirs(job);

	writeFile(job + "/run_it000_data.star", 10);
	writeFile(job + "/run_it001_data.star", 100);
	writeFile(job + "/run_it002_data.star", 1000);
	writeFile(job + "/run_it003_data.star", 20);

	const relion_cleanup::Plan plan = relion_cleanup::planIntermediateRemoval(tmp.path());
	CHECK(plan.remove.size() == 2);
	CHECK(plan.total_bytes == 1100);
}

TEST_CASE("applying a plan deletes exactly the planned files", "[cleanup]")
{
	TempDir tmp;
	const std::string job = tmp.path() + "/Class3D/job007";
	makeDirs(job);

	for (int it = 0; it <= 4; it++)
	{
		char name[64];
		snprintf(name, sizeof(name), "/run_it%03d_data.star", it);
		writeFile(job + name, 16);
	}

	relion_cleanup::Plan plan = relion_cleanup::planIntermediateRemoval(tmp.path());
	REQUIRE(plan.remove.size() == 3);

	long long freed = 0;
	std::vector<std::string> errors;
	const long n = relion_cleanup::applyRemoval(plan, freed, errors);

	CHECK(n == 3);
	CHECK(freed == 48);
	CHECK(errors.empty());

	CHECK(access((job + "/run_it000_data.star").c_str(), F_OK) == 0);
	CHECK(access((job + "/run_it004_data.star").c_str(), F_OK) == 0);
	CHECK(access((job + "/run_it002_data.star").c_str(), F_OK) != 0);

	// Running the scan again now has nothing left to do
	CHECK(relion_cleanup::planIntermediateRemoval(tmp.path()).empty());
}

TEST_CASE("only jobNNN directories and the project root are scanned", "[cleanup]")
{
	TempDir tmp;

	// A job directory: scanned
	const std::string job = tmp.path() + "/Class3D/job004";
	makeDirs(job);
	writeFile(job + "/run_it000_data.star");
	writeFile(job + "/run_it001_data.star");
	writeFile(job + "/run_it002_data.star");

	// Something that merely looks like one: left alone
	const std::string other = tmp.path() + "/Class3D/job4";
	makeDirs(other);
	writeFile(other + "/run_it000_data.star");
	writeFile(other + "/run_it001_data.star");
	writeFile(other + "/run_it002_data.star");

	const relion_cleanup::Plan plan = relion_cleanup::planIntermediateRemoval(tmp.path());
	REQUIRE(plan.remove.size() == 1);
	CHECK(plan.remove[0].path == job + "/run_it001_data.star");
}

TEST_CASE("removal reports progress and always ends at the total", "[cleanup]")
{
	TempDir tmp;
	const std::string job = tmp.path() + "/Class3D/job009";
	makeDirs(job);

	// More files than the reporting interval, and not a multiple of it, so that
	// the final call is the one that takes the count to the end
	for (int it = 0; it <= 50; it++)
	{
		char name[64];
		snprintf(name, sizeof(name), "/run_it%03d_data.star", it);
		writeFile(job + name);
	}

	relion_cleanup::Plan plan = relion_cleanup::planIntermediateRemoval(tmp.path());
	REQUIRE(plan.remove.size() == 49);

	struct Seen {
		std::vector<size_t> done;
		size_t total;
		Seen() : total(0) {}
		static void record(size_t done, size_t total, void* user)
		{
			Seen* s = (Seen*)user;
			s->done.push_back(done);
			s->total = total;
		}
	} seen;

	long long freed = 0;
	std::vector<std::string> errors;
	relion_cleanup::applyRemoval(plan, freed, errors, Seen::record, &seen);

	REQUIRE(!seen.done.empty());
	CHECK(seen.total == 49);
	// A dialog driven by this must not be left showing a partial count
	CHECK(seen.done.back() == 49);
	// and must never be told about more files than there are
	for (size_t i = 0; i < seen.done.size(); i++)
		CHECK(seen.done[i] <= 49);
	// Progress only moves forwards
	for (size_t i = 1; i < seen.done.size(); i++)
		CHECK(seen.done[i] >= seen.done[i - 1]);
}

TEST_CASE("human-readable sizes", "[cleanup]")
{
	CHECK(relion_cleanup::humanSize(0) == "0 B");
	CHECK(relion_cleanup::humanSize(512) == "512 B");
	CHECK(relion_cleanup::humanSize(1024) == "1.0 KB");
	CHECK(relion_cleanup::humanSize(1536) == "1.5 KB");
	CHECK(relion_cleanup::humanSize(3LL * 1024 * 1024 * 1024) == "3.0 GB");
}
