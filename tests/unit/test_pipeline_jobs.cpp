/*
 * tests/unit/test_pipeline_jobs.cpp
 *
 * Unit tests for RelionJob option registration and command generation.
 * These guard against regressions in joboption definitions and the
 * command-line flags that getCommands*() emits.
 */

#include <catch2/catch.hpp>

#include "src/pipeline_jobs.h"

#include <string>
#include <vector>

static bool generateCommand(RelionJob &job, std::string &command)
{
	std::string outputname, final_command, error_message;
	std::vector<std::string> commands;
	bool ok = job.getCommands(outputname, commands, final_command,
	                           false, 1, error_message);
	if (ok && !commands.empty())
		command = commands[0];
	return ok;
}

// ---------------------------------------------------------------------------
// Reconstruct3D
// ---------------------------------------------------------------------------

TEST_CASE("Reconstruct3D: cache_dir and cache_copy_threads options exist",
          "[pipeline][reconstruct3d]")
{
	RelionJob job;
	job.clear();
	job.initialise(PROC_RECONSTRUCT3D);
	REQUIRE(job.joboptions.find("cache_dir") != job.joboptions.end());
	REQUIRE(job.joboptions.find("cache_copy_threads") != job.joboptions.end());
}

// ---------------------------------------------------------------------------
// MultiBody
// ---------------------------------------------------------------------------

TEST_CASE("MultiBody: cache_dir and cache_copy_threads options exist",
          "[pipeline][multibody]")
{
	RelionJob job;
	job.clear();
	job.initialise(PROC_MULTIBODY);
	REQUIRE(job.joboptions.find("cache_dir") != job.joboptions.end());
	REQUIRE(job.joboptions.find("cache_copy_threads") != job.joboptions.end());
}

// ---------------------------------------------------------------------------
// Class3D
// ---------------------------------------------------------------------------

TEST_CASE("Class3D: cache_dir and cache_copy_threads options exist",
          "[pipeline][class3d]")
{
	RelionJob job;
	job.clear();
	job.initialise(PROC_3DCLASS);
	REQUIRE(job.joboptions.find("cache_dir") != job.joboptions.end());
	REQUIRE(job.joboptions.find("cache_copy_threads") != job.joboptions.end());
}
