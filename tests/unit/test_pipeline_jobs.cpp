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

// ---------------------------------------------------------------------------
// Select2D filament voting
// ---------------------------------------------------------------------------

TEST_CASE("Select2D: keeps runtime controls out of the pipeline window",
          "[pipeline][select2d]")
{
	RelionJob job;
	job.clear();
	job.initialise(PROC_SELECT2D);
	REQUIRE(job.joboptions.find("fn_optimiser") != job.joboptions.end());
	REQUIRE(job.joboptions.find("nr_types") == job.joboptions.end());
	REQUIRE(job.joboptions.find("do_similarity_sort") == job.joboptions.end());
}

TEST_CASE("Select2D: launches the dedicated interactive program",
          "[pipeline][select2d]")
{
	RelionJob job;
	job.clear();
	job.initialise(PROC_SELECT2D);
	job.joboptions["fn_optimiser"].setString("Class2D/job001/run_it025_optimiser.star");

	std::string command;
	REQUIRE(generateCommand(job, command));
	REQUIRE(command.find("relion_select_2d_classes") != std::string::npos);
	REQUIRE(command.find("relion_display") == std::string::npos);
	REQUIRE(command.find("--i Class2D/job001/run_it025_optimiser.star") != std::string::npos);
	REQUIRE(command.find("--o Select2D/job001/") != std::string::npos);
	REQUIRE(job.outputNodes.empty()); // The GUI registers its runtime-selected outputs dynamically.
}

TEST_CASE("Select2D: requires an optimiser input", "[pipeline][select2d]")
{
	RelionJob job;
	job.clear();
	job.initialise(PROC_SELECT2D);

	std::string command;
	REQUIRE_FALSE(generateCommand(job, command));
}
