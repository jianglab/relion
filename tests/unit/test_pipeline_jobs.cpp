/*
 * tests/unit/test_pipeline_jobs.cpp
 *
 * Unit tests for RelionJob option registration and command generation.
 * These guard against regressions in joboption definitions and the
 * command-line flags that getCommands*() emits.
 */

#include <catch2/catch.hpp>

#include "src/pipeline_jobs.h"
#include "src/pipeline_control.h"

#include <climits>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <unistd.h>

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

static bool generateCommands(RelionJob &job, std::vector<std::string> &commands,
		std::string &final_command, std::string &error_message,
		bool do_makedir = false, std::string outputname = "")
{
	return job.getCommands(outputname, commands, final_command,
	                       do_makedir, 1, error_message);
}

static RelionJob makeClass2DJob()
{
	RelionJob job;
	job.clear();
	job.initialise(PROC_2DCLASS);
	job.joboptions["fn_img"].setString("particles.star");
	job.joboptions["do_em"].setString("Yes");
	job.joboptions["do_grad"].setString("No");
	job.joboptions["nr_mpi"].setString("1");
	job.joboptions["nr_threads"].setString("1");
	job.joboptions["do_queue"].setString("No");
	job.joboptions["scratch_dir"].setString("");
	return job;
}

// ---------------------------------------------------------------------------
// Class2D parallel replicas
// ---------------------------------------------------------------------------

TEST_CASE("Class2D: parallel replica options have backward-compatible defaults",
          "[pipeline][class2d]")
{
	RelionJob job = makeClass2DJob();
	REQUIRE(job.joboptions.find("nr_parallel_runs") != job.joboptions.end());
	REQUIRE(job.joboptions.find("random_seed") != job.joboptions.end());
	REQUIRE(job.joboptions.find("do_fast_subsets") != job.joboptions.end());
	REQUIRE(job.joboptions["nr_parallel_runs"].getString() == "1");
	REQUIRE(job.joboptions["random_seed"].getString() == "-1");
	REQUIRE_FALSE(job.joboptions["do_fast_subsets"].getBoolean());

	std::vector<std::string> commands;
	std::string final_command, error_message;
	REQUIRE(generateCommands(job, commands, final_command, error_message));
	REQUIRE(commands.size() == 1);
	REQUIRE(commands[0].find(" --o Class2D/job001/run ") != std::string::npos);
	REQUIRE(commands[0].find("--random_seed") == std::string::npos);
	REQUIRE(commands[0].find("pipeline_control_task") == std::string::npos);
}

TEST_CASE("Class2D: replica options survive job STAR round trip",
          "[pipeline][class2d]")
{
	RelionJob job = makeClass2DJob();
	job.joboptions["nr_parallel_runs"].setString("3");
	job.joboptions["random_seed"].setString("12345");
	job.joboptions["do_fast_subsets"].setString("Yes");
	std::string fn = "class2d_replica_roundtrip_" + integerToString((int)std::time(NULL)) + ".star";
	job.write(fn);

	RelionJob loaded;
	loaded.clear();
	bool is_continue = false;
	REQUIRE(loaded.read(fn, is_continue, true));
	REQUIRE(loaded.joboptions["nr_parallel_runs"].getString() == "3");
	REQUIRE(loaded.joboptions["random_seed"].getString() == "12345");
	REQUIRE(loaded.joboptions["do_fast_subsets"].getBoolean());
	std::remove(fn.c_str());
}

TEST_CASE("Class2D: three replicas get independent roots seeds GPUs scratch and nodes",
          "[pipeline][class2d]")
{
	RelionJob job = makeClass2DJob();
	job.joboptions["nr_parallel_runs"].setString("3");
	job.joboptions["random_seed"].setString("4100");
	job.joboptions["do_fast_subsets"].setString("Yes");
	job.joboptions["scratch_dir"].setString("/scratch/class2d");
	job.joboptions["cache_dir"].setString("/cache/shared");
	job.joboptions["use_gpu"].setString("Yes");
	job.joboptions["gpu_ids"].setString("0;1;2:3");

	std::vector<std::string> commands;
	std::string final_command, error_message;
	REQUIRE(generateCommands(job, commands, final_command, error_message));
	REQUIRE(commands.size() == 3);
	for (int i = 0; i < 3; i++)
	{
		std::string suffix = integerToString(i + 1, 3);
		REQUIRE(commands[i].find(" --o Class2D/job001/run" + suffix) != std::string::npos);
		REQUIRE(commands[i].find("--random_seed " + integerToString(4100 + i)) != std::string::npos);
		REQUIRE(commands[i].find("--scratch_dir /scratch/class2d/run" + suffix) != std::string::npos);
		REQUIRE(commands[i].find("--cache_dir /cache/shared") != std::string::npos);
		REQUIRE(commands[i].find("--fast_subsets") != std::string::npos);
		REQUIRE(commands[i].find("--pipeline_control_task_id " + integerToString(i + 1)) != std::string::npos);
	}
	REQUIRE(commands[0].find("--gpu \"0\"") != std::string::npos);
	REQUIRE(commands[1].find("--gpu \"1\"") != std::string::npos);
	REQUIRE(commands[2].find("--gpu \"2:3\"") != std::string::npos);
	REQUIRE(job.outputNodes.size() == 6);
	REQUIRE(job.outputNodes[0].name.find("run001_it025_data.star") != std::string::npos);
	REQUIRE(job.outputNodes[5].name.find("run003_it025_optimiser.star") != std::string::npos);
	REQUIRE(final_command.find("run001.out") != std::string::npos);
	REQUIRE(final_command.find("run002.err") != std::string::npos);
	REQUIRE(final_command.find("&&") == std::string::npos);
	REQUIRE(final_command.find(") & (") != std::string::npos);
}

TEST_CASE("Class2D: automatic replica seed is frozen and consecutive",
          "[pipeline][class2d]")
{
	RelionJob job = makeClass2DJob();
	job.joboptions["nr_parallel_runs"].setString("2");
	std::vector<std::string> commands;
	std::string final_command, error_message;
	REQUIRE(generateCommands(job, commands, final_command, error_message));
	REQUIRE(job.joboptions["random_seed"].getString() != "-1");
	int seed = textToInteger(job.joboptions["random_seed"].getString());
	REQUIRE(commands[0].find("--random_seed " + integerToString(seed)) != std::string::npos);
	REQUIRE(commands[1].find("--random_seed " + integerToString(seed + 1)) != std::string::npos);
}

TEST_CASE("Class2D: invalid replica configurations are rejected",
          "[pipeline][class2d]")
{
	std::vector<std::string> commands;
	std::string final_command, error_message;

	SECTION("non-integral run count")
	{
		RelionJob job = makeClass2DJob();
		job.joboptions["nr_parallel_runs"].setString("2.5");
		REQUIRE_FALSE(generateCommands(job, commands, final_command, error_message));
	}
	SECTION("seed overflow")
	{
		RelionJob job = makeClass2DJob();
		job.joboptions["nr_parallel_runs"].setString("2");
		job.joboptions["random_seed"].setString(integerToString(INT_MAX));
		REQUIRE_FALSE(generateCommands(job, commands, final_command, error_message));
	}
	SECTION("mismatched GPU groups")
	{
		RelionJob job = makeClass2DJob();
		job.joboptions["nr_parallel_runs"].setString("3");
		job.joboptions["use_gpu"].setString("Yes");
		job.joboptions["gpu_ids"].setString("0;1");
		REQUIRE_FALSE(generateCommands(job, commands, final_command, error_message));
	}
	SECTION("non-Slurm queue")
	{
		RelionJob job = makeClass2DJob();
		job.joboptions["nr_parallel_runs"].setString("2");
		job.joboptions["do_queue"].setString("Yes");
		job.joboptions["qsub"].setString("qsub");
		REQUIRE_FALSE(generateCommands(job, commands, final_command, error_message));
		REQUIRE(error_message.find("Slurm sbatch") != std::string::npos);
	}
	SECTION("fast subsets with VDAM")
	{
		RelionJob job = makeClass2DJob();
		job.joboptions["do_em"].setString("No");
		job.joboptions["do_grad"].setString("Yes");
		job.joboptions["do_fast_subsets"].setString("Yes");
		REQUIRE_FALSE(generateCommands(job, commands, final_command, error_message));
	}
	SECTION("fast subsets with too few EM iterations")
	{
		RelionJob job = makeClass2DJob();
		job.joboptions["do_fast_subsets"].setString("Yes");
		job.joboptions["nr_iter_em"].setString("19");
		REQUIRE_FALSE(generateCommands(job, commands, final_command, error_message));
	}
}

TEST_CASE("Class2D: continuation keeps the selected replica root and runs once",
          "[pipeline][class2d]")
{
	RelionJob job = makeClass2DJob();
	job.is_continue = true;
	job.joboptions["fn_cont"].setString("Class2D/job007/run002_it025_optimiser.star");
	job.joboptions["nr_parallel_runs"].setString("3");
	job.joboptions["scratch_dir"].setString("/scratch/class2d");
	job.joboptions["use_gpu"].setString("Yes");
	job.joboptions["gpu_ids"].setString("0;1;2");
	std::vector<std::string> commands;
	std::string final_command, error_message;
	REQUIRE(generateCommands(job, commands, final_command, error_message));
	REQUIRE(commands.size() == 1);
	REQUIRE(commands[0].find(" --o Class2D/job001/run002") != std::string::npos);
	REQUIRE(commands[0].find("--scratch_dir /scratch/class2d/run002") != std::string::npos);
	REQUIRE(commands[0].find("--gpu \"1\"") != std::string::npos);
	REQUIRE(commands[0].find("--random_seed") == std::string::npos);
	REQUIRE(commands[0].find("pipeline_control_task") == std::string::npos);
}

TEST_CASE("Class2D: legacy continuation keeps the run root",
          "[pipeline][class2d]")
{
	RelionJob job = makeClass2DJob();
	job.is_continue = true;
	job.joboptions["fn_cont"].setString("Class2D/job007/run_it025_optimiser.star");
	std::vector<std::string> commands;
	std::string final_command, error_message;
	REQUIRE(generateCommands(job, commands, final_command, error_message));
	REQUIRE(commands.size() == 1);
	REQUIRE(commands[0].find(" --o Class2D/job001/run ") != std::string::npos);
}

TEST_CASE("Class2D: queued replicas use one Slurm array dispatcher",
          "[pipeline][class2d]")
{
	RelionJob job = makeClass2DJob();
	job.joboptions["nr_parallel_runs"].setString("3");
	job.joboptions["random_seed"].setString("700");
	job.joboptions["do_queue"].setString("Yes");
	job.joboptions["qsub"].setString("sbatch");
	std::string stem = "class2d_array_test_" + integerToString((int)std::clock());
	std::string templ = stem + ".template";
	std::ofstream ft(templ.c_str());
	ft << "#!/bin/sh\n#SBATCH --output=XXXoutfileXXX\n#SBATCH --error=XXXerrfileXXX\nsrun -n XXXmpinodesXXX XXXcommandXXX\n";
	ft.close();
	job.joboptions["qsubscript"].setString(templ);

	std::vector<std::string> commands;
	std::string final_command, error_message;
	std::string outputname = stem + "/";
	REQUIRE(generateCommands(job, commands, final_command, error_message, true, outputname));
	REQUIRE(final_command.find("sbatch --array=1-3") != std::string::npos);

	std::ifstream fd((outputname + "run_replicas.sh").c_str());
	std::string dispatcher((std::istreambuf_iterator<char>(fd)), std::istreambuf_iterator<char>());
	REQUIRE(dispatcher.find("SLURM_ARRAY_TASK_ID") != std::string::npos);
	REQUIRE(dispatcher.find("run001.out") != std::string::npos);
	REQUIRE(dispatcher.find("--random_seed 702") != std::string::npos);
	std::ifstream fs((outputname + "run_submit.script").c_str());
	std::string submission((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());
	REQUIRE(submission.find("sh " + outputname + "run_replicas.sh") != std::string::npos);
	REQUIRE(submission.find("slurm_%A_%a.out") != std::string::npos);
	REQUIRE(submission.find("srun -n 1") != std::string::npos);

	std::remove((outputname + "run_replicas.sh").c_str());
	std::remove((outputname + "run_submit.script").c_str());
	std::remove(templ.c_str());
	::rmdir(stem.c_str());
}

TEST_CASE("Pipeline control aggregates replica completion with status precedence",
          "[pipeline][class2d]")
{
	std::string old_outputname = pipeline_control_outputname;
	int old_task_id = pipeline_control_task_id;
	int old_task_count = pipeline_control_task_count;
	std::string prefix = "pipeline_task_test_" + integerToString((int)std::clock()) + "_";
	pipeline_control_outputname = prefix;
	pipeline_control_task_count = 3;
	{
		std::ofstream stale((prefix + RELION_JOB_EXIT_SUCCESS + "_001").c_str());
	}
	pipeline_control_delete_task_exit_files(prefix, 3);
	REQUIRE_FALSE(exists(prefix + RELION_JOB_EXIT_SUCCESS + "_001"));
	std::remove((prefix + RELION_JOB_EXIT_SUCCESS).c_str());
	std::remove((prefix + RELION_JOB_EXIT_FAILURE).c_str());
	std::remove((prefix + RELION_JOB_EXIT_ABORTED).c_str());

	pipeline_control_task_id = 1;
	REQUIRE(pipeline_control_relion_exit(0) == 0);
	REQUIRE_FALSE(exists(prefix + RELION_JOB_EXIT_SUCCESS));
	pipeline_control_task_id = 2;
	REQUIRE(pipeline_control_relion_exit(1) == 1);
	REQUIRE_FALSE(exists(prefix + RELION_JOB_EXIT_FAILURE));
	pipeline_control_task_id = 3;
	REQUIRE(pipeline_control_relion_exit(2) == 2);
	REQUIRE(exists(prefix + RELION_JOB_EXIT_FAILURE));
	REQUIRE_FALSE(exists(prefix + RELION_JOB_EXIT_ABORTED));

	pipeline_control_delete_task_exit_files(prefix, 3);
	std::remove((prefix + RELION_JOB_EXIT_FAILURE).c_str());
	pipeline_control_task_id = 1;
	pipeline_control_relion_exit(0);
	pipeline_control_task_id = 2;
	pipeline_control_relion_exit(2);
	pipeline_control_task_id = 3;
	pipeline_control_relion_exit(0);
	REQUIRE(exists(prefix + RELION_JOB_EXIT_ABORTED));
	REQUIRE_FALSE(exists(prefix + RELION_JOB_EXIT_SUCCESS));
	pipeline_control_delete_task_exit_files(prefix, 3);
	std::remove((prefix + RELION_JOB_EXIT_ABORTED).c_str());
	pipeline_control_task_id = 1;
	pipeline_control_relion_exit(0);
	pipeline_control_task_id = 2;
	pipeline_control_relion_exit(0);
	pipeline_control_task_id = 3;
	pipeline_control_relion_exit(0);
	REQUIRE(exists(prefix + RELION_JOB_EXIT_SUCCESS));
	pipeline_control_delete_task_exit_files(prefix, 3);
	std::remove((prefix + RELION_JOB_EXIT_SUCCESS).c_str());
	pipeline_control_outputname = old_outputname;
	pipeline_control_task_id = old_task_id;
	pipeline_control_task_count = old_task_count;
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
