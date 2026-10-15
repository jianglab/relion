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
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>

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

TEST_CASE("Reconstruct3D: fused-extraction options emit flags only when asked",
          "[pipeline][reconstruct3d][fused]")
{
	RelionJob job;
	job.clear();
	job.initialise(PROC_RECONSTRUCT3D);
	job.label = get_proc_label(job.type);
	job.joboptions["fn_img"].setString("particles.star");
	job.joboptions["nr_mpi"].setString("1");
	job.joboptions["nr_threads"].setString("1");
	job.joboptions["do_queue"].setString("No");
	REQUIRE(job.joboptions.find("do_fused") != job.joboptions.end());
	REQUIRE_FALSE(job.joboptions["do_fused"].getBoolean());

	std::string command;
	REQUIRE(generateCommand(job, command));
	REQUIRE(command.find("--fused_extract") == std::string::npos);
	REQUIRE(command.find("--box_x") == std::string::npos);

	job.joboptions["do_fused"].setString("Yes");
	job.joboptions["fused_angpix"].setString("2.5");
	job.joboptions["fused_box_x"].setString("200");
	job.joboptions["fused_box_y"].setString("96");
	REQUIRE(generateCommand(job, command));
	REQUIRE(command.find(" --fused_extract") != std::string::npos);
	REQUIRE(command.find(" --angpix 2.5") != std::string::npos);
	REQUIRE(command.find(" --box_x 200") != std::string::npos);
	REQUIRE(command.find(" --box_y 96") != std::string::npos);

	// -1 means "take from the stored particles": no flag
	job.joboptions["fused_angpix"].setString("-1");
	job.joboptions["fused_box_x"].setString("-1");
	job.joboptions["fused_box_y"].setString("-1");
	REQUIRE(generateCommand(job, command));
	REQUIRE(command.find(" --fused_extract") != std::string::npos);
	REQUIRE(command.find("--angpix") == std::string::npos);
	REQUIRE(command.find("--box_x") == std::string::npos);
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
// Class/half-map alignment flags
//
// These exist because the underlying bug was invisible: --align_classes and
// --align_halves default to on, but unchecking the GUI box emitted nothing at
// all, so the job kept aligning.  Assert that the negative flag is emitted, not
// merely that the positive one is.
// ---------------------------------------------------------------------------

TEST_CASE("Class3D: align-classes checkbox emits a flag either way",
          "[pipeline][class3d]")
{
	RelionJob job;
	job.clear();
	job.initialise(PROC_3DCLASS);
	job.label = get_proc_label(job.type);
	job.joboptions["fn_img"].setString("particles.star");
	job.joboptions["fn_ref"].setString("ref.mrc");
	job.joboptions["do_em"].setString("Yes");
	job.joboptions["do_grad"].setString("No");
	job.joboptions["nr_mpi"].setString("1");
	job.joboptions["nr_threads"].setString("1");
	job.joboptions["do_queue"].setString("No");
	job.joboptions["scratch_dir"].setString("");
	REQUIRE(job.joboptions.find("do_align") != job.joboptions.end());
	// Aligning classes to the largest is the default
	REQUIRE(job.joboptions["do_align"].getBoolean());

	std::string command;
	REQUIRE(generateCommand(job, command));
	REQUIRE(command.find(" --align_classes ") != std::string::npos);
	REQUIRE(command.find(" --dont_align_classes ") == std::string::npos);

	job.joboptions["do_align"].setString("No");
	command.clear();
	REQUIRE(generateCommand(job, command));
	REQUIRE(command.find(" --dont_align_classes ") != std::string::npos);
}

TEST_CASE("Class3D: GPU with skipped alignment is on by default, RELION_GPU_SKIP_ALIGN=0 refuses it",
          "[pipeline][class3d]")
{
	RelionJob job;
	job.clear();
	job.initialise(PROC_3DCLASS);
	job.label = get_proc_label(job.type);
	job.joboptions["fn_img"].setString("particles.star");
	job.joboptions["fn_ref"].setString("ref.mrc");
	job.joboptions["do_em"].setString("Yes");
	job.joboptions["do_grad"].setString("No");
	job.joboptions["nr_mpi"].setString("1");
	job.joboptions["nr_threads"].setString("1");
	job.joboptions["do_queue"].setString("No");
	job.joboptions["scratch_dir"].setString("");
	job.joboptions["dont_skip_align"].setString("No");
	job.joboptions["use_gpu"].setString("Yes");
	job.joboptions["gpu_ids"].setString("0");

	const char *saved = getenv("RELION_GPU_SKIP_ALIGN");
	const std::string saved_value = saved ? saved : "";

	unsetenv("RELION_GPU_SKIP_ALIGN");
	std::string command;
	bool ok = generateCommand(job, command);
	REQUIRE(ok);
	REQUIRE(command.find(" --skip_align ") != std::string::npos);
	REQUIRE(command.find(" --gpu \"0\"") != std::string::npos);

	setenv("RELION_GPU_SKIP_ALIGN", "0", 1);
	command.clear();
	ok = generateCommand(job, command);
	if (saved) setenv("RELION_GPU_SKIP_ALIGN", saved_value.c_str(), 1);
	else unsetenv("RELION_GPU_SKIP_ALIGN");
	REQUIRE_FALSE(ok);
}

TEST_CASE("Refine3D: align-halves checkbox emits a flag either way",
          "[pipeline][refine3d]")
{
	RelionJob job;
	job.clear();
	job.initialise(PROC_3DAUTO);
	job.label = get_proc_label(job.type);
	job.joboptions["fn_img"].setString("particles.star");
	job.joboptions["fn_ref"].setString("ref.mrc");
	job.joboptions["nr_mpi"].setString("3");
	job.joboptions["nr_threads"].setString("1");
	job.joboptions["do_queue"].setString("No");
	job.joboptions["scratch_dir"].setString("");
	REQUIRE(job.joboptions.find("do_align_halves") != job.joboptions.end());
	// Aligning the two half-maps to each other is the default
	REQUIRE(job.joboptions["do_align_halves"].getBoolean());

	std::string command;
	REQUIRE(generateCommand(job, command));
	REQUIRE(command.find(" --align_halves ") != std::string::npos);
	REQUIRE(command.find(" --dont_align_halves ") == std::string::npos);

	job.joboptions["do_align_halves"].setString("No");
	command.clear();
	REQUIRE(generateCommand(job, command));
	REQUIRE(command.find(" --dont_align_halves ") != std::string::npos);
}

TEST_CASE("Extract: virtual-particles checkbox emits a flag either way",
          "[pipeline][extract]")
{
	RelionJob job;
	job.clear();
	job.initialise(PROC_EXTRACT);
	job.label = get_proc_label(job.type);
	job.joboptions["star_mics"].setString("CtfFind/job003/micrographs_ctf.star");
	job.joboptions["coords_suffix"].setString("AutoPick/job006/autopick.star");
	job.joboptions["nr_mpi"].setString("1");
	job.joboptions["do_queue"].setString("No");
	REQUIRE(job.joboptions.find("do_virtual") != job.joboptions.end());

	// Explicit both ways, so the job does not depend on the environment it
	// happens to run in (RELION_VIRTUAL_PARTICLES only sets the GUI default)
	job.joboptions["do_virtual"].setString("Yes");
	std::string command;
	REQUIRE(generateCommand(job, command));
	REQUIRE(command.find(" --virtual ") != std::string::npos);
	REQUIRE(command.find(" --no_virtual ") == std::string::npos);

	job.joboptions["do_virtual"].setString("No");
	command.clear();
	REQUIRE(generateCommand(job, command));
	REQUIRE(command.find(" --no_virtual ") != std::string::npos);
}

TEST_CASE("Extract: rectangular box and rotation options are emitted only when used",
          "[pipeline][extract]")
{
	RelionJob job;
	job.clear();
	job.initialise(PROC_EXTRACT);
	job.label = get_proc_label(job.type);
	job.joboptions["star_mics"].setString("CtfFind/job003/micrographs_ctf.star");
	job.joboptions["coords_suffix"].setString("AutoPick/job006/autopick.star");
	job.joboptions["nr_mpi"].setString("1");
	job.joboptions["do_queue"].setString("No");

	std::string command;
	REQUIRE(generateCommand(job, command));
	REQUIRE(command.find(" --extract_size ") != std::string::npos);
	REQUIRE(command.find("--extract_size_x") == std::string::npos);
	REQUIRE(command.find("--extract_size_y") == std::string::npos);
	REQUIRE(command.find("--rotate_to_horizontal") == std::string::npos);
	REQUIRE(command.find("--interpolation") == std::string::npos);

	job.joboptions["do_rect_box"].setString("Yes");
	job.joboptions["extract_size_x"].setString("768");
	job.joboptions["extract_size_y"].setString("256");
	command.clear();
	REQUIRE(generateCommand(job, command));
	REQUIRE(command.find(" --extract_size_x 768") != std::string::npos);
	REQUIRE(command.find(" --extract_size_y 256") != std::string::npos);
	REQUIRE(command.find(" --extract_size ") == std::string::npos);
	REQUIRE(command.find("--rotate_to_horizontal") == std::string::npos);

	job.joboptions["do_extract_helix"].setString("Yes");
	job.joboptions["do_rotate_horizontal"].setString("Yes");
	job.joboptions["extract_interpolation"].setString("cubic");
	command.clear();
	REQUIRE(generateCommand(job, command));
	REQUIRE(command.find(" --rotate_to_horizontal --interpolation cubic") != std::string::npos);

	// A rectangular box is downscaled by a factor (the re-scaled size is for square boxes);
	// both sides must stay even whole numbers, which is checked before the job runs
	job.joboptions["do_rescale"].setString("Yes");
	job.joboptions["rescale"].setString("384");
	job.joboptions["rescale_factor"].setString("2");
	job.joboptions["do_norm"].setString("Yes");
	job.joboptions["bg_diameter"].setString("200");
	command.clear();
	REQUIRE(generateCommand(job, command));
	REQUIRE(command.find(" --downscale_factor 2") != std::string::npos);
	REQUIRE(command.find(" --scale ") == std::string::npos);
	REQUIRE(command.find(" --bg_radius 50") != std::string::npos); // 200 / 2 pixels before, halved

	job.joboptions["rescale_factor"].setString("3");      // 256 / 3 rows
	command.clear();
	REQUIRE_FALSE(generateCommand(job, command));
	{
		std::string outputname, final_command, error_message;
		std::vector<std::string> commands;
		REQUIRE_FALSE(job.getCommands(outputname, commands, final_command, false, 1, error_message));
		REQUIRE(error_message.find("2.97674 (258 x 86)") != std::string::npos);
	}

	job.joboptions["rescale_factor"].setString("2");
	job.joboptions["extract_size_y"].setString("250");   // 125 rows
	command.clear();
	REQUIRE_FALSE(generateCommand(job, command));

	// A square box is still re-scaled to a size
	job.joboptions["do_rect_box"].setString("No");
	job.joboptions["extract_size"].setString("256");
	command.clear();
	REQUIRE(generateCommand(job, command));
	REQUIRE(command.find(" --scale 384") != std::string::npos);
	REQUIRE(command.find("--downscale_factor") == std::string::npos);
}

// relion_refine deletes old iteration files unless told to keep them; every
// refinement job must say which, so the job does what its GUI box shows even
// when RELION_KEEP_ALL_ITERATIONS changes relion_refine's own default.
static void setUpRefineJob(RelionJob &job, int type)
{
	job.clear();
	job.initialise(type);
	job.label = get_proc_label(job.type);
	job.joboptions["fn_img"].setString("particles.star");
	if (job.joboptions.find("fn_ref") != job.joboptions.end())
		job.joboptions["fn_ref"].setString("ref.mrc");
	if (job.joboptions.find("do_grad") != job.joboptions.end())
		job.joboptions["do_grad"].setString("No");
	if (job.joboptions.find("do_em") != job.joboptions.end())
		job.joboptions["do_em"].setString("Yes");
	job.joboptions["nr_mpi"].setString(type == PROC_3DAUTO ? "3" : "1");
	job.joboptions["nr_threads"].setString("1");
	job.joboptions["do_queue"].setString("No");
	job.joboptions["scratch_dir"].setString("");
}

TEST_CASE("Refinement jobs: keep-all-iterations emits a flag either way",
          "[pipeline][keep_iterations]")
{
	unsetenv("RELION_KEEP_ALL_ITERATIONS");
	const int types[] = {PROC_2DCLASS, PROC_INIMODEL, PROC_3DCLASS, PROC_3DAUTO};
	for (int t : types)
	{
		INFO("job type " << t);
		RelionJob job;
		setUpRefineJob(job, t);
		REQUIRE(job.joboptions.find("keep_all_iter") != job.joboptions.end());
		// Pruning old iterations is the default
		REQUIRE_FALSE(job.joboptions["keep_all_iter"].getBoolean());

		std::string command;
		REQUIRE(generateCommand(job, command));
		REQUIRE(command.find(" --dont_keep_all_iterations") != std::string::npos);
		REQUIRE(command.find(" --keep_all_iterations") == std::string::npos);

		job.joboptions["keep_all_iter"].setString("Yes");
		command.clear();
		REQUIRE(generateCommand(job, command));
		REQUIRE(command.find(" --keep_all_iterations") != std::string::npos);
		REQUIRE(command.find(" --dont_keep_all_iterations") == std::string::npos);
	}
}

TEST_CASE("Refinement jobs: RELION_KEEP_ALL_ITERATIONS sets the GUI default",
          "[pipeline][keep_iterations]")
{
	setenv("RELION_KEEP_ALL_ITERATIONS", "1", 1);
	RelionJob job;
	setUpRefineJob(job, PROC_3DCLASS);
	REQUIRE(job.joboptions["keep_all_iter"].getBoolean());
	std::string command;
	REQUIRE(generateCommand(job, command));
	REQUIRE(command.find(" --keep_all_iterations") != std::string::npos);
	unsetenv("RELION_KEEP_ALL_ITERATIONS");
}

TEST_CASE("Refinement jobs: masks along the length of rectangular boxes only when asked",
          "[pipeline][rectangular]")
{
	const int types[] = {PROC_2DCLASS, PROC_3DCLASS, PROC_3DAUTO};
	for (int t : types)
	{
		INFO("job type " << t);
		RelionJob job;
		setUpRefineJob(job, t);
		REQUIRE_FALSE(job.joboptions["do_rect_mask_length"].getBoolean());
		std::string command;
		REQUIRE(generateCommand(job, command));
		REQUIRE(command.find("--rect_mask_length") == std::string::npos);

		job.joboptions["do_rect_mask_length"].setString("Yes");
		command.clear();
		REQUIRE(generateCommand(job, command));
		REQUIRE(command.find(" --rect_mask_length") != std::string::npos);
	}
}

// MotionCorr: how micrographs are stored, always passed explicitly
static void setUpMotioncorrJob(RelionJob &job)
{
	job.clear();
	job.initialise(PROC_MOTIONCORR);
	job.label = get_proc_label(job.type);
	job.joboptions["input_star_mics"].setString("Import/job001/movies.star");
	job.joboptions["nr_mpi"].setString("1");
	job.joboptions["nr_threads"].setString("1");
	job.joboptions["do_queue"].setString("No");
}

TEST_CASE("MotionCorr: movie averages choice emits a flag either way", "[pipeline][motioncorr]")
{
	unsetenv("RELION_VIRTUAL_MOVIE_AVERAGES");
	RelionJob job;
	setUpMotioncorrJob(job);
	REQUIRE(job.joboptions.find("movie_averages") != job.joboptions.end());
	REQUIRE(job.joboptions["movie_averages"].getString() == job_movie_average_options[0]);   // legacy by default

	const char *flags[] = {" --movie_averages legacy", " --movie_averages real", " --movie_averages virtual"};
	for (int i = 0; i < 3; i++)
	{
		INFO("choice " << i);
		job.joboptions["movie_averages"].setString(job_movie_average_options[i]);
		std::string command;
		REQUIRE(generateCommand(job, command));
		REQUIRE(command.find(flags[i]) != std::string::npos);
	}

	// Refused where they cannot work
	std::string command;
	job.joboptions["movie_averages"].setString(job_movie_average_options[2]);
	job.joboptions["do_save_ps"].setString("No");
	job.joboptions["do_float16"].setString("No");
	REQUIRE_FALSE(generateCommand(job, command));          // CTFFIND would have nothing to read
	job.joboptions["do_save_ps"].setString("Yes");
	job.joboptions["do_own_motioncor"].setString("No");
	job.joboptions["movie_averages"].setString(job_movie_average_options[1]);
	REQUIRE_FALSE(generateCommand(job, command));          // MotionCor2's sums cannot be regenerated
	// Legacy with MotionCor2 is fine (MotionCor2 writes no power spectra)
	job.joboptions["movie_averages"].setString(job_movie_average_options[0]);
	job.joboptions["do_save_ps"].setString("No");
	REQUIRE(generateCommand(job, command));
	REQUIRE(command.find(" --movie_averages legacy") != std::string::npos);
}

TEST_CASE("MotionCorr: RELION_VIRTUAL_MOVIE_AVERAGES sets the default choice", "[pipeline][motioncorr]")
{
	const char *env[] = {"1", "real"};
	for (int i = 0; i < 2; i++)
	{
		setenv("RELION_VIRTUAL_MOVIE_AVERAGES", env[i], 1);
		RelionJob job;
		setUpMotioncorrJob(job);
		REQUIRE(job.joboptions["movie_averages"].getString() == job_movie_average_options[i == 0 ? 2 : 1]);
	}
	unsetenv("RELION_VIRTUAL_MOVIE_AVERAGES");
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
	job.label = get_proc_label(job.type);
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

TEST_CASE("Class2D: filament context options reach the command line",
          "[pipeline][class2d][filament_context]")
{
	RelionJob job = makeClass2DJob();
	for (const char *name : {"filctx_mode", "filctx_temper", "filctx_gap", "filctx_floor", "filctx_alpha"})
		REQUIRE(job.joboptions.find(name) != job.joboptions.end());

	std::vector<std::string> commands;
	std::string final_command, error_message;

	SECTION("not a helical job: nothing is emitted")
	{
		REQUIRE(generateCommands(job, commands, final_command, error_message));
		REQUIRE(commands[0].find("--filament_") == std::string::npos);
	}

	job.joboptions["do_helix"].setString("Yes");

	SECTION("helical, default: off is emitted explicitly")
	{
		REQUIRE(generateCommands(job, commands, final_command, error_message));
		REQUIRE(commands[0].find(" --filament_context off") != std::string::npos);
		REQUIRE(commands[0].find("--filament_temper") == std::string::npos);
	}

	SECTION("chain mode with its settings")
	{
		job.joboptions["filctx_mode"].setString(job_filament_context_options[1]);
		job.joboptions["filctx_gap"].setString("3");
		job.joboptions["filctx_floor"].setString("0.05");
		REQUIRE(generateCommands(job, commands, final_command, error_message));
		REQUIRE(commands[0].find(" --filament_context chain") != std::string::npos);
		REQUIRE(commands[0].find(" --filament_temper -1") != std::string::npos);
		REQUIRE(commands[0].find(" --filament_gap 3") != std::string::npos);
		REQUIRE(commands[0].find(" --filament_floor 0.05") != std::string::npos);
		REQUIRE(commands[0].find("--filament_alpha") == std::string::npos);
	}

	SECTION("filament mode also passes alpha")
	{
		job.joboptions["filctx_mode"].setString(job_filament_context_options[2]);
		job.joboptions["filctx_temper"].setString("0.25");
		job.joboptions["filctx_alpha"].setString("2");
		REQUIRE(generateCommands(job, commands, final_command, error_message));
		REQUIRE(commands[0].find(" --filament_context filament") != std::string::npos);
		REQUIRE(commands[0].find(" --filament_temper 0.25") != std::string::npos);
		REQUIRE(commands[0].find(" --filament_alpha 2") != std::string::npos);
	}

	SECTION("a gap below 1 is refused")
	{
		job.joboptions["filctx_mode"].setString(job_filament_context_options[1]);
		job.joboptions["filctx_gap"].setString("0");
		REQUIRE_FALSE(generateCommands(job, commands, final_command, error_message));
		REQUIRE(error_message.find("gap") != std::string::npos);
	}
}

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
// Class2D consensus
// ---------------------------------------------------------------------------

// Real metadata, deliberately independent of the source job's saved K.
static void writeConsensusSourceModel(const std::string &directory, int classes)
{
	std::ofstream optimiser(directory + "run001_it025_optimiser.star");
	optimiser << "data_optimiser_general\n_rlnModelStarFile " << directory << "model.star\n";
	std::ofstream model(directory + "model.star");
	model << "data_model_general\n_rlnNrClasses " << classes << "\n_rlnReferenceDimensionality 2\n";
}

static void removeConsensusSourceModel(const std::string &directory)
{
	std::remove((directory + "run001_it025_optimiser.star").c_str());
	std::remove((directory + "model.star").c_str());
}

TEST_CASE("Class2DConsensus: generates preparation and configurable frozen-class refinement", "[pipeline][consensus]")
{
	const std::string directory = "class2d_consensus_source_" + integerToString((int)std::clock()) + "/";
	REQUIRE(::mkdir(directory.c_str(), 0700) == 0);
	RelionJob source = makeClass2DJob();
	source.joboptions["nr_parallel_runs"].setString("3");
	source.joboptions["nr_classes"].setString("7");
	source.joboptions["psi_sampling"].setString("5");
	source.write(directory);
	writeConsensusSourceModel(directory, 9);

	RelionJob job;
	job.clear();
	job.initialise(PROC_CLASS2D_CONSENSUS);
	job.joboptions["fn_optimiser"].setString(directory + "run001_it025_optimiser.star");
	job.joboptions["nr_mpi"].setString("1");
	job.joboptions["nr_threads"].setString("2");
	job.joboptions["do_queue"].setString("No");
	job.joboptions["scratch_dir"].setString("");

	REQUIRE(job.joboptions["nr_iter"].getString() == "25");
	REQUIRE(job.joboptions["do_reset_alignments"].getBoolean());
	int iterations = 25;
	std::string iteration_suffix = "025";
	bool legacy_keep = false;
	int classes = 9;
	std::string method = "categorical_em";
	REQUIRE(job.joboptions["consensus_method"].getString() == "Categorical EM");
	SECTION("defaults") {}
	SECTION("sparse NMF") { method = "sparse_nmf"; job.joboptions["consensus_method"].setString("Sparse NMF (experimental)"); }
	SECTION("one iteration") { iterations = 1; iteration_suffix = "001"; }
	SECTION("custom iteration count") { iterations = 7; iteration_suffix = "007"; }
	SECTION("above the GUI slider range") { iterations = 51; iteration_suffix = "051"; }
	SECTION("more than three digits") { iterations = 1000; iteration_suffix = "1000"; }
	SECTION("largest representable iteration count") { iterations = INT_MAX; iteration_suffix = std::to_string(INT_MAX); }
	SECTION("old keep-alignments setting is ignored")
	{
		legacy_keep = true;
		job.joboptions["do_reset_alignments"].setString("No");
	}
	SECTION("smaller consensus count") { classes = 3; job.joboptions["nr_classes"].setString("3"); }
	SECTION("larger consensus count") { classes = 12; job.joboptions["nr_classes"].setString("12"); }
	job.joboptions["nr_iter"].setString(integerToString(iterations));

	std::vector<std::string> commands;
	std::string final_command, error_message;
	REQUIRE(generateCommands(job, commands, final_command, error_message));
	REQUIRE(commands.size() == 2);
	REQUIRE(commands[0].find(" --method " + method) != std::string::npos);
	REQUIRE(commands[1].find("--method") == std::string::npos);
	REQUIRE(commands[1].find("--nmf_") == std::string::npos);
	REQUIRE(commands[0].find("relion_class2d_consensus") != std::string::npos);
	REQUIRE(commands[0].find("--nr_runs 3") != std::string::npos);
	REQUIRE((commands[0].find(" --legacy_keep_alignments") != std::string::npos) == legacy_keep);
	REQUIRE(commands[0].find(" --K " + integerToString(classes) + " ") != std::string::npos);
	REQUIRE(commands[0].find("--pipeline_control") == std::string::npos);
	REQUIRE(commands[1].find("relion_refine") != std::string::npos);
	REQUIRE(commands[1].find("--iter " + integerToString(iterations) + " ") != std::string::npos);
	REQUIRE(commands[1].find("--reset_alignments") == std::string::npos);
	REQUIRE(commands[1].find("--fix_classes") != std::string::npos);
	REQUIRE(commands[1].find("--K " + integerToString(classes) + " ") != std::string::npos);
	REQUIRE(commands[1].find(" --ref ") == std::string::npos);
	REQUIRE(commands[1].find("--random_seed 1") != std::string::npos);
	REQUIRE(commands[1].find("--psi_step 10") != std::string::npos);
	REQUIRE(commands[1].find("--pipeline_control") != std::string::npos);
	REQUIRE(final_command.find(" && ") != std::string::npos);
	REQUIRE(job.inputNodes.size() == 3);
	REQUIRE(job.inputNodes[2].name.find("run003_it025_optimiser.star") != std::string::npos);
	REQUIRE(job.outputNodes.size() == 3);
	REQUIRE(job.outputNodes[0].name.find("consensus_data.star") != std::string::npos);
	REQUIRE(job.outputNodes[1].name.find("run_it" + iteration_suffix + "_data.star") != std::string::npos);
	REQUIRE(job.outputNodes[2].name.find("run_it" + iteration_suffix + "_optimiser.star") != std::string::npos);

	removeConsensusSourceModel(directory);
	std::remove((directory + "job.star").c_str());
	::rmdir(directory.c_str());
}

TEST_CASE("Class2DConsensus: rejects invalid refinement iteration counts", "[pipeline][consensus]")
{
	const char *invalid[] = {"", "0", "-1", "2.5", "25x", "1e2", "2147483648", "9999999999999999999999"};
	RelionJob job;
	job.clear();
	job.initialise(PROC_CLASS2D_CONSENSUS);
	for (const char *iterations : invalid)
	{
		INFO("Iteration input: " << iterations);
		job.joboptions["nr_iter"].setString(iterations);
		std::vector<std::string> commands;
		std::string final_command, error_message;
		REQUIRE_FALSE(generateCommands(job, commands, final_command, error_message));
		REQUIRE(error_message.find("iterations must be a positive integer") != std::string::npos);
	}
}

TEST_CASE("Class2DConsensus: validates independent class count and initialization seed", "[pipeline][consensus]")
{
	RelionJob job;
	job.clear();
	job.initialise(PROC_CLASS2D_CONSENSUS);
	std::string option = "nr_classes", expected = "consensus classes";
	std::vector<std::string> invalid{"", "1", "-1", "2.5", "2x", "2147483648"};
	SECTION("class count") {}
	SECTION("seed")
	{
		option = "random_seed";
		expected = "random seed";
		invalid = {"", "0", "-1", "2.5", "2x", "2147483648"};
	}
	for (const auto &value : invalid)
	{
		job.joboptions[option].setString(value);
		std::vector<std::string> commands;
		std::string final_command, error_message;
		REQUIRE_FALSE(generateCommands(job, commands, final_command, error_message));
		REQUIRE(error_message.find(expected) != std::string::npos);
	}
}

TEST_CASE("Class2DConsensus: refinement options survive saved jobs and default in older jobs", "[pipeline][consensus]")
{
	const std::string filename = "class2d_consensus_options_" + integerToString((int)std::clock()) + ".star";
	RelionJob job;
	job.clear();
	job.initialise(PROC_CLASS2D_CONSENSUS);
	std::string expected_iterations = "25";
	bool expected_reset = true;
	std::string expected_method = "Categorical EM";
	SECTION("default options round trip") {}
	SECTION("custom options round trip")
	{
		expected_iterations = "51";
		expected_method = "Sparse NMF (experimental)";
		job.joboptions["consensus_method"].setString(expected_method);
		expected_reset = false;
		job.joboptions["nr_classes"].setString("12");
		job.joboptions["random_seed"].setString("17");
		job.joboptions["nr_iter"].setString(expected_iterations);
		job.joboptions["do_reset_alignments"].setString("No");
	}
	SECTION("old job missing new options")
	{
		job.joboptions.erase("consensus_method");
		job.joboptions.erase("nr_classes");
		job.joboptions.erase("random_seed");
		job.joboptions.erase("nr_iter");
		job.joboptions.erase("do_reset_alignments");
	}
	job.label = get_proc_label(job.type);
	job.write(filename);
	RelionJob restored;
	restored.clear();
	bool is_continue = false;
	REQUIRE(restored.read(filename, is_continue, true));
	REQUIRE(restored.joboptions["consensus_method"].getString() == expected_method);
	REQUIRE(restored.joboptions["nr_iter"].getString() == expected_iterations);
	REQUIRE(restored.joboptions["nr_classes"].getString() == (expected_reset ? "0" : "12"));
	REQUIRE(restored.joboptions["random_seed"].getString() == (expected_reset ? "1" : "17"));
	REQUIRE(restored.joboptions["do_reset_alignments"].getBoolean() == expected_reset);
	std::remove(filename.c_str());
}

TEST_CASE("Class2DConsensus: rejects reserved refinement overrides", "[pipeline][consensus]")
{
	const std::string directory = "class2d_consensus_reserved_" + integerToString((int)std::clock()) + "/";
	REQUIRE(::mkdir(directory.c_str(), 0700) == 0);
	RelionJob source = makeClass2DJob();
	source.joboptions["nr_parallel_runs"].setString("2");
	source.joboptions["nr_classes"].setString("3");
	source.write(directory);
	writeConsensusSourceModel(directory, 3);

	RelionJob job;
	job.clear();
	job.initialise(PROC_CLASS2D_CONSENSUS);
	job.joboptions["fn_optimiser"].setString(directory + "run001_it025_optimiser.star");
	std::string reserved = "--iter";
	SECTION("iteration") {}
	SECTION("references") { reserved = "--ref"; }
	SECTION("classes") { reserved = "--K"; }
	SECTION("seed") { reserved = "--random_seed"; }
	SECTION("SOM") { reserved = "--som"; }
	job.joboptions["other_args"].setString(reserved + " 4");
	std::vector<std::string> commands;
	std::string final_command, error_message;
	REQUIRE_FALSE(generateCommands(job, commands, final_command, error_message));
	REQUIRE(error_message.find(reserved) != std::string::npos);

	removeConsensusSourceModel(directory);
	std::remove((directory + "job.star").c_str());
	::rmdir(directory.c_str());
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

TEST_CASE("Class2DConsensus: rejects unknown assignment methods", "[pipeline][consensus]")
{
    RelionJob job;
    job.clear();
    job.initialise(PROC_CLASS2D_CONSENSUS);
    job.joboptions["consensus_method"].setString("invalid");
    std::vector<std::string> commands;
    std::string final_command, error_message;
    REQUIRE_FALSE(generateCommands(job, commands, final_command, error_message));
    REQUIRE(error_message.find("unknown consensus assignment method") != std::string::npos);
}

// ---------------------------------------------------------------------------
// MPI processes and threads chosen from a number of CPUs
// ---------------------------------------------------------------------------

TEST_CASE("Refinement jobs: MPI processes and threads chosen from the number of CPUs", "[pipeline]")
{
	setenv("RELION_CPUS_PER_GPU", "8", 1);
	RelionJob job = makeClass2DJob();
	REQUIRE(job.joboptions.find("auto_parallel") != job.joboptions.end());
	REQUIRE(job.joboptions.find("nr_cpus") != job.joboptions.end());
	job.joboptions["auto_parallel"].setString("Yes");
	job.joboptions["use_gpu"].setString("Yes");
	job.joboptions["gpu_ids"].setString("0:1");
	job.joboptions["nr_cpus"].setString("16");

	std::vector<std::string> commands;
	std::string final_command, error_message;
	REQUIRE(generateCommands(job, commands, final_command, error_message));
	REQUIRE(commands.size() == 1);
	CHECK(job.joboptions["nr_mpi"].getString() == "3");
	CHECK(job.joboptions["nr_threads"].getString() == "8");
	CHECK(commands[0].find("relion_refine_mpi") != std::string::npos);
	CHECK(commands[0].find("--j 8") != std::string::npos);

	// chosen by hand: the numbers given are used
	job.joboptions["auto_parallel"].setString("No");
	job.joboptions["nr_mpi"].setString("1");
	job.joboptions["nr_threads"].setString("2");
	REQUIRE(generateCommands(job, commands, final_command, error_message));
	CHECK(commands[0].find("relion_refine_mpi") == std::string::npos);
	CHECK(commands[0].find("--j 2") != std::string::npos);
	unsetenv("RELION_CPUS_PER_GPU");
}

TEST_CASE("Refinement jobs: jobs saved before the automatic choice keep their numbers", "[pipeline]")
{
	RelionJob job = makeClass2DJob();
	job.joboptions["auto_parallel"].setString("Yes");
	std::string fn = "auto_parallel_old_" + integerToString((int)std::time(NULL)) + ".star";
	job.write(fn);
	// remove the options from the file, as an older RELION would have written it
	{
		std::ifstream in(fn.c_str());
		std::string line, text;
		while (std::getline(in, line))
			if (line.find("auto_parallel") == std::string::npos && line.find("nr_cpus") == std::string::npos)
				text += line + "\n";
		in.close();
		std::ofstream(fn.c_str()) << text;
	}
	RelionJob loaded;
	loaded.clear();
	bool is_continue = false;
	REQUIRE(loaded.read(fn, is_continue, true));
	CHECK(loaded.joboptions["auto_parallel"].getString() == "No");
	std::remove(fn.c_str());
}

TEST_CASE("Refinement jobs: only refinement jobs choose MPI processes and threads", "[pipeline]")
{
	for (int type : {PROC_2DCLASS, PROC_3DCLASS, PROC_INIMODEL, PROC_3DAUTO, PROC_MULTIBODY})
	{
		RelionJob job;
		job.initialise(type);
		CHECK(job.joboptions.find("auto_parallel") != job.joboptions.end());
	}
	for (int type : {PROC_EXTRACT, PROC_MOTIONCORR, PROC_AUTOPICK, PROC_RECONSTRUCT3D})
	{
		RelionJob job;
		job.initialise(type);
		CHECK(job.joboptions.find("auto_parallel") == job.joboptions.end());
	}
}

TEST_CASE("Queued jobs: an empty submission-script field means RELION's own Slurm script", "[pipeline]")
{
	setenv("RELION_CPUS_PER_GPU", "8", 1);
	const char *saved = getenv("RELION_QSUB_TEMPLATE");
	const std::string saved_value = saved ? saved : "";
	unsetenv("RELION_QSUB_TEMPLATE");

	RelionJob job;
	job.clear();
	job.initialise(PROC_3DAUTO);
	job.label = get_proc_label(job.type);
	REQUIRE(job.joboptions["qsubscript"].getString() == "");
	for (const char *key : {"slurm_gpus", "slurm_mem", "slurm_time", "slurm_args"})
		REQUIRE(job.joboptions.find(key) != job.joboptions.end());
	job.joboptions["fn_img"].setString("particles.star");
	job.joboptions["fn_ref"].setString("ref.mrc");
	job.joboptions["do_queue"].setString("Yes");
	job.joboptions["qsub"].setString("bash");          // ignored by the built-in script
	job.joboptions["queuename"].setString("");
	job.joboptions["use_gpu"].setString("Yes");
	job.joboptions["slurm_gpus"].setString("1");
	job.joboptions["slurm_mem"].setString("12");
	job.joboptions["slurm_time"].setString("1-00:00:00");
	job.joboptions["auto_parallel"].setString("Yes");
	job.joboptions["nr_cpus"].setString("8");
	job.joboptions["scratch_dir"].setString("");

	std::string stem = "builtin_slurm_test_" + integerToString((int)std::clock());
	std::string outputname = stem + "/";
	std::vector<std::string> commands;
	std::string final_command, error_message;
	REQUIRE(generateCommands(job, commands, final_command, error_message, true, outputname));
	CHECK(final_command.find("sbatch " + outputname + "run_submit.script") == 0);
	std::ifstream fs((outputname + "run_submit.script").c_str());
	std::string script((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());
	INFO(script);
	// auto-refine on 1 GPU with 8 CPUs: a leader and 2 workers with 4 threads each
	CHECK(script.find("#SBATCH --ntasks=2\n") != std::string::npos);
	CHECK(script.find("#SBATCH --cpus-per-task=4\n") != std::string::npos);
	CHECK(script.find("#SBATCH --gpus=1\n") != std::string::npos);
	CHECK(script.find("#SBATCH --mem=12G\n") != std::string::npos);
	CHECK(script.find("#SBATCH --time=1-00:00:00\n") != std::string::npos);
	CHECK(script.find("#SBATCH --output=" + outputname + "run.out") != std::string::npos);
	CHECK(script.find("mpirun --oversubscribe --bind-to none -n 3 ") != std::string::npos);
	CHECK(script.find("relion_refine_mpi") != std::string::npos);
	CHECK(script.find("XXX") == std::string::npos);

	// no GPUs asked for when the job does not use them; a bad memory value is refused
	job.joboptions["use_gpu"].setString("No");
	REQUIRE(generateCommands(job, commands, final_command, error_message, true, outputname));
	std::ifstream fs2((outputname + "run_submit.script").c_str());
	std::string script2((std::istreambuf_iterator<char>(fs2)), std::istreambuf_iterator<char>());
	CHECK(script2.find("--gpus") == std::string::npos);
	job.joboptions["slurm_mem"].setString("lots");
	CHECK_FALSE(generateCommands(job, commands, final_command, error_message, true, outputname));
	CHECK(error_message.find("Memory must be a number of GB") != std::string::npos);

	if (saved) setenv("RELION_QSUB_TEMPLATE", saved_value.c_str(), 1);
	unsetenv("RELION_CPUS_PER_GPU");
	std::remove((outputname + "run_submit.script").c_str());
	std::remove((outputname + "note.txt").c_str());
	std::string cleanup = "rm -rf " + stem;
	CHECK(system(cleanup.c_str()) == 0);
}

TEST_CASE("Queued jobs: in RELION's own Slurm script every process of other programs works", "[pipeline]")
{
	const char *saved = getenv("RELION_QSUB_TEMPLATE");
	const std::string saved_value = saved ? saved : "";
	unsetenv("RELION_QSUB_TEMPLATE");
	RelionJob job;
	job.clear();
	job.initialise(PROC_CTFFIND);
	job.label = get_proc_label(job.type);
	job.joboptions["input_star_mics"].setString("mics.star");
	job.joboptions["do_queue"].setString("Yes");
	job.joboptions["queuename"].setString("");
	job.joboptions["nr_mpi"].setString("4");
	job.joboptions["slurm_mem"].setString("8");
	std::string stem = "builtin_slurm_ctf_" + integerToString((int)std::clock());
	std::string outputname = stem + "/";
	std::vector<std::string> commands;
	std::string final_command, error_message;
	REQUIRE(generateCommands(job, commands, final_command, error_message, true, outputname));
	std::ifstream fs((outputname + "run_submit.script").c_str());
	std::string script((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());
	INFO(script);
	CHECK(script.find("#SBATCH --ntasks=4\n") != std::string::npos);
	CHECK(script.find("--oversubscribe") == std::string::npos);
	CHECK(script.find("--gpus") == std::string::npos);
	if (saved) setenv("RELION_QSUB_TEMPLATE", saved_value.c_str(), 1);
	std::string cleanup = "rm -rf " + stem;
	CHECK(system(cleanup.c_str()) == 0);
}

TEST_CASE("Queued jobs: only jobs that run on GPUs ask Slurm for them", "[pipeline]")
{
	const char *saved = getenv("RELION_QSUB_TEMPLATE");
	const std::string saved_value = saved ? saved : "";
	unsetenv("RELION_QSUB_TEMPLATE");
	RelionJob mc;
	mc.initialise(PROC_MOTIONCORR);
	mc.joboptions["slurm_gpus"].setString("1");
	mc.joboptions["do_own_motioncor"].setString("Yes");
	CHECK_FALSE(mc.usesGpu());
	CHECK(mc.slurmRequest().gpus == 0);
	mc.joboptions["do_own_motioncor"].setString("No");      // MotionCor2
	CHECK(mc.usesGpu());
	CHECK(mc.slurmRequest().gpus == 1);

	RelionJob pick;
	pick.initialise(PROC_AUTOPICK);
	pick.joboptions["use_gpu"].setString("Yes");
	pick.joboptions["do_log"].setString("Yes");
	CHECK_FALSE(pick.usesGpu());
	pick.joboptions["do_log"].setString("No");
	CHECK(pick.usesGpu());

	for (int type : {PROC_CTFFIND, PROC_EXTRACT, PROC_POST, PROC_MASKCREATE})
	{
		RelionJob job;
		job.initialise(type);
		job.joboptions["slurm_gpus"].setString("2");
		CHECK_FALSE(job.usesGpu());
		CHECK(job.slurmRequest().gpus == 0);
	}
	RelionJob refine;
	refine.initialise(PROC_3DAUTO);
	refine.joboptions["slurm_gpus"].setString("2");
	refine.joboptions["use_gpu"].setString("No");
	CHECK(refine.slurmRequest().gpus == 0);
	refine.joboptions["use_gpu"].setString("Yes");
	CHECK(refine.slurmRequest().gpus == 2);

	// Class3D without alignment: the GPU unless RELION_GPU_SKIP_ALIGN=0; the default needs no
	// variable on the node
	const char *saved_skip = getenv("RELION_GPU_SKIP_ALIGN");
	const std::string saved_skip_value = saved_skip ? saved_skip : "";
	RelionJob c3d;
	c3d.initialise(PROC_3DCLASS);
	c3d.joboptions["slurm_gpus"].setString("1");
	c3d.joboptions["use_gpu"].setString("Yes");
	unsetenv("RELION_GPU_SKIP_ALIGN");
	CHECK(c3d.usesGpu());                       // with alignment
	CHECK(c3d.slurmRequest().exports.empty());
	c3d.joboptions["dont_skip_align"].setString("No");
	CHECK(c3d.usesGpu());                       // without alignment: still on the GPU
	CHECK(c3d.slurmRequest().gpus == 1);
	CHECK(c3d.slurmRequest().exports.empty());
	setenv("RELION_GPU_SKIP_ALIGN", "0", 1);
	CHECK_FALSE(c3d.usesGpu());
	CHECK(c3d.slurmRequest().gpus == 0);
	if (saved_skip) setenv("RELION_GPU_SKIP_ALIGN", saved_skip_value.c_str(), 1);
	else unsetenv("RELION_GPU_SKIP_ALIGN");
	if (saved) setenv("RELION_QSUB_TEMPLATE", saved_value.c_str(), 1);
}

TEST_CASE("Queued jobs: a site template keeps its extra fields and no built-in ones", "[pipeline]")
{
	const char *saved = getenv("RELION_QSUB_TEMPLATE");
	const std::string saved_value = saved ? saved : "";
	setenv("RELION_QSUB_TEMPLATE", "/some/site/template.sh", 1);
	RelionJob job;
	job.initialise(PROC_3DAUTO);
	CHECK(job.joboptions["qsubscript"].getString() == "/some/site/template.sh");
	CHECK(job.joboptions.find("slurm_mem") == job.joboptions.end());
	if (saved) setenv("RELION_QSUB_TEMPLATE", saved_value.c_str(), 1);
	else unsetenv("RELION_QSUB_TEMPLATE");
}
