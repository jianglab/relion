/*
 * tests/unit/test_gui_jobwindows.cpp
 *
 * Builds the GUI job window of every job type and checks that each job option
 * has a GUI entry.  A job option without one makes updateMyJob() abort the
 * whole GUI ("cannot find X in the defined GUI entries") the moment the user
 * selects that job type, which no non-GUI test can notice.
 * FLTK widgets are created but never shown, so no display is needed.
 */

#include <signal.h>
#ifndef SIGSTKSZ
#  define SIGSTKSZ 65536
#elif defined(__GLIBC__) && __GLIBC__ >= 2 && defined(__GLIBC_MINOR__) && __GLIBC_MINOR__ >= 34
#  undef  SIGSTKSZ
#  define SIGSTKSZ 65536
#endif

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "src/gui_jobwindow.h"
#include "src/pipeline_jobs.h"

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

static const std::vector<std::pair<int, std::string> > &allGuiJobTypes()
{
	static const std::vector<std::pair<int, std::string> > types = {
		{PROC_IMPORT, "Import"}, {PROC_MOTIONCORR, "MotionCorr"}, {PROC_CTFFIND, "CtfFind"},
		{PROC_MANUALPICK, "ManualPick"}, {PROC_AUTOPICK, "AutoPick"}, {PROC_EXTRACT, "Extract"},
		{PROC_CLASSSELECT, "Select"}, {PROC_SELECT2D, "Select2D"}, {PROC_COOCCURRENCE, "CoOccurrence"},
		{PROC_CLASS2D_CONSENSUS, "Class2DConsensus"}, {PROC_2DCLASS, "Class2D"},
		{PROC_3DCLASS, "Class3D"}, {PROC_3DAUTO, "Refine3D"}, {PROC_MASKCREATE, "MaskCreate"},
		{PROC_JOINSTAR, "JoinStar"}, {PROC_SUBTRACT, "Subtract"}, {PROC_POST, "PostProcess"},
		{PROC_RESMAP, "LocalRes"}, {PROC_INIMODEL, "InitialModel"}, {PROC_MULTIBODY, "MultiBody"},
		{PROC_MOTIONREFINE, "Polish"}, {PROC_CTFREFINE, "CtfRefine"}, {PROC_DYNAMIGHT, "DynaMight"},
		{PROC_MODELANGELO, "ModelAngelo"}, {PROC_RECONSTRUCT3D, "Reconstruct3D"},
		{PROC_EXTERNAL, "External"},
	};
	return types;
}

static const std::vector<std::pair<int, std::string> > &allTomoGuiJobTypes()
{
	static const std::vector<std::pair<int, std::string> > types = {
		{PROC_TOMO_IMPORT, "TomoImport"}, {PROC_TOMO_EXCLUDE_TILT_IMAGES, "TomoExcludeTilts"},
		{PROC_TOMO_ALIGN_TILTSERIES, "TomoAlignTiltSeries"}, {PROC_TOMO_RECONSTRUCT_TOMOGRAM, "TomoTomograms"},
		{PROC_TOMO_DENOISE_TOMOGRAM, "TomoDenoise"}, {PROC_TOMO_PICK_TOMOGRAM, "TomoPick"},
		{PROC_TOMO_SUBTOMO, "TomoSubtomo"}, {PROC_TOMO_CTFREFINE, "TomoCtfRefine"},
		{PROC_TOMO_ALIGN, "TomoAlign"}, {PROC_TOMO_RECONSTRUCT, "TomoReconstructParticle"},
	};
	return types;
}

TEST_CASE("GUI: every job option of every job type has a GUI entry", "[gui][jobwindow]")
{
	for (const auto &t : allGuiJobTypes())
	{
		INFO("job type " << t.second);
		{
			JobWindow window;
			REQUIRE_NOTHROW(window.initialise(t.first, false));
			REQUIRE_NOTHROW(window.updateMyJob());
		}
	}
}

TEST_CASE("GUI: every job option of every tomography job type has a GUI entry", "[gui][jobwindow]")
{
	for (const auto &t : allTomoGuiJobTypes())
	{
		INFO("job type " << t.second);
		{
			JobWindow window;
			REQUIRE_NOTHROW(window.initialise(t.first, true));
			REQUIRE_NOTHROW(window.updateMyJob());
		}
	}
}

TEST_CASE("GUI: Extract window has the rectangular-box controls", "[gui][jobwindow]")
{
	JobWindow window;
	window.initialise(PROC_EXTRACT, false);
	for (const char *name : {"do_rect_box", "extract_size_x", "extract_size_y",
	                          "do_rotate_horizontal", "extract_interpolation"})
		REQUIRE(window.guientries.find(name) != window.guientries.end());
}

TEST_CASE("GUI: Reconstruct3D window has the fused-extraction controls", "[gui][jobwindow]")
{
	JobWindow window;
	window.initialise(PROC_RECONSTRUCT3D, false);
	for (const char *name : {"do_fused", "fused_angpix", "fused_box_x", "fused_box_y"})
		REQUIRE(window.guientries.find(name) != window.guientries.end());
}

TEST_CASE("GUI: MPI and thread fields suggest numbers when the mouse enters them", "[gui][jobwindow]")
{
	setenv("RELION_CPUS_PER_GPU", "8", 1);
	JobWindow window;
	window.initialise(PROC_2DCLASS, false);
	window.guientries["use_gpu"].inp->value("Yes");
	window.guientries["do_grad"].inp->value("No");
	window.guientries["do_queue"].inp->value("No");
	window.guientries["gpu_ids"].inp->value("0:1");
	window.guientries["auto_parallel"].inp->value("No");
	for (const char *name : {"nr_mpi", "nr_threads"})
	{
		INFO(name);
		Fl_Input *inp = window.guientries[name].inp;
		inp->handle(FL_ENTER);
		REQUIRE(inp->tooltip() != NULL);
		CHECK(std::string(inp->tooltip()).find("MPI processes: 3    Threads: 8") != std::string::npos);
	}
	// chosen by RELION: the greyed-out fields show the numbers it will use
	window.guientries["auto_parallel"].inp->value("Yes");
	window.guientries["nr_cpus"].inp->value("16");
	Fl_Input *cpus = window.guientries["nr_cpus"].inp;
	cpus->handle(FL_ENTER);
	REQUIRE(cpus->tooltip() != NULL);
	CHECK(std::string(cpus->tooltip()).find("RELION splits the 16 CPUs") != std::string::npos);
	CHECK(std::string(window.guientries["nr_mpi"].inp->value()) == "3");
	CHECK(std::string(window.guientries["nr_threads"].inp->value()) == "8");
	// other jobs keep fields without a tooltip
	JobWindow extract;
	extract.initialise(PROC_EXTRACT, false);
	REQUIRE(extract.guientries.find("nr_mpi") != extract.guientries.end());
	Fl_Input *inp = extract.guientries["nr_mpi"].inp;
	inp->handle(FL_ENTER);
	CHECK(inp->tooltip() == NULL);
	unsetenv("RELION_CPUS_PER_GPU");
}

TEST_CASE("GUI: without a site template the Running tab has the built-in Slurm fields", "[gui][jobwindow]")
{
	const char *saved = getenv("RELION_QSUB_TEMPLATE");
	const std::string saved_value = saved ? saved : "";
	unsetenv("RELION_QSUB_TEMPLATE");
	for (int type : {PROC_3DAUTO, PROC_EXTRACT, PROC_MOTIONCORR})
	{
		JobWindow window;
		REQUIRE_NOTHROW(window.initialise(type, false));
		REQUIRE_NOTHROW(window.updateMyJob());
		for (const char *name : {"slurm_gpus", "slurm_mem", "slurm_time", "slurm_args"})
			CHECK(window.guientries.find(name) != window.guientries.end());
	}
	if (saved) setenv("RELION_QSUB_TEMPLATE", saved_value.c_str(), 1);
}

TEST_CASE("GUI: template-only fields are greyed out while RELION writes the Slurm script", "[gui][jobwindow]")
{
	const char *saved = getenv("RELION_QSUB_TEMPLATE");
	const std::string saved_value = saved ? saved : "";
	unsetenv("RELION_QSUB_TEMPLATE");
	JobWindow window;
	window.initialise(PROC_3DAUTO, false);
	CHECK(std::string(window.guientries["qsubscript"].inp->value()).empty());
	CHECK_FALSE(window.guientries["qsub"].inp->active());
	CHECK_FALSE(window.guientries["min_dedicated"].inp->active());
	window.guientries["qsubscript"].setValue("my_template.sh");   // as when a saved job is loaded
	CHECK(window.guientries["qsub"].inp->active());
	window.guientries["qsubscript"].setValue("");
	CHECK_FALSE(window.guientries["qsub"].inp->active());
	if (saved) setenv("RELION_QSUB_TEMPLATE", saved_value.c_str(), 1);
}
