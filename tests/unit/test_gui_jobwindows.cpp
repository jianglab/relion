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
