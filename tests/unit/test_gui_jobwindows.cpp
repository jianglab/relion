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

#include <FL/Fl_Group.H>
#include <FL/Fl_Tabs.H>
#include <cstdlib>
#include <iostream>
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

namespace
{
struct Leaf { Fl_Widget *w; std::string where; };

void collectLeaves(Fl_Group *g, const std::string &where, std::vector<Leaf> &out)
{
	for (int i = 0; i < g->children(); i++)
	{
		Fl_Widget *c = g->child(i);
		if (!c->visible() || c->w() <= 0 || c->h() <= 0)
			continue;
		if (Fl_Group *sub = c->as_group())
			collectLeaves(sub, where, out);
		else
			out.push_back({c, where});
	}
}

std::string describe(const Fl_Widget *w)
{
	std::string s = w->label() ? w->label() : "";
	if (s.empty() && w->tooltip())
		s = w->tooltip();
	return "[" + s.substr(0, 40) + "] at (" + std::to_string(w->x()) + "," + std::to_string(w->y()) + ") "
	       + std::to_string(w->w()) + "x" + std::to_string(w->h());
}

// Problems on one tab: widgets that partly overlap, or stick out of the tab.
// Widgets with identical rectangles are stacked on purpose (a label under its menu, one help
// button per toggle state), so they do not count.
std::vector<std::string> layoutProblems(Fl_Group *tab)
{
	std::vector<Leaf> leaves;
	collectLeaves(tab, "", leaves);
	std::vector<std::string> problems;
	for (size_t i = 0; i < leaves.size(); i++)
	{
		const Fl_Widget *a = leaves[i].w;
		if (a->x() < tab->x() || a->x() + a->w() > tab->x() + tab->w()
		    || a->y() < tab->y() || a->y() + a->h() > tab->y() + tab->h())
			problems.push_back("outside the tab: " + describe(a));
		for (size_t j = i + 1; j < leaves.size(); j++)
		{
			const Fl_Widget *b = leaves[j].w;
			const bool stacked = a->x() == b->x() && a->y() == b->y() && a->w() == b->w() && a->h() == b->h();
			if (!stacked && a->x() < b->x() + b->w() && b->x() < a->x() + a->w()
			    && a->y() < b->y() + b->h() && b->y() < a->y() + a->h())
				problems.push_back("overlap: " + describe(a) + " with " + describe(b));
		}
	}
	return problems;
}
}

TEST_CASE("GUI: Class3D several reference maps set the number of classes", "[gui][layout][reference_list]")
{
	JobWindow window;
	window.initialise(PROC_3DCLASS, false);
	REQUIRE(window.guientries.find("fn_ref") != window.guientries.end());
	GuiEntry &ref = window.guientries["fn_ref"];
	GuiEntry &classes = window.guientries["nr_classes"];

	// The '+' button sits at the end of the row, right of Browse, inside the tab
	REQUIRE(ref.add_button != nullptr);
	REQUIRE(ref.browse != nullptr);
	REQUIRE(ref.add_button->x() >= ref.browse->x() + ref.browse->w());
	REQUIRE(ref.add_button->y() == ref.browse->y());
	// Other file fields keep their single Browse button
	REQUIRE(window.guientries["fn_img"].add_button == nullptr);

	const std::string one_class = classes.inp->value();
	REQUIRE(classes.inp->active());

	SECTION("a list of maps overrides and greys the field")
	{
		ref.setValue("x/a.mrc,x/b.mrc,x/c.mrc");
		REQUIRE(std::string(classes.inp->value()) == "3");
		REQUIRE_FALSE(classes.inp->active());

		ref.setValue("x/a.mrc,x/b.mrc");
		REQUIRE(std::string(classes.inp->value()) == "2");
	}

	SECTION("a single map gives the field back")
	{
		ref.setValue("x/a.mrc,x/b.mrc");
		REQUIRE_FALSE(classes.inp->active());
		ref.setValue("x/a.mrc");
		REQUIRE(classes.inp->active());
	}

	SECTION("the tabs have no overlaps or out-of-tab widgets in either state")
	{
		for (const char *value : {"x/a.mrc", "x/a.mrc,x/b.mrc,x/c.mrc,x/d.mrc"})
		{
			ref.setValue(value);
			for (int k = 0; k < window.tabs->children(); k++)
			{
				Fl_Group *tab = window.tabs->child(k)->as_group();
				if (!tab) continue;
				const auto problems = layoutProblems(tab);
				std::string all;
				for (const auto &p : problems) all += "\n  " + p;
				INFO("reference " << value << ", tab " << (tab->label() ? tab->label() : "?") << all);
				REQUIRE(problems.empty());
			}
		}
	}
}

TEST_CASE("GUI: no tab of any job window has overlapping or out-of-tab widgets", "[gui][layout]")
{
	for (const auto &t : allGuiJobTypes())
	{
		JobWindow window;
		window.initialise(t.first, false);
		for (int k = 0; k < window.tabs->children(); k++)
		{
			Fl_Group *tab = window.tabs->child(k)->as_group();
			if (!tab) continue;
			INFO("job type " << t.second << ", tab " << (tab->label() ? tab->label() : "?"));
			const auto problems = layoutProblems(tab);
			std::string all;
			for (const auto &p : problems)
				all += "\n  " + p;
			INFO("layout problems:" << all);
			REQUIRE(problems.empty());
		}
	}
}

TEST_CASE("GUI: Class2D filament context parameters are greyed out while Off", "[gui][layout][filament_context]")
{
	JobWindow window;
	window.initialise(PROC_2DCLASS, false);
	REQUIRE(window.guientries.find("filctx_mode") != window.guientries.end());
	REQUIRE(window.group8 != nullptr);
	for (const char *name : {"filctx_temper", "filctx_gap", "filctx_floor", "filctx_alpha"})
		REQUIRE(window.guientries.find(name) != window.guientries.end());

	REQUIRE_FALSE(window.group8->active());

	SECTION("loading a job that uses a mode activates them")
	{
		window.guientries["filctx_mode"].setValue(job_filament_context_options[1]);
		REQUIRE(window.group8->active());
		window.guientries["filctx_mode"].setValue(job_filament_context_options[2]);
		REQUIRE(window.group8->active());
		window.guientries["filctx_mode"].setValue(job_filament_context_options[0]);
		REQUIRE_FALSE(window.group8->active());
	}
}
