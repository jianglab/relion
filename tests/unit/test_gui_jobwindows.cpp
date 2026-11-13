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
#include "src/gui_reference_table.h"
#include "src/reference_list.h"
#include "src/pipeline_jobs.h"

#include <FL/Fl_Group.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Scroll.H>
#include <FL/Fl_Slider.H>
#include <FL/Fl_Table.H>
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
	setenv("RELION_GPU_MEMORY_GB", "24", 1); // the suggestion must not depend on the GPU of the test computer
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
		INFO(inp->tooltip());
		// the measured rule: EM Class2D stops getting faster at 4 workers per GPU
		CHECK(std::string(inp->tooltip()).find("MPI processes: 9    Threads: 2") != std::string::npos);
	}
	// chosen by RELION: the greyed-out fields show the numbers it will use
	window.guientries["auto_parallel"].inp->value("Yes");
	window.guientries["nr_cpus"].inp->value("16");
	Fl_Input *cpus = window.guientries["nr_cpus"].inp;
	cpus->handle(FL_ENTER);
	REQUIRE(cpus->tooltip() != NULL);
	CHECK(std::string(cpus->tooltip()).find("RELION splits the 16 CPUs") != std::string::npos);
	// the same measured rule as the hover hint above
	CHECK(std::string(window.guientries["nr_mpi"].inp->value()) == "9");
	CHECK(std::string(window.guientries["nr_threads"].inp->value()) == "2");
	// other jobs keep fields without a tooltip
	JobWindow extract;
	extract.initialise(PROC_EXTRACT, false);
	REQUIRE(extract.guientries.find("nr_mpi") != extract.guientries.end());
	Fl_Input *inp = extract.guientries["nr_mpi"].inp;
	inp->handle(FL_ENTER);
	CHECK(inp->tooltip() == NULL);
	unsetenv("RELION_CPUS_PER_GPU");
	unsetenv("RELION_GPU_MEMORY_GB");
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
		for (const char *name : {"slurm_gpus", "slurm_mem", "slurm_time", "slurm_exclude", "slurm_args"})
			CHECK(window.guientries.find(name) != window.guientries.end());
		// the queue and the nodes to exclude can be chosen from what Slurm lists
		for (const char *name : {"queuename", "slurm_exclude"})
		{
			REQUIRE(window.guientries[name].browse != NULL);
			CHECK(std::string(window.guientries[name].browse->label()) == "Choose");
		}
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
		// a table or scroll area clips its cells: it counts as one widget
		if (dynamic_cast<Fl_Table *>(c) || dynamic_cast<Fl_Scroll *>(c))
			out.push_back({c, where});
		else if (Fl_Group *sub = c->as_group())
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
	// what is drawn: the tab's group can reach below the tabs that hold it
	const Fl_Widget *tabs = tab->parent() ? (const Fl_Widget *)tab->parent() : tab;
	const int bottom = std::min(tab->y() + tab->h(), tabs->y() + tabs->h());
	for (size_t i = 0; i < leaves.size(); i++)
	{
		const Fl_Widget *a = leaves[i].w;
		if (a->x() < tab->x() || a->x() + a->w() > tab->x() + tab->w()
		    || a->y() < tab->y() || a->y() + a->h() > bottom)
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

namespace
{
// As when the user picks a menu item
void pick(GuiEntry &entry, const std::string &value)
{
	entry.setValue(value);
	if (entry.on_pick)
		entry.on_pick();
}

std::vector<std::string> layoutProblemsOfWindow(JobWindow &window)
{
	std::vector<std::string> all;
	for (int k = 0; k < window.tabs->children(); k++)
	{
		Fl_Group *tab = window.tabs->child(k)->as_group();
		if (!tab) continue;
		for (const auto &p : layoutProblems(tab))
			all.push_back(std::string(tab->label() ? tab->label() : "?") + ": " + p);
	}
	return all;
}

reflist::ClassTable decoded(GuiEntry &text)
{
	reflist::ClassTable t;
	std::string error;
	REQUIRE(reflist::decodeTable(text.inp->value(), t, error));
	return t;
}
}

TEST_CASE("GUI: Class3D one reference per class is a table of K classes on the I/O tab", "[gui][layout][reference_list]")
{
	JobWindow window;
	window.initialise(PROC_3DCLASS, false);
	GuiEntry &mode = window.guientries["ref_mode"];
	GuiEntry &classes = window.guientries["nr_classes"];
	GuiEntry &ref = window.guientries["fn_ref"];
	GuiEntry &text = window.guientries["ref_table"];
	REQUIRE(window.reference_table != nullptr);
	ReferenceTable &table = *window.reference_table;

	// The mode and K sit on the I/O tab with the reference; one map, without a '+' button
	const Fl_Group *io = ref.inp->parent();
	CHECK(mode.inp->parent() == io);
	CHECK(classes.inp->parent() == io);
	CHECK(table.parent() == io);
	CHECK(mode.inp->y() < classes.inp->y());
	CHECK(classes.inp->y() < ref.inp->y());
	CHECK(ref.add_button == nullptr);

	// One reference, K classes: the field, no table, the job-wide fields in use
	CHECK(std::string(mode.inp->value()) == job_ref_mode_options[0]);
	CHECK(ref.inp->visible());
	CHECK_FALSE(table.visible());
	for (const char *key : {"sym_name", "helical_twist_initial", "helical_tube_outer_diameter", "fn_mask", "do_auto_mask"})
		CHECK(window.guientries[key].inp->active());

	ref.setValue("x/a.mrc");
	classes.setValue("3");
	pick(mode, job_ref_mode_options[1]);

	SECTION("one reference per class: K classes, the first from the reference field, the job-wide fields greyed out")
	{
		CHECK_FALSE(ref.inp->visible());
		CHECK(table.visible());
		REQUIRE(table.classes() == 3);
		// one column per class, one row per setting
		CHECK(table.cols() == 3);
		CHECK(table.rows() == (int)table.shownLabels().size());
		CHECK(decoded(text).maps == std::vector<std::string>{"x/a.mrc", "", ""});
		for (const char *key : {"sym_name", "helical_twist_initial", "helical_rise_initial", "helical_twist_min",
		                         "helical_twist_inistep", "helical_rise_inistep", "helical_tube_inner_diameter",
		                         "helical_tube_outer_diameter", "fn_mask", "do_auto_mask"})
		{
			INFO(key);
			CHECK_FALSE(window.guientries[key].inp->active());
		}
		// automatic masks of the classes marked "auto" use their resolution
		CHECK(window.guientries["auto_mask_start_res"].inp->active_r());
		// options the table does not hold stay
		CHECK(window.guientries["ini_high"].inp->active());

		// the table sits right under the number of classes, the fields that followed the reference move below it
		GuiEntry &mask = window.guientries["fn_mask"];
		CHECK(table.y() > classes.inp->y());
		CHECK(table.y() < classes.inp->y() + 2 * STEPY);
		CHECK(mask.inp->y() >= table.y() + table.h());
		CHECK(table.h() == table.fullHeight());  // three settings fit
		// no "Fill from STAR file..." button
		for (int i = 0; i < io->children(); i++)
			CHECK(std::string(io->child(i)->label() ? io->child(i)->label() : "").find("Fill") == std::string::npos);

		// back to one reference: as before
		const int mask_y = mask.inp->y();
		pick(mode, job_ref_mode_options[0]);
		CHECK(ref.inp->visible());
		CHECK_FALSE(table.visible());
		CHECK(window.guientries["sym_name"].inp->active());
		CHECK(mask.inp->y() < mask_y);
		CHECK(mask.inp->y() > ref.inp->y());
	}

	SECTION("typed cells go into the job option; rows beyond K are kept while K is smaller")
	{
		Fl_Input *sym = table.cell(1, "rlnSymmetryGroup");
		REQUIRE(sym != nullptr);
		sym->value("C3");
		sym->do_callback();
		Fl_Input *map = table.cell(2, "rlnReferenceImage");
		REQUIRE(map != nullptr);
		map->value("x/c.mrc");
		map->do_callback();
		reflist::ClassTable t = decoded(text);
		CHECK(t.cells[1][t.column("rlnSymmetryGroup")] == "C3");

		classes.setValue("2");
		CHECK(table.classes() == 2);
		CHECK(decoded(text).maps.size() == 2);
		// typed, with the slider following
		classes.inp->value("4");
		classes.inp->do_callback();
		CHECK(table.classes() == 4);
		CHECK(classes.slider->value() == 4);
		t = decoded(text);
		CHECK(t.maps == std::vector<std::string>{"x/a.mrc", "", "x/c.mrc", ""});
		// a new class starts with the settings of the class above it
		CHECK(t.cells[3] == t.cells[2]);
		// half-typed numbers change nothing
		classes.inp->value("");
		classes.inp->do_callback();
		CHECK(table.classes() == 4);
	}

	SECTION("helical columns are shown for helical reconstruction, search ranges with symmetry searches")
	{
		CHECK(table.cell(0, "rlnHelicalTwist") == nullptr);
		pick(window.guientries["do_helix"], "Yes");
		CHECK(table.cell(0, "rlnHelicalTwist") != nullptr);
		CHECK(table.cell(0, "rlnHelicalMaskTubeOuterDiameter") != nullptr);
		CHECK(table.cell(0, "rlnHelicalTwistMin") == nullptr);
		pick(window.guientries["do_local_search_helical_symmetry"], "Yes");
		CHECK(table.cell(2, "rlnHelicalRiseMax") != nullptr);
		CHECK(table.browseButton(2, false) != nullptr);
		CHECK(table.browseButton(2, true) != nullptr);
		// the steps of the symmetry searches, from the job-wide fields
		REQUIRE(table.cell(2, "rlnHelicalTwistInitialStep") != nullptr);
		REQUIRE(table.cell(2, "rlnHelicalRiseInitialStep") != nullptr);
		CHECK(std::string(table.cell(2, "rlnHelicalTwistInitialStep")->value()) ==
		      window.guientries["helical_twist_inistep"].inp->value());
		table.cell(1, "rlnHelicalRiseInitialStep")->value("0.05");
		table.cell(1, "rlnHelicalRiseInitialStep")->do_callback();
		reflist::ClassTable t = decoded(text);
		CHECK(t.cells[1][t.column("rlnHelicalRiseInitialStep")] == "0.05");
		// the Browse button sits in the cell of its map, beside the input
		const Fl_Widget *map = table.cell(2, "rlnReferenceImage"), *button = table.browseButton(2, false);
		CHECK(button->y() == map->y());
		CHECK(button->x() == map->x() + map->w());
	}

	SECTION("many classes and settings scroll, with the scroll bars and with the mouse wheel")
	{
		pick(window.guientries["do_helix"], "Yes");
		pick(window.guientries["do_local_search_helical_symmetry"], "Yes");
		classes.setValue("12");
		REQUIRE(table.classes() == 12);
		// the table stays inside the tab: the settings that do not fit scroll
		CHECK(table.h() < table.fullHeight());
		Fl_Input *last = table.cell(11, "rlnReferenceImage");
		REQUIRE(last != nullptr);
		const int x0 = last->x();
		CHECK(x0 > table.x() + table.w());  // out of sight
		// the wheel scrolls the settings up and down while some are hidden
		Fl_Input *step = table.cell(0, "rlnHelicalRiseInitialStep");
		const int y0 = step->y();
		Fl::e_x = table.x() + table.w() / 2;
		Fl::e_y = table.y() + table.h() / 2;
		Fl::e_dx = 0;
		Fl::e_dy = 3;
		CHECK(table.handle(FL_MOUSEWHEEL) == 1);
		CHECK(step->y() < y0);
		CHECK(last->x() == x0);
		// sideways (a horizontal wheel or a touchpad) scrolls the classes, and the cells follow
		Fl::e_dx = 5;
		Fl::e_dy = 0;
		CHECK(table.handle(FL_MOUSEWHEEL) == 1);
		CHECK(last->x() < x0);
		Fl::e_dx = 0;
	}

	SECTION("a saved job comes back with its table")
	{
		table.cell(1, "rlnReferenceImage")->value("x/b.mrc");
		table.cell(1, "rlnReferenceImage")->do_callback();
		window.updateMyJob();
		JobWindow other;
		other.initialise(PROC_3DCLASS, false);
		for (auto &opt : window.myjob.joboptions)
			other.myjob.joboptions[opt.first].value = opt.second.value;
		other.updateMyGui();
		CHECK(std::string(other.guientries["ref_mode"].inp->value()) == job_ref_mode_options[1]);
		REQUIRE(other.reference_table->classes() == 3);
		CHECK(other.reference_table->table().maps == std::vector<std::string>{"x/a.mrc", "x/b.mrc", ""});
		CHECK(other.reference_table->visible());
		CHECK_FALSE(other.guientries["sym_name"].inp->active());
	}

	SECTION("a job of an earlier version with several maps in the field becomes one reference per class")
	{
		JobWindow old;
		old.initialise(PROC_3DCLASS, false);
		old.myjob.joboptions["ref_mode"].value = job_ref_mode_options[0];
		old.myjob.joboptions["ref_table"].value = "";
		old.myjob.joboptions["fn_ref"].value = "x/a.mrc,x/b.mrc";
		old.myjob.joboptions["nr_classes"].value = "1";
		old.updateMyGui();
		CHECK(std::string(old.guientries["ref_mode"].inp->value()) == job_ref_mode_options[1]);
		CHECK(std::string(old.guientries["nr_classes"].inp->value()) == "2");
		CHECK(old.reference_table->table().maps == std::vector<std::string>{"x/a.mrc", "x/b.mrc"});
	}

	SECTION("continuing greys out the table and keeps the per-class fields greyed")
	{
		window.toggle_new_continue(true);
		CHECK_FALSE(table.active());
		CHECK_FALSE(window.guientries["sym_name"].inp->active());
		window.toggle_new_continue(false);
		CHECK(table.active());
		CHECK_FALSE(window.guientries["sym_name"].inp->active());
		CHECK_FALSE(window.guientries["fn_mask"].inp->active());
	}

	SECTION("the tabs have no overlaps or out-of-tab widgets in either mode")
	{
		pick(window.guientries["do_helix"], "Yes");
		pick(window.guientries["do_local_search_helical_symmetry"], "Yes");
		classes.setValue("30");
		for (const std::string &m : job_ref_mode_options)
		{
			pick(mode, m);
			const auto problems = layoutProblemsOfWindow(window);
			std::string all;
			for (const auto &p : problems) all += "\n  " + p;
			INFO("mode " << m << all);
			CHECK(problems.empty());
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
			CHECK(problems.empty());   // every tab reported, not only the first
		}
	}
}

TEST_CASE("GUI: Class2D filament context parameters are greyed out while Off or Helical types", "[gui][layout][filament_context]")
{
	JobWindow window;
	window.initialise(PROC_2DCLASS, false);
	REQUIRE(window.guientries.find("filctx_mode") != window.guientries.end());
	REQUIRE(window.group_filctx != nullptr);
	for (const char *name : {"filctx_temper", "filctx_gap", "filctx_floor", "filctx_alpha"})
		REQUIRE(window.guientries.find(name) != window.guientries.end());

	REQUIRE(std::string(window.guientries["filctx_mode"].inp->value()) == job_filament_context_options[0]);
	REQUIRE_FALSE(window.group_filctx->active());

	SECTION("loading a job that uses a mode activates them")
	{
		window.guientries["filctx_mode"].setValue(job_filament_context_options[1]);
		REQUIRE(window.group_filctx->active());
		window.guientries["filctx_mode"].setValue(job_filament_context_options[2]);
		REQUIRE(window.group_filctx->active());
		window.guientries["filctx_mode"].setValue(job_filament_context_options[0]);
		REQUIRE_FALSE(window.group_filctx->active());
		window.guientries["filctx_mode"].setValue(job_filament_context_options_2d[3]);
		REQUIRE_FALSE(window.group_filctx->active());
	}
}

// Pick a menu item as the user does (the menu's callback), not as loading a job does (setValue)
static void pickMenu(GuiEntry &entry, const char *item)
{
	const Fl_Menu_Item *p = entry.menu->find_item(item);
	REQUIRE(p != nullptr);
	entry.menu->picked(p);
	GuiEntry::cb_menu(entry.menu, &entry);
}

TEST_CASE("GUI: Class2D helical segments default to EM with helical types", "[gui][filament_context]")
{
	JobWindow window;
	window.initialise(PROC_2DCLASS, false);
	auto value = [&window](const char *key) { return std::string(window.guientries[key].inp->value()); };
	REQUIRE(value("do_em") == "No");
	REQUIRE(value("filctx_mode") == job_filament_context_options[0]);

	pickMenu(window.guientries["do_helix"], "Yes");
	CHECK(value("do_em") == "Yes");
	CHECK(value("do_grad") == "No");
	CHECK(value("filctx_mode") == job_filament_context_options_2d[3]);

	SECTION("back to No: VDAM and Off again")
	{
		pickMenu(window.guientries["do_helix"], "No");
		CHECK(value("do_em") == "No");
		CHECK(value("do_grad") == "Yes");
		CHECK(value("filctx_mode") == job_filament_context_options[0]);
	}
	SECTION("back to No after the user chose another mode: the user's choices stay")
	{
		pickMenu(window.guientries["filctx_mode"], job_filament_context_options[1].c_str());
		pickMenu(window.guientries["do_helix"], "No");
		CHECK(value("do_em") == "Yes");
		CHECK(value("filctx_mode") == job_filament_context_options[1]);
	}
	SECTION("loading a job changes nothing else")
	{
		window.guientries["do_helix"].setValue("No");
		window.guientries["do_em"].setValue("No");
		window.guientries["do_grad"].setValue("Yes");
		window.guientries["filctx_mode"].setValue(job_filament_context_options[0]);
		window.guientries["do_helix"].setValue("Yes");
		CHECK(value("do_em") == "No");
		CHECK(value("filctx_mode") == job_filament_context_options[0]);
	}
}

TEST_CASE("GUI: Class3D filament context sits on the Helix tab, greyed out while Off", "[gui][layout][filament_context]")
{
	JobWindow window;
	window.initialise(PROC_3DCLASS, false);
	for (const char *name : {"filctx_mode", "filctx_temper", "filctx_gap", "filctx_floor", "filctx_alpha"})
		REQUIRE(window.guientries.find(name) != window.guientries.end());
	REQUIRE(window.group_filctx != nullptr);

	auto tabWithLabel = [&](const std::string &label) -> Fl_Group * {
		for (int k = 0; k < window.tabs->children(); k++)
		{
			Fl_Group *tab = window.tabs->child(k)->as_group();
			if (tab && tab->label() && label == tab->label()) return tab;
		}
		return nullptr;
	};
	Fl_Group *helix = tabWithLabel("Helix");
	REQUIRE(helix != nullptr);

	// All five fields live on the Helix tab and nowhere else
	for (const char *name : {"filctx_mode", "filctx_temper", "filctx_gap", "filctx_floor", "filctx_alpha"})
	{
		const Fl_Widget *w = window.guientries[name].inp;
		REQUIRE(w != nullptr);
		REQUIRE(w->x() >= helix->x());
		REQUIRE(w->x() + w->w() <= helix->x() + helix->w());
		REQUIRE(w->y() >= helix->y());
		REQUIRE(w->y() + w->h() <= helix->y() + helix->h());
		REQUIRE(w->y() + w->h() <= window.tabs->y() + window.tabs->h());   // and above the bottom of the window
	}

	REQUIRE_FALSE(window.group_filctx->active());

	SECTION("loading a job that uses a mode activates the parameters")
	{
		window.guientries["filctx_mode"].setValue(job_filament_context_options[1]);
		REQUIRE(window.group_filctx->active());
		window.guientries["filctx_mode"].setValue(job_filament_context_options[2]);
		REQUIRE(window.group_filctx->active());
		window.guientries["filctx_mode"].setValue(job_filament_context_options[0]);
		REQUIRE_FALSE(window.group_filctx->active());
	}

	SECTION("the Helix tab has no overlaps or out-of-tab widgets in any state")
	{
		for (const char *helical : {"No", "Yes"})
			for (const std::string &mode : job_filament_context_options)
			{
				window.guientries["do_helix"].setValue(helical);
				window.guientries["filctx_mode"].setValue(mode);
				const auto problems = layoutProblems(helix);
				std::string all;
				for (const auto &p : problems) all += "\n  " + p;
				INFO("do_helix " << helical << ", mode " << mode << all);
				REQUIRE(problems.empty());
			}
	}

	SECTION("the parameters stay active in a mode when helical is toggled off and on")
	{
		window.guientries["do_helix"].setValue("Yes");
		REQUIRE_FALSE(window.group_filctx->active());
		window.guientries["filctx_mode"].setValue(job_filament_context_options[2]);
		window.guientries["do_helix"].setValue("No");
		window.guientries["do_helix"].setValue("Yes");
		REQUIRE(window.group_filctx->active());
	}
}

TEST_CASE("GUI: Class3D coarser-sampling options grey each other out", "[gui][layout][coarse_early]")
{
	JobWindow window;
	window.initialise(PROC_3DCLASS, false);
	GuiEntry &allow = window.guientries["allow_coarser"];
	GuiEntry &early = window.guientries["coarse_early"];
	REQUIRE(allow.menu != nullptr);
	REQUIRE(early.menu != nullptr);

	// Default: coarser sampling early is on, so allowing coarser sampling is greyed out
	REQUIRE(std::string(early.inp->value()) == "Yes");
	REQUIRE(early.menu->active_r());
	REQUIRE_FALSE(allow.menu->active_r());

	SECTION("switching early off frees the other, and allowing coarser sampling greys early out")
	{
		early.setValue("No");
		REQUIRE(allow.menu->active_r());
		allow.setValue("Yes");
		REQUIRE_FALSE(early.menu->active_r());
		allow.setValue("No");
		REQUIRE(early.menu->active_r());
	}

	SECTION("an older job that allowed coarser sampling (early missing, so Yes) keeps that choice")
	{
		allow.setValue("Yes");
		REQUIRE(std::string(early.inp->value()) == "No");
		REQUIRE(allow.menu->active_r());
		REQUIRE_FALSE(early.menu->active_r());
	}
}
