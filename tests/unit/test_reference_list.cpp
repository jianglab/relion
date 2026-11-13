/*
 * tests/unit/test_reference_list.cpp
 *
 * Several reference maps for a Class3D job: the list in the "Reference map(s)" field,
 * the STAR file that relion_refine reads from it, the checks on the maps, and the
 * command and nodes of the job.
 */

#include <catch2/catch.hpp>

#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "src/image.h"
#include "src/pipeline_jobs.h"
#include "src/reference_list.h"

namespace {

// A scratch directory that is the working directory while it exists (the job and the
// program work relative to the project directory).
class WorkDir {
public:
	WorkDir()
	{
		const char* home = getenv("HOME");
		std::string base = "/tmp";
		struct stat st;
		if (home && stat((std::string(home) + "/temp").c_str(), &st) == 0) base = std::string(home) + "/temp";
		std::string tmpl = base + "/relion_reflist_XXXXXX";
		std::vector<char> buf(tmpl.begin(), tmpl.end());
		buf.push_back('\0');
		const char* p = mkdtemp(buf.data());
		REQUIRE(p != NULL);
		path_ = p;
		char cwd[4096];
		REQUIRE(getcwd(cwd, sizeof(cwd)) != NULL);
		old_ = cwd;
		REQUIRE(chdir(path_.c_str()) == 0);
	}
	~WorkDir()
	{
		if (chdir(old_.c_str()) != 0) { /* best effort */ }
		const std::string cmd = "rm -rf '" + path_ + "'";
		if (system(cmd.c_str()) != 0) { /* best effort */ }
	}
private:
	std::string path_, old_;
};

void makeMap(const std::string& fn, int box, double angpix)
{
	Image<RFLOAT> img(box, box, box);
	img().initZeros();
	img().setXmippOrigin();
	img(0, 0, 0) = 1.;
	img.setSamplingRateInHeader(angpix);
	img.write(fn);
}

std::string readFile(const std::string& fn)
{
	std::ifstream in(fn.c_str());
	return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

} // namespace

TEST_CASE("reference list: split, join and append", "[reference_list]")
{
	using namespace reflist;
	REQUIRE(splitList("").empty());
	REQUIRE(splitList(" , ,").empty());
	REQUIRE(splitList("a.mrc") == std::vector<std::string>{"a.mrc"});
	REQUIRE(splitList(" a.mrc , b/c.mrc,d.mrc,") == std::vector<std::string>{"a.mrc", "b/c.mrc", "d.mrc"});
	REQUIRE(joinList({"a.mrc", "b.mrc"}) == "a.mrc,b.mrc");

	REQUIRE(appendToList("", {"a.mrc"}) == "a.mrc");
	REQUIRE(appendToList("a.mrc", {"b.mrc", "c.mrc"}) == "a.mrc,b.mrc,c.mrc");
	// A map is listed once, however often it is added
	REQUIRE(appendToList("a.mrc,b.mrc", {"b.mrc", "c.mrc", "a.mrc"}) == "a.mrc,b.mrc,c.mrc");
}

TEST_CASE("reference list: a single path means what it always meant", "[reference_list]")
{
	using namespace reflist;
	Resolved r;
	std::string error;

	REQUIRE(resolve("InitialModel/job015/initial_model.mrc", r, error));
	REQUIRE_FALSE(r.is_list);
	REQUIRE_FALSE(r.is_star);
	REQUIRE(r.ref_argument == "InitialModel/job015/initial_model.mrc");
	REQUIRE(r.maps.size() == 1);

	REQUIRE(resolve("None", r, error));
	REQUIRE(r.ref_argument == "None");
	REQUIRE(r.maps.empty());

	REQUIRE(resolve("", r, error));
	REQUIRE(r.ref_argument.empty());

	// A trailing comma does not make a list
	REQUIRE(resolve("a.mrc,", r, error));
	REQUIRE_FALSE(r.is_list);
	REQUIRE(r.ref_argument == "a.mrc");
}

TEST_CASE("reference list: two or more maps become a STAR file with a fixed name", "[reference_list]")
{
	using namespace reflist;
	Resolved r;
	std::string error;

	REQUIRE(resolve("a.mrc, b.mrc ,c.mrc", r, error));
	REQUIRE(r.is_list);
	REQUIRE(r.maps == std::vector<std::string>{"a.mrc", "b.mrc", "c.mrc"});
	REQUIRE(r.ref_argument.rfind("References/refs_", 0) == 0);
	REQUIRE(r.ref_argument.size() > std::string("References/refs_.star").size());

	// Same list, same file; another list, another file; the order is part of the list
	Resolved same, other, reversed;
	REQUIRE(resolve("a.mrc,b.mrc,c.mrc", same, error));
	REQUIRE(same.ref_argument == r.ref_argument);
	REQUIRE(resolve("a.mrc,b.mrc,d.mrc", other, error));
	REQUIRE(other.ref_argument != r.ref_argument);
	REQUIRE(resolve("c.mrc,b.mrc,a.mrc", reversed, error));
	REQUIRE(reversed.ref_argument != r.ref_argument);
}

TEST_CASE("reference list: bad lists are refused", "[reference_list]")
{
	using namespace reflist;
	Resolved r;
	std::string error;
	REQUIRE_FALSE(resolve("a.mrc,None", r, error));
	REQUIRE_FALSE(error.empty());
	REQUIRE_FALSE(resolve(" , ", r, error));
	REQUIRE_FALSE(resolve("a.mrc,old_model.star", r, error));
	REQUIRE(error.find("old_model.star") != std::string::npos);
}

TEST_CASE("reference list: the STAR file is written once and read back", "[reference_list]")
{
	using namespace reflist;
	WorkDir work;
	const std::vector<std::string> maps = {"x/a.mrc", "x/b.mrc", "y/c.mrc"};
	const std::string fn = starNameFor(maps);
	std::string error;

	REQUIRE(writeStar(fn, maps, error));
	const std::string text = readFile(fn);
	REQUIRE(text.find("_rlnReferenceImage") != std::string::npos);

	// Writing it again leaves the file alone
	struct stat before, after;
	REQUIRE(stat(fn.c_str(), &before) == 0);
	usleep(20000);
	REQUIRE(writeStar(fn, maps, error));
	REQUIRE(stat(fn.c_str(), &after) == 0);
	REQUIRE(before.st_mtim.tv_sec == after.st_mtim.tv_sec);
	REQUIRE(before.st_mtim.tv_nsec == after.st_mtim.tv_nsec);
	REQUIRE(readFile(fn) == text);

	std::vector<std::string> back;
	REQUIRE(readStar(fn, back, error));
	REQUIRE(back == maps);

	// The field can name the STAR file instead of the maps
	Resolved r;
	REQUIRE(resolve(fn, r, error));
	REQUIRE(r.is_star);
	REQUIRE(r.star_readable);
	REQUIRE(r.maps == maps);
	REQUIRE(r.ref_argument == fn);
}

TEST_CASE("reference list: a STAR file that is not there yet is passed on unread", "[reference_list]")
{
	using namespace reflist;
	WorkDir work;
	Resolved r;
	std::string error;
	REQUIRE(resolve("Class3D/job010/run_it025_model.star", r, error));
	REQUIRE(r.is_star);
	REQUIRE_FALSE(r.star_readable);
	REQUIRE(r.ref_argument == "Class3D/job010/run_it025_model.star");

	std::vector<std::string> maps;
	REQUIRE_FALSE(readStar("Class3D/job010/run_it025_model.star", maps, error));
}

TEST_CASE("reference list: the maps are checked before the job starts", "[reference_list]")
{
	using namespace reflist;
	WorkDir work;
	makeMap("a.mrc", 8, 3.0);
	makeMap("b.mrc", 8, 3.0);
	makeMap("big.mrc", 10, 3.0);
	makeMap("fine.mrc", 8, 2.0);
	std::string error;

	REQUIRE(checkMaps({"a.mrc", "b.mrc"}, error));

	REQUIRE_FALSE(checkMaps({"a.mrc", "missing.mrc"}, error));
	REQUIRE(error.find("missing.mrc") != std::string::npos);

	REQUIRE_FALSE(checkMaps({"a.mrc", "big.mrc"}, error));
	REQUIRE(error.find("box size") != std::string::npos);
	REQUIRE(error.find("big.mrc") != std::string::npos);

	REQUIRE_FALSE(checkMaps({"a.mrc", "b.mrc", "fine.mrc"}, error));
	REQUIRE(error.find("pixel size") != std::string::npos);
	REQUIRE(error.find("fine.mrc") != std::string::npos);
}

namespace {

void setUpClass3D(RelionJob& job, const std::string& ref, int k)
{
	job.clear();
	job.initialise(PROC_3DCLASS);
	job.label = get_proc_label(job.type);
	job.joboptions["fn_img"].setString("particles.star");
	job.joboptions["fn_ref"].setString(ref);
	job.joboptions["nr_classes"].setString(std::to_string(k));
	job.joboptions["nr_mpi"].setString("1");
	job.joboptions["nr_threads"].setString("1");
	job.joboptions["do_queue"].setString("No");
	job.joboptions["scratch_dir"].setString("");
}

int countInput(const RelionJob& job, const std::string& name)
{
	int n = 0;
	for (const Node& node : job.inputNodes)
		if (node.name == name) n++;
	return n;
}

bool commandOf(RelionJob& job, bool make, std::string& command, std::string& error)
{
	std::string outputname, final_command;
	std::vector<std::string> commands;
	const bool ok = job.getCommands(outputname, commands, final_command, make, 1, error);
	if (ok && !commands.empty()) command = commands[0];
	return ok;
}

} // namespace

TEST_CASE("Class3D: one reference map keeps the command as it was", "[pipeline][class3d][reference_list]")
{
	RelionJob job;
	setUpClass3D(job, "ref.mrc", 4);
	std::string command, error;
	REQUIRE(commandOf(job, false, command, error));
	REQUIRE(command.find(" --ref ref.mrc") != std::string::npos);
	REQUIRE(command.find(" --K 4") != std::string::npos);
	REQUIRE(countInput(job, "ref.mrc") == 1);
	// several maps are chosen in the table of "One reference per class"
	REQUIRE_FALSE(job.joboptions["fn_ref"].multiple_files);
	REQUIRE(job.joboptions["ref_mode"].getString() == job_ref_mode_options[0]);
}

TEST_CASE("Class3D: several reference maps make one class each", "[pipeline][class3d][reference_list]")
{
	WorkDir work;
	makeMap("a.mrc", 8, 3.0);
	makeMap("b.mrc", 8, 3.0);
	makeMap("c.mrc", 8, 3.0);

	RelionJob job;
	setUpClass3D(job, "a.mrc,b.mrc,c.mrc", 1);
	std::string command, error;
	REQUIRE(commandOf(job, true, command, error));

	reflist::Resolved r;
	REQUIRE(reflist::resolve("a.mrc,b.mrc,c.mrc", r, error));
	REQUIRE(command.find(" --ref " + r.ref_argument) != std::string::npos);
	// K says 1, the three maps decide
	REQUIRE(command.find(" --K 3") != std::string::npos);
	REQUIRE(command.find(" --K 1") == std::string::npos);

	std::vector<std::string> back;
	REQUIRE(reflist::readStar(r.ref_argument, back, error));
	REQUIRE(back == std::vector<std::string>{"a.mrc", "b.mrc", "c.mrc"});

	// One input node per map, none for the STAR file
	REQUIRE(countInput(job, "a.mrc") == 1);
	REQUIRE(countInput(job, "b.mrc") == 1);
	REQUIRE(countInput(job, "c.mrc") == 1);
	REQUIRE(countInput(job, r.ref_argument) == 0);

	// The output has a node for every class
	int class_maps = 0;
	for (const Node& n : job.outputNodes)
		if (n.name.find("_class00") != std::string::npos && n.name.find(".mrc") != std::string::npos) class_maps++;
	REQUIRE(class_maps == 3);

	// Printing the command, without running, writes nothing
	WorkDir other;
	RelionJob dry;
	setUpClass3D(dry, "a.mrc,b.mrc", 1);
	REQUIRE(commandOf(dry, false, command, error));
	struct stat st;
	REQUIRE(stat("References", &st) != 0);
}

TEST_CASE("Class3D: maps that do not fit each other stop the job before it starts", "[pipeline][class3d][reference_list]")
{
	WorkDir work;
	makeMap("a.mrc", 8, 3.0);
	makeMap("big.mrc", 10, 3.0);

	RelionJob job;
	setUpClass3D(job, "a.mrc,big.mrc", 2);
	std::string command, error;
	REQUIRE_FALSE(commandOf(job, true, command, error));
	REQUIRE(error.find("box size") != std::string::npos);

	RelionJob job2;
	setUpClass3D(job2, "a.mrc,nope.mrc", 2);
	REQUIRE_FALSE(commandOf(job2, true, command, error));
	REQUIRE(error.find("nope.mrc") != std::string::npos);

	RelionJob job3;
	setUpClass3D(job3, "a.mrc,None", 2);
	REQUIRE_FALSE(commandOf(job3, false, command, error));
}

TEST_CASE("Class3D: a STAR file of maps in the field gives a class per row", "[pipeline][class3d][reference_list]")
{
	WorkDir work;
	std::string error;
	REQUIRE(reflist::writeStar("my_maps.star", {"a.mrc", "b.mrc"}, error));

	RelionJob job;
	setUpClass3D(job, "my_maps.star", 7);
	std::string command;
	REQUIRE(commandOf(job, false, command, error));
	REQUIRE(command.find(" --ref my_maps.star") != std::string::npos);
	REQUIRE(command.find(" --K 2") != std::string::npos);
	REQUIRE(countInput(job, "a.mrc") == 1);
	REQUIRE(countInput(job, "b.mrc") == 1);
	REQUIRE(countInput(job, "my_maps.star") == 0);
}

TEST_CASE("Class3D: the list survives the saved job", "[pipeline][class3d][reference_list]")
{
	WorkDir work;
	RelionJob job;
	setUpClass3D(job, "x/a.mrc,x/b.mrc", 2);
	REQUIRE(system("mkdir -p saved") == 0);
	job.write("saved/");
	RelionJob again;
	again.clear();
	bool is_continue = false;
	REQUIRE(again.read("saved/", is_continue, true));
	REQUIRE(again.joboptions["fn_ref"].getString() == "x/a.mrc,x/b.mrc");
}

TEST_CASE("reference list: per-class table is completed, trimmed, written and read back", "[reference_list]")
{
	using namespace reflist;
	WorkDir work;
	const std::vector<std::string> maps = {"a.mrc", "b.mrc", "c.mrc"};
	const std::map<std::string, std::string> defaults = {
		{"rlnSymmetryGroup", "C1"}, {"rlnSolventMaskName", "None"},
		{"rlnHelicalTwist", "-1.1"}, {"rlnHelicalMaskTubeOuterDiameter", "200"}};
	const std::vector<std::string> labels = {"rlnSymmetryGroup", "rlnSolventMaskName", "rlnHelicalTwist", "rlnHelicalMaskTubeOuterDiameter"};

	// Nothing given: every cell is the job's value, and no column is per class
	ClassTable t = completeTable(ClassTable(), maps, labels, defaults);
	REQUIRE(t.cells.size() == 3);
	REQUIRE(t.cells[2][2] == "-1.1");
	ClassTable trimmed = t;
	dropDefaultColumns(trimmed, defaults);
	REQUIRE(trimmed.labels.empty());

	// Two columns differ per class; "-1.10" equals the default as a number
	t.cells[0][2] = "-1.13"; t.cells[1][2] = "-0.89"; t.cells[2][2] = "-1.10";
	t.cells[1][3] = "155";
	t.cells[0][1] = "auto";
	dropDefaultColumns(t, defaults);
	REQUIRE(t.labels == std::vector<std::string>{"rlnSolventMaskName", "rlnHelicalTwist", "rlnHelicalMaskTubeOuterDiameter"});
	std::string error;
	REQUIRE(checkTable(t, error));

	const std::string fn = tableStarNameFor(t);
	REQUIRE(fn.rfind("References/refs_", 0) == 0);
	REQUIRE(writeTable(fn, t, error));
	ClassTable back;
	REQUIRE(readTable(fn, back, error));
	REQUIRE(back.maps == maps);
	REQUIRE(back.labels == t.labels);
	REQUIRE(back.cells[0][0] == "auto");
	REQUIRE(back.cells[1][2] == "155");
	Resolved r;
	REQUIRE(resolve(fn, r, error));
	REQUIRE(r.maps == maps);

	// Reopened with a map added: rows are matched by map, the new one gets the defaults
	ClassTable again = completeTable(back, {"a.mrc", "b.mrc", "c.mrc", "d.mrc"}, labels, defaults);
	REQUIRE(again.cells[1][again.column("rlnHelicalTwist")] == "-0.89");
	REQUIRE(again.cells[3][again.column("rlnHelicalTwist")] == "-1.1");
	REQUIRE(again.cells[3][again.column("rlnSymmetryGroup")] == "C1");
}

TEST_CASE("reference list: bad per-class tables are refused", "[reference_list]")
{
	using namespace reflist;
	ClassTable t;
	t.maps = {"a.mrc"};
	std::string error;

	t.labels = {"rlnHelicalTwist"}; t.cells = {{"abc"}};
	REQUIRE_FALSE(checkTable(t, error));
	t.labels = {"rlnSolventMaskName"}; t.cells = {{"no_such_mask.mrc"}};
	REQUIRE_FALSE(checkTable(t, error));
	REQUIRE(error.find("no_such_mask.mrc") != std::string::npos);
	t.labels = {"rlnHelicalTwistMin"}; t.cells = {{"-1.2"}};
	REQUIRE_FALSE(checkTable(t, error));
	t.labels = {"rlnHelicalTwistMin", "rlnHelicalTwist", "rlnHelicalTwistMax"}; t.cells = {{"-1.2", "-0.9", "-1.0"}};
	REQUIRE_FALSE(checkTable(t, error));
	REQUIRE(error == "class 1: Twist lies outside Twist min to Twist max");  // the table's headings, not STAR labels
	t.cells = {{"-1.2", "-1.1", "-1.0"}};
	REQUIRE(checkTable(t, error));
	t.labels = {"rlnHelicalTwistInitialStep"}; t.cells = {{"-0.1"}};
	REQUIRE_FALSE(checkTable(t, error));
	REQUIRE(error.find("Twist step") != std::string::npos);
	t.cells = {{"0"}};
	REQUIRE(checkTable(t, error));
}

TEST_CASE("reference list: per-class search steps", "[reference_list]")
{
	using namespace reflist;
	ClassTable t;
	t.maps = {"a.mrc", "b.mrc"};
	t.labels = {"rlnHelicalTwist", "rlnHelicalTwistMin", "rlnHelicalTwistMax", "rlnHelicalTwistInitialStep",
	            "rlnHelicalRise", "rlnHelicalRiseMin", "rlnHelicalRiseMax", "rlnHelicalRiseInitialStep"};
	t.cells = {{"-1", "-1.5", "-0.5", "0.1", "4.7", "4.6", "4.8", "0.02"},
	           {"-1", "-1.5", "-0.5", "0.05", "4.7", "4.6", "4.8", "0.02"}};
	PerClassRefs refs;
	std::string error;
	// with symmetry searches: a step that differs stays per class, the job gets the finest; one that does not is job-wide
	REQUIRE(perClassRefs(encodeTable(t), 2, true, true, refs, error));
	REQUIRE(refs.table.column("rlnHelicalTwistInitialStep") >= 0);
	REQUIRE(refs.job["helical_twist_inistep"] == "0.05");
	REQUIRE(refs.table.column("rlnHelicalRiseInitialStep") < 0);
	REQUIRE(refs.job["helical_rise_inistep"] == "0.02");
	// without, the steps are not used
	REQUIRE(perClassRefs(encodeTable(t), 2, true, false, refs, error));
	REQUIRE(refs.table.column("rlnHelicalTwistInitialStep") < 0);
	REQUIRE(refs.job.count("helical_twist_inistep") == 0);
	// a new class gets the job's step
	const std::vector<std::string> labels = {"rlnHelicalTwistInitialStep"};
	ClassTable fresh = resizeTable(ClassTable(), 1, labels, {{"rlnHelicalTwistInitialStep", "0"}});
	REQUIRE(fresh.cells[0] == std::vector<std::string>{"0"});
}

TEST_CASE("Class3D: per-class symmetry needs C1 for the job", "[pipeline][class3d][reference_list]")
{
	WorkDir work;
	reflist::ClassTable t;
	t.maps = {"a.mrc", "b.mrc"};
	t.labels = {"rlnSymmetryGroup"};
	t.cells = {{"C3"}, {"C1"}};
	std::string error, command;
	REQUIRE(reflist::writeTable("per_class.star", t, error));

	RelionJob job;
	setUpClass3D(job, "per_class.star", 2);
	job.joboptions["sym_name"].setString("C2");
	REQUIRE_FALSE(commandOf(job, false, command, error));
	REQUIRE(error.find("C1") != std::string::npos);
	job.joboptions["sym_name"].setString("C1");
	REQUIRE(commandOf(job, false, command, error));
	REQUIRE(command.find(" --ref per_class.star") != std::string::npos);
	REQUIRE(command.find(" --K 2") != std::string::npos);
}

TEST_CASE("reference list: the per-class table survives its text", "[reference_list]")
{
	using namespace reflist;
	ClassTable t;
	t.maps = {"Refine3D/job010/run_class001.mrc", ""};
	t.labels = {"rlnSymmetryGroup", "rlnSolventMaskName", "rlnHelicalTwist"};
	t.cells = {{"C2", "None", "-1.1"}, {"C1", "", "  0.5 "}};
	const std::string text = encodeTable(t);
	REQUIRE(text == "rlnReferenceImage|rlnSymmetryGroup|rlnSolventMaskName|rlnHelicalTwist;"
	                "Refine3D/job010/run_class001.mrc|C2|None|-1.1;|C1||0.5");
	ClassTable back;
	std::string error;
	REQUIRE(decodeTable(text, back, error));
	REQUIRE(back.maps == t.maps);
	REQUIRE(back.labels == t.labels);
	REQUIRE(back.cells[1] == std::vector<std::string>{"C1", "", "0.5"});

	// the separators cannot get into a cell
	t.cells[0][0] = "C|2;";
	REQUIRE(decodeTable(encodeTable(t), back, error));
	REQUIRE(back.cells[0][0] == "C2");

	REQUIRE(encodeTable(ClassTable()).empty());
	REQUIRE(decodeTable("", back, error));
	REQUIRE(back.maps.empty());
	REQUIRE_FALSE(decodeTable("rlnSymmetryGroup;C1", back, error));
	REQUIRE_FALSE(decodeTable("rlnReferenceImage|rlnSymmetryGroup;a.mrc", back, error));
	REQUIRE(error.find("row 1") != std::string::npos);
}

TEST_CASE("reference list: the table follows the number of classes", "[reference_list]")
{
	using namespace reflist;
	const std::vector<std::string> labels = {"rlnSymmetryGroup", "rlnHelicalTwist", "rlnHelicalTwistMin", "rlnHelicalTwistMax"};
	const std::map<std::string, std::string> defaults = {{"rlnSymmetryGroup", "C1"}, {"rlnHelicalTwist", "-1"},
		{"rlnHelicalTwistMin", "-1.5"}, {"rlnHelicalTwistMax", "-0.5"}};

	// from nothing: the first class has the job's values, the others copy the row above
	ClassTable t = resizeTable(ClassTable(), 2, labels, defaults);
	REQUIRE(t.maps == std::vector<std::string>{"", ""});
	REQUIRE(t.cells[0] == std::vector<std::string>{"C1", "-1", "-1.5", "-0.5"});
	t.maps = {"a.mrc", "b.mrc"};
	t.cells[1] = {"C3", "-2", "-2.2", "-1.8"};
	ClassTable more = resizeTable(t, 4, labels, defaults);
	REQUIRE(more.maps == std::vector<std::string>{"a.mrc", "b.mrc", "", ""});
	REQUIRE(more.cells[3] == t.cells[1]);
	// fewer: the first rows stay
	ClassTable fewer = resizeTable(more, 1, labels, defaults);
	REQUIRE(fewer.maps == std::vector<std::string>{"a.mrc"});
	REQUIRE(fewer.cells[0] == t.cells[0]);

	// a table without the search ranges gets the job's range around each class's twist
	ClassTable no_range;
	no_range.maps = {"a.mrc"};
	no_range.labels = {"rlnHelicalTwist"};
	no_range.cells = {{"-3"}};
	ClassTable ranged = resizeTable(no_range, 1, labels, defaults);
	REQUIRE(ranged.cells[0] == std::vector<std::string>{"C1", "-3", "-3.5", "-2.5"});
}

TEST_CASE("reference list: a per-class table becomes the --ref file and job-wide values", "[reference_list]")
{
	using namespace reflist;
	WorkDir work;
	{ std::ofstream m("mask_a.mrc"); m << "x"; }
	ClassTable t;
	t.maps = {"a.mrc", "b.mrc", "a.mrc"};
	t.labels = {"rlnSymmetryGroup", "rlnSolventMaskName", "rlnHelicalTwist", "rlnHelicalRise",
	            "rlnHelicalTwistMin", "rlnHelicalTwistMax", "rlnHelicalRiseMin", "rlnHelicalRiseMax",
	            "rlnHelicalMaskTubeInnerDiameter", "rlnHelicalMaskTubeOuterDiameter"};
	t.cells = {{"C1", "None", "-1.1", "4.75", "-1.3", "-0.9", "4.7", "4.8", "-1", "180"},
	           {"C2", "None", "-1.1", "4.75", "-1.5", "-0.9", "4.7", "4.8", "-1", "200"},
	           {"C1", "None", "-0.8", "4.75", "-1.3", "-0.5", "4.7", "4.8", "40", "180"}};
	PerClassRefs refs;
	std::string error;

	// not helical: only symmetry and mask count
	REQUIRE(perClassRefs(encodeTable(t), 3, false, false, refs, error));
	REQUIRE(refs.table.maps == t.maps);
	REQUIRE(refs.table.labels == std::vector<std::string>{"rlnSymmetryGroup"});
	REQUIRE(refs.job["sym_name"] == "C1");  // they differ: searched in C1
	REQUIRE(refs.job["fn_mask"] == "");
	REQUIRE(refs.job["do_auto_mask"] == "No");
	REQUIRE(refs.job.count("helical_twist_initial") == 0);

	// helical, with symmetry searches: what differs stays per class, the job-wide values cover all classes
	REQUIRE(perClassRefs(encodeTable(t), 3, true, true, refs, error));
	REQUIRE(refs.table.labels == std::vector<std::string>{"rlnSymmetryGroup", "rlnHelicalTwist", "rlnHelicalTwistMin",
	                                                       "rlnHelicalTwistMax", "rlnHelicalMaskTubeInnerDiameter", "rlnHelicalMaskTubeOuterDiameter"});
	REQUIRE(refs.job["helical_rise_initial"] == "4.75");
	REQUIRE(refs.job["helical_rise_min"] == "4.7");
	REQUIRE(refs.job["helical_twist_initial"] == "-1.1");
	REQUIRE(refs.job["helical_twist_min"] == "-1.5");
	REQUIRE(refs.job["helical_twist_max"] == "-0.5");
	REQUIRE(refs.job["helical_tube_outer_diameter"] == "200");
	REQUIRE(refs.job["helical_tube_inner_diameter"] == "-1");
	// without symmetry searches the ranges are not used
	REQUIRE(perClassRefs(encodeTable(t), 3, true, false, refs, error));
	REQUIRE(refs.table.column("rlnHelicalTwistMin") < 0);
	REQUIRE(refs.job.count("helical_twist_min") == 0);

	// masks: one for all is the reference mask, auto everywhere the automatic masks, a mix stays per class
	for (auto &row : t.cells) row[1] = "mask_a.mrc";
	REQUIRE(perClassRefs(encodeTable(t), 3, false, false, refs, error));
	REQUIRE(refs.job["fn_mask"] == "mask_a.mrc");
	REQUIRE(refs.table.column("rlnSolventMaskName") < 0);
	for (auto &row : t.cells) row[1] = "auto";
	REQUIRE(perClassRefs(encodeTable(t), 3, false, false, refs, error));
	REQUIRE(refs.job["fn_mask"] == "");
	REQUIRE(refs.job["do_auto_mask"] == "Yes");
	t.cells[1][1] = "mask_a.mrc";
	t.cells[2][1] = "None";
	REQUIRE(perClassRefs(encodeTable(t), 3, false, false, refs, error));
	REQUIRE(refs.table.column("rlnSolventMaskName") >= 0);
	REQUIRE(refs.job["fn_mask"] == "");
	REQUIRE(refs.job["do_auto_mask"] == "Yes");
	REQUIRE(refs.mask_files == std::vector<std::string>{"mask_a.mrc"});

	// refused: wrong number of rows, a class without a map, a bad cell
	REQUIRE_FALSE(perClassRefs(encodeTable(t), 2, false, false, refs, error));
	REQUIRE(error.find("3 rows for 2 classes") != std::string::npos);
	REQUIRE_FALSE(perClassRefs("", 1, false, false, refs, error));
	ClassTable bad = t;
	bad.maps[1] = "";
	REQUIRE_FALSE(perClassRefs(encodeTable(bad), 3, false, false, refs, error));
	REQUIRE(error.find("class 2") != std::string::npos);
	bad = t;
	bad.cells[0][2] = "abc";
	REQUIRE_FALSE(perClassRefs(encodeTable(bad), 3, true, false, refs, error));
	REQUIRE(error.find("Twist") != std::string::npos);
}

TEST_CASE("Class3D: one reference per class makes the --ref file from the table", "[pipeline][class3d][reference_list]")
{
	WorkDir work;
	makeMap("a.mrc", 8, 3.0);
	makeMap("b.mrc", 8, 3.0);
	reflist::ClassTable t;
	t.maps = {"a.mrc", "b.mrc", "b.mrc"};
	t.labels = {"rlnSymmetryGroup", "rlnSolventMaskName", "rlnHelicalTwist", "rlnHelicalRise",
	            "rlnHelicalMaskTubeInnerDiameter", "rlnHelicalMaskTubeOuterDiameter"};
	t.cells = {{"C1", "auto", "-1.2", "4.75", "-1", "150"},
	           {"C1", "auto", "-0.9", "4.75", "-1", "180"},
	           {"C1", "auto", "1.1", "4.75", "-1", "180"}};

	RelionJob job;
	setUpClass3D(job, "ignored.mrc", 3);
	job.joboptions["ref_mode"].setString(job_ref_mode_options[1]);
	job.joboptions["ref_table"].setString(reflist::encodeTable(t));
	job.joboptions["do_helix"].setString("Yes");
	job.joboptions["helical_tube_outer_diameter"].setString("100");  // greyed out: the table decides
	job.joboptions["sym_name"].setString("D7");
	std::string command, error;
	REQUIRE(commandOf(job, true, command, error));
	INFO(command);

	reflist::PerClassRefs refs;
	REQUIRE(reflist::perClassRefs(reflist::encodeTable(t), 3, true, false, refs, error));
	const std::string fn = reflist::tableStarNameFor(refs.table);
	CHECK(command.find(" --ref " + fn) != std::string::npos);
	CHECK(command.find(" --K 3") != std::string::npos);
	CHECK(command.find(" --sym C1") != std::string::npos);
	CHECK(command.find(" --auto_mask") != std::string::npos);
	CHECK(command.find(" --helical_outer_diameter 180") != std::string::npos);
	CHECK(command.find(" --helical_twist_initial -1.2") != std::string::npos);
	CHECK(command.find(" --helical_rise_initial 4.75") != std::string::npos);
	CHECK(command.find("ignored.mrc") == std::string::npos);
	// the file holds what differs between the classes
	reflist::ClassTable written;
	REQUIRE(reflist::readTable(fn, written, error));
	CHECK(written.maps == t.maps);
	CHECK(written.labels == std::vector<std::string>{"rlnHelicalTwist", "rlnHelicalMaskTubeOuterDiameter"});
	// one input node per map
	CHECK(countInput(job, "a.mrc") == 1);
	CHECK(countInput(job, "b.mrc") == 1);
	CHECK(countInput(job, "ignored.mrc") == 0);
	int class_maps = 0;
	for (const Node& n : job.outputNodes)
		if (n.name.find("_class00") != std::string::npos) class_maps++;
	CHECK(class_maps == 3);

	// the number of classes and the table must agree
	job.joboptions["nr_classes"].setString("4");
	REQUIRE_FALSE(commandOf(job, false, command, error));
	CHECK(error.find("3 rows for 4 classes") != std::string::npos);

	// maps that do not fit each other stop the job before it starts
	makeMap("big.mrc", 10, 3.0);
	t.maps[2] = "big.mrc";
	job.joboptions["nr_classes"].setString("3");
	job.joboptions["ref_table"].setString(reflist::encodeTable(t));
	REQUIRE_FALSE(commandOf(job, true, command, error));
	CHECK(error.find("box size") != std::string::npos);

	// the mode and the table survive the saved job
	REQUIRE(system("mkdir -p saved") == 0);
	job.write("saved/");
	RelionJob again;
	again.clear();
	bool is_continue = false;
	REQUIRE(again.read("saved/", is_continue, true));
	CHECK(again.joboptions["ref_mode"].getString() == job_ref_mode_options[1]);
	CHECK(again.joboptions["ref_table"].getString() == reflist::encodeTable(t));
}

TEST_CASE("reference list: one map starting two classes keeps each row's settings", "[reference_list]")
{
	using namespace reflist;
	ClassTable given;
	given.maps = {"a.mrc", "a.mrc"};
	given.labels = {"rlnHelicalTwist", "rlnClassDistribution"};
	given.cells = {{"-1.1", "0.5"}, {"1.1", "0.5"}};
	const ClassTable t = completeTable(given, given.maps, {"rlnHelicalTwist"}, {{"rlnHelicalTwist", "0"}});
	REQUIRE(t.cells[0][0] == "-1.1");
	REQUIRE(t.cells[1][0] == "1.1");
	// a text column RELION knows is not held to be a number
	ClassTable text;
	text.maps = {"a.mrc"};
	text.labels = {"rlnHelicalTwist", "rlnMicrographName"};
	text.cells = {{"-1.1", "mic.mrc"}};
	std::string error;
	REQUIRE(checkTable(text, error));
}
