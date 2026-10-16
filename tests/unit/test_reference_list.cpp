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
	REQUIRE(job.joboptions["fn_ref"].multiple_files);
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
