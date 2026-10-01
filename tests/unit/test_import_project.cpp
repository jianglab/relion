/*
 * tests/unit/test_import_project.cpp
 *
 * Unit tests for the scope rules of Project > Import RELION project...:
 * which folder means what, which jobs are pulled in as ancestry, and what the
 * reduced pipeline file and the linked job folders look like.
 */

#include <catch2/catch.hpp>

#include "src/import_project.h"
#include "src/metadata_table.h"

#include <climits>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {

class TempDir {
public:
	TempDir()
	{
		const char* home = getenv("HOME");
		std::string base = std::string(home ? home : ".") + "/temp";
		mkdir(base.c_str(), 0755);
		std::string tmpl = base + "/relion_import_XXXXXX";
		std::vector<char> buf(tmpl.begin(), tmpl.end());
		buf.push_back('\0');
		const char* p = mkdtemp(&buf[0]);
		REQUIRE(p != NULL);
		char real[PATH_MAX];
		path_ = (realpath(p, real) != NULL) ? real : p;
	}
	~TempDir()
	{
		const std::string cmd = "rm -rf '" + path_ + "'";
		if (system(cmd.c_str()) != 0) { /* best effort */ }
	}
	const std::string& path() const { return path_; }
private:
	std::string path_;
};

void mkdirs(const std::string& p)
{
	const std::string cmd = "mkdir -p '" + p + "'";
	REQUIRE(system(cmd.c_str()) == 0);
}

bool isLink(const std::string& p)
{
	struct stat st;
	return lstat(p.c_str(), &st) == 0 && S_ISLNK(st.st_mode);
}

bool exists(const std::string& p)
{
	struct stat st;
	return lstat(p.c_str(), &st) == 0;
}

/* A small project:
 *   Import/job001 -> (movies.star)
 *   MotionCorr/job002 reads job001, writes corrected.star
 *   Extract/job003 reads job002, writes particles.star
 *   Class2D/job004 reads job003, writes it25_classes.mrcs
 *   Class2D/job005 reads job003, writes other_classes.mrcs
 *   Select/job006 reads job004
 * Class2D/job007 exists as a folder only (not in the pipeline).
 */
void makeProject(const std::string& root)
{
	mkdirs(root + "/Import/job001");
	mkdirs(root + "/MotionCorr/job002");
	mkdirs(root + "/Extract/job003");
	mkdirs(root + "/Class2D/job004");
	mkdirs(root + "/Class2D/job005");
	mkdirs(root + "/Class2D/job007");
	mkdirs(root + "/Select/job006");
	mkdirs(root + "/Other");
	std::ofstream(root + "/.gui_projectdir") << "";
	// RELION keeps a copy of the pipeline file in every job folder
	std::ofstream(root + "/Class2D/job004/default_pipeline.star") << "x\n";
	std::ofstream(root + "/Import/job001/default_pipeline.star") << "x\n";

	std::ofstream f((root + "/default_pipeline.star").c_str());
	f << "\n# version 50001\n\ndata_pipeline_general\n\n_rlnPipeLineJobCounter 7\n\n\n"
	  << "# version 50001\n\ndata_pipeline_nodes\n\nloop_ \n_rlnPipeLineNodeName #1 \n_rlnPipeLineNodeTypeLabel #2 \n_rlnPipeLineNodeTypeLabelDepth #3 \n"
	  << "Import/job001/movies.star MicrographMovieGroupMetadata.star 1\n"
	  << "MotionCorr/job002/corrected.star MicrographGroupMetadata.star 1\n"
	  << "Extract/job003/particles.star ParticleGroupMetadata.star 1\n"
	  << "Class2D/job004/it25_classes.mrcs ReferenceGroupMetadata.star 1\n"
	  << "Class2D/job005/other_classes.mrcs ReferenceGroupMetadata.star 1\n\n\n"
	  << "# version 50001\n\ndata_pipeline_processes\n\nloop_ \n_rlnPipeLineProcessName #1 \n_rlnPipeLineProcessAlias #2 \n_rlnPipeLineProcessTypeLabel #3 \n_rlnPipeLineProcessStatusLabel #4 \n"
	  << "Import/job001/ None relion.import Succeeded\n"
	  << "MotionCorr/job002/ None relion.motioncorr.own Succeeded\n"
	  << "Extract/job003/ None relion.extract Succeeded\n"
	  << "Class2D/job004/ None relion.class2d Succeeded\n"
	  << "Class2D/job005/ None relion.class2d Succeeded\n"
	  << "Select/job006/ None relion.select.onvalue Succeeded\n\n\n"
	  << "# version 50001\n\ndata_pipeline_input_edges\n\nloop_ \n_rlnPipeLineEdgeFromNode #1 \n_rlnPipeLineEdgeProcess #2 \n"
	  << "Import/job001/movies.star MotionCorr/job002/\n"
	  << "MotionCorr/job002/corrected.star Extract/job003/\n"
	  << "Extract/job003/particles.star Class2D/job004/\n"
	  << "Extract/job003/particles.star Class2D/job005/\n"
	  << "Class2D/job004/it25_classes.mrcs Select/job006/\n\n\n"
	  << "# version 50001\n\ndata_pipeline_output_edges\n\nloop_ \n_rlnPipeLineEdgeProcess #1 \n_rlnPipeLineEdgeToNode #2 \n"
	  << "Import/job001/ Import/job001/movies.star\n"
	  << "MotionCorr/job002/ MotionCorr/job002/corrected.star\n"
	  << "Extract/job003/ Extract/job003/particles.star\n"
	  << "Class2D/job004/ Class2D/job004/it25_classes.mrcs\n"
	  << "Class2D/job005/ Class2D/job005/other_classes.mrcs\n\n";
}

std::set<std::string> processNames(const std::string& pipeline)
{
	std::ifstream in(pipeline.c_str());
	MetaDataTable md;
	md.readStar(in, "pipeline_processes");
	std::set<std::string> out;
	for (size_t i = 0; i < md.numberOfObjects(); i++)
	{
		std::string n;
		md.getValue(EMDL_PIPELINE_PROCESS_NAME, n, i);
		out.insert(n);
	}
	return out;
}

size_t tableSize(const std::string& pipeline, const std::string& table)
{
	std::ifstream in(pipeline.c_str());
	MetaDataTable md;
	md.readStar(in, table);
	return md.numberOfObjects();
}

} // namespace

using namespace relion_import;

TEST_CASE("import_scope: what each folder means", "[import_scope]")
{
	TempDir t;
	const std::string root = t.path() + "/proj";
	makeProject(root);

	SECTION("project folder") {
		Selection s = classify(root);
		CHECK(s.scope == WHOLE_PROJECT);
		CHECK(s.jobs.empty());
	}
	SECTION("job folder") {
		Selection s = classify(root + "/Class2D/job004");
		REQUIRE(s.scope == SINGLE_JOB);
		CHECK(s.project == root);
		REQUIRE(s.jobs.size() == 1);
		CHECK(s.jobs[0] == "Class2D/job004/");
	}
	SECTION("job folder with a trailing slash") {
		CHECK(classify(root + "/Class2D/job004/").scope == SINGLE_JOB);
	}
	SECTION("job type folder") {
		Selection s = classify(root + "/Class2D");
		REQUIRE(s.scope == JOB_TYPE_FOLDER);
		REQUIRE(s.jobs.size() == 3);
		CHECK(s.jobs[0] == "Class2D/job004/");
		CHECK(s.jobs[2] == "Class2D/job007/");
	}
	SECTION("other folders are refused") {
		CHECK(classify(root + "/Other").scope == INVALID);        // no jobs in it
		CHECK(classify(t.path()).scope == INVALID);                // not a project
		CHECK(classify(root + "/Class2D/job004/..").scope == JOB_TYPE_FOLDER);
		CHECK(classify(root + "/does_not_exist").scope == INVALID);
		mkdirs(root + "/Class2D/job004/sub");
		CHECK(classify(root + "/Class2D/job004/sub").scope == INVALID);
	}
	SECTION("a folder that holds jobs but is not directly in a project is refused") {
		mkdirs(root + "/Deep/er/job001");
		CHECK(classify(root + "/Deep/er").scope == INVALID);
	}
	SECTION("Trash is never a job type folder") {
		mkdirs(root + "/Trash/job009");
		CHECK(classify(root + "/Trash").scope == INVALID);
	}
}

TEST_CASE("import_scope: a job brings its ancestry only", "[import_scope]")
{
	TempDir t;
	const std::string root = t.path() + "/proj";
	const std::string dest = t.path() + "/dest";
	makeProject(root);
	mkdirs(dest);

	Result r = importJobs(classify(root + "/Class2D/job004"), dest);
	REQUIRE(r.ok);

	std::set<std::string> want;
	want.insert("Import/job001/");
	want.insert("MotionCorr/job002/");
	want.insert("Extract/job003/");
	want.insert("Class2D/job004/");
	CHECK(processNames(dest + "/default_pipeline.star") == want);
	CHECK(r.linked == 4);
	CHECK(r.missing == 0);

	CHECK(isLink(dest + "/Class2D/job004"));
	CHECK(isLink(dest + "/Import/job001"));
	CHECK_FALSE(exists(dest + "/Class2D/job005"));   // a sibling, not an ancestor
	CHECK_FALSE(exists(dest + "/Select/job006"));    // a descendant, not an ancestor

	// Nodes and edges: only those touching the kept jobs
	CHECK(tableSize(dest + "/default_pipeline.star", "pipeline_nodes") == 4);
	CHECK(tableSize(dest + "/default_pipeline.star", "pipeline_input_edges") == 3);
	CHECK(tableSize(dest + "/default_pipeline.star", "pipeline_output_edges") == 4);

	// The job counter keeps the source value, so new jobs never reuse a number
	std::ifstream in((dest + "/default_pipeline.star").c_str());
	MetaDataTable gen;
	REQUIRE(gen.readStar(in, "pipeline_general"));
	int counter = -1;
	gen.getValue(EMDL_PIPELINE_JOB_COUNTER, counter);
	CHECK(counter == 7);
}

TEST_CASE("import_scope: a job type folder brings all its jobs", "[import_scope]")
{
	TempDir t;
	const std::string root = t.path() + "/proj";
	const std::string dest = t.path() + "/dest";
	makeProject(root);
	mkdirs(dest);

	Result r = importJobs(classify(root + "/Class2D"), dest);
	REQUIRE(r.ok);

	std::set<std::string> want;
	want.insert("Import/job001/");
	want.insert("MotionCorr/job002/");
	want.insert("Extract/job003/");
	want.insert("Class2D/job004/");
	want.insert("Class2D/job005/");
	CHECK(processNames(dest + "/default_pipeline.star") == want);
	CHECK_FALSE(exists(dest + "/Select/job006"));
	// job007 has a folder but is not in the pipeline: reported, not imported
	REQUIRE(r.not_in_pipeline.size() == 1);
	CHECK(r.not_in_pipeline[0] == "Class2D/job007/");
	CHECK_FALSE(exists(dest + "/Class2D/job007"));
}

TEST_CASE("import_scope: a job at the start of a pipeline imports alone", "[import_scope]")
{
	TempDir t;
	const std::string root = t.path() + "/proj";
	const std::string dest = t.path() + "/dest";
	makeProject(root);
	mkdirs(dest);

	Result r = importJobs(classify(root + "/Import/job001"), dest);
	REQUIRE(r.ok);
	CHECK(r.imported.size() == 1);
	CHECK(tableSize(dest + "/default_pipeline.star", "pipeline_nodes") == 1);
}

TEST_CASE("import_scope: repeated import leaves existing links and reports them", "[import_scope]")
{
	TempDir t;
	const std::string root = t.path() + "/proj";
	const std::string dest = t.path() + "/dest";
	makeProject(root);
	mkdirs(dest);

	REQUIRE(importJobs(classify(root + "/Extract/job003"), dest).ok);
	Result again = importJobs(classify(root + "/Class2D/job004"), dest);
	REQUIRE(again.ok);
	CHECK(again.existing == 3);
	CHECK(again.linked == 1);
}

TEST_CASE("import_scope: failures are reported, not hidden", "[import_scope]")
{
	TempDir t;
	const std::string root = t.path() + "/proj";
	makeProject(root);

	SECTION("destination does not exist") {
		CHECK_FALSE(importJobs(classify(root + "/Class2D"), t.path() + "/nope").ok);
	}
	SECTION("whole-project selections are not handled here") {
		mkdirs(t.path() + "/dest");
		CHECK_FALSE(importJobs(classify(root), t.path() + "/dest").ok);
	}
	SECTION("a job folder missing from the source is counted") {
		const std::string cmd = "rm -rf '" + root + "/Import/job001'";
		REQUIRE(system(cmd.c_str()) == 0);
		mkdirs(t.path() + "/dest");
		Selection s = classify(root + "/Class2D/job004");
		Result r = importJobs(s, t.path() + "/dest");
		REQUIRE(r.ok);
		CHECK(r.missing == 1);
	}
}
