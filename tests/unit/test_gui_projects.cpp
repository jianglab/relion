/*
 * tests/unit/test_gui_projects.cpp
 *
 * The per-user project registry ($HOME/.relion/projects.star) that the GUI's
 * project page reads and writes. Paths and names may contain spaces, which
 * a plain space-separated file splits into extra columns.
 * Linked into relion_gui_tests, whose main comes from test_gui_jobwindows.cpp.
 */

#include <catch2/catch.hpp>

#include "src/gui_projects.h"

#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>

namespace {

/// Points HOME at a fresh temporary directory for the lifetime of the object.
struct TempHome
{
	std::string dir;
	std::string old_home;
	bool had_home;

	TempHome()
	{
		char tmpl[] = "/tmp/relion_test_projects_XXXXXX";
		const char *d = mkdtemp(tmpl);
		REQUIRE(d != NULL);
		dir = d;
		const char *h = std::getenv("HOME");
		had_home = (h != NULL);
		if (had_home) old_home = h;
		setenv("HOME", dir.c_str(), 1);
	}

	~TempHome()
	{
		if (had_home) setenv("HOME", old_home.c_str(), 1);
		else unsetenv("HOME");
		const std::string cmd = "rm -rf \"" + dir + "\"";
		if (system(cmd.c_str()) != 0) {}
	}

	void writeRegistry(const std::string &body)
	{
		const std::string reg = dir + "/.relion";
		const std::string cmd = "mkdir -p \"" + reg + "\"";
		REQUIRE(system(cmd.c_str()) == 0);
		std::ofstream fh((reg + "/projects.star").c_str());
		fh << body;
	}
};

const std::string kHeader =
	"# version 50001\n\ndata_projects\n\nloop_\n"
	"_rlnProjectPath #1\n_rlnProjectName #2\n_rlnProjectLastOpened #3\n";

} // namespace

TEST_CASE("Project registry keeps paths and names with spaces", "[gui][projects]")
{
	TempHome home;

	ProjectManager pm;
	pm.add("/data/my projects/apoferritin run1", "apoferritin run1");
	pm.add("/data/plain", "plain");
	pm.add("/data/\"quoted\" dir", "\"quoted\"");
	pm.rename("/data/plain", "Renamed with spaces");
	REQUIRE(pm.save());

	ProjectManager back;
	REQUIRE(back.load());
	REQUIRE(back.getAll().size() == 3);

	const ProjectManager::Project *p = back.findByPath("/data/my projects/apoferritin run1");
	REQUIRE(p != NULL);
	CHECK(p->name == "apoferritin run1");
	CHECK(p->last_opened.size() == 19);   // YYYY-MM-DDTHH:MM:SS

	p = back.findByPath("/data/plain");
	REQUIRE(p != NULL);
	CHECK(p->name == "Renamed with spaces");

	p = back.findByPath("/data/\"quoted\" dir");
	REQUIRE(p != NULL);
	CHECK(p->name == "\"quoted\"");
}

TEST_CASE("Project registry reads files written before quoting", "[gui][projects]")
{
	TempHome home;
	home.writeRegistry(kHeader +
		"/data/projA  projA  2026-11-01T10:00:00\n"
		"/data/projB  projB  2026-11-03T09:30:00\n\n");

	ProjectManager pm;
	REQUIRE(pm.load());
	REQUIRE(pm.getAll().size() == 2);
	const ProjectManager::Project *p = pm.findByPath("/data/projB");
	REQUIRE(p != NULL);
	CHECK(p->name == "projB");
	CHECK(p->last_opened == "2026-11-03T09:30:00");
}

TEST_CASE("Project registry skips a malformed line and keeps the rest", "[gui][projects]")
{
	TempHome home;
	home.writeRegistry(kHeader +
		"\"/data/unclosed quote  broken  2026-11-01T10:00:00\n"
		"\"/data/good one\"  good  2026-11-02T10:00:00\n\n");

	ProjectManager pm;
	REQUIRE(pm.load());
	REQUIRE(pm.getAll().size() == 1);
	CHECK(pm.findByPath("/data/good one") != NULL);
}
