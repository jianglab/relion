/*
 * tests/unit/test_recover_full_filaments.cpp
 *
 * recoverFullFilaments() adds back all segments of the filaments that are well represented in a
 * subset of helical segments, as helicon images2star --recoverFullFilaments did.
 * findFullFilamentsStarFile() finds the STAR file of the job that wrote the particle images.
 */

#include <catch2/catch.hpp>

#include "src/helix.h"
#include "src/metadata_table.h"

#include <climits>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

namespace {

const std::string stack_a = "Extract/job012/Movies/a.mrcs";
const std::string stack_b = "Extract/job012/Movies/b.mrcs";

std::string imageName(int no, const std::string &stack)
{
	return integerToString(no, 6) + "@" + stack;
}

void addSegment(MetaDataTable &MD, int no, const std::string &stack, int tube_id)
{
	MD.addObject();
	MD.setValue(EMDL_IMAGE_NAME, imageName(no, stack));
	MD.setValue(EMDL_PARTICLE_HELICAL_TUBE_ID, tube_id);
	MD.setValue(EMDL_ORIENT_PSI_PRIOR, 10. * no);
}

// Micrograph a: tube 1 has images 1-4, tube 2 has images 5-6. Micrograph b: tube 1 has images 1-10.
MetaDataTable makeFull()
{
	MetaDataTable MD;
	MD.setName("particles");
	for (int no = 1; no <= 4; no++) addSegment(MD, no, stack_a, 1);
	for (int no = 5; no <= 6; no++) addSegment(MD, no, stack_a, 2);
	for (int no = 1; no <= 10; no++) addSegment(MD, no, stack_b, 1);
	return MD;
}

void addSelected(MetaDataTable &MD, const std::string &image)
{
	MD.addObject();
	MD.setValue(EMDL_IMAGE_NAME, image);
	MD.setValue(EMDL_PARTICLE_CLASS, 3);
}

std::vector<std::string> imageNames(const MetaDataTable &MD)
{
	std::vector<std::string> names;
	for (long int i = 0; i < MD.numberOfObjects(); i++)
	{
		std::string name;
		MD.getValue(EMDL_IMAGE_NAME, name, i);
		names.push_back(name);
	}
	return names;
}

std::vector<std::string> imageNames(const std::string &stack, int first, int last)
{
	std::vector<std::string> names;
	for (int no = first; no <= last; no++)
		names.push_back(imageName(no, stack));
	return names;
}

std::vector<std::string> concat(std::vector<std::string> a, const std::vector<std::string> &b)
{
	a.insert(a.end(), b.begin(), b.end());
	return a;
}

class TempDir {
public:
	TempDir()
	{
		const char* home = getenv("HOME");
		std::string base = std::string(home ? home : ".") + "/temp";
		mkdir(base.c_str(), 0755);
		std::string tmpl = base + "/relion_filaments_XXXXXX";
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

void touch(const std::string &fn)
{
	std::ofstream out(fn.c_str());
	out << "\n";
}

MetaDataTable selectionFrom(const std::string &job_dir)
{
	MetaDataTable MD;
	addSelected(MD, imageName(1, job_dir + "Movies/a.mrcs"));
	addSelected(MD, imageName(2, job_dir + "Movies/b.mrcs"));
	return MD;
}

} // namespace

TEST_CASE("recoverFullFilaments keeps filaments with enough selected segments", "[helix][recover_filaments]")
{
	const MetaDataTable MDfull = makeFull();
	MetaDataTable MDsel;
	addSelected(MDsel, imageName(2, stack_a)); // a, tube 1: 2 of 4
	addSelected(MDsel, imageName(1, stack_a));
	addSelected(MDsel, imageName(6, stack_a)); // a, tube 2: 1 of 2
	addSelected(MDsel, imageName(3, stack_b)); // b, tube 1: 1 of 10

	FullFilamentRecoveryStats stats;
	MetaDataTable MDout = recoverFullFilaments(MDsel, MDfull, 0.5, &stats);

	// All segments of the recovered filaments, in the order of the full STAR file
	REQUIRE(imageNames(MDout) == imageNames(stack_a, 1, 6));
	REQUIRE(stats.nr_selected_segments == 4);
	REQUIRE(stats.nr_selected_filaments == 3);
	REQUIRE(stats.nr_recovered_filaments == 2);
	REQUIRE(stats.nr_recovered_segments == 6);
	REQUIRE(stats.nr_full_segments == 16);

	// The rows come from the full STAR file
	REQUIRE(MDout.getName() == "particles");
	REQUIRE(MDout.containsLabel(EMDL_ORIENT_PSI_PRIOR));
	REQUIRE(MDout.containsLabel(EMDL_PARTICLE_HELICAL_TUBE_ID));
	REQUIRE_FALSE(MDout.containsLabel(EMDL_PARTICLE_CLASS));
	RFLOAT psi_prior;
	MDout.getValue(EMDL_ORIENT_PSI_PRIOR, psi_prior, 4);
	REQUIRE(psi_prior == Approx(50.));

	SECTION("a minimum fraction of 0 keeps every filament with a selected segment")
	{
		MDout = recoverFullFilaments(MDsel, MDfull, 0., &stats);
		REQUIRE(imageNames(MDout) == concat(imageNames(stack_a, 1, 6), imageNames(stack_b, 1, 10)));
		REQUIRE(stats.nr_recovered_filaments == 3);
	}

	SECTION("a minimum fraction of 1 keeps only completely selected filaments")
	{
		addSelected(MDsel, imageName(5, stack_a));
		MDout = recoverFullFilaments(MDsel, MDfull, 1., &stats);
		REQUIRE(imageNames(MDout) == imageNames(stack_a, 5, 6));
	}

	SECTION("a fraction just below the minimum is not enough")
	{
		MDout = recoverFullFilaments(MDsel, MDfull, 0.51, &stats);
		REQUIRE(MDout.numberOfObjects() == 0);
		REQUIRE(stats.nr_recovered_filaments == 0);
	}
}

TEST_CASE("recoverFullFilaments matches images by number and stack", "[helix][recover_filaments]")
{
	const MetaDataTable MDfull = makeFull();
	MetaDataTable MDsel;
	addSelected(MDsel, "1@./" + stack_a);
	addSelected(MDsel, "000002@" + stack_a);

	MetaDataTable MDout = recoverFullFilaments(MDsel, MDfull, 0.5);
	REQUIRE(imageNames(MDout) == imageNames(stack_a, 1, 4));
}

TEST_CASE("recoverFullFilaments counts duplicated images once", "[helix][recover_filaments]")
{
	SECTION("in the selection")
	{
		const MetaDataTable MDfull = makeFull();
		MetaDataTable MDsel;
		for (int i = 0; i < 3; i++)
			addSelected(MDsel, imageName(1, stack_a)); // still 1 of 4

		FullFilamentRecoveryStats stats;
		MetaDataTable MDout = recoverFullFilaments(MDsel, MDfull, 0.5, &stats);
		REQUIRE(MDout.numberOfObjects() == 0);
		REQUIRE(stats.nr_selected_segments == 1);
	}

	SECTION("in the full STAR file, where the last row is kept")
	{
		MetaDataTable MDfull = makeFull();
		addSegment(MDfull, 2, stack_a, 1);
		MDfull.setValue(EMDL_ORIENT_PSI_PRIOR, 123.);
		MetaDataTable MDsel;
		addSelected(MDsel, imageName(1, stack_a));
		addSelected(MDsel, imageName(2, stack_a));

		FullFilamentRecoveryStats stats;
		MetaDataTable MDout = recoverFullFilaments(MDsel, MDfull, 0.5, &stats);
		REQUIRE(imageNames(MDout) == concat(concat(imageNames(stack_a, 1, 1), imageNames(stack_a, 3, 4)),
		                                    imageNames(stack_a, 2, 2)));
		REQUIRE(stats.nr_full_segments == 16);
		RFLOAT psi_prior;
		MDout.getValue(EMDL_ORIENT_PSI_PRIOR, psi_prior, 3);
		REQUIRE(psi_prior == Approx(123.));
	}
}

TEST_CASE("recoverFullFilaments takes the filaments from the full STAR file", "[helix][recover_filaments]")
{
	// A tube ID in the selection that differs from the full STAR file does not matter
	const MetaDataTable MDfull = makeFull();
	MetaDataTable MDsel;
	addSelected(MDsel, imageName(5, stack_a));
	MDsel.setValue(EMDL_PARTICLE_HELICAL_TUBE_ID, 7);

	MetaDataTable MDout = recoverFullFilaments(MDsel, MDfull, 0.5);
	REQUIRE(imageNames(MDout) == imageNames(stack_a, 5, 6));
}

TEST_CASE("recoverFullFilaments rejects inconsistent input", "[helix][recover_filaments]")
{
	const MetaDataTable MDfull = makeFull();
	MetaDataTable MDsel;
	addSelected(MDsel, imageName(1, stack_a));

	SECTION("a selected segment that is not in the full STAR file")
	{
		addSelected(MDsel, imageName(11, stack_b));
		REQUIRE_THROWS(recoverFullFilaments(MDsel, MDfull, 0.5));
	}

	SECTION("a minimum fraction outside [0, 1]")
	{
		REQUIRE_THROWS(recoverFullFilaments(MDsel, MDfull, -0.1));
		REQUIRE_THROWS(recoverFullFilaments(MDsel, MDfull, 1.1));
	}

	SECTION("a full STAR file without rlnHelicalTubeID")
	{
		MetaDataTable MDnotube;
		MDnotube.addObject();
		MDnotube.setValue(EMDL_IMAGE_NAME, imageName(1, stack_a));
		REQUIRE_THROWS(recoverFullFilaments(MDsel, MDnotube, 0.5));
	}
}

TEST_CASE("findFullFilamentsStarFile finds the particles of the job that wrote the images", "[helix][recover_filaments]")
{
	TempDir tmp;
	const std::string extract_dir = tmp.path() + "/Extract/job012/";
	const std::string polish_dir = tmp.path() + "/Polish/job020/";
	REQUIRE(system(("mkdir -p '" + extract_dir + "Movies' '" + polish_dir + "Movies'").c_str()) == 0);

	SECTION("particles.star of an Extract job")
	{
		touch(extract_dir + "particles.star");
		REQUIRE(findFullFilamentsStarFile(selectionFrom(extract_dir)) == extract_dir + "particles.star");
	}

	SECTION("shiny.star of a Polish job")
	{
		touch(polish_dir + "shiny.star");
		REQUIRE(findFullFilamentsStarFile(selectionFrom(polish_dir)) == polish_dir + "shiny.star");
	}

	SECTION("a particles output node of the job, not an input node from another job")
	{
		touch(extract_dir + "particles.star");
		touch(extract_dir + "segments.star");
		MetaDataTable MDnodes;
		MDnodes.setName("pipeline_nodes");
		MDnodes.addObject();
		MDnodes.setValue(EMDL_PIPELINE_NODE_NAME, std::string("Refine3D/job005/run_data.star"));
		MDnodes.setValue(EMDL_PIPELINE_NODE_TYPE_LABEL, std::string("ParticleGroupMetadata.star.relion.refine3d"));
		MDnodes.addObject();
		MDnodes.setValue(EMDL_PIPELINE_NODE_NAME, std::string("Extract/job012/coords_suffix.star"));
		MDnodes.setValue(EMDL_PIPELINE_NODE_TYPE_LABEL, std::string("MicrographCoordsGroup.star.relion"));
		MDnodes.addObject();
		MDnodes.setValue(EMDL_PIPELINE_NODE_NAME, std::string("Extract/job012/segments.star"));
		MDnodes.setValue(EMDL_PIPELINE_NODE_TYPE_LABEL, std::string("ParticleGroupMetadata.star.relion.helicalsegments"));
		MDnodes.write(extract_dir + "job_pipeline.star");
		touch(extract_dir + "coords_suffix.star");

		REQUIRE(findFullFilamentsStarFile(selectionFrom(extract_dir)) == extract_dir + "segments.star");
	}

	SECTION("images from two jobs")
	{
		touch(extract_dir + "particles.star");
		MetaDataTable MDsel = selectionFrom(extract_dir);
		addSelected(MDsel, imageName(1, polish_dir + "Movies/a.mrcs"));
		REQUIRE_THROWS(findFullFilamentsStarFile(MDsel));
	}

	SECTION("images outside a job directory")
	{
		MetaDataTable MDsel;
		addSelected(MDsel, imageName(1, tmp.path() + "/Movies/a.mrcs"));
		REQUIRE_THROWS(findFullFilamentsStarFile(MDsel));
	}

	SECTION("a job without a particles STAR file")
	{
		REQUIRE_THROWS(findFullFilamentsStarFile(selectionFrom(extract_dir)));
	}
}
