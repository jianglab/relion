#include <catch2/catch.hpp>

#include "src/class2d_consensus_metadata.h"

#include <cstdio>
#include <ctime>

TEST_CASE("Class2DConsensus: independent-count diagnostics survive STAR round trip", "[consensus][metadata]")
{
	MetaDataTable diagnostics;
	diagnostics.setName("consensus_general");
	diagnostics.setIsList(true);
	diagnostics.addObject();
	diagnostics.setValue(EMDL_MLMODEL_NR_CLASSES, 150);
	diagnostics.setValue(EMDL_CLASS2D_CONSENSUS_SOURCE_CLASSES, 100);
	diagnostics.setValue(EMDL_CLASS2D_CONSENSUS_OCCUPIED_CLASSES, 148);
	diagnostics.setValue(EMDL_CLASS2D_CONSENSUS_PATTERNS, 10000L);
	diagnostics.setValue(EMDL_CLASS2D_CONSENSUS_MAPPING_MODE, std::string("consensus_to_source"));
	const FileName filename = "consensus_dimensions_" + integerToString((int)std::clock()) + ".star";
	diagnostics.write(filename);
	diagnostics.read(filename, "consensus_general");
	int classes = 0;
	long int patterns = 0;
	std::string mode;
	REQUIRE(diagnostics.getValue(EMDL_MLMODEL_NR_CLASSES, classes));
	REQUIRE(classes == 150);
	REQUIRE(diagnostics.getValue(EMDL_CLASS2D_CONSENSUS_SOURCE_CLASSES, classes));
	REQUIRE(classes == 100);
	REQUIRE(diagnostics.getValue(EMDL_CLASS2D_CONSENSUS_OCCUPIED_CLASSES, classes));
	REQUIRE(classes == 148);
	REQUIRE(diagnostics.getValue(EMDL_CLASS2D_CONSENSUS_PATTERNS, patterns));
	REQUIRE(patterns == 10000);
	REQUIRE(diagnostics.getValue(EMDL_CLASS2D_CONSENSUS_MAPPING_MODE, mode));
	REQUIRE(mode == "consensus_to_source");
	std::remove(filename.c_str());
}

TEST_CASE("Class2DConsensus: reset clears alignments and preserves particle metadata", "[consensus][metadata]")
{
	bool have_alignments = true, have_zoff = false;
	SECTION("existing 2D alignments") {}
	SECTION("missing 2D alignments") { have_alignments = false; }
	SECTION("existing alignments with Z shifts") { have_zoff = true; }
	SECTION("missing 2D alignments with Z shifts") { have_alignments = false; have_zoff = true; }
	const EMDLabel alignment_labels[] = {
		EMDL_ORIENT_ROT, EMDL_ORIENT_TILT, EMDL_ORIENT_PSI,
		EMDL_ORIENT_ORIGIN_X_ANGSTROM, EMDL_ORIENT_ORIGIN_Y_ANGSTROM
	};
	const EMDLabel preserved_doubles[] = {
		EMDL_ORIENT_ROT_PRIOR, EMDL_ORIENT_TILT_PRIOR, EMDL_ORIENT_PSI_PRIOR,
		EMDL_ORIENT_ORIGIN_X_PRIOR_ANGSTROM, EMDL_ORIENT_ORIGIN_Y_PRIOR_ANGSTROM,
		EMDL_ORIENT_ORIGIN_Z_PRIOR_ANGSTROM, EMDL_ORIENT_PSI_PRIOR_FLIP_RATIO,
		EMDL_CTF_DEFOCUSU, EMDL_CTF_DEFOCUSV, EMDL_CTF_DEFOCUS_ANGLE,
		EMDL_IMAGE_COORD_X, EMDL_IMAGE_COORD_Y, EMDL_IMAGE_NORM_CORRECTION,
		EMDL_PARTICLE_CLASS2D_CONSENSUS_PROBABILITY,
		EMDL_PARTICLE_CLASS2D_CONSENSUS_ENTROPY,
		EMDL_PARTICLE_CLASS2D_CONSENSUS_AGREEMENT
	};
	MetaDataTable particles;
	particles.setName("particles");
	for (long int row = 0; row < 3; ++row)
	{
		particles.addObject();
		particles.setValue(EMDL_IMAGE_NAME, std::string("particle_") + integerToString(row) + ".mrc", row);
		particles.setValue(EMDL_PARTICLE_CLASS, (int)row + 1, row);
		particles.setValue(EMDL_IMAGE_OPTICS_GROUP, (int)row + 1, row);
		for (EMDLabel label : preserved_doubles)
			particles.setValue(label, 0.125 * (row + 1), row);
		if (have_alignments)
			for (EMDLabel label : alignment_labels)
				particles.setValue(label, 17.5 * (row + 1), row);
		if (have_zoff) particles.setValue(EMDL_ORIENT_ORIGIN_Z_ANGSTROM, -2.5 * (row + 1), row);
	}

	resetClass2DConsensusAlignments(particles);
	for (bool round_trip : {false, true})
	{
		if (round_trip)
		{
			const FileName filename = "class2d_consensus_metadata_" + integerToString((int)std::clock()) + ".star";
			particles.write(filename);
			particles.read(filename, "particles");
			std::remove(filename.c_str());
		}
		REQUIRE(particles.numberOfObjects() == 3);
		REQUIRE(particles.containsLabel(EMDL_ORIENT_ORIGIN_Z_ANGSTROM) == have_zoff);
		for (long int row = 0; row < 3; ++row)
		{
			for (EMDLabel label : alignment_labels)
			{
				double value = -1.;
				REQUIRE(particles.getValue(label, value, row));
				REQUIRE(value == 0.);
			}
			if (have_zoff)
			{
				double value = -1.;
				REQUIRE(particles.getValue(EMDL_ORIENT_ORIGIN_Z_ANGSTROM, value, row));
				REQUIRE(value == 0.);
			}
			for (EMDLabel label : preserved_doubles)
			{
				double value = 0.;
				REQUIRE(particles.getValue(label, value, row));
				REQUIRE(value == Approx(0.125 * (row + 1)));
			}
			std::string name;
			int class_number = 0, optics_group = 0;
			REQUIRE(particles.getValue(EMDL_IMAGE_NAME, name, row));
			REQUIRE(name == "particle_" + integerToString(row) + ".mrc");
			REQUIRE(particles.getValue(EMDL_PARTICLE_CLASS, class_number, row));
			REQUIRE(class_number == row + 1);
			REQUIRE(particles.getValue(EMDL_IMAGE_OPTICS_GROUP, optics_group, row));
			REQUIRE(optics_group == row + 1);
		}
	}
}
