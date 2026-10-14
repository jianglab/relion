/*
 * tests/unit/test_cache_init.cpp
 *
 * A cache directory that a node cannot use (node-local scratch that only some
 * nodes have) means reading the particles where they are, with a warning.
 */

#include <catch2/catch.hpp>

#include "src/cache_init.h"
#include "src/metadata_table.h"

#include <string>

static MetaDataTable particles(const std::string &stack)
{
	MetaDataTable md;
	for (int i = 1; i <= 3; i++)
	{
		md.addObject();
		md.setValue(EMDL_IMAGE_NAME, std::to_string(i) + "@" + stack);
	}
	return md;
}

TEST_CASE("cache: an unusable cache directory is skipped", "[cache]")
{
	// a directory that cannot be created
	const FileName cache = "/proc/relion_no_such_cache_dir";
	MetaDataTable md = particles("Extract/job001/particles.mrcs");
	REQUIRE(CacheInitializer::initializeCache(cache, 1, md, 0, NULL, false, "testnode"));
	std::string name;
	md.getValue(EMDL_IMAGE_NAME, name, 0);
	CHECK(name == "1@Extract/job001/particles.mrcs");

	// names that already point into it cannot be read without it
	MetaDataTable cached = particles(cache + "/key/Extract/job001/particles.mrcs");
	CHECK_THROWS(CacheInitializer::initializeCache(cache, 1, cached, 0, NULL, false, "testnode"));
}
