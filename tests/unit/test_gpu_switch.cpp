/*
 * tests/unit/test_gpu_switch.cpp
 *
 * The GPU speed-up switches are on when unset, off for 0/off/no/false, and "check" is a mode of its own.
 */

#include <catch2/catch.hpp>

#include "src/gpu_switch.h"
#include "src/gpu_reconstruct.h"

#include <cstdlib>
#include <string>

TEST_CASE("GPU speed-up switches are on unless switched off", "[gpu_switch]")
{
	const char *name = "RELION_TEST_GPU_SWITCH";
	unsetenv(name);
	CHECK(gpuswitch::mode(name) == 1);
	CHECK(gpuswitch::on(name));
	for (const char *off : {"0", "off", "no", "false"})
	{
		setenv(name, off, 1);
		INFO("value " << off);
		CHECK(gpuswitch::mode(name) == 0);
		CHECK(!gpuswitch::on(name));
	}
	for (const char *on : {"1", "single", "double", "yes", ""})
	{
		setenv(name, on, 1);
		INFO("value " << on);
		CHECK(gpuswitch::mode(name) == 1);
	}
	setenv(name, "check", 1);
	CHECK(gpuswitch::mode(name) == 2);
	unsetenv(name);
}

TEST_CASE("GPU reconstruction and symmetrisation switches default to on", "[gpu_switch]")
{
	unsetenv("RELION_GPU_RECONSTRUCT");
	unsetenv("RELION_GPU_SYMMETRISE");
	CHECK(gpurec::requested());
	CHECK(gpurec::symmetriseMode() == 1);
	setenv("RELION_GPU_RECONSTRUCT", "0", 1);
	setenv("RELION_GPU_SYMMETRISE", "0", 1);
	CHECK(!gpurec::requested());
	CHECK(gpurec::symmetriseMode() == 0);
	setenv("RELION_GPU_SYMMETRISE", "check", 1);
	CHECK(gpurec::symmetriseMode() == 2);
	unsetenv("RELION_GPU_RECONSTRUCT");
	unsetenv("RELION_GPU_SYMMETRISE");
}

TEST_CASE("Every default-on GPU switch in the list is on when unset and off with 0", "[gpu_switch]")
{
	size_t n = 0;
	const char *const *names = gpuswitch::defaultOnNames(n);
	REQUIRE(n >= 9);
	for (size_t i = 0; i < n; i++)
	{
		INFO("switch " << names[i]);
		const char *old = getenv(names[i]);
		const std::string saved = old ? old : "";
		unsetenv(names[i]);
		CHECK(gpuswitch::on(names[i]));
		setenv(names[i], "0", 1);
		CHECK(!gpuswitch::on(names[i]));
		if (old) setenv(names[i], saved.c_str(), 1); else unsetenv(names[i]);
	}
}
