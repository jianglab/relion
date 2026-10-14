/*
 * tests/unit/test_gpu_switch.cpp
 *
 * The GPU speed-up switches are on when unset, off for 0/off/no/false, and "check" is a mode of its own.
 */

#include <catch2/catch.hpp>

#include "src/gpu_switch.h"
#include "src/gpu_reconstruct.h"

#include <cstdlib>

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
