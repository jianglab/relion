/*
 * tests/unit/test_resource_report.cpp
 *
 * The CPU/GPU balance advice of src/resource_report.h: a starved GPU asks for
 * more threads (up to a thread per core), then for MPI processes under MPS; a
 * saturated one for fewer; idle allocated cores are flagged, and a balanced run
 * gets no advice.
 */

#include <catch2/catch.hpp>

#include "src/resource_report.h"

namespace
{
resrep::ProcessStats proc(const std::string &host, int threads, int ncores, int first_core,
                          double gpu_busy, const std::string &gpu = "0000:01:00.0", double cpu_busy = 1.)
{
	resrep::ProcessStats p;
	p.host = host;
	p.threads = threads;
	p.wall = 100.;
	p.cpu = 100. * threads * cpu_busy;
	for (int c = 0; c < ncores; c++) p.cores.push_back(first_core + c);
	if (gpu_busy > -2.)
	{
		p.uses_gpu = true;
		p.gpus.push_back(gpu);
		p.gpu_busy.push_back(gpu_busy);
	}
	return p;
}

bool mentions(const std::vector<std::string> &a, const std::string &what)
{
	for (const auto &s : a) if (s.find(what) != std::string::npos) return true;
	return false;
}
}

TEST_CASE("a starved GPU with fewer threads than cores asks for more threads", "[resource_report]")
{
	std::vector<resrep::ProcessStats> p(1, proc("n1", 4, 8, 0, 0.11));
	auto a = resrep::advise(p, false);
	REQUIRE(!a.empty());
	CHECK(mentions(a, "busy only 11%"));
	CHECK(mentions(a, "Raise --j from 4 to 8"));
	CHECK(!mentions(a, "NVIDIA MPS"));
}

TEST_CASE("a starved GPU with a thread per core recommends MPI processes under MPS", "[resource_report]")
{
	std::vector<resrep::ProcessStats> p(1, proc("n1", 8, 8, 0, 0.11));
	auto a = resrep::advise(p, false);
	CHECK(mentions(a, "NVIDIA MPS"));
	CHECK(mentions(a, "about 7, each with --j 1"));
	CHECK(!mentions(a, "Raise --j from"));
}

TEST_CASE("a starved GPU without idle cores makes no guess at a core count", "[resource_report]")
{
	std::vector<resrep::ProcessStats> p(1, proc("n1", 4, 4, 0, 0.2));
	auto a = resrep::advise(p, false);
	CHECK(mentions(a, "NVIDIA MPS"));
	CHECK(!mentions(a, "request about"));
}

TEST_CASE("a saturated GPU with many threads suggests fewer, and idle cores are flagged", "[resource_report]")
{
	std::vector<resrep::ProcessStats> p(1, proc("n1", 8, 16, 0, 0.97));
	auto a = resrep::advise(p, false);
	CHECK(mentions(a, "unlikely to make this faster"));
	CHECK(mentions(a, "on average only 8 of the 16 allocated cores were busy"));
}

TEST_CASE("a balanced GPU run gets no advice", "[resource_report]")
{
	std::vector<resrep::ProcessStats> p(1, proc("n1", 2, 2, 0, 0.8));
	CHECK(resrep::advise(p, false).empty());
}

TEST_CASE("MPI: processes on a host share its cores, and low use is flagged", "[resource_report]")
{
	// leader + 2 followers (--j 3, one GPU) sharing 8 cores
	std::vector<resrep::ProcessStats> busy, waiting;
	busy.push_back(proc("n1", 3, 8, 0, -3., "", 0.33));
	busy.push_back(proc("n1", 3, 8, 0, 0.75));
	busy.push_back(proc("n1", 3, 8, 0, 0.75));
	for (size_t i = 1; i < busy.size(); i++) busy[i].mps = true; // two workers on one GPU: with MPS
	CHECK(resrep::advise(busy, true).empty());

	waiting.push_back(proc("n1", 3, 8, 0, -3., "", 0.33));
	waiting.push_back(proc("n1", 3, 8, 0, 0.75, "0000:01:00.0", 0.3));
	waiting.push_back(proc("n1", 3, 8, 0, 0.75, "0000:01:00.0", 0.3));
	for (size_t i = 1; i < waiting.size(); i++) waiting[i].mps = true;
	auto a = resrep::advise(waiting, true);
	CHECK(mentions(a, "on average only 2.8 of the 8 allocated cores were busy"));
}

TEST_CASE("several processes on one GPU without MPS are told to use it, whatever the busy figure", "[resource_report]")
{
	std::vector<resrep::ProcessStats> p;
	p.push_back(proc("n1", 1, 8, 0, -3., "", 0.1));
	for (int i = 0; i < 4; i++) p.push_back(proc("n1", 1, 8, 0, 0.98));
	auto a = resrep::advise(p, true);
	CHECK(mentions(a, "4 worker processes shared a GPU without NVIDIA MPS"));
	CHECK(!mentions(a, "unlikely to make this faster"));
	for (size_t i = 1; i < p.size(); i++) p[i].mps = true;
	CHECK(!mentions(resrep::advise(p, true), "without NVIDIA MPS"));
}

TEST_CASE("totals over several E-steps weight GPU use by time", "[resource_report]")
{
	resrep::ProcessStats t, a = proc("n1", 2, 2, 0, 0.2), b = proc("n1", 2, 2, 0, 0.8);
	b.wall = 300.;
	resrep::accumulate(t, a);
	resrep::accumulate(t, b);
	resrep::finish(t);
	CHECK(t.wall == Approx(400.));
	CHECK(t.gpu_busy[0] == Approx((0.2 * 100. + 0.8 * 300.) / 400.));
}

TEST_CASE("CPU-only runs with idle threads point at the disk", "[resource_report]")
{
	std::vector<resrep::ProcessStats> p(1, proc("n1", 8, 8, 0, -3., "", 0.3));
	auto a = resrep::advise(p, false);
	CHECK(mentions(a, "--preread_images"));
}

TEST_CASE("with MPS on, a starved GPU asks for more processes, and the summary says so", "[resource_report]")
{
	std::vector<resrep::ProcessStats> p(1, proc("n1", 8, 8, 0, 0.3));
	p[0].mps = true;
	auto a = resrep::advise(p, false);
	CHECK(mentions(a, "MPS is running"));
	CHECK(!mentions(a, "nvidia-cuda-mps-control -d"));
	auto s = resrep::summary(p, false);
	REQUIRE(s.size() == 1);
	CHECK(mentions(s, "GPU busy 30% (MPS on)"));
	CHECK(mentions(s, "8 of 8 cores busy"));
}

TEST_CASE("stats survive the MPI serialisation", "[resource_report]")
{
	resrep::ProcessStats p = proc("node-7", 3, 4, 2, 0.42, "0000:3B:00.0");
	p.mps = true;
	resrep::ProcessStats q = resrep::deserialise(resrep::serialise(p));
	CHECK(q.host == "node-7");
	CHECK(q.threads == 3);
	CHECK(q.cores == p.cores);
	REQUIRE(q.gpus.size() == 1);
	CHECK(q.gpus[0] == "0000:3B:00.0");
	CHECK(q.gpu_busy[0] == Approx(0.42));
	CHECK(q.mps);
}
