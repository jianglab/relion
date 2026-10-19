/*
 * tests/unit/test_parallel_hint.cpp
 *
 * Suggested MPI processes and threads shown by the GUI (src/parallel_hint.h).
 */

#include <catch2/catch.hpp>

#include "src/parallel_hint.h"
#include "src/pipeline_jobs.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>

using parhint::Kind;

TEST_CASE("parallel hint: suggestions per kind of job", "[parhint]")
{
	SECTION("EM classification: one process for one GPU, leader + one worker per GPU otherwise")
	{
		auto s = parhint::suggest(Kind::Classify, 1, 8);
		REQUIRE(s.valid);
		CHECK(s.mpi == 1);
		CHECK(s.threads == 8);
		s = parhint::suggest(Kind::Classify, 4, 8);
		CHECK(s.mpi == 5);
		CHECK(s.threads == 8);
	}
	SECTION("VDAM: one process, threads for all GPUs")
	{
		auto s = parhint::suggest(Kind::Vdam, 2, 8);
		CHECK(s.mpi == 1);
		CHECK(s.threads == 16);
	}
	SECTION("auto-refine: odd number of processes, at least 3, cores shared among workers")
	{
		auto s = parhint::suggest(Kind::AutoRefine, 1, 8);
		CHECK(s.mpi == 3);
		CHECK(s.threads == 4);
		s = parhint::suggest(Kind::AutoRefine, 2, 8);
		CHECK(s.mpi == 3);
		CHECK(s.threads == 8);
		s = parhint::suggest(Kind::AutoRefine, 3, 8);
		CHECK(s.mpi == 5);
		CHECK(s.threads == 6);
		s = parhint::suggest(Kind::AutoRefine, 4, 8);
		CHECK(s.mpi == 5);
		CHECK(s.threads == 8);
		for (int g = 1; g <= 9; g++)
		{
			const int mpi = parhint::suggest(Kind::AutoRefine, g, 4).mpi;
			CHECK(mpi >= 3);
			CHECK(mpi % 2 == 1);
		}
	}
	SECTION("unknown cores: no thread number")
	{
		CHECK(parhint::suggest(Kind::Classify, 1, 0).threads == 0);
		CHECK(parhint::suggest(Kind::AutoRefine, 1, 0).threads == 0);
	}
	SECTION("no suggestion for other jobs or no GPU")
	{
		CHECK_FALSE(parhint::suggest(Kind::Other, 1, 8).valid);
		CHECK_FALSE(parhint::suggest(Kind::Classify, 0, 8).valid);
	}
}

TEST_CASE("parallel hint: kinds of GUI jobs", "[parhint]")
{
	CHECK(parhint::kindOf(PROC_2DCLASS, false) == Kind::Classify);
	CHECK(parhint::kindOf(PROC_2DCLASS, true) == Kind::Vdam);
	CHECK(parhint::kindOf(PROC_3DCLASS, false) == Kind::Classify);
	CHECK(parhint::kindOf(PROC_INIMODEL, false) == Kind::Vdam);
	CHECK(parhint::kindOf(PROC_3DAUTO, false) == Kind::AutoRefine);
	CHECK(parhint::kindOf(PROC_MULTIBODY, false) == Kind::AutoRefine);
	CHECK(parhint::kindOf(PROC_EXTRACT, false) == Kind::Other);
}

TEST_CASE("parallel hint: GPU ids and Slurm node descriptions", "[parhint]")
{
	CHECK(parhint::countGpuIds("") == 0);
	CHECK(parhint::countGpuIds("0") == 1);
	CHECK(parhint::countGpuIds("0:1") == 2);
	CHECK(parhint::countGpuIds("0,0:1,1:0,0:1,1") == 2);
	CHECK(parhint::countGpuIds("0;1;2") == 3);

	CHECK(parhint::coresPerGpuFromSinfo("") == 0);
	CHECK(parhint::coresPerGpuFromSinfo("20+ (null)\n") == 0);
	CHECK(parhint::coresPerGpuFromSinfo("64 gpu:4\n") == 16);
	// the smallest ratio over node types, whatever the GRES spelling
	CHECK(parhint::coresPerGpuFromSinfo("80 gpu:typeA:3\n64 gpu:typeB:1\n16 gpu:typeC:1\n128 gpu:typeD:4(S:0-1)\n") == 16);
	CHECK(parhint::coresPerGpuFromSinfo("48 gpu:2,mps:200\n") == 24);
}

TEST_CASE("parallel hint: tooltip from the current job settings", "[parhint]")
{
	setenv("RELION_CPUS_PER_GPU", "8", 1);
	std::map<std::string, std::string> values, labels;
	parhint::JobView view;
	view.value = [&](const std::string &k) { auto it = values.find(k); return it == values.end() ? std::string() : it->second; };
	view.label = [&](const std::string &k) { auto it = labels.find(k); return it == labels.end() ? std::string() : it->second; };

	view.job_type = PROC_EXTRACT;
	CHECK(parhint::tooltip(view, parhint::Field::MpiOrThreads).empty());

	view.job_type = PROC_3DAUTO;
	values["use_gpu"] = "No";
	CHECK(parhint::tooltip(view, parhint::Field::MpiOrThreads).find("does not use GPUs") != std::string::npos);

	// a queue job: the GPU count comes from the queue option whose label mentions GPUs
	values["use_gpu"] = "Yes";
	values["do_queue"] = "Yes";
	labels["qsub_extra1"] = "Walltime";
	values["qsub_extra1"] = "2";
	labels["qsub_extra3"] = "Number of GPUs";
	values["qsub_extra3"] = "2";
	std::string t = parhint::tooltip(view, parhint::Field::MpiOrThreads);
	CHECK(t.find("Suggested for 2 GPUs with 8 CPU cores per GPU") != std::string::npos);
	CHECK(t.find("MPI processes: 3    Threads: 8") != std::string::npos);
	CHECK(t.find("\"Number of GPUs\" below") != std::string::npos);

	// a local job: the GPU count comes from "Which GPUs to use"
	values["do_queue"] = "No";
	values["gpu_ids"] = "0:1:2:3";
	view.job_type = PROC_2DCLASS;
	values["do_grad"] = "No";
	t = parhint::tooltip(view, parhint::Field::MpiOrThreads);
	CHECK(t.find("MPI processes: 5    Threads: 8") != std::string::npos);
	values["do_grad"] = "Yes";
	t = parhint::tooltip(view, parhint::Field::MpiOrThreads);
	CHECK(t.find("MPI processes: 1    Threads: 32") != std::string::npos);
	unsetenv("RELION_CPUS_PER_GPU");
}

TEST_CASE("parallel hint: splitting a number of CPUs", "[parhint]")
{
	SECTION("EM classification")
	{
		auto s = parhint::split(Kind::Classify, 1, 8);
		CHECK(s.mpi == 1);
		CHECK(s.threads == 8);
		s = parhint::split(Kind::Classify, 2, 16);
		CHECK(s.mpi == 3);
		CHECK(s.threads == 8);
		s = parhint::split(Kind::Classify, 0, 32);   // CPU only
		CHECK(s.mpi == 5);
		CHECK(s.threads == 8);
		s = parhint::split(Kind::Classify, 0, 4);
		CHECK(s.mpi == 1);
		CHECK(s.threads == 4);
	}
	SECTION("VDAM: always one process")
	{
		auto s = parhint::split(Kind::Vdam, 2, 16);
		CHECK(s.mpi == 1);
		CHECK(s.threads == 16);
	}
	SECTION("auto-refine: odd, at least 3, cores shared among the workers")
	{
		auto s = parhint::split(Kind::AutoRefine, 1, 8);
		CHECK(s.mpi == 3);
		CHECK(s.threads == 4);
		s = parhint::split(Kind::AutoRefine, 3, 24);
		CHECK(s.mpi == 5);
		CHECK(s.threads == 6);
		s = parhint::split(Kind::AutoRefine, 0, 4);   // CPU only, few cores
		CHECK(s.mpi == 3);
		CHECK(s.threads == 2);
		for (int g = 0; g <= 5; g++)
			for (int c = 1; c <= 40; c += 3)
			{
				s = parhint::split(Kind::AutoRefine, g, c);
				CHECK(s.mpi >= 3);
				CHECK(s.mpi % 2 == 1);
				CHECK(s.threads >= 1);
			}
	}
	SECTION("never fewer than one thread")
	{
		CHECK(parhint::split(Kind::Classify, 4, 2).threads == 1);
		CHECK_FALSE(parhint::split(Kind::Classify, 1, 0).valid);
	}
}

TEST_CASE("parallel hint: a job that chooses MPI processes and threads itself", "[parhint]")
{
	setenv("RELION_CPUS_PER_GPU", "8", 1);
	std::map<std::string, std::string> values, labels;
	parhint::JobView view;
	view.job_type = PROC_3DAUTO;
	view.value = [&](const std::string &k) { auto it = values.find(k); return it == values.end() ? std::string() : it->second; };
	view.label = [&](const std::string &k) { auto it = labels.find(k); return it == labels.end() ? std::string() : it->second; };
	values["use_gpu"] = "Yes";
	values["do_queue"] = "No";
	values["gpu_ids"] = "0:1";
	values["nr_cpus"] = "16";

	parhint::Suggestion s;
	values["auto_parallel"] = "No";
	CHECK_FALSE(parhint::autoSplit(view, s));
	CHECK(parhint::tooltip(view, parhint::Field::Cpus).find("Used only when") != std::string::npos);

	values["auto_parallel"] = "Yes";
	REQUIRE(parhint::autoSplit(view, s));
	CHECK(s.mpi == 3);
	CHECK(s.threads == 8);
	std::string t = parhint::tooltip(view, parhint::Field::MpiOrThreads);
	CHECK(t.find("RELION splits the 16 CPUs into") != std::string::npos);
	CHECK(t.find("MPI processes: 3    Threads: 8") != std::string::npos);
	CHECK(parhint::tooltip(view, parhint::Field::Cpus).find("Suggested number of CPUs for 2 GPUs: 16") != std::string::npos);

	// Class2D replicas: the GPUs of one replica
	view.job_type = PROC_2DCLASS;
	values["do_grad"] = "No";
	values["gpu_ids"] = "0;1;2";
	values["nr_cpus"] = "8";
	REQUIRE(parhint::autoSplit(view, s));
	CHECK(s.mpi == 1);
	CHECK(s.threads == 8);
	unsetenv("RELION_CPUS_PER_GPU");
}

TEST_CASE("auto threads: cores per worker, capped per GPU", "[parhint]")
{
	using parhint::autoThreads;
	SECTION("CPU-only job: all cores of the worker, never the cap")
	{
		CHECK(autoThreads(64, 1, 0) == 64);
		CHECK(autoThreads(64, 2, 0) == 32);
	}
	SECTION("GPU job: capped at 8 per GPU the process feeds")
	{
		CHECK(autoThreads(64, 2, 1) == 8);     // two workers, one GPU
		CHECK(autoThreads(64, 1, 1) == 8);
		CHECK(autoThreads(64, 2, 2) == 8);     // one GPU each
		CHECK(autoThreads(64, 1, 4) == 32);    // one process, four GPUs
		CHECK(autoThreads(64, 2, 3) == 16);    // 3 GPUs shared by 2 workers: up to 2 each
	}
	SECTION("the cores a job may use come first: an 8-CPU Slurm job with 2 workers gets 4")
	{
		CHECK(autoThreads(8, 2, 1) == 4);
		CHECK(autoThreads(1, 2, 1) == 1);
		CHECK(autoThreads(3, 4, 1) == 1);   // never below 1
	}
	SECTION("cap 0 switches the cap off, another number replaces it")
	{
		CHECK(autoThreads(64, 2, 1, 0) == 32);
		CHECK(autoThreads(64, 2, 1, 6) == 6);
	}
	SECTION("RELION_AUTO_THREADS_CAP")
	{
		unsetenv("RELION_AUTO_THREADS_CAP");
		CHECK(parhint::autoThreadsCapFromEnv() == parhint::THREADS_PER_GPU_CAP);
		setenv("RELION_AUTO_THREADS_CAP", "off", 1);
		CHECK(parhint::autoThreadsCapFromEnv() == 0);
		setenv("RELION_AUTO_THREADS_CAP", "5", 1);
		CHECK(parhint::autoThreadsCapFromEnv() == 5);
		setenv("RELION_AUTO_THREADS_CAP", "junk", 1);
		CHECK(parhint::autoThreadsCapFromEnv() == parhint::THREADS_PER_GPU_CAP);
		unsetenv("RELION_AUTO_THREADS_CAP");
	}
}

TEST_CASE("auto threads: GPU memory and work size limits", "[parhint]")
{
	using parhint::autoThreads;
	const double GB = 1e9;
	SECTION("GPU memory: 8 GB card, 2 workers: 4 threads each (8 per worker ran out of memory)")
	{
		CHECK(autoThreads(16, 2, 1, 8, 7.0 * GB) == 4);
		CHECK(autoThreads(16, 2, 1, 8, 7.0 * GB) < 8);
	}
	SECTION("a 24 GB card does not limit below the per-GPU cap")
	{
		CHECK(autoThreads(64, 2, 1, 8, 22.0 * GB) == 8);
	}
	SECTION("almost no GPU memory still gives one thread; unknown memory (0) is no limit")
	{
		CHECK(autoThreads(64, 2, 1, 8, 0.1 * GB) == 1);
		CHECK(autoThreads(64, 2, 1, 8, 0) == 8);
	}
	SECTION("memory counts only for GPU jobs")
	{
		CHECK(autoThreads(64, 2, 0, 8, 1.0 * GB) == 32);
	}
	SECTION("work: at least 128 particles per thread")
	{
		CHECK(autoThreads(64, 2, 1, 8, 0, 300) == 2);
		CHECK(autoThreads(64, 2, 1, 8, 0, 50) == 1);
		CHECK(autoThreads(64, 2, 1, 8, 0, 2300) == 8);
		CHECK(autoThreads(64, 2, 0, 8, 0, 300) == 2);   // CPU jobs too
	}
	SECTION("count particles in a STAR file")
	{
		const std::string fn = "test_count_particles.star";
		{
			std::ofstream o(fn);
			o << "# version 30001\n\ndata_optics\n\nloop_\n_rlnOpticsGroup #1\n1\n\n"
			  << "data_particles\n\nloop_\n_rlnImageName #1\n_rlnDefocusU #2\na@x 1\nb@x 2\nc@x 3\n\n";
		}
		CHECK(parhint::countParticlesInStar(fn) == 3);
		CHECK(parhint::countParticlesInStar("no_such_file.star") == 0);
		std::remove(fn.c_str());
	}
}
