/*
 * tests/unit/test_rank_launcher.cpp
 *
 * relion_launch: MPI processes and threads chosen on the computer a job landed on, and the
 * workers per GPU learned from earlier runs (src/rank_launcher.h).
 */

#include <catch2/catch.hpp>

#include "src/rank_launcher.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

using parhint::Work;

namespace
{
const char *SMI =
	"0, GPU-aaaa, NVIDIA RTX A5000, 24564, 300\n"
	"1, GPU-bbbb, NVIDIA RTX A5000, 24564, 8492\n"
	"2, GPU-cccc, NVIDIA RTX A5000, 24564, 0\n";

rlaunch::Node node(int gpus, double gpu_gb, int cores, double host_gb)
{
	rlaunch::Node n;
	n.gpus = gpus;
	n.gpu_name = "NVIDIA RTX A5000";
	n.gpu_bytes = gpu_gb * 1e9;
	n.cores = cores;
	n.host_bytes = host_gb * 1e9;
	return n;
}

std::string joined(const std::vector<std::string> &v)
{
	std::string s;
	for (const std::string &x : v) s += (s.empty() ? "" : " ") + x;
	return s;
}
}

TEST_CASE("relion_launch: the GPUs this job may use", "[rank_launcher]")
{
	rlaunch::Node n;
	rlaunch::parseGpus(SMI, NULL, n);
	CHECK(n.gpus == 3);
	CHECK(n.gpu_name == "NVIDIA RTX A5000");
	CHECK(n.gpu_bytes == Approx((24564. - 8492.) * 1048576.));   // the one with least free memory

	rlaunch::parseGpus(SMI, "0,2", n);                             // indices
	CHECK(n.gpus == 2);
	CHECK(n.gpu_bytes == Approx((24564. - 300.) * 1048576.));
	rlaunch::parseGpus(SMI, "GPU-cccc", n);                        // UUIDs, as some Slurm sites set them
	CHECK(n.gpus == 1);
	rlaunch::parseGpus(SMI, "", n);
	CHECK(n.gpus == 0);
	rlaunch::parseGpus("", NULL, n);                               // no nvidia-smi
	CHECK(n.gpus == 0);
}

TEST_CASE("relion_launch: kind of work from the command line", "[rank_launcher]")
{
	bool vdam = false;
	CHECK(rlaunch::workOfArgs({"--i", "a.star", "--K", "50", "--gpu"}, vdam) == Work::Class2D);
	CHECK_FALSE(vdam);
	CHECK(rlaunch::workOfArgs({"--i", "a.star", "--ref", "r.mrc", "--K", "3"}, vdam) == Work::Class3D);
	CHECK(rlaunch::workOfArgs({"--ref", "r.mrc", "--skip_align"}, vdam) == Work::SkipAlign);
	CHECK(rlaunch::workOfArgs({"--ref", "r.mrc", "--auto_refine", "--split_random_halves"}, vdam) == Work::AutoRefine);
	rlaunch::workOfArgs({"--i", "a.star", "--grad", "--K", "100"}, vdam);
	CHECK(vdam);

	// --continue: from the optimiser STAR file
	char dir[] = "/tmp/rank_launcher_XXXXXX";
	REQUIRE(mkdtemp(dir));
	const std::string fn = std::string(dir) + "/run_it005_optimiser.star";
	FILE *f = fopen(fn.c_str(), "w");
	fputs("data_optimiser_general\n\n_rlnDoAutoRefine                         1\n_rlnDoSkipAlign 0\n", f);
	fclose(f);
	CHECK(rlaunch::workOfArgs({"--continue", fn, "--o", "x"}, vdam) == Work::AutoRefine);
	CHECK_FALSE(vdam);
	unlink(fn.c_str());
	rmdir(dir);
}

TEST_CASE("relion_launch: learning the workers per GPU from E-step speed", "[rank_launcher]")
{
	rlaunch::Record r;
	// cores to spare: one more worker next time
	r = rlaunch::next(r, 2, 0.010, 0.40, 4);
	CHECK(r.next == 3);
	CHECK(r.prev == 2);
	// faster with 3: one more again
	r = rlaunch::next(r, 3, 0.008, 0.50, 3);
	CHECK(r.next == 4);
	// not 3% faster with 4: back to three, and stay
	r = rlaunch::next(r, 4, 0.0079, 0.60, 2);
	CHECK(r.next == 3);
	CHECK(r.settled);
	r = rlaunch::next(r, 3, 0.008, 0.50, 3);
	CHECK(r.next == 3);
	CHECK(r.settled);

	// the cores are the limit with one worker: nothing to try
	rlaunch::Record c = rlaunch::next(rlaunch::Record(), 1, 0.02, 0.90, 8);
	CHECK(c.next == 1);
	CHECK(c.settled);
	// one thread per worker: fewer workers, not more
	CHECK(rlaunch::next(rlaunch::Record(), 4, 0.01, 0.3, 1).next == 3);
	// a start that is too high comes down: 6 -> 5 (faster) -> 4 (faster) -> 3 (not faster) -> back to 4
	rlaunch::Record h = rlaunch::next(rlaunch::Record(), 6, 0.020, 0.95, 1);
	CHECK(h.next == 5);
	h = rlaunch::next(h, 5, 0.017, 0.95, 1);
	CHECK(h.next == 4);
	h = rlaunch::next(h, 4, 0.015, 0.90, 2);
	CHECK(h.next == 3);
	h = rlaunch::next(h, 3, 0.0149, 0.85, 2);
	CHECK(h.next == 4);
	CHECK(h.settled);
	// a down step stops at one worker
	rlaunch::Record o = rlaunch::next(rlaunch::Record(), 2, 0.02, 0.9, 4);
	o = rlaunch::next(o, 1, 0.01, 0.9, 8);
	CHECK(o.next == 1);
	CHECK(o.settled);
	// a count that ran below what was recorded (memory limits) is what is learned from
	rlaunch::Record m;
	m.next = 6;
	m.prev = 5;
	m.prev_time = 0.01;
	const rlaunch::Record mm = rlaunch::next(m, 3, 0.01, 0.9, 2);
	CHECK(mm.next == 2);
	CHECK(mm.prev == 3);
	// nothing measured: keep what ran
	CHECK(rlaunch::next(rlaunch::Record(), 2, 0, 0.3, 4).next == 2);

	rlaunch::Record d;
	REQUIRE(rlaunch::decode(rlaunch::encode(r), d));
	CHECK(d.next == r.next);
	CHECK(d.prev == r.prev);
	CHECK(d.settled == r.settled);
	CHECK_FALSE(rlaunch::decode("garbage", d));

	// the key: same data and search, same key; another K or data set, another key
	const std::string k1 = rlaunch::keyOf({"--i", "a.star", "--ref", "r.mrc", "--K", "3", "--j", "4"});
	CHECK(k1 == rlaunch::keyOf({"relion_refine_mpi", "--i", "a.star", "--ref", "r.mrc", "--K", "3", "--j", "8"}));
	CHECK(k1 != rlaunch::keyOf({"--i", "a.star", "--ref", "r.mrc", "--K", "4"}));
	CHECK(k1 != rlaunch::keyOf({"--i", "b.star", "--ref", "r.mrc", "--K", "3"}));
	CHECK(k1.find("class3d:") == 0);
}

TEST_CASE("relion_launch: the command for the computer it landed on", "[rank_launcher]")
{
	unsetenv("RELION_RANKS_PER_GPU");
	unsetenv("RELION_MPIRUN");
	const std::vector<std::string> cmd = {"/opt/relion/bin/relion_refine_mpi", "--i", "a.star", "--ref", "r.mrc", "--K", "3",
	                                      "--j", "4", "--gpu", ""};
	rlaunch::Command c;
	std::string why;

	SECTION("one 24 GB GPU, 16 cores: the measured target for Class3D, threads from the cores")
	{
		REQUIRE(rlaunch::build(cmd, node(1, 23, 16, 100), 50000, rlaunch::Record(), c, why));
		CHECK(c.advice.workers_per_gpu == 2);
		CHECK(c.advice.mpi == 3);
		CHECK(c.advice.threads == 8);
		CHECK(joined(c.argv) == "mpirun --oversubscribe --bind-to none -n 3 /opt/relion/bin/relion_refine_mpi --i a.star "
		                        "--ref r.mrc --K 3 --gpu  --j 8");
		CHECK(c.report.find("NVIDIA RTX A5000") != std::string::npos);
		CHECK(joined(c.env).find("RELION_LAUNCH_WORKERS_PER_GPU=2") != std::string::npos);
	}
	SECTION("a small card: fewer workers, limited by its free memory")
	{
		REQUIRE(rlaunch::build(cmd, node(1, 5, 16, 100), 50000, rlaunch::Record(), c, why));
		CHECK(c.advice.workers_per_gpu < 4);
	}
	SECTION("a learned count replaces the measured target, the limits still apply")
	{
		rlaunch::Record learned;
		learned.next = 6;
		REQUIRE(rlaunch::build(cmd, node(1, 46, 24, 200), 50000, learned, c, why));
		CHECK(c.advice.workers_per_gpu == 6);
		CHECK(c.report.find("are trying 6") != std::string::npos);   // still stepping
		learned.next = 4; // the same as the measured target: still reported
		learned.settled = true;
		REQUIRE(rlaunch::build(cmd, node(1, 46, 24, 200), 50000, learned, c, why));
		CHECK(c.report.find("settled on 4") != std::string::npos);
		learned.next = 6;
		REQUIRE(rlaunch::build(cmd, node(1, 5, 24, 200), 50000, learned, c, why));
		CHECK(c.advice.workers_per_gpu < 6);
	}
	SECTION("two GPUs: workers on both")
	{
		REQUIRE(rlaunch::build(cmd, node(2, 23, 16, 100), 50000, rlaunch::Record(), c, why));
		CHECK(c.advice.mpi == 2 * c.advice.workers_per_gpu + 1);
	}
	SECTION("VDAM: one process with the job's cores (at most 8 threads)")
	{
		std::vector<std::string> g = {"relion_refine", "--i", "a.star", "--grad", "--K", "100", "--gpu", ""};
		REQUIRE(rlaunch::build(g, node(1, 23, 6, 100), 50000, rlaunch::Record(), c, why));
		CHECK(c.advice.mpi == 1);
		CHECK(joined(c.argv) == "relion_refine --i a.star --grad --K 100 --gpu  --j 6");
	}
	SECTION("not a GPU refinement: run as given")
	{
		CHECK_FALSE(rlaunch::build({"relion_postprocess", "--i", "x"}, node(1, 23, 16, 100), 0, rlaunch::Record(), c, why));
		CHECK_FALSE(rlaunch::build({"relion_refine_mpi", "--i", "a.star", "--ref", "r.mrc"}, node(1, 23, 16, 100), 0,
		                           rlaunch::Record(), c, why));
		CHECK_FALSE(rlaunch::build(cmd, node(0, 0, 16, 100), 0, rlaunch::Record(), c, why));
		CHECK_FALSE(why.empty());
	}
}

TEST_CASE("relion_launch: the record is kept per GPU model and job", "[rank_launcher]")
{
	char dir[] = "/tmp/rank_launcher_XXXXXX";
	REQUIRE(mkdtemp(dir));
	const std::string file = std::string(dir) + "/tuning.csv";
	setenv("RELION_TUNING_FILE", file.c_str(), 1);
	unsetenv("RELION_TUNING_CACHE");

	rlaunch::Record r;
	CHECK_FALSE(rlaunch::load("GPU A", "class3d:1", r));
	r.next = 5;
	REQUIRE(rlaunch::store("GPU A", "class3d:1", r));
	rlaunch::Record back;
	REQUIRE(rlaunch::load("GPU A", "class3d:1", back));
	CHECK(back.next == 5);
	CHECK_FALSE(rlaunch::load("GPU B", "class3d:1", back));
	CHECK_FALSE(rlaunch::load("GPU A", "class3d:2", back));

	// the end of a run that relion_launch started: 2 workers, 600 s of E-steps, 6 of 16 cores busy
	setenv("RELION_LAUNCH_GPU_NAME", "GPU A", 1);
	setenv("RELION_LAUNCH_KEY", "class3d:9", 1);
	setenv("RELION_LAUNCH_WORKERS_PER_GPU", "2", 1);
	resrep::ProcessStats leader, worker;
	leader.wall = worker.wall = 600;
	worker.cpu = 1800;
	worker.threads = 4;
	worker.host = leader.host = "n1";
	worker.cores = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
	worker.gpu_mem_peak = {12e9};   // the card held 12 GB at most: 6 GB per worker
	setenv("RELION_LAUNCH_KIND_KEY", "class3d:any:cores16", 1);
	const std::string line = rlaunch::learnFromRun({leader, worker, worker}, true, 10000, 6);
	CHECK(line.find("10 ms of E-step per particle") != std::string::npos);
	REQUIRE(rlaunch::load("GPU A", "class3d:9", back));
	CHECK(back.next == 3);
	CHECK(back.gpu_gb == Approx(6.0));
	CHECK_FALSE(rlaunch::load("GPU A", "class3d:any:cores16", back));   // nothing settled yet
	// 3 workers were not faster: back to 2, settled, and other jobs of the kind start from 2
	setenv("RELION_LAUNCH_WORKERS_PER_GPU", "3", 1);
	worker.wall = 700;
	worker.gpu_mem_peak = {13.5e9};
	CHECK_FALSE(rlaunch::learnFromRun({leader, worker, worker, worker}, true, 10000, 6).empty());
	REQUIRE(rlaunch::load("GPU A", "class3d:9", back));
	CHECK(back.next == 2);
	CHECK(back.settled);
	CHECK(back.gpu_gb == Approx(4.5));
	REQUIRE(rlaunch::load("GPU A", "class3d:any:cores16", back));
	CHECK(back.next == 2);
	CHECK(back.gpu_gb == 0);   // memory depends on the job: not shared
	unsetenv("RELION_LAUNCH_KIND_KEY");
	setenv("RELION_LAUNCH_WORKERS_PER_GPU", "2", 1);
	worker.wall = 600;
	// not started by relion_launch, too short, or nothing to divide by: not recorded
	worker.wall = 30;
	CHECK(rlaunch::learnFromRun({leader, worker}, true, 10000, 6).empty());
	worker.wall = 600;
	CHECK(rlaunch::learnFromRun({leader, worker}, true, 0, 6).empty());
	unsetenv("RELION_LAUNCH_KEY");
	CHECK(rlaunch::learnFromRun({leader, worker}, true, 10000, 6).empty());
	unsetenv("RELION_LAUNCH_GPU_NAME");
	unsetenv("RELION_LAUNCH_WORKERS_PER_GPU");
	unsetenv("RELION_TUNING_FILE");
	unlink(file.c_str());
	unlink((file + ".lock").c_str());
	rmdir(dir);
}

TEST_CASE("relion_launch: GPUs named with --gpu become the devices the job may use", "[rank_launcher]")
{
	CHECK(rlaunch::visibleForGpuIds("0:1", NULL) == "0,1");
	CHECK(rlaunch::visibleForGpuIds("0,1:2,1", NULL) == "0,1,2");
	CHECK(rlaunch::visibleForGpuIds("1", "") == "1");
	// numbers refer to the devices already visible (as under Slurm)
	CHECK(rlaunch::visibleForGpuIds("1:0", "GPU-aaaa,GPU-bbbb") == "GPU-bbbb,GPU-aaaa");
	CHECK(rlaunch::visibleForGpuIds("2", "3,5") == "");       // not a device this job may use
	CHECK(rlaunch::visibleForGpuIds("", NULL) == "");         // all GPUs
	CHECK(rlaunch::visibleForGpuIds("all", NULL) == "");
}

TEST_CASE("relion_launch: records of the kind of work, and GPU memory measured per job", "[rank_launcher]")
{
	// old four-field records still read; the memory field round-trips
	rlaunch::Record r;
	REQUIRE(rlaunch::decode("3:2:0.01:0", r));
	CHECK(r.gpu_gb == 0);
	r.gpu_gb = 4.25;
	rlaunch::Record back;
	REQUIRE(rlaunch::decode(rlaunch::encode(r), back));
	CHECK(back.gpu_gb == Approx(4.25));

	// the kind key ignores the data and the search, but not the kind of work or the cores per GPU
	const std::vector<std::string> a = {"--i", "a.star", "--ref", "r.mrc", "--K", "3"};
	const std::vector<std::string> b = {"--i", "b.star", "--ref", "q.mrc", "--K", "8"};
	CHECK(rlaunch::kindKeyOf(a, 8) == rlaunch::kindKeyOf(b, 7));
	CHECK(rlaunch::kindKeyOf(a, 8) == "class3d:any:cores8");
	CHECK(rlaunch::kindKeyOf(a, 8) != rlaunch::kindKeyOf(a, 16));
	CHECK(rlaunch::kindKeyOf(a, 1) == "class3d:any:cores2");
	CHECK(rlaunch::kindKeyOf(a, 200) == "class3d:any:cores32");
	CHECK(rlaunch::kindKeyOf({"--i", "a.star", "--K", "50"}, 8) == "class2d:any:cores8");

	const std::vector<std::string> cmd = {"/opt/relion/bin/relion_refine_mpi", "--i", "a.star", "--ref", "r.mrc", "--K", "3", "--gpu", ""};
	rlaunch::Command c;
	std::string why;
	// started from other jobs of the kind: said so
	rlaunch::Record kind;
	kind.next = 3;
	kind.settled = true;
	kind.from_other_jobs = true;
	REQUIRE(rlaunch::build(cmd, node(1, 23, 16, 100), 50000, kind, c, why));
	CHECK(c.advice.workers_per_gpu == 3);
	CHECK(c.report.find("Other jobs of this kind") != std::string::npos);
	CHECK(joined(c.env).find("RELION_LAUNCH_KIND_KEY=class3d:any:cores16") != std::string::npos);
	// this job's measured memory sets the GPU limit: 23 GB x 75% / 8 GB = 2
	rlaunch::Record own;
	own.next = 4;
	own.gpu_gb = 8;
	REQUIRE(rlaunch::build(cmd, node(1, 23, 16, 100), 50000, own, c, why));
	CHECK(c.advice.workers_per_gpu == 2);
	CHECK(c.report.find("as measured in an earlier run of this job") != std::string::npos);
	// ... unless an expert sets the memory per worker
	setenv("RELION_GPU_GB_PER_RANK", "2", 1);
	REQUIRE(rlaunch::build(cmd, node(1, 23, 16, 100), 50000, own, c, why));
	CHECK(c.advice.workers_per_gpu == 4);
	unsetenv("RELION_GPU_GB_PER_RANK");
}

TEST_CASE("relion_launch: MPS on or off, learned per job and workers per GPU", "[rank_launcher]")
{
	rlaunch::MpsRecord none, job, kind;
	std::string why;
	// the rule: on from 4 workers per GPU, never with one
	CHECK(rlaunch::chooseMps(none, none, 8, false, why));
	CHECK_FALSE(rlaunch::chooseMps(none, none, 2, false, why));
	CHECK_FALSE(rlaunch::chooseMps(none, none, 1, true, why));
	// measured one way: the other is tried once the workers per GPU are settled, not before
	job.off = 0.010;
	CHECK_FALSE(rlaunch::chooseMps(job, none, 2, false, why));
	CHECK(rlaunch::chooseMps(job, none, 2, true, why));
	CHECK(why.find("trying it on once") != std::string::npos);
	// measured both ways: the faster
	job.on = 0.0106;
	CHECK_FALSE(rlaunch::chooseMps(job, none, 2, true, why));
	job.on = 0.004;
	CHECK(rlaunch::chooseMps(job, none, 2, false, why));
	// a job without its own record follows its kind, measured both ways
	kind.on = 0.02;
	kind.off = 0.01;
	CHECK_FALSE(rlaunch::chooseMps(none, kind, 8, false, why));
	rlaunch::MpsRecord back;
	REQUIRE(rlaunch::decodeMps(rlaunch::encodeMps(job), back));
	CHECK(back.on == Approx(job.on));
	CHECK(back.off == Approx(job.off));

	char dir[] = "/tmp/rank_launcher_XXXXXX";
	REQUIRE(mkdtemp(dir));
	const std::string file = std::string(dir) + "/tuning.csv";
	setenv("RELION_TUNING_FILE", file.c_str(), 1);
	unsetenv("RELION_TUNING_CACHE");
	unsetenv("RELION_AUTO_MPS");
	unsetenv("RELION_RANKS_PER_GPU");

	// the command: Class3D on one 24 GB card gets 2 workers, so MPS starts off
	const std::vector<std::string> cmd = {"relion_refine_mpi", "--i", "a.star", "--ref", "r.mrc", "--K", "3", "--gpu", ""};
	const rlaunch::Node n = node(1, 23, 16, 100);
	rlaunch::Command c;
	rlaunch::Record settled;
	settled.next = 2;
	settled.settled = true;
	REQUIRE(rlaunch::build(cmd, n, 50000, settled, c, why));
	rlaunch::applyMps(c, n, cmd, settled);
	CHECK(joined(c.env).find("RELION_AUTO_MPS=off") != std::string::npos);
	CHECK(c.report.find("MPS off: fewer than 4 workers per GPU") != std::string::npos);

	// the end of that run, without MPS: recorded, and the next run tries it on
	for (const std::string &e : c.env) putenv(strdup(e.c_str()));
	resrep::ProcessStats leader, worker;
	leader.wall = worker.wall = 600;
	worker.cpu = 600;
	worker.threads = 4;
	worker.host = leader.host = "n1";
	worker.cores = {0, 1, 2, 3, 4, 5, 6, 7};
	CHECK(rlaunch::learnFromRun({leader, worker, worker}, true, 10000, 6).find("without MPS") != std::string::npos);
	unsetenv("RELION_AUTO_MPS");
	REQUIRE(rlaunch::build(cmd, n, 50000, settled, c, why));
	rlaunch::applyMps(c, n, cmd, settled);
	CHECK(joined(c.env).find("RELION_AUTO_MPS=on") != std::string::npos);
	// with MPS it was slower: recorded for the job and its kind, and the next run goes without
	for (const std::string &e : c.env) putenv(strdup(e.c_str()));
	worker.mps = true;
	worker.wall = 640;
	CHECK(rlaunch::learnFromRun({leader, worker, worker}, true, 10000, 6).find("with MPS") != std::string::npos);
	REQUIRE(rlaunch::loadMps("NVIDIA RTX A5000", "class3d:any:cores16:w2", back));
	CHECK(back.on > back.off);
	unsetenv("RELION_AUTO_MPS");
	REQUIRE(rlaunch::build(cmd, n, 50000, settled, c, why));
	rlaunch::applyMps(c, n, cmd, settled);
	CHECK(joined(c.env).find("RELION_AUTO_MPS=off") != std::string::npos);
	CHECK(c.report.find("faster for this job") != std::string::npos);

	// an expert's setting is obeyed and nothing is learned
	setenv("RELION_AUTO_MPS", "on", 1);
	rlaunch::Command e;
	REQUIRE(rlaunch::build(cmd, n, 50000, settled, e, why));
	rlaunch::applyMps(e, n, cmd, settled);
	CHECK(joined(e.env).find("RELION_LAUNCH_MPS_KEY") == std::string::npos);
	CHECK(e.report.find("as set by RELION_AUTO_MPS=on") != std::string::npos);

	for (const char *v : {"RELION_AUTO_MPS", "RELION_LAUNCH_MPS_KEY", "RELION_LAUNCH_MPS_KIND_KEY", "RELION_LAUNCH_GPU_NAME",
	                      "RELION_LAUNCH_KEY", "RELION_LAUNCH_KIND_KEY", "RELION_LAUNCH_WORKERS_PER_GPU", "RELION_TUNING_FILE"})
		unsetenv(v);
	unlink(file.c_str());
	unlink((file + ".lock").c_str());
	rmdir(dir);
}
