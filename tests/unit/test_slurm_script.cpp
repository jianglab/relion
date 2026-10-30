/*
 * tests/unit/test_slurm_script.cpp
 *
 * RELION's own Slurm submission script (src/slurm_script.h).
 */

#include <catch2/catch.hpp>

#include "src/slurm_script.h"

#include <string>

using slurmscript::Request;

static const char *NODES =
	"4 128 224310 gpu:typeA:3\n"
	"1 80 256589 gpu:typeB:3\n"
	"2 64 127500 gpu:typeC:1\n"
	"3 128 256000 gpu:typeD:4(S:0-1)\n"
	"1 16 127500 gpu:typeE:1\n"
	"1 48 514000 gpu:typeF:3\n";

static Request request(int mpi, int threads, int gpus, bool leader_idle, std::string mem = "auto")
{
	Request r;
	r.partition = "q";
	r.mpi = mpi;
	r.threads = threads;
	r.gpus = gpus;
	r.leader_idle = leader_idle;
	r.mem = mem;
	return r;
}

TEST_CASE("Slurm script: node list from sinfo", "[slurm]")
{
	auto nodes = slurmscript::parseSinfo(NODES);
	REQUIRE(nodes.size() == 6);
	CHECK(nodes[3].count == 3);
	CHECK(nodes[3].cpus == 128);
	CHECK(nodes[3].mem_mb == 256000);
	CHECK(nodes[3].gpus == 4);
	CHECK(slurmscript::parseSinfo("2 20+ 64000 (null)\n")[0].gpus == 0);
	CHECK(slurmscript::parseSinfo("").empty());
}

TEST_CASE("Slurm script: queues and nodes to choose from", "[slurm]")
{
	auto parts = slurmscript::parsePartitions("typeA* up 10/2/0/12\ngpu up 6/0/0/6\nold down 0/0/1/1\nbad;name up 1/1/0/2\n");
	REQUIRE(parts.size() == 3);   // a name sinfo could not have given is dropped
	CHECK(parts[0].name == "typeA");
	CHECK(parts[0].is_default);
	CHECK(parts[0].idle == 2);
	CHECK(parts[0].total == 12);
	CHECK(parts[0].label() == "typeA (default): 2 of 12 nodes idle");
	CHECK_FALSE(parts[1].is_default);
	CHECK(parts[1].label() == "gpu: 0 of 6 nodes idle");
	CHECK_FALSE(parts[2].up);
	CHECK(parts[2].label() == "old: down");
	CHECK(slurmscript::parsePartitions("").empty());

	// one line per node and partition: each node once
	auto nodes = slurmscript::parseNodes("n1 mix gpu:typeD:4\nn2 idle (null)\nn1 mix gpu:typeD:4\n");
	REQUIRE(nodes.size() == 2);
	CHECK(nodes[0].name == "n1");
	CHECK(nodes[0].state == "mix");
	CHECK(nodes[0].gres == "gpu:typeD:4");
	CHECK(nodes[1].gres.empty());
}

TEST_CASE("Slurm script: one node, memory the job's share of the smallest node that fits", "[slurm]")
{
	auto nodes = slurmscript::parseSinfo(NODES);
	// 8 cores and 1 GPU: on typeD a GPU is a quarter of the node
	auto l = slurmscript::plan(request(1, 8, 1, true), nodes);
	REQUIRE(l.ok);
	CHECK(l.one_node);
	CHECK(l.tasks == 1);
	CHECK(l.mem_mb == 62 * 1024);   // 64000 MB, rounded down to whole GB
	CHECK_FALSE(l.mem_per_cpu);
	// a refinement: the leader runs on its workers' cores
	l = slurmscript::plan(request(3, 4, 1, true), nodes);
	CHECK(l.tasks == 2);
	// other programs: every process is a task
	l = slurmscript::plan(request(4, 1, 0, false), nodes);
	CHECK(l.tasks == 4);
	// a CPU refinement on one node: by its share of the cores, on the node type where that is least
	l = slurmscript::plan(request(1, 16, 0, true), nodes);
	CHECK(l.one_node);
	CHECK(l.mem_mb == 27 * 1024);   // 28038 MB
	// other CPU programs: Slurm places them; memory per core of the node type with the least
	l = slurmscript::plan(request(1, 16, 0, false), nodes);
	CHECK_FALSE(l.one_node);
	CHECK(l.mem_per_cpu);
	CHECK(l.mem_mb == 1700);   // 1752 MB per core, rounded down to 100 MB
	// memory given by the user
	l = slurmscript::plan(request(1, 8, 1, true, "12"), nodes);
	CHECK(l.mem_mb == 12 * 1024);
	CHECK_FALSE(slurmscript::plan(request(1, 8, 1, true, "lots"), nodes).ok);
}

TEST_CASE("Slurm script: GPU jobs on one node, CPU jobs spread when they must", "[slurm]")
{
	auto nodes = slurmscript::parseSinfo(NODES);
	// 5 GPUs: no node has them
	auto l = slurmscript::plan(request(6, 4, 5, true), nodes);
	CHECK_FALSE(l.ok);
	CHECK(l.error.find("runs GPU jobs on one node") != std::string::npos);
	// 300 CPU cores: no node has them, so several nodes; every process a task, memory per core
	l = slurmscript::plan(request(30, 10, 0, true), nodes);
	REQUIRE(l.ok);
	CHECK_FALSE(l.one_node);
	CHECK(l.tasks == 30);
	CHECK(l.mem_per_cpu);
	CHECK(l.mem_mb == 1700);   // the node type with the least memory per core
	// more cores than the queue has
	l = slurmscript::plan(request(400, 8, 0, false), nodes);
	CHECK_FALSE(l.ok);
	CHECK(l.error.find("Not enough CPU cores") != std::string::npos);
	// no node list: one node, 4 GB per core
	l = slurmscript::plan(request(3, 4, 1, true), {});
	REQUIRE(l.ok);
	CHECK(l.mem_mb == 2 * 4 * 4096);
}

TEST_CASE("Slurm script: text", "[slurm]")
{
	auto nodes = slurmscript::parseSinfo(NODES);
	Request r = request(1, 8, 1, true);
	std::string t = slurmscript::templateText(r, slurmscript::plan(r, nodes), "");
	CHECK(t.find("#SBATCH --nodes=1\n") != std::string::npos);
	CHECK(t.find("#SBATCH --ntasks=1\n") != std::string::npos);
	CHECK(t.find("#SBATCH --partition=XXXqueueXXX\n") != std::string::npos);
	CHECK(t.find("#SBATCH --gpus=1\n") != std::string::npos);
	CHECK(t.find("#SBATCH --mem=62G\n") != std::string::npos);
	CHECK(t.find("\nXXXcommandXXX\n") != std::string::npos);
	CHECK(t.find("mpirun") == std::string::npos);
	CHECK(t.find("--time") == std::string::npos);
	CHECK(t.find("--exclude") == std::string::npos);
	r.exclude = "n3, n7";
	t = slurmscript::templateText(r, slurmscript::plan(r, nodes), "");
	CHECK(t.find("#SBATCH --exclude=n3,n7\n") != std::string::npos);
	r.exclude.clear();

	r.exports.push_back("RELION_EXAMPLE_SWITCH=1");
	t = slurmscript::templateText(r, slurmscript::plan(r, nodes), "");
	CHECK(t.find("export RELION_EXAMPLE_SWITCH=1\nXXXcommandXXX\n") != std::string::npos);

	r = request(5, 4, 0, true);
	r.time = "2:00:00";
	r.args = "--account=abc --qos=high";
	t = slurmscript::templateText(r, slurmscript::plan(r, nodes), "module load openmpi");
	CHECK(t.find("#SBATCH --ntasks=4\n") != std::string::npos);
	CHECK(t.find("--gpus") == std::string::npos);
	CHECK(t.find("#SBATCH --time=2:00:00\n") != std::string::npos);
	CHECK(t.find("#SBATCH --account=abc --qos=high\n") != std::string::npos);
	CHECK(t.find("module load openmpi\nmpirun --oversubscribe --bind-to none -n XXXmpinodesXXX XXXcommandXXX\n") != std::string::npos);

	// every process working: no oversubscription, and Slurm chooses the nodes
	r = request(4, 2, 0, false);
	r.mem = "16";
	t = slurmscript::templateText(r, slurmscript::plan(r, nodes), "");
	CHECK(t.find("#SBATCH --ntasks=4\n") != std::string::npos);
	CHECK(t.find("--nodes") == std::string::npos);
	CHECK(t.find("#SBATCH --mem-per-cpu=2000M\n") != std::string::npos);
	CHECK(t.find("mpirun --bind-to none -n XXXmpinodesXXX") != std::string::npos);

	// several nodes: no --nodes=1, memory per core
	r = request(30, 10, 0, true);
	t = slurmscript::templateText(r, slurmscript::plan(r, nodes), "");
	CHECK(t.find("--nodes") == std::string::npos);
	CHECK(t.find("#SBATCH --ntasks=30\n") != std::string::npos);
	CHECK(t.find("#SBATCH --mem-per-cpu=1700M\n") != std::string::npos);
	CHECK(t.find("--oversubscribe") == std::string::npos);
}

TEST_CASE("Slurm script: auto parallel GPU refinements run through relion_launch", "[slurm]")
{
	auto nodes = slurmscript::parseSinfo(NODES);
	Request r = request(1, 24, 1, true);
	r.launch = true;
	const std::string t = slurmscript::templateText(r, slurmscript::plan(r, nodes), "");
	CHECK(t.find("#SBATCH --ntasks=1\n") != std::string::npos);
	CHECK(t.find("#SBATCH --cpus-per-task=24\n") != std::string::npos);   // all the job's cores, not XXXthreadsXXX
	CHECK(t.find("#SBATCH --gpus=1\n") != std::string::npos);
	CHECK(t.find("relion_launch XXXcommandXXX\n") != std::string::npos);
	CHECK(t.find("mpirun") == std::string::npos);
}
