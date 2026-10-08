/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef RESOURCE_REPORT_H_
#define RESOURCE_REPORT_H_

/*
 * How well a run uses the CPUs and GPUs it was given, and what to change.
 *
 * Each process measures, over a phase (an E-step): wall time, its CPU time, the
 * CPU cores it may run on, its threads, and how busy its GPUs were (sampled
 * through NVML, loaded at run time; absent NVML just means no GPU numbers).
 * The leader combines the processes per host and turns the numbers into
 * advice: a starved GPU wants more processes or threads to feed it, a saturated
 * GPU does not need the CPUs it has, and allocated cores that no thread uses are
 * wasted. Busy CPU threads alone prove little on GPU runs, since CUDA may spin
 * while it waits for the GPU, so GPU runs are judged by the GPU.
 *
 * RELION_RESOURCE_REPORT=off silences the report.
 */

#include <atomic>
#include <string>
#include <thread>
#include <vector>

namespace resrep
{

/// What one process measured over one phase.
struct ProcessStats
{
	std::string host;
	int    threads = 1;             ///< --j
	bool   uses_gpu = false;
	double wall = 0.;               ///< seconds
	double cpu = 0.;                ///< seconds of CPU time (user + system)
	std::vector<int> cores;         ///< CPU cores this process may run on
	std::vector<std::string> gpus;  ///< PCI bus IDs of its GPUs
	std::vector<double> gpu_busy;   ///< per GPU: fraction of the time a kernel ran (-1: unknown)
	bool   mps = false;             ///< an NVIDIA MPS daemon was running (processes share the GPU concurrently)
};

/// Whether an NVIDIA MPS control daemon is running for this process's environment.
bool mpsRunning();

/// One line with the measured numbers per host (GPU busy, cores busy).
std::vector<std::string> summary(const std::vector<ProcessStats> &procs, bool mpi);

/// Add one phase to a running total (GPU use weighted by time), and turn the
/// total into averages once all phases are in.
void accumulate(ProcessStats &total, const ProcessStats &phase);
void finish(ProcessStats &total);

/// The advice, one line per point; empty when the run is balanced.
std::vector<std::string> advise(const std::vector<ProcessStats> &procs, bool mpi);

/// CPU cores this process may run on (sched_getaffinity).
std::vector<int> allowedCores();

/// False when RELION_RESOURCE_REPORT is off.
bool enabled();

/// Measures one process over a phase; GPU utilisation is sampled in a thread.
class Monitor
{
public:
	~Monitor();
	void start(const std::vector<std::string> &gpu_pci_ids);
	ProcessStats stop(int threads);

private:
	void sample();
	std::vector<std::string> pci_;
	std::vector<void*> handles_;
	std::vector<double> sum_;
	std::vector<long> n_;
	std::atomic<bool> running_{false};
	std::thread thread_;
	double t0_ = 0., cpu0_ = 0.;
};

/// One-line serialisation, to gather the processes' stats over MPI.
std::string serialise(const ProcessStats &s);
ProcessStats deserialise(const std::string &s);

}

#endif
