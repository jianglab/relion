/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef CUDA_SIGNIF_SELECT_TEST_H_
#define CUDA_SIGNIF_SELECT_TEST_H_

#include <vector>

namespace gpusel
{

struct Outcome
{
	float sum_weight = 0, threshold = 0, significant_weight = 0, max_value = 0;
	unsigned long long idx = 0, idx_capped = 0;
	int max_key = 0;
	std::vector<float> weights;   // exponentiated weights
	std::vector<char> mask;       // significance mask
};

// The fused weight pass (CudaKernels::fusedWeightPass) on the given coarse log-weights, with the
// sort-and-scan significance (radix = false) or the radix selection (radix = true).
// False if no GPU is usable.
bool runFusedWeightPass(const std::vector<float> &log_weights, double adaptive_fraction, long max_significants,
		bool radix, Outcome &out);

struct StressOutcome
{
	long launches = 0;       // selections completed over all threads
	long mismatches = 0;     // results that differ from the first result of the same input
	bool stalled = false;    // no thread made progress for stall_seconds: a deadlock
	bool cooperative = false;// the card supports cooperative launches
	long fused_launches = 0; // selections that ran as one cooperative launch
	long fallbacks = 0;      // selections that fell back to the separate launches
};

// nthreads host threads, each with its own stream, run the radix selection (RELION_GPU_SELECT_FUSED as set in the
// environment) on n weights for `seconds`; with compete = true another stream keeps the multiprocessors busy with long
// kernels. A watchdog reports a stall instead of hanging. False if no GPU is usable.
bool stressRadixSelect(int nthreads, double seconds, size_t n, bool compete, double stall_seconds, StressOutcome &out);

struct OnceOutcome
{
	float sum_weight = 0, threshold = 0, significant_weight = 0;
	unsigned long long idx = 0, idx_capped = 0;
	long fused_launches = 0, fallbacks = 0;
};

// One radix selection on the given weights; with refuse = true the cooperative launch is made to fail, so the separate
// launches run. False if no GPU is usable.
bool radixSelectOnce(const std::vector<float> &weights, bool refuse, OnceOutcome &out);

}

#endif
