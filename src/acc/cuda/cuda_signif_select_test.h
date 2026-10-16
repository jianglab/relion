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

}

#endif
