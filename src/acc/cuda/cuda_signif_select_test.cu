/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

// Host entry point for the unit tests of the significance selection: src/acc/cuda/cuda_signif_select_test.h

#include "src/acc/cuda/cuda_signif_select_test.h"
#include "src/acc/acc_ptr.h"
#include "src/acc/cuda/custom_allocator.cuh"
#include "src/acc/cuda/cuda_utils_cub.cuh"

namespace gpusel
{

bool runFusedWeightPass(const std::vector<float> &log_weights, double adaptive_fraction, long max_significants,
		bool radix, Outcome &out)
{
	int ndev = 0;
	if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1)
	{
		cudaGetLastError();
		return false;
	}
	typedef CudaKernels::SignificanceResult<float> SigRes;
	const size_t n = log_weights.size();
	CudaCustomAllocator pool(0, 512);
	pool.configureLean((size_t) 512 << 20, (size_t) 128 << 20);
	AccPtrFactory factory(&pool, cudaStreamPerThread);

	AccPtr<float> w = factory.make<float>(n);
	w.allAlloc();
	for (size_t i = 0; i < n; i++)
		w[i] = log_weights[i];
	w.cpToDevice();

	AccPtr<char> res = factory.make<char>(sizeof(SigRes) + n);
	res.allAlloc();

	CudaKernels::gpuSignifSelectOverride() = radix ? 1 : 0;
	CudaKernels::fusedWeightPass<float>(w, adaptive_fraction, max_significants, ~res, true);
	CudaKernels::gpuSignifSelectOverride() = -1;
	res.cpToHostWait();
	w.cpToHost();
	cudaStreamSynchronize(cudaStreamPerThread);

	SigRes r;
	memcpy(&r, &res[0], sizeof(SigRes));
	out.sum_weight = r.sum_weight;
	out.threshold = r.threshold;
	out.significant_weight = r.significant_weight;
	out.max_value = r.max.value;
	out.max_key = r.max.key;
	out.idx = r.idx;
	out.idx_capped = r.idx_capped;
	out.weights.assign(&w[0], &w[0] + n);
	out.mask.assign(&res[sizeof(SigRes)], &res[sizeof(SigRes)] + n);
	return true;
}

}
