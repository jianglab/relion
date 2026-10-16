#ifndef COARSE_CACHE_H_
#define COARSE_CACHE_H_

#include <cuda_runtime.h>
#include <mutex>
#include <vector>
#include "src/macros.h"
#include "src/acc/cuda/custom_allocator.cuh"

/*
 * The reference projections of the coarse pass of a 3D classification: for the K pixels of the coarse mask and all
 * orientations of all classes (the same for every particle of a global search), made once per iteration and kept on the
 * device (layout R[k][orientation], orientation fastest), so that each particle's coarse differences become one pass
 * over this table instead of K * O texture projections (see cuda_kernels/coarse_cache.cuh).
 * It belongs to the MlDeviceBundle of the iteration and takes its memory from the bundle's allocator.
 */
class CoarseProjectionCache
{
public:
	std::mutex mtx;
	bool tried = false;   // a build was attempted (it is not repeated within the iteration)
	bool ok = false;      // the table is built and usable

	// What the table was made for
	int image_size = 0;                      // pixels of the coarse image
	int K = 0;                               // pixels of the mask
	std::vector<unsigned char> mask;         // per pixel of the image
	std::vector<int> class_ids;
	std::vector<const void *> eulers;        // device pointers of the classes' Euler matrices
	std::vector<size_t> n_orient, offset;    // per class
	size_t O_total = 0;

	CudaCustomAllocator *allocator = NULL;
	CudaCustomAllocator::Alloc *block = NULL;
	int *d_pix = NULL;                       // in the block: the K pixels
	float2 *d_R = NULL;                      // in the block: K * O_total
	size_t bytes = 0;

	// Run-time choice between kernel shapes (see CoarseShapeTuner in coarse_cache.cuh)
	void *tuner = NULL;

	~CoarseProjectionCache()
	{
		release();
	}

	void release()
	{
		if (block != NULL && allocator != NULL)
		{
			cudaDeviceSynchronize();
			allocator->free(block);
		}
		block = NULL;
		ok = false;
	}
};

// What a coarse pass tells the cache about its inputs
struct CoarseCacheCtx
{
	CoarseProjectionCache *cache = NULL;
	CudaCustomAllocator *allocator = NULL;
	std::vector<int> class_ids;
	const RFLOAT *host_minvsigma2 = NULL;   // support of the weights of the pass (same for all particles of an optics group)
	int verb = 0;
	int device = 0;
};

#endif /* COARSE_CACHE_H_ */
