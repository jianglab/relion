#ifndef CUDA_ML_OPTIMISER_H_
#define CUDA_ML_OPTIMISER_H_

#include "src/mpi.h"
#include "src/ml_optimiser.h"
#include "src/acc/cuda/cuda_mem_utils.h"
#include "src/acc/cuda/gpu_mem_trace.h"
#include "src/acc/acc_projector_plan.h"
#include "src/acc/acc_projector.h"
#include "src/acc/acc_backprojector.h"
#include "src/acc/acc_bp_accum.h"
#include "src/acc/cuda/coarse_cache.h"
#include "src/acc/cuda/cuda_fft.h"
#include "src/acc/cuda/cuda_benchmark_utils.h"
#include <stack>
//#include <cufft.h>

#include "src/acc/acc_ml_optimiser.h"
#include "src/acc/acc_ptr.h"

class MlDeviceBundle
{
public:

	//The CUDA accelerated projector set
	std::vector< AccProjector > projectors;

	//The CUDA accelerated back-projector set
	std::vector< AccBackprojector > backprojectors;

	//Used for precalculations of projection setup
	CudaCustomAllocator *allocator;

	//Used for precalculations of projection setup
	bool generateProjectionPlanOnTheFly;
	std::vector< AccProjectorPlan > coarseProjectionPlans;

	MlOptimiser *baseMLO;

	int device_id;

	int rank_shared_count;

	bool haveWarnedRefinementMem;

	// Hand the allocator and its chunk of GPU memory on to the next bundle on this device instead of
	// freeing and allocating them every iteration (a cudaMalloc of the chunk takes 0.1-0.3 s)
	bool keep_allocator = false;

	// Bytes of GPU memory held for the next iteration (0 when none)
	size_t retainedBytes() const;

	MlDeviceBundle(MlOptimiser *baseMLOptimiser):
			baseMLO(baseMLOptimiser),
			generateProjectionPlanOnTheFly(false),
			rank_shared_count(1),
			device_id(-1),
			haveWarnedRefinementMem(false),
			allocator(NULL)
	{};

	void setDevice(int did)
	{
		device_id = did;
	}

	size_t checkFixedSizedObjects(int shares);
	void setupFixedSizedObjects();
	void setupTunableSizedObjects(size_t allocationSize);

	// Lean GPU memory pool (RELION_GPU_LEAN_MEMORY): see cuda_ml_optimiser.cu
	std::string lean_key; // description of the problem of this iteration, for the tuning cache
	void setupLeanPool(size_t cap);
	void finishLeanIteration();

	void syncAllBackprojects()
	{
		DEBUG_HANDLE_ERROR(cudaDeviceSynchronize());
	}

	// Per-orientation accumulation for the back-projection of 3D references (see BPAccumView):
	// sized and put in the allocator's chunk by setupBackprojectAccumulator() when the
	// iteration is a global search with images against 3D references, spread into the
	// volumes and freed by flushBackprojectAccumulator()
	struct BPAccumulator
	{
		CudaCustomAllocator::Alloc *block = NULL;
		BPAccumView view = BPAccumView();
		size_t slots = 0;           // slots (class, orientation) in all
		unsigned slots_per_class = 0;
		unsigned nr_over = 0;       // oversampled rotations per orientation the slots were made for
		unsigned img_x = 0, img_y = 0;
		BPModelPtrs *d_models = NULL; // in the block
		size_t bytes = 0;
	} bpacc;
	void setupBackprojectAccumulator();
	void flushBackprojectAccumulator();

	// The reference projections of the coarse pass, made on the first use in an iteration (RELION_GPU_COARSE_CACHE)
	CoarseProjectionCache coarse_cache;

	// Make this device current on the calling thread (worker threads start on device 0)
	void bindThreadToDevice()
	{
		cudaSetDevice(device_id);
	}


	~MlDeviceBundle()
	{
		gpuMemTrace("iteration end, before the bundle is released");
		coarse_cache.release();
		finishLeanIteration();
		projectors.clear();
		backprojectors.clear();
		coarseProjectionPlans.clear();
		//Delete this lastly
		if (keep_allocator && allocator != NULL && retainAllocator(device_id, allocator))
			allocator = NULL;
		delete allocator;
		gpuMemTrace("bundle released");
	}

	static bool retainAllocator(int device, CudaCustomAllocator *a);

};
// Frees the allocators kept between iterations, for code that needs the GPU memory (the GPU
// reconstruction): the next iteration then allocates a new one
void relionReleaseRetainedGpuMemory();

class MlOptimiserCuda
{
public:
	// transformer as holder for reuse of fftw_plans
	FourierTransformer transformer;

   //Class streams ( for concurrent scheduling of class-specific kernels)
	std::vector< cudaStream_t > classStreams;
	bool classStreamsShared = false; // all classStreams are cudaStreamPerThread (see resetData)

	// Wait for the work on the class streams of classes first..last and on this thread's stream
	void syncClassStreams(long first, long last)
	{
		if (!classStreamsShared)
			for (long c = first; c <= last; c++)
				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(classStreams[c]));
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread));
	}
	cudaError_t errorStatus;

	CudaFFT transformer1;
	CudaFFT transformer2;

	MlOptimiser *baseMLO;

	bool refIs3D;
	bool dataIs3D;
    bool shiftsIs3D;

	int device_id;

	MlDeviceBundle *bundle;

	//Used for precalculations of projection setup
	CudaCustomAllocator *allocator;

	//Used for precalculations of projection setup
	bool generateProjectionPlanOnTheFly;


#ifdef TIMING_FILES
	relion_timer timer;
#endif

	MlOptimiserCuda(MlOptimiser *baseMLOptimiser, MlDeviceBundle* bundle, const char * timing_fnm) :
			baseMLO(baseMLOptimiser),
			transformer1(cudaStreamPerThread, bundle->allocator, baseMLOptimiser->mymodel.data_dim),
			transformer2(cudaStreamPerThread, bundle->allocator, baseMLOptimiser->mymodel.data_dim),
			refIs3D(baseMLO->mymodel.ref_dim == 3),
			dataIs3D(baseMLO->mymodel.data_dim == 3),
            shiftsIs3D(baseMLO->mymodel.data_dim == 3 || baseMLO->mydata.is_tomo),
			bundle(bundle),
			device_id(bundle->device_id),
#ifdef TIMING_FILES
			timer(timing_fnm),
#endif
			errorStatus((cudaError_t)0),
			allocator(bundle->allocator),
			generateProjectionPlanOnTheFly(bundle->generateProjectionPlanOnTheFly)
	{};

	void resetData();

	void doThreadExpectationSomeParticles(int thread_id);

	~MlOptimiserCuda()
	{
		for (int i = 0; i < classStreams.size(); i++)
			if (classStreams[i] != NULL && classStreams[i] != cudaStreamPerThread)
				HANDLE_ERROR(cudaStreamDestroy(classStreams[i]));
	}

	CudaCustomAllocator *getAllocator()	
	{
		return (bundle->allocator);
	};
	
};

#endif
