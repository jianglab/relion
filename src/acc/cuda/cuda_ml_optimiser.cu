#undef ALTCPU
#include <sys/time.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <ctime>
#include <vector>
#include <iostream>
#include "src/ml_optimiser.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "src/acc/acc_ptr.h"
#include "src/acc/acc_projector.h"
#include "src/acc/acc_backprojector.h"
#include "src/acc/acc_projector_plan.h"
#include "src/acc/cuda/cuda_benchmark_utils.h"
#include "src/acc/cuda/cuda_kernels/helper.cuh"
#include "src/acc/cuda/cuda_kernels/diff2.cuh"
#include "src/acc/cuda/cuda_kernels/wavg.cuh"
#include "src/acc/cuda/cuda_mem_utils.h"
#include "src/acc/group_size_tuner.h"
#include "src/acc/cuda/cuda_fft.h"
#include "src/acc/data_types.h"
#include "src/complex.h"
#include "src/helix.h"
#include "src/error.h"
#include <fstream>
#include "src/parallel.h"
#include <signal.h>
#include <map>
#include <mutex>
#include <chrono>
#include <algorithm>

#ifdef CUDA_FORCESTL
#include "src/acc/cuda/cuda_utils_stl.cuh"
#else
#include "src/acc/cuda/cuda_utils_cub.cuh"
#endif

#include "src/acc/utilities.h"
#include "src/acc/utilities_impl.h"

#include "src/acc/acc_ml_optimiser.h"
#include "src/acc/cuda/cuda_ml_optimiser.h"
#include "src/acc/acc_helper_functions.h"
#include "src/acc/acc_ml_optimiser_impl.h"

// -------------------------------  Some explicit template instantiations
template __global__ void CudaKernels::cuda_kernel_translate2D<XFLOAT>(XFLOAT *,
    XFLOAT*, int, int, int, int, int);

template __global__ void CudaKernels::cuda_kernel_translate3D<XFLOAT>(XFLOAT *,
    XFLOAT *, int, int, int, int, int, int, int);

template __global__ void cuda_kernel_multi<XFLOAT>( XFLOAT *,
	XFLOAT *, XFLOAT, int);

template __global__ void CudaKernels::cuda_kernel_multi<XFLOAT>( XFLOAT *,
	XFLOAT, int);

template __global__ void cuda_kernel_multi<XFLOAT>( XFLOAT *, XFLOAT *,
	XFLOAT *, XFLOAT, int);

// ----------------------------------------------------------------------

// High-level CUDA objects

// Allocators (with their chunk of GPU memory) kept from one iteration to the next, per device.
// Never destroyed at exit: the CUDA context is gone by then.
namespace
{
	std::map<int, CudaCustomAllocator*> &retainedAllocators()
	{
		static std::map<int, CudaCustomAllocator*> *m = new std::map<int, CudaCustomAllocator*>();
		return *m;
	}
	omp_lock_t *retainedLock()
	{
		static omp_lock_t *l = [](){ omp_lock_t *x = new omp_lock_t; omp_init_lock(x); return x; }();
		return l;
	}
	bool keepAllocatorAllowed()
	{
		const char *e = getenv("RELION_GPU_KEEP_ALLOCATOR");
		return e == NULL || atoi(e) != 0;
	}
}

bool MlDeviceBundle::retainAllocator(int device, CudaCustomAllocator *a)
{
	if (!keepAllocatorAllowed() || a->chunkSize() == 0 || a->getNumberOfAllocs() != 0)
		return false;
	omp_set_lock(retainedLock());
	CudaCustomAllocator *old = NULL;
	auto it = retainedAllocators().find(device);
	if (it != retainedAllocators().end())
		old = it->second;
	retainedAllocators()[device] = a;
	omp_unset_lock(retainedLock());
	if (old != NULL && old != a)
		delete old;
	return true;
}

size_t MlDeviceBundle::retainedBytes() const
{
	return allocator == NULL ? 0 : allocator->chunkSize();
}

void relionReleaseRetainedGpuMemory()
{
	std::map<int, CudaCustomAllocator*> mine;
	omp_set_lock(retainedLock());
	mine.swap(retainedAllocators());
	omp_unset_lock(retainedLock());
	if (mine.empty())
		return;
	int dev = 0;
	cudaGetDevice(&dev);
	for (auto &kv : mine)
	{
		cudaSetDevice(kv.first);
		delete kv.second;
	}
	cudaSetDevice(dev);
}

// How a host thread waits for the GPU. RELION_GPU_SYNC_MODE: auto (driver default), spin, yield, blocking.
// Spinning waiters keep a hardware thread busy, which slows a hyperthread sibling that does real work.
static void applyGpuSyncMode(int verb)
{
	static bool done = false;
	if (done)
		return;
	done = true;
	const char *e = getenv("RELION_GPU_SYNC_MODE");
	if (!e || !strcmp(e, "auto"))
		return;
	unsigned int flags;
	if (!strcmp(e, "spin")) flags = cudaDeviceScheduleSpin;
	else if (!strcmp(e, "yield")) flags = cudaDeviceScheduleYield;
	else if (!strcmp(e, "blocking")) flags = cudaDeviceScheduleBlockingSync;
	else return;
	cudaError_t err = cudaSetDeviceFlags(flags);
	if (verb > 0)
		printf(" GPU wait mode: %s%s\n", e, err == cudaSuccess ? "" : " (could not be set, context already exists)");
}

size_t MlDeviceBundle::checkFixedSizedObjects(int shares)
{
	int devCount;
	size_t BoxLimit;
	HANDLE_ERROR(cudaGetDeviceCount(&devCount));
	if(device_id >= devCount)
		CRITICAL(ERR_GPUID);

	applyGpuSyncMode(0);

	HANDLE_ERROR(cudaSetDevice(device_id));

	size_t free(0), total(0);
	DEBUG_HANDLE_ERROR(cudaMemGetInfo( &free, &total ));
	float margin(1.05);
	BoxLimit = pow(free/(margin*2.5*sizeof(XFLOAT)*((float)shares)),(1/3.0)) / ((float) baseMLO->mymodel.padding_factor);
	//size_t BytesNeeded = ((float)shares)*margin*2.5*sizeof(XFLOAT)*pow((baseMLO->mymodel.ori_size*baseMLO->mymodel.padding_factor),3);

	return(BoxLimit);
}
void MlDeviceBundle::setupFixedSizedObjects()
{
	int devCount;
	HANDLE_ERROR(cudaGetDeviceCount(&devCount));
	if(device_id >= devCount)
	{
		//std::cerr << " using device_id=" << device_id << " (device no. " << device_id+1 << ") which is higher than the available number of devices=" << devCount << std::endl;
		CRITICAL(ERR_GPUID);
	}
	else
	{
		applyGpuSyncMode(0);
		HANDLE_ERROR(cudaSetDevice(device_id));
	}

	//Can we pre-generate projector plan and corresponding euler matrices for all particles
	if (baseMLO->do_skip_align || baseMLO->do_skip_rotate || baseMLO->do_auto_refine || baseMLO->mymodel.orientational_prior_mode != NOPRIOR || baseMLO->mydata.is_tomo)
		generateProjectionPlanOnTheFly = true;
	else
		generateProjectionPlanOnTheFly = false;

    //std::cerr << "DEBUG: setting generateProjectionPlanOnTheFly = true;" << std::endl;
    //generateProjectionPlanOnTheFly = true;

	unsigned nr_proj = baseMLO->mymodel.PPref.size();
	unsigned nr_bproj = baseMLO->wsum_model.BPref.size();

	// The allocator kept from the last iteration, if the models of this one still fit next to it
	CudaCustomAllocator *kept = NULL;
	if (keep_allocator)
	{
		omp_set_lock(retainedLock());
		auto it = retainedAllocators().find(device_id);
		if (it != retainedAllocators().end())
		{
			kept = it->second;
			retainedAllocators().erase(it);
		}
		omp_unset_lock(retainedLock());
	}
	if (kept != NULL)
	{
		double bytes = 0;
		for (unsigned i = 0; i < nr_proj; i++)
			bytes += 2. * sizeof(XFLOAT) * (double) baseMLO->mymodel.PPref[i].data.nzyxdim;
		for (unsigned i = 0; i < nr_bproj; i++)
			bytes += 4. * sizeof(XFLOAT) * (double) baseMLO->wsum_model.BPref[i].data.nzyxdim;
		size_t free_now = 0, total_now = 0;
		cudaMemGetInfo(&free_now, &total_now);
		if ((double) free_now < 1.15 * bytes + 5e8)
		{
			delete kept; // not enough room for the models: allocate a chunk to fit, as before
			kept = NULL;
		}
	}

	projectors.resize(nr_proj);
	backprojectors.resize(nr_bproj);

	/*======================================================
	              PROJECTOR AND BACKPROJECTOR
	======================================================*/

	for (int imodel = 0; imodel < nr_proj; imodel++)
	{
		projectors[imodel].setMdlDim(
				baseMLO->mymodel.PPref[imodel].data.xdim,
				baseMLO->mymodel.PPref[imodel].data.ydim,
				baseMLO->mymodel.PPref[imodel].data.zdim,
				baseMLO->mymodel.PPref[imodel].data.yinit,
				baseMLO->mymodel.PPref[imodel].data.zinit,
				baseMLO->mymodel.PPref[imodel].r_max,
				baseMLO->mymodel.PPref[imodel].padding_factor);

		projectors[imodel].initMdl(baseMLO->mymodel.PPref[imodel].data.data);

	}

	for (int imodel = 0; imodel < nr_bproj; imodel++)
	{
		backprojectors[imodel].setMdlDim(
				baseMLO->wsum_model.BPref[imodel].data.xdim,
				baseMLO->wsum_model.BPref[imodel].data.ydim,
				baseMLO->wsum_model.BPref[imodel].data.zdim,
				baseMLO->wsum_model.BPref[imodel].data.yinit,
				baseMLO->wsum_model.BPref[imodel].data.zinit,
				baseMLO->wsum_model.BPref[imodel].r_max,
				baseMLO->wsum_model.BPref[imodel].padding_factor);

		backprojectors[imodel].initMdl();
	}

	/*======================================================
	                    CUSTOM ALLOCATOR
	======================================================*/

	int memAlignmentSize;
	cudaDeviceGetAttribute ( &memAlignmentSize, cudaDevAttrTextureAlignment, device_id );
	allocator = kept != NULL ? kept : new CudaCustomAllocator(0, memAlignmentSize);
}

void MlDeviceBundle::setupTunableSizedObjects(size_t allocationSize)
{
	unsigned nr_models = baseMLO->mymodel.nr_classes;
	int devCount;
	HANDLE_ERROR(cudaGetDeviceCount(&devCount));
	if(device_id >= devCount)
	{
		//std::cerr << " using device_id=" << device_id << " (device no. " << device_id+1 << ") which is higher than the available number of devices=" << devCount << std::endl;
		CRITICAL(ERR_GPUID);
	}
	else
		HANDLE_ERROR(cudaSetDevice(device_id));

	/*======================================================
	                    CUSTOM ALLOCATOR
	======================================================*/
#ifdef DEBUG_CUDA
	printf("DEBUG: Total GPU allocation size set to %zu MB on device id %d.\n", allocationSize / (1000*1000), device_id);
#endif
#ifndef CUDA_NO_CUSTOM_ALLOCATION
	// A chunk kept from the last iteration is used again when it is about the size that would be
	// allocated now (the size given includes the chunk's own bytes)
	const size_t have = allocator->chunkSize();
	if (have > 0 && allocationSize >= 0.97 * (double) have && allocationSize <= 1.25 * (double) have &&
	    allocator->getNumberOfAllocs() == 0)
		;
	else
		allocator->resize(allocationSize);
#endif


	/*======================================================
	                    PROJECTION PLAN
	======================================================*/

	coarseProjectionPlans.resize(nr_models, allocator);

	for (int iclass = 0; iclass < nr_models; iclass++)
	{
		//If doing predefined projector plan at all and is this class significant
		if (!generateProjectionPlanOnTheFly && baseMLO->mymodel.pdf_class[iclass] > 0.)
		{
			std::vector<int> exp_pointer_dir_nonzeroprior;
			std::vector<int> exp_pointer_psi_nonzeroprior;
			std::vector<RFLOAT> exp_directions_prior;
			std::vector<RFLOAT> exp_psi_prior;

			long unsigned itrans_max = baseMLO->sampling.NrTranslationalSamplings() - 1;
			long unsigned nr_idir = baseMLO->sampling.NrDirections(0, &exp_pointer_dir_nonzeroprior);
			long unsigned nr_ipsi = baseMLO->sampling.NrPsiSamplings(0, &exp_pointer_psi_nonzeroprior );

			coarseProjectionPlans[iclass].setup(
					baseMLO->sampling,
					exp_directions_prior,
					exp_psi_prior,
					exp_pointer_dir_nonzeroprior,
					exp_pointer_psi_nonzeroprior,
					NULL, //Mcoarse_significant
					baseMLO->mymodel.pdf_class,
					baseMLO->mymodel.pdf_direction,
					nr_idir,
					nr_ipsi,
					0, //idir_min
					nr_idir - 1, //idir_max
					0, //ipsi_min
					nr_ipsi - 1, //ipsi_max
					0, //itrans_min
					itrans_max,
					0, //current_oversampling
					1, //nr_oversampled_rot
					iclass,
					true, //coarse
					!IS_NOT_INV,
					baseMLO->do_skip_align,
					baseMLO->do_skip_rotate,
					baseMLO->mymodel.orientational_prior_mode
					);
		}
	}

	setupBackprojectAccumulator();
};

// Whether this iteration's back-projections can be accumulated per orientation, and the room for
// it: a part of the allocator's chunk. On when eligible; RELION_GPU_BP_ACCUM=0 turns it off, =N sets the part in MB:
// the sums over particles are made in separate buffers before they reach the volume, which
// loses less to float rounding than adding straight into it, so the maps differ from the
// direct way by more than the run-to-run noise of atomic additions (weights up to 4% larger
// at low resolution). Only for global searches with images against 3D references on identical
// volumes, where the (class, orientation) slot of every orientation is the same for all particles.
void MlDeviceBundle::setupBackprojectAccumulator()
{
	bpacc = BPAccumulator();
	const char *env = getenv("RELION_GPU_BP_ACCUM");
	if (!env)
		env = "1";
	if (atoi(env) <= 0)
		return;
	MlOptimiser *m = baseMLO;
	const int nr_classes = m->mymodel.nr_classes;
	if (!gpuClassBatching() || m->do_grad || m->is_som_iter || m->mymodel.nr_bodies > 1 || m->mymodel.ref_dim != 3 ||
	    m->mymodel.data_dim == 3 || m->mydata.is_tomo || m->do_skip_align || m->do_skip_rotate || m->do_only_sample_tilt ||
	    m->mymodel.orientational_prior_mode != NOPRIOR || m->grad_pseudo_halfsets ||
	    (int) backprojectors.size() != nr_classes || nr_classes < 1)
		return;
	for (int c = 1; c < nr_classes; c++)
	{
		const AccBackprojector &a = backprojectors[0], &b = backprojectors[c];
		if (b.mdlX != a.mdlX || b.mdlY != a.mdlY || b.mdlZ != a.mdlZ || b.mdlInitY != a.mdlInitY ||
		    b.mdlInitZ != a.mdlInitZ || b.maxR != a.maxR || b.padding_factor != a.padding_factor)
			return;
	}
	if (backprojectors[0].mdlZ == 1)
		return;
	for (size_t g = 1; g < m->image_current_size.size(); g++)
		if (m->image_current_size[g] != m->image_current_size[0])
			return;
	const unsigned cs = m->image_current_size[0];
	const unsigned img_x = cs / 2 + 1, img_y = cs;
	const size_t pixels = (size_t) img_x * img_y;

	std::vector<int> pdir, ppsi;
	const size_t nr_dir = m->sampling.NrDirections(0, &pdir);
	const size_t nr_psi = m->sampling.NrPsiSamplings(0, &ppsi);
	const size_t nr_over = m->sampling.oversamplingFactorOrientations(m->adaptive_oversampling);
	if (!pdir.empty() || !ppsi.empty())
		return;
	const size_t slots = (size_t) nr_classes * nr_dir * nr_psi * nr_over;
	if (slots == 0 || slots >= ((size_t) 1 << 31))
		return;

	const size_t chunk = allocator->chunkSize();
	size_t budget = std::min<size_t>((size_t) 4 << 30, chunk / 5);
	if (atoi(env) > 1)
		budget = (size_t) atoi(env) << 20;
	const size_t header = 4096 + slots * sizeof(int) + (size_t) nr_classes * sizeof(BPModelPtrs) + 4096;
	const size_t per_buf = (3 * pixels + 9 + 1) * sizeof(XFLOAT);
	if (budget < header + 1000 * per_buf)
		return;
	size_t cap = std::min<size_t>(slots, (budget - header) / per_buf);

	const size_t bytes = header + cap * per_buf;
	CudaCustomAllocator::Alloc *blk = allocator->alloc(bytes);
	char *base = (char *) blk->getPtr();
	BPAccumulator &A = bpacc;
	A.block = blk;
	A.bytes = bytes;
	A.slots = slots;
	A.slots_per_class = (unsigned) (nr_dir * nr_psi * nr_over);
	A.nr_over = (unsigned) nr_over;
	A.img_x = img_x;
	A.img_y = img_y;
	size_t pos = 0;
	A.view.counter = (int *) (base + pos);                      pos += 256;
	A.view.slot_idx = (int *) (base + pos);                     pos += ((slots * sizeof(int) + 255) / 256) * 256;
	A.d_models = (BPModelPtrs *) (base + pos);                  pos += ((nr_classes * sizeof(BPModelPtrs) + 255) / 256) * 256;
	A.view.idx_slot = (unsigned *) (base + pos);                pos += ((cap * sizeof(unsigned) + 255) / 256) * 256;
	A.view.euler = (XFLOAT *) (base + pos);                     pos += ((cap * 9 * sizeof(XFLOAT) + 255) / 256) * 256;
	A.view.acc = (XFLOAT *) (base + pos);                       pos += cap * 3 * pixels * sizeof(XFLOAT);
	A.view.cap = (int) cap;
	A.view.pixels = (unsigned) pixels;
	if (pos > bytes)
	{
		allocator->free(blk);
		bpacc = BPAccumulator();
		return;
	}
	HANDLE_ERROR(cudaMemset(A.view.counter, 0, sizeof(int)));
	HANDLE_ERROR(cudaMemset(A.view.slot_idx, 0xFF, slots * sizeof(int)));
	HANDLE_ERROR(cudaMemset(A.view.acc, 0, cap * 3 * pixels * sizeof(XFLOAT)));
	std::vector<BPModelPtrs> models(nr_classes);
	for (int c = 0; c < nr_classes; c++)
	{
		models[c].real = backprojectors[c].d_mdlReal;
		models[c].imag = backprojectors[c].d_mdlImag;
		models[c].weight = backprojectors[c].d_mdlWeight;
	}
	HANDLE_ERROR(cudaMemcpy(A.d_models, models.data(), nr_classes * sizeof(BPModelPtrs), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaDeviceSynchronize());
	if (m->verb > 0)
		printf(" GPU back-projection per orientation: %zu slots of %zu possible (%d MB), as the search is global\n",
				cap, slots, (int) (bytes >> 20));
}

void MlDeviceBundle::flushBackprojectAccumulator()
{
	HANDLE_ERROR(cudaSetDevice(device_id));
	HANDLE_ERROR(cudaDeviceSynchronize());
	if (bpacc.block == NULL)
		return;
	int n = 0;
	HANDLE_ERROR(cudaMemcpy(&n, bpacc.view.counter, sizeof(int), cudaMemcpyDeviceToHost));
	const int used = std::min(n, bpacc.view.cap);
	if (used > 0)
	{
		runBackProjectAccumFlush(backprojectors[0], bpacc.view, bpacc.d_models, bpacc.slots_per_class,
				bpacc.img_x, bpacc.img_y, used);
		HANDLE_ERROR(cudaDeviceSynchronize());
	}
	if (baseMLO->verb > 0 && n > bpacc.view.cap)
		printf(" GPU back-projection per orientation: %d orientations did not fit in the %d slots and were added directly\n",
				n - bpacc.view.cap, bpacc.view.cap);
	allocator->free(bpacc.block);
	bpacc = BPAccumulator();
}

void MlOptimiserCuda::resetData()
{
	int devCount;
	HANDLE_ERROR(cudaGetDeviceCount(&devCount));
	if(device_id >= devCount)
	{
		//std::cerr << " using device_id=" << device_id << " (device no. " << device_id+1 << ") which is higher than the available number of devices=" << devCount << std::endl;
		CRITICAL(ERR_GPUID);
	}
	else
		HANDLE_ERROR(cudaSetDevice(device_id));

	unsigned nr_classes = baseMLO->mymodel.nr_classes;

	// With class batching nearly all GPU work is issued for all classes at once on this
	// thread's stream; the few per-class steps left then use that stream too, so that
	// waiting for "all class streams" is one wait instead of one per class (each a
	// driver call under the driver's per-process lock)
	classStreamsShared = gpuClassBatching();
	classStreams.resize(nr_classes, 0);
	for (int i = 0; i < nr_classes; i++)
		if (classStreamsShared)
			classStreams[i] = cudaStreamPerThread;
		else
			HANDLE_ERROR(cudaStreamCreate(&classStreams[i])); //HANDLE_ERROR(cudaStreamCreateWithFlags(&classStreams[i],cudaStreamNonBlocking));

	transformer1.clear();
	transformer2.clear();
};

void MlOptimiserCuda::doThreadExpectationSomeParticles(int thread_id)
{
#ifdef TIMING
	// Only time one thread
	if (thread_id == 0)
		baseMLO->timer.tic(baseMLO->TIMING_ESP_THR);
#endif
//	CTOC(cudaMLO->timer,"interParticle");

	int devCount;
	HANDLE_ERROR(cudaGetDeviceCount(&devCount));
	if(device_id >= devCount)
	{
		//std::cerr << " using device_id=" << device_id << " (device no. " << device_id+1 << ") which is higher than the available number of devices=" << devCount << std::endl;
		CRITICAL(ERR_GPUID);
	}
	else
		DEBUG_HANDLE_ERROR(cudaSetDevice(device_id));
	//std::cerr << " calling on device " << device_id << std::endl;
	//put mweight allocation here
	size_t first_ipart = 0, last_ipart = 0;

	// RELION_GPU_PARTICLE_GROUP=n: particles go through the E-step n at a time in lockstep
	// (accDoExpectationParticleGroup); 1 (default): one at a time; auto: the size is chosen while running
	// (GroupSizeTuner)
	static const bool group_auto = []() {
		const char *e = getenv("RELION_GPU_PARTICLE_GROUP");
		return e && !strcmp(e, "auto");
	}();
	static const int group_fixed = []() {
		const char *e = getenv("RELION_GPU_PARTICLE_GROUP");
		return std::max(1, (e && strcmp(e, "auto")) ? atoi(e) : 1);
	}();
	if (group_auto || group_fixed > 1)
	{
		bool more = true;
		while (more)
		{
			// What a thread may take for its group: 70% of what the allocator has free now, shared by the
			// threads (the allocator holds all the memory that the job uses; cudaMemGetInfo would say little)
			const size_t budget = (size_t) (0.7 * allocator->getTotalFreeSpace() / std::max(1, (int) baseMLO->nr_threads));
			const int group_size = group_auto ? GroupSizeTuner::instance().next() : group_fixed;
			std::vector<unsigned long> ids;
			while ((int) ids.size() < group_size && (more = baseMLO->exp_ipart_ThreadTaskDistributor->getTasks(first_ipart, last_ipart)))
				for (long unsigned ipart = first_ipart; ipart <= last_ipart; ipart++)
					ids.push_back(baseMLO->exp_my_first_part_id + ipart);
			if (!ids.empty())
			{
				AccPtrFactory ptrFactory(allocator, cudaStreamPerThread);
				accDoExpectationParticleGroup<MlOptimiserCuda>(this, ids, thread_id, ptrFactory, budget);
				// only full groups tell the speed of that size
				if (group_auto && (int) ids.size() == group_size)
					GroupSizeTuner::instance().report(group_size, group_size,
							std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count());
			}
		}
	}
	else
	while (baseMLO->exp_ipart_ThreadTaskDistributor->getTasks(first_ipart, last_ipart))
	{
		CTIC(timer,"oneTask");
		for (long unsigned ipart = first_ipart; ipart <= last_ipart; ipart++)
		{
#ifdef TIMING
	// Only time one thread
	if (thread_id == 0)
		baseMLO->timer.tic(baseMLO->TIMING_ESP_DIFF2_A);
#endif

			AccPtrFactory ptrFactory(allocator, cudaStreamPerThread);
            accDoExpectationOneParticle<MlOptimiserCuda>(this, baseMLO->exp_my_first_part_id + ipart, thread_id, ptrFactory);

		}
		CTOC(timer,"oneTask");
	}

//	CTIC(cudaMLO->timer,"interParticle");
//	exit(0);

#ifdef TIMING
	// Only time one thread
	if (thread_id == 0)
		baseMLO->timer.toc(baseMLO->TIMING_ESP_THR);
#endif
}

