#ifndef ACC_HELPER_FUNCTIONS_H_
#define ACC_HELPER_FUNCTIONS_H_

#include "src/acc/acc_ml_optimiser.h"
#include "src/acc/orientation_cache.h"
#include "src/gpu_switch.h"
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#ifdef _CUDA_ENABLED
	using deviceStream_t = cudaStream_t;
#elif _HIP_ENABLED
	using deviceStream_t = hipStream_t;
#elif _SYCL_ENABLED
	#include "src/acc/sycl/sycl_virtual_dev.h"
	using deviceStream_t = virtualSYCL*;
#else
	using deviceStream_t = float;
#endif
/*
 * This assisting function goes over the orientations determined as significant for this image, and checks
 * which translations should be included in the list of those which differences will be calculated for.
 *
 * Any contiguous translations with a shared orientation are grouped together into a "job" which is supplied
 * to the difference kernel. If there are more contiguous translations than the specified "chunk" number,
 * these are split into separate jobs, to increase parallelism at the cost of redundant memory reads.
 */
long int makeJobsForDiff2Fine(
		OptimisationParamters &op,  SamplingParameters &sp,
		long int orientation_num, long int translation_num,
		ProjectionParams &FineProjectionData,
		std::vector< long unsigned > &iover_transes,
		std::vector< long unsigned > &ihiddens,
		long int nr_over_orient, long int nr_over_trans,
		IndexedDataArray &FPW, // FPW=FinePassWeights
		IndexedDataArrayMask &dataMask,
		int chunk);

/*
 * This assisting function goes over the weight-array and groups all weights with shared
 * orientations into 'jobs' which are fed into the collect-kenrel, which reduces all translations
 * with computed differences into a reduced object to be back-projected.
 */
long int  makeJobsForCollect(IndexedDataArray &FPW,
        IndexedDataArrayMask &dataMask,
        unsigned long NewJobNum); // FPW=FinePassWeights

/*
 * Maps weights to a decoupled indexing of translations and orientations
 */
void mapWeights(
		unsigned long orientation_start,
		XFLOAT *mapped_weights,
		unsigned long orientation_num,
		unsigned long idxArr_start,
		unsigned long idxArr_end,
		unsigned long translation_num,
		XFLOAT *weights,
		long unsigned *rot_idx,
		long unsigned *trans_idx,
		unsigned long current_oversampling);

void buildCorrImage(MlOptimiser *baseMLO,
		OptimisationParamters &op,
		AccPtr<XFLOAT> &corr_img,
		int img_id,
		long int group_id);

void generateEulerMatrices(
		ProjectionParams &ProjectionData,
		XFLOAT *eulers,
		bool inverse,
		Matrix2D<RFLOAT> &L,
		Matrix2D<RFLOAT> &R,
		const OrientationCache *cache = nullptr);

std::shared_ptr<OrientationCache> getOrientationCache(
		MlOptimiser *baseMLO, SamplingParameters &sp, OptimisationParamters &op);

long unsigned generateProjectionSetupFine(
		OptimisationParamters &op,
		SamplingParameters &sp,
		MlOptimiser *baseMLO,
		unsigned iclass,
		ProjectionParams &ProjectionData);

void runWavgKernel(
		AccProjectorKernel &projector,
		XFLOAT *eulers,
		XFLOAT *Fimgs_real,
		XFLOAT *Fimgs_imag,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT *sorted_weights,
		XFLOAT *ctfs,
		XFLOAT *wdiff2s_parts,
		XFLOAT *wdiff2s_AA,
		XFLOAT *wdiff2s_XA,
		OptimisationParamters &op,
		long unsigned orientation_num,
		long unsigned translation_num,
		unsigned long image_size,
		int img_id,
		int group_id,
		int exp_iclass,
		XFLOAT part_scale,
		bool refs_are_ctf_corrected,
		bool ctf_premultiplied,
		bool data_is_3D,
		deviceStream_t stream);

void runBackProjectKernel(
		AccBackprojector &BP,
		AccProjectorKernel &projector,
		XFLOAT *d_img_real,
		XFLOAT *d_img_imag,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT* d_weights,
		XFLOAT* d_Minvsigma2s,
		XFLOAT* d_ctfs,
		unsigned long translation_num,
		XFLOAT significant_weight,
		XFLOAT weight_norm,
		XFLOAT *d_eulers,
		int imgX,
		int imgY,
		int imgZ,
		unsigned long imageCount,
		bool data_is_3D,
		bool do_grad,
		bool ctf_premultiplied,
		deviceStream_t optStream);

template< typename T>
void deviceInitComplexValue(AccPtr<T> &data, XFLOAT value)
{
	AccUtilities::InitComplexValue<T>(data, value);
}

template< typename T>
void deviceInitValue(AccPtr<T> &data, T value)
{
	AccUtilities::InitValue<T>(data, value);
}

template< typename T>
void deviceInitValue(AccPtr<T> &data, T value, size_t Size)
{
	AccUtilities::InitValue<T>(data, value, Size);
}

#ifdef _CUDA_ENABLED
// All classes of a particle in one launch (see cuda_kernel_allweights_to_mweights_batched)
void mapAllWeightsToMweightsBatched(
		const std::vector<WeightMapJob> &jobs,
		XFLOAT * d_allweights,
		XFLOAT * d_mweights,
		unsigned long translation_num,
		AccPtr<char> &table);
#endif

void mapAllWeightsToMweights(
		unsigned long * d_iorient, //projectorPlan.iorientclasses
		XFLOAT * d_allweights, //allWeights
		XFLOAT * d_mweights, //Mweight
		unsigned long orientation_num, //projectorPlan.orientation_num
		unsigned long translation_num, //translation_num
		deviceStream_t stream
		);

#define OVER_THRESHOLD_BLOCK_SIZE 512
template< typename T>
void arrayOverThreshold(AccPtr<T> &data, AccPtr<bool> &passed, T threshold)
{
#ifdef _CUDA_ENABLED
	int grid_size = ceil((float)data.getSize()/(float)OVER_THRESHOLD_BLOCK_SIZE);
	cuda_kernel_array_over_threshold<T><<< grid_size, OVER_THRESHOLD_BLOCK_SIZE, 0, data.getStream() >>>(
			~data,
			~passed,
			threshold,
			data.getSize(),
			OVER_THRESHOLD_BLOCK_SIZE);
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
#elif _HIP_ENABLED
	int grid_size = ceil((float)data.getSize()/(float)OVER_THRESHOLD_BLOCK_SIZE);
	hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_array_over_threshold<T>), dim3(grid_size), dim3(OVER_THRESHOLD_BLOCK_SIZE), 0, data.getStream(),
			~data,
			~passed,
			threshold,
			data.getSize(),
			OVER_THRESHOLD_BLOCK_SIZE);
	LAUNCH_HANDLE_ERROR(hipGetLastError());
#else
	int Size = data.getSize();
	for(size_t i=0; i<Size; i++)
	{
		if (data[i] >= threshold)
			passed[i] = true;
		else
			passed[i] = false;
	}
#endif
}

#define FIND_IN_CUMULATIVE_BLOCK_SIZE 512
template< typename T>
size_t findThresholdIdxInCumulativeSum(AccPtr<T> &data, T threshold)
{
	int grid_size = ceil((float)(data.getSize()-1)/(float)FIND_IN_CUMULATIVE_BLOCK_SIZE);
	if(grid_size==0)
	{
		return(0);
	}
	else
	{
#ifdef _CUDA_ENABLED
		AccPtr<size_t >  idx(1, data.getStream(), data.getAllocator());
		idx[0] = 0;

		idx.putOnDevice();
		cuda_kernel_find_threshold_idx_in_cumulative<<< grid_size, FIND_IN_CUMULATIVE_BLOCK_SIZE, 0, data.getStream() >>>(
				~data,
				threshold,
				data.getSize()-1,
				~idx,
				FIND_IN_CUMULATIVE_BLOCK_SIZE);
		idx.cpToHost();
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(data.getStream()));
		return idx[0];
#elif _HIP_ENABLED
		AccPtr<size_t >  idx(1, data.getStream(), data.getAllocator());
		idx[0] = 0;

		idx.putOnDevice();
		hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_find_threshold_idx_in_cumulative), dim3(grid_size), dim3(FIND_IN_CUMULATIVE_BLOCK_SIZE), 0, data.getStream(),
				~data,
				threshold,
				data.getSize()-1,
				~idx,
				FIND_IN_CUMULATIVE_BLOCK_SIZE);
		idx.cpToHost();
		DEBUG_HANDLE_ERROR(hipStreamSynchronize(data.getStream()));
		return idx[0];
#elif defined(_SYCL_ENABLED) && defined(USE_ONEDPL)
		return AccUtilities::findThresholdIdxInCumulativeSum(data, threshold);
#else
		size_t idx = 0;
		size_t size_m1 = data.getSize()-1;
		for (size_t i = 0; i < size_m1; i++) {
			if (data[i] <= threshold && threshold < data[i+1])
				idx = i+1;
		}
		return idx;
#endif
	}
}

void runDiff2KernelCoarse(
		AccProjectorKernel &projector,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT *corr_img,
		XFLOAT *Fimg_real,
		XFLOAT *Fimg_imag,
		XFLOAT *d_eulers,
		XFLOAT *diff2s,
		XFLOAT local_sqrtXi2,
		long unsigned orientation_num,
		unsigned long translation_num,
		unsigned long image_size,
		deviceStream_t stream,
		bool do_CC,
		bool data_is_3D);

#ifdef _CUDA_ENABLED
// GPU class batching (several classes per kernel launch) is on by default;
// RELION_GPU_CLASS_BATCH=off goes back to one launch per class, e.g. to compare
inline bool gpuClassBatching()
{
	static const bool on = []() {
		const char *e = getenv("RELION_GPU_CLASS_BATCH");
		return !(e && (std::string(e) == "off" || std::string(e) == "0"));
	}();
	return on;
}

// RELION_GPU_MAP_PER_CLASS=1: allweights_to_mweights with one launch per class again
// (the default is one launch for all classes), e.g. to compare
inline bool gpuMapPerClass()
{
	static const bool on = []() {
		const char *e = getenv("RELION_GPU_MAP_PER_CLASS");
		return e && std::string(e) != "0" && std::string(e) != "off";
	}();
	return on;
}

// RELION_GPU_SHIFT_TABLE=1: (cos, sin) of the translations' phases of a particle's pixels made
// once per particle for the coarse differences instead of a sincosf in every block
inline bool gpuShiftTable()
{
	static const bool on = []() {
		const char *e = getenv("RELION_GPU_SHIFT_TABLE");
		return e && std::string(e) != "0" && std::string(e) != "off";
	}();
	return on;
}

// RELION_GPU_WAVG_SUM: the class-batched weighted averages of 2D data from the weights' sum and
// the weighted sum of the shifted images (see cuda_kernel_wavg_sum); float noise, on by default
inline bool gpuWavgSum()
{
	static const bool on = gpuswitch::on("RELION_GPU_WAVG_SUM");
	return on;
}

// RELION_GPU_FUSED_WEIGHTS: the significance pass of the weights (maximum, exponentiation, sort,
// scan, threshold, significance mask) as a few launches and one copy instead of about fifty driver
// calls per pass (see CudaKernels::fusedWeightPass); on by default, RELION_GPU_FUSED_WEIGHTS=0 turns it off
inline bool gpuFusedWeights()
{
	static const bool on = gpuswitch::on("RELION_GPU_FUSED_WEIGHTS");
	return on;
}

// RELION_GPU_FUSED_PREP=1: the particle preparation with fewer round trips to the host: the
// background value of the soft mask is made on the GPU from the partial sums, and the wait
// after the mask is left to the copy that follows; off by default while it is validated
inline bool gpuFusedPrep()
{
	static const bool on = []() {
		const char *e = getenv("RELION_GPU_FUSED_PREP");
		return e && std::string(e) != "0" && std::string(e) != "off";
	}();
	return on;
}

// RELION_GPU_SHARED_SUMS=1: the weighted sums and the 3D back-projection of 2D images in
// one kernel (see cuda_kernel_shared_sums); off by default while it is validated
// (2: only the weighted-sum outputs, 3: only the back-projection, from the shared-sums
// kernel, the other half from its own kernel: for checking the two halves separately)
inline int gpuSharedSums()
{
	static const int mode = []() {
		const char *e = getenv("RELION_GPU_SHARED_SUMS");
		if (!e || std::string(e) == "0" || std::string(e) == "off")
			return 0;
		const int m = atoi(e);
		return (m == 2 || m == 3) ? m : 1;
	}();
	return mode;
}

// All classes of one coarse pass in one call (two launches); `table` is uploaded and
// must outlive the launches (CUDA only, not for do_CC)
//
// With `defer` nothing is launched: the objects the launch needs stay in `defer` (they must live until the
// group's launch has finished) and runDiff2KernelCoarseGroup launches for several particles together.
struct CoarseDeferred
{
	std::unique_ptr<AccPtr<XFLOAT> > partial;
	std::unique_ptr<AccPtr<float2> > shift;
	std::unique_ptr<AccProjectorKernel> projector;
	XFLOAT *trans_x, *trans_y, *trans_z, *corr_img, *Fimg_real, *Fimg_imag;
	unsigned long translation_num, image_size;
	const CoarseClassJob *d_classes;
	const int *d_even;
	const float2 *d_shift;
	XFLOAT *d_partial;
	int n_blocks = 0, n_classes = 0, n_chunks = 1;
	size_t chunk_stride = 0, max_orient = 0;
	deviceStream_t stream;
};

bool runDiff2KernelCoarseBatched(
		std::vector<CoarseClassJob> &classes,
		std::vector<size_t> &orientation_nums,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT *corr_img,
		XFLOAT *Fimg_real,
		XFLOAT *Fimg_imag,
		unsigned long translation_num,
		unsigned long image_size,
		bool data_is_3D,
		AccPtr<char> &table,
		CoarseDeferred *defer = NULL);

// The deferred coarse launches of several particles together; `recs` holds the uploaded records
bool runDiff2KernelCoarseGroup(std::vector<CoarseDeferred *> &items, AccPtr<char> &recs);

// The minimum of each of several device arrays (one launch, one read-back); the arrays are
// (pointer, size) pairs, all on `stream`
void minOnDeviceBatched(const std::vector<std::pair<const XFLOAT *, size_t> > &arrays, std::vector<XFLOAT> &mins,
		AccPtr<char> &scratch, deviceStream_t stream);
#endif

#ifdef _CUDA_ENABLED
// All classes' mappings in one launch (RELION_GPU_CLASS_BATCH, default; the per-class
// launch is 2-3 us of GPU and a launch for each of up to 100 classes per particle)
void mapAllWeightsToMweightsBatched(
		std::vector<MapClassJob> &jobs,
		XFLOAT * d_mweights,
		unsigned long translation_num,
		AccPtr<char> &table);

// All classes' weighted averages / back-projections of one image in one launch
// (one block per class and orientation); `table` is uploaded and must outlive
// the launch. Back-projection without do_grad only.
void runWavgKernelBatched(
		std::vector<WavgClassJob> &jobs,
		std::vector<size_t> &orientation_nums,
		XFLOAT *Fimg_real,
		XFLOAT *Fimg_imag,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT *ctfs,
		XFLOAT *wdiff2s_parts,
		OptimisationParamters &op,
		long unsigned translation_num,
		unsigned long image_size,
		XFLOAT part_scale,
		bool refs_are_ctf_corrected,
		bool data_is_3D,
		AccPtr<char> &table);

void runDiff2KernelFineBatched(
		std::vector<FineClassJob> &jobs,
		std::vector<size_t> &job_nums,
		XFLOAT *corr_img,
		XFLOAT *Fimgs_real,
		XFLOAT *Fimgs_imag,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		unsigned long image_size,
		XFLOAT sum_init,
		long unsigned translation_num,
		bool data_is_3D,
		AccPtr<char> &table);

void runCollect2jobsBatched(
		std::vector<CollectClassJob> &jobs,
		std::vector<size_t> &block_nums,
		XFLOAT significant_weight,
		XFLOAT sum_weight,
		unsigned long nr_trans,
		unsigned long nr_oversampled_trans,
		unsigned long nr_oversampled_rot,
		unsigned long oversamples,
		bool skip_rots,
		XFLOAT * p_weights,
		XFLOAT * p_thr_wsum_prior_offsetx_class,
		XFLOAT * p_thr_wsum_prior_offsety_class,
		XFLOAT * p_thr_wsum_prior_offsetz_class,
		XFLOAT * p_thr_wsum_sigma2_offset,
		bool data_is_3D,
		AccPtr<char> &table);

// Weighted sums and 3D back-projection of all classes in one launch (opt-in, see
// cuda_kernel_shared_sums); false if the case is not covered (nothing launched)
bool runSharedSumsKernelBatched(
		std::vector<SharedSumsClassJob> &jobs,
		AccBackprojector &BP,
		XFLOAT *d_img_real,
		XFLOAT *d_img_imag,
		XFLOAT *d_img_real_nomask,
		XFLOAT *d_img_imag_nomask,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *d_ctfs,
		XFLOAT *d_Minvsigma2s,
		XFLOAT *wdiff2s_parts,
		unsigned long translation_num,
		XFLOAT weight_norm,
		XFLOAT significant_weight,
		XFLOAT part_scale,
		unsigned long image_size,
		int imgX,
		int imgY,
		bool refs_are_ctf_corrected,
		bool ctf_premultiplied,
		int parts,
		AccPtr<char> &table);

void runBackProjectSgdKernelBatched(
		AccBackprojector &BP,
		std::vector<BPSgdClassJob> &jobs,
		std::vector<size_t> &orientation_nums,
		XFLOAT *d_img_real,
		XFLOAT *d_img_imag,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT* d_Minvsigma2s,
		XFLOAT* d_ctfs,
		unsigned long translation_num,
		XFLOAT significant_weight,
		int imgX,
		int imgY,
		bool ctf_premultiplied,
		AccPtr<char> &table);

void runBackProjectKernelBatched(
		AccBackprojector &BP,
		std::vector<BPClassJob> &jobs,
		std::vector<size_t> &orientation_nums,
		XFLOAT *d_img_real,
		XFLOAT *d_img_imag,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT* d_Minvsigma2s,
		XFLOAT* d_ctfs,
		unsigned long translation_num,
		XFLOAT significant_weight,
		int imgX,
		int imgY,
		int imgZ,
		bool data_is_3D,
		bool ctf_premultiplied,
		AccPtr<char> &table,
		const std::vector<unsigned> *slots = NULL,
		const BPAccumView *accum = NULL);

void runBackProjectAccumFlush(
		AccBackprojector &BP,
		const BPAccumView &acc,
		const BPModelPtrs *d_models,
		unsigned slots_per_class,
		unsigned img_x,
		unsigned img_y,
		int used);
#endif

void runDiff2KernelFine(
		AccProjectorKernel &projector,
		XFLOAT *corr_img,
		XFLOAT *Fimgs_real,
		XFLOAT *Fimgs_imag,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT *eulers,
		long unsigned *rot_id,
		long unsigned *rot_idx,
		long unsigned *trans_idx,
		long unsigned *job_idx,
		long unsigned *job_num,
		XFLOAT *diff2s,
		OptimisationParamters &op,
		MlOptimiser *baseMLO,
		long unsigned orientation_num,
		long unsigned translation_num,
		long unsigned significant_num,
		unsigned long image_size,
		int img_id,
		int exp_iclass,
		deviceStream_t stream,
		long unsigned job_num_count,
		bool do_CC,
		bool data_is_3D);

void runCollect2jobs(	int grid_dim,
						XFLOAT * oo_otrans_x,          // otrans-size -> make const
						XFLOAT * oo_otrans_y,          // otrans-size -> make const
						XFLOAT * oo_otrans_z,          // otrans-size -> make const
						XFLOAT * myp_oo_otrans_x2y2z2, // otrans-size -> make const
						XFLOAT * weights,
						XFLOAT significant_weight,    // TODO Put in const
						XFLOAT sum_weight,    		  // TODO Put in const
						unsigned long nr_trans,
						unsigned long oversampled_trans,
						unsigned long oversampled_rot,
						unsigned long oversamples,
						bool skip_rots,
						XFLOAT * p_weights,
						XFLOAT * p_thr_wsum_prior_offsetx_class,
						XFLOAT * p_thr_wsum_prior_offsety_class,
						XFLOAT * p_thr_wsum_prior_offsetz_class,
						XFLOAT * p_thr_wsum_sigma2_offset,
						size_t * rot_idx,
						size_t * trans_idx,
						size_t * jobOrigin,
						size_t * jobExtent,
						bool data_is_3D,
						deviceStream_t stream);

void windowFourierTransform2(
		AccPtr<ACCCOMPLEX> &d_in,
		AccPtr<ACCCOMPLEX> &d_out,
		size_t iX, size_t iY, size_t iZ, //Input dimensions
		size_t oX, size_t oY, size_t oZ,  //Output dimensions
		size_t Npsi = 1,
		size_t pos = 0,
		deviceStream_t stream = 0);


void selfApplyBeamTilt2(MultidimArray<Complex > &Fimg, RFLOAT beamtilt_x, RFLOAT beamtilt_y,
		RFLOAT wavelength, RFLOAT Cs, RFLOAT angpix, int ori_size);

template <typename T>
void runCenterFFT(MultidimArray< T >& v, bool forward,
#ifdef _CUDA_ENABLED
	CudaCustomAllocator *allocator
#elif _HIP_ENABLED
	HipCustomAllocator *allocator
#else
	CudaCustomAllocator *allocator
#endif
)
{
	AccPtr<XFLOAT >  img_in (v.nzyxdim, allocator);   // with original data pointer
//	AccPtr<XFLOAT >  img_aux(v.nzyxdim, allocator);   // temporary holder

	for (unsigned long i = 0; i < v.nzyxdim; i ++)
		img_in[i] = (XFLOAT) v.data[i];

	img_in.putOnDevice();
//	img_aux.deviceAlloc();

	if ( v.getDim() == 1 )
	{
		std::cerr << "CenterFFT on gpu reverts to cpu for dim!=2 (now dim=1)" <<std::endl;
		// 1D
		MultidimArray< T > aux;
		int l, shift;

		l = XSIZE(v);
		aux.resize(l);
		shift = (int)(l / 2);

		if (!forward)
			shift = -shift;

		// Shift the input in an auxiliar vector
		for (int i = 0; i < l; i++)
		{
			int ip = i + shift;

			if (ip < 0)
				ip += l;
			else if (ip >= l)
				ip -= l;

			aux(ip) = DIRECT_A1D_ELEM(v, i);
		}

		// Copy the vector
		for (int i = 0; i < l; i++)
			DIRECT_A1D_ELEM(v, i) = DIRECT_A1D_ELEM(aux, i);
	}
	else if ( v.getDim() == 2 )
	{
		// 2D
		//std::cerr << "CenterFFT on gpu with dim=2!" <<std::endl;

		long int xshift = (int)(XSIZE(v) / 2);
		long int yshift = (int)(YSIZE(v) / 2);

		if (!forward)
		{
			xshift = -xshift;
			yshift = -yshift;
		}


		int dim=ceilf((float)(v.nzyxdim/(float)(2*CFTT_BLOCK_SIZE)));
		AccUtilities::centerFFT_2D(dim, 0, CFTT_BLOCK_SIZE,
#if defined _CUDA_ENABLED || defined _HIP_ENABLED
				~img_in,
#else
				&img_in[0],
#endif
				v.nzyxdim,
				XSIZE(v),
				YSIZE(v),
				xshift,
				yshift);
#ifdef _CUDA_ENABLED
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
#elif _HIP_ENABLED
		LAUNCH_HANDLE_ERROR(hipGetLastError());
#endif
		img_in.cpToHost();

		for (unsigned long i = 0; i < v.nzyxdim; i ++)
			v.data[i] = (T) img_in[i];

	}
	else if ( v.getDim() == 3 )
	{
		// TODO - convert this to use the faster AccUtilities::centerFFT_3D like the 2D case above
		// and in fftw.h when FAST_CENTERFFT is defined
		std::cerr << "CenterFFT on gpu reverts to cpu for dim!=2 (now dim=3)" <<std::endl;
		// 3D
		MultidimArray< T > aux;
		int l, shift;

		// Shift in the X direction
		l = XSIZE(v);
		aux.resize(l);
		shift = (int)(l / 2);

		if (!forward)
			shift = -shift;

		for (int k = 0; k < ZSIZE(v); k++)
			for (int i = 0; i < YSIZE(v); i++)
			{
				// Shift the input in an auxiliary vector
				for (int j = 0; j < l; j++)
				{
					int jp = j + shift;

					if (jp < 0)
						jp += l;
					else if (jp >= l)
						jp -= l;

					aux(jp) = DIRECT_A3D_ELEM(v, k, i, j);
				}

				// Copy the vector
				for (int j = 0; j < l; j++)
					DIRECT_A3D_ELEM(v, k, i, j) = DIRECT_A1D_ELEM(aux, j);
			}

		// Shift in the Y direction
		l = YSIZE(v);
		aux.resize(l);
		shift = (int)(l / 2);

		if (!forward)
			shift = -shift;

		for (int k = 0; k < ZSIZE(v); k++)
			for (int j = 0; j < XSIZE(v); j++)
			{
				// Shift the input in an auxiliary vector
				for (int i = 0; i < l; i++)
				{
					int ip = i + shift;

					if (ip < 0)
						ip += l;
					else if (ip >= l)
						ip -= l;

					aux(ip) = DIRECT_A3D_ELEM(v, k, i, j);
				}

				// Copy the vector
				for (int i = 0; i < l; i++)
					DIRECT_A3D_ELEM(v, k, i, j) = DIRECT_A1D_ELEM(aux, i);
			}

		// Shift in the Z direction
		l = ZSIZE(v);
		aux.resize(l);
		shift = (int)(l / 2);

		if (!forward)
			shift = -shift;

		for (int i = 0; i < YSIZE(v); i++)
			for (int j = 0; j < XSIZE(v); j++)
			{
				// Shift the input in an auxiliary vector
				for (int k = 0; k < l; k++)
				{
					int kp = k + shift;
					if (kp < 0)
						kp += l;
					else if (kp >= l)
						kp -= l;

					aux(kp) = DIRECT_A3D_ELEM(v, k, i, j);
				}

				// Copy the vector
				for (int k = 0; k < l; k++)
					DIRECT_A3D_ELEM(v, k, i, j) = DIRECT_A1D_ELEM(aux, k);
			}
	}
	else
	{
		v.printShape();
		REPORT_ERROR("CenterFFT ERROR: Dimension should be 1, 2 or 3");
	}
}


template <typename T>
void runCenterFFT( AccPtr< T > &img_in,
				  int xSize,
				  int ySize,
				  bool forward,
				  int batchSize = 1)
{
//	AccPtr<XFLOAT >  img_aux(img_in.h_ptr, img_in.getSize(), allocator);   // temporary holder
//	img_aux.deviceAlloc();

	int xshift = (xSize / 2);
	int yshift = (ySize / 2);

	if (!forward)
	{
		xshift = -xshift;
		yshift = -yshift;
	}

	int blocks = ceilf((float)((xSize*ySize)/(float)(2*CFTT_BLOCK_SIZE)));
	AccUtilities::centerFFT_2D(blocks, batchSize, CFTT_BLOCK_SIZE,
		img_in.getStream(),
		~img_in,
		xSize*ySize,
		xSize,
		ySize,
		xshift,
		yshift);
#ifdef _CUDA_ENABLED
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
	//	HANDLE_ERROR(cudaStreamSynchronize(0));
#elif _HIP_ENABLED
	LAUNCH_HANDLE_ERROR(hipGetLastError());
	//	HANDLE_ERROR(hipStreamSynchronize(0));
#endif
//	img_aux.cpOnDevice(img_in.d_ptr); //update input image with centered kernel-output.
}

template <typename T>
void runCenterFFT( AccPtr< T > &img_in,
				  int xSize,
				  int ySize,
				  int zSize,
				  bool forward,
				  int batchSize = 1)
{
//	AccPtr<XFLOAT >  img_aux(img_in.h_ptr, img_in.getSize(), allocator);   // temporary holder
//	img_aux.deviceAlloc();

	if(zSize>1)
	{
		int xshift = (xSize / 2);
		int yshift = (ySize / 2);
		int zshift = (ySize / 2);

		if (!forward)
		{
			xshift = -xshift;
			yshift = -yshift;
			zshift = -zshift;
		}

		int grid_size = ceilf((float)(((size_t)xSize*(size_t)ySize*(size_t)zSize)/
			(float)(2*CFTT_BLOCK_SIZE)));
		AccUtilities::centerFFT_3D(grid_size, batchSize, CFTT_BLOCK_SIZE,
			img_in.getStream(),
			~img_in,
			(size_t)xSize*(size_t)ySize*(size_t)zSize,
			xSize,
			ySize,
			zSize,
			xshift,
			yshift,
			zshift);
		#ifdef _CUDA_ENABLED
			LAUNCH_HANDLE_ERROR(cudaGetLastError());
		//	HANDLE_ERROR(cudaStreamSynchronize(0));
		#elif _HIP_ENABLED
			LAUNCH_HANDLE_ERROR(hipGetLastError());
		//	HANDLE_ERROR(hipStreamSynchronize(0));
		#endif
		//	img_aux.cpOnDevice(img_in.d_ptr); //update input image with centered kernel-output.
	}
	else
	{
		int xshift = (xSize / 2);
		int yshift = (ySize / 2);

		if (!forward)
		{
			xshift = -xshift;
			yshift = -yshift;
		}

		int blocks = ceilf((float)((xSize*ySize)/(float)(2*CFTT_BLOCK_SIZE)));
		AccUtilities::centerFFT_2D(blocks, batchSize, CFTT_BLOCK_SIZE,
			img_in.getStream(),
			~img_in,
			xSize*ySize,
			xSize,
			ySize,
			xshift,
			yshift);
		#ifdef _CUDA_ENABLED
			LAUNCH_HANDLE_ERROR(cudaGetLastError());
		#elif _HIP_ENABLED
			LAUNCH_HANDLE_ERROR(hipGetLastError());
		#endif
	}
}

template <typename T>
void lowPassFilterMapGPU(
		AccPtr< T > &img_in,
		size_t Zdim,
		size_t Ydim,
		size_t Xdim,
		long int ori_size,
		RFLOAT lowpass,
		RFLOAT highpass,
		RFLOAT angpix,
		int filter_edge_width,
		bool do_highpass)
{
	// High or low?
	RFLOAT passLimit = (do_highpass ? highpass : lowpass);

	// Which resolution shell is the filter?
	int ires_filter = ROUND((ori_size * angpix)/passLimit);
	int filter_edge_halfwidth = filter_edge_width / 2;

	// Soft-edge: from 1 shell less to one shell more:
	XFLOAT edge_low = XMIPP_MAX(0., (ires_filter - filter_edge_halfwidth) / (RFLOAT)ori_size); // in 1/pix
	XFLOAT edge_high = XMIPP_MIN(Xdim, (ires_filter + filter_edge_halfwidth) / (RFLOAT)ori_size); // in 1/pix
	XFLOAT edge_width = edge_high - edge_low;

	int blocks = ceilf( (float)((size_t)Xdim*(size_t)Ydim*(size_t)Zdim) /
		(float)(CFTT_BLOCK_SIZE) );
	if (do_highpass)
	{
		AccUtilities::frequencyPass<true>(blocks,CFTT_BLOCK_SIZE, img_in.getStream(),
				~img_in,
				ori_size,
				Xdim,
				Ydim,
				Zdim,
				edge_low,
				edge_width,
				edge_high,
				(XFLOAT)angpix,
				(size_t)Xdim*(size_t)Ydim*(size_t)Zdim);
	}
	else
	{
		AccUtilities::frequencyPass<false>(blocks,CFTT_BLOCK_SIZE, img_in.getStream(),
						~img_in,
						ori_size,
						Xdim,
						Ydim,
						Zdim,
						edge_low,
						edge_width,
						edge_high,
						(XFLOAT)angpix,
						(size_t)Xdim*(size_t)Ydim*(size_t)Zdim);
	}
#ifdef _CUDA_ENABLED
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
#elif _HIP_ENABLED
	LAUNCH_HANDLE_ERROR(hipGetLastError());
#endif
}

#endif //ACC_HELPER_FUNCTIONS_H_
