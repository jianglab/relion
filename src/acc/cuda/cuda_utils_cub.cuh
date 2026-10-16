#ifndef CUDA_UTILS_CUB_CUH_
#define CUDA_UTILS_CUB_CUH_

#include <cuda.h> // For CUDA_VERSION
#include <cuda_runtime.h>
#include "src/acc/cuda/cuda_settings.h"
#include "src/acc/cuda/cuda_mem_utils.h"
#include "src/gpu_switch.h"
#include <stdio.h>
#include <signal.h>
#include <vector>
// Because thrust uses CUB, thrust defines CubLog and CUB tries to redefine it,
// resulting in warnings. This avoids those warnings.
#if(defined(CubLog) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__<= 520)) // Intetionally force a warning for new arch
	#undef CubLog
#endif

#if (CUDA_VERSION >= 11000)
    #include <cub/cub.cuh>
#else
    // #define CUB_NS_QUALIFIER ::cub // for compatibility with CUDA 11.5
    #include <cub/device/device_radix_sort.cuh>
    #include <cub/device/device_reduce.cuh>
    #include <cub/device/device_scan.cuh>
    #include <cub/device/device_select.cuh>
#endif

namespace CudaKernels
{
template <typename T>
static std::pair<int, T> getArgMaxOnDevice(AccPtr<T> &ptr)
{
#ifdef DEBUG_CUDA
if (ptr.getSize() == 0)
	printf("DEBUG_WARNING: getArgMaxOnDevice called with pointer of zero size.\n");
if (ptr.getDevicePtr() == NULL)
	printf("DEBUG_WARNING: getArgMaxOnDevice called with null device pointer.\n");
if (ptr.getAllocator() == NULL)
	printf("DEBUG_WARNING: getArgMaxOnDevice called with null allocator.\n");
#endif
	AccPtr<cub::KeyValuePair<int, T> >  max_pair(1, ptr.getStream(), ptr.getAllocator());
	max_pair.deviceAlloc();
	size_t temp_storage_size = 0;

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::ArgMax( NULL, temp_storage_size, ~ptr, ~max_pair, ptr.getSize()));

	if(temp_storage_size==0)
		temp_storage_size=1;

	CudaCustomAllocator::Alloc* alloc = ptr.getAllocator()->alloc(temp_storage_size);

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::ArgMax( alloc->getPtr(), temp_storage_size, ~ptr, ~max_pair, ptr.getSize(), ptr.getStream()));

	max_pair.cpToHostWait();

	ptr.getAllocator()->free(alloc);

	std::pair<int, T> pair;
	pair.first = max_pair[0].key;
	pair.second = max_pair[0].value;

	return pair;
}

template <typename T>
static std::pair<int, T> getArgMinOnDevice(AccPtr<T> &ptr)
{
#ifdef DEBUG_CUDA
if (ptr.getSize() == 0)
	printf("DEBUG_WARNING: getArgMinOnDevice called with pointer of zero size.\n");
if (ptr.getDevicePtr() == NULL)
	printf("DEBUG_WARNING: getArgMinOnDevice called with null device pointer.\n");
if (ptr.getAllocator() == NULL)
	printf("DEBUG_WARNING: getArgMinOnDevice called with null allocator.\n");
#endif
	AccPtr<cub::KeyValuePair<int, T> >  min_pair(1, ptr.getStream(), ptr.getAllocator());
	min_pair.deviceAlloc();
	size_t temp_storage_size = 0;

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::ArgMin( NULL, temp_storage_size, ~ptr, ~min_pair, ptr.getSize()));

	if(temp_storage_size==0)
		temp_storage_size=1;

	CudaCustomAllocator::Alloc* alloc = ptr.getAllocator()->alloc(temp_storage_size);

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::ArgMin( alloc->getPtr(), temp_storage_size, ~ptr, ~min_pair, ptr.getSize(), ptr.getStream()));

	min_pair.cpToHostWait();

	ptr.getAllocator()->free(alloc);

	std::pair<int, T> pair;
	pair.first = min_pair[0].key;
	pair.second = min_pair[0].value;

	return pair;
}

template <typename T>
static T getMaxOnDevice(AccPtr<T> &ptr)
{
#ifdef DEBUG_CUDA
if (ptr.getSize() == 0)
	printf("DEBUG_ERROR: getMaxOnDevice called with pointer of zero size.\n");
if (ptr.getDevicePtr() == NULL)
	printf("DEBUG_ERROR: getMaxOnDevice called with null device pointer.\n");
if (ptr.getAllocator() == NULL)
	printf("DEBUG_ERROR: getMaxOnDevice called with null allocator.\n");
#endif
	AccPtr<T>  max_val(1, ptr.getStream(), ptr.getAllocator());
	max_val.deviceAlloc();
	size_t temp_storage_size = 0;

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::Max( NULL, temp_storage_size, ~ptr, ~max_val, ptr.getSize()));

	if(temp_storage_size==0)
		temp_storage_size=1;

	CudaCustomAllocator::Alloc* alloc = ptr.getAllocator()->alloc(temp_storage_size);

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::Max( alloc->getPtr(), temp_storage_size, ~ptr, ~max_val, ptr.getSize(), ptr.getStream()));

	max_val.cpToHostWait();

	ptr.getAllocator()->free(alloc);

	return max_val[0];
}

template <typename T>
static T getMinOnDevice(AccPtr<T> &ptr)
{
#ifdef DEBUG_CUDA
if (ptr.getSize() == 0)
	printf("DEBUG_ERROR: getMinOnDevice called with pointer of zero size.\n");
if (ptr.getDevicePtr() == NULL)
	printf("DEBUG_ERROR: getMinOnDevice called with null device pointer.\n");
if (ptr.getAllocator() == NULL)
	printf("DEBUG_ERROR: getMinOnDevice called with null allocator.\n");
#endif
	AccPtr<T>  min_val(1, ptr.getStream(), ptr.getAllocator());
	min_val.deviceAlloc();
	size_t temp_storage_size = 0;

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::Min( NULL, temp_storage_size, ~ptr, ~min_val, ptr.getSize()));

	if(temp_storage_size==0)
		temp_storage_size=1;

	CudaCustomAllocator::Alloc* alloc = ptr.getAllocator()->alloc(temp_storage_size);

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::Min( alloc->getPtr(), temp_storage_size, ~ptr, ~min_val, ptr.getSize(), ptr.getStream()));

	min_val.cpToHostWait();

	ptr.getAllocator()->free(alloc);

	return min_val[0];
}

template <typename T>
static T getSumOnDevice(AccPtr<T> &ptr)
{
#ifdef DEBUG_CUDA
if (ptr.getSize() == 0)
	printf("DEBUG_ERROR: getSumOnDevice called with pointer of zero size.\n");
if (ptr.getDevicePtr() == NULL)
	printf("DEBUG_ERROR: getSumOnDevice called with null device pointer.\n");
if (ptr.getAllocator() == NULL)
	printf("DEBUG_ERROR: getSumOnDevice called with null allocator.\n");
#endif
	AccPtr<T>  val(1, ptr.getStream(), ptr.getAllocator());
	val.deviceAlloc();
	size_t temp_storage_size = 0;

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::Sum( NULL, temp_storage_size, ~ptr, ~val, ptr.getSize()));

	if(temp_storage_size==0)
		temp_storage_size=1;

	CudaCustomAllocator::Alloc* alloc = ptr.getAllocator()->alloc(temp_storage_size);

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::Sum( alloc->getPtr(), temp_storage_size, ~ptr, ~val, ptr.getSize(), ptr.getStream()));

	val.cpToHostWait();

	ptr.getAllocator()->free(alloc);

	return val[0];
}

template <typename T>
static void sortOnDevice(AccPtr<T> &in, AccPtr<T> &out)
{
#ifdef DEBUG_CUDA
if (in.getSize() == 0 || out.getSize() == 0)
	printf("DEBUG_ERROR: sortOnDevice called with pointer of zero size.\n");
if (in.getDevicePtr() == NULL || out.getDevicePtr() == NULL)
	printf("DEBUG_ERROR: sortOnDevice called with null device pointer.\n");
if (in.getAllocator() == NULL)
	printf("DEBUG_ERROR: sortOnDevice called with null allocator.\n");
#endif
	size_t temp_storage_size = 0;

	cudaStream_t stream = in.getStream();

	DEBUG_HANDLE_ERROR(cub::DeviceRadixSort::SortKeys( NULL, temp_storage_size, ~in, ~out, in.getSize()));

	if(temp_storage_size==0)
		temp_storage_size=1;

	CudaCustomAllocator::Alloc* alloc = in.getAllocator()->alloc(temp_storage_size);

	DEBUG_HANDLE_ERROR(cub::DeviceRadixSort::SortKeys( alloc->getPtr(), temp_storage_size, ~in, ~out, in.getSize(), 0, sizeof(T) * 8, stream));

	alloc->markReadyEvent(stream);
	alloc->doFreeWhenReady();
}

template <typename T>
static void sortDescendingOnDevice(AccPtr<T> &in, AccPtr<T> &out)
{
#ifdef DEBUG_CUDA
if (in.getSize() == 0 || out.getSize() == 0)
	printf("DEBUG_ERROR: sortDescendingOnDevice called with pointer of zero size.\n");
if (in.getDevicePtr() == NULL || out.getDevicePtr() == NULL)
	printf("DEBUG_ERROR: sortDescendingOnDevice called with null device pointer.\n");
if (in.getAllocator() == NULL)
	printf("DEBUG_ERROR: sortDescendingOnDevice called with null allocator.\n");
#endif
	size_t temp_storage_size = 0;

	cudaStream_t stream = in.getStream();

	DEBUG_HANDLE_ERROR(cub::DeviceRadixSort::SortKeysDescending( NULL, temp_storage_size, ~in, ~out, in.getSize()));

	if(temp_storage_size==0)
		temp_storage_size=1;

	CudaCustomAllocator::Alloc* alloc = in.getAllocator()->alloc(temp_storage_size);

	DEBUG_HANDLE_ERROR(cub::DeviceRadixSort::SortKeysDescending( alloc->getPtr(), temp_storage_size, ~in, ~out, in.getSize(), 0, sizeof(T) * 8, stream));

	alloc->markReadyEvent(stream);
	alloc->doFreeWhenReady();

}

class AllocatorThrustWrapper
{
public:
    // just allocate bytes
    typedef char value_type;
	std::vector<CudaCustomAllocator::Alloc*> allocs;
	CudaCustomAllocator *allocator;

    AllocatorThrustWrapper(CudaCustomAllocator *allocator):
		allocator(allocator)
	{}

    ~AllocatorThrustWrapper()
    {
    	for (int i = 0; i < allocs.size(); i ++)
    		allocator->free(allocs[i]);
    }

    char* allocate(std::ptrdiff_t num_bytes)
    {
    	CudaCustomAllocator::Alloc* alloc = allocator->alloc(num_bytes);
    	allocs.push_back(alloc);
    	return (char*) alloc->getPtr();
    }

    void deallocate(char* ptr, size_t n)
    {
    	//TODO fix this (works fine without it though) /Dari
    }
};

template <typename T>
struct MoreThanCubOpt
{
	T compare;
	MoreThanCubOpt(T compare) : compare(compare) {}
	__device__ __forceinline__
	bool operator()(const T &a) const {
		return (a > compare);
	}
};

template <typename T, typename SelectOp>
static int filterOnDevice(AccPtr<T> &in, AccPtr<T> &out, SelectOp select_op)
{
#ifdef DEBUG_CUDA
if (in.getSize() == 0 || out.getSize() == 0)
	printf("DEBUG_ERROR: filterOnDevice called with pointer of zero size.\n");
if (in.getDevicePtr() == NULL || out.getDevicePtr() == NULL)
	printf("DEBUG_ERROR: filterOnDevice called with null device pointer.\n");
if (in.getAllocator() == NULL)
	printf("DEBUG_ERROR: filterOnDevice called with null allocator.\n");
#endif
	size_t temp_storage_size = 0;

	cudaStream_t stream = in.getStream();

	AccPtr<int>  num_selected_out(1, stream, in.getAllocator());
	num_selected_out.deviceAlloc();

	DEBUG_HANDLE_ERROR(cub::DeviceSelect::If(NULL, temp_storage_size, ~in, ~out, ~num_selected_out, in.getSize(), select_op, stream));

	if(temp_storage_size==0)
		temp_storage_size=1;

	CudaCustomAllocator::Alloc* alloc = in.getAllocator()->alloc(temp_storage_size);

	DEBUG_HANDLE_ERROR(cub::DeviceSelect::If(alloc->getPtr(), temp_storage_size, ~in, ~out, ~num_selected_out, in.getSize(), select_op, stream));

	num_selected_out.cpToHost();
	DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));

	in.getAllocator()->free(alloc);
	return num_selected_out[0];
}

template <typename T>
static void scanOnDevice(AccPtr<T> &in, AccPtr<T> &out)
{
#ifdef DEBUG_CUDA
if (in.getSize() == 0 || out.getSize() == 0)
	printf("DEBUG_ERROR: scanOnDevice called with pointer of zero size.\n");
if (in.getDevicePtr() == NULL || out.getDevicePtr() == NULL)
	printf("DEBUG_ERROR: scanOnDevice called with null device pointer.\n");
if (in.getAllocator() == NULL)
	printf("DEBUG_ERROR: scanOnDevice called with null allocator.\n");
#endif
	size_t temp_storage_size = 0;

	cudaStream_t stream = in.getStream();

	DEBUG_HANDLE_ERROR(cub::DeviceScan::InclusiveSum( NULL, temp_storage_size, ~in, ~out, in.getSize()));

	if(temp_storage_size==0)
		temp_storage_size=1;

	CudaCustomAllocator::Alloc* alloc = in.getAllocator()->alloc(temp_storage_size);

	DEBUG_HANDLE_ERROR(cub::DeviceScan::InclusiveSum( alloc->getPtr(), temp_storage_size, ~in, ~out, in.getSize(), stream));

	alloc->markReadyEvent(stream);
	alloc->doFreeWhenReady();
}


// The significance of one pass of convertAllSquaredDifferencesToWeights, computed on
// the GPU and read back in one copy instead of one round trip per value: the total
// weight, the threshold, its index in the cumulative sum, the significant weight and
// the maximum weight, with the same arithmetic as the host code.
template <typename T>
struct SignificanceResult
{
	T sum_weight;                 // cumulative[n - 1]
	T threshold;                  // (T) ((1 - adaptive_fraction) * sum_weight)
	unsigned long long idx;       // first index with cumulative > threshold (0 if none)
	unsigned long long idx_capped;// idx after the maximum_significants cap
	T significant_weight;         // sorted[idx_capped]
	cub::KeyValuePair<int, T> max;// arg max of the weights
	T wmax;                       // largest log-weight before the exponentiation (fused pass only)
};

template <typename T>
__global__ void cuda_kernel_significance_threshold(const T *cumulative, size_t n, double adaptive_fraction, SignificanceResult<T> *r)
{
	const T sum = cumulative[n - 1];
	r->sum_weight = sum;
	r->threshold = (T) ((1 - adaptive_fraction) * (double) sum);
	r->idx = 0;
}

template <typename T>
__global__ void cuda_kernel_significance_find(const T *cumulative, size_t size_m1, SignificanceResult<T> *r)
{
	const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	const T t = r->threshold;
	if (i < size_m1 && cumulative[i] <= t && t < cumulative[i + 1])
		r->idx = i + 1;
}

template <typename T>
__global__ void cuda_kernel_significance_value(const T *sorted, size_t n, long max_significants, SignificanceResult<T> *r)
{
	unsigned long long idx = r->idx;
	if (max_significants > 0 && (long) (n - idx) > max_significants)
		idx = n - max_significants;
	r->idx_capped = idx;
	r->significant_weight = (idx < n) ? sorted[idx] : (T) 0;
}

template <typename T>
__global__ void cuda_kernel_array_over_significant(const T *data, bool *passed, size_t size, const SignificanceResult<T> *r)
{
	const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
		passed[i] = (data[i] >= r->significant_weight);
}

// The three kernels above in one block, for up to a few tens of thousands of weights:
// the same values (the crossing found is unique, the cumulative sum being ordered)
template <typename T>
__global__ void cuda_kernel_significance_one_block(const T *cumulative, const T *sorted, size_t n,
		double adaptive_fraction, long max_significants, SignificanceResult<T> *r)
{
	__shared__ T t;
	__shared__ unsigned long long found;
	if (threadIdx.x == 0)
	{
		const T sum = cumulative[n - 1];
		r->sum_weight = sum;
		t = (T) ((1 - adaptive_fraction) * (double) sum);
		r->threshold = t;
		found = 0;
	}
	__syncthreads();
	for (size_t i = threadIdx.x; i + 1 < n; i += blockDim.x)
		if (cumulative[i] <= t && t < cumulative[i + 1])
			found = i + 1;
	__syncthreads();
	if (threadIdx.x == 0)
	{
		unsigned long long idx = found;
		r->idx = idx;
		if (max_significants > 0 && (long) (n - idx) > max_significants)
			idx = n - max_significants;
		r->idx_capped = idx;
		r->significant_weight = (idx < n) ? sorted[idx] : (T) 0;
	}
}

// Launch the above on sorted / cumulative (the sorted significant candidates and their
// cumulative sum) and weights (for the arg max), all on weights' stream; no wait
template <typename T>
static void significanceOnDevice(AccPtr<T> &sorted, AccPtr<T> &cumulative, AccPtr<T> &weights,
                                 double adaptive_fraction, long max_significants, SignificanceResult<T> *d_r,
                                 bool do_argmax = true)
{
	cudaStream_t stream = weights.getStream();
	const size_t n = cumulative.getSize();
	if (n <= ((size_t) 1 << 16))
		cuda_kernel_significance_one_block<T><<<1, 512, 0, stream>>>(~cumulative, ~sorted, n, adaptive_fraction, max_significants, d_r);
	else
	{
		cuda_kernel_significance_threshold<T><<<1, 1, 0, stream>>>(~cumulative, n, adaptive_fraction, d_r);
		const int bs = 512;
		cuda_kernel_significance_find<T><<<(int) ((n - 1 + bs - 1) / bs), bs, 0, stream>>>(~cumulative, n - 1, d_r);
		cuda_kernel_significance_value<T><<<1, 1, 0, stream>>>(~sorted, n, max_significants, d_r);
	}

	if (!do_argmax)
	{
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
		return;
	}
	size_t temp_storage_size = 0;
	DEBUG_HANDLE_ERROR(cub::DeviceReduce::ArgMax(NULL, temp_storage_size, ~weights, &d_r->max, weights.getSize()));
	if (temp_storage_size == 0)
		temp_storage_size = 1;
	CudaCustomAllocator::Alloc *alloc = weights.getAllocator()->alloc(temp_storage_size);
	DEBUG_HANDLE_ERROR(cub::DeviceReduce::ArgMax(alloc->getPtr(), temp_storage_size, ~weights, &d_r->max, weights.getSize(), stream));
	alloc->markReadyEvent(stream);
	alloc->doFreeWhenReady();
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
}

// ---- Sort, cumulative sum, significance (and arg max, mask) of up to BS_BLOCK * BS_ITEMS weights in ONE block ----
// (RELION_GPU_BLOCK_SORT=1). The per-particle passes of a 2D classification have a few thousand weights at most; there
// cub::DeviceRadixSort, DeviceScan, the significance kernel and DeviceReduce::ArgMax are four to seven launches of a single
// block each. The values are the ones of those calls except that the cumulative sum adds up in the order of a block scan
// (a few float ulps), and ties of the arg max go to the lowest index as in cub.
#define BS_ITEMS 4
#define BS_MAX_BLOCK 1024
#define BS_MAX_N (BS_MAX_BLOCK * BS_ITEMS)

template <typename T, int BS_BLOCK>
__global__ void cuda_kernel_block_sort_significance(const T *weights, int n, T *sorted_out, T *cumulative_out,
		double adaptive_fraction, long max_significants, SignificanceResult<T> *r, bool do_argmax, bool *mask_out)
{
	typedef cub::BlockRadixSort<T, BS_BLOCK, BS_ITEMS> Sort;
	typedef cub::BlockScan<T, BS_BLOCK> Scan;
	typedef cub::BlockReduce<cub::KeyValuePair<int, T>, BS_BLOCK> Reduce;
	__shared__ union { typename Sort::TempStorage sort; typename Scan::TempStorage scan; typename Reduce::TempStorage reduce; } temp;
	__shared__ unsigned long long s_found;
	const int tid = threadIdx.x;

	T keys[BS_ITEMS];
	for (int i = 0; i < BS_ITEMS; i++)
	{
		const int idx = tid * BS_ITEMS + i;
		keys[i] = idx < n ? weights[idx] : (T) INFINITY;
	}
	Sort(temp.sort).Sort(keys);
	__syncthreads();
	T cum[BS_ITEMS];
	Scan(temp.scan).InclusiveSum(keys, cum);
	for (int i = 0; i < BS_ITEMS; i++)
	{
		const int idx = tid * BS_ITEMS + i;
		if (idx < n)
		{
			sorted_out[idx] = keys[i];
			cumulative_out[idx] = cum[i];
		}
	}
	if (tid == 0)
		s_found = 0;
	__syncthreads(); // sorted_out and cumulative_out (global, 64 KB of shared memory would not fit) are read back by the whole block
	const T sum = cumulative_out[n - 1];
	const T t = (T) ((1 - adaptive_fraction) * (double) sum);
	for (int i = tid; i + 1 < n; i += BS_BLOCK)
		if (cumulative_out[i] <= t && t < cumulative_out[i + 1])
			s_found = i + 1;
	__syncthreads();
	unsigned long long idx = s_found;
	unsigned long long idx_capped = idx;
	if (max_significants > 0 && (long) (n - idx) > max_significants)
		idx_capped = n - max_significants;
	const T sig_w = (idx_capped < (unsigned long long) n) ? sorted_out[idx_capped] : (T) 0;
	if (tid == 0)
	{
		r->sum_weight = sum;
		r->threshold = t;
		r->idx = idx;
		r->idx_capped = idx_capped;
		r->significant_weight = sig_w;
	}
	if (do_argmax)
	{
		cub::KeyValuePair<int, T> kv(0, (T) -INFINITY);
		for (int i = tid; i < n; i += BS_BLOCK)
		{
			const T v = weights[i];
			if (v > kv.value)
			{
				kv.key = i;
				kv.value = v;
			}
		}
		__syncthreads();
		kv = Reduce(temp.reduce).Reduce(kv, cub::ArgMax());
		if (tid == 0)
			r->max = kv;
	}
	if (mask_out != NULL)
		for (int i = tid; i < n; i += BS_BLOCK)
			mask_out[i] = (weights[i] >= sig_w);
}

// RELION_GPU_BLOCK_SORT=0 always uses the radix sort of the weights (the block-wise sort is on by default)
inline bool gpuBlockSort()
{
	static const bool on = gpuswitch::on("RELION_GPU_BLOCK_SORT");
	return on;
}

// True (and launched, on weights' stream, nothing waited for) if weights fit the block; sorted and cumulative are
// filled as after sortOnDevice and scanOnDevice; r gets the SignificanceResult (with the arg max if asked)
template <typename T>
static bool blockSortSignificance(AccPtr<T> &weights, T *sorted, T *cumulative, double adaptive_fraction,
		long max_significants, SignificanceResult<T> *d_r, bool do_argmax, bool *d_mask = NULL)
{
	const size_t n = weights.getSize();
	if (n < 1 || n > BS_MAX_N)
		return false;
	// the smallest block that holds n weights: the sort is a fixed number of passes, each as long as one thread's share
#define BS_LAUNCH(B) cuda_kernel_block_sort_significance<T, B><<<1, B, 0, weights.getStream()>>>(~weights, (int) n, sorted, cumulative, \
			adaptive_fraction, max_significants, d_r, do_argmax, d_mask)
	if (n <= 128 * BS_ITEMS)      BS_LAUNCH(128);
	else if (n <= 256 * BS_ITEMS) BS_LAUNCH(256);
	else if (n <= 512 * BS_ITEMS) BS_LAUNCH(512);
	else                          BS_LAUNCH(1024);
#undef BS_LAUNCH
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
	return true;
}

// ---- The significance pass of the coarse weights in one go (RELION_GPU_FUSED_WEIGHTS) ----
// The default code makes about fifty driver calls per pass (the maximum read back to the host,
// the compaction of the positive weights with its count read back, the sort, the scan, ...).
// Here the maximum stays on the GPU (partial maxima per block, reduced again by every block
// of the exponentiation), the weights are sorted without compaction (the zeros sort to the
// front and add nothing to the cumulative sum, so the threshold and the significant weight
// are those of the compacted array), and the sum, threshold, significant weight, arg max and
// the significance mask come back in one copy.

template <typename T>
struct MaxIdx { T v; int i; };

template <typename T>
__device__ __forceinline__ bool maxidx_better(T v1, int i1, T v2, int i2)
{
	return (v1 > v2) || (v1 == v2 && i1 < i2);
}

// First stage: the largest value (lowest index on ties) of each block's slice
template <typename T>
__global__ void cuda_kernel_fused_max_partial(const T *g, size_t n, MaxIdx<T> *partial)
{
	T bv = -INFINITY;
	int bi = 0x7fffffff;
	for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
	{
		const T v = g[i];
		if (maxidx_better(v, (int) i, bv, bi))
		{
			bv = v;
			bi = (int) i;
		}
	}
	__shared__ T sv[32];
	__shared__ int si[32];
	for (int o = 16; o > 0; o >>= 1)
	{
		const T ov = __shfl_down_sync(0xffffffff, bv, o);
		const int oi = __shfl_down_sync(0xffffffff, bi, o);
		if (maxidx_better(ov, oi, bv, bi))
		{
			bv = ov;
			bi = oi;
		}
	}
	if ((threadIdx.x & 31) == 0)
	{
		sv[threadIdx.x >> 5] = bv;
		si[threadIdx.x >> 5] = bi;
	}
	__syncthreads();
	if (threadIdx.x < 32)
	{
		bv = (threadIdx.x < (blockDim.x >> 5)) ? sv[threadIdx.x] : (T) -INFINITY;
		bi = (threadIdx.x < (blockDim.x >> 5)) ? si[threadIdx.x] : 0x7fffffff;
		for (int o = 16; o > 0; o >>= 1)
		{
			const T ov = __shfl_down_sync(0xffffffff, bv, o);
			const int oi = __shfl_down_sync(0xffffffff, bi, o);
			if (maxidx_better(ov, oi, bv, bi))
			{
				bv = ov;
				bi = oi;
			}
		}
		if (threadIdx.x == 0)
			partial[blockIdx.x] = MaxIdx<T>{bv, bi};
	}
}

// Second stage: every block reduces the partial maxima (a few hundred values at most), then
// exponentiates its weights as cuda_kernel_exponentiate does, with add = target - maximum
template <typename T>
__global__ void cuda_kernel_fused_exponentiate(T *g, size_t n, const MaxIdx<T> *partial, int n_partial,
                                               T target, SignificanceResult<T> *r)
{
	__shared__ T sv[32];
	__shared__ int si[32];
	__shared__ T s_max;
	__shared__ int s_idx;
	T bv = -INFINITY;
	int bi = 0x7fffffff;
	if ((int) threadIdx.x < n_partial)
	{
		bv = partial[threadIdx.x].v;
		bi = partial[threadIdx.x].i;
	}
	for (int o = 16; o > 0; o >>= 1)
	{
		const T ov = __shfl_down_sync(0xffffffff, bv, o);
		const int oi = __shfl_down_sync(0xffffffff, bi, o);
		if (maxidx_better(ov, oi, bv, bi))
		{
			bv = ov;
			bi = oi;
		}
	}
	if ((threadIdx.x & 31) == 0)
	{
		sv[threadIdx.x >> 5] = bv;
		si[threadIdx.x >> 5] = bi;
	}
	__syncthreads();
	if (threadIdx.x < 32)
	{
		bv = (threadIdx.x < (blockDim.x >> 5)) ? sv[threadIdx.x] : (T) -INFINITY;
		bi = (threadIdx.x < (blockDim.x >> 5)) ? si[threadIdx.x] : 0x7fffffff;
		for (int o = 16; o > 0; o >>= 1)
		{
			const T ov = __shfl_down_sync(0xffffffff, bv, o);
			const int oi = __shfl_down_sync(0xffffffff, bi, o);
			if (maxidx_better(ov, oi, bv, bi))
			{
				bv = ov;
				bi = oi;
			}
		}
		if (threadIdx.x == 0)
		{
			s_max = bv;
			s_idx = bi;
		}
	}
	__syncthreads();
	const T add = target - s_max;
	const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	if (i == 0)
		r->wmax = s_max;
	if (i < n)
	{
		const T a = g[i] + add;
		const T e = (a < (T) -88.f) ? (T) 0. : expf(a);
		g[i] = e;
		if ((int) i == s_idx)
		{
			r->max.key = (int) i;
			r->max.value = e;
		}
	}
}

// Needs at most n_scratch bytes of scratch, see fusedWeightScratchBytes
template <typename T>
static size_t fusedWeightScratchBytes(size_t n)
{
	size_t sort_tmp = 0, scan_tmp = 0;
	cub::DeviceRadixSort::SortKeys((void *) NULL, sort_tmp, (const T *) NULL, (T *) NULL, (int) n);
	cub::DeviceScan::InclusiveSum((void *) NULL, scan_tmp, (const T *) NULL, (T *) NULL, (int) n);
	const size_t al = 256;
	auto up = [&](size_t b) { return (b + al - 1) / al * al; };
	return up(n * sizeof(T)) * 2 + up(128 * sizeof(MaxIdx<T>)) + up(std::max(sort_tmp, scan_tmp) + 1);
}

// weights: the log-weights of the coarse pass (excluded ones very negative); on return they
// are the exponentiated weights (as after kernel_exponentiate with 50 - maximum). res (device
// and host) receives a SignificanceResult followed by n bytes of significance mask (if
// want_mask); the caller copies it to the host and waits. Nothing here waits.
template <typename T>
static void fusedWeightPass(AccPtr<T> &weights, double adaptive_fraction, long max_significants,
                            char *d_res, bool want_mask)
{
	cudaStream_t stream = weights.getStream();
	const size_t n = weights.getSize();
	const size_t al = 256;
	auto up = [&](size_t b) { return (b + al - 1) / al * al; };
	size_t sort_tmp = 0, scan_tmp = 0;
	DEBUG_HANDLE_ERROR(cub::DeviceRadixSort::SortKeys((void *) NULL, sort_tmp, (const T *) NULL, (T *) NULL, (int) n));
	DEBUG_HANDLE_ERROR(cub::DeviceScan::InclusiveSum((void *) NULL, scan_tmp, (const T *) NULL, (T *) NULL, (int) n));
	const size_t tmp_bytes = std::max(sort_tmp, scan_tmp) + 1;
	const size_t total = fusedWeightScratchBytes<T>(n);

	CudaCustomAllocator::Alloc *alloc = weights.getAllocator()->alloc(total);
	char *base = (char *) alloc->getPtr();
	T *sorted = (T *) base;
	T *cumulative = (T *) (base + up(n * sizeof(T)));
	MaxIdx<T> *partial = (MaxIdx<T> *) (base + 2 * up(n * sizeof(T)));
	void *tmp = base + 2 * up(n * sizeof(T)) + up(128 * sizeof(MaxIdx<T>));

	SignificanceResult<T> *r = (SignificanceResult<T> *) d_res;
	const int bs = 512;
	const int n_partial = (int) std::min<size_t>((n + bs - 1) / bs, 128);
	cuda_kernel_fused_max_partial<T><<<n_partial, bs, 0, stream>>>(~weights, n, partial);
	cuda_kernel_fused_exponentiate<T><<<(unsigned) ((n + bs - 1) / bs), bs, 0, stream>>>(~weights, n, partial, n_partial, (T) 50, r);

	if (gpuBlockSort() && blockSortSignificance<T>(weights, sorted, cumulative, adaptive_fraction, max_significants, r, false,
			want_mask ? (bool *) (d_res + sizeof(SignificanceResult<T>)) : NULL))
	{
		// sort, sum, threshold and mask in one launch (the maximum was found by the first two)
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
		alloc->markReadyEvent(stream);
		alloc->doFreeWhenReady();
		return;
	}
	size_t t1 = tmp_bytes;
	DEBUG_HANDLE_ERROR(cub::DeviceRadixSort::SortKeys(tmp, t1, ~weights, sorted, (int) n, 0, sizeof(T) * 8, stream));
	size_t t2 = tmp_bytes;
	DEBUG_HANDLE_ERROR(cub::DeviceScan::InclusiveSum(tmp, t2, sorted, cumulative, (int) n, stream));

	if (n <= ((size_t) 1 << 16))
		cuda_kernel_significance_one_block<T><<<1, 512, 0, stream>>>(cumulative, sorted, n, adaptive_fraction, max_significants, r);
	else
	{
		cuda_kernel_significance_threshold<T><<<1, 1, 0, stream>>>(cumulative, n, adaptive_fraction, r);
		cuda_kernel_significance_find<T><<<(int) ((n - 1 + bs - 1) / bs), bs, 0, stream>>>(cumulative, n - 1, r);
		cuda_kernel_significance_value<T><<<1, 1, 0, stream>>>(sorted, n, max_significants, r);
	}
	if (want_mask)
		cuda_kernel_array_over_significant<T><<<(unsigned) ((n + bs - 1) / bs), bs, 0, stream>>>(
				~weights, (bool *) (d_res + sizeof(SignificanceResult<T>)), n, r);
	LAUNCH_HANDLE_ERROR(cudaGetLastError());

	alloc->markReadyEvent(stream);
	alloc->doFreeWhenReady();
}

} // namespace CudaKernels
#endif
