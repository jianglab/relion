#ifndef CUDA_UTILS_CUB_CUH_
#define CUDA_UTILS_CUB_CUH_

#include <cuda.h> // For CUDA_VERSION
#include <cuda_runtime.h>
#include "src/acc/cuda/cuda_settings.h"
#include "src/gpu_switch.h"
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
#include <cooperative_groups.h>
#include <atomic>
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
	T first;                      // sorted[0], the smallest weight (block-wise sort only)
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
	if (tid == 0)
		r->first = keys[0];
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

// ---- The significance without a sort: radix selection (RELION_GPU_SIGNIF_SELECT) ----
// The sorted array is only used to find where its running sum passes (1 - adaptive_fraction) of the total, and the
// value at that position. Both are found here from histograms of the weights' bit patterns (positive floats order
// like their bits): three levels of 11, 10 and 10 bits, each one pass over the weights (plus one pass of one block over
// the bins), pick the bin where the threshold falls, and after the last level the exact weight and the position
// inside the group of equal weights are known. The sums are made in double precision (the sorted-array way sums in
// float and so has a few ulps of noise in the threshold), so the result is the one of the sort except where the
// threshold falls within float noise of a weight. With a cap on the number of significant weights, a second chain
// selects the weight of that rank.
struct RadixSelState
{
	double target;                   // sum mode: threshold of the running sum; rank mode: the sorted position wanted
	double below_sum;                // sum of the weights whose bits are below the chosen prefix
	unsigned long long below_count;  // and their number
	unsigned long long group_count;  // number of weights with exactly the chosen prefix
	unsigned prefix;
	unsigned pad;
};

#define RS_BLOCK 256
#define RS_BINS (2048 + 1024 + 1024)

__device__ __forceinline__ void rs_level(int level, int &top, int &sh, int &nb, int &off)
{
	top = level == 0 ? 31 : (level == 1 ? 20 : 10);
	sh  = level == 0 ? 20 : (level == 1 ? 10 : 0);
	nb  = level == 0 ? 2048 : 1024;
	off = level == 0 ? 0 : (level == 1 ? 2048 : 3072);
}

// atomicAdd(double *, double) is native only from sm_60; below that, the atomicCAS loop of helper.cuh
static __device__ __forceinline__ void rs_atomic_add(double *address, double val)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
	atomicAdd(address, val);
#else
	unsigned long long int *address_as_ull = (unsigned long long int *)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
#endif
}

// body of cuda_kernel_rs_hist (shared with cuda_kernel_rs_all): s_cnt / s_sum are 2048 shared bins each
static __device__ __forceinline__ void rs_hist_body(const float *w, size_t n, const RadixSelState *st, unsigned *g_cnt, double *g_sum, int level,
		unsigned *s_cnt, double *s_sum)
{
	int top, sh, nb, off;
	rs_level(level, top, sh, nb, off);
	const unsigned prefix = st->prefix;
	for (int i = threadIdx.x; i < nb; i += blockDim.x)
	{
		s_cnt[i] = 0;
		s_sum[i] = 0.;
	}
	__syncthreads();
	for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
	{
		const float v = w[i];
		const unsigned u = __float_as_uint(v);
		if ((u >> top) == prefix)
		{
			const int d = (u >> sh) & (nb - 1);
			atomicAdd(&s_cnt[d], 1u);
			if (v != 0.f)
				rs_atomic_add(&s_sum[d], (double) v);
		}
	}
	__syncthreads();
	for (int i = threadIdx.x; i < nb; i += blockDim.x)
	{
		if (s_cnt[i] != 0)
		{
			atomicAdd(&g_cnt[off + i], s_cnt[i]);
			if (s_sum[i] != 0.)
				rs_atomic_add(&g_sum[off + i], s_sum[i]);
		}
	}
}

static __global__ void cuda_kernel_rs_hist(const float *w, size_t n, const RadixSelState *st, unsigned *g_cnt, double *g_sum, int level)
{
	__shared__ unsigned s_cnt[2048];
	__shared__ double s_sum[2048];
	rs_hist_body(w, n, st, g_cnt, g_sum, level, s_cnt, s_sum);
}

// One block of 1024 threads: scan the bins of a level, choose the bin, and after the last level write the result
// mode 0: the position where the running sum passes (1 - adaptive_fraction) of the total, and its weight
// mode 1: the weight at sorted position rank_target, used in place of that weight if the count above exceeds max_sig
static __device__ __forceinline__ void rs_select_body(RadixSelState *st, const unsigned *g_cnt, const double *g_sum,
		int level, int mode, double adaptive_fraction, double rank_target, unsigned long long n, long max_sig,
		SignificanceResult<float> *r, double *sh_s, unsigned long long *sh_c, int &s_found, int &s_last, int fast_scan)
{
	const int tid = threadIdx.x;
	__shared__ unsigned long long s_woff_c[32];
	__shared__ double s_woff_s[32];
	int top, sh, nb, off;
	rs_level(level, top, sh, nb, off);
	const int b0 = 2 * tid, b1 = b0 + 1;
	const unsigned c0 = b0 < nb ? g_cnt[off + b0] : 0u, c1 = b1 < nb ? g_cnt[off + b1] : 0u;
	const double s0 = b0 < nb ? g_sum[off + b0] : 0., s1 = b1 < nb ? g_sum[off + b1] : 0.;
	sh_c[tid] = (unsigned long long) c0 + c1;
	sh_s[tid] = s0 + s1;
	if (tid == 0)
	{
		s_found = 0x7fffffff;
		s_last = -1;
	}
	__syncthreads();
	if (!fast_scan)
	{
	for (int d = 1; d < 1024; d <<= 1)
	{
		unsigned long long cv = 0;
		double sv = 0.;
		if (tid >= d)
		{
			cv = sh_c[tid - d];
			sv = sh_s[tid - d];
		}
		__syncthreads();
		sh_c[tid] += cv;
		sh_s[tid] += sv;
		__syncthreads();
	}
	}
	else
	{
		// the same inclusive scan with warp shuffles: 5 steps in registers, the 32 warp totals scanned by warp 0 (3 barriers instead of 20)
		const int lane = tid & 31, wid = tid >> 5;
		unsigned long long cv = sh_c[tid];
		double sv = sh_s[tid];
		#pragma unroll
		for (int d = 1; d < 32; d <<= 1)
		{
			const unsigned long long cu = __shfl_up_sync(0xffffffffu, cv, d);
			const double su = __shfl_up_sync(0xffffffffu, sv, d);
			if (lane >= d)
			{
				cv += cu;
				sv += su;
			}
		}
		__syncthreads();
		sh_c[tid] = cv;
		sh_s[tid] = sv;
		__syncthreads();
		if (wid == 0)
		{
			// warp totals (the last lane of each warp) scanned inclusive in the 32 lanes of warp 0
			unsigned long long tc = sh_c[lane * 32 + 31];
			double ts = sh_s[lane * 32 + 31];
			#pragma unroll
			for (int d = 1; d < 32; d <<= 1)
			{
				const unsigned long long cu = __shfl_up_sync(0xffffffffu, tc, d);
				const double su = __shfl_up_sync(0xffffffffu, ts, d);
				if (lane >= d)
				{
					tc += cu;
					ts += su;
				}
			}
			__syncwarp();
			unsigned long long oc = __shfl_up_sync(0xffffffffu, tc, 1);
			double os = __shfl_up_sync(0xffffffffu, ts, 1);
			if (lane == 0) { oc = 0; os = 0.; }
			// exclusive offset of warp w = inclusive total of warp w - 1: through shared memory to every thread
			s_woff_c[lane] = oc;
			s_woff_s[lane] = os;
		}
		__syncthreads();
		sh_c[tid] += s_woff_c[wid];
		sh_s[tid] += s_woff_s[wid];
		__syncthreads();
	}
	const unsigned long long ex_c = tid > 0 ? sh_c[tid - 1] : 0ull;
	const double ex_s = tid > 0 ? sh_s[tid - 1] : 0.;

	double bsum, target;
	unsigned long long bcnt;
	unsigned prefix;
	if (level == 0)
	{
		const double total = sh_s[1023];
		bsum = 0.;
		bcnt = 0;
		prefix = 0;
		if (mode == 0)
		{
			const double t = (1.0 - adaptive_fraction) * total;
			target = (t >= total) ? -1.0 : t; // as the sorted array: no crossing means position 0
			if (tid == 0)
			{
				r->sum_weight = (float) total;
				r->threshold = (float) t;
				r->idx = 0;
				r->idx_capped = 0;
			}
		}
		else
			target = rank_target;
	}
	else
	{
		target = st->target;
		bsum = st->below_sum;
		bcnt = st->below_count;
		prefix = st->prefix;
	}

	double run_s = bsum + ex_s;
	unsigned long long run_c = bcnt + ex_c;
	for (int j = 0; j < 2; j++)
	{
		const int b = b0 + j;
		if (b >= nb)
			break;
		const unsigned c = j ? c1 : c0;
		const double s = j ? s1 : s0;
		if (c > 0)
		{
			atomicMax(&s_last, b);
			const bool cross = (mode == 0) ? (run_s + s > target) : ((double) (run_c + c) > target);
			if (cross)
				atomicMin(&s_found, b);
		}
		run_s += s;
		run_c += c;
	}
	__syncthreads();
	int d = s_found != 0x7fffffff ? s_found : s_last; // none: rounding at the very top, take the last bin
	if (d < 0)
		d = 0;
	if (tid == (d >> 1))
	{
		const bool odd = d & 1;
		const double nsum = bsum + ex_s + (odd ? s0 : 0.);
		const unsigned long long ncnt = bcnt + ex_c + (odd ? (unsigned long long) c0 : 0ull);
		const unsigned long long cg = odd ? c1 : c0;
		const unsigned np = (prefix << (level == 0 ? 11 : 10)) | (unsigned) d;
		st->below_sum = nsum;
		st->below_count = ncnt;
		st->group_count = cg;
		st->prefix = np;
		st->target = target;
		if (level == 2)
		{
			const float v = __uint_as_float(np);
			if (mode == 0)
			{
				double m = 1.;
				if (v > 0.f && target >= nsum)
					m = floor((target - nsum) / (double) v) + 1.0;
				if (m < 1.) m = 1.;
				if (m > (double) cg) m = (double) cg;
				const unsigned long long idx = ncnt + (unsigned long long) (m - 1.);
				r->idx = idx;
				r->idx_capped = idx;
				r->significant_weight = v;
			}
			else if (max_sig > 0 && (long) (n - r->idx) > max_sig)
			{
				r->idx_capped = n - (unsigned long long) max_sig;
				r->significant_weight = v;
			}
		}
	}
}

static __global__ void __launch_bounds__(1024) cuda_kernel_rs_select(RadixSelState *st, const unsigned *g_cnt, const double *g_sum,
		int level, int mode, double adaptive_fraction, double rank_target, unsigned long long n, long max_sig,
		SignificanceResult<float> *r)
{
	__shared__ double sh_s[1024];
	__shared__ unsigned long long sh_c[1024];
	__shared__ int s_found, s_last;
	rs_select_body(st, g_cnt, g_sum, level, mode, adaptive_fraction, rank_target, n, max_sig, r, sh_s, sh_c, s_found, s_last, 0);
}

// ---- RELION_GPU_SELECT_FUSED=1: the whole selection (3 levels, 1 or 2 chains) in ONE cooperative launch ----
// cuda_kernel_rs_hist and cuda_kernel_rs_select alternate 3 x (1 or 2) times, each its own launch (a pass of 48 blocks, then one block
// of 1024 threads on the bins): 6-12 launches and their gaps per particle. Here every block does its part of the histogram, a grid-wide
// barrier, block 0 chooses the bin, a barrier, and so on: the same two bodies, so the same values (the histogram sums are added in
// shared/global atomics as before: double rounding order only). All blocks are 1024 threads and co-resident (at most one per SM).
static __global__ void __launch_bounds__(1024) cuda_kernel_rs_all(const float *w, size_t n, char *base, size_t chain_bytes, int nchain,
		double adaptive_fraction, double rank_target, unsigned long long nn, long max_sig, SignificanceResult<float> *r, int fast_scan)
{
	__shared__ unsigned s_cnt[2048];
	__shared__ double s_sum[2048];
	__shared__ double sh_s[1024];
	__shared__ unsigned long long sh_c[1024];
	__shared__ int s_found, s_last;
	for (int c = 0; c < nchain; c++)
	{
		RadixSelState *st = (RadixSelState *) (base + c * chain_bytes);
		unsigned *cnt = (unsigned *) (base + c * chain_bytes + 64);
		double *sum = (double *) (base + c * chain_bytes + 64 + RS_BINS * sizeof(unsigned));
		for (int level = 0; level < 3; level++)
		{
			rs_hist_body(w, n, st, cnt, sum, level, s_cnt, s_sum);
			cooperative_groups::this_grid().sync();
			if (blockIdx.x == 0)
				rs_select_body(st, cnt, sum, level, c, adaptive_fraction, rank_target, nn, max_sig, r, sh_s, sh_c, s_found, s_last, fast_scan);
			cooperative_groups::this_grid().sync();
		}
	}
}

// -1: follow RELION_GPU_SIGNIF_SELECT; 0 or 1: forced (tests)
inline int &gpuSignifSelectOverride()
{
	static int v = -1;
	return v;
}

inline bool gpuSignifSelect()
{
	static const bool on = gpuswitch::on("RELION_GPU_SIGNIF_SELECT");
	return gpuSignifSelectOverride() >= 0 ? gpuSignifSelectOverride() != 0 : on;
}

inline size_t radixSelectScratchBytes()
{
	return 2 * (64 + RS_BINS * sizeof(unsigned) + RS_BINS * sizeof(double));
}

inline int radixSelectBlocks(size_t n)
{
	static int sms[64] = {0};
	int dev = 0;
	cudaGetDevice(&dev);
	if (dev < 0 || dev >= 64)
		dev = 0;
	if (sms[dev] == 0)
	{
		int v = 0;
		cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev);
		sms[dev] = v > 0 ? v : 16;
	}
	const size_t per = (size_t) RS_BLOCK * 8;
	return (int) std::max<size_t>(1, std::min<size_t>((n + per - 1) / per, 2 * (size_t) sms[dev]));
}

// RELION_GPU_SELECT_FUSED: unset or on = 2 (one cooperative launch, warp-shuffle scans), "1" = one launch with the plain scans, off = 0
inline int rsFusedMode()
{
	const char *e = getenv("RELION_GPU_SELECT_FUSED");
	if (e == NULL)
		return 2;
	if (gpuswitch::isOffValue(e))
		return 0;
	return atoi(e) == 1 ? 1 : 2;
}

// how many selections ran as one cooperative launch (0) and how many fell back to the separate launches (1)
// test hook (RELION_GPU_SELECT_FUSED_REFUSE=1 or set from a unit test): ask for more blocks than can be co-resident, so the driver refuses the cooperative launch and the fallback runs
inline std::atomic<int> &rsFusedRefuse(){ static std::atomic<int> v(getenv("RELION_GPU_SELECT_FUSED_REFUSE") != NULL && atoi(getenv("RELION_GPU_SELECT_FUSED_REFUSE")) != 0); return v; }
inline std::atomic<long> &rsFusedCount(int which)
{
	static std::atomic<long> c[2];
	return c[which];
}

// Fills r (sum_weight, threshold, idx, idx_capped, significant_weight) as the sort, scan and significance kernels do.
// scratch: radixSelectScratchBytes() bytes, device. Everything on stream, nothing waited for.
static void radixSelectSignificance(const float *weights, size_t n, cudaStream_t stream, void *scratch, double adaptive_fraction,
		long max_significants, SignificanceResult<float> *r)
{
	char *base = (char *) scratch;
	const size_t chain_bytes = 64 + RS_BINS * sizeof(unsigned) + RS_BINS * sizeof(double);
	cudaMemsetAsync(base, 0, 2 * chain_bytes, stream);
	const int nchain = (max_significants > 0 && n > (size_t) max_significants) ? 2 : 1;
	const int blocks = radixSelectBlocks(n);
	// RELION_GPU_SELECT_FUSED: on by default (2); 1 = one launch with the plain scans of the bins, 2 = also warp-shuffle scans, 0/off = separate launches
	static const int fused_mode = rsFusedMode();
	const bool fused = fused_mode > 0;
	if (fused)
	{
		// one cooperative launch: as many blocks as fit, one per SM at most (1024 threads each)
		// the co-resident capacity of the card, found once per device; a cooperative launch that does not fit fails (no hang)
		static int capacity[64] = {0};
		int dev = 0;
		cudaGetDevice(&dev);
		if (dev < 0 || dev >= 64)
			dev = 0;
		if (capacity[dev] == 0)
		{
			int sms = 0, per_sm = 0, coop = 0;
			cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
			cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev);
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, cuda_kernel_rs_all, 1024, 0);
			capacity[dev] = coop ? std::max(1, per_sm) * std::max(1, sms) : -1;
		}
		if (capacity[dev] > 0) // else: no cooperative launch on this card, the separate launches below
		{
			const int cb = rsFusedRefuse().load() ? capacity[dev] + 1 : std::max(1, std::min(blocks, capacity[dev]));
			SignificanceResult<float> *rr = r;
			double rank_target = (double) (n - (size_t) max_significants);
			unsigned long long nn = (unsigned long long) n;
			size_t cbytes = chain_bytes;
			int nch = nchain;
			long ms = max_significants;
			int fast = fused_mode >= 2 ? 1 : 0;
			void *args[] = { &weights, &n, &base, &cbytes, &nch, &adaptive_fraction, &rank_target, &nn, &ms, &rr, &fast };
			const cudaError_t coop_err = cudaLaunchCooperativeKernel((void *) cuda_kernel_rs_all, dim3(cb), dim3(1024), args, 0, stream);
			if (coop_err == cudaSuccess)
			{
				rsFusedCount(0)++;
				LAUNCH_HANDLE_ERROR(cudaGetLastError());
				return;
			}
			cudaGetLastError(); // not launchable: the separate launches below
			if (rsFusedCount(1)++ == 0)
				fprintf(stderr, " WARNING: RELION_GPU_SELECT_FUSED: the cooperative launch was refused (%s), the selection uses separate launches.\n", cudaGetErrorString(coop_err));
		}
		else if (rsFusedCount(1)++ == 0)
			fprintf(stderr, " WARNING: RELION_GPU_SELECT_FUSED: this card cannot launch cooperative kernels, the selection uses separate launches.\n");
	}
	for (int c = 0; c < nchain; c++)
	{
		RadixSelState *st = (RadixSelState *) (base + c * chain_bytes);
		unsigned *cnt = (unsigned *) (base + c * chain_bytes + 64);
		double *sum = (double *) (base + c * chain_bytes + 64 + RS_BINS * sizeof(unsigned));
		for (int level = 0; level < 3; level++)
		{
			cuda_kernel_rs_hist<<<blocks, RS_BLOCK, 0, stream>>>(weights, n, st, cnt, sum, level);
			cuda_kernel_rs_select<<<1, 1024, 0, stream>>>(st, cnt, sum, level, c, adaptive_fraction,
					(double) (n - (size_t) max_significants), (unsigned long long) n, max_significants, r);
		}
	}
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
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
	if (sizeof(T) == sizeof(float) && gpuSignifSelect() && n > ((size_t) 1 << 16) && n < ((size_t) 1 << 31) &&
			adaptive_fraction > 0. && adaptive_fraction < 1.)
	{
		// no sort: the maximum, the exponentiation, and the radix selection of the threshold weight
		const size_t part_bytes = up(128 * sizeof(MaxIdx<T>));
		CudaCustomAllocator::Alloc *ralloc = weights.getAllocator()->alloc(part_bytes + radixSelectScratchBytes());
		char *rbase = (char *) ralloc->getPtr();
		MaxIdx<T> *rpartial = (MaxIdx<T> *) rbase;
		SignificanceResult<T> *rr = (SignificanceResult<T> *) d_res;
		const int rbs = 512;
		const int rn_partial = (int) std::min<size_t>((n + rbs - 1) / rbs, 128);
		cuda_kernel_fused_max_partial<T><<<rn_partial, rbs, 0, stream>>>(~weights, n, rpartial);
		cuda_kernel_fused_exponentiate<T><<<(unsigned) ((n + rbs - 1) / rbs), rbs, 0, stream>>>(~weights, n, rpartial, rn_partial, (T) 50, rr);
		radixSelectSignificance((const float *) ~weights, n, stream, rbase + part_bytes, adaptive_fraction, max_significants,
				(SignificanceResult<float> *) rr);
		if (want_mask)
			cuda_kernel_array_over_significant<T><<<(unsigned) ((n + rbs - 1) / rbs), rbs, 0, stream>>>(
					~weights, (bool *) (d_res + sizeof(SignificanceResult<T>)), n, rr);
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
		ralloc->markReadyEvent(stream);
		ralloc->doFreeWhenReady();
		return;
	}
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

// RELION_GPU_SKIPALIGN_BLOCK (on by default, =0 turns it off): the fine pass of --skip_align (no oversampling, a few weights per particle) in three launches
// and one copy: partial maximum, exponentiation (the maximum stays on the GPU) and one block that sorts, sums and finds the
// threshold. Replaces DeviceReduce::Max, the exponentiation, DeviceRadixSort, DeviceScan, DeviceReduce::ArgMax and the reads
// of the sum, the smallest weight and the threshold. r: sum_weight, idx, first (smallest weight), max, wmax.
inline bool gpuSkipAlignBlock()
{
	static const bool on = gpuswitch::on("RELION_GPU_SKIPALIGN_BLOCK");
	return on;
}

template <typename T>
static bool skipAlignWeightPass(AccPtr<T> &weights, double adaptive_fraction, char *d_res)
{
	cudaStream_t stream = weights.getStream();
	const size_t n = weights.getSize();
	if (n < 1 || n > BS_MAX_N)
		return false;
	const size_t al = 256;
	auto up = [&](size_t b) { return (b + al - 1) / al * al; };
	CudaCustomAllocator::Alloc *alloc = weights.getAllocator()->alloc(up(n * sizeof(T)) * 2 + up(128 * sizeof(MaxIdx<T>)));
	char *base = (char *) alloc->getPtr();
	T *sorted = (T *) base;
	T *cumulative = (T *) (base + up(n * sizeof(T)));
	MaxIdx<T> *partial = (MaxIdx<T> *) (base + 2 * up(n * sizeof(T)));
	SignificanceResult<T> *r = (SignificanceResult<T> *) d_res;
	const int bs = 512;
	const int n_partial = (int) std::min<size_t>((n + bs - 1) / bs, 128);
	cuda_kernel_fused_max_partial<T><<<n_partial, bs, 0, stream>>>(~weights, n, partial);
	cuda_kernel_fused_exponentiate<T><<<(unsigned) ((n + bs - 1) / bs), bs, 0, stream>>>(~weights, n, partial, n_partial, (T) 50, r);
	blockSortSignificance<T>(weights, sorted, cumulative, adaptive_fraction, 0, r, false, NULL);
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
	alloc->markReadyEvent(stream);
	alloc->doFreeWhenReady();
	return true;
}

} // namespace CudaKernels
#endif
