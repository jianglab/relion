#ifndef CUDA_DEVICE_MEM_UTILS_H_
#define CUDA_DEVICE_MEM_UTILS_H_

#ifdef _CUDA_ENABLED
#include <cuda_runtime.h>
#include <cstring>
#include <string>
#include <algorithm>
#include <curand.h>
#include "src/acc/cuda/cuda_settings.h"
#include "src/acc/cuda/custom_allocator.cuh"
#endif


#include <signal.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "src/complex.h"

// Forward definition
template <typename T>  class AccPtr;

/**
 * Print cuda device memory info
 */
static void cudaPrintMemInfo()
{
	size_t free;
	size_t total;
	DEBUG_HANDLE_ERROR(cudaMemGetInfo( &free, &total ));
	float free_hr(free/(1024.*1024.));
	float total_hr(total/(1024.*1024.));
    printf( "free %.2fMiB, total %.2fMiB, used %.2fMiB\n",
    		free_hr, total_hr, total_hr - free_hr);
}

template< typename T>
static inline
void cudaCpyHostToDevice( T *h_ptr, T *d_ptr, size_t size)
{
	DEBUG_HANDLE_ERROR(cudaMemcpy( d_ptr, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice));
};

template< typename T>
static inline
void cudaCpyHostToDevice( T *h_ptr, T *d_ptr, size_t size, cudaStream_t &stream)
{
	DEBUG_HANDLE_ERROR(cudaMemcpyAsync( d_ptr, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice, stream));
};

template< typename T>
static inline
void cudaCpyDeviceToHost( T *d_ptr, T *h_ptr, size_t size)
{
	DEBUG_HANDLE_ERROR(cudaMemcpy( h_ptr, d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
};

// Copy from the device to (pageable) host memory through this thread's pinned staging
// buffer. A copy to pageable memory goes through the driver's own staging buffers,
// behind a lock held until the copy has completed, so threads copying at the same time
// queue for it (seen in stack samples: most threads in cudaMemcpyAsync were waiting
// for a driver mutex). Returns when the data is in h_ptr, as such a copy does.
// RELION_GPU_PINNED_STAGING=off copies directly.
static inline void stagedDeviceToHost(void *h_ptr, const void *d_ptr, size_t bytes, cudaStream_t stream)
{
	static const bool on = []() {
		const char *e = getenv("RELION_GPU_PINNED_STAGING");
		return !(e && (std::string(e) == "off" || std::string(e) == "0"));
	}();
	const size_t max_bytes = (size_t) 64 << 20;
	thread_local void *stage = NULL;
	thread_local size_t capacity = 0;
	if (!on || bytes == 0 || bytes > max_bytes)
	{
		DEBUG_HANDLE_ERROR(cudaMemcpyAsync(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost, stream));
		return;
	}
	if (bytes > capacity)
	{
		if (stage != NULL)
			DEBUG_HANDLE_ERROR(cudaFreeHost(stage));
		capacity = std::max(bytes, (size_t) 1 << 20);
		HANDLE_ERROR(cudaMallocHost(&stage, capacity));
	}
	DEBUG_HANDLE_ERROR(cudaMemcpyAsync(stage, d_ptr, bytes, cudaMemcpyDeviceToHost, stream));
	DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
	memcpy(h_ptr, stage, bytes);
}

template< typename T>
static inline
void cudaCpyDeviceToHost( T *d_ptr, T *h_ptr, size_t size, cudaStream_t &stream)
{
	stagedDeviceToHost(h_ptr, d_ptr, size * sizeof(T), stream);
};

template< typename T>
static inline
void cudaCpyDeviceToDevice( T *src, T *des, size_t size, cudaStream_t stream)
{
	DEBUG_HANDLE_ERROR(cudaMemcpyAsync( des, src, size * sizeof(T), cudaMemcpyDeviceToDevice, stream));
};

template< typename T>
static inline
void cudaMemInit( T *ptr, T value, size_t size)
{
	DEBUG_HANDLE_ERROR(cudaMemset( ptr, value, size * sizeof(T)));
};

template< typename T>
static inline
void cudaMemInit( T *ptr, T value, size_t size, cudaStream_t &stream)
{
	DEBUG_HANDLE_ERROR(cudaMemsetAsync( ptr, value, size * sizeof(T), stream));
};

#endif
