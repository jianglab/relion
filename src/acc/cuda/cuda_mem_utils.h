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
#include "src/acc/stage_timer.h"
#include "src/acc/pinned_pool.h"
#include "src/gpu_switch.h"
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

// The pinned staging buffers of all threads come from this pool (see pinned_pool.h);
// RELION_GPU_PINNED_POOL=0 makes every thread allocate and free its own, as before.
static inline pinnedpool::Pool &pinnedHostPool()
{
	static pinnedpool::Pool pool(
		[](size_t n) -> void * { void *p = NULL; HANDLE_ERROR(cudaMallocHost(&p, n)); return p; },
		[](void *p) { cudaFreeHost(p); },
		gpuswitch::on("RELION_GPU_PINNED_POOL"));
	return pool;
}

// Copy pageable host memory to the device through this thread's pinned, double-buffered
// staging. A copy straight from pageable memory makes the driver synchronise the stream
// and take a lock shared by all threads, so threads uploading at the same time (and
// behind their own queued kernels) wait for each other. Here the host only waits when
// it needs a staging buffer back. The source may be reused when this returns.
// RELION_GPU_PINNED_STAGING=off copies directly.
static inline void stagedHostToDevice(void *d_ptr, const void *h_ptr, size_t bytes, cudaStream_t stream)
{
	static const bool on = []() {
		const char *e = getenv("RELION_GPU_PINNED_STAGING");
		return !(e && (std::string(e) == "off" || std::string(e) == "0"));
	}();
	stagetimer::countUp(bytes); 
	const size_t min_bytes = (size_t) 64 << 10;
	const size_t chunk = (size_t) 8 << 20;
	if (!on || bytes < min_bytes)
	{
		DEBUG_HANDLE_ERROR(cudaMemcpyAsync(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice, stream));
		return;
	}
	struct Stage
	{
		void *buf[2] = {NULL, NULL};
		size_t cap[2] = {0, 0};
		cudaEvent_t ev[2] = {NULL, NULL};
		bool used[2] = {false, false};
		int device = -1;
		int next = 0;
		// when the thread ends the buffers go back to the pool, once the copies still reading them are done
		~Stage()
		{
			for (int i = 0; i < 2; i++)
			{
				if (ev[i] != NULL)
				{
					cudaEventSynchronize(ev[i]);
					cudaEventDestroy(ev[i]);
				}
				pinnedHostPool().release(buf[i], cap[i]);
			}
			cudaGetLastError();
		}
	};
	thread_local Stage st;
	int dev = 0;
	HANDLE_ERROR(cudaGetDevice(&dev));
	if (st.device != dev)
	{
		for (int i = 0; i < 2; i++)
		{
			if (st.ev[i] != NULL)
			{
				cudaEventSynchronize(st.ev[i]);
				cudaEventDestroy(st.ev[i]);
				st.ev[i] = NULL;
			}
			st.used[i] = false;
			if (st.buf[i] == NULL)
				st.buf[i] = pinnedHostPool().acquire(chunk, st.cap[i]);
			HANDLE_ERROR(cudaEventCreateWithFlags(&st.ev[i], cudaEventDisableTiming));
		}
		st.device = dev;
	}
	const char *src = (const char *) h_ptr;
	char *dst = (char *) d_ptr;
	for (size_t done = 0; done < bytes; )
	{
		const size_t n = std::min(chunk, bytes - done);
		const int b = st.next;
		st.next ^= 1;
		if (st.used[b])
			DEBUG_HANDLE_ERROR(cudaEventSynchronize(st.ev[b]));
		memcpy(st.buf[b], src + done, n);
		DEBUG_HANDLE_ERROR(cudaMemcpyAsync(dst + done, st.buf[b], n, cudaMemcpyHostToDevice, stream));
		DEBUG_HANDLE_ERROR(cudaEventRecord(st.ev[b], stream));
		st.used[b] = true;
		done += n;
	}
}

template< typename T>
static inline
void cudaCpyHostToDevice( T *h_ptr, T *d_ptr, size_t size, cudaStream_t &stream)
{
	stagedHostToDevice(d_ptr, h_ptr, size * sizeof(T), stream);
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
// for a driver mutex). Returns true when it has synchronised the stream (data is in h_ptr),
// false when the copy was only queued.
// RELION_GPU_PINNED_STAGING=off copies directly.
static inline bool stagedDeviceToHost(void *h_ptr, const void *d_ptr, size_t bytes, cudaStream_t stream)
{
	static const bool on = []() {
		const char *e = getenv("RELION_GPU_PINNED_STAGING");
		return !(e && (std::string(e) == "off" || std::string(e) == "0"));
	}();
	stagetimer::countDown(bytes); 
	const size_t max_bytes = (size_t) 64 << 20;
	struct Bounce
	{
		void *ptr = NULL;
		size_t capacity = 0;
		~Bounce() { pinnedHostPool().release(ptr, capacity); }
	};
	thread_local Bounce bounce;
	void *&stage = bounce.ptr;
	size_t &capacity = bounce.capacity;
	if (!on || bytes == 0 || bytes > max_bytes)
	{
		DEBUG_HANDLE_ERROR(cudaMemcpyAsync(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost, stream));
		return false;
	}
	if (bytes > capacity)
	{
		pinnedHostPool().release(stage, capacity);
		stage = pinnedHostPool().acquire(std::max(bytes, (size_t) 1 << 20), capacity);
	}
	DEBUG_HANDLE_ERROR(cudaMemcpyAsync(stage, d_ptr, bytes, cudaMemcpyDeviceToHost, stream));
	DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
	memcpy(h_ptr, stage, bytes);
	return true;
}

// Big one-off copies between a pageable host array and the device (legacy default stream), through
// two pinned buffers from the pool: the copy of one chunk overlaps the memcpy of the next, so the
// transfer runs at close to the speed of the bus instead of the driver's pageable path (which stages
// small pieces and waits for each). Both return when the data is in place. Under 1 MB, or with
// RELION_GPU_PINNED_BIG_COPY=0, they are a plain cudaMemcpy. Same bytes either way.
static inline bool pinnedBigCopyOn()
{
	static const bool on = gpuswitch::on("RELION_GPU_PINNED_BIG_COPY");
	return on;
}

static inline cudaError_t pinnedBigUpload(void *d_ptr, const void *h_ptr, size_t bytes)
{
	const size_t chunk = (size_t) 8 << 20;
	if (!pinnedBigCopyOn() || bytes < ((size_t) 1 << 20))
		return cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice);
	void *buf[2] = {NULL, NULL};
	size_t cap[2] = {0, 0};
	cudaEvent_t ev[2] = {NULL, NULL};
	cudaError_t e = cudaSuccess;
	for (int i = 0; i < 2 && e == cudaSuccess; i++)
	{
		buf[i] = pinnedHostPool().acquire(chunk, cap[i]);
		e = cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming);
	}
	const char *src = (const char *) h_ptr;
	char *dst = (char *) d_ptr;
	bool used[2] = {false, false};
	int b = 0;
	for (size_t done = 0; done < bytes && e == cudaSuccess; b ^= 1)
	{
		const size_t n = std::min(chunk, bytes - done);
		if (used[b])
			e = cudaEventSynchronize(ev[b]);
		if (e != cudaSuccess) break;
		memcpy(buf[b], src + done, n);
		e = cudaMemcpyAsync(dst + done, buf[b], n, cudaMemcpyHostToDevice, 0);
		if (e == cudaSuccess) e = cudaEventRecord(ev[b], 0);
		used[b] = true;
		done += n;
	}
	cudaError_t e2 = cudaStreamSynchronize(0);
	if (e == cudaSuccess) e = e2;
	for (int i = 0; i < 2; i++)
	{
		if (ev[i] != NULL) cudaEventDestroy(ev[i]);
		pinnedHostPool().release(buf[i], cap[i]);
	}
	return e;
}

static inline cudaError_t pinnedBigDownload(void *h_ptr, const void *d_ptr, size_t bytes)
{
	const size_t chunk = (size_t) 8 << 20;
	if (!pinnedBigCopyOn() || bytes < ((size_t) 1 << 20))
		return cudaMemcpy(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost);
	void *buf[2] = {NULL, NULL};
	size_t cap[2] = {0, 0};
	cudaEvent_t ev[2] = {NULL, NULL};
	cudaError_t e = cudaSuccess;
	for (int i = 0; i < 2 && e == cudaSuccess; i++)
	{
		buf[i] = pinnedHostPool().acquire(chunk, cap[i]);
		e = cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming);
	}
	const size_t nchunks = (bytes + chunk - 1) / chunk;
	auto issue = [&](size_t i)
	{
		const size_t off = i * chunk, n = std::min(chunk, bytes - off);
		cudaError_t r = cudaMemcpyAsync(buf[i & 1], (const char *) d_ptr + off, n, cudaMemcpyDeviceToHost, 0);
		if (r == cudaSuccess) r = cudaEventRecord(ev[i & 1], 0);
		return r;
	};
	if (e == cudaSuccess) e = issue(0);
	for (size_t i = 0; i < nchunks && e == cudaSuccess; i++)
	{
		// the other buffer was emptied into the host array at the end of the last round
		if (i + 1 < nchunks) e = issue(i + 1);
		if (e != cudaSuccess) break;
		e = cudaEventSynchronize(ev[i & 1]);
		if (e != cudaSuccess) break;
		const size_t off = i * chunk, n = std::min(chunk, bytes - off);
		memcpy((char *) h_ptr + off, buf[i & 1], n);
	}
	cudaError_t e2 = cudaStreamSynchronize(0);
	if (e == cudaSuccess) e = e2;
	for (int i = 0; i < 2; i++)
	{
		if (ev[i] != NULL) cudaEventDestroy(ev[i]);
		pinnedHostPool().release(buf[i], cap[i]);
	}
	return e;
}

template< typename T>
static inline
bool cudaCpyDeviceToHost( T *d_ptr, T *h_ptr, size_t size, cudaStream_t &stream)
{
	return stagedDeviceToHost(h_ptr, d_ptr, size * sizeof(T), stream);
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
