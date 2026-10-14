//#include <signal.h>
//#ifdef _CUDA_ENABLED
//#include <cuda_runtime.h>
//#include "src/acc/cuda/cuda_kernels/cuda_device_utils.cuh"
//#elif _HIP_ENABLED
//#include <hip/hip_runtime.h>
//#include "src/acc/hip/hip_kernels/hip_device_utils.h"
//#endif
//#include "src/acc/settings.h"
//#include "src/acc/acc_backprojector.h"
#include "src/acc/acc_projector.h"

#ifdef _CUDA_ENABLED
// Interleaved model layout (see AccBackprojector::interleaved): the default for 3D models
static bool bpInterleavedLayout()
{
	static const bool on = [] { const char *e = getenv("RELION_GPU_BP_LAYOUT"); return !(e && std::string(e) == "planar"); }();
	return on;
}

__global__ void cuda_kernel_bp_deinterleave(const XFLOAT *g_in, XFLOAT *g_out, size_t n)
{
	const size_t v = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	if (v < n)
	{
		g_out[v]         = g_in[3 * v];
		g_out[n + v]     = g_in[3 * v + 1];
		g_out[2 * n + v] = g_in[3 * v + 2];
	}
}

// Copies an interleaved model to three host arrays through a device staging buffer in
// chunks, which shrinks if memory is short
static void bpDeinterleaveToHost(const XFLOAT *d_model, size_t n, XFLOAT *r, XFLOAT *i, XFLOAT *w, cudaStream_t stream)
{
	size_t chunk = std::min(n, (size_t) 1 << 22);
	XFLOAT *d_tmp = NULL;
	while (cudaMalloc((void **) &d_tmp, 3 * chunk * sizeof(XFLOAT)) != cudaSuccess)
	{
		cudaGetLastError();
		chunk /= 2;
		if (chunk < 4096)
			CRITICAL(RAMERR);
	}
	for (size_t from = 0; from < n; from += chunk)
	{
		const size_t m = std::min(chunk, n - from);
		cuda_kernel_bp_deinterleave<<<(m + 255) / 256, 256, 0, stream>>>(d_model + 3 * from, d_tmp, m);
		DEBUG_HANDLE_ERROR(cudaMemcpyAsync(r + from, d_tmp,         m * sizeof(XFLOAT), cudaMemcpyDeviceToHost, stream));
		DEBUG_HANDLE_ERROR(cudaMemcpyAsync(i + from, d_tmp + m,     m * sizeof(XFLOAT), cudaMemcpyDeviceToHost, stream));
		DEBUG_HANDLE_ERROR(cudaMemcpyAsync(w + from, d_tmp + 2 * m, m * sizeof(XFLOAT), cudaMemcpyDeviceToHost, stream));
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
	}
	DEBUG_HANDLE_ERROR(cudaFree(d_tmp));
}
#endif

size_t AccBackprojector::setMdlDim(
#ifdef _SYCL_ENABLED
			deviceStream_t dev,
#endif
			int xdim, int ydim, int zdim,
			int inity, int initz,
			int max_r, XFLOAT paddingFactor)
{
	if (xdim != mdlX ||
		ydim != mdlY ||
		zdim != mdlZ ||
		inity != mdlInitY ||
		initz != mdlInitZ ||
		max_r != maxR ||
		paddingFactor != padding_factor)
	{
		clear();

		mdlX = xdim;
		mdlY = ydim;
		mdlZ = zdim;
		if (mdlZ < 1) mdlZ = 1;
		mdlXYZ = (size_t)xdim*(size_t)ydim*(size_t)zdim;
		mdlInitY = inity;
		mdlInitZ = initz;
		maxR = max_r;
		maxR2 = max_r*max_r;
		padding_factor = paddingFactor;

		//Allocate space for model
#ifdef _CUDA_ENABLED
		interleaved = mdlZ != 1 && bpInterleavedLayout();
		if (interleaved)
		{
			HANDLE_ERROR(cudaMalloc( (void**) &d_mdlReal, 3 * mdlXYZ * sizeof(XFLOAT)));
			d_mdlImag = d_mdlReal + 1;
			d_mdlWeight = d_mdlReal + 2;
		}
		else
		{
			HANDLE_ERROR(cudaMalloc( (void**) &d_mdlReal,   mdlXYZ * sizeof(XFLOAT)));
			HANDLE_ERROR(cudaMalloc( (void**) &d_mdlImag,   mdlXYZ * sizeof(XFLOAT)));
			HANDLE_ERROR(cudaMalloc( (void**) &d_mdlWeight, mdlXYZ * sizeof(XFLOAT)));
		}
#elif _HIP_ENABLED
		HANDLE_ERROR(hipMalloc( (void**) &d_mdlReal,   mdlXYZ * sizeof(XFLOAT)));
		HANDLE_ERROR(hipMalloc( (void**) &d_mdlImag,   mdlXYZ * sizeof(XFLOAT)));
		HANDLE_ERROR(hipMalloc( (void**) &d_mdlWeight, mdlXYZ * sizeof(XFLOAT)));
#elif _SYCL_ENABLED
		stream = dev;
		d_mdlReal   = (XFLOAT*)(stream->syclMalloc(mdlXYZ * sizeof(XFLOAT), syclMallocType::device, "d_mdlReal"));
		d_mdlImag   = (XFLOAT*)(stream->syclMalloc(mdlXYZ * sizeof(XFLOAT), syclMallocType::device, "d_mdlImag"));
		d_mdlWeight = (XFLOAT*)(stream->syclMalloc(mdlXYZ * sizeof(XFLOAT), syclMallocType::device, "d_mdlWeight"));
#else
		if (posix_memalign((void **)&d_mdlReal,   MEM_ALIGN, mdlXYZ * sizeof(XFLOAT))) CRITICAL(RAMERR);
		if (posix_memalign((void **)&d_mdlImag,   MEM_ALIGN, mdlXYZ * sizeof(XFLOAT))) CRITICAL(RAMERR);
		if (posix_memalign((void **)&d_mdlWeight, MEM_ALIGN, mdlXYZ * sizeof(XFLOAT))) CRITICAL(RAMERR);

        mutexes = new tbb::spin_mutex[mdlZ*mdlY];
#endif

		allocaton_size = mdlXYZ * sizeof(XFLOAT) * 3;
	}

	return allocaton_size;
}

void AccBackprojector::initMdl()
{
#if defined DEBUG_CUDA || defined DEBUG_HIP
	if (mdlXYZ == 0)
	{
        printf("Model dimensions must be set with setMdlDim before call to initMdl.");
        CRITICAL(ERR_MDLDIM);
	}
	if (voxelCount != 0)
	{
        printf("DEBUG_ERROR: Duplicated call to model setup");
        CRITICAL(ERR_MDLSET);
	}
#endif

	//Initiate model with zeros
#ifdef _CUDA_ENABLED
	if (interleaved)
	{
		DEBUG_HANDLE_ERROR(cudaMemset( d_mdlReal, 0, 3 * mdlXYZ * sizeof(XFLOAT)));
	}
	else
	{
		DEBUG_HANDLE_ERROR(cudaMemset( d_mdlReal,   0, mdlXYZ * sizeof(XFLOAT)));
		DEBUG_HANDLE_ERROR(cudaMemset( d_mdlImag,   0, mdlXYZ * sizeof(XFLOAT)));
		DEBUG_HANDLE_ERROR(cudaMemset( d_mdlWeight, 0, mdlXYZ * sizeof(XFLOAT)));
	}
#elif _HIP_ENABLED
	DEBUG_HANDLE_ERROR(hipMemset( d_mdlReal,   0, mdlXYZ * sizeof(XFLOAT)));
	DEBUG_HANDLE_ERROR(hipMemset( d_mdlImag,   0, mdlXYZ * sizeof(XFLOAT)));
	DEBUG_HANDLE_ERROR(hipMemset( d_mdlWeight, 0, mdlXYZ * sizeof(XFLOAT)));
#elif _SYCL_ENABLED
	stream->syclMemset(d_mdlReal, 0, mdlXYZ * sizeof(XFLOAT));
	stream->syclMemset(d_mdlImag, 0, mdlXYZ * sizeof(XFLOAT));
	stream->syclMemset(d_mdlWeight, 0, mdlXYZ * sizeof(XFLOAT));
	stream->waitAll();
#else
	memset(d_mdlReal,     0, mdlXYZ * sizeof(XFLOAT));
	memset(d_mdlImag,     0, mdlXYZ * sizeof(XFLOAT));
	memset(d_mdlWeight,   0, mdlXYZ * sizeof(XFLOAT));
#endif

    voxelCount = mdlXYZ;
}


void AccBackprojector::getMdlData(XFLOAT *r, XFLOAT *i, XFLOAT * w)
{
#ifdef _CUDA_ENABLED
	DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream)); //Make sure to wait for remaining kernel executions

	if (interleaved)
	{
		bpDeinterleaveToHost(d_mdlReal, mdlXYZ, r, i, w, stream);
		return;
	}
	DEBUG_HANDLE_ERROR(cudaMemcpyAsync( r, d_mdlReal,   mdlXYZ * sizeof(XFLOAT), cudaMemcpyDeviceToHost, stream));
	DEBUG_HANDLE_ERROR(cudaMemcpyAsync( i, d_mdlImag,   mdlXYZ * sizeof(XFLOAT), cudaMemcpyDeviceToHost, stream));
	DEBUG_HANDLE_ERROR(cudaMemcpyAsync( w, d_mdlWeight, mdlXYZ * sizeof(XFLOAT), cudaMemcpyDeviceToHost, stream));

	DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream)); //Wait for copy
#elif _HIP_ENABLED
	DEBUG_HANDLE_ERROR(hipStreamSynchronize(stream)); //Make sure to wait for remaining kernel executions

	DEBUG_HANDLE_ERROR(hipMemcpyAsync( r, d_mdlReal,   mdlXYZ * sizeof(XFLOAT), hipMemcpyDeviceToHost, stream));
	DEBUG_HANDLE_ERROR(hipMemcpyAsync( i, d_mdlImag,   mdlXYZ * sizeof(XFLOAT), hipMemcpyDeviceToHost, stream));
	DEBUG_HANDLE_ERROR(hipMemcpyAsync( w, d_mdlWeight, mdlXYZ * sizeof(XFLOAT), hipMemcpyDeviceToHost, stream));

	DEBUG_HANDLE_ERROR(hipStreamSynchronize(stream)); //Wait for copy
#elif _SYCL_ENABLED
	stream->waitAll();
	stream->syclMemcpy(r, d_mdlReal, mdlXYZ * sizeof(XFLOAT));
	stream->syclMemcpy(i, d_mdlImag, mdlXYZ * sizeof(XFLOAT));
	stream->syclMemcpy(w, d_mdlWeight, mdlXYZ * sizeof(XFLOAT));
	stream->waitAll();
#else
	memcpy(r, d_mdlReal,   mdlXYZ * sizeof(XFLOAT));
	memcpy(i, d_mdlImag,   mdlXYZ * sizeof(XFLOAT));
	memcpy(w, d_mdlWeight, mdlXYZ * sizeof(XFLOAT));
#endif
}

void AccBackprojector::getMdlDataPtrs(XFLOAT *& r, XFLOAT *& i, XFLOAT *& w)
{
#ifdef ALTCPU
	r = d_mdlReal;
	i = d_mdlImag;
	w = d_mdlWeight;
#endif
}

void AccBackprojector::clear()
{
	mdlX = 0;
	mdlY = 0;
	mdlZ = 0;
	mdlXYZ = 0;
	mdlInitY = 0;
	mdlInitZ = 0;
	maxR = 0;
	maxR2 = 0;
	padding_factor = 0;
	allocaton_size = 0;

	if (d_mdlReal != NULL)
	{
#ifdef _CUDA_ENABLED
		DEBUG_HANDLE_ERROR(cudaFree(d_mdlReal));
		if (!interleaved)
		{
			DEBUG_HANDLE_ERROR(cudaFree(d_mdlImag));
			DEBUG_HANDLE_ERROR(cudaFree(d_mdlWeight));
		}
#elif _HIP_ENABLED
		DEBUG_HANDLE_ERROR(hipFree(d_mdlReal));
		DEBUG_HANDLE_ERROR(hipFree(d_mdlImag));
		DEBUG_HANDLE_ERROR(hipFree(d_mdlWeight));
#elif _SYCL_ENABLED
		stream->waitAll();
		stream->syclFree(d_mdlReal);
		stream->syclFree(d_mdlImag);
		stream->syclFree(d_mdlWeight);
#else
		free(d_mdlReal);
		free(d_mdlImag);
		free(d_mdlWeight);
		delete [] mutexes;
#endif

		d_mdlReal = d_mdlImag = d_mdlWeight = NULL;
		interleaved = false;
	}
}

AccBackprojector::~AccBackprojector()
{
	clear();
}
