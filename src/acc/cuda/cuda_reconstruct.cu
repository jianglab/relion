/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

// BackProjector::reconstruct() with skip_gridding on the GPU: see src/gpu_reconstruct.h. Every kernel
// mirrors one loop of the CPU code in src/backprojector.cpp, with the same index arithmetic.

#include "src/gpu_reconstruct.h"
#include "src/args.h"
#include "src/error.h"

#include <cuda_runtime.h>
#include "src/acc/cuda/gpu_mem_trace.h"
#include "src/acc/cuda/cuda_mem_utils.h"
#include <cufft.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <type_traits>

void relionReleaseRetainedGpuMemory(); // cuda_ml_optimiser.cu
bool relionHasRetainedGpuMemory();     // cuda_ml_optimiser.cu

namespace gpurec
{

namespace
{

// The reconstruction runs in R (float or double) with complex type C; the host arrays are RFLOAT.
template <class R> struct CplxOf;
template <> struct CplxOf<float> { typedef float2 type; };
template <> struct CplxOf<double> { typedef double2 type; };

cufftResult planC2R(cufftHandle plan, int n, float *, size_t *work) { return cufftMakePlan3d(plan, n, n, n, CUFFT_C2R, work); }
cufftResult planC2R(cufftHandle plan, int n, double *, size_t *work) { return cufftMakePlan3d(plan, n, n, n, CUFFT_Z2D, work); }
cufftResult execC2R(cufftHandle plan, float2 *in, float *out) { return cufftExecC2R(plan, (cufftComplex*)in, (cufftReal*)out); }
cufftResult execC2R(cufftHandle plan, double2 *in, double *out) { return cufftExecZ2D(plan, (cufftDoubleComplex*)in, (cufftDoubleReal*)out); }
cufftResult planR2C(cufftHandle plan, int n, float *, size_t *work) { return cufftMakePlan3d(plan, n, n, n, CUFFT_R2C, work); }
cufftResult planR2C(cufftHandle plan, int n, double *, size_t *work) { return cufftMakePlan3d(plan, n, n, n, CUFFT_D2Z, work); }
cufftResult execR2C(cufftHandle plan, float *in, float2 *out) { return cufftExecR2C(plan, (cufftReal*)in, (cufftComplex*)out); }
cufftResult execR2C(cufftHandle plan, double *in, double2 *out) { return cufftExecD2Z(plan, (cufftDoubleReal*)in, (cufftDoubleComplex*)out); }

// The macros of src/macros.h, for the device
__device__ __forceinline__ int dROUND_R(float x) { return (x > 0.0f) ? (int)(x + 0.5f) : (int)(x - 0.5f); }
__device__ __forceinline__ int dROUND_R(double x) { return (x > 0.0) ? (int)(x + 0.5) : (int)(x - 0.5); }
__device__ __forceinline__ int dFLOOR_R(float x) { return (x == (int)x) ? (int)x : ((x > 0.0f) ? (int)x : (int)(x - 1.0f)); }
__device__ __forceinline__ int dFLOOR_R(double x) { return (x == (int)x) ? (int)x : ((x > 0.0) ? (int)x : (int)(x - 1.0)); }
__device__ __forceinline__ float dSQRT_R(float x) { return sqrtf(x); }
__device__ __forceinline__ double dSQRT_R(double x) { return sqrt(x); }
__device__ __forceinline__ float dCOS_R(float x) { return cosf(x); }
__device__ __forceinline__ double dCOS_R(double x) { return cos(x); }
__device__ __forceinline__ float dSIN_R(float x) { return sinf(x); }
__device__ __forceinline__ double dSIN_R(double x) { return sin(x); }

// FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(V): element (k, i, j) of a zdim x ydim x xdim array has
// kp = k < xdim ? k : k - zdim, ip = i < xdim ? i : i - ydim, jp = j
struct Dims { long int z, y, x; };
__device__ __forceinline__ void decompose(size_t n, const Dims &d, long int &k, long int &i, long int &j)
{
	j = n % d.x;
	const size_t r = n / d.x;
	i = r % d.y;
	k = r / d.y;
}
__device__ __forceinline__ long int wrapK(long int k, const Dims &d) { return (k < d.x) ? k : k - d.z; }
__device__ __forceinline__ long int wrapI(long int i, const Dims &d) { return (i < d.x) ? i : i - d.y; }

// atomicAdd(double *, double) is native only from sm_60; below that, the atomicCAS loop of helper.cuh
__device__ __forceinline__ double atomicAddDouble(double *address, double val)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
	return atomicAdd(address, val);
#else
	unsigned long long int *address_as_ull = (unsigned long long int *)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
		                __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
#endif
}

#define GRID_STRIDE(n, total) for (size_t n = blockIdx.x * (size_t)blockDim.x + threadIdx.x; n < (total); n += (size_t)blockDim.x * gridDim.x)

// Projector::decenter(Min, Mout, max_r2): Mout has the same shape as Min
template <class T>
__global__ void kDecenter(const T *in, T *out, Dims d, long int sz, long int sy, long int sx, int max_r2)
{
	const size_t total = (size_t)d.z * d.y * d.x;
	GRID_STRIDE(n, total)
	{
		long int k, i, j;
		decompose(n, d, k, i, j);
		const long int kp = wrapK(k, d), ip = wrapI(i, d), jp = j;
		T v{};
		if (kp * kp + ip * ip + jp * jp <= max_r2)
			v = in[((kp - sz) * d.y + (ip - sy)) * d.x + (jp - sx)];
		out[n] = v;
	}
}

// do_map: Fweight += invtau2 for r2 < max_r2 and ires >= minres_map
template <class R>
__global__ void kMap(R *Fw, Dims d, int max_r2, R pad, int minres_map,
                     const R *shell_invtau2, const char *shell_pos, int nshell)
{
	const size_t total = (size_t)d.z * d.y * d.x;
	GRID_STRIDE(n, total)
	{
		long int k, i, j;
		decompose(n, d, k, i, j);
		const long int kp = wrapK(k, d), ip = wrapI(i, d), jp = j;
		const int r2 = kp * kp + ip * ip + jp * jp;
		if (r2 < max_r2)
		{
			const int ires = dROUND_R(dSQRT_R((R)r2) / pad);
			if (ires >= nshell) continue;   // (the CPU would read past tau2)
			R invw = Fw[n];
			R invtau2;
			if (shell_pos[ires])
				invtau2 = shell_invtau2[ires];
			else if (invw > (R)1e-20)
				invtau2 = (R)1.0 / ((R)0.001 * invw);
			else
				invtau2 = (R)0.0;
			if (ires >= minres_map)
			{
				invw += invtau2;
				Fw[n] = invw;
			}
		}
	}
}

// Sums of the weight per shell (ires = FLOOR(sqrt(r2) / pad), r2 < round_max_r2), and their counts
template <class R>
__global__ void kShellSums(const R *Fw, Dims d, int round_max_r2, R pad, int r_max,
                           double *sums, unsigned long long *counts, int *bad)
{
	extern __shared__ char shbuf[];
	const bool use_smem = (r_max <= 1024);
	double *s_sums = (double *)shbuf;
	unsigned long long *s_cnt = (unsigned long long *)(s_sums + r_max);
	if (use_smem)
	{
		for (int i = threadIdx.x; i < r_max; i += blockDim.x)
		{
			s_sums[i] = 0.;
			s_cnt[i] = 0ull;
		}
		__syncthreads();
	}
	const size_t total = (size_t)d.z * d.y * d.x;
	GRID_STRIDE(n, total)
	{
		long int k, i, j;
		decompose(n, d, k, i, j);
		const long int kp = wrapK(k, d), ip = wrapI(i, d), jp = j;
		const int r2 = kp * kp + ip * ip + jp * jp;
		if (r2 < round_max_r2)
		{
			const int ires = dFLOOR_R(dSQRT_R((R)r2) / pad);
			if (ires >= r_max) { *bad = 1; continue; }
			if (use_smem)
			{
				atomicAddDouble(&s_sums[ires], (double)Fw[n]);
				atomicAdd(&s_cnt[ires], 1ull);
			}
			else
			{
				atomicAddDouble(&sums[ires], (double)Fw[n]);
				atomicAdd(&counts[ires], 1ull);
			}
		}
	}
	if (use_smem)
	{
		__syncthreads();
		for (int i = threadIdx.x; i < r_max; i += blockDim.x)
		{
			if (s_cnt[i] > 0ull)
			{
				atomicAddDouble(&sums[i], s_sums[i]);
				atomicAdd(&counts[i], s_cnt[i]);
			}
		}
	}
}

// Fconv /= max(Fweight, radavg_weight[ires]) (skipped, with a warning, where that is 0)
template <class R, class C>
__global__ void kDivide(C *Fc, const R *Fw, Dims d, R pad, int r_max, const R *radavg, int *zero_seen)
{
	const size_t total = (size_t)d.z * d.y * d.x;
	GRID_STRIDE(n, total)
	{
		long int k, i, j;
		decompose(n, d, k, i, j);
		const long int kp = wrapK(k, d), ip = wrapI(i, d), jp = j;
		const int r2 = kp * kp + ip * ip + jp * jp;
		const int ires = dFLOOR_R(dSQRT_R((R)r2) / pad);
		const R a = Fw[n], b = radavg[(ires < r_max) ? ires : (r_max - 1)];
		const R weight = (a > b) ? a : b;   // XMIPP_MAX
		C c = Fc[n];
		if (weight == (R)0.0)
		{
			if (c.x != (R)0.0 || c.y != (R)0.0) *zero_seen = 1;
		}
		else
		{
			c.x /= weight;
			c.y /= weight;
			Fc[n] = c;
		}
	}
}

// windowFourierTransform(Fin, N) followed by CenterFFTbySign: out is N x N x (N/2+1)
template <class C>
__global__ void kWindowFourierCenter(const C *in, Dims din, C *out, Dims dout)
{
	const size_t total = (size_t)dout.z * dout.y * dout.x;
	const bool grow = dout.x > din.x;
	const long int max_r2 = (din.x - 1) * (din.x - 1);
	GRID_STRIDE(n, total)
	{
		long int k, i, j;
		decompose(n, dout, k, i, j);
		const long int kp = wrapK(k, dout), ip = wrapI(i, dout), jp = j;
		C v{};
		if (!grow || kp * kp + ip * ip + jp * jp <= max_r2)
		{
			const long int ki = (kp < 0) ? kp + din.z : kp, ii = (ip < 0) ? ip + din.y : ip;
			if (ki >= 0 && ki < din.z && ii >= 0 && ii < din.y && jp < din.x)
				v = in[((size_t)ki * din.y + ii) * din.x + jp];
		}
		if (((k ^ i ^ j) & 1) != 0)
		{
			v.x *= -1;
			v.y *= -1;
		}
		out[n] = v;
	}
}

// Window the real-space result (in place C2R: rows padded to 2 * (N/2+1)) to ori^3 and divide by normfft
template <class R>
__global__ void kWindowReal(const R *in, long int N, R *out, long int ori, R normfft)
{
	const long int row = 2 * (N / 2 + 1), off = N / 2 - ori / 2;
	const size_t total = (size_t)ori * ori * ori;
	GRID_STRIDE(n, total)
	{
		const long int x = n % ori, y = (n / ori) % ori, z = n / ((size_t)ori * ori);
		out[n] = in[((size_t)(z + off) * N + (y + off)) * row + (x + off)] / normfft;
	}
}

// Convert a piece of the host's RFLOAT array (copied to the GPU) into R
template <class R>
__global__ void kConvert(const RFLOAT *in, R *out, size_t n_el)
{
	GRID_STRIDE(n, n_el)
		out[n] = (R)in[n];
}

// Device memory freed on every return
struct DevBuf
{
	void *p = nullptr;
	cudaError_t alloc(size_t bytes) { return cudaMalloc(&p, bytes); }
	void release() { if (p) cudaFree(p); p = nullptr; }
	~DevBuf() { release(); }
	template <class T> T *as() const { return (T*)p; }
};

struct CachedDevBuf
{
	void *p = nullptr;
	size_t cap = 0;
	cudaError_t ensure(size_t bytes)
	{
		if (bytes <= cap) return cudaSuccess;
		release();
		cudaError_t e = cudaMalloc(&p, bytes);
		if (e == cudaSuccess) cap = bytes;
		else { p = nullptr; cap = 0; }
		return e;
	}
	void release()
	{
		if (p) cudaFree(p);
		p = nullptr;
		cap = 0;
	}
	~CachedDevBuf() { release(); }
	template <class T> T *as() const { return (T*)p; }
};

struct BufSlot
{
	DevBuf local;
	CachedDevBuf *cached = nullptr;
	void *p = nullptr;
	explicit BufSlot(CachedDevBuf *c = nullptr) : cached(c) {}
	cudaError_t alloc(size_t bytes)
	{
		if (cached)
		{
			cudaError_t e = cached->ensure(bytes);
			p = (e == cudaSuccess) ? cached->p : nullptr;
			return e;
		}
		cudaError_t e = local.alloc(bytes);
		p = (e == cudaSuccess) ? local.p : nullptr;
		return e;
	}
	void release()
	{
		if (!cached) { local.release(); p = nullptr; }
		else { p = nullptr; }
	}
	template <class T> T *as() const { return (T*)p; }
};

struct ReconstructCacheState
{
	int devid = -1;
	cufftHandle plan = 0;
	int plan_N = 0;
	bool plan_double = false;
	size_t plan_work = 0;
	CachedDevBuf slotFw, slotFc, slotWork, slotStage;
	CachedDevBuf dRad, dSums, dCounts, dFlags, dShellInv, dShellPos, dMaskSums;

	void destroyPlan()
	{
		if (plan) { cufftDestroy(plan); plan = 0; }
		plan_N = 0;
		plan_work = 0;
	}

	void releaseAll()
	{
		destroyPlan();
		slotFw.release();
		slotFc.release();
		slotWork.release();
		slotStage.release();
		dRad.release();
		dSums.release();
		dCounts.release();
		dFlags.release();
		dShellInv.release();
		dShellPos.release();
		dMaskSums.release();
		devid = -1;
	}

	~ReconstructCacheState() { releaseAll(); }
};

struct DownsampleCacheState
{
	int devid = -1;
	int orig_size = 0;
	int new_size = 0;
	bool plan_double = false;
	cufftHandle plan_fwd = 0;
	cufftHandle plan_inv = 0;
	CachedDevBuf dIn, dFtLarge, dFtSmall, dOut, dStage;

	void releaseAll()
	{
		if (plan_fwd) { cufftDestroy(plan_fwd); plan_fwd = 0; }
		if (plan_inv) { cufftDestroy(plan_inv); plan_inv = 0; }
		dIn.release();
		dFtLarge.release();
		dFtSmall.release();
		dOut.release();
		dStage.release();
		orig_size = new_size = 0;
		plan_double = false;
		devid = -1;
	}

	~DownsampleCacheState() { releaseAll(); }
};

thread_local ReconstructCacheState *tl_rec_cache = nullptr;
thread_local DownsampleCacheState *tl_ds_cache = nullptr;

// Pass 1 of softMaskOutsideMap(vol, -1, 3.0, NULL): sum mask weight and background density in FP64
template <class R>
__global__ void kSoftMaskSum(const R *vol, long int ori, double *out_sums)
{
	const long int s0 = -(long int)((float)ori / 2.0f);
	const R radius = (R)ori / (R)2.0;
	const R cosine_width = (R)3.0;
	const R radius_p = radius + cosine_width;
	const R radius2 = radius * radius;
	const R radius_p2 = radius_p * radius_p;
	const size_t total = (size_t)ori * ori * ori;
	double lw = 0.0, lwb = 0.0;
	GRID_STRIDE(n, total)
	{
		const long int kx = n % ori;
		const long int ky = (n / ori) % ori;
		const long int kz = n / ((size_t)ori * ori);
		const long int j = s0 + kx, i = s0 + ky, k = s0 + kz;
		const R r2 = (R)(k * k + i * i + j * j);
		if (r2 >= radius2)
		{
			if (r2 > radius_p2)
			{
				lw += 1.0;
				lwb += (double)vol[n];
			}
			else
			{
				const R r = dSQRT_R(r2);
				const R rc = (R)0.5 + (R)0.5 * dCOS_R((R)M_PI * (radius_p - r) / cosine_width);
				lw += (double)rc;
				lwb += (double)(rc * vol[n]);
			}
		}
	}
	__shared__ double sw[256], swb[256];
	const int tid = threadIdx.x;
	sw[tid] = lw;
	swb[tid] = lwb;
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sw[tid] += sw[tid + s];
			swb[tid] += swb[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		if (sw[0] != 0.0) atomicAddDouble(&out_sums[0], sw[0]);
		if (swb[0] != 0.0) atomicAddDouble(&out_sums[1], swb[0]);
	}
}

// Pass 2 of softMaskOutsideMap + griddingCorrect on dOut (R, ori^3)
template <class R>
__global__ void kSoftMaskAndGridding(R *vol, long int ori, bool do_soft_mask, const double *mask_sums,
                                     int gridding_mode, R rrval)
{
	const long int s0 = -(long int)((float)ori / 2.0f);
	const R radius = (R)ori / (R)2.0;
	const R cosine_width = (R)3.0;
	const R radius_p = radius + cosine_width;
	const R radius2 = radius * radius;
	const R radius_p2 = radius_p * radius_p;
	const R sum_bg = (R)((do_soft_mask && mask_sums[0] != 0.0) ? (mask_sums[1] / mask_sums[0]) : 0.0);
	const size_t total = (size_t)ori * ori * ori;
	GRID_STRIDE(n, total)
	{
		const long int kx = n % ori;
		const long int ky = (n / ori) % ori;
		const long int kz = n / ((size_t)ori * ori);
		const long int j = s0 + kx, i = s0 + ky, k = s0 + kz;
		const R r2 = (R)(k * k + i * i + j * j);
		R val = vol[n];
		if (do_soft_mask && r2 >= radius2)
		{
			if (r2 > radius_p2)
				val = sum_bg;
			else
			{
				const R r = dSQRT_R(r2);
				const R rc = (R)0.5 + (R)0.5 * dCOS_R((R)M_PI * (radius_p - r) / cosine_width);
				val = ((R)1.0 - rc) * val + rc * sum_bg;
			}
		}
		if (gridding_mode != 0 && r2 > (R)0.0)
		{
			const R r = dSQRT_R(r2);
			const R rval = r / rrval;
			const R arg = (R)M_PI * rval;
			const R sinc = dSIN_R(arg) / arg;
			val /= (gridding_mode == 1) ? sinc : (sinc * sinc);
		}
		vol[n] = val;
	}
}

// Shrink a half-Hermitian 3D Fourier volume from orig_size to new_size with exact windowFourierTransform
// wrapping, multiplying by inv_norm = 1 / orig_size^3 to match FourierTransformer forward normalisation.
template <class R, class C>
__global__ void kWindowFourierDownsample(const C *in, int orig_size, C *out, int new_size, R inv_norm)
{
	const long int oh = orig_size / 2 + 1;
	const long int nh = new_size / 2 + 1;
	const size_t total = (size_t)new_size * new_size * nh;
	GRID_STRIDE(n, total)
	{
		const long int jp = n % nh;
		const size_t r = n / nh;
		const long int i = r % new_size;
		const long int k = r / new_size;
		const long int kp = (k < nh) ? k : k - new_size;
		const long int ip = (i < nh) ? i : i - new_size;
		const long int ki = (kp < 0) ? kp + orig_size : kp;
		const long int ii = (ip < 0) ? ip + orig_size : ip;
		C v = in[((size_t)ki * orig_size + ii) * oh + jp];
		v.x *= inv_norm;
		v.y *= inv_norm;
		out[n] = v;
	}
}

int gridFor(size_t total)
{
	const size_t b = (total + 255) / 256;
	return (int)((b > 65535 * 8) ? 65535 * 8 : (b < 1 ? 1 : b));
}

}

bool deviceAvailable()
{
	int n = 0;
	if (cudaGetDeviceCount(&n) != cudaSuccess) { cudaGetLastError(); return false; }
	return n > 0;
}

namespace
{

// Copy n_el RFLOATs from the host into dst (type R) through pooled pinned staging buffers
template <class R>
cudaError_t uploadAs(const RFLOAT *host, R *dst, size_t n_el, RFLOAT *stage, size_t stage_el)
{
	if (std::is_same<R, RFLOAT>::value)
		return pinnedBigUpload(dst, host, n_el * sizeof(RFLOAT));

	auto syncFallback = [&]() -> cudaError_t
	{
		for (size_t done = 0; done < n_el; done += stage_el)
		{
			const size_t n = std::min(stage_el, n_el - done);
			cudaError_t e = cudaMemcpy(stage, host + done, n * sizeof(RFLOAT), cudaMemcpyHostToDevice);
			if (e != cudaSuccess) return e;
			kConvert<R><<<gridFor(n), 256>>>(stage, dst + done, n);
		}
		return cudaGetLastError();
	};

	if (!pinnedBigCopyOn() || n_el * sizeof(RFLOAT) < ((size_t)1 << 20) || stage_el < 2)
		return syncFallback();

	const size_t chunk_el = std::max<size_t>(1, std::min<size_t>(stage_el / 2, (size_t)1 << 20));
	RFLOAT *h_pin[2] = {nullptr, nullptr};
	size_t cap[2] = {0, 0};
	for (int i = 0; i < 2; i++)
		h_pin[i] = (RFLOAT *)pinnedHostPool().acquire(chunk_el * sizeof(RFLOAT), cap[i]);
	if (!h_pin[0] || !h_pin[1])
	{
		for (int i = 0; i < 2; i++)
			pinnedHostPool().release(h_pin[i], cap[i]);
		return syncFallback();
	}

	cudaEvent_t ev[2] = {nullptr, nullptr};
	cudaError_t err = cudaEventCreateWithFlags(&ev[0], cudaEventDisableTiming);
	if (err == cudaSuccess) err = cudaEventCreateWithFlags(&ev[1], cudaEventDisableTiming);
	if (err != cudaSuccess)
	{
		cudaGetLastError();
		if (ev[0]) cudaEventDestroy(ev[0]);
		if (ev[1]) cudaEventDestroy(ev[1]);
		for (int i = 0; i < 2; i++)
			pinnedHostPool().release(h_pin[i], cap[i]);
		return syncFallback();
	}

	size_t idx = 0;
	for (size_t done = 0; done < n_el; done += chunk_el, idx++)
	{
		const int s = (int)(idx & 1);
		if (idx >= 2)
		{
			err = cudaEventSynchronize(ev[s]);
			if (err != cudaSuccess) break;
		}
		const size_t n = std::min(chunk_el, n_el - done);
		RFLOAT *hp = h_pin[s];
		RFLOAT *d_st = stage + (size_t)s * chunk_el;
		memcpy(hp, host + done, n * sizeof(RFLOAT));
		err = cudaMemcpyAsync(d_st, hp, n * sizeof(RFLOAT), cudaMemcpyHostToDevice, 0);
		if (err == cudaSuccess)
			kConvert<R><<<gridFor(n), 256, 0, 0>>>(d_st, dst + done, n);
		if (err != cudaSuccess) break;
		err = cudaEventRecord(ev[s], 0);
		if (err != cudaSuccess) break;
	}
	cudaError_t sync_err = cudaStreamSynchronize(0);
	if (err == cudaSuccess) err = sync_err;
	cudaEventDestroy(ev[0]);
	cudaEventDestroy(ev[1]);
	for (int i = 0; i < 2; i++)
		pinnedHostPool().release(h_pin[i], cap[i]);
	if (err == cudaSuccess) err = cudaGetLastError();
	return err;
}

// Copy n_el elements of type R from device into host (RFLOAT) through pooled pinned staging buffers
template <class R>
cudaError_t downloadAsRfloat(RFLOAT *host, const R *dev, size_t n_el)
{
	if (std::is_same<R, RFLOAT>::value)
		return pinnedBigDownload(host, dev, n_el * sizeof(RFLOAT));

	if (!pinnedBigCopyOn() || n_el * sizeof(R) < ((size_t)1 << 20))
	{
		std::vector<R> tmp(n_el);
		cudaError_t e = cudaMemcpy(tmp.data(), dev, n_el * sizeof(R), cudaMemcpyDeviceToHost);
		if (e != cudaSuccess) return e;
		for (size_t i = 0; i < n_el; i++)
			host[i] = (RFLOAT)tmp[i];
		return cudaSuccess;
	}

	const size_t chunk_el = std::min<size_t>(n_el, (size_t)2 << 20);
	R *h_pin[2] = {nullptr, nullptr};
	size_t cap[2] = {0, 0};
	cudaEvent_t ev[2] = {nullptr, nullptr};
	cudaError_t e = cudaSuccess;
	for (int i = 0; i < 2 && e == cudaSuccess; i++)
	{
		h_pin[i] = (R *)pinnedHostPool().acquire(chunk_el * sizeof(R), cap[i]);
		if (!h_pin[i]) { e = cudaErrorMemoryAllocation; break; }
		e = cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming);
	}
	if (e == cudaSuccess)
	{
		const size_t nchunks = (n_el + chunk_el - 1) / chunk_el;
		auto issue = [&](size_t ci) -> cudaError_t
		{
			const size_t off = ci * chunk_el, n = std::min(chunk_el, n_el - off);
			cudaError_t r = cudaMemcpyAsync(h_pin[ci & 1], dev + off, n * sizeof(R), cudaMemcpyDeviceToHost, 0);
			if (r == cudaSuccess) r = cudaEventRecord(ev[ci & 1], 0);
			return r;
		};
		e = issue(0);
		for (size_t ci = 0; ci < nchunks && e == cudaSuccess; ci++)
		{
			if (ci + 1 < nchunks) e = issue(ci + 1);
			if (e != cudaSuccess) break;
			e = cudaEventSynchronize(ev[ci & 1]);
			if (e != cudaSuccess) break;
			const size_t off = ci * chunk_el, n = std::min(chunk_el, n_el - off);
			const R * __restrict__ src = h_pin[ci & 1];
			RFLOAT * __restrict__ dst = host + off;
			for (size_t i = 0; i < n; i++)
				dst[i] = (RFLOAT)src[i];
		}
		cudaError_t e2 = cudaStreamSynchronize(0);
		if (e == cudaSuccess) e = e2;
	}
	for (int i = 0; i < 2; i++)
	{
		if (ev[i]) cudaEventDestroy(ev[i]);
		pinnedHostPool().release(h_pin[i], cap[i]);
	}
	return e;
}

template <class R>
bool run(const SkipGriddingInput &in, RFLOAT *vol_out, std::string &reason, bool &warned_zero_weight)
{
	typedef typename CplxOf<R>::type C;
	warned_zero_weight = false;
	const Dims d = {in.zdim, in.ydim, in.xdim};
	const long int N = in.padoridim;
	const Dims dw = {N, N, N / 2 + 1};
	const size_t n_vol = (size_t)d.z * d.y * d.x, n_win = (size_t)dw.z * dw.y * dw.x;
	const size_t n_out = (size_t)in.ori_size * in.ori_size * in.ori_size;
	const size_t stage_el = (size_t)8 << 20;   // 64 MB of RFLOAT

	ReconstructCacheState *cache = tl_rec_cache;
	int cur_dev = 0;
	if (cache)
	{
		if (cudaGetDevice(&cur_dev) != cudaSuccess)
		{
			cudaGetLastError();
			reason = "no GPU";
			return false;
		}
		if (cache->devid != -1 && cache->devid != cur_dev)
			cache->releaseAll();
		cache->devid = cur_dev;
	}

	cufftHandle plan = 0;
	size_t work = 0;
	struct PlanGuard
	{
		cufftHandle *p = nullptr;
		bool owned = true;
		~PlanGuard() { if (owned && p && *p) { cufftDestroy(*p); *p = 0; } }
	} plan_guard{&plan, true};

	const bool use_double = std::is_same<R, double>::value;
	if (cache && cache->plan && cache->plan_N == (int)N && cache->plan_double == use_double)
	{
		plan = cache->plan;
		work = cache->plan_work;
		plan_guard.owned = false;
	}
	else
	{
		if (cache && cache->plan)
			cache->destroyPlan();
		if (cufftCreate(&plan) != CUFFT_SUCCESS) { reason = "cuFFT not available"; return false; }
		if (planC2R(plan, (int)N, (R*)nullptr, &work) != CUFFT_SUCCESS)
		{
			cufftDestroy(plan);
			plan = 0;
			plan_guard.owned = false;
			cudaGetLastError();
			if (cache) cache->releaseAll();
			reason = "cuFFT could not plan the inverse transform (probably not enough GPU memory)";
			return false;
		}
	}

	const size_t bw = n_vol * sizeof(R), bc = n_vol * sizeof(C), bwin = n_win * sizeof(C);
	const size_t fw_slot_bytes = std::max(bw, n_out * sizeof(R));
	const size_t wk_slot_bytes = std::max(bc, bwin);
	const size_t st_slot_bytes = stage_el * sizeof(RFLOAT);

	size_t extra_need = 0;
	if (cache)
	{
		if (cache->slotFw.cap < fw_slot_bytes) extra_need += fw_slot_bytes;
		if (cache->slotFc.cap < bc) extra_need += bc;
		if (cache->slotWork.cap < wk_slot_bytes) extra_need += wk_slot_bytes;
		if (cache->slotStage.cap < st_slot_bytes) extra_need += st_slot_bytes;
	}
	else
	{
		extra_need = std::max(bw + 2 * bc, bw + bc + bwin);
		extra_need = std::max(extra_need, bwin + n_out * sizeof(R)) + st_slot_bytes;
	}
	size_t free_b = 0, total_b = 0;
	cudaMemGetInfo(&free_b, &total_b);
	if (free_b < extra_need + ((size_t)256 << 20))
	{
		if (cache)
			cache->releaseAll();
		std::ostringstream o;
		o << "the GPU has " << (free_b >> 20) << " MB free and this needs " << ((extra_need + work) >> 20) << " MB";
		reason = o.str();
		return false;
	}

	if (cache && !cache->plan)
	{
		cache->plan = plan;
		cache->plan_N = (int)N;
		cache->plan_double = use_double;
		cache->plan_work = work;
		plan_guard.owned = false;
	}

	auto fail = [&](const char *what, cudaError_t e) -> bool
	{
		if (cache) cache->releaseAll();
		reason = std::string(what) + ": " + cudaGetErrorString(e);
		cudaGetLastError();
		return false;
	};

	gpuMemTrace("GPU reconstruction: start (cuFFT work area already planned)", (double)work);
	BufSlot dFw(cache ? &cache->slotFw : nullptr);
	BufSlot dFc(cache ? &cache->slotFc : nullptr);
	BufSlot dIn(cache ? &cache->slotWork : nullptr);
	BufSlot dRad(cache ? &cache->dRad : nullptr);
	BufSlot dSums(cache ? &cache->dSums : nullptr);
	BufSlot dCounts(cache ? &cache->dCounts : nullptr);
	BufSlot dFlags(cache ? &cache->dFlags : nullptr);
	BufSlot dShellInv(cache ? &cache->dShellInv : nullptr);
	BufSlot dShellPos(cache ? &cache->dShellPos : nullptr);
	BufSlot dStage(cache ? &cache->slotStage : nullptr);
	BufSlot dMaskSums(cache ? &cache->dMaskSums : nullptr);

	cudaError_t e;
	const int threads = 256;
	if ((e = dStage.alloc(st_slot_bytes)) != cudaSuccess) return fail("GPU memory", e);

	// Fweight = decenter(weight)
	if ((e = dIn.alloc(cache ? wk_slot_bytes : bw)) != cudaSuccess) return fail("GPU memory", e);
	if ((e = dFw.alloc(cache ? fw_slot_bytes : bw)) != cudaSuccess) return fail("GPU memory", e);
	if ((e = uploadAs<R>(in.weight, dIn.as<R>(), n_vol, dStage.as<RFLOAT>(), stage_el)) != cudaSuccess) return fail("copy to GPU", e);
	kDecenter<R><<<gridFor(n_vol), threads>>>(dIn.as<R>(), dFw.as<R>(), d, in.start_z, in.start_y, in.start_x, in.max_r2);
	dIn.release();

	if (in.do_map)
	{
		const int nshell = (int)in.shell_invtau2.size();
		std::vector<R> inv(in.shell_invtau2.begin(), in.shell_invtau2.end());
		if ((e = dShellInv.alloc(nshell * sizeof(R))) != cudaSuccess) return fail("GPU memory", e);
		if ((e = dShellPos.alloc(nshell)) != cudaSuccess) return fail("GPU memory", e);
		cudaMemcpy(dShellInv.p, inv.data(), nshell * sizeof(R), cudaMemcpyHostToDevice);
		cudaMemcpy(dShellPos.p, in.shell_tau2_positive.data(), nshell, cudaMemcpyHostToDevice);
		kMap<R><<<gridFor(n_vol), threads>>>(dFw.as<R>(), d, in.max_r2, (R)in.padding_factor, in.minres_map,
		                                     dShellInv.as<R>(), dShellPos.as<char>(), nshell);
	}

	// Fconv = decenter(data)
	if ((e = dIn.alloc(cache ? wk_slot_bytes : bc)) != cudaSuccess) return fail("GPU memory", e);
	if ((e = dFc.alloc(bc)) != cudaSuccess) return fail("GPU memory", e);
	if ((e = uploadAs<R>(in.data, dIn.as<R>(), 2 * n_vol, dStage.as<RFLOAT>(), stage_el)) != cudaSuccess) return fail("copy to GPU", e);
	dStage.release();
	kDecenter<C><<<gridFor(n_vol), threads>>>(dIn.as<C>(), dFc.as<C>(), d, in.start_z, in.start_y, in.start_x, in.max_r2);
	dIn.release();
	gpuMemTrace("GPU reconstruction: weight and data decentred", (double)(bw + bc));

	// 1/1000th of the radially averaged weight (sums in double)
	std::vector<double> sums(in.r_max, 0.);
	std::vector<unsigned long long> counts(in.r_max, 0);
	int flags[2] = {0, 0};
	if ((e = dSums.alloc(in.r_max * sizeof(double))) != cudaSuccess) return fail("GPU memory", e);
	if ((e = dCounts.alloc(in.r_max * sizeof(unsigned long long))) != cudaSuccess) return fail("GPU memory", e);
	if ((e = dFlags.alloc(sizeof(flags))) != cudaSuccess) return fail("GPU memory", e);
	cudaMemsetAsync(dSums.p, 0, in.r_max * sizeof(double), 0);
	cudaMemsetAsync(dCounts.p, 0, in.r_max * sizeof(unsigned long long), 0);
	cudaMemsetAsync(dFlags.p, 0, sizeof(flags), 0);
	const size_t shell_smem = (in.r_max <= 1024)
		? ((size_t)in.r_max * (sizeof(double) + sizeof(unsigned long long))) : 0;
	kShellSums<R><<<gridFor(n_vol), threads, shell_smem>>>(dFw.as<R>(), d, in.round_max_r2, (R)in.padding_factor, in.r_max,
	                                                       dSums.as<double>(), dCounts.as<unsigned long long>(), dFlags.as<int>());
	if ((e = cudaMemcpy(sums.data(), dSums.p, in.r_max * sizeof(double), cudaMemcpyDeviceToHost)) != cudaSuccess) return fail("GPU shell sums", e);
	cudaMemcpy(counts.data(), dCounts.p, in.r_max * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaMemcpy(flags, dFlags.p, sizeof(flags), cudaMemcpyDeviceToHost);
	if (flags[0])
		REPORT_ERROR("BUG: ires >=XSIZE(radavg_weight) ");
	std::vector<R> radavg(in.r_max);
	for (int sh = 0; sh < in.r_max; sh++)
	{
		if (counts[sh] > 0 || sums[sh] > 0.)
			radavg[sh] = (R)((RFLOAT)sums[sh] / (1000. * (RFLOAT)counts[sh]));
		else
			REPORT_ERROR("BUG: zeros in counter or radavg_weight!");
	}
	if ((e = dRad.alloc(in.r_max * sizeof(R))) != cudaSuccess) return fail("GPU memory", e);
	cudaMemcpy(dRad.p, radavg.data(), in.r_max * sizeof(R), cudaMemcpyHostToDevice);
	cudaMemsetAsync(dFlags.p, 0, sizeof(flags), 0);
	kDivide<R, C><<<gridFor(n_vol), threads>>>(dFc.as<C>(), dFw.as<R>(), d, (R)in.padding_factor, in.r_max,
	                                           dRad.as<R>(), dFlags.as<int>());
	dFw.release();

	// Window to the padded box, centre, inverse transform in place, window to the box.
	// Note: flags from kDivide stay on GPU until after kWindowReal/post-process so cuFFT
	// proceeds without a mid-pipeline D2H stall.
	BufSlot dWin(cache ? &cache->slotWork : nullptr);
	if ((e = dWin.alloc(cache ? wk_slot_bytes : bwin)) != cudaSuccess) return fail("GPU memory", e);
	kWindowFourierCenter<C><<<gridFor(n_win), threads>>>(dFc.as<C>(), d, dWin.as<C>(), dw);
	dFc.release();
	gpuMemTrace("GPU reconstruction: padded window in the card", (double)bwin);
	if (execC2R(plan, dWin.as<C>(), dWin.as<R>()) != CUFFT_SUCCESS)
	{
		if (cache) cache->releaseAll();
		reason = "cuFFT failed";
		return false;
	}
	gpuMemTrace("GPU reconstruction: inverse transform done");

	BufSlot dOut(cache ? &cache->slotFw : nullptr);
	if ((e = dOut.alloc(cache ? fw_slot_bytes : (n_out * sizeof(R)))) != cudaSuccess) return fail("GPU memory", e);
	kWindowReal<R><<<gridFor(n_out), threads>>>(dWin.as<R>(), N, dOut.as<R>(), in.ori_size, (R)in.normfft);

	if (in.do_soft_mask || in.gridding_mode != 0)
	{
		if (in.do_soft_mask)
		{
			if ((e = dMaskSums.alloc(2 * sizeof(double))) != cudaSuccess) return fail("GPU memory", e);
			cudaMemsetAsync(dMaskSums.p, 0, 2 * sizeof(double), 0);
			kSoftMaskSum<R><<<gridFor(n_out), threads>>>(dOut.as<R>(), in.ori_size, dMaskSums.as<double>());
		}
		if (in.do_soft_mask || in.gridding_mode != 0)
		{
			kSoftMaskAndGridding<R><<<gridFor(n_out), threads>>>(dOut.as<R>(), in.ori_size, in.do_soft_mask,
			                                                     dMaskSums.as<double>(), in.gridding_mode, (R)in.gridding_rrval);
		}
	}

	if ((e = downloadAsRfloat<R>(vol_out, dOut.as<R>(), n_out)) != cudaSuccess)
		return fail("copy from GPU", e);
	if ((e = cudaMemcpy(flags, dFlags.p, sizeof(flags), cudaMemcpyDeviceToHost)) != cudaSuccess)
		return fail("GPU flags copy", e);
	warned_zero_weight = flags[0] != 0;
	return true;
}

template <class R>
bool runDownsample(const RFLOAT *in, int orig_size, RFLOAT *out, int new_size)
{
	typedef typename CplxOf<R>::type C;
	DownsampleCacheState *cache = tl_ds_cache;
	int cur_dev = 0;
	if (cudaGetDevice(&cur_dev) != cudaSuccess) { cudaGetLastError(); return false; }

	const bool use_double = std::is_same<R, double>::value;
	const size_t n_in = (size_t)orig_size * orig_size * orig_size;
	const size_t n_ft_lg = (size_t)orig_size * orig_size * (orig_size / 2 + 1);
	const size_t n_ft_sm = (size_t)new_size * new_size * (new_size / 2 + 1);
	const size_t n_out = (size_t)new_size * new_size * new_size;
	const size_t stage_el = std::is_same<R, RFLOAT>::value ? 0 : std::min<size_t>(n_in, (size_t)4 << 20);
	const size_t st_bytes = stage_el * sizeof(RFLOAT);

	cufftHandle pfwd = 0, pinv = 0;
	struct PlansGuard
	{
		cufftHandle *a = nullptr, *b = nullptr;
		bool owned = true;
		~PlansGuard()
		{
			if (owned)
			{
				if (a && *a) { cufftDestroy(*a); *a = 0; }
				if (b && *b) { cufftDestroy(*b); *b = 0; }
			}
		}
	} pguard{&pfwd, &pinv, true};

	if (cache)
	{
		if (cache->devid != -1 && (cache->devid != cur_dev || cache->orig_size != orig_size ||
		                           cache->new_size != new_size || cache->plan_double != use_double))
			cache->releaseAll();
		cache->devid = cur_dev;
	}

	if (cache && cache->plan_fwd && cache->plan_inv && cache->orig_size == orig_size &&
	    cache->new_size == new_size && cache->plan_double == use_double)
	{
		pfwd = cache->plan_fwd;
		pinv = cache->plan_inv;
		pguard.owned = false;
	}
	else
	{
		if (cache && (cache->plan_fwd || cache->plan_inv))
		{
			if (cache->plan_fwd) { cufftDestroy(cache->plan_fwd); cache->plan_fwd = 0; }
			if (cache->plan_inv) { cufftDestroy(cache->plan_inv); cache->plan_inv = 0; }
		}
		size_t w1 = 0, w2 = 0;
		if (cufftCreate(&pfwd) != CUFFT_SUCCESS || cufftCreate(&pinv) != CUFFT_SUCCESS ||
		    planR2C(pfwd, orig_size, (R*)nullptr, &w1) != CUFFT_SUCCESS ||
		    planC2R(pinv, new_size, (R*)nullptr, &w2) != CUFFT_SUCCESS)
		{
			cudaGetLastError();
			if (cache) cache->releaseAll();
			return false;
		}
		const size_t buf_need = n_in * sizeof(R) + n_ft_lg * sizeof(C) + n_ft_sm * sizeof(C) + n_out * sizeof(R) + st_bytes;
		size_t extra_need = buf_need;
		if (cache)
		{
			extra_need = (cache->dIn.cap < n_in * sizeof(R) ? n_in * sizeof(R) : 0) +
			             (cache->dFtLarge.cap < n_ft_lg * sizeof(C) ? n_ft_lg * sizeof(C) : 0) +
			             (cache->dFtSmall.cap < n_ft_sm * sizeof(C) ? n_ft_sm * sizeof(C) : 0) +
			             (cache->dOut.cap < n_out * sizeof(R) ? n_out * sizeof(R) : 0) +
			             (cache->dStage.cap < st_bytes ? st_bytes : 0);
		}
		size_t free_b = 0, total_b = 0;
		cudaMemGetInfo(&free_b, &total_b);
		if (free_b < extra_need + ((size_t)128 << 20))
		{
			if (cache) cache->releaseAll();
			return false;
		}
		if (cache)
		{
			cache->plan_fwd = pfwd;
			cache->plan_inv = pinv;
			cache->orig_size = orig_size;
			cache->new_size = new_size;
			cache->plan_double = use_double;
			pguard.owned = false;
		}
	}

	BufSlot dIn(cache ? &cache->dIn : nullptr);
	BufSlot dFtLarge(cache ? &cache->dFtLarge : nullptr);
	BufSlot dFtSmall(cache ? &cache->dFtSmall : nullptr);
	BufSlot dOut(cache ? &cache->dOut : nullptr);
	BufSlot dStage(cache ? &cache->dStage : nullptr);
	if (dIn.alloc(n_in * sizeof(R)) != cudaSuccess ||
	    dFtLarge.alloc(n_ft_lg * sizeof(C)) != cudaSuccess ||
	    dFtSmall.alloc(n_ft_sm * sizeof(C)) != cudaSuccess ||
	    dOut.alloc(n_out * sizeof(R)) != cudaSuccess ||
	    (st_bytes > 0 && dStage.alloc(st_bytes) != cudaSuccess))
	{
		cudaGetLastError();
		if (cache) cache->releaseAll();
		return false;
	}
	if (uploadAs<R>(in, dIn.as<R>(), n_in, dStage.as<RFLOAT>(), stage_el) != cudaSuccess)
	{
		cudaGetLastError();
		if (cache) cache->releaseAll();
		return false;
	}
	dStage.release();
	if (execR2C(pfwd, dIn.as<R>(), dFtLarge.as<C>()) != CUFFT_SUCCESS)
	{
		cudaGetLastError();
		if (cache) cache->releaseAll();
		return false;
	}
	const R inv_norm = (R)1.0 / ((R)orig_size * (R)orig_size * (R)orig_size);
	kWindowFourierDownsample<R, C><<<gridFor(n_ft_sm), 256>>>(dFtLarge.as<C>(), orig_size, dFtSmall.as<C>(), new_size, inv_norm);
	if (execC2R(pinv, dFtSmall.as<C>(), dOut.as<R>()) != CUFFT_SUCCESS ||
	    downloadAsRfloat<R>(out, dOut.as<R>(), n_out) != cudaSuccess)
	{
		cudaGetLastError();
		if (cache) cache->releaseAll();
		return false;
	}
	return true;
}

}

ScopedReconstructCache::ScopedReconstructCache(bool enable) : active_(false)
{
	if (enable && tl_rec_cache == nullptr)
	{
		tl_rec_cache = new ReconstructCacheState();
		active_ = true;
	}
}

ScopedReconstructCache::~ScopedReconstructCache()
{
	if (active_ && tl_rec_cache != nullptr)
	{
		ReconstructCacheState *c = tl_rec_cache;
		tl_rec_cache = nullptr;
		delete c;
	}
}

ScopedDownsampleCache::ScopedDownsampleCache(bool enable) : active_(false)
{
	if (enable && tl_ds_cache == nullptr)
	{
		tl_ds_cache = new DownsampleCacheState();
		active_ = true;
	}
}

ScopedDownsampleCache::~ScopedDownsampleCache()
{
	if (active_ && tl_ds_cache != nullptr)
	{
		DownsampleCacheState *c = tl_ds_cache;
		tl_ds_cache = nullptr;
		delete c;
	}
}

bool downsampleVolume3D(const RFLOAT *in, int orig_size, RFLOAT *out, int new_size)
{
	if (!in || !out || orig_size <= new_size || new_size <= 0 || (orig_size % 2 != 0) || (new_size % 2 != 0))
		return false;
	int ndev = 0;
	if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { cudaGetLastError(); return false; }
	auto launch = [&]() -> bool {
		return doublePrecision() ? runDownsample<double>(in, orig_size, out, new_size)
		                         : runDownsample<float>(in, orig_size, out, new_size);
	};
	bool ok = launch();
	if (!ok && keepPoolActive() && relionHasRetainedGpuMemory())
	{
		cudaGetLastError();
		relionReleaseRetainedGpuMemory();
		ok = launch();
	}
	return ok;
}

bool keepPoolActive()
{
	// RELION_GPU_KEEP_POOL: on unless 0
	static const bool on = [] { const char *e = getenv("RELION_GPU_KEEP_POOL"); return !(e && *e && atoi(e) <= 0); }();
	return on;
}

bool reconstructSkipGridding(const SkipGriddingInput &in, RFLOAT *vol_out, std::string &reason, bool &warned_zero_weight)
{
	// The chunk kept for the next expectation step stays on the device and is given back only when the
	// reconstruction does not fit next to it (RELION_GPU_KEEP_POOL=0 gives it back first, as before).
	// Freeing and allocating again the ~20 GB chunk of the old pool takes 0.1 to 0.2 s per iteration, and
	// the reconstruction of the usual box sizes fits next to it (the chunk is sized to leave room for the
	// threads' own needs). The results do not change.
	const bool keep_pool = keepPoolActive();
	if (!keep_pool)
		relionReleaseRetainedGpuMemory(); // the chunk kept for the next expectation step
	int ndev = 0;
	if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { cudaGetLastError(); reason = "no GPU"; return false; }
	bool ok = in.double_precision ? run<double>(in, vol_out, reason, warned_zero_weight)
	                              : run<float>(in, vol_out, reason, warned_zero_weight);
	if (!ok && keep_pool && relionHasRetainedGpuMemory())
	{
		cudaGetLastError();
		relionReleaseRetainedGpuMemory();
		reason.clear();
		ok = in.double_precision ? run<double>(in, vol_out, reason, warned_zero_weight)
		                         : run<float>(in, vol_out, reason, warned_zero_weight);
	}
	return ok;
}

int selectDevice(const std::string &gpu_ids, int rank, int verb)
{
	int devCount = 0;
	if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount <= 0)
	{
		cudaGetLastError();
		return -1;
	}
	int dev = (rank >= 0) ? (rank % devCount) : 0;
	if (!gpu_ids.empty() && std::isdigit((unsigned char)gpu_ids[0]))
	{
		std::string copy = gpu_ids;
		std::vector<std::vector<std::string> > allThreadIDs;
		untangleDeviceIDs(copy, allThreadIDs);
		if (!allThreadIDs.empty())
		{
			const size_t ridx = (size_t)std::max(0, rank) % allThreadIDs.size();
			if (!allThreadIDs[ridx].empty())
				dev = textToInteger(allThreadIDs[ridx][0].c_str());
		}
	}
	if (dev < 0 || dev >= devCount)
		dev = 0;
	if (cudaSetDevice(dev) != cudaSuccess)
	{
		cudaGetLastError();
		return -1;
	}
	if (verb > 0)
		std::cout << " + Using GPU device " << dev << " for reconstruction" << std::endl;
	return dev;
}

}
