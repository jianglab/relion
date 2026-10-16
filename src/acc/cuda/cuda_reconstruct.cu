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
#include "src/error.h"

#include <cuda_runtime.h>
#include "src/acc/cuda/gpu_mem_trace.h"
#include <cufft.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <type_traits>

void relionReleaseRetainedGpuMemory(); // cuda_ml_optimiser.cu

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

// The macros of src/macros.h, for the device
__device__ __forceinline__ int dROUND(RFLOAT x) { return (x > 0) ? (int)(x + 0.5) : (int)(x - 0.5); }
__device__ __forceinline__ int dFLOOR(RFLOAT x) { return (x == (int)x) ? (int)x : ((x > 0) ? (int)x : (int)(x - 1)); }

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
__global__ void kMap(R *Fw, Dims d, int max_r2, float pad, int minres_map,
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
			const int ires = dROUND(sqrt((RFLOAT)r2) / pad);
			if (ires >= nshell) continue;   // (the CPU would read past tau2)
			R invw = Fw[n];
			R invtau2;
			if (shell_pos[ires])
				invtau2 = shell_invtau2[ires];
			else if (invw > 1e-20)
				invtau2 = 1. / (0.001 * invw);
			else
				invtau2 = 0.;
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
__global__ void kShellSums(const R *Fw, Dims d, int round_max_r2, float pad, int r_max,
                           double *sums, unsigned long long *counts, int *bad)
{
	const size_t total = (size_t)d.z * d.y * d.x;
	GRID_STRIDE(n, total)
	{
		long int k, i, j;
		decompose(n, d, k, i, j);
		const long int kp = wrapK(k, d), ip = wrapI(i, d), jp = j;
		const int r2 = kp * kp + ip * ip + jp * jp;
		if (r2 < round_max_r2)
		{
			const int ires = dFLOOR(sqrt((RFLOAT)r2) / pad);
			if (ires >= r_max) { *bad = 1; continue; }
			atomicAddDouble(&sums[ires], (double)Fw[n]);
			atomicAdd(&counts[ires], 1ull);
		}
	}
}

// Fconv /= max(Fweight, radavg_weight[ires]) (skipped, with a warning, where that is 0)
template <class R, class C>
__global__ void kDivide(C *Fc, const R *Fw, Dims d, float pad, int r_max, const R *radavg, int *zero_seen)
{
	const size_t total = (size_t)d.z * d.y * d.x;
	GRID_STRIDE(n, total)
	{
		long int k, i, j;
		decompose(n, d, k, i, j);
		const long int kp = wrapK(k, d), ip = wrapI(i, d), jp = j;
		const int r2 = kp * kp + ip * ip + jp * jp;
		const int ires = dFLOOR(sqrt((RFLOAT)r2) / pad);
		const R a = Fw[n], b = radavg[(ires < r_max) ? ires : (r_max - 1)];
		const R weight = (a > b) ? a : b;   // XMIPP_MAX
		C c = Fc[n];
		if (weight == 0.)
		{
			if (c.x != 0. || c.y != 0.) *zero_seen = 1;
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
		C v; v.x = 0.; v.y = 0.;
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
__global__ void kWindowReal(const R *in, long int N, RFLOAT *out, long int ori, R normfft)
{
	const long int row = 2 * (N / 2 + 1), off = N / 2 - ori / 2;
	const size_t total = (size_t)ori * ori * ori;
	GRID_STRIDE(n, total)
	{
		const long int x = n % ori, y = (n / ori) % ori, z = n / ((size_t)ori * ori);
		out[n] = (RFLOAT)(in[((size_t)(z + off) * N + (y + off)) * row + (x + off)] / normfft);
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

// Copy n_el RFLOATs from the host into dst (type R) through a small staging buffer
template <class R>
cudaError_t uploadAs(const RFLOAT *host, R *dst, size_t n_el, RFLOAT *stage, size_t stage_el)
{
	for (size_t done = 0; done < n_el; done += stage_el)
	{
		const size_t n = std::min(stage_el, n_el - done);
		if (std::is_same<R, RFLOAT>::value)
		{
			cudaError_t e = cudaMemcpy(dst + done, host + done, n * sizeof(RFLOAT), cudaMemcpyHostToDevice);
			if (e != cudaSuccess) return e;
			continue;
		}
		cudaError_t e = cudaMemcpy(stage, host + done, n * sizeof(RFLOAT), cudaMemcpyHostToDevice);
		if (e != cudaSuccess) return e;
		kConvert<R><<<gridFor(n), 256>>>(stage, dst + done, n);
	}
	return cudaGetLastError();
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

	cufftHandle plan;
	size_t work = 0;
	if (cufftCreate(&plan) != CUFFT_SUCCESS) { reason = "cuFFT not available"; return false; }
	if (planC2R(plan, (int)N, (R*)nullptr, &work) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		cudaGetLastError();
		reason = "cuFFT could not plan the inverse transform (probably not enough GPU memory)";
		return false;
	}
	// Peak memory: Fweight + Fconv + a copy of data in R (before decentering), or Fweight + Fconv +
	// window, plus the staging buffer; the cuFFT work area is already allocated
	const size_t bw = n_vol * sizeof(R), bc = n_vol * sizeof(C), bwin = n_win * sizeof(C);
	size_t need = std::max(bw + 2 * bc, bw + bc + bwin);
	need = std::max(need, bwin + n_out * sizeof(RFLOAT)) + stage_el * sizeof(RFLOAT);
	size_t free_b = 0, total_b = 0;
	cudaMemGetInfo(&free_b, &total_b);
	if (free_b < need + ((size_t)256 << 20))
	{
		cufftDestroy(plan);
		std::ostringstream o;
		o << "the GPU has " << (free_b >> 20) << " MB free and this needs " << ((need + work) >> 20) << " MB";
		reason = o.str();
		return false;
	}

	auto fail = [&](const char *what, cudaError_t e) -> bool
	{
		cufftDestroy(plan);
		reason = std::string(what) + ": " + cudaGetErrorString(e);
		cudaGetLastError();
		return false;
	};

	gpuMemTrace("GPU reconstruction: start (cuFFT work area already planned)", (double)work);
	DevBuf dFw, dFc, dIn, dWin, dRad, dSums, dCounts, dFlags, dShellInv, dShellPos, dStage;
	cudaError_t e;
	const int threads = 256;
	if ((e = dStage.alloc(stage_el * sizeof(RFLOAT))) != cudaSuccess) return fail("GPU memory", e);

	// Fweight = decenter(weight)
	if ((e = dIn.alloc(bw)) != cudaSuccess) return fail("GPU memory", e);
	if ((e = dFw.alloc(bw)) != cudaSuccess) return fail("GPU memory", e);
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
		kMap<R><<<gridFor(n_vol), threads>>>(dFw.as<R>(), d, in.max_r2, in.padding_factor, in.minres_map,
		                                     dShellInv.as<R>(), dShellPos.as<char>(), nshell);
	}

	// Fconv = decenter(data)
	if ((e = dIn.alloc(bc)) != cudaSuccess) return fail("GPU memory", e);
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
	cudaMemset(dSums.p, 0, in.r_max * sizeof(double));
	cudaMemset(dCounts.p, 0, in.r_max * sizeof(unsigned long long));
	cudaMemset(dFlags.p, 0, sizeof(flags));
	kShellSums<R><<<gridFor(n_vol), threads>>>(dFw.as<R>(), d, in.round_max_r2, in.padding_factor, in.r_max,
	                                           dSums.as<double>(), dCounts.as<unsigned long long>(), dFlags.as<int>());
	if ((e = cudaMemcpy(sums.data(), dSums.p, in.r_max * sizeof(double), cudaMemcpyDeviceToHost)) != cudaSuccess) return fail("GPU shell sums", e);
	cudaMemcpy(counts.data(), dCounts.p, in.r_max * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaMemcpy(flags, dFlags.p, sizeof(flags), cudaMemcpyDeviceToHost);
	if (flags[0])
	{
		cufftDestroy(plan);
		REPORT_ERROR("BUG: ires >=XSIZE(radavg_weight) ");
	}
	std::vector<R> radavg(in.r_max);
	for (int sh = 0; sh < in.r_max; sh++)
	{
		if (counts[sh] > 0 || sums[sh] > 0.)
			radavg[sh] = (R)((RFLOAT)sums[sh] / (1000. * (RFLOAT)counts[sh]));
		else
		{
			cufftDestroy(plan);
			REPORT_ERROR("BUG: zeros in counter or radavg_weight!");
		}
	}
	if ((e = dRad.alloc(in.r_max * sizeof(R))) != cudaSuccess) return fail("GPU memory", e);
	cudaMemcpy(dRad.p, radavg.data(), in.r_max * sizeof(R), cudaMemcpyHostToDevice);
	cudaMemset(dFlags.p, 0, sizeof(flags));
	kDivide<R, C><<<gridFor(n_vol), threads>>>(dFc.as<C>(), dFw.as<R>(), d, in.padding_factor, in.r_max,
	                                           dRad.as<R>(), dFlags.as<int>());
	cudaMemcpy(flags, dFlags.p, sizeof(flags), cudaMemcpyDeviceToHost);
	warned_zero_weight = flags[0] != 0;
	dFw.release();

	// Window to the padded box, centre, inverse transform in place, window to the box
	if ((e = dWin.alloc(bwin)) != cudaSuccess) return fail("GPU memory", e);
	kWindowFourierCenter<C><<<gridFor(n_win), threads>>>(dFc.as<C>(), d, dWin.as<C>(), dw);
	dFc.release();
	gpuMemTrace("GPU reconstruction: padded window in the card", (double)bwin);
	if (execC2R(plan, dWin.as<C>(), dWin.as<R>()) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		reason = "cuFFT failed";
		return false;
	}
	cufftDestroy(plan);
	gpuMemTrace("GPU reconstruction: inverse transform done");
	DevBuf dOut;
	if ((e = dOut.alloc(n_out * sizeof(RFLOAT))) != cudaSuccess) { reason = std::string("GPU memory: ") + cudaGetErrorString(e); return false; }
	kWindowReal<R><<<gridFor(n_out), threads>>>(dWin.as<R>(), N, dOut.as<RFLOAT>(), in.ori_size, (R)in.normfft);
	if ((e = cudaMemcpy(vol_out, dOut.p, n_out * sizeof(RFLOAT), cudaMemcpyDeviceToHost)) != cudaSuccess)
	{
		reason = std::string("copy from GPU: ") + cudaGetErrorString(e);
		cudaGetLastError();
		return false;
	}
	return true;
}

}

bool reconstructSkipGridding(const SkipGriddingInput &in, RFLOAT *vol_out, std::string &reason, bool &warned_zero_weight)
{
	relionReleaseRetainedGpuMemory(); // the chunk kept for the next expectation step
	int ndev = 0;
	if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { cudaGetLastError(); reason = "no GPU"; return false; }
	if (in.double_precision)
		return run<double>(in, vol_out, reason, warned_zero_weight);
	return run<float>(in, vol_out, reason, warned_zero_weight);
}

}
