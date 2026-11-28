/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

// imposeHelicalSymmetryInRealSpace() on the GPU: see src/gpu_reconstruct.h and src/helix.cpp, whose
// loop body this mirrors line by line.

#include "src/gpu_reconstruct.h"

#include <cuda_runtime.h>
#include "src/acc/cuda/cuda_mem_utils.h"
#include "src/acc/cuda/gpu_mem_trace.h"
#include <algorithm>
#include <cmath>
#include <sstream>

void relionReleaseRetainedGpuMemory(); // cuda_ml_optimiser.cu
bool relionHasRetainedGpuMemory();     // cuda_ml_optimiser.cu

namespace gpurec
{

namespace
{

struct HelixDev
{
	long int zdim, ydim, xdim, sz, sy, sx;
	RFLOAT r_min, r_max, r_max2, d_min, d_min2, d_max, D_min, D_max, D_max2, z_min, z_max, rise_pix, cw;
	int rec_len;
	bool cylinder;   // r: the distance along the helical axis (Z), not from the centre
};

__device__ __forceinline__ int hFLOOR(RFLOAT x) { return (x == (int)x) ? (int)x : ((x > 0) ? (int)x : (int)(x - 1)); }
__device__ __forceinline__ int hCEIL(RFLOAT x) { return (x == (int)x) ? (int)x : ((x > 0) ? (int)(x + 1) : (int)x); }

// The mask test of the first loop of the CPU code, for logical coordinates (sqrt-free)
__device__ __forceinline__ bool isOutside(long int k, long int i, long int j, const HelixDev &p)
{
	const RFLOAT dd = (RFLOAT)(i * i + j * j);
	if (p.D_max2 < 0. || dd > p.D_max2) return true;
	if (p.d_min2 >= 0. && dd < p.d_min2) return true;
	if (p.cylinder) return (RFLOAT)(k < 0 ? -k : k) > p.r_max;
	if (p.r_max2 < 0.) return true;
	return (dd + (RFLOAT)(k * k)) > p.r_max2;
}

__global__ void kHelixRealSpace(const RFLOAT *v, RFLOAT *vout, HelixDev p, const RFLOAT *sin_rec, const RFLOAT *cos_rec, int *bad)
{
	const size_t total = (size_t)p.zdim * p.ydim * p.xdim;
	const long int vx = p.xdim, vxy = p.xdim * p.ydim;
	for (size_t here = blockIdx.x * (size_t)blockDim.x + threadIdx.x; here < total; here += (size_t)blockDim.x * gridDim.x)
	{
		const long int jd = here % p.xdim, id_ = (here / p.xdim) % p.ydim, kd = here / vxy;
		const long int k = kd + p.sz, i = id_ + p.sy, j = jd + p.sx;
		// v as the original loop saw it when it reached this voxel: voxels outside the mask that came
		// before it had been set to zero
		auto at = [&](long int z, long int y, long int x) -> RFLOAT
		{
			if (z < 0 || z >= p.zdim || y < 0 || y >= p.ydim || x < 0 || x >= p.xdim) return 0.;
			const long int n = z * vxy + y * vx + x;
			if (n < (long int)here && isOutside(z + p.sz, y + p.sy, x + p.sx, p)) return 0.;
			return v[n];
		};

		RFLOAT dd = (RFLOAT)(i * i + j * j);
		RFLOAT rr = dd + (RFLOAT)(k * k);
		RFLOAT d = sqrt(dd);
		RFLOAT r = p.cylinder ? (RFLOAT)(k < 0 ? -k : k) : sqrt(rr);
		if ((r > p.r_max) || (d < p.d_min) || (d > p.D_max))
		{
			vout[here] = 0.;
			continue;
		}

		RFLOAT zi = (RFLOAT)(k), yi = (RFLOAT)(i), xi = (RFLOAT)(j);
		int rot_max = -(hCEIL((zi - p.z_max) / p.rise_pix));
		int rot_min = -(hFLOOR((zi - p.z_min) / p.rise_pix));
		if (rot_max < rot_min)
		{
			*bad = 1;
			vout[here] = 0.;
			continue;
		}

		// (xp, yp) is a pure Z-rotation of (xi, yi), so its cylindrical radius is still d;
		// any corner of its 2x2x2 trilinear stencil shifts in X/Y by < 1.42 and in Z by < 1.1.
		const bool radial_interior = (d + 1.5 < p.D_max) && (p.d_min <= 0. || d - 1.5 > p.d_min);
		const RFLOAT d_pad = d + 1.5;
		const RFLOAT d_pad2 = d_pad * d_pad;

		RFLOAT pix_sum = 0., pix_weight = 0.;
		for (int idr = rot_min; idr <= rot_max; idr++)
		{
			RFLOAT sin_val, cos_val;
			const int a = (idr >= 0) ? idr : -idr;
			if (a >= p.rec_len) { *bad = 2; continue; }
			if (idr >= 0) { sin_val = sin_rec[idr]; cos_val = cos_rec[idr]; }
			else { sin_val = (-1.) * sin_rec[-idr]; cos_val = cos_rec[-idr]; }

			RFLOAT zp = zi + ((RFLOAT)(idr)) * p.rise_pix;
			RFLOAT yp = xi * sin_val + yi * cos_val;
			RFLOAT xp = xi * cos_val - yi * sin_val;

			int x0, y0, z0, x1, y1, z1;
			RFLOAT fx, fy, fz;
			x0 = hFLOOR(xp); fx = xp - x0; x0 -= p.sx; x1 = x0 + 1;
			y0 = hFLOOR(yp); fy = yp - y0; y0 -= p.sy; y1 = y0 + 1;
			z0 = hFLOOR(zp); fz = zp - z0; z0 -= p.sz; z1 = z0 + 1;

			RFLOAT d000, d001, d010, d011, d100, d101, d110, d111;
			const bool in_bounds = (z0 >= 0 && z1 < p.zdim && y0 >= 0 && y1 < p.ydim && x0 >= 0 && x1 < p.xdim);
			const RFLOAT z_abs_max = fabs(zp) + 1.1;
			const bool mask_interior = radial_interior &&
				(p.cylinder ? (z_abs_max < p.r_max) : (p.r_max2 >= 0. && d_pad2 + z_abs_max * z_abs_max < p.r_max2));
			if (in_bounds && mask_interior)
			{
				const size_t base00 = ((size_t)z0 * vxy + y0 * vx + x0);
				const size_t base01 = base00 + vx;
				const size_t base10 = base00 + vxy;
				const size_t base11 = base10 + vx;
				d000 = v[base00];     d001 = v[base00 + 1];
				d010 = v[base01];     d011 = v[base01 + 1];
				d100 = v[base10];     d101 = v[base10 + 1];
				d110 = v[base11];     d111 = v[base11 + 1];
			}
			else
			{
				d000 = at(z0, y0, x0); d001 = at(z0, y0, x1); d010 = at(z0, y1, x0); d011 = at(z0, y1, x1);
				d100 = at(z1, y0, x0); d101 = at(z1, y0, x1); d110 = at(z1, y1, x0); d111 = at(z1, y1, x1);
			}
			// LIN_INTERP(a, l, h) = l + (h - l) * a
			RFLOAT dx00 = d000 + (d001 - d000) * fx, dx01 = d100 + (d101 - d100) * fx;
			RFLOAT dx10 = d010 + (d011 - d010) * fx, dx11 = d110 + (d111 - d110) * fx;
			RFLOAT dxy0 = dx00 + (dx10 - dx00) * fy, dxy1 = dx01 + (dx11 - dx01) * fy;
			pix_sum += dxy0 + (dxy1 - dxy0) * fz;
			pix_weight += 1.;
		}

		if (pix_weight > 0.9)
		{
			RFLOAT out = pix_sum / pix_weight;
			if ((d > p.d_max) && (d < p.D_min) && (r < p.r_min))
			{}
			else
			{
				pix_weight = 1.;
				if (d < p.d_max)
					pix_weight = 0.5 + (0.5 * cos(M_PI * ((p.d_max - d) / p.cw)));
				else if (d > p.D_min)
					pix_weight = 0.5 + (0.5 * cos(M_PI * ((d - p.D_min) / p.cw)));
				if (r > p.r_min)
				{
					pix_sum = 0.5 + (0.5 * cos(M_PI * ((r - p.r_min) / p.cw)));
					pix_weight = (pix_sum < pix_weight) ? (pix_sum) : (pix_weight);
				}
				out *= pix_weight;
			}
			vout[here] = out;
		}
		else
			vout[here] = 0.;
	}
}

struct HelixCacheState
{
	int devid = -1;
	RFLOAT *dv = nullptr, *dout = nullptr, *dsin = nullptr, *dcos = nullptr;
	int *dbad = nullptr;
	size_t cap_dv = 0, cap_dout = 0, cap_dsin = 0, cap_dcos = 0;

	void releaseAll()
	{
		if (dv) { cudaFree(dv); dv = nullptr; }
		if (dout) { cudaFree(dout); dout = nullptr; }
		if (dsin) { cudaFree(dsin); dsin = nullptr; }
		if (dcos) { cudaFree(dcos); dcos = nullptr; }
		if (dbad) { cudaFree(dbad); dbad = nullptr; }
		cap_dv = cap_dout = cap_dsin = cap_dcos = 0;
		devid = -1;
	}

	~HelixCacheState() { releaseAll(); }
};

thread_local HelixCacheState *tl_helix_cache = nullptr;

}

ScopedHelixRealSpaceCache::ScopedHelixRealSpaceCache(bool enable) : active_(false)
{
	if (enable && tl_helix_cache == nullptr)
	{
		tl_helix_cache = new HelixCacheState();
		active_ = true;
	}
}

ScopedHelixRealSpaceCache::~ScopedHelixRealSpaceCache()
{
	if (active_ && tl_helix_cache != nullptr)
	{
		HelixCacheState *c = tl_helix_cache;
		tl_helix_cache = nullptr;
		delete c;
	}
}

bool helicalSymmetryRealSpace(const HelixRealSpaceInput &in, RFLOAT *v, std::string &reason, bool &bad)
{
	const bool keep_pool = keepPoolActive();
	if (!keep_pool)
		relionReleaseRetainedGpuMemory(); // the chunk kept for the next expectation step
	bad = false;
	gpuMemTrace("helical symmetrisation in real space: start");
	int ndev = 0;
	if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { cudaGetLastError(); reason = "no GPU"; return false; }

	HelixCacheState *cache = tl_helix_cache;
	int cur_dev = 0;
	if (cache)
	{
		if (cudaGetDevice(&cur_dev) != cudaSuccess) { cudaGetLastError(); reason = "no GPU"; return false; }
		if (cache->devid != -1 && cache->devid != cur_dev)
			cache->releaseAll();
		cache->devid = cur_dev;
	}

	const size_t n = (size_t)in.zdim * in.ydim * in.xdim, bytes = n * sizeof(RFLOAT);
	const size_t sin_bytes = in.sin_rec.size() * sizeof(RFLOAT);
	const size_t cos_bytes = in.cos_rec.size() * sizeof(RFLOAT);
	const RFLOAT r_max2 = (in.r_max >= 0.) ? (in.r_max * in.r_max) : -1.;
	const RFLOAT d_min2 = (in.d_min > 0.) ? (in.d_min * in.d_min) : -1.;
	const RFLOAT D_max2 = (in.D_max >= 0.) ? (in.D_max * in.D_max) : -1.;
	HelixDev p = {in.zdim, in.ydim, in.xdim, in.start_z, in.start_y, in.start_x,
	              in.r_min, in.r_max, r_max2, in.d_min, d_min2, in.d_max, in.D_min, in.D_max, D_max2,
	              in.z_min, in.z_max, in.rise_pix, in.cosine_width_pix,
	              (int)in.sin_rec.size(), in.cylinder};
	RFLOAT *dv = nullptr, *dout = nullptr, *dsin = nullptr, *dcos = nullptr;
	int *dbad = nullptr;
	auto cleanup = [&]()
	{
		if (cache)
			cache->releaseAll();
		else
		{
			cudaFree(dv); cudaFree(dout); cudaFree(dsin); cudaFree(dcos); cudaFree(dbad);
		}
		dv = dout = dsin = dcos = nullptr; dbad = nullptr;
	};
	cudaError_t e = cudaSuccess;
	auto ensurePtr = [&](auto *&ptr, size_t &cap, size_t req) -> cudaError_t
	{
		if (req <= cap && ptr != nullptr) return cudaSuccess;
		if (ptr) { cudaFree(ptr); ptr = nullptr; cap = 0; }
		cudaError_t r = cudaMalloc(&ptr, req);
		if (r == cudaSuccess) cap = req;
		else { ptr = nullptr; cap = 0; }
		return r;
	};
	auto allocBufs = [&]() -> bool
	{
		size_t free_b = 0, total_b = 0;
		cudaMemGetInfo(&free_b, &total_b);
		size_t extra_need = 2 * bytes;
		if (cache)
			extra_need = (cache->cap_dv < bytes ? bytes : 0) + (cache->cap_dout < bytes ? bytes : 0);
		if (free_b < extra_need + ((size_t)256 << 20))
		{
			if (cache) cache->releaseAll();
			std::ostringstream o;
			o << "the GPU has " << (free_b >> 20) << " MB free and this needs " << ((extra_need) >> 20) << " MB";
			reason = o.str();
			return false;
		}
		if (cache)
		{
			if ((e = ensurePtr(cache->dv, cache->cap_dv, bytes)) != cudaSuccess ||
			    (e = ensurePtr(cache->dout, cache->cap_dout, bytes)) != cudaSuccess ||
			    (e = ensurePtr(cache->dsin, cache->cap_dsin, sin_bytes)) != cudaSuccess ||
			    (e = ensurePtr(cache->dcos, cache->cap_dcos, cos_bytes)) != cudaSuccess ||
			    (cache->dbad == nullptr && (e = cudaMalloc(&cache->dbad, sizeof(int))) != cudaSuccess))
			{
				cleanup();
				reason = std::string("GPU memory: ") + cudaGetErrorString(e);
				cudaGetLastError();
				return false;
			}
			dv = cache->dv; dout = cache->dout; dsin = cache->dsin; dcos = cache->dcos; dbad = cache->dbad;
		}
		else
		{
			if ((e = cudaMalloc(&dv, bytes)) != cudaSuccess || (e = cudaMalloc(&dout, bytes)) != cudaSuccess ||
			    (e = cudaMalloc(&dsin, sin_bytes)) != cudaSuccess ||
			    (e = cudaMalloc(&dcos, cos_bytes)) != cudaSuccess ||
			    (e = cudaMalloc(&dbad, sizeof(int))) != cudaSuccess)
			{
				cleanup();
				reason = std::string("GPU memory: ") + cudaGetErrorString(e);
				cudaGetLastError();
				return false;
			}
		}
		return true;
	};
	if (!allocBufs())
	{
		if (keep_pool && relionHasRetainedGpuMemory())
		{
			cudaGetLastError();
			relionReleaseRetainedGpuMemory();
			reason.clear();
			if (!allocBufs())
				return false;
		}
		else
			return false;
	}
	gpuMemTrace("helical symmetrisation in real space: buffers in the card", (double) (2 * bytes));
	if ((e = pinnedBigUpload(dv, v, bytes)) != cudaSuccess)
	{
		cleanup();
		reason = std::string("copy to GPU: ") + cudaGetErrorString(e);
		cudaGetLastError();
		return false;
	}
	cudaMemcpyAsync(dsin, in.sin_rec.data(), sin_bytes, cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(dcos, in.cos_rec.data(), cos_bytes, cudaMemcpyHostToDevice, 0);
	cudaMemsetAsync(dbad, 0, sizeof(int), 0);
	const size_t blocks = std::min((n + 255) / 256, (size_t)65535 * 8);
	kHelixRealSpace<<<(int)blocks, 256>>>(dv, dout, p, dsin, dcos, dbad);
	e = pinnedBigDownload(v, dout, bytes);
	int hbad = 0;
	if (e == cudaSuccess)
		e = cudaMemcpy(&hbad, dbad, sizeof(int), cudaMemcpyDeviceToHost);
	if (!cache)
		cleanup();
	if (e != cudaSuccess)
	{
		if (cache) cache->releaseAll();
		reason = std::string("GPU: ") + cudaGetErrorString(e);
		cudaGetLastError();
		return false;
	}
	bad = hbad != 0;
	return true;
}

}
