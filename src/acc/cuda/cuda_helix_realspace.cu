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
#include "src/acc/cuda/gpu_mem_trace.h"
#include <algorithm>
#include <cmath>
#include <sstream>

void relionReleaseRetainedGpuMemory(); // cuda_ml_optimiser.cu

namespace gpurec
{

namespace
{

struct HelixDev
{
	long int zdim, ydim, xdim, sz, sy, sx;
	RFLOAT r_min, r_max, d_min, d_max, D_min, D_max, z_min, z_max, rise_pix, cw;
	int rec_len;
};

__device__ __forceinline__ int hFLOOR(RFLOAT x) { return (x == (int)x) ? (int)x : ((x > 0) ? (int)x : (int)(x - 1)); }
__device__ __forceinline__ int hCEIL(RFLOAT x) { return (x == (int)x) ? (int)x : ((x > 0) ? (int)(x + 1) : (int)x); }

// The mask test of the first loop of the CPU code, for logical coordinates
__device__ __forceinline__ bool isOutside(long int k, long int i, long int j, const HelixDev &p)
{
	const RFLOAT dd = (RFLOAT)(i * i + j * j);
	const RFLOAT rr = dd + (RFLOAT)(k * k);
	const RFLOAT d = sqrt(dd), r = sqrt(rr);
	return (r > p.r_max) || (d < p.d_min) || (d > p.D_max);
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
		RFLOAT r = sqrt(rr);
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

			RFLOAT d000 = at(z0, y0, x0), d001 = at(z0, y0, x1), d010 = at(z0, y1, x0), d011 = at(z0, y1, x1);
			RFLOAT d100 = at(z1, y0, x0), d101 = at(z1, y0, x1), d110 = at(z1, y1, x0), d111 = at(z1, y1, x1);
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

}

bool helicalSymmetryRealSpace(const HelixRealSpaceInput &in, RFLOAT *v, std::string &reason, bool &bad)
{
	relionReleaseRetainedGpuMemory(); // the chunk kept for the next expectation step
	bad = false;
	gpuMemTrace("helical symmetrisation in real space: start");
	int ndev = 0;
	if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { cudaGetLastError(); reason = "no GPU"; return false; }
	const size_t n = (size_t)in.zdim * in.ydim * in.xdim, bytes = n * sizeof(RFLOAT);
	size_t free_b = 0, total_b = 0;
	cudaMemGetInfo(&free_b, &total_b);
	if (free_b < 2 * bytes + ((size_t)256 << 20))
	{
		std::ostringstream o;
		o << "the GPU has " << (free_b >> 20) << " MB free and this needs " << ((2 * bytes) >> 20) << " MB";
		reason = o.str();
		return false;
	}
	HelixDev p = {in.zdim, in.ydim, in.xdim, in.start_z, in.start_y, in.start_x,
	              in.r_min, in.r_max, in.d_min, in.d_max, in.D_min, in.D_max, in.z_min, in.z_max, in.rise_pix, in.cosine_width_pix,
	              (int)in.sin_rec.size()};
	RFLOAT *dv = nullptr, *dout = nullptr, *dsin = nullptr, *dcos = nullptr;
	int *dbad = nullptr;
	auto cleanup = [&]() { cudaFree(dv); cudaFree(dout); cudaFree(dsin); cudaFree(dcos); cudaFree(dbad); };
	cudaError_t e;
	if ((e = cudaMalloc(&dv, bytes)) != cudaSuccess || (e = cudaMalloc(&dout, bytes)) != cudaSuccess ||
	    (e = cudaMalloc(&dsin, in.sin_rec.size() * sizeof(RFLOAT))) != cudaSuccess ||
	    (e = cudaMalloc(&dcos, in.cos_rec.size() * sizeof(RFLOAT))) != cudaSuccess ||
	    (e = cudaMalloc(&dbad, sizeof(int))) != cudaSuccess)
	{
		cleanup();
		reason = std::string("GPU memory: ") + cudaGetErrorString(e);
		cudaGetLastError();
		return false;
	}
	gpuMemTrace("helical symmetrisation in real space: buffers in the card", (double) (2 * bytes));
	cudaMemcpy(dv, v, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dsin, in.sin_rec.data(), in.sin_rec.size() * sizeof(RFLOAT), cudaMemcpyHostToDevice);
	cudaMemcpy(dcos, in.cos_rec.data(), in.cos_rec.size() * sizeof(RFLOAT), cudaMemcpyHostToDevice);
	cudaMemset(dbad, 0, sizeof(int));
	const size_t blocks = std::min((n + 255) / 256, (size_t)65535 * 8);
	kHelixRealSpace<<<(int)blocks, 256>>>(dv, dout, p, dsin, dcos, dbad);
	int hbad = 0;
	cudaMemcpy(&hbad, dbad, sizeof(int), cudaMemcpyDeviceToHost);
	e = cudaMemcpy(v, dout, bytes, cudaMemcpyDeviceToHost);
	cleanup();
	if (e != cudaSuccess)
	{
		reason = std::string("GPU: ") + cudaGetErrorString(e);
		cudaGetLastError();
		return false;
	}
	bad = hbad != 0;
	return true;
}

}
