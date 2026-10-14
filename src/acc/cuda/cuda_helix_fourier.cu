/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

// BackProjector::applyHelicalSymmetry() on the GPU: see src/gpu_reconstruct.h and src/backprojector.cpp,
// whose loop body this mirrors line by line (one thread per output voxel, the operators added in the
// same order, so the thread count and the grouping of planes do not change the result).

#include "src/gpu_reconstruct.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sstream>

void relionReleaseRetainedGpuMemory(); // cuda_ml_optimiser.cu

namespace gpurec
{

namespace
{

__device__ __forceinline__ int fFLOOR(RFLOAT x) { return (x == (int)x) ? (int)x : ((x > 0) ? (int)x : (int)(x - 1)); }

// A rotation about Z keeps z, and the partner of a voxel with negative x (the Hermitian symmetry) is in
// the plane -z, so plane k reads the planes k, k+1, -k and -k+1. The volume is therefore cut into groups
// of |z| in [a, b]: the output is the planes -b..-a (neg) and max(a,1)..b (pos), the input the planes
// -b..-a+1 (neg) and a..b+1 (pos); for a = 0 these form the single block -b..b+1.
struct GroupDev
{
	long int a, klo, pos_off, nneg_out;
	bool single;
};

__device__ __forceinline__ size_t loc(long int k, const GroupDev &g)
{
	if (g.single) return (size_t)(k - g.klo);
	return k >= g.a ? (size_t)(g.pos_off + (k - g.a)) : (size_t)(k - g.klo);
}

__global__ void kHelixFourier(const RFLOAT *din_data, const RFLOAT *din_weight, RFLOAT *dout_data, RFLOAT *dout_weight,
                              GroupDev grp, long int nout, long int zdim, long int ydim, long int xdim,
                              long int start_z, long int start_y, int rmax2, int nhh, bool has_rise,
                              const RFLOAT *R, const RFLOAT *shift_cos, const RFLOAT *shift_sin)
{
	const size_t plane_vox = (size_t)ydim * xdim;
	const size_t total = (size_t)nout * plane_vox;
	for (size_t here = blockIdx.x * (size_t)blockDim.x + threadIdx.x; here < total; here += (size_t)blockDim.x * gridDim.x)
	{
		const long int pl = here / plane_vox;           // output plane within the group
		const size_t rest = here % plane_vox;
		const long int iy = rest / xdim, jx = rest % xdim;
		const long int k = pl < grp.nneg_out ? grp.klo + pl : (grp.a > 0 ? grp.a : 1) + (pl - grp.nneg_out);
		const long int i = start_y + iy, j = jx; // start_x is zero
		const size_t src = loc(k, grp) * plane_vox + rest;

		RFLOAT sum_re = din_data[2 * src], sum_im = din_data[2 * src + 1];
		RFLOAT sum_w = din_weight[src];

		const RFLOAT x = (RFLOAT)j, y = (RFLOAT)i, z = (RFLOAT)k;
		const RFLOAT r2 = x * x + y * y + z * z;
		if (r2 <= rmax2)
		{
			for (int h = 0; h < nhh; h++)
			{
				const RFLOAT *Rh = R + 9 * h;
				RFLOAT xp = x * Rh[0] + y * Rh[1] + z * Rh[2];
				RFLOAT yp = x * Rh[3] + y * Rh[4] + z * Rh[5];
				RFLOAT zp = x * Rh[6] + y * Rh[7] + z * Rh[8];
				bool is_neg_x = false;
				if (xp < 0)
				{
					xp = -xp; yp = -yp; zp = -zp;
					is_neg_x = true;
				}
				int x0 = fFLOOR(xp);
				RFLOAT fx = xp - x0;
				int x1 = x0 + 1;
				int y0 = fFLOOR(yp);
				RFLOAT fy = yp - y0;
				y0 -= (int)start_y;
				int y1 = y0 + 1;
				int z0 = fFLOOR(zp);
				RFLOAT fz = zp - z0;
				const long int zl0 = (long int)loc(z0, grp), zl1 = (long int)loc(z0 + 1, grp);

				const size_t p00 = ((size_t)zl0 * ydim + y0) * xdim, p01 = ((size_t)zl0 * ydim + y1) * xdim;
				const size_t p10 = ((size_t)zl1 * ydim + y0) * xdim, p11 = ((size_t)zl1 * ydim + y1) * xdim;

				// LIN_INTERP(a, l, h) = l + (h - l) * a, for the real and the imaginary part
				#define LI(a, l, hh_) ((l) + ((hh_) - (l)) * (a))
				RFLOAT re000 = din_data[2 * (p00 + x0)], im000 = din_data[2 * (p00 + x0) + 1];
				RFLOAT re001 = din_data[2 * (p00 + x1)], im001 = din_data[2 * (p00 + x1) + 1];
				RFLOAT re010 = din_data[2 * (p01 + x0)], im010 = din_data[2 * (p01 + x0) + 1];
				RFLOAT re011 = din_data[2 * (p01 + x1)], im011 = din_data[2 * (p01 + x1) + 1];
				RFLOAT re100 = din_data[2 * (p10 + x0)], im100 = din_data[2 * (p10 + x0) + 1];
				RFLOAT re101 = din_data[2 * (p10 + x1)], im101 = din_data[2 * (p10 + x1) + 1];
				RFLOAT re110 = din_data[2 * (p11 + x0)], im110 = din_data[2 * (p11 + x0) + 1];
				RFLOAT re111 = din_data[2 * (p11 + x1)], im111 = din_data[2 * (p11 + x1) + 1];

				RFLOAT rx00 = LI(fx, re000, re001), ix00 = LI(fx, im000, im001);
				RFLOAT rx01 = LI(fx, re100, re101), ix01 = LI(fx, im100, im101);
				RFLOAT rx10 = LI(fx, re010, re011), ix10 = LI(fx, im010, im011);
				RFLOAT rx11 = LI(fx, re110, re111), ix11 = LI(fx, im110, im111);
				RFLOAT rxy0 = LI(fy, rx00, rx10), ixy0 = LI(fy, ix00, ix10);
				RFLOAT rxy1 = LI(fy, rx01, rx11), ixy1 = LI(fy, ix01, ix11);
				RFLOAT dre = LI(fz, rxy0, rxy1), dim_ = LI(fz, ixy0, ixy1);

				if (is_neg_x)
					dim_ = -dim_;

				if (has_rise)
				{
					RFLOAT a = shift_cos[(size_t)h * zdim + (k - start_z)];
					RFLOAT b = shift_sin[(size_t)h * zdim + (k - start_z)];
					RFLOAT c = dre;
					RFLOAT d = dim_;
					RFLOAT ac = a * c;
					RFLOAT bd = b * d;
					RFLOAT ab_cd = (a + b) * (c + d);
					dre = ac - bd;
					dim_ = ab_cd - ac - bd;
				}
				sum_re += dre;
				sum_im += dim_;

				RFLOAT w000 = din_weight[p00 + x0], w001 = din_weight[p00 + x1], w010 = din_weight[p01 + x0], w011 = din_weight[p01 + x1];
				RFLOAT w100 = din_weight[p10 + x0], w101 = din_weight[p10 + x1], w110 = din_weight[p11 + x0], w111 = din_weight[p11 + x1];
				RFLOAT wx00 = LI(fx, w000, w001), wx01 = LI(fx, w100, w101), wx10 = LI(fx, w010, w011), wx11 = LI(fx, w110, w111);
				RFLOAT wxy0 = LI(fy, wx00, wx10), wxy1 = LI(fy, wx01, wx11);
				sum_w += LI(fz, wxy0, wxy1);
				#undef LI
			}
		}
		dout_data[2 * here] = sum_re;
		dout_data[2 * here + 1] = sum_im;
		dout_weight[here] = sum_w;
	}
}

}

bool helicalSymmetryFourier(const HelixFourierInput &in, RFLOAT *data, RFLOAT *weight, std::string &reason,
                            std::string &choice, bool &partial)
{
	partial = false;
	if (in.start_x != 0) { reason = "the volume does not start at x = 0"; return false; }
	const int nhh = (int)(in.R.size() / 9);
	if (nhh < 1) { reason = "no symmetry operators"; return false; }
	for (int h = 0; h < nhh; h++)
		if (in.R[9 * h + 6] != 0. || in.R[9 * h + 7] != 0. || in.R[9 * h + 8] != 1.)
		{
			reason = "a symmetry operator is not a rotation about Z";
			return false;
		}
	// Planes with |z| up to kmax hold voxels inside the radius; the others are left as they are. The
	// volume must be symmetric in z and leave a plane of margin beyond the radius, as the arrays
	// of BackProjector do (the neighbours read by the interpolation are then all inside).
	const long int kmax = (long int)std::floor(std::sqrt((double)in.rmax2));
	const long int finish_z = in.start_z + in.zdim - 1;
	if (in.start_z != -finish_z || kmax + 1 > finish_z || kmax + 1 > in.xdim - 1 ||
	    kmax + 1 > -in.start_y || kmax + 1 > in.start_y + in.ydim - 1)
	{
		reason = "the volume is not symmetric in z or has no margin beyond the radius";
		return false;
	}
	relionReleaseRetainedGpuMemory(); // the chunk kept for the next expectation step
	int ndev = 0;
	if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { cudaGetLastError(); reason = "no GPU"; return false; }

	const size_t plane_vox = (size_t)in.ydim * in.xdim;
	const size_t plane_bytes = plane_vox * 3 * sizeof(RFLOAT); // data (2) + weight (1)
	const long int nm = kmax + 1;                              // values of |z|: 0 .. kmax

	size_t free_b = 0, total_b = 0;
	cudaMemGetInfo(&free_b, &total_b);
	const size_t reserve = (size_t)384 << 20;
	// a group of c values of |z| has 2c + 2 input planes and 2c output planes
	long int c = free_b > reserve ? (long int)(((free_b - reserve) / plane_bytes - 2) / 4) : 0;
	c = std::min(c, nm);
	// Manual override for experts and for the tests: the most values of |z| in a group
	if (const char *ep = getenv("RELION_GPU_SYMMETRISE_PLANES"))
		if (atoi(ep) > 0) c = std::min(c, (long int)atoi(ep));

	const size_t nsc = in.shift_cos.size() * sizeof(RFLOAT);
	RFLOAT *dR = nullptr, *dcos = nullptr, *dsin = nullptr, *din_d = nullptr, *din_w = nullptr, *dout_d = nullptr, *dout_w = nullptr;
	auto cleanup = [&]() { cudaFree(dR); cudaFree(dcos); cudaFree(dsin); cudaFree(din_d); cudaFree(din_w); cudaFree(dout_d); cudaFree(dout_w);
	                       dR = dcos = dsin = din_d = din_w = dout_d = dout_w = nullptr; };
	cudaError_t e = cudaSuccess;
	// back off to smaller groups when an allocation fails (another process may have taken the memory)
	for (; c >= 1; c = c / 2)
	{
		cleanup();
		if ((e = cudaMalloc(&din_d, (size_t)(2 * c + 2) * plane_vox * 2 * sizeof(RFLOAT))) == cudaSuccess &&
		    (e = cudaMalloc(&din_w, (size_t)(2 * c + 2) * plane_vox * sizeof(RFLOAT))) == cudaSuccess &&
		    (e = cudaMalloc(&dout_d, (size_t)(2 * c) * plane_vox * 2 * sizeof(RFLOAT))) == cudaSuccess &&
		    (e = cudaMalloc(&dout_w, (size_t)(2 * c) * plane_vox * sizeof(RFLOAT))) == cudaSuccess)
			break;
		cudaGetLastError();
	}
	if (c < 1)
	{
		cleanup();
		std::ostringstream o;
		o << "the GPU has " << (free_b >> 20) << " MB free and a pair of z planes needs " << ((6 * plane_bytes) >> 20) << " MB";
		reason = o.str();
		return false;
	}
	if ((e = cudaMalloc(&dR, in.R.size() * sizeof(RFLOAT))) != cudaSuccess ||
	    (in.has_rise && ((e = cudaMalloc(&dcos, nsc)) != cudaSuccess || (e = cudaMalloc(&dsin, nsc)) != cudaSuccess)))
	{
		cleanup();
		reason = std::string("GPU memory: ") + cudaGetErrorString(e);
		cudaGetLastError();
		return false;
	}
	cudaMemcpy(dR, in.R.data(), in.R.size() * sizeof(RFLOAT), cudaMemcpyHostToDevice);
	if (in.has_rise)
	{
		cudaMemcpy(dcos, in.shift_cos.data(), nsc, cudaMemcpyHostToDevice);
		cudaMemcpy(dsin, in.shift_sin.data(), nsc, cudaMemcpyHostToDevice);
	}

	// Groups go up in |z|. A group writes the planes +-m back into the arrays at once, but the next group
	// still needs the original of the plane -b (its -a + 1): that one is kept aside before the write-back.
	// Everything else a group reads has not been written yet (planes with |z| >= a).
	auto host_plane = [&](long int k) -> size_t { return (size_t)(k - in.start_z) * plane_vox; };
	std::vector<RFLOAT> stash_d(plane_vox * 2), stash_w(plane_vox);
	const size_t pb_d = plane_vox * 2 * sizeof(RFLOAT), pb_w = plane_vox * sizeof(RFLOAT);
	int ngroups = 0;
	bool written = false;
	for (long int a = 0; a < nm; a += c)
	{
		const long int b = std::min(a + c - 1, nm - 1);
		GroupDev g;
		g.a = a; g.klo = -b; g.single = (a == 0);
		const long int nneg_out = b - a + 1;                          // planes -b .. -a
		const long int npos_out = b - (a > 0 ? a : 1) + 1;            // planes max(a,1) .. b
		const long int nout = nneg_out + npos_out;
		g.nneg_out = nneg_out;
		long int nneg_in = 0, npos_in = 0;
		if (g.single)
		{
			g.pos_off = 0;
			nneg_in = 2 * b + 2;                                      // planes -b .. b+1
		}
		else
		{
			nneg_in = b - a + 2;                                      // planes -b .. -a+1
			npos_in = b - a + 2;                                      // planes a .. b+1
			g.pos_off = nneg_in;
		}
		// upload
		e = cudaSuccess;
		if (g.single)
		{
			e = cudaMemcpy(din_d, data + host_plane(-b) * 2, (size_t)nneg_in * pb_d, cudaMemcpyHostToDevice);
			if (e == cudaSuccess) e = cudaMemcpy(din_w, weight + host_plane(-b), (size_t)nneg_in * pb_w, cudaMemcpyHostToDevice);
		}
		else
		{
			// negative block: planes -b .. -a from the arrays, then the original of -a+1 kept aside
			e = cudaMemcpy(din_d, data + host_plane(-b) * 2, (size_t)(nneg_in - 1) * pb_d, cudaMemcpyHostToDevice);
			if (e == cudaSuccess) e = cudaMemcpy(din_w, weight + host_plane(-b), (size_t)(nneg_in - 1) * pb_w, cudaMemcpyHostToDevice);
			if (e == cudaSuccess) e = cudaMemcpy(din_d + (size_t)(nneg_in - 1) * plane_vox * 2, stash_d.data(), pb_d, cudaMemcpyHostToDevice);
			if (e == cudaSuccess) e = cudaMemcpy(din_w + (size_t)(nneg_in - 1) * plane_vox, stash_w.data(), pb_w, cudaMemcpyHostToDevice);
			if (e == cudaSuccess) e = cudaMemcpy(din_d + (size_t)nneg_in * plane_vox * 2, data + host_plane(a) * 2, (size_t)npos_in * pb_d, cudaMemcpyHostToDevice);
			if (e == cudaSuccess) e = cudaMemcpy(din_w + (size_t)nneg_in * plane_vox, weight + host_plane(a), (size_t)npos_in * pb_w, cudaMemcpyHostToDevice);
		}
		if (e == cudaSuccess)
		{
			const size_t total = (size_t)nout * plane_vox;
			const size_t blocks = std::min((total + 255) / 256, (size_t)65535 * 8);
			kHelixFourier<<<(int)blocks, 256>>>(din_d, din_w, dout_d, dout_w, g, nout, in.zdim, in.ydim, in.xdim,
			                                    in.start_z, in.start_y, in.rmax2, nhh, in.has_rise, dR, dcos, dsin);
			e = cudaGetLastError();
			if (e == cudaSuccess) e = cudaDeviceSynchronize();
		}
		if (e == cudaSuccess)
		{
			// the original of plane -b, for the next group
			memcpy(stash_d.data(), data + host_plane(-b) * 2, pb_d);
			memcpy(stash_w.data(), weight + host_plane(-b), pb_w);
			written = true;
			e = cudaMemcpy(data + host_plane(-b) * 2, dout_d, (size_t)nneg_out * pb_d, cudaMemcpyDeviceToHost);
			if (e == cudaSuccess) e = cudaMemcpy(weight + host_plane(-b), dout_w, (size_t)nneg_out * pb_w, cudaMemcpyDeviceToHost);
			if (e == cudaSuccess && npos_out > 0)
			{
				const long int p0 = a > 0 ? a : 1;
				e = cudaMemcpy(data + host_plane(p0) * 2, dout_d + (size_t)nneg_out * plane_vox * 2, (size_t)npos_out * pb_d, cudaMemcpyDeviceToHost);
				if (e == cudaSuccess)
					e = cudaMemcpy(weight + host_plane(p0), dout_w + (size_t)nneg_out * plane_vox, (size_t)npos_out * pb_w, cudaMemcpyDeviceToHost);
			}
		}
		if (e != cudaSuccess)
		{
			// the first group fails before anything is written back; later failures leave the arrays half done
			partial = written;
			reason = std::string("GPU: ") + cudaGetErrorString(e);
			cudaGetLastError();
			cleanup();
			return false;
		}
		ngroups++;
	}
	cleanup();
	std::ostringstream o;
	o << c << " values of |z| per group, " << ngroups << " group" << (ngroups > 1 ? "s" : "") << ", "
	  << (free_b >> 20) << " MB free at the start";
	choice = o.str();
	return true;
}

}
