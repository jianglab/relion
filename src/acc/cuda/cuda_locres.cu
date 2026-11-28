/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/gpu_locres.h"
#include "src/args.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <vector>

namespace gpulocres
{

namespace
{

template <class R> struct CplxOf;
template <> struct CplxOf<float>  { typedef float2 type; };
template <> struct CplxOf<double> { typedef double2 type; };

template <class R> struct CufftTraits;
template <> struct CufftTraits<float>
{
	static cufftType r2cType() { return CUFFT_R2C; }
	static cufftType c2cType() { return CUFFT_C2C; }
	static cufftType c2rType() { return CUFFT_C2R; }
	static cufftResult execR2C(cufftHandle p, float *in, float2 *out)
	{ return cufftExecR2C(p, (cufftReal*)in, (cufftComplex*)out); }
	static cufftResult execC2C(cufftHandle p, float2 *in, float2 *out, int dir)
	{ return cufftExecC2C(p, (cufftComplex*)in, (cufftComplex*)out, dir); }
	static cufftResult execC2R(cufftHandle p, float2 *in, float *out)
	{ return cufftExecC2R(p, (cufftComplex*)in, (cufftReal*)out); }
};
template <> struct CufftTraits<double>
{
	static cufftType r2cType() { return CUFFT_D2Z; }
	static cufftType c2cType() { return CUFFT_Z2Z; }
	static cufftType c2rType() { return CUFFT_Z2D; }
	static cufftResult execR2C(cufftHandle p, double *in, double2 *out)
	{ return cufftExecD2Z(p, (cufftDoubleReal*)in, (cufftDoubleComplex*)out); }
	static cufftResult execC2C(cufftHandle p, double2 *in, double2 *out, int dir)
	{ return cufftExecZ2Z(p, (cufftDoubleComplex*)in, (cufftDoubleComplex*)out, dir); }
	static cufftResult execC2R(cufftHandle p, double2 *in, double *out)
	{ return cufftExecZ2D(p, (cufftDoubleComplex*)in, (cufftDoubleReal*)out); }
};

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

template <class T>
struct DevPtr
{
	T *p = nullptr;
	cudaError_t alloc(size_t count)
	{
		release();
		if (count == 0) return cudaSuccess;
		return cudaMalloc(&p, count * sizeof(T));
	}
	void release()
	{
		if (p) cudaFree(p);
		p = nullptr;
	}
	~DevPtr() { release(); }
};

template <class T>
struct PinnedPtr
{
	T *p = nullptr;
	cudaError_t alloc(size_t count)
	{
		release();
		if (count == 0) return cudaSuccess;
		return cudaMallocHost((void **)&p, count * sizeof(T));
	}
	void release()
	{
		if (p) cudaFreeHost(p);
		p = nullptr;
	}
	~PinnedPtr() { release(); }
};

template <class R>
__global__ void kMaskExtractX(
		const R * __restrict__ vol,
		const R * __restrict__ mask_tmpl,
		R * __restrict__ rx,
		long int nx, long int ny, long int S,
		long int kx0, long int cx,
		long int ky0, long int cy,
		long int kz0, long int cz,
		long int dm_x0, long int dm_y0, long int dm_z0)
{
	const size_t total = (size_t)S * S * nx;
	for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	     idx < total;
	     idx += (size_t)blockDim.x * gridDim.x)
	{
		const long int kx = (long int)(idx % nx);
		const size_t row = idx / nx;
		const long int dy = (long int)(row % S);
		const long int dz = (long int)(row / S);

		R out = (R)0;
		if (dz < cz && dy < cy && kx >= kx0 && kx < kx0 + cx)
		{
			const long int dx = kx - kx0;
			const size_t v_off = ((size_t)(kz0 + dz) * ny + (ky0 + dy)) * nx + kx;
			const size_t m_off = ((size_t)(dm_z0 + dz) * S + (dm_y0 + dy)) * S + (dm_x0 + dx);
			out = vol[v_off] * mask_tmpl[m_off];
		}
		rx[idx] = out;
	}
}

template <class C>
__global__ void kScatterXToY(
		const C * __restrict__ cx,
		C * __restrict__ colY,
		long int nfx, long int ny, long int S,
		long int ky0, long int cy, long int cz)
{
	const size_t total = (size_t)S * nfx * ny;
	for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	     idx < total;
	     idx += (size_t)blockDim.x * gridDim.x)
	{
		const long int ky = (long int)(idx % ny);
		const size_t col = idx / ny;
		const long int fx = (long int)(col % nfx);
		const long int dz = (long int)(col / nfx);

		C out{};
		if (dz < cz && ky >= ky0 && ky < ky0 + cy)
		{
			const long int dy = ky - ky0;
			out = cx[((size_t)dz * S + dy) * nfx + fx];
		}
		colY[idx] = out;
	}
}

template <class C>
__global__ void kScatterYToActiveZ(
		const C * __restrict__ outY,
		const int * __restrict__ active_ky,
		const int * __restrict__ active_fx,
		C * __restrict__ colZ,
		long int M, long int nfx, long int ny, long int nz,
		long int kz0, long int cz)
{
	const size_t total = (size_t)M * nz;
	for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	     idx < total;
	     idx += (size_t)blockDim.x * gridDim.x)
	{
		const long int kz = (long int)(idx % nz);
		const long int m = (long int)(idx / nz);

		C out{};
		if (kz >= kz0 && kz < kz0 + cz)
		{
			const long int dz = kz - kz0;
			const long int ky = active_ky[m];
			const long int fx = active_fx[m];
			out = outY[((size_t)dz * nfx + fx) * ny + ky];
		}
		colZ[idx] = out;
	}
}

template <class C>
__global__ void kFscReduceActive(
		const C * __restrict__ outZ1,
		const C * __restrict__ outZ2,
		const long long * __restrict__ active_ip2fx2,
		const int * __restrict__ r2_to_shell,
		double * __restrict__ num_out,
		double * __restrict__ den1_out,
		double * __restrict__ den2_out,
		long int M, long int nz, long int nfx, long long nfx2, bool use_smem)
{
	extern __shared__ double sh[];
	double *s_num = sh;
	double *s_d1 = sh + nfx;
	double *s_d2 = sh + 2 * nfx;
	if (use_smem)
	{
		for (long int i = threadIdx.x; i < nfx; i += blockDim.x)
		{
			s_num[i] = 0.0;
			s_d1[i] = 0.0;
			s_d2[i] = 0.0;
		}
		__syncthreads();
	}

	const size_t total = (size_t)M * nz;
	for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	     idx < total;
	     idx += (size_t)blockDim.x * gridDim.x)
	{
		const long int kz = (long int)(idx % nz);
		const long int m = (long int)(idx / nz);
		const long int kp = (kz < nfx) ? kz : (kz - nz);
		const long long d2 = active_ip2fx2[m] + (long long)kp * kp;
		if (d2 >= nfx2) continue;
		const int shell = r2_to_shell[d2];
		if (shell >= nfx) continue;

		const C z1 = outZ1[idx];
		const C z2 = outZ2[idx];
		const double z1x = (double)z1.x, z1y = (double)z1.y;
		const double z2x = (double)z2.x, z2y = (double)z2.y;
		const double nm = z1x * z2x + z1y * z2y;
		const double d1 = z1x * z1x + z1y * z1y;
		const double d2v = z2x * z2x + z2y * z2y;
		if (use_smem)
		{
			atomicAddDouble(&s_num[shell], nm);
			atomicAddDouble(&s_d1[shell], d1);
			atomicAddDouble(&s_d2[shell], d2v);
		}
		else
		{
			atomicAddDouble(&num_out[shell], nm);
			atomicAddDouble(&den1_out[shell], d1);
			atomicAddDouble(&den2_out[shell], d2v);
		}
	}

	if (use_smem)
	{
		__syncthreads();
		for (long int i = threadIdx.x; i < nfx; i += blockDim.x)
		{
			if (s_d1[i] != 0.0 || s_d2[i] != 0.0 || s_num[i] != 0.0)
			{
				atomicAddDouble(&num_out[i], s_num[i]);
				atomicAddDouble(&den1_out[i], s_d1[i]);
				atomicAddDouble(&den2_out[i], s_d2[i]);
			}
		}
	}
}

template <class R, class C>
__global__ void kFilterActiveZ(
		const C * __restrict__ FTsum_yxz,
		const int * __restrict__ active_ky,
		const int * __restrict__ active_fx,
		const long long * __restrict__ active_ip2fx2,
		const R * __restrict__ inv_w,
		C * __restrict__ colZ,
		long int M, long int nfx, long int nz, long long r2_bound)
{
	const size_t total = (size_t)M * nz;
	for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	     idx < total;
	     idx += (size_t)blockDim.x * gridDim.x)
	{
		const long int kz = (long int)(idx % nz);
		const long int m = (long int)(idx / nz);
		const long long ip2fx2 = active_ip2fx2[m];
		C out{};
		if (ip2fx2 <= r2_bound)
		{
			const long int kp = (kz < nfx) ? kz : (kz - nz);
			const long long d2 = ip2fx2 + (long long)kp * kp;
			if (d2 <= r2_bound)
			{
				const R wt = inv_w[d2];
				if (wt != (R)0)
				{
					const long int ky = active_ky[m];
					const long int fx = active_fx[m];
					const C ft = FTsum_yxz[((size_t)ky * nfx + fx) * nz + kz];
					out.x = ft.x * wt;
					out.y = ft.y * wt;
				}
			}
		}
		colZ[idx] = out;
	}
}

template <class C>
__global__ void kGatherZToY(
		const C * __restrict__ outZ,
		const int * __restrict__ col_idx,
		const long long * __restrict__ active_ip2fx2,
		C * __restrict__ colY,
		long int nfx, long int ny, long int nz, long int S,
		long int kz0, long int cz, long int n_fx_inv, long long r2_bound)
{
	const size_t total = (size_t)S * nfx * ny;
	for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	     idx < total;
	     idx += (size_t)blockDim.x * gridDim.x)
	{
		const long int ky = (long int)(idx % ny);
		const size_t col = idx / ny;
		const long int fx = (long int)(col % nfx);
		const long int dz = (long int)(col / nfx);

		C out{};
		if (dz < cz && fx < n_fx_inv)
		{
			const int m = col_idx[(size_t)ky * nfx + fx];
			if (m >= 0 && active_ip2fx2[m] <= r2_bound)
				out = outZ[(size_t)m * nz + (kz0 + dz)];
		}
		colY[idx] = out;
	}
}

template <class C>
__global__ void kGatherYToX(
		const C * __restrict__ outY,
		C * __restrict__ cx,
		long int nfx, long int ny, long int S,
		long int ky0, long int cy, long int cz, long int n_fx_inv)
{
	const size_t total = (size_t)S * S * nfx;
	for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	     idx < total;
	     idx += (size_t)blockDim.x * gridDim.x)
	{
		const long int fx = (long int)(idx % nfx);
		const size_t row = idx / nfx;
		const long int dy = (long int)(row % S);
		const long int dz = (long int)(row / S);

		C out{};
		if (dz < cz && dy < cy && fx < n_fx_inv)
			out = outY[((size_t)dz * nfx + fx) * ny + (ky0 + dy)];
		cx[idx] = out;
	}
}

template <class R>
__global__ void kAccumLocresVol(
		const R * __restrict__ rx,
		const R * __restrict__ mask_tmpl,
		RFLOAT * __restrict__ Ifil,
		RFLOAT * __restrict__ Ilocres,
		RFLOAT * __restrict__ Isumw,
		long int nx, long int ny, long int S,
		long int kx0, long int cx,
		long int ky0, long int cy,
		long int kz0, long int cz,
		long int dm_x0, long int dm_y0, long int dm_z0,
		double inv_local_resol)
{
	const size_t total = (size_t)cz * cy * cx;
	for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	     idx < total;
	     idx += (size_t)blockDim.x * gridDim.x)
	{
		const long int dx = (long int)(idx % cx);
		const size_t r = idx / cx;
		const long int dy = (long int)(r % cy);
		const long int dz = (long int)(r / cy);

		const double m = (double)mask_tmpl[((size_t)(dm_z0 + dz) * S + (dm_y0 + dy)) * S + (dm_x0 + dx)];
		const double filtered = (double)rx[((size_t)dz * S + dy) * nx + (kx0 + dx)];
		const size_t v_off = ((size_t)(kz0 + dz) * ny + (ky0 + dy)) * nx + (kx0 + dx);
		Ifil[v_off] += (RFLOAT)(m * filtered);
		Ilocres[v_off] += (RFLOAT)(m * inv_local_resol);
		Isumw[v_off] += (RFLOAT)m;
	}
}

inline int gridFor(size_t total, int threads = 256)
{
	if (total == 0) return 1;
	const size_t blocks = (total + (size_t)threads - 1) / (size_t)threads;
	return (int)std::min<size_t>(blocks, 65535);
}

bool make1DBatchedPlan(cufftHandle &plan, int n, int idist, int odist,
                       cufftType type, int batch, cudaStream_t stream)
{
	plan = 0;
	int rank = 1;
	int dims[1] = { n };
	int inembed[1] = { idist };
	int onembed[1] = { odist };
	if (cufftPlanMany(&plan, rank, dims, inembed, 1, idist, onembed, 1, odist, type, batch) != CUFFT_SUCCESS)
	{
		plan = 0;
		return false;
	}
	if (cufftSetStream(plan, stream) != CUFFT_SUCCESS)
		return false;
	return true;
}

template <class R>
cudaError_t uploadRfloatAs(const RFLOAT *src, R *dst, size_t count, cudaStream_t stream, std::vector<R> &scratch)
{
	if (std::is_same<R, RFLOAT>::value)
		return cudaMemcpyAsync(dst, src, count * sizeof(RFLOAT), cudaMemcpyHostToDevice, stream);

	const size_t chunk = (size_t)4 << 20;
	const size_t scratch_n = std::min(count, chunk);
	if (scratch.size() < scratch_n)
		scratch.resize(scratch_n);
	R * __restrict__ buf = scratch.data();
	for (size_t off = 0; off < count; off += chunk)
	{
		const size_t n = std::min(chunk, count - off);
		for (size_t i = 0; i < n; i++)
			buf[i] = (R)src[off + i];
		cudaError_t e = cudaMemcpyAsync(dst + off, buf, n * sizeof(R), cudaMemcpyHostToDevice, stream);
		if (e != cudaSuccess) return e;
		e = cudaStreamSynchronize(stream);
		if (e != cudaSuccess) return e;
	}
	return cudaSuccess;
}

} // namespace

int selectDevice(const std::string &gpu_ids, int rank, int size, int verb)
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
	{
		std::cout << " + Using GPU device " << dev << " for local resolution"
		          << (size > 1 ? (" (MPI rank " + std::to_string(rank) + ")") : "")
		          << std::endl;
	}
	return dev;
}

namespace
{
struct LocresGpuOps
{
	virtual ~LocresGpuOps() {}
	virtual bool computePointFsc(const LocresPointBox &pt, double *out_stats_6nfx, std::string &reason) = 0;
	virtual bool accumFilteredPoint(const LocresPointBox &pt, long int r2_bound, long int n_fx_inv,
	                                const double *inv_w, double inv_local_resol, std::string &reason) = 0;
	virtual bool finishVolumes(RFLOAT *Ifil_out, RFLOAT *Ilocres_out, RFLOAT *Isumw_out, std::string &reason) = 0;
};
}

struct LocresGpuSession::Impl
{
	LocresGpuOps *ops = nullptr;
	explicit Impl(LocresGpuOps *o) : ops(o) {}
	~Impl() { delete ops; }
};

template <class R>
struct LocresGpuImpl : LocresGpuOps
{
	typedef typename CplxOf<R>::type C;
	typedef CufftTraits<R> FFT;

	int device_id = -1;
	cudaStream_t stream = nullptr;
	bool has_stream = false;
	cudaEvent_t ev_inv_w = nullptr;
	bool inv_w_pending = false;

	long int nx = 0, ny = 0, nz = 0, nfx = 0, S = 0, M = 0;
	long long nfx2 = 0, inv_w_cap = 0;
	size_t nvox = 0;

	cufftHandle plan_x_fwd = 0, plan_y_fwd = 0, plan_z_fwd = 0;
	cufftHandle plan_z_bwd = 0, plan_y_bwd = 0, plan_x_bwd = 0;

	DevPtr<R> d_I1, d_I2, d_I1p, d_I2p;
	DevPtr<C> d_FTsum_yxz;
	DevPtr<R> d_mask_tmpl;
	DevPtr<int> d_r2_to_shell;
	DevPtr<int> d_active_ky, d_active_fx, d_col_idx;
	DevPtr<long long> d_active_ip2fx2;

	DevPtr<RFLOAT> d_Ifil, d_Ilocres, d_Isumw;
	DevPtr<R> d_rx, d_inv_w;
	DevPtr<double> d_fsc_buf;
	DevPtr<C> d_cx, d_colY, d_outY, d_colZ1, d_outZ1, d_colZ2, d_outZ2;

	PinnedPtr<double> h_fsc_pin;
	PinnedPtr<R> h_inv_w_pin;

	~LocresGpuImpl() override
	{
		if (device_id >= 0) cudaSetDevice(device_id);
		if (has_stream && stream) cudaStreamSynchronize(stream);
		if (ev_inv_w) { cudaEventDestroy(ev_inv_w); ev_inv_w = nullptr; }
		if (plan_x_fwd) { cufftDestroy(plan_x_fwd); plan_x_fwd = 0; }
		if (plan_y_fwd) { cufftDestroy(plan_y_fwd); plan_y_fwd = 0; }
		if (plan_z_fwd) { cufftDestroy(plan_z_fwd); plan_z_fwd = 0; }
		if (plan_z_bwd) { cufftDestroy(plan_z_bwd); plan_z_bwd = 0; }
		if (plan_y_bwd) { cufftDestroy(plan_y_bwd); plan_y_bwd = 0; }
		if (plan_x_bwd) { cufftDestroy(plan_x_bwd); plan_x_bwd = 0; }
		h_inv_w_pin.release();
		h_fsc_pin.release();
		if (has_stream && stream) { cudaStreamDestroy(stream); stream = nullptr; has_stream = false; }
	}

	bool initImpl(int dev, const LocresInitData &in, std::string &reason)
	{
		device_id = dev;
		nx = in.nx; ny = in.ny; nz = in.nz; nfx = in.nfx; S = in.S;
		nfx2 = (long long)in.nfx * in.nfx;
		inv_w_cap = nfx2 * 2 + 16;
		nvox = (size_t)in.nz * in.ny * in.nx;

		// Build active (ky, fx) list inside spherical FSC cut (matches CPU prunedMaskedFsc)
		std::vector<int> h_ky, h_fx;
		std::vector<long long> h_ip2fx2;
		std::vector<int> h_col_idx((size_t)in.ny * in.nfx, -1);
		for (long int ky = 0; ky < in.ny; ky++)
		{
			const long int ip = (ky < in.nfx) ? ky : (ky - in.ny);
			const long long ip2 = (long long)ip * ip;
			for (long int fx = 0; fx < in.nfx; fx++)
			{
				const long long ip2fx2 = ip2 + (long long)fx * fx;
				if (ip2fx2 >= nfx2) continue;
				h_col_idx[(size_t)ky * in.nfx + fx] = (int)h_ky.size();
				h_ky.push_back((int)ky);
				h_fx.push_back((int)fx);
				h_ip2fx2.push_back(ip2fx2);
			}
		}
		M = (long int)h_ky.size();
		if (M <= 0)
		{
			reason = "empty active frequency set";
			return false;
		}

		if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
		    cudaEventCreateWithFlags(&ev_inv_w, cudaEventDisableTiming) != cudaSuccess)
		{
			cudaGetLastError();
			reason = "cudaStreamCreateWithFlags failed";
			return false;
		}
		has_stream = true;

		// Estimate static + output + scratch GPU memory before allocating
		const size_t bytes_needed =
			4 * nvox * sizeof(R) +
			3 * nvox * sizeof(RFLOAT) +
			(size_t)in.ny * in.nfx * in.nz * sizeof(C) +
			(size_t)in.S * in.S * in.S * sizeof(R) +
			(size_t)in.S * in.S * in.nx * sizeof(R) +
			(size_t)in.S * in.S * in.nfx * sizeof(C) +
			2 * (size_t)in.S * in.nfx * in.ny * sizeof(C) +
			4 * (size_t)M * in.nz * sizeof(C) +
			(size_t)inv_w_cap * sizeof(R) +
			6 * (size_t)in.nfx * sizeof(double) +
			(size_t)nfx2 * sizeof(int) +
			(size_t)in.ny * in.nfx * sizeof(int) +
			2 * (size_t)M * sizeof(int) +
			(size_t)M * sizeof(long long);

		size_t free_b = 0, total_b = 0;
		cudaMemGetInfo(&free_b, &total_b);
		if (free_b < bytes_needed + ((size_t)256 << 20))
		{
			std::ostringstream oss;
			oss << "not enough GPU memory (needs ~" << (bytes_needed >> 20) << " MB + workspace, "
			    << (free_b >> 20) << " MB free)";
			reason = oss.str();
			return false;
		}

		auto fail = [&](const char *what) -> bool {
			cudaError_t ce = cudaGetLastError();
			std::ostringstream oss;
			oss << what;
			if (ce != cudaSuccess) oss << ": " << cudaGetErrorString(ce);
			reason = oss.str();
			return false;
		};

		if (!make1DBatchedPlan(plan_x_fwd, (int)in.nx, (int)in.nx, (int)in.nfx, FFT::r2cType(), (int)(in.S * in.S), stream) ||
		    !make1DBatchedPlan(plan_y_fwd, (int)in.ny, (int)in.ny, (int)in.ny,  FFT::c2cType(), (int)(in.S * in.nfx), stream) ||
		    !make1DBatchedPlan(plan_z_fwd, (int)in.nz, (int)in.nz, (int)in.nz,  FFT::c2cType(), (int)M, stream) ||
		    !make1DBatchedPlan(plan_z_bwd, (int)in.nz, (int)in.nz, (int)in.nz,  FFT::c2cType(), (int)M, stream) ||
		    !make1DBatchedPlan(plan_y_bwd, (int)in.ny, (int)in.ny, (int)in.ny,  FFT::c2cType(), (int)(in.S * in.nfx), stream) ||
		    !make1DBatchedPlan(plan_x_bwd, (int)in.nx, (int)in.nfx, (int)in.nx, FFT::c2rType(), (int)(in.S * in.S), stream))
			return fail("cufftPlanMany failed");

		if (d_I1.alloc(nvox) != cudaSuccess ||
		    d_I2.alloc(nvox) != cudaSuccess ||
		    d_I1p.alloc(nvox) != cudaSuccess ||
		    d_I2p.alloc(nvox) != cudaSuccess ||
		    d_FTsum_yxz.alloc((size_t)in.ny * in.nfx * in.nz) != cudaSuccess ||
		    d_mask_tmpl.alloc((size_t)in.S * in.S * in.S) != cudaSuccess ||
		    d_r2_to_shell.alloc((size_t)nfx2) != cudaSuccess ||
		    d_active_ky.alloc((size_t)M) != cudaSuccess ||
		    d_active_fx.alloc((size_t)M) != cudaSuccess ||
		    d_col_idx.alloc((size_t)in.ny * in.nfx) != cudaSuccess ||
		    d_active_ip2fx2.alloc((size_t)M) != cudaSuccess ||
		    d_Ifil.alloc(nvox) != cudaSuccess ||
		    d_Ilocres.alloc(nvox) != cudaSuccess ||
		    d_Isumw.alloc(nvox) != cudaSuccess ||
		    d_rx.alloc((size_t)in.S * in.S * in.nx) != cudaSuccess ||
		    d_cx.alloc((size_t)in.S * in.S * in.nfx) != cudaSuccess ||
		    d_colY.alloc((size_t)in.S * in.nfx * in.ny) != cudaSuccess ||
		    d_outY.alloc((size_t)in.S * in.nfx * in.ny) != cudaSuccess ||
		    d_colZ1.alloc((size_t)M * in.nz) != cudaSuccess ||
		    d_outZ1.alloc((size_t)M * in.nz) != cudaSuccess ||
		    d_colZ2.alloc((size_t)M * in.nz) != cudaSuccess ||
		    d_outZ2.alloc((size_t)M * in.nz) != cudaSuccess ||
		    d_fsc_buf.alloc((size_t)(6 * in.nfx)) != cudaSuccess ||
		    d_inv_w.alloc((size_t)inv_w_cap) != cudaSuccess ||
		    h_fsc_pin.alloc((size_t)(6 * in.nfx)) != cudaSuccess ||
		    h_inv_w_pin.alloc((size_t)inv_w_cap) != cudaSuccess)
			return fail("cudaMalloc/cudaMallocHost failed");

		std::vector<R> h_stage;
		if (uploadRfloatAs<R>(in.I1, d_I1.p, nvox, stream, h_stage) != cudaSuccess ||
		    uploadRfloatAs<R>(in.I2, d_I2.p, nvox, stream, h_stage) != cudaSuccess ||
		    uploadRfloatAs<R>(in.I1p, d_I1p.p, nvox, stream, h_stage) != cudaSuccess ||
		    uploadRfloatAs<R>(in.I2p, d_I2p.p, nvox, stream, h_stage) != cudaSuccess ||
		    uploadRfloatAs<R>(in.FTsum_yxz, (R*)d_FTsum_yxz.p, (size_t)in.ny * in.nfx * in.nz * 2, stream, h_stage) != cudaSuccess ||
		    uploadRfloatAs<R>(in.mask_tmpl, d_mask_tmpl.p, (size_t)in.S * in.S * in.S, stream, h_stage) != cudaSuccess ||
		    cudaMemcpyAsync(d_r2_to_shell.p, in.r2_to_shell, (size_t)nfx2 * sizeof(int), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
		    cudaMemcpyAsync(d_active_ky.p, h_ky.data(), (size_t)M * sizeof(int), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
		    cudaMemcpyAsync(d_active_fx.p, h_fx.data(), (size_t)M * sizeof(int), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
		    cudaMemcpyAsync(d_col_idx.p, h_col_idx.data(), (size_t)in.ny * in.nfx * sizeof(int), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
		    cudaMemcpyAsync(d_active_ip2fx2.p, h_ip2fx2.data(), (size_t)M * sizeof(long long), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
		    cudaMemsetAsync(d_Ifil.p, 0, nvox * sizeof(RFLOAT), stream) != cudaSuccess ||
		    cudaMemsetAsync(d_Ilocres.p, 0, nvox * sizeof(RFLOAT), stream) != cudaSuccess ||
		    cudaMemsetAsync(d_Isumw.p, 0, nvox * sizeof(RFLOAT), stream) != cudaSuccess ||
		    cudaStreamSynchronize(stream) != cudaSuccess)
			return fail("initial H2D copy failed");

		return true;
	}

	bool computePointFsc(const LocresPointBox &pt, double *out_stats_6nfx, std::string &reason) override
	{
		if (cudaSetDevice(device_id) != cudaSuccess) { reason = "cudaSetDevice failed"; return false; }

		const int thr = 256;
		const bool use_smem = (nfx <= 512);
		const size_t smem_bytes = use_smem ? ((size_t)3 * nfx * sizeof(double)) : 0;

		if (cudaMemsetAsync(d_fsc_buf.p, 0, (size_t)6 * nfx * sizeof(double), stream) != cudaSuccess)
		{
			reason = "cudaMemsetAsync fsc_buf failed";
			return false;
		}

		auto runForward = [&](const R *vol, C *d_colZ, C *d_outZ) -> bool {
			kMaskExtractX<R><<<gridFor((size_t)S * S * nx, thr), thr, 0, stream>>>(
				vol, d_mask_tmpl.p, d_rx.p,
				nx, ny, S,
				pt.kx0, pt.cx, pt.ky0, pt.cy, pt.kz0, pt.cz,
				pt.dm_x0, pt.dm_y0, pt.dm_z0);
			if (FFT::execR2C(plan_x_fwd, d_rx.p, d_cx.p) != CUFFT_SUCCESS)
				return false;
			kScatterXToY<C><<<gridFor((size_t)S * nfx * ny, thr), thr, 0, stream>>>(
				d_cx.p, d_colY.p, nfx, ny, S, pt.ky0, pt.cy, pt.cz);
			if (FFT::execC2C(plan_y_fwd, d_colY.p, d_outY.p, CUFFT_FORWARD) != CUFFT_SUCCESS)
				return false;
			kScatterYToActiveZ<C><<<gridFor((size_t)M * nz, thr), thr, 0, stream>>>(
				d_outY.p, d_active_ky.p, d_active_fx.p, d_colZ,
				M, nfx, ny, nz, pt.kz0, pt.cz);
			if (FFT::execC2C(plan_z_fwd, d_colZ, d_outZ, CUFFT_FORWARD) != CUFFT_SUCCESS)
				return false;
			return true;
		};

		if (!runForward(d_I1.p, d_colZ1.p, d_outZ1.p) ||
		    !runForward(d_I2.p, d_colZ2.p, d_outZ2.p))
		{
			reason = "cuFFT forward execution failed on (I1, I2)";
			return false;
		}
		kFscReduceActive<C><<<gridFor((size_t)M * nz, thr), thr, smem_bytes, stream>>>(
			d_outZ1.p, d_outZ2.p, d_active_ip2fx2.p, d_r2_to_shell.p,
			d_fsc_buf.p, d_fsc_buf.p + nfx, d_fsc_buf.p + 2 * nfx,
			M, nz, nfx, nfx2, use_smem);

		if (!runForward(d_I1p.p, d_colZ1.p, d_outZ1.p) ||
		    !runForward(d_I2p.p, d_colZ2.p, d_outZ2.p))
		{
			reason = "cuFFT forward execution failed on (I1p, I2p)";
			return false;
		}
		kFscReduceActive<C><<<gridFor((size_t)M * nz, thr), thr, smem_bytes, stream>>>(
			d_outZ1.p, d_outZ2.p, d_active_ip2fx2.p, d_r2_to_shell.p,
			d_fsc_buf.p + 3 * nfx, d_fsc_buf.p + 4 * nfx, d_fsc_buf.p + 5 * nfx,
			M, nz, nfx, nfx2, use_smem);

		if (cudaMemcpyAsync(h_fsc_pin.p, d_fsc_buf.p, (size_t)6 * nfx * sizeof(double),
		                    cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
		    cudaStreamSynchronize(stream) != cudaSuccess)
		{
			cudaError_t ce = cudaGetLastError();
			reason = std::string("FSC D2H failed: ") + cudaGetErrorString(ce);
			return false;
		}
		inv_w_pending = false;
		std::memcpy(out_stats_6nfx, h_fsc_pin.p, (size_t)6 * nfx * sizeof(double));
		return true;
	}

	bool accumFilteredPoint(const LocresPointBox &pt,
	                        long int r2_bound, long int n_fx_inv,
	                        const double *inv_w, double inv_local_resol,
	                        std::string &reason) override
	{
		if (r2_bound < 0 || r2_bound + 1 > inv_w_cap)
		{
			reason = "r2_bound exceeds GPU inv_w buffer";
			return false;
		}
		if (cudaSetDevice(device_id) != cudaSuccess) { reason = "cudaSetDevice failed"; return false; }
		if (inv_w_pending)
		{
			if (cudaEventSynchronize(ev_inv_w) != cudaSuccess)
			{
				reason = "cudaEventSynchronize inv_w failed";
				return false;
			}
			inv_w_pending = false;
		}

		for (long int i = 0; i <= r2_bound; i++)
			h_inv_w_pin.p[i] = (R)inv_w[i];

		const int thr = 256;
		if (cudaMemcpyAsync(d_inv_w.p, h_inv_w_pin.p, (size_t)(r2_bound + 1) * sizeof(R),
		                    cudaMemcpyHostToDevice, stream) != cudaSuccess ||
		    cudaEventRecord(ev_inv_w, stream) != cudaSuccess)
		{
			reason = "H2D inv_w failed";
			return false;
		}
		inv_w_pending = true;

		kFilterActiveZ<R, C><<<gridFor((size_t)M * nz, thr), thr, 0, stream>>>(
			d_FTsum_yxz.p, d_active_ky.p, d_active_fx.p, d_active_ip2fx2.p,
			d_inv_w.p, d_colZ1.p, M, nfx, nz, (long long)r2_bound);
		if (FFT::execC2C(plan_z_bwd, d_colZ1.p, d_outZ1.p, CUFFT_INVERSE) != CUFFT_SUCCESS)
		{
			reason = "cuFFT bwd z failed";
			return false;
		}

		kGatherZToY<C><<<gridFor((size_t)S * nfx * ny, thr), thr, 0, stream>>>(
			d_outZ1.p, d_col_idx.p, d_active_ip2fx2.p, d_colY.p,
			nfx, ny, nz, S, pt.kz0, pt.cz, n_fx_inv, (long long)r2_bound);
		if (FFT::execC2C(plan_y_bwd, d_colY.p, d_outY.p, CUFFT_INVERSE) != CUFFT_SUCCESS)
		{
			reason = "cuFFT bwd y failed";
			return false;
		}

		kGatherYToX<C><<<gridFor((size_t)S * S * nfx, thr), thr, 0, stream>>>(
			d_outY.p, d_cx.p, nfx, ny, S, pt.ky0, pt.cy, pt.cz, n_fx_inv);
		if (FFT::execC2R(plan_x_bwd, d_cx.p, d_rx.p) != CUFFT_SUCCESS)
		{
			reason = "cuFFT bwd x failed";
			return false;
		}

		kAccumLocresVol<R><<<gridFor((size_t)pt.cz * pt.cy * pt.cx, thr), thr, 0, stream>>>(
			d_rx.p, d_mask_tmpl.p, d_Ifil.p, d_Ilocres.p, d_Isumw.p,
			nx, ny, S, pt.kx0, pt.cx, pt.ky0, pt.cy, pt.kz0, pt.cz,
			pt.dm_x0, pt.dm_y0, pt.dm_z0, inv_local_resol);

		return true;
	}

	bool finishVolumes(RFLOAT *Ifil_out, RFLOAT *Ilocres_out, RFLOAT *Isumw_out, std::string &reason) override
	{
		if (cudaSetDevice(device_id) != cudaSuccess) { reason = "cudaSetDevice failed"; return false; }

		const size_t bytes = nvox * sizeof(RFLOAT);
		if (cudaMemcpyAsync(Ifil_out, d_Ifil.p, bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
		    cudaMemcpyAsync(Ilocres_out, d_Ilocres.p, bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
		    cudaMemcpyAsync(Isumw_out, d_Isumw.p, bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
		    cudaStreamSynchronize(stream) != cudaSuccess)
		{
			cudaError_t ce = cudaGetLastError();
			reason = std::string("finishVolumes D2H failed: ") + cudaGetErrorString(ce);
			return false;
		}
		return true;
	}
};

LocresGpuSession::LocresGpuSession() : impl_(nullptr) {}
LocresGpuSession::~LocresGpuSession() { release(); }

void LocresGpuSession::release()
{
	delete impl_;
	impl_ = nullptr;
}

bool LocresGpuSession::init(int device_id, const LocresInitData &in, std::string &reason)
{
	release();
	if (device_id < 0 || !in.I1 || !in.I2 || !in.I1p || !in.I2p ||
	    !in.FTsum_yxz || !in.mask_tmpl || !in.r2_to_shell ||
	    in.nx <= 0 || in.ny <= 0 || in.nz <= 0 || in.nfx <= 0 || in.S <= 0)
	{
		reason = "invalid inputs or device";
		return false;
	}
	if (cudaSetDevice(device_id) != cudaSuccess)
	{
		cudaGetLastError();
		reason = "cudaSetDevice failed";
		return false;
	}

	if (in.double_precision)
	{
		LocresGpuImpl<double> *s = new LocresGpuImpl<double>();
		if (!s->initImpl(device_id, in, reason)) { delete s; return false; }
		impl_ = new Impl(s);
	}
	else
	{
		LocresGpuImpl<float> *s = new LocresGpuImpl<float>();
		if (!s->initImpl(device_id, in, reason)) { delete s; return false; }
		impl_ = new Impl(s);
	}
	return true;
}

bool LocresGpuSession::computePointFsc(const LocresPointBox &pt, double *out_stats_6nfx, std::string &reason)
{
	if (!impl_ || !impl_->ops) { reason = "not initialized"; return false; }
	return impl_->ops->computePointFsc(pt, out_stats_6nfx, reason);
}

bool LocresGpuSession::accumFilteredPoint(const LocresPointBox &pt,
                                            long int r2_bound, long int n_fx_inv,
                                            const double *inv_w, double inv_local_resol,
                                            std::string &reason)
{
	if (!impl_ || !impl_->ops) { reason = "not initialized"; return false; }
	return impl_->ops->accumFilteredPoint(pt, r2_bound, n_fx_inv, inv_w, inv_local_resol, reason);
}

bool LocresGpuSession::finishVolumes(RFLOAT *Ifil_out, RFLOAT *Ilocres_out, RFLOAT *Isumw_out, std::string &reason)
{
	if (!impl_ || !impl_->ops) { reason = "not initialized"; return false; }
	return impl_->ops->finishVolumes(Ifil_out, Ilocres_out, Isumw_out, reason);
}

} // namespace gpulocres
