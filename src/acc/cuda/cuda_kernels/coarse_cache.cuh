#ifndef COARSE_CACHE_CUH_
#define COARSE_CACHE_CUH_

#include <algorithm>
#include <cstdio>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

#ifndef ACC_DOUBLE_PRECISION

/*
 * Coarse differences from cached reference projections (single precision, 2D images against 3D references).
 *
 * With w_p = corr_p / 2 and the K pixels p of the coarse mask, the difference of orientation o and translation t is
 *   S(o,t) = sum_p w_p |R(o,p)|^2 + sum_p w_p |x_p|^2 - 2 sum_p Re conj(R(o,p)) A(t,p),   A(t,p) = w_p x_p shift(t,p),
 * the form of cuda_kernel_diff2_coarse_xt, whose A tables this makes with the same arithmetic. R(o,p), the projection
 * of the reference, does not depend on the particle, so a table of it (CoarseProjectionCache) replaces the K projections
 * through the texture that every particle needed for every orientation.
 *
 * cuda_kernel_cc_score: a thread owns CC_TM orientations (consecutive threads, consecutive orientations, so that the
 * table reads coalesce) and CC_TT translations in registers; the A values of KC pixels sit in shared memory, all
 * threads read the same address. The result is added to the array it finds (which holds Xi2/2).
 */

#define CC_SB 128   // threads per block
#define CC_KC 32    // pixels per staged chunk

// R[k * O_total + o_offset + o] = projection of pixel pix[k] for orientation o of one class (one launch per class)
static __global__ void cuda_kernel_cc_build(
		AccProjectorKernel projector,
		const XFLOAT *g_eulers,
		const int *g_pix,
		int K,
		int n_orient,
		size_t O_total,
		size_t o_offset,
		float2 *g_R)
{
	const int o = blockIdx.x * blockDim.x + threadIdx.x;
	if (o >= n_orient)
		return;
	const float *e = g_eulers + (size_t) o * 9;
	const float e0 = e[0], e1 = e[1], e3 = e[3], e4 = e[4], e6 = e[6], e7 = e[7];
	const float inv_imgX = 1.f / (float) projector.imgX;
	for (int k = 0; k < K; k++)
	{
		const int p = g_pix[k];
		// p / imgX without an integer division (as cuda_kernel_diff2_coarse_xt)
		int y = (int) (((float) p + 0.5f) * inv_imgX);
		const int x = p - y * projector.imgX;
		if (y > projector.maxR)
			y -= projector.imgY;
		float re = 0.f, im = 0.f;
		projector.project3Dmodel(x, y, e0, e1, e3, e4, e6, e7, re, im);
		g_R[(size_t) k * O_total + o_offset + o] = make_float2(re, im);
	}
}

// One particle: A[tile][k][CC_TT] (translations of a tile contiguous, padded with zeros) and w[k] = corr / 2 of the mask pixels.
// One thread per (k, translation), the same arithmetic as cuda_kernel_coarse_xt_prepare.
static __global__ void cuda_kernel_cc_prepare(
		const XFLOAT *trans_x,
		const XFLOAT *trans_y,
		const XFLOAT *g_real,
		const XFLOAT *g_imag,
		const XFLOAT *g_corr,
		const int *g_pix,
		float2 *g_A,
		float *g_w,
		int K,
		int translation_num,
		int tpad,
		int tt,
		int img_x,
		int img_y,
		int max_r)
{
	const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= (size_t) K * tpad)
		return;
	const int k = (int) (idx / tpad);
	const int t = (int) (idx - (size_t) k * tpad);
	const int p = g_pix[k];
	const int x = p % img_x;
	int y = p / img_x;
	if (y > max_r)
		y -= img_y;
	const float re = g_real[p], im = g_imag[p];
	const float sc = g_corr[p] / 2;
	if (t == 0)
		g_w[k] = sc;
	float2 a = make_float2(0.f, 0.f);
	if (t < translation_num)
	{
		float s, c;
		sincosf(x * trans_x[t] + y * trans_y[t], &s, &c);
		a = make_float2(sc * (c * re - s * im), sc * (c * im + s * re));
	}
	const int tile = t / tt;
	g_A[((size_t) tile * K + k) * tt + (t - tile * tt)] = a;
}

template <int TT, int TM>
static __global__ void __launch_bounds__(CC_SB)
cuda_kernel_cc_score(
		const float2 *__restrict__ g_R,
		int K,
		size_t O,
		const float2 *__restrict__ g_A,
		const float *__restrict__ g_w,
		const int *__restrict__ g_pix,
		const XFLOAT *__restrict__ g_real,
		const XFLOAT *__restrict__ g_imag,
		int translation_num,
		float *__restrict__ g_out)
{
	constexpr int SMEM_A = CC_KC * TT * 2; // floats
	constexpr int SMEM_IN = SMEM_A + CC_KC;
	constexpr int SMEM_OUT = CC_SB * (TT + 1); // row stride TT + 1: no bank conflicts
	constexpr int SMEM = SMEM_IN > SMEM_OUT ? SMEM_IN : SMEM_OUT;
	__shared__ __align__(16) float smem[SMEM];
	__shared__ float s_x2[CC_SB / 32];
	float4 *sA = (float4 *) smem;
	float *sW = smem + SMEM_A;

	const int tid = threadIdx.x;
	const int tile = blockIdx.y;
	const size_t obase = (size_t) blockIdx.x * CC_SB * TM;

	float acc[TM][TT];
	float nrm[TM];
#pragma unroll
	for (int j = 0; j < TM; j++)
	{
		nrm[j] = 0.f;
#pragma unroll
		for (int t = 0; t < TT; t++)
			acc[j][t] = 0.f;
	}

	const float2 *A_tile = g_A + (size_t) tile * K * TT;
	for (int k0 = 0; k0 < K; k0 += CC_KC)
	{
		const int kn = min(CC_KC, K - k0);
		__syncthreads();
		for (int idx = tid; idx < kn * TT / 2; idx += CC_SB)
			sA[idx] = ((const float4 *) (A_tile + (size_t) k0 * TT))[idx];
		if (tid < kn)
			sW[tid] = g_w[k0 + tid];
		__syncthreads();

#pragma unroll 4
		for (int kk = 0; kk < kn; kk++)
		{
			float2 r[TM];
#pragma unroll
			for (int j = 0; j < TM; j++)
			{
				const size_t o = obase + (size_t) j * CC_SB + tid;
				r[j] = (o < O) ? g_R[(size_t) (k0 + kk) * O + o] : make_float2(0.f, 0.f);
			}
			const float w = sW[kk];
#pragma unroll
			for (int j = 0; j < TM; j++)
				nrm[j] += w * (r[j].x * r[j].x + r[j].y * r[j].y);
#pragma unroll
			for (int t2 = 0; t2 < TT / 2; t2++)
			{
				const float4 a = sA[kk * (TT / 2) + t2];
#pragma unroll
				for (int j = 0; j < TM; j++)
				{
					acc[j][2 * t2] += r[j].x * a.x + r[j].y * a.y;
					acc[j][2 * t2 + 1] += r[j].x * a.z + r[j].y * a.w;
				}
			}
		}
	}

	// sum_p w_p |x_p|^2 over the mask, in a fixed order (the same value in every block)
	float x2 = 0.f;
	for (int k = tid; k < K; k += CC_SB)
	{
		const int p = g_pix[k];
		x2 += g_w[k] * (g_real[p] * g_real[p] + g_imag[p] * g_imag[p]);
	}
#pragma unroll
	for (int d = 16; d > 0; d >>= 1)
		x2 += __shfl_xor_sync(0xffffffffu, x2, d);
	if ((tid & 31) == 0)
		s_x2[tid >> 5] = x2;
	__syncthreads();
	x2 = 0.f;
#pragma unroll
	for (int i = 0; i < CC_SB / 32; i++)
		x2 += s_x2[i];

	const int t0 = tile * TT;
	const int tw = min(TT, translation_num - t0);
#pragma unroll
	for (int j = 0; j < TM; j++)
	{
		__syncthreads();
		const size_t o0 = obase + (size_t) j * CC_SB;
		#pragma unroll
		for (int t = 0; t < TT; t++)
			smem[tid * (TT + 1) + t] = x2 + nrm[j] - 2.f * acc[j][t];
		__syncthreads();
		if (o0 < O)
		{
			const int n_o = (int) min((size_t) CC_SB, O - o0);
			float *dst = g_out + o0 * translation_num + t0;
			for (int idx = tid; idx < n_o * tw; idx += CC_SB)
			{
				const int ol = idx / tw;
				const int tt = idx - ol * tw;
				dst[(size_t) ol * translation_num + tt] += smem[ol * (TT + 1) + tt];
			}
		}
	}
}

// Chooses between kernel shapes by measuring (per device and translation tiling): each candidate is timed on a few
// launches, the faster one is used, and the choice is checked again every 256 launches. A candidate whose launch fails is dropped.
struct CoarseShapeTuner
{
	static const int MAXC = 4, SAMPLES = 3;
	struct Pending { cudaEvent_t t0, t1; int cand; double work; };
	struct State
	{
		int ncand = 0, cands[MAXC] = {0};
		double samples[MAXC][SAMPLES];
		int count[MAXC] = {0};
		bool dead[MAXC] = {false};
		int best = -1;
		bool round = true; // measuring (the first round, and one more every 256 launches)
		long calls = 0;
		bool announced = false;
		std::vector<Pending> pending;
	};
	std::mutex mtx;
	std::map<std::pair<int, int>, State> states; // (device, tt)
	std::vector<std::pair<cudaEvent_t, cudaEvent_t> > free_events;

	static double median3(const double *v)
	{
		double a = v[0], b = v[1], c = v[2];
		return std::max(std::min(a, b), std::min(std::max(a, b), c));
	}

	void collect(State &st)
	{
		for (size_t i = 0; i < st.pending.size();)
		{
			Pending &pd = st.pending[i];
			if (cudaEventQuery(pd.t1) == cudaErrorNotReady)
			{
				i++;
				continue;
			}
			float ms = 0;
			if (cudaEventElapsedTime(&ms, pd.t0, pd.t1) == cudaSuccess && ms > 0 && st.count[pd.cand] < SAMPLES)
				st.samples[pd.cand][st.count[pd.cand]++] = 1e3 * ms / pd.work;
			cudaGetLastError();
			free_events.push_back(std::make_pair(pd.t0, pd.t1));
			st.pending.erase(st.pending.begin() + i);
		}
	}

	void decide(State &st, int device, int tt)
	{
		if (!st.round)
			return;
		for (int c = 0; c < st.ncand; c++)
			if (!st.dead[c] && st.count[c] < SAMPLES)
				return;
		int pick = -1;
		double pm = 1e300;
		for (int c = 0; c < st.ncand; c++)
		{
			if (st.dead[c]) continue;
			const double m = median3(st.samples[c]);
			if (m < pm) { pm = m; pick = c; }
		}
		if (pick < 0) pick = 0;
		if (pick != st.best || !st.announced)
		{
			fprintf(stderr, " GPU coarse cache: device %d, %d translations per tile -> %d orientations per thread (measured, ns per orientation and pixel:",
					device, tt, st.cands[pick]);
			for (int c = 0; c < st.ncand; c++)
				if (!st.dead[c]) fprintf(stderr, " %d:%.4f", st.cands[c], median3(st.samples[c]));
			fprintf(stderr, "; checked again every 256 launches)\n");
			st.best = pick;
			st.announced = true;
		}
		st.round = false;
	}

	// The candidate to launch (an index into cands) and, when it is to be timed, the events to record around it
	int begin(int device, int tt, int ncand, const int *cands, std::pair<cudaEvent_t, cudaEvent_t> &ev)
	{
		std::lock_guard<std::mutex> lock(mtx);
		State &st = states[std::make_pair(device, tt)];
		if (st.ncand == 0)
		{
			st.ncand = std::min(ncand, (int) MAXC);
			for (int c = 0; c < st.ncand; c++) st.cands[c] = cands[c];
		}
		collect(st);
		decide(st, device, tt);
		ev = std::make_pair((cudaEvent_t) NULL, (cudaEvent_t) NULL);
		st.calls++;
		if (!st.round && st.ncand > 1 && (st.calls % 256) == 0)
		{
			st.round = true;
			for (int c = 0; c < st.ncand; c++) st.count[c] = 0;
		}
		int pick = st.best >= 0 ? st.best : 0;
		if (st.round && st.ncand > 1)
		{
			// the candidate with the fewest samples (not dead) that is not already being timed too often
			int fewest = -1;
			for (int c = 0; c < st.ncand; c++)
			{
				if (st.dead[c]) continue;
				int pend = 0;
				for (auto &pd : st.pending) pend += pd.cand == c;
				if (st.count[c] + pend >= SAMPLES) continue;
				if (fewest < 0 || st.count[c] + pend < st.count[fewest]) fewest = c;
			}
			if (fewest >= 0)
			{
				pick = fewest;
				if (!free_events.empty()) { ev = free_events.back(); free_events.pop_back(); }
				else { cudaEventCreate(&ev.first); cudaEventCreate(&ev.second); }
			}
		}
		if (st.dead[pick])
		{
			for (int c = 0; c < st.ncand; c++) if (!st.dead[c]) { pick = c; break; }
		}
		return pick;
	}

	void finish(int device, int tt, std::pair<cudaEvent_t, cudaEvent_t> &ev, int cand, double work)
	{
		std::lock_guard<std::mutex> lock(mtx);
		State &st = states[std::make_pair(device, tt)];
		st.pending.push_back(Pending{ev.first, ev.second, cand, work});
	}

	void discard(std::pair<cudaEvent_t, cudaEvent_t> &ev)
	{
		std::lock_guard<std::mutex> lock(mtx);
		if (ev.first != NULL) free_events.push_back(ev);
		ev = std::make_pair((cudaEvent_t) NULL, (cudaEvent_t) NULL);
	}

	void mark_dead(int device, int tt, int cand)
	{
		std::lock_guard<std::mutex> lock(mtx);
		State &st = states[std::make_pair(device, tt)];
		st.dead[cand] = true;
		if (st.best == cand) st.best = -1;
	}
};

#endif // ACC_DOUBLE_PRECISION
#endif /* COARSE_CACHE_CUH_ */
