#ifndef CUDA_BP_KERNELS_CUH_
#define CUDA_BP_KERNELS_CUH_

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "src/acc/acc_projector.h"
#include "src/acc/acc_backprojector.h"
#include "src/acc/cuda/cuda_settings.h"
#include "src/acc/cuda/cuda_kernels/cuda_device_utils.cuh"


/*
 *   	BP KERNELS
 */

template < bool CTF_PREMULTIPLIED >
__global__ void cuda_kernel_backproject2D(
		XFLOAT *g_img_real,
		XFLOAT *g_img_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT* g_weights,
		XFLOAT* g_Minvsigma2s,
		XFLOAT* g_ctfs,
		unsigned long translation_num,
		XFLOAT significant_weight,
		XFLOAT weight_norm,
		XFLOAT *g_eulers,
		XFLOAT *g_model_real,
		XFLOAT *g_model_imag,
		XFLOAT *g_model_weight,
		int max_r,
		int max_r2,
		XFLOAT padding_factor,
		unsigned img_x,
		unsigned img_y,
		unsigned img_xy,
		unsigned mdl_x,
		int mdl_inity,
		const BPClassJob *g_classes = NULL, // class batching: one block per class and
		const int *g_class_start = NULL,    // orientation (see cuda_kernel_diff2_coarse)
		int n_classes = 0,
		bool compact = false,              // see below; needs translation_num * 8 bytes
		                                    // of dynamic shared memory
		int n_chunks = 1)                   // pixel chunks: blockIdx.y takes one of n_chunks ranges of the passes
{
	unsigned tid = threadIdx.x;
	unsigned img = blockIdx.x;
	if (n_classes > 0)
	{
		int local;
		const BPClassJob &c = g_classes[findBatchClass(g_class_start, n_classes, blockIdx.x, local)];
		img = local;
		g_eulers = c.eulers;
		g_weights = c.weights;
		g_model_real = c.model_real;
		g_model_imag = c.model_imag;
		g_model_weight = c.model_weight;
		weight_norm = c.weight_norm;
	}

	int img_y_half = img_y / 2;
	int max_r2_out = max_r2 * padding_factor * padding_factor;

	__shared__ XFLOAT s_eulers[4];

	XFLOAT minvsigma2, ctf, img_real, img_imag, Fweight, real, imag, weight;

	if (tid == 0)
		s_eulers[0] = g_eulers[img*9+0] * padding_factor;
	else if (tid == 1)
		s_eulers[1] = g_eulers[img*9+1] * padding_factor;
	else if (tid == 2)
		s_eulers[2] = g_eulers[img*9+3] * padding_factor;
	else if (tid == 3)
		s_eulers[3] = g_eulers[img*9+4] * padding_factor;

	// compact: the significant translations of this orientation, in order, are listed
	// once in shared memory (most are not significant; every pixel used to read and test
	// every weight), and an orientation without any is skipped. Each pixel sums the same
	// terms in the same order as before.
	extern __shared__ unsigned char bp_dynamic[];
	XFLOAT *s_w = (XFLOAT *) bp_dynamic;
	int *s_t = (int *) (s_w + (compact ? translation_num : 0));
	__shared__ int s_nr;
	if (compact)
	{
		for (unsigned long t = tid; t < translation_num; t += BP_2D_BLOCK_SIZE)
			s_w[t] = g_weights[img * translation_num + t];
		__syncthreads();
		if (tid == 0)
		{
			int k = 0;
			for (unsigned long t = 0; t < translation_num; t++)
				if (s_w[t] >= significant_weight)
				{
					s_w[k] = s_w[t];
					s_t[k] = (int) t;
					k++;
				}
			s_nr = k;
		}
	}

	__syncthreads();

	if (compact && s_nr == 0)
		return;

	int pixel_pass_num(ceilf((float)img_xy/(float)BP_2D_BLOCK_SIZE));

	// A pixel's sums over the translations are made by one thread whatever the chunks,
	// and the model is added to with atomics anyway, so chunks change no value
	const unsigned pass_begin = (unsigned) (((unsigned long) blockIdx.y * pixel_pass_num) / n_chunks);
	const unsigned pass_end   = (unsigned) (((unsigned long) (blockIdx.y + 1) * pixel_pass_num) / n_chunks);
	for (unsigned pass = pass_begin; pass < pass_end; pass++)
	{
		unsigned pixel = (pass * BP_2D_BLOCK_SIZE) + tid;

		if (pixel >= img_xy)
			continue;

		int x = pixel % img_x;
		int y = (int)floorf( (float)pixel / (float)img_x);

		if (y > img_y_half)
		{
			y -= img_y;
		}

		// Logical coordinates in the 3D map; a pixel projected outside the allowed circle in output
		// coordinates (--JZ, Nov. 26th 2018) adds nothing, so it is dropped before its sums are made
		XFLOAT xp = (s_eulers[0] * x + s_eulers[1] * y );
		XFLOAT yp = (s_eulers[2] * x + s_eulers[3] * y );
		if ( ( xp * xp + yp * yp ) > max_r2_out )
			continue;

		//WAVG
		minvsigma2 = __ldg(&g_Minvsigma2s[pixel]);
		ctf = __ldg(&g_ctfs[pixel]);
		img_real = __ldg(&g_img_real[pixel]);
		img_imag = __ldg(&g_img_imag[pixel]);
		Fweight = (XFLOAT) 0.0;
		real = (XFLOAT) 0.0;
		imag = (XFLOAT) 0.0;

		XFLOAT temp_real, temp_imag;

		const unsigned long nr_loop = compact ? (unsigned long) s_nr : translation_num;
		for (unsigned long k = 0; k < nr_loop; k++)
		{
			const unsigned long itrans = compact ? (unsigned long) s_t[k] : k;
			weight = compact ? s_w[k] : g_weights[img * translation_num + itrans];

			if (compact || weight >= significant_weight)
			{
				if(CTF_PREMULTIPLIED)
				{
					weight = (weight / weight_norm) * minvsigma2;
					Fweight += weight * ctf; // SHWS 13feb2020: from now on when ctf_premultiplied, the ctf array actually contains ctf^2!
				}
				else
				{
					weight = (weight / weight_norm) * ctf * minvsigma2;
					Fweight += weight * ctf;
				}

				translatePixel(x, y, g_trans_x[itrans], g_trans_y[itrans], img_real, img_imag, temp_real, temp_imag);

				real += temp_real * weight;
				imag += temp_imag * weight;

			}
		}

		if (Fweight > (XFLOAT) 0.0)
		{

			// Only asymmetric half is stored
			if (xp < 0)
			{
				// Get complex conjugated hermitian symmetry pair
				xp = -xp;
				yp = -yp;
				imag = -imag;
			}

			int x0 = floorf(xp);
			XFLOAT fx = xp - x0;
			int x1 = x0 + 1;

			int y0 = floorf(yp);
			XFLOAT fy = yp - y0;
			y0 -= mdl_inity;
			int y1 = y0 + 1;

			XFLOAT mfx = (XFLOAT) 1.0 - fx;
			XFLOAT mfy = (XFLOAT) 1.0 - fy;

			XFLOAT dd00 = mfy * mfx;
			XFLOAT dd01 = mfy *  fx;
			XFLOAT dd10 =  fy * mfx;
			XFLOAT dd11 =  fy *  fx;

			cuda_atomic_add(&g_model_real  [y0 * mdl_x + x0], dd00 * real);
			cuda_atomic_add(&g_model_imag  [y0 * mdl_x + x0], dd00 * imag);
			cuda_atomic_add(&g_model_weight[y0 * mdl_x + x0], dd00 * Fweight);

			cuda_atomic_add(&g_model_real  [y0 * mdl_x + x1], dd01 * real);
			cuda_atomic_add(&g_model_imag  [y0 * mdl_x + x1], dd01 * imag);
			cuda_atomic_add(&g_model_weight[y0 * mdl_x + x1], dd01 * Fweight);

			cuda_atomic_add(&g_model_real  [y1 * mdl_x + x0], dd10 * real);
			cuda_atomic_add(&g_model_imag  [y1 * mdl_x + x0], dd10 * imag);
			cuda_atomic_add(&g_model_weight[y1 * mdl_x + x0], dd10 * Fweight);

			cuda_atomic_add(&g_model_real  [y1 * mdl_x + x1], dd11 * real);
			cuda_atomic_add(&g_model_imag  [y1 * mdl_x + x1], dd11 * imag);
			cuda_atomic_add(&g_model_weight[y1 * mdl_x + x1], dd11 * Fweight);
		}
	}
}

__device__ inline unsigned long long gcdU(unsigned long long a, unsigned long long b)
{
	while (b) { unsigned long long t = a % b; a = b; b = t; }
	return a;
}

// The three arrays of a 3D model are either separate (stride 1) or interleaved per voxel
// (real, imag, weight adjacent: imag = real + 1, weight = real + 2, stride 3), which puts the
// eight corners of a trilinear scatter in four 24-byte runs instead of 24 separate cache lines.
__device__ __forceinline__ size_t bpModelStride(const XFLOAT *g_model_real, const XFLOAT *g_model_imag)
{
	return (g_model_imag - g_model_real == 1) ? 3 : 1;
}

template < bool DATA3D, bool CTF_PREMULTIPLIED, bool USE_SHIFT = false, bool HOIST_DIV = false >
__global__ void cuda_kernel_backproject3D(
		XFLOAT *g_img_real,
		XFLOAT *g_img_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		XFLOAT* g_weights,
		XFLOAT* g_Minvsigma2s,
		XFLOAT* g_ctfs,
		unsigned long translation_num,
		XFLOAT significant_weight,
		XFLOAT weight_norm,
		XFLOAT *g_eulers,
		XFLOAT *g_model_real,
		XFLOAT *g_model_imag,
		XFLOAT *g_model_weight,
		int max_r,
		int max_r2,
		XFLOAT padding_factor,
		unsigned img_x,
		unsigned img_y,
		unsigned img_z,
		unsigned img_xyz,
		unsigned mdl_x,
		unsigned mdl_y,
		int mdl_inity,
		int mdl_initz,
		const BPClassJob *g_classes = NULL, // class batching: one block per class and
		const int *g_class_start = NULL,    // orientation (see cuda_kernel_diff2_coarse)
		int n_classes = 0,
		int n_chunks = 1, // pixel chunks: blockIdx.y takes one of n_chunks ranges of the passes
		bool compact = false, // significant translations listed once per block in dynamic
		                      // shared memory (translation_num * 8 bytes), as in cuda_kernel_backproject2D
		bool scatter = false, // class batching: blocks running at the same time take orientations a
		                      // fixed stride apart instead of neighbours, which overlap in the volume
		                      // and so collide in the atomic additions; only the order of the sums changes
		BPAccumView acc = BPAccumView(), // per-orientation accumulation (see BPAccumView), with
		const unsigned *g_slots = NULL,  // each orientation's slot, class after class
		int tile = 0,  // 2D data only (RELION_GPU_BP_TILE=1): a warp takes a tile of 16 x 2 pixels
		               // instead of 32 consecutive ones; only the order of the atomic additions changes
		const float2 *g_shift = NULL, // USE_SHIFT: (cos, sin) of every translation's phase at every pixel of
		               // 2D data, table[translation * img_xyz + pixel] (cuda_kernel_shift_table), read instead of
		               // a sincosf per pixel and translation; float noise
		int PR = BP_PACK_RUNS) // tile == 2: runs (of 4) staged in shared memory at a time, 1, 2 or 4; fewer runs
		               // use less shared memory (more blocks per SM), the same terms are added in the same loop order within a run
{
	unsigned tid = threadIdx.x;
	unsigned img = blockIdx.x;
	unsigned long long slot_pos = 0;
	if (n_classes > 0)
	{
		int local;
		const int ci = findBatchClass(g_class_start, n_classes, blockIdx.x, local);
		const BPClassJob &c = g_classes[ci];
		img = local;
		if (scatter)
		{
			const unsigned long long cnt = g_class_start[ci + 1] - g_class_start[ci];
			unsigned long long stride = c.scatter_stride; // made on the host (runBackProjectKernelBatched)
			if (stride == 0)
			{
				stride = (unsigned long long) (cnt * 0.6180339887) | 1;
				while (stride < cnt && gcdU(stride, cnt) != 1)
					stride += 2;
			}
			if (cnt > 2)
				img = (unsigned) (((unsigned long long) local * stride) % cnt);
		}
		slot_pos = (unsigned long long) g_class_start[ci] + img;
		g_eulers = c.eulers;
		g_weights = c.weights;
		g_model_real = c.model_real;
		g_model_imag = c.model_imag;
		g_model_weight = c.model_weight;
		weight_norm = c.weight_norm;
	}
	const size_t S = bpModelStride(g_model_real, g_model_imag);
	
	int img_y_half = img_y / 2;
	int img_z_half = img_z / 2;
	
	int max_r2_vol = max_r2 * padding_factor * padding_factor;

	__shared__ XFLOAT s_eulers[9];
	XFLOAT minvsigma2, ctf, img_real, img_imag, Fweight, real, imag, weight;

	if (tid < 9)
		s_eulers[tid] = g_eulers[img*9+tid];

	// Same terms in the same order as the uncompacted loop; an orientation without any
	// significant translation is skipped.
	extern __shared__ unsigned char bp_dynamic[];
	XFLOAT *s_w = (XFLOAT *) bp_dynamic;
	int *s_t = (int *) (s_w + (compact ? translation_num : 0));
	__shared__ int s_nr;
	if (compact)
	{
		for (unsigned long t = tid; t < translation_num; t += blockDim.x)
			s_w[t] = g_weights[img * translation_num + t];
		__syncthreads();
		if (tid == 0)
		{
			int kk = 0;
			for (unsigned long t = 0; t < translation_num; t++)
				if (s_w[t] >= significant_weight)
				{
					// HOIST_DIV (only launched with the compacted list): weight / weight_norm once per translation here,
					// not per pixel in the loops; the same quotient
					s_w[kk] = HOIST_DIV ? s_w[t] / weight_norm : s_w[t];
					s_t[kk] = (int) t;
					kk++;
				}
			s_nr = kk;
		}
	}

	__syncthreads();

	if (compact && s_nr == 0)
		return;

	// The orientation's buffer in the accumulator, claimed by the first block to come here
	// (-1: add into the volume directly)
	__shared__ int s_abuf;
	if (acc.cap > 0 && g_slots != NULL)
	{
		if (tid == 0)
		{
			const unsigned slot = g_slots[slot_pos];
			volatile int *vslot = acc.slot_idx + slot;
			int idx = *vslot;
			if (idx == -1)
			{
				if (atomicCAS(acc.slot_idx + slot, -1, -2) == -1)
				{
					const int n = atomicAdd(acc.counter, 1);
					if (n < acc.cap)
					{
						for (int e = 0; e < 9; e++)
							acc.euler[(size_t) n * 9 + e] = s_eulers[e];
						acc.idx_slot[n] = slot;
						idx = n;
					}
					else
						idx = -3;
					__threadfence();
					*vslot = idx;
				}
				else
					idx = -2;
			}
			while (idx == -2)
				idx = *vslot;
			s_abuf = idx >= 0 ? idx : -1;
		}
	}
	else if (tid == 0)
		s_abuf = -1;
	__syncthreads();
	XFLOAT *a_buf = s_abuf >= 0 ? acc.acc + (size_t) s_abuf * 3 * acc.pixels : NULL;

	// tile == 2 (RELION_GPU_BP_PACK, on by default): the atomic additions of a warp are arranged so that a lane takes
	// one float of a run (two neighbouring voxels, three floats each) of one pixel instead of one voxel
	// of one component: the warp's requests land in about a fifth of the sectors. Same terms, only
	// the order of the additions changes. Images only, interleaved model, no accumulation buffer.
	if (tile == 2 && !DATA3D && a_buf == NULL && S == 3)
	{
		const unsigned lane = tid & 31, wid = tid >> 5;
		const size_t pk_off = compact ? (((size_t) translation_num * (sizeof(XFLOAT) + sizeof(int)) + 15) & ~(size_t) 15) : 0;
		XFLOAT *pk_val = (XFLOAT *) (bp_dynamic + pk_off) + (size_t) wid * (32 * 6 * PR);
		size_t *pk_base = (size_t *) (bp_dynamic + pk_off + (size_t) (BP_REF3D_BLOCK_SIZE / 32) * (32 * 6 * PR) * sizeof(XFLOAT)) + (size_t) wid * (32 * PR);
		const unsigned n_pass = (img_xyz + BP_REF3D_BLOCK_SIZE - 1) / BP_REF3D_BLOCK_SIZE;
		const unsigned pb = (unsigned) (((unsigned long) blockIdx.y * n_pass) / n_chunks);
		const unsigned pe = (unsigned) (((unsigned long) (blockIdx.y + 1) * n_pass) / n_chunks);
		const unsigned long nr_loop = compact ? (unsigned long) s_nr : translation_num;
		for (unsigned pass = pb; pass < pe; pass++)
		{
			const unsigned pixel = pass * BP_REF3D_BLOCK_SIZE + tid;
			bool go = false;
			XFLOAT xp = 0, yp = 0, zp = 0;
			if (pixel < img_xyz)
			{
				int x = pixel % img_x;
				int y = floorfracf(pixel, img_x);
				if (y > img_y_half)
					y = y - img_y;
				minvsigma2 = __ldg(&g_Minvsigma2s[pixel]);
				ctf = __ldg(&g_ctfs[pixel]);
				img_real = __ldg(&g_img_real[pixel]);
				img_imag = __ldg(&g_img_imag[pixel]);
				Fweight = (XFLOAT) 0.0;
				real = (XFLOAT) 0.0;
				imag = (XFLOAT) 0.0;
				XFLOAT temp_real, temp_imag;
				#pragma unroll 1
				for (unsigned long kk = 0; kk < nr_loop; kk++)
				{
					const unsigned long itrans = compact ? (unsigned long) s_t[kk] : kk;
					weight = compact ? s_w[kk] : g_weights[img * translation_num + itrans];
					if (compact || weight >= significant_weight)
					{
						if (CTF_PREMULTIPLIED)
						{
							weight = (HOIST_DIV ? weight : weight / weight_norm) * minvsigma2;
							Fweight += weight * ctf;
						}
						else
						{
							weight = (HOIST_DIV ? weight : weight / weight_norm) * ctf * minvsigma2;
							Fweight += weight * ctf;
						}
						if (USE_SHIFT)
						{
							const float2 cs = __ldg(&g_shift[itrans * img_xyz + pixel]);
							temp_real = cs.x * img_real - cs.y * img_imag;
							temp_imag = cs.x * img_imag + cs.y * img_real;
						}
						else
							translatePixel(x, y, g_trans_x[itrans], g_trans_y[itrans], img_real, img_imag, temp_real, temp_imag);
						real += temp_real * weight;
						imag += temp_imag * weight;
					}
				}
				if (Fweight > (XFLOAT) 0.0)
				{
					xp = (s_eulers[0] * x + s_eulers[1] * y) * padding_factor;
					yp = (s_eulers[3] * x + s_eulers[4] * y) * padding_factor;
					zp = (s_eulers[6] * x + s_eulers[7] * y) * padding_factor;
					go = !((xp * xp + yp * yp + zp * zp) > max_r2_vol);
				}
			}
			const unsigned ballot = __ballot_sync(0xffffffffu, go);
			const unsigned n_act = __popc(ballot);
			__syncwarp();
			XFLOAT fx = 0, fy = 0, fz = 0;
			int x0 = 0, y0 = 0, z0 = 0;
			unsigned pos = 0;
			if (go)
			{
				if (xp < (XFLOAT) 0.0)
				{
					xp = -xp;
					yp = -yp;
					zp = -zp;
					imag = -imag;
				}
				x0 = floorf(xp);
				fx = xp - x0;
				y0 = floorf(yp);
				fy = yp - y0;
				y0 -= mdl_inity;
				z0 = floorf(zp);
				fz = zp - z0;
				z0 -= mdl_initz;
				pos = __popc(ballot & ((1u << lane) - 1u));
			}
			// the four runs (y, z neighbours) go through the staging buffer PR at a time
			for (int rb = 0; rb < 4; rb += PR)
			{
				__syncwarp();
				if (go)
				{
					const XFLOAT mfx = (XFLOAT) 1.0 - fx, mfy = (XFLOAT) 1.0 - fy, mfz = (XFLOAT) 1.0 - fz;
					XFLOAT *v = pk_val + pos * (6 * PR);
					size_t *b = pk_base + pos * PR;
#pragma unroll
					for (int r = 0; r < PR; r++)
					{
						const int run = rb + r;
						const int zz = z0 + (run >> 1), yy = y0 + (run & 1);
						const XFLOAT wz = (run >> 1) ? fz : mfz, wy = (run & 1) ? fy : mfy;
						const XFLOAT d0 = wz * wy * mfx, d1 = wz * wy * fx;
						b[r] = (size_t) (zz * mdl_x * mdl_y + yy * mdl_x + x0) * 3;
						v[r * 6 + 0] = d0 * real;
						v[r * 6 + 1] = d0 * imag;
						v[r * 6 + 2] = d0 * Fweight;
						v[r * 6 + 3] = d1 * real;
						v[r * 6 + 4] = d1 * imag;
						v[r * 6 + 5] = d1 * Fweight;
					}
				}
				__syncwarp();
				const unsigned total = n_act * (6 * PR);
				for (unsigned k = lane; k < total; k += 32)
				{
					const unsigned p = k / (6 * PR), f = k - p * (6 * PR), r = f / 6, e = f - r * 6;
					cuda_atomic_add(&g_model_real[pk_base[p * PR + r] + e], pk_val[k]);
				}
			}
			__syncwarp();
		}
		return;
	}
	if (tile == 2)
		tile = 0;

	int pixel_pass_num(0);
	const unsigned tiles_x = (img_x + 15) / 16;
	if(DATA3D)
		pixel_pass_num = (ceilf((float)img_xyz/(float)BP_DATA3D_BLOCK_SIZE));
	else if (tile)
		pixel_pass_num = (tiles_x * 16 * ((img_y + 1) & ~1u) + BP_REF3D_BLOCK_SIZE - 1) / BP_REF3D_BLOCK_SIZE;
	else
		pixel_pass_num = (ceilf((float)img_xyz/(float)BP_REF3D_BLOCK_SIZE));

	// A pixel's sums over the translations are made by one thread whatever the chunks,
	// and the volume is added to with atomics anyway, so chunks change no value
	const unsigned pass_begin = (unsigned) (((unsigned long) blockIdx.y * pixel_pass_num) / n_chunks);
	const unsigned pass_end   = (unsigned) (((unsigned long) (blockIdx.y + 1) * pixel_pass_num) / n_chunks);
	for (unsigned pass = pass_begin; pass < pass_end; pass++)
    {
		unsigned pixel(0);
		if(DATA3D)
			pixel = (pass * BP_DATA3D_BLOCK_SIZE) + tid;
		else if (tile)
		{
			// slot -> pixel: tile t = slot / 32 is 16 columns x 2 rows, the tiles run along the rows
			const unsigned slot = (pass * BP_REF3D_BLOCK_SIZE) + tid;
			const unsigned t = slot >> 5, l = slot & 31;
			const unsigned col = (t % tiles_x) * 16 + (l & 15);
			const unsigned row = (t / tiles_x) * 2 + (l >> 4);
			if (col >= img_x || row >= img_y)
				continue;
			pixel = row * img_x + col;
		}
		else
			pixel = (pass * BP_REF3D_BLOCK_SIZE) + tid;

		if (pixel >= img_xyz)
			continue;

		int x,y,z,xy;

		if(DATA3D)
		{
			z =  floorfracf(pixel, img_x*img_y);
			xy = pixel % (img_x*img_y);
			x =             xy  % img_x;
			y = floorfracf( xy,   img_x);
			
			if (z > img_z_half)
			{
				z = z - img_z;

				if(x==0)
					continue;
			}
		}
		else
		{
			x =             pixel % img_x;
			y = floorfracf( pixel , img_x);
		}
		if (y > img_y_half)
		{
			y = y - img_y;
		}
		
		//WAVG
		minvsigma2 = __ldg(&g_Minvsigma2s[pixel]);
		ctf = __ldg(&g_ctfs[pixel]);
		img_real = __ldg(&g_img_real[pixel]);
		img_imag = __ldg(&g_img_imag[pixel]);
		Fweight = (XFLOAT) 0.0;
		real = (XFLOAT) 0.0;
		imag = (XFLOAT) 0.0;

		XFLOAT temp_real, temp_imag;
		const unsigned long nr_loop = compact ? (unsigned long) s_nr : translation_num;
		#pragma unroll 1
				for (unsigned long kk = 0; kk < nr_loop; kk++)
		{
			const unsigned long itrans = compact ? (unsigned long) s_t[kk] : kk;
			weight = compact ? s_w[kk] : g_weights[img * translation_num + itrans];

			if (compact || weight >= significant_weight)
			{
				if(CTF_PREMULTIPLIED)
				{
					weight = (HOIST_DIV ? weight : weight / weight_norm) * minvsigma2;
					Fweight += weight * ctf; // SHWS 13feb2020: from now on when ctf_premultiplied, the ctf array actually contains ctf^2!
				}
				else
				{
					weight = (HOIST_DIV ? weight : weight / weight_norm) * ctf * minvsigma2;
					Fweight += weight * ctf;
				}
				
				if(DATA3D)
					translatePixel(x, y, z, g_trans_x[itrans], g_trans_y[itrans], g_trans_z[itrans], img_real, img_imag, temp_real, temp_imag);
				else if (USE_SHIFT)
				{
					const float2 cs = __ldg(&g_shift[itrans * img_xyz + pixel]);
					temp_real = cs.x * img_real - cs.y * img_imag;
					temp_imag = cs.x * img_imag + cs.y * img_real;
				}
				else
					translatePixel(x, y,    g_trans_x[itrans], g_trans_y[itrans],                    img_real, img_imag, temp_real, temp_imag);

				real += temp_real * weight;
				imag += temp_imag * weight;
			}
		}

		//BP
		if (Fweight > (XFLOAT) 0.0 && a_buf != NULL)
		{
			// Added up over the particles; the position in the volume is found in the flush
			cuda_atomic_add(&a_buf[pixel], real);
			cuda_atomic_add(&a_buf[acc.pixels + pixel], imag);
			cuda_atomic_add(&a_buf[2 * (size_t) acc.pixels + pixel], Fweight);
		}
		else if (Fweight > (XFLOAT) 0.0)
		{
			// Get logical coordinates in the 3D map

			XFLOAT xp,yp,zp;
			if(DATA3D)
			{
				xp = (s_eulers[0] * x + s_eulers[1] * y + s_eulers[2] * z) * padding_factor;
				yp = (s_eulers[3] * x + s_eulers[4] * y + s_eulers[5] * z) * padding_factor;
				zp = (s_eulers[6] * x + s_eulers[7] * y + s_eulers[8] * z) * padding_factor;
			}
			else
			{
				xp = (s_eulers[0] * x + s_eulers[1] * y ) * padding_factor;
				yp = (s_eulers[3] * x + s_eulers[4] * y ) * padding_factor;
				zp = (s_eulers[6] * x + s_eulers[7] * y ) * padding_factor;
			}
			
			// Only consider pixels that are projected inside the sphere in output coordinates.
			//     --JZ, Oct. 18. 2018			
			if ( ( xp * xp + yp * yp  + zp * zp ) > max_r2_vol)
				continue;
		
			// Only asymmetric half is stored
			if (xp < (XFLOAT) 0.0)
			{
				// Get complex conjugated hermitian symmetry pair
				xp = -xp;
				yp = -yp;
				zp = -zp;
				imag = -imag;
			}

			int x0 = floorf(xp);
			XFLOAT fx = xp - x0;
			int x1 = x0 + 1;

			int y0 = floorf(yp);
			XFLOAT fy = yp - y0;
			y0 -= mdl_inity;
			int y1 = y0 + 1;

			int z0 = floorf(zp);
			XFLOAT fz = zp - z0;
			z0 -= mdl_initz;
			int z1 = z0 + 1;

			XFLOAT mfx = (XFLOAT)1.0 - fx;
			XFLOAT mfy = (XFLOAT)1.0 - fy;
			XFLOAT mfz = (XFLOAT)1.0 - fz;

			XFLOAT dd000 = mfz * mfy * mfx;

			cuda_atomic_add(&g_model_real  [(size_t)(z0 * mdl_x * mdl_y + y0 * mdl_x + x0) * S], dd000 * real);
			cuda_atomic_add(&g_model_imag  [(size_t)(z0 * mdl_x * mdl_y + y0 * mdl_x + x0) * S], dd000 * imag);
			cuda_atomic_add(&g_model_weight[(size_t)(z0 * mdl_x * mdl_y + y0 * mdl_x + x0) * S], dd000 * Fweight);

			XFLOAT dd001 = mfz * mfy *  fx;

			cuda_atomic_add(&g_model_real  [(size_t)(z0 * mdl_x * mdl_y + y0 * mdl_x + x1) * S], dd001 * real);
			cuda_atomic_add(&g_model_imag  [(size_t)(z0 * mdl_x * mdl_y + y0 * mdl_x + x1) * S], dd001 * imag);
			cuda_atomic_add(&g_model_weight[(size_t)(z0 * mdl_x * mdl_y + y0 * mdl_x + x1) * S], dd001 * Fweight);

			XFLOAT dd010 = mfz *  fy * mfx;

			cuda_atomic_add(&g_model_real  [(size_t)(z0 * mdl_x * mdl_y + y1 * mdl_x + x0) * S], dd010 * real);
			cuda_atomic_add(&g_model_imag  [(size_t)(z0 * mdl_x * mdl_y + y1 * mdl_x + x0) * S], dd010 * imag);
			cuda_atomic_add(&g_model_weight[(size_t)(z0 * mdl_x * mdl_y + y1 * mdl_x + x0) * S], dd010 * Fweight);

			XFLOAT dd011 = mfz *  fy *  fx;

			cuda_atomic_add(&g_model_real  [(size_t)(z0 * mdl_x * mdl_y + y1 * mdl_x + x1) * S], dd011 * real);
			cuda_atomic_add(&g_model_imag  [(size_t)(z0 * mdl_x * mdl_y + y1 * mdl_x + x1) * S], dd011 * imag);
			cuda_atomic_add(&g_model_weight[(size_t)(z0 * mdl_x * mdl_y + y1 * mdl_x + x1) * S], dd011 * Fweight);

			XFLOAT dd100 =  fz * mfy * mfx;

			cuda_atomic_add(&g_model_real  [(size_t)(z1 * mdl_x * mdl_y + y0 * mdl_x + x0) * S], dd100 * real);
			cuda_atomic_add(&g_model_imag  [(size_t)(z1 * mdl_x * mdl_y + y0 * mdl_x + x0) * S], dd100 * imag);
			cuda_atomic_add(&g_model_weight[(size_t)(z1 * mdl_x * mdl_y + y0 * mdl_x + x0) * S], dd100 * Fweight);

			XFLOAT dd101 =  fz * mfy *  fx;

			cuda_atomic_add(&g_model_real  [(size_t)(z1 * mdl_x * mdl_y + y0 * mdl_x + x1) * S], dd101 * real);
			cuda_atomic_add(&g_model_imag  [(size_t)(z1 * mdl_x * mdl_y + y0 * mdl_x + x1) * S], dd101 * imag);
			cuda_atomic_add(&g_model_weight[(size_t)(z1 * mdl_x * mdl_y + y0 * mdl_x + x1) * S], dd101 * Fweight);

			XFLOAT dd110 =  fz *  fy * mfx;

			cuda_atomic_add(&g_model_real  [(size_t)(z1 * mdl_x * mdl_y + y1 * mdl_x + x0) * S], dd110 * real);
			cuda_atomic_add(&g_model_imag  [(size_t)(z1 * mdl_x * mdl_y + y1 * mdl_x + x0) * S], dd110 * imag);
			cuda_atomic_add(&g_model_weight[(size_t)(z1 * mdl_x * mdl_y + y1 * mdl_x + x0) * S], dd110 * Fweight);

			XFLOAT dd111 =  fz *  fy *  fx;

			cuda_atomic_add(&g_model_real  [(size_t)(z1 * mdl_x * mdl_y + y1 * mdl_x + x1) * S], dd111 * real);
			cuda_atomic_add(&g_model_imag  [(size_t)(z1 * mdl_x * mdl_y + y1 * mdl_x + x1) * S], dd111 * imag);
			cuda_atomic_add(&g_model_weight[(size_t)(z1 * mdl_x * mdl_y + y1 * mdl_x + x1) * S], dd111 * Fweight);

		}
	}
}


// RELION_GPU_BP_TILE=check: compares the model after the tiled launch (m2) with the one after the
// default launch (m1), both started from m0. d = m2 - m1, r = m1 - m0 (the update of the launch).
__global__ void cuda_kernel_bp_check_diff(const float *m0, const float *m1, const float *m2, size_t n,
		unsigned *max_d, unsigned *max_r, double *sum_d2, double *sum_r2)
{
	double sd = 0, sr = 0;
	unsigned md = 0, mr = 0;
	for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
	{
		const float d = m2[i] - m1[i], r = m1[i] - m0[i];
		md = max(md, __float_as_uint(fabsf(d)));
		mr = max(mr, __float_as_uint(fabsf(r)));
		sd += (double) d * d;
		sr += (double) r * r;
	}
	atomicMax(max_d, md);
	atomicMax(max_r, mr);
	atomicAdd(sum_d2, sd);
	atomicAdd(sum_r2, sr);
}

// Spreads the accumulated sums of one orientation (one block per buffer in use) into its class's
// volume, as cuda_kernel_backproject3D does for a single particle, and clears the buffer and
// its slot for the next iteration. For images (not volumes) only.
__global__ void cuda_kernel_backproject3D_flush(
		BPAccumView acc,
		const BPModelPtrs *g_models,
		unsigned slots_per_class,
		unsigned img_x,
		unsigned img_y,
		int max_r2,
		XFLOAT padding_factor,
		unsigned mdl_x,
		unsigned mdl_y,
		int mdl_inity,
		int mdl_initz)
{
	const unsigned tid = threadIdx.x;
	const int buf = blockIdx.x;
	const unsigned slot = acc.idx_slot[buf];
	const BPModelPtrs &m = g_models[slot / slots_per_class];
	XFLOAT *g_model_real = m.real, *g_model_imag = m.imag, *g_model_weight = m.weight;
	const size_t S = bpModelStride(g_model_real, g_model_imag);
	XFLOAT *a_buf = acc.acc + (size_t) buf * 3 * acc.pixels;

	__shared__ XFLOAT s_eulers[9];
	if (tid < 9)
		s_eulers[tid] = acc.euler[(size_t) buf * 9 + tid];
	__syncthreads();

	const int img_y_half = img_y / 2;
	const int max_r2_vol = max_r2 * padding_factor * padding_factor;

	for (unsigned pixel = tid; pixel < acc.pixels; pixel += blockDim.x)
	{
		XFLOAT real = a_buf[pixel];
		XFLOAT imag = a_buf[acc.pixels + pixel];
		XFLOAT Fweight = a_buf[2 * (size_t) acc.pixels + pixel];
		if (real == (XFLOAT) 0.0 && imag == (XFLOAT) 0.0 && Fweight == (XFLOAT) 0.0)
			continue;
		a_buf[pixel] = (XFLOAT) 0.0;
		a_buf[acc.pixels + pixel] = (XFLOAT) 0.0;
		a_buf[2 * (size_t) acc.pixels + pixel] = (XFLOAT) 0.0;

		if (!(Fweight > (XFLOAT) 0.0))
			continue;

		int x = pixel % img_x;
		int y = floorfracf(pixel, img_x);
		if (y > img_y_half)
			y = y - img_y;

		XFLOAT xp = (s_eulers[0] * x + s_eulers[1] * y) * padding_factor;
		XFLOAT yp = (s_eulers[3] * x + s_eulers[4] * y) * padding_factor;
		XFLOAT zp = (s_eulers[6] * x + s_eulers[7] * y) * padding_factor;

		if ((xp * xp + yp * yp + zp * zp) > max_r2_vol)
			continue;

		if (xp < (XFLOAT) 0.0)
		{
			xp = -xp;
			yp = -yp;
			zp = -zp;
			imag = -imag;
		}

		int x0 = floorf(xp);
		XFLOAT fx = xp - x0;
		int x1 = x0 + 1;

		int y0 = floorf(yp);
		XFLOAT fy = yp - y0;
		y0 -= mdl_inity;
		int y1 = y0 + 1;

		int z0 = floorf(zp);
		XFLOAT fz = zp - z0;
		z0 -= mdl_initz;
		int z1 = z0 + 1;

		XFLOAT mfx = (XFLOAT)1.0 - fx;
		XFLOAT mfy = (XFLOAT)1.0 - fy;
		XFLOAT mfz = (XFLOAT)1.0 - fz;

		const XFLOAT dd[8] = { mfz * mfy * mfx, mfz * mfy * fx, mfz * fy * mfx, mfz * fy * fx,
		                       fz * mfy * mfx, fz * mfy * fx, fz * fy * mfx, fz * fy * fx };
		const size_t at[8] = {
				(size_t) z0 * mdl_x * mdl_y + (size_t) y0 * mdl_x + x0, (size_t) z0 * mdl_x * mdl_y + (size_t) y0 * mdl_x + x1,
				(size_t) z0 * mdl_x * mdl_y + (size_t) y1 * mdl_x + x0, (size_t) z0 * mdl_x * mdl_y + (size_t) y1 * mdl_x + x1,
				(size_t) z1 * mdl_x * mdl_y + (size_t) y0 * mdl_x + x0, (size_t) z1 * mdl_x * mdl_y + (size_t) y0 * mdl_x + x1,
				(size_t) z1 * mdl_x * mdl_y + (size_t) y1 * mdl_x + x0, (size_t) z1 * mdl_x * mdl_y + (size_t) y1 * mdl_x + x1 };
		#pragma unroll
		for (int c = 0; c < 8; c++)
		{
			cuda_atomic_add(&g_model_real  [(size_t) at[c] * S], dd[c] * real);
			cuda_atomic_add(&g_model_imag  [(size_t) at[c] * S], dd[c] * imag);
			cuda_atomic_add(&g_model_weight[(size_t) at[c] * S], dd[c] * Fweight);
		}
	}
	if (tid == 0)
		acc.slot_idx[slot] = -1;
}

template < bool DATA3D, bool CTF_PREMULTIPLIED >
__global__ void cuda_kernel_backproject3D_SGD(
		AccProjectorKernel projector,
		XFLOAT *g_img_real,
		XFLOAT *g_img_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		XFLOAT* g_weights,
		XFLOAT* g_Minvsigma2s,
		XFLOAT* g_ctfs,
		unsigned long translation_num,
		XFLOAT significant_weight,
		XFLOAT weight_norm,
		XFLOAT *g_eulers,
		XFLOAT *g_model_real,
		XFLOAT *g_model_imag,
		XFLOAT *g_model_weight,
		int max_r,
		int max_r2,
		XFLOAT padding_factor,
		unsigned img_x,
		unsigned img_y,
		unsigned img_z,
		unsigned img_xyz,
		unsigned mdl_x,
		unsigned mdl_y,
		int mdl_inity,
		int mdl_initz)
{
	const size_t S = bpModelStride(g_model_real, g_model_imag);
	unsigned tid = threadIdx.x;
	unsigned img = blockIdx.x;
	
	int img_y_half = img_y / 2;
	int img_z_half = img_z / 2;
	
	int max_r2_vol = max_r2 * padding_factor * padding_factor;

	__shared__ XFLOAT s_eulers[9];
	XFLOAT minvsigma2, ctf, img_real, img_imag, Fweight, real, imag, weight;

	if (tid < 9)
		s_eulers[tid] = g_eulers[img*9+tid];

	__syncthreads();

	int pixel_pass_num(0);
	if(DATA3D)
		pixel_pass_num = (ceilf((float)img_xyz/(float)BP_DATA3D_BLOCK_SIZE));
	else
		pixel_pass_num = (ceilf((float)img_xyz/(float)BP_REF3D_BLOCK_SIZE));

	for (unsigned pass = 0; pass < pixel_pass_num; pass++)
    {
		unsigned pixel(0);
		if(DATA3D)
			pixel = (pass * BP_DATA3D_BLOCK_SIZE) + tid;
		else
			pixel = (pass * BP_REF3D_BLOCK_SIZE) + tid;

		if (pixel >= img_xyz)
			continue;

		int x,y,z,xy;

		if(DATA3D)
		{
			z =  floorfracf(pixel, img_x*img_y);
			xy = pixel % (img_x*img_y);
			x =             xy  % img_x;
			y = floorfracf( xy,   img_x);
			
			if (z > img_z_half)
			{
				z = z - img_z;

				if(x==0)
					continue;
			}
		}
		else
		{
			x =             pixel % img_x;
			y = floorfracf( pixel , img_x);
		}
		if (y > img_y_half)
		{
			y = y - img_y;
		}

		XFLOAT ref_real = (XFLOAT) 0.0;
		XFLOAT ref_imag = (XFLOAT) 0.0;

		if(DATA3D)
			projector.project3Dmodel(
				x,y,z,
				s_eulers[0], s_eulers[1], s_eulers[2],
				s_eulers[3], s_eulers[4], s_eulers[5],
				s_eulers[6], s_eulers[7], s_eulers[8],
				ref_real, ref_imag);
		else
			projector.project3Dmodel(
				x,y,
				s_eulers[0], s_eulers[1],
				s_eulers[3], s_eulers[4],
				s_eulers[6], s_eulers[7],
				ref_real, ref_imag);

		//WAVG
		minvsigma2 = __ldg(&g_Minvsigma2s[pixel]);
		ctf = __ldg(&g_ctfs[pixel]);
		img_real = __ldg(&g_img_real[pixel]);
		img_imag = __ldg(&g_img_imag[pixel]);
		Fweight = (XFLOAT) 0.0;
		real = (XFLOAT) 0.0;
		imag = (XFLOAT) 0.0;
		ref_real *= ctf;
		ref_imag *= ctf;

		XFLOAT temp_real, temp_imag;

		for (unsigned long itrans = 0; itrans < translation_num; itrans++)
		{
			weight = g_weights[img * translation_num + itrans];

			if (weight >= significant_weight)
			{
				if(CTF_PREMULTIPLIED)
				{
					weight = (weight / weight_norm) * minvsigma2;
					Fweight += weight * ctf; // SHWS 13feb2020: from now on when ctf_premultiplied, the ctf array actually contains ctf^2!
				}
				else
				{
					weight = (weight / weight_norm) * ctf * minvsigma2;
					Fweight += weight * ctf;
				}
				if(DATA3D)
					translatePixel(x, y, z, g_trans_x[itrans], g_trans_y[itrans], g_trans_z[itrans], img_real, img_imag, temp_real, temp_imag);
				else
					translatePixel(x, y,    g_trans_x[itrans], g_trans_y[itrans],                    img_real, img_imag, temp_real, temp_imag);

				real += (temp_real-ref_real) * weight;
				imag += (temp_imag-ref_imag) * weight;
			}
		}

		//BP
		if (Fweight > (XFLOAT) 0.0)
		{
			// Get logical coordinates in the 3D map

			XFLOAT xp,yp,zp;
			if(DATA3D)
			{
				xp = (s_eulers[0] * x + s_eulers[1] * y + s_eulers[2] * z) * padding_factor;
				yp = (s_eulers[3] * x + s_eulers[4] * y + s_eulers[5] * z) * padding_factor;
				zp = (s_eulers[6] * x + s_eulers[7] * y + s_eulers[8] * z) * padding_factor;
			}
			else
			{
				xp = (s_eulers[0] * x + s_eulers[1] * y ) * padding_factor;
				yp = (s_eulers[3] * x + s_eulers[4] * y ) * padding_factor;
				zp = (s_eulers[6] * x + s_eulers[7] * y ) * padding_factor;
			}
			
			// Only consider pixels that are projected inside the sphere in output coordinates.
			//     --JZ, Nov. 26th 2018			
			if ( ( xp * xp + yp * yp  + zp * zp ) > max_r2_vol)
				continue;
			
			// Only asymmetric half is stored
			if (xp < (XFLOAT) 0.0)
			{
				// Get complex conjugated hermitian symmetry pair
				xp = -xp;
				yp = -yp;
				zp = -zp;
				imag = -imag;
			}

			int x0 = floorf(xp);
			XFLOAT fx = xp - x0;
			int x1 = x0 + 1;

			int y0 = floorf(yp);
			XFLOAT fy = yp - y0;
			y0 -= mdl_inity;
			int y1 = y0 + 1;

			int z0 = floorf(zp);
			XFLOAT fz = zp - z0;
			z0 -= mdl_initz;
			int z1 = z0 + 1;

			XFLOAT mfx = (XFLOAT)1.0 - fx;
			XFLOAT mfy = (XFLOAT)1.0 - fy;
			XFLOAT mfz = (XFLOAT)1.0 - fz;

			XFLOAT dd000 = mfz * mfy * mfx;

			cuda_atomic_add(&g_model_real  [(size_t)(z0 * mdl_x * mdl_y + y0 * mdl_x + x0) * S], dd000 * real);
			cuda_atomic_add(&g_model_imag  [(size_t)(z0 * mdl_x * mdl_y + y0 * mdl_x + x0) * S], dd000 * imag);
			cuda_atomic_add(&g_model_weight[(size_t)(z0 * mdl_x * mdl_y + y0 * mdl_x + x0) * S], dd000 * Fweight);

			XFLOAT dd001 = mfz * mfy *  fx;

			cuda_atomic_add(&g_model_real  [(size_t)(z0 * mdl_x * mdl_y + y0 * mdl_x + x1) * S], dd001 * real);
			cuda_atomic_add(&g_model_imag  [(size_t)(z0 * mdl_x * mdl_y + y0 * mdl_x + x1) * S], dd001 * imag);
			cuda_atomic_add(&g_model_weight[(size_t)(z0 * mdl_x * mdl_y + y0 * mdl_x + x1) * S], dd001 * Fweight);

			XFLOAT dd010 = mfz *  fy * mfx;

			cuda_atomic_add(&g_model_real  [(size_t)(z0 * mdl_x * mdl_y + y1 * mdl_x + x0) * S], dd010 * real);
			cuda_atomic_add(&g_model_imag  [(size_t)(z0 * mdl_x * mdl_y + y1 * mdl_x + x0) * S], dd010 * imag);
			cuda_atomic_add(&g_model_weight[(size_t)(z0 * mdl_x * mdl_y + y1 * mdl_x + x0) * S], dd010 * Fweight);

			XFLOAT dd011 = mfz *  fy *  fx;

			cuda_atomic_add(&g_model_real  [(size_t)(z0 * mdl_x * mdl_y + y1 * mdl_x + x1) * S], dd011 * real);
			cuda_atomic_add(&g_model_imag  [(size_t)(z0 * mdl_x * mdl_y + y1 * mdl_x + x1) * S], dd011 * imag);
			cuda_atomic_add(&g_model_weight[(size_t)(z0 * mdl_x * mdl_y + y1 * mdl_x + x1) * S], dd011 * Fweight);

			XFLOAT dd100 =  fz * mfy * mfx;

			cuda_atomic_add(&g_model_real  [(size_t)(z1 * mdl_x * mdl_y + y0 * mdl_x + x0) * S], dd100 * real);
			cuda_atomic_add(&g_model_imag  [(size_t)(z1 * mdl_x * mdl_y + y0 * mdl_x + x0) * S], dd100 * imag);
			cuda_atomic_add(&g_model_weight[(size_t)(z1 * mdl_x * mdl_y + y0 * mdl_x + x0) * S], dd100 * Fweight);

			XFLOAT dd101 =  fz * mfy *  fx;

			cuda_atomic_add(&g_model_real  [(size_t)(z1 * mdl_x * mdl_y + y0 * mdl_x + x1) * S], dd101 * real);
			cuda_atomic_add(&g_model_imag  [(size_t)(z1 * mdl_x * mdl_y + y0 * mdl_x + x1) * S], dd101 * imag);
			cuda_atomic_add(&g_model_weight[(size_t)(z1 * mdl_x * mdl_y + y0 * mdl_x + x1) * S], dd101 * Fweight);

			XFLOAT dd110 =  fz *  fy * mfx;

			cuda_atomic_add(&g_model_real  [(size_t)(z1 * mdl_x * mdl_y + y1 * mdl_x + x0) * S], dd110 * real);
			cuda_atomic_add(&g_model_imag  [(size_t)(z1 * mdl_x * mdl_y + y1 * mdl_x + x0) * S], dd110 * imag);
			cuda_atomic_add(&g_model_weight[(size_t)(z1 * mdl_x * mdl_y + y1 * mdl_x + x0) * S], dd110 * Fweight);

			XFLOAT dd111 =  fz *  fy *  fx;

			cuda_atomic_add(&g_model_real  [(size_t)(z1 * mdl_x * mdl_y + y1 * mdl_x + x1) * S], dd111 * real);
			cuda_atomic_add(&g_model_imag  [(size_t)(z1 * mdl_x * mdl_y + y1 * mdl_x + x1) * S], dd111 * imag);
			cuda_atomic_add(&g_model_weight[(size_t)(z1 * mdl_x * mdl_y + y1 * mdl_x + x1) * S], dd111 * Fweight);

		}
	}
}


template < bool CTF_PREMULTIPLIED >
__global__ void cuda_kernel_backproject2D_SGD(
		AccProjectorKernel projector_arg,
		XFLOAT *g_img_real,
		XFLOAT *g_img_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT* g_weights,
		XFLOAT* g_Minvsigma2s,
		XFLOAT* g_ctfs,
		unsigned long translation_num,
		XFLOAT significant_weight,
		XFLOAT weight_norm,
		XFLOAT *g_eulers,
		XFLOAT *g_model_real,
		XFLOAT *g_model_imag,
		XFLOAT *g_model_weight,
		int max_r,
		int max_r2,
		XFLOAT padding_factor,
		unsigned img_x,
		unsigned img_y,
		unsigned img_xy,
		unsigned mdl_x,
		int mdl_inity,
		const BPSgdClassJob *g_classes = NULL, // class batching: one block per class and
		const int *g_class_start = NULL,       // orientation (see cuda_kernel_diff2_coarse)
		int n_classes = 0)
{
	unsigned tid = threadIdx.x;
	unsigned img = blockIdx.x;

	AccProjectorKernel *pp = &projector_arg; // its methods are not const
	if (n_classes > 0)
	{
		int local;
		const BPSgdClassJob &c = g_classes[findBatchClass(g_class_start, n_classes, blockIdx.x, local)];
		img = local;
		pp = const_cast<AccProjectorKernel *>(&c.projector);
		g_eulers = c.eulers;
		g_weights = c.weights;
		g_model_real = c.model_real;
		g_model_imag = c.model_imag;
		g_model_weight = c.model_weight;
		weight_norm = c.weight_norm;
	}
	AccProjectorKernel &projector = *pp;

	int img_y_half = img_y / 2;
	int max_r2_out = max_r2 * padding_factor * padding_factor;

	__shared__ XFLOAT s_eulers[4];

	XFLOAT minvsigma2, ctf, img_real, img_imag, Fweight, real, imag, weight;

	if (tid == 0)
		s_eulers[0] = g_eulers[img*9+0];
	else if (tid == 1)
		s_eulers[1] = g_eulers[img*9+1];
	else if (tid == 2)
		s_eulers[2] = g_eulers[img*9+3];
	else if (tid == 3)
		s_eulers[3] = g_eulers[img*9+4];

	// Most orientations have no significant weight and add nothing: skip them at once
	// instead of one pixel at a time (as in cuda_kernel_wavg); also the barrier for s_eulers
	{
		int any = 0;
		for (unsigned long itrans = tid; itrans < translation_num; itrans += BP_2D_BLOCK_SIZE)
			any |= (g_weights[img * translation_num + itrans] >= significant_weight);
		if (!__syncthreads_or(any))
			return;
	}

	int pixel_pass_num(ceilf((float)img_xy/(float)BP_2D_BLOCK_SIZE));

	for (unsigned pass = 0; pass < pixel_pass_num; pass++)
	{
		unsigned pixel = (pass * BP_2D_BLOCK_SIZE) + tid;

		if (pixel >= img_xy)
			continue;

		int x = pixel % img_x;
		int y = (int)floorf( (float)pixel / (float)img_x);

		if (y > img_y_half)
		{
			y -= img_y;
		}

		//WAVG
		minvsigma2 = __ldg(&g_Minvsigma2s[pixel]);
		ctf = __ldg(&g_ctfs[pixel]);
		img_real = __ldg(&g_img_real[pixel]);
		img_imag = __ldg(&g_img_imag[pixel]);
		Fweight = (XFLOAT) 0.0;
		real = (XFLOAT) 0.0;
		imag = (XFLOAT) 0.0;

		XFLOAT temp_real, temp_imag;

		XFLOAT ref_real = (XFLOAT) 0.0;
		XFLOAT ref_imag = (XFLOAT) 0.0;

		projector.project2Dmodel(
			x,y,
			s_eulers[0], s_eulers[1],
			s_eulers[2], s_eulers[3],
			ref_real, ref_imag
		);
		ref_real *= ctf;
		ref_imag *= ctf;

		for (unsigned long itrans = 0; itrans < translation_num; itrans++)
		{
			weight = g_weights[img * translation_num + itrans];

			if (weight >= significant_weight)
			{
				if(CTF_PREMULTIPLIED)
				{
					weight = (weight / weight_norm) * minvsigma2;
					Fweight += weight * ctf; // SHWS 13feb2020: from now on when ctf_premultiplied, the ctf array actually contains ctf^2!
				}
				else
				{
					weight = (weight / weight_norm) * ctf * minvsigma2;
					Fweight += weight * ctf;
				}

				translatePixel(x, y, g_trans_x[itrans], g_trans_y[itrans], img_real, img_imag, temp_real, temp_imag);

				real += (temp_real-ref_real) * weight;
				imag += (temp_imag-ref_imag) * weight;
			}
		}

		if (Fweight > (XFLOAT) 0.0)
		{

			// Get logical coordinates in the 3D map
			XFLOAT xp = (s_eulers[0] * x + s_eulers[1] * y ) * padding_factor;
			XFLOAT yp = (s_eulers[2] * x + s_eulers[3] * y ) * padding_factor;

			// Only consider pixels that are projected inside the allowed circle in output coordinates.
			//     --JZ, Nov. 26th 2018
			if ( ( xp * xp + yp * yp ) > max_r2_out )
				continue;

			// Only asymmetric half is stored
			if (xp < 0)
			{
				// Get complex conjugated hermitian symmetry pair
				xp = -xp;
				yp = -yp;
				imag = -imag;
			}

			int x0 = floorf(xp);
			XFLOAT fx = xp - x0;
			int x1 = x0 + 1;

			int y0 = floorf(yp);
			XFLOAT fy = yp - y0;
			y0 -= mdl_inity;
			int y1 = y0 + 1;

			XFLOAT mfx = (XFLOAT) 1.0 - fx;
			XFLOAT mfy = (XFLOAT) 1.0 - fy;

			XFLOAT dd00 = mfy * mfx;
			XFLOAT dd01 = mfy *  fx;
			XFLOAT dd10 =  fy * mfx;
			XFLOAT dd11 =  fy *  fx;

			cuda_atomic_add(&g_model_real  [y0 * mdl_x + x0], dd00 * real);
			cuda_atomic_add(&g_model_imag  [y0 * mdl_x + x0], dd00 * imag);
			cuda_atomic_add(&g_model_weight[y0 * mdl_x + x0], dd00 * Fweight);

			cuda_atomic_add(&g_model_real  [y0 * mdl_x + x1], dd01 * real);
			cuda_atomic_add(&g_model_imag  [y0 * mdl_x + x1], dd01 * imag);
			cuda_atomic_add(&g_model_weight[y0 * mdl_x + x1], dd01 * Fweight);

			cuda_atomic_add(&g_model_real  [y1 * mdl_x + x0], dd10 * real);
			cuda_atomic_add(&g_model_imag  [y1 * mdl_x + x0], dd10 * imag);
			cuda_atomic_add(&g_model_weight[y1 * mdl_x + x0], dd10 * Fweight);

			cuda_atomic_add(&g_model_real  [y1 * mdl_x + x1], dd11 * real);
			cuda_atomic_add(&g_model_imag  [y1 * mdl_x + x1], dd11 * imag);
			cuda_atomic_add(&g_model_weight[y1 * mdl_x + x1], dd11 * Fweight);
		}
	}
}

#ifndef ACC_DOUBLE_PRECISION
// cuda_kernel_backproject2D_SGD (class batching) from the block's list of significant translations
// and the weighted sum of the shifted image, as cuda_kernel_wavg_sum does for the weighted averages:
// with w'_t = f w_t / weight_norm (f = minvsigma2, times ctf unless premultiplied),
//     Fweight = sum_t w'_t ctf = f ctf W,   real/imag = sum_t w'_t (X_t - ref) = f (Y - W ref)
// where W = sum_t w_t / weight_norm and Y = sum_t w_t / weight_norm X_t. g_shift is the table of the
// particle's shifts [translation * img_xy + pixel] (valid when the image size is even and max_r is
// half of it, where this kernel and cuda_kernel_shift_table number the pixels alike). Float noise.
// Dynamic shared memory: 16 + 2 * translation_num words.
template < bool CTF_PREMULTIPLIED >
__global__ void cuda_kernel_backproject2D_SGD_sum(
		AccProjectorKernel projector_arg,
		XFLOAT *g_img_real,
		XFLOAT *g_img_imag,
		XFLOAT* g_Minvsigma2s,
		XFLOAT* g_ctfs,
		unsigned long translation_num,
		XFLOAT significant_weight,
		int max_r2,
		XFLOAT padding_factor,
		unsigned img_x,
		unsigned img_y,
		unsigned img_xy,
		unsigned mdl_x,
		int mdl_inity,
		const float2 *g_shift,
		const BPSgdClassJob *g_classes,
		const int *g_class_start,
		int n_classes)
{
	const unsigned tid = threadIdx.x;
	int img;
	const BPSgdClassJob &c = g_classes[findBatchClass(g_class_start, n_classes, blockIdx.x, img)];
	AccProjectorKernel &projector = *const_cast<AccProjectorKernel *>(&c.projector);
	const XFLOAT weight_norm = c.weight_norm;
	const int img_y_half = img_y / 2;
	const int max_r2_out = max_r2 * padding_factor * padding_factor;

	extern __shared__ XFLOAT buffer[];
	XFLOAT *s_eulers = &buffer[0];
	int *s_idx = (int *) &buffer[16];
	XFLOAT *s_w = &buffer[16 + translation_num];
	__shared__ int s_n;
	__shared__ XFLOAT s_W;

	if (tid == 0)
		s_eulers[0] = c.eulers[img*9+0];
	else if (tid == 1)
		s_eulers[1] = c.eulers[img*9+1];
	else if (tid == 2)
		s_eulers[2] = c.eulers[img*9+3];
	else if (tid == 3)
		s_eulers[3] = c.eulers[img*9+4];
	if (tid < 32)
	{
		int n = 0;
		for (unsigned long base = 0; base < translation_num; base += 32)
		{
			const unsigned long t = base + tid;
			XFLOAT w = (t < translation_num) ? c.weights[img * translation_num + t] : (XFLOAT) 0.;
			const bool sig = (t < translation_num) && (w >= significant_weight);
			const unsigned m = __ballot_sync(0xffffffffu, sig);
			if (sig)
			{
				const int pos = n + __popc(m & ((1u << tid) - 1));
				s_idx[pos] = (int) t;
				s_w[pos] = w / weight_norm;
			}
			n += __popc(m);
		}
		if (tid == 0)
			s_n = n;
	}
	__syncthreads();
	const int n = s_n;
	if (n == 0)
		return;
	if (tid == 0)
	{
		XFLOAT W = (XFLOAT) 0.;
		for (int k = 0; k < n; k++)
			W += s_w[k];
		s_W = W;
	}
	__syncthreads();
	const XFLOAT W = s_W;

	const int pixel_pass_num(ceilf((float)img_xy/(float)BP_2D_BLOCK_SIZE));
	for (int pass = 0; pass < pixel_pass_num; pass++)
	{
		const unsigned pixel = (pass * BP_2D_BLOCK_SIZE) + tid;
		if (pixel >= img_xy)
			continue;
		const int x = pixel % img_x;
		int y = (int)floorf( (float)pixel / (float)img_x);
		if (y > img_y_half)
			y -= img_y;

		const XFLOAT minvsigma2 = __ldg(&g_Minvsigma2s[pixel]);
		const XFLOAT ctf = __ldg(&g_ctfs[pixel]);
		const XFLOAT xr = __ldg(&g_img_real[pixel]), xi = __ldg(&g_img_imag[pixel]);
		XFLOAT ref_real = (XFLOAT) 0.0, ref_imag = (XFLOAT) 0.0;
		projector.project2Dmodel(x, y, s_eulers[0], s_eulers[1], s_eulers[2], s_eulers[3], ref_real, ref_imag);
		ref_real *= ctf;
		ref_imag *= ctf;

		XFLOAT Yr = 0, Yi = 0;
		for (int k = 0; k < n; k++)
		{
			const float2 sc = __ldg(&g_shift[(size_t) s_idx[k] * img_xy + pixel]);
			const XFLOAT w = s_w[k];
			Yr += w * (sc.x * xr - sc.y * xi);
			Yi += w * (sc.x * xi + sc.y * xr);
		}
		const XFLOAT f = CTF_PREMULTIPLIED ? minvsigma2 : ctf * minvsigma2;
		const XFLOAT Fweight = f * ctf * W;
		XFLOAT real = f * (Yr - W * ref_real);
		XFLOAT imag = f * (Yi - W * ref_imag);

		if (Fweight > (XFLOAT) 0.0)
		{
			XFLOAT xp = (s_eulers[0] * x + s_eulers[1] * y ) * padding_factor;
			XFLOAT yp = (s_eulers[2] * x + s_eulers[3] * y ) * padding_factor;
			if ( ( xp * xp + yp * yp ) > max_r2_out )
				continue;
			if (xp < 0)
			{
				xp = -xp;
				yp = -yp;
				imag = -imag;
			}
			const int x0 = floorf(xp);
			const XFLOAT fx = xp - x0;
			const int x1 = x0 + 1;
			int y0 = floorf(yp);
			const XFLOAT fy = yp - y0;
			y0 -= mdl_inity;
			const int y1 = y0 + 1;
			const XFLOAT mfx = (XFLOAT) 1.0 - fx;
			const XFLOAT mfy = (XFLOAT) 1.0 - fy;
			const XFLOAT dd00 = mfy * mfx, dd01 = mfy * fx, dd10 = fy * mfx, dd11 = fy * fx;
			cuda_atomic_add(&c.model_real  [y0 * mdl_x + x0], dd00 * real);
			cuda_atomic_add(&c.model_imag  [y0 * mdl_x + x0], dd00 * imag);
			cuda_atomic_add(&c.model_weight[y0 * mdl_x + x0], dd00 * Fweight);
			cuda_atomic_add(&c.model_real  [y0 * mdl_x + x1], dd01 * real);
			cuda_atomic_add(&c.model_imag  [y0 * mdl_x + x1], dd01 * imag);
			cuda_atomic_add(&c.model_weight[y0 * mdl_x + x1], dd01 * Fweight);
			cuda_atomic_add(&c.model_real  [y1 * mdl_x + x0], dd10 * real);
			cuda_atomic_add(&c.model_imag  [y1 * mdl_x + x0], dd10 * imag);
			cuda_atomic_add(&c.model_weight[y1 * mdl_x + x0], dd10 * Fweight);
			cuda_atomic_add(&c.model_real  [y1 * mdl_x + x1], dd11 * real);
			cuda_atomic_add(&c.model_imag  [y1 * mdl_x + x1], dd11 * imag);
			cuda_atomic_add(&c.model_weight[y1 * mdl_x + x1], dd11 * Fweight);
		}
	}
}
#endif

#endif /* CUDA_PB_KERNELS_CUH_ */
