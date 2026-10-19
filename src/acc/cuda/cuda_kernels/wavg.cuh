#ifndef CUDA_WAVG_KERNEL_CUH_
#define CUDA_WAVG_KERNEL_CUH_

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "src/acc/acc_projector.h"
#include "src/acc/acc_projectorkernel_impl.h"
#include "src/acc/cuda/cuda_settings.h"
#include "src/acc/cuda/cuda_kernels/cuda_device_utils.cuh"

template<bool REFCTF, bool REF3D, bool DATA3D, int block_sz>
__global__ void cuda_kernel_wavg(
		XFLOAT *g_eulers,
		AccProjectorKernel projector_arg,
		unsigned image_size,
		unsigned long orientation_num,
		XFLOAT *g_img_real,
		XFLOAT *g_img_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		XFLOAT* g_weights,
		XFLOAT* g_ctfs,
		XFLOAT *g_wdiff2s_parts,
		XFLOAT *g_wdiff2s_AA,
		XFLOAT *g_wdiff2s_XA,
		unsigned long translation_num,
		XFLOAT weight_norm,
		XFLOAT significant_weight,
		XFLOAT part_scale,
		const WavgClassJob *g_classes = NULL, // class batching: one block per class and
		const int *g_class_start = NULL,      // orientation (see cuda_kernel_diff2_coarse)
		int n_classes = 0,
		int n_chunks = 1, // pixel chunks: blockIdx.y takes one of n_chunks ranges of the passes
		bool sig_list = false) // dynamic shared memory has room for translation_num ints after the eulers
{
	XFLOAT ref_real, ref_imag, img_real, img_imag, trans_real, trans_imag;

	int bid = blockIdx.x; //block ID
	int tid = threadIdx.x;

	AccProjectorKernel projector = projector_arg; // by value (registers): a pointer to the kernel parameter copies the struct to local memory in every thread
	if (n_classes > 0)
	{
		const WavgClassJob &c = g_classes[findBatchClass(g_class_start, n_classes, blockIdx.x, bid)];
		projector = c.projector;
		g_eulers = c.eulers;
		g_weights = c.weights;
		g_wdiff2s_AA = c.wdiff2s_AA;
		g_wdiff2s_XA = c.wdiff2s_XA;
	}

	extern __shared__ XFLOAT buffer[];

	unsigned pass_num(ceilfracf(image_size,block_sz)),pixel;
	XFLOAT * s_wdiff2s_parts	= &buffer[0];
	XFLOAT * s_sumXA			= &buffer[block_sz];
	XFLOAT * s_sumA2			= &buffer[2*block_sz];
	XFLOAT * s_eulers           = &buffer[3*block_sz];

	if (tid < 9)
		s_eulers[tid] = g_eulers[bid*9+tid];

	// The significant translations of this orientation, in increasing order (warp 0,
	// 32 at a time). Most orientations that reach this kernel have none above
	// significant_weight: such a block adds nothing (it only added zeros), so it ends
	// here. With sig_list, the pixel loop below visits only the listed translations, in
	// the same order, so every sum is the same as when it tested all of them.
	__shared__ int s_nsig;
	int *s_sig = (int *) &buffer[3*block_sz + 9];
	if (tid < 32)
	{
		int n = 0;
		for (unsigned long base = 0; base < translation_num; base += 32)
		{
			const unsigned long itrans = base + tid;
			const bool sig = itrans < translation_num &&
					__ldg(&g_weights[bid * translation_num + itrans]) >= significant_weight;
			const unsigned mask = __ballot_sync(0xffffffff, sig);
			if (sig && sig_list)
				s_sig[n + __popc(mask & ((1u << tid) - 1))] = (int) itrans;
			n += __popc(mask);
		}
		if (tid == 0)
			s_nsig = n;
	}
	__syncthreads();
	if (s_nsig == 0)
		return;

	// A pixel's sums over the translations are made by one thread whatever the chunks,
	// and orientations are added with atomics anyway, so chunks change no value
	const unsigned pass_begin = (unsigned) (((unsigned long) blockIdx.y * pass_num) / n_chunks);
	const unsigned pass_end   = (unsigned) (((unsigned long) (blockIdx.y + 1) * pass_num) / n_chunks);
	for (unsigned pass = pass_begin; pass < pass_end; pass++) // finish a reference proj in each block
	{
		s_wdiff2s_parts[tid] = 0.0f;
		s_sumXA[tid] = 0.0f;
		s_sumA2[tid] = 0.0f;

		pixel = pass * block_sz + tid;

		if(pixel<image_size)
		{
			int x,y,z,xy;
			if(DATA3D)
			{
				z =  floorfracf(pixel, projector.imgX*projector.imgY);
				xy = pixel % (projector.imgX*projector.imgY);
				x =             xy  % projector.imgX;
				y = floorfracf( xy,   projector.imgX);
				if (z > projector.maxR)
				{
					if (z >= projector.imgZ - projector.maxR)
						z = z - projector.imgZ;
					else
						x = projector.maxR;
				}
			}
			else
			{
				x =             pixel % projector.imgX;
				y = floorfracf( pixel , projector.imgX);
			}
			if (y > projector.maxR)
			{
				if (y >= projector.imgY - projector.maxR)
					y = y - projector.imgY;
				else
					x = projector.maxR;
			}

			if(DATA3D)
				projector.project3Dmodel(
					x,y,z,
					s_eulers[0], s_eulers[1], s_eulers[2],
					s_eulers[3], s_eulers[4], s_eulers[5],
					s_eulers[6], s_eulers[7], s_eulers[8],
					ref_real, ref_imag);
			else if(REF3D)
				projector.project3Dmodel(
					x,y,
					s_eulers[0], s_eulers[1],
					s_eulers[3], s_eulers[4],
					s_eulers[6], s_eulers[7],
					ref_real, ref_imag);
			else
				projector.project2Dmodel(
						x,y,
					s_eulers[0], s_eulers[1],
					s_eulers[3], s_eulers[4],
					ref_real, ref_imag);

			if (REFCTF)
			{
				ref_real *= __ldg(&g_ctfs[pixel]);
				ref_imag *= __ldg(&g_ctfs[pixel]);
			}
			else
			{
				ref_real *= part_scale;
				ref_imag *= part_scale;
			}

			img_real = __ldg(&g_img_real[pixel]);
			img_imag = __ldg(&g_img_imag[pixel]);

			const unsigned long n_visit = sig_list ? (unsigned long) s_nsig : translation_num;
			for (unsigned long k = 0; k < n_visit; k++)
			{
				const unsigned long itrans = sig_list ? (unsigned long) s_sig[k] : k;
				XFLOAT weight = __ldg(&g_weights[bid * translation_num + itrans]);

				if (weight >= significant_weight)
				{
					weight /= weight_norm;

					if(DATA3D)
						translatePixel(x, y, z, g_trans_x[itrans], g_trans_y[itrans], g_trans_z[itrans], img_real, img_imag, trans_real, trans_imag);
					else
						translatePixel(x, y,    g_trans_x[itrans], g_trans_y[itrans],                    img_real, img_imag, trans_real, trans_imag);

					XFLOAT diff_real = ref_real - trans_real;
					XFLOAT diff_imag = ref_imag - trans_imag;

					s_wdiff2s_parts[tid] += weight * (diff_real*diff_real + diff_imag*diff_imag);

					s_sumXA[tid] +=  weight * ( ref_real * trans_real + ref_imag * trans_imag);
					s_sumA2[tid] +=  weight * ( ref_real*ref_real  +  ref_imag*ref_imag );
				}
			}

			cuda_atomic_add(&g_wdiff2s_XA[pixel], s_sumXA[tid]);
			cuda_atomic_add(&g_wdiff2s_AA[pixel], s_sumA2[tid]);
			cuda_atomic_add(&g_wdiff2s_parts[pixel], s_wdiff2s_parts[tid]);
		}
	}
}

#ifndef ACC_DOUBLE_PRECISION
// cuda_kernel_wavg for 2D data and class batching, from the sums that do not need a
// pass over the translations for every output: with W = sum_t w_t and Y(p) = sum_t w_t X_t(p)
// over the significant translations (|X_t| = |X|),
//     sum_t w Re(ref conj(X_t))      = Re(ref conj(Y))
//     sum_t w |ref|^2                = W |ref|^2
//     sum_t w |ref - X_t|^2          = W |ref|^2 - 2 Re(ref conj(Y)) + W |X|^2
// The block lists its orientation's significant translations first (warp 0, in order), so
// a pixel's thread takes the (cos, sin) of the shift of each from the table of the particle
// (g_shift[translation * image_size + pixel]) and adds it in registers: no weight read and
// test for every translation, no sincosf, no shared-memory sums. Sums in another order:
// float noise. Dynamic shared memory: 16 + translation_num (index) + translation_num (weight) words.
template<bool REFCTF, bool REF3D, int block_sz>
__global__ void cuda_kernel_wavg_sum(
		unsigned image_size,
		XFLOAT *g_img_real,
		XFLOAT *g_img_imag,
		XFLOAT* g_ctfs,
		XFLOAT *g_wdiff2s_parts,
		unsigned long translation_num,
		XFLOAT weight_norm,
		XFLOAT significant_weight,
		XFLOAT part_scale,
		const float2 *g_shift,
		const WavgClassJob *g_classes,
		const int *g_class_start,
		int n_classes,
		int n_chunks)
{
	int bid;
	const int tid = threadIdx.x;
	const WavgClassJob &c = g_classes[findBatchClass(g_class_start, n_classes, blockIdx.x, bid)];
	const AccProjectorKernel &cp = c.projector;
	AccProjectorKernel projector = cp;

	extern __shared__ XFLOAT buffer[];
	XFLOAT *s_eulers = &buffer[0];
	int *s_idx = (int *) &buffer[16];
	XFLOAT *s_w = &buffer[16 + translation_num];
	__shared__ int s_n;
	__shared__ XFLOAT s_W;

	if (tid < 9)
		s_eulers[tid] = c.eulers[bid * 9 + tid];
	if (tid < 32)
	{
		int n = 0;
		for (unsigned long base = 0; base < translation_num; base += 32)
		{
			const unsigned long t = base + tid;
			XFLOAT w = (t < translation_num) ? __ldg(&c.weights[bid * translation_num + t]) : (XFLOAT) 0.;
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

	const unsigned pass_num = ceilfracf(image_size, block_sz);
	const unsigned pass_begin = (unsigned) (((unsigned long) blockIdx.y * pass_num) / n_chunks);
	const unsigned pass_end   = (unsigned) (((unsigned long) (blockIdx.y + 1) * pass_num) / n_chunks);
	for (unsigned pass = pass_begin; pass < pass_end; pass++)
	{
		const unsigned pixel = pass * block_sz + tid;
		if (pixel >= image_size)
			continue;
		int x = pixel % projector.imgX;
		int y = floorfracf(pixel, projector.imgX);
		if (y > projector.maxR)
		{
			if (y >= projector.imgY - projector.maxR)
				y = y - projector.imgY;
			else
				x = projector.maxR;
		}
		XFLOAT ref_real, ref_imag;
		if (REF3D)
			projector.project3Dmodel(x, y, s_eulers[0], s_eulers[1], s_eulers[3], s_eulers[4], s_eulers[6], s_eulers[7], ref_real, ref_imag);
		else
			projector.project2Dmodel(x, y, s_eulers[0], s_eulers[1], s_eulers[3], s_eulers[4], ref_real, ref_imag);
		if (REFCTF)
		{
			const XFLOAT ctf = __ldg(&g_ctfs[pixel]);
			ref_real *= ctf;
			ref_imag *= ctf;
		}
		else
		{
			ref_real *= part_scale;
			ref_imag *= part_scale;
		}
		const XFLOAT xr = __ldg(&g_img_real[pixel]), xi = __ldg(&g_img_imag[pixel]);
		XFLOAT Yr = 0, Yi = 0;
		for (int k = 0; k < n; k++)
		{
			const float2 sc = __ldg(&g_shift[(size_t) s_idx[k] * image_size + pixel]);
			const XFLOAT w = s_w[k];
			Yr += w * (sc.x * xr - sc.y * xi);
			Yi += w * (sc.x * xi + sc.y * xr);
		}
		const XFLOAT ref2 = ref_real * ref_real + ref_imag * ref_imag;
		const XFLOAT XA = ref_real * Yr + ref_imag * Yi;
		cuda_atomic_add(&c.wdiff2s_XA[pixel], XA);
		cuda_atomic_add(&c.wdiff2s_AA[pixel], W * ref2);
		cuda_atomic_add(&g_wdiff2s_parts[pixel], W * ref2 - (XFLOAT) 2. * XA + W * (xr * xr + xi * xi));
	}
}
#endif

#endif /* CUDA_WAVG_KERNEL_CUH_ */
