#ifndef CUDA_SHARED_SUMS_CUH_
#define CUDA_SHARED_SUMS_CUH_

// Weighted sums and 3D back-projection of 2D images against a 3D reference in one kernel
// (opt-in, RELION_GPU_SHARED_SUMS=1; see runSharedSumsKernelBatched).
//
// cuda_kernel_wavg and cuda_kernel_backproject3D each loop, for every orientation and
// pixel, over all translations: they shift the particle image (a sincosf per pixel and
// translation, the same for every orientation) and add it up with the weight of the
// (orientation, translation) pair. The sums are the same up to the image they use (the
// masked one for the weighted sums, the unmasked one for the back-projection):
//
//     Y_o(p) = sum_t w_{o,t} X_t(p)        W_o = sum_t w_{o,t}
//
// (X_t = X shifted by translation t; translations only change the phase, so |X_t| = |X|).
// The three outputs of cuda_kernel_wavg follow from Y_o of the masked image and W_o:
//     sum_t w Re(ref conj(X_t)) = Re(ref conj(Y_o))
//     sum_t w |ref|^2           = W_o |ref|^2
//     sum_t w |ref - X_t|^2     = W_o |ref|^2 - 2 Re(ref conj(Y_o)) + W_o |X|^2
// and the back-projection adds ctf*Minvsigma2*Y_o of the unmasked image with weight
// ctf^2*Minvsigma2*W_o. So here
//  - the table of cos/sin of the shifts of a particle is made once (cuda_kernel_shift_table),
//  - a block takes G orientations of a class and a tile of pixels, loops over the
//    translations once (one table load and the two complex products of the pixel for
//    all G orientations), and adds the weights of its G orientations from shared memory,
//  - then the weighted-sum outputs (added up over the G orientations: one atomic per
//    pixel and output instead of one per orientation) and the back-projection of its
//    orientations are made as in the two kernels.
// Sums are made in a different order and with |X_t|^2 = |X|^2: float noise.

#ifndef ACC_DOUBLE_PRECISION

#include <cuda_runtime.h>
#include "src/acc/acc_projector.h"
#include "src/acc/acc_projectorkernel_impl.h"
#include "src/acc/cuda/cuda_settings.h"
#include "src/acc/cuda/cuda_kernels/cuda_device_utils.cuh"

#define SHSUM_THREADS 128

// (cos, sin) of the phase of every (translation, pixel), as translatePixel makes them:
// table[t * image_size + pixel]. Pixel coordinates as in cuda_kernel_wavg (2D data).
static __global__ void cuda_kernel_shift_table(
		const XFLOAT *g_trans_x,
		const XFLOAT *g_trans_y,
		float2 *g_table,
		unsigned image_size,
		int img_x,
		int img_y,
		int max_r)
{
	const unsigned pixel = blockIdx.x * blockDim.x + threadIdx.x;
	if (pixel >= image_size)
		return;
	const int t = blockIdx.y;
	int x = pixel % img_x;
	int y = floorfracf(pixel, img_x);
	if (y > max_r)
	{
		if (y >= img_y - max_r)
			y = y - img_y;
		else
			x = max_r;
	}
	XFLOAT s, c;
	sincosf( x * g_trans_x[t] + y * g_trans_y[t], &s, &c );
	g_table[(size_t) t * image_size + pixel] = make_float2(c, s);
}

// Blocks: x = groups of G orientations of all classes (class table as for the other
// class-batched kernels, with groups instead of orientations), y = tiles of
// SHSUM_THREADS pixels. Dynamic shared memory: G * translation_num floats.
template < int G, bool REFCTF, bool CTF_PREMULTIPLIED >
__global__ void cuda_kernel_shared_sums(
		const float2 *g_table,
		const XFLOAT *g_img_real,       // masked image (weighted sums)
		const XFLOAT *g_img_imag,
		const XFLOAT *g_img_real_nomask, // unmasked image (back-projection)
		const XFLOAT *g_img_imag_nomask,
		const XFLOAT *g_ctfs,
		const XFLOAT *g_Minvsigma2s,
		XFLOAT *g_wdiff2s_parts,
		unsigned long translation_num,
		XFLOAT weight_norm,
		XFLOAT significant_weight,
		XFLOAT part_scale,
		unsigned image_size,
		int img_x,
		int img_y,
		int max_r,            // of the projector (= xdim - 1)
		int bp_max_r2,
		XFLOAT padding_factor,
		unsigned mdl_x,
		unsigned mdl_y,
		int mdl_inity,
		int mdl_initz,
		const SharedSumsClassJob *g_classes,
		const int *g_class_start,
		int n_classes,
		int parts) // 1: weighted-sum outputs, 2: back-projection, 3: both
{
	const unsigned tid = threadIdx.x;
	int local;
	const SharedSumsClassJob &c = g_classes[findBatchClass(g_class_start, n_classes, blockIdx.x, local)];
	AccProjectorKernel *pp = const_cast<AccProjectorKernel *>(&c.projector);
	AccProjectorKernel &projector = *pp;
	const int o0 = local * G;
	const int ng = min(G, c.n_orient - o0);

	extern __shared__ XFLOAT s_w[]; // [G][translation_num], 0 for insignificant translations
	__shared__ XFLOAT s_eul[G * 9];
	__shared__ XFLOAT s_W[G];
	__shared__ int s_any;

	if (tid == 0)
		s_any = 0;
	if (tid < ng * 9)
		s_eul[tid] = c.eulers[(size_t) o0 * 9 + tid];
	__syncthreads();
	for (unsigned long i = tid; i < (unsigned long) ng * translation_num; i += SHSUM_THREADS)
	{
		XFLOAT w = c.weights[(size_t) o0 * translation_num + i];
		if (w >= significant_weight)
		{
			s_w[i] = w / weight_norm;
			s_any = 1;
		}
		else
			s_w[i] = (XFLOAT) 0.0;
	}
	__syncthreads();
	if (!s_any)
		return;
	if (tid < ng)
	{
		XFLOAT W = (XFLOAT) 0.0;
		for (unsigned long t = 0; t < translation_num; t++)
			W += s_w[tid * translation_num + t];
		s_W[tid] = W;
	}
	__syncthreads();

	const unsigned pixel = blockIdx.y * SHSUM_THREADS + tid;
	if (pixel >= image_size)
		return;

	int x = pixel % img_x;
	int y = floorfracf(pixel, img_x);
	if (y > max_r)
	{
		if (y >= img_y - max_r)
			y = y - img_y;
		else
			x = max_r;
	}

	const XFLOAT xmr = __ldg(&g_img_real[pixel]),        xmi = __ldg(&g_img_imag[pixel]);
	const XFLOAT xnr = __ldg(&g_img_real_nomask[pixel]), xni = __ldg(&g_img_imag_nomask[pixel]);

	XFLOAT Ymr[G], Ymi[G], Ynr[G], Yni[G];
	#pragma unroll
	for (int g = 0; g < G; g++)
		Ymr[g] = Ymi[g] = Ynr[g] = Yni[g] = (XFLOAT) 0.0;

	for (unsigned long t = 0; t < translation_num; t++)
	{
		const float2 sc = __ldg(&g_table[t * image_size + pixel]);
		const XFLOAT zmr = sc.x * xmr - sc.y * xmi, zmi = sc.x * xmi + sc.y * xmr;
		const XFLOAT znr = sc.x * xnr - sc.y * xni, zni = sc.x * xni + sc.y * xnr;
		#pragma unroll
		for (int g = 0; g < G; g++)
		{
			const XFLOAT w = s_w[g * translation_num + t];
			Ymr[g] += w * zmr;
			Ymi[g] += w * zmi;
			Ynr[g] += w * znr;
			Yni[g] += w * zni;
		}
	}

	const XFLOAT ctf = __ldg(&g_ctfs[pixel]);
	const XFLOAT minvsigma2 = __ldg(&g_Minvsigma2s[pixel]);
	XFLOAT sXA = 0, sAA = 0, sParts = 0;
	const int bp_max_r2_vol = bp_max_r2 * padding_factor * padding_factor;

	#pragma unroll
	for (int g = 0; g < G; g++)
	{
		if (g >= ng)
			break;
		const XFLOAT W = s_W[g];
		if (W == (XFLOAT) 0.0)
			continue;
		const XFLOAT *e = &s_eul[g * 9];

		// weighted sums
		XFLOAT ref_real, ref_imag;
		projector.project3Dmodel(x, y, e[0], e[1], e[3], e[4], e[6], e[7], ref_real, ref_imag);
		if (REFCTF)
		{
			ref_real *= ctf;
			ref_imag *= ctf;
		}
		else
		{
			ref_real *= part_scale;
			ref_imag *= part_scale;
		}
		const XFLOAT ref2 = ref_real * ref_real + ref_imag * ref_imag;
		const XFLOAT XA = ref_real * Ymr[g] + ref_imag * Ymi[g];
		sXA += XA;
		sAA += W * ref2;
		sParts += W * ref2 - (XFLOAT) 2.0 * XA + W * (xmr * xmr + xmi * xmi);

		// back-projection
		const XFLOAT f = CTF_PREMULTIPLIED ? minvsigma2 : ctf * minvsigma2;
		const XFLOAT Fweight = f * ctf * W;
		if ((parts & 2) && Fweight > (XFLOAT) 0.0)
		{
			XFLOAT real = f * Ynr[g];
			XFLOAT imag = f * Yni[g];
			XFLOAT xp = (e[0] * x + e[1] * y) * padding_factor;
			XFLOAT yp = (e[3] * x + e[4] * y) * padding_factor;
			XFLOAT zp = (e[6] * x + e[7] * y) * padding_factor;
			if ( ( xp * xp + yp * yp + zp * zp ) > bp_max_r2_vol)
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
			                       fz * mfy * mfx,  fz * mfy * fx,  fz * fy * mfx,  fz * fy * fx };
			#pragma unroll
			for (int k = 0; k < 8; k++)
			{
				const size_t idx = (size_t) ((k & 4) ? z1 : z0) * mdl_x * mdl_y
				                 + (size_t) ((k & 2) ? y1 : y0) * mdl_x + ((k & 1) ? x1 : x0);
				cuda_atomic_add(&c.model_real  [idx], dd[k] * real);
				cuda_atomic_add(&c.model_imag  [idx], dd[k] * imag);
				cuda_atomic_add(&c.model_weight[idx], dd[k] * Fweight);
			}
		}
	}

	if (parts & 1)
	{
		cuda_atomic_add(&c.wdiff2s_XA[pixel], sXA);
		cuda_atomic_add(&c.wdiff2s_AA[pixel], sAA);
		cuda_atomic_add(&g_wdiff2s_parts[pixel], sParts);
	}
}

#endif // ACC_DOUBLE_PRECISION
#endif /* CUDA_SHARED_SUMS_CUH_ */
