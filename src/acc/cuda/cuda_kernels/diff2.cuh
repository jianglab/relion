#ifndef CUDA_DIFF2_KERNELS_CUH_
#define CUDA_DIFF2_KERNELS_CUH_

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "src/acc/acc_projector.h"
#include "src/acc/acc_projectorkernel_impl.h"
#include "src/acc/cuda/cuda_settings.h"
#include "src/acc/cuda/cuda_kernels/cuda_device_utils.cuh"


/*
 *   	DIFFERNECE-BASED KERNELS
 */

/*
 * Class batching: one launch for several classes. A block finds its class from
 * the first block of each class (g_class_start, n_classes entries), and takes
 * that class's projector, eulers and output from g_classes; a launch covers either
 * the whole blocks of eulers_per_block orientations or (rest_blocks) the remaining
 * orientations, one per block. With n_classes == 0 a launch is for one class, as
 * before. A block computes exactly what it computed in a per-class launch, so
 * results do not change, and the tables take a few dozen bytes per class
 * whatever the number of orientations. (CoarseClassJob: see
 * acc_projectorkernel_impl.h)
 *
 * Pixel chunks (class batching only, g_partial != NULL): blockIdx.y picks one of
 * n_chunks ranges of the pixel passes, so a particle gets n_chunks times as many
 * blocks. A block adds up its threads' sums for each translation in a fixed order
 * and stores them in g_partial [chunk][orientation of all classes][translation]
 * (chunk_stride elements per chunk); cuda_kernel_diff2_coarse_sum_chunks then adds
 * the chunks in order to diff2s. No atomics, so the result is deterministic and
 * differs from the unchunked kernel only in the order of the sums.
 */

#ifndef ACC_DOUBLE_PRECISION
// (cos, sin) of the phase translatePixel applies to every (pixel, translation) of 2D data,
// table[pixel * translation_num + t], for the coarse differences
static __global__ void cuda_kernel_shift_table_pt(
		const XFLOAT *trans_x,
		const XFLOAT *trans_y,
		float2 *table,
		int translation_num,
		int image_size,
		int img_x,
		int img_y,
		int max_r)
{
	const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
	if (pixel >= image_size)
		return;
	const int x = pixel % img_x;
	int y = floorfracf(pixel, img_x);
	if (y > max_r)
		y -= img_y;
	for (int t = 0; t < translation_num; t++)
	{
		XFLOAT s, c;
		sincosf( x * trans_x[t] + y * trans_y[t], &s, &c );
		table[(size_t) pixel * translation_num + t] = make_float2(c, s);
	}
}
#endif

/*
 * Assuming block_sz % prefetch_fraction == 0 and prefetch_fraction < block_sz
 * Assuming block_sz % eulers_per_block == 0
 * Assuming eulers_per_block * 3 < block_sz
 */
// Particle batching (prototype, 2D data, pixel chunks): blockIdx.z selects a particle; everything that belongs to a
// particle (images, translations, shift table, class table, partial sums) comes from this record; the blocks of a
// particle beyond its n_blocks are idle (particles have different numbers of orientations)
struct CoarseParticle
{
	XFLOAT *trans_x;
	XFLOAT *trans_y;
	XFLOAT *g_real;
	XFLOAT *g_imag;
	XFLOAT *g_corr;
	const float2 *shift;
	const CoarseClassJob *classes;
	const int *class_start;
	XFLOAT *partial;
	int n_blocks;
	int n_classes;
	int n_chunks;
	size_t chunk_stride;
};

template<bool REF3D, bool DATA3D, int block_sz, int eulers_per_block, int prefetch_fraction>
__global__ void cuda_kernel_diff2_coarse(
		XFLOAT *g_eulers,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT *g_real,
		XFLOAT *g_imag,
		AccProjectorKernel projector_arg,
		XFLOAT *g_corr,
		XFLOAT *g_diff2s,
		int translation_num,
		int image_size,
		const CoarseClassJob *g_classes,
		const int *g_class_start,
		int n_classes,
		bool rest_blocks,
		XFLOAT *g_partial = NULL,
		int n_chunks = 1,
		size_t chunk_stride = 0,
		const float2 *g_shift = NULL, // (cos, sin) of the shifts [pixel][translation] (2D data, see cuda_kernel_shift_table_pt)
		const CoarseParticle *g_particles = NULL // particle batching: see CoarseParticle (blockIdx.z)
		)
{
	int tid = threadIdx.x;
	int orient_offset = 0; // of the class in g_partial
	if (g_particles != NULL)
	{
		const CoarseParticle &P = g_particles[blockIdx.z];
		if ((int) blockIdx.x >= P.n_blocks || (int) blockIdx.y >= P.n_chunks)
			return;
		n_classes = P.n_classes; n_chunks = P.n_chunks; chunk_stride = P.chunk_stride;
		trans_x = P.trans_x; trans_y = P.trans_y; g_real = P.g_real; g_imag = P.g_imag; g_corr = P.g_corr;
		g_shift = P.shift; g_classes = P.classes; g_class_start = P.class_start; g_partial = P.partial;
	}

	AccProjectorKernel *pp = &projector_arg; // its methods are not const
	int orient0 = blockIdx.x * eulers_per_block;
	// Orientations of this block: fewer than eulers_per_block in the last block of a class
	// (class batching), whose empty places are computed with zero matrices and not stored
	int n_valid = eulers_per_block;
	if (n_classes > 0)
	{
		int local;
		const CoarseClassJob &c = g_classes[findBatchClass(g_class_start, n_classes, blockIdx.x, local)];
		pp = const_cast<AccProjectorKernel *>(&c.projector);
		g_eulers = c.eulers;
		g_diff2s = c.diff2s;
		orient0 = rest_blocks ? c.n_even + local : local * eulers_per_block;
		if (!rest_blocks)
			n_valid = min(eulers_per_block, c.n_orient - orient0);
		orient_offset = c.orient_offset;
	}
	AccProjectorKernel &projector = *pp;

	//Prefetch euler matrices
	__shared__ XFLOAT s_eulers[eulers_per_block * 9];

	int max_block_pass_euler( ceilfracf(eulers_per_block*9, block_sz) * block_sz);

	for (int i = tid; i < max_block_pass_euler; i += block_sz)
		if (i < eulers_per_block * 9)
			s_eulers[i] = (i < n_valid * 9) ? g_eulers[orient0 * 9 + i] : (XFLOAT) 0.;


	//Setup variables
	__shared__ XFLOAT s_ref_real[block_sz/prefetch_fraction * eulers_per_block];
	__shared__ XFLOAT s_ref_imag[block_sz/prefetch_fraction * eulers_per_block];

	__shared__ XFLOAT s_real[block_sz];
	__shared__ XFLOAT s_imag[block_sz];
	__shared__ XFLOAT s_corr[block_sz];

	XFLOAT diff2s[eulers_per_block] = {0.f};

	XFLOAT tx = trans_x[tid%translation_num];
	XFLOAT ty = trans_y[tid%translation_num];
	XFLOAT tz = trans_z[tid%translation_num];

	//Step through data: all pixel passes, or this block's chunk of them
	int n_pass = ceilfracf(image_size,block_sz);
	int pass_begin = 0, pass_end = n_pass;
	if (g_partial != NULL)
	{
		pass_begin = (int) (((long) blockIdx.y * n_pass) / n_chunks);
		pass_end   = (int) (((long) (blockIdx.y + 1) * n_pass) / n_chunks);
	}
	int max_block_pass_pixel( pass_end * block_sz );

	for (int init_pixel = pass_begin * block_sz; init_pixel < max_block_pass_pixel; init_pixel += block_sz/prefetch_fraction)
	{
		__syncthreads();

		//Prefetch block-fraction-wise
		if(init_pixel + tid/prefetch_fraction < image_size)
		{
			int x,y,z,xy;
			if(DATA3D)
			{
				z =  floorfracf(init_pixel + tid/prefetch_fraction, projector.imgX*projector.imgY);
				xy = (init_pixel + tid/prefetch_fraction) % (projector.imgX*projector.imgY);
				x =             xy  % projector.imgX;
				y = floorfracf( xy,   projector.imgX);
				if (z > projector.maxR)
					z -= projector.imgZ;
			}
			else
			{
				x =           ( init_pixel + tid/prefetch_fraction) % projector.imgX;
				y = floorfracf( init_pixel + tid/prefetch_fraction  , projector.imgX);
			}
			if (y > projector.maxR)
				y -= projector.imgY;

//			#pragma unroll
			for (int i = tid%prefetch_fraction; i < eulers_per_block; i += prefetch_fraction)
			{
				if(DATA3D) // if DATA3D, then REF3D as well.
					projector.project3Dmodel(
						x,y,z,
						s_eulers[i*9  ],
						s_eulers[i*9+1],
						s_eulers[i*9+2],
						s_eulers[i*9+3],
						s_eulers[i*9+4],
						s_eulers[i*9+5],
						s_eulers[i*9+6],
						s_eulers[i*9+7],
						s_eulers[i*9+8],
						s_ref_real[eulers_per_block * (tid/prefetch_fraction) + i],
						s_ref_imag[eulers_per_block * (tid/prefetch_fraction) + i]);
				else if(REF3D)
					projector.project3Dmodel(
						x,y,
						s_eulers[i*9  ],
						s_eulers[i*9+1],
						s_eulers[i*9+3],
						s_eulers[i*9+4],
						s_eulers[i*9+6],
						s_eulers[i*9+7],
						s_ref_real[eulers_per_block * (tid/prefetch_fraction) + i],
						s_ref_imag[eulers_per_block * (tid/prefetch_fraction) + i]);
				else
					projector.project2Dmodel(
						x,y,
						s_eulers[i*9  ],
						s_eulers[i*9+1],
						s_eulers[i*9+3],
						s_eulers[i*9+4],
						s_ref_real[eulers_per_block * (tid/prefetch_fraction) + i],
						s_ref_imag[eulers_per_block * (tid/prefetch_fraction) + i]);
			}
		}

		//Prefetch block-wise
		if (init_pixel % block_sz == 0 && init_pixel + tid < image_size)
		{
			s_real[tid] = g_real[init_pixel + tid];
			s_imag[tid] = g_imag[init_pixel + tid];
			s_corr[tid] = g_corr[init_pixel + tid] / 2;
		}

		__syncthreads();

		if (tid/translation_num < block_sz/translation_num) // NOTE int division A/B==C/B !=> A==C
		for (int i = tid / translation_num;
				i < block_sz/prefetch_fraction;
				i += block_sz/translation_num)
		{
			if((init_pixel + i) >= image_size) break;

			// the pixel's coordinates are only needed to make the phase here (not with the table)
			int x = 0,y = 0,z = 0,xy;
			if (DATA3D || g_shift == NULL)
			{
				if(DATA3D)
				{
					z =  floorfracf( init_pixel + i   ,  projector.imgX*projector.imgY); //TODO optimize index extraction.
					xy =           ( init_pixel + i ) % (projector.imgX*projector.imgY);
					x =             xy  % projector.imgX;
					y = floorfracf( xy,   projector.imgX);
					if (z > projector.maxR)
						z -= projector.imgZ;
				}
				else
				{
					x =           ( init_pixel + i ) % projector.imgX;
					y = floorfracf( init_pixel + i   , projector.imgX);
				}
				if (y > projector.maxR)
					y -= projector.imgY;
			}

			XFLOAT real, imag;

			if(DATA3D)
				translatePixel(x, y, z, tx, ty, tz, s_real[i + init_pixel % block_sz], s_imag[i + init_pixel % block_sz], real, imag);
			else if (g_shift != NULL)
			{
				const float2 sc = __ldg(&g_shift[(size_t) (init_pixel + i) * translation_num + tid % translation_num]);
				real = sc.x * s_real[i + init_pixel % block_sz] - sc.y * s_imag[i + init_pixel % block_sz];
				imag = sc.x * s_imag[i + init_pixel % block_sz] + sc.y * s_real[i + init_pixel % block_sz];
			}
			else
				translatePixel(x, y,    tx, ty,     s_real[i + init_pixel % block_sz], s_imag[i + init_pixel % block_sz], real, imag);


			#pragma unroll
			for (int j = 0; j < eulers_per_block; j ++)
			{
				XFLOAT diff_real =  s_ref_real[eulers_per_block * i + j] - real;
				XFLOAT diff_imag =  s_ref_imag[eulers_per_block * i + j] - imag;
				diff2s[j] += (diff_real * diff_real + diff_imag * diff_imag) * s_corr[i + init_pixel % block_sz];
			}
		}
	}

	//Set global
	if (g_partial == NULL)
	{
		#pragma unroll
		for (int i = 0; i < eulers_per_block; i ++)
			if (i < n_valid)
				cuda_atomic_add(&g_diff2s[(orient0 + i) * translation_num + tid % translation_num], diff2s[i]);
		return;
	}

	// Chunk: the threads tid, tid + translation_num, ... (the groups that took part,
	// tid / translation_num < block_sz / translation_num) hold sums for the same
	// translation; add them in group order and store the block's sum
	__shared__ XFLOAT s_sum[block_sz];
	const int groups = block_sz / translation_num;
	XFLOAT *out = &g_partial[blockIdx.y * chunk_stride + (size_t) (orient_offset + orient0) * translation_num];
	#pragma unroll
	for (int i = 0; i < eulers_per_block; i ++)
	{
		__syncthreads();
		s_sum[tid] = diff2s[i];
		__syncthreads();
		if (i < n_valid && tid < translation_num)
		{
			XFLOAT sum = s_sum[tid];
			for (int g = 1; g < groups; g++)
				sum += s_sum[g * translation_num + tid];
			out[i * translation_num + tid] = sum;
		}
	}
}

// Adds the chunks of cuda_kernel_diff2_coarse (pixel chunks) in chunk order to the
// differences of each class (blockIdx.y)
static __global__ void cuda_kernel_diff2_coarse_sum_chunks(
		const CoarseClassJob *g_classes,
		const XFLOAT *g_partial,
		int n_chunks,
		size_t chunk_stride,
		int translation_num)
{
	const CoarseClassJob &c = g_classes[blockIdx.y];
	const size_t n = (size_t) c.n_orient * translation_num;
	const XFLOAT *in = &g_partial[(size_t) c.orient_offset * translation_num];
	for (size_t k = blockIdx.x * (size_t) blockDim.x + threadIdx.x; k < n; k += (size_t) gridDim.x * blockDim.x)
	{
		XFLOAT sum = in[k];
		for (int ch = 1; ch < n_chunks; ch++)
			sum += in[ch * chunk_stride + k];
		c.diff2s[k] += sum;
	}
}

// The same for a batch of particles (blockIdx.z); dry: only the sums are made (written back over the first chunk), for timing
static __global__ void cuda_kernel_diff2_coarse_sum_chunks_batched(
		const CoarseParticle *g_particles,
		int translation_num,
		bool dry)
{
	const CoarseParticle &P = g_particles[blockIdx.z];
	if ((int) blockIdx.y >= P.n_classes)
		return;
	const int n_chunks = P.n_chunks;
	const size_t chunk_stride = P.chunk_stride;
	const CoarseClassJob &c = P.classes[blockIdx.y];
	const size_t n = (size_t) c.n_orient * translation_num;
	XFLOAT *in = &P.partial[(size_t) c.orient_offset * translation_num];
	for (size_t k = blockIdx.x * (size_t) blockDim.x + threadIdx.x; k < n; k += (size_t) gridDim.x * blockDim.x)
	{
		XFLOAT sum = in[k];
		for (int ch = 1; ch < n_chunks; ch++)
			sum += in[ch * chunk_stride + k];
		if (dry)
			in[k] = sum;
		else
			c.diff2s[k] += sum;
	}
}


// Minimum of several arrays (blockIdx.x = array), one launch for a group of particles' coarse differences
struct MinJob
{
	const XFLOAT *data;
	size_t n;
};

static __global__ void cuda_kernel_min_batched(const MinJob *g_jobs, XFLOAT *g_out)
{
	const MinJob &j = g_jobs[blockIdx.x];
	XFLOAT m = (XFLOAT) 3.4e38;
	for (size_t k = threadIdx.x; k < j.n; k += blockDim.x)
		m = min(m, j.data[k]);
	__shared__ XFLOAT s[256];
	s[threadIdx.x] = m;
	__syncthreads();
	for (int h = blockDim.x / 2; h > 0; h >>= 1)
	{
		if ((int) threadIdx.x < h)
			s[threadIdx.x] = min(s[threadIdx.x], s[threadIdx.x + h]);
		__syncthreads();
	}
	if (threadIdx.x == 0)
		g_out[blockIdx.x] = s[0];
}

template<bool REF3D, bool DATA3D, int block_sz, int chunk_sz>
__global__ void cuda_kernel_diff2_fine(
		XFLOAT *g_eulers,
		XFLOAT *g_imgs_real,
		XFLOAT *g_imgs_imag,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		AccProjectorKernel projector_arg,
		XFLOAT *g_corr_img,
		XFLOAT *g_diff2s,
		unsigned image_size,
		XFLOAT sum_init,
		unsigned long orientation_num,
		unsigned long translation_num,
		unsigned long todo_blocks,
		unsigned long *d_rot_idx,
		unsigned long *d_trans_idx,
		unsigned long *d_job_idx,
		unsigned long *d_job_num,
		const FineClassJob *g_classes = NULL, // class batching: one block per class and
		const int *g_class_start = NULL,      // job (see cuda_kernel_diff2_coarse)
		int n_classes = 0,
		int n_chunks = 1, // pixel chunks (2D references): blockIdx.y takes one of n_chunks ranges of the passes
		XFLOAT *g_partial = NULL // with chunks: each chunk's partial sums, [chunk][block][chunk_sz], added in a fixed order by cuda_kernel_diff2_fine_sum_chunks
		)
{
	unsigned long bid = blockIdx.x;
	unsigned long tid = threadIdx.x;

	AccProjectorKernel *pp = &projector_arg;
	if (n_classes > 0)
	{
		int local;
		const FineClassJob &c = g_classes[findBatchClass(g_class_start, n_classes, blockIdx.x, local)];
		pp = const_cast<AccProjectorKernel *>(&c.projector);
		g_eulers = c.eulers;
		g_diff2s = c.diff2s;
		d_rot_idx = c.rot_idx;
		d_trans_idx = c.trans_idx;
		d_job_idx = c.job_idx;
		d_job_num = c.job_num;
		bid = local;
	}
	AccProjectorKernel &projector = *pp;

//    // Specialize BlockReduce for a 1D block of 128 threads on type XFLOAT
//    typedef cub::BlockReduce<XFLOAT, 128> BlockReduce;
//    // Allocate shared memory for BlockReduce
//    __shared__ typename BlockReduce::TempStorage temp_storage;

	unsigned long pixel;
	XFLOAT ref_real, ref_imag,
		shifted_real, shifted_imag,
		diff_real, diff_imag;

	__shared__ XFLOAT s[block_sz*chunk_sz]; //We MAY have to do up to chunk_sz translations in each block
	__shared__ XFLOAT s_outs[chunk_sz];
	__shared__ XFLOAT s_tx[chunk_sz], s_ty[chunk_sz], s_tz[DATA3D ? chunk_sz : 1];
	// inside the padded 2D orientation gri
//	if( bid < todo_blocks ) // we only need to make
	{
		unsigned trans_num  = (unsigned)d_job_num[bid]; //how many transes we have for this rot
		for (int itrans=0; itrans<trans_num; itrans++)
		{
			s[itrans*block_sz+tid] = (XFLOAT)0.0;
		}
		// the translations of this job are the same for every pixel: read them once per block
		if (tid < trans_num)
		{
			const unsigned long jt = d_trans_idx[d_job_idx[bid]] + tid;
			s_tx[tid] = trans_x[jt];
			s_ty[tid] = trans_y[jt];
			if (DATA3D) s_tz[tid] = trans_z[jt];
		}
		__syncthreads();
		// index of comparison
		unsigned long int ix = d_rot_idx[d_job_idx[bid]];
		unsigned long int iy;
		unsigned pass_num(ceilfracf(image_size,block_sz));

		const unsigned pass_begin = (unsigned) (((unsigned long) blockIdx.y * pass_num) / n_chunks);
		const unsigned pass_end   = (unsigned) (((unsigned long) (blockIdx.y + 1) * pass_num) / n_chunks);
		for (unsigned pass = pass_begin; pass < pass_end; pass++) // finish an entire ref image each block (or its chunk)
		{
			pixel = (pass * block_sz) + tid;

			if(pixel < image_size)
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
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]), __ldg(&g_eulers[ix*9+2]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]), __ldg(&g_eulers[ix*9+5]),
						__ldg(&g_eulers[ix*9+6]), __ldg(&g_eulers[ix*9+7]), __ldg(&g_eulers[ix*9+8]),
						ref_real, ref_imag);
				else if(REF3D)
					projector.project3Dmodel(
						x,y,
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]),
						__ldg(&g_eulers[ix*9+6]), __ldg(&g_eulers[ix*9+7]),
						ref_real, ref_imag);
				else
					projector.project2Dmodel(
						x,y,
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]),
						ref_real, ref_imag);

				// the pixel's data are read once, not once per translation
				XFLOAT img_real = g_imgs_real[pixel], img_imag = g_imgs_imag[pixel], corr = __ldg(&g_corr_img[pixel]);
				for (int itrans=0; itrans<trans_num; itrans++) // finish all translations in each partial pass
				{
					if(DATA3D)
						translatePixel(x, y, z, s_tx[itrans], s_ty[itrans], s_tz[DATA3D ? itrans : 0], img_real, img_imag, shifted_real, shifted_imag);
					else
						translatePixel(x, y, s_tx[itrans], s_ty[itrans], img_real, img_imag, shifted_real, shifted_imag);

					diff_real =  ref_real - shifted_real;
					diff_imag =  ref_imag - shifted_imag;
					s[itrans*block_sz + tid] += (diff_real * diff_real + diff_imag * diff_imag) * (XFLOAT)0.5 * corr;
				}
			}
			__syncthreads();
		}
		for(int j=(block_sz/2); j>0; j/=2)
		{
			if(tid<j)
			{
				for (int itrans=0; itrans<trans_num; itrans++) // finish all translations in each partial pass
				{
					s[itrans*block_sz+tid] += s[itrans*block_sz+tid+j];
				}
			}
			__syncthreads();
		}
		if (tid < trans_num)
		{
			iy=d_job_idx[bid]+tid;
			if (n_chunks > 1)
			{
				// not added here: the chunks' partial sums are added in a fixed order by cuda_kernel_diff2_fine_sum_chunks
				g_partial[((size_t) blockIdx.y * gridDim.x + blockIdx.x) * chunk_sz + tid] = s[tid*block_sz];
			}
			else
			{
				s_outs[tid]=s[tid*block_sz]+sum_init;
				g_diff2s[iy] += s_outs[tid];
			}
		}
	}
}

// Adds the chunks' partial sums of each job in chunk order, then sum_init, to the job's differences: the same result
// whatever the order the chunk blocks ran in (one block per fine block, a thread per translation)
template<int chunk_sz>
__global__ void cuda_kernel_diff2_fine_sum_chunks(
		const FineClassJob *g_classes, const int *g_class_start, int n_classes,
		const XFLOAT *g_partial, int n_chunks, XFLOAT sum_init)
{
	int local;
	const FineClassJob &c = g_classes[findBatchClass(g_class_start, n_classes, blockIdx.x, local)];
	const unsigned trans_num = (unsigned) c.job_num[local];
	if (threadIdx.x < trans_num)
	{
		XFLOAT sum = (XFLOAT) 0.0;
		for (int ch = 0; ch < n_chunks; ch++)
			sum += g_partial[((size_t) ch * gridDim.x + blockIdx.x) * chunk_sz + threadIdx.x];
		c.diff2s[c.job_idx[local] + threadIdx.x] += sum + sum_init;
	}
}




/*
 *   	CROSS-CORRELATION-BASED KERNELS
 */

template<bool REF3D, bool DATA3D, int block_sz>
__global__ void cuda_kernel_diff2_CC_coarse(
		XFLOAT *g_eulers,
		XFLOAT *g_imgs_real,
		XFLOAT *g_imgs_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		AccProjectorKernel projector,
		XFLOAT *g_corr_img,
		XFLOAT *g_diff2s,
		unsigned translation_num,
		int image_size,
		XFLOAT exp_local_sqrtXi2
		)
{

	int iorient = blockIdx.x;
	int itrans =  blockIdx.y;
	int tid = threadIdx.x;

    __shared__ XFLOAT s_weight[block_sz];
    s_weight[tid] = (XFLOAT)0.0;
	__shared__ XFLOAT s_norm[block_sz];
	s_norm[tid] = (XFLOAT)0.0;

	XFLOAT real, imag, ref_real, ref_imag;

	XFLOAT e0,e1,e2,e3,e4,e5,e6,e7,e8;
	e0 = __ldg(&g_eulers[iorient*9  ]);
	e1 = __ldg(&g_eulers[iorient*9+1]);
	e2 = __ldg(&g_eulers[iorient*9+2]);
	e3 = __ldg(&g_eulers[iorient*9+3]);
	e4 = __ldg(&g_eulers[iorient*9+4]);
	e5 = __ldg(&g_eulers[iorient*9+5]);
	e6 = __ldg(&g_eulers[iorient*9+6]);
	e7 = __ldg(&g_eulers[iorient*9+7]);
	e8 = __ldg(&g_eulers[iorient*9+8]);

	__syncthreads();

	unsigned pixel_pass_num( ceilfracf(image_size,block_sz) );
	for (unsigned pass = 0; pass < pixel_pass_num; pass++)
	{
		unsigned pixel = (pass * block_sz) + tid;

		if(pixel < image_size)
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
					e0,e1,e2,e3,e4,e5,e6,e7,e8,
					ref_real, ref_imag);
			else if(REF3D)
				projector.project3Dmodel(
					x,y,
					e0,e1,e3,e4,e6,e7,
					ref_real, ref_imag);
			else
				projector.project2Dmodel(
					x,y,
					e0,e1,e3,e4,
					ref_real, ref_imag);

			if(DATA3D)
				translatePixel(x, y, z, g_trans_x[itrans], g_trans_y[itrans], g_trans_z[itrans], g_imgs_real[pixel], g_imgs_imag[pixel], real, imag);
			else
				translatePixel(x, y,    g_trans_x[itrans], g_trans_y[itrans],                    g_imgs_real[pixel], g_imgs_imag[pixel], real, imag);

			s_weight[tid] += (ref_real * real     + ref_imag * imag)      * __ldg(&g_corr_img[pixel]);
			s_norm[tid]   += (ref_real * ref_real + ref_imag * ref_imag ) * __ldg(&g_corr_img[pixel]);
		}
		__syncthreads();
	}


	for(int j=(block_sz/2); j>0; j/=2)
	{
		if(tid<j)
		{
			s_weight[tid] += s_weight[tid+j];
			s_norm[tid]   += s_norm[tid+j];
		}
		__syncthreads();
	}
#ifdef ACC_DOUBLE_PRECISION
    cuda_atomic_add(&g_diff2s[iorient * translation_num + itrans], - ( s_weight[0] / (block_sz * sqrt(s_norm[0]))) );
    //g_diff2s[iorient * translation_num + itrans] += - ( s_weight[0] / sqrt(s_norm[0]));
#else
    cuda_atomic_add(&g_diff2s[iorient * translation_num + itrans], - ( s_weight[0] / (block_sz * sqrtf(s_norm[0]))) );
    //g_diff2s[iorient * translation_num + itrans] += - ( s_weight[0] / sqrtf(s_norm[0]));

#endif
}

template<bool REF3D, bool DATA3D, int block_sz,int chunk_sz>
__global__ void cuda_kernel_diff2_CC_fine(
		XFLOAT *g_eulers,
		XFLOAT *g_imgs_real,
		XFLOAT *g_imgs_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		AccProjectorKernel projector,
		XFLOAT *g_corr_img,
		XFLOAT *g_diff2s,
		unsigned image_size,
		XFLOAT sum_init,
		XFLOAT exp_local_sqrtXi2,
		unsigned long orientation_num,
		unsigned long translation_num,
		unsigned long todo_blocks,
		unsigned long *d_rot_idx,
		unsigned long *d_trans_idx,
		unsigned long *d_job_idx,
		unsigned long *d_job_num
		)
{
	int bid = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = threadIdx.x;

//    // Specialize BlockReduce for a 1D block of 128 threads on type XFLOAT
//    typedef cub::BlockReduce<XFLOAT, 128> BlockReduce;
//    // Allocate shared memory for BlockReduce
//    __shared__ typename BlockReduce::TempStorage temp_storage;

	int pixel;
	XFLOAT ref_real, ref_imag, shifted_real, shifted_imag;

	__shared__ XFLOAT      s[block_sz*chunk_sz]; //We MAY have to do up to chunk_sz translations in each block
	__shared__ XFLOAT   s_cc[block_sz*chunk_sz];
	__shared__ XFLOAT s_outs[chunk_sz];

	if( bid < todo_blocks ) // we only need to make
	{
		unsigned trans_num   = d_job_num[bid]; //how many transes we have for this rot
		for (int itrans=0; itrans<trans_num; itrans++)
		{
			s[   itrans*block_sz+tid] = 0.0f;
			s_cc[itrans*block_sz+tid] = 0.0f;
		}
		__syncthreads();
		// index of comparison
		unsigned long int ix = d_rot_idx[d_job_idx[bid]];
		unsigned long int iy;
		unsigned pass_num( ceilfracf(image_size,block_sz) );

		for (unsigned pass = 0; pass < pass_num; pass++) // finish an entire ref image each block
		{
			pixel = (pass * block_sz) + tid;

			if(pixel < image_size)
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
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]), __ldg(&g_eulers[ix*9+2]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]), __ldg(&g_eulers[ix*9+5]),
						__ldg(&g_eulers[ix*9+6]), __ldg(&g_eulers[ix*9+7]), __ldg(&g_eulers[ix*9+8]),
						ref_real, ref_imag);
				else if(REF3D)
					projector.project3Dmodel(
						x,y,
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]),
						__ldg(&g_eulers[ix*9+6]), __ldg(&g_eulers[ix*9+7]),
						ref_real, ref_imag);
				else
					projector.project2Dmodel(
						x,y,
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]),
						ref_real, ref_imag);

				for (int itrans=0; itrans<trans_num; itrans++) // finish all translations in each partial pass
				{
					iy = d_trans_idx[d_job_idx[bid]] + itrans;

					if(DATA3D)
						translatePixel(x, y, z, g_trans_x[iy], g_trans_y[iy], g_trans_z[iy], g_imgs_real[pixel], g_imgs_imag[pixel], shifted_real, shifted_imag);
					else
						translatePixel(x, y,    g_trans_x[iy], g_trans_y[iy],                g_imgs_real[pixel], g_imgs_imag[pixel], shifted_real, shifted_imag);

					s[   itrans*block_sz + tid] += (ref_real * shifted_real + ref_imag * shifted_imag) * __ldg(&g_corr_img[pixel]);
					s_cc[itrans*block_sz + tid] += (ref_real*ref_real + ref_imag*ref_imag) * __ldg(&g_corr_img[pixel]);
				}
			}
			__syncthreads();
		}
		for(int j=(block_sz/2); j>0; j/=2)
		{
			if(tid<j)
			{
				for (int itrans=0; itrans<trans_num; itrans++) // finish all translations in each partial pass
				{
					s[   itrans*block_sz+tid] += s[   itrans*block_sz+tid+j];
					s_cc[itrans*block_sz+tid] += s_cc[itrans*block_sz+tid+j];
				}
			}
			__syncthreads();
		}
		if (tid < trans_num)
		{
#ifdef ACC_DOUBLE_PRECISION
			s_outs[tid]= - s[tid*block_sz] / (sqrt(s_cc[tid*block_sz]));
#else
			s_outs[tid]= - s[tid*block_sz] / (sqrtf(s_cc[tid*block_sz]));
#endif
		}
		if (tid < trans_num)
		{
			iy=d_job_idx[bid]+tid;
			g_diff2s[iy] += s_outs[tid];
		}
	}
}

#ifndef ACC_DOUBLE_PRECISION
/*
 * Cross-term form of the coarse differences (2D data, 3D reference, single precision),
 *   diff2(o, t) = sum_p c_p |ref(o,p) - x_p S(t,p)|^2
 *               = sum_p c_p |ref(o,p)|^2 + sum_p c_p |x_p|^2 - 2 sum_p Re conj(ref(o,p)) A(t,p),
 * with A(t,p) = c_p x_p S(t,p) made once per particle (cuda_kernel_coarse_xt_prepare).
 * The cross term is a small matrix product: each thread keeps a 4 orientation x 4
 * translation tile and takes four reference and four A values from shared memory for
 * every 32 multiply-adds, instead of two shared loads and a handful of operations for each.
 * Orientations are blocks of 16, translations tiles of 16 (blockIdx.z), pixel chunks
 * blockIdx.y as in cuda_kernel_diff2_coarse, whose partial-sum layout and order of
 * additions (fixed, no atomics) it shares. The terms are whitened (order 1 each), so the
 * single precision sums differ from the direct differences by rounding only.
 */
#define XT_BLOCK 128
#define XT_ORIENT 16
#define XT_TRANS 16
#define XT_PIXELS 32
#ifndef XT_MAXNREG_VALUE
#define XT_MAXNREG_VALUE 0
#endif
#if XT_MAXNREG_VALUE > 0
#define XT_BOUNDS __maxnreg__(XT_MAXNREG_VALUE)  // registers per thread capped (cannot be combined with __launch_bounds__)
#else
#define XT_BOUNDS __launch_bounds__(XT_BLOCK)
#endif

static __global__ void cuda_kernel_coarse_xt_prepare(
		const XFLOAT *trans_x,
		const XFLOAT *trans_y,
		const XFLOAT *g_real,
		const XFLOAT *g_imag,
		const XFLOAT *g_corr,
		float2 *g_A,
		float *g_x2,
		int translation_num,
		int tpad,
		int image_size,
		int img_x,
		int img_y,
		int max_r)
{
	const int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p >= image_size)
		return;
	const int x = p % img_x;
	int y = p / img_x;
	if (y > max_r)
		y -= img_y;
	const float re = g_real[p], im = g_imag[p];
	const float sc = g_corr[p] / 2;
	g_x2[p] = sc * (re * re + im * im);
	for (int t = 0; t < tpad; t++)
	{
		float2 a = make_float2(0.f, 0.f);
		if (t < translation_num)
		{
			float s, c;
			sincosf(x * trans_x[t] + y * trans_y[t], &s, &c);
			a = make_float2(sc * (c * re - s * im), sc * (c * im + s * re));
		}
		g_A[(size_t) p * tpad + t] = a;
	}
}

static __global__ void XT_BOUNDS cuda_kernel_diff2_coarse_xt(
		const CoarseClassJob *g_classes,
		const int *g_class_start,
		int n_classes,
		const float2 *g_A,
		const float *g_x2,
		const XFLOAT *g_corr,
		int translation_num,
		int tpad,
		int image_size,
		float *g_partial,
		int n_chunks,
		size_t chunk_stride,
		int ablate)
{
	const int tid = threadIdx.x;
	int local;
	const CoarseClassJob &c = g_classes[findBatchClass(g_class_start, n_classes, blockIdx.x, local)];
	AccProjectorKernel projector = c.projector;
	const float inv_imgX = 1.f / (float) projector.imgX;
	const int orient0 = local * XT_ORIENT;
	const int n_valid = min(XT_ORIENT, c.n_orient - orient0);
	const int ttile = blockIdx.z;

	// Projection role: orientation o of the block, pixels plane, plane + 8, ...
	const int o = tid & (XT_ORIENT - 1);
	const int plane = tid >> 4;
	float e0 = 0.f, e1 = 0.f, e3 = 0.f, e4 = 0.f, e6 = 0.f, e7 = 0.f;
	if (o < n_valid)
	{
		const float *e = c.eulers + (size_t) (orient0 + o) * 9;
		e0 = e[0]; e1 = e[1]; e3 = e[3]; e4 = e[4]; e6 = e[6]; e7 = e[7];
	}
	// Product role: orientations 4 * po .. +3, translations 4 * pt .. +3
	const int lane16 = tid & 15;
	const int po = lane16 & 3;
	const int pt = lane16 >> 2;

	// s_ref and s_A are only used in the pass loop and s_red (same size) after it, so they share
	// one buffer: 8 KB less shared memory per block, same values
	__shared__ __align__(16) float2 s_buf[XT_PIXELS * XT_ORIENT + XT_PIXELS * XT_TRANS];
	float2 *const s_ref = s_buf;
	float2 *const s_A = s_buf + XT_PIXELS * XT_ORIENT;
	__shared__ float2 s_refp[XT_ORIENT * (XT_PIXELS + 1)];
	__shared__ float s_rn2[XT_ORIENT];
	const bool pixfast = (ablate & 8) != 0;
	// pixfast: lane = pixel, warp w projects orientations 4w .. 4w+3
	float pe[4][6];
	float prn[4] = {0.f, 0.f, 0.f, 0.f};
	if (pixfast)
	{
		#pragma unroll
		for (int k = 0; k < 4; k++)
		{
			const int oo = (tid >> 5) * 4 + k;
			pe[k][0] = pe[k][1] = pe[k][2] = pe[k][3] = pe[k][4] = pe[k][5] = 0.f;
			if (oo < n_valid)
			{
				const float *e = c.eulers + (size_t) (orient0 + oo) * 9;
				pe[k][0] = e[0]; pe[k][1] = e[1]; pe[k][2] = e[3]; pe[k][3] = e[4]; pe[k][4] = e[6]; pe[k][5] = e[7];
			}
		}
	}

	float acc[4][4];
	#pragma unroll
	for (int i = 0; i < 4; i++)
		#pragma unroll
		for (int j = 0; j < 4; j++)
			acc[i][j] = 0.f;
	float rn = 0.f; // sum_p c_p |ref|^2 of this thread's orientation over its pixels

	const int n_pass = (image_size + XT_PIXELS - 1) / XT_PIXELS;
	const int pass_begin = (int) (((long) blockIdx.y * n_pass) / n_chunks);
	const int pass_end   = (int) (((long) (blockIdx.y + 1) * n_pass) / n_chunks);

	for (int pass = pass_begin; pass < pass_end; pass++)
	{
		const int base = pass * XT_PIXELS;
		__syncthreads();
		if (pixfast)
		{
			const int lane = tid & 31;
			const int p = base + lane;
			int x = 0, y = 0;
			float cw = 0.f;
			if (p < image_size)
			{
				y = (int) (((float) p + 0.5f) * inv_imgX);
				x = p - y * projector.imgX;
				if (y > projector.maxR)
					y -= projector.imgY;
				cw = g_corr[p] * 0.5f;
			}
			#pragma unroll
			for (int k = 0; k < 4; k++)
			{
				float re = 0.f, im = 0.f;
				if (p < image_size)
				{
					projector.project3Dmodel(x, y, pe[k][0], pe[k][1], pe[k][2], pe[k][3], pe[k][4], pe[k][5], re, im);
					prn[k] += cw * (re * re + im * im);
				}
				s_refp[((tid >> 5) * 4 + k) * (XT_PIXELS + 1) + lane] = make_float2(re, im);
			}
		}
		else
		{
		#pragma unroll
		for (int k = 0; k < XT_PIXELS / 8; k++)
		{
			const int row = plane + 8 * k;
			const int p = base + row;
			float re = 0.f, im = 0.f;
			if (p < image_size)
			{
				// p / imgX without an integer division: exact, p / imgX is at least 0.5 / imgX from an integer
				int y = (int) (((float) p + 0.5f) * inv_imgX);
				const int x = p - y * projector.imgX;
				if (y > projector.maxR)
					y -= projector.imgY;
				if (ablate & 1) { re = e0 * x + e1 * y; im = e3 * x + e4 * y; }
				else projector.project3Dmodel(x, y, e0, e1, e3, e4, e6, e7, re, im);
				rn += g_corr[p] * 0.5f * (re * re + im * im);
			}
			s_ref[row * XT_ORIENT + o] = make_float2(re, im);
		}
		}
		for (int idx = tid; idx < XT_PIXELS * XT_TRANS; idx += XT_BLOCK)
		{
			const int row = idx / XT_TRANS, t = idx % XT_TRANS;
			const int p = base + row;
			s_A[idx] = (ablate & 4) ? make_float2(p, t) : (p < image_size && ttile * XT_TRANS + t < tpad) ? g_A[(size_t) p * tpad + ttile * XT_TRANS + t] : make_float2(0.f, 0.f);
		}
		__syncthreads();
		#pragma unroll
		for (int k = 0; k < XT_PIXELS / 8; k++)
		{
			const int row = plane + 8 * k;
			float rr[4], ri[4];
			if (pixfast)
			{
				#pragma unroll
				for (int i = 0; i < 4; i++)
				{
					const float2 r = s_refp[(po * 4 + i) * (XT_PIXELS + 1) + row];
					rr[i] = r.x; ri[i] = r.y;
				}
			}
			else
			{
				const float4 r01 = *(const float4 *) &s_ref[row * XT_ORIENT + po * 4];
				const float4 r23 = *(const float4 *) &s_ref[row * XT_ORIENT + po * 4 + 2];
				rr[0] = r01.x; rr[1] = r01.z; rr[2] = r23.x; rr[3] = r23.z;
				ri[0] = r01.y; ri[1] = r01.w; ri[2] = r23.y; ri[3] = r23.w;
			}
			const float4 a01 = *(const float4 *) &s_A[row * XT_TRANS + pt * 4];
			const float4 a23 = *(const float4 *) &s_A[row * XT_TRANS + pt * 4 + 2];
			const float ar[4] = {a01.x, a01.z, a23.x, a23.z}, ai[4] = {a01.y, a01.w, a23.y, a23.w};
			if (!(ablate & 2))
			#pragma unroll
			for (int i = 0; i < 4; i++)
				#pragma unroll
				for (int j = 0; j < 4; j++)
					acc[i][j] += rr[i] * ar[j] + ri[i] * ai[j];
			else acc[0][0] += rr[0] + ri[1] + ar[2] + ai[3];
		}
	}

	// Sum over the 8 pixel planes in plane order, then store
	__syncthreads();
	float (*const s_red)[XT_ORIENT * XT_TRANS] = (float (*)[XT_ORIENT * XT_TRANS]) s_buf;
	__shared__ float s_rn[8][XT_ORIENT];
	__shared__ float s_x2[XT_BLOCK];
	#pragma unroll
	for (int i = 0; i < 4; i++)
		#pragma unroll
		for (int j = 0; j < 4; j++)
			s_red[plane][(po * 4 + i) * XT_TRANS + pt * 4 + j] = acc[i][j];
	s_rn[plane][o] = rn;
	if (pixfast)
	{
		#pragma unroll
		for (int k = 0; k < 4; k++)
		{
			float v = prn[k];
			#pragma unroll
			for (int d = 16; d > 0; d >>= 1)
				v += __shfl_xor_sync(0xffffffffu, v, d);
			if ((tid & 31) == 0) s_rn2[(tid >> 5) * 4 + k] = v;
		}
	}
	float x2 = 0.f;
	for (int p = pass_begin * XT_PIXELS + tid; p < min(pass_end * XT_PIXELS, image_size); p += XT_BLOCK)
		x2 += g_x2[p];
	s_x2[tid] = x2;
	__syncthreads();
	float x2_sum = 0.f;
	for (int i = 0; i < XT_BLOCK; i++)
		x2_sum += s_x2[i];
	float *out = &g_partial[(size_t) blockIdx.y * chunk_stride + (size_t) (c.orient_offset + orient0) * translation_num];
	for (int idx = tid; idx < XT_ORIENT * XT_TRANS; idx += XT_BLOCK)
	{
		const int oo = idx / XT_TRANS, t = idx % XT_TRANS;
		const int tg = ttile * XT_TRANS + t;
		if (oo < n_valid && tg < translation_num)
		{
			float cross = 0.f, norm = 0.f;
			for (int pl = 0; pl < 8; pl++)
			{
				cross += s_red[pl][idx];
				norm += s_rn[pl][oo];
			}
			if (pixfast) norm = s_rn2[oo];
			out[(size_t) oo * translation_num + tg] = norm + x2_sum - 2.f * cross;
		}
	}
}
#endif

#endif /* CUDA_DIFF2_KERNELS_CUH_ */
