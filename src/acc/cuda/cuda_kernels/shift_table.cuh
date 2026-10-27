#ifndef CUDA_SHIFT_TABLE_CUH_
#define CUDA_SHIFT_TABLE_CUH_

#ifndef ACC_DOUBLE_PRECISION

#include <cuda_runtime.h>
#include "src/acc/acc_projector.h"
#include "src/acc/cuda/cuda_settings.h"
#include "src/acc/cuda/cuda_kernels/cuda_device_utils.cuh"

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


// Separable form of the same phases (RELION_GPU_WAVG_SEPARABLE): the phase of a pixel is x*tx + y*ty, so
// (cos, sin) of it is the complex product of those of x*tx and of y*ty. g_shx[t * img_x + x] holds the first,
// g_shy[t * img_y + row] the second for the image row `row` (the y coordinate of the row, wrapped as the
// pixel loops do: rows beyond max_r count from the far side; the rows between, whose pixels use x = max_r,
// keep their own y). Sizes (img_x + img_y) per translation instead of image_size.
static __global__ void cuda_kernel_shift_table_sep(
		const XFLOAT *g_trans_x,
		const XFLOAT *g_trans_y,
		float2 *g_shx,
		float2 *g_shy,
		int img_x,
		int img_y,
		int max_r)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int t = blockIdx.y;
	XFLOAT s, c;
	if (i < img_x)
	{
		sincosf( i * g_trans_x[t], &s, &c );
		g_shx[(size_t) t * img_x + i] = make_float2(c, s);
	}
	else if (i < img_x + img_y)
	{
		const int row = i - img_x;
		const int y = (row > max_r && row >= img_y - max_r) ? row - img_y : row;
		sincosf( y * g_trans_y[t], &s, &c );
		g_shy[(size_t) t * img_y + row] = make_float2(c, s);
	}
}

#endif // ACC_DOUBLE_PRECISION
#endif /* CUDA_SHIFT_TABLE_CUH_ */
