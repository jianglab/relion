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

#endif // ACC_DOUBLE_PRECISION
#endif /* CUDA_SHIFT_TABLE_CUH_ */
