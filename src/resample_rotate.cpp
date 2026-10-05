/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/resample_rotate.h"
#include "src/fftw.h"
#include "src/relion_finufft.h"

#include <cmath>
#include <iostream>
#include <vector>

ResampleMethod parseResampleMethod(const std::string &name)
{
	if (name == "linear") return RESAMPLE_LINEAR;
	if (name == "cubic") return RESAMPLE_CUBIC;
	if (name == "nufft") return RESAMPLE_NUFFT;
	REPORT_ERROR("Unknown resampling method '" + name + "'; use linear, cubic or nufft.");
	return RESAMPLE_LINEAR;
}

std::string resampleMethodName(ResampleMethod method)
{
	switch (method)
	{
		case RESAMPLE_LINEAR: return "linear";
		case RESAMPLE_CUBIC: return "cubic";
		default: return "nufft";
	}
}

namespace
{

inline double pixelAt(const MultidimArray<RFLOAT> &src, long ix, long iy)
{
	if (ix < 0 || iy < 0 || ix >= (long)XSIZE(src) || iy >= (long)YSIZE(src)) return 0.0;
	return DIRECT_A2D_ELEM(src, iy, ix);
}

inline double cubicWeight(double t)
{
	// Catmull-Rom
	t = std::abs(t);
	if (t < 1.0) return 1.5 * t * t * t - 2.5 * t * t + 1.0;
	if (t < 2.0) return -0.5 * t * t * t + 2.5 * t * t - 4.0 * t + 2.0;
	return 0.0;
}

inline bool outside(const MultidimArray<RFLOAT> &src, double px, double py)
{
	return px < 0.0 || py < 0.0 || px > (double)(XSIZE(src) - 1) || py > (double)(YSIZE(src) - 1);
}

}

void resampleRotatedRectangle(const MultidimArray<RFLOAT> &src, MultidimArray<RFLOAT> &dest,
                              int nx, int ny, double centre_x, double centre_y,
                              double angle_degrees, ResampleMethod method)
{
	if (nx < 1 || ny < 1)
		REPORT_ERROR("resampleRotatedRectangle: the output size must be positive.");
	if (ZSIZE(src) != 1 || NSIZE(src) != 1)
		REPORT_ERROR("resampleRotatedRectangle: the source must be a 2D image.");

	const double a = DEG2RAD(angle_degrees);
	const double ca = std::cos(a), sa = std::sin(a);

	dest.initZeros(ny, nx);

	std::vector<double> px((size_t)nx * ny), py((size_t)nx * ny);
	for (int y = 0; y < ny; y++)
		for (int x = 0; x < nx; x++)
		{
			const double dx = x - nx / 2, dy = y - ny / 2;
			px[(size_t)y * nx + x] = centre_x + dx * ca - dy * sa;
			py[(size_t)y * nx + x] = centre_y + dx * sa + dy * ca;
		}

	if (method == RESAMPLE_LINEAR)
	{
		for (size_t i = 0; i < px.size(); i++)
		{
			if (outside(src, px[i], py[i])) continue;
			const long x0 = (long)std::floor(px[i]), y0 = (long)std::floor(py[i]);
			const double fx = px[i] - x0, fy = py[i] - y0;
			const double v = (1 - fy) * ((1 - fx) * pixelAt(src, x0, y0) + fx * pixelAt(src, x0 + 1, y0))
			               + fy * ((1 - fx) * pixelAt(src, x0, y0 + 1) + fx * pixelAt(src, x0 + 1, y0 + 1));
			DIRECT_MULTIDIM_ELEM(dest, i) = v;
		}
	}
	else if (method == RESAMPLE_CUBIC)
	{
		for (size_t i = 0; i < px.size(); i++)
		{
			if (outside(src, px[i], py[i])) continue;
			const long x0 = (long)std::floor(px[i]), y0 = (long)std::floor(py[i]);
			const double fx = px[i] - x0, fy = py[i] - y0;
			double v = 0.0;
			for (int j = -1; j <= 2; j++)
			{
				const double wy = cubicWeight(fy - j);
				for (int k = -1; k <= 2; k++)
					v += wy * cubicWeight(fx - k) * pixelAt(src, x0 + k, y0 + j);
			}
			DIRECT_MULTIDIM_ELEM(dest, i) = v;
		}
	}
	else
	{
#ifndef RELION_USE_FINUFFT
		REPORT_ERROR("Resampling method nufft needs a RELION build with RELION_USE_FINUFFT=ON.");
#else
		const int Nx = XSIZE(src), Ny = YSIZE(src);

		// RELION's forward transform already divides by Nx*Ny.
		// Full (not half) transform, modes -(N/2)..N-1-N/2 along each axis, as modeord=0 expects.
		MultidimArray<RFLOAT> copy = src;
		FourierTransformer transformer;
		MultidimArray<Complex> half;
		transformer.FourierTransform(copy, half, true);

		std::vector<std::complex<RFLOAT> > modes((size_t)Nx * Ny);
		for (int iy = 0; iy < Ny; iy++)
		{
			const int ky = iy - Ny / 2;
			for (int ix = 0; ix < Nx; ix++)
			{
				const int kx = ix - Nx / 2;
				Complex c;
				if (kx >= 0)
					c = DIRECT_A2D_ELEM(half, ((ky % Ny) + Ny) % Ny, kx);
				else
				{
					c = DIRECT_A2D_ELEM(half, (((-ky) % Ny) + Ny) % Ny, -kx);
					c.imag = -c.imag;
				}
				modes[(size_t)iy * Nx + ix] = std::complex<RFLOAT>(c.real, c.imag);
			}
		}

		std::vector<RFLOAT> tx, ty;
		std::vector<size_t> index;
		for (size_t i = 0; i < px.size(); i++)
		{
			if (outside(src, px[i], py[i])) continue;
			tx.push_back((RFLOAT)(2.0 * PI * px[i] / Nx));
			ty.push_back((RFLOAT)(2.0 * PI * py[i] / Ny));
			index.push_back(i);
		}
		if (index.empty()) return;

		std::vector<std::complex<RFLOAT> > out(index.size());
		// FINUFFT leaves std::cout in scientific format with raised precision
		struct StreamState {
			std::ios_base::fmtflags flags; std::streamsize prec;
			StreamState() : flags(std::cout.flags()), prec(std::cout.precision()) {}
			~StreamState() { std::cout.flags(flags); std::cout.precision(prec); }
		} stream_state;
		FinufftPlanGuard guard;
		finufft_opts opts;
		finufft_default_opts(&opts);
		opts.modeord = 0;
		opts.nthreads = 1;
		int64_t n_modes[3] = {Nx, Ny, 1};
		int ier = FINUFFT_MAKEPLAN(2, 2, n_modes, +1, 1, 1e-12, &guard.plan, &opts);
		if (ier > 1) handleFinufftError(ier, "finufft_makeplan type2 (resample)");
		ier = FINUFFT_SETPTS(guard.plan, (int64_t)index.size(), tx.data(), ty.data(), NULL, 0, NULL, NULL, NULL);
		if (ier > 1) handleFinufftError(ier, "finufft_setpts type2 (resample)");
		ier = FINUFFT_EXECUTE(guard.plan, out.data(), modes.data());
		if (ier > 1) handleFinufftError(ier, "finufft_execute type2 (resample)");
		guard.clear();

		for (size_t j = 0; j < index.size(); j++)
			DIRECT_MULTIDIM_ELEM(dest, index[j]) = out[j].real();
#endif
	}
}
