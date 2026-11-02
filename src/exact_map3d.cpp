/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/exact_map3d.h"

#ifdef RELION_USE_FINUFFT

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <set>
#include <fftw3.h>

#include "src/async_file_writer.h"
#include "src/fftw.h"
#include "src/macros.h"

double exactmap::esKernelFT(double kappa, int w, double beta)
{
	// t = (w/2) sin(th) makes the integrand smooth; midpoint rule
	const int n = 400;
	double s = 0;
	for (int i = 0; i < n; i++)
	{
		const double th = -0.5 * PI + (i + 0.5) * PI / n;
		s += std::exp(beta * (std::cos(th) - 1.)) * std::cos(kappa * 0.5 * w * std::sin(th)) * 0.5 * w * std::cos(th);
	}
	return s * PI / n;
}

int exactmap::smoothEvenSize(int n)
{
	for (n += n & 1;; n += 2)
	{
		int m = n;
		for (int p : {2, 3, 5, 7}) while (m % p == 0) m /= p;
		if (m == 1) return n;
	}
}

ExactMap3D::ExactMap3D(const MultidimArray<RFLOAT> &map, int threads, int w_) : w(w_)
{
	nx = XSIZE(map); ny = YSIZE(map); nz = ZSIZE(map);
	x0 = STARTINGX(map); y0 = STARTINGY(map); z0 = STARTINGZ(map);
	n1x = exactmap::smoothEvenSize(2 * nx); n1y = exactmap::smoothEvenSize(2 * ny); n1z = exactmap::smoothEvenSize(2 * nz);
	sx = (double)n1x / nx; sy = (double)n1y / ny; sz = (double)n1z / nz;
	row = 2 * (size_t)(n1x / 2 + 1);
	const double beta = 2.30 * w;

	// the map's transform (normalised forward, as RELION's FourierTransformer)
	MultidimArray<RFLOAT> m = map;
	m.setXmippOrigin();
	MultidimArray<Complex> F;
	{
		FourierTransformer ft;
		ft.FourierTransform(m, F);
	}
	m.clear();

	std::vector<double> dx(nx / 2 + 1), dy(ny / 2 + 1), dz(nz / 2 + 1);
	for (int k = 0; k <= nx / 2; k++) dx[k] = 1. / exactmap::esKernelFT(2 * PI * k / n1x, w, beta);
	for (int k = 0; k <= ny / 2; k++) dy[k] = 1. / exactmap::esKernelFT(2 * PI * k / n1y, w, beta);
	for (int k = 0; k <= nz / 2; k++) dz[k] = 1. / exactmap::esKernelFT(2 * PI * k / n1z, w, beta);

	// divided by the kernel's transform, into the 2x grid (half complex, in place); a Nyquist plane is split in halves
	// between +N/2 and -N/2 (the real interpolant)
	grid.assign((size_t)n1z * n1y * row, 0.f);
	fftwf_complex *G = (fftwf_complex *)grid.data();
	const size_t crow = n1x / 2 + 1;
	#pragma omp parallel for num_threads(threads)
	for (long k = 0; k < nz; k++)
	{
		const long lz = k > nz / 2 ? k - nz : k;
		const bool nyq_z = 2 * lz == nz;
		for (long i = 0; i < ny; i++)
		{
			const long ly = i > ny / 2 ? i - ny : i;
			const bool nyq_y = 2 * ly == ny;
			const double fzy = dz[std::abs(lz)] * dy[std::abs(ly)] * (nyq_z ? 0.5 : 1.) * (nyq_y ? 0.5 : 1.);
			for (long j = 0; j <= nx / 2; j++)
			{
				const double f = fzy * dx[j] * (2 * j == nx ? 0.5 : 1.);
				const Complex v = DIRECT_A3D_ELEM(F, k, i, j);
				for (int az = 0; az < (nyq_z ? 2 : 1); az++)
					for (int ay = 0; ay < (nyq_y ? 2 : 1); ay++)
					{
						const long gz = ((az ? -lz : lz) + n1z) % n1z, gy = ((ay ? -ly : ly) + n1y) % n1y;
						fftwf_complex &g = G[((size_t)gz * n1y + gy) * crow + j];
						g[0] = (float)(v.real * f);
						g[1] = (float)(v.imag * f);
					}
			}
		}
	}
	F.clear();

	// one inverse FFT (unnormalised: the Fourier series)
	fftwf_plan plan;
	#pragma omp critical(FourierTransformer_fftw_plan)
	{
#if defined(RELION_HAVE_FFTW_THREADS) && !defined(MKLFFT)
		// float threads, as RELION's FFTs use them (RELION_FFTW_THREADS=1 keeps transforms single-threaded)
		const char *e = getenv("RELION_FFTW_THREADS");
		const int nt = (e != NULL && atoi(e) == 1) ? 1 : threads;
		static bool threads_ready = false;
		if (nt > 1 && !threads_ready) threads_ready = fftwf_init_threads() != 0;
		if (threads_ready) fftwf_plan_with_nthreads(nt);
#endif
		plan = fftwf_plan_dft_c2r_3d(n1z, n1y, n1x, G, grid.data(), FFTW_ESTIMATE);
#if defined(RELION_HAVE_FFTW_THREADS) && !defined(MKLFFT)
		if (threads_ready) fftwf_plan_with_nthreads(1);
#endif
	}
	fftwf_execute(plan);
	#pragma omp critical(FourierTransformer_fftw_plan)
	fftwf_destroy_plan(plan);

	// kernel by distance d in [0, w] from the first tap, tabulated: linear interpolation in it is ~1e-6 accurate
	tab.resize(w * L + 2);
	for (int i = 0; i < (int)tab.size(); i++)
	{
		const double t = std::min(1., 2. * i / (w * L) - 1.);
		tab[i] = std::exp(beta * (std::sqrt(std::max(0., 1. - t * t)) - 1.));
	}
}

RFLOAT ExactMap3D::value(RFLOAT x, RFLOAT y, RFLOAT z) const
{
	const double u = (x - x0) * sx - 0.5 * w, v = (y - y0) * sy - 0.5 * w, s = (z - z0) * sz - 0.5 * w;
	const long i0 = (long)std::ceil(u), j0 = (long)std::ceil(v), k0 = (long)std::ceil(s);
	float wx[16], wy[16], wz[16];
	long cx[16], cy[16], cz[16];
	for (int j = 0; j < w; j++)
	{
		const double ex = (i0 + j - u) * L, ey = (j0 + j - v) * L, ez = (k0 + j - s) * L;
		const int kx = (int)ex, ky = (int)ey, kz = (int)ez;
		wx[j] = tab[kx] + (float)(ex - kx) * (tab[kx + 1] - tab[kx]);
		wy[j] = tab[ky] + (float)(ey - ky) * (tab[ky + 1] - tab[ky]);
		wz[j] = tab[kz] + (float)(ez - kz) * (tab[kz + 1] - tab[kz]);
		cx[j] = ((i0 + j) % n1x + n1x) % n1x;
		cy[j] = ((j0 + j) % n1y + n1y) % n1y;
		cz[j] = ((k0 + j) % n1z + n1z) % n1z;
	}
	double val = 0;
	for (int jz = 0; jz < w; jz++)
	{
		double sz_ = 0;
		for (int jy = 0; jy < w; jy++)
		{
			const float *r = grid.data() + ((size_t)cz[jz] * n1y + cy[jy]) * row;
			float sy_ = 0;
			for (int jx = 0; jx < w; jx++) sy_ += wx[jx] * r[cx[jx]];
			sz_ += wy[jy] * sy_;
		}
		val += wz[jz] * sz_;
	}
	return val;
}

void ExactMap3D::applyGeometry(MultidimArray<RFLOAT> &out, const Matrix2D<RFLOAT> &A, bool inv, int threads,
                               const MultidimArray<RFLOAT> *where) const
{
	if (MAT_XSIZE(A) != 4 || MAT_YSIZE(A) != 4)
		REPORT_ERROR("ExactMap3D::applyGeometry: the transformation matrix is not 4x4");
	const Matrix2D<RFLOAT> Ai = inv ? A : A.inv();
	out.resize(nz, ny, nx);
	STARTINGZ(out) = z0; STARTINGY(out) = y0; STARTINGX(out) = x0;
	// coordinates from the box centre, size/2, as applyGeometry
	const long cx = nx / 2, cy = ny / 2, cz = nz / 2;
	const double eps = XMIPP_EQUAL_ACCURACY;
	#pragma omp parallel for num_threads(threads)
	for (long k = 0; k < nz; k++)
		for (long i = 0; i < ny; i++)
			for (long j = 0; j < nx; j++)
			{
				RFLOAT &o = DIRECT_A3D_ELEM(out, k, i, j);
				o = 0;
				if (where != NULL && !(DIRECT_A3D_ELEM(*where, k, i, j) > 0)) continue;
				const double x = j - cx, y = i - cy, z = k - cz;
				const double xp = x * Ai(0, 0) + y * Ai(0, 1) + z * Ai(0, 2) + Ai(0, 3);
				const double yp = x * Ai(1, 0) + y * Ai(1, 1) + z * Ai(1, 2) + Ai(1, 3);
				const double zp = x * Ai(2, 0) + y * Ai(2, 1) + z * Ai(2, 2) + Ai(2, 3);
				if (xp < -cx - eps || xp > nx - cx - 1 + eps || yp < -cy - eps || yp > ny - cy - 1 + eps ||
				    zp < -cz - eps || zp > nz - cz - 1 + eps)
					continue;
				o = value(xp + cx + x0, yp + cy + y0, zp + cz + z0);
			}
}

size_t ExactMap3D::bytes(const MultidimArray<RFLOAT> &map)
{
	const size_t n1x = exactmap::smoothEvenSize(2 * XSIZE(map)), n1y = exactmap::smoothEvenSize(2 * YSIZE(map)),
	             n1z = exactmap::smoothEvenSize(2 * ZSIZE(map));
	// the grid, plus the map's transform and a copy of the map while it is made
	return n1z * n1y * 2 * (n1x / 2 + 1) * sizeof(float) + MULTIDIM_SIZE(map) * (sizeof(RFLOAT) + sizeof(Complex) / 2 + sizeof(Complex));
}

bool ExactMap3D::fits(const MultidimArray<RFLOAT> &map)
{
	const long long avail = asyncwrite::memoryAvailableBytes();
	return avail < 0 || bytes(map) <= (size_t)(avail / 2);
}

bool ExactMap3D::requested()
{
	const char *v = getenv("RELION_MAP_INTERPOLATION");
	const std::string s = v ? v : "";
	if (s == "nufft") return true;
	if (!s.empty() && s != "linear" && s != "trilinear")
		REPORT_ERROR("RELION_MAP_INTERPOLATION must be linear (the default) or nufft, not " + s);
	return false;
}

bool ExactMap3D::use(const MultidimArray<RFLOAT> &map, const std::string &where)
{
	if (!requested()) return false;
	const bool ok = fits(map);
	bool first;
	#pragma omp critical(ExactMap3D_report)
	{
		static std::set<std::string> told;
		first = told.insert(where).second;
	}
	if (first)
	{
		if (ok) std::cout << " Exact (NUFFT) map resampling for " << where << "." << std::endl;
		else std::cout << " Exact (NUFFT) map resampling for " << where << " would need " << bytes(map) / (1 << 20)
		               << " MB, more than half the memory available: using trilinear interpolation." << std::endl;
	}
	return ok;
}

#endif // RELION_USE_FINUFFT
