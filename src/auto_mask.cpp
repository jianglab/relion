/***************************************************************************
 *
 * Automatic solvent masks for 3D maps, after the masks of trueFSC in helicon:
 * https://github.com/jianglab/helicon/blob/main/src/helicon/commands/trueFSC.py
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 ***************************************************************************/

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <random>
#include <vector>
#include "src/auto_mask.h"
#include "src/fftw.h"
#include "src/error.h"

// Frequency (cycles per pixel) of index idx along a dimension of n pixels in FFTW order
static inline RFLOAT fftwFrequency(long int idx, long int n)
{
	return (RFLOAT)((idx <= n / 2) ? idx : idx - n) / n;
}

// The FSC shell, in units of 1 / XSIZE, of element (k, i, j) of the transform of a real
// array of xdim x ydim x zdim pixels
static inline long int fftwShell(long int k, long int i, long int j, long int xdim, long int ydim, long int zdim)
{
	const RFLOAT fx = (RFLOAT)j / xdim;
	const RFLOAT fy = fftwFrequency(i, ydim);
	const RFLOAT fz = (zdim > 1) ? fftwFrequency(k, zdim) : 0.;
	return ROUND(xdim * sqrt(fx * fx + fy * fy + fz * fz));
}

static void fscOfTransforms(const MultidimArray<Complex> &F1, const MultidimArray<Complex> &F2,
		long int xdim, long int ydim, long int zdim, MultidimArray<RFLOAT> &fsc)
{
	const long int nr_shells = xdim / 2 + 1;
	std::vector<double> num(nr_shells, 0.), den1(nr_shells, 0.), den2(nr_shells, 0.);
	for (long int k = 0; k < ZSIZE(F1); k++)
	for (long int i = 0; i < YSIZE(F1); i++)
	for (long int j = 0; j < XSIZE(F1); j++)
	{
		const long int r = fftwShell(k, i, j, xdim, ydim, zdim);
		if (r >= nr_shells)
			continue;
		const Complex z1 = DIRECT_A3D_ELEM(F1, k, i, j);
		const Complex z2 = DIRECT_A3D_ELEM(F2, k, i, j);
		num[r] += z1.real * z2.real + z1.imag * z2.imag;
		den1[r] += z1.real * z1.real + z1.imag * z1.imag;
		den2[r] += z2.real * z2.real + z2.imag * z2.imag;
	}
	fsc.initZeros(nr_shells);
	for (long int r = 0; r < nr_shells; r++)
		if (den1[r] > 0. && den2[r] > 0.)
			DIRECT_A1D_ELEM(fsc, r) = num[r] / sqrt(den1[r] * den2[r]);
}

void autoMaskFsc(const MultidimArray<RFLOAT> &m1, const MultidimArray<RFLOAT> &m2, MultidimArray<RFLOAT> &fsc)
{
	if (!m1.sameShape(m2))
		REPORT_ERROR("autoMaskFsc ERROR: the maps have different sizes");
	MultidimArray<RFLOAT> c1 = m1, c2 = m2;
	MultidimArray<Complex> F1, F2;
	FourierTransformer transformer1, transformer2;
	transformer1.FourierTransform(c1, F1, false);
	transformer2.FourierTransform(c2, F2, false);
	fscOfTransforms(F1, F2, XSIZE(m1), YSIZE(m1), ZSIZE(m1), fsc);
}

RFLOAT autoMaskResolutionAtFsc(const MultidimArray<RFLOAT> &fsc, RFLOAT threshold, int box, RFLOAT angpix)
{
	for (long int i = 1; i < XSIZE(fsc); i++)
	{
		const RFLOAT y1 = DIRECT_A1D_ELEM(fsc, i);
		if (y1 >= threshold)
			continue;
		const RFLOAT y0 = DIRECT_A1D_ELEM(fsc, i - 1);
		const RFLOAT x = (y0 == y1) ? i : (i - 1) + (threshold - y0) / (y1 - y0);
		return (x > 0.) ? box * angpix / x : box * angpix;
	}
	return -1.;
}

void autoMaskGaussianLowPass(MultidimArray<RFLOAT> &map, RFLOAT lowpass, RFLOAT angpix)
{
	const RFLOAT sigma = lowpass / (3.81 * angpix);
	const RFLOAT c = -2. * PI * PI * sigma * sigma;
	const long int xdim = XSIZE(map), ydim = YSIZE(map), zdim = ZSIZE(map);
	MultidimArray<Complex> FT;
	FourierTransformer transformer;
	transformer.FourierTransform(map, FT, false);
	for (long int k = 0; k < ZSIZE(FT); k++)
	{
		const RFLOAT fz = (zdim > 1) ? fftwFrequency(k, zdim) : 0.;
		for (long int i = 0; i < YSIZE(FT); i++)
		{
			const RFLOAT fy = fftwFrequency(i, ydim);
			for (long int j = 0; j < XSIZE(FT); j++)
			{
				const RFLOAT fx = (RFLOAT)j / xdim;
				const RFLOAT g = exp(c * (fx * fx + fy * fy + fz * fz));
				DIRECT_A3D_ELEM(FT, k, i, j).real *= g;
				DIRECT_A3D_ELEM(FT, k, i, j).imag *= g;
			}
		}
	}
	transformer.inverseFourierTransform();
}

RFLOAT autoMaskOtsuThreshold(const MultidimArray<RFLOAT> &map, int nr_bins)
{
	const RFLOAT hmin = map.computeMin(), hmax = map.computeMax();
	const RFLOAT bin_width = (hmax - hmin) / nr_bins;
	if (bin_width <= 0.)
		return hmin;

	std::vector<double> hist(nr_bins, 0.);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(map)
	{
		const RFLOAT v = DIRECT_MULTIDIM_ELEM(map, n);
		if (v == 0.)
			continue;
		const int b = std::min(nr_bins - 1, (int)((v - hmin) / bin_width));
		hist[b] += 1.;
	}

	double total = 0., sum_all = 0.;
	for (int b = 0; b < nr_bins; b++)
	{
		total += hist[b];
		sum_all += b * hist[b];
	}
	if (total == 0.)
		return hmin;

	// The between-class variance of each split; the first bin is never the split
	double cum = 0., cum_val = 0., best = -1.;
	int best_bin = 1;
	for (int b = 0; b < nr_bins; b++)
	{
		cum += hist[b];
		cum_val += b * hist[b];
		if (b == 0)
			continue;
		const double wb = cum, wf = total - cum;
		double between = 0.;
		if (wb > 0. && wf > 0.)
		{
			const double mb = cum_val / wb, mf = (sum_all - cum_val) / wf;
			between = wb * wf * (mb - mf) * (mb - mf);
		}
		if (between > best)
		{
			best = between;
			best_bin = b;
		}
	}
	return hmin + (best_bin + 1) * bin_width;
}

void autoMaskBinary(const MultidimArray<RFLOAT> &map, RFLOAT angpix, RFLOAT lowpass,
		const AutoMaskOptions &opt, MultidimArray<RFLOAT> &binary, AutoMaskReport &report)
{
	MultidimArray<RFLOAT> lp = map;
	if (lowpass > 2. * angpix)
	{
		autoMaskGaussianLowPass(lp, lowpass, angpix);
		report.lowpass = lowpass;
	}
	else
		report.lowpass = -1.;

	const long int nr_voxels = MULTIDIM_SIZE(lp);
	RFLOAT threshold;
	if (opt.threshold_fraction > 0.)
		threshold = opt.threshold_fraction * lp.computeMax();
	else if (opt.threshold > 0.)
		threshold = opt.threshold;
	else if (opt.mass > 0.)
	{
		// The value of the voxel that encloses the mass, counting from the brightest one
		const long int nr_enclosed = (long int)(opt.mass * 1e3 / (0.81 * angpix * angpix * angpix));
		const long int idx = std::min(nr_enclosed, nr_voxels - 1);
		std::vector<RFLOAT> values(MULTIDIM_ARRAY(lp), MULTIDIM_ARRAY(lp) + nr_voxels);
		std::nth_element(values.begin(), values.begin() + idx, values.end(), std::greater<RFLOAT>());
		threshold = values[idx];
	}
	else
		threshold = autoMaskOtsuThreshold(lp);
	report.threshold = threshold;

	// The brightest voxels seed the regions to keep
	typedef std::pair<RFLOAT, long int> ValueIndex;
	std::priority_queue<ValueIndex, std::vector<ValueIndex>, std::greater<ValueIndex> > brightest;
	const int nr_seeds = std::max(1, opt.nr_seeds);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(lp)
	{
		const RFLOAT v = DIRECT_MULTIDIM_ELEM(lp, n);
		if ((int)brightest.size() < nr_seeds)
			brightest.push(ValueIndex(v, n));
		else if (v > brightest.top().first)
		{
			brightest.pop();
			brightest.push(ValueIndex(v, n));
		}
	}

	// Grow them through the 26-connected voxels above the threshold
	const long int xdim = XSIZE(lp), ydim = YSIZE(lp), zdim = ZSIZE(lp);
	binary.initZeros(lp);
	long int nr_inside = 0;
	std::vector<long int> todo;
	for (; !brightest.empty(); brightest.pop())
	{
		const long int seed = brightest.top().second;
		if (DIRECT_MULTIDIM_ELEM(lp, seed) <= threshold || DIRECT_MULTIDIM_ELEM(binary, seed) > 0.)
			continue;
		DIRECT_MULTIDIM_ELEM(binary, seed) = 1.;
		nr_inside++;
		todo.push_back(seed);
		while (!todo.empty())
		{
			const long int n = todo.back();
			todo.pop_back();
			const long int k = n / (xdim * ydim), i = (n / xdim) % ydim, j = n % xdim;
			for (long int kk = std::max(0L, k - 1); kk <= std::min(zdim - 1, k + 1); kk++)
			for (long int ii = std::max(0L, i - 1); ii <= std::min(ydim - 1, i + 1); ii++)
			for (long int jj = std::max(0L, j - 1); jj <= std::min(xdim - 1, j + 1); jj++)
			{
				const long int nn = (kk * ydim + ii) * xdim + jj;
				if (DIRECT_MULTIDIM_ELEM(binary, nn) == 0. && DIRECT_MULTIDIM_ELEM(lp, nn) > threshold)
				{
					DIRECT_MULTIDIM_ELEM(binary, nn) = 1.;
					nr_inside++;
					todo.push_back(nn);
				}
			}
		}
	}

	// No seed above the threshold: everything above it
	if (nr_inside == 0)
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(lp)
		{
			if (DIRECT_MULTIDIM_ELEM(lp, n) > threshold)
			{
				DIRECT_MULTIDIM_ELEM(binary, n) = 1.;
				nr_inside++;
			}
		}
	}
	report.mass = nr_inside * angpix * angpix * angpix * 0.81e-3;

	if (opt.verb)
	{
		std::cout << "  + auto mask threshold: " << threshold
		          << ((opt.threshold_fraction > 0. || opt.threshold > 0.) ? "" : (opt.mass > 0.) ? " (from the mass)" : " (Otsu)") << std::endl;
		std::cout << "  + auto mask mass inside: " << ROUND(report.mass) << " kDa" << std::endl;
	}
}

// Squared Euclidean distance transform of one line (Felzenszwalb & Huttenlocher): d[q] is the
// smallest (q - p)^2 + f[p]; entries of f at or above far are not sources
static void squaredDistance1D(const std::vector<double> &f, std::vector<double> &d,
		std::vector<long int> &v, std::vector<double> &z, long int n, double far)
{
	long int k = -1;
	for (long int q = 0; q < n; q++)
	{
		if (f[q] >= far)
			continue;
		if (k < 0)
		{
			k = 0;
			v[0] = q;
			z[0] = -far;
			z[1] = far;
			continue;
		}
		double s;
		while (true)
		{
			const long int p = v[k];
			s = ((f[q] + (double)q * q) - (f[p] + (double)p * p)) / (2. * (q - p));
			if (k > 0 && s <= z[k])
				k--;
			else
				break;
		}
		k++;
		v[k] = q;
		z[k] = s;
		z[k + 1] = far;
	}
	if (k < 0)
	{
		for (long int q = 0; q < n; q++)
			d[q] = far;
		return;
	}
	long int j = 0;
	for (long int q = 0; q < n; q++)
	{
		while (z[j + 1] < q)
			j++;
		const long int p = v[j];
		d[q] = (double)(q - p) * (q - p) + f[p];
	}
}

void autoMaskDistance(const MultidimArray<RFLOAT> &binary, MultidimArray<RFLOAT> &dist, int n_threads)
{
	const double far = 1e30;
	const long int xdim = XSIZE(binary), ydim = YSIZE(binary), zdim = ZSIZE(binary);
	dist.resize(binary);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(binary)
		DIRECT_MULTIDIM_ELEM(dist, n) = (DIRECT_MULTIDIM_ELEM(binary, n) > 0.5) ? 0. : far;

	// One pass along each axis: lines of length len, nr_lines of them, the elements of line l at
	// offset(l) + q * stride
	const long int lens[3] = {xdim, ydim, zdim};
	const long int strides[3] = {1, xdim, xdim * ydim};
	for (int axis = 0; axis < 3; axis++)
	{
		const long int len = lens[axis], stride = strides[axis];
		if (len < 2)
			continue;
		const long int nr_lines = MULTIDIM_SIZE(dist) / len;
		#pragma omp parallel num_threads(n_threads)
		{
			std::vector<double> f(len), d(len), z(len + 1);
			std::vector<long int> v(len);
			#pragma omp for
			for (long int l = 0; l < nr_lines; l++)
			{
				// The first element of line l: the lines run along x, y or z
				long int offset;
				if (axis == 0)
					offset = l * xdim;
				else if (axis == 1)
					offset = (l / xdim) * xdim * ydim + l % xdim;
				else
					offset = l;
				for (long int q = 0; q < len; q++)
					f[q] = DIRECT_MULTIDIM_ELEM(dist, offset + q * stride);
				squaredDistance1D(f, d, v, z, len, far);
				for (long int q = 0; q < len; q++)
					DIRECT_MULTIDIM_ELEM(dist, offset + q * stride) = d[q];
			}
		}
	}

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(dist)
		DIRECT_MULTIDIM_ELEM(dist, n) = sqrt(DIRECT_MULTIDIM_ELEM(dist, n));
}

static inline RFLOAT softEdgeValue(RFLOAT d, RFLOAT width)
{
	if (d <= 0.)
		return 1.;
	if (width <= 0. || d > width)
		return 0.;
	return 0.5 + 0.5 * cos(PI * d / width);
}

void autoMaskSoftEdge(const MultidimArray<RFLOAT> &dist, RFLOAT width, MultidimArray<RFLOAT> &mask)
{
	mask.resize(dist);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(dist)
		DIRECT_MULTIDIM_ELEM(mask, n) = softEdgeValue(DIRECT_MULTIDIM_ELEM(dist, n), width);
}

// The distance to the binary mask, after growing or shrinking it by extend pixels
static void distanceToExtendedMask(MultidimArray<RFLOAT> &binary, RFLOAT extend,
		MultidimArray<RFLOAT> &dist, int n_threads)
{
	if (extend < 0.)
	{
		// Keep the voxels at least -extend from the outside
		MultidimArray<RFLOAT> outside(binary), dist_in;
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(binary)
			DIRECT_MULTIDIM_ELEM(outside, n) = (DIRECT_MULTIDIM_ELEM(binary, n) > 0.5) ? 0. : 1.;
		autoMaskDistance(outside, dist_in, n_threads);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(binary)
			if (DIRECT_MULTIDIM_ELEM(dist_in, n) < -extend)
				DIRECT_MULTIDIM_ELEM(binary, n) = 0.;
	}
	autoMaskDistance(binary, dist, n_threads);
	// The distance to the mask grown by extend is extend less than that to the mask
	if (extend > 0.)
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(dist)
			DIRECT_MULTIDIM_ELEM(dist, n) = XMIPP_MAX(0., DIRECT_MULTIDIM_ELEM(dist, n) - extend);
}

static void randomisePhasesFromShell(MultidimArray<RFLOAT> &map, long int shell, std::mt19937 &rng)
{
	const long int xdim = XSIZE(map), ydim = YSIZE(map), zdim = ZSIZE(map);
	std::uniform_real_distribution<double> phase(0., 2. * PI);
	MultidimArray<Complex> FT;
	FourierTransformer transformer;
	transformer.FourierTransform(map, FT, false);
	for (long int k = 0; k < ZSIZE(FT); k++)
	for (long int i = 0; i < YSIZE(FT); i++)
	for (long int j = 0; j < XSIZE(FT); j++)
	{
		if (fftwShell(k, i, j, xdim, ydim, zdim) < shell)
			continue;
		Complex &z = DIRECT_A3D_ELEM(FT, k, i, j);
		const double amplitude = sqrt(z.real * z.real + z.imag * z.imag);
		const double phi = phase(rng);
		z = Complex(amplitude * cos(phi), amplitude * sin(phi));
	}
	transformer.inverseFourierTransform();
}

// Minimise f on [a, b] to within xatol, as scipy's minimize_scalar(method="bounded")
static RFLOAT boundedBrent(const std::function<RFLOAT(RFLOAT)> &f, RFLOAT a, RFLOAT b, RFLOAT xatol, int max_evaluations = 500)
{
	const RFLOAT sqrt_eps = sqrt(2.2e-16);
	const RFLOAT golden_mean = 0.5 * (3. - sqrt(5.));
	RFLOAT fulc = a + golden_mean * (b - a);
	RFLOAT nfc = fulc, xf = fulc;
	RFLOAT rat = 0., e = 0.;
	RFLOAT fx = f(xf);
	int nr_evaluations = 1;
	RFLOAT ffulc = fx, fnfc = fx;
	RFLOAT xm = 0.5 * (a + b);
	RFLOAT tol1 = sqrt_eps * fabs(xf) + xatol / 3.;
	RFLOAT tol2 = 2. * tol1;

	while (fabs(xf - xm) > (tol2 - 0.5 * (b - a)))
	{
		bool golden = true;
		// A parabolic step?
		if (fabs(e) > tol1)
		{
			golden = false;
			RFLOAT r = (xf - nfc) * (fx - ffulc);
			RFLOAT q = (xf - fulc) * (fx - fnfc);
			RFLOAT p = (xf - fulc) * q - (xf - nfc) * r;
			q = 2. * (q - r);
			if (q > 0.)
				p = -p;
			q = fabs(q);
			r = e;
			e = rat;
			if (fabs(p) < fabs(0.5 * q * r) && p > q * (a - xf) && p < q * (b - xf))
			{
				rat = p / q;
				const RFLOAT x = xf + rat;
				if ((x - a) < tol2 || (b - x) < tol2)
					rat = (xm - xf >= 0.) ? tol1 : -tol1;
			}
			else
				golden = true;
		}
		if (golden)
		{
			e = (xf >= xm) ? a - xf : b - xf;
			rat = golden_mean * e;
		}
		const RFLOAT x = xf + ((rat >= 0.) ? 1. : -1.) * XMIPP_MAX(fabs(rat), tol1);
		const RFLOAT fu = f(x);
		nr_evaluations++;
		if (fu <= fx)
		{
			if (x >= xf)
				a = xf;
			else
				b = xf;
			fulc = nfc; ffulc = fnfc;
			nfc = xf; fnfc = fx;
			xf = x; fx = fu;
		}
		else
		{
			if (x < xf)
				a = x;
			else
				b = x;
			if (fu <= fnfc || nfc == xf)
			{
				fulc = nfc; ffulc = fnfc;
				nfc = x; fnfc = fu;
			}
			else if (fu <= ffulc || fulc == xf || fulc == nfc)
			{
				fulc = x; ffulc = fu;
			}
		}
		xm = 0.5 * (a + b);
		tol1 = sqrt_eps * fabs(xf) + xatol / 3.;
		tol2 = 2. * tol1;
		if (nr_evaluations >= max_evaluations)
			break;
	}
	return xf;
}

// The width of the soft edge (pixels) that best separates signal from noise: the FSC of the
// masked half-maps (fsc_t) against that of the masked phase-randomised half-maps (fsc_n) and
// the true FSC (fsc_t - fsc_n) / (1 - fsc_n), in shells first_shell to last_shell
static RFLOAT optimiseSoftEdge(const MultidimArray<RFLOAT> &half1, const MultidimArray<RFLOAT> &half2,
		const MultidimArray<RFLOAT> &dist, long int randomise_shell, long int first_shell, long int last_shell,
		RFLOAT angpix, bool verb)
{
	const long int xdim = XSIZE(half1), ydim = YSIZE(half1), zdim = ZSIZE(half1);

	// The same random phases every time, for the same mask from the same maps
	std::mt19937 rng(1993);
	MultidimArray<RFLOAT> rand1 = half1, rand2 = half2;
	randomisePhasesFromShell(rand1, randomise_shell, rng);
	randomisePhasesFromShell(rand2, randomise_shell, rng);

	MultidimArray<RFLOAT> m1, m2, fsc_t, fsc_n;
	m1.resize(half1);
	m2.resize(half1);
	MultidimArray<Complex> F1, F2;
	FourierTransformer transformer1, transformer2;

	auto maskedFsc = [&](const MultidimArray<RFLOAT> &a, const MultidimArray<RFLOAT> &b, RFLOAT width, MultidimArray<RFLOAT> &fsc)
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(m1)
		{
			const RFLOAT w = softEdgeValue(DIRECT_MULTIDIM_ELEM(dist, n), width);
			DIRECT_MULTIDIM_ELEM(m1, n) = w * DIRECT_MULTIDIM_ELEM(a, n);
			DIRECT_MULTIDIM_ELEM(m2, n) = w * DIRECT_MULTIDIM_ELEM(b, n);
		}
		transformer1.FourierTransform(m1, F1, false);
		transformer2.FourierTransform(m2, F2, false);
		fscOfTransforms(F1, F2, xdim, ydim, zdim, fsc);
	};

	auto score = [&](RFLOAT width) -> RFLOAT
	{
		maskedFsc(half1, half2, width, fsc_t);
		maskedFsc(rand1, rand2, width, fsc_n);
		double s1 = 0., s2 = 0., s3 = 0., s4 = 0.;
		long int nr_shells = 0, crossing = -1;
		for (long int r = first_shell; r <= last_shell; r++)
		{
			const RFLOAT t = DIRECT_A1D_ELEM(fsc_t, r), fn = DIRECT_A1D_ELEM(fsc_n, r);
			const RFLOAT tr = (fabs(1. - fn) > 1e-12) ? (t - fn) / (1. - fn) : 1.;
			s1 += 1. - fabs(tr);
			s2 += fabs(fn);
			s3 += fabs(t - tr);
			s4 += 1. - fabs(tr - fn);
			nr_shells++;
			if (crossing < 0 && tr < 0.143)
				crossing = r;
		}
		const RFLOAT value = (s1 + s2 + s3 + s4) / nr_shells;
		if (verb)
		{
			std::cout << "    soft edge " << width * angpix << " A (" << width << " pixels): ";
			if (crossing > 0)
				std::cout << "true FSC = 0.143 at shell " << crossing << " (" << xdim * angpix / crossing << " A); ";
			std::cout << "score " << value << std::endl;
		}
		return value;
	};

	return boundedBrent(score, 0., xdim / 3., 2.);
}

static RFLOAT roundedLowPass(RFLOAT res)
{
	if (res > 10.)
		return std::round(res);
	if (res > 5.)
		return std::round(res * 2.) / 2.;
	return std::round(res * 4.) / 4.;
}

void autoMaskFromHalfMaps(const MultidimArray<RFLOAT> &half1, const MultidimArray<RFLOAT> &half2,
		RFLOAT angpix, const AutoMaskOptions &opt, MultidimArray<RFLOAT> &mask,
		AutoMaskReport *report, int n_threads)
{
	if (!half1.sameShape(half2))
		REPORT_ERROR("autoMaskFromHalfMaps ERROR: the half-maps have different sizes");
	if (ZSIZE(half1) < 2)
		REPORT_ERROR("autoMaskFromHalfMaps ERROR: the maps are not 3D");

	AutoMaskReport rep;
	const long int box = XSIZE(half1);
	long int last_shell = box / 2;
	if (opt.max_shell > 0 && opt.max_shell < last_shell)
		last_shell = opt.max_shell;

	// FSC of the unmasked half-maps
	MultidimArray<RFLOAT> fsc;
	autoMaskFsc(half1, half2, fsc);
	for (long int r = last_shell + 1; r < XSIZE(fsc); r++)
		DIRECT_A1D_ELEM(fsc, r) = 0.;
	rep.resolution = autoMaskResolutionAtFsc(fsc, 0.143, box, angpix);
	if (rep.resolution < 0.)
		rep.resolution = box * angpix / last_shell;

	// Low-pass filter and phase randomisation from where the FSC drops below 0.8
	RFLOAT lowpass = opt.lowpass;
	if (lowpass <= 0.)
	{
		lowpass = autoMaskResolutionAtFsc(fsc, 0.8, box, angpix);
		lowpass = (lowpass < 0.) ? 2. * angpix : roundedLowPass(lowpass);
	}
	if (opt.verb)
	{
		std::cout << "  + auto mask: resolution at FSC = 0.143 of the unmasked half-maps: " << rep.resolution << " A" << std::endl;
		std::cout << "  + auto mask: low-pass filter (and phase randomisation from): " << lowpass << " A" << std::endl;
	}

	MultidimArray<RFLOAT> avg = half1, binary, dist;
	avg += half2;
	avg *= 0.5;
	autoMaskBinary(avg, angpix, lowpass, opt, binary, rep);
	avg.clear();
	distanceToExtendedMask(binary, opt.extend, dist, n_threads);
	binary.clear();

	RFLOAT width = opt.soft_edge;
	const long int randomise_shell = (long int)(box * angpix / lowpass);
	const long int first_shell = randomise_shell + 2;
	if (width < 0. && first_shell < last_shell)
	{
		if (opt.verb)
			std::cout << "  + auto mask: searching for the soft edge that best separates signal from noise ..." << std::endl;
		width = optimiseSoftEdge(half1, half2, dist, randomise_shell, first_shell, last_shell, angpix, opt.verb);
		rep.soft_edge_optimised = true;
	}
	else if (width < 0.)
	{
		if (opt.verb)
			std::cout << "  + auto mask: too few shells beyond the phase randomisation to optimise the soft edge: 3 times the resolution" << std::endl;
		width = 3. * rep.resolution / angpix;
	}
	width = XMIPP_MIN(XMIPP_MAX(width, 0.), box / 3.);
	rep.soft_edge = width;
	if (opt.verb)
		std::cout << "  + auto mask soft edge: " << width * angpix << " A (" << width << " pixels)" << std::endl;

	autoMaskSoftEdge(dist, width, mask);
	mask.setXmippOrigin();
	if (report != NULL)
		*report = rep;
}

void autoMaskFromMap(const MultidimArray<RFLOAT> &map, RFLOAT angpix, RFLOAT resolution,
		const AutoMaskOptions &opt, MultidimArray<RFLOAT> &mask,
		AutoMaskReport *report, int n_threads)
{
	if (ZSIZE(map) < 2)
		REPORT_ERROR("autoMaskFromMap ERROR: the map is not 3D");

	AutoMaskReport rep;
	rep.resolution = resolution;
	MultidimArray<RFLOAT> binary, dist;
	autoMaskBinary(map, angpix, opt.lowpass, opt, binary, rep);
	distanceToExtendedMask(binary, opt.extend, dist, n_threads);
	binary.clear();

	RFLOAT width = opt.soft_edge;
	if (width < 0.)
		width = (resolution > 0.) ? 3. * resolution / angpix : 0.;
	width = XMIPP_MIN(XMIPP_MAX(width, 0.), XSIZE(map) / 3.);
	rep.soft_edge = width;
	if (opt.verb)
		std::cout << "  + auto mask soft edge: " << width * angpix << " A (" << width << " pixels)" << std::endl;

	autoMaskSoftEdge(dist, width, mask);
	mask.setXmippOrigin();
	if (report != NULL)
		*report = rep;
}
