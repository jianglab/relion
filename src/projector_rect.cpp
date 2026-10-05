/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

/* Projector for rectangular images and cuboid volumes.
 *
 * Everything that has to know the per-axis size lives here, so the square code in
 * projector.cpp stays untouched.
 *
 * Geometry: ori_size is the nominal size L = max(nx, ny, nz). A Fourier index
 * along an axis of n pixels is turned into nominal "shell units" by multiplying
 * with L/n; rotations happen in shell units (where space is isotropic), and the
 * rotated position is turned back into a data-array index by multiplying with
 * n_ref/L. For nx == ny == nz == L every factor is 1.
 */

#include "src/projector.h"
#include "src/fftw_rect.h"

void Projector::setBoxSize(int nx, int ny, int nz)
{
	if (nx < 1 || ny < 1 || nz < 0)
		REPORT_ERROR("Projector::setBoxSize: invalid box size");
	int L = XMIPP_MAX(nx, ny);
	if (nz > 0)
		L = XMIPP_MAX(L, nz);
	ori_size = L;
	rect = (nx != ny) || (nz > 0 && nz != nx);
	if (rect)
	{
		box_nx = nx;
		box_ny = ny;
		box_nz = (nz > 0) ? nz : 1;
	}
	else
	{
		box_nx = box_ny = box_nz = 0;
	}
}

void Projector::initialiseDataRect(int current_size)
{
	if (current_size < 0)
		r_max = ori_size / 2;
	else
		r_max = current_size / 2;
	r_max = XMIPP_MIN(r_max, ori_size / 2);

	pad_size = 2 * (ROUND(padding_factor * r_max) + 1) + 1;

	const int psx = 2 * (ROUND(padding_factor * r_max * box_nx / ori_size) + 1) + 1;
	const int psy = 2 * (ROUND(padding_factor * r_max * box_ny / ori_size) + 1) + 1;
	const int psz = 2 * (ROUND(padding_factor * r_max * box_nz / ori_size) + 1) + 1;

	switch (ref_dim)
	{
	case 2:
		data.resize(psy, psx / 2 + 1);
		break;
	case 3:
		data.resize(psz, psy, psx / 2 + 1);
		break;
	default:
		REPORT_ERROR("Projector::initialiseDataRect: dimension of the data array should be 2 or 3");
	}
	data.setXmippOrigin();
	data.xinit = 0;
}

void Projector::griddingCorrectRect(MultidimArray<RFLOAT> &vol_in)
{
	vol_in.setXmippOrigin();
	if (interpolator == FINUFFT)
		return;

	const RFLOAT ix = 1. / box_nx, iy = 1. / box_ny, iz = (ZSIZE(vol_in) > 1) ? 1. / box_nz : 0.;
	FOR_ALL_ELEMENTS_IN_ARRAY3D(vol_in)
	{
		const RFLOAT fx = j * ix, fy = i * iy, fz = k * iz;
		const RFLOAT r = sqrt(fx*fx + fy*fy + fz*fz);
		if (r > 0.)
		{
			// the same radial sinc as the square code, in units where the box side is 1
			const RFLOAT rval = r / padding_factor;
			const RFLOAT sinc = sin(PI * rval) / (PI * rval);
			if (interpolator == NEAREST_NEIGHBOUR && r_min_nn == 0)
				A3D_ELEM(vol_in, k, i, j) /= sinc;
			else if (interpolator == TRILINEAR || (interpolator == NEAREST_NEIGHBOUR && r_min_nn > 0))
				A3D_ELEM(vol_in, k, i, j) /= sinc * sinc;
			else
				REPORT_ERROR("BUG Projector::griddingCorrectRect: unrecognised interpolator scheme.");
		}
	}
}

void Projector::computeFourierTransformMapRect(MultidimArray<RFLOAT> &vol_in, MultidimArray<RFLOAT> &power_spectrum,
                                               int current_size, bool do_gridding, bool do_heavy, int min_ires,
                                               const MultidimArray<RFLOAT> *fourier_mask)
{
	if (interpolator == FINUFFT)
		REPORT_ERROR("RELION_INTERPOLATION=nufft is not supported for rectangular boxes yet.");

	ref_dim = vol_in.getDim();
	if (ref_dim != 2 && ref_dim != 3)
		REPORT_ERROR("Projector::computeFourierTransformMapRect: dimension of the data array should be 2 or 3");
	if (XSIZE(vol_in) != box_nx || YSIZE(vol_in) != box_ny || (ref_dim == 3 && ZSIZE(vol_in) != box_nz))
		REPORT_ERROR("Projector::computeFourierTransformMapRect: the map is " + integerToString(XSIZE(vol_in)) + "x" +
		             integerToString(YSIZE(vol_in)) + "x" + integerToString(ZSIZE(vol_in)) + " but the box is " +
		             integerToString(box_nx) + "x" + integerToString(box_ny) + "x" + integerToString(box_nz));

	// The padded box must have an even integer size on every axis
	const int nax[3] = { box_nx, box_ny, ref_dim == 3 ? box_nz : 1 };
	int pn[3];
	for (int a = 0; a < 3; a++)
	{
		pn[a] = (a == 2 && ref_dim == 2) ? 1 : ROUND(padding_factor * nax[a]);
		if (!(a == 2 && ref_dim == 2) && (pn[a] % 2 != 0 || std::abs(padding_factor * nax[a] - pn[a]) > 1e-4))
			REPORT_ERROR("Projector: padding factor " + floatToString(padding_factor) + " times box size " + integerToString(nax[a]) +
			             " must be an even integer for a rectangular box.");
	}
	padded_real_size = XMIPP_MAX(pn[0], pn[1]);

	RFLOAT normfft;
	if (ref_dim == 2)
	{
		if (data_dim != 2)
			REPORT_ERROR("Projector: rectangular boxes support 2D references only for 2D images.");
		normfft = (RFLOAT)(padding_factor * padding_factor);
	}
	else
	{
		normfft = (data_dim == 3) ? (RFLOAT)(padding_factor * padding_factor * padding_factor)
		                          : (RFLOAT)(padding_factor * padding_factor * padding_factor * box_nx);
	}

	MultidimArray<RFLOAT> Mpad;
	MultidimArray<Complex> Faux;
	FourierTransformer transformer;

	if (do_gridding)
	{
		if (do_heavy)
			griddingCorrectRect(vol_in);
		else
			vol_in.setXmippOrigin();
	}
	else
		vol_in.setXmippOrigin();

	if (ref_dim == 2)
		Mpad.initZeros(pn[1], pn[0]);
	else
		Mpad.initZeros(pn[2], pn[1], pn[0]);
	Mpad.setXmippOrigin();
	if (do_heavy)
	{
		FOR_ALL_ELEMENTS_IN_ARRAY3D(vol_in)
			A3D_ELEM(Mpad, k, i, j) = A3D_ELEM(vol_in, k, i, j);
		transformer.FourierTransform(Mpad, Faux, false);
		CenterFFTbySign(Faux);
	}
	Mpad.clear();

	initZeros(current_size);

	power_spectrum.initZeros(ori_size / 2 + 1);
	MultidimArray<RFLOAT> counter(power_spectrum);
	counter.initZeros();

	const RFLOAT max_r = ROUND(r_max * padding_factor);
	const RFLOAT max_r2 = max_r * max_r;
	const RFLOAT min_r = (min_ires > 0) ? ROUND(min_ires * padding_factor) : -1.;
	const RFLOAT min_r2 = (min_ires > 0) ? min_r * min_r : -1.;
	// padded index * (L / n_axis) is the shell in units of 1/padding
	const RFLOAT px = shellScaleX(), py = shellScaleY(), pz = (ref_dim == 3) ? shellScaleZ() : 0.;

	if (do_heavy)
	{
		RFLOAT weight = 1.;
		const bool do_fourier_mask = (fourier_mask != NULL);
		FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM_RECT(Faux)
		{
			const RFLOAT ux = jp * px, uy = ip * py, uz = kp * pz;
			const RFLOAT r2 = ux*ux + uy*uy + uz*uz;
			if (r2 <= max_r2)
			{
				if (do_fourier_mask)
					weight = FFTW_ELEM(*fourier_mask, ROUND(kp / padding_factor), ROUND(ip / padding_factor), ROUND(jp / padding_factor));
				A3D_ELEM(data, kp, ip, jp) = weight * DIRECT_A3D_ELEM(Faux, k, i, j) * normfft;

				const int ires = ROUND(sqrt(r2) / padding_factor);
				DIRECT_A1D_ELEM(power_spectrum, ires) += norm(A3D_ELEM(data, kp, ip, jp)) / 2.;
				DIRECT_A1D_ELEM(counter, ires) += weight;

				if (r2 <= min_r2)
					A3D_ELEM(data, kp, ip, jp) = 0;
			}
		}
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(power_spectrum)
		{
			if (DIRECT_A1D_ELEM(counter, i) < 1.)
				DIRECT_A1D_ELEM(power_spectrum, i) = 0.;
			else
				DIRECT_A1D_ELEM(power_spectrum, i) /= DIRECT_A1D_ELEM(counter, i);
		}
	}
}

namespace
{
// Trilinear value of a half-stored Fourier array at a position given in its own index units,
// using Hermitian symmetry for negative x. Returns false when the stencil falls outside the array.
inline bool interpolate3D(const MultidimArray<Complex> &data, RFLOAT xd, RFLOAT yd, RFLOAT zd, Complex &out)
{
	const bool neg = (xd < 0);
	if (neg) { xd = -xd; yd = -yd; zd = -zd; }
	const int x0 = FLOOR(xd);
	const RFLOAT fx = xd - x0;
	int y0 = FLOOR(yd);
	const RFLOAT fy = yd - y0;
	y0 -= STARTINGY(data);
	int z0 = FLOOR(zd);
	const RFLOAT fz = zd - z0;
	z0 -= STARTINGZ(data);
	if (x0 < 0 || x0 + 1 >= data.xdim || y0 < 0 || y0 + 1 >= data.ydim || z0 < 0 || z0 + 1 >= data.zdim)
		return false;
	const Complex d000 = DIRECT_A3D_ELEM(data, z0, y0, x0),     d001 = DIRECT_A3D_ELEM(data, z0, y0, x0 + 1);
	const Complex d010 = DIRECT_A3D_ELEM(data, z0, y0 + 1, x0), d011 = DIRECT_A3D_ELEM(data, z0, y0 + 1, x0 + 1);
	const Complex d100 = DIRECT_A3D_ELEM(data, z0 + 1, y0, x0), d101 = DIRECT_A3D_ELEM(data, z0 + 1, y0, x0 + 1);
	const Complex d110 = DIRECT_A3D_ELEM(data, z0 + 1, y0 + 1, x0), d111 = DIRECT_A3D_ELEM(data, z0 + 1, y0 + 1, x0 + 1);
	const Complex dx00 = LIN_INTERP(fx, d000, d001), dx01 = LIN_INTERP(fx, d100, d101);
	const Complex dx10 = LIN_INTERP(fx, d010, d011), dx11 = LIN_INTERP(fx, d110, d111);
	const Complex dxy0 = LIN_INTERP(fy, dx00, dx10), dxy1 = LIN_INTERP(fy, dx01, dx11);
	out = LIN_INTERP(fz, dxy0, dxy1);
	if (neg) out = conj(out);
	return true;
}

inline bool interpolate2D(const MultidimArray<Complex> &data, RFLOAT xd, RFLOAT yd, Complex &out)
{
	const bool neg = (xd < 0);
	if (neg) { xd = -xd; yd = -yd; }
	const int x0 = FLOOR(xd);
	const RFLOAT fx = xd - x0;
	int y0 = FLOOR(yd);
	const RFLOAT fy = yd - y0;
	y0 -= STARTINGY(data);
	if (x0 < 0 || x0 + 1 >= data.xdim || y0 < 0 || y0 + 1 >= data.ydim)
		return false;
	const Complex d00 = DIRECT_A2D_ELEM(data, y0, x0), d01 = DIRECT_A2D_ELEM(data, y0, x0 + 1);
	const Complex d10 = DIRECT_A2D_ELEM(data, y0 + 1, x0), d11 = DIRECT_A2D_ELEM(data, y0 + 1, x0 + 1);
	const Complex dx0 = LIN_INTERP(fx, d00, d01), dx1 = LIN_INTERP(fx, d10, d11);
	out = LIN_INTERP(fy, dx0, dx1);
	if (neg) out = conj(out);
	return true;
}
}

void Projector::projectRect(MultidimArray<Complex> &f2d, Matrix2D<RFLOAT> &A)
{
	if (interpolator != TRILINEAR)
		REPORT_ERROR("Projector::project: rectangular boxes support the linear interpolator only (unset RELION_INTERPOLATION).");

	Matrix2D<RFLOAT> Ainv = A.inv();
	Ainv *= (RFLOAT)padding_factor;

	const RFLOAT sx = imgScaleX(), sy = imgScaleY();
	// shell units -> index in the (padded) data array, per axis
	const RFLOAT tx = 1. / shellScaleX(), ty = 1. / shellScaleY(), tz = 1. / shellScaleZ();

	const RFLOAT xmax = XSIZE(f2d) - 1, ymax = XMIPP_MAX(YSIZE(f2d) / 2, 1);
	const RFLOAT r_max_ref = r_max * padding_factor;
	const RFLOAT r_max_ref_2 = r_max_ref * r_max_ref;

	for (int i = 0; i < YSIZE(f2d); i++)
	{
		const int y = (i <= YSIZE(f2d) / 2) ? i : i - YSIZE(f2d);
		const RFLOAT ey = y / ymax;
		const RFLOAT frac = 1. - ey * ey;
		if (frac < 0.) continue;
		const int x_max = FLOOR(xmax * sqrt(frac));
		const RFLOAT uy = y * sy;
		for (int x = 0; x <= x_max; x++)
		{
			const RFLOAT ux = x * sx;
			const RFLOAT xp = Ainv(0,0) * ux + Ainv(0,1) * uy;
			const RFLOAT yp = Ainv(1,0) * ux + Ainv(1,1) * uy;
			const RFLOAT zp = Ainv(2,0) * ux + Ainv(2,1) * uy;
			if (xp*xp + yp*yp + zp*zp > r_max_ref_2) continue;

			Complex v;
			if (interpolate3D(data, xp * tx, yp * ty, zp * tz, v))
				DIRECT_A2D_ELEM(f2d, i, x) = v;
		}
	}
}

void Projector::rotate2DRect(MultidimArray<Complex> &f2d, Matrix2D<RFLOAT> &A)
{
	if (interpolator != TRILINEAR)
		REPORT_ERROR("Projector::rotate2D: rectangular boxes support the linear interpolator only (unset RELION_INTERPOLATION).");

	Matrix2D<RFLOAT> Ainv = A.inv();
	Ainv *= (RFLOAT)padding_factor;

	const RFLOAT sx = imgScaleX(), sy = imgScaleY();
	const RFLOAT tx = 1. / shellScaleX(), ty = 1. / shellScaleY();

	const RFLOAT xmax = XSIZE(f2d) - 1, ymax = XMIPP_MAX(YSIZE(f2d) / 2, 1);
	const RFLOAT r_max_ref = r_max * padding_factor;
	const RFLOAT r_max_ref_2 = r_max_ref * r_max_ref;

	for (int i = 0; i < YSIZE(f2d); i++)
	{
		const int y = (i <= YSIZE(f2d) / 2) ? i : i - YSIZE(f2d);
		const RFLOAT ey = y / ymax;
		const RFLOAT frac = 1. - ey * ey;
		if (frac < 0.) continue;
		const int x_max = FLOOR(xmax * sqrt(frac));
		const RFLOAT uy = y * sy;
		for (int x = 0; x <= x_max; x++)
		{
			const RFLOAT ux = x * sx;
			const RFLOAT xp = Ainv(0,0) * ux + Ainv(0,1) * uy;
			const RFLOAT yp = Ainv(1,0) * ux + Ainv(1,1) * uy;
			if (xp*xp + yp*yp > r_max_ref_2) continue;

			Complex v;
			if (interpolate2D(data, xp * tx, yp * ty, v))
				DIRECT_A2D_ELEM(f2d, i, x) = v;
		}
	}
}
