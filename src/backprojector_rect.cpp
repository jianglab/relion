/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

/* BackProjector for rectangular images and cuboid volumes.
 *
 * Mirrors projector_rect.cpp. Everything that has to know the per-axis size
 * lives here; the square code in backprojector.cpp only has a dispatch line at
 * the top of each entry point and is otherwise untouched.
 *
 * Geometry (see projector_rect.cpp): ori_size is the nominal size
 * L = max(nx, ny, nz). An index along an axis of the (padded) data array is
 * turned into "shell units" (isotropic, nominal pixels of the L box, times the
 * padding factor) by multiplying with shellScale = L / n_axis. Rotations and
 * radius tests happen in shell units; the rotated position is converted back
 * to an array index by dividing by the same scale. Per-shell spectra keep their
 * L/2+1 entries. For nx == ny == nz == L every scale is 1.
 */

#include <cstring>
#include <vector>
#include "src/backprojector.h"
#include "src/fftw_rect.h"
#include "src/mask.h"

namespace
{
// Shell-unit scale of the three axes of the data array (z is 0 for a 2D reference)
struct AxisScale
{
	double x, y, z;
	double r2(long int k, long int i, long int j) const
	{
		const double ux = j * x, uy = i * y, uz = k * z;
		return ux*ux + uy*uy + uz*uz;
	}
};

// Trilinear splat of one complex value and weight into half-stored Fourier arrays,
// using Hermitian symmetry for negative x. Positions are array indices.
inline void splat3D(MultidimArray<Complex> &data, MultidimArray<RFLOAT> &weight,
                    RFLOAT xp, RFLOAT yp, RFLOAT zp, Complex my_val, RFLOAT my_weight)
{
	const bool is_neg_x = (xp < 0);
	if (is_neg_x) { xp = -xp; yp = -yp; zp = -zp; }

	const int x0 = FLOOR(xp);
	const RFLOAT fx = xp - x0;
	const int x1 = x0 + 1;
	int y0 = FLOOR(yp);
	const RFLOAT fy = yp - y0;
	y0 -= STARTINGY(data);
	const int y1 = y0 + 1;
	int z0 = FLOOR(zp);
	const RFLOAT fz = zp - z0;
	z0 -= STARTINGZ(data);
	const int z1 = z0 + 1;

	if (x0 < 0 || x0 + 1 >= data.xdim || y0 < 0 || y0 + 1 >= data.ydim || z0 < 0 || z0 + 1 >= data.zdim)
		return;

	const RFLOAT mfx = 1. - fx, mfy = 1. - fy, mfz = 1. - fz;
	const RFLOAT dd000 = mfz * mfy * mfx, dd001 = mfz * mfy * fx;
	const RFLOAT dd010 = mfz * fy * mfx,  dd011 = mfz * fy * fx;
	const RFLOAT dd100 = fz * mfy * mfx,  dd101 = fz * mfy * fx;
	const RFLOAT dd110 = fz * fy * mfx,   dd111 = fz * fy * fx;

	if (is_neg_x) my_val = conj(my_val);

	DIRECT_A3D_ELEM(data, z0, y0, x0) += dd000 * my_val;
	DIRECT_A3D_ELEM(data, z0, y0, x1) += dd001 * my_val;
	DIRECT_A3D_ELEM(data, z0, y1, x0) += dd010 * my_val;
	DIRECT_A3D_ELEM(data, z0, y1, x1) += dd011 * my_val;
	DIRECT_A3D_ELEM(data, z1, y0, x0) += dd100 * my_val;
	DIRECT_A3D_ELEM(data, z1, y0, x1) += dd101 * my_val;
	DIRECT_A3D_ELEM(data, z1, y1, x0) += dd110 * my_val;
	DIRECT_A3D_ELEM(data, z1, y1, x1) += dd111 * my_val;

	DIRECT_A3D_ELEM(weight, z0, y0, x0) += dd000 * my_weight;
	DIRECT_A3D_ELEM(weight, z0, y0, x1) += dd001 * my_weight;
	DIRECT_A3D_ELEM(weight, z0, y1, x0) += dd010 * my_weight;
	DIRECT_A3D_ELEM(weight, z0, y1, x1) += dd011 * my_weight;
	DIRECT_A3D_ELEM(weight, z1, y0, x0) += dd100 * my_weight;
	DIRECT_A3D_ELEM(weight, z1, y0, x1) += dd101 * my_weight;
	DIRECT_A3D_ELEM(weight, z1, y1, x0) += dd110 * my_weight;
	DIRECT_A3D_ELEM(weight, z1, y1, x1) += dd111 * my_weight;
}

inline void splat2D(MultidimArray<Complex> &data, MultidimArray<RFLOAT> &weight,
                    RFLOAT xp, RFLOAT yp, Complex my_val, RFLOAT my_weight)
{
	const bool is_neg_x = (xp < 0);
	if (is_neg_x) { xp = -xp; yp = -yp; }

	const int x0 = FLOOR(xp);
	const RFLOAT fx = xp - x0;
	const int x1 = x0 + 1;
	int y0 = FLOOR(yp);
	const RFLOAT fy = yp - y0;
	y0 -= STARTINGY(data);
	const int y1 = y0 + 1;

	if (x0 < 0 || x0 + 1 >= data.xdim || y0 < 0 || y0 + 1 >= data.ydim)
		return;

	const RFLOAT mfx = 1. - fx, mfy = 1. - fy;
	const RFLOAT dd00 = mfy * mfx, dd01 = mfy * fx, dd10 = fy * mfx, dd11 = fy * fx;

	if (is_neg_x) my_val = conj(my_val);

	DIRECT_A2D_ELEM(data, y0, x0) += dd00 * my_val;
	DIRECT_A2D_ELEM(data, y0, x1) += dd01 * my_val;
	DIRECT_A2D_ELEM(data, y1, x0) += dd10 * my_val;
	DIRECT_A2D_ELEM(data, y1, x1) += dd11 * my_val;

	DIRECT_A2D_ELEM(weight, y0, x0) += dd00 * my_weight;
	DIRECT_A2D_ELEM(weight, y0, x1) += dd01 * my_weight;
	DIRECT_A2D_ELEM(weight, y1, x0) += dd10 * my_weight;
	DIRECT_A2D_ELEM(weight, y1, x1) += dd11 * my_weight;
}

// Trilinear read of the half-stored data and weight arrays at an array-index position.
// Returns false when the stencil falls outside the arrays.
inline bool sample3D(const MultidimArray<Complex> &data, const MultidimArray<RFLOAT> &weight,
                     RFLOAT xp, RFLOAT yp, RFLOAT zp, Complex &out_val, RFLOAT &out_w)
{
	const bool is_neg_x = (xp < 0);
	if (is_neg_x) { xp = -xp; yp = -yp; zp = -zp; }

	const int x0 = FLOOR(xp);
	const RFLOAT fx = xp - x0;
	const int x1 = x0 + 1;
	int y0 = FLOOR(yp);
	const RFLOAT fy = yp - y0;
	y0 -= STARTINGY(data);
	const int y1 = y0 + 1;
	int z0 = FLOOR(zp);
	const RFLOAT fz = zp - z0;
	z0 -= STARTINGZ(data);
	const int z1 = z0 + 1;

	if (x0 < 0 || x1 >= data.xdim || y0 < 0 || y1 >= data.ydim || z0 < 0 || z1 >= data.zdim)
		return false;

	const Complex d000 = DIRECT_A3D_ELEM(data, z0, y0, x0), d001 = DIRECT_A3D_ELEM(data, z0, y0, x1);
	const Complex d010 = DIRECT_A3D_ELEM(data, z0, y1, x0), d011 = DIRECT_A3D_ELEM(data, z0, y1, x1);
	const Complex d100 = DIRECT_A3D_ELEM(data, z1, y0, x0), d101 = DIRECT_A3D_ELEM(data, z1, y0, x1);
	const Complex d110 = DIRECT_A3D_ELEM(data, z1, y1, x0), d111 = DIRECT_A3D_ELEM(data, z1, y1, x1);
	const Complex dx00 = LIN_INTERP(fx, d000, d001), dx01 = LIN_INTERP(fx, d100, d101);
	const Complex dx10 = LIN_INTERP(fx, d010, d011), dx11 = LIN_INTERP(fx, d110, d111);
	const Complex dxy0 = LIN_INTERP(fy, dx00, dx10), dxy1 = LIN_INTERP(fy, dx01, dx11);
	out_val = LIN_INTERP(fz, dxy0, dxy1);
	if (is_neg_x) out_val = conj(out_val);

	const RFLOAT w000 = DIRECT_A3D_ELEM(weight, z0, y0, x0), w001 = DIRECT_A3D_ELEM(weight, z0, y0, x1);
	const RFLOAT w010 = DIRECT_A3D_ELEM(weight, z0, y1, x0), w011 = DIRECT_A3D_ELEM(weight, z0, y1, x1);
	const RFLOAT w100 = DIRECT_A3D_ELEM(weight, z1, y0, x0), w101 = DIRECT_A3D_ELEM(weight, z1, y0, x1);
	const RFLOAT w110 = DIRECT_A3D_ELEM(weight, z1, y1, x0), w111 = DIRECT_A3D_ELEM(weight, z1, y1, x1);
	const RFLOAT wx00 = LIN_INTERP(fx, w000, w001), wx01 = LIN_INTERP(fx, w100, w101);
	const RFLOAT wx10 = LIN_INTERP(fx, w010, w011), wx11 = LIN_INTERP(fx, w110, w111);
	const RFLOAT wxy0 = LIN_INTERP(fy, wx00, wx10), wxy1 = LIN_INTERP(fy, wx01, wx11);
	out_w = LIN_INTERP(fz, wxy0, wxy1);
	return true;
}

// From the shell-unit position of an FFTW-layout array, put the value at the matching
// element of a padded-centred (Projector-style) array. Template on the output type so the
// double-precision weight can be used in single-precision builds.
template <typename T1, typename T2>
void decenterRect(MultidimArray<T1> &Min, MultidimArray<T2> &Mout, double my_rmax2, const AxisScale &sc)
{
	const long int nz = ZSIZE(Mout), ny = YSIZE(Mout), nx = XSIZE(Mout);
	#pragma omp parallel for
	for (long int k = 0; k < nz; k++)
	{
		const long int kp = (k <= nz / 2) ? k : k - nz;
		std::memset((void*)&DIRECT_A3D_ELEM(Mout, k, 0, 0), 0, sizeof(T2) * ny * nx);
		for (long int i = 0; i < ny; i++)
		{
			const long int ip = (i <= ny / 2) ? i : i - ny;
			for (long int j = 0; j < nx; j++)
			{
				if (sc.r2(kp, ip, j) <= my_rmax2)
					DIRECT_A3D_ELEM(Mout, k, i, j) = (T2)A3D_ELEM(Min, kp, ip, j);
			}
		}
	}
}

// CenterFFTbySign() with the planes shared between the threads
static void centerFFTbySignParallel(MultidimArray<Complex> &v)
{
	const long int nz = ZSIZE(v), ny = YSIZE(v), nx = XSIZE(v);
	#pragma omp parallel for
	for (long int k = 0; k < nz; k++)
		for (long int i = 0; i < ny; i++)
			for (long int j = 0; j < nx; j++)
				if (((k ^ i ^ j) & 1) != 0)
					DIRECT_A3D_ELEM(v, k, i, j) *= -1;
}

// Soft mask outside the ellipse (ellipsoid) inscribed in the box, i.e. the radial mask of
// softMaskOutsideMap() in shell units. For a square box it is the same as softMaskOutsideMap().
void softMaskOutsideMapRect(MultidimArray<RFLOAT> &vol, int L, int nx, int ny, int nz)
{
	const RFLOAT cosine_width = 3.;
	const RFLOAT radius = (RFLOAT)L / 2., radius_p = radius + cosine_width;
	const RFLOAT sx = (RFLOAT)L / nx, sy = (RFLOAT)L / ny, sz = (ZSIZE(vol) > 1) ? (RFLOAT)L / nz : 0.;
	vol.setXmippOrigin();

	RFLOAT sum_bg = 0., sum = 0.;
	FOR_ALL_ELEMENTS_IN_ARRAY3D(vol)
	{
		const RFLOAT ux = j * sx, uy = i * sy, uz = k * sz;
		const RFLOAT r = sqrt(ux*ux + uy*uy + uz*uz);
		if (r < radius)
			continue;
		else if (r > radius_p)
		{
			sum += 1.;
			sum_bg += A3D_ELEM(vol, k, i, j);
		}
		else
		{
			const RFLOAT raisedcos = 0.5 + 0.5 * cos(PI * (radius_p - r) / cosine_width);
			sum += raisedcos;
			sum_bg += raisedcos * A3D_ELEM(vol, k, i, j);
		}
	}
	sum_bg /= sum;

	FOR_ALL_ELEMENTS_IN_ARRAY3D(vol)
	{
		const RFLOAT ux = j * sx, uy = i * sy, uz = k * sz;
		const RFLOAT r = sqrt(ux*ux + uy*uy + uz*uz);
		if (r < radius)
			continue;
		else if (r > radius_p)
			A3D_ELEM(vol, k, i, j) = sum_bg;
		else
		{
			const RFLOAT raisedcos = 0.5 + 0.5 * cos(PI * (radius_p - r) / cosine_width);
			A3D_ELEM(vol, k, i, j) = (1 - raisedcos) * A3D_ELEM(vol, k, i, j) + raisedcos * sum_bg;
		}
	}
}
} // namespace

// ---------------------------------------------------------------------------------------

void BackProjector::checkRectBox(const char *who) const
{
	if (!rect)
		REPORT_ERROR(std::string(who) + ": internal error, called for a square box");
	if (interpolator != TRILINEAR)
		REPORT_ERROR(std::string(who) + ": rectangular boxes support the linear interpolator only (unset RELION_INTERPOLATION).");
	if (ref_dim != 2 && ref_dim != 3)
		REPORT_ERROR(std::string(who) + ": dimension of the data array should be 2 or 3");
	if (ref_dim == 2 && data_dim != 2)
		REPORT_ERROR(std::string(who) + ": rectangular boxes support 2D references only with 2D images.");
	if (ref_dim == 3 && data_dim != 2 && data_dim != 3)
		REPORT_ERROR(std::string(who) + ": rectangular boxes support 3D references only with 2D images or 3D maps.");
	const int nax[3] = { box_nx, box_ny, ref_dim == 3 ? box_nz : 1 };
	for (int a = 0; a < (ref_dim == 3 ? 3 : 2); a++)
	{
		const int pn = ROUND(padding_factor * nax[a]);
		if (pn % 2 != 0 || std::abs(padding_factor * nax[a] - pn) > 1e-4)
			REPORT_ERROR(std::string(who) + ": padding factor " + floatToString(padding_factor) + " times box size " +
			             integerToString(nax[a]) + " must be an even integer for a rectangular box.");
	}
}

// ---------------------------------------------------------------------------------------
// Backprojection / rotation of Fourier transforms into the data array

void BackProjector::backproject2Dto3DRect(const MultidimArray<Complex> &f2d, const Matrix2D<RFLOAT> &A,
                                          const MultidimArray<RFLOAT> *Mweight, RFLOAT r_ewald_sphere,
                                          bool is_positive_curvature, Matrix2D<RFLOAT> *magMatrix)
{
	checkRectBox("BackProjector::backproject2Dto3D");
	if (ref_dim != 3 || data_dim != 2)
		REPORT_ERROR("BackProjector::backproject2Dto3D: rectangular box needs a 3D reference and 2D images");

	RFLOAT m00 = 1., m10 = 0., m01 = 0., m11 = 1.;
	if (magMatrix != 0)
	{
		m00 = (*magMatrix)(0,0); m10 = (*magMatrix)(1,0);
		m01 = (*magMatrix)(0,1); m11 = (*magMatrix)(1,1);
	}

	Matrix2D<RFLOAT> Ainv = A.inv();
	Ainv *= (RFLOAT)padding_factor;

	const RFLOAT sx = imgScaleX(), sy = imgScaleY();
	const RFLOAT tx = 1. / shellScaleX(), ty = 1. / shellScaleY(), tz = 1. / shellScaleZ();

	const int max_r2 = ROUND(r_max * padding_factor) * ROUND(r_max * padding_factor);

	// ellipse of the image that corresponds to the sphere in 3D (shell units), see backproject2Dto3D
	const RFLOAT Am_Xx = (Ainv(0,0) * m00 + Ainv(0,1) * m10) * sx;
	const RFLOAT Am_Xy = (Ainv(0,0) * m01 + Ainv(0,1) * m11) * sy;
	const RFLOAT Am_Yx = (Ainv(1,0) * m00 + Ainv(1,1) * m10) * sx;
	const RFLOAT Am_Yy = (Ainv(1,0) * m01 + Ainv(1,1) * m11) * sy;
	const RFLOAT Am_Zx = (Ainv(2,0) * m00 + Ainv(2,1) * m10) * sx;
	const RFLOAT Am_Zy = (Ainv(2,0) * m01 + Ainv(2,1) * m11) * sy;
	const RFLOAT AtA_xx = Am_Xx * Am_Xx + Am_Yx * Am_Yx + Am_Zx * Am_Zx;
	const RFLOAT AtA_xy = Am_Xx * Am_Xy + Am_Yx * Am_Yy + Am_Zx * Am_Zy;
	const RFLOAT AtA_yy = Am_Xy * Am_Xy + Am_Yy * Am_Yy + Am_Zy * Am_Zy;
	const RFLOAT AtA_xy2 = AtA_xy * AtA_xy;

	RFLOAT inv_diam_ewald = (r_ewald_sphere > 0.0) ? 1.0 / (2.0 * r_ewald_sphere) : 0.0;
	if (!is_positive_curvature) inv_diam_ewald *= -1.0;

	const int s = YSIZE(f2d);
	const int sh = XSIZE(f2d);

	for (int i = 0; i < s; i++)
	{
		int y, first_allowed_x;
		if (i <= s / 2)
		{
			y = i;
			first_allowed_x = 0;
		}
		else
		{
			y = i - s;
			first_allowed_x = 1; // x == 0 is stored twice in the FFTW format
		}

		const RFLOAT discr = AtA_xy2 * y * y - AtA_xx * (AtA_yy * y * y - max_r2);
		if (discr < 0.0) continue;

		const RFLOAT d = sqrt(discr) / AtA_xx;
		const RFLOAT q = -AtA_xy * y / AtA_xx;

		int first_x = CEIL(q - d);
		int last_x = FLOOR(q + d);
		if (first_x < first_allowed_x) first_x = first_allowed_x;
		if (last_x > sh - 1) last_x = sh - 1;

		for (int x = first_x; x <= last_x; x++)
		{
			const Complex my_val = DIRECT_A2D_ELEM(f2d, i, x);
			const RFLOAT my_weight = (Mweight != NULL) ? DIRECT_A2D_ELEM(*Mweight, i, x) : 1.0;
			if (my_weight <= 0.) continue;

			// shell units, then undistort for anisotropic magnification
			const RFLOAT ux = x * sx, uy = y * sy;
			const RFLOAT xu = m00 * ux + m01 * uy;
			const RFLOAT yu = m10 * ux + m11 * uy;
			const RFLOAT z_on_ewaldp = inv_diam_ewald * (xu * xu + yu * yu);

			const RFLOAT xp = Ainv(0,0) * xu + Ainv(0,1) * yu + Ainv(0,2) * z_on_ewaldp;
			const RFLOAT yp = Ainv(1,0) * xu + Ainv(1,1) * yu + Ainv(1,2) * z_on_ewaldp;
			const RFLOAT zp = Ainv(2,0) * xu + Ainv(2,1) * yu + Ainv(2,2) * z_on_ewaldp;

			const double r2_3D = xp * xp + yp * yp + zp * zp;
			if (r2_3D > max_r2) continue;

			splat3D(data, weight, xp * tx, yp * ty, zp * tz, my_val, my_weight);
		}
	}
}

void BackProjector::backrotate2DRect(const MultidimArray<Complex> &f2d, const Matrix2D<RFLOAT> &A,
                                     const MultidimArray<RFLOAT> *Mweight, Matrix2D<RFLOAT> *magMatrix)
{
	checkRectBox("BackProjector::backrotate2D");
	if (ref_dim != 2)
		REPORT_ERROR("BackProjector::backrotate2D: rectangular box needs a 2D reference");

	Matrix2D<RFLOAT> Ainv = A.inv();
	Ainv *= (RFLOAT)padding_factor;

	RFLOAT m00 = 1., m10 = 0., m01 = 0., m11 = 1.;
	if (magMatrix != 0)
	{
		m00 = (*magMatrix)(0,0); m10 = (*magMatrix)(1,0);
		m01 = (*magMatrix)(0,1); m11 = (*magMatrix)(1,1);
	}

	const RFLOAT sx = imgScaleX(), sy = imgScaleY();
	const RFLOAT tx = 1. / shellScaleX(), ty = 1. / shellScaleY();

	const int r_max_ref = r_max * padding_factor;
	const int r_max_ref_2 = r_max_ref * r_max_ref;

	const RFLOAT Am_Xx = (Ainv(0,0) * m00 + Ainv(0,1) * m10) * sx;
	const RFLOAT Am_Xy = (Ainv(0,0) * m01 + Ainv(0,1) * m11) * sy;
	const RFLOAT Am_Yx = (Ainv(1,0) * m00 + Ainv(1,1) * m10) * sx;
	const RFLOAT Am_Yy = (Ainv(1,0) * m01 + Ainv(1,1) * m11) * sy;
	const RFLOAT AtA_xx = Am_Xx * Am_Xx + Am_Yx * Am_Yx;
	const RFLOAT AtA_xy = Am_Xx * Am_Xy + Am_Yx * Am_Yy;
	const RFLOAT AtA_yy = Am_Xy * Am_Xy + Am_Yy * Am_Yy;
	const RFLOAT AtA_xy2 = AtA_xy * AtA_xy;

	const int s = YSIZE(f2d);
	const int sh = XSIZE(f2d);

	for (int i = 0; i < s; i++)
	{
		int y, first_allowed_x;
		if (i <= s / 2)
		{
			y = i;
			first_allowed_x = 0;
		}
		else
		{
			y = i - s;
			first_allowed_x = 1;
		}

		const RFLOAT discr = AtA_xy2 * y * y - AtA_xx * (AtA_yy * y * y - r_max_ref_2);
		if (discr < 0.0) continue;

		const RFLOAT d = sqrt(discr) / AtA_xx;
		const RFLOAT q = -AtA_xy * y / AtA_xx;

		int first_x = CEIL(q - d);
		int last_x = FLOOR(q + d);
		if (first_x < first_allowed_x) first_x = first_allowed_x;
		if (last_x > sh - 1) last_x = sh - 1;

		for (int x = first_x; x <= last_x; x++)
		{
			RFLOAT my_weight = 1.;
			if (Mweight != NULL)
			{
				my_weight = DIRECT_A2D_ELEM(*Mweight, i, x);
				if (my_weight <= 0.) continue;
			}
			const Complex my_val = DIRECT_A2D_ELEM(f2d, i, x);

			const RFLOAT ux = x * sx, uy = y * sy;
			const RFLOAT xu = m00 * ux + m01 * uy;
			const RFLOAT yu = m10 * ux + m11 * uy;
			const RFLOAT xp = Ainv(0,0) * xu + Ainv(0,1) * yu;
			const RFLOAT yp = Ainv(1,0) * xu + Ainv(1,1) * yu;

			splat2D(data, weight, xp * tx, yp * ty, my_val, my_weight);
		}
	}
}

void BackProjector::backrotate3DRect(const MultidimArray<Complex> &f3d, const Matrix2D<RFLOAT> &A,
                                     const MultidimArray<RFLOAT> *Mweight)
{
	checkRectBox("BackProjector::backrotate3D");
	if (ref_dim != 3 || data_dim != 3)
		REPORT_ERROR("BackProjector::backrotate3D: rectangular box needs a 3D reference and a 3D map");

	Matrix2D<RFLOAT> Ainv = A.inv();
	Ainv *= (RFLOAT)padding_factor;

	const RFLOAT sx = shellScaleX(), sy = shellScaleY(), sz = shellScaleZ();
	const RFLOAT tx = 1. / sx, ty = 1. / sy, tz = 1. / sz;

	const RFLOAT r_max_src = (XSIZE(f3d) - 1) * sx;
	const RFLOAT r_max_src_2 = r_max_src * r_max_src;
	const int r_max_ref = r_max * padding_factor;
	const int r_max_ref_2 = r_max_ref * r_max_ref;

	for (int k = 0; k < ZSIZE(f3d); k++)
	{
		int z, x_min;
		if (k <= ZSIZE(f3d) / 2)
		{
			z = k;
			x_min = 0;
		}
		else
		{
			z = k - ZSIZE(f3d);
			x_min = 1;
		}
		const RFLOAT uz = z * sz;

		for (int i = 0; i < YSIZE(f3d); i++)
		{
			const int y = (i <= YSIZE(f3d) / 2) ? i : i - YSIZE(f3d);
			const RFLOAT uy = y * sy;
			const RFLOAT yz2 = uy * uy + uz * uz;
			if (yz2 > r_max_src_2) continue;
			const int x_max = FLOOR(sqrt(r_max_src_2 - yz2) / sx);

			for (int x = x_min; x <= x_max && x < XSIZE(f3d); x++)
			{
				const RFLOAT ux = x * sx;
				const RFLOAT xp = Ainv(0,0) * ux + Ainv(0,1) * uy + Ainv(0,2) * uz;
				const RFLOAT yp = Ainv(1,0) * ux + Ainv(1,1) * uy + Ainv(1,2) * uz;
				const RFLOAT zp = Ainv(2,0) * ux + Ainv(2,1) * uy + Ainv(2,2) * uz;

				const int r_ref_2 = xp * xp + yp * yp + zp * zp;
				if (r_ref_2 > r_max_ref_2) continue;

				RFLOAT my_weight = 1.;
				if (Mweight != NULL)
				{
					my_weight = DIRECT_A3D_ELEM(*Mweight, k, i, x);
					if (my_weight <= 0.) continue;
				}
				const Complex my_val = DIRECT_A3D_ELEM(f3d, k, i, x);
				splat3D(data, weight, xp * tx, yp * ty, zp * tz, my_val, my_weight);
			}
		}
	}
}

// ---------------------------------------------------------------------------------------
// Low-resolution exchange between the two half sets

namespace
{
// Extent (in array indices) of a data array holding shells up to lowres_r_max
void lowresExtents(const BackProjector &bp, int lowres_r_max, int &psx, int &psy, int &psz)
{
	const double pf = bp.padding_factor, L = bp.ori_size;
	psx = 2 * (ROUND(pf * lowres_r_max * bp.box_nx / L) + 1) + 1;
	psy = 2 * (ROUND(pf * lowres_r_max * bp.box_ny / L) + 1) + 1;
	psz = (bp.ref_dim == 3) ? 2 * (ROUND(pf * lowres_r_max * bp.box_nz / L) + 1) + 1 : 1;
}
}

void BackProjector::getLowResDataAndWeightRect(MultidimArray<Complex> &lowres_data, MultidimArray<RFLOAT> &lowres_weight,
                                               int lowres_r_max)
{
	checkRectBox("BackProjector::getLowResDataAndWeight");
	if (lowres_r_max > r_max)
		REPORT_ERROR("BackProjector::getLowResDataAndWeight%%ERROR: lowres_r_max is bigger than r_max");

	const double lowres_r = ROUND(padding_factor * lowres_r_max);
	const double lowres_r2_max = lowres_r * lowres_r;
	int psx, psy, psz;
	lowresExtents(*this, lowres_r_max, psx, psy, psz);

	lowres_data.clear();
	lowres_weight.clear();
	if (ref_dim == 2)
	{
		lowres_data.resize(psy, psx / 2 + 1);
		lowres_weight.resize(psy, psx / 2 + 1);
	}
	else
	{
		lowres_data.resize(psz, psy, psx / 2 + 1);
		lowres_weight.resize(psz, psy, psx / 2 + 1);
	}
	lowres_data.setXmippOrigin();
	lowres_data.xinit = 0;
	lowres_weight.setXmippOrigin();
	lowres_weight.xinit = 0;

	AxisScale sc = { shellScaleX(), shellScaleY(), ref_dim == 3 ? shellScaleZ() : 0. };
	FOR_ALL_ELEMENTS_IN_ARRAY3D(lowres_data)
	{
		if (sc.r2(k, i, j) <= lowres_r2_max)
		{
			A3D_ELEM(lowres_data, k, i, j) = A3D_ELEM(data, k, i, j);
			A3D_ELEM(lowres_weight, k, i, j) = A3D_ELEM(weight, k, i, j);
		}
	}
}

void BackProjector::setLowResDataAndWeightRect(MultidimArray<Complex> &lowres_data, MultidimArray<RFLOAT> &lowres_weight,
                                               int lowres_r_max)
{
	checkRectBox("BackProjector::setLowResDataAndWeight");
	if (lowres_r_max > r_max)
		REPORT_ERROR("BackProjector::setLowResDataAndWeight%%ERROR: lowres_r_max is bigger than r_max");

	const double lowres_r = ROUND(padding_factor * lowres_r_max);
	const double lowres_r2_max = lowres_r * lowres_r;
	int psx, psy, psz;
	lowresExtents(*this, lowres_r_max, psx, psy, psz);

	if (YSIZE(lowres_data) != psy || XSIZE(lowres_data) != psx / 2 + 1 || (ref_dim == 3 && ZSIZE(lowres_data) != psz))
		REPORT_ERROR("BackProjector::setLowResDataAndWeight%%ERROR: lowres_data is not of expected size...");
	if (YSIZE(lowres_weight) != psy || XSIZE(lowres_weight) != psx / 2 + 1 || (ref_dim == 3 && ZSIZE(lowres_weight) != psz))
		REPORT_ERROR("BackProjector::setLowResDataAndWeight%%ERROR: lowres_weight is not of expected size...");

	lowres_data.setXmippOrigin();
	lowres_data.xinit = 0;
	lowres_weight.setXmippOrigin();
	lowres_weight.xinit = 0;

	AxisScale sc = { shellScaleX(), shellScaleY(), ref_dim == 3 ? shellScaleZ() : 0. };
	FOR_ALL_ELEMENTS_IN_ARRAY3D(lowres_data)
	{
		if (sc.r2(k, i, j) <= lowres_r2_max)
		{
			A3D_ELEM(data, k, i, j) = A3D_ELEM(lowres_data, k, i, j);
			A3D_ELEM(weight, k, i, j) = A3D_ELEM(lowres_weight, k, i, j);
		}
	}
}

// ---------------------------------------------------------------------------------------
// FSC between the halves and SSNR arrays, in shell units

void BackProjector::getDownsampledAverageRect(MultidimArray<Complex> &avg, bool divide) const
{
	if (!rect || interpolator != TRILINEAR)
		REPORT_ERROR("BackProjector::getDownsampledAverage: rectangular boxes support the linear interpolator only.");
	MultidimArray<RFLOAT> down_weight;

	const double L = ori_size;
	const int hx = ROUND(r_max * box_nx / L) + 1;
	const int hy = ROUND(r_max * box_ny / L) + 1;
	const int hz = ROUND(r_max * box_nz / L) + 1;

	switch (ref_dim)
	{
	case 2:
		avg.initZeros(2 * hy + 1, hx + 1);
		break;
	case 3:
		avg.initZeros(2 * hz + 1, 2 * hy + 1, hx + 1);
		break;
	default:
		REPORT_ERROR("BackProjector::getDownsampledAverage%%ERROR: Dimension of the data array should be 2 or 3");
	}
	avg.setXmippOrigin();
	avg.xinit = 0;
	down_weight.initZeros(avg);

	FOR_ALL_ELEMENTS_IN_ARRAY3D(data)
	{
		const int kp = ROUND((RFLOAT)k / padding_factor);
		const int ip = ROUND((RFLOAT)i / padding_factor);
		const int jp = ROUND((RFLOAT)j / padding_factor);
		// guard cells beyond the last full shell have no counterpart in the average
		if (kp > FINISHINGZ(avg) || ip > FINISHINGY(avg) || jp > FINISHINGX(avg) ||
		    kp < STARTINGZ(avg) || ip < STARTINGY(avg) || jp < STARTINGX(avg))
			continue;
		A3D_ELEM(avg, kp, ip, jp) += A3D_ELEM(data, k, i, j);
		A3D_ELEM(down_weight, kp, ip, jp) += (divide ? A3D_ELEM(weight, k, i, j) : 1.0);
	}

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(avg)
	{
		if (DIRECT_MULTIDIM_ELEM(down_weight, n) > 0.)
			DIRECT_MULTIDIM_ELEM(avg, n) /= DIRECT_MULTIDIM_ELEM(down_weight, n);
		else
			DIRECT_MULTIDIM_ELEM(avg, n) = 0.;
	}
}

void BackProjector::calculateDownSampledFourierShellCorrelationRect(const MultidimArray<Complex> &avg1,
                                                                    const MultidimArray<Complex> &avg2,
                                                                    MultidimArray<RFLOAT> &fsc) const
{
	if (!avg1.sameShape(avg2))
		REPORT_ERROR("ERROR BackProjector::calculateDownSampledFourierShellCorrelation: two arrays have different sizes");

	MultidimArray<RFLOAT> num, den1, den2;
	num.initZeros(ori_size / 2 + 1);
	den1.initZeros(num);
	den2.initZeros(num);
	fsc.initZeros(num);

	const AxisScale sc = { shellScaleX(), shellScaleY(), ref_dim == 3 ? shellScaleZ() : 0. };
	FOR_ALL_ELEMENTS_IN_ARRAY3D(avg1)
	{
		const RFLOAT R = sqrt(sc.r2(k, i, j));
		if (R > r_max) continue;
		const int idx = ROUND(R);

		const Complex z1 = A3D_ELEM(avg1, k, i, j);
		const Complex z2 = A3D_ELEM(avg2, k, i, j);
		num(idx) += z1.real * z2.real + z1.imag * z2.imag;
		den1(idx) += z1.norm();
		den2(idx) += z2.norm();
	}

	FOR_ALL_ELEMENTS_IN_ARRAY1D(fsc)
	{
		if (den1(i) * den2(i) > 0.)
			fsc(i) = num(i) / sqrt(den1(i) * den2(i));
	}
	fsc(0) = 1.;
}

void BackProjector::updateSSNRarraysRect(RFLOAT tau2_fudge,
                                         MultidimArray<RFLOAT> &tau2_io,
                                         MultidimArray<RFLOAT> &sigma2_out,
                                         MultidimArray<RFLOAT> &data_vs_prior_out,
                                         MultidimArray<RFLOAT> &fourier_coverage_out,
                                         const MultidimArray<RFLOAT> &fsc,
                                         const MultidimArray<RFLOAT> &avgctf2,
                                         bool update_tau2_with_fsc,
                                         bool is_whole_instead_of_half,
                                         bool correct_tau2_by_avgctf2)
{
	checkRectBox("BackProjector::updateSSNRarrays");
	MultidimArray<RFLOAT> sigma2, data_vs_prior, fourier_coverage;
	MultidimArray<RFLOAT> tau2 = tau2_io;
	MultidimArray<RFLOAT> counter;
	const double max_r = ROUND(r_max * padding_factor);
	const double max_r2 = max_r * max_r;
	const RFLOAT oversampling_correction = (ref_dim == 3) ? (padding_factor * padding_factor * padding_factor) : (padding_factor * padding_factor);
	const AxisScale sc = { shellScaleX(), shellScaleY(), ref_dim == 3 ? shellScaleZ() : 0. };

	sigma2.initZeros(ori_size / 2 + 1);
	counter.initZeros(ori_size / 2 + 1);
	FOR_ALL_ELEMENTS_IN_ARRAY3D(weight)
	{
		const double r2 = sc.r2(k, i, j);
		if (r2 < max_r2)
		{
			const int ires = ROUND(sqrt(r2) / padding_factor);
			const RFLOAT invw = oversampling_correction * A3D_ELEM(weight, k, i, j);
			DIRECT_A1D_ELEM(sigma2, ires) += invw;
			DIRECT_A1D_ELEM(counter, ires) += 1.;
		}
	}

	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(sigma2)
	{
		if (DIRECT_A1D_ELEM(sigma2, i) > 1e-20)
			DIRECT_A1D_ELEM(sigma2, i) = DIRECT_A1D_ELEM(counter, i) / DIRECT_A1D_ELEM(sigma2, i);
		else if (DIRECT_A1D_ELEM(sigma2, i) == 0)
			DIRECT_A1D_ELEM(sigma2, i) = 0.;
		else
			REPORT_ERROR("BackProjector::reconstruct: ERROR: unexpectedly small, yet non-zero sigma2 value, this should not happen...");
	}

	tau2.reshape(ori_size / 2 + 1);
	data_vs_prior.initZeros(ori_size / 2 + 1);
	fourier_coverage.initZeros(ori_size / 2 + 1);
	counter.initZeros(ori_size / 2 + 1);
	if (update_tau2_with_fsc)
	{
		if (!fsc.sameShape(sigma2) || !fsc.sameShape(tau2))
			REPORT_ERROR("ERROR BackProjector::reconstruct: sigma2, tau2 and fsc have different sizes");
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(sigma2)
		{
			RFLOAT myfsc = XMIPP_MAX(0.001, DIRECT_A1D_ELEM(fsc, i));
			if (is_whole_instead_of_half)
				myfsc = sqrt(2. * myfsc / (myfsc + 1.));
			myfsc = XMIPP_MIN(0.999, myfsc);
			RFLOAT myssnr = myfsc / (1. - myfsc);
			myssnr *= tau2_fudge;
			DIRECT_A1D_ELEM(tau2, i) = myssnr * DIRECT_A1D_ELEM(sigma2, i);
			DIRECT_A1D_ELEM(data_vs_prior, i) = myssnr;
		}
	}

	FOR_ALL_ELEMENTS_IN_ARRAY3D(weight)
	{
		const double r2 = sc.r2(k, i, j);
		if (r2 < max_r2)
		{
			const int ires = ROUND(sqrt(r2) / padding_factor);
			const RFLOAT invw = A3D_ELEM(weight, k, i, j);

			RFLOAT invtau2;
			if (DIRECT_A1D_ELEM(tau2, ires) > 0.)
			{
				invtau2 = 1. / (oversampling_correction * tau2_fudge * DIRECT_A1D_ELEM(tau2, ires));
				if (correct_tau2_by_avgctf2 && DIRECT_A1D_ELEM(avgctf2, ires) > 0.)
					invtau2 *= 1. / DIRECT_A1D_ELEM(avgctf2, ires);
			}
			else if (DIRECT_A1D_ELEM(tau2, ires) == 0.)
			{
				invtau2 = 1. / (0.001 * invw);
			}
			else
				REPORT_ERROR("ERROR BackProjector::reconstruct: Negative or zero values encountered for tau2 spectrum!");

			if (!update_tau2_with_fsc)
				DIRECT_A1D_ELEM(data_vs_prior, ires) += invw / invtau2;
			if (invw / invtau2 >= 1.)
				DIRECT_A1D_ELEM(fourier_coverage, ires) += 1.;
			DIRECT_A1D_ELEM(counter, ires) += 1.;
		}
	}

	if (!update_tau2_with_fsc)
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(data_vs_prior)
		{
			if (i > r_max)
				DIRECT_A1D_ELEM(data_vs_prior, i) = 0.;
			else if (DIRECT_A1D_ELEM(counter, i) < 0.001)
				DIRECT_A1D_ELEM(data_vs_prior, i) = 999.;
			else
				DIRECT_A1D_ELEM(data_vs_prior, i) /= DIRECT_A1D_ELEM(counter, i);
		}
	}

	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(fourier_coverage)
	{
		if (DIRECT_A1D_ELEM(counter, i) > 0.)
			DIRECT_A1D_ELEM(fourier_coverage, i) /= DIRECT_A1D_ELEM(counter, i);
	}

	tau2_io = tau2;
	sigma2_out = sigma2;
	data_vs_prior_out = data_vs_prior;
	fourier_coverage_out = fourier_coverage;
}

// ---------------------------------------------------------------------------------------
// Symmetry

void BackProjector::applyHelicalSymmetryRect(int nr_helical_asu, RFLOAT helical_twist, RFLOAT helical_rise)
{
	if ((nr_helical_asu < 2) || (ref_dim != 3))
		return;
	checkRectBox("BackProjector::applyHelicalSymmetry");

	const double rr = ROUND(r_max * padding_factor);
	const double rmax2 = rr * rr;
	const double sx = shellScaleX(), sy = shellScaleY(), sz = shellScaleZ();
	const double tx = 1. / sx, ty = 1. / sy, tz = 1. / sz;

	Matrix2D<RFLOAT> R(4, 4);
	MultidimArray<RFLOAT> sum_weight = weight;
	MultidimArray<Complex> sum_data = data;
	const int h_min = -nr_helical_asu / 2;
	const int h_max = -h_min + nr_helical_asu % 2;
	for (int hh = h_min; hh < h_max; hh++)
	{
		if (hh == 0) continue;
		const RFLOAT rot_ang = hh * (-helical_twist);
		rotation3DMatrix(rot_ang, 'Z', R);
		R.setSmallValuesToZero();

		FOR_ALL_ELEMENTS_IN_ARRAY3D(sum_weight)
		{
			const RFLOAT x = j * sx, y = i * sy, z = k * sz; // shell units
			if (x * x + y * y + z * z <= rmax2)
			{
				const RFLOAT xp = x * R(0, 0) + y * R(0, 1) + z * R(0, 2);
				const RFLOAT yp = x * R(1, 0) + y * R(1, 1) + z * R(1, 2);
				const RFLOAT zp = x * R(2, 0) + y * R(2, 1) + z * R(2, 2);

				Complex ddd;
				RFLOAT ww;
				if (!sample3D(data, weight, xp * tx, yp * ty, zp * tz, ddd, ww))
					continue;

				if (ABS(helical_rise) > 0.)
				{
					RFLOAT zshift = hh * helical_rise;
					zshift /= -ori_size * (RFLOAT)padding_factor;
					const RFLOAT dotp = 2 * PI * (z * zshift);
					const RFLOAT a = cos(dotp), b = sin(dotp);
					const RFLOAT c = ddd.real, d = ddd.imag;
					const RFLOAT ac = a * c, bd = b * d;
					const RFLOAT ab_cd = (a + b) * (c + d);
					ddd = Complex(ac - bd, ab_cd - ac - bd);
				}
				A3D_ELEM(sum_data, k, i, j) += ddd;
				A3D_ELEM(sum_weight, k, i, j) += ww;
			}
		}
	}
	data = sum_data;
	weight = sum_weight;
}

void BackProjector::applyPointGroupSymmetryRect(int threads)
{
	if (SL.SymsNo() <= 0 || ref_dim != 3)
		return;
	checkRectBox("BackProjector::applyPointGroupSymmetry");

	const double rr = ROUND(r_max * padding_factor);
	const double rmax2 = rr * rr;
	const double sx = shellScaleX(), sy = shellScaleY(), sz = shellScaleZ();
	const double tx = 1. / sx, ty = 1. / sy, tz = 1. / sz;

	Matrix2D<RFLOAT> L(4, 4), R(4, 4);
	MultidimArray<RFLOAT> sum_weight = weight;
	MultidimArray<Complex> sum_data = data;
	for (int isym = 0; isym < SL.SymsNo(); isym++)
	{
		SL.get_matrices(isym, L, R);

		#pragma omp parallel for num_threads(threads)
		for (long int k = STARTINGZ(sum_weight); k <= FINISHINGZ(sum_weight); k++)
		for (long int i = STARTINGY(sum_weight); i <= FINISHINGY(sum_weight); i++)
		for (long int j = STARTINGX(sum_weight); j <= FINISHINGX(sum_weight); j++)
		{
			const RFLOAT x = j * sx, y = i * sy, z = k * sz;
			if (x * x + y * y + z * z <= rmax2)
			{
				const RFLOAT xp = x * R(0, 0) + y * R(0, 1) + z * R(0, 2);
				const RFLOAT yp = x * R(1, 0) + y * R(1, 1) + z * R(1, 2);
				const RFLOAT zp = x * R(2, 0) + y * R(2, 1) + z * R(2, 2);

				Complex dd;
				RFLOAT ww;
				if (!sample3D(data, weight, xp * tx, yp * ty, zp * tz, dd, ww))
					continue;
				A3D_ELEM(sum_data, k, i, j) += dd;
				A3D_ELEM(sum_weight, k, i, j) += ww;
			}
		}
	}
	data = sum_data;
	weight = sum_weight;
}

// ---------------------------------------------------------------------------------------
// Reconstruction

void BackProjector::convoluteBlobRealSpaceRect(FourierTransformer &transformer, bool do_mask)
{
	// the real-space grid has the size of the data array on every axis
	const int psx = 2 * (XSIZE(data) - 1) + 1, psy = YSIZE(data), psz = ZSIZE(data);

	MultidimArray<RFLOAT> Mconv;
	if (ref_dim == 2)
		Mconv.reshape(psy, psx);
	else
		Mconv.reshape(psz, psy, psx);

	transformer.setReal(Mconv);
	transformer.inverseFourierTransform();

	const RFLOAT normftblob = tab_ftblob(0.);
	const RFLOAT ix = 1. / box_nx, iy = 1. / box_ny, iz = (ref_dim == 3) ? 1. / box_nz : 0.;

	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(Mconv)
	{
		const int kp = (k < psz / 2) ? k : k - psz;
		const int ip = (i < psy / 2) ? i : i - psy;
		const int jp = (j < psx / 2) ? j : j - psx;
		const RFLOAT fx = jp * ix, fy = ip * iy, fz = kp * iz;
		const RFLOAT rval = sqrt(fx * fx + fy * fy + fz * fz) / padding_factor;
		if (do_mask && rval > 1. / (2. * padding_factor))
			DIRECT_A3D_ELEM(Mconv, k, i, j) = 0.;
		else
			DIRECT_A3D_ELEM(Mconv, k, i, j) *= (tab_ftblob(rval) / normftblob);
	}

	transformer.FourierTransform();
}

void BackProjector::windowToOridimRealSpaceRect(FourierTransformer &transformer, MultidimArray<RFLOAT> &Mout)
{
	checkRectBox("BackProjector::windowToOridimRealSpace");
	MultidimArray<Complex> &Fin = transformer.getFourierReference();

	const int pnx = ROUND(padding_factor * box_nx), pny = ROUND(padding_factor * box_ny);
	const int pnz = (ref_dim == 3) ? ROUND(padding_factor * box_nz) : 1;

	// Usually the padded grid is already this size: nothing to window, and no copy of it
	if (!(pnx == 2 * (XSIZE(Fin) - 1) && pny == YSIZE(Fin) && pnz == ZSIZE(Fin)))
	{
		MultidimArray<Complex> Ftmp;
		windowFourierTransformRect(Fin, Ftmp, pnx, pny, pnz);
		Fin.moveFrom(Ftmp);
	}

	RFLOAT normfft;
	if (ref_dim == 2)
	{
		Mout.reshape(pny, pnx);
		normfft = (RFLOAT)(padding_factor * padding_factor);
	}
	else
	{
		Mout.reshape(pnz, pny, pnx);
		normfft = (data_dim == 3) ? (RFLOAT)(padding_factor * padding_factor * padding_factor)
		                          : (RFLOAT)(padding_factor * padding_factor * padding_factor * box_nx);
	}
	Mout.setXmippOrigin();

	centerFFTbySignParallel(Fin);

	transformer.setReal(Mout);
	transformer.inverseFourierTransform();
	Fin.clear();
	transformer.fReal = NULL; // make sure the fftw plan is re-calculated
	Mout.setXmippOrigin();

	if (ref_dim == 2)
		Mout.window(FIRST_XMIPP_INDEX(box_ny), FIRST_XMIPP_INDEX(box_nx),
		            LAST_XMIPP_INDEX(box_ny), LAST_XMIPP_INDEX(box_nx));
	else
		Mout.window(FIRST_XMIPP_INDEX(box_nz), FIRST_XMIPP_INDEX(box_ny), FIRST_XMIPP_INDEX(box_nx),
		            LAST_XMIPP_INDEX(box_nz), LAST_XMIPP_INDEX(box_ny), LAST_XMIPP_INDEX(box_nx));
	Mout.setXmippOrigin();

	Mout /= normfft;

	softMaskOutsideMapRect(Mout, ori_size, box_nx, box_ny, box_nz);
}

void BackProjector::reconstructRect(MultidimArray<RFLOAT> &vol_out,
                                    int max_iter_preweight,
                                    bool do_map,
                                    const MultidimArray<RFLOAT> &tau2,
                                    RFLOAT tau2_fudge,
                                    RFLOAT normalise,
                                    int minres_map,
                                    bool printTimes,
                                    Image<RFLOAT> *weight_out)
{
	checkRectBox("BackProjector::reconstruct");
	if (weight_out != 0)
		REPORT_ERROR("BackProjector::reconstruct: weight_out is not supported for rectangular boxes yet.");

	const double max_r = ROUND(r_max * padding_factor);
	const double max_r2 = max_r * max_r;
	const RFLOAT oversampling_correction = (ref_dim == 3) ? (padding_factor * padding_factor * padding_factor) : (padding_factor * padding_factor);
	const AxisScale sc = { shellScaleX(), shellScaleY(), ref_dim == 3 ? shellScaleZ() : 0. };

	// the FFT grid is the data array, per axis
	const int psx = 2 * (XSIZE(data) - 1) + 1, psy = YSIZE(data), psz = (ref_dim == 3) ? ZSIZE(data) : 1;
	vol_out.setDimensions(psx, psy, psz, 1);

	FourierTransformer transformer;
	transformer.setReal(vol_out); // fake set real: allocates Fconv and calculates the plans
	MultidimArray<Complex> &Fconv = transformer.getFourierReference();
	vol_out.clear();

	MultidimArray<RFLOAT> Fweight;
	Fweight.reshape(Fconv);
	decenterRect(weight, Fweight, max_r2, sc);

	if (do_map)
	{
		FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM_RECT(Fconv)
		{
			const double r2 = sc.r2(kp, ip, jp);
			if (r2 < max_r2)
			{
				const int ires = ROUND(sqrt(r2) / padding_factor);
				RFLOAT invw = DIRECT_A3D_ELEM(Fweight, k, i, j);

				RFLOAT invtau2;
				if (DIRECT_A1D_ELEM(tau2, ires) > 0.)
				{
					invtau2 = 1. / (oversampling_correction * tau2_fudge * DIRECT_A1D_ELEM(tau2, ires));
				}
				else if (DIRECT_A1D_ELEM(tau2, ires) < 1e-20)
				{
					if (invw > 1e-20) invtau2 = 1. / (0.001 * invw);
					else invtau2 = 0.;
				}
				else
					REPORT_ERROR("ERROR BackProjector::reconstruct: Negative or zero values encountered for tau2 spectrum!");

				if (ires >= minres_map)
				{
					invw += invtau2;
					DIRECT_A3D_ELEM(Fweight, k, i, j) = invw;
				}
			}
		}
	}

	if (skip_gridding || max_iter_preweight <= 0)
	{
		Fconv.initZeros();
		decenterRect(data, Fconv, max_r2, sc);

		MultidimArray<RFLOAT> radavg_weight(r_max), counter(r_max);
		radavg_weight.initZeros();
		counter.initZeros();
		const double round_max_r2 = (double)(r_max * padding_factor * r_max * padding_factor);
		const long int fz = ZSIZE(Fweight), fy = YSIZE(Fweight), fx = XSIZE(Fweight);
		#pragma omp parallel
		{
			std::vector<RFLOAT> my_sum(r_max, 0.), my_count(r_max, 0.);
			#pragma omp for nowait
			for (long int k = 0; k < fz; k++)
			{
				const long int kp = (k <= fz / 2) ? k : k - fz;
				for (long int i = 0; i < fy; i++)
				{
					const long int ip = (i <= fy / 2) ? i : i - fy;
					for (long int j = 0; j < fx; j++)
					{
						const double r2 = sc.r2(kp, ip, j);
						if (r2 < round_max_r2)
						{
							const int ires = FLOOR(sqrt(r2) / padding_factor);
							if (ires >= r_max)
								REPORT_ERROR("BUG: ires >=XSIZE(radavg_weight) ");
							my_sum[ires] += DIRECT_A3D_ELEM(Fweight, k, i, j);
							my_count[ires] += 1.;
						}
					}
				}
			}
			#pragma omp critical(radavg_merge)
			for (int r = 0; r < r_max; r++)
			{
				DIRECT_A1D_ELEM(radavg_weight, r) += my_sum[r];
				DIRECT_A1D_ELEM(counter, r) += my_count[r];
			}
		}

		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(radavg_weight)
		{
			if (DIRECT_A1D_ELEM(counter, i) > 0. || DIRECT_A1D_ELEM(radavg_weight, i) > 0.)
				DIRECT_A1D_ELEM(radavg_weight, i) /= 1000. * DIRECT_A1D_ELEM(counter, i);
			else
				REPORT_ERROR("BUG: zeros in counter or radavg_weight!");
		}

		bool have_warned = false;
		#pragma omp parallel for
		for (long int k = 0; k < fz; k++)
		{
			const long int kp = (k <= fz / 2) ? k : k - fz;
			for (long int i = 0; i < fy; i++)
			{
				const long int ip = (i <= fy / 2) ? i : i - fy;
				for (long int j = 0; j < fx; j++)
				{
					const double r2 = sc.r2(kp, ip, j);
					const int ires = FLOOR(sqrt(r2) / padding_factor);
					const RFLOAT w = XMIPP_MAX(DIRECT_A3D_ELEM(Fweight, k, i, j), DIRECT_A1D_ELEM(radavg_weight, (ires < r_max) ? ires : (r_max - 1)));
					if (w == 0.)
					{
						if (abs(DIRECT_A3D_ELEM(Fconv, k, i, j)) > 0.)
						{
							#pragma omp critical(radavg_warn)
							if (!have_warned)
							{
								std::cerr << " WARNING: ignoring divide by zero in skip_gridding: ires = " << ires << " kp = " << kp << " ip = " << ip << " jp = " << j << std::endl;
								have_warned = true;
							}
						}
					}
					else
						DIRECT_A3D_ELEM(Fconv, k, i, j) /= w;
				}
			}
		}
	}
	else
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fweight)
			DIRECT_MULTIDIM_ELEM(Fweight, n) /= normalise;
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(data)
			DIRECT_MULTIDIM_ELEM(data, n) /= normalise;

		FOR_ALL_ELEMENTS_IN_ARRAY3D(weight)
		{
			if (sc.r2(k, i, j) < max_r2)
				A3D_ELEM(weight, k, i, j) = 1.;
			else
				A3D_ELEM(weight, k, i, j) = 0.;
		}
		MultidimArray<double> Fnewweight;
		Fnewweight.reshape(Fconv);
		decenterRect(weight, Fnewweight, max_r2, sc);

		// Iterative algorithm as in Eq. [14] in Pipe & Menon (1999)
		for (int iter = 0; iter < max_iter_preweight; iter++)
		{
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
				DIRECT_MULTIDIM_ELEM(Fconv, n) = DIRECT_MULTIDIM_ELEM(Fnewweight, n) * DIRECT_MULTIDIM_ELEM(Fweight, n);

			convoluteBlobRealSpaceRect(transformer, false);

			FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM_RECT(Fconv)
			{
				if (sc.r2(kp, ip, jp) < max_r2)
				{
					const RFLOAT w = XMIPP_MAX(1e-6, abs(DIRECT_A3D_ELEM(Fconv, k, i, j)));
					DIRECT_A3D_ELEM(Fnewweight, k, i, j) /= w;
				}
			}
		}

		Fweight.clear();

		Fconv.initZeros();
		decenterRect(data, Fconv, max_r2, sc);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
#ifdef RELION_SINGLE_PRECISION
			if (DIRECT_MULTIDIM_ELEM(Fnewweight, n) > 1e20)
				DIRECT_MULTIDIM_ELEM(Fnewweight, n) = 1e20;
#endif
			DIRECT_MULTIDIM_ELEM(Fconv, n) *= DIRECT_MULTIDIM_ELEM(Fnewweight, n);
		}
		Fnewweight.clear();
	}

	windowToOridimRealSpaceRect(transformer, vol_out);

	// Correct for the linear interpolation that led to the data array: the radial sinc^2
	// of the square code, in units where the box side is 1
	vol_out.setXmippOrigin();
	{
		const RFLOAT ix = 1. / box_nx, iy = 1. / box_ny, iz = (ref_dim == 3) ? 1. / box_nz : 0.;
		#pragma omp parallel for
		for (long int k = STARTINGZ(vol_out); k <= FINISHINGZ(vol_out); k++)
			for (long int i = STARTINGY(vol_out); i <= FINISHINGY(vol_out); i++)
				for (long int j = STARTINGX(vol_out); j <= FINISHINGX(vol_out); j++)
				{
					const RFLOAT fx = j * ix, fy = i * iy, fz = k * iz;
					const RFLOAT r = sqrt(fx * fx + fy * fy + fz * fz);
					if (r > 0.)
					{
						const RFLOAT rval = r / padding_factor;
						const RFLOAT sinc = sin(PI * rval) / (PI * rval);
						A3D_ELEM(vol_out, k, i, j) /= sinc * sinc;
					}
				}
	}
	transformer.cleanup();
	vol_out.shrinkToFit();
}
