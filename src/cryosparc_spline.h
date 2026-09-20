#ifndef CRYOSPARC_SPLINE_H
#define CRYOSPARC_SPLINE_H

/* Evaluation of CryoSPARC's patch-motion local ("bending") field.
 *
 * CryoSPARC's Patch Motion Correction job stores its local motion field as
 * cubic B-spline *coefficients* on a coarse (K_Z, K_Y, K_X) control grid
 * spanning the movie's time and space - not as samples. CryoSPARC publishes no
 * library for reading this back, so the evaluation is reimplemented here,
 * following cryosparc_compute/sigproc.py (pix_to_spline_coords,
 * prepare_spl_naturalbc, prepare_coords_naturalbc, spline_interp,
 * spline_interp_traj).
 *
 * Ported from the cs2relion Python package, which validated this path
 * end-to-end against a real RELION installation including Bayesian Polishing.
 * Two details there are load-bearing and easy to lose in translation:
 *
 *  - The coefficient grid is padded by 1 on every axis with an ODD reflection
 *    (numpy's pad(..., "reflect", reflect_type="odd")) and the query
 *    coordinates are shifted by +1 to match. That is what gives the spline
 *    natural boundary conditions at the edges of the control grid.
 *  - The array holds spline COEFFICIENTS, so evaluation must not pre-filter
 *    (scipy's prefilter=False). Pre-filtering here would silently distort the
 *    field rather than fail.
 *
 * Because of the +1 padding, every tap of the 4-wide cubic support lands
 * inside the padded array for any in-range query, so no boundary mode is
 * needed at evaluation time; out-of-range taps are nevertheless treated as
 * zero, matching scipy's mode="constant".
 */

#include <vector>
#include <cmath>
#include <cstddef>

namespace cryosparc {

/// Cubic B-spline weights for the four taps at floor(c)-1 .. floor(c)+2,
/// given the fractional part f = c - floor(c) in [0, 1). Sums to 1.
inline void cubicBSplineWeights(double f, double w[4])
{
	const double g = 1. - f;
	w[0] = g * g * g / 6.;
	w[1] = 2. / 3. - f * f + f * f * f / 2.;
	w[2] = 2. / 3. - g * g + g * g * g / 2.;
	w[3] = f * f * f / 6.;
}

/// A cubic B-spline coefficient volume with natural boundary conditions,
/// i.e. padded by one on every axis with an odd reflection.
class SplineVolume {
public:
	/// spl is (KZ, KY, KX) in C order.
	SplineVolume(const double* spl, int KZ, int KY, int KX)
		: kz_(KZ), ky_(KY), kx_(KX),
		  nz_(KZ + 2), ny_(KY + 2), nx_(KX + 2)
	{
		// Pad axis by axis, in numpy's own order, so that corner values match
		// what np.pad produces (it pads axis 0 first, then 1, then 2, each time
		// operating on the already-partly-padded result).
		std::vector<double> a0, a1;

		padAxis0(spl, KZ, KY, KX, a0);            // -> (KZ+2, KY,   KX)
		padAxis1(a0, KZ + 2, KY, KX, a1);         // -> (KZ+2, KY+2, KX)
		padAxis2(a1, KZ + 2, KY + 2, KX, data_);  // -> (KZ+2, KY+2, KX+2)
	}

	/// Evaluate at a coordinate given in the UNPADDED control-grid index space.
	double at(double z, double y, double x) const
	{
		// Coordinates outside the control grid produce no signal, matching
		// scipy's mode="constant" with cval=0.
		if (z < 0. || z > kz_ - 1. || y < 0. || y > ky_ - 1. || x < 0. || x > kx_ - 1.)
			return 0.;

		// prepare_coords_naturalbc: shift into the padded array's index space
		const double pz = z + 1., py = y + 1., px = x + 1.;

		const int iz = (int)std::floor(pz);
		const int iy = (int)std::floor(py);
		const int ix = (int)std::floor(px);

		double wz[4], wy[4], wx[4];
		cubicBSplineWeights(pz - iz, wz);
		cubicBSplineWeights(py - iy, wy);
		cubicBSplineWeights(px - ix, wx);

		double sum = 0.;
		for (int dz = 0; dz < 4; dz++)
		{
			const int zz = iz - 1 + dz;
			if (zz < 0 || zz >= nz_ || wz[dz] == 0.) continue;
			double sy = 0.;
			for (int dy = 0; dy < 4; dy++)
			{
				const int yy = iy - 1 + dy;
				if (yy < 0 || yy >= ny_ || wy[dy] == 0.) continue;
				double sx = 0.;
				for (int dx = 0; dx < 4; dx++)
				{
					const int xx = ix - 1 + dx;
					if (xx < 0 || xx >= nx_ || wx[dx] == 0.) continue;
					sx += wx[dx] * data_[((size_t)zz * ny_ + yy) * nx_ + xx];
				}
				sy += wy[dy] * sx;
			}
			sum += wz[dz] * sy;
		}
		return sum;
	}

private:
	int kz_, ky_, kx_;   // unpadded control-grid shape
	int nz_, ny_, nx_;   // padded shape
	std::vector<double> data_;

	/// new[0] = 2*a[0] - a[1], new[n+1] = 2*a[n-1] - a[n-2] (numpy's odd reflect).
	static void oddReflect(double first, double second, double last, double second_last,
	                       double& lo, double& hi)
	{
		lo = 2. * first - second;
		hi = 2. * last - second_last;
	}

	static void padAxis0(const double* a, int nz, int ny, int nx, std::vector<double>& out)
	{
		out.assign((size_t)(nz + 2) * ny * nx, 0.);
		for (int z = 0; z < nz; z++)
		for (int y = 0; y < ny; y++)
		for (int x = 0; x < nx; x++)
			out[((size_t)(z + 1) * ny + y) * nx + x] = a[((size_t)z * ny + y) * nx + x];

		for (int y = 0; y < ny; y++)
		for (int x = 0; x < nx; x++)
		{
			const size_t s = (size_t)y * nx + x;
			if (nz == 1)
			{
				out[s] = out[((size_t)1 * ny + y) * nx + x];
				out[((size_t)(nz + 1) * ny + y) * nx + x] = out[((size_t)1 * ny + y) * nx + x];
				continue;
			}
			double lo, hi;
			oddReflect(out[((size_t)1 * ny + y) * nx + x],
			           out[((size_t)2 * ny + y) * nx + x],
			           out[((size_t)nz * ny + y) * nx + x],
			           out[((size_t)(nz - 1) * ny + y) * nx + x], lo, hi);
			out[s] = lo;
			out[((size_t)(nz + 1) * ny + y) * nx + x] = hi;
		}
	}

	static void padAxis1(const std::vector<double>& a, int nz, int ny, int nx, std::vector<double>& out)
	{
		out.assign((size_t)nz * (ny + 2) * nx, 0.);
		for (int z = 0; z < nz; z++)
		for (int y = 0; y < ny; y++)
		for (int x = 0; x < nx; x++)
			out[((size_t)z * (ny + 2) + (y + 1)) * nx + x] = a[((size_t)z * ny + y) * nx + x];

		for (int z = 0; z < nz; z++)
		for (int x = 0; x < nx; x++)
		{
			if (ny == 1)
			{
				const double v = out[((size_t)z * (ny + 2) + 1) * nx + x];
				out[((size_t)z * (ny + 2) + 0) * nx + x] = v;
				out[((size_t)z * (ny + 2) + (ny + 1)) * nx + x] = v;
				continue;
			}
			double lo, hi;
			oddReflect(out[((size_t)z * (ny + 2) + 1) * nx + x],
			           out[((size_t)z * (ny + 2) + 2) * nx + x],
			           out[((size_t)z * (ny + 2) + ny) * nx + x],
			           out[((size_t)z * (ny + 2) + (ny - 1)) * nx + x], lo, hi);
			out[((size_t)z * (ny + 2) + 0) * nx + x] = lo;
			out[((size_t)z * (ny + 2) + (ny + 1)) * nx + x] = hi;
		}
	}

	static void padAxis2(const std::vector<double>& a, int nz, int ny, int nx, std::vector<double>& out)
	{
		out.assign((size_t)nz * ny * (nx + 2), 0.);
		for (int z = 0; z < nz; z++)
		for (int y = 0; y < ny; y++)
		for (int x = 0; x < nx; x++)
			out[((size_t)z * ny + y) * (nx + 2) + (x + 1)] = a[((size_t)z * ny + y) * nx + x];

		for (int z = 0; z < nz; z++)
		for (int y = 0; y < ny; y++)
		{
			const size_t row = ((size_t)z * ny + y) * (nx + 2);
			if (nx == 1)
			{
				out[row + 0] = out[row + 1];
				out[row + nx + 1] = out[row + 1];
				continue;
			}
			double lo, hi;
			oddReflect(out[row + 1], out[row + 2], out[row + nx], out[row + nx - 1], lo, hi);
			out[row + 0] = lo;
			out[row + nx + 1] = hi;
		}
	}
};

/// One component (x or y) of the local field at arbitrary movie pixel
/// coordinates. pix_to_spline_coords rescales a movie pixel coordinate onto the
/// control grid's own index space.
inline double splineInterpAt(const SplineVolume& vol,
                             int KZ, int KY, int KX,
                             int n_frames, int height, int width,
                             double frame, double y_px, double x_px)
{
	const double sz = (KZ - 1) / (double)std::max(n_frames - 1, 1);
	const double sy = (KY - 1) / (double)std::max(height - 1, 1);
	const double sx = (KX - 1) / (double)std::max(width - 1, 1);
	return vol.at(frame * sz, y_px * sy, x_px * sx);
}

/// Port of spline_interp_traj: the local-only displacement at each of the given
/// (x, y) movie pixel positions, for every frame, in CryoSPARC's own raw
/// sign/axis convention. out_lx/out_ly are indexed [pos * n_frames + frame].
inline void splineInterpTraj(int n_frames, int height, int width,
                             const double* splx, const double* sply,
                             int KZ, int KY, int KX,
                             const std::vector<double>& pos_x,
                             const std::vector<double>& pos_y,
                             std::vector<double>& out_lx,
                             std::vector<double>& out_ly)
{
	const SplineVolume vx(splx, KZ, KY, KX);
	const SplineVolume vy(sply, KZ, KY, KX);

	const size_t n_pos = pos_x.size();
	out_lx.assign(n_pos * n_frames, 0.);
	out_ly.assign(n_pos * n_frames, 0.);

	for (size_t p = 0; p < n_pos; p++)
	for (int f = 0; f < n_frames; f++)
	{
		const size_t k = p * n_frames + f;
		out_lx[k] = splineInterpAt(vx, KZ, KY, KX, n_frames, height, width, f, pos_y[p], pos_x[p]);
		out_ly[k] = splineInterpAt(vy, KZ, KY, KX, n_frames, height, width, f, pos_y[p], pos_x[p]);
	}
}

} // namespace cryosparc

#endif // CRYOSPARC_SPLINE_H
