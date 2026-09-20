#ifndef CRYOSPARC_MOTION_MODEL_H
#define CRYOSPARC_MOTION_MODEL_H

/* RELION's ThirdOrderPolynomialModel for local (patch) motion.
 *
 * This is the same least-squares design matrix motioncorr_runner.cpp uses to
 * fit its local motion model from raw per-patch cross-correlation
 * measurements. CryoSPARC's local field is a cubic B-spline over its own
 * spatial x temporal control grid (see cryosparc_spline.h); the two model
 * families are different and non-nested, so the import samples CryoSPARC's
 * field at a RELION-style patch grid and least-squares refits RELION's basis to
 * those samples. That is RELION's best polynomial approximation of CryoSPARC's
 * field, not a literal re-encoding, and some information is necessarily lost.
 *
 * Note that RELION only reloads the fitted data_local_motion_model downstream -
 * Micrograph::read() does not reload the raw data_local_shift table - so the
 * fitted coefficients are what actually gets applied.
 *
 * None of the 18 basis functions has a constant (z^0) term, so the model
 * predicts exactly zero displacement everywhere at frame 1 by construction,
 * rather than as a fitted approximation.
 *
 * Ported from the cs2relion Python package, where the fit was verified against
 * RELION's own stored coefficients, on RELION's own raw measurements, at
 * correlation 0.997-1.000.
 */

#include <vector>
#include <cstddef>
#include <utility>

#include "src/Eigen/Dense"

namespace cryosparc {

const int NUM_MOTION_COEFFS_PER_DIM = 18;

/// One row of RELION's design matrix, for a patch at (x_px, y_px) on a
/// width x height movie, observed at `frame`.
inline void motionDesignRow(double width, double height, double frame,
                            double x_px, double y_px, int first_frame,
                            double row[NUM_MOTION_COEFFS_PER_DIM])
{
	const double x_n = x_px / width - 0.5;
	const double y_n = y_px / height - 0.5;
	const double z = frame - first_frame;

	const double z2 = z * z, z3 = z2 * z;
	const double x2 = x_n * x_n, y2 = y_n * y_n, xy = x_n * y_n;

	row[0]  = z;        row[1]  = z2;        row[2]  = z3;
	row[3]  = x_n * z;  row[4]  = x_n * z2;  row[5]  = x_n * z3;
	row[6]  = x2 * z;   row[7]  = x2 * z2;   row[8]  = x2 * z3;
	row[9]  = y_n * z;  row[10] = y_n * z2;  row[11] = y_n * z3;
	row[12] = y2 * z;   row[13] = y2 * z2;   row[14] = y2 * z3;
	row[15] = xy * z;   row[16] = xy * z2;   row[17] = xy * z3;
}

/// Evaluate one dimension's fitted polynomial at normalised (x_n, y_n) in
/// [-0.5, 0.5] and frame offset z = frame - first_frame.
inline double motionShiftAt(const double* coeff, double z, double x_n, double y_n)
{
	const double z2 = z * z, z3 = z2 * z;
	const double x2 = x_n * x_n, y2 = y_n * y_n, xy = x_n * y_n;

	return (coeff[0]  * z + coeff[1]  * z2 + coeff[2]  * z3)
	     + (coeff[3]  * z + coeff[4]  * z2 + coeff[5]  * z3) * x_n
	     + (coeff[6]  * z + coeff[7]  * z2 + coeff[8]  * z3) * x2
	     + (coeff[9]  * z + coeff[10] * z2 + coeff[11] * z3) * y_n
	     + (coeff[12] * z + coeff[13] * z2 + coeff[14] * z3) * y2
	     + (coeff[15] * z + coeff[16] * z2 + coeff[17] * z3) * xy;
}

/// Least-squares fit of RELION's 18 coefficients per dimension from raw
/// (frame, x_px, y_px, shift_x, shift_y) observations.
inline void fitMotionModel(double width, double height, int first_frame,
                           const std::vector<double>& frame,
                           const std::vector<double>& x_px,
                           const std::vector<double>& y_px,
                           const std::vector<double>& shift_x,
                           const std::vector<double>& shift_y,
                           std::vector<double>& coeff_x,
                           std::vector<double>& coeff_y)
{
	const size_t n = frame.size();
	Eigen::MatrixXd A((int)n, NUM_MOTION_COEFFS_PER_DIM);
	Eigen::VectorXd bx((int)n), by((int)n);

	for (size_t i = 0; i < n; i++)
	{
		double row[NUM_MOTION_COEFFS_PER_DIM];
		motionDesignRow(width, height, frame[i], x_px[i], y_px[i], first_frame, row);
		for (int j = 0; j < NUM_MOTION_COEFFS_PER_DIM; j++) A((int)i, j) = row[j];
		bx((int)i) = shift_x[i];
		by((int)i) = shift_y[i];
	}

	// Same solver family as numpy's lstsq (SVD-based), so a rank-deficient or
	// ill-conditioned patch layout degrades gracefully instead of blowing up.
	Eigen::BDCSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
	const Eigen::VectorXd cx = svd.solve(bx);
	const Eigen::VectorXd cy = svd.solve(by);

	coeff_x.assign(NUM_MOTION_COEFFS_PER_DIM, 0.);
	coeff_y.assign(NUM_MOTION_COEFFS_PER_DIM, 0.);
	for (int j = 0; j < NUM_MOTION_COEFFS_PER_DIM; j++)
	{
		coeff_x[j] = cx(j);
		coeff_y[j] = cy(j);
	}
}

/// RELION's own bin-centre convention for a patch grid, ordered with x varying
/// fastest (matching RELION's own data_local_shift table ordering).
inline std::vector<std::pair<double, double> > motionPatchGrid(
		double width, double height, int patches_x, int patches_y)
{
	std::vector<std::pair<double, double> > out;
	out.reserve((size_t)patches_x * patches_y);
	for (int j = 0; j < patches_y; j++)
	{
		const double y = (j + 0.5) * height / patches_y;
		for (int i = 0; i < patches_x; i++)
		{
			const double x = (i + 0.5) * width / patches_x;
			out.push_back(std::make_pair(x, y));
		}
	}
	return out;
}

/// RELION's early/late accumulated motion statistic (motioncorr_runner.cpp):
/// the summed frame-to-frame displacement magnitude of the GLOBAL trajectory
/// only - RELION evaluates this with use_local=false - split at a dose-based
/// cutoff frame. Shifts are in pixels; results in Angstrom.
inline void accumulatedMotion(const std::vector<double>& shift_x,
                              const std::vector<double>& shift_y,
                              double pixel_size_A, double dose_per_frame_A2,
                              double pre_exposure_A2, double cutoff_A2,
                              double& total, double& early, double& late)
{
	total = early = late = 0.;
	if (dose_per_frame_A2 <= 0.) return;

	const double cutoff_frame = (cutoff_A2 - pre_exposure_A2) / dose_per_frame_A2;

	for (size_t i = 1; i < shift_x.size(); i++)
	{
		const double dx = (shift_x[i] - shift_x[i - 1]) * pixel_size_A;
		const double dy = (shift_y[i] - shift_y[i - 1]) * pixel_size_A;
		const double d = std::sqrt(dx * dx + dy * dy);

		total += d;
		// 1-indexed, matching motioncorr_runner.cpp's own `frame` loop variable
		if ((double)(i + 1) <= cutoff_frame) early += d;
		else                                 late  += d;
	}
}

} // namespace cryosparc

#endif // CRYOSPARC_MOTION_MODEL_H
