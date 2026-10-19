/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef GPU_RECONSTRUCT_H_
#define GPU_RECONSTRUCT_H_

/*
 * BackProjector::reconstruct() on the GPU, for 3D maps with skip_gridding (RELION's default): the
 * same steps as the CPU code, with the large arrays (the padded Fourier volume, about 9 GB in double
 * precision for a 384 box at padding 2) on the GPU and the inverse transform done by cuFFT.
 *
 * Opt-in while being validated: RELION_GPU_RECONSTRUCT=1, and only when a GPU was requested. When the
 * GPU cannot be used (no device, not enough free memory, any CUDA error) the function returns false
 * with the reason, and the caller does the reconstruction on the CPU.
 *
 * Results agree with the CPU to rounding: cuFFT and FFTW round differently, and the per-shell sums
 * of the weights are added up in a different order.
 */

#include "src/macros.h"
#include <string>
#include <vector>

namespace gpurec
{

struct SkipGriddingInput
{
	// BackProjector::data (complex, interleaved) and ::weight: zdim x ydim x xdim, logical origin
	// (start_z, start_y, start_x); the Fourier volume of the reconstruction has the same shape
	const RFLOAT *data = nullptr;
	const RFLOAT *weight = nullptr;
	long int zdim = 0, ydim = 0, xdim = 0;
	long int start_z = 0, start_y = 0, start_x = 0;

	int max_r2 = 0;            // ROUND(r_max * padding_factor)^2
	int round_max_r2 = 0;      // ROUND(r_max * padding_factor * r_max * padding_factor)
	int r_max = 0;
	float padding_factor = 2.f;

	// MAP: 1 / (oversampling_correction * tau2_fudge * tau2[ires]) per shell, and whether tau2 > 0
	bool do_map = false;
	int minres_map = 0;
	std::vector<RFLOAT> shell_invtau2;
	std::vector<char> shell_tau2_positive;

	int ori_size = 0;
	int padoridim = 0;         // the padded box of the inverse transform (even)
	RFLOAT normfft = 1.;

	// Precision on the GPU: by default that of RELION's GPU code (single unless built with
	// DoublePrec_ACC); RELION_GPU_RECONSTRUCT=double forces double
	bool double_precision = false;
};

/// Reconstruct; vol_out receives ori_size^3 values (z, y, x order), before softMaskOutsideMap() and
/// griddingCorrect(). Returns false, with the reason, when the CPU has to do it.
bool reconstructSkipGridding(const SkipGriddingInput &in, RFLOAT *vol_out, std::string &reason,
                             bool &warned_zero_weight);

/// Whether the GPU memory kept for the next expectation step stays in the card during the reconstruction
/// (default; RELION_GPU_KEEP_POOL=0 frees it first). False in a build without CUDA.
bool keepPoolActive();

/// Whether RELION_GPU_RECONSTRUCT asks for it: 1 (precision of the GPU code), single or double
bool requested();
/// Whether it asks for double precision (RELION_GPU_RECONSTRUCT=double, or a DoublePrec_ACC build)
bool doublePrecision();

/// Set by relion_refine: whether the job asked for a GPU (--gpu), also when --skip_align then turns
/// the GPU off for the expectation step
void setGpuRequested(bool r);
bool gpuRequested();

/// Whether a CUDA device is visible
bool deviceAvailable();

/// imposeHelicalSymmetryInRealSpace() on the GPU (double precision): the same mask, the same reads of
/// neighbours (as the original in-place loop saw them) and the same interpolation, one thread per voxel.
struct HelixRealSpaceInput
{
	long int zdim = 0, ydim = 0, xdim = 0, start_z = 0, start_y = 0, start_x = 0;
	RFLOAT r_min = 0., r_max = 0., d_min = 0., d_max = 0., D_min = 0., D_max = 0.;
	RFLOAT z_min = 0., z_max = 0., rise_pix = 1., cosine_width_pix = 1.;
	std::vector<RFLOAT> sin_rec, cos_rec;
};
/// v is replaced by the result; false (with the reason) when the CPU has to do it. bad: the
/// "ERROR in imposing symmetry" case of the CPU code
bool helicalSymmetryRealSpace(const HelixRealSpaceInput &in, RFLOAT *v, std::string &reason, bool &bad);

/// BackProjector::applyHelicalSymmetry() on the GPU (Fourier space, 3D): for every voxel the same
/// rotations, interpolation and sums, in the same order as the CPU loop, in the precision of RFLOAT.
/// A rotation about Z keeps z, so the volume is cut into groups of z planes that fit in the free GPU
/// memory (the group size is chosen from it); planes beyond the radius are not touched.
/// Opt-in: RELION_GPU_SYMMETRISE=1 (=check also runs the CPU and reports the difference), and only
/// when a GPU was requested.
struct HelixFourierInput
{
	long int zdim = 0, ydim = 0, xdim = 0, start_z = 0, start_y = 0, start_x = 0;
	int rmax2 = 0;
	// one entry per symmetry operator hh (identity excluded): R is the 3x3 rotation (row-major), and
	// cos/sin of the phase shift for every z plane (zdim values each; 1 and 0 without a rise)
	std::vector<RFLOAT> R;
	std::vector<RFLOAT> shift_cos, shift_sin;
	bool has_rise = false;
};
/// data (interleaved complex) and weight are replaced by the symmetrised arrays; false (with the
/// reason) when the CPU has to do it, in which case they are unchanged. partial: a CUDA error came after
/// some planes had been written back, so the arrays are unusable and the job must stop.
bool helicalSymmetryFourier(const HelixFourierInput &in, RFLOAT *data, RFLOAT *weight, std::string &reason,
                            std::string &choice, bool &partial);
/// RELION_GPU_SYMMETRISE: 0 off, 1 on, 2 on and checked against the CPU
int symmetriseMode();

}

#endif /* GPU_RECONSTRUCT_H_ */
