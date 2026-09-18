#ifndef SRC_FINUFFT_CENTRAL_SLICE_H_
#define SRC_FINUFFT_CENTRAL_SLICE_H_

// Exact central-slice extraction from a gridded Fourier volume.
//
// Kept in its own header (rather than in relion_finufft.h) so that projector.h
// can pull in FinufftProjectorModes without also pulling in the s2-mode helpers.

#include <vector>
#include <complex>

#include "src/macros.h"
#include "src/complex.h"

// ---------------------------------------------------------------------------
// Exact central-slice extraction from a gridded Fourier volume
// ---------------------------------------------------------------------------
//
// Projector::project() has to evaluate a uniformly sampled Fourier volume
// (Projector::data) at non-integer positions (xp, yp, zp) that lie on a plane
// through the origin.  The classical implementation approximates that with a
// trilinear stencil on a padding_factor=2 grid, which leaks power near the
// padded grid's own Nyquist frequency.
//
// The exact alternative is the band-limited (trigonometric-polynomial)
// interpolant of the sampled volume.  Writing P for the size of the periodic
// real-space box that the Fourier grid samples (Projector's "padoridim"), the
// interpolant of a Fourier grid F[n] at an arbitrary real-valued position q is
//
//     g(q) = (1 / P^dim) * sum_r  V[r] * exp(-2*pi*i * (q . r) / P)
//     with  V[r] = sum_n F[n] * exp(+2*pi*i * (n . r) / P)
//
// i.e. V is the unnormalised inverse DFT of the Fourier volume - which is just
// the padded *real-space* volume.  Evaluating g at many nonuniform q is
// precisely a FINUFFT type-2 transform whose *mode array is V*, not F.
//
// Two consequences that are worth spelling out, because they are easy to get
// backwards:
//
//   * The half-Hermitian storage of Projector::data needs no special handling
//     on the nonuniform side.  V is real, so the interpolant automatically
//     satisfies g(-q) = conj(g(q)); there is no need to split the query points
//     by the sign of xp and conjugate one group, as a naive reading of
//     "feed data to finufft3d2" would require.
//
//   * Because FINUFFT does its own internal upsampling and gridding, V does not
//     have to be kept at the full padded size.  V is (up to the spherical
//     band-limit applied to `data`) supported inside the original, unpadded box,
//     so cropping it to `mode_size` samples per dimension is both exact enough
//     and a large saving in memory and in the size of FINUFFT's internal FFT.
//
// FinufftProjectorModes holds V in the layout FINUFFT expects (CMCL mode
// ordering: index 0 is mode -mode_size/2).  It is deliberately free of any
// FINUFFT type so that Projector can hold one even in builds without FINUFFT.

struct FinufftProjectorModes
{
	// 2 for a 2D reference (in-plane rotation), 3 for a 3D reference (projection)
	int dim;
	// Number of retained samples of V per dimension (even)
	int mode_size;
	// Size of the periodic real-space box that the Fourier grid samples
	int padded_size;
	// V, cropped to mode_size^dim, in CMCL mode order, x fastest
	std::vector<std::complex<RFLOAT> > modes;

	FinufftProjectorModes() : dim(0), mode_size(0), padded_size(0) {}

	bool isPrepared() const { return !modes.empty(); }

	void clear()
	{
		dim = mode_size = padded_size = 0;
		std::vector<std::complex<RFLOAT> >().swap(modes);
	}

	size_t memoryBytes() const { return modes.size() * sizeof(std::complex<RFLOAT>); }
};

// Evaluate the interpolant described above at n_points nonuniform positions.
// qz is ignored (and may be NULL) when modes.dim == 2.  The query positions are
// in the same units as Projector::data's logical indices, i.e. padded Fourier
// pixels.  samples_out must have room for n_points entries.
//
// Throws (REPORT_ERROR) in builds without FINUFFT support.
// upsampfac is FINUFFT's internal upsampling factor: its FFT is
// (upsampfac * mode_size)^dim, so 1.25 costs roughly a quarter of the FFT work of
// 2.0 in 3D at the price of a wider spreading kernel.  0 lets FINUFFT choose.
void evaluateNonuniformFourierSamplesFromFourierVolume3D(
		const FinufftProjectorModes& modes,
		const RFLOAT* qx, const RFLOAT* qy, const RFLOAT* qz,
		size_t n_points,
		Complex* samples_out,
		double tol = 1e-6,
		double upsampfac = 0.0);

// True when this build can actually run the above.
bool haveFinufftSupport();

#endif // SRC_FINUFFT_CENTRAL_SLICE_H_
