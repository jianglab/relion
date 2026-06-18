#ifndef SRC_RELION_FINUFFT_H_
#define SRC_RELION_FINUFFT_H_

#include <vector>
#include <complex>
#include <string>
#include <map>

#include "src/macros.h"
#include "src/complex.h"
#include "src/multidim_array.h"

struct SpatialFrequencyGrid2D;
class ObservationModel;
class CTF;
template <typename T> class BufferedImage;
template <typename T> class MultidimArray;

#ifdef RELION_USE_FINUFFT
#include <finufft.h>

#ifdef RELION_SINGLE_PRECISION
typedef finufftf_plan FinufftPlanHandle;
#define FINUFFT_MAKEPLAN finufftf_makeplan
#define FINUFFT_SETPTS   finufftf_setpts
#define FINUFFT_EXECUTE  finufftf_execute
#define FINUFFT_DESTROY  finufftf_destroy
#else
typedef finufft_plan FinufftPlanHandle;
#define FINUFFT_MAKEPLAN finufft_makeplan
#define FINUFFT_SETPTS   finufft_setpts
#define FINUFFT_EXECUTE  finufft_execute
#define FINUFFT_DESTROY  finufft_destroy
#endif

struct FinufftPlanGuard
{
        FinufftPlanHandle plan;
        FinufftPlanGuard() : plan(nullptr) {}
        ~FinufftPlanGuard() { /* defer to explicit clear() to avoid FFTW teardown race */ }
        void clear() { if (plan) { FINUFFT_DESTROY(plan); plan = nullptr; } }
        FinufftPlanGuard(const FinufftPlanGuard&) = delete;
        FinufftPlanGuard& operator=(const FinufftPlanGuard&) = delete;
        FinufftPlanGuard(FinufftPlanGuard&& other) : plan(other.plan) { other.plan = nullptr; }
        FinufftPlanGuard& operator=(FinufftPlanGuard&& other)
        {
                clear();
                plan = other.plan;
                other.plan = nullptr;
                return *this;
        }
};

void handleFinufftError(int status, const std::string& context);

// Evaluate the forward Fourier transform of a real-space image at the
// nonuniform frequency positions defined by grid.finufft_target_x/y.
// Uses FINUFFT type 2 (uniform real-space grid -> nonuniform Fourier samples).
// The image must have setXmippOrigin() called (DIRECT_A2D_ELEM layout
// must be modeord=0 compatible: index [0] = spatial coordinate -N/2).
void evaluateNonuniformFourierSamples2D(
const MultidimArray<RFLOAT>& image,
const SpatialFrequencyGrid2D& grid,
std::vector<Complex>& samples);

void setFinufftThreadCount(int nthreads);

#endif // RELION_USE_FINUFFT

#ifndef RELION_USE_FINUFFT
inline void setFinufftThreadCount(int) {}
#endif

// Bilinear interpolation helpers for MultidimArray containers
inline Complex bilinearSampleCoeffs(const MultidimArray<Complex>& img,
		int x0, int x1, int y0, int y1,
		RFLOAT tx, RFLOAT ty)
{
	const RFLOAT v00_r = DIRECT_A2D_ELEM(img, y0, x0).real;
	const RFLOAT v10_r = DIRECT_A2D_ELEM(img, y0, x1).real;
	const RFLOAT v01_r = DIRECT_A2D_ELEM(img, y1, x0).real;
	const RFLOAT v11_r = DIRECT_A2D_ELEM(img, y1, x1).real;
	const RFLOAT v00_i = DIRECT_A2D_ELEM(img, y0, x0).imag;
	const RFLOAT v10_i = DIRECT_A2D_ELEM(img, y0, x1).imag;
	const RFLOAT v01_i = DIRECT_A2D_ELEM(img, y1, x0).imag;
	const RFLOAT v11_i = DIRECT_A2D_ELEM(img, y1, x1).imag;

	const RFLOAT r0 = (1.0 - tx) * v00_r + tx * v10_r;
	const RFLOAT r1 = (1.0 - tx) * v01_r + tx * v11_r;
	const RFLOAT i0 = (1.0 - tx) * v00_i + tx * v10_i;
	const RFLOAT i1 = (1.0 - tx) * v01_i + tx * v11_i;

	return Complex((1.0 - ty) * r0 + ty * r1,
		       (1.0 - ty) * i0 + ty * i1);
}

inline RFLOAT bilinearRealCoeffs(const MultidimArray<RFLOAT>& img,
		int x0, int x1, int y0, int y1,
		RFLOAT tx, RFLOAT ty)
{
	const RFLOAT v00 = DIRECT_A2D_ELEM(img, y0, x0);
	const RFLOAT v10 = DIRECT_A2D_ELEM(img, y0, x1);
	const RFLOAT v01 = DIRECT_A2D_ELEM(img, y1, x0);
	const RFLOAT v11 = DIRECT_A2D_ELEM(img, y1, x1);

	const RFLOAT v0 = (1.0 - tx) * v00 + tx * v10;
	const RFLOAT v1 = (1.0 - tx) * v01 + tx * v11;

	return (1.0 - ty) * v0 + ty * v1;
}

Complex sampleComplexFromFftwHalfBilinear(
	const MultidimArray<Complex>& img,
	RFLOAT x, RFLOAT y_signed, int size);

RFLOAT sampleRealFromFftwHalfBilinear(
	const MultidimArray<RFLOAT>& img,
	RFLOAT x, RFLOAT y_signed, int size);

// Observation model corrections at s2 sample positions
void applyObsModelPhaseCorrectionToSamples(
	std::vector<Complex>& samples,
	const SpatialFrequencyGrid2D& grid,
	ObservationModel& obsModel,
	int optics_group, int size);

void applyObsModelMtfCorrectionToSamples(
	std::vector<Complex>& samples,
	const SpatialFrequencyGrid2D& grid,
	ObservationModel& obsModel,
	int optics_group, int size);

void evaluateSymmetricAberrationAtSamplePositions(
	std::vector<RFLOAT>& gamma_values,
	const SpatialFrequencyGrid2D& grid,
	ObservationModel& obsModel,
	int optics_group, int size);

// CTF evaluation at s2 sample positions
void evaluateCtfAtSamplePositions(
	std::vector<RFLOAT>& ctf_values,
	const SpatialFrequencyGrid2D& grid,
	CTF& ctf, int size, RFLOAT angpix,
	bool ctf_phase_flipped, bool only_flip_phases,
	bool intact_ctf_first_peak,
	const std::vector<RFLOAT>* gamma_offsets);

// Combined CTF application: evaluates CTF at each s2 position and multiplies
// samples (if !ctf_premultiplied) and sample_weight (always) by CTF/CTF^2.
// The ctf_values vector is filled as a side effect if non-NULL.
void applyCtfToSamples(
	std::vector<Complex>& samples,
	std::vector<RFLOAT>& sample_weight,
	const SpatialFrequencyGrid2D& grid,
	CTF& ctf, int size, RFLOAT angpix,
	bool ctf_phase_flipped, bool only_flip_phases,
	bool intact_ctf_first_peak, bool ctf_premultiplied,
	const std::vector<RFLOAT>* gamma_offsets,
	std::vector<RFLOAT>* ctf_values = NULL);

// FOM weighting
void applyFomToSamples(
	std::vector<Complex>& samples,
	std::vector<RFLOAT>& sample_weight,
	RFLOAT fom);

#endif // SRC_S2_FINUFFT_H_
