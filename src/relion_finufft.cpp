#include "src/relion_finufft.h"

#include "src/spatial_frequency_grid.h"
#include "src/multidim_array.h"
#include "src/ctf.h"
#include "src/jaz/single_particle/obs_model.h"
#include "src/jaz/single_particle/image_log.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

#ifdef RELION_USE_FINUFFT
static thread_local int tls_finufft_nthreads = 1;
#endif

namespace
{

constexpr RFLOAT TWO_PI = 2.0 * PI;

// ---- BufferedImage bilinear helpers (file-local, used by obsModel helpers) ----

RFLOAT sampleRealFromHalfPlaneBilinear(const BufferedImage<RFLOAT>& img,
		RFLOAT x, RFLOAT y_signed, int size)
{
	const int sh = size / 2 + 1;
	RFLOAT xc = x;
	if (xc < 0.0) xc = 0.0;
	if (xc > (RFLOAT)(sh - 1)) xc = (RFLOAT)(sh - 1);

	RFLOAT yw = y_signed;
	while (yw < 0.0) yw += (RFLOAT)size;
	while (yw >= (RFLOAT)size) yw -= (RFLOAT)size;

	const int x0 = (int)FLOOR(xc);
	const int x1 = (x0 + 1 < sh) ? x0 + 1 : x0;
	const int y0 = (int)FLOOR(yw);
	const int y1 = (y0 + 1 < size) ? y0 + 1 : 0;

	const RFLOAT tx = xc - x0;
	const RFLOAT ty = yw - y0;

	const RFLOAT v00 = img(x0, y0);
	const RFLOAT v10 = img(x1, y0);
	const RFLOAT v01 = img(x0, y1);
	const RFLOAT v11 = img(x1, y1);

	const RFLOAT v0 = (1.0 - tx) * v00 + tx * v10;
	const RFLOAT v1 = (1.0 - tx) * v01 + tx * v11;

	return (1.0 - ty) * v0 + ty * v1;
}

Complex sampleComplexFromHalfPlaneBilinear(const BufferedImage<Complex>& img,
		RFLOAT x, RFLOAT y_signed, int size)
{
	const int sh = size / 2 + 1;
	RFLOAT xc = x;
	if (xc < 0.0) xc = 0.0;
	if (xc > (RFLOAT)(sh - 1)) xc = (RFLOAT)(sh - 1);

	RFLOAT yw = y_signed;
	while (yw < 0.0) yw += (RFLOAT)size;
	while (yw >= (RFLOAT)size) yw -= (RFLOAT)size;

	const int x0 = (int)FLOOR(xc);
	const int x1 = (x0 + 1 < sh) ? x0 + 1 : x0;
	const int y0 = (int)FLOOR(yw);
	const int y1 = (y0 + 1 < size) ? y0 + 1 : 0;

	const RFLOAT tx = xc - x0;
	const RFLOAT ty = yw - y0;

	const Complex c00 = img(x0, y0);
	const Complex c10 = img(x1, y0);
	const Complex c01 = img(x0, y1);
	const Complex c11 = img(x1, y1);

	const RFLOAT r00 = c00.real;
	const RFLOAT r10 = c10.real;
	const RFLOAT r01 = c01.real;
	const RFLOAT r11 = c11.real;

	const RFLOAT i00 = c00.imag;
	const RFLOAT i10 = c10.imag;
	const RFLOAT i01 = c01.imag;
	const RFLOAT i11 = c11.imag;

	const RFLOAT r0 = (1.0 - tx) * r00 + tx * r10;
	const RFLOAT r1 = (1.0 - tx) * r01 + tx * r11;
	const RFLOAT i0 = (1.0 - tx) * i00 + tx * i10;
	const RFLOAT i1 = (1.0 - tx) * i01 + tx * i11;

	return Complex((1.0 - ty) * r0 + ty * r1,
		       (1.0 - ty) * i0 + ty * i1);
}

// ---- type-3 FINUFFT evaluation (used only by debug comparison mode) ----

#ifdef RELION_USE_FINUFFT
void evaluateNonuniformFourierSamples2D_type3(
	const MultidimArray<RFLOAT>& image,
	const SpatialFrequencyGrid2D& grid,
	std::vector<Complex>& samples)
{
	const int size = XSIZE(image);
	const int64_t nj = (int64_t)size * (int64_t)size;
	const int64_t nk = (int64_t)grid.sample_x.size();

	// Source coordinates (not cached; only used for debug comparison)
	std::vector<RFLOAT> xj(nj), yj(nj);
	{
		const int x0 = STARTINGX(image);
		const int y0 = STARTINGY(image);
		int64_t idx = 0;
		for (int y = 0; y < size; y++)
		for (int x = 0; x < size; x++, idx++)
		{
			const RFLOAT xc = (RFLOAT)(x + x0);
			const RFLOAT yc = (RFLOAT)(y + y0);
			xj[idx] = TWO_PI * xc / (RFLOAT)size;
			yj[idx] = TWO_PI * yc / (RFLOAT)size;
		}
	}

	std::vector<std::complex<RFLOAT>> source_buf(nj), output_buf(nk);
	{
		int64_t idx = 0;
		for (int y = 0; y < size; y++)
		for (int x = 0; x < size; x++, idx++)
		{
			source_buf[idx] = std::complex<RFLOAT>(
				DIRECT_A2D_ELEM(image, y, x), 0.0);
		}
	}

	const std::vector<RFLOAT>& target_x = grid.finufft_target_x;
	const std::vector<RFLOAT>& target_y = grid.finufft_target_y;

	finufft_plan plan = NULL;
	finufft_opts opts;
	finufft_default_opts(&opts);
	opts.modeord = 0;
	const double tol = 1e-12;

	int ier = finufft_makeplan(3, 2, NULL, -1, 1, tol, &plan, &opts);
	if (ier > 1) handleFinufftError(ier, "finufft_makeplan type3 (debug)");

	ier = finufft_setpts(plan,
		nj, xj.data(), yj.data(), NULL,
		nk,
		const_cast<RFLOAT*>(target_x.data()),
		const_cast<RFLOAT*>(target_y.data()),
		NULL);
	if (ier > 1) handleFinufftError(ier, "finufft_setpts type3 (debug)");

	ier = finufft_execute(plan, source_buf.data(), output_buf.data());
	if (ier > 1) handleFinufftError(ier, "finufft_execute type3 (debug)");

	finufft_destroy(plan);

	const RFLOAT norm = 1.0 / ((RFLOAT)size * (RFLOAT)size);
	samples.resize(nk);
	for (long int idx = 0; idx < nk; idx++)
	{
		samples[idx] = Complex(
			output_buf[idx].real() * norm,
			output_buf[idx].imag() * norm);
	}
}
#endif

} // anonymous namespace

// ---- Public API ----

#ifdef RELION_USE_FINUFFT
void handleFinufftError(int status, const std::string& context)
{
	REPORT_ERROR("FINUFFT ERROR (ier=" + integerToString(status) + ") in " + context);
}

void setFinufftThreadCount(int nthreads)
{
	tls_finufft_nthreads = nthreads;
}

void evaluateNonuniformFourierSamples2D(
	const MultidimArray<RFLOAT>& image,
	const SpatialFrequencyGrid2D& grid,
	std::vector<Complex>& samples)
{
	if (image.getDim() != 2 || XSIZE(image) != YSIZE(image))
	{
		REPORT_ERROR("evaluateNonuniformFourierSamples2D requires a square 2D image.");
	}

	const int size = XSIZE(image);
	const int64_t nj = (int64_t)size * (int64_t)size;
	const int64_t nk = (int64_t)grid.finufft_target_x.size();

	// thread_local buffers and plan cache for OpenMP safety
	thread_local std::vector<std::complex<RFLOAT>> tls_source_buf, tls_output_buf;
	thread_local std::map<std::pair<int, const SpatialFrequencyGrid2D*>, FinufftPlanGuard> tls_plan_cache;

	tls_source_buf.resize(nj);
	tls_output_buf.resize(nk);

	// Fill source values from the real-space image.
	// After setXmippOrigin(), DIRECT_A2D_ELEM(img, 0, 0) = pixel at (-N/2, -N/2)
	// which matches FINUFFT modeord=0: index [0] = mode -N/2.
	{
		int64_t idx = 0;
		for (int y = 0; y < size; y++)
		for (int x = 0; x < size; x++, idx++)
		{
			tls_source_buf[idx] = std::complex<RFLOAT>(
				DIRECT_A2D_ELEM(image, y, x), 0.0);
		}
	}

	const std::vector<RFLOAT>& target_x = grid.finufft_target_x;
	const std::vector<RFLOAT>& target_y = grid.finufft_target_y;

	// Plan cache lookup
	auto cache_key = std::make_pair(size, &grid);
	auto plan_it = tls_plan_cache.find(cache_key);
	if (plan_it == tls_plan_cache.end())
	{
	FinufftPlanGuard guard;
	finufft_opts opts;
	finufft_default_opts(&opts);
	const double tol = 1e-6;
	opts.modeord = 0;
	opts.nthreads = tls_finufft_nthreads;
	int64_t n_modes[2] = {size, size};

		int ier = FINUFFT_MAKEPLAN(2, 2, n_modes, -1, 1, tol, &guard.plan, &opts);
		if (ier > 1) handleFinufftError(ier, "finufft_makeplan type2");

		// For type 2, the "nonuniform points" (xj/yj) are the target
		// frequency positions. No type-3-style "source-in-k" target set.
		ier = FINUFFT_SETPTS(guard.plan, nk,
			const_cast<RFLOAT*>(target_x.data()),
			const_cast<RFLOAT*>(target_y.data()),
			NULL,
			0, NULL, NULL, NULL);
		if (ier > 1) handleFinufftError(ier, "finufft_setpts type2");

		plan_it = tls_plan_cache.emplace(cache_key, std::move(guard)).first;
	}

	{
		// type 2: first arg (c) = output at nonuniform targets, second arg (f) = input uniform grid
		int ier_ex = FINUFFT_EXECUTE(plan_it->second.plan, tls_output_buf.data(), tls_source_buf.data());
		if (ier_ex > 1) handleFinufftError(ier_ex, "finufft_execute type2");
	}

	// FINUFFT computes the unnormalized DFT sum. RELION's FourierTransformer
	// normalizes the forward FFT by 1/(N*N). Apply the same normalization.
	const RFLOAT norm = 1.0 / ((RFLOAT)size * (RFLOAT)size);
	samples.resize(nk);
	for (long int idx = 0; idx < nk; idx++)
	{
		samples[idx] = Complex(
			tls_output_buf[idx].real() * norm,
			tls_output_buf[idx].imag() * norm);
	}

	// Debug comparison mode: run type 3 as well and compare
	const char* debug_env = std::getenv("RELION_S2_DEBUG_COMPARE_FINUFFT_TYPES");
	if (debug_env != NULL && std::string(debug_env) == "1")
	{
		std::vector<Complex> t3_samples;
		evaluateNonuniformFourierSamples2D_type3(image, grid, t3_samples);

		double max_diff = 0.0, sum_diff = 0.0;
		for (long int idx = 0; idx < nk; idx++)
		{
			const double dr = std::abs(samples[idx].real - t3_samples[idx].real);
			const double di = std::abs(samples[idx].imag - t3_samples[idx].imag);
			const double d = std::sqrt(dr*dr + di*di);
			if (d > max_diff) max_diff = d;
			sum_diff += d;
		}
		const double mean_diff = sum_diff / (double)nk;
		std::cerr << "FINUFFT DEBUG: type2 vs type3 comparison over "
			  << nk << " samples:" << std::endl;
		std::cerr << "  max absolute difference = " << max_diff << std::endl;
		std::cerr << "  mean absolute difference = " << mean_diff << std::endl;
	}
}

#endif // RELION_USE_FINUFFT

// ---- MultidimArray bilinear helpers ----

Complex sampleComplexFromFftwHalfBilinear(
	const MultidimArray<Complex>& img,
	RFLOAT x, RFLOAT y_signed, int size)
{
	const int sh = size / 2 + 1;
	RFLOAT xc = x;
	if (xc < 0.0) xc = 0.0;
	if (xc > (RFLOAT)(sh - 1)) xc = (RFLOAT)(sh - 1);

	RFLOAT yw = y_signed;
	while (yw < 0.0) yw += (RFLOAT)size;
	while (yw >= (RFLOAT)size) yw -= (RFLOAT)size;

	const int x0 = (int)FLOOR(xc);
	const int x1 = (x0 + 1 < sh) ? x0 + 1 : x0;
	const int y0 = (int)FLOOR(yw);
	const int y1 = (y0 + 1 < size) ? y0 + 1 : 0;

	const RFLOAT tx = xc - x0;
	const RFLOAT ty = yw - y0;

	const Complex c00 = DIRECT_A2D_ELEM(img, y0, x0);
	const Complex c10 = DIRECT_A2D_ELEM(img, y0, x1);
	const Complex c01 = DIRECT_A2D_ELEM(img, y1, x0);
	const Complex c11 = DIRECT_A2D_ELEM(img, y1, x1);

	const RFLOAT r0 = (1.0 - tx) * c00.real + tx * c10.real;
	const RFLOAT r1 = (1.0 - tx) * c01.real + tx * c11.real;
	const RFLOAT i0 = (1.0 - tx) * c00.imag + tx * c10.imag;
	const RFLOAT i1 = (1.0 - tx) * c01.imag + tx * c11.imag;

	return Complex((1.0 - ty) * r0 + ty * r1,
		       (1.0 - ty) * i0 + ty * i1);
}

RFLOAT sampleRealFromFftwHalfBilinear(
	const MultidimArray<RFLOAT>& img,
	RFLOAT x, RFLOAT y_signed, int size)
{
	const int sh = size / 2 + 1;
	RFLOAT xc = x;
	if (xc < 0.0) xc = 0.0;
	if (xc > (RFLOAT)(sh - 1)) xc = (RFLOAT)(sh - 1);

	RFLOAT yw = y_signed;
	while (yw < 0.0) yw += (RFLOAT)size;
	while (yw >= (RFLOAT)size) yw -= (RFLOAT)size;

	const int x0 = (int)FLOOR(xc);
	const int x1 = (x0 + 1 < sh) ? x0 + 1 : x0;
	const int y0 = (int)FLOOR(yw);
	const int y1 = (y0 + 1 < size) ? y0 + 1 : 0;

	const RFLOAT tx = xc - x0;
	const RFLOAT ty = yw - y0;

	const RFLOAT v00 = DIRECT_A2D_ELEM(img, y0, x0);
	const RFLOAT v10 = DIRECT_A2D_ELEM(img, y0, x1);
	const RFLOAT v01 = DIRECT_A2D_ELEM(img, y1, x0);
	const RFLOAT v11 = DIRECT_A2D_ELEM(img, y1, x1);

	const RFLOAT v0 = (1.0 - tx) * v00 + tx * v10;
	const RFLOAT v1 = (1.0 - tx) * v01 + tx * v11;

	return (1.0 - ty) * v0 + ty * v1;
}

// ---- Observation model corrections ----

void applyObsModelPhaseCorrectionToSamples(
	std::vector<Complex>& samples,
	const SpatialFrequencyGrid2D& grid,
	ObservationModel& obsModel,
	int optics_group,
	int size)
{
	if (!obsModel.hasOddZernike)
	{
		return;
	}

	const BufferedImage<Complex>& corr = obsModel.getPhaseCorrection(optics_group, size);

	for (long int idx = 0; idx < (long int)samples.size(); idx++)
	{
		Complex c = sampleComplexFromHalfPlaneBilinear(corr,
			grid.sample_x[idx],
			grid.sample_y[idx],
			size);

		const RFLOAT abs_c = abs(c);
		if (abs_c > 1e-12)
		{
			c /= abs_c;
		}

		samples[idx] *= conj(c);
	}
}

void applyObsModelMtfCorrectionToSamples(
	std::vector<Complex>& samples,
	const SpatialFrequencyGrid2D& grid,
	ObservationModel& obsModel,
	int optics_group,
	int size)
{
	if (!obsModel.hasMultipleMtfs)
	{
		return;
	}

	const BufferedImage<RFLOAT>& mtf = obsModel.getMtfImage(optics_group, size);
	const BufferedImage<RFLOAT>& avg_mtf = obsModel.getAverageMtfImage(size);

	for (long int idx = 0; idx < (long int)samples.size(); idx++)
	{
		const RFLOAT m = sampleRealFromHalfPlaneBilinear(mtf,
			grid.sample_x[idx],
			grid.sample_y[idx],
			size);
		const RFLOAT a = sampleRealFromHalfPlaneBilinear(avg_mtf,
			grid.sample_x[idx],
			grid.sample_y[idx],
			size);

		if (ABS(m) > 1e-12)
		{
			samples[idx] /= m;
			samples[idx] *= a;
		}
	}
}

void evaluateSymmetricAberrationAtSamplePositions(
	std::vector<RFLOAT>& gamma_values,
	const SpatialFrequencyGrid2D& grid,
	ObservationModel& obsModel,
	int optics_group, int size)
{
	gamma_values.assign(grid.sample_x.size(), 0.0);

	if (!obsModel.hasEvenZernike)
	{
		return;
	}

	const BufferedImage<RFLOAT>& gamma = obsModel.getGammaOffset(optics_group, size);

	for (long int idx = 0; idx < (long int)gamma_values.size(); idx++)
	{
		gamma_values[idx] = sampleRealFromHalfPlaneBilinear(gamma,
			grid.sample_x[idx],
			grid.sample_y[idx],
			size);
	}
}

// ---- CTF evaluation and application ----

void evaluateCtfAtSamplePositions(
	std::vector<RFLOAT>& ctf_values,
	const SpatialFrequencyGrid2D& grid,
	CTF& ctf, int size, RFLOAT angpix,
	bool ctf_phase_flipped, bool only_flip_phases,
	bool intact_ctf_first_peak,
	const std::vector<RFLOAT>* gamma_offsets)
{
	const RFLOAT xs = size * angpix;
	ctf_values.resize(grid.sample_x.size());

	for (long int idx = 0; idx < (long int)grid.sample_x.size(); idx++)
	{
		const RFLOAT x = grid.sample_x[idx] / xs;
		const RFLOAT y = grid.sample_y[idx] / xs;
		const RFLOAT gamma_offset = (gamma_offsets != NULL) ? (*gamma_offsets)[idx] : 0.0;
		ctf_values[idx] = ctf.getCTF(x, y,
			ctf_phase_flipped, only_flip_phases,
			intact_ctf_first_peak, true, gamma_offset);
	}
}

void applyCtfToSamples(
	std::vector<Complex>& samples,
	std::vector<RFLOAT>& sample_weight,
	const SpatialFrequencyGrid2D& grid,
	CTF& ctf, int size, RFLOAT angpix,
	bool ctf_phase_flipped, bool only_flip_phases,
	bool intact_ctf_first_peak, bool ctf_premultiplied,
	const std::vector<RFLOAT>* gamma_offsets,
	std::vector<RFLOAT>* ctf_values)
{
	std::vector<RFLOAT> local_ctf_values;
	if (ctf_values == NULL)
		ctf_values = &local_ctf_values;

	evaluateCtfAtSamplePositions(*ctf_values,
		grid, ctf, size, angpix,
		ctf_phase_flipped, only_flip_phases,
		intact_ctf_first_peak, gamma_offsets);

	for (long int idx = 0; idx < (long int)samples.size(); idx++)
	{
		const RFLOAT ctf_value = (*ctf_values)[idx];

		if (!ctf_premultiplied)
		{
			samples[idx] *= ctf_value;
		}
		sample_weight[idx] *= ctf_value * ctf_value;
	}
}

// ---- FOM weighting ----

void applyFomToSamples(
	std::vector<Complex>& samples,
	std::vector<RFLOAT>& sample_weight,
	RFLOAT fom)
{
	for (long int idx = 0; idx < (long int)samples.size(); idx++)
	{
		samples[idx] *= fom;
		sample_weight[idx] *= fom;
	}
}
