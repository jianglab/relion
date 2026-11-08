/***************************************************************************
 *
 * Author: "Jasenko Zivanov"
 * MRC Laboratory of Molecular Biology
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
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "new_ft.h"

#include "src/macros.h"
#include "src/fftw.h"
#include "src/args.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fstream>
#include <string>
#include <math.h>


void NewFFT::FourierTransform(
		MultidimArray<double>& src,
		MultidimArray<dComplex>& dest,
		const NewFFT::DoublePlan& plan,
		Normalization normalization)
{
	if (!plan.isCompatible(src))
	{
		REPORT_ERROR("ERROR: plan incompatible with input array\n");
	}
	
	if (!plan.isCompatible(dest))
	{
		if (plan.isReusable())
		{
			dest.resizeNoCp(src.ndim, src.zdim, src.ydim, src.xdim/2 + 1);
		}
		else
		{
			REPORT_ERROR("NewFFT::FourierTransform: plan incompatible with output array\n");
		}
	}
	
	_FourierTransform(src, dest, plan, normalization);
}

void NewFFT::inverseFourierTransform(
		MultidimArray<dComplex>& src,
		MultidimArray<double>& dest,
		const NewFFT::DoublePlan& plan,
		Normalization normalization,
		bool preserveInput)
{
	if (preserveInput && !plan.isReusable())
	{
		REPORT_ERROR("NewFFT::inverseFourierTransform: preserveInput is only supported for reusable plans\n");
	}
	
	if (!plan.isCompatible(src))
	{
		REPORT_ERROR("NewFFT::inverseFourierTransform: plan incompatible with input array\n");
	}
	
	if (!plan.isCompatible(dest))
	{
		if (plan.isReusable())
		{
			dest.resizeNoCp(src.ndim, src.zdim, src.ydim, 2*(src.xdim-1));
		}
		else
		{
			REPORT_ERROR("NewFFT::inverseFourierTransform: plan incompatible with output array\n");
		}
	}
	
	MultidimArray<dComplex> src2;
	
	if (preserveInput)
	{
		src2 = src;
		_inverseFourierTransform(src2, dest, plan, normalization);
	}
	else
	{
		_inverseFourierTransform(src, dest, plan, normalization);
	}
}

void NewFFT::FourierTransform(
		MultidimArray<float>& src,
		MultidimArray<fComplex>& dest,
		const NewFFT::FloatPlan& plan,
		Normalization normalization)
{
	if (!plan.isCompatible(src))
	{
		REPORT_ERROR("NewFFT::FourierTransform: plan incompatible with input array\n");
	}
	
	if (!plan.isCompatible(dest))
	{
		if (plan.isReusable())
		{
			dest.resizeNoCp(src.ndim, src.zdim, src.ydim, src.xdim/2 + 1);
		}
		else
		{
			REPORT_ERROR("NewFFT::FourierTransform: plan incompatible with output array\n");
		}
	}
	
	_FourierTransform(src, dest, plan, normalization);
}

void NewFFT::inverseFourierTransform(
		MultidimArray<fComplex>& src,
		MultidimArray<float>& dest,
		const NewFFT::FloatPlan& plan,
		Normalization normalization,
		bool preserveInput)
{
	if (preserveInput && !plan.isReusable())
	{
		REPORT_ERROR("NewFFT::inverseFourierTransform: preserveInput is only supported for reusable plans\n");
	}
	
	if (!plan.isCompatible(src))
	{
		REPORT_ERROR("NewFFT::inverseFourierTransform: plan incompatible with input array\n");
	}
	
	if (!plan.isCompatible(dest))
	{
		if (plan.isReusable())
		{
			dest.resizeNoCp(src.ndim, src.zdim, src.ydim, 2*(src.xdim-1));
		}
		else
		{
			REPORT_ERROR("NewFFT::inverseFourierTransform: plan incompatible with output array\n");
		}
	}
	
	MultidimArray<fComplex> src2;
	
	if (preserveInput)
	{
		src2 = src;
		_inverseFourierTransform(src2, dest, plan, normalization);
	}
	else
	{
		_inverseFourierTransform(src, dest, plan, normalization);
	}
}



void NewFFT::FourierTransform(
		MultidimArray<double>& src,
		MultidimArray<dComplex>& dest,
		Normalization normalization)
{
	if (!areSizesCompatible(src, dest))
	{
		resizeComplexToMatch(src, dest);
	}
			
	DoublePlan p(src, dest);
	_FourierTransform(src, dest, p, normalization);
}

void NewFFT::inverseFourierTransform(
		MultidimArray<dComplex>& src,
		MultidimArray<double>& dest,
		Normalization normalization,
		bool preserveInput)
{
	if (!areSizesCompatible(dest, src))
	{
		resizeRealToMatch(dest, src);
	}
			
	if (preserveInput)
	{
		MultidimArray<dComplex> src2 = src;
		DoublePlan p(dest, src2);
		_inverseFourierTransform(src2, dest, p, normalization);
	}
	else
	{
		DoublePlan p(dest, src);
		_inverseFourierTransform(src, dest, p, normalization);
	}
}

void NewFFT::FourierTransform(
		MultidimArray<float>& src,
		MultidimArray<fComplex>& dest,
		Normalization normalization,
		unsigned int plan_flags)
{
	if (!areSizesCompatible(src, dest))
	{
		resizeComplexToMatch(src, dest);
	}
			
	FloatPlan p(src, dest, plan_flags);
	_FourierTransform(src, dest, p, normalization);
}

void NewFFT::inverseFourierTransform(
		MultidimArray<fComplex>& src,
		MultidimArray<float>& dest,
		Normalization normalization,
		bool preserveInput,
		unsigned int plan_flags)
{
	if (!areSizesCompatible(dest, src))
	{
		resizeRealToMatch(dest, src);
	}
			
	if (preserveInput)
	{
		MultidimArray<fComplex> src2 = src;
		FloatPlan p(dest, src2, plan_flags);
		_inverseFourierTransform(src2, dest, p, normalization);
	}
	else
	{
		FloatPlan p(dest, src, plan_flags);
		_inverseFourierTransform(src, dest, p, normalization);
	}
}


void NewFFT::_FourierTransform(
		MultidimArray<double>& src,
		MultidimArray<dComplex>& dest,
		const NewFFT::DoublePlan& plan,
		Normalization normalization)
{
	fftw_execute_dft_r2c(plan.getForward(),
						 MULTIDIM_ARRAY(src), (fftw_complex*) MULTIDIM_ARRAY(dest));
	
	if (normalization == FwdOnly)
	{
		const double scale = MULTIDIM_SIZE(src);
		
		for (long int i = 0; i < NZYXSIZE(dest); i++)
		{
			dest.data[i] /= scale;
		}
	}
	else if (normalization == Both)
	{
		const double scale = sqrt(MULTIDIM_SIZE(src));
		
		for (long int i = 0; i < NZYXSIZE(dest); i++)
		{
			dest.data[i] /= scale;
		}
	}
}

void NewFFT::_inverseFourierTransform(
		MultidimArray<dComplex>& src,
		MultidimArray<double>& dest,
		const NewFFT::DoublePlan& plan,
		Normalization normalization)
{
	fftw_complex* in = (fftw_complex*) MULTIDIM_ARRAY(src);
	
	fftw_execute_dft_c2r(plan.getBackward(), in, MULTIDIM_ARRAY(dest));
	
	if (normalization == Both)
	{
		const double scale = sqrt(MULTIDIM_SIZE(dest));
		
		for (long int i = 0; i < NZYXSIZE(dest); i++)
		{
			dest.data[i] /= scale;
		}
	}
}

void NewFFT::_FourierTransform(
		MultidimArray<float>& src,
		MultidimArray<fComplex>& dest,
		const NewFFT::FloatPlan& plan,
		Normalization normalization)
{
	fftwf_execute_dft_r2c(plan.getForward(),
						  MULTIDIM_ARRAY(src), (fftwf_complex*) MULTIDIM_ARRAY(dest));
	
	if (normalization == FwdOnly)
	{
		const float scale = MULTIDIM_SIZE(src);
		const long int n = MULTIDIM_SIZE(src);

		const float cd = scale * scale;
		if ((n & (n - 1)) == 0 && cd == (double) n * n)
		{
			// What dest.data[i] /= scale computes (a complex division by (scale, 0)), with the division by
			// scale^2 done as a multiplication: for a power of two the two give the same bits, signed zeros included
			const float inv_cd = 1.f / cd;
			for (long int i = 0; i < NZYXSIZE(dest); i++)
			{
				const float re = dest.data[i].real * scale + dest.data[i].imag * 0.f;
				const float im = dest.data[i].imag * scale - dest.data[i].real * 0.f;
				dest.data[i].real = re * inv_cd;
				dest.data[i].imag = im * inv_cd;
			}
		}
		else
		for (long int i = 0; i < NZYXSIZE(dest); i++)
		{
			dest.data[i] /= scale;
		}
	}
	else if (normalization == Both)
	{
		const float scale = sqrt(MULTIDIM_SIZE(src));
		
		for (long int i = 0; i < NZYXSIZE(dest); i++)
		{
			dest.data[i] /= scale;
		}
	}
}

void NewFFT::_inverseFourierTransform(
		MultidimArray<fComplex>& src,
		MultidimArray<float>& dest,
		const NewFFT::FloatPlan& plan,
		Normalization normalization)
{	
	fftwf_complex* in = (fftwf_complex*) MULTIDIM_ARRAY(src);
	
	fftwf_execute_dft_c2r(plan.getBackward(), in, MULTIDIM_ARRAY(dest));
	
	if (normalization == Both)
	{
		const float scale = sqrt(MULTIDIM_SIZE(dest));
		
		for (long int i = 0; i < NZYXSIZE(dest); i++)
		{
			dest.data[i] /= scale;
		}
	}
}


NewFFT::DoublePlan::DoublePlan(int w, int h, int d, unsigned int flags)
:   
	reusable(true), 
	w(w), h(h), d(d),
	realPtr(0), 
	complexPtr(0)
{
	MultidimArray<double> realDummy(d,h,w);
	MultidimArray<dComplex> complexDummy(d,h,w/2+1);
	
	std::vector<int> N(0);
	if (d > 1) N.push_back(d);
	if (h > 1) N.push_back(h);
	           N.push_back(w);
	
	const int ndim = N.size();

	fftw_plan planForward, planBackward;	
	#pragma omp critical(FourierTransformer_fftw_plan)
	{
		planForward = fftw_plan_dft_r2c(
				ndim, &N[0],
				MULTIDIM_ARRAY(realDummy),
				(fftw_complex*) MULTIDIM_ARRAY(complexDummy),
				FFTW_UNALIGNED | flags);
		
		planBackward = fftw_plan_dft_c2r(
				ndim, &N[0],
				(fftw_complex*) MULTIDIM_ARRAY(complexDummy),
				MULTIDIM_ARRAY(realDummy),
				FFTW_UNALIGNED | flags);
	}	
	
	if (planForward == NULL || planBackward == NULL)
	{
		REPORT_ERROR("FFTW plans cannot be created");
	}
	
	plan = std::shared_ptr<Plan>(new Plan(planForward, planBackward));
}

NewFFT::DoublePlan::DoublePlan(
		MultidimArray<double>& real,
		MultidimArray<dComplex>& complex,
		unsigned int flags)
:   
	reusable(flags & FFTW_UNALIGNED), 
	w(real.xdim), h(real.ydim), d(real.zdim),
	realPtr(MULTIDIM_ARRAY(real)), 
	complexPtr((double*)MULTIDIM_ARRAY(complex))
{
	std::vector<int> N(0);
	if (d > 1) N.push_back(d);
	if (h > 1) N.push_back(h);
	           N.push_back(w);
	
	const int ndim = N.size();
	
	fftw_plan planForward, planBackward;	
	#pragma omp critical(FourierTransformer_fftw_plan)
	{
		planForward = fftw_plan_dft_r2c(
				ndim, &N[0],
				MULTIDIM_ARRAY(real),
				(fftw_complex*) MULTIDIM_ARRAY(complex),
				flags);
		
		planBackward = fftw_plan_dft_c2r(
				ndim, &N[0],
				(fftw_complex*) MULTIDIM_ARRAY(complex),
				MULTIDIM_ARRAY(real),
				flags);
	}	
	
	if (planForward == NULL || planBackward == NULL)
	{
		REPORT_ERROR("FFTW plans cannot be created");
	}
	
	plan = std::shared_ptr<Plan>(new Plan(planForward, planBackward));
}

NewFFT::FloatPlan::FloatPlan(int w, int h, int d, unsigned int flags)
:	reusable(true), 
	w(w), h(h), d(d),
	realPtr(0), 
	complexPtr(0)
{
	MultidimArray<float> realDummy(d,h,w);
	MultidimArray<fComplex> complexDummy(d,h,w/2+1);
	
	std::vector<int> N(0);
	if (d > 1) N.push_back(d);
	if (h > 1) N.push_back(h);
	           N.push_back(w);
	
	const int ndim = N.size();
	
	
	fftwf_plan planForward, planBackward;	
	#pragma omp critical(FourierTransformer_fftw_plan)
	{
		planForward = fftwf_plan_dft_r2c(
				ndim, &N[0],
				MULTIDIM_ARRAY(realDummy),
				(fftwf_complex*) MULTIDIM_ARRAY(complexDummy),
				FFTW_UNALIGNED | flags);
		
		planBackward = fftwf_plan_dft_c2r(
				ndim, &N[0],
				(fftwf_complex*) MULTIDIM_ARRAY(complexDummy),
				MULTIDIM_ARRAY(realDummy),
				FFTW_UNALIGNED | flags);
	}

	if (planForward == NULL || planBackward == NULL)
	{
		REPORT_ERROR("FFTW plans cannot be created");
	}
	
	plan = std::shared_ptr<Plan>(new Plan(planForward, planBackward));
}


// Measured plans (NewFFT::MEASURED): FFTW_MEASURE overwrites the arrays it plans on, so a plan that is not yet in
// the wisdom is measured on scratch arrays with the same alignment, and the wisdom is saved for later runs.
namespace
{

bool measuredPlansWanted()
{
	// On by default (only MotionCorr asks for them); RELION_FFTW_PLANS=estimate turns them off
	const char *mode = getenv("RELION_FFTW_PLANS");
	return mode == NULL || std::string(mode) != "estimate";
}

std::string newFftWisdomFile()
{
	const char *wf = getenv("RELION_FFTW_NEWFFT_WISDOM");
	if (wf != NULL && wf[0] != '\0') return wf;
	const char *xdg = getenv("XDG_CACHE_HOME"), *home = getenv("HOME");
	const std::string dir = (xdg != NULL && xdg[0] != '\0') ? std::string(xdg) :
	                        (home != NULL ? std::string(home) + "/.cache" : std::string());
	if (dir.empty()) return "";
	std::string cpu = "unknown_cpu";
	std::ifstream in("/proc/cpuinfo");
	std::string line;
	while (std::getline(in, line))
		if (line.compare(0, 10, "model name") == 0)
		{
			std::string out;
			for (char ch : line.substr(line.find(':') + 1))
				if (isalnum((unsigned char)ch)) out += ch;
				else if (!out.empty() && out.back() != '_') out += '_';
			while (!out.empty() && out.back() == '_') out.pop_back();
			if (!out.empty()) cpu = out;
			break;
		}
	return dir + "/relion/fftw_wisdom_newfft_single_" + cpu + ".txt";
}

// Inside the planner's critical section
void measuredFloatPlans(int ndim, const int *N, float *real, fftwf_complex *complex, unsigned int flags,
                        fftwf_plan &forward, fftwf_plan &backward)
{
	static bool loaded = false;
	static const std::string file = newFftWisdomFile();
	if (!loaded)
	{
		loaded = true;
		if (!file.empty()) fftwf_import_wisdom_from_filename(file.c_str());   // a missing file is fine
	}

	const unsigned int wo = FFTW_MEASURE | FFTW_WISDOM_ONLY | flags;
	forward = fftwf_plan_dft_r2c(ndim, N, real, complex, wo);
	backward = fftwf_plan_dft_c2r(ndim, N, complex, real, wo);
	if (forward != NULL && backward != NULL) return;
	if (forward != NULL) fftwf_destroy_plan(forward);
	if (backward != NULL) fftwf_destroy_plan(backward);

	size_t n = 1;
	for (int d = 0; d < ndim; d++) n *= N[d];
	const size_t nhalf = n / N[ndim - 1] * (N[ndim - 1] / 2 + 1);
	// scratch whose address has the same remainder modulo 64 bytes as the real arrays
	char *rb = (char*) malloc(n * sizeof(float) + 128), *cb = (char*) malloc(nhalf * sizeof(fftwf_complex) + 128);
	forward = backward = NULL;
	if (rb != NULL && cb != NULL)
	{
		float *sr = (float*)(rb + (64 - (uintptr_t)rb % 64) % 64 + (uintptr_t)real % 64);
		fftwf_complex *sc = (fftwf_complex*)(cb + (64 - (uintptr_t)cb % 64) % 64 + (uintptr_t)complex % 64);
		fftwf_plan f = fftwf_plan_dft_r2c(ndim, N, sr, sc, FFTW_MEASURE | flags);
		fftwf_plan b = fftwf_plan_dft_c2r(ndim, N, sc, sr, FFTW_MEASURE | flags);
		if (f != NULL) fftwf_destroy_plan(f);
		if (b != NULL) fftwf_destroy_plan(b);
		if (f != NULL && b != NULL && !file.empty())
		{
			// merge with what other processes saved meanwhile, then replace the file in one step
			const std::string dir = file.substr(0, file.rfind('/'));
			mkdir(dir.substr(0, dir.rfind('/')).c_str(), 0755);
			mkdir(dir.c_str(), 0755);
			fftwf_import_wisdom_from_filename(file.c_str());
			const std::string tmp = file + ".tmp" + std::to_string((long) getpid());
			if (fftwf_export_wisdom_to_filename(tmp.c_str())) rename(tmp.c_str(), file.c_str());
			else remove(tmp.c_str());
		}
	}
	free(rb);
	free(cb);

	// the wisdom now holds the plan; the plans for the real arrays use it (an estimate if it still fails)
	forward = fftwf_plan_dft_r2c(ndim, N, real, complex, FFTW_MEASURE | FFTW_WISDOM_ONLY | flags);
	backward = fftwf_plan_dft_c2r(ndim, N, complex, real, FFTW_MEASURE | FFTW_WISDOM_ONLY | flags);
	if (forward == NULL || backward == NULL)
	{
		if (forward != NULL) fftwf_destroy_plan(forward);
		if (backward != NULL) fftwf_destroy_plan(backward);
		forward = fftwf_plan_dft_r2c(ndim, N, real, complex, FFTW_ESTIMATE | flags);
		backward = fftwf_plan_dft_c2r(ndim, N, complex, real, FFTW_ESTIMATE | flags);
	}
}

}

NewFFT::FloatPlan::FloatPlan(
		MultidimArray<float>& real,
		MultidimArray<fComplex>& complex,
		unsigned int flags)
:
	reusable(flags & FFTW_UNALIGNED), 
	w(real.xdim), h(real.ydim), d(real.zdim),
	realPtr(MULTIDIM_ARRAY(real)), 
	complexPtr((float*)MULTIDIM_ARRAY(complex))
{
	std::vector<int> N(0);
	if (d > 1) N.push_back(d);
	if (h > 1) N.push_back(h);
	           N.push_back(w);
	
	const int ndim = N.size();
	
	
	fftwf_plan planForward, planBackward;	
	const bool measured = (flags & NewFFT::MEASURED) != 0 && measuredPlansWanted();
	if (flags & NewFFT::MEASURED)
	{
		flags &= ~NewFFT::MEASURED;
		if (!measured) flags |= FFTW_ESTIMATE;   // planning with no flag would measure, on the caller's arrays
	}
	#pragma omp critical(FourierTransformer_fftw_plan)
	{
		if (measured)
		{
			measuredFloatPlans(ndim, &N[0], MULTIDIM_ARRAY(real), (fftwf_complex*) MULTIDIM_ARRAY(complex),
			                   flags, planForward, planBackward);
		}
		else
		{
			planForward = fftwf_plan_dft_r2c(
					ndim, &N[0],
					MULTIDIM_ARRAY(real),
					(fftwf_complex*) MULTIDIM_ARRAY(complex),
					flags);

			planBackward = fftwf_plan_dft_c2r(
					ndim, &N[0],
					(fftwf_complex*) MULTIDIM_ARRAY(complex),
					MULTIDIM_ARRAY(real),
					flags);
		}
	}
	
	if (planForward == NULL || planBackward == NULL)
	{
		REPORT_ERROR("FFTW plans cannot be created");
	}
	
	plan = std::shared_ptr<Plan>(new Plan(planForward, planBackward));
}
