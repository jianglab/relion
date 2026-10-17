/***************************************************************************
 *
 * Author: "Sjors H.W. Scheres"
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
/***************************************************************************
 *
 * Authors:    Roberto Marabini					(roberto@cnb.csic.es)
 *			   Carlos Oscar S. Sorzano			(coss@cnb.csic.es)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *	All comments concerning this program package may be sent to the
 *	e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#include "src/macros.h"
#include "src/fftw.h"
#include "src/rect_refusal.h"
#include "src/args.h"
#include <string.h>
#include <math.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>

//#define TIMING_FFTW
#ifdef TIMING_FFTW
	#define RCTIC(label) (timer_fftw.tic(label))
	#define RCTOC(label) (timer_fftw.toc(label))

	Timer timer_fftw;
	int TIMING_FFTW_PLAN = timer_fftw.setNew("fftw - plan");
	int TIMING_FFTW_EXECUTE = timer_fftw.setNew("fftw - exec");
	int TIMING_FFTW_NORMALISE = timer_fftw.setNew("fftw - normalise");
	int TIMING_FFTW_COPY = timer_fftw.setNew("fftw - copy");
#else
	#define RCTIC(label)
	#define RCTOC(label)
#endif

//#define DEBUG_PLANS

// FFTW plans ---------------------------------------------------------------
// RELION plans with FFTW_ESTIMATE, a guess that can be several times slower than a plan FFTW has
// measured (FFTW_MEASURE). Measuring takes from a fraction of a second (2D particle images) to tens
// of seconds (large 3D maps), and it overwrites the arrays it is given, so with
// RELION_FFTW_PLANS=measure a plan is measured once on scratch arrays with the same alignment, and
// kept in a wisdom file per CPU model that later runs read. If anything fails, the plan is the
// FFTW_ESTIMATE one, as before.
//   RELION_FFTW_PLANS=measure|estimate (default estimate, until validated)
//   RELION_FFTW_WISDOM=<file> (default $XDG_CACHE_HOME or ~/.cache, relion/fftw_wisdom_<precision>_<cpu>.txt)
//   RELION_FFTW_PLAN_TIMELIMIT=<seconds per measured plan> (default 30)
#if defined(RELION_SINGLE_PRECISION)
	#define RFFTW(f) fftwf_##f
	typedef fftwf_complex rfftw_complex;
	typedef fftwf_plan rfftw_plan;
#else
	#define RFFTW(f) fftw_##f
	typedef fftw_complex rfftw_complex;
	typedef fftw_plan rfftw_plan;
#endif

namespace
{

enum PlanKind { PLAN_R2C, PLAN_C2R, PLAN_C2C };

// Set once relionFftwInitThreads() has switched on FFTW's threads (see FourierTransformer::cleanup)
bool g_fftw_threads_inited = false;

struct PlanSettings
{
	bool measure = false;
	std::string wisdom_file;
	double timelimit = 30.;
	bool loaded = false;
};

std::string cpuModelName()
{
	std::ifstream in("/proc/cpuinfo");
	std::string line;
	while (std::getline(in, line))
		if (line.compare(0, 10, "model name") == 0)
		{
			std::string name = line.substr(line.find(':') + 1), out;
			for (char c : name)
				if (isalnum((unsigned char)c)) out += c;
				else if (!out.empty() && out.back() != '_') out += '_';
			while (!out.empty() && out.back() == '_') out.pop_back();
			return out.empty() ? "unknown_cpu" : out;
		}
	return "unknown_cpu";
}

PlanSettings &planSettings()
{
	static PlanSettings ps = []()
	{
		PlanSettings p;
#ifndef MKLFFT
		const char *mode = getenv("RELION_FFTW_PLANS");
		p.measure = (mode != NULL && std::string(mode) == "measure");
		const char *tl = getenv("RELION_FFTW_PLAN_TIMELIMIT");
		if (tl != NULL && atof(tl) > 0.) p.timelimit = atof(tl);
		const char *wf = getenv("RELION_FFTW_WISDOM");
		if (wf != NULL && wf[0] != '\0')
			p.wisdom_file = wf;
		else
		{
			const char *xdg = getenv("XDG_CACHE_HOME"), *home = getenv("HOME");
			std::string dir = (xdg != NULL && xdg[0] != '\0') ? std::string(xdg) :
			                  (home != NULL ? std::string(home) + "/.cache" : std::string());
			if (!dir.empty())
				p.wisdom_file = dir + "/relion/fftw_wisdom_" +
#ifdef RELION_SINGLE_PRECISION
					"single_" +
#else
					"double_" +
#endif
					cpuModelName() + ".txt";
		}
#endif
		return p;
	}();
	return ps;
}

// Called inside the planner's critical section
void loadWisdomOnce(PlanSettings &ps)
{
	if (ps.loaded) return;
	ps.loaded = true;
	if (!ps.wisdom_file.empty())
		RFFTW(import_wisdom_from_filename)(ps.wisdom_file.c_str()); // a missing file is fine
	// Measured plans and the lookups of saved ones use the same limit (FFTW keeps it with the wisdom)
	RFFTW(set_timelimit)(ps.timelimit);
}

// Merge with what other processes saved meanwhile, then replace the file in one step
void saveWisdom(const PlanSettings &ps)
{
	if (ps.wisdom_file.empty()) return;
	const std::string dir = ps.wisdom_file.substr(0, ps.wisdom_file.rfind('/'));
	if (!dir.empty()) { mkdir(dir.substr(0, dir.rfind('/')).c_str(), 0755); mkdir(dir.c_str(), 0755); }
	RFFTW(import_wisdom_from_filename)(ps.wisdom_file.c_str());
	const std::string tmp = ps.wisdom_file + ".tmp" + std::to_string((long)getpid());
	if (RFFTW(export_wisdom_to_filename)(tmp.c_str()))
		rename(tmp.c_str(), ps.wisdom_file.c_str());
	else
		remove(tmp.c_str());
}

rfftw_plan planWith(PlanKind kind, int ndim, const int *N, void *in, void *out, int sign, unsigned flags)
{
	switch (kind)
	{
	case PLAN_R2C: return RFFTW(plan_dft_r2c)(ndim, N, (RFLOAT*)in, (rfftw_complex*)out, flags);
	case PLAN_C2R: return RFFTW(plan_dft_c2r)(ndim, N, (rfftw_complex*)in, (RFLOAT*)out, flags);
	default:       return RFFTW(plan_dft)(ndim, N, (rfftw_complex*)in, (rfftw_complex*)out, sign, flags);
	}
}

// Scratch memory whose address has the same remainder modulo 64 bytes as p: FFTW's plans depend on
// the alignment of the arrays, and RELION executes them on its own arrays
struct Scratch
{
	char *base = NULL;
	void *ptr = NULL;
	Scratch(const void *p, size_t bytes)
	{
		base = (char*)malloc(bytes + 128);
		if (base == NULL) return;
		char *aligned = base + (64 - ((uintptr_t)base % 64)) % 64;
		ptr = aligned + ((uintptr_t)p % 64);
	}
	~Scratch() { free(base); }
};

// Plan thread count requested by the calling thread (0: use whatever the planner has)
static thread_local int g_plan_threads_override = 0;

// Within the planner's critical section: plan with the calling thread's own thread count, then give the planner its old one back
struct PlanThreadsScope
{
	int old = 0;
	PlanThreadsScope()
	{
#if defined(RELION_HAVE_FFTW_THREADS) && !defined(MKLFFT)
		if (g_plan_threads_override > 0 && g_fftw_threads_inited)
		{
			old = RFFTW(planner_nthreads)();
			RFFTW(plan_with_nthreads)(g_plan_threads_override);
		}
#endif
	}
	~PlanThreadsScope()
	{
#if defined(RELION_HAVE_FFTW_THREADS) && !defined(MKLFFT)
		if (old > 0)
			RFFTW(plan_with_nthreads)(old);
#endif
	}
};

// Make a plan (called inside the planner's critical section)
rfftw_plan makePlan(PlanKind kind, int ndim, const int *N, void *in, void *out, int sign = FFTW_FORWARD)
{
	PlanThreadsScope plan_threads_scope;
	PlanSettings &ps = planSettings();
	if (getenv("RELION_FFTW_DEBUG") != NULL)
	{
		size_t n = 1;
		for (int d = 0; d < ndim; d++) n *= N[d];
		if (n >= (1 << 21))
		{
			std::string dims;
			for (int d = 0; d < ndim; d++) dims += (d ? "x" : "") + std::to_string(N[d]);
#if defined(RELION_HAVE_FFTW_THREADS) && !defined(MKLFFT)
			const int nt = RFFTW(planner_nthreads)();
#else
			const int nt = 1;
#endif
			std::cerr << " FFTW debug: plan " << dims << " kind " << kind << " on " << nt << " threads" << std::endl;
		}
	}
	if (!ps.measure)
		return planWith(kind, ndim, N, in, out, sign, FFTW_ESTIMATE);

	loadWisdomOnce(ps);
	rfftw_plan p = planWith(kind, ndim, N, in, out, sign, FFTW_MEASURE | FFTW_WISDOM_ONLY);
	if (p != NULL) return p;

	// Measure on scratch arrays of the same sizes and alignment
	size_t n = 1;
	for (int d = 0; d < ndim; d++) n *= N[d];
	const size_t nhalf = n / N[ndim - 1] * (N[ndim - 1] / 2 + 1);
	size_t in_bytes, out_bytes;
	switch (kind)
	{
	case PLAN_R2C: in_bytes = n * sizeof(RFLOAT); out_bytes = nhalf * sizeof(rfftw_complex); break;
	case PLAN_C2R: in_bytes = nhalf * sizeof(rfftw_complex); out_bytes = n * sizeof(RFLOAT); break;
	default:       in_bytes = out_bytes = n * sizeof(rfftw_complex); break;
	}
	{
		Scratch sin(in, in_bytes), sout(out, out_bytes);
		if (sin.ptr != NULL && sout.ptr != NULL)
		{
			const auto t0 = std::chrono::steady_clock::now();
			rfftw_plan m = planWith(kind, ndim, N, sin.ptr, (in == out) ? sin.ptr : sout.ptr, sign, FFTW_MEASURE);
			const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
			if (m != NULL)
			{
				RFFTW(destroy_plan)(m);
				saveWisdom(ps);
				if (secs > 1.)
				{
					std::string dims;
					for (int d = 0; d < ndim; d++) dims += (d ? "x" : "") + std::to_string(N[d]);
					std::cout << " FFTW: measured the best plan for a " << dims << " transform in "
					          << std::fixed << std::setprecision(1) << secs << " s (once per CPU model; kept in "
					          << ps.wisdom_file << ")" << std::endl;
				}
			}
		}
	}
	p = planWith(kind, ndim, N, in, out, sign, FFTW_MEASURE | FFTW_WISDOM_ONLY);
	if (p != NULL) return p;
	return planWith(kind, ndim, N, in, out, sign, FFTW_ESTIMATE);
}

}

// Threads for RELION's own loops over a transform (normalisation): as many as FFTW plans with,
// so one inside the particle loop and all of them outside it
static int relionFftwLoopThreads()
{
#if defined(RELION_HAVE_FFTW_THREADS) && !defined(MKLFFT)
	return std::max(1, RFFTW(planner_nthreads)());
#else
	return 1;
#endif
}

// RELION_FFTW_DEBUG=1: report plans and execution times of large transforms
static bool relionFftwDebug()
{
	static const bool d = getenv("RELION_FFTW_DEBUG") != NULL;
	return d;
}

void relionFftwSetPlanMode(bool measure, const std::string &wisdom_file, double timelimit_s)
{
	#pragma omp critical(FourierTransformer_fftw_plan)
	{
		PlanSettings &ps = planSettings();
		ps.measure = measure;
		ps.wisdom_file = wisdom_file;
		ps.timelimit = timelimit_s;
		ps.loaded = false;
	}
}


// Constructors and destructors --------------------------------------------
FourierTransformer::FourierTransformer():
		plans_are_set(false)
{
	init();

#ifdef DEBUG_PLANS
	std::cerr << "INIT this= "<<this<< std::endl;
#endif
}

FourierTransformer::~FourierTransformer()
{
	clear();
#ifdef DEBUG_PLANS
	std::cerr << "CLEARED this= "<<this<< std::endl;
#endif
}

FourierTransformer::FourierTransformer(const FourierTransformer& op) :
		plans_are_set(false)
{
	// Clear current object
	clear();
	// New object is an extact copy of op
	*this = op;
}

void FourierTransformer::init()
{
	fReal = NULL;
	fComplex = NULL;
	fPlanForward = NULL;
	fPlanBackward = NULL;
	dataPtr = NULL;
	complexDataPtr = NULL;
}

void FourierTransformer::clear()
{
	fFourier.clear();
	// Clean-up all other FFTW-allocated memory
	destroyPlans();
	// Initialise all pointers to NULL
	init();

}

void FourierTransformer::cleanup()
{
	// First clear object and destroy plans
	clear();
	// Then clean up all the junk fftw keeps lying around
	// SOMEHOW THE FOLLOWING IS NOT ALLOWED WHEN USING MULTPLE TRANSFORMER OBJECTS....
	// fftw_cleanup() also replaces FFTW's planner. The new planner plans on one thread, has forgotten
	// the wisdom and, worse, lacks FFTW's threaded algorithms, which fftw_init_threads() registers on
	// the planner of the moment: every later transform then ran mostly on one core, whatever the number
	// of threads. With threads on, clean up through FFTW's thread layer and switch it on again.
	#pragma omp critical(FourierTransformer_fftw_plan)
	{
#if defined(MKLFFT) || defined(RELION_HAVE_FFTW_THREADS)
		if (g_fftw_threads_inited)
		{
			const int nthreads = relionFftwLoopThreads();
			RFFTW(cleanup_threads)();
			if (0 == RFFTW(init_threads)())
				REPORT_ERROR("Multithreaded FFTW failed to initialize");
			RFFTW(plan_with_nthreads)(nthreads);
		}
		else
#endif
			RFFTW(cleanup)();
		planSettings().loaded = false;
	}

#ifdef DEBUG_PLANS
	std::cerr << "CLEANED-UP this= "<<this<< std::endl;
#endif

}

void FourierTransformer::destroyPlans()
{
	// Anything to do with plans has to be protected for threads!
	#pragma omp critical(FourierTransformer_fftw_plan)
	{
		if (plans_are_set)
		{
#ifdef RELION_SINGLE_PRECISION
			fftwf_destroy_plan(fPlanForward);
			fftwf_destroy_plan(fPlanBackward);
#else
			fftw_destroy_plan(fPlanForward);
			fftw_destroy_plan(fPlanBackward);
#endif
			plans_are_set = false;
		}
	}
}

// Initialization ----------------------------------------------------------
const MultidimArray<RFLOAT> &FourierTransformer::getReal() const
{
	return (*fReal);
}

const MultidimArray<Complex > &FourierTransformer::getComplex() const
{
	return (*fComplex);
}


void FourierTransformer::setReal(MultidimArray<RFLOAT> &input, bool force_new_plans)
{
	bool recomputePlan = false;

	if (   (fReal == NULL)
	    || (dataPtr != MULTIDIM_ARRAY(input))
	    || (!fReal->sameShape(input))
		|| (XSIZE(fFourier) != XSIZE(input)/2+1)
	    || (complexDataPtr != MULTIDIM_ARRAY(fFourier)) )
	{
		recomputePlan = true;
	}

	if (recomputePlan || force_new_plans)
	{
		fFourier.reshape(ZSIZE(input),YSIZE(input),XSIZE(input)/2+1);
		fReal=&input;

		int ndim=3;
		if (ZSIZE(input)==1)
		{
			ndim=2;
			if (YSIZE(input)==1)
				ndim=1;
		}
		int *N = new int[ndim];
		switch (ndim)
		{
		case 1:
			N[0]=XSIZE(input);
			break;
		case 2:
			N[0]=YSIZE(input);
			N[1]=XSIZE(input);
			break;
		case 3:
			N[0]=ZSIZE(input);
			N[1]=YSIZE(input);
			N[2]=XSIZE(input);
			break;
		}

		// Destroy both forward and backward plans if they already exist
		destroyPlans();

		// Make new plans
		plans_are_set = true;

		RCTIC(TIMING_FFTW_PLAN);

		#pragma omp critical(FourierTransformer_fftw_plan)
		{
			fPlanForward = makePlan(PLAN_R2C, ndim, N, MULTIDIM_ARRAY(*fReal), MULTIDIM_ARRAY(fFourier));
			fPlanBackward = makePlan(PLAN_C2R, ndim, N, MULTIDIM_ARRAY(fFourier), MULTIDIM_ARRAY(*fReal));
		}	
		RCTOC(TIMING_FFTW_PLAN);

		if (fPlanForward == NULL || fPlanBackward == NULL)
			REPORT_ERROR("FFTW plans cannot be created");

#ifdef DEBUG_PLANS
		std::cerr << " SETREAL fPlanForward= " << fPlanForward << " fPlanBackward= " << fPlanBackward  <<" this= "<<this<< std::endl;
#endif

		delete [] N;
		dataPtr=MULTIDIM_ARRAY(*fReal);
		complexDataPtr = MULTIDIM_ARRAY(fFourier);

	}
}

void FourierTransformer::setReal(MultidimArray<Complex > &input, bool force_new_plans)
{
	bool recomputePlan=false;
	if (fComplex==NULL)
		recomputePlan=true;
	else if (complexDataPtr!=MULTIDIM_ARRAY(input))
		recomputePlan=true;
	else
		recomputePlan=!(fComplex->sameShape(input));

	fFourier.resize(input);
	fComplex=&input;

	if (recomputePlan || force_new_plans)
	{
		int ndim=3;
		if (ZSIZE(input)==1)
		{
			ndim=2;
			if (YSIZE(input)==1)
				ndim=1;
		}
		int *N = new int[ndim];
		switch (ndim)
		{
		case 1:
			N[0]=XSIZE(input);
			break;
		case 2:
			N[0]=YSIZE(input);
			N[1]=XSIZE(input);
			break;
		case 3:
			N[0]=ZSIZE(input);
			N[1]=YSIZE(input);
			N[2]=XSIZE(input);
			break;
		}

		// Destroy both forward and backward plans if they already exist
		destroyPlans();

		plans_are_set = true;

		RCTIC(TIMING_FFTW_PLAN);
		#pragma omp critical(FourierTransformer_fftw_plan)
		{
			fPlanForward = makePlan(PLAN_C2C, ndim, N, MULTIDIM_ARRAY(*fComplex), MULTIDIM_ARRAY(fFourier), FFTW_FORWARD);
			fPlanBackward = makePlan(PLAN_C2C, ndim, N, MULTIDIM_ARRAY(fFourier), MULTIDIM_ARRAY(*fComplex), FFTW_BACKWARD);
		}		
		RCTOC(TIMING_FFTW_PLAN);

		if (fPlanForward == NULL || fPlanBackward == NULL)
			REPORT_ERROR("FFTW plans cannot be created");

		delete [] N;
		complexDataPtr=MULTIDIM_ARRAY(*fComplex);
	}
}

void FourierTransformer::setFourier(const MultidimArray<Complex> &inputFourier)
{
	RCTIC(TIMING_FFTW_COPY);

	if (!fFourier.sameShape(inputFourier))
	{
		std::cerr << " fFourier= "; fFourier.printShape(std::cerr);
		std::cerr << " inputFourier= "; inputFourier.printShape(std::cerr);
		REPORT_ERROR("BUG: incompatible shaped in setFourier part of FFTW transformer");
	}
	memcpy(MULTIDIM_ARRAY(fFourier),MULTIDIM_ARRAY(inputFourier),
		   MULTIDIM_SIZE(inputFourier)*2*sizeof(RFLOAT));
	RCTOC(TIMING_FFTW_COPY);
}

// Transform ---------------------------------------------------------------
void FourierTransformer::Transform(int sign)
{
	const bool debug_time = relionFftwDebug() && fReal != NULL && MULTIDIM_SIZE(*fReal) >= (1 << 21);
	const auto debug_t0 = std::chrono::steady_clock::now();
	TransformTimed(sign);
	if (debug_time)
		std::cerr << " FFTW debug: " << (sign == FFTW_FORWARD ? "forward " : "inverse ")
		          << ZSIZE(*fReal) << "x" << YSIZE(*fReal) << "x" << XSIZE(*fReal) << " "
		          << std::chrono::duration<double>(std::chrono::steady_clock::now() - debug_t0).count() << " s" << std::endl;
}

void FourierTransformer::TransformTimed(int sign)
{
	if (sign == FFTW_FORWARD)
	{
		RCTIC(TIMING_FFTW_EXECUTE);
#ifdef RELION_SINGLE_PRECISION
		fftwf_execute_dft_r2c(fPlanForward,MULTIDIM_ARRAY(*fReal),
				(fftwf_complex*) MULTIDIM_ARRAY(fFourier));
#else
		fftw_execute_dft_r2c(fPlanForward,MULTIDIM_ARRAY(*fReal),
				(fftw_complex*) MULTIDIM_ARRAY(fFourier));
#endif
		RCTOC(TIMING_FFTW_EXECUTE);

		// Normalisation of the transform
		unsigned long int size=0;
		if(fReal!=NULL)
			size = MULTIDIM_SIZE(*fReal);
		else if (fComplex!= NULL)
			size = MULTIDIM_SIZE(*fComplex);
		else
			REPORT_ERROR("No complex nor real data defined");

		RCTIC(TIMING_FFTW_NORMALISE);
		// Every element is divided exactly as before, so the result does not depend on the threads
		const long int nF = MULTIDIM_SIZE(fFourier);
		Complex *pF = MULTIDIM_ARRAY(fFourier);
		const int nt = (nF > (1 << 20)) ? relionFftwLoopThreads() : 1;
		#pragma omp parallel for num_threads(nt) if(nt > 1)
		for (long int n = 0; n < nF; n++)
			pF[n] /= size;
		RCTOC(TIMING_FFTW_NORMALISE);
	}
	else if (sign == FFTW_BACKWARD)
	{
		RCTIC(TIMING_FFTW_EXECUTE);
#ifdef RELION_SINGLE_PRECISION
		fftwf_execute_dft_c2r(fPlanBackward,
				(fftwf_complex*) MULTIDIM_ARRAY(fFourier), MULTIDIM_ARRAY(*fReal));
#else
		fftw_execute_dft_c2r(fPlanBackward,
				(fftw_complex*) MULTIDIM_ARRAY(fFourier), MULTIDIM_ARRAY(*fReal));
#endif
		RCTOC(TIMING_FFTW_EXECUTE);
	}
}

void FourierTransformer::FourierTransform()
{
	Transform(FFTW_FORWARD);
}

void FourierTransformer::inverseFourierTransform()
{
	Transform(FFTW_BACKWARD);
}

// Inforce Hermitian symmetry ---------------------------------------------
void FourierTransformer::enforceHermitianSymmetry()
{
	int ndim = 3;
	if (ZSIZE(*fReal) == 1)
	{
		ndim = 2;
		if (YSIZE(*fReal) == 1)
			ndim = 1;
	}
	long int yHalf = YSIZE(*fReal) / 2;
	if (YSIZE(*fReal) % 2 == 0)
		yHalf--;
	long int zHalf = ZSIZE(*fReal) / 2;
	if (ZSIZE(*fReal) % 2 == 0)
		zHalf--;
	switch (ndim)
	{
	case 2:
		for (long int i = 1; i <= yHalf; i++)
		{
			long int isym = intWRAP(-i, 0, YSIZE(*fReal) - 1);
			Complex mean = 0.5 * (DIRECT_A2D_ELEM(fFourier, i, 0) + conj(DIRECT_A2D_ELEM(fFourier, isym, 0)));
			DIRECT_A2D_ELEM(fFourier, i, 0) = mean;
			DIRECT_A2D_ELEM(fFourier, isym, 0) = conj(mean);
		}
		break;
	case 3:
		for (long int k  =0; k < ZSIZE(*fReal); k++)
		{
			long int ksym = intWRAP(-k, 0, ZSIZE(*fReal) - 1);
			for (long int i = 1; i <= yHalf; i++)
			{
				long int isym = intWRAP(-i, 0, YSIZE(*fReal) - 1);
				Complex mean = 0.5 * (DIRECT_A3D_ELEM(fFourier, k, i, 0) + conj(DIRECT_A3D_ELEM(fFourier, ksym, isym, 0)));
				DIRECT_A3D_ELEM(fFourier, k, i, 0) = mean;
				DIRECT_A3D_ELEM(fFourier, ksym, isym, 0) = conj(mean);
			}
		}
		for (long int k = 1; k <= zHalf; k++)
		{
			long int ksym = intWRAP(-k, 0, ZSIZE(*fReal) - 1);
			Complex mean = 0.5*(DIRECT_A3D_ELEM(fFourier, k, 0, 0) + conj(DIRECT_A3D_ELEM(fFourier, ksym, 0, 0)));
			DIRECT_A3D_ELEM(fFourier, k, 0, 0) = mean;
			DIRECT_A3D_ELEM(fFourier, ksym, 0, 0) = conj(mean);
		}
		break;
	}
}


void randomizePhasesBeyond(MultidimArray<RFLOAT> &v, int index)
{
	MultidimArray< Complex > FT;
	FourierTransformer transformer;

	transformer.FourierTransform(v, FT, false);

	int index2 = index*index;
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(FT)
	{
		if (kp*kp + ip*ip + jp*jp >= index2)
		{
			RFLOAT mag = abs(DIRECT_A3D_ELEM(FT, k, i, j));
			RFLOAT phas = rnd_unif(0., 2.*PI);
			RFLOAT realval = mag * cos(phas);
			RFLOAT imagval = mag * sin(phas);
			DIRECT_A3D_ELEM(FT, k, i, j) = Complex(realval, imagval);
		}
	}

	// Inverse transform
	transformer.inverseFourierTransform();

}

/*
void randomizePhasesBeyond(MultidimArray<Complex> &v, int index)
{
	int index2 = index*index;
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(v)
	{
	   if (kp*kp + ip*ip + jp*jp >= index2)
	   {
			   RFLOAT mag = abs(DIRECT_A3D_ELEM(v, k, i, j));
			   RFLOAT phas = rnd_unif(0., 2.*PI);
			   RFLOAT realval = mag * cos(phas);
			   RFLOAT imagval = mag * sin(phas);
			   DIRECT_A3D_ELEM(v, k, i, j) = Complex(realval, imagval);
	   }
	}
}
*/

// Fourier ring correlation -----------------------------------------------
// from precalculated Fourier Transforms, and without sampling rate etc.
void getFSC(MultidimArray< Complex > &FT1,
			MultidimArray< Complex > &FT2,
			MultidimArray< RFLOAT > &fsc)
{
	if (!FT1.sameShape(FT2))
		REPORT_ERROR("fourierShellCorrelation ERROR: MultidimArrays have different shapes!");

	MultidimArray<RFLOAT> num(XSIZE(FT1)), den1(XSIZE(FT1)), den2(XSIZE(FT1));
	Matrix1D<RFLOAT> f(3);
	num.initZeros();
	den1.initZeros();
	den2.initZeros();
	fsc.initZeros(num);
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(FT1)
	{
		int idx = ROUND(sqrt(kp*kp + ip*ip + jp*jp));
		if (idx >= XSIZE(FT1))
			continue;
		Complex z1=DIRECT_A3D_ELEM(FT1, k, i, j);
		Complex z2=DIRECT_A3D_ELEM(FT2, k, i, j);
		RFLOAT absz1=abs(z1);
		RFLOAT absz2=abs(z2);
		num(idx)+= (conj(z1) * z2).real;
		den1(idx)+= absz1*absz1;
		den2(idx)+= absz2*absz2;
	}

	FOR_ALL_ELEMENTS_IN_ARRAY1D(fsc)
	{
		fsc(i) = num(i)/sqrt(den1(i)*den2(i));
	}

}


void getFSC(MultidimArray< RFLOAT > &m1,
			MultidimArray< RFLOAT > &m2,
			MultidimArray< RFLOAT > &fsc)
{
	MultidimArray< Complex > FT1, FT2;
	FourierTransformer transformer;
	transformer.FourierTransform(m1, FT1);
	transformer.FourierTransform(m2, FT2);
	getFSC(FT1, FT2, fsc);
}

void getAmplitudeCorrelationAndDifferentialPhaseResidual(MultidimArray< Complex > &FT1,
			MultidimArray< Complex > &FT2,
			MultidimArray< RFLOAT > &acorr,
			MultidimArray< RFLOAT > &dpr)
{

	MultidimArray< int > radial_count(XSIZE(FT1));
	MultidimArray<RFLOAT> num, mu1, mu2, sig1, sig2;
	Matrix1D<RFLOAT> f(3);
	mu1.initZeros(radial_count);
	mu2.initZeros(radial_count);
	sig1.initZeros(radial_count);
	sig2.initZeros(radial_count);
	acorr.initZeros(radial_count);
	dpr.initZeros(radial_count);
	num.initZeros(radial_count);
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(FT1)
	{
		// Amplitudes
		int idx = ROUND(sqrt(kp*kp + ip*ip + jp*jp));
		if (idx >= XSIZE(FT1))
			continue;
		RFLOAT abs1 = abs(DIRECT_A3D_ELEM(FT1, k, i, j));
		RFLOAT abs2 = abs(DIRECT_A3D_ELEM(FT2, k, i, j));
		mu1(idx)+= abs1;
		mu2(idx)+= abs2;
		radial_count(idx)++;

		//phases
		RFLOAT phas1 = RAD2DEG(atan2((DIRECT_A3D_ELEM(FT1, k, i, j)).imag, (DIRECT_A3D_ELEM(FT1, k, i, j)).real));
		RFLOAT phas2 = RAD2DEG(atan2((DIRECT_A3D_ELEM(FT2, k, i, j)).imag, (DIRECT_A3D_ELEM(FT2, k, i, j)).real));
		RFLOAT delta_phas = phas1 - phas2;
		if (delta_phas > 180.)
			delta_phas -= 360.;
		else if (delta_phas < -180.)
			delta_phas += 360.;
		dpr(idx) += delta_phas*delta_phas*(abs1+abs2);
		num(idx) += (abs1+abs2);
	}

	// Get average amplitudes in each shell for both maps
	FOR_ALL_ELEMENTS_IN_ARRAY1D(mu1)
	{
		if (radial_count(i) > 0)
		{
			mu1(i) /= radial_count(i);
			mu2(i) /= radial_count(i);
			dpr(i) = sqrt(dpr(i)/num(i));
		}
	}

	// Now calculate Pearson's correlation coefficient
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(FT1)
	{
		int idx = ROUND(sqrt(kp*kp + ip*ip + jp*jp));
		if (idx >= XSIZE(FT1))
			continue;
		RFLOAT z1=abs(DIRECT_A3D_ELEM(FT1, k, i, j)) - mu1(idx);
		RFLOAT z2=abs(DIRECT_A3D_ELEM(FT2, k, i, j)) - mu2(idx);
		acorr(idx)	+= z1*z2;
		sig1(idx) += z1*z1;
		sig2(idx) += z2*z2;
	}

	FOR_ALL_ELEMENTS_IN_ARRAY1D(acorr)
	{
		RFLOAT aux = sqrt(sig1(i))*sqrt(sig2(i));
		if (aux > 0.)
			acorr(i) /= sqrt(sig1(i))*sqrt(sig2(i));
		else
			acorr(i) = 1.;
	}

}

void getCosDeltaPhase(MultidimArray< Complex > &FT1,
					  MultidimArray< Complex > &FT2,
					  MultidimArray< RFLOAT > &cosPhi)
{
	MultidimArray< int > radial_count(XSIZE(FT1));
	cosPhi.initZeros(XSIZE(FT1));

	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(FT1)
	{
		int idx = ROUND(sqrt(kp*kp + ip*ip + jp*jp));
		if (idx >= XSIZE(FT1))
			continue;

		RFLOAT phas1 = RAD2DEG(atan2((DIRECT_A3D_ELEM(FT1, k, i, j)).imag, (DIRECT_A3D_ELEM(FT1, k, i, j)).real));
		RFLOAT phas2 = RAD2DEG(atan2((DIRECT_A3D_ELEM(FT2, k, i, j)).imag, (DIRECT_A3D_ELEM(FT2, k, i, j)).real));
		cosPhi(idx) += cos(phas1 - phas2);
		radial_count(idx) ++;
	}

	FOR_ALL_ELEMENTS_IN_ARRAY1D(cosPhi)
	{
		if (radial_count(i) > 0)
			cosPhi(i) /= radial_count(i);
	}
}

void getAmplitudeCorrelationAndDifferentialPhaseResidual(MultidimArray< RFLOAT > &m1,
			MultidimArray< RFLOAT > &m2,
			MultidimArray< RFLOAT > &acorr,
			MultidimArray< RFLOAT > &dpr)
{
	MultidimArray< Complex > FT1, FT2;
	FourierTransformer transformer;
	transformer.FourierTransform(m1, FT1);
	transformer.FourierTransform(m2, FT2);
	getAmplitudeCorrelationAndDifferentialPhaseResidual(FT1, FT2, acorr, dpr);
}


/*
void selfScaleToSizeFourier(long int Ydim, long int Xdim, MultidimArray<RFLOAT>& Mpmem, int nThreads)
{

	//Mmem = *this
	//memory for fourier transform output
	MultidimArray<Complex > MmemFourier;
	// Perform the Fourier transform
	FourierTransformer transformerM;
	transformerM.setThreadsNumber(nThreads);
	transformerM.FourierTransform(Mpmem, MmemFourier, true);

	// Create space for the downsampled image and its Fourier transform
	Mpmem.resize(Ydim, Xdim);
	MultidimArray<Complex > MpmemFourier;
	FourierTransformer transformerMp;
	transformerMp.setReal(Mpmem);
	transformerMp.getFourierAlias(MpmemFourier);
	long int ihalf = XMIPP_MIN((YSIZE(MpmemFourier)/2+1),(YSIZE(MmemFourier)/2+1));
	long int xsize = XMIPP_MIN((XSIZE(MmemFourier)),(XSIZE(MpmemFourier)));
	//Init with zero
	MpmemFourier.initZeros();
	for (long int i=0; i<ihalf; i++)
		for (long int j=0; j<xsize; j++)
			MpmemFourier(i,j)=MmemFourier(i,j);
	for (long int i=YSIZE(MpmemFourier)-1, n=1; n < ihalf-1; i--, n++)
	{
		long int ip = YSIZE(MmemFourier) - n;
		for (long int j=0; j<xsize; j++)
			MpmemFourier(i,j)=MmemFourier(ip,j);
	}

	// Transform data
	transformerMp.inverseFourierTransform();
}
*/


void getAbMatricesForShiftImageInFourierTransform(MultidimArray<Complex > &in,
		MultidimArray<Complex > &out,
		  RFLOAT oridim, RFLOAT xshift, RFLOAT yshift, RFLOAT zshift)
{
	out.resize(in);
	RFLOAT dotp, a, b, x, y, z;
	switch (in.getDim())
	{
	case 1:
		xshift /= -oridim;
		for (long int j = 0; j < XSIZE(in); j++)
		{
			x = j;
			dotp = 2 * PI * (x * xshift);
#ifdef RELION_SINGLE_PRECISION
						SINCOSF(dotp, &b, &a);
#else
						SINCOS(dotp, &b, &a);
#endif
			DIRECT_A1D_ELEM(out, j) = Complex(a, b);
		}
		break;
	case 2:
		xshift /= -oridim;
		yshift /= -oridim;
		for (long int i=0; i<XSIZE(in); i++)
			for (long int j=0; j<XSIZE(in); j++)
			{
				x = j;
				y = i;
				dotp = 2 * PI * (x * xshift + y * yshift);
#ifdef RELION_SINGLE_PRECISION
				SINCOSF(dotp, &b, &a);
#else
				SINCOS(dotp, &b, &a);
#endif
				DIRECT_A2D_ELEM(out, i, j) = Complex(a, b);
			}
		for (long int i=YSIZE(in)-1; i>=XSIZE(in); i--)
		{
			y = i - YSIZE(in);
			for (long int j=0; j<XSIZE(in); j++)
			{
				x = j;
				dotp = 2 * PI * (x * xshift + y * yshift);
#ifdef RELION_SINGLE_PRECISION
				SINCOSF(dotp, &b, &a);
#else
				SINCOS(dotp, &b, &a);
#endif
				DIRECT_A2D_ELEM(out, i, j) = Complex(a, b);
			}
		}
		break;
	case 3:
		xshift /= -oridim;
		yshift /= -oridim;
		zshift /= -oridim;
		for (long int k=0; k<ZSIZE(in); k++)
		{
			z = (k < XSIZE(in)) ? k : k - ZSIZE(in);
			for (long int i=0; i<YSIZE(in); i++)
			{
				y = (i < XSIZE(in)) ? i : i - YSIZE(in);
				for (long int j=0; j<XSIZE(in); j++)
				{
					x = j;
					dotp = 2 * PI * (x * xshift + y * yshift + z * zshift);
#ifdef RELION_SINGLE_PRECISION
					SINCOSF(dotp, &b, &a);
#else
					SINCOS(dotp, &b, &a);
#endif
					DIRECT_A3D_ELEM(out, k, i, j) = Complex(a, b);
				}
			}
		}
		break;
	default:
		REPORT_ERROR("getAbMatricesForShiftImageInFourierTransform ERROR: dimension should be 1, 2 or 3!");
	}
}

void shiftImageInFourierTransformWithTabSincos(MultidimArray<Complex > &in,
								  MultidimArray<Complex > &out,
								  RFLOAT oridim, long int newdim,
								  TabSine& tabsin, TabCosine& tabcos,
								  RFLOAT xshift, RFLOAT yshift, RFLOAT zshift)
{
	RFLOAT a = 0., b = 0., c = 0., d = 0., ac = 0., bd = 0., ab_cd = 0., dotp = 0., x = 0., y = 0., z = 0.;
	RFLOAT twopi = 2. * PI;

	if (&in == &out)
		REPORT_ERROR("shiftImageInFourierTransformWithTabSincos ERROR: Input and output images should be different!");
	// Check size of the input array
	if ( (YSIZE(in) > 1) && ( (YSIZE(in)/2 + 1) != XSIZE(in) ) )
		REPORT_ERROR("shiftImageInFourierTransformWithTabSincos ERROR: the Fourier transform should be of an image with equal sizes in all dimensions!");

	long int newhdim = newdim/2 + 1;
	if (newhdim > XSIZE(in))
		REPORT_ERROR("shiftImageInFourierTransformWithTabSincos ERROR: 'newdim' should be equal or smaller than the size of the original array!");

	// Initialise output array
	out.clear();
	switch (in.getDim())
	{
	case 2: out.initZeros(newdim, newhdim); break;
	case 3: out.initZeros(newdim, newdim, newhdim); break;
	default: REPORT_ERROR("shiftImageInFourierTransformWithTabSincos ERROR: dimension should be 2 or 3!");
	}

	if (in.getDim() == 2)
	{
		xshift /= -oridim;
		yshift /= -oridim;
		if ( (ABS(xshift) < XMIPP_EQUAL_ACCURACY) && (ABS(yshift) < XMIPP_EQUAL_ACCURACY) )
		{
			windowFourierTransform(in, out, newdim);
			return;
		}

		FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(out)
		{
			dotp = twopi * (jp * xshift + ip * yshift);

			a = tabcos(dotp);
			b = tabsin(dotp);
			c = DIRECT_A2D_ELEM(in, i, j).real;
			d = DIRECT_A2D_ELEM(in, i, j).imag;
			ac = a * c;
			bd = b * d;
			ab_cd = (a + b) * (c + d);
			DIRECT_A2D_ELEM(out, i, j) = Complex(ac - bd, ab_cd - ac - bd);
		}
	}
	else if (in.getDim() == 3)
	{
		xshift /= -oridim;
		yshift /= -oridim;
		zshift /= -oridim;
		if ( (ABS(xshift) < XMIPP_EQUAL_ACCURACY) && (ABS(yshift) < XMIPP_EQUAL_ACCURACY) && (ABS(zshift) < XMIPP_EQUAL_ACCURACY) )
		{
			windowFourierTransform(in, out, newdim);
			return;
		}

		FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(out)
		{
			dotp = twopi * (jp * xshift + ip * yshift + kp * zshift);

			a = tabcos(dotp);
			b = tabsin(dotp);
			c = DIRECT_A3D_ELEM(in, k, i, j).real;
			d = DIRECT_A3D_ELEM(in, k, i, j).imag;
			ac = a * c;
			bd = b * d;
			ab_cd = (a + b) * (c + d);
			DIRECT_A3D_ELEM(out, k, i, j) = Complex(ac - bd, ab_cd - ac - bd);
		}
	}
}

// Shift an image through phase-shifts in its Fourier Transform (without pretabulated sine and cosine)
void shiftImageInFourierTransform(MultidimArray<Complex > &in,
								  MultidimArray<Complex > &out,
								  RFLOAT oridim, RFLOAT xshift, RFLOAT yshift, RFLOAT zshift)
{
	out.resize(in);
	RFLOAT dotp, a, b, c, d, ac, bd, ab_cd, x, y, z;
	switch (in.getDim())
	{
	case 1:
		xshift /= -oridim;
		if (ABS(xshift) < XMIPP_EQUAL_ACCURACY)
		{
			out = in;
			return;
		}
		for (long int j = 0; j < XSIZE(in); j++)
		{
			x = j;
			dotp = 2 * PI * (x * xshift);
#ifdef RELION_SINGLE_PRECISION
			SINCOSF(dotp, &b, &a);
#else
			SINCOS(dotp, &b, &a);
#endif
			c = DIRECT_A1D_ELEM(in, j).real;
			d = DIRECT_A1D_ELEM(in, j).imag;
			ac = a * c;
			bd = b * d;
			ab_cd = (a + b) * (c + d); // (ab_cd-ac-bd = ad+bc : but needs 4 multiplications)
			DIRECT_A1D_ELEM(out, j) = Complex(ac - bd, ab_cd - ac - bd);
		}
		break;
	case 2:
		xshift /= -oridim;
		yshift /= -oridim;
		if (ABS(xshift) < XMIPP_EQUAL_ACCURACY && ABS(yshift) < XMIPP_EQUAL_ACCURACY)
		{
			out = in;
			return;
		}
		for (long int i=0; i<XSIZE(in); i++)
			for (long int j=0; j<XSIZE(in); j++)
			{
				x = j;
				y = i;
				dotp = 2 * PI * (x * xshift + y * yshift);
#ifdef RELION_SINGLE_PRECISION
				SINCOSF(dotp, &b, &a);
#else
				SINCOS(dotp, &b, &a);
#endif
				c = DIRECT_A2D_ELEM(in, i, j).real;
				d = DIRECT_A2D_ELEM(in, i, j).imag;
				ac = a * c;
				bd = b * d;
				ab_cd = (a + b) * (c + d);
				DIRECT_A2D_ELEM(out, i, j) = Complex(ac - bd, ab_cd - ac - bd);
			}
		for (long int i=YSIZE(in)-1; i>=XSIZE(in); i--)
		{
			y = i - YSIZE(in);
			for (long int j=0; j<XSIZE(in); j++)
			{
				x = j;
				dotp = 2 * PI * (x * xshift + y * yshift);
#ifdef RELION_SINGLE_PRECISION
				SINCOSF(dotp, &b, &a);
#else
				SINCOS(dotp, &b, &a);
#endif
				c = DIRECT_A2D_ELEM(in, i, j).real;
				d = DIRECT_A2D_ELEM(in, i, j).imag;
				ac = a * c;
				bd = b * d;
				ab_cd = (a + b) * (c + d);
				DIRECT_A2D_ELEM(out, i, j) = Complex(ac - bd, ab_cd - ac - bd);
			}
		}
		break;
	case 3:
		xshift /= -oridim;
		yshift /= -oridim;
		zshift /= -oridim;
		if (ABS(xshift) < XMIPP_EQUAL_ACCURACY && ABS(yshift) < XMIPP_EQUAL_ACCURACY && ABS(zshift) < XMIPP_EQUAL_ACCURACY)
		{
			out = in;
			return;
		}
		for (long int k=0; k<ZSIZE(in); k++)
		{
			z = (k < XSIZE(in)) ? k : k - ZSIZE(in);
			for (long int i=0; i<YSIZE(in); i++)
			{
				y = (i < XSIZE(in)) ? i : i - YSIZE(in);
				for (long int j=0; j<XSIZE(in); j++)
				{
					x = j;
					dotp = 2 * PI * (x * xshift + y * yshift + z * zshift);
#ifdef RELION_SINGLE_PRECISION
					SINCOSF(dotp, &b, &a);
#else
					SINCOS(dotp, &b, &a);
#endif
					c = DIRECT_A3D_ELEM(in, k, i, j).real;
					d = DIRECT_A3D_ELEM(in, k, i, j).imag;
					ac = a * c;
					bd = b * d;
					ab_cd = (a + b) * (c + d);
					DIRECT_A3D_ELEM(out, k, i, j) = Complex(ac - bd, ab_cd - ac - bd);
				}
			}
		}
		break;
	default:
		REPORT_ERROR("shiftImageInFourierTransform ERROR: dimension should be 1, 2 or 3!");
	}
}

// Shift an image through phase-shifts in its Fourier Transform (without pretabulated sine and cosine)
void shiftImageInContinuousFourierTransform(MultidimArray<Complex > &in, MultidimArray<Complex > &out,
                                            RFLOAT oridim, RFLOAT xshift, RFLOAT yshift, RFLOAT zshift)
{
	out.resize(in);
	RFLOAT dotp, a, b, c, d, ac, bd, ab_cd, x, y, z;

	xshift /= -oridim;
	yshift /= -oridim;
	zshift /= -oridim;
	if (ABS(xshift) < XMIPP_EQUAL_ACCURACY && ABS(yshift) < XMIPP_EQUAL_ACCURACY && ABS(zshift) < XMIPP_EQUAL_ACCURACY)
	{
		out = in;
		return;
	}

	long iniY(0), iniZ(0);

	if (YSIZE(in) > 1)
		iniY = YSIZE(in) / 2;

	if (ZSIZE(in) > 1)
		iniZ = ZSIZE(in) / 2;

	for (long int k=0; k<ZSIZE(in); k++)
	{
		z = k - iniZ;
		for (long int i=0; i<YSIZE(in); i++)
		{
			y = i - iniY;
			for (long int j=0; j<XSIZE(in); j++)
			{
				x = j;
				dotp = 2 * PI * (x * xshift + y * yshift + z * zshift);
#ifdef RELION_SINGLE_PRECISION
				SINCOSF(dotp, &b, &a);
#else
				SINCOS(dotp, &b, &a);
#endif
				c = DIRECT_A3D_ELEM(in, k, i, j).real;
				d = DIRECT_A3D_ELEM(in, k, i, j).imag;
				ac = a * c;
				bd = b * d;
				ab_cd = (a + b) * (c + d);
				DIRECT_A3D_ELEM(out, k, i, j) = Complex(ac - bd, ab_cd - ac - bd);
			}
		}
	}
}

void getSpectrum(MultidimArray<RFLOAT> &Min,
				 MultidimArray<RFLOAT> &spectrum,
				 int spectrum_type)
{

	MultidimArray<Complex > Faux;
	int xsize = XSIZE(Min);
	// Takanori: The above line should be XSIZE(Min) / 2 + 1 but for compatibility reasons, I keep this as it is.
	Matrix1D<RFLOAT> f(3);
	MultidimArray<RFLOAT> count(xsize);
	FourierTransformer transformer;

	spectrum.initZeros(xsize);
	count.initZeros();
	transformer.FourierTransform(Min, Faux, false);
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Faux)
	{
		long int idx = ROUND(sqrt(kp*kp + ip*ip + jp*jp));
		if (spectrum_type == AMPLITUDE_SPECTRUM)
			spectrum(idx) += abs(dAkij(Faux, k, i, j));
		else
			spectrum(idx) += norm(dAkij(Faux, k, i, j));
		count(idx) += 1.;
	}

	for (long int i = 0; i < xsize; i++)
		if (count(i) > 0.)
			spectrum(i) /= count(i);

}

void divideBySpectrum(MultidimArray<RFLOAT> &Min,
					  MultidimArray<RFLOAT> &spectrum,
					  bool leave_origin_intact)
{

	MultidimArray<RFLOAT> div_spec(spectrum);
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(spectrum)
	{
		if (ABS(dAi(spectrum,i)) > 0.)
			dAi(div_spec,i) = 1./dAi(spectrum,i);
		else
			dAi(div_spec,i) = 1.;
	}
	multiplyBySpectrum(Min,div_spec,leave_origin_intact);
}

void multiplyBySpectrum(MultidimArray<RFLOAT> &Min,
						MultidimArray<RFLOAT> &spectrum,
						bool leave_origin_intact)
{

	MultidimArray<Complex > Faux;
	Matrix1D<RFLOAT> f(3);
	MultidimArray<RFLOAT> lspectrum;
	FourierTransformer transformer;
	//RFLOAT dim3 = XSIZE(Min)*YSIZE(Min)*ZSIZE(Min);
	transformer.FourierTransform(Min, Faux, false);
	lspectrum=spectrum;
	if (leave_origin_intact)
		lspectrum(0)=1.;
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Faux)
	{
		long int idx = ROUND(sqrt(kp*kp + ip*ip + jp*jp));
		dAkij(Faux, k, i, j) *=  lspectrum(idx);// * dim3;
	}
	transformer.inverseFourierTransform();

}


void whitenSpectrum(MultidimArray<RFLOAT> &Min,
					MultidimArray<RFLOAT> &Mout,
					int spectrum_type,
					bool leave_origin_intact)
{

	MultidimArray<RFLOAT> spectrum;
	getSpectrum(Min,spectrum,spectrum_type);
	Mout=Min;
	divideBySpectrum(Mout,spectrum,leave_origin_intact);

}

void adaptSpectrum(MultidimArray<RFLOAT> &Min,
				   MultidimArray<RFLOAT> &Mout,
				   const MultidimArray<RFLOAT> &spectrum_ref,
				   int spectrum_type,
				   bool leave_origin_intact)
{

	MultidimArray<RFLOAT> spectrum;
	getSpectrum(Min,spectrum,spectrum_type);
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(spectrum)
	{
		dAi(spectrum, i) = (dAi(spectrum, i) > 0.) ? dAi(spectrum_ref,i)/ dAi(spectrum, i) : 1.;
	}
	Mout=Min;
	multiplyBySpectrum(Mout,spectrum,leave_origin_intact);
}

/** Kullback-Leibler divergence */
RFLOAT getKullbackLeiblerDivergence(MultidimArray<Complex > &Fimg,
		MultidimArray<Complex > &Fref, MultidimArray<RFLOAT> &sigma2,
		MultidimArray<RFLOAT> &p_i, MultidimArray<RFLOAT> &q_i, int highshell, int lowshell )
{
	// First check dimensions are OK
	if (!Fimg.sameShape(Fref))
		REPORT_ERROR("getKullbackLeiblerDivergence ERROR: Fimg and Fref are not of the same shape.");

	if (highshell < 0)
		highshell = XSIZE(Fimg) - 1;
	if (lowshell < 0)
		lowshell = 0;

	if (highshell > XSIZE(sigma2))
		REPORT_ERROR("getKullbackLeiblerDivergence ERROR: highshell is larger than size of sigma2 array.");

	if (highshell < lowshell)
		REPORT_ERROR("getKullbackLeiblerDivergence ERROR: highshell is smaller than lowshell.");

	// Initialize the histogram
	MultidimArray<int> histogram;
	int histogram_size = 101;
	int histogram_origin = histogram_size / 2;
	RFLOAT sigma_max = 10.;
	RFLOAT histogram_factor = histogram_origin / sigma_max;
	histogram.initZeros(histogram_size);

	// This way this will work in both 2D and 3D
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fimg)
	{
		int ires = ROUND(sqrt(kp*kp + ip*ip + jp*jp));
		if (ires >= lowshell && ires <= highshell)
		{
			// Use FT of masked image for noise estimation!
			RFLOAT diff_real = (DIRECT_A3D_ELEM(Fref, k, i, j)).real - (DIRECT_A3D_ELEM(Fimg, k, i, j)).real;
			RFLOAT diff_imag = (DIRECT_A3D_ELEM(Fref, k, i, j)).imag - (DIRECT_A3D_ELEM(Fimg, k, i, j)).imag;
			RFLOAT sigma = sqrt(DIRECT_A1D_ELEM(sigma2, ires));

			// Divide by standard deviation to normalise all the difference
			diff_real /= sigma;
			diff_imag /= sigma;

			// Histogram runs from -10 sigma to +10 sigma
			diff_real += sigma_max;
			diff_imag += sigma_max;

			// Make histogram on-the-fly;
			// Real part
			int ihis = ROUND(diff_real * histogram_factor);
			if (ihis < 0)
				ihis = 0;
			else if (ihis >= histogram_size)
				ihis = histogram_size - 1;
			histogram(ihis)++;
			// Imaginary part
			ihis = ROUND(diff_imag * histogram_factor);
			if (ihis < 0)
				ihis = 0;
			else if (ihis > histogram_size)
				ihis = histogram_size;
			histogram(ihis)++;

		}
	}

	// Normalise the histogram and the discretised analytical Gaussian
	RFLOAT norm = (RFLOAT)histogram.sum();
	RFLOAT gaussnorm = 0.;
	for (int i = 0; i < histogram_size; i++)
	{
		RFLOAT x = (RFLOAT)i / histogram_factor;
		gaussnorm += gaussian1D(x - sigma_max, 1. , 0.);
	}

	// Now calculate the actual Kullback-Leibler divergence
	RFLOAT kl_divergence = 0.;
	p_i.resize(histogram_size);
	q_i.resize(histogram_size);
	for (int i = 0; i < histogram_size; i++)
	{
		// Data distribution
		p_i(i) = (RFLOAT)histogram(i) / norm;
		// Theoretical distribution
		RFLOAT x = (RFLOAT)i / histogram_factor;
		q_i(i) = gaussian1D(x - sigma_max, 1. , 0.) / gaussnorm;

		if (p_i(i) > 0.)
			kl_divergence += p_i(i) * log (p_i(i) / q_i(i));
	}
	kl_divergence /= (RFLOAT)histogram_size;

	return kl_divergence;

}
void resizeMap(MultidimArray<RFLOAT > &img, int newsize)
{

	FourierTransformer transformer;
	MultidimArray<Complex > FT, FT2;
	transformer.FourierTransform(img, FT, false);
	windowFourierTransform(FT, FT2, newsize);
	if (img.getDim() == 2)
		img.resize(newsize, newsize);
	else if (img.getDim() == 3)
		img.resize(newsize, newsize, newsize);
	transformer.inverseFourierTransform(FT2, img);

}

void applyBFactorToMap(MultidimArray<Complex > &FT, int ori_size, RFLOAT bfactor, RFLOAT angpix)
{
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(FT)
	{
		int r2 = kp * kp + ip * ip + jp * jp;
		RFLOAT res = sqrt((RFLOAT)r2)/(ori_size * angpix); // get resolution in 1/Angstrom
		if (res <= 1. / (angpix * 2.) ) // Apply B-factor sharpening until Nyquist, then low-pass filter later on (with a soft edge)
		{
			DIRECT_A3D_ELEM(FT, k, i, j) *= exp( -(bfactor / 4.)  * res * res);
		}
		else
		{
			DIRECT_A3D_ELEM(FT, k, i, j) = 0.;
		}
	}
}

void applyBFactorToMap(MultidimArray<RFLOAT > &img, RFLOAT bfactor, RFLOAT angpix)
{
	refuseNonCubicMap(img, "B-factor filtering (applyBFactorToMap)");

	FourierTransformer transformer;
	MultidimArray<Complex > FT;
	transformer.FourierTransform(img, FT, false);
	applyBFactorToMap(FT, XSIZE(img), bfactor, angpix);
	transformer.inverseFourierTransform();
}



void LoGFilterMap(MultidimArray<Complex > &FT, int ori_size, RFLOAT sigma, RFLOAT angpix)
{

	// Calculate sigma in reciprocal pixels (input is in Angstroms) and pre-calculate its square
	// Factor of 1/2 because input is diameter, and filter uses radius
	RFLOAT isigma2 = (0.5 * ori_size * angpix) / sigma;
	isigma2 *= isigma2;

	// Gunn Pattern Recognition 32 (1999) 1463-1472
	// The Laplacian filter is: 1/(PI*sigma2)*(r^2/2*sigma2 - 1) * exp(-r^2/(2*sigma2))
	// and its Fourier Transform is: r^2 * exp(-0.5*r2/isigma2);
	// Then to normalise for different scales: divide by isigma2;
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(FT)
	{
		RFLOAT r2 = (RFLOAT)kp * (RFLOAT)kp + (RFLOAT)ip * (RFLOAT)ip + (RFLOAT)jp * (RFLOAT)jp;
		DIRECT_A3D_ELEM(FT, k, i, j) *= r2 * exp(-0.5*r2/isigma2) / isigma2;
	}

}

void LoGFilterMap(MultidimArray<RFLOAT > &img, RFLOAT sigma, RFLOAT angpix)
{
	refuseNonCubicMap(img, "LoG filtering (LoGFilterMap)");
	FourierTransformer transformer;
	MultidimArray<Complex > FT;

	// Make this work for maps (or more likely 2D images) that have unequal X and Y dimensions
	img.setXmippOrigin();
	int my_xsize = XSIZE(img);
	int my_ysize = YSIZE(img);
	int my_size = (my_xsize != my_ysize) ? XMIPP_MAX(my_xsize, my_ysize) : my_xsize;
	if (my_xsize != my_ysize)
	{
		if (img.getDim() == 2)
		{
			int my_small_size = XMIPP_MIN(my_xsize, my_ysize);
			RFLOAT avg,stddev,minn,maxx;
			img.computeStats(avg,stddev,minn,maxx);
			img.window(FIRST_XMIPP_INDEX(my_size), FIRST_XMIPP_INDEX(my_size),
					   LAST_XMIPP_INDEX(my_size),  LAST_XMIPP_INDEX(my_size));
			if (my_small_size == my_xsize)
			{
				FOR_ALL_ELEMENTS_IN_ARRAY2D(img)
				{
					if (j <  FIRST_XMIPP_INDEX(my_small_size) || j >  LAST_XMIPP_INDEX(my_small_size))
						A2D_ELEM(img, i, j) = rnd_gaus(avg, stddev);
				}
			}
			else
			{
				FOR_ALL_ELEMENTS_IN_ARRAY2D(img)
				{
					if (i <  FIRST_XMIPP_INDEX(my_small_size) || i >  LAST_XMIPP_INDEX(my_small_size))
						A2D_ELEM(img, i, j) = rnd_gaus(avg, stddev);
				}

			}
		}
		else
		{
			REPORT_ERROR("lowPassFilterMap: filtering of non-cube maps is not implemented...");
		}
	}
	transformer.FourierTransform(img, FT, false);
	LoGFilterMap(FT, XSIZE(img), sigma, angpix);
	transformer.inverseFourierTransform();
	img.setXmippOrigin();
	if (my_xsize != my_ysize)
	{
		if (img.getDim() == 2)
		{
			img.window(FIRST_XMIPP_INDEX(my_ysize), FIRST_XMIPP_INDEX(my_xsize),
					   LAST_XMIPP_INDEX(my_ysize),	LAST_XMIPP_INDEX(my_xsize));
		}
		else
		{
			REPORT_ERROR("lowPassFilterMap: filtering of non-cube maps is not implemented...");
		}
	}


}

void lowPassFilterMap(MultidimArray<Complex > &FT, int ori_size,
		RFLOAT low_pass, RFLOAT angpix, int filter_edge_width, bool do_highpass_instead)
{

	// Which resolution shell is the filter?
	int ires_filter = ROUND((ori_size * angpix)/low_pass);
	int filter_edge_halfwidth = filter_edge_width / 2;

	// Soft-edge: from 1 shell less to one shell more:
	RFLOAT edge_low = XMIPP_MAX(0., (ires_filter - filter_edge_halfwidth) / (RFLOAT)ori_size); // in 1/pix
	RFLOAT edge_high = XMIPP_MIN(XSIZE(FT), (ires_filter + filter_edge_halfwidth) / (RFLOAT)ori_size); // in 1/pix
	RFLOAT edge_width = edge_high - edge_low;

	// Put a raised cosine from edge_low to edge_high
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(FT)
	{
		RFLOAT r2 = (RFLOAT)kp * (RFLOAT)kp + (RFLOAT)ip * (RFLOAT)ip + (RFLOAT)jp * (RFLOAT)jp;
		RFLOAT res = sqrt((RFLOAT)r2)/ori_size; // get resolution in 1/pixel

		if (do_highpass_instead)
		{
			if (res < edge_low)
				DIRECT_A3D_ELEM(FT, k, i, j) = 0.;
			else if (res > edge_high)
				continue;
			else
				DIRECT_A3D_ELEM(FT, k, i, j) *= 0.5 - 0.5 * cos( PI * (res-edge_low)/edge_width);
		}
		else
		{
			if (res < edge_low)
				continue;
			else if (res > edge_high)
				DIRECT_A3D_ELEM(FT, k, i, j) = 0.;
			else
				DIRECT_A3D_ELEM(FT, k, i, j) *= 0.5 + 0.5 * cos( PI * (res-edge_low)/edge_width);
		}
	}


}
void lowPassFilterMap(MultidimArray<RFLOAT > &img, RFLOAT low_pass, RFLOAT angpix, int filter_edge_width)
{
	refuseNonCubicMap(img, "low-pass filtering (lowPassFilterMap)");
	FourierTransformer transformer;
	MultidimArray<Complex > FT;

	// Make this work for maps (or more likely 2D images) that have unequal X and Y dimensions
	img.setXmippOrigin();
	int my_xsize = XSIZE(img);
	int my_ysize = YSIZE(img);
	int my_size = (my_xsize != my_ysize) ? XMIPP_MAX(my_xsize, my_ysize) : my_xsize;
	if (my_xsize != my_ysize)
	{
		if (img.getDim() == 2)
		{
			int my_small_size = XMIPP_MIN(my_xsize, my_ysize);
			RFLOAT avg,stddev,minn,maxx;
			img.computeStats(avg,stddev,minn,maxx);
			img.window(FIRST_XMIPP_INDEX(my_size), FIRST_XMIPP_INDEX(my_size),
					   LAST_XMIPP_INDEX(my_size),  LAST_XMIPP_INDEX(my_size));
			if (my_small_size == my_xsize)
			{
				FOR_ALL_ELEMENTS_IN_ARRAY2D(img)
				{
					if (j <  FIRST_XMIPP_INDEX(my_small_size) || j >  LAST_XMIPP_INDEX(my_small_size))
						A2D_ELEM(img, i, j) = rnd_gaus(avg, stddev);
				}
			}
			else
			{
				FOR_ALL_ELEMENTS_IN_ARRAY2D(img)
				{
					if (i <  FIRST_XMIPP_INDEX(my_small_size) || i >  LAST_XMIPP_INDEX(my_small_size))
						A2D_ELEM(img, i, j) = rnd_gaus(avg, stddev);
				}

			}
		}
		else
		{
			REPORT_ERROR("lowPassFilterMap: filtering of non-cube maps is not implemented...");
		}
	}
	transformer.FourierTransform(img, FT, false);
	lowPassFilterMap(FT, XSIZE(img), low_pass, angpix, filter_edge_width, false);
	transformer.inverseFourierTransform();
	img.setXmippOrigin();
	if (my_xsize != my_ysize)
	{
		if (img.getDim() == 2)
		{
			img.window(FIRST_XMIPP_INDEX(my_ysize), FIRST_XMIPP_INDEX(my_xsize),
					   LAST_XMIPP_INDEX(my_ysize),	LAST_XMIPP_INDEX(my_xsize));
		}
		else
		{
			REPORT_ERROR("lowPassFilterMap: filtering of non-cube maps is not implemented...");
		}
	}


}

void highPassFilterMap(MultidimArray<RFLOAT > &img, RFLOAT low_pass, RFLOAT angpix, int filter_edge_width)
{
	refuseNonCubicMap(img, "high-pass filtering (highPassFilterMap)");
	FourierTransformer transformer;
	MultidimArray<Complex > FT;
	transformer.FourierTransform(img, FT, false);
	lowPassFilterMap(FT, XSIZE(img), low_pass, angpix, filter_edge_width, true);
	transformer.inverseFourierTransform();
}

void directionalFilterMap(MultidimArray<Complex > &FT, int ori_size,
		RFLOAT low_pass, RFLOAT angpix, std::string axis, int filter_edge_width)
{


	// Which resolution shell is the filter?
	int ires_filter = ROUND((ori_size * angpix)/low_pass);
	int filter_edge_halfwidth = filter_edge_width / 2;

	// Soft-edge: from 1 shell less to one shell more:
	RFLOAT edge_low = XMIPP_MAX(0., (ires_filter - filter_edge_halfwidth) / (RFLOAT)ori_size); // in 1/pix
	RFLOAT edge_high = XMIPP_MIN(XSIZE(FT), (ires_filter + filter_edge_halfwidth) / (RFLOAT)ori_size); // in 1/pix
	RFLOAT edge_width = edge_high - edge_low;

	if (axis == "x" || axis == "X")
	{
		FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(FT)
		{

			RFLOAT r2 = (RFLOAT)jp * (RFLOAT)jp;
			RFLOAT res = sqrt((RFLOAT)r2)/ori_size; // get resolution in 1/pixel

			if (res < edge_low)
				continue;
			else if (res > edge_high)
				DIRECT_A3D_ELEM(FT, k, i, j) = 0.;
			else
				DIRECT_A3D_ELEM(FT, k, i, j) *= 0.5 + 0.5 * cos( PI * (res-edge_low)/edge_width);
		}
	}
	else if (axis == "y" || axis == "Y")
	{

		FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(FT)
		{

			RFLOAT r2 = (RFLOAT)ip * (RFLOAT)ip;
			RFLOAT res = sqrt((RFLOAT)r2)/ori_size; // get resolution in 1/pixel

			if (res < edge_low)
				continue;
			else if (res > edge_high)
				DIRECT_A3D_ELEM(FT, k, i, j) = 0.;
			else
				DIRECT_A3D_ELEM(FT, k, i, j) *= 0.5 + 0.5 * cos( PI * (res-edge_low)/edge_width);
		}
	}
	else if  (axis == "z" || axis == "Z")
	{
		FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(FT)
		{

			RFLOAT r2 = (RFLOAT)kp * (RFLOAT)kp;
			RFLOAT res = sqrt((RFLOAT)r2)/ori_size; // get resolution in 1/pixel

			if (res < edge_low)
				continue;
			else if (res > edge_high)
				DIRECT_A3D_ELEM(FT, k, i, j) = 0.;
			else
				DIRECT_A3D_ELEM(FT, k, i, j) *= 0.5 + 0.5 * cos( PI * (res-edge_low)/edge_width);
		}
	}


}

void directionalFilterMap(MultidimArray<RFLOAT > &img, RFLOAT low_pass, RFLOAT angpix, std::string axis, int filter_edge_width)
{
	refuseNonCubicMap(img, "directional filtering (directionalFilterMap)");
	FourierTransformer transformer;
	MultidimArray<Complex > FT;

	// Make this work for maps (or more likely 2D images) that have unequal X and Y dimensions
	img.setXmippOrigin();
	int my_xsize = XSIZE(img);
	int my_ysize = YSIZE(img);
	int my_size = (my_xsize != my_ysize) ? XMIPP_MAX(my_xsize, my_ysize) : my_xsize;
	if (my_xsize != my_ysize)
	{
		if (img.getDim() == 2)
		{
			int my_small_size = XMIPP_MIN(my_xsize, my_ysize);
			RFLOAT avg,stddev,minn,maxx;
			img.computeStats(avg,stddev,minn,maxx);
			img.window(FIRST_XMIPP_INDEX(my_size), FIRST_XMIPP_INDEX(my_size),
					   LAST_XMIPP_INDEX(my_size),  LAST_XMIPP_INDEX(my_size));
			if (my_small_size == my_xsize)
			{
				FOR_ALL_ELEMENTS_IN_ARRAY2D(img)
				{
					if (j <  FIRST_XMIPP_INDEX(my_small_size) || j >  LAST_XMIPP_INDEX(my_small_size))
						A2D_ELEM(img, i, j) = rnd_gaus(avg, stddev);
				}
			}
			else
			{
				FOR_ALL_ELEMENTS_IN_ARRAY2D(img)
				{
					if (i <  FIRST_XMIPP_INDEX(my_small_size) || i >  LAST_XMIPP_INDEX(my_small_size))
						A2D_ELEM(img, i, j) = rnd_gaus(avg, stddev);
				}

			}
		}
		else
		{
			REPORT_ERROR("lowPassFilterMap: filtering of non-cube maps is not implemented...");
		}
	}
	transformer.FourierTransform(img, FT, false);
	directionalFilterMap(FT, XSIZE(img), low_pass, angpix, axis, filter_edge_width);
	transformer.inverseFourierTransform();
	img.setXmippOrigin();
	if (my_xsize != my_ysize)
	{
		if (img.getDim() == 2)
		{
			img.window(FIRST_XMIPP_INDEX(my_ysize), FIRST_XMIPP_INDEX(my_xsize),
					   LAST_XMIPP_INDEX(my_ysize),	LAST_XMIPP_INDEX(my_xsize));
		}
		else
		{
			REPORT_ERROR("lowPassFilterMap: filtering of non-cube maps is not implemented...");
		}
	}

}


void applyBeamTilt(const MultidimArray<Complex > &Fin, MultidimArray<Complex > &Fout, RFLOAT beamtilt_x, RFLOAT beamtilt_y,
		RFLOAT wavelength, RFLOAT Cs, RFLOAT angpix, int ori_size)
{

	Fout = Fin;
	selfApplyBeamTilt(Fout, beamtilt_x, beamtilt_y, wavelength, Cs, angpix, ori_size);
}

void selfApplyBeamTilt(MultidimArray<Complex > &Fimg, RFLOAT beamtilt_x, RFLOAT beamtilt_y,
		RFLOAT wavelength, RFLOAT Cs, RFLOAT angpix, int ori_size)
{
	if (Fimg.getDim() != 2)
		REPORT_ERROR("applyBeamTilt can only be done on 2D Fourier Transforms!");

	RFLOAT boxsize = angpix * ori_size;
	RFLOAT factor = 0.360 * Cs * 10000000 * wavelength * wavelength / (boxsize * boxsize * boxsize);
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(Fimg)
	{
		RFLOAT delta_phase = factor * (ip * ip + jp * jp) * (ip * beamtilt_y + jp * beamtilt_x);
		RFLOAT realval = DIRECT_A2D_ELEM(Fimg, i, j).real;
		RFLOAT imagval = DIRECT_A2D_ELEM(Fimg, i, j).imag;
		RFLOAT mag = sqrt(realval * realval + imagval * imagval);
		RFLOAT phas = atan2(imagval, realval) + DEG2RAD(delta_phase); // apply phase shift!
		realval = mag * cos(phas);
		imagval = mag * sin(phas);
		DIRECT_A2D_ELEM(Fimg, i, j) = Complex(realval, imagval);
	}

}

void selfApplyBeamTilt(MultidimArray<Complex > &Fimg,
		RFLOAT beamtilt_x, RFLOAT beamtilt_y,
		RFLOAT beamtilt_xx, RFLOAT beamtilt_xy, RFLOAT beamtilt_yy,
		RFLOAT wavelength, RFLOAT Cs, RFLOAT angpix, int ori_size)
{
	if (Fimg.getDim() != 2)
		REPORT_ERROR("applyBeamTilt can only be done on 2D Fourier Transforms!");

	RFLOAT boxsize = angpix * ori_size;
	RFLOAT factor = 0.360 * Cs * 10000000 * wavelength * wavelength / (boxsize * boxsize * boxsize);

	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(Fimg)
	{
		RFLOAT q = beamtilt_xx * jp * jp + 2.0 * beamtilt_xy * ip * jp + beamtilt_yy * ip * ip;

		RFLOAT delta_phase = factor * q * (ip * beamtilt_y + jp * beamtilt_x);
		RFLOAT realval = DIRECT_A2D_ELEM(Fimg, i, j).real;
		RFLOAT imagval = DIRECT_A2D_ELEM(Fimg, i, j).imag;
		RFLOAT mag = sqrt(realval * realval + imagval * imagval);
		RFLOAT phas = atan2(imagval, realval) + DEG2RAD(delta_phase); // apply phase shift!
		realval = mag * cos(phas);
		imagval = mag * sin(phas);
		DIRECT_A2D_ELEM(Fimg, i, j) = Complex(realval, imagval);
	}

}

void padAndFloat2DMap(const MultidimArray<RFLOAT > &v, MultidimArray<RFLOAT> &out, int factor)
{
	long int Xdim, Ydim, Zdim, Ndim, XYdim;
	RFLOAT bg_val, bg_pix, bd_val, bd_pix;

	out.clear();

	// Check dimensions
	v.getDimensions(Xdim, Ydim, Zdim, Ndim);
	if ( (Zdim > 1) || (Ndim > 1) )
		REPORT_ERROR("fftw.cpp::padAndFloat2DMap(): ERROR MultidimArray should be 2D.");
	if (Xdim * Ydim <= 16)
		REPORT_ERROR("fftw.cpp::padAndFloat2DMap(): ERROR MultidimArray is too small.");
	if (factor <= 1)
		REPORT_ERROR("fftw.cpp::padAndFloat2DMap(): ERROR Padding factor should be larger than 1.");

	// Calculate background and border values
	bg_val = bg_pix = bd_val = bd_pix = 0.;
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(v)
	{
		bg_val += DIRECT_A2D_ELEM(v, i, j);
		bg_pix += 1.;
		if ( (i == 0) || (j == 0) || (i == (YSIZE(v) - 1)) || (j == (XSIZE(v) - 1)) )
		{
			bd_val += DIRECT_A2D_ELEM(v, i, j);
			bd_pix += 1.;
		}
	}
	if ( (bg_pix < 1.) || (bd_pix < 1.) )
		REPORT_ERROR("fftw.cpp::padAndFloat2DMap(): ERROR MultidimArray is too small.");
	bg_val /= bg_pix;
	bd_val /= bd_pix;
	// DEBUG
	//std::cout << "bg_val = " << bg_val << ", bg_pix = " << bg_pix << std::endl;
	//std::cout << "bd_val = " << bd_val << ", bd_pix = " << bd_pix << std::endl;

	// Pad and float output MultidimArray (2x original size by default)
	XYdim = (Xdim > Ydim) ? (Xdim * factor) : (Ydim * factor);
	out.resize(XYdim, XYdim);
	out.initConstant(bd_val - bg_val);
	out.setXmippOrigin();
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(v)
	{
		A2D_ELEM(out, i + FIRST_XMIPP_INDEX(YSIZE(v)), j + FIRST_XMIPP_INDEX(XSIZE(v))) = DIRECT_A2D_ELEM(v, i, j) - bg_val;
	}
}

void amplitudeOrPhaseMap(const MultidimArray<RFLOAT > &v, MultidimArray<RFLOAT > &amp, int output_map_type)
{
	long int XYdim, maxr2;
	RFLOAT val;
	FourierTransformer transformer;
	MultidimArray<Complex > Faux;
	MultidimArray<RFLOAT > out;

	transformer.clear();
	Faux.clear();
	out.clear();

	// Pad and float
	padAndFloat2DMap(v, out);
	if ( (XSIZE(out) != YSIZE(out)) || (ZSIZE(out) > 1) || (NSIZE(out) > 1) )
		REPORT_ERROR("fftw.cpp::amplitudeOrPhaseMap(): ERROR MultidimArray should be 2D square.");
	XYdim = XSIZE(out);

	// Fourier Transform
	transformer.FourierTransform(out, Faux, false); // TODO: false???
	CenterFFTbySign(Faux);

	// Write to output files
	out.setXmippOrigin();
	out.initZeros(XYdim, XYdim);
	maxr2 = (XYdim - 1) * (XYdim - 1) / 4;
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(Faux)
	{
		if ( (ip > STARTINGY(out)) && (ip < FINISHINGY(out))
				&& (jp > STARTINGX(out)) && (jp < FINISHINGX(out))
				&& ((ip * ip + jp * jp) < maxr2) )
		{
			if (output_map_type == AMPLITUDE_MAP)
				val = FFTW2D_ELEM(Faux, ip, jp).abs();
			else if (output_map_type == PHASE_MAP)
				val = (180.) * (FFTW2D_ELEM(Faux, ip, jp).arg()) / PI;
			else
				REPORT_ERROR("fftw.cpp::amplitudeOrPhaseMap(): ERROR Unknown type of output map.");

			A2D_ELEM(out, -ip, -jp) = A2D_ELEM(out, ip, jp) = val;
		}
	}
	A2D_ELEM(out, 0, 0) = 0.;
	amp.clear();
	amp = out;
}

void helicalLayerLineProfile(const MultidimArray<RFLOAT > &v, std::string title, std::string fn_eps)
{
	long int XYdim, maxr2;
	FourierTransformer transformer;
	MultidimArray<Complex > Faux;
	MultidimArray<RFLOAT > out;
	std::vector<RFLOAT > ampl_list, ampr_list, nr_pix_list;

	transformer.clear();
	Faux.clear();
	out.clear();

	// TODO: DO I NEED TO ROTATE THE ORIGINAL MULTIDINARRAY BY 90 DEGREES ?

	// Pad and float
	padAndFloat2DMap(v, out);
	if ( (XSIZE(out) != YSIZE(out)) || (ZSIZE(out) > 1) || (NSIZE(out) > 1) )
		REPORT_ERROR("fftw.cpp::helicalLayerLineProfile(): ERROR MultidimArray should be 2D square.");
	XYdim = XSIZE(out);

	// Fourier Transform
	transformer.FourierTransform(out, Faux, false); // TODO: false???
	CenterFFTbySign(Faux);

	// Statistics
	out.setXmippOrigin();
	maxr2 = (XYdim - 1) * (XYdim - 1) / 4;
	ampl_list.resize(XSIZE(Faux) + 2);
	ampr_list.resize(XSIZE(Faux) + 2);
	nr_pix_list.resize(XSIZE(Faux) + 2);
	for (int ii = 0; ii < ampl_list.size(); ii++)
		ampl_list[ii] = ampr_list[ii] = nr_pix_list[ii] = 0.;

	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(Faux)
	{
		if ( ((ip * ip + jp * jp) < maxr2) && (ip > 0) )
		{
			nr_pix_list[jp] += 1.;
			ampl_list[jp] += FFTW2D_ELEM(Faux, ip, jp).abs();
			ampr_list[jp] += FFTW2D_ELEM(Faux, -ip, jp).abs();
		}
	}
	CDataSet dataSetAmpl, dataSetAmpr;
	RFLOAT linewidth = 1.0;
	std::string figTitle = "Helical Layer Line Profile - " + title;
	std::string yTitle = "Reciprocal pixels (padded box size = " + integerToString(XYdim) + ")";
	for (int ii = 0; ii < (3 * ampl_list.size() / 4 + 1); ii++)
	{
		if (nr_pix_list[ii] < 1.)
			break; // TODO: IS THIS CORRECT?
		dataSetAmpl.AddDataPoint(CDataPoint(ii, log(ampl_list[ii] / nr_pix_list[ii])));
		dataSetAmpr.AddDataPoint(CDataPoint(ii, log(ampr_list[ii] / nr_pix_list[ii])));
	}
	dataSetAmpl.SetDrawMarker(false);
	dataSetAmpl.SetLineWidth(linewidth);
	dataSetAmpl.SetDatasetColor(1., 0., 0.);
	dataSetAmpl.SetDatasetTitle("ln(amplitudes) (left)");
	dataSetAmpr.SetDrawMarker(false);
	dataSetAmpr.SetLineWidth(linewidth);
	dataSetAmpr.SetDatasetColor(0., 1., 0.);
	dataSetAmpr.SetDatasetTitle("ln(amplitudes) (right)");
	CPlot2D *plot2D = new CPlot2D(figTitle);
	plot2D->SetXAxisSize(600);
	plot2D->SetYAxisSize(400);
	plot2D->SetXAxisTitle(yTitle);
	plot2D->SetYAxisTitle("ln(amplitudes)");
	plot2D->AddDataSet(dataSetAmpl);
	plot2D->AddDataSet(dataSetAmpr);
	plot2D->OutputPostScriptPlot(fn_eps);
	delete plot2D;
}

void generateBinaryHelicalFourierMask(MultidimArray<RFLOAT> &mask, std::vector<RFLOAT> exclude_begin, std::vector<RFLOAT> exclude_end, RFLOAT angpix)
{
	if (exclude_begin.size() != exclude_end.size()) REPORT_ERROR("BUG: generateHelicalFourierMask: provide start-end resolutions for each shell.");

	mask.initConstant(1.);

	bool is_2d = (mask.getDim() == 2);
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(mask)
	{
		RFLOAT res;
		if (is_2d) res = (jp == 0) ? 999. : YSIZE(mask) * angpix / fabs(jp); // helical axis along X-axis, so only jp matters!
		else res = (kp == 0) ? 999. : ZSIZE(mask) * angpix / fabs(kp); // helical axis along Z-axis, so only kp matters!

		for (int ishell = 0; ishell < exclude_begin.size(); ishell++)
		{
			if (res <= exclude_begin[ishell] && res >= exclude_end[ishell]) DIRECT_A3D_ELEM(mask, k, i, j) = 0.;
		}
	}

}

#if defined(MKLFFT) || defined(RELION_HAVE_FFTW_THREADS)
#if defined(RELION_SINGLE_PRECISION) && !defined(MKLFFT)
#define RELION_FFTW_THREADS_API(f) fftwf_##f
#else
#define RELION_FFTW_THREADS_API(f) fftw_##f
#endif

// RELION_FFTW_THREADS=1 keeps every transform single-threaded
static bool relionFftwThreadsAllowed()
{
	static const bool allowed = []()
	{
		const char *e = getenv("RELION_FFTW_THREADS");
		return !(e != NULL && atoi(e) == 1);
	}();
	return allowed;
}

void relionFftwInitThreads()
{
	if (!relionFftwThreadsAllowed())
		return;
	if (0 == RELION_FFTW_THREADS_API(init_threads)())
		REPORT_ERROR("Multithreaded FFTW failed to initialize");
	g_fftw_threads_inited = true;
}

// The setter takes the planner's lock: a background thread that plans with its own thread count
// (relionFftwSetThreadPlanThreads) changes the planner's count and restores it inside that lock
void relionFftwPlanThreads(int nr_threads)
{
	if (relionFftwThreadsAllowed())
	{
		#pragma omp critical(FourierTransformer_fftw_plan)
		RELION_FFTW_THREADS_API(plan_with_nthreads)(nr_threads);
	}
}

void relionFftwSetThreadPlanThreads(int nr_threads)
{
	g_plan_threads_override = std::max(0, nr_threads);
}

void relionFftwCleanupThreads()
{
	if (relionFftwThreadsAllowed())
		RELION_FFTW_THREADS_API(cleanup_threads)();
	g_fftw_threads_inited = false;
}
#else
// Without FFTW's thread library every transform is single-threaded
void relionFftwInitThreads() {}
void relionFftwPlanThreads(int) {}
void relionFftwSetThreadPlanThreads(int) {}
void relionFftwCleanupThreads() {}
#endif

int relionBulkThreads(size_t n)
{
	return (n >= ((size_t)1 << 20)) ? relionFftwLoopThreads() : 1;
}

void relionZeroMemory(void *p, size_t bytes)
{
	if (bytes == 0)
		return;
	const int nt = (bytes >= ((size_t)32 << 20)) ? relionBulkThreads(bytes) : 1;
	if (nt <= 1)
	{
		memset(p, 0, bytes);
		return;
	}
	char *c = (char *)p;
	#pragma omp parallel for num_threads(nt)
	for (int t = 0; t < nt; t++)
	{
		const size_t b0 = bytes / nt * t, b1 = (t == nt - 1) ? bytes : bytes / nt * (t + 1);
		memset(c + b0, 0, b1 - b0);
	}
}
