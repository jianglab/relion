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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/
#ifndef __PROJECTOR_H
#define __PROJECTOR_H

#include "src/fftw.h"
#include "src/multidim_array.h"
#include "src/image.h"
#include "src/finufft_central_slice.h"

#include <src/jaz/single_particle/volume.h>
#include <src/jaz/gravis/t2Vector.h>

#ifdef _CUDA_ENABLED
#include "src/acc/cuda/cuda_mem_utils.h"
using deviceStream_t = cudaStream_t;
#elif _HIP_ENABLED
#include "src/acc/hip/hip_mem_utils.h"
using deviceStream_t = hipStream_t;
#endif

#if defined _CUDA_ENABLED || defined _HIP_ENABLED
#include "src/acc/acc_ptr.h"
void run_griddingCorrect(RFLOAT *vol, int interpolator, RFLOAT rrval, RFLOAT r_min_nn,
						 size_t iX, size_t iY, size_t iZ);

void run_padTranslatedMap(RFLOAT *d_in, RFLOAT *d_out,
						  size_t isX, size_t ieX, size_t isY, size_t ieY, size_t isZ, size_t ieZ, //Input dimensions
						  size_t osX, size_t oeX, size_t osY, size_t oeY, size_t osZ, size_t oeZ,  //Output dimensions
						  deviceStream_t stream = 0);

void run_CenterFFTbySign(Complex *img_in, int xSize, int ySize, int zSize, deviceStream_t = 0);

void run_calcPowerSpectrum(Complex *dFaux, int padoridim, Complex *ddata, int data_sz, RFLOAT *dpower_spectrum, RFLOAT *dcounter,
											  int max_r2, int min_r2, RFLOAT normfft, RFLOAT padding_factor, RFLOAT weight,
											  RFLOAT *dfourier_mask, int fx, int fy, int fz, bool do_fourier_mask, bool if3D);

void run_updatePowerSpectrum(RFLOAT *dcounter, int sz, RFLOAT *dpower_spectrum);

extern void scale(RFLOAT *img, size_t sz, RFLOAT val, deviceStream_t stream = 0);
#endif



#define NEAREST_NEIGHBOUR 0
#define TRILINEAR 1
#define CONVOLUTE_BLOB 2
// Exact band-limited interpolation of the gridded Fourier volume, evaluated
// with a FINUFFT type-2 transform (forward projection only; BackProjector
// still uses TRILINEAR).  See src/relion_finufft.h for the maths.
#define FINUFFT 3

#define FORWARD_PROJECTION 0
#define BACKWARD_PROJECTION 1

#define ACT_ON_DATA 0
#define ACT_ON_WEIGHT 1

class Projector
{
public:
	// The Fourier-space image data array
	MultidimArray<Complex > data;

	// Only points within this many pixels from the origin (in the original size) will be interpolated
	int r_max;

	// Radius of sphere within TRILINEAR interpolation will be used in NEAREST_NEIGHBOUR interpolator
	int r_min_nn;

	// Original size of the real-space map
	int ori_size;

	// Padded size of the map in Fourier-space
	int pad_size;

	// Interpolation scheme (TRILINEAR or NEAREST_NEIGHBOUR, for BackProjector also CONVOLUTE_BLOB)
	int interpolator;

	// Oversample FT by padding in real space
	float padding_factor;

	// Dimension of the reference (currently allowed 2 or 3)
	int ref_dim;

	// Dimension of the projections (1 or 2 or 3)
	int data_dim;

	// Only used when interpolator == FINUFFT: the conjugate-domain (real-space)
	// representation of `data` that serves as the FINUFFT type-2 mode array.
	// Built by prepareFinufft(), which computeFourierTransformMap() calls.
	FinufftProjectorModes finufft_modes;

	// Size of the periodic real-space box that `data` samples ("padoridim").
	// Recorded by computeFourierTransformMap(); 0 when not yet known.
	int padded_real_size;

	// --- Static configuration of the FINUFFT projector (set from the command line) ---

	// Number of retained real-space samples per dimension, as a multiple of
	// ori_size.  1.0 keeps exactly the unpadded box (cheapest); 2.0 keeps the
	// whole padded box, which makes the interpolation of `data` exact at the
	// price of 8x the memory and a 64x larger internal FFT inside FINUFFT.
	static float finufft_mode_crop;

	// Requested relative accuracy of the FINUFFT transform.
	static double finufft_tol;

	// FINUFFT's internal upsampling factor.  Its FFT is
	// (finufft_upsampfac * mode_size)^dim, so 1.25 does roughly a quarter of the
	// FFT work of 2.0 in 3D at the price of a wider spreading kernel; that FFT
	// dominates the cost here, so 1.25 is the default.  0 lets FINUFFT choose.
	static double finufft_upsampfac;

	/* Which interpolator the FORWARD projectors should use.
	 *
	 * The default is FINUFFT (exact band-limited central slices).  Set the
	 * environment variable
	 *
	 *     RELION_INTERPOLATION=linear
	 *
	 * to get the classic trilinear/bilinear resampling back; "nufft" (or leaving
	 * the variable unset) selects FINUFFT.  Backprojection is not affected and
	 * always uses TRILINEAR.
	 *
	 * FINUFFT is a CPU-only path, so accelerated runs fall back to TRILINEAR:
	 * pass accelerator_in_use = true for those.  The fallback is silent-ish (a
	 * one-line note when verb > 0) unless the user asked for NUFFT explicitly via
	 * the environment variable, in which case it is an error.
	 */
	static int resolveForwardInterpolator(bool accelerator_in_use, int verb = 1);

	/* Seed finufft_mode_crop / finufft_tol from RELION_FINUFFT_MODE_CROP and
	 * RELION_FINUFFT_TOL.  Call this before parsing the command line, so that an
	 * explicit --finufft_mode_crop / --finufft_tol still wins.
	 */
	static void applyFinufftEnvDefaults();

	/* The current values of the two knobs above, formatted for use as the default
	 * string of a command-line option (so that --finufft_* overrides the
	 * environment rather than the other way round).
	 */
	static std::string finufftModeCropAsString();
	static std::string finufftTolAsString();

public:

	/** Empty constructor
	 *
	 * A default Projector is created.
	 *
	 * @code
	 * Projector PPref;
	 * @endcode
	 */
	Projector()
	{
		clear();
	}

	/** Constructor with parameters
	  *
	  * A default Projector is created.
	  *
	  * @code
	  * Projector PPref(ori_size, NEAREST_NEIGHBOUR);
	  * @endcode
	  */
	Projector(int _ori_size, int _interpolator = TRILINEAR, float _padding_factor_3d = 2., int _r_min_nn = 10, int _data_dim = 2)
	{

		clear();

		// Store original dimension
		ori_size = _ori_size;

		// Padding factor for the map
		if (_padding_factor_3d < 1.0)
			REPORT_ERROR("Padding factor cannot be less than 1.");

		padding_factor = _padding_factor_3d;

		// Interpolation scheme
		interpolator = _interpolator;

		// Minimum radius for NN interpolation
		r_min_nn = _r_min_nn;

		// Dimension of the projections
		data_dim = _data_dim;

	}

	/** Copy constructor
	 *
	 * The created Projector is a perfect copy of the input array but with a
	 * different memory assignment.
	 *
	 * @code
	 * Projector V2(V1);
	 * @endcode
	 */
	Projector(const Projector& op)
	{
		clear();
		*this = op;
	}

	/** Assignment.
	 *
	 * You can build as complex assignment expressions as you like. Multiple
	 * assignment is allowed.
	 */
	Projector& operator=(const Projector& op)
	{
		if (&op != this)
		{
			data = op.data;
			ori_size = op.ori_size;
			pad_size = op.pad_size;
			r_max = op.r_max;
			r_min_nn = op.r_min_nn;
			interpolator = op.interpolator;
			padding_factor = op.padding_factor;
			ref_dim = op.ref_dim;
			data_dim  = op.data_dim;
			padded_real_size = op.padded_real_size;
			finufft_modes = op.finufft_modes;
		}
		return *this;
	}

	/** Destructor
	 *
	 * Clears everything
	 *
	 * @code
	 * FourierInterpolator fourint;
	 * @endcode
	 */
	~Projector()
	{
	    clear();
	}

	/** Clear.
	  * Initialize everything to back to default and empty arrays
	  */
	void clear()
	{
		data.clear();
		r_max = r_min_nn = interpolator = ref_dim = data_dim = pad_size = 0;
		padding_factor = 0.;
		padded_real_size = 0;
		finufft_modes.clear();
	}

	/*
	 * Resize data array to the given size
	 */
	void initialiseData(int current_size = -1);

	/*
	 * Initialise data array to all zeros
	 */
	void initZeros(int current_size = -1);

	/*
	 *  Only get the size of the data array
	 */
	long int getSize();

	/* ** Prepares a 3D map for taking slices in its 3D Fourier Transform
	 *
	 * This routine does the following:
	 * 1. It pads the input map with zeros to obtain oversampling in the Fourier transform
	 * 2. It does the Fourier transform
	 * 3. It sets values beyond Nyquist for images of current_size to zero in the transform and windows the transform at max_r+1
	 * Depending on whether 2D or 3D Fourier Transforms will be extracted, the map is normalized internally in a different manner
	 *
	 * If fourier_mask!= NULL: then apply this oriinally-size, FFTw-centered Fourier mask, also in power_spectrum calculations!
	 *
	 */
	void computeFourierTransformMap(MultidimArray<RFLOAT> &vol_in, MultidimArray<RFLOAT> &power_spectrum,
                                        int current_size = -1, int nr_threads = 1, bool do_gridding = true, bool do_heavy = true,
	                                int min_ires = -1, const MultidimArray<RFLOAT> *fourier_mask = NULL, bool do_gpu = false);

	/* Because we interpolate in Fourier space to make projections and/or reconstructions, we have to correct
	 * the real-space maps by dividing them by the Fourier Transform of the interpolator
	 * Note these corrections are made on the not-oversampled, i.e. originally sized real-space map
	 */
	void griddingCorrect(MultidimArray<RFLOAT> &vol_in);

	/*
	* Go from the Projector-centered fourier transform back to FFTW-uncentered one
	*/
	template <typename T>
	void decenter(MultidimArray<T> &Min, MultidimArray<T> &Mout, int my_rmax2)
	{

		// Mout should already have the right size
		// Initialize to zero
		Mout.initZeros();
		FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Mout)
		{
			if (kp*kp + ip*ip + jp*jp <= my_rmax2)
				DIRECT_A3D_ELEM(Mout, k, i, j) = A3D_ELEM(Min, kp, ip, jp);
		}
	}


	/*
	* Get a 2D Fourier Transform from the 2D or 3D data array
	* Depending on the dimension of the map, this will be a projection or a rotation operation
	*/
	void get2DFourierTransform(MultidimArray<Complex > &img_out, Matrix2D<RFLOAT> &A)
	{
		// Rotation of a 3D Fourier Transform
		if (data_dim == 3)
		{
			if (ref_dim != 3)
				REPORT_ERROR("Projector::get3DFourierTransform%%ERROR: Dimension of the data array should be 3");
			rotate3D(img_out, A);
		}
		else if (data_dim == 1)
		{
			project2Dto1D(img_out, A);
		}
		else
		{
			switch (ref_dim)
			{
			case 2:
				rotate2D(img_out, A);
				break;
			case 3:
				project(img_out, A);
				break;
			default:
				REPORT_ERROR("Projector::get2DSlice%%ERROR: Dimension of the data array should be 2 or 3");
			}
		}
	}

	/*
	* Get a 2D slice from the 3D map (forward projection)
	*/
	void project(MultidimArray<Complex > &img_out, Matrix2D<RFLOAT> &A);

	/*
	* Build the FINUFFT mode array from the current contents of `data`.
	* A no-op unless interpolator == FINUFFT.  Called at the end of
	* computeFourierTransformMap(); call it explicitly after filling `data`
	* by any other route.
	*/
	void prepareFinufft();

	/*
	* Batched equivalent of calling get2DFourierTransform() once per entry of A.
	* All slices are evaluated against the same mode array in a single FINUFFT
	* call, which is what makes the FINUFFT interpolator affordable: FINUFFT's
	* per-call cost is dominated by one FFT of its internal upsampled grid,
	* independent of how many points are evaluated.
	*
	* img_out must already have the right number of entries, each sized and
	* zeroed as for get2DFourierTransform().  Falls back to a plain loop for
	* every interpolator other than FINUFFT.
	*/
	void get2DFourierTransformMany(std::vector<MultidimArray<Complex > > &img_out,
	                               std::vector<Matrix2D<RFLOAT> > &A);

	/* As above, but writing into caller-owned arrays, so a caller that already
	 * holds the destinations does not have to copy them in and out.
	 */
	void get2DFourierTransformMany(std::vector<MultidimArray<Complex > *> &img_out,
	                               std::vector<Matrix2D<RFLOAT> > &A);

	/*
	* Get the two gradients (real and imaginary) of that slice.
	* Note: the gradient has to be computed in 3D and then mapped to 2D.
	* Computing the gradient from a 2D projection would systematically
	* underestimate the magnitude of the gradient.
	*/
	void projectGradient(Volume<gravis::t2Vector<Complex> >& img_out, Matrix2D<RFLOAT>& A);

	/*
	* Get a 1D slice from the 2D map (forward projection)
	*/
	void project2Dto1D(MultidimArray<Complex > &img_out, Matrix2D<RFLOAT> &A);

	/*
	* Get an in-plane rotated version of the 2D map (mere interpolation)
	*/
	void rotate2D(MultidimArray<Complex > &img_out, Matrix2D<RFLOAT> &A);

	/*
	* Get a rotated version of the 3D map (mere interpolation)
	*/
	void rotate3D(MultidimArray<Complex > &img_out, Matrix2D<RFLOAT> &A);
};
#endif
