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
#include "src/reconstructor.h"
#include "src/cache_manager.h"

#include "src/spatial_frequency_grid.h"
#include "src/relion_finufft.h"

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <map>
#include <unistd.h>
#ifdef S2_PROFILE
#include <chrono>
#endif

namespace
{

constexpr RFLOAT TWO_PI = 2.0 * PI;

#ifdef S2_PROFILE
using Clock = std::chrono::high_resolution_clock;

struct S2Profile {
	long long ns_read_img = 0;
	long long ns_fft = 0;
	long long ns_ctf_precomp = 0;
	long long ns_ewald_precomp = 0;
	long long ns_grid_ensure = 0;
	long long ns_finufft = 0;
	long long ns_shift_jac = 0;
	long long ns_dc_zero = 0;
	long long ns_bilinear = 0;
	long long ns_ctf_analytical = 0;

	long long ns_backproject = 0;
	long long ns_total = 0;
	int count = 0;
	bool is_ewald = false;
	int nk = 0;
	int boxsize = 0;

	void accumulate(const S2Profile& o) {
		ns_read_img += o.ns_read_img;
		ns_fft += o.ns_fft;
		ns_ctf_precomp += o.ns_ctf_precomp;
		ns_ewald_precomp += o.ns_ewald_precomp;
		ns_grid_ensure += o.ns_grid_ensure;
		ns_finufft += o.ns_finufft;
		ns_shift_jac += o.ns_shift_jac;
		ns_dc_zero += o.ns_dc_zero;
		ns_bilinear += o.ns_bilinear;
		ns_ctf_analytical += o.ns_ctf_analytical;
		ns_backproject += o.ns_backproject;
		ns_total += o.ns_total;
		count += o.count;
		if (o.count > 0) {
			is_ewald = o.is_ewald;
			nk = o.nk;
			boxsize = o.boxsize;
		}
	}

	void print(int rank) const {
		auto ms = [](long long ns) { return (double)ns / 1e6; };
		fprintf(stderr, "=== S2 PROFILE rank=%d n_particles=%d box=%d nk=%d ewald=%d ===\n",
			rank, count, boxsize, nk, is_ewald ? 1 : 0);
		if (count > 0) {
			fprintf(stderr, " total_per_particle: %.3f ms\n", ms(ns_total / count));
			fprintf(stderr, " read_img: %.3f ms\n", ms(ns_read_img / count));
			fprintf(stderr, " fft: %.3f ms\n", ms(ns_fft / count));
			fprintf(stderr, " ctf_precomp: %.3f ms\n", ms(ns_ctf_precomp / count));
			fprintf(stderr, " ewald_precomp: %.3f ms\n", ms(ns_ewald_precomp / count));
			fprintf(stderr, " grid_ensure: %.3f ms\n", ms(ns_grid_ensure / count));
			fprintf(stderr, " finufft: %.3f ms\n", ms(ns_finufft / count));
			fprintf(stderr, " shift_jacobian: %.3f ms\n", ms(ns_shift_jac / count));
			fprintf(stderr, " dc_zero: %.3f ms\n", ms(ns_dc_zero / count));
			fprintf(stderr, " bilinear_interp: %.3f ms\n", ms(ns_bilinear / count));
			fprintf(stderr, " ctf_analytical: %.3f ms\n", ms(ns_ctf_analytical / count));
			fprintf(stderr, " backproject: %.3f ms\n", ms(ns_backproject / count));
		} else {
			fprintf(stderr, " (no particles profiled)\n");
		}
		fprintf(stderr, "=== END S2 PROFILE ===\n");
		fflush(stderr);
	}
};

static thread_local S2Profile tls_s2_profile;
#endif

SpatialFrequencyMode parseSpatialFrequencyMode(const std::string& mode)
{
	if (mode == "s")
	{
		return SPATIAL_FREQUENCY_MODE_S;
	}

	if (mode == "s2")
	{
		return SPATIAL_FREQUENCY_MODE_S2;
	}

	REPORT_ERROR("ERROR: --spatial_frequency_mode must be either 's' or 's2'.");
	return SPATIAL_FREQUENCY_MODE_S;
}




void extractCartesianSamplesAndWeightsFromSPath(const MultidimArray<Complex>& F2D,
	                                            const MultidimArray<RFLOAT>& Fctf,
	                                            bool do_ctf,
	                                            bool ctf_premultiplied,
	                                            bool do_fom_weighting,
	                                            RFLOAT fom,
	                                            std::vector<Complex>& samples,
	                                            std::vector<RFLOAT>& weights)
{
	const int size = YSIZE(F2D);
	const int sh = XSIZE(F2D);

	samples.clear();
	weights.clear();
	samples.reserve(size * sh);
	weights.reserve(size * sh);

	for (int i = 0; i < size; i++)
	{
		const int first_x = (i < sh) ? 0 : 1;

		for (int x = first_x; x < sh; x++)
		{
			Complex val = DIRECT_A2D_ELEM(F2D, i, x);
			RFLOAT w = 1.0;

			if (do_ctf)
			{
				const RFLOAT ctf_value = DIRECT_A2D_ELEM(Fctf, i, x);
				if (!ctf_premultiplied)
				{
					val *= ctf_value;
				}
				w *= ctf_value * ctf_value;
			}

			if (do_fom_weighting)
			{
				val *= fom;
				w *= fom;
			}

			samples.push_back(val);
			weights.push_back(w);
		}
	}

	if (!samples.empty())
	{
		samples[0] = Complex(0.0, 0.0);
	}
}

void applyFourierShiftToSamples(std::vector<Complex>& samples,
	                            const SpatialFrequencyGrid2D& grid,
	                            const Matrix1D<RFLOAT>& trans,
	                            int size)
{
	if (ABS(XX(trans)) < XMIPP_EQUAL_ACCURACY && ABS(YY(trans)) < XMIPP_EQUAL_ACCURACY)
	{
		return;
	}

	for (long int idx = 0; idx < (long int)samples.size(); idx++)
	{
		const RFLOAT phase = (RFLOAT)(-TWO_PI * (grid.sample_x[idx] * XX(trans) + grid.sample_y[idx] * YY(trans)) / size);
		const Complex shift(cos(phase), sin(phase));
		samples[idx] *= shift;
	}
}

}

void Reconstructor::read(int argc, char **argv)
{
	parser.setCommandLine(argc, argv);

	int general_section = parser.addSection("General options");
	fn_sel = parser.getOption("--i", "Input STAR file with the projection images and their orientations", "");
	fn_out = parser.getOption("--o", "Name for output reconstruction","relion.mrc");
	fn_sym = parser.getOption("--sym", "Symmetry group", "c1");
	maxres = textToFloat(parser.getOption("--maxres", "Maximum resolution (in Angstrom) to consider in Fourier space (default Nyquist)", "-1"));
	padding_factor = textToFloat(parser.getOption("--pad", "Padding factor", "2"));
	image_path = parser.getOption("--img", "Optional: image path prefix", "");
	subset = textToInteger(parser.getOption("--subset", "Subset of images to consider (1: only reconstruct half1; 2: only half2; other: reconstruct all)", "-1"));
	chosen_class = textToInteger(parser.getOption("--class", "Consider only this class (-1: use all classes)", "-1"));
	angpix  = textToFloat(parser.getOption("--angpix", "Pixel size in the reconstruction (take from first optics group by default)", "-1"));
	spatial_frequency_mode = parseSpatialFrequencyMode(parser.getOption("--spatial_frequency_mode", "Spatial-frequency sampling mode for reconstruction ('s' for the current Cartesian frequency grid; 's2' for signed s^2 Cartesian sampling)", "s"));
s2_ctf_oversampling_min = textToInteger(parser.getOption("--s2_ctf_oversampling_min",
            "For --spatial_frequency_mode s2: minimum number of s2 samples per CTF oscillation (0 = no fill/Cartesian-only, default 2)", "2"));
	do_s2_grid_diagnostic = parser.checkOption("--save_s2_grid_diagnostic", "Save s2 grid diagnostic data and plot script");
	s2_uniform_fill_ = parser.checkOption("--s2_uniform_fill", "For --spatial_frequency_mode s2: use uniform 2x fill (all half-grid positions, equivalent to 2x real-space padding) instead of adaptive s2 oversampling");
	do_half1 = parser.checkOption("--do_half1", "Reconstruct half-1 map (subset 1)");
	do_half2 = parser.checkOption("--do_half2", "Reconstruct half-2 map (subset 2)");
	do_alldata = parser.checkOption("--do_alldata", "Reconstruct full map from all data");
	do_prefetch = !parser.checkOption("--no_prefetch", "Disable background prefetching of particle images while backprojecting");
	random_subset_size = textToInteger(parser.getOption("--random_subset_size", "After reconstruction, also reconstruct a random subset of this many particles (for quick validation)", "-1"));
	random_subset_seed = textToInteger(parser.getOption("--random_subset_seed", "Seed for the random subset selection (deterministic across MPI ranks)", "42"));
	fn_cache = parser.getOption("--cache_dir", "Local SSD cache directory for persistent particle stack caching", "");
	cache_copy_threads = textToInteger(parser.getOption("--cache_copy_threads", "Number of parallel threads for cache file copy (default 4)", "4"));
	nr_threads = textToInteger(parser.getOption("--j", "Number of threads to run in parallel (only useful on multi-core machines)", "1"));
	int ctf_section = parser.addSection("CTF options");
	do_ctf = parser.checkOption("--ctf", "Apply CTF correction");
	intact_ctf_first_peak = parser.checkOption("--ctf_intact_first_peak", "Leave CTFs intact until first peak");
	ctf_phase_flipped = parser.checkOption("--ctf_phase_flipped", "Images have been phase flipped");
	only_flip_phases = parser.checkOption("--only_flip_phases", "Do not correct CTF-amplitudes, only flip phases");

	int ewald_section = parser.addSection("Ewald-sphere correction options");
	do_ewald = parser.checkOption("--ewald", "Correct for Ewald-sphere curvature (developmental)");
	mask_diameter  = textToFloat(parser.getOption("--mask_diameter", "Diameter (in A) of mask for Ewald-sphere curvature correction", "-1."));
	width_mask_edge = textToInteger(parser.getOption("--width_mask_edge", "Width (in pixels) of the soft edge on the mask", "3"));
	is_reverse = parser.checkOption("--reverse_curvature", "Try curvature the other way around");
	newbox = textToInteger(parser.getOption("--newbox", "Box size of reconstruction after Ewald sphere correction", "-1"));
	nr_sectors = textToInteger(parser.getOption("--sectors", "Number of sectors for Ewald sphere correction", "2"));
	skip_mask = parser.checkOption("--skip_mask", "Do not apply real space mask during Ewald sphere correction");
	skip_weighting = parser.checkOption("--skip_weighting", "Do not apply weighting during Ewald sphere correction");

	if (verb > 0 && do_ewald && mask_diameter < 0 && !(skip_mask && skip_weighting))
		REPORT_ERROR("To apply Ewald sphere correction (--ewald), you have to specify the mask diameter(--mask_diameter).");

	int helical_section = parser.addSection("Helical options");
	nr_helical_asu = textToInteger(parser.getOption("--nr_helical_asu", "Number of helical asymmetrical units", "1"));
	helical_rise = textToFloat(parser.getOption("--helical_rise", "Helical rise (in Angstroms)", "0."));
	helical_twist = textToFloat(parser.getOption("--helical_twist", "Helical twist (in degrees, + for right-handedness)", "0."));

	int subtomogram_section = parser.addSection("Subtomogram averaging");
	normalised_subtomo = parser.checkOption("--normalised_subtomo", "Have subtomograms been multiplicity normalised? (Default=False)");
	skip_subtomo_correction = parser.checkOption("--skip_subtomo_multi", "Skip subtomo multiplicity correction? (For nomalised subtomos only)");
	ctf3d_squared = !parser.checkOption("--ctf3d_not_squared", "CTF3D files contain sqrt(CTF^2) patterns");

int expert_section = parser.addSection("Expert options");
do_invert_contrast = parser.checkOption("--invert_contrast", "Invert the contrast in the reconstruction");
fn_sub = parser.getOption("--subtract","Subtract projections of this map from the images used for reconstruction", "");
	if (parser.checkOption("--NN", "Use nearest-neighbour instead of linear interpolation before gridding correction"))
		interpolator = NEAREST_NEIGHBOUR;
	else
		interpolator = TRILINEAR;
	blob_radius = textToFloat(parser.getOption("--blob_r", "Radius of blob for gridding interpolation", "1.9"));
	blob_order = textToInteger(parser.getOption("--blob_m", "Order of blob for gridding interpolation", "0"));
	blob_alpha = textToFloat(parser.getOption("--blob_a", "Alpha-value of blob for gridding interpolation", "15"));
	iter = textToInteger(parser.getOption("--iter", "Number of gridding-correction iterations", "10"));
	ref_dim = textToInteger(parser.getOption("--refdim", "Dimension of the reconstruction (2D or 3D)", "3"));
	angular_error = textToFloat(parser.getOption("--angular_error", "Apply random deviations with this standard deviation (in degrees) to each of the 3 Euler angles", "0."));
	shift_error = textToFloat(parser.getOption("--shift_error", "Apply random deviations with this standard deviation (in Angstrom) to each of the 2 translations", "0."));
	do_fom_weighting = parser.checkOption("--fom_weighting", "Weight particles according to their figure-of-merit (_rlnParticleFigureOfMerit)");
	fn_fsc = parser.getOption("--fsc", "FSC-curve for regularized reconstruction", "");
	do_3d_rot = parser.checkOption("--3d_rot", "Perform 3D rotations instead of backprojections from 2D images");
	ctf_dim  = textToInteger(parser.getOption("--reconstruct_ctf", "Perform a 3D reconstruction from 2D CTF-images, with the given size in pixels", "-1"));
	do_reconstruct_ctf2 = parser.checkOption("--ctf2", "Reconstruct CTF^2 and then take the sqrt of that");
	skip_gridding = !parser.checkOption("--dont_skip_gridding", "Perform gridding in the reconstruction (obsolete?)");
	fn_debug = parser.getOption("--debug", "Rootname for debug reconstruction files", "");
	debug_ori_size =  textToInteger(parser.getOption("--debug_ori_size", "Rootname for debug reconstruction files", "1"));
	debug_size =  textToInteger(parser.getOption("--debug_size", "Rootname for debug reconstruction files", "1"));
	fn_noise = parser.getOption("--reconstruct_noise","Reconstruct noise using sigma2 values in this model STAR file", "");
	read_weights = parser.checkOption("--read_weights", "Developmental: read freq. weight files");
	do_debug = parser.checkOption("--write_debug_output", "Write out arrays with data and weight terms prior to reconstruct");
	do_external_reconstruct = parser.checkOption("--external_reconstruct", "Write out BP denominator and numerator for external_reconstruct program");
	verb = textToInteger(parser.getOption("--verb", "Verbosity", "1"));

	// Hidden
	r_min_nn = textToInteger(getParameter(argc, argv, "--r_min_nn", "10"));

	// Check for errors in the command-line option
	if (parser.checkForErrors())
		REPORT_ERROR("Errors encountered on the command line (see above), exiting...");
}

void Reconstructor::usage()
{
	parser.writeUsage(std::cout);
}

void Reconstructor::initialise()
{
	fftw_init_threads();
	fftw_plan_with_nthreads(nr_threads);

	do_reconstruct_ctf = (ctf_dim > 0);
	if (do_reconstruct_ctf)
	{
		do_ctf = false;
		padding_factor = 1.;
	}

	do_ignore_optics = false;
	// Read MetaData file, which should have the image names and their angles!
	if (fn_debug == "")
	{
        ObservationModel::loadSafely(fn_sel, obsModel, DF, "particles", 0, false);
        if (obsModel.isTomoStack2D) REPORT_ERROR("ERROR: reconstruct does not yet work with 2D subtomogram stacks....");

		if (obsModel.opticsMdt.numberOfObjects() == 0)
		{
			do_ignore_optics = true;
			DF.read(fn_sel);
		}
	}

	if (verb > 0 && (subset == 1 || subset == 2) && !DF.containsLabel(EMDL_PARTICLE_RANDOM_SUBSET))
	{
		REPORT_ERROR("The rlnRandomSubset column is missing in the input STAR file.");
	}

	if (verb > 0 && (chosen_class >= 0) && !DF.containsLabel(EMDL_PARTICLE_CLASS))
	{
		REPORT_ERROR("The rlnClassNumber column is missing in the input STAR file.");
	}

	if (fn_cache != "" && !skip_cache_init_in_read_)
	{
		ObservationModel *obsModel_ptr = do_ignore_optics ? NULL : &obsModel;
		CacheInitializer::initializeCache(fn_cache, cache_copy_threads, DF, verb, obsModel_ptr);
	}

	randomize_random_generator();

	if (do_ewald) do_ctf = true;

	// Is this 2D or 3D data?
	data_dim = 2; // Initial default value

	if (fn_noise != "")
	{
		model.read(fn_noise, obsModel.numberOfOpticsGroups());
	}

	// Get dimension of the images
	if (do_reconstruct_ctf)
	{
		output_boxsize = ctf_dim;
	}
	else
	{
		(DF).firstObject();
		DF.getValue(EMDL_IMAGE_NAME, fn_img);

		if (image_path != "")
		{
			fn_img = image_path + "/" + fn_img.substr(fn_img.find_last_of("/")+1);
		}

		Image<RFLOAT> img0;
		{
			FileName fn_stack;
			long int imgno_dummy;
			fn_img.decompose(imgno_dummy, fn_stack);
			if (!exists(fn_stack))
				REPORT_ERROR("First image stack not found: " + fn_stack);
		}
		img0.read(fn_img, false);
		output_boxsize=(int)XSIZE(img0());
		// When doing Ewald-curvature correction or when having optics groups: allow reconstructing smaller box than the input images (which should have large boxes!!)
		if ((do_ewald || !do_ignore_optics) && newbox > 0)
		{
			output_boxsize = newbox;
		}

		if (do_3d_rot)
			data_dim = 3;
		else // If not specifically provided, we autodetect it
		{
			if (do_ignore_optics)
			{
				data_dim = img0().getDim();
				std::cout << " + Taking data dimensions from the first image: " << data_dim << std::endl;
			}
			else
			{
				obsModel.opticsMdt.getValue(EMDL_IMAGE_DIMENSIONALITY, data_dim, 0);
				std::cout << " + Taking data dimensions from the first optics group: " << data_dim << std::endl;
			}
		}
	}

	if (angpix < 0.)
	{
		if (do_ignore_optics)
		{
			if (DF.containsLabel(EMDL_CTF_MAGNIFICATION) && DF.containsLabel(EMDL_CTF_DETECTOR_PIXEL_SIZE))
			{
				RFLOAT mag, dstep;
				DF.getValue(EMDL_CTF_MAGNIFICATION, mag);
				DF.getValue(EMDL_CTF_DETECTOR_PIXEL_SIZE, dstep);
				angpix = 10000. * dstep / mag;
				if (verb > 0)
					std::cout << " + Using pixel size calculated from magnification and detector pixel size in the input STAR file: " << angpix << std::endl;
			}
			else
			{
				REPORT_ERROR("ERROR: cannot find pixel size in input STAR file, provide it using --angpix");
			}
		}
		else
		{
			angpix = obsModel.getPixelSize(0);
			std::cout << " + Taking angpix from the first optics group: " << angpix << std::endl;
		}
	}

	if (maxres < 0.)
		r_max = -1;
	else
		r_max = CEIL(output_boxsize * angpix / maxres);

	if (spatial_frequency_mode == SPATIAL_FREQUENCY_MODE_S2)
	{
		if (data_dim != 2)
		{
			REPORT_ERROR("ERROR: --spatial_frequency_mode s2 is currently only supported for 2D particle images.");
		}

		if (do_reconstruct_ctf)
		{
			REPORT_ERROR("ERROR: --spatial_frequency_mode s2 is not yet supported for --reconstruct_ctf.");
		}

		if (fn_sub != "")
		{
			REPORT_ERROR("ERROR: --spatial_frequency_mode s2 is not yet supported together with --subtract.");
		}

		if (fn_noise != "")
		{
			REPORT_ERROR("ERROR: --spatial_frequency_mode s2 is not yet supported for --reconstruct_noise.");
		}

		if (read_weights)
		{
			REPORT_ERROR("ERROR: --spatial_frequency_mode s2 is not yet supported together with --read_weights.");
		}

#ifndef RELION_USE_FINUFFT
		REPORT_ERROR("ERROR: --spatial_frequency_mode s2 requires a build configured with -DRELION_USE_FINUFFT=ON.");
#endif
	}

	// Print defocus statistics and cache s2 grids for CTF oversampling
	if (spatial_frequency_mode == SPATIAL_FREQUENCY_MODE_S2 && s2_ctf_oversampling_min > 0)
	{
		if (!do_ignore_optics && obsModel.opticsMdt.containsLabel(EMDL_CTF_VOLTAGE))
		{
			obsModel.opticsMdt.getValue(EMDL_CTF_VOLTAGE, voltage_for_s2_, 0);
		}

		if (voltage_for_s2_ > 0.0 && DF.containsLabel(EMDL_CTF_DEFOCUSU) && verb > 0)
		{
			std::vector<RFLOAT> defoci;
			DF.firstObject();
			for (long int p = 0; p < DF.numberOfObjects(); p++)
			{
				RFLOAT du, dv;
				DF.getValue(EMDL_CTF_DEFOCUSU, du);
				DF.getValue(EMDL_CTF_DEFOCUSV, dv);
				if (du > 0.0 && dv > 0.0)
					defoci.push_back(std::max(du, dv));
				DF.nextObject();
			}
			DF.firstObject();

 if (!defoci.empty())
 {
 std::sort(defoci.begin(), defoci.end());
 RFLOAT dmin = defoci.front();
 RFLOAT dmax = defoci.back();
 RFLOAT dmean = 0.0;
 for (auto d : defoci) dmean += d;
 dmean /= defoci.size();
 RFLOAT dmedian = defoci[defoci.size() / 2];
			printS2CtfOversamplingStats(output_boxsize / 2, angpix,
				dmin, dmean, dmedian, dmax,
				s2_ctf_oversampling_min, voltage_for_s2_);

			if (do_s2_grid_diagnostic)
			{
				FileName diag_prefix = fn_out.withoutExtension() + "_s2_grid_diag";
				writeS2GridDiagnostic(output_boxsize, angpix, voltage_for_s2_,
					dmin, dmean, dmedian, dmax,
					s2_ctf_oversampling_min, diag_prefix);
			}
		}
	}
	}

}

void Reconstructor::run()
{
	if (fn_debug != "")
	{
		readDebugArrays();
		reconstruct();
		return;
	}

	initialise();
	if (do_half1 || do_half2 || do_alldata)
	{
		FileName fn_out_orig = fn_out;
		// Half-1
		if (do_half1)
		{
			subset = 1;
			fn_out = fn_out_orig.insertBeforeExtension("_half1");
			MetaDataTable DF_restore = DF;
			DF = selectRandomSubset(DF, random_subset_size, 1, random_subset_seed, verb);
			if (verb > 0)
				std::cout << "=== Reconstructing half-1 (" << DF.numberOfObjects() << " particles) ===" << std::endl;
			backproject();
			reconstruct();
			DF = DF_restore;
		}
		// Half-2
		if (do_half2)
		{
			subset = 2;
			fn_out = fn_out_orig.insertBeforeExtension("_half2");
			MetaDataTable DF_restore = DF;
			DF = selectRandomSubset(DF, random_subset_size, 2, random_subset_seed, verb);
			if (verb > 0)
				std::cout << "=== Reconstructing half-2 (" << DF.numberOfObjects() << " particles) ===" << std::endl;
			backproject();
			reconstruct();
			DF = DF_restore;
		}
		// All data
		if (do_alldata)
		{
			subset = -1;
			fn_out = fn_out_orig;
			MetaDataTable DF_restore = DF;
			DF = selectRandomSubset(DF, random_subset_size, -1, random_subset_seed, verb);
			if (verb > 0)
				std::cout << "=== Reconstructing full map (" << DF.numberOfObjects() << " particles) ===" << std::endl;
			backproject();
			reconstruct();
			DF = DF_restore;
		}
		fn_out = fn_out_orig;
	}
	else
	{
		backproject();
		reconstruct();
	}
}

MetaDataTable Reconstructor::selectRandomSubset(const MetaDataTable &DF_in,
                                                 long int sample_size,
                                                 int random_subset_filter,
                                                 int seed, int verb) const
{
	// Collect candidate indices (filtering by random_subset if needed)
	std::vector<long int> candidates;
	for (long int p = 0; p < DF_in.numberOfObjects(); p++)
	{
		if (random_subset_filter >= 1 && random_subset_filter <= 2)
		{
			int rs = 0;
			DF_in.getValue(EMDL_PARTICLE_RANDOM_SUBSET, rs, p);
			if (rs != random_subset_filter)
				continue;
		}
		candidates.push_back(p);
	}

	long int n = (long int)candidates.size();
	if (sample_size <= 0 || sample_size >= n)
	{
		MetaDataTable out;
		out.reserve(n);
		for (long int i = 0; i < n; i++)
			out.addObject(DF_in.getObject(candidates[i]));
		return out;
	}

	// Fisher-Yates partial shuffle to select sample_size random indices
	std::mt19937 gen(static_cast<unsigned int>(seed));
	for (long int i = 0; i < sample_size; i++)
	{
		long int j = i + (long int)(gen() % (n - i));
		std::swap(candidates[i], candidates[j]);
	}

	MetaDataTable out;
	out.reserve(sample_size);
	for (long int i = 0; i < sample_size; i++)
		out.addObject(DF_in.getObject(candidates[i]));

	if (verb > 0)
		std::cout << "  Selected random subset of " << sample_size
		          << " particles (from " << n << " candidates)." << std::endl;

	return out;
}

void Reconstructor::readDebugArrays()
{
	if (verb > 0)
		std::cout << " + Reading in the debug arrays ... " << std::endl;

	// We first read the image to set the data_dim automatically from backprojector data
	Image<RFLOAT> It;
	It.read(fn_debug+"_data_real.mrc");
	data_dim = It().getDim();

	backprojector = BackProjector(debug_ori_size, 3, fn_sym, interpolator, padding_factor, r_min_nn, blob_order, blob_radius, blob_alpha, data_dim, skip_gridding);

	backprojector.initialiseDataAndWeight(debug_size);
	if (verb > 0)
	{
		std::cout << " Size of data array: " ;
		backprojector.data.printShape();
		std::cout << " Size of weight array: " ;
		backprojector.weight.printShape();
	}

	It().setXmippOrigin();
	It().xinit=0;

	if (verb > 0)
	{
		std::cout << " Size of reconstruction: " ;
		It().printShape();
	}
	FOR_ALL_ELEMENTS_IN_ARRAY3D(It())
	{
		A3D_ELEM(backprojector.data, k, i, j).real = A3D_ELEM(It(), k, i, j);
	}
	It.read(fn_debug+"_data_imag.mrc");
	It().setXmippOrigin();
	It().xinit=0;
	FOR_ALL_ELEMENTS_IN_ARRAY3D(It())
	{
		A3D_ELEM(backprojector.data, k, i, j).imag = A3D_ELEM(It(), k, i, j);
	}
	It.read(fn_debug+"_weight.mrc");
	It().setXmippOrigin();
	It().xinit=0;
	FOR_ALL_ELEMENTS_IN_ARRAY3D(It())
	{
		A3D_ELEM(backprojector.weight, k, i, j) = A3D_ELEM(It(), k, i, j);
	}
	output_boxsize = debug_ori_size;
}

static long int packS2GridKey(int box_size, int int_step)
{
    return ((long int)box_size << 20) | (int_step + 1);
}

void Reconstructor::ensureS2GridCached(int myBoxSize, RFLOAT myPixelSize, RFLOAT s2_step)
{
    int int_step = (s2_step > 0) ? ROUND(s2_step) : -1;
    long int key = packS2GridKey(myBoxSize, int_step);
    {
        std::lock_guard<std::mutex> lock(s2_grid_mutex_);
        auto it = s2_grid_cache_.find(key);
        if (it != s2_grid_cache_.end())
            return;
    }

	SpatialFrequencyGrid2D grid = makeAdaptiveS2HybridGrid2D(myBoxSize, nullptr, s2_step);

	computeBilinearCoeffs(grid);

    {
        std::lock_guard<std::mutex> lock(s2_grid_mutex_);
        s2_grid_cache_.emplace(key, std::move(grid));
    }
}

const SpatialFrequencyGrid2D& Reconstructor::getS2Grid(long int cache_key)
{
    std::lock_guard<std::mutex> lock(s2_grid_mutex_);
    auto it = s2_grid_cache_.find(cache_key);
    if (it == s2_grid_cache_.end())
        REPORT_ERROR("Reconstructor::getS2Grid: grid not found for key " + std::to_string(cache_key));
    return it->second;
}

void Reconstructor::backproject(int rank, int size)
{
	if (fn_sub != "")
	{
		projector = Projector(output_boxsize, interpolator, padding_factor, r_min_nn);
		Image<RFLOAT> sub;
		sub.read(fn_sub);
		MultidimArray<RFLOAT> dummy;
		projector.computeFourierTransformMap(sub(), dummy, 2 * r_max);
	}

	backprojector = BackProjector(output_boxsize, ref_dim, fn_sym, interpolator,
					padding_factor, r_min_nn, blob_order,
					blob_radius, blob_alpha, data_dim, skip_gridding);
	backprojector.initZeros(2 * r_max);

	long int nr_parts = DF.numberOfObjects();
	long int barstep = XMIPP_MAX(1, nr_parts/(size*120));
	if (verb > 0)
	{
		std::cout << " + Back-projecting all images ..." << std::endl;
		time_config();
		init_progress_bar(nr_parts);
	}

	prefetcher_.reset();
	const bool use_prefetch = do_prefetch && !do_reconstruct_ctf && fn_noise == "";
	if (use_prefetch)
	{
		prefetcher_.reset(new AsyncReconstructPrefetcher(&DF, rank, size, subset, chosen_class, nr_threads + 2));
		prefetcher_->start(nr_parts);
	}

	// Collect this rank's particle indices, then parallelize over them with --j threads.
	std::vector<long int> my_parts;
	my_parts.reserve(nr_parts / size + 1);
	for (long int ipart = 0; ipart < nr_parts; ipart++)
	{
		if (ipart % size == rank)
			my_parts.push_back(ipart);
	}

	if (nr_threads > 1)
	{
		// Inside the OMP parallel loop, each thread uses single-threaded FFTW/FINUFFT.
		// Parallelism comes from OMP at the particle level.
		fftw_plan_with_nthreads(1);
		setFinufftThreadCount(1);
	}

#pragma omp parallel for num_threads(nr_threads) schedule(dynamic, 1)
	for (long int idx = 0; idx < (long int)my_parts.size(); idx++)
	{
		backprojectOneParticle(my_parts[idx]);
	}

	if (nr_threads > 1)
	{
		fftw_plan_with_nthreads(nr_threads);
	}

	if (verb > 0)
	{
		for (long int ipart : my_parts)
		{
			if (ipart % barstep == 0)
				progress_bar(ipart);
		}
		progress_bar(nr_parts);
	}

        if (prefetcher_)
        {
                prefetcher_->stop();
                prefetcher_.reset();
        }

        if (verb > 0)
                progress_bar(nr_parts);

	if (spatial_frequency_mode == SPATIAL_FREQUENCY_MODE_S2)
	{
#ifdef S2_PROFILE
		tls_s2_profile.print(rank);
#endif
	}
}

void Reconstructor::backprojectOneParticle(long int p)
{
	RFLOAT rot, tilt, psi, fom, r_ewald_sphere;
	Matrix2D<RFLOAT> A3D;
	MultidimArray<RFLOAT> Fctf, FstMulti;
	Matrix1D<RFLOAT> trans(3); // Always 3 elements so ZZ(trans) is valid for 2D data too
	FourierTransformer transformer;

	bool do_subtomo_correction = false;

	int randSubset = 0, classid = 0;
	DF.getValue(EMDL_PARTICLE_RANDOM_SUBSET, randSubset, p);
	DF.getValue(EMDL_PARTICLE_CLASS, classid, p);

	if (subset >= 1 && subset <= 2 && randSubset != subset)
		return;

	if (chosen_class >= 0 && chosen_class != classid)
		return;

	// Rotations
	if (ref_dim == 2)
	{
		rot = tilt = 0.;
	}
	else
	{
		DF.getValue(EMDL_ORIENT_ROT, rot, p);
		DF.getValue(EMDL_ORIENT_TILT, tilt, p);
	}

	psi = 0.;
	DF.getValue(EMDL_ORIENT_PSI, psi, p);

	if (angular_error > 0.)
	{
		rot += rnd_gaus(0., angular_error);
		tilt += rnd_gaus(0., angular_error);
		psi += rnd_gaus(0., angular_error);
		//std::cout << rnd_gaus(0., angular_error) << std::endl;
	}

	Euler_angles2matrix(rot, tilt, psi, A3D);

	// If we are considering Ewald sphere curvature, the mag. matrix
	// has to be provided to the backprojector explicitly
	// (to avoid creating an Ewald ellipsoid)
	int opticsGroup=-1;
	int myBoxSize = output_boxsize; // Without optics groups, the output box size is always the same as the one from the input images
	RFLOAT myPixelSize = angpix; // Without optics groups, the pixel size is always the same as the one from the input images
	bool ctf_premultiplied = false;
	if (!do_ignore_optics)
	{
		opticsGroup = obsModel.getOpticsGroup(DF, p);
		myBoxSize = obsModel.getBoxSize(opticsGroup);
		myPixelSize = obsModel.getPixelSize(opticsGroup);
		ctf_premultiplied = obsModel.getCtfPremultiplied(opticsGroup);
		if (do_ewald && ctf_premultiplied)
			REPORT_ERROR("We cannot perform Ewald sphere correction on CTF premultiplied particles.");
		Matrix2D<RFLOAT> magMat;
		if (!do_ewald)
		{
			A3D = obsModel.applyAnisoMag(A3D, opticsGroup);
		}
		A3D = obsModel.applyScaleDifference(A3D, opticsGroup, output_boxsize, angpix);
	}

	// Translations (either through phase-shifts or in real space
	trans.initZeros();
	DF.getValue( EMDL_ORIENT_ORIGIN_X_ANGSTROM, XX(trans), p);
	DF.getValue( EMDL_ORIENT_ORIGIN_Y_ANGSTROM, YY(trans), p);

	if (shift_error > 0.)
	{
		XX(trans) += rnd_gaus(0., shift_error);
		YY(trans) += rnd_gaus(0., shift_error);
	}

	if (data_dim == 3)
	{
		trans.resize(3);
		DF.getValue( EMDL_ORIENT_ORIGIN_Z_ANGSTROM, ZZ(trans), p);

		if (shift_error > 0.)
		{
			ZZ(trans) += rnd_gaus(0., shift_error);
		}
	}

	// As of v3.1, shifts are in Angstroms in the STAR files, convert back to pixels here
	trans/= myPixelSize;

	if (do_fom_weighting)
	{
		DF.getValue( EMDL_PARTICLE_FOM, fom, p);
	}

	// Use either selfTranslate OR shiftImageInFourierTransform!!
	//selfTranslate(img(), trans, WRAP);

MultidimArray<Complex> Fsub, F2D, F2DP, F2DQ;
FileName fn_img;
Image<RFLOAT> img;
std::vector<Complex> nonuniformSamples;
std::vector<RFLOAT> nonuniformSampleWeight;
std::vector<Complex> nonuniformSamplesP, nonuniformSamplesQ;
std::vector<RFLOAT> gamma_offsets;
std::vector<RFLOAT> ctf_values;
bool use_nonuniform_s2 = (spatial_frequency_mode == SPATIAL_FREQUENCY_MODE_S2);
#ifdef S2_PROFILE
	auto t_s2_read = Clock::now();
	auto t_s2_fft = Clock::now();
	auto t_s2_ctf = Clock::now();
	auto t_s2_ewald = Clock::now();
	(void)t_s2_read; (void)t_s2_fft; (void)t_s2_ctf; (void)t_s2_ewald;
#endif

        if (!do_reconstruct_ctf && fn_noise == "")
        {
                if (prefetcher_)
                {
                        if (!prefetcher_->waitAndPop(p, img))
                        {
                                REPORT_ERROR("Unexpected prefetch underflow while reconstructing particles.");
                        }
                }
                else
                {
DF.getValue(EMDL_IMAGE_NAME, fn_img, p);
#ifdef S2_PROFILE
			t_s2_read = Clock::now();
#endif
			img.read(fn_img);
		}
		img().setXmippOrigin();
#ifdef S2_PROFILE
		t_s2_fft = Clock::now();
#endif
		transformer.FourierTransform(img(), F2D);
		CenterFFTbySign(F2D);
#ifdef S2_PROFILE
		if (use_nonuniform_s2) t_s2_ctf = Clock::now();
#endif

		if (ABS(XX(trans)) > 0. || ABS(YY(trans)) > 0. || ABS(ZZ(trans)) > 0. ) // ZZ(trans) is 0 in case data_dim=2
		{
			shiftImageInFourierTransform(F2D, F2D, XSIZE(img()), XX(trans), YY(trans), ZZ(trans));
		}
	}
	else
	{
		if (data_dim == 3) F2D.resize(myBoxSize, myBoxSize, myBoxSize / 2 + 1);
		else F2D.resize(myBoxSize, myBoxSize / 2 + 1);
	}

	if (fn_noise != "")
	{

		int optics_group = 0;
		DF.getValue(EMDL_IMAGE_OPTICS_GROUP, optics_group);

		// Make coloured noise image
		FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(F2D)
		{
			int ires = ROUND(sqrt((RFLOAT)(kp*kp + ip*ip + jp*jp)));
			ires = XMIPP_MIN(ires, myBoxSize/2); // at freqs higher than Nyquist: use last sigma2 value

			RFLOAT sigma = sqrt(DIRECT_A1D_ELEM(model.sigma2_noise[optics_group], ires));
			DIRECT_A3D_ELEM(F2D, k, i, j).real += rnd_gaus(0., sigma);
			DIRECT_A3D_ELEM(F2D, k, i, j).imag += rnd_gaus(0., sigma);
		}
	}

	Fctf.resize(F2D);
	Fctf.initConstant(1.);

	// Apply CTF if necessary
	if (do_ctf || do_reconstruct_ctf)
	{

		// Also allow 3D CTF correction here
		if (data_dim == 3)
		{
			Image<RFLOAT> Ictf;
			FileName fn_ctf;
			if (!DF.getValue(EMDL_CTF_IMAGE, fn_ctf, p))
				REPORT_ERROR("ERROR: cannot find rlnCtfImage for 3D CTF correction!");
			Ictf.read(fn_ctf);

			// If there is a redundant half, get rid of it
			if (XSIZE(Ictf()) == YSIZE(Ictf()))
			{
				Ictf().setXmippOrigin();
				FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fctf)
				{
					// Use negative kp,ip and jp indices, because the origin in the ctf_img lies half a pixel to the right of the actual center....
					DIRECT_A3D_ELEM(Fctf, k, i, j) = A3D_ELEM(Ictf(), -kp, -ip, -jp);
				}
			}
			// otherwise, just window the CTF to the current resolution
			else if (XSIZE(Ictf()) == YSIZE(Ictf()) / 2 + 1)
			{
				// If subtomos are not normalised MULTI is included and we don't need to read it
				if (ZSIZE(Ictf()) == YSIZE(Ictf()))
				{
					windowFourierTransform(Ictf(), Fctf, YSIZE(Fctf));
				}
				else if (ZSIZE(Ictf()) == YSIZE(Ictf())*2) // Subtomo multiplicity weights included in the CTF file
				{
					MultidimArray<RFLOAT> &Mctf = Ictf();
					long int max_r2 = (XSIZE(Mctf) - 1) * (XSIZE(Mctf) - 1);

					if (!normalised_subtomo || skip_subtomo_correction)
					{
						FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fctf)
						{
							// Make sure windowed FT has nothing in the corners, otherwise we end up with an asymmetric FT!
							if (kp * kp + ip * ip + jp * jp <= max_r2)
							{
								FFTW_ELEM(Fctf, kp, ip, jp) = DIRECT_A3D_ELEM(Mctf, ((kp < 0) ? (kp + YSIZE(Mctf)) : (kp)), \
								((ip < 0) ? (ip + YSIZE(Mctf)) : (ip)), jp);
							}
							else
								FFTW_ELEM(Fctf, kp, ip, jp) = 0.;
						}
					}
					else
					{
						FstMulti.resize(F2D);
						do_subtomo_correction = true;
						FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fctf)
						{
							// Make sure windowed FT has nothing in the corners, otherwise we end up with an asymmetric FT!
							if (kp * kp + ip * ip + jp * jp <= max_r2)
							{
								FFTW_ELEM(Fctf, kp, ip, jp) = DIRECT_A3D_ELEM(Mctf, ((kp < 0) ? (kp + YSIZE(Mctf)): (kp)), \
								((ip < 0) ? (ip + YSIZE(Mctf)) : (ip)), jp);
								FFTW_ELEM(FstMulti, kp, ip, jp) = DIRECT_A3D_ELEM(Mctf, ((kp < 0) ? (kp + ZSIZE(Mctf)) : (kp + YSIZE(Mctf))), \
								((ip < 0) ? (ip + YSIZE(Mctf)) : (ip)), jp);
							}
							else
							{
								FFTW_ELEM(Fctf, kp, ip, jp) = 0.;
								FFTW_ELEM(FstMulti, kp, ip, jp) = 0.;
							}
						}
					}
				}
			}
			// if dimensions are neither cubical nor FFTW, stop
			else
			{
				REPORT_ERROR("3D CTF volume must be either cubical or adhere to FFTW format!");
			}
		}
        else
        {
                CTF ctf;
                if (do_ignore_optics)
                {
                        ctf.read(DF, DF, p);
                }
                else
                {
                        ctf.readByGroup(DF, &obsModel, p);
                }

                ctf.getFftwImage(Fctf, myBoxSize, myBoxSize, myPixelSize,
                        ctf_phase_flipped, only_flip_phases,
                        intact_ctf_first_peak, true);

                if (!do_ignore_optics)
                {
                        obsModel.demodulatePhase(DF, p, F2D);
                        obsModel.divideByMtf(DF, p, F2D);
                }

                // Ewald-sphere curvature correction
if (do_ewald)
	{
#ifdef S2_PROFILE
		if (use_nonuniform_s2) t_s2_ewald = Clock::now();
#endif
		applyCTFPandCTFQ(F2D, ctf, transformer, F2DP, F2DQ, skip_mask);

		if (!skip_weighting)
		{
			// Also calculate W, store again in Fctf
			ctf.applyWeightEwaldSphereCurvature_noAniso(Fctf, myBoxSize, myBoxSize, myPixelSize, mask_diameter);
		}

		// Also calculate the radius of the Ewald sphere (in pixels)
		r_ewald_sphere = myBoxSize * myPixelSize / ctf.lambda;
	}
	else if (use_nonuniform_s2)
	{
#ifdef S2_PROFILE
		t_s2_ewald = Clock::now();
#endif
	}
        }
        }
else if (use_nonuniform_s2)
{
#ifdef S2_PROFILE
	t_s2_ctf = Clock::now();
	t_s2_ewald = Clock::now();
#endif
}

        // Subtract reference projection
if (use_nonuniform_s2)
	{
#ifdef S2_PROFILE
		auto t_s2_total = Clock::now();
		S2Profile prof;
		prof.is_ewald = do_ewald;
		prof.boxsize = myBoxSize;
#endif

                RFLOAT s2_step = -1.0;
    if (s2_uniform_fill_)
    {
        s2_step = -2.0; // sentinel for uniform ns=2 fill
    }
    else if (voltage_for_s2_ > 0.0 && s2_ctf_oversampling_min > 0 && DF.containsLabel(EMDL_CTF_DEFOCUSU))
    {
        RFLOAT du, dv;
        DF.getValue(EMDL_CTF_DEFOCUSU, du, p);
        DF.getValue(EMDL_CTF_DEFOCUSV, dv, p);
        RFLOAT defocus = std::max(du, dv);
        if (defocus > 0.0)
        {
            int half = myBoxSize / 2;
            s2_step = computeS2StepForCtfOversampling(half, myPixelSize,
                defocus, s2_ctf_oversampling_min, voltage_for_s2_);
        }
    }
                ensureS2GridCached(myBoxSize, myPixelSize, s2_step);
                int int_step = (s2_step > 0) ? ROUND(s2_step) : -1;
                long int grid_key = packS2GridKey(myBoxSize, int_step);
const SpatialFrequencyGrid2D& s2_grid = getS2Grid(grid_key);
#ifdef S2_PROFILE
		prof.nk = (int)s2_grid.sample_x.size();
#endif

#ifdef RELION_USE_FINUFFT
#ifdef S2_PROFILE
		auto t0 = Clock::now();
#endif
		evaluateNonuniformFourierSamples2D(img(), s2_grid, nonuniformSamples);
#ifdef S2_PROFILE
		prof.ns_finufft = (Clock::now() - t0).count();
#endif
#endif

#ifdef S2_PROFILE
		auto t0_shift_jac = Clock::now();
#endif
		// FINUFFT was evaluated on the unshifted real-space image, so apply
		// the image translation and the s^2 Jacobian afterwards.
		// Fused single pass: Fourier shift (if nonzero) + Jacobian weight.
		const bool hasShift = (ABS(XX(trans)) >= XMIPP_EQUAL_ACCURACY
			|| ABS(YY(trans)) >= XMIPP_EQUAL_ACCURACY);
		nonuniformSampleWeight = s2_grid.sample_weight;
		for (long int idx = 0; idx < (long int)nonuniformSamples.size(); idx++)
		{
			if (hasShift)
			{
				const RFLOAT phase = (RFLOAT)(-TWO_PI
					* (s2_grid.sample_x[idx] * XX(trans)
					+ s2_grid.sample_y[idx] * YY(trans)) / myBoxSize);
				const Complex shift(cos(phase), sin(phase));
				nonuniformSamples[idx] *= shift;
			}
			nonuniformSamples[idx] *= nonuniformSampleWeight[idx];
		}
#ifdef S2_PROFILE
		prof.ns_shift_jac = (Clock::now() - t0_shift_jac).count();
#endif

{
#ifdef S2_PROFILE
			auto t0 = Clock::now();
#endif
			for (long int idx = 0; idx < (long int)nonuniformSamples.size(); idx++)
			{
				if (ABS(s2_grid.sample_x[idx]) < XMIPP_EQUAL_ACCURACY
					&& ABS(s2_grid.sample_y[idx]) < XMIPP_EQUAL_ACCURACY)
				{
					nonuniformSamples[idx] = Complex(0.0, 0.0);
					break;
				}
			}
#ifdef S2_PROFILE
			prof.ns_dc_zero = (Clock::now() - t0).count();
#endif
		}

        if (do_ctf)
        {
if (do_ewald)
		{
#ifdef S2_PROFILE
			auto t0_bilin = Clock::now();
#endif
			std::vector<Complex> nonuniformSamplesP(nonuniformSamples.size());
                        std::vector<Complex> nonuniformSamplesQ(nonuniformSamples.size());

                        for (long int idx = 0; idx < (long int)nonuniformSamples.size(); idx++)
                        {
                                const RFLOAT sx = s2_grid.sample_x[idx];
                                const RFLOAT sy = s2_grid.sample_y[idx];
                                const RFLOAT jac = s2_grid.sample_weight[idx];

                                nonuniformSamplesP[idx] = sampleComplexFromFftwHalfBilinear(F2DP, sx, sy, myBoxSize) * jac;
                                nonuniformSamplesQ[idx] = sampleComplexFromFftwHalfBilinear(F2DQ, sx, sy, myBoxSize) * jac;
                                nonuniformSampleWeight[idx] = sampleRealFromFftwHalfBilinear(Fctf, sx, sy, myBoxSize) * jac;
                        }

                        if (do_fom_weighting)
                        {
                                for (long int idx = 0; idx < (long int)nonuniformSamples.size(); idx++)
                                {
                                        nonuniformSamplesP[idx] *= fom;
                                        nonuniformSamplesQ[idx] *= fom;
                                        nonuniformSampleWeight[idx] *= fom;
                                }
}
#ifdef S2_PROFILE
			prof.ns_bilinear = (Clock::now() - t0_bilin).count();
#endif

                        Matrix2D<RFLOAT> magMat;
                        if (!do_ignore_optics && obsModel.hasMagMatrices)
                        {
                                magMat = obsModel.getMagMatrix(obsModel.getOpticsGroup(DF, p));
                        }
                        else
                        {
                                magMat = Matrix2D<RFLOAT>(2,2);
                                magMat.initIdentity();
                        }

#ifdef S2_PROFILE
			auto t0_bp = Clock::now();
#endif
			#pragma omp critical(backproject)
			{
				backprojector.backprojectNonuniform2Dto3D(nonuniformSamplesP, s2_grid.sample_x,
					s2_grid.sample_y, A3D,
					&nonuniformSampleWeight,
					r_ewald_sphere, true, &magMat);
				backprojector.backprojectNonuniform2Dto3D(nonuniformSamplesQ, s2_grid.sample_x,
					s2_grid.sample_y, A3D,
					&nonuniformSampleWeight,
					r_ewald_sphere, false, &magMat);
			}
#ifdef S2_PROFILE
			prof.ns_backproject = (Clock::now() - t0_bp).count();
#endif

#ifdef S2_PROFILE
			prof.count = 1;
			prof.ns_total = (Clock::now() - t_s2_total).count();
			prof.ns_read_img = (t_s2_fft - t_s2_read).count();
			prof.ns_fft = (t_s2_ctf - t_s2_fft).count();
			prof.ns_ctf_precomp = (t_s2_ewald - t_s2_ctf).count();
			prof.ns_ewald_precomp = (t_s2_total - t_s2_ewald).count();
			tls_s2_profile.accumulate(prof);
#endif

	return;
                }

                // Apply the analytical 2D CTF directly at the actual s2 sample
                // positions to both the data samples and their weights.
#ifdef S2_PROFILE
		auto t0_ctf = Clock::now();
#endif
		CTF ctf;
                std::vector<RFLOAT> gamma_offsets;
                const std::vector<RFLOAT>* gamma_offsets_ptr = NULL;

                if (do_ignore_optics)
                {
                        ctf.read(DF, DF, p);
                }
                else
                {
                        ctf.readByGroup(DF, &obsModel, p);

                        const int optics_group = obsModel.getOpticsGroup(DF, p);

                        // Match the s-path optics handling in sample space for nonuniform s2 points.
                        applyObsModelPhaseCorrectionToSamples(nonuniformSamples, s2_grid, obsModel,
                                optics_group, myBoxSize);
                        applyObsModelMtfCorrectionToSamples(nonuniformSamples, s2_grid, obsModel,
                                optics_group, myBoxSize);

                        if (obsModel.hasEvenZernike)
                        {
                                evaluateSymmetricAberrationAtSamplePositions(gamma_offsets, s2_grid, obsModel, optics_group, myBoxSize);
                                gamma_offsets_ptr = &gamma_offsets;
                        }
                }

                // Fused single-pass: CTF evaluation at each exact s2 position,
                // CTF application to sample/weight, and FOM weighting.
                {
                        const RFLOAT xs = (RFLOAT)myBoxSize * myPixelSize;
                        for (long int idx = 0; idx < (long int)nonuniformSamples.size(); idx++)
                        {
                                const RFLOAT x = s2_grid.sample_x[idx] / xs;
                                const RFLOAT y = s2_grid.sample_y[idx] / xs;
                                const RFLOAT gamma_offset = (gamma_offsets_ptr != NULL)
                                        ? (*gamma_offsets_ptr)[idx] : 0.0;

                                const RFLOAT ctf_value = ctf.getCTF(x, y,
                                        ctf_phase_flipped, only_flip_phases,
                                        intact_ctf_first_peak, true, gamma_offset);

                                if (!ctf_premultiplied)
                                {
                                        nonuniformSamples[idx] *= ctf_value;
                                }
                                nonuniformSampleWeight[idx] *= ctf_value * ctf_value;

                                if (do_fom_weighting)
                                {
                                        nonuniformSamples[idx] *= fom;
                                        nonuniformSampleWeight[idx] *= fom;
                                }
                        }
}
#ifdef S2_PROFILE
		prof.ns_ctf_analytical = (Clock::now() - t0_ctf).count();
#endif
	}

	{
#ifdef S2_PROFILE
		auto t0_bp = Clock::now();
#endif
		#pragma omp critical(backproject)
		backprojector.backprojectNonuniform2Dto3D(nonuniformSamples, s2_grid.sample_x,
			s2_grid.sample_y, A3D,
			&nonuniformSampleWeight);
#ifdef S2_PROFILE
		prof.ns_backproject = (Clock::now() - t0_bp).count();
#endif
	}

#ifdef S2_PROFILE
		prof.count = 1;
		prof.ns_total = (Clock::now() - t_s2_total).count();
		prof.ns_read_img = (t_s2_fft - t_s2_read).count();
		prof.ns_fft = (t_s2_ctf - t_s2_fft).count();
		prof.ns_ctf_precomp = (t_s2_ewald - t_s2_ctf).count();
		prof.ns_ewald_precomp = (t_s2_total - t_s2_ewald).count();
		tls_s2_profile.accumulate(prof);
#endif
}


        else if (fn_sub != "")
	{
		Fsub.resize(F2D);
		projector.get2DFourierTransform(Fsub, A3D);

		// Apply CTF if necessary
		if (do_ctf)
		{
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fsub)
			{
				DIRECT_MULTIDIM_ELEM(Fsub, n) *= DIRECT_MULTIDIM_ELEM(Fctf, n);
			}
		}

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fsub)
		{
			DIRECT_MULTIDIM_ELEM(F2D, n) -= DIRECT_MULTIDIM_ELEM(Fsub, n);
		}
	// Back-project difference image
#pragma omp critical(backproject)
	backprojector.set2DFourierTransform(F2D, A3D);
	}
	else
	{
		if (do_reconstruct_ctf)
		{
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(F2D)
			{
				DIRECT_MULTIDIM_ELEM(F2D, n)  = DIRECT_MULTIDIM_ELEM(Fctf, n);
				if (do_reconstruct_ctf2)
					DIRECT_MULTIDIM_ELEM(F2D, n) *= DIRECT_MULTIDIM_ELEM(Fctf, n);
				DIRECT_MULTIDIM_ELEM(Fctf, n) = 1.;
			}
		}

		if (read_weights)
		{
			std::string name, fullName;

			DF.getValue(EMDL_IMAGE_NAME, fullName, 0);
			name = fullName.substr(fullName.find("@")+1);

			if (image_path != "")
			{
				name = image_path + "/" + name.substr(name.find_last_of("/")+1);
			}

			std::string wghName = name;
			wghName = wghName.substr(0, wghName.find_last_of('.')) + "_weight.mrc";

			Image<RFLOAT> wgh;
			wgh.read(wghName);

			if (   Fctf.ndim != wgh().ndim
			    || Fctf.zdim != wgh().zdim
			    || Fctf.ydim != wgh().ydim
			    || Fctf.xdim != wgh().xdim)
			{
				REPORT_ERROR(wghName + " and " + name + " are of unequal size.\n");
			}

			for (long int n = 0; n < Fctf.ndim; n++)
			for (long int z = 0; z < Fctf.zdim; z++)
			for (long int y = 0; y < Fctf.ydim; y++)
			for (long int x = 0; x < Fctf.xdim; x++)
			{
				DIRECT_NZYX_ELEM(Fctf, n, z, y, x) *= DIRECT_NZYX_ELEM(wgh(), n, z, y, x);
			}
		}

		DIRECT_A2D_ELEM(F2D, 0, 0) = 0.0;

		if (do_ewald)
		{
			Matrix2D<RFLOAT> magMat;

			if (!do_ignore_optics && obsModel.hasMagMatrices)
			{
				magMat = obsModel.getMagMatrix(opticsGroup);
			}
			else
			{
				magMat = Matrix2D<RFLOAT>(2,2);
				magMat.initIdentity();
			}

#pragma omp critical(backproject)
	{
	backprojector.set2DFourierTransform(F2DP, A3D, &Fctf, r_ewald_sphere, true, &magMat);
	backprojector.set2DFourierTransform(F2DQ, A3D, &Fctf, r_ewald_sphere, false, &magMat);
	}
		}
		else
		{
			if (do_ctf)
			{
				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(F2D)
				{
					if (!ctf_premultiplied)
						DIRECT_MULTIDIM_ELEM(F2D, n) *= DIRECT_MULTIDIM_ELEM(Fctf, n);
					DIRECT_MULTIDIM_ELEM(Fctf, n) *= DIRECT_MULTIDIM_ELEM(Fctf, n);
				}
			}
		#pragma omp critical(backproject)
		backprojector.set2DFourierTransform(F2D, A3D, &Fctf);
		}
	}


}

void Reconstructor::reconstruct()
{
	bool do_map = false;
	bool do_use_fsc = false;
	MultidimArray<RFLOAT> fsc, dummy;
	Image<RFLOAT> vol;
	fsc.resize(output_boxsize/2+1);

	if (fn_fsc != "")
	{
		do_map = true;
		do_use_fsc =true;
		MetaDataTable MDfsc;
		MDfsc.read(fn_fsc);
		FOR_ALL_OBJECTS_IN_METADATA_TABLE(MDfsc)
		{
			int idx;
			RFLOAT val;
			MDfsc.getValue(EMDL_SPECTRAL_IDX, idx);
			MDfsc.getValue(EMDL_MLMODEL_FSC_HALVES_REF, val);
			fsc(idx) = val;
		}
	}

	if (verb > 0)
		std::cout << " + Starting the reconstruction ..." << std::endl;

	backprojector.symmetrise(nr_helical_asu, helical_twist, helical_rise/angpix);

	if (do_reconstruct_ctf)
	{

		vol().initZeros(ctf_dim, ctf_dim, ctf_dim);
		vol().setXmippOrigin();

		FOR_ALL_ELEMENTS_IN_ARRAY3D(vol())
		{
			int jp = j;
			int ip = i;
			int kp = k;

			// for negative j's: use inverse
			if (j < 0)
			{
				jp = -j;
				ip = -i;
				kp = -k;
			}

			if (jp >= STARTINGX(backprojector.data) && jp <= FINISHINGX(backprojector.data) &&
					ip >= STARTINGY(backprojector.data) && ip <= FINISHINGY(backprojector.data) &&
					kp >= STARTINGZ(backprojector.data) && kp <= FINISHINGZ(backprojector.data))
			{
				if (A3D_ELEM(backprojector.weight, kp, ip, jp) > 0.)
				{
					A3D_ELEM(vol(), k, i, j) = A3D_ELEM(backprojector.data, kp, ip, jp) / A3D_ELEM(backprojector.weight, kp, ip, jp);
					if (do_reconstruct_ctf2)
						A3D_ELEM(vol(), k, i, j) = sqrt(A3D_ELEM(vol(), k, i, j));
				}
			}
		}
	}
	else
	{

		if (do_debug)
		{
			Image<RFLOAT> It;
			FileName fn_tmp = fn_out.withoutExtension();
			It().resize(backprojector.data);
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(It())
			{
				DIRECT_MULTIDIM_ELEM(It(), n) = (DIRECT_MULTIDIM_ELEM(backprojector.data, n)).real;
			}
			It.write(fn_tmp+"_data_real.mrc");
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(It())
			{
				DIRECT_MULTIDIM_ELEM(It(), n) = (DIRECT_MULTIDIM_ELEM(backprojector.data, n)).imag;
			}
			It.write(fn_tmp+"_data_imag.mrc");
			It()=backprojector.weight;
			It.write(fn_tmp+"_weight.mrc");
		}

		MultidimArray<RFLOAT> tau2;
		if (do_use_fsc) backprojector.updateSSNRarrays(1., tau2, dummy, dummy, dummy, fsc, do_use_fsc, true);

		if (do_external_reconstruct)
		{
			FileName fn_root = fn_out.withoutExtension();
			backprojector.externalReconstruct(vol(),
					fn_root,
					tau2, dummy, dummy, dummy, false, 1., 1);
		}
		else
		{
			backprojector.reconstruct(vol(), iter, do_map, tau2);
		}
	}


vol.setSamplingRateInHeader(angpix);
if (do_invert_contrast)
	invert_contrast(vol);
vol.write(fn_out);
	if (verb > 0)
		std::cout << " + Done! Written output map in: "<<fn_out<<std::endl;


}

void Reconstructor::applyCTFPandCTFQ(MultidimArray<Complex> &Fin, CTF &ctf, FourierTransformer &transformer,
		MultidimArray<Complex> &outP, MultidimArray<Complex> &outQ, bool skip_mask)
{
	//FourierTransformer transformer;
	outP.resize(Fin);
	outQ.resize(Fin);
	float angle_step = 180./nr_sectors;
	for (float angle = 0.; angle < 180.;  angle +=angle_step)
	{
		MultidimArray<Complex> CTFP(Fin), Fapp(Fin);
		MultidimArray<RFLOAT> Iapp(YSIZE(Fin), YSIZE(Fin));
		// Two passes: one for CTFP, one for CTFQ
		for (int ipass = 0; ipass < 2; ipass++)
		{
			bool is_my_positive = (ipass == 1) ? is_reverse : !is_reverse;

			// Get CTFP and multiply the Fapp with it
			ctf.getCTFPImage(CTFP, YSIZE(Fin), YSIZE(Fin), angpix, is_my_positive, angle);

			Fapp = Fin * CTFP; // element-wise complex multiplication!

			if (!skip_mask)
			{
				// inverse transform and mask out the particle....
				CenterFFTbySign(Fapp);
				transformer.inverseFourierTransform(Fapp, Iapp);

				softMaskOutsideMap(Iapp, ROUND(mask_diameter/(angpix*2.)), (RFLOAT)width_mask_edge);

				// Back into Fourier-space
				transformer.FourierTransform(Iapp, Fapp, false); // false means: leave Fapp in the transformer
				CenterFFTbySign(Fapp);
			}

			// First time round: resize the output arrays
			if (ipass == 0 && fabs(angle) < XMIPP_EQUAL_ACCURACY)
			{
				outP.resize(Fapp);
				outQ.resize(Fapp);
			}

			// Now set back the right parts into outP (first pass) or outQ (second pass)
			float anglemin = angle + 90. - (0.5*angle_step);
			float anglemax = angle + 90. + (0.5*angle_step);

			// angles larger than 180
			bool is_angle_reverse = false;
			if (anglemin >= 180.)
			{
				anglemin -= 180.;
				anglemax -= 180.;
				is_angle_reverse = true;
			}
			MultidimArray<Complex> *myCTFPorQ, *myCTFPorQb;
			if (is_angle_reverse)
			{
				myCTFPorQ  = (ipass == 0) ? &outQ : &outP;
				myCTFPorQb = (ipass == 0) ? &outP : &outQ;
			}
			else
			{
				myCTFPorQ  = (ipass == 0) ? &outP : &outQ;
				myCTFPorQb = (ipass == 0) ? &outQ : &outP;
			}

			// Deal with sectors with the Y-axis in the middle of the sector...
			bool do_wrap_max = false;
			if (anglemin < 180. && anglemax > 180.)
			{
				anglemax -= 180.;
				do_wrap_max = true;
			}

			// use radians instead of degrees
			anglemin = DEG2RAD(anglemin);
			anglemax = DEG2RAD(anglemax);
			FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(CTFP)
			{
				RFLOAT x = (RFLOAT)jp;
				RFLOAT y = (RFLOAT)ip;
				RFLOAT myangle = (x*x+y*y > 0) ? acos(std::max(-1.0, std::min(1.0, y/sqrt(x*x+y*y)))) : 0; // dot-product with Y-axis: (0,1)
				// Only take the relevant sector now...
				if (do_wrap_max)
				{
					if (myangle >= anglemin)
						DIRECT_A2D_ELEM(*myCTFPorQ, i, j) = DIRECT_A2D_ELEM(Fapp, i, j);
					else if (myangle < anglemax)
						DIRECT_A2D_ELEM(*myCTFPorQb, i, j) = DIRECT_A2D_ELEM(Fapp, i, j);
				}
				else
				{
					if (myangle >= anglemin && myangle < anglemax)
						DIRECT_A2D_ELEM(*myCTFPorQ, i, j) = DIRECT_A2D_ELEM(Fapp, i, j);
				}
			}
		}
	}
}
