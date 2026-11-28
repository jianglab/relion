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

#include <src/rect_refusal.h>
#include "src/postprocessing.h"
#include "src/async_file_writer.h"
#include "src/gpu_locres.h"
#include <omp.h>
#include <algorithm>
#include <cstring>

void Postprocessing::read(int argc, char **argv)
{
	parser.setCommandLine(argc, argv);
	int gen_section = parser.addSection("General options");
	fn_I1 = parser.getOption("--i", "Input name of half1, e.g. run_half1_class001_unfil.mrc", "");
	fn_I2 = parser.getOption("--i2", "Input name of half2, (default replaces half1 from --i with half2)", "");
	fn_OS = parser.getOption("--ios", "Input tomo optimiser set file. It is used to set --i if not provided. Updated output optimiser set is created.", "");
	fn_out = parser.getOption("--o", "Output rootname", "postprocess");
	angpix = textToFloat(parser.getOption("--angpix", "Pixel size in Angstroms", "-1"));
	write_halfmaps = parser.checkOption("--half_maps", "Write post-processed half maps for validation");
	mtf_angpix  = textToFloat(parser.getOption("--mtf_angpix", "Pixel size in the original micrographs/movies (in Angstroms)", "-1."));
	molweight = textToFloat(parser.getOption("--molweight", "Molecular weight (in kDa) of ordered protein mass", "-1"));

	int mask_section = parser.addSection("Masking options");
	do_auto_mask = parser.checkOption("--auto_mask", "Perform automated masking, based on a density threshold");
	ini_mask_density_threshold = textToFloat(parser.getOption("--inimask_threshold", "Density at which to threshold the map for the initial seed mask", "0.02"));
	extend_ini_mask = textToFloat(parser.getOption("--extend_inimask", "Number of pixels to extend the initial seed mask", "3."));
	width_soft_mask_edge  = textToFloat(parser.getOption("--width_mask_edge", "Width for the raised cosine soft mask edge (in pixels)", "6."));
	fn_mask = parser.getOption("--mask", "Filename of a user-provided mask (1=protein, 0=solvent, all values in range [0,1])", "");
	force_mask = parser.checkOption("--force_mask", "Use the mask even when the masked resolution is worse than the unmasked resolution");

	int sharp_section = parser.addSection("Sharpening options");
	fn_mtf = parser.getOption("--mtf", "User-provided STAR-file with the MTF-curve of the detector", "");
	do_auto_bfac = parser.checkOption("--auto_bfac", "Perform automated B-factor determination (Rosenthal and Henderson, 2003)");
	fit_minres = textToFloat(parser.getOption("--autob_lowres", "Lowest resolution (in A) to include in fitting of the B-factor", "10."));
	fit_maxres = textToFloat(parser.getOption("--autob_highres", "Highest resolution (in A) to include in fitting of the B-factor", "0."));
	adhoc_bfac =  textToFloat(parser.getOption("--adhoc_bfac", "User-provided B-factor (in A^2) for map sharpening, e.g. -400", "0."));

	int filter_section = parser.addSection("Filtering options");
	do_fsc_weighting = !parser.checkOption("--skip_fsc_weighting", "Do not use FSC-weighting (Rosenthal and Henderson, 2003) in the sharpening process");
	// include low-pass filter option in the program? This could be useful for structurally heterogeneous reconstructions (instead of FSC-weighting)
	low_pass_freq = textToFloat(parser.getOption("--low_pass", "Resolution (in Angstroms) at which to low-pass filter the final map (0: disable, negative: resolution at FSC=0.143)", "0"));

	int locres_section = parser.addSection("Local-resolution options");
	do_locres = parser.checkOption("--locres", "Perform local resolution estimation");
	locres_sampling = textToFloat(parser.getOption("--locres_sampling", "Sampling rate (in Angstroms) with which to sample the local-resolution map", "25."));
	locres_maskrad = textToFloat(parser.getOption("--locres_maskrad", "Radius (in A) of spherical mask for local-resolution map (default = 0.5*sampling)", "-1"));
	locres_edgwidth = textToFloat(parser.getOption("--locres_edgwidth", "Width of soft edge (in A) on masks for local-resolution map (default = sampling)", "-1"));
	locres_randomize_fsc = textToFloat(parser.getOption("--locres_randomize_at", "Randomize phases from this resolution (in A)", "25."));
	locres_minres = textToFloat(parser.getOption("--locres_minres", "Lowest local resolution allowed (in A)", "50."));
	locres_threads = textToInteger(parser.getOption("--j", "Threads for --locres (default: chosen from the cores and the free memory)", "-1"));
	do_gpu = parser.checkOption("--gpu", "Use GPU acceleration for local-resolution estimation (default: auto when a CUDA GPU is available; optional device IDs)");
	gpu_ids = parser.getOption("--gpu", "Device ids for --gpu, e.g. 0 or \"0:1\"", "default");
	dont_gpu = parser.checkOption("--dont_gpu", "Disable automatic GPU acceleration for --locres (force CPU)");

	int expert_section = parser.addSection("Expert options");
	do_ampl_corr = parser.checkOption("--ampl_corr", "Perform amplitude correlation and DPR, also re-normalize amplitudes for non-uniform angular distributions");
	randomize_fsc_at = textToFloat(parser.getOption("--randomize_at_fsc", "Randomize phases from the resolution where FSC drops below this value", "0.8"));
	randomize_at_A  = textToFloat(parser.getOption("--randomize_at_A", "Randomize phases from this resolution (in A) onwards (if positive)", "-1"));
	filter_edge_width = textToInteger(parser.getOption("--filter_edge_width", "Width of the raised cosine on the low-pass filter edge (in resolution shells)", "2"));
	do_interpolate = parser.checkOption("--interpolate", "Interpolate the FSC to obtain an additional, more precise resolution estimate");
	verb = textToInteger(parser.getOption("--verb", "Verbosity", "1"));
	int random_seed = textToInteger(parser.getOption("--random_seed", "Seed for random number generator (negative value for truly random)", "0"));

	if (random_seed >= 0)
	{
		init_random_generator(random_seed);
	}

	// Check for errors in the command-line option
	if (parser.checkForErrors())
		REPORT_ERROR("Errors encountered on the command line (see above), exiting...");
}

void Postprocessing::usage()
{
	parser.writeUsage(std::cout);
}

void Postprocessing::clear()
{
	fn_I1 = fn_I2 = fn_OS = "";
	fn_out="postprocess";
	angpix = 1.;
	mtf_angpix = 1.;
	do_auto_mask = false;
	ini_mask_density_threshold = 0.02;
	width_soft_mask_edge = 6.;
	fn_mask = "";
	fn_mtf = "";
	do_auto_bfac = false;
	fit_minres = 10.;
	fit_maxres = 0.;
	adhoc_bfac = 0.;
	do_fsc_weighting = true;
	low_pass_freq = 0.;
	randomize_fsc_at = 0.8;
	randomize_at_A = -1.;
	filter_edge_width = 2.;
	verb = 1;
	do_ampl_corr = false;
	do_gpu = false;
	dont_gpu = false;
	gpu_ids = "default";
}

void Postprocessing::initialise()
{
	// Check if input is a tomo optimisation set file
	if (fn_OS != "")
	{
		optimisationSet.read(fn_OS);

		if (fn_I1 == "")
		{
			if (!optimisationSet.getValue(EMDL_TOMO_REFERENCE_MAP_1_FILE_NAME, fn_I1) ||
				!optimisationSet.getValue(EMDL_TOMO_REFERENCE_MAP_2_FILE_NAME, fn_I2))
				REPORT_ERROR("No halfmap filenames were found in file " + fn_OS);
		}
		else
		{
			if (fn_I2 == "" && !fn_I1.getTheOtherHalf(fn_I2))
				REPORT_ERROR("The input filename does not contain 'half1' or 'half2'");

			optimisationSet.setValue(EMDL_TOMO_REFERENCE_MAP_1_FILE_NAME, fn_I1);
			optimisationSet.setValue(EMDL_TOMO_REFERENCE_MAP_2_FILE_NAME, fn_I2);
		}
	}
	else if (fn_I1 != "") // Read in the input maps
	{
		if (fn_I2 == "" && !fn_I1.getTheOtherHalf(fn_I2))
			REPORT_ERROR("The input filename does not contain 'half1' or 'half2'");
	}
	else
	{
		REPORT_ERROR("Input file is missing. Provide, at least, --i or --ios input.");
	}

	if (verb > 0)
	{
		std::cout <<"== Reading input half-reconstructions: " <<std::endl;
		std::cout.width(35); std::cout << std::left <<"  + half1-map: "; std::cout << fn_I1 << std::endl;
		std::cout.width(35); std::cout << std::left <<"  + half2-map: "; std::cout << fn_I2 << std::endl;
	}

    // Check if output directory exists, otherwise create it
    if (fn_out.contains("/"))
    {
        FileName fn_dir = fn_out.beforeLastOf("/");
        if (!exists(fn_dir)) mktree(fn_dir);
    }


	I1.read(fn_I1);
	I2.read(fn_I2);
	refuseNonCubicMap(I1(), "relion_postprocess", "half-map 1");
	refuseNonCubicMap(I2(), "relion_postprocess", "half-map 2");
	I1().setXmippOrigin();
	I2().setXmippOrigin();

	if (angpix <= 0)
	{
		angpix = I1.samplingRateX();
		std::cerr << "WARNING: You did not specify --angpix. The pixel size in the image header, " << angpix << " A/px, is used." << std::endl;
	}

	if (mtf_angpix < 0.)
	{
		if (verb > 0) std::cout << " + --mtf_angpix was not provided, assuming pixel size in raw micrographs is the same as in particles, " << angpix << " A/px." << std::endl;
		mtf_angpix = angpix;
	}

	// Calculate what fraction of voxels in the box is protein according to the expected ordered molecular weight
	// Protein density is 1.35 g/cm^3, Nav=6.022 E+23, so protein volume = (MW /0.81 Da) A^3
	// 47.6% is volume of sphere relative to box
	if (molweight > 0.)
	{
		frac_molweight = 0.476 * std::pow(XSIZE(I1())*angpix, 3) * 0.81 / ( molweight*1000) ;
		if (verb > 0)
		{
			std::cout.width(35); std::cout << std::left   << "  + ordered molecular weight (kDa): "; std::cout  << molweight <<std::endl;
			std::cout.width(35); std::cout << std::left   << "  + fraction f (molweight based): "; std::cout  << frac_molweight <<std::endl;
		}
	}

	if (!I1().sameShape(I2()))
	{
		std::cerr << " Size of half1 map: "; I1().printShape(std::cerr); std::cerr << std::endl;
		std::cerr << " Size of half2 map: "; I2().printShape(std::cerr); std::cerr << std::endl;
		REPORT_ERROR("Postprocessing::initialise ERROR: The two half reconstructions are not of the same size!");
	}

	if (do_locres)
	{
		if (locres_maskrad < 0.0)
			locres_maskrad = 0.5*locres_sampling;
		if (locres_edgwidth < 0.0)
			locres_edgwidth = locres_sampling;

		if (fn_mask != "" && verb > 0)
			std::cerr << " WARNING: --mask is used only to make a histogram of local resolutions; it is not used for local resolution calculation itself." << std::endl;
		if (do_auto_bfac)
			REPORT_ERROR("Postprocessing::initialise ERROR: for --locres, you cannot do --auto_bfac, use --adhoc_bfac instead!");
	}

	if (do_auto_mask)
		REPORT_ERROR("Postprocessing:: --auto_mask has been removed. Please make a mask with relion_mask_create beforehand.");

	if (do_auto_bfac && ABS(adhoc_bfac) > 0.)
		REPORT_ERROR("Postprocessing::initialise ERROR: provide either --auto_bfac OR --adhoc_bfac, but not both!");
}

bool Postprocessing::getMask()
{
	if (fn_mask != "")
	{
		if (verb > 0)
		{
			std::cout << "== Using a user-provided mask ... " <<std::endl;
			std::cout.width(35); std::cout << std::left   << "  + input mask: "; std::cout  << fn_mask <<std::endl;
		}

		// Read the mask in memory
		Im.read(fn_mask);
		Im().setXmippOrigin();

		// Check values are between 0 and 1
		RFLOAT avg, stddev, minval, maxval;
		Im().computeStats(avg, stddev, minval, maxval);

		long summask = 0;
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Im())
		{
			if (DIRECT_MULTIDIM_ELEM(Im(), n) > 0.5) summask++;
		}
		avg = (RFLOAT)summask / (RFLOAT)NZYXSIZE(Im());
		frac_solvent_mask = 0.476 /avg;
		molweight_frommask = avg * std::pow(XSIZE(Im()) * angpix, 3) * 0.81;

		if (verb > 0)
		{
			std::cout.width(35); std::cout << std::left   << "  + fraction f (solvent mask based): "; std::cout  << frac_solvent_mask <<std::endl;
			std::cout.width(35); std::cout << std::left   << "  + molecular weight inside protein mask: "; std::cout  << molweight_frommask <<std::endl;

		}

		if (minval < -1e-6 || maxval - 1. > 1.e-6)
		{
			std::cerr << " minval= " << minval << " maxval= " << maxval << std::endl;
			REPORT_ERROR("Postprocessing::mask ERROR: mask values not in range [0,1]!");
		}

		// Also check the mask is the same size as the input maps
		if (!Im().sameShape(I2()))
		{
			std::cerr << " Size of input mask: "; Im().printShape(std::cerr); std::cerr<< std::endl;
			std::cerr << " Size of input maps: "; I1().printShape(std::cerr); std::cerr<< std::endl;
			REPORT_ERROR("Postprocessing::mask ERROR: mask and input maps do not have the same size!");
		}

	}
	else
	{
		if (verb > 0)
		{
			std::cout << "== Not performing any masking ... " << std::endl;
			frac_solvent_mask = 0.;
		}
		return false;
	}

	return true;
}

void Postprocessing::divideByMtf(MultidimArray<Complex > &FT)
{
	if (fn_mtf != "")
	{
		if (verb > 0)
		{
			std::cout << "== Dividing map by the MTF of the detector ..." << std::endl;
			std::cout.width(35); std::cout << std::left <<"  + mtf STAR-file: "; std::cout << fn_mtf << std::endl;
		}

		MetaDataTable MDmtf;

		if (!fn_mtf.isStarFile())
			REPORT_ERROR("Postprocessing::divideByMtf ERROR: input MTF file is not a STAR file.");

		MDmtf.read(fn_mtf);
		MultidimArray<RFLOAT> mtf_resol, mtf_value;
		mtf_resol.resize(MDmtf.numberOfObjects());
		mtf_value.resize(mtf_resol);

		RFLOAT resol_inv_pixel;
		int i =0;
		FOR_ALL_OBJECTS_IN_METADATA_TABLE(MDmtf)
		{
			MDmtf.getValue(EMDL_RESOLUTION_INVPIXEL, resol_inv_pixel);
			DIRECT_A1D_ELEM(mtf_resol, i) = resol_inv_pixel/mtf_angpix; // resolution needs to be given in 1/Ang
			MDmtf.getValue(EMDL_POSTPROCESS_MTF_VALUE, DIRECT_A1D_ELEM(mtf_value, i) );
			if (DIRECT_A1D_ELEM(mtf_value, i) < 1e-10)
			{
				std::cerr << " i= " << i <<  " mtf_value[i]= " << DIRECT_A1D_ELEM(mtf_value, i) << std::endl;
				REPORT_ERROR("Postprocessing::sharpenMap ERROR: zero or negative values encountered in MTF curve!");
			}
			i++;
		}

		// Calculate slope of resolution (in 1/A) per element in the MTF array, in order to interpolate below
		RFLOAT res_per_elem = (DIRECT_A1D_ELEM(mtf_resol, i-1) - DIRECT_A1D_ELEM(mtf_resol, 0)) / (RFLOAT)(i);
		if (res_per_elem < 1e-10) REPORT_ERROR(" ERROR: the resolution in the MTF star file does not go up....");

		RFLOAT xsize_ang = angpix * XSIZE(I1());
		const long int nx = XSIZE(FT), ny = YSIZE(FT), nz = ZSIZE(FT);
		const long int n_mtf = MULTIDIM_SIZE(mtf_value);
		const RFLOAT nyq_inv_ang = 1. / (2. * angpix);
		const RFLOAT nyq2_len = nyq_inv_ang * xsize_ang;
		const RFLOAT nyq2_safe = nyq2_len * nyq2_len * (1.0 + 1e-12);
		const RFLOAT * __restrict mtf_r = MULTIDIM_ARRAY(mtf_resol);
		const RFLOAT * __restrict mtf_v = MULTIDIM_ARRAY(mtf_value);
		const int nt = relionBulkThreads(NZYXSIZE(FT));
		#pragma omp parallel for collapse(2) num_threads(nt) if(nt > 1)
		for (long int k = 0; k < nz; k++)
		{
			for (long int i = 0; i < ny; i++)
			{
				const long int kp = (k < nx) ? k : k - nz;
				const long int ip = (i < nx) ? i : i - ny;
				const long int k2i2 = kp * kp + ip * ip;
				Complex * __restrict row = &DIRECT_A3D_ELEM(FT, k, i, 0);
				for (long int j = 0; j < nx; j++)
				{
					const long int r2 = k2i2 + j * j;
					if ((RFLOAT)r2 > nyq2_safe)
						continue;
					const RFLOAT res = sqrt((RFLOAT)r2) / xsize_ang; // get resolution in 1/Ang
					if (res < nyq_inv_ang)
					{
						const int i_0 = FLOOR(res / res_per_elem);
						RFLOAT mtf;
						// check boundaries of the array
						if (i_0 >= n_mtf - 1)
							mtf = mtf_v[n_mtf - 1];
						else if (i_0 <= 0)
							mtf = mtf_v[0];
						else
						{
							// linear interpolation:
							const RFLOAT x_0 = mtf_r[i_0];
							const RFLOAT y_0 = mtf_v[i_0];
							const RFLOAT x_1 = mtf_r[i_0 + 1];
							const RFLOAT y_1 = mtf_v[i_0 + 1];
							mtf = y_0 + (y_1 - y_0) * (res - x_0) / (x_1 - x_0);
						}

						// Divide Fourier component by the MTF
						row[j] /= mtf;
					}
				}
			}
		}
	}
}

bool Postprocessing::findSurfacePixel(int idx, int kp, int ip, int jp,
		int &best_kpp, int &best_ipp, int &best_jpp,
		int myradius_count, int search)
{
	// bring kp, ip, jp onto the sphere
	RFLOAT frac = (RFLOAT)(myradius_count)/(RFLOAT)idx;
	int kpp = ROUND(frac*kp);
	int ipp = ROUND(frac*ip);
	int jpp = ROUND(frac*jp);

	// Search +/- 2 pixels in all directions and choose voxel closest to the circle
	int best_dist = 999;
	best_kpp=kpp;
	best_ipp=ipp;
	best_jpp=jpp;
	bool found = false;
	for (int kppp = kpp-search; kppp <= kpp+search; kppp++)
	{
		for (int ippp = ipp-search; ippp <= ipp+search; ippp++)
		{
			for (int jppp = jpp-search; jppp <= jpp+search; jppp++)
	 		{
				// Distance to surface on the sphere
				int dist = ABS(ROUND(sqrt((RFLOAT)(kppp*kppp + ippp*ippp + jppp*jppp))) - myradius_count);
				int reldist2 = (kppp-kpp)*(kppp-kpp) + (ippp-ipp)*(ippp-ipp) + (jppp-jpp)*(jppp-jpp);
				if (dist < 0.5 && reldist2 < best_dist)
				{
					best_kpp=kppp;
					best_ipp=ippp;
					best_jpp=jppp;
					best_dist=reldist2;
					found=true;
				}
			}
    		}
	}

	return found;
}

void Postprocessing::correctRadialAmplitudeDistribution(MultidimArray<RFLOAT > &I)
{
	MultidimArray<Complex > FT;
	FourierTransformer transformer;
	transformer.FourierTransform(I, FT, false);

	// First calculate radial average, to normalize the power spectrum
	int myradius = XSIZE(FT);
	MultidimArray< int > radial_count(myradius);
	MultidimArray<RFLOAT> num, ravg;
	num.initZeros(myradius);
	ravg.initZeros(myradius);
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(FT)
	{
		int idx = ROUND(sqrt(kp*kp + ip*ip + jp*jp));
		if (idx >= myradius)
	    		continue;
		ravg(idx)+= norm(DIRECT_A3D_ELEM(FT, k, i, j));
		radial_count(idx)++;
	}
	FOR_ALL_ELEMENTS_IN_ARRAY1D(ravg)
	{
		if (radial_count(i) > 0)
		{
    			ravg(i) /= radial_count(i);
	    	}
	}

	// Apply correction only beyond low-res fitting of B-factors
	int minr = FLOOR(XSIZE(FT) * angpix / fit_minres);
	int myradius_count = minr;
	MultidimArray<RFLOAT> sum3d;
	MultidimArray<int> count3d;
	sum3d.resize(2*myradius_count+4, 2*myradius_count+4, 2*myradius_count+4);
	sum3d.setXmippOrigin();
	count3d.resize(sum3d);
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(FT)
	{
		int idx = ROUND(sqrt(kp*kp + ip*ip + jp*jp));
		// only correct from fit_minres to Nyquist
		if (idx < minr || idx >= myradius)
	     		continue;

		int best_kpp, best_ipp, best_jpp;
		if (!findSurfacePixel(idx, kp, ip, jp, best_kpp, best_ipp, best_jpp, myradius_count, 2))
		{
			std::cerr << "Postprocessing::correctRadialAmplitudeDistribution ERROR! kp= " << kp << " ip= " << ip << " jp= " << jp << std::endl;
		}

		// Apply correction on the spectrum-corrected values!
		RFLOAT aux = norm(DIRECT_A3D_ELEM(FT, k, i, j)) / ravg(idx);
		A3D_ELEM(sum3d, best_kpp, best_ipp, best_jpp) += aux;
		A3D_ELEM(count3d, best_kpp, best_ipp, best_jpp) += 1;
	}

	// Average
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(sum3d)
	{
		if (DIRECT_MULTIDIM_ELEM(count3d, n) > 0)
		{
			DIRECT_MULTIDIM_ELEM(sum3d, n) /= DIRECT_MULTIDIM_ELEM(count3d, n);
		}
	}

	// Now divide all elements by the normalized correction term
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(FT)
	{
		int idx = ROUND(sqrt(kp*kp + ip*ip + jp*jp));
		// only correct from fit_minres to Nyquist
		if (idx < minr || idx >= myradius)
		     	continue;

		int best_kpp, best_ipp, best_jpp;
		if (!findSurfacePixel(idx, kp, ip, jp, best_kpp, best_ipp, best_jpp, myradius_count, 2))
		{
			std::cerr << "Postprocessing::correctRadialAmplitudeDistribution ERROR!  kp= " << kp << " ip= " << ip << " jp= " << jp << std::endl;
		}

		// Apply correction on the spectrum-corrected values!
		RFLOAT aux = sqrt(A3D_ELEM(sum3d, best_kpp, best_ipp, best_jpp));
		DIRECT_A3D_ELEM(FT, k, i, j) /= aux;
	}

	transformer.inverseFourierTransform(FT, I);
}

RFLOAT Postprocessing::sharpenMap()
{
	MultidimArray<Complex > FT;
	FourierTransformer transformer;
	transformer.FourierTransform(I1(), FT, true);

	makeGuinierPlot(FT, guinierin);

	// A. If MTF curve is given, first divide by the MTF
	divideByMtf(FT);
	makeGuinierPlot(FT, guinierinvmtf);

	// B. Then perform B-factor sharpening
	if (do_fsc_weighting)
	{
		if (verb > 0)
		{
			std::cout <<"== Applying sqrt(2*FSC/(FSC+1)) weighting (as in Rosenthal & Henderson, 2003) ..." <<std::endl;
		}
		applyFscWeighting(FT, fsc_true);
	}
	makeGuinierPlot(FT, guinierweighted);

	global_bfactor = 0.;
	if (do_auto_bfac)
	{
		if (verb > 0)
		{
			std::cout <<"== Fitting straight line through Guinier plot to find B-factor ..." <<std::endl;
			std::cout.width(35); std::cout << std::left <<"  + fit from resolution: "; std::cout << fit_minres << std::endl;
			std::cout.width(35); std::cout << std::left  <<"  + fit until resolution: "; std::cout << fit_maxres << std::endl;
		}

		fitStraightLine(guinierweighted, global_slope, global_intercept, global_corr_coeff);
		global_bfactor = 4. * global_slope;
		if (verb > 0)
		{
			std::cout.width(35); std::cout << std::left  <<"  + slope of fit: "; std::cout << global_slope << std::endl;
			std::cout.width(35); std::cout << std::left  <<"  + intercept of fit: "; std::cout << global_intercept << std::endl;
			std::cout.width(35); std::cout << std::left  <<"  + correlation of fit: "; std::cout << global_corr_coeff << std::endl;
		}
	}
	else if (ABS(adhoc_bfac) > 0.)
	{
		if (verb > 0)
		{
			std::cout <<"== Using a user-provided (ad-hoc) B-factor ..." <<std::endl;
		}
		if (adhoc_bfac > 0.)
			std::cout <<" WARNING: using a positive B-factor. This will effectively dampen your map. Use negative value to sharpen it!" << std::endl;
		global_bfactor = adhoc_bfac;
	}

	// Now apply the B-factor
	if (ABS(global_bfactor) > 0.)
	{
		if (verb > 0)
		{
			std::cout.width(35); std::cout << std::left  <<"  + apply b-factor of: "; std::cout << global_bfactor << std::endl;
		}
		applyBFactorToMap(FT, XSIZE(I1()), global_bfactor, angpix);
	}
        else
        {
            // Write a warning that the map will not be sharpened
            std::cerr << " WARNING: map will not be sharpened. This map may be used as reference for subsequent refinements, but you would need to run a postprocessing job with B-factor sharpening to better visualise high-resolution features." << std::endl;
        }

	makeGuinierPlot(FT, guiniersharpen);

	RFLOAT applied_filter = low_pass_freq;
	if (low_pass_freq != 0)
	{
		if (low_pass_freq < 0)
			applied_filter = global_resol;

		if (verb > 0)
		{
			std::cout << "== Low-pass filtering final map ... " << std::endl;
			std::cout.width(35); std::cout << std::left  <<"  + filter frequency: "; std::cout << applied_filter << std::endl;
		}

		lowPassFilterMap(FT, XSIZE(I1()), applied_filter, angpix, filter_edge_width);
	}

	transformer.inverseFourierTransform(FT, I1());

	return applied_filter;
}

void Postprocessing::calculateFSCtrue(MultidimArray<RFLOAT> &fsc_true, MultidimArray<RFLOAT> &fsc_unmasked,
		MultidimArray<RFLOAT> &fsc_masked, MultidimArray<RFLOAT> &fsc_random_masked, int randomize_at)
{
	// Now that we have fsc_masked and fsc_random_masked, calculate fsc_true according to Richard's formula
	// FSC_true = FSC_t - FSC_n / ( )

	// Sometimes FSc at origin becomes -1!
	if (DIRECT_A1D_ELEM(fsc_masked, 0) <= 0.)
		DIRECT_A1D_ELEM(fsc_masked, 0) = 1.;
	if (DIRECT_A1D_ELEM(fsc_unmasked, 0) <= 0.)
		DIRECT_A1D_ELEM(fsc_unmasked, 0) = 1.;
	if (DIRECT_A1D_ELEM(fsc_random_masked, 0) <= 0.)
		DIRECT_A1D_ELEM(fsc_random_masked, 0) = 1.;


	fsc_true.resize(fsc_masked);
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(fsc_true)
	{
		// 29jan2015: let's move this 2 shells upwards, because of small artefacts near the resolution of randomisation!
		if (i < randomize_at + 2)
		{
			DIRECT_A1D_ELEM(fsc_true, i) = DIRECT_A1D_ELEM(fsc_masked, i);
		}
		else
		{
			RFLOAT fsct = DIRECT_A1D_ELEM(fsc_masked, i);
			RFLOAT fscn = DIRECT_A1D_ELEM(fsc_random_masked, i);
			DIRECT_A1D_ELEM(fsc_true, i) = (fsct - fscn) / (1. - fscn);
		}
	}
    
}

void Postprocessing::calculateFSCpart(const MultidimArray<RFLOAT> fsc_unmasked, RFLOAT fraction, MultidimArray<RFLOAT> &fsc_part)
{
	// Now that we have fsc_masked and fsc_random_masked, calculate fsc_true according to Richard's formula
	// FSC_true = FSC_t - FSC_n / ( )

	// Sometimes FSc at origin becomes -1!
	if (DIRECT_A1D_ELEM(fsc_masked, 0) <= 0.)
		DIRECT_A1D_ELEM(fsc_masked, 0) = 1.;
	if (DIRECT_A1D_ELEM(fsc_random_masked, 0) <= 0.)
		DIRECT_A1D_ELEM(fsc_random_masked, 0) = 1.;


	fsc_part.resize(fsc_unmasked);
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(fsc_part)
	{
		DIRECT_A1D_ELEM(fsc_part, i) = fraction * DIRECT_A1D_ELEM(fsc_unmasked, i) / (1. + (fraction-1) * DIRECT_A1D_ELEM(fsc_unmasked, i));
	}

}

void Postprocessing::applyFscWeighting(MultidimArray<Complex > &FT, MultidimArray<RFLOAT> my_fsc)
{
	// Find resolution where fsc_true drops below zero for the first time
	// Set all weights to zero beyond that resolution
	int ires_max = 0 ;

	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(my_fsc)
	{
		if (DIRECT_A1D_ELEM(my_fsc, i) < 0.0001)
			break;
		ires_max = i;
	}

	std::vector<RFLOAT> fsc_w(ires_max + 1, 0.);
	for (int i = 0; i <= ires_max; i++)
	{
		const RFLOAT fsc = DIRECT_A1D_ELEM(my_fsc, i);
		if (fsc > 0.)
			fsc_w[i] = sqrt((2. * fsc) / (1. + fsc));
	}

	const long int nx = XSIZE(FT), ny = YSIZE(FT), nz = ZSIZE(FT);
	const RFLOAT max_r2_safe = ((RFLOAT)ires_max + 0.5) * ((RFLOAT)ires_max + 0.5) + 1e-6;
	const RFLOAT * __restrict fw = fsc_w.data();
	const int nt = relionBulkThreads(NZYXSIZE(FT));
	#pragma omp parallel for collapse(2) num_threads(nt) if(nt > 1)
	for (long int k = 0; k < nz; k++)
	{
		for (long int i = 0; i < ny; i++)
		{
			const long int kp = (k < nx) ? k : k - nz;
			const long int ip = (i < nx) ? i : i - ny;
			const long int k2i2 = kp * kp + ip * ip;
			Complex * __restrict row = &DIRECT_A3D_ELEM(FT, k, i, 0);
			for (long int j = 0; j < nx; j++)
			{
				const long int r2 = k2i2 + j * j;
				if ((RFLOAT)r2 > max_r2_safe)
				{
					row[j] = 0.;
					continue;
				}
				const int ires = ROUND(sqrt((RFLOAT)r2));
				if (ires <= ires_max)
					row[j] *= fw[ires];
				else
					row[j] = 0.;
			}
		}
	}
}

void Postprocessing::makeGuinierPlot(MultidimArray<Complex > &FT, std::vector<fit_point2D> &guinier)
{
	MultidimArray<int> radial_count(XSIZE(FT));
	MultidimArray<RFLOAT> lnF(XSIZE(FT));
	fit_point2D      onepoint;

	const long int nx = XSIZE(FT), ny = YSIZE(FT), nz = ZSIZE(FT);
	const RFLOAT max_r2_safe = ((RFLOAT)nx - 0.5) * ((RFLOAT)nx - 0.5) + 1e-6;
	const int nt = relionBulkThreads(NZYXSIZE(FT));
	if (nt <= 1)
	{
		FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(FT)
		{
			int r2 = kp * kp + ip * ip + jp * jp;
			if ((RFLOAT)r2 >= max_r2_safe) continue;
			int ires = ROUND(sqrt((RFLOAT)r2));
			if (ires < nx)
			{
				lnF(ires) += abs(DIRECT_A3D_ELEM(FT, k, i, j));
				radial_count(ires)++;
			}
		}
	}
	else
	{
		std::vector<RFLOAT> lnF_t((size_t)nt * nx, 0.);
		std::vector<int> cnt_t((size_t)nt * nx, 0);
		#pragma omp parallel num_threads(nt)
		{
			const int tid =
#ifdef _OPENMP
				omp_get_thread_num();
#else
				0;
#endif
			RFLOAT * __restrict l_ln = &lnF_t[(size_t)tid * nx];
			int * __restrict l_cnt = &cnt_t[(size_t)tid * nx];
			#pragma omp for collapse(2) schedule(static)
			for (long int k = 0; k < nz; k++)
			{
				for (long int i = 0; i < ny; i++)
				{
					const long int kp = (k < nx) ? k : k - nz;
					const long int ip = (i < nx) ? i : i - ny;
					const long int k2i2 = kp * kp + ip * ip;
					const Complex * __restrict row = &DIRECT_A3D_ELEM(FT, k, i, 0);
					for (long int j = 0; j < nx; j++)
					{
						const long int r2 = k2i2 + j * j;
						if ((RFLOAT)r2 >= max_r2_safe) continue;
						const int ires = ROUND(sqrt((RFLOAT)r2));
						if (ires < nx)
						{
							l_ln[ires] += abs(row[j]);
							l_cnt[ires]++;
						}
					}
				}
			}
		}
		for (int t = 0; t < nt; t++)
		{
			const size_t base = (size_t)t * nx;
			for (long int idx = 0; idx < nx; idx++)
			{
				lnF(idx) += lnF_t[base + idx];
				radial_count(idx) += cnt_t[base + idx];
			}
		}
	}

	RFLOAT xsize = XSIZE(I1());
	guinier.clear();
	FOR_ALL_ELEMENTS_IN_ARRAY1D(radial_count)
	{

		RFLOAT res = (xsize * angpix)/(RFLOAT)i; // resolution in Angstrom
		if (res >= angpix * 2.) // Apply B-factor sharpening until Nyquist, then low-pass filter later on (with a soft edge)
		{
			onepoint.x = 1. / (res * res);
			if (DIRECT_A1D_ELEM(lnF, i) > 0.)
			{
				onepoint.y = log ( DIRECT_A1D_ELEM(lnF, i) / DIRECT_A1D_ELEM(radial_count, i) );
				if (res <= fit_minres && res >= fit_maxres)
				{
					onepoint.w = 1.;
				}
				else
				{
					onepoint.w = 0.;
				}
			}
			else
			{
				onepoint.y = -99.;
				onepoint.w = 0.;
			}
			//std::cerr << " onepoint.x= " << onepoint.x << " onepoint.y= " << onepoint.y << " onepoint.w= " << onepoint.w << std::endl;
			guinier.push_back(onepoint);
		}
	}
}

void Postprocessing::writeOutput()
{
	FileName fn_tmp;

	if (verb > 0)
	{
		std::cout <<"== Writing output files ..." <<std::endl;
	}

	writeMaps(fn_out);

	// Write an output STAR file with FSC curves, Guinier plots etc
	std::ofstream  fh;
	fn_tmp = fn_out + ".star";
	if (verb > 0)
	{
		std::cout.width(35); std::cout << std::left <<"  + Metadata file: "; std::cout << fn_tmp<< std::endl;
	}

	fh.open((fn_tmp).c_str(), std::ios::out);
	if (!fh)
		REPORT_ERROR( (std::string)"MlOptimiser::write: Cannot write file: " + fn_tmp);

	// Write the command line as a comment in the header
	fh << "# RELION postprocess; version " << g_RELION_VERSION << std::endl;
	fh << "# ";
	parser.writeCommandLine(fh);

	MetaDataTable MDlist, MDfsc, MDguinier;

	MDlist.setIsList(true);
	MDlist.setName("general");
	MDlist.addObject();
	MDlist.setValue(EMDL_POSTPROCESS_FINAL_RESOLUTION, global_resol);
	MDlist.setValue(EMDL_POSTPROCESS_BFACTOR, global_bfactor );
	MDlist.setValue(EMDL_POSTPROCESS_UNFIL_HALFMAP1, fn_I1);
	MDlist.setValue(EMDL_POSTPROCESS_UNFIL_HALFMAP2, fn_I2);
	MDlist.setValue(EMDL_POSTPROCESSED_MAP, fn_out + ".mrc");
	if (molweight > 0.)
	{
		MDlist.setValue(EMDL_POSTPROCESS_MOLWEIGHT, molweight);
		MDlist.setValue(EMDL_POSTPROCESS_FRACTION_MOLWEIGHT, frac_molweight);
	}
	if (do_mask)
	{
		MDlist.setValue(EMDL_POSTPROCESS_FRACTION_SOLVENT_MASK, frac_solvent_mask);
		RFLOAT randomize_at_Ang = XSIZE(I1())* angpix / randomize_at;
		MDlist.setValue(EMDL_MASK_NAME, fn_mask);
		MDlist.setValue(EMDL_POSTPROCESS_RANDOMISE_FROM, randomize_at_Ang);
		MDlist.setValue(EMDL_POSTPROCESSED_MAP_MASKED, fn_out + "_masked.mrc");
	}
	if (do_auto_bfac)
	{
		MDlist.setValue(EMDL_POSTPROCESS_GUINIER_FIT_SLOPE, global_slope);
		MDlist.setValue(EMDL_POSTPROCESS_GUINIER_FIT_INTERCEPT, global_intercept);
		MDlist.setValue(EMDL_POSTPROCESS_GUINIER_FIT_CORRELATION, global_corr_coeff);
	}
	MDlist.write(fh);

	// If input optimisation set is provided, also crete it as output
	if (!optimisationSet.isEmpty())
	{
		optimisationSet.setValue(EMDL_TOMO_REFERENCE_FSC_FILE_NAME, fn_tmp);
		optimisationSet.write(fn_out + "_optimisation_set.star");
	}

	MDfsc.setName("fsc");
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(fsc_true)
	{
		MDfsc.addObject();
		RFLOAT res = (i > 0) ? (XSIZE(I1()) * angpix / (RFLOAT)i) : 999.;
		MDfsc.setValue(EMDL_SPECTRAL_IDX, (int)i);
		MDfsc.setValue(EMDL_RESOLUTION, 1./res);
		MDfsc.setValue(EMDL_RESOLUTION_ANGSTROM, res);
		if (do_mask)
		{
			MDfsc.setValue(EMDL_POSTPROCESS_FSC_TRUE, DIRECT_A1D_ELEM(fsc_true, i) );
			MDfsc.setValue(EMDL_POSTPROCESS_FSC_PART_FRACMASK, DIRECT_A1D_ELEM(fsc_part_fracmask, i) );
			if (molweight > 0.)
				MDfsc.setValue(EMDL_POSTPROCESS_FSC_PART_MOLWEIGHT, DIRECT_A1D_ELEM(fsc_part_molweight, i) );
			MDfsc.setValue(EMDL_POSTPROCESS_FSC_UNMASKED, DIRECT_A1D_ELEM(fsc_unmasked, i) );
			MDfsc.setValue(EMDL_POSTPROCESS_FSC_MASKED, DIRECT_A1D_ELEM(fsc_masked, i) );
			MDfsc.setValue(EMDL_POSTPROCESS_FSC_RANDOM_MASKED, DIRECT_A1D_ELEM(fsc_random_masked, i) );
			if (do_ampl_corr)
			{
				MDfsc.setValue(EMDL_POSTPROCESS_AMPLCORR_UNMASKED, DIRECT_A1D_ELEM(acorr_unmasked, i) );
				MDfsc.setValue(EMDL_POSTPROCESS_AMPLCORR_MASKED, DIRECT_A1D_ELEM(acorr_masked, i) );
				MDfsc.setValue(EMDL_POSTPROCESS_DPR_UNMASKED, DIRECT_A1D_ELEM(dpr_unmasked, i) );
				MDfsc.setValue(EMDL_POSTPROCESS_DPR_MASKED, DIRECT_A1D_ELEM(dpr_masked, i) );
			}
		}
		else
		{
			MDfsc.setValue(EMDL_POSTPROCESS_FSC_UNMASKED, DIRECT_A1D_ELEM(fsc_true, i) );
			if (molweight > 0.)
				MDfsc.setValue(EMDL_POSTPROCESS_FSC_PART_MOLWEIGHT, DIRECT_A1D_ELEM(fsc_part_molweight, i) );
			if (do_ampl_corr)
			{
				MDfsc.setValue(EMDL_POSTPROCESS_AMPLCORR_UNMASKED, DIRECT_A1D_ELEM(acorr_unmasked, i) );
				MDfsc.setValue(EMDL_POSTPROCESS_DPR_UNMASKED, DIRECT_A1D_ELEM(dpr_unmasked, i) );
			}
		}
	}
	MDfsc.write(fh);

	// Write a plot with the FSC curves
	std::string title= "Final resolution = " + floatToString(global_resol, 5, 2) + " Angstroms";
	if (do_mask)
    {
        CPlot2D *plot2D = new CPlot2D(title);
        plot2D->SetXAxisSize(600);
        plot2D->SetYAxisSize(400);
        MDfsc.addToCPlot2D(plot2D, EMDL_RESOLUTION, EMDL_POSTPROCESS_FSC_TRUE, 0., 0., 0., 2.);
        MDfsc.addToCPlot2D(plot2D, EMDL_RESOLUTION, EMDL_POSTPROCESS_FSC_UNMASKED, 0., 1., 0.);
        MDfsc.addToCPlot2D(plot2D, EMDL_RESOLUTION, EMDL_POSTPROCESS_FSC_MASKED, 0., 0., 1.);
        MDfsc.addToCPlot2D(plot2D, EMDL_RESOLUTION, EMDL_POSTPROCESS_FSC_RANDOM_MASKED, 1., 0., 0.);
        plot2D->SetXAxisTitle("resolution (1/A)");
        plot2D->SetYAxisTitle("Fourier Shell Correlation");
        plot2D->OutputPostScriptPlot(fn_out + "_fsc.eps");
        delete plot2D;
    }
//#define CISTEMFSC
#ifdef CISTEMFSC
	// Write a plot with the FSC curves
	std::string title2= "RELION/cisTEM FSC comparison; MW_mask = " +  floatToString(molweight_frommask/1000., 8,2) + " kDa";
	CPlot2D *plot2Db = new CPlot2D(title2);
	plot2Db->SetXAxisSize(600);
	plot2Db->SetYAxisSize(400);
	MDfsc.addToCPlot2D(plot2Db, EMDL_RESOLUTION, EMDL_POSTPROCESS_FSC_TRUE, 0., 0., 0., 2.);
	MDfsc.addToCPlot2D(plot2Db, EMDL_RESOLUTION, EMDL_POSTPROCESS_FSC_PART_FRACMASK, 1., 0.66, 0., 1.);
	if (molweight > 0.)
		MDfsc.addToCPlot2D(plot2Db, EMDL_RESOLUTION, EMDL_POSTPROCESS_FSC_PART_MOLWEIGHT, 0., 1., 1., 2.);
	plot2Db->SetXAxisTitle("resolution (1/A)");
	plot2Db->SetYAxisTitle("Fourier Shell Correlation");
	plot2Db->OutputPostScriptPlot(fn_out + "_fsc_part.eps");
	delete plot2Db;
#endif

	// Also write XML file with FSC_true curve for EMDB submission
	writeFscXml(MDfsc);

	// Write a DAT file for easier plotting in xmgrace (JZ, June 2nd, 2020)
	writeFscDat(MDfsc);

	MDguinier.setName("guinier");
	MetaDataTable MDextra1, MDextra2; // for postscript plot
	for (int i = 0; i < guinierin.size(); i++)
	{
		MDguinier.addObject();
		MDguinier.setValue(EMDL_POSTPROCESS_GUINIER_RESOL_SQUARED, guinierin[i].x);
		MDguinier.setValue(EMDL_POSTPROCESS_GUINIER_VALUE_IN, guinierin[i].y);
		if (fn_mtf != "")
			MDguinier.setValue(EMDL_POSTPROCESS_GUINIER_VALUE_INVMTF, guinierinvmtf[i].y);
		if (do_fsc_weighting)
		{
			MDguinier.setValue(EMDL_POSTPROCESS_GUINIER_VALUE_WEIGHTED, guinierweighted[i].y);
			if (guinierweighted[i].y > -99.)
			{
				MDextra1.addObject();
				MDextra1.setValue(EMDL_POSTPROCESS_GUINIER_RESOL_SQUARED, guinierin[i].x);
				MDextra1.setValue(EMDL_POSTPROCESS_GUINIER_VALUE_WEIGHTED, guinierweighted[i].y);
			}
		}
		if (do_auto_bfac || ABS(adhoc_bfac) > 0.)
		{
			MDguinier.setValue(EMDL_POSTPROCESS_GUINIER_VALUE_SHARPENED, guiniersharpen[i].y);
			if (guiniersharpen[i].y > -99.)
			{
				MDextra2.addObject();
				MDextra2.setValue(EMDL_POSTPROCESS_GUINIER_RESOL_SQUARED, guinierin[i].x);
				MDextra2.setValue(EMDL_POSTPROCESS_GUINIER_VALUE_SHARPENED, guiniersharpen[i].y);
			}
		}
		if (do_auto_bfac)
			MDguinier.setValue(EMDL_POSTPROCESS_GUINIER_VALUE_INTERCEPT, global_intercept);
	}
	MDguinier.write(fh);
	fh.close();

	CPlot2D *plot2Dc = new CPlot2D("Guinier plots");
	plot2Dc->SetXAxisSize(600);
	plot2Dc->SetYAxisSize(400);
	MDguinier.addToCPlot2D(plot2Dc, EMDL_POSTPROCESS_GUINIER_RESOL_SQUARED, EMDL_POSTPROCESS_GUINIER_VALUE_IN, 0., 0., 0.);
	if (fn_mtf != "")
		MDguinier.addToCPlot2D(plot2Dc, EMDL_POSTPROCESS_GUINIER_RESOL_SQUARED, EMDL_POSTPROCESS_GUINIER_VALUE_INVMTF, 0., 1., 0.);
	if (do_fsc_weighting)
	{
		MDextra1.addToCPlot2D(plot2Dc, EMDL_POSTPROCESS_GUINIER_RESOL_SQUARED, EMDL_POSTPROCESS_GUINIER_VALUE_WEIGHTED, 0., 0., 1.);
	}
	if (do_auto_bfac || ABS(adhoc_bfac) > 0.)
	{
		MDextra2.addToCPlot2D(plot2Dc, EMDL_POSTPROCESS_GUINIER_RESOL_SQUARED, EMDL_POSTPROCESS_GUINIER_VALUE_SHARPENED, 1., 0., 0.);
	}
	plot2Dc->SetXAxisTitle("resolution^2 (1/A^2)");
	plot2Dc->SetYAxisTitle("ln(amplitudes)");
	plot2Dc->OutputPostScriptPlot(fn_out + "_guinier.eps");
	delete plot2Dc;

	FileName fn_log = fn_out.beforeLastOf("/") + "/logfile.pdf";
	std::vector<FileName> fn_eps;
	fn_eps.push_back(fn_out + "_fsc.eps");
	fn_eps.push_back(fn_out + "_fsc_part.eps");
	fn_eps.push_back(fn_out + "_guinier.eps");
	joinMultipleEPSIntoSinglePDF(fn_log, fn_eps);

	if (verb > 0)
	{
		if (do_interpolate)
		{
			std::cout.width(35);
			std::cout << std::left   <<"  + FINAL RESOLUTION: ";
			std::cout << global_resol << " (" << fract_resol << ')' << std::endl;
		}
		else
		{
			std::cout.width(35); std::cout << std::left   <<"  + FINAL RESOLUTION: "; std::cout << global_resol<< std::endl;
		}
	}
}

// This masks I1!
void Postprocessing::writeMaps(FileName fn_root) {
	FileName fn_tmp = fn_root + ".mrc";

	I1.setStatisticsInHeader();
	I1.setSamplingRateInHeader(angpix);
	I1.write(fn_tmp);
	if (verb > 0)
	{
		std::cout.width(35); std::cout << std::left   <<"  + Processed map: "; std::cout << fn_tmp<< std::endl;
	}

	// Also write the masked postprocessed map
	if (fn_mask != "")
	{
		fn_tmp = fn_root + "_masked.mrc";
		I1() *= Im();
		I1.setStatisticsInHeader();
		I1.write(fn_tmp);
		if (verb > 0)
		{
			std::cout.width(35); std::cout << std::left   <<"  + Processed masked map: "; std::cout << fn_tmp<< std::endl;
		}
	}
}

void Postprocessing::writeFscXml(MetaDataTable &MDfsc)
{
	FileName fn_fsc = fn_out + "_fsc.xml";
	std::ofstream  fh;
	fh.open((fn_fsc).c_str(), std::ios::out);
	if (!fh)
		REPORT_ERROR( (std::string)"MetaDataTable::write Cannot write to file: " + fn_fsc);

	fh << "<fsc title=\"RELION masked-corrected FSC\" xaxis=\"Resolution (A-1)\" yaxis=\"Correlation Coefficient\">"<<std::endl;

	FOR_ALL_OBJECTS_IN_METADATA_TABLE(MDfsc)
	{
		RFLOAT xx, yy;
		MDfsc.getValue(EMDL_RESOLUTION, xx);
		MDfsc.getValue(EMDL_POSTPROCESS_FSC_TRUE, yy);
		fh << "  <coordinate>" << std::endl;
		fh << "    <x>" << xx << "</x>" << std::endl;
		fh << "    <y>" << yy << "</y>" << std::endl;
		fh << "  </coordinate>" << std::endl;
	}
	fh << "</fsc>" << std::endl;
	fh.close();
}

void Postprocessing::writeFscDat(MetaDataTable &MDfsc)
{
	const std::string fn_fsc = fn_out + "_fsc.dat";

	std::ofstream fh(fn_fsc);

	if (!fh)
	{
		REPORT_ERROR( (std::string)"MetaDataTable::write Cannot write to file: " + fn_fsc);
	}

	for (int i = 0; i < MDfsc.numberOfObjects(); i++)
	{
		const RFLOAT xx = MDfsc.getRfloat(EMDL_RESOLUTION, i);
		const RFLOAT yy = MDfsc.getRfloat(EMDL_POSTPROCESS_FSC_TRUE, i);

		fh << xx << ' ' << yy << '\n';
	}
}

void Postprocessing::run_locres(int rank, int size)
{
	// Read input maps and perform some checks
	initialise();

	const long int nx = XSIZE(I1()), ny = YSIZE(I1()), nz = ZSIZE(I1());
	const long int nfx = nx / 2 + 1;
	int ranks_here = 1;
	if (size > 1)
	{
		MPI_Comm node;
		MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
		MPI_Comm_size(node, &ranks_here);
		MPI_Comm_free(&node);
	}
	const int setup_threads = (locres_threads > 0) ? locres_threads : std::max(1, omp_get_num_procs() / ranks_here);
	relionFftwPlanThreads(setup_threads);

	MultidimArray<RFLOAT> I1p, I2p, Isum, Ilocres, Ifil, Isumw;

	// Get sum of two half-maps and sharpen according to estimated or ad-hoc B-factor
	Isum.resize(I1());
	I1p.resize(I1());
	I2p.resize(I1());
	// Initialise local-resolution maps, weights etc
	Ifil.initZeros(I1());
	Ilocres.initZeros(I1());
	Isumw.initZeros(I1());
	{
		const long int nr_vox = MULTIDIM_SIZE(I1());
		const RFLOAT * __restrict p1 = MULTIDIM_ARRAY(I1());
		const RFLOAT * __restrict p2 = MULTIDIM_ARRAY(I2());
		RFLOAT * __restrict ps = MULTIDIM_ARRAY(Isum);
		RFLOAT * __restrict pp1 = MULTIDIM_ARRAY(I1p);
		RFLOAT * __restrict pp2 = MULTIDIM_ARRAY(I2p);
		#pragma omp parallel for num_threads(setup_threads) if(setup_threads > 1)
		for (long int n = 0; n < nr_vox; n++)
		{
			ps[n] = 0.5 * (p1[n] + p2[n]);
			pp1[n] = p1[n];
			pp2[n] = p2[n];
		}
	}

	// Pre-sharpen the sum of the two half-maps with the provided MTF curve and adhoc B-factor
	do_fsc_weighting = false;
	MultidimArray<Complex > FTsum;
	FourierTransformer transformer;
	transformer.FourierTransform(Isum, FTsum, true);
	Isum.clear();
	divideByMtf(FTsum);
	applyBFactorToMap(FTsum, nx, adhoc_bfac, angpix);

	// Step size of locres-sampling in pixels
	int step_size = ROUND(locres_sampling / angpix);
	int maskrad_pix = ROUND(locres_maskrad / angpix);
	int edgewidth_pix = ROUND(locres_edgwidth / angpix);

	// Get the unmasked FSC curve
	getFSC(I1(), I2(), fsc_unmasked);

	// Randomize phases of unmasked maps from user-provided resolution
	int randomize_at = nx * angpix / locres_randomize_fsc;
	if (verb > 0)
	{
		std::cout.width(35); std::cout << std::left << "  + randomize phases beyond: "; std::cout << nx * angpix / randomize_at << " Angstroms" << std::endl;
	}
	// Randomize phases
	randomizePhasesBeyond(I1p, randomize_at);
	randomizePhasesBeyond(I2p, randomize_at);

	// Write an output STAR file with FSC curves, Guinier plots etc
	FileName fn_tmp = fn_out + "_locres_fscs.star";
	std::vector<char> fh_buf;
	std::ofstream fh;
	if (rank == 0)
	{
		if (verb > 0)
		{
			std::cout.width(35); std::cout << std::left <<"  + Metadata output file: "; std::cout << fn_tmp<< std::endl;
		}

		fh_buf.resize(1 << 20);
		fh.rdbuf()->pubsetbuf(fh_buf.data(), fh_buf.size());
		fh.open((fn_tmp).c_str(), std::ios::out);
		if (!fh)
			REPORT_ERROR( (std::string)"MlOptimiser::write: Cannot write file: " + fn_tmp);
	}

	// Sample the entire volume (within the provided mask)
	int myrad = nx / 2 - maskrad_pix;
	float myradf = (float)myrad/(float)step_size;
	long int nr_samplings = ROUND((4.* PI / 3.) * (myradf*myradf*myradf));

	// The sampling points of this rank, in the order of the loop over the volume
	struct Point { long int kk, ii, jj, nn; };
	std::vector<Point> points;
	long int nn = 0;
	for (long int kk=((I1()).zinit); kk<=((I1()).zinit + (I1()).zdim - 1); kk+= step_size)
		for (long int ii=((I1()).yinit); ii<=((I1()).yinit + (I1()).ydim - 1); ii+= step_size)
			for (long int jj=((I1()).xinit); jj<=((I1()).xinit + (I1()).xdim - 1); jj+= step_size)
			{
				// Only calculate local-resolution inside a spherical mask with radius less than half-box-size minus maskrad_pix
				float rad = sqrt(kk*kk + ii*ii + jj*jj);
				if (rad < myrad)
				{
					if (nn%size == rank)
						points.push_back({kk, ii, jj, nn});
					nn++;
				}
			}

	const RFLOAT radius = maskrad_pix, radius_p = maskrad_pix + edgewidth_pix;
	const long int R = (long int)radius_p + 1;
	const long int S = 2 * R + 1;
	const double per_thread_bytes = 2.0 * (double)ny * nfx * S * sizeof(Complex) + (double)S * nfx * ny * sizeof(Complex);
	int n_threads = locres_threads;
	std::string why = "--j";
	if (n_threads < 1)
	{
		const int by_cores = std::max(1, omp_get_num_procs() / ranks_here);
		const long long avail = asyncwrite::memoryAvailableBytes();
		const int by_memory = (avail < 0 || per_thread_bytes <= 0.) ? by_cores : std::max(1, (int)(0.7 * avail / ranks_here / per_thread_bytes));
		n_threads = std::min(by_cores, by_memory);
		why = (by_memory < by_cores) ? "free memory" : "cores";
	}
	n_threads = (int)std::max(1L, std::min((long int)n_threads, (long int)points.size()));

	// Precompute the spherical raised-cosine template once
	std::vector<RFLOAT> mask_tmpl((size_t)S * S * S, 0.);
	for (long int dk = -R; dk <= R; dk++)
	for (long int di = -R; di <= R; di++)
	for (long int dj = -R; dj <= R; dj++)
	{
		const RFLOAT d = sqrt((RFLOAT)(dk * dk + di * di + dj * dj));
		RFLOAT val = 0.;
		if (d > radius_p) val = 0.;
		else if (d < radius) val = 1.;
		else val = 0.5 - 0.5 * cos(PI * (radius_p - d) / (radius_p - radius));
		mask_tmpl[((size_t)(dk + R) * S + (di + R)) * S + (dj + R)] = val;
	}

	// Transpose FTsum once into [ny][nfx][nz] so every pruned Z column inverse is contiguous
	std::vector<Complex> FTsum_yxz((size_t)ny * nfx * nz);
	#pragma omp parallel for collapse(2) num_threads(n_threads) if(n_threads > 1)
	for (long int ky = 0; ky < ny; ky++)
	{
		for (long int fx = 0; fx < nfx; fx++)
		{
			Complex * __restrict dst = &FTsum_yxz[((size_t)ky * nfx + fx) * nz];
			for (long int kz = 0; kz < nz; kz++)
				dst[kz] = DIRECT_A3D_ELEM(FTsum, kz, ky, fx);
		}
	}
	FTsum.clear();

	const long int sx0 = STARTINGX(I1()), sy0 = STARTINGY(I1()), sz0 = STARTINGZ(I1());
	const long int nfx2 = nfx * nfx;
	std::vector<int> r2_to_shell((size_t)nfx2);
	for (long int d2 = 0; d2 < nfx2; d2++)
		r2_to_shell[d2] = ROUND(sqrt((RFLOAT)d2));

	auto finalizePoint = [&](MultidimArray<RFLOAT> &fsc_masked,
	                         MultidimArray<RFLOAT> &fsc_random_masked,
	                         MultidimArray<RFLOAT> &fsc_unm,
	                         MultidimArray<RFLOAT> &fsc_true,
	                         float &local_resol,
	                         std::vector<RFLOAT> &fsc_w,
	                         std::vector<RFLOAT> &inv_w,
	                         long int &r2_bound,
	                         long int &n_fx_inv)
	{
		// FSC_true = FSC_t - FSC_n / ( ); it sets element 0 of the unmasked FSC, so each point gets a copy
		fsc_unm = fsc_unmasked;
		calculateFSCtrue(fsc_true, fsc_unm, fsc_masked, fsc_random_masked, randomize_at);

		local_resol = 999.;
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(fsc_true)
		{
			if (DIRECT_A1D_ELEM(fsc_true, i) < 0.143)
				break;
			local_resol = (i > 0) ? nx * angpix / (RFLOAT)i : 999.;
		}
		local_resol = XMIPP_MIN(locres_minres, local_resol);

		int ires_max = 0;
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(fsc_true)
		{
			if (DIRECT_A1D_ELEM(fsc_true, i) < 0.0001)
				break;
			ires_max = i;
		}
		fsc_w.resize((size_t)nfx);
		for (int i = 0; i <= ires_max; i++)
		{
			const RFLOAT fsc = DIRECT_A1D_ELEM(fsc_true, i);
			fsc_w[i] = (fsc > 0.) ? sqrt((2. * fsc) / (1. + fsc)) : 0.;
		}
		const int ires_filter = ROUND((nx * angpix) / local_resol);
		const int filter_edge_halfwidth = filter_edge_width / 2;
		const RFLOAT edge_low = XMIPP_MAX(0., (ires_filter - filter_edge_halfwidth) / (RFLOAT)nx);
		const RFLOAT edge_high = XMIPP_MIN((RFLOAT)nfx, (ires_filter + filter_edge_halfwidth) / (RFLOAT)nx);
		const RFLOAT edge_width = edge_high - edge_low;
		const RFLOAT r_hi = std::min((RFLOAT)ires_max + 0.5, edge_high * (RFLOAT)nx) + 1e-6;
		r2_bound = (long int)ceil(r_hi * r_hi);
		n_fx_inv = std::min(nfx, (long int)ceil(r_hi) + 1);

		inv_w.assign((size_t)r2_bound + 1, 0.);
		for (long int d2 = 0; d2 <= r2_bound; d2++)
		{
			const RFLOAT r = sqrt((RFLOAT)d2);
			const int ires = ROUND(r);
			const RFLOAT res_f = r / nx;
			if (ires <= ires_max && res_f <= edge_high)
			{
				const RFLOAT w_lp = (res_f < edge_low) ? 1.0 : ((edge_width > 0.) ? (0.5 + 0.5 * cos(PI * (res_f - edge_low) / edge_width)) : 1.0);
				inv_w[d2] = fsc_w[ires] * w_lp;
			}
		}
	};

	auto writePointStar = [&](long int kk, long int ii, long int jj, float local_resol,
	                          const MultidimArray<RFLOAT> &fsc_true,
	                          const MultidimArray<RFLOAT> &fsc_unm,
	                          const MultidimArray<RFLOAT> &fsc_masked,
	                          const MultidimArray<RFLOAT> &fsc_random_masked)
	{
		if (rank != 0) return;
		MetaDataTable MDfsc;
		FileName fn_name = "fsc_"+integerToString(kk, 5)+"_"+integerToString(ii, 5)+"_"+integerToString(jj, 5);
		MDfsc.setName(fn_name);
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(fsc_true)
		{
			MDfsc.addObject();
			RFLOAT resol = (i > 0) ? (nx * angpix / (RFLOAT)i) : 999.;
			MDfsc.setValue(EMDL_SPECTRAL_IDX, (int)i);
			MDfsc.setValue(EMDL_RESOLUTION, 1./resol);
			MDfsc.setValue(EMDL_RESOLUTION_ANGSTROM, resol);
			MDfsc.setValue(EMDL_POSTPROCESS_FSC_TRUE, DIRECT_A1D_ELEM(fsc_true, i) );
			MDfsc.setValue(EMDL_POSTPROCESS_FSC_UNMASKED, DIRECT_A1D_ELEM(fsc_unm, i) );
			MDfsc.setValue(EMDL_POSTPROCESS_FSC_MASKED, DIRECT_A1D_ELEM(fsc_masked, i) );
			MDfsc.setValue(EMDL_POSTPROCESS_FSC_RANDOM_MASKED, DIRECT_A1D_ELEM(fsc_random_masked, i) );
		}
		MDfsc.write(fh);
		fh << " kk= " << kk << " ii= " << ii << " jj= " << jj << " local resolution= " << local_resol << std::endl;
	};

	bool used_gpu = false;
	if (gpulocres::requested(do_gpu, dont_gpu) && !points.empty())
	{
		const int gdev = gpulocres::selectDevice(gpu_ids, rank, size, verb);
		gpulocres::LocresGpuSession gpu_sess;
		std::string gpu_reason;
		if (gdev >= 0)
		{
			gpulocres::LocresInitData in;
			in.I1 = MULTIDIM_ARRAY(I1());
			in.I2 = MULTIDIM_ARRAY(I2());
			in.I1p = MULTIDIM_ARRAY(I1p);
			in.I2p = MULTIDIM_ARRAY(I2p);
			in.FTsum_yxz = (const RFLOAT*)FTsum_yxz.data();
			in.mask_tmpl = mask_tmpl.data();
			in.r2_to_shell = r2_to_shell.data();
			in.nx = nx; in.ny = ny; in.nz = nz; in.nfx = nfx; in.S = S;
			in.double_precision = gpulocres::doublePrecision();
			if (gpu_sess.init(gdev, in, gpu_reason))
			{
				used_gpu = true;
				if (verb > 0)
				{
					std::cout << " Calculating local resolution in " << nr_samplings
					          << " sampling points on GPU (" << (in.double_precision ? "FP64" : "FP32") << ") ..." << std::endl;
					init_progress_bar(nr_samplings);
				}
				std::vector<double> stats((size_t)6 * nfx);
				std::vector<RFLOAT> fsc_w(nfx), inv_w;
				MultidimArray<RFLOAT> fsc_masked, fsc_random_masked, fsc_unm, fsc_true;
				fsc_masked.initZeros(nfx);
				fsc_random_masked.initZeros(nfx);
				for (size_t ip = 0; ip < points.size(); ip++)
				{
					if (pipeline_control_check_abort_job())
						exit(RELION_EXIT_ABORTED);
					const Point &pt = points[ip];
					const long int x = pt.kk, y = pt.ii, z = pt.jj;
					const long int z0 = std::max(sz0, z - R), z1 = std::min(FINISHINGZ(I1()), z + R);
					const long int y0 = std::max(sy0, y - R), y1 = std::min(FINISHINGY(I1()), y + R);
					const long int x0 = std::max(sx0, x - R), x1 = std::min(FINISHINGX(I1()), x + R);
					gpulocres::LocresPointBox box;
					box.kz0 = z0 - sz0; box.cz = z1 - z0 + 1;
					box.ky0 = y0 - sy0; box.cy = y1 - y0 + 1;
					box.kx0 = x0 - sx0; box.cx = x1 - x0 + 1;
					box.dm_z0 = z0 - z + R; box.dm_y0 = y0 - y + R; box.dm_x0 = x0 - x + R;

					if (!gpu_sess.computePointFsc(box, stats.data(), gpu_reason))
						REPORT_ERROR("LocRes GPU computePointFsc failed: " + gpu_reason);

					for (long int i = 0; i < nfx; i++)
					{
						DIRECT_A1D_ELEM(fsc_masked, i) = stats[i] / sqrt(stats[nfx + i] * stats[2 * nfx + i]);
						DIRECT_A1D_ELEM(fsc_random_masked, i) = stats[3 * nfx + i] / sqrt(stats[4 * nfx + i] * stats[5 * nfx + i]);
					}

					float local_resol = 999.;
					long int r2_bound = 0, n_fx_inv = 0;
					finalizePoint(fsc_masked, fsc_random_masked, fsc_unm, fsc_true,
					              local_resol, fsc_w, inv_w, r2_bound, n_fx_inv);
					if (!gpu_sess.accumFilteredPoint(box, r2_bound, n_fx_inv, inv_w.data(),
					                                   1.0 / (double)local_resol, gpu_reason))
						REPORT_ERROR("LocRes GPU accumFilteredPoint failed: " + gpu_reason);

					writePointStar(pt.kk, pt.ii, pt.jj, local_resol, fsc_true, fsc_unm, fsc_masked, fsc_random_masked);

					if (verb > 0 && ((ip + 1) % 16 == 0 || ip + 1 == points.size()))
						progress_bar(std::min(pt.nn + 1, nr_samplings));
				}
				if (!gpu_sess.finishVolumes(MULTIDIM_ARRAY(Ifil), MULTIDIM_ARRAY(Ilocres), MULTIDIM_ARRAY(Isumw), gpu_reason))
					REPORT_ERROR("LocRes GPU finishVolumes failed: " + gpu_reason);
			}
			else if (verb > 0)
			{
				std::cerr << " + LocRes GPU unavailable (" << gpu_reason << "); falling back to CPU." << std::endl;
			}
		}
		else if (verb > 0 && do_gpu)
		{
			std::cerr << " + LocRes GPU unavailable; falling back to CPU." << std::endl;
		}
	}

	if (!used_gpu && !points.empty())
	{
		if (verb > 0)
		{
			std::cout << " Calculating local resolution in " << nr_samplings << " sampling points on " << n_threads
			          << " thread(s) per process (from the " << why << "; --j sets it) ..." << std::endl;
			init_progress_bar(nr_samplings);
		}

		// Plan 6 single-threaded 1D FFTW plans once with FFTW_MEASURE | FFTW_UNALIGNED before the sampling loop
		relionFftwPlanThreads(1);
#ifdef RELION_SINGLE_PRECISION
		#define LR_FFTW(f) fftwf_##f
		typedef fftwf_complex lr_fftw_complex;
		typedef fftwf_plan lr_fftw_plan;
#else
		#define LR_FFTW(f) fftw_##f
		typedef fftw_complex lr_fftw_complex;
		typedef fftw_plan lr_fftw_plan;
#endif
		struct LocresPlans
		{
			lr_fftw_plan x_fwd = NULL, y_fwd = NULL, z_fwd = NULL;
			lr_fftw_plan z_bwd = NULL, y_bwd = NULL, x_bwd = NULL;
			~LocresPlans()
			{
				#pragma omp critical(FourierTransformer_fftw_plan)
				{
					if (x_fwd) LR_FFTW(destroy_plan)(x_fwd);
					if (y_fwd) LR_FFTW(destroy_plan)(y_fwd);
					if (z_fwd) LR_FFTW(destroy_plan)(z_fwd);
					if (z_bwd) LR_FFTW(destroy_plan)(z_bwd);
					if (y_bwd) LR_FFTW(destroy_plan)(y_bwd);
					if (x_bwd) LR_FFTW(destroy_plan)(x_bwd);
				}
			}
		} plans;
		{
			const long int max_dim = std::max(nx, std::max(ny, nz));
			std::vector<RFLOAT> dr(nx, 0.);
			std::vector<Complex> dc1(max_dim), dc2(max_dim);
			#pragma omp critical(FourierTransformer_fftw_plan)
			{
				plans.x_fwd = LR_FFTW(plan_dft_r2c_1d)((int)nx, dr.data(), (lr_fftw_complex*)dc1.data(), FFTW_MEASURE | FFTW_UNALIGNED);
				plans.y_fwd = LR_FFTW(plan_dft_1d)((int)ny, (lr_fftw_complex*)dc1.data(), (lr_fftw_complex*)dc2.data(), FFTW_FORWARD, FFTW_MEASURE | FFTW_UNALIGNED);
				plans.z_fwd = LR_FFTW(plan_dft_1d)((int)nz, (lr_fftw_complex*)dc1.data(), (lr_fftw_complex*)dc2.data(), FFTW_FORWARD, FFTW_MEASURE | FFTW_UNALIGNED);
				plans.z_bwd = LR_FFTW(plan_dft_1d)((int)nz, (lr_fftw_complex*)dc1.data(), (lr_fftw_complex*)dc2.data(), FFTW_BACKWARD, FFTW_MEASURE | FFTW_UNALIGNED);
				plans.y_bwd = LR_FFTW(plan_dft_1d)((int)ny, (lr_fftw_complex*)dc1.data(), (lr_fftw_complex*)dc2.data(), FFTW_BACKWARD, FFTW_MEASURE | FFTW_UNALIGNED);
				plans.x_bwd = LR_FFTW(plan_dft_c2r_1d)((int)nx, (lr_fftw_complex*)dc1.data(), dr.data(), FFTW_MEASURE | FFTW_UNALIGNED);
			}
			if (!plans.x_fwd || !plans.y_fwd || !plans.z_fwd || !plans.z_bwd || !plans.y_bwd || !plans.x_bwd)
				REPORT_ERROR("LocRes 1D FFTW plans cannot be created");
		}

		struct Work
		{
			std::vector<RFLOAT> rx;
			std::vector<Complex> cx, stageX, colY, outY, colZ1, outZ1, colZ2, outZ2;
			std::vector<Complex> tmpZ1, tmpZ2, invZ, invY;
			std::vector<RFLOAT> num, den1, den2, fsc_w, inv_w;
		};
		struct Result
		{
			MultidimArray<RFLOAT> fsc_true, fsc_masked, fsc_random_masked, fsc_unm;
			float local_resol;
			long int z0, z1, y0, y1, x0, x1;
			std::vector<RFLOAT> mask, filtered;
		};
		std::vector<Work> work(n_threads);
		for (Work &w : work)
		{
			w.rx.resize(nx);
			w.cx.resize(nfx);
			w.stageX.resize((size_t)nfx * S);
			w.colY.resize(ny);
			w.outY.resize(ny);
			w.colZ1.resize(nz);
			w.outZ1.resize(nz);
			w.colZ2.resize(nz);
			w.outZ2.resize(nz);
			w.tmpZ1.resize((size_t)ny * nfx * S);
			w.tmpZ2.resize((size_t)ny * nfx * S);
			w.invZ.resize((size_t)S * nfx * ny);
			w.invY.resize((size_t)S * S * nfx);
			w.num.resize(nfx);
			w.den1.resize(nfx);
			w.den2.resize(nfx);
			w.fsc_w.resize(nfx);
		}

		auto forwardXY = [&](Work &w, const MultidimArray<RFLOAT> &vol,
		                     long int kx0, long int cx, long int ky0, long int cy, long int kz0, long int cz,
		                     long int dm_x0, long int dm_y0, long int dm_z0, std::vector<Complex> &tmpZ)
		{
			RFLOAT * __restrict rx = w.rx.data();
			Complex * __restrict cxp = w.cx.data();
			Complex * __restrict stageX = w.stageX.data();
			Complex * __restrict colY = w.colY.data();
			Complex * __restrict outY = w.outY.data();
			Complex * __restrict tz = tmpZ.data();
			for (long int dz = 0; dz < cz; dz++)
			{
				for (long int dy = 0; dy < cy; dy++)
				{
					std::memset(rx, 0, (size_t)nx * sizeof(RFLOAT));
					const RFLOAT * __restrict vrow = &DIRECT_A3D_ELEM(vol, kz0 + dz, ky0 + dy, kx0);
					const RFLOAT * __restrict mrow = &mask_tmpl[((size_t)(dm_z0 + dz) * S + (dm_y0 + dy)) * S + dm_x0];
					RFLOAT * __restrict rx_in = rx + kx0;
					for (long int dx = 0; dx < cx; dx++)
						rx_in[dx] = vrow[dx] * mrow[dx];
					LR_FFTW(execute_dft_r2c)(plans.x_fwd, rx, (lr_fftw_complex*)cxp);
					for (long int fx = 0; fx < nfx; fx++)
						stageX[(size_t)fx * cy + dy] = cxp[fx];
				}
				for (long int fx = 0; fx < nfx; fx++)
				{
					std::memset(colY, 0, (size_t)ny * sizeof(Complex));
					std::memcpy(colY + ky0, stageX + (size_t)fx * cy, (size_t)cy * sizeof(Complex));
					LR_FFTW(execute_dft)(plans.y_fwd, (lr_fftw_complex*)colY, (lr_fftw_complex*)outY);
					for (long int ky = 0; ky < ny; ky++)
						tz[((size_t)ky * nfx + fx) * cz + dz] = outY[ky];
				}
			}
		};

		auto prunedMaskedFsc = [&](Work &w, const MultidimArray<RFLOAT> &A, const MultidimArray<RFLOAT> &B,
		                           long int kx0, long int cx, long int ky0, long int cy, long int kz0, long int cz,
		                           long int dm_x0, long int dm_y0, long int dm_z0, MultidimArray<RFLOAT> &fsc)
		{
			forwardXY(w, A, kx0, cx, ky0, cy, kz0, cz, dm_x0, dm_y0, dm_z0, w.tmpZ1);
			forwardXY(w, B, kx0, cx, ky0, cy, kz0, cz, dm_x0, dm_y0, dm_z0, w.tmpZ2);
			std::fill(w.num.begin(), w.num.end(), 0.);
			std::fill(w.den1.begin(), w.den1.end(), 0.);
			std::fill(w.den2.begin(), w.den2.end(), 0.);
			Complex * __restrict colZ1 = w.colZ1.data();
			Complex * __restrict outZ1 = w.outZ1.data();
			Complex * __restrict colZ2 = w.colZ2.data();
			Complex * __restrict outZ2 = w.outZ2.data();
			RFLOAT * __restrict num = w.num.data();
			RFLOAT * __restrict den1 = w.den1.data();
			RFLOAT * __restrict den2 = w.den2.data();
			const int * __restrict r2tab = r2_to_shell.data();
			for (long int ky = 0; ky < ny; ky++)
			{
				const long int ip = (ky < nfx) ? ky : ky - ny;
				const long int ip2 = ip * ip;
				for (long int fx = 0; fx < nfx; fx++)
				{
					const long int k2i2 = ip2 + fx * fx;
					if (k2i2 >= nfx2) continue;
					const size_t off = ((size_t)ky * nfx + fx) * cz;
					std::memset(colZ1, 0, (size_t)nz * sizeof(Complex));
					std::memcpy(colZ1 + kz0, w.tmpZ1.data() + off, (size_t)cz * sizeof(Complex));
					LR_FFTW(execute_dft)(plans.z_fwd, (lr_fftw_complex*)colZ1, (lr_fftw_complex*)outZ1);
					std::memset(colZ2, 0, (size_t)nz * sizeof(Complex));
					std::memcpy(colZ2 + kz0, w.tmpZ2.data() + off, (size_t)cz * sizeof(Complex));
					LR_FFTW(execute_dft)(plans.z_fwd, (lr_fftw_complex*)colZ2, (lr_fftw_complex*)outZ2);
					for (long int kz = 0; kz < nz; kz++)
					{
						const long int kp = (kz < nfx) ? kz : kz - nz;
						const long int d2 = k2i2 + kp * kp;
						if (d2 >= nfx2) continue;
						const int idx = r2tab[d2];
						if (idx >= nfx) continue;
						const Complex z1 = outZ1[kz];
						const Complex z2 = outZ2[kz];
						num[idx] += z1.real * z2.real + z1.imag * z2.imag;
						den1[idx] += z1.real * z1.real + z1.imag * z1.imag;
						den2[idx] += z2.real * z2.real + z2.imag * z2.imag;
					}
				}
			}
			fsc.initZeros(nfx);
			for (long int i = 0; i < nfx; i++)
				DIRECT_A1D_ELEM(fsc, i) = num[i] / sqrt(den1[i] * den2[i]);
		};

		const int batch = 2 * n_threads;
		std::vector<Result> results(batch);
		long int nn_done = 0;
		for (size_t first = 0; first < points.size(); first += batch)
		{
			// Abort through the pipeline_control system, TODO: check how this goes with MPI....
			if (pipeline_control_check_abort_job())
				exit(RELION_EXIT_ABORTED);

			const size_t last = std::min(points.size(), first + batch);
			#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 1)
			for (size_t ip = first; ip < last; ip++)
			{
				Work &w = work[omp_get_thread_num()];
				Result &res = results[ip - first];
				const Point &pt = points[ip];

				// Make a spherical mask around (k,i,j), diameter is step_size pixels, soft-edge width is edgewidth_pix
				// (raisedCosineMask(locmask, radius, radius_p, kk, ii, jj): its x is kk and its z is jj)
				const long int x = pt.kk, y = pt.ii, z = pt.jj;
				res.z0 = std::max(sz0, z - R); res.z1 = std::min(FINISHINGZ(I1()), z + R);
				res.y0 = std::max(sy0, y - R); res.y1 = std::min(FINISHINGY(I1()), y + R);
				res.x0 = std::max(sx0, x - R); res.x1 = std::min(FINISHINGX(I1()), x + R);
				const long int kz0 = res.z0 - sz0, cz = res.z1 - res.z0 + 1;
				const long int ky0 = res.y0 - sy0, cy = res.y1 - res.y0 + 1;
				const long int kx0 = res.x0 - sx0, cx = res.x1 - res.x0 + 1;
				const long int dm_z0 = res.z0 - z + R, dm_y0 = res.y0 - y + R, dm_x0 = res.x0 - x + R;

				// FSC of masked maps and masked randomized-phase maps via pruned separable 3D FFTs
				prunedMaskedFsc(w, I1(), I2(), kx0, cx, ky0, cy, kz0, cz, dm_x0, dm_y0, dm_z0, res.fsc_masked);
				prunedMaskedFsc(w, I1p,  I2p,  kx0, cx, ky0, cy, kz0, cz, dm_x0, dm_y0, dm_z0, res.fsc_random_masked);

				long int r2_bound = 0, n_fx_inv = 0;
				finalizePoint(res.fsc_masked, res.fsc_random_masked, res.fsc_unm, res.fsc_true,
				              res.local_resol, w.fsc_w, w.inv_w, r2_bound, n_fx_inv);

				// Fused applyFscWeighting + lowPassFilterMap + pruned 3D inverse c2r FFT into [kz0..kz1]x[ky0..ky1]x[kx0..kx1]
				std::memset(w.invZ.data(), 0, (size_t)cz * n_fx_inv * ny * sizeof(Complex));
				Complex * __restrict colZ1 = w.colZ1.data();
				Complex * __restrict outZ1 = w.outZ1.data();
				const RFLOAT * __restrict inv_w = w.inv_w.data();
				for (long int ky = 0; ky < ny; ky++)
				{
					const long int ipy = (ky < nfx) ? ky : ky - ny;
					const long int ip2 = ipy * ipy;
					for (long int fx = 0; fx < n_fx_inv; fx++)
					{
						const long int ip2fx2 = ip2 + fx * fx;
						if (ip2fx2 > r2_bound) continue;
						std::memset(colZ1, 0, (size_t)nz * sizeof(Complex));
						const Complex * __restrict ft_col = &FTsum_yxz[((size_t)ky * nfx + fx) * nz];
						const long int kz_max = (long int)floor(sqrt((RFLOAT)(r2_bound - ip2fx2)));
						const long int kz_pos_end = std::min((long int)nfx - 1, kz_max);
						for (long int kz = 0; kz <= kz_pos_end; kz++)
						{
							const RFLOAT wt = inv_w[ip2fx2 + kz * kz];
							if (wt != 0.) colZ1[kz] = ft_col[kz] * wt;
						}
						const long int kz_neg_beg = std::max(nfx, nz - kz_max);
						for (long int kz = kz_neg_beg; kz < nz; kz++)
						{
							const long int kp = kz - nz;
							const RFLOAT wt = inv_w[ip2fx2 + kp * kp];
							if (wt != 0.) colZ1[kz] = ft_col[kz] * wt;
						}
						LR_FFTW(execute_dft)(plans.z_bwd, (lr_fftw_complex*)colZ1, (lr_fftw_complex*)outZ1);
						for (long int dz = 0; dz < cz; dz++)
							w.invZ[((size_t)dz * n_fx_inv + fx) * ny + ky] = outZ1[kz0 + dz];
					}
				}

				std::memset(w.invY.data(), 0, (size_t)cz * cy * nfx * sizeof(Complex));
				Complex * __restrict outY = w.outY.data();
				for (long int dz = 0; dz < cz; dz++)
				{
					for (long int fx = 0; fx < n_fx_inv; fx++)
					{
						LR_FFTW(execute_dft)(plans.y_bwd, (lr_fftw_complex*)&w.invZ[((size_t)dz * n_fx_inv + fx) * ny], (lr_fftw_complex*)outY);
						for (long int dy = 0; dy < cy; dy++)
							w.invY[((size_t)dz * cy + dy) * nfx + fx] = outY[ky0 + dy];
					}
				}

				const size_t cube_n = (size_t)cz * cy * cx;
				res.mask.resize(cube_n);
				res.filtered.resize(cube_n);
				RFLOAT * __restrict rx = w.rx.data();
				for (long int dz = 0; dz < cz; dz++)
				{
					for (long int dy = 0; dy < cy; dy++)
					{
						const size_t row_off = ((size_t)dz * cy + dy);
						LR_FFTW(execute_dft_c2r)(plans.x_bwd, (lr_fftw_complex*)&w.invY[row_off * nfx], rx);
						std::memcpy(res.filtered.data() + row_off * cx, rx + kx0, (size_t)cx * sizeof(RFLOAT));
						std::memcpy(res.mask.data() + row_off * cx,
						            &mask_tmpl[((size_t)(dm_z0 + dz) * S + (dm_y0 + dy)) * S + dm_x0],
						            (size_t)cx * sizeof(RFLOAT));
					}
				}
			}
#undef LR_FFTW

			// Write and add the results in order
			for (size_t ip = first; ip < last; ip++)
			{
				const Point &pt = points[ip];
				const Result &res = results[ip - first];
				writePointStar(pt.kk, pt.ii, pt.jj, res.local_resol, res.fsc_true, res.fsc_unm, res.fsc_masked, res.fsc_random_masked);

				// Store weighted sum of local resolution and filtered map (the mask is zero outside the cube)
				size_t c = 0;
				for (long int k = res.z0; k <= res.z1; k++)
					for (long int i = res.y0; i <= res.y1; i++)
						for (long int j = res.x0; j <= res.x1; j++, c++)
						{
							const RFLOAT m = res.mask[c];
							A3D_ELEM(Ifil, k, i, j) += m * res.filtered[c];
							A3D_ELEM(Ilocres, k, i, j) += m / res.local_resol;
							A3D_ELEM(Isumw, k, i, j) += m;
						}
				nn_done = pt.nn + 1;
			}
			if (verb > 0)
				progress_bar(std::min(nn_done, nr_samplings));
		}
	}

	fh.close();
	if (verb > 0)
		init_progress_bar(nr_samplings);


	if (size > 1)
	{
		MultidimArray<RFLOAT> I1m(I1());
		I1m.initZeros();
		MPI_Reduce(MULTIDIM_ARRAY(Ifil), MULTIDIM_ARRAY(I1m), MULTIDIM_SIZE(Ifil), MY_MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if (rank == 0) Ifil = I1m;
		I1m.initZeros();
		MPI_Reduce(MULTIDIM_ARRAY(Ilocres), MULTIDIM_ARRAY(I1m), MULTIDIM_SIZE(Ilocres), MY_MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if (rank == 0) Ilocres = I1m;
		I1m.initZeros();
		MPI_Reduce(MULTIDIM_ARRAY(Isumw), MULTIDIM_ARRAY(I1m), MULTIDIM_SIZE(Isumw), MY_MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if (rank == 0) Isumw = I1m;
	}

	if (rank == 0)
	{
		// Now write out the local-resolution map and
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Isumw)
		{
			if (DIRECT_MULTIDIM_ELEM(Isumw, n ) > 0.)
			{
				DIRECT_MULTIDIM_ELEM(I1(), n) = 1. / (DIRECT_MULTIDIM_ELEM(Ilocres, n) / DIRECT_MULTIDIM_ELEM(Isumw, n));
				DIRECT_MULTIDIM_ELEM(I2(), n) = DIRECT_MULTIDIM_ELEM(Ifil, n) / DIRECT_MULTIDIM_ELEM(Isumw, n);
			}
			else
			{
				DIRECT_MULTIDIM_ELEM(I1(), n) = 0.;
				DIRECT_MULTIDIM_ELEM(I2(), n) = 0.;
			}
		}

		fn_tmp = fn_out + "_locres.mrc";
		I1.setStatisticsInHeader();
		I1.setSamplingRateInHeader(angpix);
		I1.write(fn_tmp);
		fn_tmp = fn_out + "_locres_filtered.mrc";
		I2.setStatisticsInHeader();
		I2.setSamplingRateInHeader(angpix);
		I2.write(fn_tmp);

#ifdef DEBUG
		I1() = Isumw;
		fn_tmp = fn_out + "_locres_sumw.mrc";
		I1.write(fn_tmp);
#endif

		if (fn_mask != "")
		{
			std::cout << "Calculating a histogram of local resolutions within the mask." << std::endl;

			std::vector<RFLOAT> values;
			Image<RFLOAT> Imask;
			Imask.read(fn_mask);

			if (!I1().sameShape(Imask(), true))
				REPORT_ERROR("The shape of the input half maps and the mask is not the same.");

			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Imask())
				if (DIRECT_MULTIDIM_ELEM(Imask(), n) > 0.5)
					values.push_back(DIRECT_MULTIDIM_ELEM(I1(), n));

			std::vector <RFLOAT> histX, histY;
			CPlot2D *plot2D=new CPlot2D("");
                	FileName fn_eps = fn_out + "_histogram.eps";
			MetaDataTable::histogram(values, histX, histY, verb, "local resolution", plot2D);
			plot2D->OutputPostScriptPlot(fn_eps);
			FileName fn_log = fn_out.beforeLastOf("/") + "/histogram.pdf";
			std::vector<FileName> to_convert;
			to_convert.push_back(fn_eps);
			joinMultipleEPSIntoSinglePDF(fn_log, to_convert);
			std::cout << "Written the histogram to " << fn_log << std::endl;
		}
	}

	if (verb > 0)
		std::cout << " done! " << std::endl;

	if (size > 1)
		MPI_Barrier(MPI_COMM_WORLD);
}

void Postprocessing::run()
{
	// Read input maps and perform some checks
	initialise();

	const int nt = relionBulkThreads(MULTIDIM_SIZE(I1()));
	relionFftwPlanThreads(nt);

	// For amplitude correlation curves: first do radial amplitude correction for non-uniform angular distributions
	if (do_ampl_corr)
	{
		correctRadialAmplitudeDistribution(I1());
		correctRadialAmplitudeDistribution(I2());
	}

	// Calculate FSC of the unmask maps
	getFSC(I1(), I2(), fsc_unmasked);
	if (do_ampl_corr)
		getAmplitudeCorrelationAndDifferentialPhaseResidual(I1(), I2(), acorr_unmasked, dpr_unmasked);

	// Check whether we'll do masking
	do_mask = getMask();
	const long long avail = asyncwrite::memoryAvailableBytes();
	const double vol_bytes = (double)MULTIDIM_SIZE(I1()) * sizeof(RFLOAT);
	const bool keep_halves = (do_mask || write_halfmaps) && (avail < 0 || (double)avail > 8.0 * vol_bytes);
	MultidimArray<RFLOAT> I1_orig, I2_orig;
	if (keep_halves)
	{
		I1_orig = I1();
		I2_orig = I2();
	}

	if (do_mask)
	{
		if (verb > 0)
		{
			std::cout <<"== Masking input maps ..." <<std::endl;
		}
		// Mask I1 and I2 and calculated fsc_masked
		{
			const long int nr_vox = MULTIDIM_SIZE(I1());
			RFLOAT * __restrict p1 = MULTIDIM_ARRAY(I1());
			RFLOAT * __restrict p2 = MULTIDIM_ARRAY(I2());
			const RFLOAT * __restrict pm = MULTIDIM_ARRAY(Im());
			#pragma omp parallel for num_threads(nt) if(nt > 1)
			for (long int n = 0; n < nr_vox; n++)
			{
				const RFLOAT m = pm[n];
				p1[n] *= m;
				p2[n] *= m;
			}
		}
		getFSC(I1(), I2(), fsc_masked);
		if (do_ampl_corr)
			getAmplitudeCorrelationAndDifferentialPhaseResidual(I1(), I2(), acorr_masked, dpr_masked);

		// Restore original maps for phase randomisation (in memory when free RAM allows)
		if (keep_halves)
		{
			std::memcpy(MULTIDIM_ARRAY(I1()), MULTIDIM_ARRAY(I1_orig), MULTIDIM_SIZE(I1()) * sizeof(RFLOAT));
			std::memcpy(MULTIDIM_ARRAY(I2()), MULTIDIM_ARRAY(I2_orig), MULTIDIM_SIZE(I2()) * sizeof(RFLOAT));
		}
		else
		{
			I1.read(fn_I1);
			I2.read(fn_I2);
			I1().setXmippOrigin();
			I2().setXmippOrigin();
		}

		// Check at which resolution shell the FSC drops below randomize_fsc_at
		randomize_at = -1;

		if (randomize_at_A > 0.)
		{
			randomize_at = (int)(angpix * XSIZE(I1()) / randomize_at_A );
		}
		else
		{
			// Check when FSC drops below randomize_fsc_at
			FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(fsc_unmasked)
			{
				if (i > 0 && DIRECT_A1D_ELEM(fsc_unmasked, i) < randomize_fsc_at)
				{
					randomize_at = i;
					break;
				}
			}
		}
		if (verb > 0)
		{
			std::cout.width(35); std::cout << std::left << "  + randomize phases beyond: "; std::cout << XSIZE(I1())* angpix / randomize_at << " Angstroms" << std::endl;
		}
		if (randomize_at > 0)
		{
			randomizePhasesBeyond(I1(), randomize_at);
			randomizePhasesBeyond(I2(), randomize_at);
			// Mask randomized phases maps and calculated fsc_random_masked
			{
				const long int nr_vox = MULTIDIM_SIZE(I1());
				RFLOAT * __restrict p1 = MULTIDIM_ARRAY(I1());
				RFLOAT * __restrict p2 = MULTIDIM_ARRAY(I2());
				const RFLOAT * __restrict pm = MULTIDIM_ARRAY(Im());
				#pragma omp parallel for num_threads(nt) if(nt > 1)
				for (long int n = 0; n < nr_vox; n++)
				{
					const RFLOAT m = pm[n];
					p1[n] *= m;
					p2[n] *= m;
				}
			}
			getFSC(I1(), I2(), fsc_random_masked);
		}
		else
			REPORT_ERROR("Postprocessing::run ERROR: FSC curve never drops below randomize_fsc_at.");

		// Now that we have fsc_masked and fsc_random_masked, calculate fsc_true according to Richard's formula
		// FSC_true = FSC_t - FSC_n / ( )
		calculateFSCtrue(fsc_true, fsc_unmasked, fsc_masked, fsc_random_masked, randomize_at);

		// Also calculate cisTEM-like corrected part_FSC based on expected ordered molecular weight
		calculateFSCpart(fsc_unmasked, frac_molweight, fsc_part_molweight);
		// and based on fraction of white voxels in the solvent mask used for RELION-correction
		calculateFSCpart(fsc_unmasked, frac_solvent_mask, fsc_part_fracmask);

		if (!keep_halves)
		{
			I1.read(fn_I1);
			I2.read(fn_I2);
			I1().setXmippOrigin();
			I2().setXmippOrigin();
		}
	}
	else
	{
		fsc_true = fsc_unmasked;
	}

	global_resol = 999.;
	// See where corrected FSC drops below 0.143
	int global_resol_i = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(fsc_true)
	{
		if ( DIRECT_A1D_ELEM(fsc_true, i) < 0.143)
			break;
		global_resol = XSIZE(I1())*angpix/(RFLOAT)i;
		global_resol_i = i;
	}

	// Determine the fractional resolution (needed for development purposes)
	if (do_interpolate && global_resol_i < fsc_true.xdim - 1)
	{
		const double v0 = DIRECT_A1D_ELEM(fsc_true, global_resol_i) - 0.143;
		const double v1 = DIRECT_A1D_ELEM(fsc_true, global_resol_i + 1) - 0.143;
		const double r = v0 / (v0 - v1);

		fract_resol = XSIZE(I1()) * angpix / (global_resol_i + r);
	}
	else
	{
		fract_resol = global_resol;
	}

	// Perform some checks on phase-randomisation..
	if (do_mask)
	{
		int unmasked_resol_i = 0;
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(fsc_unmasked)
		{
			if ( DIRECT_A1D_ELEM(fsc_unmasked, i) < 0.143)
				break;
			unmasked_resol_i = i;
		}
		// Check whether global_resol is worse than the unmasked one
		if (unmasked_resol_i > global_resol_i)
		{
			if (force_mask)
			{
				std::cerr << " WARNING: the unmasked FSC extends beyond the solvent-corrected FSC." << std::endl;
			}
			else
			{
				std::cerr << " WARNING: the unmasked FSC extends beyond the solvent-corrected FSC. Skip masking for now, but you may want to adjust you mask!" << std::endl;
				std::cerr << "          You can force the mask by the '--force_mask' option." << std::endl;
				fsc_true = fsc_unmasked;
				global_resol = XSIZE(I1())*angpix/(RFLOAT)unmasked_resol_i;
			}
		}
		// Check whether the phase-randomised FSC is less than 5% at the resolution estimate, otherwise warn the user
		else if (DIRECT_A1D_ELEM(fsc_random_masked, global_resol_i) > 0.1)
		{
			std::cerr << " WARNING: The phase-randomised FSC is larger than 0.10 at the estimated resolution!" << std::endl;
			std::cerr << " WARNING: This may result in an incorrect resolution estimation. Provide a softer mask with less features to get lower phase-randomised FSCs." << std::endl;
		}
	}

	// Add the two half-maps together for subsequent sharpening
	{
		const long int nr_vox = MULTIDIM_SIZE(I1());
		RFLOAT * __restrict p1 = MULTIDIM_ARRAY(I1());
		const RFLOAT * __restrict src1 = (do_mask && keep_halves) ? MULTIDIM_ARRAY(I1_orig) : p1;
		const RFLOAT * __restrict src2 = (do_mask && keep_halves) ? MULTIDIM_ARRAY(I2_orig) : MULTIDIM_ARRAY(I2());
		#pragma omp parallel for num_threads(nt) if(nt > 1)
		for (long int n = 0; n < nr_vox; n++)
			p1[n] = (src1[n] + src2[n]) * 0.5;
	}
	if (!write_halfmaps)
	{
		I1_orig.clear();
		I2_orig.clear();
	}

	// Divide by MTF and perform FSC-weighted B-factor sharpening, as in Rosenthal and Henderson, 2003
	// also low-pass filters...
	RFLOAT applied_filter = sharpenMap();

	// Write original and corrected FSC curve, Guinier plot, etc.
	writeOutput();

	if (write_halfmaps) {
		std::cout << "== Writing half maps after applying same sharpening and low-pass filter" << std::endl;
		// Force the filtering as merged map
		do_auto_bfac = false;
		adhoc_bfac = global_bfactor;
		low_pass_freq = applied_filter;
		verb = 0;

		std::cout << "  + half1" << std::endl;
		if (keep_halves)
		{
			std::memcpy(MULTIDIM_ARRAY(I1()), MULTIDIM_ARRAY(I1_orig), MULTIDIM_SIZE(I1()) * sizeof(RFLOAT));
			I1_orig.clear();
		}
		else
		{
			I1.read(fn_I1);
			I1().setXmippOrigin();
		}
		sharpenMap();
		writeMaps(fn_out + "_half1");

		std::cout << "  + half2" << std::endl;
		if (keep_halves)
		{
			std::memcpy(MULTIDIM_ARRAY(I1()), MULTIDIM_ARRAY(I2_orig), MULTIDIM_SIZE(I1()) * sizeof(RFLOAT));
			I2_orig.clear();
		}
		else
		{
			I1.read(fn_I2);
			I1().setXmippOrigin();
		}
		sharpenMap();
		writeMaps(fn_out + "_half2");
	}
}
