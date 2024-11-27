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
#include "src/amyloid_finder.h"
//#define DEBUG_BOUNDS

void AmyloidFinder::read(int argc, char **argv, int rank)
{
    parser.setCommandLine(argc, argv);


    int general_section = parser.addSection("General options");
    fn_in = parser.getOption("--i", "Input image (.mrc) or STAR file with micrographs");
    fn_out = parser.getOption("--pickname", "Rootname for coordinate STAR files", "autopick");
    fn_odir = parser.getOption("--odir", "Output directory for coordinate files (default is to store next to micrographs)", "AutoPick/");
    do_only_unfinished = parser.checkOption("--only_do_unfinished", "Only estimate CTFs for those tomograms for which there is not yet a logfile with Final values.");
    do_write_intermediate = parser.checkOption("--write_intermediates", "Write out intermediate FOM & PSI maps for faster testing of tracing parameters?");
    do_only_write_intermediate = parser.checkOption("--only_write_intermediates", "Write out only intermediate FOM & PSI maps?");
    nr_threads = textToInteger(parser.getOption("--j", "Number of threads to us in parallel", "1"));

    int search_section = parser.addSection("Filament searching options ");
    psi_step = textToFloat(parser.getOption("--psi_step", "Angular sampling rate (in degrees)", "10."));
    shift_step = textToInteger(parser.getOption("--shift_step", "Step in shifts to search (in downscaled pixels)", "5"));
    search_filament_width = textToFloat(parser.getOption("--search_filament_width", "Width of searching image (in A)", "50"));
    search_filament_length = textToFloat(parser.getOption("--search_filament_length", "Length of searching image (in A)", "250"));

    int pick_section = parser.addSection("Filament tracing options ");
    zscore_threshold = textToFloat(parser.getOption("--threshold", "Threshold in Z-scores for coordinate picking", "0.5"));
    nr_rungs_per_segment = textToInteger(parser.getOption("--rungs_per_segment", "Number of new amyloid rungs per segment", "3"));
    trace_filament_width = textToFloat(parser.getOption("--trace_filament_width", "Minimum width occupied by a traced filaments (in A)", "200"));
    trace_filament_length = textToFloat(parser.getOption("--trace_filament_length", "Minimum length of traced filaments (in A)", "300"));
    RFLOAT kappa = textToFloat(parser.getOption("--kappa", "Curvature parameter kappa ", "0.07"));
    if (kappa > 0.101)
        REPORT_ERROR("ERROR: for amyloids you cannot use kappa larger than 0.1!");
    psidiff_per_segment = RAD2DEG(kappa*2.);

    int expert_section = parser.addSection("Expert options (typically no need to change)");
    signal_minres = textToFloat(parser.getOption("--signal_minres", "Minimum resolution value for signal (in A)", "4.85"));
    signal_maxres = textToFloat(parser.getOption("--signal_maxres", "Maximum resolution value for signal (in A)", "4.65"));
    inifactor_threshold = textToFloat(parser.getOption("--inifactor_threshold", "What fraction of the Z-score threshold for allowing to grow filaments?", "0.33"));
    down_angpix = textToFloat(parser.getOption("--down_angpix", "Pixel size for downscaled images (needs to include signal frequency!)", "2.25"));
    angpix = textToFloat(parser.getOption("--force_angpix", "Force this pixel size, regardless of what is in the image header", "-1"));
    verb =textToInteger(parser.getOption("--verb", "Verbosity", "1"));

    // Check for errors in the command-line option
    if (parser.checkForErrors())
        REPORT_ERROR("Errors encountered on the command line (see above), exiting...");
}

void AmyloidFinder::usage()
{
    parser.writeUsage(std::cout);
}

void AmyloidFinder::initialise(bool is_leader)
{
    // Make sure fn_odir ends with a slash
    if (fn_odir[fn_odir.length()-1] != '/')
        fn_odir += "/";

    // Make sure inifactor is <0.1]
    if (inifactor_threshold > 1 || inifactor_threshold <= 0.)
        REPORT_ERROR("ERROR: --inifactor_threshold should be between <0,1]");

    fn_micrographs.clear();
    if (fn_in.isStarFile())
    {
        MetaDataTable MDin;
        MDin.read(fn_in, "micrographs");
        FOR_ALL_OBJECTS_IN_METADATA_TABLE(MDin)
        {
            FileName fn_mic;
            MDin.getValue(EMDL_MICROGRAPH_NAME, fn_mic);
            fn_micrographs.push_back(fn_mic);
        }
    }
    else
    {
        // Read a single micrograph
        fn_micrographs.push_back(fn_in);
    }

    fn_ori_micrographs = fn_micrographs;
    // If we're continuing an old run, see which micrographs have not been finished yet...
    if (do_only_unfinished && !do_write_intermediate)
    {
        if (verb > 0)
        {
            std::cout << " + Skipping those micrographs for which coordinate file already exists" << std::endl;
        }

        std::vector<FileName> fns_todo;
        for (long int imic = 0; imic < fn_micrographs.size(); imic++)
        {
            FileName fn_tmp = getOutputRootName(fn_micrographs[imic]) + "_" + fn_out + ".star";
            if (!exists(fn_tmp))
                fns_todo.push_back(fn_micrographs[imic]);
        }

        fn_micrographs = fns_todo;
    }

    // If there is nothing to do, then go out of initialise
    if (fn_micrographs.size() == 0)
    {
        if (verb > 0)
            std::cout << " + No new micrographs to do, so exiting finding amyloids ..." << std::endl;
        return;
    }

    if (verb > 0) std::cout << " + Finding amyloids in the " << fn_micrographs.size() << " micrographs... " << std::endl;

    // Read in header of first image
    Image<RFLOAT> Iin;
    Iin.read(fn_micrographs[0], false);
    ori_xsize = XSIZE(Iin());
    ori_ysize = YSIZE(Iin());
    Iin().setXmippOrigin();
    if (angpix < 0.)
    {
        angpix = Iin.samplingRateX();
        if (verb > 0) std::cout << " - Using pixel size from the header of : " << fn_in << " = " << angpix << std::endl;
    }

    if (signal_maxres < 2*down_angpix) REPORT_ERROR("ERROR: the down_angpix is not enough to support the maximum resolution of the signal!");
    if (angpix > down_angpix) REPORT_ERROR("ERROR: this program requires input images with a pixel size of at least down_angpix (" + floatToString(down_angpix) + ")!");

    // Width and length in the downscaled pixels
    iwidthmax = ROUND(search_filament_width / down_angpix );
    ilengthmax = CEIL(search_filament_length / down_angpix );

    down_xsize = FLOOR( (ori_xsize * angpix) / down_angpix );
    down_ysize = FLOOR( (ori_ysize * angpix) / down_angpix );
    if (ilengthmax %2 != 0) ilengthmax++;
    nr_psi = ROUND(180./psi_step);
    psi_step = 180./nr_psi;

    // Calculate Fourier shells for amyloid signal
    imin_signal = FLOOR(ilengthmax*down_angpix/signal_minres);
    imax_signal = CEIL(ilengthmax*down_angpix/signal_maxres);

    // Box size, orginal and cropped: set size of rectangular image to largest dimension
    large_box = sqrt(2.)*XMIPP_MAX(ori_xsize, ori_ysize);
    large_box += ROUND(XMIPP_MAX(search_filament_width, search_filament_length) / angpix);
    if (large_box%2 != 0) large_box++;
    // Also calculate size of cropped box:
    crop_box = large_box * angpix/down_angpix;
    if (crop_box%2 != 0) crop_box++;

    // Output some information to the user
    if (verb > 0)
    {
        std::cout << " + Number of 1D rows for filament width (in downscaled pixels): " << iwidthmax << std::endl;
        std::cout << " + Length of 1D rows for filament (in downscaled pixels): " << ilengthmax << std::endl;
        std::cout << " + Number of in-plane rotations to sample: " << nr_psi << " with step of " << psi_step << " degrees" << std::endl;
        std::cout << " + Original size of the input micrographs: " << ori_xsize << " x " << ori_ysize << " pixels" << std::endl;
        std::cout << " + Size of image to sample (in downscaled pixels): " <<  down_xsize << " x " << down_ysize << std::endl;
        std::cout << " + Fourier shells for the amyloid signal (in downscaled pixels): " << imin_signal  << " - " << imax_signal << std::endl;
        std::cout << "  ========================== " << std::endl;
   }

    // For filament tracing
    // Set up a vector with coordinates of feasible next coordinates regarding distance and psi-angle
    amyloid_rung = (signal_minres + signal_maxres)/2.;
	int myrad = ROUND(nr_rungs_per_segment * amyloid_rung / angpix);
    int myradb = myrad + 1;
    float myrad2 = (float)myrad * (float)myrad;
    float myradb2 = (float)myradb * (float)myradb;
    for (int ii = -myradb; ii <= myradb; ii++)
    {
    	for (int jj = 0; jj <= myradb; jj++)
    	{
    		float r2 = (float)(ii*ii) + (float)(jj*jj);
    		if (r2 > myrad2 && r2 <= myradb2)
    		{
                float myang = RAD2DEG(atan2((float)(ii),(float)(jj)));
                if (myang > 90.) myang -= 180.;
                if (myang < -90.) myang += 180.;
                if (myang < 90. && fabs(myang) < psidiff_per_segment)
                {
                	AmyloidCoordinate circlecoord;
                	circlecoord.x = (RFLOAT)jj;
                	circlecoord.y = (RFLOAT)ii;
                	circlecoord.fom =0.;
                	circlecoord.psi =myang;
                	circle.push_back(circlecoord);
                	//std::cerr << " circlecoord.x= " << circlecoord.x << " circlecoord.y= " << circlecoord.y << " psi= " << circlecoord.psi << std::endl;
                }
    		}
    	}
    }



}

FileName AmyloidFinder::getOutputRootName(FileName fn_mic)
{
	FileName fn_pre, fn_jobnr, fn_post;
	decomposePipelineFileName(fn_mic, fn_pre, fn_jobnr, fn_post);
	return fn_odir + fn_post.withoutExtension();
}

RFLOAT AmyloidFinder::getPsiAngle(int ipsi)
{
    return 2.3 + ipsi * psi_step;
}

RFLOAT AmyloidFinder::getPsiDiff(RFLOAT psi1, RFLOAT psi2)
{
    RFLOAT psidiff = fabs(psi1 - psi2);
    if (psidiff > 90.) psidiff -= 180.;
    return fabs(psidiff);

}

void AmyloidFinder::getScoreForOneMicrograph(MultidimArray<RFLOAT> &image, MultidimArray<RFLOAT> &Mscore,
                                             MultidimArray<RFLOAT> &Mangle, RFLOAT &skew, RFLOAT &kurt, bool myverb)
{

    MultidimArray<RFLOAT> Mbig(large_box, large_box);
    Mbig.setXmippOrigin();
    for (long int i=STARTINGY(Mbig); i<=FINISHINGY(Mbig); i++)
    {
        long int ip = i;
        //if (i < STARTINGY(image)) ip += YSIZE(image);
        //else if (i > FINISHINGY(image)) ip -= YSIZE(image);
        if (i < STARTINGY(image)) ip = 2*STARTINGY(image) - i;
        else if (i > FINISHINGY(image)) ip = 2*FINISHINGY(image) - i;

        for (long int j=STARTINGX(Mbig); j<=FINISHINGX(Mbig); j++)
        {
            long int jp = j;
            if (j < STARTINGX(image)) jp = 2*STARTINGX(image) - j;
            else if (j > FINISHINGX(image)) jp = 2*FINISHINGX(image) - j;

            A2D_ELEM(Mbig, i, j) = A2D_ELEM(image, ip, jp);
        }
    }

    // Rotate the large image, and store downscaled images by cropping their Fourier Transform
    std::vector<MultidimArray<RFLOAT> > rotated_imgs(nr_psi), rotated_scores_perline(nr_psi), rotated_scores(nr_psi);
    std::vector<FourierTransformer> transformer(nr_threads);
    // TODO: in principle, only need to rotate to 90 degrees, as I can use both the X and the Y direction for the 1D FFTs!
    if (myverb)
    {
        std::cout << " - Rotating the input image ..." << std::endl;
        init_progress_bar(nr_psi);
    }

#pragma omp parallel for num_threads(nr_threads)
    for (int ipsi = 0; ipsi < nr_psi; ipsi++)
    {
        const int tid = omp_get_thread_num();
        RFLOAT psi = getPsiAngle(ipsi);

        // Rotate the images in their original size to prevent interpolation artefacts near the signal frequencies
        MultidimArray<RFLOAT> Mrot;
        Mrot.setXmippOrigin();
        Mrot.initZeros(large_box, large_box);
        rotate(Mbig, Mrot, psi, 'Z', true);

        //Image<RFLOAT> Ir;
        //Ir()=Mrot;
        //Ir.write("Ir_psi"+ integerToString(ipsi)+".spi");
        //std::cerr << " written: " << "Ir_psi"<< integerToString(ipsi)<<".spi" << std::endl;

        // Re-scale image so that Nyquist is at down_angpix
        MultidimArray<Complex > FT, FT2;
        transformer[tid].FourierTransform(Mrot, FT, false);
        windowFourierTransform(FT, FT2, crop_box);
        Mrot.resize(crop_box, crop_box);
        transformer[tid].inverseFourierTransform(FT2, Mrot);
        Mrot.setXmippOrigin();
        rotated_imgs[ipsi] = Mrot;
        // Just prepare the rotated_scores vector too
        rotated_scores_perline[ipsi].initZeros(crop_box, crop_box);
        rotated_scores_perline[ipsi].setXmippOrigin();
        rotated_scores[ipsi].initZeros(crop_box/shift_step, crop_box/shift_step);
        rotated_scores[ipsi].setXmippOrigin();

    }
    if (myverb) progress_bar(nr_psi);

    // Prepare all the transformers for the 1D lines
    MultidimArray<RFLOAT> oneline_tmp(ilengthmax);
    for (int i = 0; i < nr_threads; i++)
        transformer[i].setReal(oneline_tmp);

    // Now loop over all positions to calculate 1D FFTs
    if (myverb)
    {
        std::cout << " - Searching over all coordinates ..." << std::endl;
        init_progress_bar(nr_psi);
    }

    int my_skip_side_length = ilengthmax/2;
    for (int ipsi = 0; ipsi < nr_psi; ipsi++)
    {
#pragma omp parallel for num_threads(nr_threads)
        for (int ypos = my_skip_side_length; ypos < YSIZE(rotated_imgs[ipsi]) - my_skip_side_length; ypos += 1)
        {
            int cen_ypos = ypos - YSIZE(rotated_imgs[ipsi])/2;
            const int tid = omp_get_thread_num();
            MultidimArray<RFLOAT> oneline(ilengthmax);
            MultidimArray<Complex> FTline(ilengthmax/2 + 1);

            for (int xpos = my_skip_side_length; xpos < XSIZE(rotated_imgs[ipsi]) - my_skip_side_length; xpos += 1)
            {
                int cen_xpos = xpos - XSIZE(rotated_imgs[ipsi])/2;

                // Grab the line from the rotated image, in X and in Y directions
                for (int iline = 0; iline < ilengthmax; iline++)
                    DIRECT_A1D_ELEM(oneline, iline) = A2D_ELEM(rotated_imgs[ipsi], cen_ypos, cen_xpos+iline-ilengthmax/2);

                transformer[tid].FourierTransform(oneline, FTline, false);

                for (int isig = imin_signal; isig <= imax_signal; isig++)
                {
                    A2D_ELEM(rotated_scores_perline[ipsi], cen_ypos, cen_xpos)  += norm(DIRECT_A1D_ELEM(FTline, isig));
                }

            } // end loop ypos
        } // end for xpos

        //Image<RFLOAT> It0;
        //It0()= rotated_scores_perline[ipsi];
        //FileName fnt0="It0_psi"+ integerToString(ipsi)+".spi";
        //It0.write(fnt0);
        //std::cerr <<" written: "<<fnt0 << std::endl;


        // Now that we have signal per individual line for each coordinate, sum over the width of the search box
        // The below is split in two halves, becauses otherwise cen_pos=0 may be sampled twice!!!
        int my_skip_side_width = iwidthmax/2;
 #pragma omp parallel for num_threads(nr_threads)
       for (int ypos = 0; ypos < YSIZE(rotated_imgs[ipsi])/2 - my_skip_side_width; ypos += shift_step)
       {
           for (int ipassy = 0; ipassy < 2; ipassy++)
           {
               int cen_ypos = (ipassy == 0) ? ypos : -ypos;
               if (ypos == 0 && ipassy == 1) continue;

               for (int xpos = 0; xpos < XSIZE(rotated_imgs[ipsi])/2 - my_skip_side_width; xpos += shift_step)
               {
                   for (int ipass = 0; ipass < 2; ipass++)
                   {
                       int cen_xpos = (ipass == 0) ? xpos : -xpos;
                       if (xpos == 0 && ipass == 1) continue;
                       for (int iwidth = 0; iwidth < iwidthmax; iwidth++)
                       {
                           // Grab the line from the rotated image, in X and in Y directions
                           A2D_ELEM(rotated_scores[ipsi], cen_ypos/shift_step, cen_xpos/shift_step) +=
                                   A2D_ELEM(rotated_scores_perline[ipsi], cen_ypos+iwidth-iwidthmax/2, cen_xpos);
                       } // end loop iwidth
                   } // end loop ipass
               } // end loop xpos
           } // end for ipassy
       } // end for ypos

       //Image<RFLOAT> It;
       //It()= rotated_scores[ipsi];
       //FileName fnt="It_psi"+ integerToString(ipsi)+".spi";
       //It.write(fnt);
       //std::cerr <<" written: "<<fnt << std::endl;

       if (myverb) progress_bar(ipsi);

    } // end for ipsi
    if (myverb) progress_bar(nr_psi);


    // Now loop over all positions ad find the best Zscore and the best ipsi
    // Note that each translation in the original image has a different coordinate in the rotated_score images!
    // So, rotate those back first
    if (myverb)
    {
        std::cout << " - Gathering search results ..." << std::endl;
        init_progress_bar(nr_psi);
    }

#pragma omp parallel for num_threads(nr_threads)
    for (int ipsi = 0; ipsi < nr_psi; ipsi++)
    {
        RFLOAT psi = getPsiAngle(ipsi);
        selfRotate(rotated_scores[ipsi], -psi);
    }

    Mangle.resize(down_ysize, down_xsize);
    Mangle.setXmippOrigin();
    Mscore.resize(Mangle);
    MultidimArray<RFLOAT> Msum;
    Msum.resize(Mangle);

    // This can't be parallelised efficiently because need to protect Msums, Mscore and Mangle from simultaneous writing...
    // Calculate Z-scores over psi: (max_psi - avg_psi) /stddev_psi
    for (int ipsi = 0; ipsi < nr_psi; ipsi++)
    {
        RFLOAT mypsi = getPsiAngle(ipsi);
        for (int ypos = 0; ypos < down_ysize; ypos ++)
        {
            int cen_ypos = ypos - down_ysize/2;
            for (int xpos = 0; xpos < down_xsize; xpos ++)
            {
                int cen_xpos = xpos - down_xsize/2;

                RFLOAT myscore = A2D_ELEM(rotated_scores[ipsi], cen_ypos/shift_step, cen_xpos/shift_step);
                A2D_ELEM(Msum, cen_ypos, cen_xpos) += myscore;

                if (myscore > A2D_ELEM(Mscore, cen_ypos, cen_xpos))
                {
                    A2D_ELEM(Mscore, cen_ypos, cen_xpos) = myscore;
                    A2D_ELEM(Mangle, cen_ypos, cen_xpos) = mypsi;
                }

            }
        }
    }

    // Now need to subtract the Mscore for the best psi, as the mean and stddev should be calculated for the non-signal!
    RFLOAT sum=0., sum2=0.;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Msum)
    {
        DIRECT_MULTIDIM_ELEM(Msum, n) -= DIRECT_MULTIDIM_ELEM(Mscore, n);
        DIRECT_MULTIDIM_ELEM(Msum, n) /= (RFLOAT)(nr_psi-1);
        // calculate as normalised score:
        // (max_psi - adjusted_mean_psi) / adjusted_mean_psi
        // where adjusted_mean is the average of the score over all psi-values, except the maximum
        if (DIRECT_MULTIDIM_ELEM(Msum, n) > 0.)
            DIRECT_MULTIDIM_ELEM(Mscore, n) = (DIRECT_MULTIDIM_ELEM(Mscore, n) - DIRECT_MULTIDIM_ELEM(Msum, n)) / DIRECT_MULTIDIM_ELEM(Msum, n);
        sum  += DIRECT_MULTIDIM_ELEM(Mscore, n);
        sum2 += DIRECT_MULTIDIM_ELEM(Mscore, n) * DIRECT_MULTIDIM_ELEM(Mscore, n);
    }

    // Output skewness and kurtosis of Mscore distribution to detect which micrographs have filaments
    RFLOAT n = NZYXSIZE(Msum);
    sum /= n;
    sum2 /= n;
    sum2 = sqrt(sum2-sum*sum);
    skew = 0.;
    kurt = 0.;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Msum)
    {
        RFLOAT aux = (DIRECT_MULTIDIM_ELEM(Mscore, n) - sum)/sum2;
        skew += aux*aux*aux;
        kurt += aux*aux*aux*aux;
    }
    kurt *= n*(n+1)/((n-1)*(n-2)*(n-3));
    skew *= n/((n-1)*(n-2));

    /*
    std::cerr << " skewness= " << skew << " excess kurtosis= " << kurt -3 << std::endl;
    Image<RFLOAT> Itt;
    Itt()=Msum;
    Itt.write("Mscore_avg_over_psi.spi");
    Itt()=Mscore;
    Itt.write("Mscore_Zscore_over_psi.spi");
    */

    if (myverb) progress_bar(nr_psi);

}


std::vector<AmyloidCoordinate> AmyloidFinder::findNextCandidateCoordinates(AmyloidCoordinate &mycoord, MultidimArray<RFLOAT> &Mscore, MultidimArray<RFLOAT> &Mpsi)
{

    std::vector<AmyloidCoordinate> result;

	Matrix2D<RFLOAT> A2D;
	Matrix1D<RFLOAT> vec_c(2), vec_p(2);
	rotation2DMatrix(-mycoord.psi, A2D, false);

    // How many psi-steps away is this segment allowed to be from the previous one: just a bit more than original psi_steps?
    RFLOAT max_psidiff = (1.5 * psi_step);
	for (int icoor = 0; icoor < circle.size(); icoor++)
	{
		// Rotate the circle-vector coordinates along the mycoord.psi
		XX(vec_c) = (circle[icoor]).x;
		YY(vec_c) = (circle[icoor]).y;
		vec_p = A2D * vec_c;

		long int jj = ROUND(mycoord.x + XX(vec_p));
		long int ii = ROUND(mycoord.y + YY(vec_p));

        RFLOAT myscore = A2D_ELEM(Mscore, ii, jj);
        RFLOAT mypsi = A2D_ELEM(Mpsi, ii, jj);
        // Impose small difference in psi-angle with mycoord
        RFLOAT psidiff = getPsiDiff(mycoord.psi, mypsi);
        //std::cerr <<" icoor= " << icoor <<"x= "<<jj<<" y=" << ii<< " elem=" << A2D_ELEM(Mpsi, ii, jj)<< " mypsi= " <<mypsi << " mycoord.psi=" << mycoord.psi<< " max_psidiff= " << max_psidiff << " myscore= " << myscore << " zscore_threshold= " << zscore_threshold << std::endl;
        if (fabs(psidiff) < max_psidiff && myscore > inifactor_threshold * zscore_threshold)
        {
            AmyloidCoordinate newcoord;
            newcoord.x = mycoord.x + XX(vec_p);
            newcoord.y = mycoord.y + YY(vec_p);
#ifdef DEBUG_BOUNDS
            if (ii < STARTINGY(Mscore) || ii > FINISHINGY(Mscore) ||
                    jj < STARTINGX(Mscore) || jj > FINISHINGX(Mscore))
            {
                 REPORT_ERROR("ERROR: out of bounds in findNextCandidateCoordinates");
            }
#endif

            newcoord.psi = A2D_ELEM(Mpsi, ii, jj);
            newcoord.fom = myscore;
            //std::cerr << " myccf= " << myccf << " psi= " << newcoord.psi << std::endl;
            result.push_back(newcoord);
        }
	}

	return result;

}



AmyloidCoordinate AmyloidFinder::findNextAmyloidCoordinate(AmyloidCoordinate &mycoord, MultidimArray<RFLOAT> &Mscore, MultidimArray<RFLOAT> &Mpsi)
{

	// Return if this one has been done already.
	AmyloidCoordinate result;
	result.x = result.y = result.psi = 0.;
	result.fom = -2.;
    RFLOAT myfom = A2D_ELEM(Mscore, ROUND(mycoord.y), ROUND(mycoord.x));
    if (myfom < inifactor_threshold * zscore_threshold)
    {
        //std::cerr << " myfom=" << myfom << " mycoord.fom= " << mycoord.fom << " x= " << mycoord.x << " y= " << mycoord.y << std::endl;
        return result;
    }

    // Set FOM to small value (-2.) in rectangle around mycoord (filament-width wide, distance to new segment high)
    Matrix2D<RFLOAT> A2D;
    Matrix1D<RFLOAT> vec_c(2), vec_p(2);
    rotation2DMatrix(mycoord.psi, A2D, false);
    int rectangle_hwidth = CEIL(0.5 * trace_filament_width / angpix);
    int rectangle_hheight = CEIL(0.6 * nr_rungs_per_segment*amyloid_rung / angpix);
    int maxr = XMIPP_MAX(rectangle_hwidth, rectangle_hheight);
    for (int ii = -maxr; ii <= maxr; ii++)
    {
        for (int jj = -maxr; jj <= maxr; jj++)
        {
            XX(vec_c) = jj;
            YY(vec_c) = ii;
            vec_p = A2D * vec_c;
            if (YY(vec_p) > -rectangle_hwidth && YY(vec_p) < rectangle_hwidth &&
                XX(vec_p) > -rectangle_hheight && XX(vec_p) < rectangle_hheight )
            {
                long int jp = ROUND(mycoord.x + jj);
                long int ip = ROUND(mycoord.y + ii);
#ifdef DEBUG_BOUNDS
                if (ip < STARTINGY(Mscore) || ip > FINISHINGY(Mscore) ||
                    jp < STARTINGX(Mscore) || jp > FINISHINGX(Mscore))
                {
                    Mscore.printShape();
                    std::cerr << " ii= " << ii << " + " << mycoord.y << " = " << ip << std::endl;
                    std::cerr << " jj= " << jj << " + " << mycoord.x << " = " << jp << std::endl;
                    REPORT_ERROR("ERROR: coordinates out of bound");
                }
#endif
                A2D_ELEM(Mscore, ip, jp) = -2.;
            }
        }
    }

    /*
    Image<RFLOAT> It;
    It()=Mpsi;
    It.write("Mpsi.spi");
    It()=Mscore;
    It.write("Mscore.spi");
    char c;
    std::cerr << "press any key" << std::endl;
    std::cin >> c;
    */

	// See how far we can grow in any of the circle directions.
	// Note recursive calls to findNextCandidateCoordinates below.
	// Let's search 3 layers deep...
	std::vector<AmyloidCoordinate> new1coords;
	new1coords = findNextCandidateCoordinates(mycoord, Mscore, Mpsi);

	long int N = new1coords.size();
	std::vector<int> max_depths(N, 0);
	std::vector<RFLOAT> max_sumfoms(N, -9999.);

	RFLOAT sumfom = 0.;
	RFLOAT max_sumfom = -9999.;
	int best_inew1=-1;
	for (int inew1 = 0; inew1 < new1coords.size(); inew1++)
	{
		sumfom = new1coords[inew1].fom;
        // Just select for highest score from all candidate coordinates
        // TODO: should I avoid changes in psi instead?
		if (sumfom > max_sumfom)
		{
			max_sumfom = sumfom;
			best_inew1 = inew1;
		}

		std::vector<AmyloidCoordinate> new2coords;
		new2coords = findNextCandidateCoordinates(new1coords[inew1], Mscore, Mpsi);
		for (int inew2 = 0; inew2 < new2coords.size(); inew2++)
		{

			sumfom = new1coords[inew1].fom + new2coords[inew2].fom;
			if (sumfom > max_sumfom)
			{
				max_sumfom = sumfom;
				best_inew1 = inew1;
			}

			std::vector<AmyloidCoordinate> new3coords;
			new3coords = findNextCandidateCoordinates(new2coords[inew2], Mscore, Mpsi);
			for (int inew3 = 0; inew3 < new3coords.size(); inew3++)
			{
				sumfom = new1coords[inew1].fom + new2coords[inew2].fom + new3coords[inew3].fom;
				if (sumfom > max_sumfom)
				{
					max_sumfom = sumfom;
					best_inew1 = inew1;
				}

				std::vector<AmyloidCoordinate> new4coords;
				new4coords = findNextCandidateCoordinates(new3coords[inew3], Mscore, Mpsi);
				for (int inew4 = 0; inew4 < new4coords.size(); inew4++)
				{
					sumfom = new1coords[inew1].fom + new2coords[inew2].fom + new3coords[inew3].fom + new4coords[inew4].fom;
					if (sumfom > max_sumfom)
					{
						max_sumfom = sumfom;
						best_inew1 = inew1;
					}
				}
			}
		}
	}

	if (best_inew1 < 0)
	{
		return result;
	}
	else
	{
		return new1coords[best_inew1];
	}

}


RFLOAT AmyloidFinder::maxIndex_multithreaded(MultidimArray<RFLOAT> &m, long int &imax, long int &jmax, int skip_sides)
{
    RFLOAT maxval= -99.e99;
    std::vector<RFLOAT> maxval_per_thread(nr_threads, -99.e99);
    std::vector<size_t> maxi_per_thread(nr_threads), maxj_per_thread(nr_threads);

#pragma omp parallel for num_threads(nr_threads)
    for (long int i=STARTINGY(m)+skip_sides; i<=FINISHINGY(m)-skip_sides; i++)
    {
        const int tid = omp_get_thread_num();
        for (long int j=STARTINGX(m)+skip_sides; j<=FINISHINGX(m)-skip_sides; j++)
        {
            const int tid = omp_get_thread_num();
            RFLOAT aux = A2D_ELEM(m, i, j);
            if (aux > maxval_per_thread[tid])
            {
                maxval_per_thread[tid] = aux;
                maxi_per_thread[tid] = i;
                maxj_per_thread[tid] = j;

            }
        }
    }

    for (int tid = 0; tid < nr_threads; tid++)
    {
        if (maxval_per_thread[tid] > maxval)
        {
            maxval = maxval_per_thread[tid];
            imax = maxi_per_thread[tid];
            jmax = maxj_per_thread[tid];
        }
    }

    return maxval;

}

MetaDataTable AmyloidFinder::traceFilaments(MultidimArray<RFLOAT> &Mscore, MultidimArray<RFLOAT> &Mpsi)
{

    // Re-scale and re-box image to original size
    int new_box  = XMIPP_MAX(YSIZE(Mscore), XSIZE(Mscore));
    Mscore.window(FIRST_XMIPP_INDEX(new_box), FIRST_XMIPP_INDEX(new_box), LAST_XMIPP_INDEX(new_box),
                LAST_XMIPP_INDEX(new_box));
    Mpsi.window(FIRST_XMIPP_INDEX(new_box), FIRST_XMIPP_INDEX(new_box), LAST_XMIPP_INDEX(new_box),
                  LAST_XMIPP_INDEX(new_box));
    Mscore.setXmippOrigin();
    Mpsi.setXmippOrigin();

    int newsize = (XSIZE(Mpsi)*down_angpix) / angpix;
    if (newsize%2 !=0) newsize++;
    resizeMap(Mscore, newsize);
    resizeMap(Mpsi, newsize);
    Mscore.setXmippOrigin();
    Mpsi.setXmippOrigin();

    // Again, leave some space near the edges to account for setting the rectangles to zero
    int larger_side = CEIL(1.5 * trace_filament_width / angpix);
    if (larger_side%2 != 0) larger_side++;

    Mscore.window(FIRST_XMIPP_INDEX(ori_ysize + larger_side), FIRST_XMIPP_INDEX(ori_xsize + larger_side),
                  LAST_XMIPP_INDEX(ori_ysize + larger_side), LAST_XMIPP_INDEX(ori_xsize + larger_side));
    Mpsi.window(FIRST_XMIPP_INDEX(ori_ysize + larger_side), FIRST_XMIPP_INDEX(ori_xsize + larger_side),
                  LAST_XMIPP_INDEX(ori_ysize + larger_side), LAST_XMIPP_INDEX(ori_xsize + larger_side));
    Mscore.setXmippOrigin();
    Mpsi.setXmippOrigin();

    std::vector< std::vector <AmyloidCoordinate> > helices;
	bool no_more_ccf_peaks = false;
	int nr_fail = 0;
    int skip_sides = 25;
    while (!no_more_ccf_peaks)
	{
		long int imax, jmax;
		float myscore = maxIndex_multithreaded(Mscore, imax, jmax, skip_sides);
		float mypsi = Mpsi(imax, jmax);
		// Peaks to seed new filaments have to be at least the zscore_threshold
        if (myscore < zscore_threshold )
        {
            //std::cerr << " myscore= " << myscore << " nr_fail= " << nr_fail << std::endl;
            no_more_ccf_peaks = true;
        }

		std::vector<AmyloidCoordinate> helix;
		AmyloidCoordinate coord, newcoord;
		coord.x = jmax;
		coord.y = imax;
		coord.fom = myscore;
		coord.psi = mypsi;
        coord.order = 0;
		helix.push_back(coord);

        // Try grow from the start
        bool is_done_start = false;
        while ( !is_done_start  )
        {
            //std::cerr << " START-in: x= "<< helix[0].x <<" y= " << helix[0].y << " psi= " <<  helix[0].psi << " fom= " <<  helix[0].fom << std::endl;
            newcoord = findNextAmyloidCoordinate(helix[0],Mscore, Mpsi);
            //std::cerr << " START newcoord.x= " << newcoord.x << " newcoord.y= " << newcoord.y << " newcoord.fom= " << newcoord.fom << " newcoord.psi= " << newcoord.psi << " helix.size()= " << helix.size() << std::endl;
            if (newcoord.fom > inifactor_threshold * zscore_threshold)
            {
                newcoord.order = helix[0].order - 1;
                helix.insert(helix.begin(), newcoord);
            }
            else
            {
                is_done_start = true;
                //std::cerr << " START IS DONE!" << std::endl;
            }
        }

        // And try grow from the end, but only if any segments were added in growing from the start
        if (helix.size() > 1)
        {
            // Now that we're done with the START, reset the value for Qscore in the original coordinate, otherwise END won't run
            A2D_ELEM(Mscore, ROUND(helix[helix.size()-1].y), ROUND(helix[helix.size()-1].x)) = helix[helix.size()-1].fom;

            bool is_done_end = false;
            while ( !is_done_end  )
            {
                //std::cerr << " END-in: x= "<< helix[helix.size()-1].x <<" y= " << helix[helix.size()-1].y  << " psi= " <<  helix[helix.size()-1].psi << " fom= " <<  helix[helix.size()-1].fom << std::endl;
                newcoord = findNextAmyloidCoordinate(helix[helix.size()-1], Mscore, Mpsi);
                //std::cerr << " END newcoord.x= " << newcoord.x << " newcoord.y= " << newcoord.y << " newcoord.fom= " << newcoord.fom << " newcoord.psi= " << newcoord.psi << " helix.size()= " << helix.size() << std::endl;
                if (newcoord.fom > inifactor_threshold * zscore_threshold)
                {
                    newcoord.order = helix[0].order + 1;
                    helix.push_back(newcoord);
                }
                else
                {
                    is_done_end = true;
                    //std::cerr << " END IS DONE!" << std::endl;
                }

            }
        }


		if (nr_rungs_per_segment * amyloid_rung * helix.size() > trace_filament_length)
		{
			helices.push_back(helix);

#ifdef DEBUG_TRACE
            std::cerr << "PUSHING BACK HELIX " << helices.size() << " << WITH SIZE= " << helix.size() << std::endl;
			char c;
			std::cerr << " helices.size()= " << helices.size() << std::endl;
			std::cerr << "press any key" << std::endl;
			std::cin >> c;
            Image<RFLOAT> It;
            It()=Mscore;
            It.write("Mscore.spi");
#endif

		}
        else
        {
            nr_fail++;
        }

	} // end while (!no_more_ccf_peaks)

	// Now write out in a STAR file
	// Write out a STAR file with the coordinates
	FileName fn_tmp;
	MetaDataTable MDout;

	// Only output STAR header if there are no tubes...
	MDout.clear();
	MDout.addLabel(EMDL_IMAGE_COORD_X);
	MDout.addLabel(EMDL_IMAGE_COORD_Y);
	MDout.addLabel(EMDL_PARTICLE_AUTOPICK_FOM);
	MDout.addLabel(EMDL_PARTICLE_HELICAL_TUBE_ID);
	MDout.addLabel(EMDL_ORIENT_TILT_PRIOR);
	MDout.addLabel(EMDL_ORIENT_PSI_PRIOR);
	MDout.addLabel(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM);
	MDout.addLabel(EMDL_ORIENT_PSI_PRIOR_FLIP_RATIO);
	MDout.addLabel(EMDL_ORIENT_ROT_PRIOR_FLIP_RATIO);	// KThurber


	// Write out segments for all helices
	int helixid = 0;
	for (int ihelix = 0; ihelix < helices.size(); ihelix++)
	{
		// Sort all coordinates in the helix based on their position
        std::sort(helices[ihelix].begin(), helices[ihelix].end(),
                  [](const AmyloidCoordinate& a, const AmyloidCoordinate& b) { return a.order < b.order;});

        RFLOAT tube_length = 0.;
		RFLOAT mypsi, old_mypsi;
        for (long int iseg = 0; iseg < helices[ihelix].size(); iseg++)
		{

            RFLOAT xpos =  (helices[ihelix][iseg].x ) - (RFLOAT)(FIRST_XMIPP_INDEX(ori_xsize));
            RFLOAT ypos =  (helices[ihelix][iseg].y ) - (RFLOAT)(FIRST_XMIPP_INDEX(ori_ysize));

            // Angle with the next segment (for last one, take mypsi from previous segment!)
			if (iseg < helices[ihelix].size()-1)
            {
                float dx = (float)(helices[ihelix][iseg+1].x - helices[ihelix][iseg].x);
                float dy = (float)(helices[ihelix][iseg+1].y - helices[ihelix][iseg].y);
                mypsi = -1. * RAD2DEG(atan2(dy,dx));
                if (mypsi < 0.)
                {
                    if (fabs(mypsi+180. - old_mypsi) < 90.)
                        mypsi+= 180.;
                }
            }
            else
            {
                mypsi = old_mypsi;
            }

            MDout.addObject();
            MDout.setValue(EMDL_IMAGE_COORD_X, xpos);
            MDout.setValue(EMDL_IMAGE_COORD_Y, ypos);
            MDout.setValue(EMDL_PARTICLE_AUTOPICK_FOM, helices[ihelix][iseg].fom);
            MDout.setValue(EMDL_PARTICLE_HELICAL_TUBE_ID, ihelix+1); // start counting at 1
            MDout.setValue(EMDL_ORIENT_TILT_PRIOR, 90.);
            MDout.setValue(EMDL_ORIENT_PSI_PRIOR, mypsi);
            MDout.setValue(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM, tube_length);
            MDout.setValue(EMDL_ORIENT_PSI_PRIOR_FLIP_RATIO, 0.5);
            MDout.setValue(EMDL_ORIENT_ROT_PRIOR_FLIP_RATIO, 0.5);	// KThurber

            tube_length += nr_rungs_per_segment * amyloid_rung;
		}
		helixid++;
	}

    return MDout;

}



void AmyloidFinder::processOneMicrograph(FileName fn_mic, bool myverb)
{

    FileName fn_root = getOutputRootName(fn_mic);
    FileName fn_fom = fn_root + "_" + fn_out + "_fom.mrc";
    FileName fn_psi = fn_root + "_" + fn_out + "_psi.mrc";
    FileName fn_skew = fn_root + "_" + fn_out + "_skew.star";
    MultidimArray<RFLOAT> Mscore, Mangle;

    if (!(do_write_intermediate || do_only_write_intermediate) || !exists(fn_fom) || !exists(fn_psi))
    {

        Image<RFLOAT> Iin;
        Iin.read(fn_mic);
        Iin().setXmippOrigin();
        if (XSIZE(Iin()) != ori_xsize || YSIZE(Iin()) != ori_ysize || fabs(angpix - Iin.samplingRateX()) > 0.001)
            REPORT_ERROR("ERROR: incorrect size or pixel size for image " + fn_mic);

        RFLOAT skew, kurt;
        getScoreForOneMicrograph(Iin(), Mscore, Mangle, skew, kurt, myverb);

        MetaDataTable MDskew;
        MDskew.setIsList(true);
        MDskew.addObject();
        MDskew.setValue(EMDL_MICROGRAPH_SCORE_SKEWNESS, skew);
        MDskew.setValue(EMDL_MICROGRAPH_SCORE_KURTOSIS, kurt);
        MDskew.write(fn_skew);

        if (do_write_intermediate || do_only_write_intermediate)
        {
            Image<RFLOAT> Ipsi, Izscore;
            Ipsi.setSamplingRateInHeader(down_angpix);
            Ipsi()=Mangle;
            Izscore()=Mscore;
            Ipsi.write(fn_psi);
            Izscore.write(fn_fom);
        }
    }
    else
    {
        Image<RFLOAT> Ipsi, Izscore;
        Ipsi.read(fn_psi);
        Izscore.read(fn_fom);
        Mangle=Ipsi();
        Mscore=Izscore();
        Mangle.setXmippOrigin();
        Mscore.setXmippOrigin();
    }

    if (!do_only_write_intermediate)
    {
        if (myverb)
        {
            std::cout << " - Tracing filaments ..." << std::endl;
            init_progress_bar(1);
        }

        MetaDataTable MDmic = traceFilaments(Mscore, Mangle);

        if (myverb) progress_bar(1);

        if (MDmic.numberOfObjects() > 0)
        {
            if (myverb) std::cout << "Picked " << MDmic.numberOfObjects() << " particles " << std::endl;
            FileName fn_pick = getOutputRootName(fn_mic) + "_" + fn_out + ".star";
            MDmic.write(fn_pick);
        }
    }

    if (myverb) std::cout << "done!" << std::endl;



}


void AmyloidFinder::run()
{
	int barstep;
	if (verb > 0)
	{
		std::cout << " Finding amyloids ..." << std::endl;
		init_progress_bar(fn_micrographs.size());
		barstep = XMIPP_MAX(1, fn_micrographs.size() / 60);
	}

	FileName fn_olddir="";
	for (long int imic = 0; imic < fn_micrographs.size(); imic++)
	{

		// Abort through the pipeline_control system
		if (pipeline_control_check_abort_job())
			exit(RELION_EXIT_ABORTED);

		if (verb > 0 && imic % barstep == 0)
			progress_bar(imic);

		// Check new-style outputdirectory exists and make it if not!
		FileName fn_oroot = getOutputRootName(fn_micrographs[imic]);
		FileName fn_dir = fn_oroot.beforeLastOf("/");
		if (fn_dir != fn_olddir)
		{
			// Make a Particles directory
			mktree(fn_dir);
			fn_olddir = fn_dir;
		}
#ifdef TIMING
		timer.tic(TIMING_A5);
#endif
        processOneMicrograph(fn_micrographs[imic], fn_micrographs.size()==1);

#ifdef TIMING
		timer.toc(TIMING_A5);
#endif
	}

	if (verb > 0)
		progress_bar(fn_micrographs.size());

}

void AmyloidFinder::finalise()
{

    long int barstep = XMIPP_MAX(1, fn_ori_micrographs.size() / 60);
	if (verb > 0)
	{
		std::cout << " Generating  output list of coordinate files ... " << std::endl;
		init_progress_bar(fn_ori_micrographs.size());
	}

    MetaDataTable MDin;
    ObservationModel obsModel;
    ObservationModel::loadSafely(fn_in, obsModel, MDin, "micrographs", verb);

	MetaDataTable MDcoords;
	MetaDataTable MDresult;
	long total_nr_picked = 0;
	int nr_coord_files = 0;
	for (long int imic = 0; imic < fn_ori_micrographs.size(); imic++)
	{

        FileName fn_root = getOutputRootName(fn_ori_micrographs[imic]);
        FileName fn_fom = fn_root + "_" + fn_out + "_fom.mrc";
        FileName fn_psi = fn_root + "_" + fn_out + "_psi.mrc";
        FileName fn_skew = fn_root + "_" + fn_out + "_skew.star";
        FileName fn_pick = fn_root + "_" + fn_out + ".star";

		MetaDataTable MD;
		if (exists(fn_pick))
		{

			MDcoords.addObject();
			MDcoords.setValue(EMDL_MICROGRAPH_NAME, fn_ori_micrographs[imic]);
			MDcoords.setValue(EMDL_MICROGRAPH_COORDINATES, fn_pick);
			nr_coord_files++;

			MD.read(fn_pick);
			long nr_pick = MD.numberOfObjects();
			total_nr_picked += nr_pick;
			if (MD.containsLabel(EMDL_PARTICLE_AUTOPICK_FOM))
			{
				RFLOAT fom, avg_fom = 0.;
				FOR_ALL_OBJECTS_IN_METADATA_TABLE(MD)
				{
					MD.getValue(EMDL_PARTICLE_AUTOPICK_FOM, fom);
					avg_fom += fom;
				}
				avg_fom /= nr_pick;
				// mis-use MetadataTable to conveniently make histograms and value-plots
				MDresult.addObject();
				MDresult.setValue(EMDL_MICROGRAPH_NAME, fn_ori_micrographs[imic]);
				MDresult.setValue(EMDL_PARTICLE_AUTOPICK_FOM, avg_fom);
				MDresult.setValue(EMDL_MLMODEL_GROUP_NR_PARTICLES, nr_pick);
			}
		}

        if (exists(fn_skew))
		{
            MetaDataTable MDskew;
            RFLOAT kurt, skew;
            MDskew.read(fn_skew);
            MDskew.getValue(EMDL_MICROGRAPH_SCORE_KURTOSIS, kurt);
            MDskew.getValue(EMDL_MICROGRAPH_SCORE_SKEWNESS, skew);
            MDin.setValue(EMDL_MICROGRAPH_SCORE_KURTOSIS, kurt, imic);
            MDin.setValue(EMDL_MICROGRAPH_SCORE_SKEWNESS, skew, imic);

            if (do_write_intermediate || do_only_write_intermediate)
            {
                MDin.setValue(EMDL_MICROGRAPH_AUTOPICK_FOM, fn_fom, imic);
                MDin.setValue(EMDL_MICROGRAPH_AUTOPICK_PSI, fn_psi, imic);
            }
        }

		if (verb > 0 && imic % 60 == 0) progress_bar(imic);

	}


	FileName fn_coords = fn_odir + fn_out + ".star";
	MDcoords.setName("coordinate_files");
	MDcoords.write(fn_coords);

    FileName fn_mics = fn_odir + "micrographs_" + fn_out + ".star";
    obsModel.save(MDin, fn_mics, "micrographs");

	if (verb > 0 )
	{
		progress_bar(fn_ori_micrographs.size());
		std::cout << " Saved list with " << nr_coord_files << " coordinate files in: " << fn_coords << std::endl;
	}

	if (verb > 0)
	{
		std::cout << " Total number of particles from " << fn_ori_micrographs.size() << " micrographs is " << total_nr_picked << std::endl;

		long avg = 0;
		if (fn_ori_micrographs.size() > 0) avg = ROUND((RFLOAT)total_nr_picked/fn_ori_micrographs.size());
		std::cout << " i.e. on average there were " << avg << " particles per micrograph" << std::endl;

		std::cout << " Now generating logfile.pdf ... " << std::endl;
	}

	// Values for all micrographs
	FileName fn_eps;
	std::vector<FileName> all_fn_eps;
	std::vector<RFLOAT> histX, histY;

	MDresult.write(fn_odir + "summary.star");
	CPlot2D *plot2Db=new CPlot2D("Nr of picked particles for all micrographs");
	MDresult.addToCPlot2D(plot2Db, EMDL_UNDEFINED, EMDL_MLMODEL_GROUP_NR_PARTICLES, 1.);
	plot2Db->SetDrawLegend(false);
	fn_eps = fn_odir + "all_nr_parts.eps";
	plot2Db->OutputPostScriptPlot(fn_eps);
	all_fn_eps.push_back(fn_eps);
	delete plot2Db;
	if (MDresult.numberOfObjects() > 3)
	{
		CPlot2D *plot2D=new CPlot2D("");
		MDresult.columnHistogram(EMDL_MLMODEL_GROUP_NR_PARTICLES,histX,histY,0, plot2D);
		fn_eps = fn_odir + "histogram_nrparts.eps";
		plot2D->SetTitle("Histogram of nr of picked particles per micrograph");
		plot2D->OutputPostScriptPlot(fn_eps);
		all_fn_eps.push_back(fn_eps);
		delete plot2D;
	}

	CPlot2D *plot2Dc=new CPlot2D("Average autopick FOM for all micrographs");
	MDresult.addToCPlot2D(plot2Dc, EMDL_UNDEFINED, EMDL_PARTICLE_AUTOPICK_FOM, 1.);
	plot2Dc->SetDrawLegend(false);
	fn_eps = fn_odir + "all_FOMs.eps";
	plot2Dc->OutputPostScriptPlot(fn_eps);
	all_fn_eps.push_back(fn_eps);
	delete plot2Dc;
	if (MDresult.numberOfObjects() > 3)
	{
		CPlot2D *plot2Dd=new CPlot2D("");
		MDresult.columnHistogram(EMDL_PARTICLE_AUTOPICK_FOM,histX,histY,0, plot2Dd);
		fn_eps = fn_odir + "histogram_FOMs.eps";
		plot2Dd->SetTitle("Histogram of average autopick FOM per micrograph");
		plot2Dd->OutputPostScriptPlot(fn_eps);
		all_fn_eps.push_back(fn_eps);
		delete plot2Dd;
	}

    CPlot2D *plot2De=new CPlot2D("Skewness of FOM for all micrographs");
    MDresult.addToCPlot2D(plot2De, EMDL_UNDEFINED, EMDL_MICROGRAPH_SCORE_SKEWNESS, 1.);
    plot2De->SetDrawLegend(false);
    fn_eps = fn_odir + "all_FOM_skew.eps";
    plot2De->OutputPostScriptPlot(fn_eps);
    all_fn_eps.push_back(fn_eps);
    delete plot2De;
    if (MDresult.numberOfObjects() > 3)
    {
        CPlot2D *plot2Df=new CPlot2D("");
        MDresult.columnHistogram(EMDL_MICROGRAPH_SCORE_SKEWNESS,histX,histY,0, plot2Df);
        fn_eps = fn_odir + "histogram_FOM_skew.eps";
        plot2Df->SetTitle("Histogram of FOM skewness per micrograph");
        plot2Df->OutputPostScriptPlot(fn_eps);
        all_fn_eps.push_back(fn_eps);
        delete plot2Df;
    }
    
    CPlot2D *plot2Dg=new CPlot2D("Kurtosis of FOM for all micrographs");
    MDresult.addToCPlot2D(plot2Dg, EMDL_UNDEFINED, EMDL_MICROGRAPH_SCORE_KURTOSIS, 1.);
    plot2Dg->SetDrawLegend(false);
    fn_eps = fn_odir + "all_FOM_kurt.eps";
    plot2Dg->OutputPostScriptPlot(fn_eps);
    all_fn_eps.push_back(fn_eps);
    delete plot2Dg;
    if (MDresult.numberOfObjects() > 3)
    {
        CPlot2D *plot2Dh=new CPlot2D("");
        MDresult.columnHistogram(EMDL_MICROGRAPH_SCORE_KURTOSIS,histX,histY,0, plot2Dh);
        fn_eps = fn_odir + "histogram_FOM_kurt.eps";
        plot2Dh->SetTitle("Histogram of FOM kurtosis per micrograph");
        plot2Dh->OutputPostScriptPlot(fn_eps);
        all_fn_eps.push_back(fn_eps);
        delete plot2Dh;
    }

	joinMultipleEPSIntoSinglePDF(fn_odir + "logfile.pdf", all_fn_eps);

}