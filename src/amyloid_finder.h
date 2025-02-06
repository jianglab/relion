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
#ifndef RELION_AMYLOID_FINDER_H
#define RELION_AMYLOID_FINDER_H

#include <src/image.h>
#include <src/funcs.h>
#include <src/args.h>
#include <src/fftw.h>
#include <src/filename.h>
#include <src/time.h>
#include <src/transformations.h>
#include <src/jaz/single_particle/obs_model.h>
#include <omp.h>

struct AmyloidCoordinate
{
	RFLOAT x, y, psi, fom, order;
};

class AmyloidFinder
{
public:

    // I/O Parser
    IOParser parser;

    // Verbosity
    int verb;

    // Output rootname
    FileName fn_in, fn_odir, fn_out;

    // Psi sampling step
    RFLOAT psi_step;

    // Original pixel size, and the downscaled one
    RFLOAT angpix, down_angpix;

    // Fourier shells for amyloid signal and non-signal
    RFLOAT signal_minres, signal_maxres, nonsignal_minres, nonsignal_maxres;

    // Threshold for selecting peaks in Zscore map
    RFLOAT threshold;

    // Plot filament tracing results?
    bool do_plot;

    // width and length of filaments (in A) for searching of 4.7A signal
    RFLOAT search_filament_length, search_filament_width;

    // Minimum length for tracing filaments
    RFLOAT trace_filament_length, trace_filament_width;

    // Redo existing tracings?
    bool do_redo_tracing;
    
    // Python executable name
    FileName fn_exe;

    // Sampling of positions
    int shift_step, ori_xsize, ori_ysize, down_xsize, down_ysize;

    // Number of psi samplings
    int nr_psi;

    //Number of openMP trhreads
    int nr_threads;

    // Some public parameters
    int iwidthmax, ilengthmax, imin_signal, imax_signal, imin_nonsignal, imax_nonsignal, large_box, crop_box;
    std::vector<AmyloidCoordinate> circle;

    // All and selected micrographs to autopick from
    std::vector<FileName> fn_ori_micrographs, fn_micrographs;

    // Continue an old run: only estimate CTF if logfile WITH Final Values line does not yet exist, otherwise skip the tomogram
    bool do_only_unfinished;

public:
    // Read command line arguments
    void read(int argc, char **argv, int rank = 0);

    // Print usage instructions
    void usage();

    // Initialise some stuff after reading
    void initialise(bool is_leader = true);

    // Get the output coordinate filename given the micrograph filename
    FileName getOutputRootName(FileName fn_mic);

    // Make sure all pieces of code use same psi angles from ipsi
    RFLOAT getPsiAngle(int ipsi);

    // Difference between two psi angles that range from 0-180
    RFLOAT getPsiDiff(RFLOAT psi1, RFLOAT psi2);

    // Loop over all psi-angles and coordinates to get accumulated score and angle image for a given micrograph
    void getScoreForOneMicrograph(MultidimArray<RFLOAT> &image, MultidimArray<RFLOAT> &Mscore,
                                  MultidimArray<RFLOAT> &Mangle, RFLOAT &skew, RFLOAT &kurt, bool myverb = false);

    // Trace individual segments in the images based on the scores and angles
    void traceFilaments(FileName &fn_fom, FileName &fn_star);

    // Run on one micrograph
    void processOneMicrograph(FileName fn_mic, bool myverb = false);

    // Execute all CTFFIND jobs to get CTF parameters
    void run();

    // Write out combined star files, etc
    void finalise();


};


#endif //RELION_AMYLOID_FINDER_H
