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

    // Fourier shells for amyloid signal
    RFLOAT signal_minres, signal_maxres;

    // Threshold for selecting peaks in Zscore map
    RFLOAT zscore_threshold;

    // Number of new amyloid rungs per segment (particle)
    int nr_rungs_per_segment;

    // Allowed psi-deviation per segment (particle)
    RFLOAT psidiff_per_segment;

    // Minimum width and length of filaments (in A) for tracing of segments
    RFLOAT minimum_filament_length, filament_width;

    // Width and length of the filament searching motifs (in A)
    int width, length;

    // Sampling of positions
    int shift_step, ori_xsize, ori_ysize, down_xsize, down_ysize;

    // Number of psi samplings
    int nr_psi;

    //Number of openMP trhreads
    int nr_threads;

    // Some public parameters
    int iwidthmax, ilengthmax, imin_signal, imax_signal, large_box, crop_box;
    RFLOAT amyloid_rung;
    std::vector<AmyloidCoordinate> circle;

    // All micrographs to autopick from
    std::vector<FileName> fn_micrographs, fn_ori_micrographs;

    // Continue an old run: only estimate CTF if logfile WITH Final Values line does not yet exist, otherwise skip the tomogram
    bool do_only_unfinished;

    // Write out intermediate FOM and Psi maps?
    bool do_write_intermediate;

    // Is there any work to be done?
    bool todo_anything;


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

    // Loop over all psi-angles and coordinates to get accumulated score and angle image for a given micrograph
    void getScoreForOneMicrograph(MultidimArray<RFLOAT> &image, MultidimArray<RFLOAT> &Mscore, MultidimArray<RFLOAT> &Mangle, bool myverb = false);

    // Find next candidate coordinates around a previous coordinate
    std::vector<AmyloidCoordinate> findNextCandidateCoordinates(AmyloidCoordinate &mycoord, MultidimArray<RFLOAT> &Mscore, MultidimArray<RFLOAT> &Mpsi);

    // Find next segment in an amyloid
    AmyloidCoordinate findNextAmyloidCoordinate(AmyloidCoordinate &mycoord, MultidimArray<RFLOAT> &Mscore, MultidimArray<RFLOAT> &Mpsi);

    // Multi-threaded maxIndex
    RFLOAT maxIndex_multithreaded(MultidimArray<RFLOAT> &Mscore, long int &imax, long int &jmax);

    // Trace individual segments in the images based on the scores and angles
    MetaDataTable traceFilaments(MultidimArray<RFLOAT> &Mscore, MultidimArray<RFLOAT> &Mpsi);

    // Run on one micrograph
    void processOneMicrograph(FileName fn_mic, bool myverb = false);

    // Execute all CTFFIND jobs to get CTF parameters
    void run();

    // Write out combined star files, etc
    void finalise();


};


#endif //RELION_AMYLOID_FINDER_H
