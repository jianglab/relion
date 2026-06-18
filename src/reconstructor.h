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
#ifndef SRC_RECONSTRUCTOR_H_
#define SRC_RECONSTRUCTOR_H_

#include <src/backprojector.h>
#include <src/funcs.h>
#include <src/ctf.h>
#include <src/args.h>
#include <src/error.h>
#include <src/euler.h>
#include <src/time.h>
#include <src/ml_model.h>
#include <mutex>
#include <src/jaz/single_particle/obs_model.h>
#include <src/cache_init.h>
#include <src/prefetch.h>
#include <src/spatial_frequency_grid.h>
#include <memory>

enum SpatialFrequencyMode
{
	SPATIAL_FREQUENCY_MODE_S,
	SPATIAL_FREQUENCY_MODE_S2
};

class Reconstructor
{
public:
	// I/O Parser
	IOParser parser;

	FileName fn_out, fn_sel, fn_img, fn_sym, fn_sub, fn_fsc, fn_debug, fn_noise, image_path;
	FileName fn_cache;
	bool do_s2_grid_diagnostic = false;

	MetaDataTable DF;
	ObservationModel obsModel;
	MlModel model;

	int r_max, r_min_nn, blob_order, ref_dim, interpolator, iter,
	debug_ori_size, debug_size,
	ctf_dim, nr_helical_asu, newbox, width_mask_edge, nr_sectors, subset, chosen_class,
	data_dim, output_boxsize, cache_copy_threads, nr_threads;
	int verb = 1;

	RFLOAT blob_radius, blob_alpha, angular_error, shift_error, angpix, maxres,
	       helical_rise, helical_twist;
	int s2_ctf_oversampling_min = 2;
	RFLOAT voltage_for_s2_ = -1.0;
	bool s2_uniform_fill_ = false;

	bool do_ctf, ctf_phase_flipped, only_flip_phases, intact_ctf_first_peak,
	     do_fom_weighting, do_3d_rot, do_reconstruct_ctf, do_ewald, skip_weighting, skip_mask, do_debug,
	     do_ignore_optics, skip_subtomo_correction, normalised_subtomo, ctf3d_squared, is_tomo;


	bool skip_gridding, do_reconstruct_ctf2, do_reconstruct_meas, is_reverse, read_weights, do_external_reconstruct;
	bool do_half1, do_half2, do_alldata, do_prefetch, do_invert_contrast;
	long int random_subset_size;
	int random_subset_seed;
	SpatialFrequencyMode spatial_frequency_mode;

	float padding_factor, mask_diameter;

	// All backprojectors needed for parallel reconstruction
	BackProjector backprojector;

	std::unique_ptr<AsyncReconstructPrefetcher> prefetcher_;

	// A single projector is needed for parallel reconstruction
	Projector projector;

public:
	/** Empty constructor
	 *
	 * A default Projector is created.
	 *
	 * @code
	 * Projector PPref;
	 * @endcode
	 */
	Reconstructor() { }

	// Read command line arguments
	void read(int argc, char **argv);

	// Print usage instructions
	void usage();

	// Initialise some stuff after reading
	void initialise();

	// Execute
	void run();

	// Reconstruct with debug arrays
	void readDebugArrays();

	// Loop over all particles to be back-projected
	void backproject(int rank = 0, int size = 1);

	// For parallelisation purposes
	void backprojectOneParticle(long int ipart);

protected:
    bool skip_cache_init_in_read_ = false;

    std::map<long int, SpatialFrequencyGrid2D> s2_grid_cache_;
    std::mutex s2_grid_mutex_;

    void ensureS2GridCached(int myBoxSize, RFLOAT myPixelSize, RFLOAT s2_step);
    const SpatialFrequencyGrid2D& getS2Grid(long int cache_key);

    // perform the gridding reconstruction
    void reconstruct();

	// Select a random subset of particles from DF based on subset filter
	MetaDataTable selectRandomSubset(const MetaDataTable &DF_in, long int sample_size,
	                                 int random_subset_filter, int seed, int verb) const;

	void applyCTFPandCTFQ(MultidimArray<Complex> &Fin, CTF &ctf, FourierTransformer &transformer,
	                      MultidimArray<Complex> &outP, MultidimArray<Complex> &outQ, bool skip_mask=false);
};

#endif /* SRC_RECONSTRUCTOR_H_ */
