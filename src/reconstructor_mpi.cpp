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
#include "src/reconstructor_mpi.h"
#include <algorithm>
#include <chrono>
#include <iomanip>

void ReconstructorMpi::read(int argc, char **argv)
{
	// Define a new MpiNode
	node = new MpiNode(argc, argv);

	// Defer cache init in Reconstructor::initialise(): let only rank 0 do the actual copying
	skip_cache_init_in_read_ = true;

	// First read in non-parallelisation-dependent variables
	Reconstructor::read(argc, argv);

	// Don't put any output to screen for mpi followers
	verb = (node->isLeader()) ? verb : 0;

	// Possibly also read parallelisation-dependent variables here

	if (node->size < 2)
		REPORT_ERROR("ReconstductMpi::read ERROR: this program needs to be run with at least two MPI processes!");

	// Print out MPI info
	printMpiNodesMachineNames(*node);

}

// Sum an array onto the leader, in place and in pieces (an MPI count is an int). Followers keep
// their own partial sums, and no second copy of the volume is made.
static void reduceToLeader(double *p, long int n, bool is_leader)
{
	const long int chunk = 1L << 26;
	for (long int off = 0; off < n; off += chunk)
	{
		const int c = (int)std::min(chunk, n - off);
		if (is_leader)
			MPI_Reduce(MPI_IN_PLACE, p + off, c, MY_MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		else
			MPI_Reduce(p + off, NULL, c, MY_MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}
}

void ReconstructorMpi::run()
{

	if (fn_debug != "")
	{
		Reconstructor::readDebugArrays();
		if (node->isLeader())
			reconstruct();
		return;
	}

	typedef std::chrono::steady_clock WallClock;
	WallClock::time_point t_last = WallClock::now();
	auto lap = [&](const char *what)
	{
		WallClock::time_point t = WallClock::now();
		if (node->isLeader())
			std::cout << std::fixed << std::setprecision(1) << " + [timing] " << what << ": " << std::chrono::duration<double>(t - t_last).count() << " s" << std::endl;
		t_last = t;
	};

    Reconstructor::initialise();
	lap("initialise");

    // MPI-guarded cache init: leader copies + registers, barrier, then followers register
    if (fn_cache != "")
    {
        ObservationModel *obsModel_ptr = do_ignore_optics ? NULL : &obsModel;
        CacheInitializer::initializeCacheMpi(fn_cache, cache_copy_threads, DF, verb, obsModel_ptr, node->isLeader(), (intptr_t)MPI_COMM_WORLD);
    }

	// Helper for MPI reduce + reconstruct per subset
	auto reduceAndReconstruct = [&](const FileName &fn_out_orig, bool is_half1, bool is_half2)
	{
		lap("back-projection");
		reduceToLeader((double*)MULTIDIM_ARRAY(backprojector.data), 2 * MULTIDIM_SIZE(backprojector.data), node->isLeader());
		reduceToLeader((double*)MULTIDIM_ARRAY(backprojector.weight), MULTIDIM_SIZE(backprojector.weight), node->isLeader());
		lap("reduce across ranks");

		if (node->isLeader())
		{
			if (is_half1)
				fn_out = fn_out_orig.insertBeforeExtension("_half1");
			else if (is_half2)
				fn_out = fn_out_orig.insertBeforeExtension("_half2");
			else
				fn_out = fn_out_orig;
			reconstruct();
		}
		lap("reconstruct (leader)");
		MPI_Barrier(MPI_COMM_WORLD);
	};

	if (do_half1 || do_half2 || do_alldata)
	{
		FileName fn_out_orig = fn_out;
		if (do_half1)
		{
			subset = 1;
			MetaDataTable DF_restore = DF;
			DF = selectRandomSubset(DF, random_subset_size, 1, random_subset_seed, verb);
			if (verb > 0 && node->isLeader())
				std::cout << "=== Reconstructing half-1 (" << DF.numberOfObjects() << " particles) ===" << std::endl;
			Reconstructor::backproject(node->rank, node->size);
			reduceAndReconstruct(fn_out_orig, true, false);
			DF = DF_restore;
		}
		if (do_half2)
		{
			subset = 2;
			MetaDataTable DF_restore = DF;
			DF = selectRandomSubset(DF, random_subset_size, 2, random_subset_seed, verb);
			if (verb > 0 && node->isLeader())
				std::cout << "=== Reconstructing half-2 (" << DF.numberOfObjects() << " particles) ===" << std::endl;
			Reconstructor::backproject(node->rank, node->size);
			reduceAndReconstruct(fn_out_orig, false, true);
			DF = DF_restore;
		}
		if (do_alldata)
		{
			subset = -1;
			MetaDataTable DF_restore = DF;
			DF = selectRandomSubset(DF, random_subset_size, -1, random_subset_seed, verb);
			if (verb > 0 && node->isLeader())
				std::cout << "=== Reconstructing full map (" << DF.numberOfObjects() << " particles) ===" << std::endl;
			Reconstructor::backproject(node->rank, node->size);
			reduceAndReconstruct(fn_out_orig, false, false);
			DF = DF_restore;
		}
		fn_out = fn_out_orig;
	}
	else
	{
		Reconstructor::backproject(node->rank, node->size);
		lap("back-projection");
		reduceToLeader((double*)MULTIDIM_ARRAY(backprojector.data), 2 * MULTIDIM_SIZE(backprojector.data), node->isLeader());
		reduceToLeader((double*)MULTIDIM_ARRAY(backprojector.weight), MULTIDIM_SIZE(backprojector.weight), node->isLeader());
		lap("reduce across ranks");
		if (node->isLeader())
			reconstruct();
		lap("reconstruct (leader)");
	}

}
