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

void ReconstructorMpi::run()
{

	if (fn_debug != "")
	{
		Reconstructor::readDebugArrays();
		if (node->isLeader())
			reconstruct();
		return;
	}

    Reconstructor::initialise();

    // MPI-guarded cache init: leader copies + registers, barrier, then followers register
    if (fn_cache != "")
    {
        ObservationModel *obsModel_ptr = do_ignore_optics ? NULL : &obsModel;
        CacheInitializer::initializeCacheMpi(fn_cache, cache_copy_threads, DF, verb, obsModel_ptr, node->isLeader(), (intptr_t)MPI_COMM_WORLD);
    }

	// Helper for MPI reduce + reconstruct per subset
	auto reduceAndReconstruct = [&](const FileName &fn_out_orig, bool is_half1, bool is_half2)
	{
		MultidimArray<Complex> sumd(backprojector.data);
		MultidimArray<RFLOAT> sumw(backprojector.weight);
		MPI_Allreduce(MULTIDIM_ARRAY(backprojector.data), MULTIDIM_ARRAY(sumd),
			      2 * MULTIDIM_SIZE(backprojector.data), MY_MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MULTIDIM_ARRAY(backprojector.weight), MULTIDIM_ARRAY(sumw),
			      MULTIDIM_SIZE(backprojector.weight), MY_MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		if (node->isLeader())
		{
			backprojector.data = sumd;
			backprojector.weight = sumw;
			if (is_half1)
				fn_out = fn_out_orig.insertBeforeExtension("_half1");
			else if (is_half2)
				fn_out = fn_out_orig.insertBeforeExtension("_half2");
			else
				fn_out = fn_out_orig;
			reconstruct();
		}
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
		MultidimArray<Complex> sumd(backprojector.data);
		MultidimArray<RFLOAT> sumw(backprojector.weight);
		MPI_Allreduce(MULTIDIM_ARRAY(backprojector.data), MULTIDIM_ARRAY(sumd), 2*MULTIDIM_SIZE(backprojector.data), MY_MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MULTIDIM_ARRAY(backprojector.weight), MULTIDIM_ARRAY(sumw), MULTIDIM_SIZE(backprojector.weight), MY_MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		if (node->isLeader())
		{
			backprojector.data = sumd;
			backprojector.weight = sumw;
		}
		if (node->isLeader())
			reconstruct();
	}

}
