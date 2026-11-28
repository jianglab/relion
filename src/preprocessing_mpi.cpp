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
#include "src/preprocessing_mpi.h"
#include "src/async_file_writer.h"
#include "src/virtual_movie_averages.h"

void PreprocessingMpi::read(int argc, char **argv)
{
	// Define a new MpiNode
	node = new MpiNode(argc, argv);

	// First read in non-parallelisation-dependent variables
	Preprocessing::read(argc, argv, node->rank);

	int mpi_section = parser.addSection("MPI options");
	max_mpi_nodes =textToInteger(parser.getOption("--max_mpi_nodes", "Limit the number of effective MPI nodes to protect from too heavy disk I/O (thus ignoring larger values from mpirun)", "8"));

	// Don't put any output to screen for mpi followers
	verb = (node->isLeader()) ? 1 : 0;

	// Possibly also read parallelisation-dependent variables here

	// Print out MPI info
	printMpiNodesMachineNames(*node);
}

void PreprocessingMpi::runExtractParticles()
{
	int shm_size = 1;
	{
		MPI_Comm shm_comm;
		if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm) == MPI_SUCCESS)
		{
			MPI_Comm_size(shm_comm, &shm_size);
			MPI_Comm_free(&shm_comm);
		}
	}
	mpi_local_size = XMIPP_MAX(1, shm_size);

	// Total number of nodes is limited to max_mpi_nodes
	long int my_nr_mics;
	if (node->rank < max_mpi_nodes)
	{

		// Each node until max_mpi_nodes does part of the work
		long int nr_mics = MDmics.numberOfObjects();
		long int my_first_mic, my_last_mic;
		int my_nr_nodes = XMIPP_MIN(max_mpi_nodes, node->size);
		divide_equally(nr_mics, my_nr_nodes, node->rank, my_first_mic, my_last_mic);
		my_nr_mics = my_last_mic - my_first_mic + 1;
		//std::cerr << " rank= " << node->rank << " my_first_mic= "<<my_first_mic<< " mylastmic= "<< my_last_mic<< " max_mpi_nodes= "<<max_mpi_nodes<<std::endl;

		int barstep;
		if (verb > 0)
		{
			std::cout << " Extracting particles from the micrographs ..." << std::endl;
			init_progress_bar(my_nr_mics);
			barstep = XMIPP_MAX(1, my_nr_mics / 60);

		}

		vmovies::ReadAhead read_ahead(do_virtual ? 0 : -1);
		read_ahead.expectMicrographs(my_nr_mics);
		MicReadAhead mic_ahead(!do_virtual);
		struct MicPrefetchScope
		{
			Preprocessing *p;
			MicPrefetchScope(Preprocessing *p, MicReadAhead *m) : p(p) { p->active_mic_prefetch_ = m; }
			~MicPrefetchScope() { p->active_mic_prefetch_ = NULL; }
		} mic_prefetch_scope(this, &mic_ahead);

		FileName fn_mic, fn_olddir = "";
		long int imic = 0;
		FOR_ALL_OBJECTS_IN_METADATA_TABLE(MDmics)
		{
			if (imic >= my_first_mic && imic <= my_last_mic)
			{

				// Abort through the pipeline_control system
				if (pipeline_control_check_abort_job())
					MPI_Abort(MPI_COMM_WORLD, RELION_EXIT_ABORTED);

				MDmics.getValue(EMDL_MICROGRAPH_NAME, fn_mic);
				if (read_ahead.enabled())
				{
					std::vector<std::string> next;
					for (long int k = imic + 1; k <= my_last_mic && (int) next.size() < vmovies::ReadAhead::WINDOW; k++)
					{
						FileName fn_next;
						MDmics.getValue(EMDL_MICROGRAPH_NAME, fn_next, k);
						if (fn_data == "" && !exists(micname2coordname[fn_next]))
							continue;
						next.push_back(fn_next);
					}
					read_ahead.upcoming(next);
				}
				if (!do_virtual)
				{
					for (long int k = imic + 1; k <= my_last_mic; k++)
					{
						FileName fn_next;
						MDmics.getValue(EMDL_MICROGRAPH_NAME, fn_next, k);
						if (only_extract_unfinished && exists(getOutputFileNameRoot(fn_next) + "_extract.star"))
							continue;
						if (fn_data == "")
						{
							if (!exists(micname2coordname[fn_next]))
								continue;
						}
						else if (!hasCoordinatesForMic(fn_next))
							continue;
						mic_ahead.request(fn_next);
						break;
					}
				}
				int optics_group = obsModelMic.getOpticsGroup(MDmics);

				// Set the pixel size for this micrograph
				angpix = obsModelMic.getPixelSize(optics_group);
				// Also set the output_angpix (which could be rescaled)
				output_angpix = angpix;
				if (do_rescale)
					output_angpix *= (RFLOAT)extract_size / (RFLOAT)scale;

				// Check new-style outputdirectory exists and make it if not!
				FileName fn_dir = getOutputFileNameRoot(fn_mic);
				fn_dir = fn_dir.beforeLastOf("/");
				if (fn_dir != fn_olddir && !exists(fn_dir))
				{
					// Make a Particles directory
					mktree(fn_dir);
					fn_olddir = fn_dir;
				}

				if (verb > 0 && imic % barstep == 0)
					progress_bar(imic);

				extractParticlesFromFieldOfView(fn_mic, imic);
			}
			imic++;
		}
	}

	// Every rank's stacks must be complete (an error ends the job) before the final star file lists them
	AsyncFileWriter::instance().waitOrThrow();

	// Wait until all nodes have finished to make final star file
	MPI_Barrier(MPI_COMM_WORLD);
	int pm_lo = premultiply_size_min, pm_hi = premultiply_size_max;
	MPI_Allreduce(MPI_IN_PLACE, &pm_lo, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &pm_hi, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	if (verb > 0)
		reportPremultiplySizes(pm_lo, pm_hi);

	if (node->isLeader())
	{
		if (verb > 0)
			progress_bar(my_nr_mics);
		Preprocessing::joinAllStarFiles();
	}
}

void PreprocessingMpi::run()
{
	// Extract and operate on particles in parallel
	if (do_extract)
	{
		runExtractParticles();
	}
	// The following has not been parallelised....
	else if (fn_operate_in != "" && node->isLeader())
		Preprocessing::runOperateOnInputFile();

	if (verb > 0)
		std::cout << " Done preprocessing!" <<std::endl;
}
