/***************************************************************************
 *
 * Author: "Jasenko Zivanov & Sjors H.W. Scheres"
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

#include "motion_refiner_mpi.h"
#include <algorithm>

namespace
{

// Partition [0, weights.size()) into contiguous non-overlapping ranges balancing total weight
void divideWeighted(
		const std::vector<double>& weights,
		int nr_ranks, int my_rank,
		long int& first, long int& last)
{
	const long int n = (long int)weights.size();
	if (n <= 0 || my_rank < 0 || my_rank >= nr_ranks)
	{
		first = 1;
		last = 0;
		return;
	}
	if (nr_ranks == 1)
	{
		first = 0;
		last = n - 1;
		return;
	}
	if (n <= nr_ranks)
	{
		if (my_rank < n) { first = my_rank; last = my_rank; }
		else { first = 1; last = 0; }
		return;
	}

	std::vector<double> pref(n + 1, 0.0);
	for (long int i = 0; i < n; i++)
		pref[i + 1] = pref[i] + std::max(1.0, weights[i]);

	const double total = pref[n];
	std::vector<long int> cut(nr_ranks + 1, 0);
	cut[nr_ranks] = n;

	for (int r = 1; r < nr_ranks; r++)
	{
		const double target = total * ((double)r / (double)nr_ranks);
		const long int lo = cut[r - 1] + 1;
		const long int hi = n - (nr_ranks - r);
		auto it = std::lower_bound(pref.begin() + lo, pref.begin() + hi + 1, target);
		long int idx = (long int)(it - pref.begin());
		if (idx > lo && std::abs(pref[idx - 1] - target) < std::abs(pref[idx] - target))
			idx--;
		cut[r] = std::max(lo, std::min(hi, idx));
	}

	first = cut[my_rank];
	last = cut[my_rank + 1] - 1;
}

} // namespace

void MotionRefinerMpi::read(int argc, char **argv)
{
	// Define a new MpiNode
	node = new MpiNode(argc, argv);
	
	// First read in non-parallelisation-dependent variables
	MotionRefiner::read(argc, argv);
	readFirstMovieEarly = false;
	
	// Don't put any output to screen for mpi followers!
	verb = (node->isLeader()) ? verb : 0;
	
	// Possibly also read parallelisation-dependent variables here
	if (node->size < 2)
	{
		REPORT_ERROR("ERROR: this program needs to be run with at least two MPI processes!");
	}
	
	if (node->isLeader() && (motionParamEstimator.anythingToDo()))
	{
		REPORT_ERROR("Parameter estimation is not supported in MPI mode.");
		return;
	}
	
	// Print out MPI info
	printMpiNodesMachineNames(*node);
}

void MotionRefinerMpi::runWithFccUpdate()
{
	if (estimateParams)
	{
		REPORT_ERROR("Parameter estimation is not supported in MPI mode.");
		return;
	}

	const int lastTotalMgForFCC = lastTotalMicrographForFCC();
	int lastMotionMgForFCC = subtractFinishedMicrographs(lastTotalMgForFCC, motionUnfinished);
	
	// Parallel loop over micrographs

	if (debug)
	{
		std::cout << "Node " << node->rank << ": "
				  << "begin FCC batch \n";
	}
	
	if (estimateMotion)
	{
		// Each node does part of the work (balanced by particle^2 workload)
		long int my_first_motion_micrograph_FCC, my_last_motion_micrograph_FCC;
		const long int n_fcc = lastMotionMgForFCC + 1;
		std::vector<double> weights_fcc(std::max<long int>(0, n_fcc));
		for (long int i = 0; i < n_fcc; i++)
		{
			const double pc = (double)motionMdts[i].numberOfObjects();
			weights_fcc[i] = std::max(1.0, pc * pc);
		}
		divideWeighted(weights_fcc, node->size, node->rank,
			my_first_motion_micrograph_FCC, my_last_motion_micrograph_FCC);

		if (debug)
		{
			std::vector<int> motion2total = getBackwardIndices(motionUnfinished);

			std::cout << "Node " << node->rank << ": "
					  << "motion: " << my_first_motion_micrograph_FCC << '(' << motion2total[my_first_motion_micrograph_FCC] << ')' <<
						 " ... " << my_last_motion_micrograph_FCC << '(' << motion2total[my_last_motion_micrograph_FCC] << ')' << "\n";
		}
		
		motionEstimator.process(motionMdts, my_first_motion_micrograph_FCC, my_last_motion_micrograph_FCC, true);
	}

	if (debug)
	{
		std::cout << "Node " << node->rank << ": "
				  << "end FCC batch \n";
	}
}

void MotionRefinerMpi::runWithRecombination()
{
	double k_out_A = reference.pixToAng(reference.k_out);

	const int total_nr_micrographs = chosenMdts.size();
	const int motion_nr_micrographs = motionMdts.size();

	const int lastTotalMgForFCC = lastTotalMicrographForFCC();

	int lastMotionMgForFCC = subtractFinishedMicrographs(lastTotalMgForFCC, motionUnfinished);
	int lastRecombMgForFCC = subtractFinishedMicrographs(lastTotalMgForFCC, recombUnfinished);

	const int firstMotionMgWithoutFCC = lastMotionMgForFCC + 1;
	const int firstTotalMgWithoutFCC = lastTotalMgForFCC + 1;

	if (debug)
	{
		std::cout << "Node " << node->rank << ": "
				  << "begin post-FCC batch \n";
	}
	
	if (recombineFrames)
	{
		std::vector<int> total2motion = getForwardIndices(motionUnfinished);
		std::vector<int> total2recomb = getForwardIndices(recombUnfinished);


		long int my_first_recomb_micrograph_FCC, my_last_recomb_micrograph_FCC;
		const long int n_rec = lastRecombMgForFCC + 1;
		std::vector<double> weights_rec(std::max<long int>(0, n_rec));
		for (long int i = 0; i < n_rec; i++)
			weights_rec[i] = std::max(1.0, (double)recombMdts[i].numberOfObjects());
		divideWeighted(weights_rec, node->size, node->rank,
			my_first_recomb_micrograph_FCC, my_last_recomb_micrograph_FCC);


		long int my_first_total_micrograph_no_FCC, my_last_total_micrograph_no_FCC;
		const long int n_no_fcc = total_nr_micrographs - firstTotalMgWithoutFCC;
		std::vector<double> weights_no_fcc(std::max<long int>(0, n_no_fcc));
		for (long int i = 0; i < n_no_fcc; i++)
			weights_no_fcc[i] = std::max(1.0, (double)chosenMdts[firstTotalMgWithoutFCC + i].numberOfObjects());
		divideWeighted(weights_no_fcc, node->size, node->rank,
			my_first_total_micrograph_no_FCC, my_last_total_micrograph_no_FCC);

		my_first_total_micrograph_no_FCC += firstTotalMgWithoutFCC;
		my_last_total_micrograph_no_FCC += firstTotalMgWithoutFCC;


		// Recombine movies that have already been aligned:

		frameRecombiner.init(
			allMdts, 
			verb, reference.s, fc, k_out_A, reference.angpix,
			nr_omp_threads, outPath, debug,
			&reference, &obsModel, &micrographHandler);

		if (debug)
		{
			std::vector<int> recomb2total = getBackwardIndices(recombUnfinished);

			std::cout << "Node " << node->rank << ": "
					  << "motion: " << my_first_recomb_micrograph_FCC << '(' << recomb2total[my_first_recomb_micrograph_FCC] << ')' <<
						 " ... " << my_last_recomb_micrograph_FCC << '(' << recomb2total[my_last_recomb_micrograph_FCC] << ')' << "\n";
		}
		
		frameRecombiner.process(recombMdts, my_first_recomb_micrograph_FCC, my_last_recomb_micrograph_FCC);
		

		// Then, align and recombine in an alternating pattern to minimize disk I/O:

		motionEstimator.setVerbosity(0);
		frameRecombiner.setVerbosity(0);

		const int left = my_last_total_micrograph_no_FCC - my_first_total_micrograph_no_FCC + 1;
		const int barstep = XMIPP_MAX(1, left / 60);

		if (verb > 0)
		{
			std::cout << " + Aligning and combining frames for micrographs ... " << std::endl;
			init_progress_bar(left);
		}

		for (int m = my_first_total_micrograph_no_FCC; m <= my_last_total_micrograph_no_FCC; m++)
		{
			if (estimateMotion && motionUnfinished[m])
			{
				if (debug)
				{
					std::cout << "Node " << node->rank << ": "
							  << "motion: " << total2motion[m] << '(' << m << ')' << "\n";
				}

				motionEstimator.process(motionMdts, total2motion[m], total2motion[m], false,
					m < my_last_total_micrograph_no_FCC? &chosenMdts[m+1] : 0);
			}

			if (recombUnfinished[m])
			{
				if (debug)
				{
					std::cout << "Node " << node->rank << ": "
							  << "recomb: " << total2recomb[m] << '(' << m << ')' << "\n";
				}

				frameRecombiner.process(recombMdts, total2recomb[m], total2recomb[m],
					m < my_last_total_micrograph_no_FCC? &chosenMdts[m+1] : 0);
			}

			const int nr_done = m - firstTotalMgWithoutFCC;

			if (verb > 0 && nr_done % barstep == 0)
			{
				progress_bar(nr_done);
			}
		}

		if (verb > 0)
		{
			progress_bar(left);
		}
	}
	else
	{
		long int my_first_motion_micrograph_no_FCC, my_last_motion_micrograph_no_FCC;
		const long int n_mot_no_fcc = motion_nr_micrographs - firstMotionMgWithoutFCC;
		std::vector<double> weights_mot_no_fcc(std::max<long int>(0, n_mot_no_fcc));
		for (long int i = 0; i < n_mot_no_fcc; i++)
		{
			const double pc = (double)motionMdts[firstMotionMgWithoutFCC + i].numberOfObjects();
			weights_mot_no_fcc[i] = std::max(1.0, pc * pc);
		}
		divideWeighted(weights_mot_no_fcc, node->size, node->rank,
			my_first_motion_micrograph_no_FCC, my_last_motion_micrograph_no_FCC);

		my_first_motion_micrograph_no_FCC += firstMotionMgWithoutFCC;
		my_last_motion_micrograph_no_FCC += firstMotionMgWithoutFCC;

		// no recombination: just align the remaining movies

		if (debug)
		{
			std::vector<int> motion2total = getBackwardIndices(motionUnfinished);

			std::cout << "Node " << node->rank << ": "
					  << "motion: " << my_first_motion_micrograph_no_FCC << '(' << motion2total[my_first_motion_micrograph_no_FCC] << ')' <<
						 " ... " << my_last_motion_micrograph_no_FCC << '(' << motion2total[my_last_motion_micrograph_no_FCC] << ')' << "\n";
		}

		motionEstimator.process(
				motionMdts, my_first_motion_micrograph_no_FCC, my_last_motion_micrograph_no_FCC, false);
	}

	if (debug)
	{
		std::cout << "Node " << node->rank << ": "
				  << "end post-FCC batch \n";
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	if (generateStar && node->isLeader())
	{
		combineEPSAndSTARfiles();
	}
}
