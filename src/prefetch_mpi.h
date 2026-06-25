/***************************************************************************
 *
 * Author: "Wen Jiang"
 * Pennsylvania State University
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

#ifndef PREFETCH_MPI_H_
#define PREFETCH_MPI_H_

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <string>
#include "src/filename.h"
#include "src/image.h"
#include "src/exp_model.h"
#include "src/metadata_table.h"
#include "src/mpi.h"

class MpiAsyncImagePrefetcher
{
public:
	MpiAsyncImagePrefetcher(Experiment *mydata, MpiNode *node);
	~MpiAsyncImagePrefetcher();

	void start();

	bool waitForJob(std::vector<MultidimArray<RFLOAT>>& images,
	                MultidimArray<RFLOAT>& metadata,
	                long int& job_first,
	                long int& job_last);

	void stop();

private:
	struct JobData
	{
		std::vector<MultidimArray<RFLOAT>> images;
		MultidimArray<RFLOAT> metadata;
		long int job_first;
		long int job_last;
		int nimg;
	};

	void workerThread();

	Experiment *mydata_;
	MpiNode *node_;

	JobData buffers_[2];
	int fill_idx_;
	int ready_idx_;

	bool has_work_;
	bool shutdown_;

	std::mutex mtx_;
	std::condition_variable cv_fill_;
	std::condition_variable cv_ready_;

	std::thread worker_thread_;
};

#endif
