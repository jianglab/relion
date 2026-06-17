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

#ifndef PREFETCH_H_
#define PREFETCH_H_

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <vector>
#include <unordered_map>
#include "src/filename.h"
#include "src/image.h"
#include "src/exp_model.h"
#include "src/metadata_table.h"

class AsyncImagePrefetcher
{
public:
	AsyncImagePrefetcher(Experiment *mydata);
	~AsyncImagePrefetcher();

	void startPrefetch(long int first_part_id, long int last_part_id);
	bool waitAndSwap(std::vector<MultidimArray<RFLOAT>>& target);
	void stop();

	long int readTimeOriginalUs() const { return read_time_original_us_; }
	long int readTimeScratchUs() const  { return read_time_scratch_us_; }
	long int readTimeCacheUs() const    { return read_time_cache_us_; }

private:
	void workerThread();

	Experiment *mydata_;

	std::vector<MultidimArray<RFLOAT>> buffers_[2];
	int fill_idx_;
	int ready_idx_;

	long int first_part_id_;
	long int last_part_id_;
	bool has_work_;
	bool work_done_;

	std::mutex mtx_;
	std::condition_variable cv_start_;
	std::condition_variable cv_done_;

	bool shutdown_;
	std::thread worker_thread_;

	std::atomic<long int> read_time_original_us_{0};
	std::atomic<long int> read_time_scratch_us_{0};
	std::atomic<long int> read_time_cache_us_{0};
};

class AsyncReconstructPrefetcher
{
public:
	struct Item
	{
		long int ipart;
		Image<RFLOAT> img;
	};

	AsyncReconstructPrefetcher(const MetaDataTable *df,
	                          int rank,
	                          int size,
	                          int subset,
	                          int chosen_class,
	                          int max_queue = 3);
	~AsyncReconstructPrefetcher();

	void start(long int nr_parts);
	bool waitAndPop(long int expected_ipart, Image<RFLOAT> &img);
	void stop();

private:
	void workerThread();
	bool shouldRead(long int ipart) const;

	const MetaDataTable *df_;
	int rank_;
	int size_;
	int subset_;
	int chosen_class_;
	int max_queue_;

	std::thread worker_thread_;
	std::unordered_map<long int, Image<RFLOAT>> store_;
	std::mutex mtx_;
	std::condition_variable cv_item_ready_;
	std::condition_variable cv_slot_free_;
	bool stop_requested_;
	bool done_;
	long int nr_parts_;
};

#endif
