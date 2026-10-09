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

#include "src/prefetch.h"
#include "src/error.h"
#include <sys/time.h>
#include <sys/resource.h>
#include <list>
#include <memory>

// How many particle stacks the prefetcher keeps open: an eighth of the process's file
// descriptor limit (each stack may hold two streams, and the rest is left for other
// files), at least 8, at most 4096
static size_t maxOpenStacks()
{
	struct rlimit rl;
	size_t lim = 1024;
	if (getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY)
		lim = rl.rlim_cur;
	return std::max<size_t>(8, std::min<size_t>(4096, lim / 8));
}

AsyncImagePrefetcher::AsyncImagePrefetcher(Experiment *mydata)
	: mydata_(mydata),
	  fill_idx_(0),
	  ready_idx_(-1),
	  first_part_id_(0),
	  last_part_id_(0),
	  has_work_(false),
	  work_done_(false),
	  shutdown_(false)
{
	worker_thread_ = std::thread(&AsyncImagePrefetcher::workerThread, this);
}

AsyncImagePrefetcher::~AsyncImagePrefetcher()
{
	stop();
}

void AsyncImagePrefetcher::startPrefetch(long int first_part_id, long int last_part_id)
{
	std::lock_guard<std::mutex> lock(mtx_);
	first_part_id_ = first_part_id;
	last_part_id_ = last_part_id;
	has_work_ = true;
	work_done_ = false;
	cv_start_.notify_one();
}

bool AsyncImagePrefetcher::waitAndSwap(std::vector<MultidimArray<RFLOAT>>& target, long int first_part_id, long int last_part_id)
{
	std::unique_lock<std::mutex> lock(mtx_);
	cv_done_.wait(lock, [this]() { return ready_idx_ != -1 || shutdown_; });
	if (shutdown_)
		return false;
	if (ready_first_ != first_part_id || ready_last_ != last_part_id)
		REPORT_ERROR("AsyncImagePrefetcher: the prefetched images are for particles " + integerToString(ready_first_) + "-" +
		             integerToString(ready_last_) + ", not the requested " + integerToString(first_part_id) + "-" + integerToString(last_part_id));

	target.swap(buffers_[ready_idx_]);
	ready_idx_ = -1;
	work_done_ = false;
	has_work_ = false;

	return true;
}

void AsyncImagePrefetcher::stop()
{
	{
		std::lock_guard<std::mutex> lock(mtx_);
		shutdown_ = true;
		cv_start_.notify_one();
	}
	if (worker_thread_.joinable())
		worker_thread_.join();
}

void AsyncImagePrefetcher::workerThread()
{
	// Stacks stay open across pools, the least recently used closed beyond the limit.
	// Particles come in random order, so keeping only the last stack open meant
	// opening a stack for nearly every image - over a network filesystem the reading
	// thread then could not keep up with the GPU, and all threads waited for it.
	const size_t max_open = maxOpenStacks();
	std::list<std::string> lru; // most recently used first
	std::unordered_map<std::string, std::pair<std::unique_ptr<fImageHandler>, std::list<std::string>::iterator> > open_stacks;

	while (true)
	{
		long int my_first, my_last;
		{
			std::unique_lock<std::mutex> lock(mtx_);
			cv_start_.wait(lock, [this]() { return has_work_ || shutdown_; });
			if (shutdown_)
				return;
			my_first = first_part_id_;
			my_last = last_part_id_;
			// Take the job: left set, the loop came straight back and read the same range
			// again into the other buffer, and that stale copy could be handed out for the
			// next range when the main thread was fast (particles got the wrong images)
			has_work_ = false;
		}

		std::vector<MultidimArray<RFLOAT>>& buffer = buffers_[fill_idx_];
		buffer.clear();
		buffer.reserve(my_last - my_first + 1);

		for (long int part_id_sorted = my_first; part_id_sorted <= my_last; part_id_sorted++)
		{
			long int part_id = mydata_->sorted_idx[part_id_sorted];

			FileName fn_img;
			bool on_scratch = mydata_->getImageNameOnScratch(part_id, fn_img);
			if (!on_scratch)
				fn_img = mydata_->particles[part_id].name;

			long int imgno;
			FileName fn_stack;
			fn_img.decompose(imgno, fn_stack);
			auto it = open_stacks.find(fn_stack);
			if (it == open_stacks.end())
			{
				if (open_stacks.size() >= max_open)
				{
					open_stacks.erase(lru.back());
					lru.pop_back();
				}
				std::unique_ptr<fImageHandler> h(new fImageHandler());
				h->openFile(fn_stack, WRITE_READONLY);
				lru.push_front(fn_stack);
				it = open_stacks.emplace(fn_stack, std::make_pair(std::move(h), lru.begin())).first;
			}
			else
				lru.splice(lru.begin(), lru, it->second.second);
			fImageHandler &hFile = *it->second.first;

			Image<RFLOAT> img;
			struct timeval t0, t1;
			gettimeofday(&t0, NULL);
			img.readFromOpenFile(fn_img, hFile, -1, false);
			gettimeofday(&t1, NULL);
			long int dt = (t1.tv_sec - t0.tv_sec) * 1000000 + (t1.tv_usec - t0.tv_usec);
			if (mydata_->cacheMode)
				read_time_cache_us_ += dt;
			else if (!on_scratch)
				read_time_original_us_ += dt;
			else
				read_time_scratch_us_ += dt;

			img().setXmippOrigin();
			buffer.push_back(img());
		}

		{
			std::lock_guard<std::mutex> lock(mtx_);
			ready_idx_ = fill_idx_;
			ready_first_ = my_first;
			ready_last_ = my_last;
			fill_idx_ = 1 - fill_idx_;
			cv_done_.notify_one();
		}
	}
}

AsyncReconstructPrefetcher::AsyncReconstructPrefetcher(const MetaDataTable *df,
	                                                   const std::vector<long int> &my_parts,
	                                                   int subset,
	                                                   int chosen_class,
	                                                   int max_queue)
	: df_(df),
	  my_parts_(my_parts),
	  subset_(subset),
	  chosen_class_(chosen_class),
	  max_queue_(max_queue),
	  stop_requested_(false),
	  done_(false),
	  nr_parts_(0)
{}

AsyncReconstructPrefetcher::~AsyncReconstructPrefetcher()
{
	stop();
}

void AsyncReconstructPrefetcher::start(long int nr_parts)
{
	nr_parts_ = nr_parts;
	stop_requested_ = false;
	done_ = false;
	worker_thread_ = std::thread(&AsyncReconstructPrefetcher::workerThread, this);
}

bool AsyncReconstructPrefetcher::waitAndPop(long int expected_ipart, Image<RFLOAT> &img)
{
	std::unique_lock<std::mutex> lock(mtx_);
	cv_item_ready_.wait(lock, [this, expected_ipart]() {
		return store_.count(expected_ipart) || done_ || stop_requested_;
	});

	auto it = store_.find(expected_ipart);
	if (it == store_.end())
		return false;

	img = std::move(it->second);
	store_.erase(it);
	lock.unlock();
	cv_slot_free_.notify_one();

	return true;
}

void AsyncReconstructPrefetcher::stop()
{
	{
		std::lock_guard<std::mutex> lock(mtx_);
		stop_requested_ = true;
	}
	cv_item_ready_.notify_all();
	cv_slot_free_.notify_all();

	if (worker_thread_.joinable())
		worker_thread_.join();

	std::lock_guard<std::mutex> lock(mtx_);
	store_.clear();
}

bool AsyncReconstructPrefetcher::shouldRead(long int ipart) const
{
	int randSubset = 0, classid = 0;
	df_->getValue(EMDL_PARTICLE_RANDOM_SUBSET, randSubset, ipart);
	df_->getValue(EMDL_PARTICLE_CLASS, classid, ipart);

	if (subset_ >= 1 && subset_ <= 2 && randSubset != subset_)
		return false;

	if (chosen_class_ >= 0 && chosen_class_ != classid)
		return false;

	return true;
}

void AsyncReconstructPrefetcher::workerThread()
{
	for (long int ipart : my_parts_)
	{
		if (stop_requested_)
			break;

		if (!shouldRead(ipart))
			continue;

		FileName fn_img;
		df_->getValue(EMDL_IMAGE_NAME, fn_img, ipart);
		Image<RFLOAT> img_tmp;
		img_tmp.read(fn_img);
		img_tmp().setXmippOrigin();

		std::unique_lock<std::mutex> lock(mtx_);
		cv_slot_free_.wait(lock, [this]()
		{
			return stop_requested_ || (int)store_.size() < max_queue_;
		});

		if (stop_requested_)
			break;

		store_.emplace(ipart, std::move(img_tmp));
		lock.unlock();
		cv_item_ready_.notify_one();
	}

	{
		std::lock_guard<std::mutex> lock(mtx_);
		done_ = true;
	}
	cv_item_ready_.notify_all();
}
