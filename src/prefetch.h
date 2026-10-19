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
#include <list>
#include <memory>
#include <string>
#include "src/filename.h"
#include "src/image.h"
#include "src/exp_model.h"
#include "src/metadata_table.h"

// Particle stacks kept open between reads, the least recently used closed beyond a
// limit derived from the open-file limit (which is raised as needed). Particles are
// read in random order, so keeping only the last stack open meant opening a stack
// for nearly every image - slow on a network filesystem. Not thread-safe: one
// reading thread per instance.
class OpenStacks
{
public:
	OpenStacks();
	/// at most max_open stacks (several instances share the open-file budget)
	explicit OpenStacks(size_t max_open);
	fImageHandler &get(const FileName &fn_stack);
private:
	size_t max_open_;
	std::list<std::string> lru_; // most recently used first
	std::unordered_map<std::string, std::pair<std::unique_ptr<fImageHandler>, std::list<std::string>::iterator> > open_;
};

/// Reads the needed images of whole particle stacks ahead (stack by stack, in order) of the E-step so that the per-particle reads of the prefetcher
/// find them in the page cache (RELION_PREFETCH_WARM=1). Particles are read in random order in small
/// pieces, which a network file system answers slowly; reading whole stacks in parallel is several times
/// faster. Only reads data, so results are unchanged. The look-ahead starts at a few pools, doubles
/// whenever the E-step had to wait for a pool, and is capped by a share of the free memory (the page
/// cache gives way under memory pressure, so the worst case is lost speed).
class StackWarmer
{
public:
	StackWarmer(Experiment *mydata, int nr_threads);
	~StackWarmer();
	/// the pool [first,last] (positions in sorted order) is being handed to the E-step
	void consumed(long int first, long int last, bool late);
	std::string report() const;
private:
	void run();
	Experiment *mydata_;
	int nr_threads_;
	std::mutex mtx_;
	std::condition_variable cv_;
	bool stop_ = false;
	long int consumed_ = -1, pool_size_ = 0, lookahead_ = 0, lookahead_max_ = 0, pos_ = 0, first_ = -1;
	std::atomic<long int> bytes_{0}, nr_warmed_{0}, nr_cached_{0}, nr_sparse_{0}, busy_us_{0};
	std::thread thread_;
};

class AsyncImagePrefetcher
{
public:
	/// max_readers: the most threads that may read one pool; starts with one and adds readers while the
	/// E-step has to wait for pools (a network file system answers many requests at once)
	AsyncImagePrefetcher(Experiment *mydata, int max_readers = 1);
	~AsyncImagePrefetcher();

	void startPrefetch(long int first_part_id, long int last_part_id);
	/// Wait for the images of particles [first_part_id, last_part_id] (the range passed to
	/// the last startPrefetch) and swap them into `target`. Any other range is an error, unless
	/// `speculative` (the range was only a guess): then nothing is swapped, the guess is dropped and
	/// false is returned (also when no pool was started).
	bool waitAndSwap(std::vector<MultidimArray<RFLOAT>>& target, long int first_part_id, long int last_part_id, bool speculative = false);
	void stop();

	long int readTimeOriginalUs() const { return read_time_original_us_; }
	long int readTimeScratchUs() const  { return read_time_scratch_us_; }
	long int readTimeCacheUs() const    { return read_time_cache_us_; }
	/// Time the main thread spent waiting for a pool, and the number of pools and of pools that were late
	long int waitTimeUs() const { return wait_time_us_; }
	long int nrPools() const    { return nr_pools_; }
	long int nrLatePools() const { return nr_late_pools_; }
	/// Wall time the reader needed for each pool, summed
	long int poolReadTimeUs() const { return pool_read_time_us_; }
	int readers() const { return active_readers_; }
	/// one line about the stack warm-up, empty when it is off
	std::string warmReport() const { return warmer_ ? warmer_->report() : std::string(); }

private:
	void workerThread();

	Experiment *mydata_;

	std::vector<MultidimArray<RFLOAT>> buffers_[2];
	int fill_idx_;
	int ready_idx_;

	long int first_part_id_;
	long int last_part_id_;
	long int ready_first_ = -1, ready_last_ = -1; // range held by buffers_[ready_idx_]
	bool has_work_;
	bool work_done_;
	bool pending_ = false; // a pool was started and not yet taken

	std::mutex mtx_;
	std::condition_variable cv_start_;
	std::condition_variable cv_done_;

	bool shutdown_;
	std::thread worker_thread_;

	std::atomic<long int> read_time_original_us_{0};
	std::atomic<long int> read_time_scratch_us_{0};
	std::atomic<long int> read_time_cache_us_{0};
	std::atomic<long int> pool_read_time_us_{0};
	int max_readers_ = 1;
	std::atomic<int> active_readers_{1};
	std::atomic<bool> last_pool_late_{false};
	long int wait_time_us_ = 0, nr_pools_ = 0, nr_late_pools_ = 0;
	std::unique_ptr<StackWarmer> warmer_;
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
	                          const std::vector<long int> &my_parts,
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
	std::vector<long int> my_parts_;
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
