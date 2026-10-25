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
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

// How many particle stacks the prefetcher keeps open. The soft limit on open files is
// first raised towards the hard limit (at most 65536), as many programs do: it is
// often 1024, while a data set can have thousands of stacks. Then an eighth of it is
// used (each stack may hold two streams, and the rest is left for other files), at
// least 8 and at most 8192.
static size_t maxOpenStacks()
{
	struct rlimit rl;
	size_t lim = 1024;
	if (getrlimit(RLIMIT_NOFILE, &rl) == 0)
	{
		rlim_t want = (rl.rlim_max == RLIM_INFINITY) ? 65536 : std::min<rlim_t>(rl.rlim_max, 65536);
		if (rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < want)
		{
			struct rlimit raised = rl;
			raised.rlim_cur = want;
			if (setrlimit(RLIMIT_NOFILE, &raised) == 0)
				rl = raised;
		}
		lim = (rl.rlim_cur == RLIM_INFINITY) ? 65536 : rl.rlim_cur;
	}
	return std::max<size_t>(8, std::min<size_t>(8192, lim / 8));
}

OpenStacks::OpenStacks() : max_open_(maxOpenStacks()) {}
OpenStacks::OpenStacks(size_t max_open) : max_open_(std::max<size_t>(2, std::min(max_open, maxOpenStacks()))) {}

fImageHandler &OpenStacks::get(const FileName &fn_stack)
{
	auto it = open_.find(fn_stack);
	if (it == open_.end())
	{
		if (open_.size() >= max_open_)
		{
			open_.erase(lru_.back());
			lru_.pop_back();
		}
		std::unique_ptr<fImageHandler> h(new fImageHandler());
		h->openFile(fn_stack, WRITE_READONLY);
		lru_.push_front(fn_stack);
		it = open_.emplace(fn_stack, std::make_pair(std::move(h), lru_.begin())).first;
	}
	else
		lru_.splice(lru_.begin(), lru_, it->second.second);
	return *it->second.first;
}

// ------------------------------- StackWarmer -------------------------------

static long int memAvailableBytes()
{
	FILE *f = fopen("/proc/meminfo", "r");
	if (!f) return 0;
	char line[256];
	long int kb = 0;
	while (fgets(line, sizeof(line), f))
		if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break;
	fclose(f);
	return kb * 1024;
}

// fraction of the file that is in the page cache (-1 when it cannot be told)
static double cachedFraction(int fd, size_t size)
{
	if (size == 0) return 1.;
	void *m = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
	if (m == MAP_FAILED) return -1.;
	const size_t page = sysconf(_SC_PAGESIZE), np = (size + page - 1) / page;
	std::vector<unsigned char> v(np);
	double r = (mincore(m, size, v.data()) == 0) ? (double) std::count_if(v.begin(), v.end(), [](unsigned char c) { return c & 1; }) / np : -1.;
	munmap(m, size);
	return r;
}

StackWarmer::StackWarmer(Experiment *mydata, int nr_threads) : mydata_(mydata), nr_threads_(std::max(1, nr_threads))
{
	thread_ = std::thread(&StackWarmer::run, this);
}

StackWarmer::~StackWarmer()
{
	{
		std::lock_guard<std::mutex> lock(mtx_);
		stop_ = true;
		cv_.notify_one();
	}
	if (thread_.joinable())
		thread_.join();
}

void StackWarmer::consumed(long int first, long int last, bool late)
{
	std::lock_guard<std::mutex> lock(mtx_);
	if (first_ < 0)
	{
		first_ = first;
		pos_ = first;
	}
	consumed_ = std::max(consumed_, last);
	pool_size_ = std::max(pool_size_, last - first + 1);
	if (lookahead_ == 0)
		lookahead_ = 8 * pool_size_;
	else if (late)
		lookahead_ = std::min(2 * lookahead_, std::max(lookahead_max_, lookahead_));
	cv_.notify_one();
}

std::string StackWarmer::report() const
{
	char buf[300];
	snprintf(buf, sizeof(buf), "Stack warm-up: %ld stacks read ahead (%.2f GB, %.1f s of reading with up to %d threads); look-ahead %ld particles (RELION_PREFETCH_WARM=0 turns it off).",
	         nr_warmed_.load(), bytes_.load() / 1e9, busy_us_.load() / 1e6, nr_threads_, lookahead_);
	return buf;
}

void StackWarmer::run()
{
	// the stack of every particle in sorted order is found when needed; how many particles of the
	// data set use each stack tells whether reading it whole is worth it
	const long int nr_total = mydata_->numberOfParticles();
	std::unordered_map<std::string, std::vector<long int> > needs;
	auto stackOf = [&](long int pos, long int *imgno_out = NULL) -> std::string
	{
		long int part_id = mydata_->sorted_idx[pos];
		FileName fn_img, fn_stack;
		long int imgno;
		if (!mydata_->getImageNameOnScratch(part_id, fn_img))
			fn_img = mydata_->particles[part_id].name;
		fn_img.decompose(imgno, fn_stack);
		if (imgno_out) *imgno_out = imgno;
		return fn_stack;
	};
	for (long int p = 0; p < nr_total; p++)
	{
		long int imgno;
		std::string s = stackOf(p, &imgno);
		needs[s].push_back(imgno);
	}
	for (auto &n : needs) std::sort(n.second.begin(), n.second.end());
	std::unordered_map<std::string, long int> uses, sizes;
	double img_bytes = 0.;
	for (auto &n : needs)
	{
		uses[n.first] = n.second.size();
		struct stat st;
		if (stat(n.first.c_str(), &st) != 0) { sizes[n.first] = -1; continue; }
		sizes[n.first] = st.st_size;
		double per = (double) st.st_size / n.second.size();
		if (img_bytes == 0. || per < img_bytes) img_bytes = per;
	}
	std::unordered_set<std::string> done;
	double bytes_per_particle = 131072.;
	long int scanned = 0;

	while (true)
	{
		std::vector<std::string> batch;
		long int batch_bytes = 0;
		{
			std::unique_lock<std::mutex> lock(mtx_);
			cv_.wait(lock, [&]() { return stop_ || (first_ >= 0 && pos_ < nr_total && pos_ <= consumed_ + lookahead_); });
			if (stop_) return;
			// memory: what the look-ahead may hold is a quarter of the available memory, at most 16 GB
			double budget = std::min(0.25 * memAvailableBytes(), 16e9);
			lookahead_max_ = std::max<long int>(2 * pool_size_, (long int) (budget / bytes_per_particle));
			lookahead_ = std::min(lookahead_, lookahead_max_);
			while (pos_ < nr_total && pos_ <= consumed_ + lookahead_ && batch.size() < 4 * (size_t) nr_threads_)
			{
				std::string s = stackOf(pos_++);
				scanned++;
				if (!done.insert(s).second) continue;
				long int sz = sizes[s];
				if (sz <= 0) continue;
				batch.push_back(s);
				batch_bytes += sz;
			}
		}
		if (batch.empty()) continue;
		struct timeval t0, t1;
		gettimeofday(&t0, NULL);
		std::atomic<size_t> next(0);
		auto work = [&]()
		{
			std::vector<char> buf(4 << 20);
			size_t i;
			while ((i = next++) < batch.size())
			{
				int fd = open(batch[i].c_str(), O_RDONLY);
				if (fd < 0) continue;
				struct stat st;
				if (fstat(fd, &st) == 0)
				{
					// MRC layout: 1024-byte header, nsymbt bytes of extended header, then nz images; only the
					// images that are needed are read, neighbours in one request. Other formats: the whole file.
					long int img = 0, start = 0;
					int hdr[256];
					if (pread(fd, hdr, sizeof(hdr), 0) == (ssize_t) sizeof(hdr))
					{
						int bpp = 0;
						switch (hdr[3]) { case 0: bpp = 1; break; case 1: case 6: case 12: bpp = 2; break; case 2: bpp = 4; break; default: bpp = 0; }
						if (bpp > 0 && hdr[0] > 0 && hdr[1] > 0 && hdr[23] >= 0)
						{
							img = (long int) hdr[0] * hdr[1] * bpp;
							start = 1024 + hdr[23];
						}
					}
					long int got = 0;
					if (img > 0)
					{
						const std::vector<long int> &v = needs.at(batch[i]);
						size_t k = 0;
						while (k < v.size())
						{
							size_t k2 = k;
							while (k2 + 1 < v.size() && v[k2 + 1] - v[k2] <= 3 && (v[k2 + 1] - v[k]) * img < (4 << 20)) k2++;
							off_t o0 = start + (v[k] - 1) * img;
							size_t len = (size_t) ((v[k2] - v[k] + 1) * img);
							if (o0 + (off_t) len <= st.st_size)
							{
								ssize_t n = pread(fd, buf.data(), std::min(len, buf.size()), o0);
								if (n > 0) got += n;
							}
							k = k2 + 1;
						}
					}
					else
					{
						posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
						ssize_t n;
						off_t off = 0;
						while ((n = pread(fd, buf.data(), buf.size(), off)) > 0) off += n;
						got = off;
					}
					bytes_ += got;
					nr_warmed_++;
				}
				close(fd);
			}
		};
		std::vector<std::thread> helpers;
		for (int r = 1; r < nr_threads_; r++) helpers.emplace_back(work);
		work();
		for (auto &h : helpers) h.join();
		gettimeofday(&t1, NULL);
		busy_us_ += (t1.tv_sec - t0.tv_sec) * 1000000 + (t1.tv_usec - t0.tv_usec);
		if (scanned > 0 && bytes_ > 0) bytes_per_particle = std::max(65536., (double) bytes_ / scanned);
	}
}

AsyncImagePrefetcher::AsyncImagePrefetcher(Experiment *mydata, int max_readers)
	: mydata_(mydata),
	  max_readers_(std::max(1, max_readers))
{
	// opt-in: RELION_PREFETCH_WARM=1 (or a thread count) reads whole stacks ahead (see StackWarmer)
	if (const char *e = getenv("RELION_PREFETCH_WARM"))
		if (strcmp(e, "0") != 0 && strcmp(e, "off") != 0 && mydata_->particles.size() > 0)
		{
			int nt = atoi(e);
			warmer_.reset(new StackWarmer(mydata_, nt > 1 ? nt : 8));
		}
	if (const char *e = getenv("RELION_PREFETCH_DEPTH"))
		if (atoi(e) > 0)
		{
			depth_ = max_depth_ = atoi(e);
			fixed_depth_ = true;
		}
	worker_thread_ = std::thread(&AsyncImagePrefetcher::workerThread, this);
}

AsyncImagePrefetcher::~AsyncImagePrefetcher()
{
	stop();
}

void AsyncImagePrefetcher::setReadAhead(long int last_part_id)
{
	std::lock_guard<std::mutex> lock(mtx_);
	ahead_last_ = last_part_id;
	cv_start_.notify_one();
}

void AsyncImagePrefetcher::startPrefetch(long int first_part_id, long int last_part_id)
{
	std::lock_guard<std::mutex> lock(mtx_);
	want_first_ = first_part_id;
	want_last_ = last_part_id;
	pool_size_ = last_part_id - first_part_id + 1;
	pending_ = true;
	cv_start_.notify_one();
	if (warmer_)
		warmer_->consumed(first_part_id, last_part_id, false);
}

bool AsyncImagePrefetcher::inReady(long int first) const
{
	for (const Pool &q : ready_)
		if (q.first == first) return true;
	return false;
}

bool AsyncImagePrefetcher::nextPool(long int &first, long int &last)
{
	// the pool asked for, unless it is read or being read; pools read ahead on another guess are of no use
	if (want_first_ >= 0 && reading_first_ != want_first_ && !inReady(want_first_))
	{
		ready_.clear();
		first = want_first_;
		last = want_last_;
		return true;
	}
	// read ahead: the pool after the last one read (or taken)
	if (ahead_last_ < 0 || pool_size_ <= 0 || reading_first_ >= 0 || (long int) ready_.size() >= depth_)
		return false;
	const long int tail = !ready_.empty() ? ready_.back().last : (want_first_ >= 0 ? want_last_ : taken_last_);
	if (tail < 0 || tail + 1 > ahead_last_)
		return false;
	first = tail + 1;
	last = std::min(ahead_last_, tail + pool_size_);
	return true;
}

bool AsyncImagePrefetcher::waitAndSwap(std::vector<MultidimArray<RFLOAT>>& target, long int first_part_id, long int last_part_id, bool speculative)
{
	std::unique_lock<std::mutex> lock(mtx_);
	if (speculative && !pending_ && !inReady(first_part_id))
		return false;
	struct timeval w0, w1;
	gettimeofday(&w0, NULL);
	bool late = !inReady(first_part_id);
	// until the pool is there, or will not come: neither asked for nor being read
	cv_done_.wait(lock, [&]() { return shutdown_ || inReady(first_part_id) ||
	                                   (want_first_ != first_part_id && reading_first_ != first_part_id); });
	gettimeofday(&w1, NULL);
	wait_time_us_ += (w1.tv_sec - w0.tv_sec) * 1000000 + (w1.tv_usec - w0.tv_usec);
	nr_pools_++;
	if (late)
	{
		nr_late_pools_++;
		last_pool_late_ = true;
	}
	if (warmer_)
		warmer_->consumed(first_part_id, last_part_id, late);
	if (shutdown_)
		return false;

	// pools before it are stale guesses
	while (!ready_.empty() && ready_.front().first != first_part_id && inReady(first_part_id))
		ready_.pop_front();
	if (ready_.empty() || ready_.front().first != first_part_id || ready_.front().last != last_part_id)
	{
		if (speculative)
		{
			ready_.clear();
			want_first_ = -1;
			pending_ = false;
			cv_start_.notify_one();
			return false;
		}
		REPORT_ERROR("AsyncImagePrefetcher: no prefetched images for particles " + integerToString(first_part_id) + "-" +
		             integerToString(last_part_id) + (want_first_ >= 0 ? ", the pool asked for was " + integerToString(want_first_) + "-" +
		             integerToString(want_last_) : std::string("")));
	}

	target.swap(ready_.front().imgs);
	ready_.pop_front();
	want_first_ = -1;
	taken_last_ = last_part_id;
	pending_ = false;
	cv_start_.notify_one();   // room to read ahead
	return true;
}

void AsyncImagePrefetcher::stop()
{
	{
		std::lock_guard<std::mutex> lock(mtx_);
		shutdown_ = true;
		cv_start_.notify_one();
		cv_done_.notify_all();
	}
	if (worker_thread_.joinable())
		worker_thread_.join();
}

void AsyncImagePrefetcher::adaptReaders()
{
	if (!last_pool_late_.exchange(false))
		return;
	const double us_per_img = level_imgs_ > 0 ? level_us_ / level_imgs_ : 0.;
	if (!readers_settled_ && prev_readers_ > 0 && prev_us_per_img_ > 0.)
	{
		if (us_per_img <= 0.)
			return;   // the last doubling is not measured yet
		if (us_per_img > 0.9 * prev_us_per_img_)
		{
			// it did not make reading faster: back to the readers before it
			active_readers_ = prev_readers_;
			readers_settled_ = true;
		}
		prev_readers_ = 0;
	}
	if (!readers_settled_)
	{
		if (active_readers_ < max_readers_)
		{
			prev_readers_ = active_readers_;
			prev_us_per_img_ = us_per_img;
			active_readers_ = std::min(max_readers_, 2 * active_readers_);
			level_us_ = 0.;
			level_imgs_ = 0;
			return;
		}
		readers_settled_ = true;
	}
	// more threads do not help: hold more pools ahead
	if (!fixed_depth_ && depth_ < max_depth_)
		depth_++;
}

void AsyncImagePrefetcher::readPool(Pool &p, std::vector<std::unique_ptr<OpenStacks> > &readers)
{
	struct timeval p0, p1;
	gettimeofday(&p0, NULL);
	const long int n_img = p.last - p.first + 1;
	p.imgs.resize(n_img);
	const int nr = std::min<long int>(active_readers_, n_img);

	std::atomic<long int> next(0);
	auto read_some = [&](int rid)
	{
		long int i;
		while ((i = next++) < n_img)
		{
			long int part_id = mydata_->sorted_idx[p.first + i];

			FileName fn_img;
			bool on_scratch = mydata_->getImageNameOnScratch(part_id, fn_img);
			if (!on_scratch)
				fn_img = mydata_->particles[part_id].name;

			long int imgno;
			FileName fn_stack;
			fn_img.decompose(imgno, fn_stack);
			fImageHandler &hFile = readers[rid]->get(fn_stack);

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
			p.imgs[i] = img();
		}
	};
	std::vector<std::thread> helpers;
	for (int r = 1; r < nr; r++)
		helpers.emplace_back(read_some, r);
	read_some(0);
	for (auto &h : helpers) h.join();

	gettimeofday(&p1, NULL);
	const long int us = (p1.tv_sec - p0.tv_sec) * 1000000 + (p1.tv_usec - p0.tv_usec);
	pool_read_time_us_ += us;
	level_us_ += us;
	level_imgs_ += n_img;

	// what the pools held ahead may take: a quarter of the available memory, at most 8 pools
	if (!fixed_depth_ && n_img > 0)
	{
		const double pool_bytes = (double) n_img * p.imgs[0].nzyxdim * sizeof(RFLOAT);
		max_depth_ = std::max(1, std::min(8, (int) (0.25 * memAvailableBytes() / std::max(1., pool_bytes))));
		if (depth_ > max_depth_) depth_ = max_depth_;
	}
}

void AsyncImagePrefetcher::workerThread()
{
	// one set of open stacks per reader, kept across pools (see OpenStacks); they share the open-file budget
	std::vector<std::unique_ptr<OpenStacks> > readers;
	for (int r = 0; r < max_readers_; r++)
		readers.emplace_back(new OpenStacks(maxOpenStacks() / max_readers_));

	while (true)
	{
		Pool p;
		{
			std::unique_lock<std::mutex> lock(mtx_);
			cv_start_.wait(lock, [&]() { return shutdown_ || nextPool(p.first, p.last); });
			if (shutdown_)
				return;
			reading_first_ = p.first;
		}
		adaptReaders();
		readPool(p, readers);
		{
			std::lock_guard<std::mutex> lock(mtx_);
			reading_first_ = -1;
			// a pool asked for meanwhile on another guess has made this one stale
			if (want_first_ < 0 || want_first_ <= p.first)
				ready_.push_back(std::move(p));
			cv_done_.notify_all();
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
