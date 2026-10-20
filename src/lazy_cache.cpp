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

#include "src/lazy_cache.h"

#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lazycache
{

namespace
{

struct Item
{
	std::string cache, source;
};

struct State
{
	std::mutex mtx;
	std::string root;
	std::deque<Item> items;
	std::unordered_map<std::string, std::string> source_of; // cache path -> source path
	size_t next = 0;                                        // first item not yet taken by a worker

	std::atomic<long> bytes_copied{0}, nr_copied{0}, nr_present{0}, nr_elsewhere{0}, nr_fallback_opens{0};
	long bytes_queued = 0, nr_queued = 0;
	std::atomic<bool> stop_requested{false}, failed{false};
	std::atomic<bool> manager_running{false};
	std::atomic<int> active_workers{0};
	std::atomic<int> threads_used{0};
	std::thread manager;
	int max_threads = 1;
	int verb = 0;
	bool reported = true; // nothing started yet
	std::chrono::steady_clock::time_point t_start;
	std::atomic<long> ms_last_copy{0}; // time of the last finished copy, since t_start

	~State() { stopAndReport(); }
	void stopAndReport();
};

State &state()
{
	static State s;
	return s;
}

bool sameFile(const struct stat &a, const struct stat &b)
{
	return a.st_size == b.st_size && a.st_mtime == b.st_mtime;
}

void makeParents(const std::string &path)
{
	for (size_t pos = path.find('/', 1); pos != std::string::npos; pos = path.find('/', pos + 1))
		mkdir(path.substr(0, pos).c_str(), 0777);
}

void warnOnce(State &s, const std::string &msg)
{
	if (!s.failed.exchange(true))
		std::cerr << " WARNING (lazy cache): " << msg << " The cache will not be filled any further; "
		          << "particles keep being read from the original files." << std::endl;
}

// Copy one stack into the cache, or leave it to the process that holds its lock. Returns false when
// copying should stop altogether (out of space and the like).
bool copyOne(State &s, const Item &it)
{
	struct stat ss, cs;
	if (stat(it.source.c_str(), &ss) != 0)
		return true; // source gone: reads of it will report the error

	if (stat(it.cache.c_str(), &cs) == 0 && sameFile(ss, cs))
	{
		s.nr_present++;
		return true;
	}

	makeParents(it.cache);
	const std::string lock_path = it.cache + ".lazylock";
	const std::string tmp_path = it.cache + ".lazytmp";

	int lfd = open(lock_path.c_str(), O_CREAT | O_RDWR, 0666);
	if (lfd < 0)
	{
		warnOnce(s, "cannot create " + lock_path + ": " + strerror(errno) + ".");
		return false;
	}
	if (flock(lfd, LOCK_EX | LOCK_NB) != 0)
	{
		close(lfd);
		s.nr_elsewhere++; // another process is copying it
		return true;
	}

	// finished by someone else between the first check and taking the lock
	if (stat(it.cache.c_str(), &cs) == 0 && sameFile(ss, cs))
	{
		unlink(lock_path.c_str());
		close(lfd);
		s.nr_present++;
		return true;
	}

	bool ok = false;
	std::string why;
	int in = open(it.source.c_str(), O_RDONLY);
	int out = (in >= 0) ? open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666) : -1;
	if (in < 0 || out < 0)
		why = std::string("cannot open ") + (in < 0 ? it.source : tmp_path) + ": " + strerror(errno) + ".";
	else
	{
		posix_fadvise(in, 0, 0, POSIX_FADV_SEQUENTIAL);
		std::vector<char> buf(4 << 20);
		long int done = 0;
		ok = true;
		while (done < (long int)ss.st_size && ok)
		{
			if (s.stop_requested)
			{
				ok = false;
				break;
			}
			ssize_t n = read(in, buf.data(), buf.size());
			if (n <= 0)
			{
				why = std::string("reading ") + it.source + " failed: " + (n < 0 ? strerror(errno) : "unexpected end of file") + ".";
				ok = false;
				break;
			}
			for (ssize_t w = 0; w < n && ok;)
			{
				ssize_t m = write(out, buf.data() + w, n - w);
				if (m < 0)
				{
					why = std::string("writing ") + tmp_path + " failed: " + strerror(errno) + ".";
					ok = false;
				}
				else
					w += m;
			}
			done += n;
			s.bytes_copied += n;
		}
		if (ok)
		{
			struct timespec times[2] = { ss.st_atim, ss.st_mtim };
			futimens(out, times);
		}
	}
	if (in >= 0) close(in);
	if (out >= 0) close(out);

	if (ok && rename(tmp_path.c_str(), it.cache.c_str()) != 0)
	{
		why = std::string("cannot rename ") + tmp_path + ": " + strerror(errno) + ".";
		ok = false;
	}
	if (!ok)
		unlink(tmp_path.c_str());
	unlink(lock_path.c_str());
	close(lfd);

	if (ok)
		s.nr_copied++;
		s.ms_last_copy = (long)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - s.t_start).count();
	if (!why.empty())
		warnOnce(s, why);
	return ok || why.empty(); // a stop request is not an error
}

bool take(State &s, Item &out)
{
	std::lock_guard<std::mutex> lock(s.mtx);
	if (s.next >= s.items.size())
		return false;
	out = s.items[s.next++];
	return true;
}

void managerMain(State *sp)
{
	State &s = *sp;
	while (true)
	{
		{
			std::lock_guard<std::mutex> lock(s.mtx);
			if (s.next >= s.items.size() || s.stop_requested || s.failed)
			{
				s.manager_running = false;
				return;
			}
		}

		std::vector<std::thread> workers;
		int n_threads = 0;
		auto launch = [&]()
		{
			s.active_workers++;
			n_threads++;
			if (n_threads > s.threads_used)
				s.threads_used = n_threads;
			workers.emplace_back([&s]()
			{
				Item it;
				while (!s.stop_requested && !s.failed && take(s, it))
					if (!copyOne(s, it))
						break;
				s.active_workers--;
			});
		};
		launch();

		// Add a thread while the copy rate keeps growing by at least 15% per added thread (a network
		// file system answers many requests at once, a single disc does not), up to max_threads
		const double interval_s = 3.0;
		double prev_rate = -1.;
		long prev_bytes = s.bytes_copied;
		bool growing = true;
		auto prev_t = std::chrono::steady_clock::now();
		while (s.active_workers > 0 && !s.stop_requested)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			auto now = std::chrono::steady_clock::now();
			double dt = std::chrono::duration<double>(now - prev_t).count();
			if (dt < interval_s)
				continue;
			long b = s.bytes_copied;
			double rate = (b - prev_bytes) / dt;
			if (growing && n_threads < s.max_threads && s.active_workers > 0)
			{
				if (prev_rate < 0. || rate > 1.15 * prev_rate)
					launch();
				else
					growing = false;
			}
			prev_rate = rate;
			prev_bytes = b;
			prev_t = now;
		}
		for (auto &w : workers)
			w.join();
	}
}

} // namespace

void State::stopAndReport()
{
	stop_requested = true;
	if (manager.joinable())
		manager.join();
	if (!reported && verb > 0)
	{
		double secs = ms_last_copy / 1000.;
		std::cout << " Lazy cache: copied " << nr_copied << " of " << nr_queued << " stacks ("
		          << std::fixed << std::setprecision(1) << bytes_copied / 1073741824.0 << " GB) in the first "
		          << std::setprecision(0) << secs << " s with up to " << threads_used << " thread(s); "
		          << nr_present << " were already cached, " << nr_elsewhere << " copied by another process, "
		          << nr_queued - nr_copied - nr_present - nr_elsewhere << " not copied (job ended or copy failed). "
		          << nr_fallback_opens << " stack opens read the original file." << std::endl;
	}
	reported = true;

	// back to the initial state (at exit nothing follows; a unit test can start over)
	std::lock_guard<std::mutex> lock(mtx);
	items.clear();
	source_of.clear();
	next = 0;
	bytes_queued = nr_queued = 0;
	bytes_copied = nr_copied = nr_present = nr_elsewhere = nr_fallback_opens = ms_last_copy = 0;
	threads_used = 0;
	failed = false;
	stop_requested = false;
}

bool enabled()
{
	// read each time: only called when a file is missing or a cache is set up
	const char *e = getenv("RELION_CACHE_LAZY");
	return e != NULL && atoi(e) != 0;
}

void setRoot(const std::string &cache_root)
{
	if (!enabled())
		return;
	State &s = state();
	std::lock_guard<std::mutex> lock(s.mtx);
	s.root = cache_root;
}

void add(const std::string &cache_path, const std::string &source_path)
{
	State &s = state();
	struct stat ss, cs;
	if (stat(source_path.c_str(), &ss) != 0)
		return;
	if (stat(cache_path.c_str(), &cs) == 0)
	{
		if (sameFile(ss, cs))
			return; // already cached
		unlink(cache_path.c_str()); // stale: read the source until the new copy is in place
	}
	std::lock_guard<std::mutex> lock(s.mtx);
	if (!s.source_of.emplace(cache_path, source_path).second)
		return;
	s.items.push_back({cache_path, source_path});
	s.bytes_queued += ss.st_size;
	s.nr_queued++;
}

void start(int max_threads, int verb)
{
	State &s = state();
	std::lock_guard<std::mutex> lock(s.mtx);
	if (s.next >= s.items.size() || s.failed)
		return;
	if (s.manager_running)
		return; // picks up the new items by itself
	if (s.manager.joinable())
		s.manager.join();
	s.max_threads = max_threads < 1 ? 1 : max_threads;
	s.verb = verb;
	if (s.reported)
	{
		s.t_start = std::chrono::steady_clock::now();
		s.reported = false;
		if (verb > 0)
			std::cout << " Lazy cache (RELION_CACHE_LAZY): " << s.items.size() - s.next << " stacks ("
			          << std::fixed << std::setprecision(1) << s.bytes_queued / 1073741824.0
			          << " GB) are copied in the background while the job runs, with up to " << s.max_threads
			          << " threads (more only while the copy rate grows). Until a stack is copied, its particles are read from the original file."
			          << std::endl;
	}
	s.manager_running = true;
	s.manager = std::thread(managerMain, &s);
}

std::string resolve(const std::string &cache_path)
{
	if (!enabled())
		return "";
	State &s = state();
	std::lock_guard<std::mutex> lock(s.mtx);
	auto it = s.source_of.find(cache_path);
	if (it != s.source_of.end())
	{
		s.nr_fallback_opens++;
		return it->second;
	}
	// A process that did not queue the stack: <root><key>/<path of the original, without leading '/'>
	if (!s.root.empty() && cache_path.compare(0, s.root.size(), s.root) == 0)
	{
		size_t slash = cache_path.find('/', s.root.size());
		if (slash != std::string::npos)
		{
			std::string rel = cache_path.substr(slash + 1);
			for (const std::string &cand : { rel, "/" + rel })
				if (access(cand.c_str(), R_OK) == 0)
				{
					s.nr_fallback_opens++;
					return cand;
				}
		}
	}
	return "";
}

void stop()
{
	state().stopAndReport();
}

} // namespace lazycache
