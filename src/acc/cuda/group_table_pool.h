#ifndef ACC_CUDA_GROUP_TABLE_POOL_H_
#define ACC_CUDA_GROUP_TABLE_POOL_H_
// Scratch for the reference-projection table of the particle-group coarse kernel (RELION_GPU_GROUP_PROJ_TABLE=1).
// One pool per process: the threads of a rank share a per-device byte budget (the budget divided by the threads that use
// the device, and at most a quarter of the free memory when it grows), a buffer is leased for one call, and the idle
// buffers are freed when the iteration's device bundle is released. Several MPI ranks on one GPU see each other through
// the free memory (each takes at most a quarter of what is free); RELION_GPU_GROUP_PROJ_TABLE_SHARERS divides the budget further.
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <vector>

struct GroupTablePool
{
	struct Slot { void *p = NULL; size_t cap = 0; int dev = 0; cudaEvent_t ev = NULL; bool busy = false; };
	std::mutex m;
	std::vector<Slot> slots;
	std::set<std::pair<int, void *> > users; // (device, thread token)
	size_t peak_held = 0;
	std::atomic<long> n_used{0}, n_texture{0};
	std::atomic<bool> said{false};
	std::vector<std::pair<cudaEvent_t, cudaEvent_t> > times; // RELION_GPU_GROUP_TIME=1: events around table+group launches of each set (guarded by m)
	std::atomic<long> n_parts{0};
	void addTime(cudaEvent_t a, cudaEvent_t b) { std::lock_guard<std::mutex> lk(m); times.push_back(std::make_pair(a, b)); }

	static GroupTablePool &get() { static GroupTablePool p; return p; }

	size_t held(int dev) const { size_t h = 0; for (const Slot &s : slots) if (s.dev == dev) h += s.cap; return h; }

	// A buffer of up to `want` bytes (at least `min_bytes`) for the calling thread, or NULL. `*idx` identifies the lease.
	void *acquire(int dev, size_t budget, size_t min_bytes, size_t want, cudaStream_t stream, int *idx, size_t *cap_out, size_t *free_out, size_t *share_out = NULL, size_t *total_cap_out = NULL, int *users_out = NULL)
	{
		static thread_local char token;
		static const double sharers = getenv("RELION_GPU_GROUP_PROJ_TABLE_SHARERS") ? std::max(1., atof(getenv("RELION_GPU_GROUP_PROJ_TABLE_SHARERS"))) : 1.;
		std::lock_guard<std::mutex> lk(m);
		users.insert(std::make_pair(dev, (void *) &token));
		int n_users = 0;
		for (const auto &u : users) if (u.first == dev) n_users++;
		size_t free_b = 0, total_b = 0;
		cudaMemGetInfo(&free_b, &total_b);
		*free_out = free_b;
		const size_t h = held(dev);
		const size_t total_cap = std::min((size_t) ((double) budget / sharers), (free_b + h) / 4);
		const size_t share = std::max(min_bytes, std::min(want, total_cap / (size_t) std::max(1, n_users)));
		*idx = -1;
		*cap_out = 0;
		if (share_out) *share_out = share;
		if (total_cap_out) *total_cap_out = total_cap;
		if (users_out) *users_out = n_users;
		// an idle buffer that is big enough (the smallest of those), else the largest idle one
		int best = -1, big = -1;
		for (size_t i = 0; i < slots.size(); i++)
		{
			const Slot &s = slots[i];
			if (s.busy || s.dev != dev || s.p == NULL) continue;
			if (s.cap >= share && (best < 0 || s.cap < slots[best].cap)) best = (int) i;
			if (big < 0 || s.cap > slots[big].cap) big = (int) i;
		}
		int use = best;
		if (use < 0)
		{
			size_t room = total_cap > h ? total_cap - h : 0;
			if (big >= 0)
			{
				// grow the largest idle buffer if the budget allows
				const size_t grown = std::min(share, slots[big].cap + room);
				if (grown >= min_bytes && grown > slots[big].cap)
				{
					cudaFree(slots[big].p);
					slots[big].p = NULL; slots[big].cap = 0;
					if (cudaMalloc(&slots[big].p, grown) == cudaSuccess) slots[big].cap = grown; else slots[big].p = NULL;
					cudaGetLastError();
					if (slots[big].p != NULL) use = big;
				}
				else if (slots[big].cap >= min_bytes)
					use = big;
				else if (slots[big].cap < min_bytes)
				{
					// a stale small buffer (made when little was free) that cannot grow: give it back
					cudaFree(slots[big].p);
					slots[big].p = NULL; slots[big].cap = 0;
					cudaGetLastError();
				}
			}
			else
			{
				const size_t make = std::min(share, room);
				if (make >= min_bytes)
				{
					Slot s;
					s.dev = dev;
					if (cudaMalloc(&s.p, make) == cudaSuccess)
					{
						s.cap = make;
						cudaEventCreateWithFlags(&s.ev, cudaEventDisableTiming);
						slots.push_back(s);
						use = (int) slots.size() - 1;
					}
					cudaGetLastError();
				}
			}
		}
		if (use < 0 || slots[use].p == NULL || slots[use].cap < min_bytes) return NULL;
		Slot &s = slots[use];
		if (s.ev == NULL) cudaEventCreateWithFlags(&s.ev, cudaEventDisableTiming);
		s.busy = true;
		cudaStreamWaitEvent(stream, s.ev, 0); // the previous lessee's launches have finished with it
		peak_held = std::max(peak_held, held(dev));
		*idx = use;
		*cap_out = s.cap;
		return s.p;
	}

	void release(int idx, cudaStream_t stream)
	{
		if (idx < 0) return;
		std::lock_guard<std::mutex> lk(m);
		cudaEventRecord(slots[idx].ev, stream);
		slots[idx].busy = false;
	}

	// Frees the idle buffers of a device (at the end of the E-step).
	void trim(int dev)
	{
		std::lock_guard<std::mutex> lk(m);
		if (!times.empty())
		{
			double ms = 0.;
			for (auto &e : times)
			{
				float f = 0.f;
				cudaEventSynchronize(e.second);
				if (cudaEventElapsedTime(&f, e.first, e.second) == cudaSuccess) ms += f;
				cudaEventDestroy(e.first); cudaEventDestroy(e.second);
			}
			printf(" GPU coarse pass: group kernel launches (table + group) %zu sets, %ld table parts, %.1f ms elapsed on the streams\n", times.size(), n_parts.load(), ms);
			times.clear();
			n_parts = 0;
		}
		int cur = 0;
		cudaGetDevice(&cur);
		cudaSetDevice(dev);
		size_t freed = 0;
		for (Slot &s : slots)
			if (s.dev == dev && !s.busy && s.p != NULL)
			{
				cudaFree(s.p);
				freed += s.cap;
				s.p = NULL; s.cap = 0;
				if (s.ev != NULL) { cudaEventDestroy(s.ev); s.ev = NULL; }
			}
		cudaGetLastError();
		cudaSetDevice(cur);
		for (auto it = users.begin(); it != users.end(); ) it = (it->first == dev) ? users.erase(it) : std::next(it);
		if ((freed > 0 || n_used + n_texture > 0) && getenv("RELION_GPU_GROUP_PROJ_TABLE_STATS"))
			printf(" GPU coarse pass: reference projection table pool released %.1f MB (peak held %.1f MB; %ld sets with the table, %ld on the texture path)\n",
			       (double) freed / 1048576., (double) peak_held / 1048576., n_used.load(), n_texture.load());
		peak_held = 0;
		n_used = 0;
		n_texture = 0;
	}
};

inline void groupTablePoolTrim(int dev) { GroupTablePool::get().trim(dev); }

#endif
