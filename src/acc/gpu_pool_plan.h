#ifndef GPU_POOL_PLAN_H_
#define GPU_POOL_PLAN_H_

// Sizing rules of the "lean" GPU memory pool (RELION_GPU_LEAN_MEMORY, see custom_allocator.cuh).
//
// By default the expectation step takes ALL free GPU memory in one block and hands it out in pieces.
// The lean pool holds a first chunk sized from what the job really used and adds chunks only when an
// allocation does not fit, up to the same limit (the "cap": what the old pool would have taken). Pure
// C++ with no GPU calls, so the rules can be tested without a GPU.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>

namespace gpupool
{

// Smallest chunk worth a driver call. cudaMalloc has a granularity of a few MB and each call
// stalls the device, so smaller chunks only add calls.
constexpr size_t kMinChunk = (size_t) 64 << 20;

// Margin on top of a measured peak for the next iteration's first chunk. The peak moves from one
// iteration to the next (finer sampling, more image pixels), and an allocation that finds no room
// costs a driver call that stalls the GPU, whereas a too large chunk only holds memory for one more
// iteration. 15% covers the fragmentation of a first-fit pool; when two successive peaks differed by
// a fraction r, the next one may differ by about as much again, so up to 10% more is added, 2 r.
// With one measurement only (a cached value from an earlier run) 20%.
inline double marginFor(size_t previous_peak, size_t last_peak)
{
	if (last_peak == 0)
		return 1.2;
	if (previous_peak == 0)
		return 1.2;
	const double r = std::fabs((double) last_peak - (double) previous_peak) / (double) std::max(previous_peak, last_peak);
	return 1.15 + std::min(0.10, 2. * r);
}

struct Plan
{
	size_t cap = 0;          // never hold more than this (what the old pool would have taken)
	size_t first_chunk = 0;  // bytes to take at the start of the iteration
	double margin = 0.;      // 0 when no measurement was used
	const char *reason = ""; // where first_chunk came from
};

// peak_last: peak bytes in use during the previous iteration of this process (0 if none),
// peak_previous: the one before it, cached_peak: from the tuning cache (0 if none)
inline Plan plan(size_t cap, size_t peak_last, size_t peak_previous, size_t cached_peak)
{
	Plan p;
	p.cap = cap;
	size_t known = 0;
	if (peak_last > 0)
	{
		known = peak_last;
		p.reason = "peak of the previous iteration";
	}
	else if (cached_peak > 0)
	{
		known = cached_peak;
		p.reason = "peak remembered from an earlier run";
	}
	if (known > 0)
	{
		p.margin = marginFor(peak_last > 0 ? peak_previous : 0, known);
		p.first_chunk = (size_t) std::ceil((double) known * p.margin);
	}
	else
	{
		// Nothing measured: a small chunk that grows as needed. Growing costs one driver call per
		// chunk in this first iteration only, and it gets the peak measured.
		p.first_chunk = cap / 64;
		p.reason = "no measurement yet";
	}
	p.first_chunk = std::min(cap, std::max(kMinChunk, p.first_chunk));
	return p;
}

// Size of the chunk to add when `request` bytes found no room, `held` bytes are held, and `room`
// more bytes may be taken (cap - held). 0: cannot grow (the request exceeds the room).
// Geometric steps (half of what is held) make the number of driver calls logarithmic in the final
// size, and the overshoot is at most half of it, corrected by the next iteration's plan.
inline size_t nextChunk(size_t held, size_t request, size_t room)
{
	if (request > room)
		return 0;
	size_t want = std::max(request, std::max(held / 2, kMinChunk));
	return std::min(want, room);
}

// Whether a chunk list kept from the last iteration (all of it free) can be used as it is for a first
// chunk of `target` bytes: one chunk, large enough, and not holding much more than needed.
inline bool keepChunks(size_t nr_chunks, size_t held, size_t target)
{
	return nr_chunks == 1 && held >= target && (double) held <= 1.3 * (double) target;
}

// Description of the problem a peak was measured for. The peak depends on the box, the references,
// the threads sharing the pool and the sizes of the iteration (current/coarse image size and the number of
// orientations); a different job with the same description needs about the same.
struct Problem
{
	int ori_size = 0, nr_classes = 0, ref_dim = 0;
	size_t model_bytes = 0;     // padded reference + back-projection arrays on the device
	int threads_on_device = 0;  // host threads using this pool
	int ranks_on_device = 1;    // MPI processes sharing the device
	int current_size = 0, coarse_size = 0;
	long nr_orientations = 0;   // orientations x psi x oversampling of one pass
	int nr_pool = 0;
};

inline std::string problemKey(const Problem &p)
{
	char b[256];
	snprintf(b, sizeof(b), "box=%d;K=%d;dim=%d;model=%zu;thr=%d;ranks=%d;cur=%d;coarse=%d;orient=%ld;pool=%d",
	         p.ori_size, p.nr_classes, p.ref_dim, p.model_bytes, p.threads_on_device, p.ranks_on_device,
	         p.current_size, p.coarse_size, p.nr_orientations, p.nr_pool);
	return b;
}

} // namespace gpupool

#endif
