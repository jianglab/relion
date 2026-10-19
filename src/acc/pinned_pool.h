#ifndef PINNED_POOL_H_
#define PINNED_POOL_H_

#include <cstddef>
#include <mutex>
#include <vector>

/*
 * A process-wide pool of pinned host buffers. cudaMallocHost and cudaFreeHost are slow (10 to 30 ms
 * each, behind a driver lock that stalls the other threads) and the staging buffers of the GPU copies
 * used to be made again by every new thread and every time a copy was larger than the last. Here a
 * buffer goes back to the pool when its thread finishes or outgrows it, and the next request takes the
 * smallest free buffer that is big enough. Sizes are rounded up to powers of two, so a handful of sizes
 * serve every request. Free buffers beyond max_free_bytes are given back at once.
 * The allocator is a parameter, so the pool can be tested without a GPU.
 * enabled == false turns the pool into plain allocate/free (RELION_GPU_PINNED_POOL=0).
 */
namespace pinnedpool
{

inline size_t roundUp(size_t bytes, size_t min_bytes = (size_t) 1 << 20)
{
	size_t c = min_bytes;
	while (c < bytes) c <<= 1;
	return c;
}

class Pool
{
public:
	typedef void *(*AllocFn)(size_t);
	typedef void (*FreeFn)(void *);

	Pool(AllocFn a, FreeFn f, bool on = true, size_t max_free = (size_t) 2 << 30)
		: alloc_(a), free_(f), enabled_(on), max_free_(max_free) {}

	// A buffer of at least bytes; cap is its real size (give it back to release() with it)
	void *acquire(size_t bytes, size_t &cap)
	{
		if (enabled_)
		{
			std::lock_guard<std::mutex> lock(m_);
			size_t best = free_blocks_.size();
			for (size_t i = 0; i < free_blocks_.size(); i++)
				if (free_blocks_[i].cap >= bytes && (best == free_blocks_.size() || free_blocks_[i].cap < free_blocks_[best].cap))
					best = i;
			if (best < free_blocks_.size())
			{
				void *p = free_blocks_[best].ptr;
				cap = free_blocks_[best].cap;
				free_bytes_ -= cap;
				free_blocks_.erase(free_blocks_.begin() + best);
				reuses_++;
				return p;
			}
			allocations_++;
		}
		cap = enabled_ ? roundUp(bytes) : bytes;
		return alloc_(cap);
	}

	void release(void *p, size_t cap)
	{
		if (p == NULL) return;
		if (enabled_)
		{
			std::lock_guard<std::mutex> lock(m_);
			if (free_bytes_ + cap <= max_free_)
			{
				free_blocks_.push_back({p, cap});
				free_bytes_ += cap;
				return;
			}
		}
		free_(p);
	}

	size_t allocations() const { std::lock_guard<std::mutex> lock(m_); return allocations_; }
	size_t reuses() const { std::lock_guard<std::mutex> lock(m_); return reuses_; }
	size_t freeBlocks() const { std::lock_guard<std::mutex> lock(m_); return free_blocks_.size(); }
	size_t freeBytes() const { std::lock_guard<std::mutex> lock(m_); return free_bytes_; }

private:
	struct Block { void *ptr; size_t cap; };
	AllocFn alloc_;
	FreeFn free_;
	bool enabled_;
	size_t max_free_;
	mutable std::mutex m_;
	std::vector<Block> free_blocks_;
	size_t free_bytes_ = 0, allocations_ = 0, reuses_ = 0;
};

}

#endif /* PINNED_POOL_H_ */
