#ifndef ACC_HOST_BLOCK_CACHE_H_
#define ACC_HOST_BLOCK_CACHE_H_

#include <cstdlib>
#include <cstring>
#include <vector>
#include "src/gpu_switch.h"

// Large host arrays that are allocated and freed once per particle (the fine-pass index
// lists) are kept per thread and handed out again, so the memory is not returned to the
// system and faulted in again for every particle. Opt-in: RELION_GPU_HOST_REUSE=1. On the
// test case it cut the pair-list stage by a further 40% but gave no measurable wall time.
namespace hostcache
{
inline bool enabled()
{
	const char *e = getenv("RELION_GPU_HOST_REUSE");
	static const bool on = e != NULL && *e != 0 && !gpuswitch::isOffValue(e);
	return on;
}

struct Block { void *ptr; size_t capacity; };

struct Cache
{
	std::vector<Block> free_blocks;
	~Cache() { for (const Block &b : free_blocks) free(b.ptr); }
};

inline Cache &local() { thread_local Cache c; return c; }

// The smallest cached block that is large enough, or a new one. Pages of a block that were
// never written cost nothing, so a block much larger than needed does no harm.
inline Block acquire(size_t bytes, size_t alignment)
{
	if (bytes == 0) bytes = 1;
	Cache &c = local();
	int best = -1;
	for (size_t i = 0; i < c.free_blocks.size(); i++)
		if (c.free_blocks[i].capacity >= bytes && (best < 0 || c.free_blocks[i].capacity < c.free_blocks[best].capacity))
			best = (int) i;
	if (best >= 0)
	{
		Block b = c.free_blocks[best];
		c.free_blocks.erase(c.free_blocks.begin() + best);
		return b;
	}
	Block b;
	b.capacity = bytes;
	if (posix_memalign(&b.ptr, alignment, bytes))
		b.ptr = NULL;
	return b;
}

inline void release(Block b)
{
	if (b.ptr == NULL) return;
	Cache &c = local();
	c.free_blocks.push_back(b);
	// no more than a few blocks are kept: drop the smallest
	while (c.free_blocks.size() > 8)
	{
		size_t smallest = 0;
		for (size_t i = 1; i < c.free_blocks.size(); i++)
			if (c.free_blocks[i].capacity < c.free_blocks[smallest].capacity)
				smallest = i;
		free(c.free_blocks[smallest].ptr);
		c.free_blocks.erase(c.free_blocks.begin() + smallest);
	}
}
}

#endif
