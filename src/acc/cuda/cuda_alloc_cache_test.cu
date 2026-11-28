/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/acc/cuda/cuda_alloc_cache_test.h"
#include "src/acc/cuda/custom_allocator.cuh"

#include <thread>
#include <vector>

namespace gpucache
{

typedef CudaCustomAllocator::Alloc Block;

static void handBack(Block *b, cudaStream_t s)
{
	b->markReadyEvent(s);
	b->doFreeWhenReady();
}

bool run(Outcome &o)
{
	int ndev = 0;
	if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1)
	{
		cudaGetLastError();
		return false;
	}
	o.gpu = true;
	cudaStream_t s1, s2;
	cudaStreamCreate(&s1);
	cudaStreamCreate(&s2);

	CudaCustomAllocator::setCacheMode(true);
	{
		CudaCustomAllocator pool(0, 512);
		pool.configureLean((size_t) 512 << 20, (size_t) 256 << 20);
		pool.setCacheCapMB(8);

		// same stream, same thread
		Block *a = pool.alloc(100000, s1);
		unsigned char *p = a->getPtr();
		cudaMemsetAsync(p, 0x5a, 100000, s1);
		handBack(a, s1);
		Block *b = pool.alloc(100000, s1);
		o.reusedSameStream = b->getPtr() == p;
		o.hitsSameStream = pool.cachedHits();
		cudaMemsetAsync(b->getPtr(), 0x11, 100000, s1);
		std::vector<unsigned char> h(100000);
		cudaMemcpyAsync(h.data(), b->getPtr(), 100000, cudaMemcpyDeviceToHost, s1);
		cudaStreamSynchronize(s1);
		o.dataOk = h[0] == 0x11 && h[99999] == 0x11;
		handBack(b, s1);

		// another stream: the cached block is not given
		Block *c = pool.alloc(100000, s2);
		o.reusedOtherStream = c->getPtr() == p;
		handBack(c, s2);

		// another thread, same stream: not given either (it may not be the thread that issued the work)
		unsigned char *q = NULL;
		std::thread t([&] { Block *d = pool.alloc(100000, s1); q = d->getPtr(); handBack(d, s1); });
		t.join();
		o.reusedOtherThread = q == p;

		// the cache is capped (1 MB) and gives everything back on a sync
		pool.setCacheCapMB(1);
		pool.syncReadyEvents();
		std::vector<Block*> v;
		for (int i = 0; i < 20; i++) v.push_back(pool.alloc(300000, s1));
		for (Block *x : v) handBack(x, s1);
		o.cap = 1000000; // setCacheCapMB(1)
		o.cachedUnderCap = pool.cachedBytes();
		pool.syncReadyEvents();
		o.cachedAfterSync = pool.cachedBytes();
		o.usedAfterSync = pool.getTotalUsedSpace();
	}
	// the whole pool asked for after many small blocks went to the cache or wait for events
	{
		CudaCustomAllocator pool(0, 512);
		pool.configureLean((size_t) 96 << 20, (size_t) 96 << 20);
		pool.setCacheCapMB(8);
		std::vector<Block*> v;
		for (int i = 0; i < 30; i++) v.push_back(pool.alloc(3000000, s1));
		for (Block *x : v) handBack(x, s1);
		Block *big = pool.alloc((size_t) 90 << 20, s1);
		o.bigAfterSmallOk = big != NULL;
		handBack(big, s1);
	}
	// a pool that holds exactly the request, in a size the cache would round up
	{
		const size_t exact = 3000320;
		CudaCustomAllocator pool(0, 512);
		pool.configureLean(exact, exact);
		Block *x = pool.alloc(exact, s1);
		o.exactFitOk = x != NULL;
		handBack(x, s1);
	}
	// nearly full pool (> 7/8 in use): blocks are not cached any more
	{
		CudaCustomAllocator pool(0, 512);
		pool.configureLean((size_t) 3200000, (size_t) 3200000);
		Block *x = pool.alloc((size_t) 3 << 20, s1);
		handBack(x, s1);
		Block *y = pool.alloc((size_t) 3 << 20, s1);
		o.hitsWhenFull = pool.cachedHits();
		handBack(y, s1);
	}
	// cache off: nothing is cached
	{
		CudaCustomAllocator::setCacheMode(false);
		CudaCustomAllocator pool(0, 512);
		pool.configureLean((size_t) 128 << 20, (size_t) 128 << 20);
		Block *a = pool.alloc(100000, s1);
		handBack(a, s1);
		Block *b = pool.alloc(100000, s1);
		(void) b;
		o.hitsCacheOff = pool.cachedHits();
	}
	// free-list integrity under fragmentation, prev/next/both coalescing, split, exact reuse, and multi-chunk growth
	{
		CudaCustomAllocator::setCacheMode(true);
		CudaCustomAllocator pool(0, 512);
		pool.configureLean((size_t) 64 << 20, (size_t) 16 << 20);
		pool.setCacheCapMB(1); // small cap forces evictions + _free coalescing
		CudaCustomAllocator::allowCache(false); // bypass ThreadCache so every alloc/free hits freeHead
		Block *b1 = pool.alloc(512 * 1024);
		Block *b2 = pool.alloc(512 * 1024);
		Block *b3 = pool.alloc(512 * 1024);
		Block *b4 = pool.alloc(512 * 1024);
		Block *b5 = pool.alloc(512 * 1024);
		pool.free(b1); // head free
		pool.free(b3); // middle free (isolated)
		pool.free(b5); // merges with tail free
		pool.free(b4); // merges with prev (b3) AND next (b5) -> both-neighbor coalesce
		pool.free(b2); // merges with prev (b1) AND next (merged b3..tail) -> single free block
		bool ok = (b1 && b2 && b3 && b4 && b5 && pool.checkFreeListInvariant());
		// exact match vs split + multi-chunk _addChunk
		Block *e1 = pool.alloc((size_t) 16 << 20);
		ok = ok && (e1 != NULL) && pool.checkFreeListInvariant();
		Block *g1 = pool.alloc((size_t) 8 << 20); // triggers _addChunk
		Block *g2 = pool.alloc((size_t) 4 << 20);
		ok = ok && (g1 != NULL && g2 != NULL) && pool.checkFreeListInvariant();
		pool.free(g1);
		pool.free(e1);
		pool.free(g2);
		// cross-chunk boundary must NOT coalesce (b in chunk1 vs chunk0), exact reuse of 4 MB block
		Block *r1 = pool.alloc((size_t) 4 << 20);
		ok = ok && (r1 != NULL) && pool.checkFreeListInvariant();
		pool.free(r1);
		ok = ok && pool.checkFreeListInvariant() && (pool.getTotalUsedSpace() == 0);
		CudaCustomAllocator::allowCache(true);
		o.freeListOk = ok;
	}
	cudaStreamDestroy(s1);
	cudaStreamDestroy(s2);
	return true;
}

}
