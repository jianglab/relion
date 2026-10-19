/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef CUDA_ALLOC_CACHE_TEST_H_
#define CUDA_ALLOC_CACHE_TEST_H_

// Host entry points for the unit tests of the per-thread block cache of the GPU pool (RELION_GPU_ALLOC_CACHE)
namespace gpucache
{

struct Outcome
{
	bool gpu = false;
	// same stream, same thread: the block comes back at once
	bool reusedSameStream = false;
	long hitsSameStream = 0;
	// another stream, or another thread, must not get it
	bool reusedOtherStream = true;
	bool reusedOtherThread = true;
	// with the cache off nothing is cached
	long hitsCacheOff = -1;
	// a nearly full pool does not cache
	long hitsWhenFull = -1;
	// the cache stays under its cap, and gives everything back on syncReadyEvents()
	size_t cachedUnderCap = 0, cap = 0, cachedAfterSync = 1, usedAfterSync = 1;
	// a request that needs the whole pool succeeds although blocks sit in caches or wait for events
	bool bigAfterSmallOk = false;
	// a request that fits exactly succeeds although the cache rounds sizes up
	bool exactFitOk = false;
	// contents survive reuse (the block is the same memory, written on the same stream)
	bool dataOk = false;
};

// False if no GPU is usable.
bool run(Outcome &out);

}

#endif
