#ifndef CUDA_CUSTOM_ALLOCATOR_CUH_
#define CUDA_CUSTOM_ALLOCATOR_CUH_
// This is where custom allocator should be. Commented out for now, to avoid double declaration.

#ifdef _CUDA_ENABLED
#include "src/acc/cuda/cuda_settings.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#endif

#include <signal.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <deque>
#include <map>
#include <thread>
#include <cstring>
#include <mutex>
#include <atomic>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "src/macros.h"
#include "src/gpu_switch.h"
#include "src/acc/stage_timer.h"
#include "src/error.h"
#include "src/parallel.h"
#include "src/acc/gpu_pool_plan.h"

#ifdef CUSTOM_ALLOCATOR_MEMGUARD
#include "src/acc/cuda/shortcuts.cuh"
#include <execinfo.h>
#include <cxxabi.h>
#endif

#ifdef DUMP_CUSTOM_ALLOCATOR_ACTIVITY
#define CUSTOM_ALLOCATOR_REGION_NAME( name ) (fprintf(stderr, "\n%s", name))
#else
#define CUSTOM_ALLOCATOR_REGION_NAME( name ) //Do nothing
#endif


class CudaCustomAllocator
{

	typedef unsigned char BYTE;

	const static unsigned GUARD_SIZE = 4;
	const static BYTE GUARD_VALUE = 145;
	const static int ALLOC_RETRY = 500;

public:

	class Alloc
	{
		friend class CudaCustomAllocator;

	public:
		// Only ever polled, never timed: an event without timing is cheaper to record and query
		static cudaEvent_t newEvent()
		{
			cudaEvent_t e;
			DEBUG_HANDLE_ERROR(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
			return e;
		}
	private:

	private:
		Alloc *prev, *next;
		BYTE *ptr;
		size_t size;
		bool free;
		cudaEvent_t readyEvent; //Event record used for auto free
		bool freeWhenReady;
		int chunk;                  // the chunk of device memory the block lies in (lean pool); blocks only merge within a chunk
		CudaCustomAllocator *owner; // set by alloc(); its pool supplies the ready events
		bool readyMarked;           // markReadyEvent() was called: free after the work
		cudaStream_t readyStream;   // issued so far on this stream by this thread
		std::thread::id readyThread;
		cudaStream_t allocStream;   // RELION_GPU_ALLOC_CACHE: the stream the block was asked for on
		bool allocStreamValid;


#ifdef CUSTOM_ALLOCATOR_MEMGUARD
		BYTE *guardPtr;
		void *backtrace[20];
		size_t backtraceSize;
#endif

		Alloc():
			prev(NULL), next(NULL),
			ptr(NULL),
			size(0),
			free(0),
			readyEvent(0),
			freeWhenReady(false),
			chunk(0),
			owner(NULL),
			readyMarked(false),
			readyStream(0),
			allocStream(0),
			allocStreamValid(false)
		{}

		~Alloc()
		{
			prev = NULL;
			next = NULL;
			ptr = NULL;

			if (readyEvent != 0)
				DEBUG_HANDLE_ERROR(cudaEventDestroy(readyEvent));
		}

	public:
		inline
		BYTE *getPtr() { return ptr; }

		inline
		size_t getSize() { return size; }

		inline
		bool isFree() { return free; }

		inline
		bool hasReadyMark() { return readyMarked || readyEvent != 0; }

		// The block may be reused once the work issued so far on `stream` by this thread
		// has completed. With an owner, no event is recorded here: the owner records one
		// event for a batch of such blocks later (see CudaCustomAllocator::pending), which
		// covers at least the same work - one event per freed buffer was a few hundred
		// driver calls per particle, each taking the CUDA driver's per-process lock.
		inline
		void markReadyEvent(cudaStream_t stream = 0)
		{
			readyMarked = true;
			readyStream = stream;
			readyThread = std::this_thread::get_id();
			if (!owner)
			{
				readyEvent = newEvent();
				DEBUG_HANDLE_ERROR(cudaEventRecord(readyEvent, stream));
			}
		}

		inline
		void doFreeWhenReady()
		{
			freeWhenReady = true;
			if (owner && readyMarked) // never marked: never freed, as before
			{
				if (!owner->tryCache(this))
					owner->enqueuePending(this);
			}
		}
	};

private:

	Alloc *first;
	size_t totalSize;
	size_t alignmentSize;

	bool cache;

	// Lean pool (see configureLean()): instead of one chunk of all the memory there may be several,
	// added when an allocation finds no room, up to capBytes held in all. Without it there is exactly
	// one chunk (or none) and nothing below changes the old behaviour.
	struct Chunk { BYTE *base; size_t size; int id; };
	std::vector<Chunk> chunks;
	int nextChunkId;
	bool lean;
	size_t capBytes;
	size_t heldBytes;     // sum of the chunks
	size_t usedBytes;     // sum of the blocks handed out and not yet freed (blocks waiting for their work count)
	size_t peakUsed;      // high-water marks since resetPeaks()
	size_t peakHeld;
	size_t chunksAdded;   // chunks added by growth since resetPeaks()

	omp_lock_t mutex;

	// Recycled ready events (all made on this allocator's device)
	std::vector<cudaEvent_t> eventPool;
	omp_lock_t eventMutex;

	// Blocks handed back with doFreeWhenReady() wait for the work issued before on their
	// stream, by the thread that freed them. They are kept per (thread, stream):
	// - open: no event yet. The thread itself closes its batches in its next alloc(),
	//   recording one event per stream for all blocks freed on it since; recording later
	//   than the free only covers more work. Only that thread may record:
	//   cudaStreamPerThread means a different stream in every thread. (Keeping batches
	//   open for longer, to save more events, was slower: the free-space search grew.)
	// - closed: queued behind their event. Events on one stream complete in the order
	//   they were recorded, so a queue is only queried from the front.
	// Finding freed blocks used to take a walk over every block, querying each one's
	// event, on every alloc() and under `mutex`; with several threads most of their time
	// went waiting for that lock. When memory runs short, _syncReadyEvents() waits for the
	// whole device instead and takes every block, open or closed: all work that used them
	// was issued before they were handed back.
	// Lock order: mutex, scanMutex, pendingMutex, openMutex (eventMutex innermost).
	typedef std::pair<std::thread::id, cudaStream_t> PendingKey;
	struct OpenBatch { std::vector<Alloc*> blocks; };
	struct ClosedBatch { cudaEvent_t event; std::vector<Alloc*> blocks; };
	std::map<PendingKey, OpenBatch> openBatches;
	std::map<PendingKey, std::deque<ClosedBatch> > pending;
	omp_lock_t pendingMutex; // guards pending
	omp_lock_t openMutex;    // guards openBatches (held only to add or take blocks)
	omp_lock_t scanMutex;    // one thread at a time polls the pending events

	// Close this thread's open batches. The events are recorded outside the locks:
	// recording is a driver call, and every thread freeing memory needs openMutex.
	void _closeOwnBatches()
	{
		const std::thread::id me = std::this_thread::get_id();
		std::vector<std::pair<PendingKey, std::vector<Alloc*> > > mine;
		{
			Lock ol(&openMutex);
			for (auto it = openBatches.lower_bound(PendingKey(me, (cudaStream_t) 0));
			     it != openBatches.end() && it->first.first == me; )
			{
				mine.push_back(std::make_pair(it->first, std::vector<Alloc*>()));
				mine.back().second.swap(it->second.blocks);
				it = openBatches.erase(it);
			}
		}
		for (auto &kb : mine)
		{
			ClosedBatch c;
			c.event = takeEvent();
			DEBUG_HANDLE_ERROR(cudaEventRecord(c.event, kb.first.second));
			c.blocks.swap(kb.second);
			Lock pl(&pendingMutex);
			pending[kb.first].push_back(std::move(c));
		}
	}

	// Take the blocks whose batches' events have completed. Returns at once if another
	// thread is already scanning: it will find the same blocks. The events are queried
	// outside pendingMutex (only the scanning thread removes batches, so the fronts
	// it saw stay the fronts).
	void _collectReady(std::vector<Alloc*> &ready)
	{
		if (!omp_test_lock(&scanMutex))
			return;
		while (true)
		{
			std::vector<std::pair<PendingKey, cudaEvent_t> > fronts;
			{
				Lock pl(&pendingMutex);
				for (auto &kv : pending)
					if (!kv.second.empty())
						fronts.push_back(std::make_pair(kv.first, kv.second.front().event));
			}
			std::vector<PendingKey> done;
			for (auto &f : fronts)
			{
				cudaError_t e = cudaEventQuery(f.second);
				if (e == cudaSuccess)
					done.push_back(f.first);
				else if (e != cudaErrorNotReady)
					HandleError( e, __FILE__, __LINE__ );
			}
			if (done.empty())
				break;
			Lock pl(&pendingMutex);
			for (const PendingKey &k : done)
			{
				auto it = pending.find(k);
				ClosedBatch &c = it->second.front();
				ready.insert(ready.end(), c.blocks.begin(), c.blocks.end());
				returnEvent(c.event);
				it->second.pop_front();
				if (it->second.empty())
					pending.erase(it);
			}
		}
		omp_unset_lock(&scanMutex);
	}

	// Take every handed-back block, open or closed, then wait for the device: their work
	// was all issued before they were handed back. (Taken first: a block handed back
	// after the wait could still be in use, and must not be freed here.)
	void _collectAll(std::vector<Alloc*> &ready)
	{
		Lock sl(&scanMutex);
		{
			Lock pl(&pendingMutex);
			for (auto &kv : pending)
				for (ClosedBatch &c : kv.second)
				{
					ready.insert(ready.end(), c.blocks.begin(), c.blocks.end());
					returnEvent(c.event);
				}
			pending.clear();
			Lock ol(&openMutex);
			for (auto &kv : openBatches)
				ready.insert(ready.end(), kv.second.blocks.begin(), kv.second.blocks.end());
			openBatches.clear();
		}
		_takeCachedBlocks(ready);
		DEBUG_HANDLE_ERROR(cudaDeviceSynchronize());
	}

	//Look for the first suited space
	Alloc *_getFirstSuitedFree(size_t size)
	{
		Alloc *a = first;
		//If not the last and too small or not free go to next allocation region
		while (a != NULL && ( a->size < size || ! a->free ) )
			a = a->next;

		return a;
	}

	// Wait for all handed-back blocks and free them (caller holds mutex)
	bool _syncReadyEvents()
	{
		std::vector<Alloc*> ready;
		_collectAll(ready);
		for (Alloc *a : ready)
			_free(a);
		return !ready.empty();
	}

	// Free the pending blocks whose events have completed (caller holds mutex)
	bool _freeReadyAllocs()
	{
		std::vector<Alloc*> ready;
		_collectReady(ready);
		for (Alloc *a : ready)
			_free(a);
		return !ready.empty();
	}

	size_t _getTotalFreeSpace()
	{
		if (cache)
		{
			size_t total = 0;
			Alloc *a = first;

			while (a != NULL)
			{
				if (a->free)
					total += a->size;
				a = a->next;
			}

			return total;
		}
		else
		{
			size_t free, total;
			DEBUG_HANDLE_ERROR(cudaMemGetInfo( &free, &total ));
			return free;
		}
	}

	size_t _getTotalUsedSpace()
	{
		size_t total = 0;
		Alloc *a = first;

		while (a != NULL)
		{
			if (!a->free)
				total += a->size;
			a = a->next;
		}

		return total;
	}

	size_t _getNumberOfAllocs()
	{

		size_t total = 0;
		Alloc *a = first;

		while (a != NULL)
		{
			if (!a->free)
				total ++;
			a = a->next;
		}

		return total;
	}

	size_t _getLargestContinuousFreeSpace()
	{
		if (cache)
		{
			size_t largest = 0;
			Alloc *a = first;

			while (a != NULL)
			{
				if (a->free && a->size > largest)
					largest = a->size;
				a = a->next;
			}

			return largest;
		}
		else
			return _getTotalFreeSpace();
	}

	void _printState()
	{
		size_t total = 0;
		Alloc *a = first;

		while (a != NULL)
		{
			total += a->size;
			if (a->free)
				printf("[%luB] ", (unsigned long) a->size);
			else if (a->freeWhenReady)
				printf("<%luB> ", (unsigned long) a->size);
			else
				printf("(%luB) ", (unsigned long) a->size);

			a = a->next;
		}

		printf("= %luB\n", (unsigned long) total);
		fflush(stdout);
	}

	void _free(Alloc* a)
	{
//		printf("free: %u ", a->size);
//		_printState();


#ifdef CUSTOM_ALLOCATOR_MEMGUARD
		size_t guardCount = a->size - (a->guardPtr - a->ptr);
		BYTE *guards = new BYTE[guardCount];
		cudaStream_t stream = 0;
		CudaShortcuts::cpyDeviceToHost<BYTE>( a->guardPtr, guards, guardCount, stream);
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
		for (int i = 0; i < guardCount; i ++)
			if (guards[i] != GUARD_VALUE)
			{
				fprintf (stderr, "ERROR: CORRUPTED BYTE GUARDS DETECTED\n");

				char ** messages = backtrace_symbols(a->backtrace, a->backtraceSize);

				// skip first stack frame (points here)
				for (int i = 1; i < a->backtraceSize && messages != NULL; ++i)
				{
					char *mangled_name = 0, *offset_begin = 0, *offset_end = 0;

					// find parantheses and +address offset surrounding mangled name
					for (char *p = messages[i]; *p; ++p)
					{
						if (*p == '(')
						{
							mangled_name = p;
						}
						else if (*p == '+')
						{
							offset_begin = p;
						}
						else if (*p == ')')
						{
							offset_end = p;
							break;
						}
					}

					// if the line could be processed, attempt to demangle the symbol
					if (mangled_name && offset_begin && offset_end &&
						mangled_name < offset_begin)
					{
						*mangled_name++ = '\0';
						*offset_begin++ = '\0';
						*offset_end++ = '\0';

						int status;
						char * real_name = abi::__cxa_demangle(mangled_name, 0, 0, &status);

						// if demangling is successful, output the demangled function name
						if (status == 0)
						{
							std::cerr << "[bt]: (" << i << ") " << messages[i] << " : "
									  << real_name << "+" << offset_begin << offset_end
									  << std::endl;

						}
						// otherwise, output the mangled function name
						else
						{
							std::cerr << "[bt]: (" << i << ") " << messages[i] << " : "
									  << mangled_name << "+" << offset_begin << offset_end
									  << std::endl;
						}
//						free(real_name);
					}
					// otherwise, print the whole line
					else
					{
						std::cerr << "[bt]: (" << i << ") " << messages[i] << std::endl;
					}
				}
				std::cerr << std::endl;

//				free(messages);

				exit(EXIT_FAILURE);
			}
		delete[] guards;
#endif

		usedBytes -= std::min(usedBytes, a->size);
		a->free = true;

		/* A free region must not carry a pending completion event.
		 *
		 * An exact-size match hands back the Alloc object itself rather than
		 * splitting a larger one, so these fields survive into the next
		 * allocation. Left set, _freeReadyAllocs() sees a live allocation whose
		 * (stale, long-completed) event queries ready and frees it while it is
		 * still in use, handing the same device memory to something else.
		 */
		a->freeWhenReady = false;
		a->readyMarked = false;
		if (a->readyEvent != 0)
		{
			returnEvent(a->readyEvent);
			a->readyEvent = 0;
		}

		if (cache)
		{
			//Previous neighbor is free, concatenate
			if ( a->prev != NULL && a->prev->free && a->prev->chunk == a->chunk)
			{
				//Resize and set pointer
				a->size += a->prev->size;
				a->ptr = a->prev->ptr;

				//Fetch secondary neighbor
				Alloc *ppL = a->prev->prev;

				//Remove primary neighbor
				if (ppL == NULL) //If the previous is first in chain
					first = a;
				else
					ppL->next = a;

				delete a->prev;

				//Attach secondary neighbor
				a->prev = ppL;
			}

			//Next neighbor is free, concatenate
			if ( a->next != NULL && a->next->free && a->next->chunk == a->chunk)
			{
				//Resize and set pointer
				a->size += a->next->size;

				//Fetch secondary neighbor
				Alloc *nnL = a->next->next;

				//Remove primary neighbor
				if (nnL != NULL)
					nnL->prev = a;
				delete a->next;

				//Attach secondary neighbor
				a->next = nnL;
			}
		}
		else
		{
			DEBUG_HANDLE_ERROR(cudaFree( a->ptr ));
			a->ptr = NULL;

			if ( a->prev != NULL)
				a->prev->next = a->next;
			else
				first = a->next; //This is the first link

			if ( a->next != NULL)
				a->next->prev = a->prev;

			delete a;
		}
	};

	void _setup()
	{
		first = new Alloc();
		first->prev = NULL;
		first->next = NULL;
		first->size = totalSize;
		first->free = true;

		if (totalSize > 0)
		{
			// Back off instead of failing when the memory measured free a moment ago is gone: another
			// process on the same GPU (several MPI ranks per GPU, under MPS) may have allocated since.
			// Retry with 90% of what is free now, down to a quarter of the size asked for.
			const size_t asked = totalSize;
			cudaError_t err = cudaMalloc( (void**) &(first->ptr), totalSize);
			while (err == cudaErrorMemoryAllocation && totalSize > asked / 4)
			{
				cudaGetLastError();
				size_t free_now = 0, total_now = 0;
				cudaMemGetInfo(&free_now, &total_now);
				const size_t next = std::min((size_t)(0.9 * (double)free_now), (size_t)(0.9 * (double)totalSize));
				if (next < asked / 4) break;
				totalSize = next;
				first->size = totalSize;
				err = cudaMalloc( (void**) &(first->ptr), totalSize);
			}
			if (err == cudaSuccess && totalSize < asked)
				printf(" GPU memory for the expectation step reduced from %zu to %zu MB: the GPU had less free memory than measured (other processes on it).\n",
				       asked / (1000*1000), totalSize / (1000*1000));
			HANDLE_ERROR(err);
			cache = true;
			first->chunk = nextChunkId++;
			chunks.push_back(Chunk{first->ptr, totalSize, first->chunk});
			heldBytes = totalSize;
			peakHeld = std::max(peakHeld, heldBytes);
		}
		else
			cache = false;
	}

	// An empty list: one free block of no size, as a pool without memory has
	void _makeEmptyList()
	{
		first = new Alloc();
		first->prev = NULL;
		first->next = NULL;
		first->size = 0;
		first->free = true;
	}

	// Take a new chunk of `size` bytes and put it at the end of the list as one free block. False if
	// the device has no such memory (the caller then waits for blocks to come free, as before).
	bool _addChunk(size_t size)
	{
		BYTE *p = NULL;
		cudaError_t err = cudaMalloc( (void**) &p, size);
		if (err == cudaErrorMemoryAllocation)
		{
			cudaGetLastError();
			return false;
		}
		HANDLE_ERROR(err);
		Alloc *a = new Alloc();
		a->ptr = p;
		a->size = size;
		a->free = true;
		a->chunk = nextChunkId++;
		chunks.push_back(Chunk{p, size, a->chunk});
		heldBytes += size;
		peakHeld = std::max(peakHeld, heldBytes);
		if (first == NULL)
			first = a;
		else if (first->ptr == NULL && first->size == 0 && first->next == NULL) // the empty list
		{
			delete first;
			first = a;
		}
		else
		{
			Alloc *t = first;
			while (t->next != NULL)
				t = t->next;
			t->next = a;
			a->prev = t;
		}
		return true;
	}

	// Lean pool: no room for `size` bytes, take a new chunk instead of waiting for the GPU
	bool _growFor(size_t size)
	{
		if (!lean || heldBytes >= capBytes)
			return false;
		const size_t room = capBytes - heldBytes;
		size_t want = gpupool::nextChunk(heldBytes, size, room);
		if (want == 0)
			return false;
		bool ok = _addChunk(want);
		if (!ok && want > size) // the device has less than hoped: the request alone
			ok = _addChunk(size);
		if (ok)
			chunksAdded++;
		return ok;
	}

	void _clear()
	{
		if (chunks.empty() && first->ptr != NULL)
			DEBUG_HANDLE_ERROR(cudaFree( first->ptr ));
		for (const Chunk &c : chunks)
			DEBUG_HANDLE_ERROR(cudaFree( c.base ));
		chunks.clear();
		heldBytes = 0;
		usedBytes = 0;

		first->ptr = NULL;

		Alloc *a = first, *nL;

		while (a != NULL)
		{
			nL = a->next;
			delete a;
			a = nL;
		}
	}


	// ---- RELION_GPU_ALLOC_CACHE (opt-in) ---------------------------------------------------------
	// A block handed back by a thread, for work on one stream, may be given to the same thread again for
	// work on the same stream at once: the stream runs its work in order, so no event is needed. This
	// leaves the shared list, its lock and the event calls out of most allocations (about 60 per particle,
	// a seventh of the host time of the E-step). The blocks of a thread's cache stay "in use" for the
	// pool; they are given back through the usual path when the cache is over its cap, and all of them
	// when the pool has to wait for the device (_collectAll), is resized or is destroyed.
	struct ThreadCache
	{
		std::mutex m;
		std::map<std::pair<cudaStream_t, size_t>, std::vector<Alloc*> > blocks;
		size_t bytes;
		ThreadCache(): bytes(0) {}
	};
	std::mutex cacheRegMutex;
	std::vector<ThreadCache*> caches;
	std::atomic<unsigned long> cacheStamp;       // changes when the caches are dropped (resize, destruction)
	std::atomic<long> cacheHits, cacheMisses;
	std::atomic<int> cacheCapMB;                 // per thread; halved when the pool ran out of memory
	// The pool is nearly full (see the end of alloc()): the cache then
	// stops adding blocks and asks the pool for exact sizes, so that it never makes a run fail that fits without it
	std::atomic<bool> cacheTight;
	static std::atomic<unsigned long> &stampSource() { static std::atomic<unsigned long> s(1); return s; }

	// RELION_GPU_ALLOC_CACHE_MB: most MB cached per thread (expert override; default 8)
	static int cacheCapFromEnv()
	{
		const char *e = getenv("RELION_GPU_ALLOC_CACHE_MB");
		return (e && *e && atoi(e) > 0) ? atoi(e) : 8;
	}
	static std::atomic<int> &cacheModeVar()
	{
		static std::atomic<int> m(gpuswitch::on("RELION_GPU_ALLOC_CACHE") ? 1 : 0); // on unless =0
		return m;
	}
	static int cacheMode() { return cacheModeVar().load(); }
	// Set by the code that issues all its work of one thread on that thread's own stream
	static std::atomic<bool> &cacheAllowed() { static std::atomic<bool> a(true); return a; }

	ThreadCache *myCache()
	{
		// one entry per allocator (a thread may use several); the stamp tells a cache that was dropped
		struct Tl { unsigned long stamp; ThreadCache *tc; };
		static thread_local std::map<CudaCustomAllocator*, Tl> tl;
		Tl &e = tl[this];
		if (e.tc != NULL && e.stamp == cacheStamp.load())
			return e.tc;
		ThreadCache *tc = new ThreadCache();
		{
			std::lock_guard<std::mutex> g(cacheRegMutex);
			caches.push_back(tc);
		}
		e.stamp = cacheStamp.load(); e.tc = tc;
		return tc;
	}
	size_t cacheCapBytes()
	{
		size_t c = (size_t) cacheCapMB.load() * 1000 * 1000;
		return std::min(c, std::max((size_t) 4000000, heldBytes / 64));
	}
	// Size classes: aligned, and above 4 kB in steps of 1/32 of the power of two below, so that buffers whose
	// sizes vary a little from particle to particle find each other
	// Only blocks up to this size are cached or rounded up: the big ones are few per particle, and cached big blocks
	// fragment a nearly full pool (a cap run failed 6% earlier with 64 MB blocks cached)
	static constexpr size_t kCacheMaxClassBytes = (size_t) 4 << 20;
	size_t cacheClass(size_t size)
	{
		size = alignmentSize * ((size + alignmentSize - 1) / alignmentSize);
		if (size >= 4096 && size <= kCacheMaxClassBytes) // larger blocks are never cached: exact size, no waste
		{
			size_t step = 1;
			while ((step << 6) <= size) step <<= 1; // step = 2^k with 2^(k+5) <= size < 2^(k+6)
			size = ((size + step - 1) / step) * step;
		}
		return size;
	}
	// Called without `mutex`. The caches are dropped (their blocks are not touched) before the list is deleted.
	void _dropCaches()
	{
		std::lock_guard<std::mutex> g(cacheRegMutex);
		for (ThreadCache *tc : caches) delete tc;
		caches.clear();
		cacheStamp = stampSource().fetch_add(1) + 1;
	}
	// Caller holds `mutex`; the device is waited for by the caller before these blocks are freed
	void _takeCachedBlocks(std::vector<Alloc*> &out)
	{
		std::lock_guard<std::mutex> g(cacheRegMutex);
		for (ThreadCache *tc : caches)
		{
			std::lock_guard<std::mutex> g2(tc->m);
			for (auto &kv : tc->blocks)
				out.insert(out.end(), kv.second.begin(), kv.second.end());
			tc->blocks.clear();
			tc->bytes = 0;
		}
	}

public:
	// Hand a freed block to this thread's cache. False: it goes the usual way.
	bool tryCache(Alloc *a)
	{
		if (!cacheMode() || !cacheAllowed().load() || cacheTight.load() || !a->allocStreamValid || a->readyStream != a->allocStream ||
		    a->readyThread != std::this_thread::get_id())
			return false;
		ThreadCache *tc = myCache();
		std::vector<Alloc*> evict;
		{
			std::lock_guard<std::mutex> g(tc->m);
			const size_t cap = cacheCapBytes();
			if (a->size > cap || a->size > kCacheMaxClassBytes)
				return false;
			tc->blocks[std::make_pair(a->allocStream, a->size)].push_back(a);
			tc->bytes += a->size;
			// over the cap: the blocks of the other sizes first, oldest key first
			for (auto it = tc->blocks.begin(); tc->bytes > cap && it != tc->blocks.end(); )
			{
				if (it->first.second == a->size && it->first.first == a->allocStream) { ++it; continue; }
				for (Alloc *b : it->second) { tc->bytes -= b->size; evict.push_back(b); }
				it = tc->blocks.erase(it);
			}
			// still over: the oldest blocks of the key just freed (the cap holds for one size class too)
			auto own = tc->blocks.find(std::make_pair(a->allocStream, a->size));
			while (tc->bytes > cap && own != tc->blocks.end() && own->second.size() > 1)
			{
				evict.push_back(own->second.front());
				tc->bytes -= own->second.front()->size;
				own->second.erase(own->second.begin());
			}
		}
		for (Alloc *b : evict)
			enqueuePending(b);
		return true;
	}
	long cachedHits() { return cacheHits.load(); }
	// bytes held by the thread caches now (they count as in use for the pool)
	size_t cachedBytes()
	{
		std::lock_guard<std::mutex> g(cacheRegMutex);
		size_t b = 0;
		for (ThreadCache *tc : caches) { std::lock_guard<std::mutex> g2(tc->m); b += tc->bytes; }
		return b;
	}
	static void setCacheMode(bool on) { cacheModeVar().store(on ? 1 : 0); }
	void setCacheCapMB(int mb) { cacheCapMB = mb; }
	long cachedMisses() { return cacheMisses.load(); }
	static void allowCache(bool on) { cacheAllowed().store(on); }

private:
public:

	CudaCustomAllocator(size_t size, size_t alignmentSize):
		totalSize(size), alignmentSize(alignmentSize), first(0), cache(true),
		nextChunkId(0), lean(false), capBytes(size), heldBytes(0), usedBytes(0), peakUsed(0), peakHeld(0), chunksAdded(0),
		cacheStamp(stampSource().fetch_add(1) + 1), cacheHits(0), cacheMisses(0), cacheCapMB(cacheCapFromEnv()), cacheTight(false)
	{
		_setup();

		omp_init_lock(&mutex);
		omp_init_lock(&eventMutex);
		omp_init_lock(&pendingMutex);
		omp_init_lock(&openMutex);
		omp_init_lock(&scanMutex);
	}

	void enqueuePending(Alloc *a)
	{
		Lock ol(&openMutex);
		openBatches[PendingKey(a->readyThread, a->readyStream)].blocks.push_back(a);
	}

	cudaEvent_t takeEvent()
	{
		{
			Lock el(&eventMutex);
			if (!eventPool.empty())
			{
				cudaEvent_t e = eventPool.back();
				eventPool.pop_back();
				return e;
			}
		}
		return Alloc::newEvent();
	}

	void returnEvent(cudaEvent_t e)
	{
		Lock el(&eventMutex);
		eventPool.push_back(e);
	}

	void resize(size_t size)
	{
		Lock ml(&mutex);
		_dropCaches();
		{
			Lock pl(&pendingMutex);
			pending.clear();
			Lock ol(&openMutex);
			openBatches.clear();
		}
		_clear();
		totalSize = size;
		_setup();
	}


	// exactSize (0: none): what the caller needs at least; tried before waiting for the device when the
	// rounded-up requestedSize does not fit, so that the cache never makes a request fail that fits without it
	Alloc* alloc(size_t requestedSize, size_t exactSize = 0)
	{
		stagetimer::Scope st_alloc(stagetimer::ALLOC);
		if (stagetimer::on()) stagetimer::allocCalls()++;
		// Poll the events before taking the lock that every allocation waits for
		std::vector<Alloc*> ready;
		_closeOwnBatches();
		_collectReady(ready);

		Lock ml(&mutex);

		for (Alloc *a : ready)
			_free(a);

//		printf("alloc: %u ", size);
//		_printState();

		size_t size = requestedSize;

#ifdef CUSTOM_ALLOCATOR_MEMGUARD
		//Ad byte-guards
		size += alignmentSize * GUARD_SIZE; //Ad an integer multiple of alignment size as byte guard size
#endif

#ifdef DUMP_CUSTOM_ALLOCATOR_ACTIVITY
		fprintf(stderr, " %.4f", 100.*(float)size/(float)totalSize);
#endif

		Alloc *newAlloc(NULL);

		if (cache)
		{
			size = alignmentSize*ceilf( (float)size / (float)alignmentSize) ; //To prevent miss-aligned memory

			Alloc *curAlloc = _getFirstSuitedFree(size);

			// Lean pool: add a chunk rather than wait for the GPU (a wait stalls this thread, a
			// chunk only costs a driver call, and the next iteration starts with the size found)
			if (curAlloc == NULL && _growFor(size))
				curAlloc = _getFirstSuitedFree(size);

			if (curAlloc == NULL && exactSize > 0 && exactSize < requestedSize)
			{
				size = alignmentSize*ceilf( (float)exactSize / (float)alignmentSize);
				curAlloc = _getFirstSuitedFree(size);
			}

			//If out of memory
			if (curAlloc == NULL)
			{
	#ifdef DEBUG_CUDA
				size_t spaceDiff = _getTotalFreeSpace();
	#endif
				//Try to recover before throwing error
				if (cacheMode()) cacheCapMB = std::max(1, cacheCapMB.load() / 2); // the caches of the threads hold memory: smaller from now on
				for (int i = 0; i <= ALLOC_RETRY; i ++)
				{
					if (_syncReadyEvents())
					{
						curAlloc = _getFirstSuitedFree(size); //Is there space now?
						if (curAlloc != NULL)
							break; //Success
					}
					else
						usleep(10000); // 10 ms, Order of magnitude of largest kernels
				}
	#ifdef DEBUG_CUDA
				spaceDiff =  _getTotalFreeSpace() - spaceDiff;
				printf("DEBUG_INFO: Out of memory handled by waiting for unfinished tasks, which freed %lu B.\n", spaceDiff);
	#endif

				//Did we manage to recover?
				if (curAlloc == NULL)
				{
					printf("ERROR: CudaCustomAllocator out of memory\n [requestedSpace:             %lu B]\n [largestContinuousFreeSpace: %lu B]\n [totalFreeSpace:             %lu B]\n",
							(unsigned long) size, (unsigned long) _getLargestContinuousFreeSpace(), (unsigned long) _getTotalFreeSpace());

					_printState();

					fflush(stdout);
					CRITICAL(ERRGPUCAOOM);
				}
			}

			if (curAlloc->size == size)
			{
				curAlloc->free = false;
				newAlloc = curAlloc;
			}
			else //Or curAlloc->size is smaller than size
			{
				//Setup new pointer
				newAlloc = new Alloc();
				newAlloc->next = curAlloc;
				newAlloc->ptr = curAlloc->ptr;
				newAlloc->size = size;
				newAlloc->free = false;
				newAlloc->chunk = curAlloc->chunk;

				//Modify old pointer
				curAlloc->ptr = &(curAlloc->ptr[size]);
				curAlloc->size -= size;

				//Insert new allocation region into chain
				if(curAlloc->prev == NULL) //If the first allocation region
					first = newAlloc;
				else
					curAlloc->prev->next = newAlloc;
				newAlloc->prev = curAlloc->prev;
				newAlloc->next = curAlloc;
				curAlloc->prev = newAlloc;
			}
		}
		else
		{
			newAlloc = new Alloc();
			newAlloc->size = size;
			newAlloc->free = false;
			DEBUG_HANDLE_ERROR(cudaMalloc( (void**) &(newAlloc->ptr), size));

			//Just add to start by replacing first
			newAlloc->next = first;
			first->prev = newAlloc;
			first = newAlloc;
		}

#ifdef CUSTOM_ALLOCATOR_MEMGUARD
		newAlloc->backtraceSize = backtrace(newAlloc->backtrace, 20);
		newAlloc->guardPtr = newAlloc->ptr + requestedSize;
		cudaStream_t stream = 0;
		CudaShortcuts::memInit<BYTE>( newAlloc->guardPtr, GUARD_VALUE, size - requestedSize, stream); //TODO switch to specialized stream
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
#endif

		usedBytes += newAlloc->size;
		peakUsed = std::max(peakUsed, usedBytes);
		{
			const size_t room = lean ? capBytes : heldBytes;
			// in use above 7/8 of what the pool may hold, or (lean) more than half of the cap already taken from the
			// device: a pool that has to grow that far is the one where fragmentation decides, and cached blocks add to it
			cacheTight = usedBytes > room - room / 8 || (lean && heldBytes > capBytes / 2);
		}
		newAlloc->owner = this;
		return newAlloc;
	};

	// With a stream: a block of this thread's cache for that stream is used if there is one (RELION_GPU_ALLOC_CACHE)
	Alloc* alloc(size_t requestedSize, cudaStream_t stream)
	{
		if (!cacheMode() || !cacheAllowed().load())
			return alloc(requestedSize);
		if (requestedSize + alignmentSize > kCacheMaxClassBytes)
			return alloc(requestedSize);
		const size_t cls = cacheClass(requestedSize);
		ThreadCache *tc = myCache();
		{
			std::lock_guard<std::mutex> g(tc->m);
			auto it = tc->blocks.find(std::make_pair(stream, cls));
			if (it != tc->blocks.end() && !it->second.empty())
			{
				Alloc *a = it->second.back();
				it->second.pop_back();
				if (it->second.empty())
					tc->blocks.erase(it);
				tc->bytes -= a->size;
				a->readyMarked = false;
				a->freeWhenReady = false;
				cacheHits++;
				if (stagetimer::on()) stagetimer::allocHits()++;
				return a;
			}
		}
		cacheMisses++;
		if (cacheTight.load())
			return alloc(requestedSize); // nearly full: exact size, and the block is not cached when freed
		Alloc *a = alloc(cls, requestedSize);
		a->allocStream = stream;
		a->allocStreamValid = true;
		return a;
	}

	~CudaCustomAllocator()
	{
		_dropCaches();
		{
			Lock ml(&mutex);
			for (auto &kv : pending)
				for (ClosedBatch &c : kv.second)
					DEBUG_HANDLE_ERROR(cudaEventDestroy(c.event));
			pending.clear();
			openBatches.clear();
			_clear();
		}
		omp_destroy_lock(&scanMutex);
		omp_destroy_lock(&openMutex);
		omp_destroy_lock(&pendingMutex);
		for (cudaEvent_t e : eventPool)
			DEBUG_HANDLE_ERROR(cudaEventDestroy(e));
		eventPool.clear();
		omp_destroy_lock(&eventMutex);
		omp_destroy_lock(&mutex);
	}

	//Thread-safe wrapper functions

	void free(Alloc* a)
	{
		Lock ml(&mutex);
		_free(a);
	}

	void syncReadyEvents()
	{
		Lock ml(&mutex);
		_syncReadyEvents();
	}

	void freeReadyAllocs()
	{
		Lock ml(&mutex);
		_freeReadyAllocs();
	}

	size_t getTotalFreeSpace()
	{
		Lock ml(&mutex);
		size_t size = _getTotalFreeSpace();
		return size;
	}

	size_t getTotalUsedSpace()
	{
		Lock ml(&mutex);
		size_t size = _getTotalUsedSpace();
		return size;
	}

	// Bytes held on the device (the one chunk, or all the chunks of a lean pool; 0 while none is allocated)
	size_t chunkSize()
	{
		Lock ml(&mutex);
		return heldBytes;
	}

	// ---- lean pool (RELION_GPU_LEAN_MEMORY) ----------------------------------------------------

	// Make this a lean pool for the next iteration: hold about `target` bytes now, add chunks when
	// something does not fit, never hold more than `cap` bytes. Chunks of the last iteration are kept
	// when one of them fits the target, else (nothing in use) given back and replaced by one of
	// the right size, which also joins the pieces that growth left. The high-water marks restart.
	void configureLean(size_t cap, size_t target)
	{
		Lock ml(&mutex);
		_syncReadyEvents();
		lean = true;
		capBytes = cap;
		cache = true;
		target = std::min(target, cap);
		if (_getNumberOfAllocs() == 0 && !gpupool::keepChunks(chunks.size(), heldBytes, target))
		{
			_clear();
			_makeEmptyList();
			size_t t = target;
			while (t > 0 && !_addChunk(t))
			{
				// Another process took the memory that was free a moment ago: take what is there, 90%
				size_t free_now = 0, total_now = 0;
				cudaMemGetInfo(&free_now, &total_now);
				const size_t next = std::min((size_t)(0.9 * (double) free_now), (size_t)(0.9 * (double) t));
				if (next < gpupool::kMinChunk || next < target / 4)
					break; // growth will try again when something is allocated
				t = next;
			}
			if (heldBytes < target)
				printf(" GPU memory: first chunk %zu MB instead of %zu MB, the GPU has less free memory than measured (other processes on it).\n",
				       heldBytes / (1000*1000), target / (1000*1000));
		}
		totalSize = heldBytes;
		peakUsed = usedBytes;
		peakHeld = heldBytes;
		chunksAdded = 0;
	}

	bool isLean()
	{
		Lock ml(&mutex);
		return lean;
	}

	// The most this pool may hold: the cap of a lean pool, the one chunk of the old pool. Sizes that
	// are chosen from the memory available to the job (not from what is held now) use this.
	size_t capSize()
	{
		Lock ml(&mutex);
		return lean ? capBytes : heldBytes;
	}

	// Free bytes in the chunks plus, for a lean pool, the memory it may still add
	size_t getPotentialFreeSpace()
	{
		Lock ml(&mutex);
		size_t size = _getTotalFreeSpace();
		if (lean && capBytes > heldBytes)
			size += capBytes - heldBytes;
		return size;
	}

	// High-water marks since configureLean(): bytes in use (blocks waiting for their work count as in
	// use, as they are), bytes held, chunks added by growth
	size_t peakUsedBytes()
	{
		Lock ml(&mutex);
		return peakUsed;
	}
	size_t peakHeldBytes()
	{
		Lock ml(&mutex);
		return peakHeld;
	}
	size_t chunksGrown()
	{
		Lock ml(&mutex);
		return chunksAdded;
	}
	size_t numberOfChunks()
	{
		Lock ml(&mutex);
		return chunks.size();
	}

	size_t getNumberOfAllocs()
	{
		Lock ml(&mutex);
		size_t size = _getNumberOfAllocs();
		return size;
	}

	size_t getLargestContinuousFreeSpace()
	{
		Lock ml(&mutex);
		size_t size = _getLargestContinuousFreeSpace();
		return size;
	}

	void printState()
	{
		Lock ml(&mutex);
		_printState();
	}
};
//

#endif
