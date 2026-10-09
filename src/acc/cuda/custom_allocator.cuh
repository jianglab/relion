#ifndef CUDA_CUSTOM_ALLOCATOR_CUH_
#define CUDA_CUSTOM_ALLOCATOR_CUH_
// This is where custom allocator should be. Commented out for now, to avoid double declaration.

#ifdef _CUDA_ENABLED
#include "src/acc/cuda/cuda_settings.h"
#include <cuda_runtime.h>
#endif

#include <signal.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <deque>
#include <map>
#include <thread>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "src/macros.h"
#include "src/error.h"
#include "src/parallel.h"

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
		CudaCustomAllocator *owner; // set by alloc(); its pool supplies the ready events
		bool readyMarked;           // markReadyEvent() was called: free after the work
		cudaStream_t readyStream;   // issued so far on this stream by this thread
		std::thread::id readyThread;


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
			owner(NULL),
			readyMarked(false),
			readyStream(0)
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
				owner->enqueuePending(this);
		}
	};

private:

	Alloc *first;
	size_t totalSize;
	size_t alignmentSize;

	bool cache;

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

	// Wait for the device, then take every handed-back block, open or closed
	void _collectAll(std::vector<Alloc*> &ready)
	{
		Lock sl(&scanMutex);
		DEBUG_HANDLE_ERROR(cudaDeviceSynchronize());
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
			if ( a->prev != NULL && a->prev->free)
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
			if ( a->next != NULL && a->next->free)
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
			HANDLE_ERROR(cudaMalloc( (void**) &(first->ptr), totalSize));
			cache = true;
		}
		else
			cache = false;
	}

	void _clear()
	{
		if (first->ptr != NULL)
			DEBUG_HANDLE_ERROR(cudaFree( first->ptr ));

		first->ptr = NULL;

		Alloc *a = first, *nL;

		while (a != NULL)
		{
			nL = a->next;
			delete a;
			a = nL;
		}
	}

public:

	CudaCustomAllocator(size_t size, size_t alignmentSize):
		totalSize(size), alignmentSize(alignmentSize), first(0), cache(true)
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


	Alloc* alloc(size_t requestedSize)
	{
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

			//If out of memory
			if (curAlloc == NULL)
			{
	#ifdef DEBUG_CUDA
				size_t spaceDiff = _getTotalFreeSpace();
	#endif
				//Try to recover before throwing error
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

		newAlloc->owner = this;
		return newAlloc;
	};

	~CudaCustomAllocator()
	{
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
