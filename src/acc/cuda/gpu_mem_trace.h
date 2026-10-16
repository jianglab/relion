#ifndef GPU_MEM_TRACE_H_
#define GPU_MEM_TRACE_H_

// RELION_GPU_MEM_TRACE=1 prints, from every process, how much of the card is in use at the marks
// placed where memory outside the allocator pool is taken or given back (models, cuFFT plans,
// transforms of the references, GPU reconstruction). "change" is the change of the card's total
// since the last mark of this process, so other processes on the card show up in it too.
#include <cuda_runtime.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>

inline bool gpuMemTraceOn()
{
	static const bool on = []() { const char *e = getenv("RELION_GPU_MEM_TRACE"); return e && *e && atoi(e) > 0; }();
	return on;
}

inline void gpuMemTrace(const char *what, double expected_bytes = 0.)
{
	if (!gpuMemTraceOn())
		return;
	static std::atomic<long> last(0);
	static const char *rank = []() {
		const char *r = getenv("OMPI_COMM_WORLD_RANK");
		if (!r) r = getenv("PMI_RANK");
		if (!r) r = getenv("SLURM_PROCID");
		return r ? r : "0";
	}();
	size_t f = 0, t = 0;
	cudaMemGetInfo(&f, &t);
	const long used = (long) ((t - f) >> 20);
	const long before = last.exchange(used);
	fprintf(stderr, "[gpu-mem] rank %s: card in use %ld MB (change %+ld MB) %s", rank, used, used - before, what);
	if (expected_bytes > 0.)
		fprintf(stderr, " [expected %.0f MB]", expected_bytes / (1024. * 1024.));
	fprintf(stderr, "\n");
}

#endif
