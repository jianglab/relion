#ifndef ACC_STAGE_TIMER_H_
#define ACC_STAGE_TIMER_H_

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

// RELION_ITER_TIMING=1: thread-seconds spent in a few host-side stages of the E-step,
// summed over all threads, printed (and reset) once per iteration
namespace stagetimer
{
enum Stage { PAIR_LIST, INDEX_UPLOAD, COLLECT_JOBS, WAVG_SETUP, COARSE, CONVERT, FINE, STORE, FOURIER, MWEIGHT_INIT, PRIORS, PARTICLE, ALLOC, N_STAGES };
inline const char *name(int s)
{
	static const char *n[] = { "pair_list", "index_upload", "collect_jobs", "wavg_setup", "coarse", "convert", "fine_total", "store_total", "fourier", "mweight_init", "priors", "particle_total", "alloc" };
	return n[s];
}
inline bool on() { static const bool v = getenv("RELION_ITER_TIMING") != NULL; return v; }
inline std::atomic<long long> &slot(int s) { static std::atomic<long long> a[N_STAGES]; return a[s]; }
// host<->device traffic: 0 upload bytes, 1 upload calls, 2 download bytes, 3 download calls
inline std::atomic<long long> &xfer(int i) { static std::atomic<long long> a[4]; return a[i]; }
inline std::atomic<long long> &allocCalls() { static std::atomic<long long> a; return a; }
inline std::atomic<long long> &allocHits() { static std::atomic<long long> a; return a; }
inline int &curStage() { static thread_local int c = N_STAGES; return c; }
inline std::atomic<long long> &stageUp(int s) { static std::atomic<long long> a[N_STAGES + 1]; return a[s]; }
inline std::atomic<long long> &stageUpCalls(int s) { static std::atomic<long long> a[N_STAGES + 1]; return a[s]; }
inline std::atomic<long long> &stageDownCalls(int s) { static std::atomic<long long> a[N_STAGES + 1]; return a[s]; }
inline void countUp(size_t bytes) { if (on()) { xfer(0) += bytes; xfer(1)++; stageUp(curStage()) += bytes; stageUpCalls(curStage())++; } }
inline void countDown(size_t bytes) { if (on()) { xfer(2) += bytes; xfer(3)++; stageDownCalls(curStage())++; } }
struct Scope
{
	int s, prev; std::chrono::steady_clock::time_point t0; bool active;
	explicit Scope(int stage) : s(stage), prev(N_STAGES), active(on()) { if (active) { t0 = std::chrono::steady_clock::now(); prev = curStage(); curStage() = s; } }
	~Scope() { if (active) { slot(s) += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(); curStage() = prev; } }
};
inline void reportAndReset(int iter)
{
	if (!on()) return;
	fprintf(stderr, "ITER_TIMING stages it%d thread-s:", iter);
	for (int s = 0; s < N_STAGES; s++)
		fprintf(stderr, " %s %.2f", name(s), 1e-9 * slot(s).exchange(0));
	fprintf(stderr, "\n");
	fprintf(stderr, "ITER_TIMING upload MB by innermost stage it%d:", iter);
	for (int s = 0; s <= N_STAGES; s++)
		fprintf(stderr, " %s %.0f", s < N_STAGES ? name(s) : "other", 1e-6 * stageUp(s).exchange(0));
	fprintf(stderr, "\n");
	fprintf(stderr, "ITER_TIMING calls by innermost stage it%d: up/down", iter);
	for (int s = 0; s <= N_STAGES; s++)
		fprintf(stderr, " %s %lld/%lld", s < N_STAGES ? name(s) : "other", stageUpCalls(s).exchange(0), stageDownCalls(s).exchange(0));
	fprintf(stderr, "\n");
	fprintf(stderr, "ITER_TIMING traffic it%d: allocs %lld, cache hits %lld, up %.0f MB in %lld calls, down %.0f MB in %lld calls\n", iter, allocCalls().exchange(0), allocHits().exchange(0),
			1e-6 * xfer(0).exchange(0), xfer(1).exchange(0), 1e-6 * xfer(2).exchange(0), xfer(3).exchange(0));
}
}

#endif
