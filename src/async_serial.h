#ifndef ASYNC_SERIAL_H_
#define ASYNC_SERIAL_H_

#include <cstddef>
#include <thread>
#include <vector>

/*
 * RELION_ASYNC_SERIAL: run host-side serial work of an iteration beside the E-step instead of
 * after it, so the GPU does not wait for it. On by default; "0"/"off" keeps the
 * older sequential order. The work is the same code on the same inputs, so results are
 * bit-identical; only the time at which it runs changes.
 *
 *  - the half-map alignment search of auto-refine (rank 1 and 2 exchange their reduced maps and
 *    rank 1 searches while the E-step runs; only the orientation update stays in place);
 *  - the zero-fill of the large weighted-sum arrays (needed only when the E-step ends).
 *
 * RELION_ASYNC_SERIAL_THREADS overrides the number of threads the background search uses
 * (default: the job's --j; the background threads run at the lowest priority so that they only
 * use cores the E-step leaves idle).
 */
namespace asyncserial
{

/// The switch (read once)
bool enabled();

/// Threads for background work: RELION_ASYNC_SERIAL_THREADS, else nr_threads
int helperThreads(int nr_threads);

/// Lowest scheduling priority for the calling thread (threads it creates later inherit it)
void lowerThreadPriority();

struct ZeroSpan
{
	void *p;
	size_t bytes;
};

/// Zero-fills memory spans on background threads; wait() joins. Spans must not be touched until wait().
class DeferredZero
{
public:
	~DeferredZero() { wait(); }
	/// Returns false, doing nothing, when the thread could not be started
	bool start(const std::vector<ZeroSpan> &spans, int nr_threads);
	/// Waits for the background zero-fill; returns the seconds it had to wait (0 when none was running)
	double wait();
	bool pending() const { return running_; }
	/// Seconds the zero-fill itself took (valid after wait())
	double fillSeconds() const { return fill_seconds_; }
private:
	std::thread th_;
	bool running_ = false;
	double fill_seconds_ = 0.;
};

}

#endif /* ASYNC_SERIAL_H_ */
