#include "src/async_serial.h"
#include "src/gpu_switch.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace asyncserial
{

bool enabled()
{
	static const bool on = []()
	{
		const char *e = getenv("RELION_ASYNC_SERIAL");
		return e == NULL || e[0] == '\0' || !gpuswitch::isOffValue(e);
	}();
	return on;
}

int helperThreads(int nr_threads)
{
	if (const char *e = getenv("RELION_ASYNC_SERIAL_THREADS"))
	{
		const int n = atoi(e);
		if (n >= 1) return n;
	}
	return std::max(1, nr_threads);
}

void lowerThreadPriority()
{
#ifdef SYS_gettid
	setpriority(PRIO_PROCESS, (id_t) syscall(SYS_gettid), 19);
#endif
}

bool DeferredZero::start(const std::vector<ZeroSpan> &spans, int nr_threads)
{
	wait();
	if (spans.empty()) return true;
	const int nt = std::max(1, nr_threads);
	try
	{
		th_ = std::thread([this, spans, nt]()
		{
			lowerThreadPriority();
			const auto t0 = std::chrono::steady_clock::now();
			for (const ZeroSpan &s : spans)
			{
				if (s.p == NULL || s.bytes == 0) continue;
				char *c = (char *) s.p;
				const int n = (s.bytes >= ((size_t) 32 << 20)) ? nt : 1;
				if (n <= 1)
				{
					memset(c, 0, s.bytes);
					continue;
				}
				#pragma omp parallel for num_threads(n)
				for (int t = 0; t < n; t++)
				{
					const size_t b0 = s.bytes / n * t, b1 = (t == n - 1) ? s.bytes : s.bytes / n * (t + 1);
					memset(c + b0, 0, b1 - b0);
				}
			}
			fill_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		});
	}
	catch (...)
	{
		return false;
	}
	running_ = true;
	return true;
}

double DeferredZero::wait()
{
	if (!running_) return 0.;
	const auto t0 = std::chrono::steady_clock::now();
	th_.join();
	running_ = false;
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}
