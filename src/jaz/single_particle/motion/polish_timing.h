#ifndef POLISH_TIMING_H
#define POLISH_TIMING_H

// Wall-clock time per Polish stage, summed over micrographs and printed at the end
// when RELION_POLISH_TIMING=1.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

namespace polishtiming
{
	inline bool enabled()
	{
		static const bool on = getenv("RELION_POLISH_TIMING") && std::string(getenv("RELION_POLISH_TIMING")) != "0";
		return on;
	}

	struct Table
	{
		std::mutex m;
		std::map<std::string, double> sec;
		~Table()
		{
			if (!enabled()) return;
			for (auto &kv : sec) fprintf(stderr, " Polish timing: %-28s %9.2f s\n", kv.first.c_str(), kv.second);
		}
	};

	inline Table &table() { static Table t; return t; }

	inline void add(const char *label, double seconds)
	{
		if (!enabled()) return;
		std::lock_guard<std::mutex> g(table().m);
		table().sec[label] += seconds;
	}

	class Scope
	{
		public:
			explicit Scope(const char *label) : label(label), t0(std::chrono::steady_clock::now()) {}
			~Scope()
			{
				add(label, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
			}
		private:
			const char *label;
			std::chrono::steady_clock::time_point t0;
	};
}

#endif
