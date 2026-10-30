#ifndef TUNING_CACHE_H_
#define TUNING_CACHE_H_

// A small per-user file of tuning results, shared by all jobs and MPI ranks on the machine: what a
// run found best for a tunable parameter on a given GPU (for example the tile shape of the GPU
// back-projection), kept as a CSV so that the next run can START from it. It is only a starting
// point: callers must still check their choice while they run, and it can never change results.
//
// File: $HOME/.relion/tuning.csv (RELION_TUNING_FILE overrides; RELION_TUNING_CACHE=off disables).
// Columns: tunable,device,sm_count,driver,runtime,build,key,value,rel_time,date
//   rel_time = time of the chosen value / time of the runner-up (< 1: faster), 0 if unknown.
// A row is appended in one write under an exclusive flock on "<file>.lock" (also taken, shared, to
// read), so concurrent processes and threads cannot corrupt it; the newest matching row wins.
// A file that cannot be read or written is ignored silently (get() false, put() false).
// Pure C++ (POSIX), no dependency on the rest of RELION.

#include <string>

namespace tuning
{

// Where the results of this run are measured: recorded in every row. device, sm_count, tunable and
// key must match for a row to be used; driver, runtime and build are kept for information.
struct Context
{
	std::string device;
	int sm_count = 0;
	std::string driver, runtime, build;
};

struct Entry
{
	std::string value;
	double rel_time = 0.;
	std::string date;
};

// The file from the environment, or "" if the cache is switched off or there is no home directory
std::string defaultPath();

class Cache
{
public:
	// path == "": defaultPath(). An explicit path is used as given (tests, tools).
	explicit Cache(const Context &ctx, const std::string &path = std::string());

	bool enabled() const { return !path_.empty(); }
	const std::string &path() const { return path_; }

	// the newest row of this tunable, key and context
	bool get(const std::string &tunable, const std::string &key, Entry &out) const;
	// appends a row; false if it could not be written. Nothing is appended (true) when the newest row of
	// this tunable, key and context has the same value, or, with same_within > 0, a number within that
	// fraction of the new one: the file stays a short list of results, not a log of every run.
	bool put(const std::string &tunable, const std::string &key, const std::string &value, double rel_time = 0., double same_within = 0.);

	// rows above which put() rewrites the file with the newest row per (tunable, device, sm, key)
	static constexpr size_t kCompactRows = 2000;

private:
	Context ctx_;
	std::string path_;
};

} // namespace tuning

#endif
