/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef VIRTUAL_MOVIE_AVERAGES_H
#define VIRTUAL_MOVIE_AVERAGES_H

/* Virtual movie averages: motion-corrected micrographs computed from their
 * movies when they are read.
 *
 * With RELION_VIRTUAL_MOVIE_AVERAGES=1, RELION's own motion correction writes,
 * under each micrograph's usual name (MotionCorr/jobNNN/.../x.mrc), a small
 * descriptor instead of the image: the motion record (the .star written next to
 * it) behind the first line "# RELION virtual movie average". Image::read
 * recognises it by that line and computes the micrograph from the movie with
 * MotioncorrRunner::sumFromRecord() - bit for bit the micrograph motion
 * correction would have written, float16 rounding included. Micrographs thereby
 * become an accelerator, like the particle stacks of virtual particles: they
 * live in a disposable cache and can be deleted at any time.
 *
 * Cache: RELION_VMOVIE_AVERAGE_CACHE (a directory, or "off"), else
 * Cache/virtual_movie_averages/ in the project; capped by
 * RELION_VMOVIE_AVERAGE_CACHE_MAX_GB (default 0, no limit), least
 * recently used entries first. See documentation/virtual_movie_averages.md.
 */

#include <memory>
#include <string>
#include <vector>

#include "src/multidim_array.h"

namespace vmovies {

enum Mode { OFF, VIRTUAL, REAL };

/// RELION_VIRTUAL_MOVIE_AVERAGES: unset/0/no -> OFF; "real" -> REAL (write real
/// micrographs, made reproducibly); any other true value -> VIRTUAL.
Mode mode();

/// True if the file is a virtual movie average descriptor, whatever its name.
bool isVirtualMovieAverageFile(int fd);
bool isVirtualMovieAverageFile(const std::string& path);

/// Write the descriptor for micrograph `fn_mic` from its motion record (atomic).
/// `replaces` is the identity (vparticles::micrographChecksum) of a real
/// micrograph file this descriptor replaces, after it was verified to hold the
/// same pixels: virtual particles extracted from that file keep verifying.
void writeDescriptor(const std::string& fn_mic, const std::string& fn_record,
                     const std::string& replaces = "");

/// The identity recorded by writeDescriptor(..., replaces), or "".
std::string replacedChecksum(const std::string& fn_descriptor);

/// What a header-only read needs; no movie is touched.
struct Header {
	int nx, ny;
	double angpix;
	bool float16;
};
Header readHeader(const std::string& fn_descriptor);

/// The micrograph's pixels (as float16 or float32 values, as stored).
/// Built from the movie on first use and kept in the cache.
void readMicrograph(const std::string& fn_descriptor, MultidimArray<RFLOAT>& out);

/// Path of a real MRC file holding the micrograph, built into the cache if
/// needed; "" when the cache is off or unwritable (then use readMicrograph).
std::string materialise(const std::string& fn_descriptor);

/// Where the cache lives by default, relative to the project root.
const char PROJECT_CACHE_DIR[] = "Cache/virtual_movie_averages";

/// Cache directory in use, or "" when caching is off.
std::string cacheDirectory();

/// Delete a project's own cache (PROJECT_CACHE_DIR), as the clean-up tools do.
/// Returns true if there was one to remove.
bool removeProjectCache(const std::string& project_dir);

/// Builds the cache entries of the micrographs a program will read next, on background threads
/// while it works on the current one, so that the read finds a cache hit. Only the order of the
/// work changes: the entries are those materialise() makes, and a read that comes first waits for
/// the build in progress (per-key lock) instead of repeating it; one that is not queued is built
/// by the reader as before.
///
/// A build scales poorly with threads, so several run at once with few threads each: K builders
/// of t threads, K*t about omp_get_max_threads(). Chosen automatically: the first build runs
/// alone and its peak memory is measured (K is cut to fit the available memory, and a build
/// waits while less than ~1.5 builds' worth is free); then t = 1 is compared with t = 2 on the
/// following builds and the faster throughput is kept. One line says what was chosen.
/// Overrides: RELION_VMOVIE_READ_AHEAD (micrographs kept queued or in progress; default 2K,
/// 0 = off), RELION_VMOVIE_BUILDERS (K), RELION_VMOVIE_BUILD_THREADS (t).
/// Does nothing when the cache is off, paused after a failed write or not writable, and for
/// files that are not virtual movie averages or are cached already.
class ReadAhead
{
public:
	/// How many of the next micrographs a caller should pass to upcoming()
	static const int WINDOW = 64;

	/// depth < 0: from RELION_VMOVIE_READ_AHEAD (default: automatic)
	explicit ReadAhead(int depth = -1);
	/// Waits for the builds in progress, drops the rest
	~ReadAhead();
	/// How many micrographs the program will go through in all; the automatic trials (about 3K builds) run only for a list
	/// of at least 4K, so without this call (or for a short list) the choice made from the first build stands
	void expectMicrographs(long n);
	/// At most k builders (interactive programs); call before upcoming()
	void limitBuilders(int k);
	/// False when switched off (depth 0): callers need not build the list
	bool enabled() const;
	/// Micrographs kept queued or in progress now
	int depth() const;
	/// Builders running at once and the threads of each (the current choice)
	int builders() const;
	int buildThreads() const;
	/// The micrographs after the current one, in reading order (up to WINDOW; the first `depth` that
	/// need building are kept queued or in progress, replacing what was queued before).
	void upcoming(const std::vector<std::string>& next);
	/// Wait until nothing is queued or being built (for tests)
	void waitIdle();
	/// Entries this object built, and micrographs it was asked to build (RELION_VMOVIE_READ_AHEAD_STATS=1 prints them at the end)
	long built() const;
	long requested() const;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
	ReadAhead(const ReadAhead&);
	ReadAhead& operator=(const ReadAhead&);
};

/// Memory a new process could use: MemAvailable, less if a cgroup v2 limit (any parent's) or the Slurm job's memory
/// (SLURM_MEM_PER_NODE, or SLURM_MEM_PER_CPU x SLURM_CPUS_ON_NODE, less this process's RSS) is nearer; -1 if unknown.
long long memoryAvailable();

/// Forget memoised headers (for tests).
void clearProcessState();

} // namespace vmovies

#endif // VIRTUAL_MOVIE_AVERAGES_H
