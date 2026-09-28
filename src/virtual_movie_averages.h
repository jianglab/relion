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
 * RELION_VMOVIE_AVERAGE_CACHE_MAX_GB (default 100, 0 for no limit), least
 * recently used entries first. See documentation/virtual_movie_averages.md.
 */

#include <string>

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
void writeDescriptor(const std::string& fn_mic, const std::string& fn_record);

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

/// Forget memoised headers (for tests).
void clearProcessState();

} // namespace vmovies

#endif // VIRTUAL_MOVIE_AVERAGES_H
