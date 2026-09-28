/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef VIRTUAL_PARTICLES_H
#define VIRTUAL_PARTICLES_H

/* Virtual particles: particles read straight from their micrographs.
 *
 * A virtual particle stack is a small descriptor file (".vstack") written by
 * relion_preprocess --virtual in place of an .mrcs. It names the micrograph,
 * records the extraction recipe and lists the particle coordinates. Image names
 * keep the usual form, "N@Extract/job012/Movies/mic.vstack", so every program
 * that passes names around is unaffected; only Image::_read dispatches a
 * .vstack to readParticle() here.
 *
 * Contract: a virtual particle is bitwise identical to the particle
 * relion_preprocess would have written with the same options, including the
 * float32 or float16 rounding of the stack it replaces.
 *
 * Reads go through a disposable on-disk cache (one file per micrograph and
 * recipe) so that random access costs no more than a stack; deleting the cache
 * only costs speed. See documentation/virtual_particles.md.
 */

#include <string>
#include <utility>
#include <vector>

#include "src/multidim_array.h"

/// Helpers shared by the caches of virtual data (particles, movie averages).
namespace vcache {

/// False for unset, "", "0", "no", "off", "false", "none".
bool envTruthy(const char* v);

/// mkdir -p
void makeDirs(const std::string& path);

/// Remove the least recently used entries (files ending in `suffix`, one level
/// of shard directories below `dir`) until the cache is back under 90% of
/// `limit_bytes`; dead writers' temporary files older than a day go too.
void prune(const std::string& dir, long long limit_bytes, const std::string& suffix);

/// Delete a directory tree without a shell and without following symlinks
/// out of it. True if everything was removed.
bool removeTree(const std::string& path);

} // namespace vcache

namespace vparticles {

/// How particles are cut from a micrograph and processed: the plain 2D case of
/// relion_preprocess --extract.
struct Recipe {
	int    extract_size;   ///< box cut from the micrograph (micrograph pixels)
	int    scale;          ///< rescaled size, or -1
	int    window;         ///< re-windowed size after rescaling, or -1
	bool   normalise;
	int    bg_radius;      ///< in output pixels
	bool   ramp;
	double white_dust;     ///< sigma, or -1
	double black_dust;
	bool   invert_contrast;
	bool   float16;        ///< round as a --float16 stack would, else as float32
	double angpix;         ///< pixel size of the output particles
	bool   helical;        ///< helical segments: normalised against a tube-shaped background
	double helical_radius; ///< that tube's radius, exactly as relion_preprocess computed it

	Recipe();

	/// Side length of the particles this recipe produces.
	int outputSize() const;
};

/// True for the format extension of a virtual stack ("vstack").
bool isVirtualStackFormat(const std::string& ext);

/// True if the file is a descriptor whatever its name. Stacks converted by
/// relion_stacks_to_virtual keep their .mrcs names, so that no STAR file that
/// refers to them has to change; this is how the reader recognises them.
bool isVirtualStackFile(int fd);
bool isVirtualStackFile(const std::string& path);

/// Every particle of a descriptor, computed from the micrograph without
/// touching the cache (for verification before a stack is replaced).
void computeParticles(const std::string& fn_vstack, std::vector<MultidimArray<RFLOAT> >& out);

/// Drop a descriptor from this process's memo (after it was renamed or replaced).
void forget(const std::string& fn_vstack);

/// Identity of a micrograph file: its size and a hash of its header and of
/// samples of its data. Recorded at extraction, checked before extracting.
std::string micrographChecksum(const std::string& fn_mic);

/// Write a descriptor. `centres` are the whole-pixel particle centres the
/// extraction windows around, i.e. rlnCoordinateX/Y after its truncation;
/// storing those rather than the raw coordinates means text round-off in the
/// STAR file can never move a particle by a pixel.
/// For helical recipes `psi` holds each segment's in-plane angle (the tube
/// mask's orientation); otherwise it is empty.
void writeDescriptor(const std::string& fn_vstack, const std::string& fn_mic,
                     const Recipe& recipe,
                     const std::vector<std::pair<long, long> >& centres,
                     const std::vector<double>& psi = std::vector<double>());

/// What a header-only read needs.
struct Header {
	long   n;        ///< number of particles
	int    box;      ///< output side length
	double angpix;
	bool   float16;
};
Header readHeader(const std::string& fn_vstack);

/// Particle `index` (0-based) of a virtual stack, into `out` (box x box).
/// Thread-safe.
void readParticle(const std::string& fn_vstack, long index, MultidimArray<RFLOAT>& out);

/// Where the cache lives by default, relative to the project root.
const char PROJECT_CACHE_DIR[] = "Cache/virtual_particles";

/// Cache directory in use, or "" when caching is off: RELION_VPARTICLE_CACHE
/// (a directory, or "off"), else PROJECT_CACHE_DIR in the project. Its size is
/// capped by RELION_VPARTICLE_CACHE_MAX_GB (default 100, 0 for no limit),
/// evicting the least recently used entries.
std::string cacheDirectory();

/// Delete a project's own cache (PROJECT_CACHE_DIR), as the clean-up tools do.
/// A cache placed elsewhere with RELION_VPARTICLE_CACHE may be shared between
/// projects and is left alone. Returns true if there was one to remove.
bool removeProjectCache(const std::string& project_dir);

/// Forget memoised descriptors, mappings and cache state (for tests).
void clearProcessState();

} // namespace vparticles

#endif // VIRTUAL_PARTICLES_H
