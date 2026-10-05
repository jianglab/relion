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

#include "src/complex.h"
#include "src/multidim_array.h"
#include "src/resample_rotate.h"

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
	int    extract_size_y; ///< height of a rectangular box (extract_size is then its width), or -1 for a square box
	bool   rotated;        ///< each particle is cut with its own box angle around a sub-pixel centre
	ResampleMethod interpolation; ///< resampling of rotated boxes

	Recipe();

	/// Width and height of the particles this recipe produces.
	int outputSize() const;
	int outputSizeY() const;
	bool rectangular() const { return extract_size_y > 0 && extract_size_y != extract_size; }
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
/// For rotated recipes `cx`, `cy` hold each particle's exact centre and `angle`
/// the angle its box is turned by (degrees); otherwise they are empty.
void writeDescriptor(const std::string& fn_vstack, const std::string& fn_mic,
                     const Recipe& recipe,
                     const std::vector<std::pair<long, long> >& centres,
                     const std::vector<double>& psi = std::vector<double>(),
                     const std::vector<double>& cx = std::vector<double>(),
                     const std::vector<double>& cy = std::vector<double>(),
                     const std::vector<double>& angle = std::vector<double>());

/// What a header-only read needs.
struct Header {
	long   n;        ///< number of particles
	int    box;      ///< output width (the side length of a square box)
	int    box_y;    ///< output height
	double angpix;
	bool   float16;
};
Header readHeader(const std::string& fn_vstack);

/// Particle `index` (0-based) of a virtual stack, into `out` (box_y rows of box).
/// Thread-safe.
void readParticle(const std::string& fn_vstack, long index, MultidimArray<RFLOAT>& out);

/// Pixel size of the micrograph the particles of a virtual stack were cut from.
double micrographPixelSize(const std::string& fn_vstack);

/// Box (width, height) of the same physical size as the stack's particles at
/// pixel size `angpix`, rounded up to even numbers.
void sameExtentBox(const std::string& fn_vstack, double angpix, int& nx, int& ny);

/// Fused extraction (see fused_extract.h): the Fourier transform of particle
/// `index` on a box of nx x ny pixels of size `angpix`, evaluated directly from
/// the micrograph pixels, as FourierTransform() + CenterFFTbySign() would give
/// for the particle cut at that pixel size. The recipe's normalisation applies,
/// with its radii converted to the new pixel size. Thread-safe.
void readParticleFourier(const std::string& fn_vstack, long index, double angpix, int nx, int ny,
                         MultidimArray<Complex>& F2D);

/// Fused extraction without a virtual stack: what relion_preprocess's options
/// would have said. Radii are in output pixels, except the helical diameter (A).
struct DirectRecipe {
	bool   normalise = false, ramp = true, invert_contrast = false;
	bool   helical = false;      ///< tube-shaped background (needs psi prior)
	bool   rotate = false;       ///< turn each box to put its tube horizontal (--rotate_to_horizontal)
	double bg_radius = -1.;
	double helical_diameter = -1.;
	double mic_angpix = -1.;
};

/// Same as readParticleFourier, for a particle given by its micrograph, its
/// coordinate (micrograph pixels), box angle (degrees; turned boxes) and psi prior (degrees; helical segments).
void readParticleFourierDirect(const std::string& fn_mic, double cx, double cy, double box_angle, double psi_prior,
                               const DirectRecipe& d, double angpix, int nx, int ny,
                               MultidimArray<Complex>& F2D);

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
