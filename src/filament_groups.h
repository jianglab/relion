#ifndef FILAMENT_GROUPS_H_
#define FILAMENT_GROUPS_H_

#include <atomic>
#include <cstdlib>
#include <cstring>

/*
 * Shared coarse projections for neighbouring helical segments on the GPU: filament order
 * (RELION_PARTICLE_ORDER), particle groups of 8 (RELION_GPU_PARTICLE_GROUP) and the shared coarse
 * pass (RELION_GPU_GROUP_SHARED). On by default for runs with 3D references whose data have
 * rlnHelicalTubeID (MlOptimiser::initialiseGeneral sets the default before the first E-step): the order
 * on CPU and GPU alike, so that both visit the particles in the same order; groups and the shared pass
 * exist on the GPU only. Each variable, when set, overrides its part.
 */
namespace filgroup
{

inline std::atomic<bool> &defaultOn() { static std::atomic<bool> v(false); return v; }

inline bool envSet(const char *name) { const char *e = getenv(name); return e != NULL && *e != 0; }

}

#endif
