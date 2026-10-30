#ifndef FILAMENT_GROUPS_H_
#define FILAMENT_GROUPS_H_

#include <atomic>
#include <cstdio>
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

// How much neighbouring segments share their searched orientations (own / union, smoothed over the
// shared coarse passes so far). Local searches around different priors share little, and the shared pass
// then costs more than it saves (measured 6-10% slower on late helical auto-refine), so it is skipped
// unless the ratio reaches ShareMin; one group in ShareProbe still runs shared to keep the ratio current.
// Global searches overlap about 8x and gain 12-37%. RELION_GPU_GROUP_SHARED=1 always shares.
inline std::atomic<double> &shareRatio() { static std::atomic<double> v(-1.); return v; }
constexpr double ShareMin = 2.;
constexpr int ShareProbe = 32;

inline void shareRatioUpdate(double r)
{
	const double o = shareRatio().load(), n = o < 0. ? r : 0.7 * o + 0.3 * r;
	shareRatio().store(n);
	static std::atomic<int> state(1); // 1: sharing, 0: skipped
	const int now = n >= ShareMin ? 1 : 0;
	if (state.exchange(now) != now)
		printf(" GPU coarse pass: orientation sets of neighbouring segments overlap %.2fx: %s (RELION_GPU_GROUP_SHARED=1 always shares)\n",
				n, now ? "sharing projections again" : "per-particle passes instead of the shared pass");
}

inline bool sharePays()
{
	static std::atomic<int> n(0);
	const double r = shareRatio().load();
	return r < 0. || r >= ShareMin || n.fetch_add(1) % ShareProbe == 0;
}

inline bool envSet(const char *name) { const char *e = getenv(name); return e != NULL && *e != 0; }

}

#endif
