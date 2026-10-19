/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/gpu_reconstruct.h"
#include "src/gpu_switch.h"

#include <cstdlib>
#include <cstring>

namespace gpurec
{

bool requested()
{
	return gpuswitch::on("RELION_GPU_RECONSTRUCT");
}

bool doublePrecision()
{
	const char *e = getenv("RELION_GPU_RECONSTRUCT");
	if (e != NULL && strcmp(e, "double") == 0) return true;
	if (e != NULL && strcmp(e, "single") == 0) return false;
#ifdef ACC_DOUBLE_PRECISION
	return true;
#else
	return false;
#endif
}

int symmetriseMode()
{
	return gpuswitch::mode("RELION_GPU_SYMMETRISE");
}

static bool g_gpu_requested = false;
void setGpuRequested(bool r) { g_gpu_requested = r; }
bool gpuRequested() { return g_gpu_requested; }

#ifndef _CUDA_ENABLED
// Without CUDA the reconstruction always runs on the CPU
bool reconstructSkipGridding(const SkipGriddingInput &, RFLOAT *, std::string &reason, bool &warned_zero_weight)
{
	warned_zero_weight = false;
	reason = "this build has no CUDA";
	return false;
}

bool keepPoolActive() { return false; }

bool deviceAvailable() { return false; }

bool helicalSymmetryFourier(const HelixFourierInput &, RFLOAT *, RFLOAT *, std::string &reason, std::string &, bool &partial)
{
	partial = false;
	reason = "this build has no CUDA";
	return false;
}

bool helicalSymmetryRealSpace(const HelixRealSpaceInput &, RFLOAT *, std::string &reason, bool &bad)
{
	bad = false;
	reason = "this build has no CUDA";
	return false;
}
#endif

}
