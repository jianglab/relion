/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/gpu_locres.h"
#include "src/gpu_switch.h"

#include <cstdlib>
#include <cstring>

namespace gpulocres
{

bool requested(bool cli_do_gpu, bool cli_dont_gpu)
{
	if (cli_dont_gpu) return false;
	if (cli_do_gpu) return true;
	return gpuswitch::on("RELION_GPU_LOCRES");
}

bool doublePrecision()
{
	const char *e = getenv("RELION_GPU_LOCRES");
	if (e != NULL && strcmp(e, "double") == 0) return true;
	if (e != NULL && strcmp(e, "single") == 0) return false;
#ifdef ACC_DOUBLE_PRECISION
	return true;
#else
	return false;
#endif
}

#ifndef _CUDA_ENABLED

int selectDevice(const std::string &, int, int, int)
{
	return -1;
}

struct LocresGpuSession::Impl {};

LocresGpuSession::LocresGpuSession() : impl_(nullptr) {}
LocresGpuSession::~LocresGpuSession() { release(); }

bool LocresGpuSession::init(int, const LocresInitData &, std::string &reason)
{
	reason = "this build has no CUDA";
	return false;
}

bool LocresGpuSession::computePointFsc(const LocresPointBox &, double *, std::string &reason)
{
	reason = "this build has no CUDA";
	return false;
}

bool LocresGpuSession::accumFilteredPoint(const LocresPointBox &, long int, long int,
                                          const double *, double, std::string &reason)
{
	reason = "this build has no CUDA";
	return false;
}

bool LocresGpuSession::finishVolumes(RFLOAT *, RFLOAT *, RFLOAT *, std::string &reason)
{
	reason = "this build has no CUDA";
	return false;
}

void LocresGpuSession::release() {}

#endif /* !_CUDA_ENABLED */

} // namespace gpulocres
