/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef GPU_LOCRES_H_
#define GPU_LOCRES_H_

#include "src/macros.h"
#include <string>

namespace gpulocres
{

struct LocresPointBox
{
	long int kz0 = 0, cz = 0;
	long int ky0 = 0, cy = 0;
	long int kx0 = 0, cx = 0;
	long int dm_z0 = 0, dm_y0 = 0, dm_x0 = 0;
};

struct LocresInitData
{
	const RFLOAT *I1 = nullptr;
	const RFLOAT *I2 = nullptr;
	const RFLOAT *I1p = nullptr;
	const RFLOAT *I2p = nullptr;
	// Interleaved complex FTsum_yxz stored in [ny][nfx][nz] order (2*double per Complex)
	const RFLOAT *FTsum_yxz = nullptr;
	const RFLOAT *mask_tmpl = nullptr;
	const int *r2_to_shell = nullptr;
	long int nx = 0, ny = 0, nz = 0, nfx = 0, S = 0;
	bool double_precision = false;
};

/// True if GPU local resolution should be attempted:
/// --dont_gpu disables, --gpu enables; otherwise RELION_GPU_LOCRES controls it (default ON; =0/off disables).
bool requested(bool cli_do_gpu, bool cli_dont_gpu = false);

/// Whether GPU local resolution runs in FP64 (RELION_GPU_LOCRES=double) or FP32 (default).
bool doublePrecision();

/// Select a CUDA device for local resolution (returns device index, or -1 if unavailable).
int selectDevice(const std::string &gpu_ids, int rank, int size, int verb);

/// Persistent GPU workspace for Postprocessing::run_locres() in full double precision.
class LocresGpuSession
{
public:
	LocresGpuSession();
	~LocresGpuSession();
	LocresGpuSession(const LocresGpuSession &) = delete;
	LocresGpuSession &operator=(const LocresGpuSession &) = delete;

	bool init(int device_id, const LocresInitData &in, std::string &reason);

	/// Compute unnormalized FSC accumulators for (I1, I2) and (I1p, I2p) on the GPU
	/// for one sampling point; writes 6*nfx values into out_stats_6nfx:
	/// [num_masked, den1_masked, den2_masked, num_rand, den1_rand, den2_rand].
	bool computePointFsc(const LocresPointBox &pt, double *out_stats_6nfx, std::string &reason);

	/// Filter FTsum_yxz by inv_w[0..r2_bound], run pruned 3D iFFT, and accumulate
	/// into GPU Ifil, Ilocres (m / local_resol), and Isumw volumes.
	bool accumFilteredPoint(const LocresPointBox &pt,
	                        long int r2_bound, long int n_fx_inv,
	                        const double *inv_w, double inv_local_resol,
	                        std::string &reason);

	/// Download accumulated volumes from GPU to host buffers (nvox = nz * ny * nx).
	bool finishVolumes(RFLOAT *Ifil_out, RFLOAT *Ilocres_out, RFLOAT *Isumw_out, std::string &reason);

	void release();

private:
	struct Impl;
	Impl *impl_ = nullptr;
};

} // namespace gpulocres

#endif /* GPU_LOCRES_H_ */
