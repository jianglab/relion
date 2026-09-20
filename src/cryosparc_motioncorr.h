#ifndef CRYOSPARC_MOTIONCORR_H
#define CRYOSPARC_MOTIONCORR_H

/* Write a CryoSPARC motion-correction job as a RELION MotionCorr job.
 *
 * Produces exactly what relion_run_motioncorr produces, so that CtfFind,
 * Extract, Refine3D and - the demanding one - Bayesian Polishing all run on it
 * unmodified:
 *
 *   MotionCorr/jobNNN/corrected_micrographs.star   data_optics + data_micrographs
 *   MotionCorr/jobNNN/Movies/<name>.star           data_general, data_global_shift,
 *                                                  data_local_motion_model, data_local_shift
 *   MotionCorr/jobNNN/Movies/<name>.mrc            the aligned average (a real copy)
 *   MotionCorr/jobNNN/Movies/<movie>               symlink to the raw movie
 *
 * Ported from the cs2relion Python package, which validated this end-to-end
 * against a real RELION installation. Four of its findings are load-bearing and
 * are preserved here; each of the first three fails silently if dropped:
 *
 *  - CryoSPARC's aligned average is Y-flipped relative to RELION's own
 *    MotionCorr convention. Uncorrected, coordinates land on the vertically
 *    mirrored position and refinement quietly degrades (3.7 A -> 10.3 A with no
 *    error). Because a flip rewrites pixel data, the average is copied, not
 *    symlinked - the one place this importer cannot use a link.
 *  - The local motion field must be sampled at the correspondingly Y-flipped
 *    position, since CryoSPARC's spline lives in its own pre-flip row frame.
 *  - The global and local trajectories need (negate x, keep y) to reach
 *    RELION's convention, calibrated against a RELION data_global_shift table
 *    for a shared movie (correlation +0.9999, RMSE 0.14 px).
 *  - CryoSPARC writes float16 (MRC mode 12) averages when output_f16 is set.
 *    RELION reads those fine but external CTFFIND4 aborts on them, so the copy
 *    is written as float32.
 *
 * Every referenced file is linked into a short in-project path, because RELION
 * mirrors out-of-project absolute paths under each downstream job directory and
 * a deep CryoSPARC path can then exceed external tools' filename buffers.
 */

#include <string>

#include "src/cryosparc_project.h"

namespace cryosparc {

struct MotionCorrOptions {
	int patches_x;
	int patches_y;
	double pre_exposure_A2;
	double amplitude_contrast;
	double dose_per_frame_A2;   ///< <= 0: derive from the import job's total dose
	bool   write_images;        ///< false: metadata only (used by --dry_run)
	int    verb;

	MotionCorrOptions()
		: patches_x(5), patches_y(5), pre_exposure_A2(0.), amplitude_contrast(0.1),
		  dose_per_frame_A2(-1.), write_images(true), verb(1) {}
};

/* Convert one CryoSPARC motion-correction job.
 *
 * cs_job        the job to convert (must be a motion-correction job)
 * out_job_dir   absolute path of the RELION job directory to create, e.g.
 *               <project>/MotionCorr/job001
 * out_project   absolute path of the RELION project root, so that written
 *               paths can be made project-relative the way RELION expects
 *
 * Returns the number of micrographs written.
 */
long writeMotionCorrJob(const Project& project,
                        const Job& cs_job,
                        const std::string& out_project,
                        const std::string& out_job_dir,
                        const Project::Acquisition& acq,
                        const MotionCorrOptions& opts);

/// True if this CryoSPARC job type is a motion-correction job this can convert.
bool isMotionCorrectionJob(const std::string& cs_job_type);

} // namespace cryosparc

#endif // CRYOSPARC_MOTIONCORR_H
