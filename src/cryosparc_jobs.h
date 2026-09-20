#ifndef CRYOSPARC_JOBS_H
#define CRYOSPARC_JOBS_H

/* Writing the non-motion-correction CryoSPARC job types as RELION jobs.
 *
 * Each of these is, at heart, "convert the job's final .cs table to a STAR file
 * under a RELION-conventional name, and make the image files it references
 * reachable". cryosparc_import.h already does the .cs -> .star conversion
 * (including merging passthrough files on uid), so this layer decides:
 *
 *   - which output group of the job carries the metadata worth importing,
 *   - what RELION calls that file inside the job directory,
 *   - which MRC/MRCS files to link, and under what name.
 *
 * Image files are symlinked, never copied - only the motion-corrected averages
 * need rewriting (see cryosparc_motioncorr.h). Where RELION expects a stack
 * extension it does not use (.mrcs for what CryoSPARC writes as .mrc), the
 * symlink simply carries the name RELION expects, which costs nothing.
 */

#include <string>

#include "src/cryosparc_project.h"

namespace cryosparc {

struct JobWriteOptions {
	double amplitude_contrast;
	int verb;

	JobWriteOptions() : amplitude_contrast(0.1), verb(1) {}
};

struct JobWriteResult {
	long n_rows;                  ///< particles / micrographs written
	long n_links;                 ///< image files linked
	std::string main_star;        ///< project-relative path of the job's main STAR
	std::string note;             ///< anything the user should know

	JobWriteResult() : n_rows(0), n_links(0) {}
};

/* Convert one job. `relion_kind` is the RELION job directory prefix chosen for
 * it ("Import", "CtfFind", "Extract", "Class2D", "Select", "InitialModel",
 * "Refine3D", "Class3D", "CtfRefine", "AutoPick", "ManualPick").
 *
 * Throws std::runtime_error if the job has nothing importable.
 */
JobWriteResult writeGenericJob(const Project& project,
                               const Job& cs_job,
                               const std::string& relion_kind,
                               const std::string& out_project,
                               const std::string& out_job_dir,
                               const Project::Acquisition& acq,
                               const JobWriteOptions& opts);

} // namespace cryosparc

#endif // CRYOSPARC_JOBS_H
