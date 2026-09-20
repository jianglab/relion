#ifndef CRYOSPARC_PIPELINE_H
#define CRYOSPARC_PIPELINE_H

/* Emit default_pipeline.star for an imported CryoSPARC project.
 *
 * Without it the imported job directories exist on disk but the RELION GUI
 * shows nothing, and downstream jobs cannot be created by picking an input from
 * the job browser. The file records, for every imported job: the process (its
 * directory, type label and status), the nodes it produced (each output file
 * and its type label), and the edges connecting one job's output to the next
 * job's input - which is where the CryoSPARC parent relationships end up.
 */

#include <string>
#include <vector>

namespace cryosparc {

/// One imported job, as it should appear in the pipeline.
struct PipelineJob {
	std::string process;          ///< "Refine3D/job005/" (with trailing slash)
	std::string type_label;       ///< "relion.refine3d"
	std::string cs_uid;           ///< "J48", to resolve parent links
	std::vector<std::string> parent_uids;

	/// Output files, project-relative, paired with their RELION node type label.
	std::vector<std::pair<std::string, std::string> > nodes;

	/// The output other jobs consume (normally the main STAR). Empty if none.
	std::string primary_node;
};

/// RELION process type label for a job directory kind and CryoSPARC job type.
std::string processTypeLabel(const std::string& relion_kind, const std::string& cs_type);

/// RELION node type label for one of a job's output files.
std::string nodeTypeLabel(const std::string& relion_kind, const std::string& filename);

/* Write a job.star into each imported job directory.
 *
 * The GUI loads a job by reading this file: it gives the job type, which tells
 * the GUI which parameter panel to show, and the parameter values to put in it.
 * Without it, clicking an imported job only warns "unrecognised job type, only
 * updating lower half of the GUI". CryoSPARC parameters do not map onto RELION
 * ones, so what is written is the RELION defaults for the type with the input
 * filled in from the job this one was imported downstream of; it makes the job
 * readable and continuable rather than claiming to reproduce the original run.
 *
 * Returns the number of files written.
 */
long writeJobStars(const std::string& out_project, const std::vector<PipelineJob>& jobs);

/// Write <out_project>/default_pipeline.star. Returns the number of processes.
/// `job_counter` is the number RELION should give the next job created in the
/// GUI; it is not simply jobs.size()+1, because a job that failed to import has
/// still taken its number and left a directory behind.
long writePipeline(const std::string& out_project, const std::vector<PipelineJob>& jobs,
                   long job_counter);

} // namespace cryosparc

#endif // CRYOSPARC_PIPELINE_H
