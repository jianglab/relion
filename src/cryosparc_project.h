#ifndef CRYOSPARC_PROJECT_H
#define CRYOSPARC_PROJECT_H

/* The job graph of a CryoSPARC project, read from each job's job.json.
 *
 * A CryoSPARC project directory holds one JNN/ directory per job, each with a
 * job.json describing the job's type, status, parents and outputs. That is
 * everything needed to walk the project, decide what is worth importing, and
 * find the *final* metadata of each job.
 *
 * Finding the final round matters: an iterative job writes one .cs per
 * iteration (J48_000_*, J48_001_*, ... for a 4-round helix_refine) and lists
 * all of them. Each output_results entry carries parallel `metafiles` and
 * `versions` arrays, so the final one is metafiles[argmax(versions)]. The job
 * also writes .csg group descriptors that reference only the final iteration;
 * those were checked to agree with the argmax rule on both an iterative refine
 * and a 2D classification, so this reads job.json alone and needs no YAML
 * parser.
 */

#include <string>
#include <vector>
#include <map>

namespace cryosparc {

/// One declared output of a job, e.g. the particle blob or the refined volume.
struct OutputResult {
	std::string group_name;   ///< e.g. "particles"
	std::string name;         ///< e.g. "blob", "ctf", "alignments3D"
	std::string type;         ///< e.g. "particle.blob", "volume.blob"
	std::string metafile;     ///< FINAL version, project-relative
	bool passthrough;         ///< merged onto the primary table by uid
	long num_items;

	OutputResult() : passthrough(false), num_items(0) {}
};

/// A group of outputs that together describe one entity (particles, a volume,
/// a set of exposures).
struct OutputGroup {
	std::string name;         ///< e.g. "particles"
	std::string type;         ///< e.g. "particle", "volume", "exposure", "mask"
	std::string title;
	long num_items;
	std::vector<OutputResult> results;

	OutputGroup() : num_items(0) {}

	/// The .cs file carrying the group's primary table: the most frequently
	/// referenced non-passthrough metafile. CryoSPARC splits a group's columns
	/// over several results that usually share one file.
	std::string primaryMetafile() const;

	/// Distinct passthrough files, to be merged onto the primary table by uid.
	std::vector<std::string> passthroughMetafiles() const;

	/// The result with this name, or NULL.
	const OutputResult* result(const std::string& name) const;

	bool hasResultType(const std::string& type) const;
};

/// One CryoSPARC job.
struct Job {
	std::string uid;          ///< e.g. "J48"
	std::string type;         ///< e.g. "helix_refine"
	std::string status;       ///< e.g. "completed", "building", "killed", "failed"
	std::string title;
	std::vector<std::string> parents;
	std::vector<OutputGroup> groups;

	bool isCompleted() const { return status == "completed"; }

	/// Output group by name, or NULL.
	const OutputGroup* group(const std::string& name) const;

	/// First output group of this CryoSPARC type ("particle", "volume",
	/// "exposure", "mask"), or NULL.
	const OutputGroup* groupOfType(const std::string& type) const;
};

/// A CryoSPARC project directory.
class Project {
public:
	/// Read every JNN/job.json under `dir`. Jobs whose job.json is missing or
	/// unreadable are skipped with a warning rather than aborting the import, so
	/// one damaged job does not make the whole project unimportable.
	static Project read(const std::string& dir, int verb = 1);

	const std::string& dir() const { return dir_; }
	const std::map<std::string, Job>& jobs() const { return jobs_; }

	/// The job with this uid, or NULL.
	const Job* job(const std::string& uid) const;

	/// Completed jobs in dependency order (parents before children). Jobs whose
	/// parents are missing or incomplete still appear; it is up to the caller to
	/// decide whether it can import them.
	std::vector<std::string> topologicalOrder() const;

	/// `uid` and all of its ancestors, in dependency order. Used by
	/// --job to import only what a chosen result actually depends on.
	std::vector<std::string> ancestryOf(const std::string& uid) const;

	/// Turn a project-relative path from a .cs file into an absolute one.
	std::string resolve(const std::string& relative) const;

	/// Acquisition metadata from an import_movies / import_micrographs job:
	/// voltage, pixel size, Cs, total dose, gain reference. Values absent from
	/// the project are left untouched, so callers can supply their own defaults.
	struct Acquisition {
		double voltage;               ///< kV
		double pixel_size_A;
		double spherical_aberration;  ///< mm
		double total_dose_e_per_A2;
		std::string gainref_path;
		bool found;

		Acquisition()
			: voltage(0.), pixel_size_A(0.), spherical_aberration(0.),
			  total_dose_e_per_A2(0.), found(false) {}
	};

	Acquisition acquisition() const;

private:
	std::string dir_;
	std::map<std::string, Job> jobs_;

	void addJobFromFile(const std::string& job_dir, const std::string& uid, int verb);
};

/// True if `uid` looks like a CryoSPARC job directory name (J followed by digits).
bool isJobDirName(const std::string& name);

/// Sort key for job uids, so J2 comes before J10.
long jobNumber(const std::string& uid);

} // namespace cryosparc

#endif // CRYOSPARC_PROJECT_H
