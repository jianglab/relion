#ifndef REMOVE_INTERMEDIATES_H
#define REMOVE_INTERMEDIATES_H

/* Find and delete the per-iteration files that refinement jobs leave behind.
 *
 * A Class2D/Class3D/Refine3D job writes a full set of files for every
 * iteration - maps, models, optimiser state, data tables - and on a large
 * project those intermediate rounds dominate the disk usage while only the
 * first and the last are normally of any use. This scans a project for such
 * files and reports what could go, so that a caller can show the user the
 * cost before anything is deleted.
 *
 * The rule matches the remove-relion-intermediate-files.py script this
 * replaces: within one job directory, files named run*_it<NNN>_* are grouped by
 * the part of the name before _it (so that a continuation, run_ct7_it012_*, is
 * a group of its own), and the lowest- and highest-numbered iteration of each
 * group is kept.
 */

#include <string>
#include <vector>

namespace relion_cleanup {

/// One file that can be deleted.
struct Candidate {
	std::string path;          ///< path as given to the scan, not canonicalised
	long long   size_bytes;
};

/// What a scan found, ready to be shown to the user before anything happens.
struct Plan {
	std::vector<Candidate> remove;
	std::vector<std::string> kept;   ///< "Class3D/job005/run_it000" etc., for reporting
	long long total_bytes;

	Plan() : total_bytes(0) {}
	bool empty() const { return remove.empty(); }
};

/// Scan a project directory (and every jobNNN directory under it) for
/// intermediate iteration files. Nothing is deleted.
Plan planIntermediateRemoval(const std::string& project_dir);

/// Called every so often while deleting, so that a GUI can show progress and
/// stay responsive. `done` counts files attempted so far, out of `total`.
typedef void (*ProgressFn)(size_t done, size_t total, void* user_data);

/* Delete the files of a plan. Returns the number deleted; `freed_bytes` gets
 * the total size of those that were, and `errors` the ones that could not be.
 *
 * On a large project this takes long enough to be worth reporting, which is
 * what `progress` is for; it is called at intervals rather than per file, and
 * once more when everything has been attempted.
 */
long applyRemoval(const Plan& plan, long long& freed_bytes,
                  std::vector<std::string>& errors,
                  ProgressFn progress = NULL, void* user_data = NULL);

/// "1.2 GB", for dialogs.
std::string humanSize(long long bytes);

} // namespace relion_cleanup

#endif // REMOVE_INTERMEDIATES_H
