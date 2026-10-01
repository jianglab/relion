/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef IMPORT_PROJECT_H
#define IMPORT_PROJECT_H

/* What "Project > Import RELION project..." imports, decided by the folder the
 * user picks:
 *
 *   a project folder           the whole project
 *   a job folder (Class2D/job008)   that job and every job it depends on
 *   a job type folder (Class2D)     all jobs in it and every job they depend on
 *   anything else                   nothing; the caller tells the user
 *
 * Imported jobs are symbolic links into the source project, so no data is
 * copied. For a part of a project the pipeline file written to the destination
 * lists only the imported jobs and the files they read or wrote.
 *
 * Nothing here depends on the GUI, so it is covered by unit tests.
 */

#include <string>
#include <vector>

namespace relion_import {

enum Scope { INVALID, WHOLE_PROJECT, SINGLE_JOB, JOB_TYPE_FOLDER };

struct Selection
{
	Scope scope;
	std::string chosen;               ///< the folder as picked, without a trailing slash
	std::string project;              ///< the project folder everything is imported from
	std::vector<std::string> jobs;    ///< jobs asked for, as named in the pipeline, e.g. "Class2D/job008/"; empty for a whole project
	std::string what;                 ///< a phrase for messages, e.g. "job Class2D/job008 and the jobs it depends on"

	Selection() : scope(INVALID) {}
};

/// True if the folder is a RELION project (has .gui_projectdir or default_pipeline.star).
bool isProject(const std::string &dir);

/// Decides what the folder stands for. Scope INVALID if it is none of the cases above.
Selection classify(const std::string &dir);

struct Result
{
	bool ok;
	std::string error;                       ///< set when !ok
	int linked;                              ///< job folders newly linked into the destination
	int existing;                            ///< already present in the destination, left as they were
	int missing;                             ///< in the pipeline but with no folder in the source project
	std::vector<std::string> not_in_pipeline;///< asked for, but not listed in the source pipeline (not imported)
	std::vector<std::string> imported;       ///< all jobs imported, the asked-for ones and their ancestors

	Result() : ok(false), linked(0), existing(0), missing(0) {}
};

/// Imports the jobs of a SINGLE_JOB or JOB_TYPE_FOLDER selection into destDir
/// and writes destDir/default_pipeline.star for them. destDir must exist.
Result importJobs(const Selection &sel, const std::string &destDir);

} // namespace relion_import

#endif
