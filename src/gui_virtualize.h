/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef GUI_VIRTUALIZE_H
#define GUI_VIRTUALIZE_H

#include <string>

/* Project > Virtualize movie averages and particles...
 *
 * Runs relion_stacks_to_virtual and relion_movie_averages_to_virtual on the
 * project (see src/virtualize_project.h), first as a dry run, then - after a
 * confirmation - converting. The work runs as a separate background process
 * with its log in the project, so the GUI stays responsive, and closing the
 * window (or the GUI) does not stop it. The window also shows the equivalent
 * command line, to run on a cluster node instead.
 */
// `modal`: opened from a modal window (Manage projects), which would otherwise
// keep every event from it.
void runVirtualizeDialog(const std::string &project_dir, bool modal = false);

#endif
