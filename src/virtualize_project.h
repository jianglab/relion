/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef VIRTUALIZE_PROJECT_H
#define VIRTUALIZE_PROJECT_H

/* Converting a project to virtual data in place, as the GUI's
 * "Project > Virtualize movie averages and particles..." runs it:
 * relion_stacks_to_virtual (particle stacks of Extract jobs), then
 * relion_movie_averages_to_virtual (micrographs of MotionCorr jobs).
 *
 * Particles go first: verifying a stack reads its micrograph, which is fast
 * while the micrograph is still a real file. Either converter replaces a file
 * only after checking it is reproduced bit for bit, and is a dry run unless
 * told to convert.
 */

#include <string>
#include <vector>

namespace relion_virtualize {

struct Options {
	bool particles;
	bool movie_averages;
	bool accept_float16_fix;   ///< stacks with the pre-5.0 float16 writer's doubled pixels
	bool convert;              ///< false: dry run, nothing is written
	int  threads;
	Options() : particles(true), movie_averages(true), accept_float16_fix(false), convert(false), threads(8) {}
};

/// The commands, in the order they must run, each a single shell word list
/// (arguments quoted where needed). Empty if nothing is selected.
std::vector<std::string> commands(const Options& opt, const std::string& project = ".");

/// One shell command running all of them in turn, each followed by a line
/// "[exit N] <program>", and a final "=== finished ===".
std::string script(const Options& opt, const std::string& project = ".");

/// Where the GUI keeps a run's log: <project>/virtualization_<time>_<dryrun|convert>.log
std::string logPath(const std::string& project, bool convert);

} // namespace relion_virtualize

#endif
