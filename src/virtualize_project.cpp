/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/virtualize_project.h"

#include <cctype>
#include <ctime>
#include <sstream>

namespace relion_virtualize {

namespace {

/// Single-quoted for sh, whatever the path holds
std::string quote(const std::string& s)
{
	// Plain paths stay readable; anything else is quoted
	bool plain = !s.empty();
	for (size_t i = 0; i < s.size(); i++)
		if (!(isalnum((unsigned char)s[i]) || s[i] == '.' || s[i] == '/' || s[i] == '_' || s[i] == '-' || s[i] == '+'))
			plain = false;
	if (plain) return s;
	std::string out = "'";
	for (size_t i = 0; i < s.size(); i++)
	{
		if (s[i] == '\'') out += "'\\''";
		else out += s[i];
	}
	return out + "'";
}

} // anonymous namespace

std::vector<std::string> commands(const Options& opt, const std::string& project)
{
	std::vector<std::string> out;
	const int j = (opt.threads > 0) ? opt.threads : 1;
	std::ostringstream threads;
	threads << j;
	if (opt.particles)
	{
		std::string c = "relion_stacks_to_virtual --project " + quote(project) + " --j " + threads.str();
		if (opt.accept_float16_fix) c += " --accept_float16_fix";
		if (opt.convert) c += " --convert";
		out.push_back(c);
	}
	if (opt.movie_averages)
	{
		std::string c = "relion_movie_averages_to_virtual --project " + quote(project) + " --j " + threads.str();
		if (opt.convert) c += " --convert";
		out.push_back(c);
	}
	return out;
}

std::string script(const Options& opt, const std::string& project)
{
	const std::vector<std::string> cmds = commands(opt, project);
	std::string s;
	for (size_t i = 0; i < cmds.size(); i++)
	{
		const std::string program = cmds[i].substr(0, cmds[i].find(' '));
		s += "echo '$ " + program + " ...'; " + cmds[i] + "; echo \"[exit $?] " + program + "\"; ";
	}
	return s + "echo '=== finished ==='";
}

std::string logPath(const std::string& project, bool convert)
{
	char stamp[32];
	const time_t now = time(NULL);
	strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", localtime(&now));
	return project + "/virtualization_" + stamp + (convert ? "_convert" : "_dryrun") + ".log";
}

} // namespace relion_virtualize
