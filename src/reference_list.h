/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef REFERENCE_LIST_H
#define REFERENCE_LIST_H

/* Several reference maps for one Class3D job.
 *
 * relion_refine accepts a STAR file with a column _rlnReferenceImage as --ref and then
 * starts one class from each listed map; the number of classes is the number of rows and
 * --K is ignored. The GUI field "Reference map" holds either one path (a map or such a
 * STAR file) or a comma-separated list of maps. A list of two or more maps is turned into
 * such a STAR file when the job is run. No FLTK here, so everything can be unit tested.
 */

#include <string>
#include <vector>

namespace reflist {

/// Directory (relative to the project) that holds the STAR files made from lists.
extern const char* const STAR_DIR;

/// "a.mrc, b.mrc" -> {"a.mrc", "b.mrc"}: split at commas, trim blanks, drop empty items.
std::vector<std::string> splitList(const std::string& field);

/// {"a.mrc", "b.mrc"} -> "a.mrc,b.mrc"
std::string joinList(const std::vector<std::string>& maps);

/// Add `added` to the end of `field`; a map that is already listed is not added twice.
std::string appendToList(const std::string& field, const std::vector<std::string>& added);

/// The maps listed in the _rlnReferenceImage column of a STAR file (table model_classes
/// if there is one, else the first table). False, with a message, if it cannot be read.
bool readStar(const std::string& fn_star, std::vector<std::string>& maps, std::string& error);

/// Name of the STAR file made for this list: "<STAR_DIR>/refs_<12 hex digits of a hash of the list>.star".
std::string starNameFor(const std::vector<std::string>& maps);

/// Write the STAR file (creating its directory). An existing file with the same content is left alone.
bool writeStar(const std::string& fn_star, const std::vector<std::string>& maps, std::string& error);

/// Every map exists and has the same box size and pixel size as the first one (the
/// program would stop with an unhelpful message, or crash, otherwise). Reads headers only.
bool checkMaps(const std::vector<std::string>& maps, std::string& error);

/// What the field of a job stands for.
struct Resolved
{
	std::vector<std::string> maps; ///< every map that will be a class
	std::string ref_argument;      ///< what to pass as --ref (the map, the user's STAR file, or the STAR file made from a list)
	bool is_list = false;          ///< two or more maps: the STAR file has to be written
	bool is_star = false;          ///< the user gave a STAR file
	bool star_readable = true;     ///< a user's STAR file could be read (it need not exist yet, e.g. in a Scheduler)
};

/// Work out the maps and the --ref argument from the field. False, with a message, for an empty list or "None" among maps.
bool resolve(const std::string& field, Resolved& result, std::string& error);

} // namespace reflist

#endif
