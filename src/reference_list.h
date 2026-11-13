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
#include <map>
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

/// Per-class settings beside the maps of a --ref STAR file (relion_refine reads them into
/// MlModel::ClassSettings): one row per map, columns keyed by label name ("rlnHelicalTwist", ...).
struct ClassTable
{
	std::vector<std::string> maps;
	std::vector<std::string> labels;              ///< the settings columns, in order
	std::vector<std::vector<std::string> > cells; ///< cells[row][column]
	int column(const std::string& label) const;   ///< -1 if absent
};

/// A column the per-class table offers: STAR label, short heading, job option holding its job-wide value, helical only?
struct TableColumn
{
	const char* label;
	const char* heading;
	const char* job_option; ///< "" when there is no job-wide value
	bool helical;
};
const std::vector<TableColumn>& tableColumns();

/// The maps and every other column of a STAR file (table model_classes if there is one, else the first table).
bool readTable(const std::string& fn_star, ClassTable& table, std::string& error);

/// A table with these maps and these columns: cells of `from` are kept (rows matched by map, else by
/// position), missing cells get the value in `defaults` (keyed by label).
ClassTable completeTable(const ClassTable& from, const std::vector<std::string>& maps,
                         const std::vector<std::string>& labels, const std::map<std::string, std::string>& defaults);

/// Drop the columns whose every cell is its default (compared as numbers when both are numbers): not per class.
void dropDefaultColumns(ClassTable& table, const std::map<std::string, std::string>& defaults);

/// Numbers where numbers belong, a search range around its start, mask files that exist ("None" and "auto" allowed).
bool checkTable(const ClassTable& table, std::string& error);

/// "<STAR_DIR>/refs_<12 hex digits of a hash of the table>.star"
std::string tableStarNameFor(const ClassTable& table);

/// Write the table as a --ref STAR file (creating its directory; an identical file is left alone).
bool writeTable(const std::string& fn_star, const ClassTable& table, std::string& error);

/* The table of the "One reference per class" mode of the Class3D GUI: one row per class with its map
 * and every column of tableColumns(). It is kept as text in a job option:
 *   "rlnReferenceImage|<label>|...;<map>|<cell>|...;<map>|<cell>|..."
 * (the first part names the columns). '|' and ';' cannot be part of a cell.
 */

/// The table as text; "" for a table without rows.
std::string encodeTable(const ClassTable& table);

/// Read the text back; "" is a table without rows. False, with a message, if it is not such a text.
bool decodeTable(const std::string& text, ClassTable& table, std::string& error);

/// The table for `nr_classes` classes with the columns `labels`: rows beyond it are dropped; a new row has
/// no map and the settings of the row above it (the defaults for the first row). A column that `from` does
/// not have gets the defaults, except a helical search range (rlnHelicalTwistMin, ...), which is centred on the
/// class's twist or rise with the width of the default range (as relion_refine does without the column).
ClassTable resizeTable(const ClassTable& from, int nr_classes, const std::vector<std::string>& labels,
                       const std::map<std::string, std::string>& defaults);

/// What a per-class table asks of relion_refine.
struct PerClassRefs
{
	/// The maps, and the columns whose cells differ between classes: the --ref STAR file.
	ClassTable table;
	/// Job option -> value, for the job-wide options that the table decides (they are greyed out in the GUI):
	/// a column that is the same for every class becomes that value; one that differs gives a value that
	/// covers all classes (symmetry C1, the widest tube, the widest search range, the finest search step, the first
	/// class's twist and rise).
	/// "fn_mask" and "do_auto_mask" come from the Mask column.
	std::map<std::string, std::string> job;
	/// Mask files named in the table (for the input nodes)
	std::vector<std::string> mask_files;
};

/// Check the table of `nr_classes` rows and split it as above. Only the symmetry and mask columns are used,
/// plus twist, rise and tube diameters if `helical`, plus the search ranges and steps if `symmetry_search`.
/// A column the table lacks leaves its job option alone. Mask files must exist.
bool perClassRefs(const std::string& encoded, int nr_classes, bool helical, bool symmetry_search,
                  PerClassRefs& result, std::string& error);

} // namespace reflist

#endif
