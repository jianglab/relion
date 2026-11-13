/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/reference_list.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

#include "src/filename.h"
#include "src/image.h"
#include "src/metadata_table.h"

namespace reflist {

const char* const STAR_DIR = "References";

namespace {

std::string trim(const std::string& s)
{
	const char* blanks = " \t\r\n";
	const size_t b = s.find_first_not_of(blanks);
	if (b == std::string::npos) return "";
	return s.substr(b, s.find_last_not_of(blanks) - b + 1);
}

bool endsWith(const std::string& s, const std::string& suffix)
{
	return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string starText(const std::vector<std::string>& maps)
{
	std::ostringstream out;
	out << "\n# RELION: reference maps of a Class3D job, one class per row\n\ndata_model_classes\n\nloop_\n_rlnReferenceImage #1\n";
	for (const std::string& m : maps) out << m << "\n";
	return out.str();
}

std::string hashedName(const std::string& text)
{
	// FNV-1a: only used to give the same content the same file name
	uint64_t h = 1469598103934665603ULL;
	for (char c : text)
	{
		h ^= (unsigned char)c;
		h *= 1099511628211ULL;
	}
	char name[64];
	snprintf(name, sizeof(name), "refs_%012llx.star", (unsigned long long)(h & 0xffffffffffffULL));
	return std::string(STAR_DIR) + "/" + name;
}

bool writeText(const std::string& fn_star, const std::string& text, std::string& error)
{
	{
		std::ifstream existing(fn_star.c_str());
		if (existing)
		{
			std::stringstream have;
			have << existing.rdbuf();
			if (have.str() == text) return true;
		}
	}
	const size_t slash = fn_star.rfind('/');
	if (slash != std::string::npos)
	{
		FileName dir = fn_star.substr(0, slash);
		mktree(dir);
	}
	std::ofstream out(fn_star.c_str());
	out << text;
	out.close();
	if (!out)
	{
		error = "cannot write " + fn_star;
		return false;
	}
	return true;
}

}

std::vector<std::string> splitList(const std::string& field)
{
	std::vector<std::string> result;
	std::istringstream in(field);
	std::string item;
	while (std::getline(in, item, ','))
	{
		item = trim(item);
		if (!item.empty()) result.push_back(item);
	}
	return result;
}

std::string joinList(const std::vector<std::string>& maps)
{
	std::string result;
	for (size_t i = 0; i < maps.size(); i++)
		result += (i ? "," : "") + maps[i];
	return result;
}

std::string appendToList(const std::string& field, const std::vector<std::string>& added)
{
	std::vector<std::string> maps = splitList(field);
	for (const std::string& a : added)
	{
		const std::string item = trim(a);
		if (!item.empty() && std::find(maps.begin(), maps.end(), item) == maps.end())
			maps.push_back(item);
	}
	return joinList(maps);
}

bool readStar(const std::string& fn_star, std::vector<std::string>& maps, std::string& error)
{
	maps.clear();
	struct stat st;
	if (stat(fn_star.c_str(), &st) != 0)
	{
		error = "cannot find the STAR file " + fn_star;
		return false;
	}
	try
	{
		MetaDataTable md;
		std::string first;
		md.read(fn_star, "model_classes");
		if (!md.getValue(EMDL_MLMODEL_REF_IMAGE, first)) md.read(fn_star);
		if (!md.getValue(EMDL_MLMODEL_REF_IMAGE, first))
		{
			error = fn_star + " has no _rlnReferenceImage column";
			return false;
		}
		FOR_ALL_OBJECTS_IN_METADATA_TABLE(md)
		{
			std::string m;
			md.getValue(EMDL_MLMODEL_REF_IMAGE, m);
			maps.push_back(m);
		}
	}
	catch (RelionError& e)
	{
		error = "cannot read the STAR file " + fn_star;
		maps.clear();
		return false;
	}
	return true;
}

std::string starNameFor(const std::vector<std::string>& maps)
{
	return hashedName(joinList(maps));
}

bool writeStar(const std::string& fn_star, const std::vector<std::string>& maps, std::string& error)
{
	return writeText(fn_star, starText(maps), error);
}

bool checkMaps(const std::vector<std::string>& maps, std::string& error)
{
	long first_box[3] = {0, 0, 0};
	double first_pix = 0.;
	for (size_t i = 0; i < maps.size(); i++)
	{
		struct stat st;
		if (stat(maps[i].c_str(), &st) != 0)
		{
			error = "cannot find the reference map " + maps[i];
			return false;
		}
		long box[3];
		double pix = 0.;
		try
		{
			Image<RFLOAT> img;
			img.read(maps[i], false);
			box[0] = XSIZE(img());
			box[1] = YSIZE(img());
			box[2] = ZSIZE(img());
			img.MDMainHeader.getValue(EMDL_IMAGE_SAMPLINGRATE_X, pix);
		}
		catch (RelionError& e)
		{
			error = "cannot read the reference map " + maps[i];
			return false;
		}
		if (i == 0)
		{
			for (int k = 0; k < 3; k++) first_box[k] = box[k];
			first_pix = pix;
			continue;
		}
		if (box[0] != first_box[0] || box[1] != first_box[1] || box[2] != first_box[2])
		{
			std::ostringstream msg;
			msg << "the reference maps differ in box size: " << maps[0] << " is " << first_box[0] << "x" << first_box[1] << "x" << first_box[2]
			    << " pixels, " << maps[i] << " is " << box[0] << "x" << box[1] << "x" << box[2];
			error = msg.str();
			return false;
		}
		if (std::fabs(pix - first_pix) > 0.001)
		{
			std::ostringstream msg;
			msg << "the reference maps differ in pixel size: " << maps[0] << " has " << first_pix << " A, " << maps[i] << " has " << pix << " A";
			error = msg.str();
			return false;
		}
	}
	return true;
}

bool resolve(const std::string& field, Resolved& result, std::string& error)
{
	result = Resolved();
	const std::string whole = trim(field);
	if (whole.find(',') == std::string::npos)
	{
		// One path: exactly what the field has always meant
		result.ref_argument = whole;
		if (whole.empty() || whole == "None") return true;
		if (endsWith(whole, ".star"))
		{
			result.is_star = true;
			std::string ignored;
			result.star_readable = readStar(whole, result.maps, ignored);
		}
		else
			result.maps.push_back(whole);
		return true;
	}
	result.maps = splitList(whole);
	if (std::find(result.maps.begin(), result.maps.end(), "None") != result.maps.end())
	{
		error = "\"None\" cannot be one of several reference maps.";
		return false;
	}
	for (const std::string& m : result.maps)
		if (endsWith(m, ".star"))
		{
			error = "a STAR file (" + m + ") cannot be one of several reference maps; give the maps themselves.";
			return false;
		}
	if (result.maps.empty())
	{
		error = "the list of reference maps is empty.";
		return false;
	}
	if (result.maps.size() == 1)
	{
		result.ref_argument = result.maps[0];
		return true;
	}
	result.is_list = true;
	result.ref_argument = starNameFor(result.maps);
	return true;
}


int ClassTable::column(const std::string& label) const
{
	for (size_t i = 0; i < labels.size(); i++)
		if (labels[i] == label) return (int)i;
	return -1;
}

const std::vector<TableColumn>& tableColumns()
{
	static const std::vector<TableColumn> columns = {
		{"rlnSymmetryGroup", "Symmetry", "sym_name", false},
		{"rlnSolventMaskName", "Mask", "", false},
		{"rlnHelicalTwist", "Twist", "helical_twist_initial", true},
		{"rlnHelicalRise", "Rise", "helical_rise_initial", true},
		{"rlnHelicalTwistMin", "Twist min", "helical_twist_min", true},
		{"rlnHelicalTwistMax", "Twist max", "helical_twist_max", true},
		{"rlnHelicalTwistInitialStep", "Twist step", "helical_twist_inistep", true},
		{"rlnHelicalRiseMin", "Rise min", "helical_rise_min", true},
		{"rlnHelicalRiseMax", "Rise max", "helical_rise_max", true},
		{"rlnHelicalRiseInitialStep", "Rise step", "helical_rise_inistep", true},
		{"rlnHelicalMaskTubeInnerDiameter", "Tube inner", "helical_tube_inner_diameter", true},
		{"rlnHelicalMaskTubeOuterDiameter", "Tube outer", "helical_tube_outer_diameter", true},
	};
	return columns;
}

bool readTable(const std::string& fn_star, ClassTable& table, std::string& error)
{
	table = ClassTable();
	if (!readStar(fn_star, table.maps, error)) return false;
	try
	{
		MetaDataTable md;
		std::string first;
		md.read(fn_star, "model_classes");
		if (!md.getValue(EMDL_MLMODEL_REF_IMAGE, first)) md.read(fn_star);
		for (EMDLabel l : md.getActiveLabels())
			if (l != EMDL_MLMODEL_REF_IMAGE) table.labels.push_back(EMDL::label2Str(l));
		table.cells.assign(table.maps.size(), std::vector<std::string>(table.labels.size()));
		for (size_t r = 0; r < table.maps.size(); r++)
			for (size_t c = 0; c < table.labels.size(); c++)
			{
				const EMDLabel l = EMDL::str2Label(table.labels[c]);
				double x;
				if (EMDL::isDouble(l) && md.getValue(l, x, (long int)r))
				{
					// "155", not the padded "  155.000000" of getValueToString
					char s[32];
					snprintf(s, sizeof(s), "%.10g", x);
					table.cells[r][c] = s;
				}
				else
					md.getValueToString(l, table.cells[r][c], (long int)r);
			}
	}
	catch (RelionError& e)
	{
		error = "cannot read the STAR file " + fn_star;
		return false;
	}
	return true;
}

ClassTable completeTable(const ClassTable& from, const std::vector<std::string>& maps,
                         const std::vector<std::string>& labels, const std::map<std::string, std::string>& defaults)
{
	ClassTable t;
	t.maps = maps;
	t.labels = labels;
	t.cells.assign(maps.size(), std::vector<std::string>(labels.size()));
	for (size_t r = 0; r < maps.size(); r++)
	{
		// the same row when it holds the same map (one map may start several classes), else the map's row, else the same row
		int fr = (r < from.maps.size() && from.maps[r] == maps[r]) ? (int)r : -1;
		for (size_t i = 0; fr < 0 && i < from.maps.size(); i++)
			if (from.maps[i] == maps[r]) fr = (int)i;
		if (fr < 0 && r < from.maps.size()) fr = (int)r;
		for (size_t c = 0; c < labels.size(); c++)
		{
			const int fc = from.column(labels[c]);
			if (fr >= 0 && fc >= 0)
				t.cells[r][c] = from.cells[fr][fc];
			else
			{
				auto d = defaults.find(labels[c]);
				t.cells[r][c] = (d != defaults.end()) ? d->second : "";
			}
		}
	}
	return t;
}

namespace {

bool toNumber(const std::string& s, double& x)
{
	const std::string t = trim(s);
	if (t.empty()) return false;
	char* end = NULL;
	x = strtod(t.c_str(), &end);
	return end != NULL && *end == '\0';
}

bool sameValue(const std::string& a, const std::string& b)
{
	double x, y;
	if (toNumber(a, x) && toNumber(b, y)) return std::fabs(x - y) <= 1e-9 * std::max(1., std::fabs(x));
	return trim(a) == trim(b);
}

}

void dropDefaultColumns(ClassTable& table, const std::map<std::string, std::string>& defaults)
{
	for (int c = (int)table.labels.size() - 1; c >= 0; c--)
	{
		auto d = defaults.find(table.labels[c]);
		if (d == defaults.end()) continue;
		bool all = true;
		for (const auto& row : table.cells)
			all = all && sameValue(row[c], d->second);
		if (!all) continue;
		table.labels.erase(table.labels.begin() + c);
		for (auto& row : table.cells) row.erase(row.begin() + c);
	}
}

bool checkTable(const ClassTable& table, std::string& error)
{
	for (size_t r = 0; r < table.maps.size(); r++)
	{
		const std::string where = "class " + std::to_string(r + 1) + ": ";
		auto num = [&](const char* label, double& x) {
			const int c = table.column(label);
			return c >= 0 && toNumber(table.cells[r][c], x);
		};
		auto name = [](const std::string& label) {  // the heading the table shows
			for (const TableColumn& c : tableColumns())
				if (label == c.label) return std::string(c.heading);
			return label;
		};
		for (size_t c = 0; c < table.labels.size(); c++)
		{
			const std::string& l = table.labels[c];
			const std::string& v = trim(table.cells[r][c]);
			double x;
			if (l == "rlnSymmetryGroup")
			{
				if (v.empty()) { error = where + "give a point group (C1 for none)"; return false; }
			}
			else if (l == "rlnSolventMaskName")
			{
				struct stat st;
				if (v.empty() || (v != "None" && v != "auto" && stat(v.c_str(), &st) != 0))
				{
					error = where + "the mask must be None, auto or an existing file, not \"" + v + "\"";
					return false;
				}
			}
			else if (EMDL::isNumber(EMDL::str2Label(l)) && !toNumber(v, x))
			{
				error = where + name(l) + " must be a number, not \"" + v + "\"";
				return false;
			}
			else if (l.find("InitialStep") != std::string::npos && x < 0.)
			{
				error = where + name(l) + " must not be negative (0: chosen by RELION)";
				return false;
			}
		}
		const char* range[2][3] = {{"rlnHelicalTwistMin", "rlnHelicalTwist", "rlnHelicalTwistMax"},
		                           {"rlnHelicalRiseMin", "rlnHelicalRise", "rlnHelicalRiseMax"}};
		for (auto& g : range)
		{
			double lo, mid, hi;
			const bool has_lo = num(g[0], lo), has_hi = num(g[2], hi);
			if (has_lo != has_hi) { error = "give both " + name(g[0]) + " and " + name(g[2]); return false; }
			if (has_lo && lo > hi) { error = where + name(g[0]) + " is larger than " + name(g[2]); return false; }
			if (has_lo && num(g[1], mid) && (mid < lo || mid > hi))
			{
				error = where + name(g[1]) + " lies outside " + name(g[0]) + " to " + name(g[2]);
				return false;
			}
		}
	}
	return true;
}

namespace {

std::string tableText(const ClassTable& t)
{
	std::ostringstream out;
	out << "\n# RELION: reference maps of a Class3D job, one class per row, with per-class settings\n\ndata_model_classes\n\nloop_\n_rlnReferenceImage #1\n";
	for (size_t c = 0; c < t.labels.size(); c++)
		out << "_" << t.labels[c] << " #" << c + 2 << "\n";
	for (size_t r = 0; r < t.maps.size(); r++)
	{
		out << t.maps[r];
		for (const std::string& v : t.cells[r]) out << " " << trim(v);
		out << "\n";
	}
	return out.str();
}

}

std::string tableStarNameFor(const ClassTable& table)
{
	return hashedName(tableText(table));
}

bool writeTable(const std::string& fn_star, const ClassTable& table, std::string& error)
{
	return writeText(fn_star, tableText(table), error);
}

namespace {

const char* const MAP_LABEL = "rlnReferenceImage";

// Split at `sep`, keeping empty fields ("a||" -> {"a", "", ""})
std::vector<std::string> splitKeep(const std::string& s, char sep)
{
	std::vector<std::string> out;
	size_t start = 0;
	for (;;)
	{
		const size_t end = s.find(sep, start);
		out.push_back(s.substr(start, end == std::string::npos ? std::string::npos : end - start));
		if (end == std::string::npos) return out;
		start = end + 1;
	}
}

std::string cellText(const std::string& v)
{
	std::string t = trim(v);
	t.erase(std::remove_if(t.begin(), t.end(), [](char c) { return c == '|' || c == ';'; }), t.end());
	return t;
}

// The other end of a search range, and the centre it is laid around
struct Range { const char* label; const char* centre; const char* lo; const char* hi; double sign; };
const Range RANGES[] = {
	{"rlnHelicalTwistMin", "rlnHelicalTwist", "rlnHelicalTwistMin", "rlnHelicalTwistMax", -1.},
	{"rlnHelicalTwistMax", "rlnHelicalTwist", "rlnHelicalTwistMin", "rlnHelicalTwistMax", 1.},
	{"rlnHelicalRiseMin", "rlnHelicalRise", "rlnHelicalRiseMin", "rlnHelicalRiseMax", -1.},
	{"rlnHelicalRiseMax", "rlnHelicalRise", "rlnHelicalRiseMin", "rlnHelicalRiseMax", 1.},
};

std::string numberText(double x)
{
	char s[32];
	snprintf(s, sizeof(s), "%.10g", x);
	return s;
}

}

std::string encodeTable(const ClassTable& table)
{
	if (table.maps.empty()) return "";
	std::string text = MAP_LABEL;
	for (const std::string& l : table.labels) text += "|" + l;
	for (size_t r = 0; r < table.maps.size(); r++)
	{
		text += ";" + cellText(table.maps[r]);
		for (const std::string& v : table.cells[r]) text += "|" + cellText(v);
	}
	return text;
}

bool decodeTable(const std::string& text, ClassTable& table, std::string& error)
{
	table = ClassTable();
	if (trim(text).empty()) return true;
	const std::vector<std::string> rows = splitKeep(trim(text), ';');
	const std::vector<std::string> header = splitKeep(rows[0], '|');
	if (header[0] != MAP_LABEL)
	{
		error = "the table of reference maps does not start with " + std::string(MAP_LABEL);
		return false;
	}
	table.labels.assign(header.begin() + 1, header.end());
	for (size_t r = 1; r < rows.size(); r++)
	{
		std::vector<std::string> cells = splitKeep(rows[r], '|');
		if (cells.size() != header.size())
		{
			error = "row " + std::to_string(r) + " of the table of reference maps has " + std::to_string(cells.size()) +
			        " cells instead of " + std::to_string(header.size());
			table = ClassTable();
			return false;
		}
		table.maps.push_back(trim(cells[0]));
		table.cells.push_back(std::vector<std::string>());
		for (size_t c = 1; c < cells.size(); c++) table.cells.back().push_back(trim(cells[c]));
	}
	return true;
}

ClassTable resizeTable(const ClassTable& from, int nr_classes, const std::vector<std::string>& labels,
                       const std::map<std::string, std::string>& defaults)
{
	ClassTable t;
	t.labels = labels;
	auto deflt = [&](const std::string& label) {
		auto d = defaults.find(label);
		return d != defaults.end() ? d->second : std::string();
	};
	auto isRange = [](const std::string& label) {
		for (const Range& g : RANGES) if (label == g.label) return true;
		return false;
	};
	for (int r = 0; r < nr_classes; r++)
	{
		const bool kept = r < (int)from.maps.size();
		t.maps.push_back(kept ? from.maps[r] : "");
		std::vector<std::string> row(labels.size());
		std::vector<bool> fill(labels.size(), false);
		for (size_t c = 0; c < labels.size(); c++)
		{
			const int fc = from.column(labels[c]);
			if (kept && fc >= 0)
				row[c] = from.cells[r][fc];
			else if (!kept && r > 0)
				row[c] = t.cells[r - 1][c];  // as the row above
			else
				fill[c] = true;
		}
		for (size_t c = 0; c < labels.size(); c++)
			if (fill[c] && !isRange(labels[c])) row[c] = deflt(labels[c]);
		for (size_t c = 0; c < labels.size(); c++)
		{
			if (!fill[c] || !isRange(labels[c])) continue;
			for (const Range& g : RANGES)
			{
				if (labels[c] != g.label) continue;
				double centre, lo, hi;
				const int cc = (int)(std::find(labels.begin(), labels.end(), g.centre) - labels.begin());
				if (cc < (int)labels.size() && toNumber(row[cc], centre) && toNumber(deflt(g.lo), lo) && toNumber(deflt(g.hi), hi))
					row[c] = numberText(centre + g.sign * 0.5 * std::fabs(hi - lo));
				else
					row[c] = deflt(labels[c]);
			}
		}
		t.cells.push_back(row);
	}
	return t;
}

bool perClassRefs(const std::string& encoded, int nr_classes, bool helical, bool symmetry_search,
                  PerClassRefs& result, std::string& error)
{
	result = PerClassRefs();
	ClassTable all;
	if (!decodeTable(encoded, all, error)) return false;
	if (all.maps.empty())
	{
		error = "the table of reference maps is empty.";
		return false;
	}
	if ((int)all.maps.size() != nr_classes)
	{
		error = "the table of reference maps has " + std::to_string(all.maps.size()) + " rows for " +
		        std::to_string(nr_classes) + " classes.";
		return false;
	}
	for (size_t r = 0; r < all.maps.size(); r++)
	{
		const std::string& m = all.maps[r];
		const std::string where = "class " + std::to_string(r + 1) + ": ";
		if (m.empty() || m == "None")
		{
			error = where + "give its reference map.";
			return false;
		}
		if (endsWith(m, ".star"))
		{
			error = where + "a STAR file (" + m + ") cannot be the map of one class.";
			return false;
		}
	}

	// The columns used, in the order of tableColumns()
	std::vector<std::string> used;
	for (const TableColumn& c : tableColumns())
	{
		const std::string l = c.label;
		const bool range = l.find("Min") != std::string::npos || l.find("Max") != std::string::npos ||
		                   l.find("InitialStep") != std::string::npos;
		if (all.column(l) >= 0 && (!c.helical || (helical && (!range || symmetry_search))))
			used.push_back(l);
	}
	ClassTable t;
	t.maps = all.maps;
	t.labels = used;
	for (size_t r = 0; r < all.maps.size(); r++)
	{
		t.cells.push_back(std::vector<std::string>());
		for (const std::string& l : used) t.cells.back().push_back(all.cells[r][all.column(l)]);
	}
	if (!checkTable(t, error)) return false;

	// Same for every class: a job-wide value, and no column
	for (int c = (int)t.labels.size() - 1; c >= 0; c--)
	{
		const std::string l = t.labels[c];
		const char* option = "";
		for (const TableColumn& tc : tableColumns())
			if (l == tc.label) option = tc.job_option;
		bool same = true;
		for (const auto& row : t.cells) same = same && sameValue(row[c], t.cells[0][c]);
		std::vector<double> x;
		double v;
		for (const auto& row : t.cells) if (toNumber(row[c], v)) x.push_back(v);
		if (l == "rlnSolventMaskName")
		{
			bool any_auto = false;
			for (const auto& row : t.cells)
			{
				const std::string m = trim(row[c]);
				any_auto = any_auto || m == "auto";
				if (m != "None" && m != "auto" && std::find(result.mask_files.begin(), result.mask_files.end(), m) == result.mask_files.end())
					result.mask_files.push_back(m);
			}
			const std::string first = trim(t.cells[0][c]);
			result.job["fn_mask"] = (same && first != "None" && first != "auto") ? first : "";
			result.job["do_auto_mask"] = any_auto ? "Yes" : "No";
		}
		else if (same)
			result.job[option] = trim(t.cells[0][c]);
		else if (l == "rlnSymmetryGroup")
			result.job[option] = "C1";  // the orientations of all classes are searched together
		else if (l == "rlnHelicalMaskTubeOuterDiameter" || l == "rlnHelicalTwistMax" || l == "rlnHelicalRiseMax")
			result.job[option] = numberText(*std::max_element(x.begin(), x.end()));
		else if (l == "rlnHelicalMaskTubeInnerDiameter" || l == "rlnHelicalTwistMin" || l == "rlnHelicalRiseMin" ||
		         l == "rlnHelicalTwistInitialStep" || l == "rlnHelicalRiseInitialStep")  // (the finest step)
			result.job[option] = numberText(*std::min_element(x.begin(), x.end()));
		else
			result.job[option] = trim(t.cells[0][c]);
		if (same)
		{
			t.labels.erase(t.labels.begin() + c);
			for (auto& row : t.cells) row.erase(row.begin() + c);
		}
	}
	result.table = t;
	return true;
}

} // namespace reflist
