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
	// FNV-1a: only used to give the same list the same file name
	uint64_t h = 1469598103934665603ULL;
	for (char c : joinList(maps))
	{
		h ^= (unsigned char)c;
		h *= 1099511628211ULL;
	}
	char name[64];
	snprintf(name, sizeof(name), "refs_%012llx.star", (unsigned long long)(h & 0xffffffffffffULL));
	return std::string(STAR_DIR) + "/" + name;
}

bool writeStar(const std::string& fn_star, const std::vector<std::string>& maps, std::string& error)
{
	const std::string text = starText(maps);
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

} // namespace reflist
