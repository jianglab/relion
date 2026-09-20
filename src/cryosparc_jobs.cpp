#include "src/cryosparc_jobs.h"
#include "src/cryosparc_cs.h"

#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <cstdio>
#include <iostream>
#include <set>
#include <algorithm>

namespace cryosparc {

namespace {

std::string baseName(const std::string& p)
{
	const size_t s = p.find_last_of('/');
	return (s == std::string::npos) ? p : p.substr(s + 1);
}

std::string relativeTo(const std::string& abs, const std::string& root)
{
	std::string r = root;
	if (!r.empty() && r[r.size() - 1] != '/') r += "/";
	if (abs.compare(0, r.size(), r) == 0) return abs.substr(r.size());
	return abs;
}

bool symlinkOnce(const std::string& target, const std::string& link)
{
	struct stat st;
	if (lstat(link.c_str(), &st) == 0) return false;   // already there
	if (symlink(target.c_str(), link.c_str()) != 0 && errno != EEXIST)
	{
		std::cerr << " WARNING: could not symlink " << link << " -> " << target << std::endl;
		return false;
	}
	return true;
}

/* Link every distinct image file referenced by a blob column of a .cs table
 * into the job directory. RELION mirrors out-of-project absolute paths under
 * each downstream job, which for a deep CryoSPARC tree can overflow external
 * tools' filename buffers, so short in-project names are worth the links.
 *
 * `force_ext` renames the link (e.g. CryoSPARC writes particle stacks as .mrc
 * where RELION expects .mrcs); empty keeps the original extension.
 */
long linkBlobFiles(const Project& project, const CsTable& cs,
                   const std::string& field, const std::string& dest_dir,
                   const std::string& force_ext)
{
	if (!cs.has(field)) return 0;

	std::set<std::string> seen;
	long n = 0;
	for (size_t i = 0; i < cs.rows(); i++)
	{
		std::string p = cs.getString(i, field);
		if (p.empty()) continue;

		// A particle blob path may be "path/to/stack.mrc" for a whole stack; the
		// index into it lives in a separate column, so the file is what matters.
		if (!seen.insert(p).second) continue;

		std::string name = baseName(p);
		if (!force_ext.empty())
		{
			const size_t dot = name.find_last_of('.');
			if (dot != std::string::npos) name = name.substr(0, dot);
			name += force_ext;
		}
		if (symlinkOnce(project.resolve(p), dest_dir + "/" + name)) n++;
	}
	return n;
}

/* The column holding a file path in a blob table.
 *
 * CryoSPARC names it after the result rather than uniformly: a volume written
 * as the "map" result stores "map/path", "map_half_A" stores "map_half_A/path",
 * while particle and class-average tables use "blob/path". Looking only for
 * "blob/path" silently finds nothing for volumes, so try the result's own name
 * first, then the common cases, then any column ending in "/path".
 */
std::string findPathField(const CsTable& t, const std::string& result_name)
{
	if (!result_name.empty() && t.has(result_name + "/path")) return result_name + "/path";
	if (t.has("blob/path")) return "blob/path";
	if (t.has("map/path"))  return "map/path";

	const std::vector<std::string> names = t.fieldNames();
	for (size_t i = 0; i < names.size(); i++)
	{
		const std::string& n = names[i];
		if (n.size() >= 5 && n.compare(n.size() - 5, 5, "/path") == 0) return n;
	}
	return "";
}

/// Convert a group's primary table (plus passthroughs) to a STAR file.
long convertGroup(const Project& project, const OutputGroup& grp,
                  const std::string& star_path,
                  const Project::Acquisition& acq, double amplitude_contrast)
{
	const std::string primary = grp.primaryMetafile();
	if (primary.empty()) return 0;

	const std::vector<std::string> pts = grp.passthroughMetafiles();
	std::string pt_joined;
	for (size_t i = 0; i < pts.size(); i++)
	{
		if (!pt_joined.empty()) pt_joined += ",";
		pt_joined += project.resolve(pts[i]);
	}

	convertCsToStar(project.resolve(primary), star_path, "opticsGroup1",
	                acq.pixel_size_A, acq.voltage, acq.spherical_aberration,
	                amplitude_contrast, pt_joined);

	// Report how many rows the source had, which is what ends up in the STAR
	try { CsTable t(project.resolve(primary)); return (long)t.rows(); }
	catch (const std::exception&) { return 0; }
}

/// The group most worth importing for a given RELION job kind, by CryoSPARC type.
const OutputGroup* pickGroup(const Job& j, const std::string& relion_kind)
{
	// Selections: the kept particles, not the discarded ones
	if (relion_kind == "Select")
	{
		const OutputGroup* g = j.group("particles_selected");
		if (g) return g;
	}

	// Ab-initio: the first class's particles carry the alignments
	if (relion_kind == "InitialModel")
	{
		const OutputGroup* g = j.group("particles_class_0");
		if (g) return g;
		g = j.group("particles_all_classes");
		if (g) return g;
	}

	// Anything particle-shaped
	const OutputGroup* g = j.group("particles");
	if (g) return g;
	g = j.groupOfType("particle");
	if (g) return g;

	// Otherwise the exposures (import / CTF / picking jobs)
	g = j.group("micrographs");
	if (g) return g;
	g = j.group("exposures");
	if (g) return g;
	g = j.group("imported_micrographs");
	if (g) return g;
	g = j.group("imported_movies");
	if (g) return g;
	return j.groupOfType("exposure");
}

/// RELION's conventional name for a job kind's main metadata file.
std::string mainStarName(const std::string& relion_kind, const Job& j)
{
	if (relion_kind == "Import")
	{
		if (j.type == "import_movies")     return "movies.star";
		if (j.type == "import_particles")  return "particles.star";
		if (j.type == "import_volumes")    return "volumes.star";
		return "micrographs.star";
	}
	if (relion_kind == "CtfFind")      return "micrographs_ctf.star";
	if (relion_kind == "Extract")      return "particles.star";
	if (relion_kind == "Select")       return "particles.star";
	if (relion_kind == "Class2D")      return "run_it025_data.star";
	if (relion_kind == "Class3D")      return "run_it025_data.star";
	if (relion_kind == "InitialModel") return "run_it300_data.star";
	if (relion_kind == "Refine3D")     return "run_data.star";
	if (relion_kind == "CtfRefine")    return "particles_ctf_refine.star";
	if (relion_kind == "AutoPick" || relion_kind == "ManualPick") return "particles.star";
	return "particles.star";
}

/* Link a job's volumes under the names RELION expects, so that a Refine3D or
 * InitialModel directory looks like one RELION produced. CryoSPARC names its
 * maps per iteration; only the final ones are referenced here. */
long linkVolumes(const Project& project, const Job& j, const std::string& relion_kind,
                 const std::string& job_dir, std::string& note)
{
	long n = 0;

	const OutputGroup* vol = j.group("volume");
	if (vol == NULL) vol = j.groupOfType("volume");
	if (vol == NULL)
	{
		// Ab-initio names its volume group per class
		vol = j.group("volume_class_0");
		if (vol == NULL) return 0;
	}

	// Each result of a volume group is one map; its .cs holds the blob path.
	struct Want { const char* cs_name; const char* relion_name; };
	static const Want wants[] = {
		{"map",        "run_class001.mrc"},
		{"map_sharp",  "run_class001_sharpened.mrc"},
		{"map_half_A", "run_half1_class001_unfil.mrc"},
		{"map_half_B", "run_half2_class001_unfil.mrc"},
		// Helical and symmetrised refinements also write a symmetry-applied map,
		// which is usually the one worth looking at
		{"map_sym",    "run_class001_symmetrised.mrc"},
		{"mask_fsc",   "run_mask.mrc"},
		{NULL, NULL}
	};

	for (int i = 0; wants[i].cs_name; i++)
	{
		const OutputResult* r = vol->result(wants[i].cs_name);
		if (r == NULL || r->metafile.empty()) continue;
		try
		{
			CsTable t(project.resolve(r->metafile));
			if (t.rows() == 0) continue;
			const std::string pf = findPathField(t, wants[i].cs_name);
			if (pf.empty()) continue;
			const std::string src = project.resolve(t.getString(0, pf));
			std::string dest = job_dir + "/";
			dest += (relion_kind == "InitialModel" && i == 0)
			      ? "run_it300_class001.mrc" : wants[i].relion_name;
			if (symlinkOnce(src, dest)) n++;
		}
		catch (const std::exception& e)
		{
			note += std::string(note.empty() ? "" : "; ") + wants[i].cs_name + ": " + e.what();
		}
	}
	return n;
}

/// Link 2D class averages as a RELION class stack.
long linkClassAverages(const Project& project, const Job& j, const std::string& job_dir)
{
	const OutputGroup* g = j.group("class_averages");
	if (g == NULL) g = j.groupOfType("particle_class_averages");
	if (g == NULL) return 0;

	const std::string primary = g->primaryMetafile();
	if (primary.empty()) return 0;

	try
	{
		CsTable t(project.resolve(primary));
		if (t.rows() == 0) return 0;
		const std::string pf = findPathField(t, "");
		if (pf.empty()) return 0;
		// All rows of a class-average table share one stack
		const std::string src = project.resolve(t.getString(0, pf));
		return symlinkOnce(src, job_dir + "/run_it025_classes.mrcs") ? 1 : 0;
	}
	catch (const std::exception&) { return 0; }
}

} // anonymous namespace

JobWriteResult writeGenericJob(const Project& project,
                               const Job& cs_job,
                               const std::string& relion_kind,
                               const std::string& out_project,
                               const std::string& out_job_dir,
                               const Project::Acquisition& acq,
                               const JobWriteOptions& opts)
{
	JobWriteResult res;

	const OutputGroup* grp = pickGroup(cs_job, relion_kind);
	if (grp == NULL)
		throw std::runtime_error("no importable output group");

	const std::string star_name = mainStarName(relion_kind, cs_job);
	const std::string star_path = out_job_dir + "/" + star_name;

	res.n_rows = convertGroup(project, *grp, star_path, acq, opts.amplitude_contrast);
	res.main_star = relativeTo(star_path, out_project);

	// Make the images the table refers to reachable under short in-project names
	const std::string primary = grp->primaryMetafile();
	if (!primary.empty())
	{
		try
		{
			CsTable cs(project.resolve(primary));

			if (relion_kind == "Extract" || relion_kind == "Select" ||
			    relion_kind == "Class2D" || relion_kind == "Class3D" ||
			    relion_kind == "Refine3D" || relion_kind == "CtfRefine" ||
			    relion_kind == "InitialModel")
			{
				// Particle stacks: RELION expects .mrcs for a stack
				res.n_links += linkBlobFiles(project, cs, "blob/path", out_job_dir, ".mrcs");
			}
			else
			{
				// Micrographs / movies keep their own names
				res.n_links += linkBlobFiles(project, cs, "micrograph_blob/path", out_job_dir, "");
				res.n_links += linkBlobFiles(project, cs, "movie_blob/path", out_job_dir, "");
			}
		}
		catch (const std::exception& e)
		{
			res.note = std::string("could not link images: ") + e.what();
		}
	}

	if (relion_kind == "Refine3D" || relion_kind == "Class3D" || relion_kind == "InitialModel")
		res.n_links += linkVolumes(project, cs_job, relion_kind, out_job_dir, res.note);

	if (relion_kind == "Class2D")
		res.n_links += linkClassAverages(project, cs_job, out_job_dir);

	return res;
}

} // namespace cryosparc
