#include "src/cryosparc_pipeline.h"
#include "src/pipeline_jobs.h"

#include <fstream>
#include <iostream>
#include <map>
#include <set>

namespace cryosparc {

std::string processTypeLabel(const std::string& kind, const std::string& cs_type)
{
	if (kind == "Import")
		return (cs_type == "import_movies") ? "relion.import.movies" : "relion.import.other";
	if (kind == "MotionCorr")   return "relion.motioncorr.own";
	if (kind == "CtfFind")      return "relion.ctffind.ctffind4";
	if (kind == "AutoPick")     return "relion.autopick.log";
	if (kind == "ManualPick")   return "relion.manualpick";
	if (kind == "Extract")      return "relion.extract";
	if (kind == "Class2D")      return "relion.class2d";
	if (kind == "Select")       return "relion.select.interactive";
	if (kind == "InitialModel") return "relion.initialmodel";
	if (kind == "Class3D")      return "relion.class3d";
	if (kind == "Refine3D")     return "relion.refine3d";
	if (kind == "CtfRefine")    return "relion.ctfrefine";
	return "relion.import.other";
}

std::string nodeTypeLabel(const std::string& kind, const std::string& filename)
{
	// Volumes and masks are recognised by name, since a job can emit several
	const bool is_mrc = filename.size() > 4 &&
	                    filename.compare(filename.size() - 4, 4, ".mrc") == 0;

	if (is_mrc)
	{
		if (filename.find("_mask") != std::string::npos) return "Mask3D.mrc.relion";
		if (filename.find("_unfil") != std::string::npos)
			return "DensityMap.mrc.relion.halfmap.refine3d";
		if (kind == "InitialModel") return "DensityMap.mrc.relion.initialmodel";
		if (kind == "Class3D")      return "DensityMap.mrc.relion.class3d";
		return "DensityMap.mrc.relion.refine3d";
	}

	if (filename.size() > 5 && filename.compare(filename.size() - 5, 5, ".mrcs") == 0)
		return "ParticleGroupMetadata.star.relion";   // class average stacks

	// STAR files
	if (kind == "Import")
	{
		if (filename.find("movies") != std::string::npos)
			return "MicrographMovieGroupMetadata.star.relion";
		if (filename.find("particles") != std::string::npos)
			return "ParticleGroupMetadata.star.relion";
		return "MicrographGroupMetadata.star.relion";
	}
	if (kind == "MotionCorr") return "MicrographGroupMetadata.star.relion.motioncorr";
	if (kind == "CtfFind")    return "MicrographGroupMetadata.star.relion.ctf";
	if (kind == "Class2D")    return "ParticleGroupMetadata.star.relion.class2d";
	if (kind == "Refine3D")   return "ParticleGroupMetadata.star.relion.refine3d";
	return "ParticleGroupMetadata.star.relion";
}

namespace {

/// CryoSPARC uid -> the RELION output that downstream jobs should consume.
std::map<std::string, std::string> primaryOf(const std::vector<PipelineJob>& jobs)
{
	std::map<std::string, std::string> m;
	for (size_t i = 0; i < jobs.size(); i++)
		if (!jobs[i].primary_node.empty())
			m[jobs[i].cs_uid] = jobs[i].primary_node;
	return m;
}

/* The job option that holds a job type's main input.
 *
 * Filling it in is what makes the GUI panel show where the job's particles or
 * micrographs came from, and is also what a Continue would start from. Import
 * jobs are deliberately absent: their input is a CryoSPARC-side path that means
 * nothing inside the RELION project, so the default empty value is honest.
 */
const char* mainInputOption(const std::string& type_label)
{
	if (type_label.compare(0, 17, "relion.motioncorr") == 0) return "input_star_mics";
	if (type_label.compare(0, 15, "relion.ctffind") == 0)     return "input_star_mics";
	if (type_label.compare(0, 16, "relion.autopick") == 0)    return "fn_input_autopick";
	if (type_label == "relion.manualpick")                    return "fn_in";
	if (type_label == "relion.extract")                       return "star_mics";
	if (type_label.compare(0, 13, "relion.select") == 0)      return "fn_data";
	if (type_label == "relion.class2d")                       return "fn_img";
	if (type_label == "relion.class3d")                       return "fn_img";
	if (type_label == "relion.initialmodel")                  return "fn_img";
	if (type_label == "relion.refine3d")                      return "fn_img";
	if (type_label == "relion.ctfrefine")                     return "fn_data";
	return NULL;
}

} // anonymous namespace

long writeJobStars(const std::string& out_project, const std::vector<PipelineJob>& jobs)
{
	const std::map<std::string, std::string> primary_of = primaryOf(jobs);
	long n = 0;

	for (size_t i = 0; i < jobs.size(); i++)
	{
		const int type = get_proc_type(jobs[i].type_label);
		if (type < 0) continue;   // get_proc_type warns and returns -1

		RelionJob job;
		job.initialise(type);
		job.label = get_proc_label(type);
		job.is_continue = false;

		// An Import job's input came from outside the RELION project, so the
		// default wildcard would claim files that were never read
		if (jobs[i].type_label.compare(0, 13, "relion.import") == 0)
		{
			if (job.joboptions.find("fn_in_raw") != job.joboptions.end())
				job.joboptions["fn_in_raw"].value = "";
			if (job.joboptions.find("fn_in_other") != job.joboptions.end())
				job.joboptions["fn_in_other"].value = "";
		}

		// Point the job at whichever imported parent it consumed
		const char* input_option = mainInputOption(jobs[i].type_label);
		if (input_option != NULL && job.joboptions.find(input_option) != job.joboptions.end())
		{
			for (size_t p = 0; p < jobs[i].parent_uids.size(); p++)
			{
				std::map<std::string, std::string>::const_iterator it =
					primary_of.find(jobs[i].parent_uids[p]);
				if (it == primary_of.end()) continue;
				job.joboptions[input_option].value = it->second;
				break;
			}
		}

		try
		{
			job.write(out_project + "/" + jobs[i].process);
			n++;
		}
		catch (const std::exception& e)
		{
			std::cerr << " WARNING: could not write job.star for " << jobs[i].process
			          << ": " << e.what() << std::endl;
		}
	}
	return n;
}

long writePipeline(const std::string& out_project, const std::vector<PipelineJob>& jobs,
                   long job_counter)
{
	const std::string path = out_project + "/default_pipeline.star";
	std::ofstream f(path.c_str());
	if (!f)
	{
		std::cerr << " WARNING: cannot write " << path << std::endl;
		return 0;
	}

	const std::map<std::string, std::string> primary_of = primaryOf(jobs);

	f << "\n# version 50001\n\ndata_pipeline_general\n\n";
	f << "_rlnPipeLineJobCounter                     " << job_counter << "\n \n";

	f << "\n# version 50001\n\ndata_pipeline_processes\n\nloop_ \n";
	f << "_rlnPipeLineProcessName #1 \n";
	f << "_rlnPipeLineProcessAlias #2 \n";
	f << "_rlnPipeLineProcessTypeLabel #3 \n";
	f << "_rlnPipeLineProcessStatusLabel #4 \n";
	for (size_t i = 0; i < jobs.size(); i++)
		f << jobs[i].process << "       None " << jobs[i].type_label << "  Succeeded \n";
	f << " \n";

	f << "\n# version 50001\n\ndata_pipeline_nodes\n\nloop_ \n";
	f << "_rlnPipeLineNodeName #1 \n";
	f << "_rlnPipeLineNodeTypeLabel #2 \n";
	f << "_rlnPipeLineNodeTypeLabelDepth #3 \n";
	for (size_t i = 0; i < jobs.size(); i++)
	for (size_t n = 0; n < jobs[i].nodes.size(); n++)
		f << jobs[i].nodes[n].first << " " << jobs[i].nodes[n].second << "            1 \n";
	f << " \n";

	f << "\n# version 50001\n\ndata_pipeline_input_edges\n\nloop_ \n";
	f << "_rlnPipeLineEdgeFromNode #1 \n";
	f << "_rlnPipeLineEdgeProcess #2 \n";
	for (size_t i = 0; i < jobs.size(); i++)
	{
		// One edge per imported parent. A parent that was skipped (no RELION
		// equivalent, or not part of a --job subset) simply contributes none,
		// which leaves the job looking like a starting point rather than
		// pointing at something that is not there.
		std::set<std::string> done;
		for (size_t p = 0; p < jobs[i].parent_uids.size(); p++)
		{
			std::map<std::string, std::string>::const_iterator it =
				primary_of.find(jobs[i].parent_uids[p]);
			if (it == primary_of.end()) continue;
			if (!done.insert(it->second).second) continue;
			f << it->second << " " << jobs[i].process << " \n";
		}
	}
	f << " \n";

	f << "\n# version 50001\n\ndata_pipeline_output_edges\n\nloop_ \n";
	f << "_rlnPipeLineEdgeProcess #1 \n";
	f << "_rlnPipeLineEdgeToNode #2 \n";
	for (size_t i = 0; i < jobs.size(); i++)
	for (size_t n = 0; n < jobs[i].nodes.size(); n++)
		f << jobs[i].process << " " << jobs[i].nodes[n].first << " \n";
	f << " \n";

	return (long)jobs.size();
}

} // namespace cryosparc
