/***************************************************************************
 *
 * Author: "Sjors H.W. Scheres"
 * MRC Laboratory of Molecular Biology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/
#include "src/pipeline_control.h"
#include <cstdio>
#include <iomanip>
#include <sstream>

std::string pipeline_control_outputname = "";
int pipeline_control_task_id = 0;
int pipeline_control_task_count = 1;

static std::string pipeline_control_task_marker(const std::string &marker, int task_id)
{
	std::ostringstream oss;
	oss << marker << "_" << std::setfill('0') << std::setw(3) << task_id;
	return oss.str();
}

static bool pipeline_control_file_exists(const std::string &fn)
{
	struct stat buffer;
	return stat(fn.c_str(), &buffer) == 0;
}

static bool pipeline_control_touch(const std::string &fn)
{
	std::ofstream fh(fn.c_str(), std::ios::out);
	return !!fh;
}

int pipeline_control_relion_exit(int mode)
{

    if (pipeline_control_outputname != "")
    {
		std::string fn = pipeline_control_outputname;
		std::string marker;
		if (mode==0)
		{
			marker = RELION_JOB_EXIT_SUCCESS;
		}
		else if (mode==1)
		{
			marker = RELION_JOB_EXIT_FAILURE;
			std::cout << std::endl << " RELION version: " << g_RELION_VERSION << std::endl << " exiting with an error ..." << std::endl;
		}
		else if (mode==2)
		{
			marker = RELION_JOB_EXIT_ABORTED;
			std::cout << std::endl << " exiting with an abort ..." << std::endl;
		}
		else
		{
			std::cerr << " ERROR: undefined mode! " << std::endl;
			return 12;
		}
		if (pipeline_control_task_count > 1 &&
		    pipeline_control_task_id >= 1 &&
		    pipeline_control_task_id <= pipeline_control_task_count)
		{
			fn += pipeline_control_task_marker(marker, pipeline_control_task_id);
			if (!pipeline_control_touch(fn))
			{
				std::cerr << " ERROR: cannot touch file: " << fn << std::endl;
				return 13;
			}

			bool all_finished = true;
			bool any_failure = false;
			bool any_aborted = false;
			for (int task = 1; task <= pipeline_control_task_count; task++)
			{
				bool success = pipeline_control_file_exists(pipeline_control_outputname + pipeline_control_task_marker(RELION_JOB_EXIT_SUCCESS, task));
				bool failure = pipeline_control_file_exists(pipeline_control_outputname + pipeline_control_task_marker(RELION_JOB_EXIT_FAILURE, task));
				bool aborted = pipeline_control_file_exists(pipeline_control_outputname + pipeline_control_task_marker(RELION_JOB_EXIT_ABORTED, task));
				all_finished = all_finished && (success || failure || aborted);
				any_failure = any_failure || failure;
				any_aborted = any_aborted || aborted;
			}

			if (!all_finished)
				return mode;

			if (any_failure)
				fn = pipeline_control_outputname + RELION_JOB_EXIT_FAILURE;
			else if (any_aborted)
				fn = pipeline_control_outputname + RELION_JOB_EXIT_ABORTED;
			else
				fn = pipeline_control_outputname + RELION_JOB_EXIT_SUCCESS;
		}
		else
		{
			fn += marker;
		}

		if (!pipeline_control_touch(fn))
		{
			std::cerr << " ERROR: cannot touch file: " << fn << std::endl;
			return 13;
		}
	}

	// Still return 0 for success, and non-zero for failure/abort as in stdlib
	return mode;

}

bool is_under_pipeline_control()
{
	return (pipeline_control_outputname != "");
}

bool pipeline_control_check_abort_job()
{
	if (pipeline_control_outputname == "")
    	return false;

	struct stat buffer;
	if (stat((pipeline_control_outputname+RELION_JOB_ABORT_NOW).c_str(), &buffer) == 0)
	{
		return true;
	}
	else
	{
		return false;
	}

}

void pipeline_control_delete_exit_files()
{
	struct stat buffer;
	if (stat((pipeline_control_outputname+RELION_JOB_EXIT_SUCCESS).c_str(), &buffer) == 0)
	{
		remove((pipeline_control_outputname+RELION_JOB_EXIT_SUCCESS).c_str());
	}

	if (stat((pipeline_control_outputname+RELION_JOB_EXIT_FAILURE).c_str(), &buffer) == 0)
	{
		remove((pipeline_control_outputname+RELION_JOB_EXIT_FAILURE).c_str());
	}

	if (stat((pipeline_control_outputname+RELION_JOB_EXIT_ABORTED).c_str(), &buffer) == 0)
	{
		remove((pipeline_control_outputname+RELION_JOB_EXIT_ABORTED).c_str());
	}

	if (pipeline_control_task_count > 1)
		pipeline_control_delete_task_exit_files(pipeline_control_outputname, pipeline_control_task_count);
}

void pipeline_control_delete_task_exit_files(const std::string &outputname, int task_count)
{
	for (int task = 1; task <= task_count; task++)
	{
		std::remove((outputname + pipeline_control_task_marker(RELION_JOB_EXIT_SUCCESS, task)).c_str());
		std::remove((outputname + pipeline_control_task_marker(RELION_JOB_EXIT_FAILURE, task)).c_str());
		std::remove((outputname + pipeline_control_task_marker(RELION_JOB_EXIT_ABORTED, task)).c_str());
	}
}
