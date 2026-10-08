/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/mps_auto.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>

namespace mpsauto
{

namespace
{

std::string g_dir;       // pipe/log directory of the daemon this rank started
bool g_started = false;  // this rank started it (and stops it at exit)

bool exists(const std::string &p)
{
	return access(p.c_str(), F_OK) == 0;
}

// Stop in the background: "quit" waits until all clients have disconnected,
// which includes this process, so waiting for it here would never return
void stopAtExit()
{
	if (!g_started)
		return;
	const std::string cmd = "( CUDA_MPS_PIPE_DIRECTORY='" + g_dir + "' CUDA_MPS_LOG_DIRECTORY='" + g_dir +
	                        "' sh -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1; rm -rf '" + g_dir + "' ) &";
	if (system(cmd.c_str())) {}
}

}

std::string start(MPI_Comm world, bool uses_gpu, int world_rank)
{
	MPI_Comm local;
	MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &local);
	int local_rank;
	MPI_Comm_rank(local, &local_rank);

	// GPU workers on this node: every rank except the leader (world rank 0)
	int is_worker = (world_rank != 0) ? 1 : 0, workers = 0;
	MPI_Allreduce(&is_worker, &workers, 1, MPI_INT, MPI_SUM, local);

	// The node's first rank decides and starts the daemon; 0 = not started
	int ok = 0;
	char dir[256] = "";
	std::string note;
	if (local_rank == 0)
	{
		const char *env = getenv("RELION_AUTO_MPS");
		const bool disabled = env && (std::string(env) == "off" || std::string(env) == "0");
		const char *user_dir = getenv("CUDA_MPS_PIPE_DIRECTORY");
		const bool running = (user_dir && *user_dir) || exists("/tmp/nvidia-mps/control");
		if (!disabled && uses_gpu && workers >= 2 && !running)
		{
			if (system("command -v nvidia-cuda-mps-control >/dev/null 2>&1") != 0)
				note = "NVIDIA MPS would let the " + std::to_string(workers) + " GPU workers on this node run at the same time, but nvidia-cuda-mps-control was not found.";
			else
			{
				// Short path: MPS uses Unix sockets, whose names are limited to ~100 characters
				const char *job = getenv("SLURM_JOB_ID");
				std::string d = "/tmp/relion_mps_" + std::string(job ? job : "") + (job ? "_" : "") + std::to_string(getpid());
				mkdir(d.c_str(), 0700);
				setenv("CUDA_MPS_PIPE_DIRECTORY", d.c_str(), 1);
				setenv("CUDA_MPS_LOG_DIRECTORY", d.c_str(), 1);
				if (system("nvidia-cuda-mps-control -d >/dev/null 2>&1") == 0)
				{
					for (int i = 0; i < 50 && !exists(d + "/control"); i++)
						std::this_thread::sleep_for(std::chrono::milliseconds(100));
				}
				if (exists(d + "/control"))
				{
					ok = 1;
					g_dir = d;
					g_started = true;
					std::atexit(stopAtExit);
					strncpy(dir, d.c_str(), sizeof(dir) - 1);
					note = "Started NVIDIA MPS so that the " + std::to_string(workers) +
					       " GPU workers on this node run on the GPU at the same time (RELION_AUTO_MPS=off to disable).";
				}
				else
				{
					unsetenv("CUDA_MPS_PIPE_DIRECTORY");
					unsetenv("CUDA_MPS_LOG_DIRECTORY");
					rmdir(d.c_str());
					note = "Could not start NVIDIA MPS; the GPU workers on this node take turns on the GPU instead.";
				}
			}
		}
	}
	MPI_Bcast(&ok, 1, MPI_INT, 0, local);
	MPI_Bcast(dir, sizeof(dir), MPI_CHAR, 0, local);
	if (ok && local_rank != 0)
	{
		setenv("CUDA_MPS_PIPE_DIRECTORY", dir, 1);
		setenv("CUDA_MPS_LOG_DIRECTORY", dir, 1);
	}
	MPI_Comm_free(&local);
	return note;
}

}
