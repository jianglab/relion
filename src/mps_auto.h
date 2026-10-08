/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef MPS_AUTO_H_
#define MPS_AUTO_H_

/*
 * Automatic NVIDIA MPS for MPI runs that put several GPU workers on one node.
 *
 * Without MPS, GPU work from different processes is time-sliced; with it, the
 * kernels of all processes run on the GPU at the same time. For small-search
 * jobs (helical Class2D) that made 8 workers on one GPU 4x faster than one
 * process. Users cannot be expected to start it, and cannot know in advance
 * which node or GPU a job lands on, so relion_refine_mpi starts a private
 * daemon per node when it helps and nothing is in the way:
 *   --gpu is given, two or more workers share the node, no MPS is running
 *   already (CUDA_MPS_PIPE_DIRECTORY set, or the default /tmp/nvidia-mps),
 *   and nvidia-cuda-mps-control is available.
 * It must run before any process creates a CUDA context, since processes
 * join MPS when their context is created. The daemon is stopped when the run
 * ends. RELION_AUTO_MPS=off disables it; any failure falls back to no MPS.
 */

#include <mpi.h>
#include <string>

namespace mpsauto
{

/// Decide, start the daemon on each node's first rank, and point every rank's
/// CUDA_MPS_* variables at it. Collective over `world`. Returns a one-line
/// note for the output ("" when nothing was done and there is nothing to say).
std::string start(MPI_Comm world, bool uses_gpu, int world_rank);

}

#endif
