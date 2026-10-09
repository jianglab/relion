# GPU performance: getting more out of one GPU

## Summary

For jobs where each particle gives the GPU little work — helical Class2D with
orientation priors is the clearest case — a single `relion_refine` process
cannot keep a GPU busy, however many threads it is given. Running several MPI
processes on the same GPU **under NVIDIA MPS** fixes that. On a realistic
helical Class2D job it was 4.2× faster, with equivalent results:

| Layout (one RTX 2080 Ti, 8 cores) | Time | GPU busy | Cores busy |
|---|---|---|---|
| 1 process × 4 threads (`--j 4`) | 783 s | 54% | 2.5 of 8 |
| 8 MPI processes × 1 thread, **with MPS** | 188 s | 94% | 7.3 of 8 |

(EMPIAR-10944 subset: 37,022 segments, 50 classes, 5 iterations, box 128 at
4.944 Å/pixel, `--helix --bimodal_psi --sigma_psi 2`.)

`relion_refine_mpi` now does this by itself: when two or more GPU workers share
a node, it starts a private MPS daemon for the job and stops it at the end (see
"Automatic MPS" below). At the end of every run it also prints how busy the GPU
and the cores were, and what to change (see "The resource report").

## Why one process cannot fill the GPU

With orientation priors each segment is compared with only a few thousand
orientation/translation combinations. The GPU finishes that in microseconds,
and the time goes on the fixed cost per particle: hundreds of small kernel
launches, memory copies and synchronisations, each a round trip between the
CPU thread and the GPU. A thread spends most of its time waiting for its own
GPU work to come back.

The obvious remedies do not work:

- **More threads in one process** (`--j 8` instead of `--j 4`) made it
  *slower*: about 40% longer before the allocator changes below, still about
  20% longer after them (31 s instead of 26 s). The threads contend inside the process.
- **More MPI processes on one GPU** without MPS also made it slower: GPU work
  from different processes is time-sliced, so the processes take turns rather
  than overlapping.

NVIDIA MPS (Multi-Process Service) lets the kernels of several processes run
on one GPU *at the same time*. Each process still waits for its own work, but
the GPU now has eight streams of work to interleave.

Measurements on a smaller test (4,590 segments, 2 iterations, same GPU):

| MPI processes × threads | without MPS | with MPS |
|---|---|---|
| 1 × 4 (no MPI) | 44 s | – |
| 2 workers × 4 | 70 s | 34 s |
| 4 workers × 2 | 77 s | 21 s |
| 8 workers × 1 | – | 16 s |
| 12 workers × 1 | – | 15 s |

## Automatic MPS

`relion_refine_mpi` starts MPS itself, right after MPI starts and before any
process opens the GPU (processes join MPS only when their CUDA context is
created), when all of these hold:

- `--gpu` is given and two or more worker processes run on the node;
- no MPS is running already (neither `CUDA_MPS_PIPE_DIRECTORY` is set nor the
  system daemon's `/tmp/nvidia-mps` exists);
- `nvidia-cuda-mps-control` is available.

One process per node starts the daemon in a private directory
(`/tmp/relion_mps_<Slurm job>_<pid>`; short, because MPS uses Unix sockets) and
all processes of that node point `CUDA_MPS_PIPE_DIRECTORY` at it. The daemon is
told to quit when the run ends; it does so once the last process has
disconnected, and its directory is removed. The output says what was done:

```
 Started NVIDIA MPS so that the 8 GPU workers on this node run on the GPU at the same time (RELION_AUTO_MPS=off to disable).
```

`RELION_AUTO_MPS=off` disables it. If the daemon cannot be started, RELION
says so and runs without MPS. Same run as above, 8 workers × 1 thread on one
GPU: 16 s with the automatic MPS, 117 s with `RELION_AUTO_MPS=off`; nothing
was left running or on disk afterwards.

So the only thing a user has to choose is the number of MPI processes: about
one worker per core of the job, each with `--j 1`, works well.

## Starting MPS by hand

Only needed for programs other than `relion_refine_mpi`, or to share one
daemon between several runs. MPS needs no administrator rights. Start the control daemon inside the GPU job,
before `mpirun`, and stop it at the end. Put its pipe and log directories in a
private temporary directory so that jobs on the same node do not share one:

```bash
export CUDA_MPS_PIPE_DIRECTORY=$TMPDIR/mps
export CUDA_MPS_LOG_DIRECTORY=$TMPDIR/mps
mkdir -p $TMPDIR/mps
nvidia-cuda-mps-control -d                 # start

mpirun -np 9 relion_refine_mpi ... --gpu --j 1

echo quit | nvidia-cuda-mps-control        # stop
```

A Slurm job for one GPU and 8 cores (adapt partition, memory and time to your
cluster):

```bash
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
export CUDA_MPS_PIPE_DIRECTORY=$TMPDIR/mps CUDA_MPS_LOG_DIRECTORY=$TMPDIR/mps
mkdir -p $TMPDIR/mps
nvidia-cuda-mps-control -d
mpirun -np 9 relion_refine_mpi --o Class2D/job042/run ... --gpu --j 1
echo quit | nvidia-cuda-mps-control
```

Points to keep in mind:

- **Count:** one MPI process is the leader and does almost no work, so
  `-np 9` gives 8 workers. About one worker per core works well; going beyond
  the number of cores gained little in the tests above.
- **GPU memory:** every worker keeps its own copy of the references on the GPU.
  Small 2D jobs are fine; large 3D boxes may run out of GPU memory with many
  workers - use fewer workers then.
- **Several GPUs:** start one MPS daemon per job as above; it serves all GPUs
  visible to the job.
- **Results** are equivalent to an ordinary MPI run: two MPI runs of RELION
  never assign quite the same particles to the same classes (here about 4%
  differ), and runs with and without MPS differ by the same amount.

## The resource report

At the end of every `relion_refine` / `relion_refine_mpi` run the leader prints
a short report, for example:

```
 Resource use over 5 E-steps (770 s):
  1 worker process, GPU busy 54%, on average 2.5 of 8 cores busy
  - The GPU was busy only 54% of the time: the work per particle is small and
    the GPU waits for the CPU side. Run several MPI processes per GPU (about 7,
    each with --j 1) under NVIDIA MPS, ...
```

- **Measured over all E-steps**, per host: wall time, CPU time, the cores the
  job may use (`sched_getaffinity`, so Slurm's allocation), and the GPU busy
  fraction (NVML, loaded at run time; without it there are no GPU numbers).
- **GPU busy below 60%:** run MPI processes under MPS (with the commands), or
  more of them if MPS is already running (it detects a running daemon).
- **Several processes on one GPU without MPS:** recommend MPS whatever the busy
  figure says - processes that take turns keep the GPU "busy" switching
  between them (98% busy, yet 7x slower than with MPS, in the test above).
- **GPU busy above 90%:** more CPU processes or threads will not help.
- **Allocated cores mostly idle:** request fewer cores (it suggests how many).
- `RELION_RESOURCE_REPORT=off` silences it.

"GPU busy" is the fraction of time at least one kernel was running. It says
whether the GPU is fed, not how efficiently its kernels use it.

## Other changes made while investigating

All verified against upstream RELION to leave results unchanged, except the
two bug fixes (centring and prefetcher), which change them as intended:

- **`--center_classes` in EM runs** (not `--grad`): `do_grad_next_iter` was
  never initialised and decides whether classes are centred each iteration, so
  centring happened or not depending on a leftover byte of memory. It now
  always happens. With centring really on, GPU runs are more sensitive to the
  order of floating-point sums (GPU atomic additions): repeated runs can end up
  with noticeably different class assignments after a few iterations. The CPU
  code is unaffected.
- **GPU allocator:** freeing ready blocks in one pass instead of restarting the
  list after every block, and recycling the CUDA events instead of creating and
  destroying one per freed buffer (a few hundred driver calls per particle).
- **Image prefetcher** (fixed separately in JiangLab): the background reader
  could hand a pool of particles the previous pool's images when the GPU code
  was fast. With the faster allocator this happened often enough to change
  results, which at first looked like a GPU race.
- **Helical mask:** no heap allocation per pixel (`softMaskOutsideMapForHelix`);
  bitwise identical output.

Together these made one process 1.6–1.8× faster on the GPU above (1 × 4
threads 46 s → 26 s, 1 × 8 threads 49 s → 31 s, small test); MPS is the
larger gain.

## Class batching (one GPU call for all classes)

For each particle the GPU code used to work class by class: build an
orientation plan, launch the difference kernel, convert the weights, average
and back-project, each with its own allocations, copies and launches. With
small searches those driver calls cost more than the GPU work, and threads in
one process queue for the driver's lock. Now (CUDA) each of these steps is one
call for all classes:

- coarse pass: the plans of all classes are built together
  (`AccProjectorPlan::setupBatch`) and the difference kernel covers all
  classes in one launch;
- fine pass: the weights of all classes are converted in one go;
- weighted sums: one weighted-average and one back-projection launch for all
  classes (back-projection stays per class with `--grad` and in SOM
  iterations).

Each GPU block looks up its class in a small table (a few dozen bytes per
class), so batching needs no extra GPU memory as the number of classes grows;
the large per-particle arrays are the ones upstream already allocates for all
classes. A block computes exactly what it computed in a per-class launch.
`RELION_GPU_CLASS_BATCH=off` restores the per-class calls.

| Same test (4,590 segments, 50 classes) | per class | batched |
|---|---|---|
| 1 process × 4 threads | 26 s | 18 s |
| 1 process × 8 threads | 28 s | 18.5 s |
| 8 MPI workers × 1 thread, MPS | 16 s | 13.5 s |

Results are the same as upstream RELION: every particle in the same class in
helical Class2D, and Class3D (tutorial data) as close to upstream as two
upstream runs are to each other.

One process still leaves the GPU idle about half the time: its threads share
the CUDA driver's per-process lock and wait on many small round trips per
particle. Several MPI processes under MPS (started automatically) avoid that.

## What limits it now, and what would help next

With MPS the GPU is busy about 94% of the time, and the time is inside two
kernels: back-projection (about 37% of GPU time) and weighted averaging (23%).
Cheaper trigonometry in them gained 1–2%, so they are not limited by
arithmetic but probably by memory traffic (atomic additions in particular).
Improving them needs the GPU performance counters (`ncu`), which the NVIDIA
driver restricts to administrators by default
(`NVreg_RestrictProfilingToAdminUsers`, shown as `RmProfilingAdminOnly` in
`/proc/driver/nvidia/params`).

Upstream RELION 5.1 (`05cd1ad9`) aborts on this job with an illegal GPU memory
access; that is the exact-size allocator reuse bug fixed in JiangLab by
`f7365189`.
