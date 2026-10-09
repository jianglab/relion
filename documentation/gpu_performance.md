# GPU performance of relion_refine

## Summary

On a realistic helical Class2D job the GPU E-step is now 16–20 times faster
than upstream RELION, with the same results:

| One RTX 2080 Ti, 8 cores | upstream RELION | this branch |
|---|---|---|
| 1 process × 4 threads (`--j 4`) | 1577 s | 100 s |
| 1 process × 8 threads (`--j 8`) | – | 81 s |
| 8 MPI workers × 1 thread, MPS | – | 97 s |

(EMPIAR-10944 subset: 37,022 segments from 1,000 micrographs, 50 classes,
5 iterations, box 128 at 4.944 Å/pixel, `--helix --bimodal_psi --sigma_psi 2`;
upstream is `05cd1ad9` with the exact-size allocator fix `f7365189`, without
which it aborts on this job.)

After 5 iterations 99.99% (4 threads) and 99.92% (8 threads) of the segments
are in the same class as with upstream, and the class averages correlate at
1.0000. MPI runs differ more (92%): splitting the particles over processes
changes the order of floating-point sums, and after 2 iterations two MPI runs
already differ by about 4%.

**What to run:** one process with a thread per core (`--j` = the number of
cores of the job) now uses one GPU about as well as several processes under
MPS. MPS still helps when one process cannot use the cores (see "Automatic
MPS"). At the end of every run RELION reports how busy the GPU and the cores
were and what to change (see "The resource report").

## Where the time went

With orientation priors each segment is compared with only a few thousand
orientations and translations, so the GPU finishes its work in microseconds and
the time went on everything around it. Measured with stack samples, per-stage
timers and nsys timelines, in the order found:

1. **Image prefetcher (a bug, fixed in JiangLab).** The background reader could
   hand a pool of particles the previous pool's images once the GPU code was
   fast; it now reads each pool once and checks the range it hands over.
2. **GPU memory allocator.** One lock shared by all threads, held while walking
   every block and querying an event per freed block on every allocation (44%
   of the threads' time with 8 threads). Freed blocks now wait in a queue per
   stream behind one event per batch, polled outside the lock.
3. **One GPU call per class.** Plans, differences, weight conversion, averaging
   and back-projection were done class by class, each with allocations, copies
   and launches (about 1,150 driver calls per particle with 50 classes). Each
   step is now one call for all classes ("Class batching", below).
4. **Image reading.** The prefetcher kept only the last stack open; particles
   come in random order, so it reopened a stack for nearly every image. Stacks
   now stay open (the open-file limit is raised as needed).
5. **Round trips per particle.** Single values were read back one at a time,
   each with a wait for the GPU, and transforms waited between steps on the
   same stream. The significance of each pass is now computed on the GPU and
   read back at once; the helical mask runs on the GPU instead of a copy to
   the host and back; needless waits are gone.
6. **The legacy default stream.** The power spectrum buffer and the soft-mask
   sums were created on stream 0 (`ptrFactory.make(size, 0)`: the 0 is the
   stream). Every operation there waits for all threads' GPU work and holds
   up all later work: three barriers across the whole GPU per particle. They
   now use the thread's stream. This affects all GPU refinements, not only
   helical ones (upstream RELION has the same calls).

Smaller fixes on the way: `--center_classes` was applied or not depending on
an uninitialised flag in EM runs (fixed: it now always applies); the helical
mask no longer allocates per pixel.

## Class batching (one GPU call for all classes)

Each of these steps is one call for all classes of a particle (CUDA):

- coarse pass: the orientation plans of all classes are built together
  (`AccProjectorPlan::setupBatch`), and the difference kernel and the mapping
  of its results cover all classes in one launch each;
- fine pass: the difference kernel for all classes, and the weights of all
  classes converted in one go;
- weighted sums: one launch each for collecting, averaging and back-projecting
  (back-projection stays per class with `--grad` and in SOM iterations).

Each GPU block looks up its class in a small table (a few dozen bytes per
class), so batching needs no extra GPU memory as the number of classes grows;
the large per-particle arrays are the ones upstream already allocates for all
classes. A block computes exactly what it computed in a per-class launch.

## Switches

All on by default; for comparisons or if a problem is suspected:

| Variable | Effect |
|---|---|
| `RELION_GPU_CLASS_BATCH=off` | one GPU call per class, as upstream |
| `RELION_GPU_HELICAL_MASK=off` | helical mask on the CPU, as upstream |
| `RELION_AUTO_MPS=off` | `relion_refine_mpi` does not start MPS |
| `RELION_RESOURCE_REPORT=off` | no resource report at the end |

## Automatic MPS

Without MPS, GPU work from different processes is time-sliced: several MPI
processes on one GPU take turns rather than overlap. NVIDIA MPS lets their
kernels run at the same time. `relion_refine_mpi` starts MPS itself, right
after MPI starts and before any process opens the GPU (processes join MPS only
when their CUDA context is created), when all of these hold:

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

If the daemon cannot be started, RELION says so and runs without MPS. Every
worker keeps its own copy of the references on the GPU, so large 3D boxes with
many workers can run out of GPU memory: use fewer workers then.

To start MPS by hand (other programs, or one daemon for several runs), inside
the GPU job and before `mpirun`, with private pipe and log directories:

```bash
export CUDA_MPS_PIPE_DIRECTORY=$TMPDIR/mps CUDA_MPS_LOG_DIRECTORY=$TMPDIR/mps
mkdir -p $TMPDIR/mps
nvidia-cuda-mps-control -d                 # start
mpirun -np 9 relion_refine_mpi ... --gpu --j 1
echo quit | nvidia-cuda-mps-control        # stop
```

## The resource report

At the end of every `relion_refine` / `relion_refine_mpi` run the leader prints
a short report, for example:

```
 Resource use over 2 E-steps (32 s):
  1 worker process, GPU busy 80%, on average 6.2 of 8 cores busy
  CPUs and GPUs were well balanced.
```

- **Measured over all E-steps**, per host: wall time, CPU time, the cores the
  job may use (`sched_getaffinity`, so Slurm's allocation), and the GPU busy
  fraction (NVML, loaded at run time; without it there are no GPU numbers).
- **GPU busy below 60%:** with fewer threads than allocated cores, raise
  `--j` to a thread per core; with a thread per core already, run MPI
  processes under MPS (or more of them, if MPS is running).
- **Several processes on one GPU without MPS:** recommend MPS whatever the busy
  figure says - processes that take turns keep the GPU "busy" switching
  between them.
- **GPU busy above 90%:** more CPU processes or threads will not help.
- **Allocated cores mostly idle:** request fewer cores (it suggests how many).

"GPU busy" is the fraction of time at least one kernel was running. It says
whether the GPU is fed, not how efficiently its kernels use it.

## What limits it now, and what would help next

An nsys timeline of one iteration (8 threads) shows the real work - the
difference, averaging and back-projection kernels - running during 43% of
the E-step; the rest is smaller kernels (sorting, reductions, transforms, the
mask), copies, and gaps where every thread is busy on the CPU. With 6 to 12
threads the time is the same, and the GPU is busy about 80% of the time, so
the next gains are less GPU time per particle outside the real kernels, and
faster real kernels. Back-projection and averaging are probably limited by
memory traffic (atomic additions in particular); tuning them needs the GPU
performance counters (`ncu`), which the NVIDIA driver restricts to
administrators by default (`NVreg_RestrictProfilingToAdminUsers`, shown as
`RmProfilingAdminOnly` in `/proc/driver/nvidia/params`).

Two oddities in upstream RELION noticed on the way, left unchanged here:

- In SOM iterations the per-class back-projection loop skips a class without
  moving past its weights, so the classes after it appear to read the wrong
  weights.
- In the coarse pass the significant weight returned to the caller is never
  set (`my_significant_weight`); the fine pass overwrites it before it seems
  to be used.
