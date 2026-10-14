# GPU performance of relion_refine

## Summary

On a realistic helical Class2D job relion_refine is now 17–20 times faster
than upstream RELION, with the same results:

| One RTX 2080 Ti, 8 cores | upstream RELION | this branch |
|---|---|---|
| 1 process × 4 threads (`--j 4`) | 1577 s | 90 s |
| 1 process × 8 threads (`--j 8`) | 1471 s | 73 s |
| 8 MPI workers × 1 thread, MPS | – | 76 s |

(EMPIAR-10944 subset: 37,022 segments from 1,000 micrographs, 50 classes,
5 iterations, box 128 at 4.944 Å/pixel, `--helix --bimodal_psi --sigma_psi 2`;
upstream is `05cd1ad9` with the exact-size allocator fix `f7365189`, without
which it aborts on this job.)

After 5 iterations 99.95–99.97% of the segments are in the same class as with
upstream at the same number of threads (4 or 8), and the class averages
correlate at 1.0000; two upstream runs with 4 and with 8 threads agree on
99.93%. MPI runs differ more (92%): splitting the particles over processes
changes the order of floating-point sums, and after 2 iterations two MPI runs
already differ by about 4%.

**What to run:** one process with a thread per core now uses one GPU about as
well as several processes under MPS. MPS still helps when one process cannot
use the cores (see "Automatic MPS"). Without `--j`, `relion_refine` already
runs a thread per core it may use (see "Number of threads"), and the GUI
suggests numbers of processes and threads for the GPUs a job asks for (see
"Suggested numbers in the GUI"). At the end of every run RELION reports how
busy the GPU and the cores were and what to change (see "The resource
report").

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
4. **Image reading.** Only the last stack was kept open; particles come in
   random order, so a stack was reopened for nearly every image. Stacks now
   stay open across pools, in the prefetcher and in MPI workers (which read
   without it; with 2–4 threads per worker this made them 2–3 times faster),
   and the open-file limit is raised as needed.
5. **Round trips per particle.** Single values were read back one at a time,
   each with a wait for the GPU, and transforms waited between steps on the
   same stream. The significance of each pass is now computed on the GPU and
   read back at once; the helical mask runs on the GPU instead of a copy to
   the host and back; needless waits are gone.
6. **Copies to the host.** A copy to pageable memory goes through the
   driver's own staging buffers behind a lock held until it completes, so the
   threads' copies queued for it; they now go through a pinned buffer per
   thread.
7. **The legacy default stream.** The power spectrum buffer and the soft-mask
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
| `RELION_CPUS_PER_GPU=n` | CPU cores per GPU assumed by the GUI's suggestions |
| `RELION_AUTO_PARALLEL=No` | GUI default: give MPI processes and threads by hand |
| `RELION_SLURM_PREAMBLE` | site lines for RELION's own Slurm script |

Opt-in (off by default):

| Variable | Effect |
|---|---|
| `RELION_GPU_SKIP_ALIGN=1` | keep the GPU for `--skip_align` / `--skip_rotate` (Class3D without alignment); upstream switches the GPU off for these jobs |

The GPU code for `--skip_align` had several defects that never showed because upstream
never runs it: the job arrays of the fine pass were one element too short with a single
orientation and translation (a heap overflow), the translation of a particle was looked up
from 0 instead of from its own stored offset (`itrans_min`, its place in the pool), the sum
of the weights, the maximum, the best orientation and the count of significant samples were
not those of the fine pass, and the best orientation was overwritten by that of the first
particle of the pool. All are fixed, and the directions are not counted, as on the CPU.

## Number of threads

Without `--j`, `relion_refine` and `relion_refine_mpi` run a thread per CPU
core the process may use (`sched_getaffinity`, so what Slurm or `taskset`
allows), and say so:

```
 Using 8 threads: --j was not given, and this process may use 8 CPU cores.
```

MPI worker processes on one node that may use the same cores share them (with
`srun --cpus-per-task` each has its own cores; with `mpirun --bind-to none` they
share all of them). All processes get the same number - the smallest over the
workers - because the mapping of threads to GPUs assumes one number for all.
`--j n` sets the number as before; the GUI always passes it.

## Number of CPUs in the GUI

For Class2D, Class3D, InitialModel, Refine3D and MultiBody the Running tab
asks "Choose MPI and threads automatically?" (default Yes;
`RELION_AUTO_PARALLEL=No` changes the default). With Yes the user gives only
"Number of CPUs", and RELION splits it into MPI processes and threads when the
job is run, using the number of GPUs (below); the greyed-out MPI and thread
fields show the split. With No the user sets both numbers, as before. Jobs
saved before this option existed open with No and keep their numbers. Submit
templates are unchanged: they still receive `XXXmpinodesXXX` and
`XXXthreadsXXX`.

Resting the mouse on any of the three fields explains the numbers. The split,
for n GPUs (0 for a CPU-only job) and C CPUs:

| Job | MPI processes | Threads per process |
|---|---|---|
| Class2D (EM), Class3D, 1 GPU | 1 | C |
| Class2D (EM), Class3D, n GPUs | n + 1 (leader + a worker per GPU) | C / n |
| InitialModel, Class2D with VDAM | 1 (VDAM does not support MPI) | C |
| Refine3D, MultiBody | odd, at least 3: leader + workers in two equal halves | C / workers |
| CPU-only Class2D (EM), Class3D, Refine3D, MultiBody | a worker per 8 CPUs (+ leader) | C / workers |

The CPU-only split (8 threads per worker) has not been measured. For
Class2D replicas, the GPUs are those of one replica.

The suggested number of CPUs is n × the CPU cores per GPU, at most 8 per GPU:
on an RTX 2080 Ti the E-step time is the same with 6 to 16 threads per GPU,
because the threads of a process issue their GPU commands one at a time, and a
job that takes fewer cores leaves the rest of a shared node to other jobs.
With "No", the MPI and thread fields suggest numbers the same way.

- **Number of GPUs:** the queue option whose label mentions GPUs (for
  example a `RELION_QSUB_EXTRA3` labelled "Number of GPUs"), else the GPUs
  listed in "Which GPUs to use", else, for jobs run on this computer, its
  GPUs.
- **CPU cores per GPU:** `RELION_CPUS_PER_GPU` if set; otherwise at most 8,
  and no more than the nodes have: for queue jobs the nodes of the queue as
  Slurm describes them (`sinfo`; the smallest ratio of cores to GPUs, since
  the job may land on any of them); for jobs run on this computer, its cores
  and GPUs.

Suggestions are advice only: the fields keep what the user enters.

## Slurm without a template

When "Submission script template (optional)" on the Running tab is empty - the default
when `RELION_QSUB_TEMPLATE` is not set - RELION writes the Slurm script
itself (`src/slurm_script.{h,cpp}`) and submits it with `sbatch`. The site
needs no template, and the Running tab then has "Number of GPUs", "Memory
(GB)", "Walltime" and "Additional sbatch arguments" in place of the template's
extra fields. A job given a template works as before.

- **Each MPI process is a Slurm task** with a core per thread (`--ntasks`,
  `--cpus-per-task`), its cores on one node.
- **CPU-only programs whose processes work on their own** (MotionCorr,
  CtfFind, Extract, AutoPick, Polish, ...): Slurm places the tasks on one node
  or several, wherever it finds room, and memory is given per core.
- **GPU jobs, and refinements** (Class2D, Class3D, InitialModel, Refine3D,
  MultiBody) **that fit on one node: one node.** The leader of a refinement
  only coordinates: it is started on top of the workers (`mpirun
  --oversubscribe --bind-to none`), so no cores are reserved for it. A CPU
  refinement too large for any node is spread like the programs above, its
  leader with a task of its own.
- **GPUs** (`--gpus`) only for jobs that, as set up, run on them, whatever
  "Number of GPUs" says: refinements and AutoPick with "Use GPU acceleration?"
  (not the Laplacian-of-Gaussian picker), MotionCorr with MotionCor2 (not
  RELION's own), tilt-series alignment with AreTomo2, tomogram denoising,
  DynaMight and ModelAngelo. Resting the mouse on the field says which.
- **Memory** is always given, since without it some sites hand a job a whole
  node's memory. `auto` (the default) is, on one node, the job's share of the
  memory of the smallest node of the queue that fits it, by its share of that
  node's cores or of its GPUs, whichever is larger; on several nodes, the
  memory per core of the node type with the least (from `sinfo`; 4 GB per
  core when Slurm's node list is not available). A number is the GB for the
  whole job. Resting the mouse on the field shows the amount and the layout.
- **A job the queue cannot hold is refused** with a message (a GPU job no
  node has room for, or more cores than the queue has), instead of waiting in
  the queue for ever.
- `RELION_SLURM_PREAMBLE`: lines for the script before the command (module
  loads, environment), or the name of a file that holds them.
  `RELION_MPIRUN` names the MPI launcher (default `mpirun`).

The script is saved as `run_submit.script` in the job directory, as with a
template.

## Cache directory missing on a node

A cache directory (`--cache_dir`, `RELION_CACHE_DIRECTORY`) that does not
exist on a node and cannot be made there, or cannot be written, no longer
stops the job: that node's processes say so and read the particles where they
are. Only a continued run whose particle names already point into the cache
still stops, since those particles exist only there.


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
difference, averaging and back-projection kernels - running during about 43%
of the E-step; the rest is smaller kernels (sorting, reductions, transforms,
the mask), copies, and gaps. The E-step time is now the same with 6 to 16
threads, with pools of 30 to 300 particles, with waiting threads spinning or
blocking, with several processes under MPS, and with a 27% faster
back-projection kernel: what it follows is the number of CUDA driver calls,
about 230 per particle (launches, copies, waits, events), which threads of a
process make one at a time. The next large step is therefore fewer calls per
particle - several particles per GPU call, the way classes are batched now. Back-projection and averaging are probably limited by
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
