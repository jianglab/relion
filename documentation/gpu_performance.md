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
| `RELION_PARTICLE_ORDER=micrograph` | shuffle whole micrographs instead of particles (see below); off by default |
| `RELION_GPU_BP_LAYOUT=planar` | keep the real, imaginary and weight arrays of the 3D back-projection model apart, as upstream (default: interleaved, see "Interleaved back-projection model") |
| `RELION_GPU_BP_PACK=0` | back-projection atomics one voxel component per lane, as before (default: on for 2D images with the interleaved 3D model; the packed path is used per launch when it fits, otherwise the launch falls back and the reason is printed once; see "Packed back-projection atomics") |
| `RELION_GPU_COARSE_XT=0` | coarse pass by the direct kernel instead of the cross-term kernel (default: on where it applies, i.e. 3D references and 2D images with a table up to 128 MB, otherwise the direct kernel is used automatically; `check` runs both and reports the largest relative difference) |
| `RELION_GPU_BLOCKS_PER_SM=N` | thread blocks per multiprocessor that the pixel chunks of the coarse, weighted-average and back-projection kernels aim for (default: chosen from the device, one full wave of 128-thread blocks per multiprocessor, e.g. 12 on sm_86, 16 on sm_80/90, 8 on sm_75; 3 when the GPU has under 2 GiB free; the choice and its reason are printed once per device) |
| `RELION_GPU_RECONSTRUCT=1` | the M-step reconstructions (3D, skip_gridding) on the GPU whenever `--gpu` was given, also with `--skip_align`; `=0` turns it off. Default: on. Falls back to the CPU with a message when the GPU lacks memory or no GPU is visible |
| `RELION_GPU_FLOAT2_TEX=0` | two single-channel projection textures instead of one two-channel texture (default: on for 3D references, CUDA, single precision) |
| `RELION_GPU_SYMMETRISE=0` | helical Fourier-space symmetrisation on the CPU (default: on with `--gpu`; `check` runs both) |
| `RELION_JOIN_SKIP_BPREF=0` | send the summed back-projection arrays back to the second follower at the join as upstream does (default: skipped, 3 ranks, converged auto-refine only) |
| `RELION_FFTW_THREADS=1` | every FFT single-threaded, as upstream's non-MKL builds (see below) |
| `RELION_STARTUP_THREADS=N` | threads for the start-up work: reading and writing STAR tables, and the noise estimate from the first particles. (Loops over large volumes at start-up, such as zeroing the Projector arrays, use the FFT thread count, see "Multi-threaded FFTs".) Default: the cores the process may use (up to 8), and one thread per 8192 table rows; `0` = the original one-thread code. Results are identical for every value |
| `RELION_PREFETCH_THREADS=N` | most threads that read one pool of particle images ahead of the E-step. The prefetcher starts with one reader and doubles it (up to this limit, default `min(--j, 8)`) whenever a pool was not ready in time; `--no_prefetch` turns it off |
| `RELION_CPUS_PER_GPU=n` | CPU cores per GPU assumed by the GUI's suggestions |
| `RELION_AUTO_PARALLEL=No` | GUI default: give MPI processes and threads by hand |
| `RELION_SLURM_PREAMBLE` | site lines for RELION's own Slurm script |

On by default in CUDA builds (`=0` switches it off):

| Variable | Effect |
|---|---|
| `RELION_GPU_SKIP_ALIGN=0` | switch the GPU off for `--skip_align` / `--skip_rotate` (Class3D without alignment), as upstream does; the job window then refuses the GPU for such jobs. By default the GPU is kept (a HIP build keeps upstream's behaviour unless `=1`) |

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

## Multi-threaded FFTs outside the particle loop

The large transforms of each iteration (the references at the start of the E-step, the
reconstructions in the M-step) used to run on one core unless RELION was built with Intel MKL.
FFTW's own thread library does the same job, and it is now used whenever the build finds it
(CMake defines `RELION_HAVE_FFTW_THREADS`): these transforms use all `--j` threads, while the
transforms inside the particle loop stay single-threaded (every thread does its own), exactly as
in the MKL builds. A 768³ forward + inverse transform (box 384 at `--pad 2`) went from 24.9 s to
3.4 s on 8 cores of an A5000 node. `RELION_FFTW_THREADS=1` turns it off.

A trap that hid most of this gain at first: `FourierTransformer::cleanup()` calls
`fftw_cleanup()`, which replaces FFTW's planner, and a new planner plans on **one** thread (and has
forgotten all wisdom). RELION calls it at the end of every class reconstruction and during start-up,
so every transform after the first class ran on one core again. `cleanup()` now restores the thread
count. `RELION_FFTW_DEBUG=1` prints, for every transform of 2²¹ points or more, its size, the number
of threads it was planned with and how long it took. The normalisation after a forward transform
(a division of every coefficient) uses the same number of threads.

With a GPU, the reference transforms normally run on the GPU (cuFFT); if the GPU has too little
free memory for them (a 768³ double-precision transform needs about 16 GB), they fall back to the
CPU, and the output now says so once.

MKL versus FFTW, measured on our nodes (forward + inverse, double precision):

| | AMD nodes | Intel node |
|---|---|---|
| 2D particle images, 1 thread | MKL no faster | MKL 1.6–3× faster |
| 3D 256–768³, 8 threads, FFTW plans as RELION makes them (`FFTW_ESTIMATE`) | MKL 1.6–2.6× faster | MKL 1.6–1.8× faster |
| same, FFTW with measured plans (`FFTW_MEASURE`) | FFTW as fast or faster | MKL 20–40% faster |

## Reconstructions on the GPU (on by default, `RELION_GPU_RECONSTRUCT=0` turns it off)

With the E-step on the GPU, the M-step's reconstructions became the largest remaining cost of a
Class3D iteration: per class, a padded Fourier volume (768 x 768 x 385 complex numbers for a 384 box at
`--pad 2`, about 3.6 GB in double precision) goes through several passes and an inverse FFT on the CPU.
`RELION_GPU_RECONSTRUCT` (on unless `=0`) does `BackProjector::reconstruct()` for 3D maps with skip_gridding (the
default) on the GPU (`src/acc/cuda/cuda_reconstruct.cu`): the same steps, kernel by kernel, with the
inverse transform by cuFFT; the soft mask and the gridding correction of the 384³ result stay on the
CPU.

Precision follows RELION's GPU code, as upstream does it: CPU code in double (`DoublePrec_CPU=ON`),
GPU code in single (`DoublePrec_ACC=OFF`). So the GPU reconstruction is in single
precision (the double arrays are converted on their way to the GPU, 64 MB at a time); `=double`
forces double, `=single` single. Memory for a 384 box at padding 2: about 4.6 GB in single, 9 GB in
double. With less free GPU memory, or on any CUDA error, the reconstruction runs on the CPU and the
output says why, once.

Accuracy (`tests/unit/test_gpu_reconstruct.cpp`, on a GPU node): double equals the CPU to rounding
(7e-15 on values up to 14); single within 1e-5 relative RMS (cuFFT single precision alone: 2.4e-7).
Two cautions found on the way, both about invalid input that an inverse real FFT is free to treat in
its own way: the zero frequency must be real, and at full resolution the few coefficients on the
Nyquist planes of the padded box are not Hermitian partners after windowing. FFTW and cuFFT treat
these differently in single precision; in real data they are single coefficients at the highest
frequency, of no consequence. End to end (Class3D without alignment, 5000 tau segments, box 384,
2 iterations): single and double precision on the GPU put every particle in the same class, with map
correlations of 1.0000; against upstream 99.88% of particles keep their class (float noise of the
GPU E-step) at map correlation 1.0000.

## Particle order by micrograph (opt-in)

RELION shuffles the particles once at the start of a run (each gold-standard
half separately), so every iteration reads single images scattered over many
stack files. With `RELION_PARTICLE_ORDER=micrograph`, runs that visit every
particle (no random subsets) shuffle whole micrographs instead and keep each
micrograph's particles together in file order (by stack file when the data
have no micrograph names), so a pool of particles reads a few long stretches
of few files. The sums over particles do not depend on the order, apart from
rounding. Runs with random subsets (VDAM, `--subset`) keep the particle-level
shuffle, since a subset should be a random sample.

On a 70,000-segment Class3D without alignment (39 GB on a networked file
system, warm caches) it made no difference: 924 s against 919 s. It is kept
for large data sets of thousands of micrographs, where scattered reads may
cost more than file caches hide. Tests: `[particle_order]`.

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

## Writing output files in the background (`RELION_ASYNC_WRITE`)

`relion_refine` writes the class images, gradient moments and `_data.star` of
an iteration on one background thread, after formatting or copying them in
memory, so the next iteration no longer waits for a slow (network) file
system. Files are written in submission order, and the job waits for all of
them at the end of the run; `pruneOldIterations` only ever deletes files of
earlier iterations. In the Class2D benchmark on a network file system this
removes about 4 s of 46 s. `RELION_ASYNC_WRITE=0` writes synchronously.
`RELION_ITER_TIMING=1` prints a per-iteration breakdown (`ITER_TIMING` lines on
stderr) of where the host time goes outside the E-step.

## Per-orientation back-projection (`RELION_GPU_BP_ACCUM`, on by default)

For a global search with 3D references on identical volumes and no orientational
priors, every particle back-projects into the same (class, orientation) slots.
The GPU therefore adds each particle's contribution into one 2D buffer per slot
(three coalesced atomic additions per pixel) and scatters the buffers into the
volumes once per iteration, instead of 24 scattered atomic additions per pixel
for every particle. Anything not eligible (priors, helical, body, gradient, SOM,
2D, tomography, too little memory for the buffers) uses the direct path;
`RELION_GPU_BP_ACCUM=0` forces it. The job prints which path it took.

B3 (Class3D, 4598 particles, K=4, healpix order 2, 25 iterations, one A5000,
8 cores): wall time 97 s -> 84 s; the iteration-2 E-step 25.9 s -> 15.5 s.
Agreement with upstream, against upstream's own run-to-run spread (two full
upstream runs, same seed):

| comparison | same class | map correlation, main class |
|---|---|---|
| upstream vs upstream | 0.9826 | 0.9873 |
| upstream vs default GPU | 0.9843 | 0.9875 |
| upstream vs accumulator | 0.9828 | 0.9866 |
| default GPU vs accumulator | 0.9804 | 0.9865 |

Log-likelihood at iteration 25: upstream 4.980915e7 and 4.980963e7, default
4.981028e7, accumulator 4.980940e7, so the accumulator lies inside upstream's
spread (about 1e-5 relative). At iteration 3 the accumulator differs from the
upstream value by 1.6e-5 (relative). The sums over particles are made in
separate buffers before they reach the volume, which loses less to float
rounding than adding straight into a voxel that already holds a large sum.

## Packed back-projection atomics (`RELION_GPU_BP_PACK`, on by default)

The back-projection adds 24 floats per pixel into the 3D model (two neighbouring voxels in x, four
(y, z) neighbours, real, imaginary and weight). With one voxel component per lane the atomic
requests of a warp touch many sectors, and in local search (where the accumulator path of
`RELION_GPU_BP_ACCUM` does not apply, about 1.1 visits per orientation) the kernel was bound by
atomic request issue. The packed path lets a lane take one float of a run of one pixel: the warp
compacts its active pixels with a ballot, stages values and addresses in shared memory (two of the
four runs at a time, about 8 KB per block) and issues dense atomic additions. The terms are the
same, only the order of the additions changes, so results differ from the default only by the
noise of atomic float additions (the GPU back-projection is never bit-reproducible).

Measured (helical auto-refine B5, 3 MPI ranks, one GPU): the back-projection kernel is 3.3x shorter
(ncu, same launches) with 3.4x fewer L2 reduction sectors; local-search iterations are about 15%
faster on an RTX 3090, and the full run on an RTX A5000 takes 862 s against 1005 s with the same
final resolution (4.16337 A). Helical Class2D E-step -23%, Class3D B3 -19%.

Automatic fallbacks, decided per launch and printed once per reason as
`RELION_GPU_BP_PACK: ...`: 3D data, planar model layout (`RELION_GPU_BP_LAYOUT=planar`),
`RELION_GPU_BP_TILE` set, and per-orientation accumulator buffers in use (silent, that launch only).
`RELION_GPU_BP_PACK=0` turns the path off.
Integration test: `tests/integration/test_bp_pack.py`.

## Interleaved back-projection model (`RELION_GPU_BP_LAYOUT`, on by default for 3D)

Back-projection adds each pixel of a particle into the 8 neighbouring voxels of the
padded 3D model, as three separate float atomic additions (real part, imaginary
part, weight). The model is three arrays of about 0.9 GB each in the helical
benchmark, and the L2 cache is 6 MB, so almost every atomic addition misses the
cache, and in the planar layout the three additions of one voxel hit three
different cache lines. With the three values of a voxel next to each other in one
allocation (`real`, `imag`, `weight` at stride 3), they share a line. The kernels
find the layout from the two pointers they are given, so nothing else changes.

Measured on the helical auto-refine benchmark (20k particles, box 384, one A5000,
two ranks): back-projection per particle 8.8 -> 6.0 ms and 20.6 -> 8.8 ms of GPU time
on the two ranks; a continued iteration 307-324 s -> 196-211 s. Maps differ only by
the order of the float atomic additions (relative difference 4e-8 in the half maps).
2D models are always planar. Not used with HIP, SYCL or the CPU build.

Before the change, skipping back-projection altogether made an iteration 2.8x
faster, and removing the atomic additions 3.2x, which is how the atomics were found to
be the limit. The earlier idea that the run was bound by the host was wrong: host threads
spin in `cudaStreamSynchronize` while the GPU works.

## Cross-term coarse kernel (`RELION_GPU_COARSE_XT`, on by default)

The coarse pass sums, per orientation and translation, the squared difference between
the shifted image and the CTF-modified reference projection. Expanding the square
gives a term that does not depend on the translation, so the translations are
evaluated as a cross-term. On the helical benchmark the kernel takes 2.6 ms per
particle instead of 3.4-10 ms. The largest relative difference of the diff2 values
against the direct kernel is 3-4e-7 (`RELION_GPU_COARSE_XT=check` measures it). The
final resolution of the full run is identical to upstream (4.16337 A).

`RELION_GPU_BLOCKS_PER_SM` sets how many blocks per multiprocessor the pixel chunks
aim for. For the coarse kernel, 3/6/12/24/48 blocks gave 10.2/7.6/4.2/2.4/1.6 ms per
call, but the throughput of the whole iteration changed by 2% between 12 and 48,
because back-projection and the fine pass then set the pace.

The default is chosen from the device (threads per multiprocessor / 128, so 12 on sm_86), with the
old 3 when the GPU has under 2 GiB free. Refining it from timings during the first iterations (try a few
values, keep the fastest) is future work; the 12 was measured on sm_86 GPUs only.

## Full B5 run, and things that did not work

Helical auto-refine, 20k particles (RELION_GPU_COARSE_XT=1, RELION_GPU_BLOCKS_PER_SM=12): wall
time 1592 s against about 5000 s for upstream (3.1x), same final resolution. Iterations 1-13
take about 53 s, 14-19 about 100 s, and the last (all data to Nyquist, both halves joined)
292 s, of which the E-step is 192 s. Continued runs started from a finished iteration are
not a fair test bed: they restart with finer sampling and cost about twice as much per
iteration as the same iteration of a full run.

Tried and dropped:

- Caching the reference projections: with orientational priors every particle uses its own set of
  orientations (3.6 million distinct ones per rank, about 650 GB of projections).
- MPS beyond the automatic one, and `cudaStreamSynchronize` modes: no gain.
- More blocks per multiprocessor for the coarse kernel beyond 12: see above.

## Later full runs, stage timers, and what the coarse kernel is bound by

Full helical runs on one A5000 (leader plus two followers under MPS), with the interleaved
model and the cross-term kernel:

| Settings | Wall | Final resolution |
|---|---|---|
| base (above) | 1592 s (3.1x) | 4.16337 A |
| + `RELION_GPU_RECONSTRUCT=1`, `RELION_FFTW_DEBUG=1` | 1453 s (3.4x) | 4.16337 A |
| + `RELION_ITER_TIMING=1` | 1398 s (3.6x) | 4.21888 A (next shell) |
| + `RELION_GPU_FLOAT2_TEX=1` | 1231 s (4.1x) | 4.16337 A |

The last value is the adjacent Fourier shell: the gold-standard FSC curves of upstream and
these runs agree within about 0.005, and at the 4.16 A shell they are 0.147, 0.145 and
0.141 around the 0.143 threshold. This is the resolution noise of the threshold crossing,
not a systematic loss.

`RELION_ITER_TIMING=1` also prints per-stage `MPI_TIMING` lines on the leader. In the last
run the work outside the E-step was about 73 s for the half-map alignment, 69 s for
maximisation, 32 s for symmetrisation and 7-8 s for solvent flattening; the E-step was about
1190 s.

What the cross-term coarse kernel is bound by (measured with `RELION_XT_ABLATE`, a
timing-only switch that removes one part of the kernel; the results are garbage and only the
kernel time is meaningful, so compare runs at similar GPU load):

- Without the projection the kernel falls from 3.6 ms to 0.2 ms per call; without the
  translation load to 1.5 ms; without the multiply-adds only to 3.3 ms. The kernel is limited
  by the texture fetches of the projection, not by arithmetic.
- A layout where a warp holds 32 consecutive pixels of one orientation was slower (4.9 ms).

`RELION_GPU_FLOAT2_TEX` (on by default; `=0` off; 3D, CUDA, single precision) stores real and imaginary
parts of the projector in one two-channel texture, so a projection needs one fetch instead of
two. On the helical benchmark the coarse kernel goes from 3.6 to 2.7 ms and the fine kernel
from 3.2 to 2.5 ms per call, with 11-12% more calls in the same time window. The classification
benchmark (box 64) does not gain (83 s against 85 s). Results are the same as without it: the
log-likelihoods of the first three iterations agree to all printed digits and the class maps
differ by less than 6e-5 relative, which is the noise of the floating-point atomics.

With the float2 texture the E-step of the full helical run fell from 1192 s to 1021 s and the
final resolution is again identical to upstream. Outside the E-step the leader spends about
105 s in the half-map alignment, 70 s in maximisation, 9 s in solvent flattening, 7 s comparing
halves and 6 s writing; reaching 5x (1000 s) needs the E-step near 800 s or these host stages
cut.

Half-map alignment on a reduced map. The two half maps are only compared at low resolution,
so the alignment search ran on a 64-cube copy already; but each rank also resized the full
384-cube map, and rank 2 sent the whole map to rank 1. The ranks now reduce their own half
map first and send only the small cube (`reduceMapForAlignment`, `alignWorkSize`), and the
search is run with `apply_result = false` so the full-size map is transformed once, afterwards,
by the caller where it is needed. The parameters found are exactly the same as with the
full-size inputs (unit test `[alignmap]`, "maps reduced by the caller give the same
parameters"). `RELION_ALIGN_TIMING=1` prints the time of each stage (`ALIGN_TIMING ...`).
The half-map alignment on the leader fell from 105 s to 60 s over a full run, and the full
helical run took 1146 s (4.36x) with the final resolution again identical to upstream
(4.16337). The E-step is still 991 s of that; the next big items are the final iteration's
host stages (symmetrisation 8.7 s, maximisation 35 s, alignment 11 s).

Unregularised reconstruction without the disc. In the final iteration of an auto-refine run the
data and weight arrays are saved before the regularised reconstruction (which changes them in
place) and read back for the unregularised one. For the helical benchmark that was about 2.7 GB
written and read per half, 11-13 s in the alignment stage and about 9 s more in maximisation.
The arrays are now kept in memory, as single-precision copies (what the files held), when the
host has at least four times the needed memory available; otherwise the disc path is used as
before. `RELION_UNREG_VIA_DISC=1` forces the disc path, and `RELION_UNREG_VIA_DISC=check` does
both and compares: all 688 million values of each half were identical. The final iteration's
alignment stage fell from 13.5 s to 2.6 s and maximisation from 37.9 s to 28.5 s, with the same
final resolution.

### Reading the next pool in the background on MPI followers

In MPI runs each follower read its pool of images serially before the threaded E-step started, and this was not prefetched (only the single-process path had a prefetcher). `RELION_ITER_TIMING=1` now prints, per follower and iteration, the time spent waiting for the leader, receiving job data, in the serial set-up and reading, and in the threaded part. On the helical auto-refine benchmark (2 followers, box 384) the serial part was 3-10 s per iteration, about 15% of the E-step.

Followers now start reading the pool after the current one while the current one is processed. The leader hands a follower consecutive pools of its half set, so the next range is a guess; a wrong guess is dropped and the pool is read the normal way, and after 3 misses with no hit the follower stops guessing (several followers sharing a half set). `RELION_MPI_PREFETCH=0` turns it off. The images are the same, so results are unchanged beyond the usual run-to-run noise. Serial part: 3-10 s to 0.1 s per iteration; whole run 1208 s to 1066 s. The misses left are the first pool of each iteration and the short last pool of the first half set (about 0.2 s per iteration).

### Joining the two half sets and symmetrising in the final iteration

The final iteration of auto-refine joins the weighted sums of the two half sets (about 5 GB in double precision) and symmetrises each half's reconstruction while the GPUs are idle. `RELION_ITER_TIMING=1` prints `join_lap`, `join_gather`, `join_scatter` and `SYM_TIMING` lines for these steps.

- **Pack and unpack** of the weighted sums (`MlWsumModel::pack` / `unpack`) copied element by element with a range test per element. The large arrays are now copied as blocks, on up to 4 threads (`RELION_PACK_THREADS` overrides). The join fell from about 25 s to 8.7 s; the final-iteration maximisation stage from 30.8 s to 15.1 s. The packed values are the same.
- **Helical symmetry in Fourier space** evaluated cos and sin for every voxel although the phase depends only on z. They are now computed once per z plane (same values), and the number of threads follows the cores available divided by the ranks that symmetrise at the same time (never fewer than `--j`; `RELION_SYMMETRISE_THREADS` overrides, and the choice is printed with `RELION_ITER_TIMING=1`). On the helical benchmark the step fell from 9.6 s to 5.1 s with 3 to 4 threads. A unit test (`[helix_symmetry]`) checks that the result is bit for bit that of the original serial loop, with and without rise, at 1 and 4 threads.

- **Skipping the send-back of the summed BPref arrays** (`RELION_JOIN_SKIP_BPREF`, on by default, `=0` off; 3 ranks, converged auto-refine only). After the join the leader scattered the summed weighted sums, including the large BPref arrays, back to the follower of the second half. That follower only needs the unregularised reconstruction of its half, which re-reads BPref from the stored copy, so the arrays are no longer sent. Scatter 3.5 s to 0; final resolution and log-likelihood identical; final maps equal at float-atomic noise level (correlation of the full map 0.999997, half maps 0.9999 or better).
- **Helical symmetry in Fourier space on the GPU** (`RELION_GPU_SYMMETRISE`, on by default with `--gpu`, `=0` off; `=check` runs both and compares). One thread per output voxel, operators summed in the same order as the CPU, in the model's precision. Planes are processed in groups of mirrored |z| values, sized from the free GPU memory (halved on allocation failure; `RELION_GPU_SYMMETRISE_PLANES` overrides for experts and tests). If the GPU cannot be used (no CUDA, rise/rotation layout not supported, no memory) the CPU code runs and the reason is printed once. On the helical benchmark the step took 3.0 s against 6.9-7.5 s on the CPU with 4 threads; the results differ by about 1e-16 relative (FMA contraction), not bit for bit. Most of the GPU time is pageable host-device transfer. Unit test: `[gpu_symmetrise]`.

- **Accuracy estimate on all cores** (`calculateExpectedAngularErrors`, automatic; `RELION_ACCURACY_THREADS=N` overrides). Under MPI only the first follower computes it, with its `--j` threads, while the GPU and the other ranks wait (about 3.3 s per iteration on the helical benchmark). The trials are independent, use a per-thread generator that reproduces the serial random sequence and are accumulated in particle order, so the result does not depend on the thread count. It now uses all cores the process may run on (`sched_getaffinity`), and the choice is printed once. Checked on a single-particle benchmark: the estimated accuracies of every class in iterations 1 and 2 are identical with 3 and with 8 threads; the maps differ only by float atomic-order noise (about 5e-6 relative).

- **Particle groups in the GPU E-step** (`RELION_GPU_PARTICLE_GROUP=n` or `=auto`, opt-in, single-body refinement). A thread takes n particles through the stages together. For 2D references (Class2D) the coarse differences of the whole group are then computed in one grouped launch (grid z = particle, `RELION_GPU_GROUP_COARSE=0` turns this off) followed by one batched sum kernel, and each particle's smallest difference comes back in one copy. With `RELION_GPU_COARSE_DETERMINISTIC=1` (partial sums plus the sum kernel for any number of chunks) every particle's coarse weights and smallest difference are bit-identical to the single-particle path: `RELION_GPU_GROUP_CHECK=1` recomputes each particle the single way and counts differences (0 of 36,000 particles in a helical Class2D benchmark). `auto` measures blocks of real work at the sizes 1, 2, 4, 8, 16 (particles per second of wall time, medians over rounds, first round dropped), takes the smallest size within 3% of the best, and re-checks the neighbours later, changing only for a 4% gain (`GroupSizeTuner`, unit test `[group_size_tuner]`); the choice and the speeds are printed. A group that does not fit 70% of the allocator's free memory divided by the threads is run in slices. Measured on that benchmark the speed is the same for n = 1 and 2 and 5-10% worse for n = 8 and 16: the per-particle host work and the per-particle copies and synchronisations are unchanged, so grouping the coarse launches alone does not pay (launches per particle 23.9 -> 23.3, synchronisations 24.4 -> 22.3).
