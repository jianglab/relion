# Worker processes (MPI ranks) per GPU

One RELION process issues about 120 CUDA calls per particle, mostly to kernels of 10 to 20
microseconds, so it keeps a GPU only about a third busy. More threads in the same process do not
help (they submit their work one at a time); more worker processes on the same GPU overlap their
submissions. `parhint::adviseRanks()` (src/parallel_hint.cpp) turns the measurements below into a
recommendation, shown as a hover hint on the GUI's MPI and thread fields, and used by `relion_launch`
(below). RELION never changes the number of ranks of a command you start yourself.

E-step seconds, one RTX 3090 (24 GB), 8 cores (workers x threads = 8, so 8 workers have one
thread each), JiangLab 49c2c5b6 (2026-10-23), average of two interleaved passes (a first run's
warm-up is left out; B1/B3/B4 are small data sets continued from a fixed state):

| job | 1 worker | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| EM Class2D, 1000 helical segments, K=50 (B1) | 12.6 | 11.5 | **9.8** | 10.5 | 9.8 |
| Class3D with alignment, 4.6k particles (B3) | 2.3 | **1.95** | 1.9 | 2.25 | 2.2 |
| Class3D --skip_align, 4.5k particles (B4) | 3.0 | 2.95 | **2.7** | 3.4 | 3.5 |
| auto-refine, 20k particles (B5; even worker counts only) | | **77.8** | out of memory (iteration 25) | | |

Cost of a worker: GPU memory about 0.4 GB (Class2D), 2.5 GB (Class3D at 2 workers; less with
more), 2.5 GB (skip-align), 7.8 GB (auto-refine, and its last iterations need more); host memory
0.4, 0.7, 1.8 and 14 GB. RELION starts NVIDIA MPS itself when several workers share a GPU.

The earlier table (2026-10-19, 12 threads, before the CC kernel, --coarse_early, the shared coarse
pass and the host work beside the GPU) gave 6 workers for Class2D and 4 for Class3D; faster kernels
lowered both. On a smaller card without MPS (RTX 3060 Ti) fewer workers were best (see below), which
is what the per-job learning of relion_launch is for.

Other cards, same runs (8 cores; best count first, then the 3090 rule's count when different and
how much slower it was): B1 Class2D: RTX 6000 (48 GB) 6 (4: +10%), RTX 4090 8 (4: +4%), A5000 8
(4: +16%), A4000 8 (4: +19%, noisy), a second RTX 6000 8 (4: +8%). B3 Class3D: 2 on all, except the RTX
4090, 4 (2: +13%). B4 skip-align: 4 on all six models on this small set, but not on a real-sized
one (below). With 4 cores (1, 2 or 4 workers): Class2D is
still fastest with 4 workers of one thread each, Class3D is flat, and skip-align with 4 one-thread
workers is up to 25% slower than 2 workers of 2 threads on every card, hence its 2-thread minimum.
So one rule fits all six models within about 10% (the learning of relion_launch closes the rest),
and no per-model table is needed.

Skip-align on a real-sized set (2026-10-27): helical Class3D --skip_align, 70k segments, 5
iterations, 8 cores, RTX A4000, two runs each, warm file cache: 1 worker x 8 threads 415 s, 2 x 4
416 s, 4 x 2 450 s (E-step 2-5: 78, 78-93 and 84-99 s each). 4 workers kept the GPU no busier
(42-54% for all three). Its rule is therefore 2 workers: as fast as 1 when the images are in memory,
and two readers when they are not (a first, cold run spent most of iteration 1 waiting for images).
An A5000 node with a slow shared disk gave 1100-3500 s for all three layouts, too noisy to rank them.

Rule: workers per GPU = 4 (Class2D), 2 (Class3D, skip-align), 2 (auto-refine; the number of
workers must be even), limited by 75% of the GPU memory / memory per worker, 60% of the host
memory, one core per worker (two for skip-align), and 300 particles per worker; threads per worker = cores / workers,
at most 8. `RELION_RANKS_PER_GPU` overrides the workers per GPU, `RELION_GPU_GB_PER_RANK` the
GPU memory per worker, `RELION_GPU_MEMORY_GB` and `RELION_HOST_MEMORY_GB` the memory the GUI
assumes (it reads this computer's; for queue jobs the GPU memory is unknown unless given).

Identity: continuing from a fixed state for 1 to 3 iterations, every rank count gives the same
log-likelihood (relative difference below 2e-5), the same class assignments up to near-tie
particles (about 1% for 3D classification, against 0.4% between two runs of the same setting) and,
for auto-refine, the same resolution and Pmax within 0.1%. Class2D: no change at all in class or
pose. A 3D classification started from scratch (4 classes from one reference, 6 iterations, B3
data) follows the same path with 1, 2, 4 and 6 workers on JiangLab 49c2c5b6: resolution 45, 45, 28,
25, 21, 19 A per iteration and 17.4 A at the end, log-likelihood within 1e-5 relative, for two seeds.
(On the 2026-10-19 tip the same run stayed at 45 A with 2 workers; that no longer happens.)

## relion_launch: chosen on the node the job lands on

On a shared cluster the GPU model a queue job gets is only known when it starts, so a count chosen at
submission either wastes a large card or overfills a small one. With "Choose MPI and threads
automatically?" set to Yes, the built-in Slurm script of a GPU refinement asks for one task with all
the job's CPUs (and its GPUs) and runs `relion_launch <command>`. On the node, `relion_launch`
(src/rank_launcher.h):

- reads what the job got: the GPUs it may use (`CUDA_VISIBLE_DEVICES`; model and free memory from
  nvidia-smi), the cores it may run on, and its memory (Slurm's limit, the cgroup's, or what is
  available), plus the kind of work and the particles from the command;
- chooses the workers per GPU and the threads with the rule above, starting from the count learned
  for this GPU model and kind of work, if any;
- prints one line with the choice and why, and starts `mpirun ... relion_refine_mpi` (or
  `relion_refine` for one process) with `--j` set.

A GPU refinement run on the GUI's own computer with the same setting goes through `relion_launch` too.
The GPUs given in "Which GPUs to use" become the job's devices (`CUDA_VISIBLE_DEVICES`) and the workers
are spread over them, since a per-process mapping such as "0:1" cannot fit a count chosen at the
start. The MPI and thread fields the GUI greys out show the same measured rule as the hover hint;
`relion_launch` makes the final choice. When it cannot choose (no GPU found), a `relion_refine_mpi`
command runs as one `relion_refine` process with a thread per core.

Learning. At the end of a run that `relion_launch` started (longer than a minute, on one computer),
RELION records its E-step seconds per particle in the tuning cache (`ranks_per_gpu`, device = GPU
model, key = kind of work plus a hash of the input data, K, healpix orders, offset search, particle
diameter and symmetry). Those times compare only between runs of the same job, so each job learns
for itself: when the run left cores to spare (below 75% busy) and every worker had two threads or
more, the next run of the job starts one more worker per GPU, otherwise one fewer (more threads
each). A step that was at least 3% faster is followed by another step the same way; one that was not
goes back to the earlier count, which stays. So a start that is too high (WSL found Class2D fastest
at 2 workers per GPU, where the rule said 6) is corrected too. The memory, core and particle limits
always apply on top of the learned count.

Jobs without a record of their own. When a job settles, its count is also stored under a record of
its kind of work on that GPU model with about as many cores per GPU (1-2, 3-4, 5-8, 9-16, more), e.g.
`class3d:any:cores8`; the latest settled job wins. A new job starts from that record ("Other jobs of
this kind on this GPU model ... settled on N") and then learns for itself; only when no job of its kind
has settled on that GPU model does it start from the measured rule.

GPU memory measured per job. The resource report samples the memory in use on each card (NVML) during
the run; the card's peak divided by its workers is stored with the job's record, and the next run of
the job limits its workers by that figure instead of the rule's per-worker figure (which comes from
our data sets; it grows with box size and classes). Memory is mostly per thread, not per process, so
this figure overestimates the cost of an extra worker: it only lowers the count when a run came close
to filling the card. `RELION_GPU_GB_PER_RANK` still overrides both.

MPS on or off. `relion_refine_mpi` starts NVIDIA MPS when several workers share a node (see "Automatic
MPS" in gpu_performance.md), but MPS does not always help. Measured: helical Class2D with 8 single-thread
workers on one GPU ran 7x faster with it (many processes with small launches); a helical Class3D with 2
workers of 3 threads, whose long coarse kernels each fill the GPU, ran 6% slower (427 vs 452 s, three
pairs, identical results). So `relion_launch` chooses: MPS on from 4 workers per GPU, off below (always
off with one). Once the workers per GPU of a job have settled, its next run tries the other setting once;
after that the faster one is kept (tuning cache `mps`, key = the job key plus the workers per GPU), and a
job measured both ways also sets the start for other jobs of its kind. The time is filed under what the
workers actually ran with, so a run where MPS could not start counts as one without it. The launcher line
says the choice and why ("MPS off: fewer than 4 workers per GPU"); `RELION_AUTO_MPS=on` or `off` set by
hand is obeyed and nothing is learned. `relion_refine_mpi` started without `relion_launch` keeps starting
MPS whenever two or more workers share the node.

GPU busy % is deliberately not the signal: measured on an RTX 3060 Ti without MPS, it reads 93-98% as
soon as two processes share the GPU (context switching counts as busy) while the E-step got slower
from 3 workers on; and `--skip_align` read 32% busy with one worker but gained nothing from a second,
because the host side was the limit. Measured there (E-step speed-up against 1 worker, 16 threads in
all): Class2D 1.28 / 1.22 / 1.17 / 1.00 at 2 / 3 / 4 / 6 workers; Class3D 1.22 / 1.14, out of memory
at 4 on the 8 GB card; skip-align 1.01 / 0.92 / 0.79 / 0.75. Without MPS, two workers per GPU was the
best on that card, which is why the count is learned per GPU model and job rather than fixed.

A run that runs out of GPU memory is not recorded (shortcut: no automatic back-off after an
out-of-memory failure yet; set `RELION_RANKS_PER_GPU` lower).

Overrides: `RELION_RANKS_PER_GPU` (workers per GPU), `RELION_MPIRUN` (the MPI launcher),
`RELION_TUNING_CACHE=off` (no learning). `relion_launch --dry_run <command>` prints the command it
would run.
