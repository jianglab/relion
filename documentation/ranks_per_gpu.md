# Worker processes (MPI ranks) per GPU

One RELION process issues about 120 CUDA calls per particle, mostly to kernels of 10 to 20
microseconds, so it keeps a GPU only about a third busy. More threads in the same process do not
help (they submit their work one at a time); more worker processes on the same GPU overlap their
submissions. `parhint::adviseRanks()` (src/parallel_hint.cpp) turns the measurements below into a
recommendation. RELION never changes the number of ranks you start; the GUI only shows the advice
as a hover hint on the MPI and thread fields.

E-step seconds, one RTX 3090 (24 GB), 12 worker threads in total (workers x threads), two
interleaved repeats (noise about 1 s in B1, 2 s in B5; B1/B3/B4 are small data sets):

| job | 1 worker | 2 | 3 | 4 | 6 | 12 |
|---|---|---|---|---|---|---|
| EM Class2D, 37k helical segments, K=50 | 10.0 | 8.5 | 6.3 | 5.9 | 5.4 | 5.3 |
| Class3D with alignment, 4.6k particles (continued from a fixed state) | 1.7 | 1.4 | 1.3 | 1.2 | 1.3 | 1.5 |
| Class3D --skip_align, 4.5k particles | 2.4 | 2.0 | 1.7 | 1.6 | 1.6 | 2.2 |
| auto-refine, 20k particles, box 128 (2 workers = one per half-set) | | 46 | | 48 (4) | out of memory | out of memory |

Cost of a worker: GPU memory about 0.4 GB (Class2D), 2 GB (Class3D), 2.5 GB (skip-align), 5 GB
(auto-refine, and its last iteration needs more); host memory 0.4, 0.7, 1.8 and 14 GB.
RELION starts NVIDIA MPS itself when several workers share a GPU.

Rule: workers per GPU = 6 (Class2D), 4 (Class3D, skip-align), 2 (auto-refine; the number of
workers must be even), limited by 75% of the GPU memory / memory per worker, 60% of the host
memory, one core per worker, and 300 particles per worker; threads per worker = cores / workers,
at most 8. `RELION_RANKS_PER_GPU` overrides the workers per GPU, `RELION_GPU_GB_PER_RANK` the
GPU memory per worker, `RELION_GPU_MEMORY_GB` and `RELION_HOST_MEMORY_GB` the memory the GUI
assumes (it reads this computer's; for queue jobs the GPU memory is unknown unless given).

Identity: continuing from a fixed state for 1 to 3 iterations, every rank count gives the same
log-likelihood (relative difference below 2e-5), the same class assignments up to near-tie
particles (about 1% for 3D classification, against 0.4% between two runs of the same setting) and,
for auto-refine, the same resolution and Pmax within 0.1%. Class2D: no change at all in class or
pose. Caveat: a 3D classification started from scratch (4 classes from one reference) stayed at
45 A with 2 workers (3 seeds and thread counts tried) and reached 25 A with 4 or more workers, so
the path of a from-scratch run depends on the number of ranks.
