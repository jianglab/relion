# Lean GPU memory (default; `RELION_GPU_LEAN_MEMORY=0` for the old pool)

## The problem

The old behaviour of `relion_refine --gpu`: every expectation step takes **all free memory of the card**
in one `cudaMalloc` (minus a small reserve) and hands it out in pieces. That is fine for a dedicated queue
node, but on a workstation it leaves nothing for other programs, even when the job needs a small part of it.
The single large block exists for a reason: `cudaMalloc` and `cudaFree` stall the whole GPU, so allocating
many small blocks during the E-step would be slow.

## What the lean pool does

On by default. `RELION_GPU_LEAN_MEMORY=0` restores the old grab-all pool with identical allocations
(it prints one line per device saying so); `=2` also prints a line every iteration. Unset or `1` is the
default. The choice, the first chunk, its source (tuning cache, measured, or default) and the cap are printed
once per device at the start of the first iteration.

* The pool starts with a **first chunk** sized from what the job really used, and adds more chunks only when
  an allocation does not fit and no pending block will be freed in time. Growth is geometric (half of what is
  held, at least 64 MB), so the number of driver calls is logarithmic in the final size.
* The pool never holds more than its **cap**: the amount the old pool would have taken (free memory plus what
  the pool already holds, minus the reserve `required_free`). Everything that fits today still fits.
* At the start of each iteration the pool is rebuilt into one chunk if nothing is allocated and the held size
  does not fit the plan (not one chunk, too small, or more than 30% too large). Blocks never merge across
  chunks, so a block is never split over two `cudaMalloc` areas.
* If growing fails (another program took the memory), the allocator does what it does today: wait for the
  pending frees, then give the normal out-of-memory error. It never fails earlier than the old pool.
* The peak is the largest number of bytes in use at any moment (blocks waiting for their events count as in
  use). It is tracked per device, and kept for the whole process, because MPI followers build a new pool every
  iteration.

## Size of the first chunk

`gpu_pool_plan.h` (no GPU code, unit tested):

| Situation | First chunk | Margin |
|---|---|---|
| peak of the previous iteration known | peak x margin | 1.15 + min(0.10, 2r), where r is the relative change between the last two peaks; 1.2 with one peak only |
| no previous iteration, value in the tuning cache | cached peak x 1.2 | 1.2 |
| nothing known | cap / 64 | none |

Always at least 64 MB, never more than the cap. 15% covers the fragmentation of a first-fit pool, the rest
covers the drift of the peak between iterations (finer sampling, larger images).

## Remembering the peak between runs

The peak is stored in the tuning cache (`src/tuning_cache.h`, default `~/.relion/tuning.csv`,
`RELION_TUNING_FILE` to move it, `RELION_TUNING_CACHE=off` to disable): tunable `gpu_pool_peak`, per GPU model
and SM count, key = the sizes the peak depends on (box, number of classes, dimension, model bytes, threads and
ranks sharing the card, current and coarse image size, number of orientations, pool size). The key holds
numbers only, no names or paths. A row is written when the peak differs by more than 10% from the last row
written for the same key. The cache is only a starting point: a wrong value costs a few extra chunk
allocations in the first iteration, never a different result.

## Overrides and output

* `RELION_GPU_LEAN_FIRST_MB=<n>` fixes the first chunk (expert override; still limited by the cap).
* `--free_gpu_memory <MB>` is honoured: it is part of `required_free`, so it lowers the cap exactly as it
  lowered the old pool. A user who wants a fixed pool of the old kind sets `RELION_GPU_LEAN_MEMORY=0`.
* `RELION_GPU_LEAN_INPLACE_TRANSFORM=0` turns off the in-place reference transform below (lean mode only).
* The first iteration prints the first chunk, the reason and the cap. Later lines (mode 2, or when the pool
  had to grow) report peak in use, peak held and number of added chunks. A final line is printed at exit.

## Memory outside the pool (Refine3D with helical symmetry)

Most of the card in a large Refine3D job (box 384, padding 2) is not the pool. With
`RELION_GPU_MEM_TRACE=1` every rank prints `[gpu-mem] rank N: card in use X MB ... what` at the places
below (card total from `cudaMemGetInfo`, so with several ranks on one card the changes also show the other
ranks' allocations).

* **Reference Fourier transform** (`Projector::computeFourierTransformMap`, once per iteration and half-set,
  serialised across ranks): in lean mode the cuFFT plan is made first (`cufftSetAutoAllocation(0)`), which
  reports the exact work size, and the block is the exact sum of the buffers plus 2% and 16 MB
  (`gpupool::transformPoolBytes`). The old block was cuFFT's upper-bound estimate times 1.5. The plan and
  the buffers are the same, so the result is bit-identical.
  For 3D references the transform is also done **in place**: the padded map is written with a row pitch of
  `2*(N/2+1)` reals, the real-to-complex cuFFT runs in place, and the complex array aliases the padded map.
  This saves the second array (3.6 GB at N=768). The unit test `[gpu_inplace_fft]` shows the in-place numbers
  are bit-identical to the out-of-place ones.
* **Helical symmetrisation in Fourier space** (`cuda_helix_fourier.cu`): planes are processed in groups sized
  from the free memory. With a big card the whole volume was one group (about 10.5 GB per rank). In lean
  mode a group holds about 1 GB (`free/8`, between 256 MB and 1 GB). The group size does not change the
  result; the log line of the choice says when it was reduced.
* Projector textures, back-projection accumulators and the context are needed for the whole iteration and
  are not changed.

Measured on a large helical Refine3D (box 768 for the transform, two half-set ranks on one card, speed-ups
switched on): the card peak fell from 21.9 GB to 9.9 GB, which is the steady-state use of the two ranks
(textures 1.9 GB + accumulators 2.6 GB + pool 0.07 GB per rank, plus the contexts). Wall time and the final
resolution did not change.

## Limits

* Memory taken outside the pool (the references and their back-projection arrays, cuFFT plans, the
  per-thread overhead) is not part of the pool and not counted in its peak.
* The first iteration of a job with no cache row grows in several steps; each costs one driver call.
* Only the CUDA allocator is changed; the HIP allocator is untouched.
