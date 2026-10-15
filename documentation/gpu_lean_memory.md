# Lean GPU memory (`RELION_GPU_LEAN_MEMORY`)

## The problem

Without this switch, every expectation step of `relion_refine --gpu` takes **all free memory of the card**
in one `cudaMalloc` (minus a small reserve) and hands it out in pieces. That is fine for a dedicated queue
node, but on a workstation it leaves nothing for other programs, even when the job needs a small part of it.
The single large block exists for a reason: `cudaMalloc` and `cudaFree` stall the whole GPU, so allocating
many small blocks during the E-step would be slow.

## What the lean pool does

Opt-in: `RELION_GPU_LEAN_MEMORY=1` (`=2` also prints a line every iteration; unset or `0` is the old
behaviour with identical allocations).

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
* `--free_gpu_memory` keeps working as before.
* The first iteration prints the first chunk, the reason and the cap. Later lines (mode 2, or when the pool
  had to grow) report peak in use, peak held and number of added chunks. A final line is printed at exit.

## Limits

* Memory taken outside the pool (the references and their back-projection arrays, cuFFT plans, the
  per-thread overhead) is not part of the pool and not counted in its peak.
* The first iteration of a job with no cache row grows in several steps; each costs one driver call.
* Only the CUDA allocator is changed; the HIP allocator is untouched.
