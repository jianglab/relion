# Lazy SSD cache fill - `RELION_CACHE_LAZY=1`

`--cache_dir` normally copies every particle stack to the local disc before the
job starts. For a large data set that copy can take longer than several
iterations. With `RELION_CACHE_LAZY=1` the job starts at once and the copy runs
in the background.

## Behaviour

- The stacks are queued and copied by background threads. A stack that is not yet
  copied is read from its original place; once the copy is complete, the next open
  reads the cache. A stack that is already open keeps its handle, so every
  particle is read from exactly one complete file and **results do not depend on
  how far the copy has come** (bit-identical, tested).
- Copies are written as `<name>.lazytmp`, given the source modification time, and
  renamed. A cache file is therefore either absent or complete; a crash leaves at
  most a `.lazytmp` file that is never read.
- A per-stack lock (`<name>.lazylock`, `flock`) keeps processes that share one cache
  (several MPI ranks, several jobs) from copying the same stack twice.
- A cache file whose size or modification time differs from the source is
  replaced. Stacks that match are skipped, so a second job on the same project
  reuses the cache at no cost, and a job that ended early leaves a partial cache
  that the next job simply continues.
- Number of copy threads: starts with one, adds one every 3 s while the copy rate
  grows by at least 15% per added thread (a network file system answers several
  requests at once, one disc does not), up to `--cache_copy_threads` (default 4).
- If a copy fails (disc full, permission), a warning is printed once, copying stops,
  and the job carries on reading the originals.
- At the end of the job the leader prints one line: stacks copied, time of the last
  finished copy, threads used, stacks already cached, copied by another process,
  not copied, and how many opens fell back to the original.

## When it helps

Prefetching already hides file reads when the read pool is faster than the
compute. The cache still pays for fast jobs (Class2D), slow or network storage and
data larger than the page cache. Decide from the `E-step waited X s` and
`prefetch:` lines of `RELION_ITER_TIMING=1`. With the lazy fill the cost of
trying the cache is small: the first iterations read the originals, as without a
cache.

## Scope and limits

- Opt-in and not on by default. Without the variable the code path is unchanged.
- Only the MPI programs use `--cache_dir` (`relion_refine_mpi`, `relion_reconstruct_mpi`);
  `relion_refine` without MPI ignores it, as before.
- With a node-local cache on several nodes, only one rank per cache queues the copy
  (unchanged from the eager implementation).
- `.vstack` virtual particles are not handled.
- The cache registry (size and eviction) sees a partly filled cache as it is.

Code: `src/lazy_cache.{h,cpp}`; hooks in `CacheManager::populateCache`,
`CacheInitializer::initializeCache` and `fImageHandler::openFile`.
Tests: `[lazy_cache]` unit tests, `tests/integration/test_lazy_cache.py`.
