# Environment switches added in this tree

This page lists every `RELION_*` environment variable that this tree reads and
that upstream RELION does not. (Upstream's own variables, such as
`RELION_QSUB_COMMAND`, `RELION_SCRATCH_DIR` or `RELION_MPIRUN`, are not listed.)
It was made by searching `src` for `getenv`, then reading the code around each
use. Each entry gives the default, what the switch does, how to turn it off, and
when you would use it.

How to read the tables:

- **Default** is what happens when the variable is not set.
- **Off** is the value that turns a default-on feature off. For the GPU speed-ups
  this is `0` (some also accept `off`, `no` or `false`). Wherever a switch
  cannot work for a job (for example 2D data, or no GPU), RELION quietly uses the
  older path, so it is safe to leave the defaults alone.
- Most switches are read **once, when the program starts**. Set them in the
  environment of `relion_refine` (for MPI jobs, make sure `mpirun` passes them
  on: `mpirun -x NAME ...` with OpenMPI).
- A command-line option, where one exists, wins over the variable.
- Rule of thumb: ordinary users need none of these. Experts use them to compare
  against the older code path (`=0`), to find a problem (`check`, timing and
  trace switches), or to try a feature that is still opt-in.

Switches that print a line at the start of a job (the GPU ones do, at
verbosity above 0) name themselves in that line, together with the value that
turns them off.

## 1. Opt-in features (off unless you ask)

These change the method or the numbers beyond noise, or are still being
validated. They stay off until experts have tried them on real projects.

| Variable | Effect | When to use it |
|---|---|---|
| `RELION_INTERPOLATION=nufft` | Replaces the trilinear central slice in `Projector::project()` by an exact NUFFT slice (`relion_refine`, `relion_autopick`, `relion_particle_subtract`). CPU only and slower. Needs a FINUFFT build; with `--gpu`, `--sycl`, `--cpu` or no FINUFFT it is an **error**, never a silent fallback. `linear` (or unset) is the default. | Accuracy studies. See `finufft_central_slice_projector.md`. |
| `RELION_FINUFFT_TOL`, `RELION_FINUFFT_UPSAMPFAC`, `RELION_FINUFFT_MODE_CROP` | Tuning of the NUFFT projector: tolerance (between 0 and 1), upsampling factor (0 lets FINUFFT choose, otherwise 1.06 to 4) and mode cropping (positive). Defaults are the built-in ones of the options of the same names. | Only together with `RELION_INTERPOLATION=nufft`. |
| `RELION_FINUFFT_BATCH_MB`, `RELION_FINUFFT_CACHE_MB` | Defaults of `--finufft_batch_mb` (512) and `--finufft_cache_mb` (4096). | Memory budget of the NUFFT paths, if the defaults do not fit the node. |
| `RELION_EXTRACT_INTERPOLATION` | Default of `relion_preprocess --interpolation` (`linear`, `cubic` or `nufft`) when a helical box is turned. Default `linear`. | Same as above, for extraction. |
| `RELION_VIRTUAL_PARTICLES=1` | Makes `relion_preprocess --virtual` the default (`--no_virtual` overrides). Extract then writes small `.vstack` descriptors, and particles are computed from the micrographs when read. | Saving disc space. See `virtual_particles.md`. |
| `RELION_VIRTUAL_MOVIE_AVERAGES=1` or `real` | RELION's own MotionCorr writes a descriptor (`1`) or a reproducible real micrograph (`real`) instead of the usual file. Off by default; MotionCor2 with the switch is an error. | Saving disc space. See `virtual_movie_averages.md`. |
| `RELION_VPARTICLE_CACHE`, `RELION_VMOVIE_AVERAGE_CACHE` | Directory of the cache of computed virtual particles / micrographs. Default: a cache folder inside the project. `off` (or `0`, `no`) disables it. Always safe to delete. | Put the cache on fast local disc, or turn it off. |
| `RELION_VPARTICLE_CACHE_MAX_GB`, `RELION_VMOVIE_AVERAGE_CACHE_MAX_GB` | Size limit of those caches, default 100 GB; the least recently used entries go first. `0` means no caching. | Small project disc. |
| `RELION_FILAMENT_CONTEXT=off\|chain\|filament` | Filament context in Class2D/3D: neighbouring segments of one filament share information (a Markov chain along the filament, or a filament-wide class distribution). Default `off`. An unknown value is an error. Command-line `--filament_context` wins. | Helical classification of filaments with weak single-segment signal. |
| `RELION_FILAMENT_TEMPER`, `_GAP`, `_FLOOR`, `_ALPHA` | Parameters of the filament context: power on the segment likelihoods (default: segment step divided by box length), minimum spacing of segments used (1 or more, default 1), floor mixed in with the global class distribution (0 to 1, default 0.02) and weight of the global distribution in filament mode (default 1). | Only with `RELION_FILAMENT_CONTEXT`. |
| `RELION_AUTO_MAXSIG=1` | Caps the number of significant orientations in the early iterations, when the posterior is nearly flat and tens of thousands of orientations survive. Off. `--maxsig` on the command line wins. | Faster Class3D with unchanged quality in our tests; not yet default. |
| `RELION_PARTICLE_ORDER=micrograph` | Shuffles whole micrographs instead of single particles, so a pool of particles touches fewer stack files. Off. | Slow storage and many small stacks. See `gpu_performance.md`. |
| `RELION_GPU_PARTICLE_GROUP=n` or `auto` | The GPU expectation step takes `n` particles (or an automatic number, chosen while running) through each stage together. Default 1 (one at a time). Single-body refinement only. | Many small kernels, for example 2D classification. |
| `RELION_FFTW_PLANS=measure` | Measures FFTW plans once instead of estimating them. Default `estimate`. Plans are kept in a wisdom file (`RELION_FFTW_WISDOM`, default under `$XDG_CACHE_HOME` or `~/.cache/relion`, one per CPU model); each measurement is limited to `RELION_FFTW_PLAN_TIMELIMIT` seconds (default 30). | Long jobs with big volumes, after you have checked the gain. |
| `RELION_GPU_SYNC_MODE=spin\|yield\|blocking` | How host threads wait for the GPU. Default `auto` (CUDA's own choice). | Many jobs sharing a node: `yield` or `blocking` leave CPU for others. |
| `RELION_GPU_SMALL_FT=1` | Keeps the GPU Fourier transform for small 2D references (below 2^20 elements), which normally go to the CPU. Off. | Only to compare. |
| `RELION_GPU_MAP_PER_CLASS=1` | One GPU launch per class where the default is one for all classes. Off. | Comparing against the batched path. |
| `RELION_GPU_FUSED_PREP`, `RELION_GPU_MASK_FUSED`, `RELION_GPU_SHIFT_TABLE`, `RELION_GPU_SHARED_SUMS` (`1`, `2` or `3` for variants) | Experimental kernel variants (fused preparation, fused helical mask, shift table, shared-memory sums). All off. Several change float rounding; none changed a map beyond noise in our tests. | Kernel development and A/B timing. |
| `RELION_GPU_FINE_SPLIT=auto` or `N` | Pixel chunks for the fine pass of 2D references. Off. Chunk sums are added in a fixed order, so runs repeat exactly but differ from the unchunked kernel in the last digits. | 2D classification on a card with many multiprocessors. |
| `RELION_GPU_BP_TILE=1` or `check` | Tiled back-projection of 2D data (a warp takes 16 x 2 pixels). Off. `check` also runs the default path on the first `RELION_GPU_BP_TILE_CHECKS` launches (default 2) and reports the difference; valid with `--j 1` only. | Kernel development. |
| `RELION_GPU_COARSE_DETERMINISTIC=1` | The coarse pass adds its sums in a fixed order, so a particle gives bit-identical results alone or in a group. Off. | Together with `RELION_GPU_GROUP_CHECK`, or when you need exact repeatability. |

## 2. GPU speed-ups that are on by default

All of these keep results the same as the older path (floating-point rounding
aside), print themselves at verbosity above 0, and fall back by themselves when
they do not apply. The value that switches the feature off is in the first
column. Details and measurements are in `gpu_performance.md` and
`gpu_lean_memory.md`.

| Off | What it does | When to turn it off |
|---|---|---|
| `RELION_GPU_LEAN_MEMORY=0` | The memory pool grows as needed (and remembers the peak in the tuning cache) instead of taking all free GPU memory. `=2` also reports the pool every iteration. | Only to restore the old pool. |
| `RELION_GPU_LEAN_FIRST_MB=N` (a size, not a switch) | Overrides the size of the first chunk of the lean pool. Default: chosen from the measured peak. | Odd memory situations. |
| `RELION_GPU_LEAN_INPLACE_TRANSFORM=0` | Fourier transform of the padded reference in place, so the output array is not needed (3D). | Comparing memory use. |
| `RELION_GPU_KEEP_POOL=0` | Keeps the expectation-step memory on the card during GPU reconstructions; frees it first if the reconstruction does not fit. | A card shared with other programs. |
| `RELION_GPU_KEEP_ALLOCATOR=0` | Keeps each device's memory pool (the allocator) alive between runs of the expectation step in one program, instead of freeing and creating it again. | Debugging memory. |
| `RELION_KEEP_PPREF=0` | With `--gpu`, keeps the host copy of the projector data (so it need not be rebuilt) when enough host memory is available. | Low host memory. |
| `RELION_GPU_CLASS_BATCH=off` | One GPU call for all classes. Off gives one per class, as upstream. | Comparison. |
| `RELION_GPU_BP_ACCUM=0` | Per-orientation accumulation for 3D back-projection when the setup allows it. | Comparison. |
| `RELION_GPU_BP_LAYOUT=planar` | Real, imaginary and weight arrays of the 3D model apart, as upstream. Default is interleaved. | Comparison. |
| `RELION_GPU_BP_PACK=0` | Packs the atomic additions of a warp into runs of neighbouring floats. Same terms, different order. | Exact comparison with the older sums. |
| `RELION_GPU_BP_SCATTER=0` | Scatter variant of the 3D back-projection kernel (passed to the kernel as a flag). | Comparison. |
| `RELION_GPU_BP_SPLIT=N` / `RELION_GPU_WAVG_SPLIT=N` / `RELION_GPU_COARSE_SPLIT=N` | Number of pixel chunks of the back-projection, weighted-average and coarse kernels. Default: chosen from the device (about 3 blocks per multiprocessor); `1` means no chunks. | Expert tuning. |
| `RELION_GPU_BLOCKS_PER_SM=N` | Blocks per multiprocessor that the chunking aims for. Default: from the device (threads per multiprocessor divided by 128, at least 3). | Expert tuning. |
| `RELION_GPU_COARSE_XT=0` | Cross-term coarse kernel (3D references, 2D with a small table). `=check` measures the difference to the direct kernel. | Comparison. |
| `RELION_GPU_XT_TILES=N` or `off` | Tiles of translations per launch in the cross-term kernel. Default `auto`: tried while running, the fastest is kept. | Expert tuning. |
| `RELION_GPU_FLOAT2_TEX=0` | One two-channel projection texture instead of two single-channel ones (3D, single precision). | Comparison. |
| `RELION_GPU_HELICAL_MASK=off` | Helical mask on the GPU. Off uses the CPU, as upstream. | Comparison. |
| `RELION_GPU_ORIENT_CACHE=0` | Caches the orientation lists between particles. `RELION_GPU_ORIENT_CACHE_VERBOSE` (any value) says why a cache was or was not used. | Debugging memory. |
| `RELION_GPU_GROUP_COARSE=0` | With particle groups, the coarse differences of a group in two launches. `RELION_GPU_GROUP_CHECK=1` compares them with the single-particle pass (use with `RELION_GPU_COARSE_DETERMINISTIC=1`). | Debugging. |
| `RELION_GPU_PINNED_STAGING=off` | Copies host data to the GPU through a pinned staging buffer. | Comparison. |
| `RELION_GPU_PINNED_POOL=0` | The pinned staging buffers are shared by all threads of a program (a pool, at most 2 GB kept idle) instead of one `cudaMallocHost` per thread. `=0` allocates and frees one per thread, as before. | Debugging memory. |
| `RELION_GPU_PINNED_BIG_COPY=0` | Copies of 1 MB or more in the GPU helical symmetrisation go through two pinned 8 MB buffers, overlapped with the transfer. `=0` is a plain `cudaMemcpy`. Same bytes either way. | Comparison. |
| `RELION_GPU_WAVG_SUM=0` | The class-batched weighted averages of 2D data computed from the sum of the weights and the weighted sum of the shifted images, with the shifts taken from a table made once per particle (no `sincosf` and no shared-memory sums per pixel). Sums are added in another order, so results differ in float rounding only. `=0` is the older kernel. | Comparison. |
| `RELION_GPU_BLOCK_SORT=0` | The sort, running sum, significance threshold and arg max of one particle's weights in a single thread block (up to the block's capacity; larger passes use the radix sort as before). Same selection; float rounding of the running sum may differ in the last digits. `=0` always uses the radix sort. | Comparison. |
| `RELION_GPU_SKIPALIGN_BLOCK=0` | The fine pass of `--skip_align` (no oversampling, at most 4096 weights per particle, as in Class3D without alignment) in three launches and one copy: partial maximum, exponentiation, and one block that sorts, sums and finds the threshold and the arg max. Replaces the radix sort, the scan, the arg max reduce and three host reads. Same threshold and significant set; for one weight per particle the sum is exact, for several the block scan may differ from the cub scan in the last float digits. Used only with `--skip_align`; other cases keep the earlier path. `=0` always uses the earlier path. | Comparison. |
| `RELION_GPU_FUSED_WEIGHTS=0` | The significance pass of the weights (maximum, exponentiation, sort, scan, threshold, significance mask) as a few launches and one copy instead of about fifty driver calls per pass. Same selection of orientations; float rounding of the sums may differ in the last digits. `=0` is the older step-by-step pass. | Comparison. |
| `RELION_GPU_SELECT_FUSED=0` | The radix selection of the significance threshold (the histogram passes and the bin choice, 3 levels, 1 or 2 chains per particle) runs as ONE cooperative kernel launch instead of 6-12 launches; `=1` uses the plain scans of the bins, the default `=2` also warp-shuffle scans (about 34% less kernel time, about 4.5% of the GPU time of a tutorial 2D class average run). The selected weight, threshold and mask are the same; the double sums of the histogram are added in atomic order as before, so the total weight differs in the last digits only. The grid is limited to the blocks that fit on the card at once; a card or a launch that cannot do a cooperative launch uses the separate launches (a warning is printed once). | Comparison, or a card that misbehaves with cooperative launches. Test hook: `RELION_GPU_SELECT_FUSED_REFUSE=1` makes the driver refuse the cooperative launch, to exercise the fallback (results are identical). |
| `RELION_GPU_RECONSTRUCT=0` | The M-step reconstruction on the GPU, also with `--skip_align`. | A card too small for the reconstruction. |
| `RELION_GPU_SYMMETRISE=0` | Helical Fourier-space symmetrisation on the GPU. `=check` runs both and compares. `RELION_GPU_SYMMETRISE_PLANES=N` limits the planes per group (default: from free memory). | Comparison. |
| `RELION_GPU_SKIP_ALIGN=0` | GPU for `--skip_align` / `--skip_rotate`. `=0` switches the GPU off for them as upstream does (the job window then refuses the GPU). | Comparison. |
| `RELION_JOIN_SKIP_BPREF=0` | At the final join of a converged 3-rank auto-refine, the summed back-projection arrays are not sent back. `=0` sends them as upstream. | Comparison. |
| `RELION_AUTO_MPS=off` | Starts NVIDIA MPS when two or more GPU workers share a node and none is running. | MPS is managed by your site. |
| `RELION_AUTO_PARALLEL=No` | GUI default of "Choose MPI and threads automatically?". | You always give MPI and threads by hand. |
| `RELION_CPUS_PER_GPU=n` | CPU cores per GPU that the GUI's suggestions assume. Default: from the Slurm partition or at most 8. | Site-specific sizing. |

## 3. CPU-side speed-ups that are on by default

| Variable | Default and effect | When to change it |
|---|---|---|
| `RELION_ASYNC_WRITE=0` | On. File writes are queued to one background thread, in order, so the next iteration does not wait for a slow file system. `0` or `off` writes at once. | Debugging a write problem. |
| `RELION_MPI_PREFETCH=0` | On (when parallel disc I/O is used, 2D data). Followers read the next pool of particles while processing the current one; a wrong guess only costs a wasted read. `RELION_PREFETCH_THREADS=N` limits reader threads (default the smaller of `--j` and 8). | Very slow storage with few readers, or comparison. |
| `RELION_UNREG_VIA_DISC=1` or `check` | Default: the unregularised reconstruction arrays stay in memory when there is room (4 times what they need), else go through disc. `1` forces disc; `check` does both and compares. | Self-test, or tight memory. |
| `RELION_FFTW_THREADS=1` | Multi-threaded FFT of large transforms. `1` makes every FFT single-threaded, as upstream's non-MKL builds. `RELION_FFTW_DEBUG=1` prints each large transform. | Comparison or debugging. |
| `RELION_STARTUP_THREADS=N` | Threads for the start-up work (reading and writing star tables, noise estimate). Default all `--j`; `0` turns the threaded noise estimate off. | Comparison. |
| `RELION_SYMMETRISE_THREADS=N`, `RELION_ACCURACY_THREADS=N`, `RELION_CLASS_THREADS=N`, `RELION_PACK_THREADS=N` | Thread counts of the symmetrisation (default `--j`), the accuracy estimate (default `--j`), the per-class loops (default chosen from the size of the reference) and the packing of weighted sums (default the smaller of 4 and the OpenMP maximum). Results do not depend on them. | A node shared with other jobs. |
| `RELION_ALIGN_THREADS=N` | Threads of `alignMapToMap` (class and half-map alignment). Default: `--j`. | Shared node. |
| `RELION_TUNING_CACHE=off` | On. Values the program chooses by measurement (tile shapes, memory peaks) are kept in a file and reused. `RELION_TUNING_FILE=path` sets the file (default `~/.relion/tuning.csv`). | `off` for clean timing; give a project file to keep tunings with the project. |
| `RELION_RESOURCE_REPORT=off` | On. A short report of time, memory and GPU use at the end of a job. | Cleaner logs. |
| `RELION_SLURM_PREAMBLE=file` | Lines of this file are put before the command in RELION's own Slurm script (modules, environment). Default none. | Site setup. |
| `RELION_CACHE_DIRECTORY=dir` | GUI default of "Local SSD cache directory" for the frozen-class refinement. Default empty. | Fixed local SSD path at your site. |
| `RELION_KEEP_ALL_ITERATIONS=1` | GUI default of "Keep all iterations?". Default No: only the last iteration (and `it000` when needed) is kept. `--keep_all_iterations` / `--dont_keep_all_iterations` win. | Runs whose early iterations you compare, or `--continue` from an earlier iteration. |

## 4. Timing, tracing and debugging (off unless set)

None of these changes results.

| Variable | Effect |
|---|---|
| `RELION_ITER_TIMING` | Per-iteration time breakdown (`ITER_TIMING`, `MPI_TIMING` lines). |
| `RELION_ALIGN_TIMING` | Time of each stage of the half-map alignment. |
| `RELION_GPU_COARSE_STATS` | Statistics of the coarse pass, every 500th call. |
| `RELION_GPU_MEM_TRACE=1` | Traces GPU memory use. |
| `RELION_XT_ABLATE=n` | Switches parts of the cross-term kernel off to see what bounds it. **Gives wrong numbers.** Development only. |
| `RELION_S2_DEBUG_COMPARE_FINUFFT_TYPES=1` | s-squared mode: also runs FINUFFT type 3 and compares. Slow. |

## Notes

- `RELION_ASYNC_STAR_WRITE` (a background write of the large `data.star`,
  `model.star` and `optimiser.star`, with a `check` mode) is not in this tree yet;
  it is on a work branch.
- Switches added in this tree are meant to be safe to leave at their defaults.
  When you add one, give it a default, an off value and a line in this page, and
  register an option in both parse paths (`parseInitial` and `parseContinue`) if
  it has a command-line twin.
