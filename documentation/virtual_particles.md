# Virtual particles: reading particles straight from micrographs

## Goal

Today `relion_preprocess` (the Extract job) cuts every particle out of its
micrograph, processes it, and writes it to a stack. Those stacks are the only
thing downstream jobs read. Every re-extraction — a new box size, binning,
recentring after refinement — writes another full copy, and on a real project
the stacks rival the micrographs in size (EMPIAR-13508: 692 GB of stacks
against 872 GB of micrographs).

A virtual particle is a particle whose pixels are *defined* by a micrograph, a
coordinate and an extraction recipe, and are computed when read. The Extract job
then writes metadata only: user-facing job directories contain no particle
pixels, re-extraction costs nothing, and the storage goes away. An internal,
disposable cache keeps reads fast.

The one rule everything else follows from:

> **Micrographs + recipe are the source of truth. The cache is only an
> accelerator.** Deleting the cache at any moment changes speed, never results.

Carried one step further (phase 5), the same rule applies to micrographs
themselves: they are derived from the movies and the motion-correction metadata
RELION already records. The end state is

    movies + motion-correction metadata + extraction recipe   (source of truth)
          │
          ├─► micrographs          (accelerator — regenerable, deletable)
          └─► particle cache       (accelerator — regenerable, deletable)

so that everything between the movies and the particle metadata can be deleted
at any time to reclaim storage, and is rebuilt on demand. Phases 1–4 still
treat the micrographs as required.

## What was measured first

A standalone benchmark (kept outside the repository, in
`test-outputs/virtual_extract/`) established three facts before any of this was
built:

1. Re-doing the extraction from the micrograph reproduces `relion_preprocess`
   output to float16 precision on all 11,281 tutorial particles, and bitwise
   when the result is rounded to float16 the way the stack writer rounds it.
2. I/O cost follows **pixels fetched**, not particle count. Micrograph reads are
   bandwidth-bound (a fixed pixel rate); stack reads are latency-bound random
   requests. Reading from micrographs wins while
   `micrograph pixels / stack pixels < k`, with k = 1.2–4.4 measured on our
   MooseFS (dense EMPIAR-13508: 2–3× faster from micrographs; a 10% subset:
   2–2.5× faster from stacks).
3. Reading a whole micrograph per particle is 20–50× slower than stacks.
   Access must be micrograph-major, or go through a cache. Memory-mapping the
   micrograph and visiting its particles sorted by (y, x) is 4× faster than the
   same reads unsorted.

## Design

### Names: nothing downstream changes

A virtual particle keeps the ordinary RELION image name, `N@file`, but `file` is
a small **virtual stack descriptor** (`.vstack`) instead of an `.mrcs`:

    _rlnImageName   000012@Extract/job018/Movies/20170629_00021_frameImage.vstack

Every program that passes image names around, splits them at `@`, groups
particles by stack, or counts images per stack keeps working unmodified. Only
the image I/O layer (`Image::_read`) needs to know: a `.vstack` extension is
dispatched to the virtual reader before any other format test (the MRC test
matches any extension containing "st", so the order matters).

Because the descriptor pins the exact windows, particle N of a virtual stack is
as immutable as particle N of a real one — later edits to `rlnCoordinateX/Y` in
a STAR file do not move it, exactly as today.

### The descriptor

One per micrograph, written by the Extract job in place of the `.mrcs`. It is a
STAR file (so it can be read and diffed by hand):

    data_vstack
    _rlnVirtualStackVersion        1
    _rlnMicrographName             MotionCorr/job002/Movies/20170629_00021_frameImage.mrc
    _rlnVirtualMicrographChecksum  <size>:<hash of header, middle and end>
    _rlnVirtualExtractSize         360        # box cut from the micrograph
    _rlnVirtualRescaleSize         256        # --scale, or -1
    _rlnVirtualWindowSize          -1         # --window, or -1
    _rlnImageSize                  256        # final particle size
    _rlnImagePixelSize             1.244      # of the output particles
    _rlnVirtualNormalise           1
    _rlnVirtualBgRadius            71
    _rlnVirtualRampBackground      1
    _rlnVirtualWhiteDust           -1
    _rlnVirtualBlackDust           -1
    _rlnVirtualInvertContrast      1
    _rlnVirtualFloat16             1          # round like --float16 stacks

    data_particles
    loop_
    _rlnCoordinateX #1
    _rlnCoordinateY #2
    1829.000000  511.000000
    ...

The coordinates are the **whole-pixel centres the extraction windows around** —
`rlnCoordinateX/Y` after `relion_preprocess` truncates them (and after any
recentring) — not the raw coordinates. Storing the raw values would let text
round-off move a particle by a pixel: 1829.9999999 truncates to 1829, but
prints as 1830.000000.

The recipe covers plain 2D extraction, helical segments, rectangular and turned
boxes, and CTF premultiplication (below). Phase flipping and 3D sub-tomograms
are refused in virtual mode with a clear error rather than approximated; they
can be added later.

**Helical segments** normalise against the background outside a tube rather than
a circle, so the recipe also records the tube radius in pixels of the extracted
box, and each particle its in-plane angle psi (the tube's orientation). Both are
stored as `%.17g` strings, not as STAR doubles, because `%12.6f` would move the
tube mask by a rounding step and change pixels. The radius is computed by the
same helper `relion_preprocess` uses for real stacks,
`helicalBackgroundRadius()`, including its integer division when downscaling.

**CTF premultiplication** (`--premultiply_ctf`, optionally
`--premultiply_extract_size`, `--ctf_intact_first_peak`) multiplies each particle
with its CTF in a box bigger than the final one and then cuts it down. The CTF
moves high-resolution signal sideways by about λΔf·s (at 3 Å and 1.3 µm defocus
on the tutorial, ~40 Å); a plain cut loses what moved out of the box, while
multiplying in the big box brings that signal back to the particle before the
cut (`relion_refine` then weights with CTF², as for any premultiplied optics
group). The descriptor (version 3; older builds refuse it) records the box, the
flag and the micrograph pixel size in the header, and each particle's CTF as a
`%.17g` string `rlnVirtualCtf` (defocus U, V, angle, voltage, Cs, Q0, B-factor,
scale, phase shift). The reader repeats `relion_preprocess` exactly: the big box
with the micrograph average off the micrograph, times the CTF, cut to the
extraction box, then pixels off the micrograph take the nearest edge pixel.

`--premultiply_extract_size auto` (stacks and virtual particles) gives each
particle its own big box: the extraction box plus twice the largest
delocalisation |λΔf·s − Cs·λ³s³| up to the output Nyquist, for the larger of
its two defoci, rounded up to an even size with factors 2, 3 and 5 only, at
most 4× the extraction box (`vparticles::autoPremultiplySize`, shared by both).
The descriptor records `-2` as the box, and the reader computes the box from the
CTF in effect, so a defocus change from CTF refinement changes the box too.
Extraction prints the range of boxes it chose.

The CTF follows the particle STAR file. `ObservationModel::loadSafely`, which
every program uses to read particles, hands the CTF of every particle of a
premultiplied optics group to `vparticles::setCurrentCtfs()`. Where it differs
from the recorded one by more than STAR-file round-off (1e-5 + 1e-6·|value|),
the particle is computed with the new CTF, and the CTF values are part of the
cache key, so after CTF refinement the particles are rebuilt once and the old
entries simply stop being used. A real premultiplied stack cannot do this: it
keeps the CTF it was extracted with while refinement weights with the new one.
Programs that read images without a particle STAR file (`relion_image_handler`
on the descriptor) get the recorded CTF. The CTF includes the optics group's
anisotropic magnification and symmetric (even Zernike) aberrations, as
`relion_preprocess` applies them, read through the same `CTF::readByGroup` path
(`ParticleCtf::aberr`, recorded in `rlnVirtualCtf` after the 9 CTF values), so
when CTF refinement fits them the particles follow too. The auto box ignores them.
Fused extraction in `relion_reconstruct` refuses premultiplied stacks.

CTF refinement fits premultiplied virtual particles as the plain particles the
same extraction gives without premultiplication (`vparticles::readUnpremultiplied()`,
called by `relion_ctf_refine` before it reads anything; the output STAR file stays
premultiplied). On premultiplied pixels the fit is poor: on EMPIAR-10940 (box 256)
the per-particle defocus scattered 152 Å within a micrograph against 61 Å on plain
stacks, and Refine3D after CtfRefine fell from 2.67 to 2.74 Å; with the defocus
fitted on plain particles it reached 2.54 Å. `RELION_CTF_REFINE_VIRTUAL_PLAIN=0`
fits the premultiplied pixels instead. Real premultiplied stacks have no plain
version: CtfRefine multiplies their predictions by the STAR file's CTF and weights
by 1/(CTF² + 0.1) (`RELION_CTF_REFINE_PREMULT_EPS`), which helps only a little.

**Dust removal is refused permanently, not just for now.** `removeDust()`
replaces each dust pixel with `rnd_gaus(avg, stddev)` from a process-wide random
generator, so the value depends on how many random numbers that process has
drawn before - on the order particles are read, on threads, on what else ran.
The same particle would come back different on every read. (Found when the
converter, run with four threads, disagreed with itself on a dust-removed job.)
RELION's default is -1 (off), so most jobs are unaffected.

**Bitwise contract.** A virtual particle is bitwise identical to the particle
`relion_preprocess` would have written with the same options, including the
float16 rounding when the recipe says `--float16`. This is what makes the
feature safe to adopt: a pipeline run on virtual particles is numerically the
same pipeline. An integration test enforces it.

### Reading

`vparticles::readParticle(descriptor, index)`:

1. Parse the descriptor (memoised per process, thread-safe; CtfRefine reads
   particles from OpenMP threads).
2. **Cache hit:** read the one box from the cache entry with a single `pread`.
3. **Cache miss:** extract *all* of that micrograph's particles in one pass
   (memory-mapped micrograph, particles sorted by (y, x)), write the cache entry,
   serve the request. The first touch of a micrograph pays for all its
   particles; afterwards random access is as cheap as a stack.
4. **Cache disabled or unwritable:** extract just the requested particle from the
   mapped micrograph (only the pages under its box are read).

Header-only reads (`Image::read(name, false)`) return the dimensions and pixel
size from the descriptor without touching the micrograph.

Before extracting, the micrograph's checksum is compared with the one recorded
at extraction time; a mismatch is an error naming the micrograph ("changed since
extraction — re-extract"), never a silent change of particles.

### The cache

One entry per (descriptor content), i.e. per micrograph × recipe × coordinate
list: a single file holding that micrograph's processed particles back to back,
in descriptor order, behind a small header (magic, version, count, box, data
type, key). Reading an entry is sequential; reading one particle is one `pread`.

- **Key:** a hash of everything that determines the pixels (micrograph checksum,
  recipe, coordinates). A stale entry cannot be hit: change anything and the key
  changes.
- **Location**, first match wins:
  1. `RELION_VPARTICLE_CACHE=<dir>` — or `off` to disable caching;
  2. `<project>/Cache/virtual_particles/` — visible to the user, and shared by
     every node running a job of the project (RELION programs run from the
     project root).
- **Concurrency:** entries are written to a temporary name and `rename()`d into
  place, so readers only ever see complete files. Several MPI ranks may build the
  same entry at once; the last rename wins with identical content. Within a
  process, a per-descriptor lock stops threads building the same entry twice.
- **Validation:** an entry whose header or size does not match is ignored and
  rebuilt.
- **Removal:** `rm -rf` the directory at any time. Open file descriptors survive
  an unlink, and a missing entry is just a miss. RELION's clean-up removes it
  too: "Clean all jobs" (`cleanupAllJobs`) and "Remove intermediate files"
  (counted in its plan and in the GUI's confirmation).
- **Capacity:** `RELION_VPARTICLE_CACHE_MAX_GB` (default 0 = no limit). Without
  a limit the cache holds at most what the real particle stacks would, which is
  no worse than upstream, and unlike stacks it can be deleted at any time to
  free the space (the particles are rebuilt when next used). A cap smaller than
  the particles of one run is worse, because every iteration then evicts them
  and rebuilds them from the micrographs. When a limit is set and exceeded the least recently used entries are removed down to 90% of
  it; recency is refreshed once per process per entry, and the directory is only
  rescanned after every ~5% of the limit written. Dead writers' temporary files
  older than a day are removed on the way.
- **Write failures:** a failed write (full or briefly failing file system) is
  reported once and pauses writing for a minute; entries already written are
  still read meanwhile, and writing is then tried again.
- **Data type:** float16 when the recipe says so (the default for Extract),
  float32 otherwise — the same bytes a stack would hold.

Which micrographs end up cached is decided by use: every micrograph that is read
gets an entry. The pixel rule above says a *dense* micrograph could be streamed
without loss, but streaming redoes the rescale/normalise work (≈2.3 ms per
particle per thread) on every read, and a refinement reads each particle tens of
times, so caching wins whenever there is space. The pixel rule matters for
choosing between *reading orders* (phase 2), not for whether to cache.

### Writing (Extract)

`relion_preprocess --extract --virtual` does everything it does today — reads
coordinates, applies FOM thresholds and bias, recentres re-extracted particles,
writes the same `particles.star` with the same CTF and optics columns — except
that it reads only the micrograph *header*, and writes a `.vstack` instead of
cutting and writing pixels. It is metadata-only and finishes in seconds.

The GUI's Extract job has "Write virtual particles?" and "Premultiply with the
CTF?", both Yes for new jobs (`RELION_VIRTUAL_PARTICLES=0` makes No the default
for the first). Premultiplication passes `--premultiply_ctf
--premultiply_extract_size auto`: each particle is multiplied by its CTF in a box
as large as its defocus needs, then cut to the particle box. It is left off for
rectangular and rotated boxes, which `relion_preprocess` cannot premultiply. Job
files saved before these defaults (no such option in them) keep both No. On the
command line `relion_preprocess` is unchanged: virtual and premultiplied only when
asked.

### Writing (Polish): virtual polished particles

`relion_motion_refine --combine_frames --virtual` (or `RELION_VIRTUAL_POLISH=1`
as the default) estimates the motion as usual, then writes, per micrograph, a
descriptor `<mic>_shiny.vstack` instead of `<mic>_shiny.mrcs`; the movies are
not read a second time for the frame combination. The descriptor (version 4)
records what Polish combines the frames with:

- the corrected micrographs STAR file (micrograph -> motion record), the frame
  range, EER upsampling and grouping, the movie, coordinate and particle pixel
  sizes and the hot-pixel threshold (`rlnVirtualPolish`);
- the box the frames are combined in, the output pixel size and the size the
  particles are cut to (`--crop`);
- the dose weight of every frame as (sigma, scale) (`rlnVirtualFrameWeights`),
  from which `DamageHelper::computeWeights` gives the weights Polish used;
- per particle its exact coordinates (after `--recenter`), its motion track in
  pixels (`rlnVirtualTrack`, the values Polish reads back from `_tracks.star`)
  and, with `--ctf_multiply`, its CTF (`rlnVirtualCtf`, with magnification and
  even Zernike terms);
- an identity checksum of the movie, the whole motion record, and the gain
  reference and defect file that record names: if any of them has changed (a
  re-run motion correction, a new gain), reading is an error, never a
  silently different particle.

Reading a particle recombines the whole micrograph (the movie has to be read
anyway) with `FrameRecombiner::recombineMicrograph`, the function Polish itself
now uses for its stacks. From descriptor version 5 on, each particle's frame sum
is rounded to 16 bits before the CTF is applied, so that the sums can be cached at
half the size; the particles therefore equal Polish's to within that rounding
(about 5e-4 of the largest pixel value), not bit for bit. They are bit for bit the
same whether the sums come from the cache or are made from the movie, so a
virtual particle never depends on the cache
(`tests/integration/test_polish_premultiplied.py`). Version-4 descriptors keep
the unrounded sums and are Polish's bit for bit.
The recombined micrographs are kept in memory, as many as fit in a tenth of the free
memory (at least two; the size is printed once, `RELION_VPOLISH_MEMORY_GB` sets it); a
cache entry being written holds its own micrograph, so readers elsewhere cannot push
it out and force a recombination per particle. The on-disk cache holds them as it
holds extracted particles. Premultiplied polished particles follow the
CTF of their STAR file like premultiplied extracted ones, and CtfRefine reads
them plain (unmultiplied) in the same way. Fused extraction refuses them. The GUI's Polish job has "Write virtual polished
particles?" and "Premultiply with the CTF?", both Yes for new jobs
(`RELION_VIRTUAL_POLISH=0` makes No the default for the first; the job always passes
`--virtual` or `--no_virtual`, and `--ctf_multiply` when asked). Job files saved
before these defaults keep both No.

#### CTF premultiplication in Polish: window, per-particle windows, frame sums

- **Window and box, chosen by the program.** `--ctf_multiply` without `--window`,
  `--scale` or `--crop` combines the frames in a window as large as the largest
  defocus of the optics group needs (`autoPremultiplySize`, the rule
  `--premultiply_extract_size auto` uses), then crops to the particle box of the
  input; both are printed. Giving `--window`/`--crop` (or the GUI's extraction and
  re-scaled sizes) sets them by hand, e.g. a smaller final box than the input's.
- **Per-particle windows** (with `--ctf_multiply`, stacks and virtual polished
  particles alike, on by default; `RELION_VPOLISH_PARTICLE_WINDOWS=0` combines every
  particle in the maximum window): each particle is combined in a window of its
  own, as large as its own defocus needs and at most the maximum window (the one
  above, which the largest defocus needs). Less work for the lower defoci; the
  particles differ slightly from maximum-window premultiplication (correlation
  0.999 on EMPIAR-10940 at 384 px, 0.965 on the test's noise-only 64 px particles,
  where the weak CTF tails reach different noise) but keep the signal their CTF
  spreads. Polish makes them in one pass over the movie per window size. Virtual
  particles record the window per particle (`rlnVirtualPremultiplySize`,
  descriptor version 6). On EMPIAR-10940 (286 movies, windows 576-1080 px) the
  end-to-end run gave the same resolution as the maximum window,
  and a 256 px final box the same final resolution as 384 px at about a tenth of
  the refinement time.
- **Frame sums cache** (`RELION_VPOLISH_SUMS=0` makes them from the movie whenever
  needed): each particle's frame sum before the CTF, rounded to 16 bits, kept as
  one `.vpc` file per micrograph in the virtual-particle cache, so that a new CTF
  (CtfRefine) does not need the movie again. They are large: about 0.2 GB per
  micrograph with windows up to 1080 px (61-86 GB for 286 micrographs). The cache
  has no size limit unless `RELION_VPARTICLE_CACHE_MAX_GB` sets one, and can be
  deleted at any time.
- **Pre-build before the first iteration** (`relion_refine_mpi`;
  `RELION_VPARTICLE_PREBUILD=0` lets the readers build entries as they go): the
  missing cache entries are shared out over all processes, largest micrograph
  first. Each process reads the frame sums of its next micrographs in the
  background; how many at a time is chosen from the measured wait (start at one,
  one more while the wait per file falls by a fifth, at most a quarter of the
  free memory and 8; `RELION_VPOLISH_SUM_READ_AHEAD=N` fixes it, `0` turns it
  off). Process 1 prints the reads, the read-ahead depth and the time it waited.
  Measured on a network file system (65 GB of sums, cold): 630 s without
  read-ahead, about 510 s with it, the rest bounded by the disc (~125 MB/s).
- **Background cache writers**: `RELION_VPARTICLE_BUILDERS` threads (default half
  the cores a process may use, shared by the processes on a node; `0` makes
  readers write the entries themselves).

### Converting existing projects: `relion_stacks_to_virtual`

Projects extracted before this existed hold their particles in `.mrcs` stacks,
often over a terabyte per project. `relion_stacks_to_virtual` reclaims that:

    relion_stacks_to_virtual --project /path/to/project --j 16            # dry run: verify, report
    relion_stacks_to_virtual --project /path/to/project --j 16 --convert  # replace verified stacks

- **Same names.** Each stack is replaced, under its own `.mrcs` name, by a
  descriptor; the reader recognises a descriptor by its first line
  (`# RELION virtual particle stack`) whatever its extension. Downstream STAR
  files (Class2D, Select, Refine3D, ...) all name the stacks, and none of them
  has to change.
- **Nothing is replaced on trust.** The recipe comes from the last
  `relion_preprocess` command in the job's `note.txt`, the centres from the
  per-micrograph `*_extract.star`. Every particle is recomputed from its
  micrograph and compared pixel for pixel with the stack; only a stack
  reproduced exactly is replaced, atomically (`rename`), so a reader sees the
  old stack or the descriptor, never a mixture. Without `--convert` nothing is
  written. Each converted job gets `virtual_conversion.log`.
- **Old float16 stacks.** Stacks written by RELION 5.0 betas carry the float16
  rounding bug fixed upstream in `1b8adc2e` (a value that should round to a power
  of two was stored as twice that). Such stacks are reported as "identical but
  for pre-5.0 float16 pixels" and converted only with `--accept_float16_fix`;
  the particles then come back with those pixels corrected.
- **Left alone:** dust-removed, phase-flipped, CTF-premultiplied and 3D
  extractions; jobs without a `note.txt`; stacks whose micrograph is gone.
- **Verification does not fill the cache**: it computes particles directly.

On the tutorial's three Extract jobs (13,655 particles, written by a 5.0 beta):
628.5 MB of stacks became 417 KB of descriptors; a downstream Select set read
through the converted stacks differs from before only in the 3,258 pixels the
old writer had doubled.

**Before converting a shared project, every RELION build that will open it
must include this reader.** An older build reads a descriptor as an MRC stack
and stops with "unsupported MRC mode"; external programs (cryoSPARC, cryoDRGN,
...) cannot read it at all. Materialise a copy for them first.

### Getting real stacks back

`relion_stack_create --i particles.star --o out` reads every image through
`Image::read` and writes a real stack, so it already materialises virtual
particles. That is the export path for external tools (cryoSPARC, cryoDRGN) and
the thing to run before deleting micrographs.

Derived particles — subtraction output, simulations — are not windows of a
micrograph and stay real stacks (polished particles can be virtual: see above). A STAR file can mix both
kinds freely, since every name says what it is.

## What changes for the user

- Extract jobs (with the option on) hold `particles.star` plus one small
  `.vstack` per micrograph, instead of gigabytes of `.mrcs`.
- Re-extraction and recentring cost seconds and no storage.
- **The micrographs must stay — until phase 5.** A particle set depends on its
  micrographs the way it used to depend on its stacks. Removing a micrograph
  makes its particles unreadable (with an error naming the file) unless they
  were cached or materialised. Phase 5 lifts this by regenerating micrographs
  from the movies.
- A cache directory appears (`Cache/virtual_particles/` in the project by
  default). It is safe to delete, and RELION's clean-up tools delete it.

## Phases

1. **Virtual stacks, reader, Extract option, cache** — implemented. See
   "Status" below.
2. **Micrograph-major reading in `relion_refine`.** Order each rank's particles by
   micrograph within an iteration so cold reads stream micrographs rather than
   seeking. Open question to test: whether VDAM mini-batches drawn from whole
   micrographs are statistically acceptable, or whether mini-batches should keep
   random particles and rely on the cache.
3. **Cache management** alongside the existing stack cache: size limit and LRU
   eviction and removal by the clean-up tools are done; a GUI entry under
   "Manage cache" is still open.
4. **Recipe extensions**: helical segments and CTF premultiplication (with the
   CTF following the STAR file) are done; phase flip is open.
5. **Regenerable micrographs: micrographs become an accelerator too.**
   Implemented as virtual movie averages (`RELION_VIRTUAL_MOVIE_AVERAGES`):
   see `virtual_movie_averages.md`. A
   micrograph is derived from its movie and the motion-correction metadata
   RELION records per micrograph (global shifts and the local polynomial model,
   the same metadata the CryoSPARC importer writes). When a micrograph is
   missing, the reader regenerates it from the movie instead of failing; users
   can then delete micrographs as freely as the particle cache.

   The requirement that keeps the guarantees intact: **regeneration must be
   deterministic** — applying the recorded motion with a fixed summation order
   and the same float16 rounding, so a regenerated micrograph is bitwise
   identical to the one it replaces. Then the recorded checksum still matches,
   particles stay bitwise identical, and nothing downstream can tell. RELION's
   own motion correction sums frames across threads, so this needs its own
   deterministic apply path rather than re-running `relion_motioncorr`, and has
   to be verified against micrographs RELION wrote. Where bitwise identity is
   not achievable (micrographs from another program), the checksum check needs
   an explicit "regenerated" state: particles are then equivalent rather than
   identical, and their cache entries are keyed by the new checksum.

## Status (phase 1)

Implemented and tested:

- `relion_preprocess --virtual` / `--no_virtual`, default from
  `RELION_VIRTUAL_PARTICLES`; GUI Extract job option "Write virtual particles?".
- `.vstack` reading everywhere through `Image::read`, including header-only reads,
  whole-stack reads and OpenMP-parallel loaders (CtfRefine).
- The cache, with the location rules, size limit and LRU eviction above;
  removed by "Clean all jobs" and "Remove intermediate files".
- Helical segments, in extraction and conversion.
- `relion_stacks_to_virtual`, converting existing Extract jobs in place.
- CTF premultiplication in a big box, recomputed when the CTF in the particle
  STAR file changes.
- Refusal of unsupported recipes (phase flip, dust removal, 3D,
  `--operate_on`) and of writes to a `.vstack`.

Measured on the RELION 5 tutorial re-extraction (Extract/job018: 4,452
particles, 360 → 256 px, float16):

| | stacks | virtual |
|---|---|---|
| Extract job | 64 s | **1 s** |
| job directory | 565 MB | **7.6 MB** (the STAR files) |
| first full read (builds the cache) | 14.5 s | 34.7 s |
| later full reads | 14.5 s | 14.2 s |
| pixels | — | **bitwise identical**, all 4,452 particles |

The full tutorial pipeline (motion correction to the polished Refine3D), run
once with stacks and once with virtual particles, gives the same PostProcess
resolutions to every printed digit: 3.122, 3.033 and 2.922 Å.

Dry runs of `relion_stacks_to_virtual` on two real helical projects (234 and
211 Extract jobs, 5.0 and 7.4 TB; 3 stacks per job recomputed and compared)
found every stack either reproduced exactly or differing only by the old
float16 bug, except in jobs whose micrographs were rewritten after extraction.
No real project has yet been converted and refined afterwards.

Tests: `tests/unit/test_virtual_particles.cpp` (reader and cache: miss/hit/direct
agreement, rounding, deleted and truncated cache entries, changed micrographs,
header-only reads, refused writes, mapped vs conventional micrographs, parallel
reads, unwritable cache) and `tests/integration/test_virtual_particles.py`
(`relion_preprocess` both ways on synthetic micrographs — float32 and float16
micrographs, three recipes, particles off every edge, a recentred
re-extraction, helical segments at full size and downscaled — compared pixel
for pixel; refusals; the cache removed by clean-up; the converter end to end,
including skipped, damaged and helical jobs).

## Code map

| piece | where |
|---|---|
| descriptor, recipe, extraction, cache | `src/virtual_particles.{h,cpp}` |
| dispatch of `.vstack` in image reads | `Image::_read` in `src/image.h` |
| new STAR labels | `src/metadata_label.{h,cpp}` (`rlnVirtual*`) |
| `--virtual` in extraction | `src/preprocessing.{h,cpp}` |
| GUI option | Extract job in `src/pipeline_jobs.cpp`, `src/gui_jobwindow.cpp` |
| converter | `src/apps/stacks_to_virtual.cpp` |
| cache removal by clean-up | `src/pipeliner.cpp`, `src/remove_intermediates.{h,cpp}`, `src/gui_projects.cpp` |
| tests | `tests/unit/test_virtual_particles.cpp`, `tests/integration/test_virtual_particles.py` |
