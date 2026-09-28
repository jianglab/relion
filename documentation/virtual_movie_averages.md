# Virtual movie averages — `RELION_VIRTUAL_MOVIE_AVERAGES` (WIP)

Builds on virtual particles (`virtual_particles.md`). Phase 5
of `virtual_particles.md`: motion-corrected micrographs become an accelerator,
computed from the movies, instead of data that must be kept. With virtual
particles as well, nothing between the movies and the particle metadata has to
be stored.

## Why the existing micrographs cannot be regenerated

Measured on a real project (RELION 5.0 own motion correction, 5760×4092×50
TIFF movies, float16 output): re-summing a movie with the motion recorded in its
per-micrograph STAR file gives a micrograph with correlation 0.9996–0.99999 to
the stored one, but only 89–95 % of pixels are bitwise identical (RMS difference
0.5–3 % of the pixel standard deviation — far below the noise, but it breaks
anything keyed on a checksum, such as virtual particles). The causes, all in how
the sum is made and recorded, not in the movie:

1. **The record is rounded.** STAR doubles are written in 12 columns, so the
   local-motion polynomial keeps 3–4 significant digits (`-0.00206`), and these
   are multiplied by frame² and frame³ (≈2,400 and ≈118,000 at frame 49). The
   dominant cause, ~0.01 px everywhere.
2. **Hot pixels are filled randomly**: each one, in every frame, by
   `rand() % n` (called from OpenMP threads, so not even reproducible run to
   run) or `rnd_gaus()`. The fill values are not recorded.
3. **Global shifts are applied in steps** — once per alignment iteration — which
   rounds differently from one shift by the recorded total.
4. **FFT plans depend on array alignment.** `NewFFT`'s ad-hoc plans use SIMD
   codelets only when the arrays happen to be aligned, so the bits of a
   transform can depend on `malloc`.

## What the switch does

Opt-in, RELION's own motion correction only (with `--use_motioncor2` it is an
error, not a silent fallback):

| `RELION_VIRTUAL_MOVIE_AVERAGES` | MotionCorr writes, under each micrograph's name |
|---|---|
| unset / `0` | the micrograph, as before (legacy, not reproducible) |
| `1` | a **descriptor**: the motion record behind the line `# RELION virtual movie average` (≈100 KB instead of ≈27–47 MB) |
| `real` | the real micrograph, made the reproducible way (it can be replaced by a descriptor later) |

Either way alignment and power spectra are unchanged, except that hot pixels
are filled deterministically, and the micrograph is **defined** as what
`MotioncorrRunner::sumFromRecord()` makes of the movie and the record written
next to it:

- the record is written with exact doubles (`%.17g`: global shifts, local
  model; `MetaDataTable::setExactDoubles`), and carries
  `rlnMicrographSumRecipe` — e.g.
  `version=1 dose_weighted=1 early_binning=0 fix_defects=1` — plus the dose per
  frame that was actually used;
- bad pixels (defect map, dead gain pixels, recorded hot pixels) are filled with
  a neighbour chosen by a hash of (pixel, frame); frame statistics for isolated
  clusters are summed sequentially, so nothing depends on the thread count;
- the movie is read again, the recorded global shift applied in one Fourier
  step, then dose weighting, the local model, binning, all through FFTW plans
  made by size with `FFTW_UNALIGNED`.

The recipe (`rlnMicrographSumRecipe`) also records the micrograph's size and
whether it is stored in float16, so a header-only read needs no movie.

### Reading a virtual movie average

`Image::read` recognises a descriptor by its first line, whatever the file is
called, as it does for converted particle stacks:

- **header only** (size, pixel size, data type): from the record, no movie read.
  Writing virtual particles from virtual micrographs therefore reads no movie
  at all;
- **pixels**: from a real MRC file in the cache, built from the movie on first
  use (tmp file + `rename`, so concurrent MPI ranks are safe) — bit for bit the
  micrograph MotionCorr would have written, float16 rounding included;
- the virtual-particle micrograph reader memory-maps that cached file.

Cache, named and placed like the particle cache:

| | particles | movie averages |
|---|---|---|
| switch | `RELION_VIRTUAL_PARTICLES` | `RELION_VIRTUAL_MOVIE_AVERAGES` |
| cache location override (or `off`) | `RELION_VPARTICLE_CACHE` | `RELION_VMOVIE_AVERAGE_CACHE` |
| default location | `Cache/virtual_particles/` | `Cache/virtual_movie_averages/` |
| size limit, LRU | `RELION_VPARTICLE_CACHE_MAX_GB` (100) | `RELION_VMOVIE_AVERAGE_CACHE_MAX_GB` (100) |

"Clean all jobs" and "Remove intermediate files" delete both caches. With the
cache off, micrographs are computed in memory on every read.

Regenerating a real file explicitly (e.g. for an external program):

    relion_regenerate_micrograph --i MotionCorr/job002/Movies/x.mrc --o x_real.mrc --j 16 [--float16] [--compare other.mrc]

`--i` takes the descriptor or the `.star` record. A record without a recipe
(every micrograph made before this, or with the switch off) is refused with an
explanation, never approximated.

**External programs cannot read descriptors.** RELION's own jobs read
micrographs through `Image::read` (AutoPick LoG and template picking, Topaz —
RELION preprocesses its micrographs — ManualPick, Extract, Polish). CTFFIND
reads the file itself: use the power spectra from motion correction
(`--use_given_ps`); without it CtfFind stops with an explanation. For
anything else (crYOLO, cryoSPARC, ...), regenerate real files first.

Cost with `=real`: the movie is read twice (once to align, once to sum), and the
summing FFTs run without SIMD; +25 % motion-correction time on real movies.
With `=1` the final sum is skipped altogether at motion correction and paid on
first read instead (≈12 s per 5760×4092×50 movie). Memory does not grow: the
aligned frames are released before the second read.

Not covered: the non-dose-weighted `_noDW` sum written *alongside* a
dose-weighted one (`--save_noDW`) and the odd/even sums keep the legacy path;
only the main micrograph is reproducible.

## Status

Implemented and tested:

- **Tutorial movies** (3 × 3710×3838×24 TIFF, gain reference): five
  configurations — float16 and float32, with and without dose weighting,
  binning 1.25 with early and late binning — each regenerated at 1, 7 and 16
  threads: **45 of 45 bitwise identical** to the micrograph MotionCorr wrote.
- **Real movies** (5760×4092×50 TIFF from a 2024 project): regenerated
  micrographs bitwise identical. MotionCorr takes ~25 s per movie with the
  switch against ~20 s without (+25 %, 16 threads); regeneration takes 12.3 s.
- The output does not depend on MotionCorr's own thread count either (1 vs 5
  threads give the same pixels; only the time stamp in the MRC header label
  differs).
- A legacy record is refused, and the legacy path is unchanged: with the switch
  off, records are written in 12 columns exactly as before.
- `tests/integration/test_virtual_movie_averages.py` (synthetic movies with a
  dead gain pixel and hot pixels): the `=real` cases above plus the refusal;
  with `=1`, descriptors read through RELION equal the real micrographs bit for
  bit (from the cache, after deleting it, with the cache off), and particles
  extracted from them — real or virtual — equal those extracted from the real
  micrographs; writing virtual particles reads no movie.
- End to end on the RELION tutorial: see below.

### End to end: the RELION 5 tutorial

The whole tutorial — motion correction, CTFFIND (`--use_given_ps`), LoG
picking, Topaz training and picking, Class2D ×2, class ranking, InitialModel,
Class3D, Refine3D, CtfRefine, Polish, Refine3D, PostProcess — with the
tutorial's parameters, in two arms on the same GPU node:

- **vall**: `RELION_VIRTUAL_MOVIE_AVERAGES=1` and virtual particles: no
  micrograph and no particle stack stored;
- **vreal**: the same micrographs written as real files (`=real`) and real
  stacks (Topaz reuses the model vall trained, since GPU training is not
  deterministic).

| | vall | vreal |
|---|---|---|
| all 24 micrographs, read through RELION | bitwise identical to vreal's files | |
| CTF fits, LoG picks, Topaz picks | identical files | |
| particles of Extract/job007 and job012 (38.8 MB, 90.8 MB) | bitwise identical | |
| PostProcess job021 / job026 / job030 | 3.153 / 3.062 / 3.004 Å | 3.153 / 3.062 / 2.976 Å |
| MotionCorr + Extract stored | **55 MB** | 1,223 MB |
| caches (disposable) | 693 MB movie averages, 652 MB particles | — |
| MotionCorr / each Extract | 230 s / 1–3 s | 363 s / 49–50 s |

Everything up to the first GPU refinement is bit for bit the same. From Class2D
on, the arms differ in the last digits (class fractions 0.021852 vs 0.021853 in
the first Class2D) and drift apart as noise accumulates: GPU `relion_refine` is
not bitwise deterministic — the earlier stack-vs-virtual-particles comparison
shows the same line-for-line differences in every model file, even where its
final resolutions happened to agree. The 0.03 Å spread at job030 is that noise.
(The tutorial's own micrographs, picked with its own Topaz model, reached
3.12 / 3.03 / 2.92 Å; this run trains its own Topaz model and so picks
differently.)

Two environment problems, not RELION's, had to be worked around for Topaz in
the `relion-5.0` env: it imports `pkg_resources` (gone from setuptools ≥ 81),
and PyTorch ≥ 2.6 refuses its pickled models (`weights_only`). Without the
workarounds RELION's AutoPick only warns, and writes empty picks.

The fix that mattered most after rounding the record: `--dose_per_frame` is
parsed in single precision (`textToFloat`), so the dose used was 1.27699995…,
while the old record printed `1.277000`. Every double in a reproducible record,
including the `general` table, is therefore written exactly.

## Next

- **Converting `=real` micrographs** to descriptors in place (after checking
  that each regenerates bit for bit), like `relion_stacks_to_virtual`.
- **Picking images from movies.** Picking and display need only a binned,
  low-pass image: crop each shifted frame in Fourier space before summing
  (measured 6–8 s per 5760×4092×50 movie, most of it the movie read and FFT).
  They are under 1 MB, so they are cached rather than regenerated per run.
- **Prefetch in ManualPick.** The picker knows the order it visits micrographs
  in; while the user picks one, a background thread builds the picking image of
  the next (or next few), so moving on never waits the 6–8 s (or 10–20 s for a
  full micrograph). Built images stay in the cache, so going back is instant.
- **Local CTF / tilt fitting from tile power spectra** computed from frame
  groups, like the whole-micrograph `_PS.mrc` used for global fits today.
- **The movies become the only copy** of the data once micrographs and particle
  stacks are virtual: they need verified replication before anything derived
  from them is deleted (the real-project scan found files lost to MooseFS chunk
  loss).
