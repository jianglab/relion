# Interpolation survey and a plan to extend FINUFFT to `Projector::project()`

This note records a survey of where RELION uses low-order (linear/trilinear)
interpolation for image resampling, and lays out a concrete plan for
extending the existing local FINUFFT integration (`relion_finufft.h/.cpp`,
`spatial_frequency_grid.h/.cpp`) to cover the one major gap it currently
does not reach: central-slice extraction from the 3D Fourier volume in
`Projector::project()`, used on every orientation-search iteration in 2D
classification, 3D classification, and 3D auto-refine.

The existing FINUFFT/`s2` work (author: Wen Jiang) is a **local addition on
top of upstream RELION**, not something upstream ships. It is currently
wired only into `reconstructor.cpp` and `apps/movie_reconstruct.cpp` — i.e.
the standalone reconstruction path — and targets a different (related but
distinct) problem: CTF-oscillation undersampling near Nyquist at high
defocus, via an adaptive "s2" sampling grid keyed to CTF oscillation rate.

## 1. Survey: where interpolation happens

| Area | File(s) | Scheme | Notes |
|---|---|---|---|
| Global motion correction | `motioncorr_runner.cpp` (`shiftNonSquareImageInFourierTransform`) | **Exact** Fourier phase shift | No loss. |
| Local/patch motion correction | `motioncorr_runner.cpp` (`realSpaceInterpolation_ThirdOrderPolynomial`) | Bilinear gather | Measurable "interpolation tax" near the resampling grid's own Nyquist; validated this session (see `stella` repo, `configs/relion_counting_movie_local.yaml`, ROUND 4-6). |
| Bayesian Polishing | `jaz/single_particle/motion/motion_estimator.cpp:635` (`shiftImageInFourierTransform`) | **Exact** Fourier phase shift | No loss — confirmed by reading the call site; per-particle motion tracks are applied as an exact shift, not spatial resampling. |
| **2D classification, 3D classification, 3D auto-refine (core EM engine)** | `ml_optimiser.cpp` lines ~6869, 7562, 7569, 7903 (`mymodel.PPref[iclass].get2DFourierTransform(Fref, A)`) → `projector.cpp` `Projector::project()` / `rotate2D()` | **Trilinear** (bilinear for the `ref_dim==2` in-plane-rotation case), on the `padding_factor=2` grid, with `sinc`/`sinc^2` gridding pre-correction | Paid on **every** orientation-search iteration for **every** particle, not just at final reconstruction. This is the main gap — see §2. |
| Final reconstruction (backprojection) | `backprojector.cpp`, `griddingCorrect()` | Trilinear, `padding_factor_3d=2`, analytic `sinc^2` roll-off correction | Deterministic-roll-off only; doesn't address noise amplification near the padded grid's own Nyquist. |
| Template-based particle picking | `autopicker.cpp:526,917` (`Projector(..., TRILINEAR, ...)`) | Trilinear | Projects 3D reference to generate picking templates. |
| Particle subtraction | `particle_subtractor.cpp` (sets up `Projector`s from the model) | Trilinear | Same central-slice mechanism as `ml_optimiser.cpp`. |
| Helical reconstruction | `helix_inimodel2d.cpp:689,1010,1060` (`Projector`/`BackProjector` with `TRILINEAR`); `helix.cpp:203-232` (raw `LIN_INTERP`) | Trilinear | Real-space volume interpolation for helical symmetry search/averaging. |
| Cryo-ET / subtomogram averaging, real-space backprojection | `jaz/tomography/projection/real_backprojection.h:40` | `enum InterpolationType {Linear, Cubic}` — **cubic option exists** | The one place with a built-in non-linear alternative already. |
| Cryo-ET / subtomogram averaging, Fourier-domain forward/central-slice projection | `jaz/tomography/projection/fwd_projection.h:111,152,197,265` (`Interpolation::linearXYZ_FftwHalf_complex`) | **Linear only**, no cubic/NUFFT option | The tomography analog of `Projector::project()`; same gap as SPA, and not even offered a cubic fallback. |
| General interpolation library | `jaz/single_particle/interpolation.cpp` (`linearXY`, `linearFFTW2D`, `linear3D`, `linearFFTW3D`, `cubic1D`, `cubicXY`) and `jaz/single_particle/resampling_helper.h` (`upsample2D_linear`/`upsample2D_cubic`/`subsample2D_cubic`) | Both linear and cubic primitives exist | Cubic is available as a library primitive but is **not** wired into the main SPA `Projector`/`BackProjector`/`ml_optimiser` path. |
| Existing local FINUFFT/`s2` work | `relion_finufft.h/.cpp`, `spatial_frequency_grid.h/.cpp` | FINUFFT type-2, **real-space image → nonuniform Fourier samples** (2D) | Targets CTF-oscillation undersampling in `reconstructor.cpp`/`movie_reconstruct.cpp` only. Does not touch `projector.cpp` or `ml_optimiser.cpp`. |

## 2. The gap: `Projector::project()`

```cpp
// projector.cpp, current implementation (abbreviated)
void Projector::project(MultidimArray<Complex> &f2d, Matrix2D<RFLOAT> &A)
{
    Matrix2D<RFLOAT> Ainv = A.inv();
    Ainv *= (RFLOAT)padding_factor;
    for (int i = 0; i < YSIZE(f2d); i++)
        for (int x = 0; ...; x++)
        {
            // rotate output pixel (x,y) into 3D Fourier coords via Ainv
            RFLOAT xp = Ainv(0,0)*x + Ainv(0,1)*y;
            RFLOAT yp = Ainv(1,0)*x + Ainv(1,1)*y;
            RFLOAT zp = Ainv(2,0)*x + Ainv(2,1)*y;
            // trilinearly interpolate `data` (the padded, half-Hermitian
            // 3D Fourier volume) at (xp, yp, zp)
            ...
        }
}
```

This is, structurally, exactly a type-2 NUFFT problem: nonuniform query
points `(xp, yp, zp)` (all lying on a plane through the origin, one plane
per requested orientation `A`) evaluated against data on a uniform 3D
Fourier grid (`data`). Mathematically, treating `data[k1,k2,k3]` as
Fourier-series coefficients and `(xp,yp,zp)` as the (fractional, radian-
scaled) evaluation points, FINUFFT's type-2 transform computes the ideal
band-limited trigonometric-polynomial interpolant at those points — exactly
the operation that trilinear interpolation approximates, but exact instead
of a leaky, low-order stencil. This is the same 2D correlate already
validated this session in the `stella` project (`finufft.nufft2d2` vs
bilinear `_bilinear_gather`, corr 0.9998 against exact Fourier-shift ground
truth, interpolation tax eliminated) — see `stella` repo
`configs/relion_counting_movie_local.yaml` ROUND 4.

This is a **different role for FINUFFT** than the existing `s2` work:
`evaluateNonuniformFourierSamples2D()` goes real-space image → nonuniform
Fourier samples (2D). What `project()` needs is Fourier volume →
nonuniform Fourier samples on a plane (3D), i.e. a new function, not a
reuse of the existing one.

### Complication: Hermitian half-volume storage

`data` only stores the `xp >= 0` half of the Fourier volume (Hermitian
symmetry); `project()` handles `xp < 0` points by negating and conjugating.
A generic `finufft3d2` call expects a full mode array. Two options for the
coding agent to weigh, not decided here:

1. **Split-and-conjugate** (mirrors current logic): partition query points
   into `xp>=0` and `xp<0` groups, negate the latter, run two `nufft3d2`
   (or one batched "many" call with both point sets concatenated and
   flagged), conjugate the second group's results. No extra memory.
2. **Materialize the full Hermitian-symmetric volume once per class per
   refinement iteration**, then run a single `nufft3d2` call per batch of
   orientations. Roughly doubles the memory of `data` for that class
   (e.g. ~3 GB for a 720^3 padded box in double precision) but simplifies
   the call and may let FINUFFT's internal plan/spreader setup be reused
   more cheaply across many orientations.

Option 1 is very likely preferable on memory grounds for typical
refinement box sizes; the coding agent should benchmark both before
committing.

### Batching recommendation

`ml_optimiser.cpp` calls `get2DFourierTransform()` once per (particle,
orientation) pair, for every orientation in the sampled grid, for every
particle, every iteration. FINUFFT's per-call setup cost (spreader/plan
build) is amortized by evaluating **many nonuniform points against the same
mode array in one call** (`finufft3d2many` / the vectorized interface).
Since all orientations tested for a given class in a given iteration share
the same `data` volume, the natural design is: **for a given class,
flatten the query points from many orientations (or all orientations in
the current sampling grid) into one point list, and issue one batched
`nufft3d2many` call**, rather than one `finufft3d2` call per orientation.
The coding agent must read the actual loop nesting around
`ml_optimiser.cpp:6869,7562,7569,7903` before implementing this, to
determine what's already grouped per-class vs. per-particle in the current
code, and restructure the minimum necessary to enable batching without a
wholesale rewrite of the optimiser's control flow.

## 3. Implementation plan for a coding agent

1. **Add a new function to `relion_finufft.h`/`.cpp`**, e.g.
   `evaluateNonuniformFourierSamplesFromFourierVolume3D()`, guarded by
   `#ifdef RELION_USE_FINUFFT` like the existing code. Signature sketch:
   ```cpp
   void evaluateNonuniformFourierSamplesFromFourierVolume3D(
       const MultidimArray<Complex>& data,      // Projector::data, half-Hermitian
       const std::vector<RFLOAT>& xp,           // nonuniform query coords (padded/scaled units)
       const std::vector<RFLOAT>& yp,
       const std::vector<RFLOAT>& zp,
       std::vector<Complex>& samples_out);
   ```
   Internally: split by sign of `xp` per §2 option 1, call `FINUFFT_MAKEPLAN`
   (type 2, 3D) + `FINUFFT_SETPTS` + `FINUFFT_EXECUTE`, conjugate the
   negative-`xp` group's outputs, recombine in original order. Follow the
   existing `FinufftPlanGuard` pattern already used in this header for
   plan lifetime management.

2. **Add a new interpolator constant**, e.g. `#define FINUFFT 3` in
   `projector.h` alongside `NEAREST_NEIGHBOUR`/`TRILINEAR`/`CONVOLUTE_BLOB`,
   and a corresponding branch in `Projector::project()` (and `rotate2D()`
   for the `ref_dim==2` case, and `rotate3D()` for the "rotate a 3D map"
   case if in scope) that builds the `xp,yp,zp` arrays already computed
   per-pixel today and calls the new FINUFFT path instead of the inline
   trilinear stencil, when `interpolator == FINUFFT`.

3. **Wire a CLI flag** to select it, e.g. extend the existing
   `--spatial_frequency_mode` option (already read in `reconstructor.cpp`/
   `movie_reconstruct.cpp` — check its argument parsing there and in
   `ml_optimiser.cpp`'s own option parsing) or add a new
   `--projector_finufft` flag threaded through to wherever `Projector`
   objects are constructed with `TRILINEAR` today (`ml_optimiser.cpp`'s
   `PPref` initialization, `autopicker.cpp`, `particle_subtractor.cpp`).
   Default should remain `TRILINEAR` (off) until validated.

4. **Batch the call sites** per the recommendation in §2 — this is the
   part requiring the most judgment; read the surrounding loops at
   `ml_optimiser.cpp:6869,7562,7569,7903` first and restructure minimally.

5. **Extend `tests/unit/test_relion_finufft.cpp`** with a unit test for
   the new 3D function: build a small synthetic Fourier volume (e.g. a
   known analytic object with a closed-form Fourier transform, or a
   discrete random volume compared against a brute-force DFT-sum ground
   truth at a handful of query points), and check the FINUFFT path
   matches to near machine precision — mirroring the existing 2D test's
   structure.

## 4. Validation plan (mirrors the methodology already proven in the `stella` project)

Before/alongside merging, validate using the same two-stage approach used
this session for the 2D local-motion-correction case (full details and
exact numbers in the `stella` repo's
`configs/relion_counting_movie_local.yaml`, ROUND 4-6):

1. **Isolated interpolation-quality test**: build a synthetic or real 3D
   Fourier volume, extract central slices at various rotations with both
   `TRILINEAR` and the new `FINUFFT` path, and compare radial power spectra
   against a slice extracted by a brute-force/ground-truth method (e.g. a
   direct DFT sum, or resampling a much higher-resolution reference volume
   and comparing). Confirm the trilinear path shows the same near-Nyquist
   power loss pattern already found for 2D bilinear gather, and that
   FINUFFT eliminates it.

2. **Real-project resolution/accuracy test, with a genuinely independent
   comparison** — critically, avoid the confound already caught and fixed
   this session: reusing one interpolation scheme's refined orientations
   to score another scheme will bias the comparison. Instead run two fully
   independent `--auto_refine` (or `--tomo` equivalent) jobs from the same
   starting reference/particles, one with `TRILINEAR`, one with `FINUFFT`,
   each doing its own fresh orientation search, then compare final
   `PostProcess` resolution and per-shell FSC. Use the RELION tutorial
   dataset at `/home/jiang/jiang12/data/relion/relion_tutorial/Tutorial5.0`
   as a first pass (same dataset already used this session for the 2D
   downstream-resolution test), given box sizes there are small enough for
   fast iteration; then re-test on a near-atomic-resolution dataset where
   the padded grid's Nyquist is more likely to matter, per the "proximity
   to the resampling grid's own Nyquist is the real risk criterion"
   correction established this session.

3. Track wall-clock/memory cost per iteration for both interpolator
   options at realistic box sizes, since this is a per-iteration,
   per-particle, per-orientation cost — the batching design in §2 directly
   determines whether this is viable in production.

## 5. Open questions left for the coding agent

- Whether to also add the `Cubic` option already available in
  `jaz/tomography/projection/real_backprojection.h` to
  `fwd_projection.h`'s Fourier-domain path (a cheaper partial fix, in case
  FINUFFT proves too costly for cryo-ET's typically much larger tilt-series
  data volumes).
- Whether `rotate2D()`/`rotate3D()` (mere in-plane rotation / 3D-map
  rotation, as opposed to `project()`'s dimensionality-reducing central-
  slice extraction) warrant the same treatment — likely yes for
  consistency, but lower priority since they're not on the main
  classification/refinement hot path in the same way.
- Precision mode: `relion_finufft.h` already branches on
  `RELION_SINGLE_PRECISION` for the plan/execute function pointers; the new
  3D function must follow the same pattern.

---

# Implementation notes (as built)

This section records what was actually implemented, and where it departs from
the plan above.  The feature is **opt-in**: see §0.

## 0. The switch: `RELION_INTERPOLATION`

Contrary to §3.3 of the plan the switch is an **environment variable**, not a
command-line flag:

```sh
# classic trilinear / bilinear interpolation (the default - nothing to set)
relion_refine ...

# exact NUFFT central slices
RELION_INTERPOLATION=nufft relion_refine ...
```

Accepted values are `linear` / `trilinear` / `bilinear` (the default, also used
when the variable is unset) and `nufft` / `finufft`.  Anything else is an error.
It applies to `relion_refine`, `relion_autopick` and `relion_particle_subtract` -
i.e. everywhere the plan's §1 survey found a forward `Projector` on the main SPA
path.  **Backprojection is not affected** and always uses `TRILINEAR`.

**The default is deliberately the old behaviour.**  The NUFFT path is much more
accurate (§5) but it is new, CPU-only and appreciably slower, so nothing changes
for an end user who does not opt in.  The intention is to make it the default
once a few experts have exercised it on real projects and the results hold up;
that is a one-line change in `Projector::resolveForwardInterpolator()`.

Because NUFFT is only ever selected explicitly, anything that prevents it from
running is reported as an **error** rather than a silent fallback - a build
without `-DRELION_USE_FINUFFT=ON`, or a combination with `--gpu` / `--sycl` /
`--cpu` (the NUFFT path is CPU-only).  That way an expert testing the new path is
never misled into thinking they measured it when they did not.

Tuning knobs, each settable from the environment (site-wide default) or the
command line (per job, and the command line wins):

| Environment | Option | Default | Meaning |
|---|---|---|---|
| `RELION_FINUFFT_MODE_CROP` | `--finufft_mode_crop` | `1.0` | mode-array size as a multiple of the box size (§3) |
| `RELION_FINUFFT_TOL` | `--finufft_tol` | `1e-6` | requested relative accuracy |
| `RELION_FINUFFT_UPSAMPFAC` | - | `1.25` | FINUFFT internal upsampling factor; `0` lets FINUFFT choose |
| `RELION_FINUFFT_BATCH_MB` | `--finufft_batch_mb` | `512` | per-thread memory budget for the pre-projected slice batch (§4) |
| `RELION_FINUFFT_CACHE_MB` | `--finufft_cache_mb` | `4096` | budget for the shared once-per-iteration projection cache (§4); `0` disables it |

## 1. Correction to §2/§3: the mode array is the *real-space* volume

The plan proposed feeding `Projector::data` to `finufft3d2` as the mode array,
and then worried at length about the half-Hermitian storage ("split-and-
conjugate" vs. "materialise the full volume").  That framing is wrong, and the
complication it creates is an artefact of the framing.

A FINUFFT type-2 transform computes

```
c_j = sum_n f[n] exp(i s n.x_j)
```

so feeding `data` as `f` evaluates the *inverse DFT* of `data` at nonuniform
positions - i.e. the real-space volume - not an interpolant of `data`.  At a
grid node `x_j = 2*pi*n0/N` it returns `V[n0]`, not `data[n0]`, so it is not
interpolatory and is not the quantity `project()` needs.

The band-limited interpolant of a sampled function is obtained by transforming
to the *conjugate* domain first.  Writing `P` for the size of the periodic
real-space box that the Fourier grid samples (`padoridim`):

```
g(q) = (1/P^dim) * sum_r V[r] exp(-2*pi*i * q.r / P),
V[r] =             sum_n data[n] exp(+2*pi*i * n.r / P)
```

`V` is the unnormalised inverse DFT of `data`, which is exactly the padded
real-space volume, and `g(n0) = data[n0]` at every grid node, as an interpolant
must.  So the mode array is `V`, and `q` (RELION's `xp, yp, zp`) supplies the
nonuniform points, scaled to radians by `2*pi/P`.

The pay-off is that §2's "complication" disappears: **`V` is real**, so the
interpolant automatically satisfies `g(-q) = conj(g(q))`.  Neither of the two
options weighed in §2 is needed - no splitting of the query points by the sign
of `xp`, no materialising a full Hermitian volume.  `Projector::project()`'s
`is_neg_x` branch simply has no counterpart in the FINUFFT path.

## 2. Gridding correction must be switched off

`computeFourierTransformMap()` divides the real-space map by `sinc^2` before
padding, to pre-compensate for the transfer function of the *trilinear*
interpolation kernel.  Exact interpolation convolves with nothing, so there is
no transfer function to divide out and applying the correction would leave the
reference sharpened by an unmatched `1/sinc^2`.  Both `computeFourierTransformMap()`
and `griddingCorrect()` therefore skip the correction when
`interpolator == FINUFFT`.

This means the FINUFFT and trilinear projectors build *different* `data` arrays
from the same input map.  That is intended, and it is why a fair comparison
requires building two separate `Projector`s from the same map rather than
sharing one.

## 3. Mode-array cropping

Because FINUFFT does its own internal upsampling, `V` does not have to be kept
at the full padded size `P`.  `V` is supported inside the original unpadded box
(up to the ringing of the spherical band-limit that `computeFourierTransformMap()`
applies to `data`), so it is cropped to `mode_size = finufft_mode_crop * ori_size`
samples per dimension, rounded up to even and clamped to `[4, P]`.

`--finufft_mode_crop` defaults to `1.0` (keep exactly the unpadded box).  This
matters a lot for cost, not just for memory: FINUFFT's internal FFT is
`(upsampfac * mode_size)^dim`, so going from a crop of 1.0 to 2.0 makes that FFT
**64x** larger in 3D.  Use `2.0` when you want the interpolation of `data` to be
exact to FINUFFT's tolerance - the unit tests do, to assert that the projector
is interpolatory at grid nodes.

Memory for the mode array is `mode_size^dim * 16` bytes per class (complex
double): 32 MB per class at box 128, 256 MB at box 256, 2 GB at box 512.

## 4. Batching

`Projector::get2DFourierTransformMany()` evaluates many orientations against the
same mode array in a single FINUFFT call.  This is not an optimisation, it is a
precondition for the feature being usable: a type-2 call costs one FFT of
FINUFFT's internal upsampled grid **however few points it evaluates**, so an
unbatched `project()` pays an `(2*ori_size)^3` FFT per slice, which is roughly
three orders of magnitude more work than the trilinear stencil it replaces.

### The shared once-per-iteration cache

The batching above still repeats every projection for every particle.  But under
a **global angular search** the orientation that a given `(idir, ipsi, iover_rot)`
index stands for is the same for every particle, and the only other thing the
projection depends on is the optics group - so the entire set of reference slices
can be computed **once per iteration** and reused by every particle.  That is what
makes the NUFFT projector affordable: its cost stops scaling with the number of
particles.

`MlOptimiser::getCachedReferenceProjections()` owns this, keyed on
`(iclass, optics_group, pass)` and rebuilt whenever `iter` changes.  Both passes
of `getAllSquaredDifferences()` and the projection loop in `storeWeightedSums()`
read from it; `storeWeightedSums()` visits a subset of the same orientations at
the same oversampling, so it shares the fine-pass entry.

It returns NULL - and the caller falls back to the per-particle batching below -
under every condition that makes the index-to-orientation mapping particle
dependent:

* `mymodel.nr_bodies > 1`: the Euler matrix is composed with the particle's own
  `Aori`.
* `mydata.is_tomo`: composed with the per-image tilt-series rotation.
* `mymodel.orientational_prior_mode != NOPRIOR`: `idir` / `ipsi` index into a
  per-particle subset, because `HealpixSampling::getOrientations()` redirects
  through `pointer_dir_nonzeroprior` / `pointer_psi_nonzeroprior`.  This rules out
  local angular searches, i.e. the later iterations of an auto-refine.
* `do_skip_align` / `do_skip_rotate`: the angles come from the particle metadata.
* the entry would not fit in `--finufft_cache_mb`, or the stored slices no longer
  match the requested shape (e.g. after `--strict_highres_exp` resizes them).

Entries are built inside an OpenMP critical section; `std::map` keeps existing
elements at stable addresses across inserts, so the pointer handed to a worker
thread stays valid while another thread adds a different class.  Building a whole
entry is itself chunked, because one batched FINUFFT call holds coordinate and
output buffers for every query point at once.

`calculateExpectedAngularErrors()` cannot use the cache - its angles come from
each particle's metadata and its step size depends on the previous step's result.
What it *can* avoid is re-projecting the unperturbed reference `F1`, whose angles
do not change inside the search loop at all; that projection is now done once per
`(class, particle, image)` and copied out.  This is a pure saving with no change
in results (verified: the trilinear path reports the same accuracies to the last
digit).

### Per-particle batching (the fallback)

`getAllSquaredDifferences()` processes directions in **blocks**, sized so that
the pre-projected slices fit in `--finufft_batch_mb` per thread.  On entering a
block it enumerates every `(idir, ipsi, iover_rot, img_id)` slice the loops below
will ask for - replaying the same `pdf_orientation` and
`isSignificantAnyImageAnyTranslation()` filters - and projects them all in one
call.  The resulting cache is keyed on `(idir, ipsi, iover_rot, img_id)` rather
than on iteration order, so if the enumeration ever drifts out of step with the
main loop the cost is a fallback `get2DFourierTransform()` call, never a wrong
answer.  At the default budget a whole particle's orientation grid usually fits
in a single batch.

Batching the `getAllSquaredDifferences()` loop alone was chosen deliberately:
it is the only site that projects every orientation of the sampling grid.
`storeWeightedSums()` and `getFourierTransformsAndCtfs()` visit only the
significant orientations - a much smaller set - and still project one at a time.

One behavioural difference: batched slices are zeroed before projection, whereas
the existing code reuses a single `Fref` buffer and therefore leaves whatever
the previous orientation wrote in the corners outside `r_max`.  The FINUFFT path
zeroes those corners consistently.

## 5. Measured cost (§4.3)

Tutorial5.0, 150 particles, 64-pixel reference, `--pad 2 --healpix_order 1
--oversampling 1 --K 1 --iter 1 --j 4`, one CPU node, double precision:

| | wall | of which accuracy estimation | of which expectation | peak RSS |
|---|---|---|---|---|
| `RELION_INTERPOLATION=linear` | 4.5 s | - | - | 74 MB |
| NUFFT, per-particle batching only | 771 s | 124 s | 645 s | 791 MB |
| NUFFT, with the shared cache | **56.5 s** | 50 s | 6 s | 262 MB |

The shared cache leaves the expectation step at 6 s against linear's ~3 s, and
the two runs agree on the science: the reconstructed maps correlate at
**0.999984** in real space, and the cached and uncached NUFFT reconstructions are
bit-identical.

What remains is the 50 s of angular-accuracy estimation, which cannot be cached
(see §4) and is a *fixed* per-iteration cost, independent of the number of
particles - so on a production job with thousands of particles and iterations
measured in hours it is noise.  The residual per-slice gap is inherent: even
perfectly batched, spreading costs on the order of `w^dim` operations per
evaluated point against roughly 30 for the trilinear stencil.

The important caveat is that the cache only applies to a **global** angular
search.  Once an auto-refine switches to local searches
(`orientational_prior_mode != NOPRIOR`), and for multi-body and tomography, every
particle needs its own projections and the cost returns to the per-particle
batching row above.

`upsampfac` trades those two against each other - it sets the internal FFT to
`(upsampfac * mode_size)^dim` and, inversely, the spreading kernel width - and
`1.25` measured better than `2.0` in *both* phases at this box size
(2.1 vs 3.4 min for the accuracy estimation, ~12.9 vs ~20.9 min ETA for the
expectation step), so it is the default.  Lowering `--finufft_tol` from `1e-6` to
`1e-4` or `1e-3` made no measurable difference, confirming that the FFT rather
than the spreading dominates here.  Note that at `upsampfac 1.25` FINUFFT clips
its kernel width at `ns = 16`, so tolerances below about `1e-9` are not actually
reached; ask for `RELION_FINUFFT_UPSAMPFAC=2.0` if you need them (the unit tests
do).

A ~170x slowdown of a CPU refinement is a real cost, and it is the open question
for whether NUFFT should stay the default.  Flipping it back is a one-line change
in `Projector::resolveForwardInterpolator()`.

## 6. What was added

| File | Change |
|---|---|
| `src/finufft_central_slice.h` | `FinufftProjectorModes` + `evaluateNonuniformFourierSamplesFromFourierVolume3D()` + `haveFinufftSupport()`.  Split out of `relion_finufft.h` so `projector.h` does not drag in the s2-mode helpers. |
| `src/relion_finufft.cpp` | Implementation: scales the query points, reuses a thread-local plan cache keyed on `(dim, mode_size, tol)`, runs `setpts`/`execute`, applies the `1/P^dim` normalisation.  `REPORT_ERROR`s in builds without FINUFFT. |
| `src/projector.h/.cpp` | `#define FINUFFT 3`; `finufft_modes`, `padded_real_size`, static `finufft_mode_crop` / `finufft_tol`; `prepareFinufft()`; `get2DFourierTransformMany()`; FINUFFT branches in `project()`, `project2Dto1D()`, `rotate2D()` and `rotate3D()`; gridding correction skipped. |
| `src/ml_model.h/.cpp` | `projector_interpolator`, applied to `PPref` in `setFourierTransformMaps()` so the backprojectors keep `TRILINEAR`.  Runtime-only - it is not written to the model star file, so the environment variable has to be set again on `--continue`. |
| `src/ml_optimiser.h/.cpp` | `--finufft_mode_crop`, `--finufft_tol`, `--finufft_batch_mb`, `--finufft_cache_mb`; `getCachedReferenceProjections()` (the shared once-per-iteration cache, used by both passes of `getAllSquaredDifferences()` and by `storeWeightedSums()`); blocked per-particle batching as the fallback; `F1` hoisted out of the search loop in `calculateExpectedAngularErrors()`. |
| `src/autopicker.h/.cpp`, `src/particle_subtractor.h/.cpp` | Same resolution of the forward interpolator and the same tuning options. |
| `tests/unit/test_finufft_projector.cpp` | Direct-sum agreement in 2D and 3D; interpolatory at grid nodes for `project()` and `rotate2D()`; accuracy against a brute-force exact central slice vs. trilinear; batched vs. unbatched equivalence. |

## 7. Still open

- `rotate3D()` and `project2Dto1D()` have FINUFFT branches but no dedicated
  unit test.
- Local angular searches (the later iterations of an auto-refine), multi-body and
  tomography cannot use the shared cache, so they still pay the per-particle
  cost.  Caching per (particle-prior, class) is possible in principle but the
  index space is per-particle by construction.
- `getFourierTransformsAndCtfs()` (multi-body only) is neither cached nor
  batched.
- The tomography Fourier-domain path (`jaz/tomography/projection/fwd_projection.h`)
  is untouched; §5's question about adding a cubic option there is still open.
- The GPU/accelerated projector is untouched, hence the hard error when
  `--projector_finufft` is combined with `--gpu`.
- Validation per §4 (isolated radial-power-spectrum comparison and two
  independent `--auto_refine` runs) has not been run.
