# Rectangular particles and physical-unit volumes: design

Status: **proposal, nothing implemented yet.** Written for discussion before any code.

## 1. Why

RELION assumes that every particle image is a square of `ori_size` pixels and
every reference volume is a cube of the same `ori_size`. That is a poor fit
for helical specimens, and above all for low-twist amyloids. To see the twist
or the register of a nearly untwisted filament, one needs long segments, but a
long segment in a square box leaves most of the box empty. The cost of a job
grows with the box area (2D) or volume (3D), so a segment three times longer
than it is wide costs about three times too much in memory and time.

The proposal has three parts:

1. **Rectangular images and cuboid volumes** at every step of the workflow, so
   that a particle can be `nx` by `ny` pixels, and a volume `nx` by `ny` by `nz`.
   Almost every project would use `nx = ny` (`= nz`), which must behave exactly
   as today.
2. **Sizes in physical units.** Internally a volume or image is described by its
   extent in Angstrom and its content by frequencies in 1/Angstrom, not by
   pixel and voxel counts. A reference can then be of any size and pixel size,
   and combining data of different magnification or origin becomes ordinary
   resampling.
3. **Segments rotated to horizontal at extraction**, using a choice of
   interpolation methods, and in virtual-particle workflows a **fused
   extract-and-backproject** step that interpolates only once.

## 2. Principles

* **Opt-in and invisible when unused.** Following the other new algorithms in
  this fork, the feature is off unless asked for. When all boxes are square and
  the pixel size is shared, results must be **bit-identical to today's** so that
  all existing tests and published runs stay valid. This is the main acceptance
  test of every stage.
* **No silent fallback.** If a program or the GPU path does not yet support
  non-square data, it stops with a clear error. It must never quietly crop,
  pad or square the data.
* **Frequencies are physical.** Wherever code now indexes a Fourier shell by a
  pixel radius, it moves to a frequency in 1/Angstrom. Resolution limits,
  noise spectra, FSC, CTF evaluation and masks are then the same number
  whatever the box shape.
* **One implementation of resampling.** Extraction, virtual particles and the
  fused reconstruction all call the same function, so they cannot disagree.
  (See the bit-for-bit contract in `virtual_particles.md`.)

## 3. How big a change is it?

A rough count of the current code (`src/`, including GPU code):

| Pattern | Occurrences | Meaning |
|---|---|---|
| `ori_size` | about 320, in 36 files | the single box size |
| `current_size`, `coarse_size`, `image_full_size` | about 225 | the box actually used in an iteration |
| `padding_factor` | about 400 | Fourier oversampling of the volume |
| `XSIZE(` | about 790 | width taken from an array, many assuming `= YSIZE` |
| `angpix` | about 1,600 | pixel size, one value per optics group |

Plus 89 programs in `src/apps`. The work is wide rather than deep: mostly
replacing one integer by an object, but touching nearly everything. That is why
the plan below is staged, with the square case checked for identity at each
step.

## 4. The core abstractions

### 4.1 A `Box` (or grid description)

One small value type replacing the lone integer:

* pixel counts `nx, ny, nz` (`nz = 1` for images),
* pixel size per axis (equal in practice; kept separate in case of
  anisotropic data),
* helpers: physical extent, physical frequency of a Fourier index,
  Nyquist, half-transform width `nx/2 + 1`, and "is this square".

Everything that now takes `ori_size` takes a `Box`. A square box converts to
and from the old integer, which lets the code change over gradually.

### 4.2 Physical frequency everywhere

In a half-transform of a rectangular image the step between Fourier samples is
`1 / (nx * pixel_size)` along x and `1 / (ny * pixel_size)` along y, so the grid
is finer along the long axis. A pixel-radius shell is then no longer a circle in
frequency. Replace it with the physical frequency
`s = sqrt((kx/(nx p))^2 + (ky/(ny p))^2)` and bin by `s`:

* **Noise spectra** (`sigma2_noise`) and **signal-to-noise** (`tau2`): binned in
  `s`. Bin width stays constant, so a square box gives the same bins as now.
* **FSC and resolution**: same binning; reported resolution is unchanged by
  box shape.
* **CTF**: already analytic in `s`; only the grid it is sampled on changes.
* **Masks, `current_size`, `coarse_size`**: defined by a resolution limit
  `s_max` in 1/Angstrom, not a pixel count. Each axis gets its own cut-off pixel
  count from `s_max`.

### 4.3 Cuboid volumes and the projector

The projector takes a central slice out of a 3D Fourier volume. For a
rectangular image and cuboid volume, take each 2D Fourier sample `(kx, ky, 0)` in
physical frequency, rotate it by the particle orientation, and convert the
result to a volume index with the volume's own per-axis frequency step.
Interpolation in Fourier space then works on a grid with unequal spacing.

* This is the same idea as the `RELION_INTERPOLATION=nufft` projector, which
  already evaluates at arbitrary frequencies; it is the natural first
  implementation and reference for correctness.
* The trilinear projector and backprojector need a per-axis scale factor in
  place of the single padding factor; their inner loops otherwise stay.
* Frequency limit `r_max` becomes `s_max`, so the number of samples used in
  each direction follows the box.

**Cuboid convention.** Segments are extracted along image x, and a filament lying
along image x lies along volume z (the helical axis is z, and the view runs along
volume x at tilt 90). So for images `nx` by `ny` the cuboid map is
`nx_map = ny_map = ny` (image width, y) and `nz_map = nx` (image length, x). Only
`nx_map = ny_map` is accepted for a cuboid; the image shape is taken from the map
(`MlModel::imgX() = nz_map`, `imgY() = nx_map`), and the projector and
backprojector have separate scales for the image axes and the map axes. The map
normalisation divides by the depth `nx_map`, not by `nz_map`.

**Terminology (GUI and messages):** the image x-dimension, along the helix axis, is the *length*; the y-dimension, across the helix, is the *width* (it plays the role of the usual box size). Command-line options keep the axis names `x`/`y`.

Volume size is then free. A helical reconstruction with no imposed symmetry,
for instance an initial model, uses a long cuboid; a short model with symmetry
imposed uses a smaller one. The image box and volume box need not be related by
anything but the physical content.

### 4.4 Pixel size and mixed datasets

If a reference is stored in Angstrom units with its own pixel size, a data set
of a different pixel size can be used against it by resampling in Fourier
space, which is exactly interpolation of the reference at the needed
frequencies. Merging data from different microscopes or magnifications then
reduces to: each optics group keeps its own pixel size and box; the reference
is sampled at each group's frequencies. This removes the need today to rescale
every particle set to one pixel size and box before a joint refinement.

Open point: how far to go here in the first version. Merging groups of
different pixel size inside one refinement touches the optics model, per-group
noise spectra and the half-set reconstruction. It should be a **later stage**,
built on top of working per-group boxes.

## 5. Storing sizes in STAR files

Today the optics table has `rlnImageSize` (and `rlnImagePixelSize`). New
columns `rlnImageSizeX`, `rlnImageSizeY` (and `rlnImageSizeZ` for volumes) are
added; when absent, `rlnImageSize` applies to all axes. Writers emit only the
old column for square boxes, so files stay readable by older RELION when
nothing non-square is used. Readers stop with a clear message if they find
`rlnImageSizeX` != `rlnImageSizeY` but do not support it yet.

Particle stacks: MRC headers already hold `nx, ny` separately; no format change.

## 6. Extraction with rotation to horizontal

For helical segments the picker gives a position and an in-plane angle
`psi`. Extraction cuts an `nx` by `ny` rectangle whose long axis lies along the
filament, so the saved image has the filament horizontal and a small residual
`psi` prior (for example about 0 degrees plus the usual spread). Consequences:

* **Smaller angular search.** The in-plane search range shrinks, which is a
  speed gain on top of the box saving.
* **CTF.** The CTF is defined in the micrograph frame, so its astigmatism angle
  must be rotated by the extraction angle. This is easy to forget and silently
  degrades high resolution; it needs a dedicated test (analytical CTF on a
  synthetic rotated image).
* **Normalization** (background noise, ramp, dust removal) uses a rectangular or
  elliptical mask in place of the circle.
* **Stored geometry.** The extraction angle and centre are recorded
  (`rlnAnglePsiPrior` already exists for helices) so that later steps and
  polishing can undo or reuse it.

### 6.1 Interpolation choices

Rotation by an arbitrary angle needs a method to resample the micrograph.
Offering three, selectable per job and recorded with the particles:

| Method | Cost | High-frequency behaviour | Notes |
|---|---|---|---|
| Bilinear | lowest | damps near Nyquist, smooths noise | fine for low-resolution work, 2D classes |
| Cubic (e.g. Catmull-Rom/B-spline) | low-medium | much less damping | sensible default |
| NUFFT (band-limited) | highest | essentially none | uses the micrograph's FFT, evaluated at the rotated sample positions; reuses the FINUFFT machinery |

All of them change the noise correlations of the image slightly. RELION
estimates noise spectra from the images, which absorbs most of it, but
anisotropic damping after rotation is a known limitation of the simple
methods and should be measured, not assumed. The resampling function is the
single code path used by every program (extraction, virtual particles, fused
reconstruction).

## 7. Fused extraction and backprojection (implemented)

`relion_reconstruct --fused_extract` takes virtual stacks (`relion_preprocess
--virtual`) and never forms the particle image. Each particle's Fourier
transform is computed straight from its micrograph pixels by a FINUFFT type-1
transform: the micrograph pixels, at their positions in the (shifted, rotated,
scaled) box frame, are the nonuniform sources and the output Fourier grid is
the target. The result equals `FourierTransform()` + `CenterFFTbySign()` of the
extracted image (origin at box pixel (nx/2, ny/2), 1/(nx*ny) normalisation),
so the rest of the reconstruction (shift, CTF, weights, backprojection) is
unchanged. Code: `src/fused_extract.{h,cpp}` (gather, normalise, transform) and
`vparticles::readParticleFourier()`.

Because the grid is the output grid, the output pixel size and box need not
match the stored particles:

* `--angpix` sets the output pixel size, `--box_x` / `--box_y` the image size
  (even; default: the same physical extent as the stored particle, rounded up
  to even), `--box_z` the depth as before.
* Coarser than the micrograph is an exact band limit. Finer adds no
  information: frequencies above the micrograph Nyquist are zero. A box larger
  than the stored one reads more of the micrograph (edge pixels are replicated
  outside it).
* The window is the same as extraction's, shifted by half an input pixel so
  that it lands on the same pixels at the stored pixel size.
* Normalisation (ramp plane, background mean/std, invert) is done on the
  samples, with the background ellipse (rect) or tube (helical, psi = 0) scaled
  to the output pixel size.

Without virtual stacks: if the first particle's image name is missing or not a
virtual stack, the particles are cut from `rlnMicrographName` at
`rlnCoordinateX/Y` (micrograph pixels). There is no recipe to read, so the
options say what `relion_preprocess` would have: `--angpix`, `--box_x`,
`--box_y` (required), `--fused_norm` with `--fused_bg_radius` (output pixels),
`--fused_no_ramp`, `--fused_invert_contrast`, `--fused_helical_diameter` (A,
needs `rlnAnglePsiPrior`) and `--fused_rotate_to_horizontal` (turns the box by
`rlnParticleExtractionAngle` when the STAR file has it, else by minus
`rlnAnglePsiPrior`; the orientations must belong to the turned images). The
micrograph pixel size comes from `--fused_mic_angpix`,
`rlnMicrographOriginalPixelSize`, or the micrograph header, in that order.
Unturned boxes are centred on the coordinate rounded to a whole pixel, as
extraction does (`relion_preprocess` now rounds instead of truncating, so that
both agree and no particle is off by up to a pixel). With the same settings the map equals the one from virtual
stacks (correlation > 0.999 in the integration tests), but nothing checks that
the options match how the particles were really extracted, so the virtual stack
remains the safer route.

Limits: no dust removal; optics with Zernike aberrations,
magnification matrices or several MTFs are refused; incompatible with
`--ewald`, `--newbox`, `--spatial_frequency_mode s2`, `--subtract`,
`--reconstruct_noise`, `--read_weights`; needs FINUFFT. It is not bitwise
identical to the unfused path (different arithmetic), so the contract of
virtual particles still applies only to the image path. Refinement still needs
real-space images.

Validation: unit tests (`[fused]`) agree with the standard route to 1e-6 at the
same pixel size for square and rect boxes, including particles hanging off
the micrograph, and to 1e-5 with Fourier-cropped binned particles; integration
tests in `tests/integration/test_fused_reconstruct.py`. On EMPIAR-10019
(563 rect 480 x 512 segments, C1, with refined orientations) the fused and
unfused maps have the same FSC against EMD-2699 (0.143 at 20.5 A for both) and
the fused run was a little faster (47 s vs 57 s, 8 threads); at 1.5 A/pixel
the fused run took 16 s with the same FSC (0.143 at 20.6 A, the 1.5 A Nyquist being far beyond that).

## 8. Staged plan

Each stage ends with: all existing tests pass unchanged, and square runs are
bit-identical.

1. **Foundations.** The `Box` type, physical-frequency helpers, STAR
   columns, rectangular `Image`/FFT support, resampling function with the three
   methods, unit tests (rotation recovers a synthetic image, frequency maps
   are correct, square case identical).
2. **Extraction.** `relion_preprocess` and virtual particles with a
   rectangular box and rotation to horizontal; CTF angle handling; GUI Extract
   tab (length, width, interpolation method, rotate-to-horizontal).
3. **2D path on the CPU.** Class2D and the 2D parts of the optimiser:
   noise spectra and masks in `s`, rectangular images, per-axis `current_size`.
4. **3D path on the CPU.** Cuboid `Projector` and `BackProjector`; Refine3D,
   Class3D, InitialModel, `relion_reconstruct`, PostProcess, MaskCreate,
   LocalRes, Select, Subtract, helical symmetry search and application.
5. **Remaining programs.** CtfRefine, Polish (Bayesian polishing),
   MultiBody, Dynamight, and the rest of `src/apps`, each either supported or
   refusing with a clear error. *Done (audited; see item 15 of
   `rectangular_particles_limits.md`):* every audited program works or
   refuses with "rectangular particle images" and the program name; a few
   programs could not be probed and are listed there.
6. **GPU.** CUDA (and SYCL) kernels take strides and per-axis limits in place
   of a single size.
7. **Fused extract-and-backproject**, and **mixed pixel size/box** joint
   refinement, which build on all the above.
8. **Documentation and validation** on real helical data (for example a
   low-twist amyloid) against the square-box result: same resolution at lower
   cost, or better resolution at equal cost.

Stages 1 to 4 deliver the scientific benefit for helical work; the rest
completes "all tasks in the workflow" so no job type silently lacks support.

## 9. Testing

* **Identity tests.** For every touched program, a square run before and after
  produces identical STAR/MRC output.
* **Synthetic rectangles.** Particle images from a known density at known
  orientations: recovery of orientations, resolution and FSC in rectangular
  boxes equal to the equivalent square box.
* **Frequency bookkeeping.** An analytical test that a physical frequency maps
  to the same value from any axis and box shape.
* **CTF under rotation.** Analytic comparison, with and without astigmatism.
* **Interpolation.** Rotation of a band-limited image by each method: error
  versus frequency, to document the tradeoff honestly.
* **Fused versus unfused reconstruction** within tolerance.
* **GPU against CPU** within the existing documented tolerance.
* **Real data.** Three public data sets, described in section 9.1, each
  processed before and after.
### 9.1 Validation data sets

Three EMPIAR data sets cover the three cases the change must handle. The
unmodified (square-box) workflow is the reference for each; the new workflow
must match it exactly where the box is square, and do at least as well, at lower
cost, where it is rectangular. Raw data live in the lab's shared EMPIAR folder
(not recorded here); runs go through Slurm per the project rules.

| Data set | Case | What it checks |
|---|---|---|
| EMPIAR-10940 (in vitro tau, "easy" set from the amyloid tutorial) | **Amyloid**: long, low-twist filaments; the main target | Long rectangular segments: does the longer segment give better in-plane/tilt determination and a better twist/register, at lower cost than the same length in a square box? Also the initial-model path without helical symmetry (cuboid volume), and Class2D/Class3D of long segments. Compare the final resolution, the refined helical twist and rise, and run time and memory against the square-box result. |
| EMPIAR-10019 (VipA/VipB sheath) | **Non-amyloid helical**: a standard, well-twisted helix | That nothing is lost where long segments are not needed: rectangular and square results should agree. Also helical symmetry search and application in a cuboid volume. |
| EMPIAR-10204 (RELION 5 tutorial, beta-galactosidase) | **Ordinary single particle**, small and fast | The square case: every step must give **bit-identical** output to the current build. Also a deliberately non-square run on the same particles (for example, padding the box to an elongated shape) to check that the rectangular code reproduces the square result. |

Suggested use during development:

* **EMPIAR-10204** is the everyday regression set: small enough to run after
  every change, and the place where the identity requirement is checked first.
* **EMPIAR-10019** comes in at stage 4, when cuboid volumes and the helical
  tools exist.
* **EMPIAR-10940** is the scientific test for stages 2 to 4, and again for the
  fused extract-and-backproject path. Because the harder tau sets from the same
  tutorial series (medium, hard) exist, they can be used afterwards to see
  whether longer segments help more on harder data.

## 10. Risks

* **Scale.** About 3,000 touch points; mitigated by staging, the `Box` type
  that converts from the old integer, and identity tests.
* **Performance regression** in the common square case from extra indexing
  arithmetic: the inner loops should keep square fast paths, checked by
  benchmark.
* **Hidden square assumptions** (loops written with `XSIZE` meaning both
  width and height, FFT plans, GPU strides). Search by pattern, plus a test
  suite that runs everything on a deliberately non-square synthetic data set so
  such cases fail loudly.
* **Interpolation damping** after rotation lowers apparent high-resolution
  signal; mitigated by offering NUFFT and by measuring it.
* **Mixed pixel size** raises optics-model questions that are deferred to a
  later stage on purpose.
* **File compatibility.** New STAR columns must not break older readers in the
  square case.

## 11. Open questions

1. Where the volume's pixel size can differ from the images': in the first
   version, can a refinement use a reference at a different pixel size from the
   data, or only the same?
2. Should the rotation to horizontal be an extraction option only, or can a
   later job (for example "re-extract rotated") change an existing set without
   going back to the micrographs, accepting a second interpolation?
3. Default interpolation method for extraction (suggest cubic) and for the fused
   path (NUFFT).
4. Whether to keep supporting `.mrcs` stacks of non-square images through the
   old tools that assume squares, or refuse them there.
5. How to present rectangular sizes in the GUI: separate length (x) and width (y) boxes
   next to the existing box size, which keeps its current meaning.

## Extraction (`relion_preprocess`)

- `--extract_size_x W --extract_size_y H` writes rectangular particles; the optics
  table then has `rlnImageSizeX/Y` instead of `rlnImageSize`.
- `--rotate_to_horizontal` (with `--helix`) turns each segment's box by `-rlnAnglePsiPrior`
  so the tube is horizontal. The output `rlnAnglePsiPrior` is then 0, the box angle goes
  in `rlnParticleExtractionAngle`, and `rlnDefocusAngle` is rotated with the box
  (a direction at angle t in the micrograph appears at t - a in the particle).
- `--interpolation linear|cubic|nufft` (default from `RELION_EXTRACT_INTERPOLATION`).
- Not supported: 3D, CTF phase flip/premultiply, scale/window, recentering.
- Beam tilt and Zernike aberrations are not rotated; a warning says so.
- Virtual particles compute the same pixels with the same code (`src/extract_rect.h`).
  Rectangular or rotated recipes are written as version 2 descriptors; square ones remain version 1.

## End-to-end validation results

All runs are Refine3D without GPU. Rect runs use a cuboid reference with the
long dimension along z (the cuboid convention in section 4.3). RELION runs are
not bit-for-bit repeatable even with a fixed seed, so only the numbers below,
not exact maps, can be compared.

Every reconstruction is also compared with the deposited EMDB map using
`scripts/map_fsc.py --mask_from_b`. This tool resamples both maps to a common
grid and searches the hand, polarity, rotation about the helix axis and shifts.
`--mask_from_b` takes the mask from the EMDB density: without it, solvent noise
inside a plain cylinder mask lowers the FSC (this is why the 10940 square
control once looked poor).

| Data set | Run | Box (px) | Half-map resolution | FSC with EMDB, 0.5 / 0.143 |
|---|---|---|---|---|
| Tau, EMPIAR-10940 (EMD-14046) | rect, long z | 384 x 256 | 5.4 A | 6.5 / 4.9 A |
| Tau, EMPIAR-10940 | square control | 384 | 7.0 A | 5.9 A at 0.143 |
| Tau, EMPIAR-10940 | rect, long z | 768 x 256 | 11.9 A | 18.2 / 11.2 A |
| VipA/VipB, EMPIAR-10019 (EMD-2699) | rect, long z | 480 x 384 | 8.1 A | 5.8 / 3.8 A |
| VipA/VipB | square | 512 | 6.7 A | 5.8 / 3.8 A |
| VipA/VipB | square | 384 | poor | 30.2 / 20.2 A |
| beta-galactosidase, EMPIAR-10204 (1500 particles, D2) | rect | 320 x 224 | 12.9 A | not compared |

Half-map resolutions are the unmasked gold-standard values printed by Refine3D.

- **Tau (10940):** twist and rise refined to the same values (179.40 degrees, 2.381 A). The 384 x 256 rect run beats the square control. A 768 x 256 segment (about 630 A) gives only 11.9 A, even when started from the EMDB map; a 316 A segment gives 5.4 A. Long segments of a bending filament average poorly, so this is a property of the data, not of the rect code. Start the refinement from a map with the right twist: a hand-made featureless cylinder stalled at 15 to 17 A.
- **VipA/VipB (10019):** pixel size 1.0 A; twist (29.4 degrees) and rise (21.8 A) held fixed. The voltage (300 kV), Cs (2.7 mm) and amplitude contrast (0.1) were assumed, so this is a functional validation. CTF ripples spread signal far from the particle, so the box needs enough background (`--pad 2` does not help; it only samples Fourier space twice as finely). The 384-pixel square box is poor; the rect box with 384 pixels across the helix is fine because the helix is long in z.
- **Cuboid orientation matters:** with the long dimension along x (the first implementation) the same 10019 run reached only 18.1 A against EMD-2699. The convention in section 4.3 fixes this.
- **Rect 480 x 512 (10019):** this box is taller than wide, so z is the short axis. Seed 1 gave a poor low-resolution FSC with EMD-2699 (18 A at 0.5, 4.2 A at 0.143). Seed 2 gave 30.6 / 20.5 A with the opposite hand, and the square 512 control with seed 2 was also poor at 0.5 (29.6 A; 4.3 A at 0.143, correlation 0.59). So this is run-to-run variation from the featureless hand-made start, not a rect defect; short-z projector and back-projector unit tests pass. A start map with the correct hand avoids it.
- **beta-galactosidase:** the 1500-particle subset and coarse sampling (healpix 1, local 2) kept the run short.

Problems found and fixed during validation:

- `calculateExpectedAngularErrors` still allocated square images, so the final (Nyquist) iteration of a rect Refine3D crashed.
- `--grad` (VDAM) is refused for rectangular images with a clear message.
- `--helical_z_percentage` must be small enough for the box (a Z percentage out of range is an error); `particle_diameter` must fit in the image height for helical soft masking.
- The cuboid convention and the image-side shell scales (`imgScaleX/Y`) and depth normalisation were corrected, as described in section 4.3.

Limits found in validation and how each was resolved are tracked in
`rectangular_particles_limits.md`. The fused extract-and-backproject step (item 10) is implemented (section 7).

## `relion_reconstruct` with rectangular images

Rect images (detected from the first image header) give a cuboid map of
`ny` by `ny` by `nx` (x = y = image width (y), z = image length (x), the cuboid convention
above); `--box_z` overrides the z size. Only the
default s mode (grid, trilinear) is supported; `--spatial_frequency_mode s2`,
`--ewald`, `--subtract`, `--newbox`, `--reconstruct_ctf`, `--reconstruct_noise`,
`--read_weights` and 3D data are refused with a clear message.

## Work split by micrograph in relion_reconstruct

`relion_reconstruct` (and `_mpi`) sort the particles by (micrograph or stack, y, x) and give
each MPI rank a contiguous block of whole micrographs, balanced by particle count. A big
micrograph is cut into strips of y only when there are fewer micrographs than ranks. Threads
take consecutive particles from the sorted list, so they work on the same micrograph. The
map is unchanged (largest difference to the old split 1e-10 of a peak of 0.39).

Measured with `--fused_extract` on 563 particles from 9 micrographs (10019 rectangular
test, 8 CPUs in total, back-projection phase only, seconds):

| ranks x threads | old split | by micrograph |
|---|---|---|
| 1 x 8 | 17.5 | 16 |
| 2 x 4 | 13 | 11 |
| 4 x 2 | 12.5 | 9 |

The leader prints where the rest of the time goes (`[timing]` lines). Before: with 2 x 4
about 15 s back-projection, 12 s reducing the volumes, 37 s in the final reconstruction.
Changes: the volumes are summed onto the leader with `MPI_Reduce` in place (the old
`MPI_Allreduce` made every rank hold a second copy of both volumes, which also ran 8 ranks
out of memory); and the large serial loops of the final reconstruction (decentring, windowing,
the skip-gridding division, the sinc correction) run on `--j` threads.

Total wall time on the same test (seconds, 8 CPUs, `--fused_extract`):

| ranks x threads | before | now |
|---|---|---|
| 1 x 8 | 59 | 40 |
| 2 x 4 | 81 | 50 |
| 4 x 2 | 105 | 75 |

More ranks are still slower in total than one rank with 8 threads on this small data set: each
rank holds a full padded volume (about 11 GB here) and they are summed through memory. MPI
ranks only pay off for many more particles, or for several nodes. The maps agree with the old
ones to 4e-9 (peak 0.39).
