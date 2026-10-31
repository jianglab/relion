# Automatic masks (as trueFSC in helicon)

Masks made from the map itself, after the masks of `trueFSC` in helicon
(`src/helicon/commands/trueFSC.py` in https://github.com/jianglab/helicon).
The code is in `src/auto_mask.h` and `src/auto_mask.cpp`; it is used by
`relion_mask_create --auto` and by `relion_refine --auto_mask` (3D
classification and 3D auto-refinement).

## The recipe

1. Low-pass filter the map with a Gaussian of sigma = lowpass / (3.81 x pixel
   size) pixels. With two half-maps, the map is their average and the low-pass
   filter is the resolution at which their FSC drops below 0.8 (rounded to 1 A
   above 10 A, 0.5 A above 5 A, 0.25 A below).
2. Binarise it at Otsu's threshold of its histogram (256 bins; voxels that are
   exactly 0 left out, as in EMAN2), or at a given value, fraction of the
   maximum, or the threshold that encloses a given mass (0.81 Da/A^3).
3. Keep the 26-connected regions above the threshold that hold any of the
   1000 brightest voxels of the low-passed map.
4. Add a raised-cosine soft edge outside the binary mask, using the exact
   Euclidean distance to it. With two half-maps the width is the one that best
   separates signal from noise: a bounded Brent search (as scipy's
   `minimize_scalar(method="bounded")`, 0 to box/3 pixels, to within 2 pixels)
   of trueFSC's score of the FSC of the masked half-maps (FSC_t), the FSC of the
   masked phase-randomised half-maps (FSC_n, phases randomised beyond the
   low-pass filter) and the true FSC (FSC_t - FSC_n) / (1 - FSC_n), from two
   shells beyond the phase randomisation. Without half-maps (or with too few
   shells to search) the width is 3 times the resolution at FSC = 0.143.

Differences from trueFSC: the Gaussian is applied in Fourier space (periodic
edges, not reflected); the distance to the mask is exact, not computed on a
downsampled grid; the FSC uses the shells of the half transform; the
FSC-curve fit that trueFSC uses when the FSC drops below 0.8 only beyond
100 A is left out; the soft edge is at most box/3 pixels also when it is
3 times the resolution.

## relion_mask_create

    relion_mask_create --i run_half1_class001_unfil.mrc --i2 run_half2_class001_unfil.mrc --auto --o mask.mrc --j 8

- `--auto` makes the mask automatically; `--ini_threshold` is not used.
- `--i2` gives the second half-map (`--i` is the first). Without `--lowpass`
  the low-pass filter comes from their FSC, and without `--width_soft_edge` the
  soft edge is optimised.
- With one map, `--lowpass` is the low-pass filter (none without it, with a
  warning) and `--width_soft_edge` the soft edge (pixels; 0 by default).
- `--extend_inimask` grows (or shrinks, if negative) the binary mask before
  the soft edge, as without `--auto`. `--invert` and `--helix` still apply.
- `--auto_threshold`, `--auto_fraction` and `--auto_mass` replace Otsu's
  threshold; `--auto_seeds` changes the number of brightest voxels (1000).

GUI (Mask creation, I/O tab): "Make the mask automatically?", with the
second half-map, the mass inside the mask (0: Otsu) and "Optimise the soft
edge?". With a second half-map the low-pass filter on the Mask tab is not
passed, and with soft-edge optimisation neither is its soft-edge width; the
extension of the binary mask is passed as set (set it to 0 for trueFSC's mask).

## relion_refine --auto_mask

- `--auto_mask` (needs `--flatten_solvent`, not with `--solvent_mask`): from
  the first iteration whose resolution is better than `--auto_mask_start_res`
  (12 A by default) the references are masked at every iteration with
  automatic masks made from them, inside the usual spherical (or helical)
  mask of the mask diameter. Until then the spherical mask alone is used, and
  the latest automatic mask stays in use if the resolution gets worse again.
- 3D auto-refinement: the two half-maps meet on the first MPI follower, which
  makes one mask from both (low-pass filter from their FSC, soft edge
  optimised; FSC shells beyond the current image size are left out, as the
  references have no signal there) and sends it to all ranks. Both halves are
  masked with it. With `--solvent_correct_fsc` the gold-standard FSC of the
  next iteration is corrected for this mask, as for a user-provided mask
  (so the first iteration after a continuation, which has no mask yet, uses
  the unmasked FSC).
- 3D classification: each class has its own mask, made on the MPI follower of
  the class from its reference alone. The low-pass filter and the resolution
  that sets the soft edge (3 times it) come from the class's spectral
  signal-to-noise ratio (data_vs_prior, SSNR): the FSC between half-maps that
  goes with it is SSNR / (2 + SSNR). A class gets a mask only once its own
  resolution is better than `--auto_mask_start_res`.
- `--auto_mask_soft` sets the soft edge (A) instead (no search);
  `--auto_mask_mass` binarises at the threshold that encloses this mass (kDa).
- Every iteration writes the masks as `<out>_itNNN_automask.mrc`
  (`_itNNN_classNNN_automask.mrc` in classification), and the output says the
  low-pass filter, threshold, mass, soft edge and resolution of each.
- The settings are kept in the optimiser STAR file (`rlnAutoMask`,
  `rlnAutoMaskStartResolution`, `rlnAutoMaskSoftEdge`, `rlnAutoMaskMass`).
  A continuation can switch the masks on (`--auto_mask`), off
  (`--dont_auto_mask`) or change their settings. Multi-body refinement does not
  use them.
- Not at the last iteration of auto-refinement (the references are not masked
  there for validation).

GUI (3D classification and 3D auto-refine, I/O tab): "Use automatic masks?"
and "Automatic masks from (A)". In auto-refine, "Use solvent-flattened FSCs?"
(Optimisation tab) also works with automatic masks.

Cost: the soft-edge search on the first follower takes two masked FSCs (four
Fourier transforms of the box) per step, about 10 steps, and holds about ten
copies of the box in memory there; `--auto_mask_soft` skips it.

## Status

Not compiled or run here (no C++ compiler on the development machine). The
distance transform, the bounded Brent search, Otsu's threshold, the
seed-and-grow step and the FSC crossing were checked with a line-by-line
Python transliteration against scipy and against trueFSC's own functions.
To do: build, then compare `relion_mask_create --auto --i2` with trueFSC's
`.common_mask.mrc` on the same half-maps, and run a Refine3D and a Class3D
with `--auto_mask`.
