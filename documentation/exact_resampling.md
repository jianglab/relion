# Exact resampling with a NUFFT (opt-in, FINUFFT builds)

Wherever RELION moves an image or a map by a fraction of a pixel, it interpolates
bilinearly (2D) or trilinearly (3D). That damps high frequencies, by a different
amount at each fractional offset. A damping that is the same for every copy costs
no signal-to-noise ratio (SNR), but when copies or frames that were each moved by
a different offset are added up, the parts that no longer agree are lost.

Two switches replace the interpolation by the exact value of the image's or the
map's Fourier series at the moved positions (a type-2 NUFFT: the transform,
divided by the kernel's, on a 2x grid, one inverse FFT, then a 5-point
exponential-of-semicircle kernel per axis; error about 1e-4 of the RMS). Both
exist only in builds with `-DRELION_USE_FINUFFT=ON`, and both are off by default.

## MotionCorr: local (patch) motion

`--local_interpolation nufft` (default from `RELION_MOTIONCORR_INTERP`; GUI,
Motion tab: "Exact local shifts (NUFFT)?") applies the local polynomial motion
model exactly in RELION's own implementation. Global shifts were already exact
(a Fourier phase shift).

- The first frame is the model's reference and is not interpolated; later frames
  are. With bilinear interpolation, SNR kept against exact (tutorial motion
  models, dose weighted): 0.987, 0.947, 0.858, 0.796 at 0.5, 0.7, 0.9, 1.0 of
  Nyquist. The even/odd frame FRC of the tutorial shows no change up to about 0.5
  of Nyquist.
- Cost: about 7 s per summed pass per 3710 x 3838 micrograph with 8 threads.
  Frames are done in parallel, as many as half the free memory holds.
- Bayesian polishing re-extracts with exact shifts, so the gain is mainly before
  polishing, or for micrographs used elsewhere.
- Virtual movie averages follow it: the sum recipe becomes version 2
  (`local_nufft=5`), which older builds refuse; see `virtual_movie_averages.md`.
  The result does not depend on the thread count.

## Maps: `RELION_MAP_INTERPOLATION=nufft`

Used where copies of a map are averaged or a map is moved:

| Where | What changes |
|---|---|
| Helical symmetry in real space (`imposeHelicalSymmetryInRealSpace`: refinement, `relion_reconstruct`, `relion_helix_toolbox --impose`) | Each copy is read from the exact interpolant of the map (voxels outside the mask set to zero). Runs on the CPU; the GPU path is skipped. |
| Local symmetry (`applyLocalSymmetry`) | Each copy is moved in one exact step (trilinear used a translation and a rotation, two interpolations), only inside its mask. |
| Class and half-map alignment (`applyMapTransformation`) | Rotation and shift in one exact step. |
| Class centring (`centerClasses`) | The shift is exact. |

If the 2x grid (about 32 bytes per voxel) would take more than half the free
memory, the program says so once and uses trilinear interpolation.
`--sym` in the backprojector (Fourier space, padded) is not changed: octahedral
and other grid-aligned symmetries are exact already, and C5 agreed with a
reconstruction of symmetry-expanded particles to FSC 0.991-0.994 at all shells.
Symmetry expansion plus C1 is the exact route if it is needed.

Measured, EMPIAR-10940 tau half maps reconstructed in C1 (0.824 A/px), helical
symmetry imposed with `relion_helix_toolbox --impose`, FSC between the halves:

| | FSC at 0.6 Nyquist | at 0.7 Nyquist | FSC = 0.143 | time per map |
|---|---|---|---|---|
| trilinear | 0.543 | 0.367 | 2.29 A | 4 s |
| exact | 0.546 | 0.375 | 2.29 A | 17 s |

A small gain where there is signal; the resolution is unchanged.

Local symmetry on the same C1 half maps (`relion_localsym --apply`, 17 helical
operators k x twist, k x rise inside a central slab mask), FSC inside the mask
at 0.5 / 0.6 / 0.7 of Nyquist: between the symmetrised halves 0.902 / 0.601 /
0.324 (trilinear) and 0.898 / 0.589 / 0.337 (exact); against the other,
unprocessed half (independent noise, so it follows agreement with the true
map) 0.669 / 0.269 / 0.112 and 0.670 / 0.268 / 0.116. No real difference;
about 38 s (trilinear) and 31 s (exact) per map.

Class alignment and centring move a map once. A move there and back with the
same rotation and a fractional shift keeps, at 0.5 / 0.9 of Nyquist, 0.16 /
0.002 (alignment) and 0.29 / 0.01 (centring) of the power with trilinear
interpolation, and all of it (FSC 1.000 with the original) when exact. The
half-map FSC after one move is the same both ways: trilinear damps signal and
noise alike, so exact resampling gives sharper maps, not more resolution.
About 5 s per 384^3 map, against 0.7-1.6 s.

Code: `src/exact_map3d.{h,cpp}` (`ExactMap3D`), `MotioncorrRunner::nufftLocalShifts`.
Tests: `relion_unit_tests "[exact_map3d]"` and `"[motioncorr_nufft]"`.
