# Class alignment under MPI (`--align_classes`)

## What changed

`relion_refine_mpi` now honours `--align_classes` (the default for Class3D; switch off with
`--dont_align_classes`). Before, the option was accepted and silently did nothing under MPI: the
serial `relion_refine` aligned the class maps to the largest class at the last iteration, the MPI
program did not. Commit 3b407f7d had removed the call from `MlOptimiserMpi::iterate()`, because
`alignClasses()` cannot simply run on every rank (see below), and nothing replaced it.

Scope is the same as the serial code: 3D classes, `K > 1`, no multi-body, no random halves
(auto-refine uses `--align_halves`), and not for Dn/T/O/I symmetry, where the symmetry fixes the
orientation. `--fix_classes` classes without particles are left alone.

## Why the serial function cannot just be called on every rank

* The leader (rank 0) never runs `maximizationOtherParameters()`, so its `pdf_class` is stale and
  would choose the wrong "largest class".
* Only the leader's `mydata.MDimg` holds the authoritative metadata that is written to
  `data.star`; the followers hold only the particles they were given.
* The class searches are expensive; every rank repeating all of them wastes the cluster.

## Design (`MlOptimiserMpi::alignClassesMpi()`, after `centerClasses()`, last iteration only)

1. Rank 1 (which has a valid `pdf_class`) chooses the largest class and broadcasts its index.
2. Every rank already holds every class map (the references are broadcast after reconstruction).
   Class *i* is searched by the follower that reconstructed it (rank `i % (size-1) + 1`) against the
   reference class, without applying the result. The six parameters (rot, tilt, psi, dx, dy, dz)
   are broadcast from that rank.
3. Every rank applies the transformation to its own copy of the map, so all copies stay identical
   and the maps written by rank 1 are the aligned ones.
4. The leader applies the inverse transformation to orientations, offsets and offset priors of the
   particles of that class in `MDimg` (`MlOptimiser::applyClassAlignmentToParticles`; for 2D
   particle images the offset follows from the 3D map shift t as the first two components of
   A * t, A being the particle's new orientation matrix, not as a plain subtraction), so the
   written `data.star` and the written maps are mutually consistent. The serial `alignClasses()`
   uses the same two helpers (`classAlignmentFreedom()`, `applyClassAlignmentToParticles()`), so
   serial and MPI cannot drift apart.

The search itself is unchanged: ±3 steps of 1 degree / 1 Angstrom with level halving, so only
small misalignments are recovered, as in the serial program.

## Offsets after alignment changed too (classes, serial and MPI; half-map alignment)

The serial `alignClasses()` subtracted the 3D shift from the image-plane offsets directly. That is
only right for a particle looking down the map's z axis; for all other orientations the offsets were
wrong whenever the shift was non-zero (a reconstruction from the written `data.star` matched the
aligned map worse than the unaligned one: correlation 0.54 against 0.86).
`MlOptimiserMpi::alignHalves()` (auto-refine, on by default) had the same flaw for the half-2 particles.
All three paths now use `MlOptimiser::applyMapAlignmentToParticle()`. Measured on a small 3D auto-refine
under MPI (half 2 in a rotated, shifted frame, local searches): correlation of the half maps
reconstructed from the final `data.star` was 0.64 without half alignment, 0.52 with the old
`alignHalves` (worse than none) and 0.89 with the corrected one. The offset priors of the adjusted
particles now move with the offsets as well.

Set `RELION_ALIGN_TIMING=1` to print the parameters `alignHalves` finds at every iteration (the existing
message was unreachable: rank 1 is a follower and its verbosity is 0).

## Score: Fourier shell correlation, not squared difference (class and half-map alignment)

`alignMapToMap` scored each candidate by its squared difference to the reference in real space. That
rewards whatever blurs a noisy map most: a half-pixel shift or a rotation off the grid lowers the noise
power more than it loses signal, and rotating through the projector also damps the noisy edge of the box.
On half maps of helices the search therefore settled at about ±0.5 pixel along the axis, with a random
sign, and moved the halves apart instead of together. Candidates are now scored by the mean Fourier shell
correlation with the reference (shells weighted by the reference's power), shifts applied as exact phase
ramps, and the reference goes through the same interpolation as the candidates (turned by a fixed 7.3°
about z, which every candidate is scored against), so no candidate gains from blur.

Helices and Cn (rotation about z and shift along z only) get their own half-map path, because the
register of the two halves along the axis shows only at fine detail (the 4.75 Å repeat of amyloids):

* the search works on the full-size maps, not the 64-cube, with shells weighted by their size, up to the
  current resolution or 0.8 × the helical rise, whichever is finer;
* it is local (one step each way, then halved steps);
* it runs after maximization, on the maps just made, and the half-2 map is moved along with its
  particles. Measured on the maps of the previous iteration and applied to the particles only, the
  correction lagged an iteration behind and kept overshooting.

On EMPIAR-10940 (amyloid, auto-refine with helical symmetry search) the final half-map offset along z is
now about 0.01 pixel, and a rerun of the project's refinements from the first Extract on stays within
one FSC shell of the original project at every step (final PostProcess 2.276 Å against 2.244 Å). Unit tests: `[alignmap]`, including noisy maps that must be left alone and an axial register that
needs the full-size map.

## What could change for existing MPI jobs

Any MPI Class3D job with `K > 1` and a Cn/C1/helical-without-symmetry-constraint setup now aligns its
class maps and the particle orientations of the non-largest classes at the final iteration. The maps
in `_it<last>_class*.mrc` and `_it<last>_data.star` differ from before by that rigid transformation
(and by nothing else). A job that relied on the old behaviour passes `--dont_align_classes`. Auto-refine jobs change too: the half-2
particle offsets are now adjusted correctly whenever `alignHalves` finds a non-zero shift, so the refinement
and the final data.star differ from before in those iterations (and are more consistent).
Continuing from an earlier iteration is unaffected; the alignment only happens when the run reaches
its last iteration.

## Tests

* Unit: `[align_classes]` (degrees of freedom; the inverse is applied only to the aligned class,
  offsets and priors move together; Cn changes only rot).
* Integration: `tests/integration/test_mpi_align_classes.py` (three ranks, two classes from the same
  density, one shifted and rotated).
