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
