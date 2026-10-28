# Cylindrical mask for helical segments

Class2D, Class3D and Refine3D mask each helical segment before comparing it
with the references. Until now the mask kept what lies inside the tube (the
"Tube diameter" on the Helix tab) **and** inside a circle (a sphere in 3D) of
the "Mask diameter" (Optimisation tab). The ends of the segment were therefore
rounded off: a pixel near the end of the segment and near the edge of the tube
was masked even though it is inside both the tube and the segment's length.

**Use cylindrical mask?** (Helix tab, default Yes) keeps a cylinder instead:

- across the helical axis, the tube diameter, as before;
- along the helical axis, the mask diameter: a pixel is kept when its
  distance from the centre *along the axis* is within half the mask diameter,
  whatever its distance across;
- the same soft cosine edge (`--maskedge`) in both directions.

In 2D the mask is a rectangle turned with the segment's psi; in 3D a cylinder
along the helical axis. The background value outside the mask is taken from
outside the tube, as before. The references get the same shape: solvent
flattening of 3D references cuts the tube at the mask diameter along Z, that of
helical 2D classes (axis along X) keeps a rectangle of the tube diameter by the
mask diameter, and the mask used when helical symmetry is imposed is a cylinder
too (on the GPU, like the sphere, unless RELION_GPU_RECONSTRUCT=0).

| | `relion_refine` |
|---|---|
| Yes | `--helical_cylindrical_mask` (also the default without either flag) |
| No | `--dont_helical_cylindrical_mask` |

The choice is saved in `run_itNNN_optimiser.star`
(`rlnHelicalCylindricalMask`), so `--continue` keeps it unless one of the flags
is given; a run from before the option, which has no such entry, is continued
with the old mask. Jobs saved by the GUI before the option existed open with No,
so that re-running or continuing them does not change their mask.

Helical auto-refine (B5: 20k segments, 220 A mask diameter, 180 A tube, two
same-seed runs each), half maps post-processed with the same masks: 4.94 -> 3.16 A
with one mask, 4.87 -> 2.88 A with another; angular accuracy 1.02 -> 0.57 deg. The
FSC of phase-randomised maps stays near zero beyond 4.5 A in both, and every
filament's segments are in one half set, so the gain is not mask correlation or
shared segments. With a mask much wider than the tube (helical Class2D, 610 A
around 200 A) the two masks differ little and so do the results. The GPU version
(`RELION_GPU_HELICAL_MASK`, on by default) computes the same mask as the CPU
version.

Code: `softMaskOutsideMapForHelix()` and `helixMaskParameters2D()` in
`src/mask.cpp`; the GPU kernel `cuda_kernel_helixMaskApply2D`. Tests:
`[helix_mask]` unit tests.

## Tube diameter for 3D reconstruction

The Helix tab of 3D reconstruction now has **Tube diameter (A)** (default -1:
not helical segments), passed to `relion_reconstruct` as
`--helical_outer_diameter` (the name `relion_refine` and `relion_preprocess`
use):

- **With helical symmetry**, the symmetry is imposed in real space as well,
  as 3D auto-refine does after each reconstruction: after the Fourier-space
  symmetrisation, each voxel within the tube becomes the average of its
  symmetry copies taken from the central part of the box along Z ("Central Z
  length (%)", `--helical_z_percentage`, default 30%), and the map outside the
  tube is set to zero with a soft edge (`--width_mask_edge`, default 3
  pixels). `--helical_inner_diameter` (command line only) leaves a hollow
  core. Without a tube diameter the symmetry is applied in Fourier space only,
  as before.
- **With fused extraction** of helical segments (Reconstruction tab), the
  background of each segment is taken from outside the tube along its psi
  prior, as when segments are extracted. `--fused_helical_diameter`, the
  earlier name of this option, is still accepted.
