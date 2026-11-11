# Unmasked maps from 3D classification and auto-refinement

Both 3D jobs run `relion_refine` with `--flatten_solvent` (the GUI always adds
it), so at the end of every iteration the references are multiplied by the
solvent mask before they are written and used in the next iteration:

| Job | Mask | Masked maps | Not masked |
|---|---|---|---|
| Class3D | soft sphere of `--particle_diameter`, or `--solvent_mask` ("Reference mask"), times `--auto_mask` / per-class masks | every `run_itNNN_classNNN.mrc`, the last iteration included | — |
| Refine3D | the same | every intermediate `run_itNNN_half[12]_class001.mrc` | the final `run_class001.mrc` and `run_half[12]_class001_unfil.mrc`. After convergence the mask is not applied. |

The mask is a plain multiplication (`map *= mask`). Inside the mask the voxels
are unchanged; outside the soft edge they are zero. `--solvent_mask2` and
`--lowpass_mask` add their own terms.

## --write_unmasked_maps

    relion_refine ... --flatten_solvent --write_unmasked_maps

At every iteration with masking, this also writes each 3D reference just before
the mask, next to the masked one:

    run_itNNN_classNNN_unmasked.mrc            (classification)
    run_itNNN_half1_class001_unmasked.mrc      (auto-refinement, per half)
    run_itNNN_half2_class001_unmasked.mrc

These copies are written only. The masked maps are still the ones the run uses,
so the result is exactly the same with or without the option. The copies follow
the iteration clean-up: with the default, only the last iteration's files are
kept, and `--keep_all_iterations` keeps them all. A converged auto-refinement
writes no extra final copy because its final maps are already unmasked.

In the GUI, "Also write unmasked maps?" is on the I/O tab of Class3D and
3D auto-refine, below the automatic-mask options. It is on by default (No
saves the disk space). Jobs saved before the option existed also get Yes when
they are loaded. On the command line the option is off unless given.
The option is not saved in `run_optimiser.star`, so a continuation needs it
again. The GUI adds it whenever the checkbox is set, and the checkbox can be
changed when you continue a job.

Notes:

- The copy is taken after local symmetry, real-space helical symmetrisation,
  class centring and class alignment, and before automatic masks are updated
  and applied. For helical refinement, the real-space symmetrisation already
  confines the map to the tube (`imposeHelicalSymmetryInRealSpace`). The
  "unmasked" helical map therefore lacks only the solvent mask and the
  per-class masks.
- 2D classification already writes its unmasked class averages, overwritten
  at every iteration, as `run_unmasked_classes.mrcs`. The option has no effect
  there, nor in multi-body refinement, where each body has its own mask.
- Tests: `tests/integration/test_write_unmasked_maps.py` (the files) and
  `tests/unit/test_pipeline_jobs.cpp` (the GUI checkbox's command flag).
