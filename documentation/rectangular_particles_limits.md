# Rectangular particles: known limits and how to resolve them

Status of each limit found while validating the design in
`rectangular_particles.md`. Update the status column as items are resolved.
A limit counts as resolved when the feature works (or is refused with a clear
message by design) and a test covers it.

| # | Limit | Plan | Status |
|---|---|---|---|
| 1 | Rect VipA/VipB Refine3D (14.7 A) is worse than square (10.6 A); cause unknown | Compare rect and square on the same particles | resolved: the gap came from a cuboid built long in x (wrong convention) and from boxes with too little background (CTF delocalisation), not from the rect code. With the long dimension along z, rect 480x384 matches square 512 against EMD-2699 (5.8 / 3.8 A). See the validation results in `rectangular_particles.md` |
| 2 | `--align_classes` and `--align_halves` are skipped for rect (`alignMapToMap` is cubic only) | Make `alignMapToMap` accept cuboid maps | resolved: `alignCuboidMapToMap` searches in the central cube and applies the transform to the whole cuboid; unit-tested |
| 3 | Rect extraction does not support `--scale` | Rescale the extracted box (pixel size and box change together) | resolved: rect `--scale` sets the new length and needs an even rescaled width; integration-tested |
| 4 | Aberrations (beam tilt, higher-order) are not rotated with the extraction angle | Rotate the aberration frame, or rotate the per-particle phase terms | resolved by design: `--rotate_to_horizontal` with aberrations is refused unless `--allow_unrotated_aberrations` is given; integration-tested |
| 5 | Cuboid references with a long z axis use a lot of memory | Measure; reduce padding copies in the rect Projector and BackProjector | resolved: the optimiser prints the estimated memory of the padded references and back-projectors and warns above 16 GB, pointing to `--pad 1` or a shallower reference; tested |
| 6 | `relion_reconstruct` does not support rect | Add rect support through the Rect BackProjector | resolved: `relion_reconstruct` reconstructs a cuboid from rect images (`--box_z` sets the depth); unsupported options are refused; tested |
| 7 | Class3D, InitialModel and MultiBody are not gated or tested for rect | Class3D: test and enable. InitialModel: VDAM refused (clear error). MultiBody: refused by design | resolved: Class3D works and is tested; VDAM (`--grad`) and GPU are refused with a clear message; MultiBody is refused by design |
| 8 | The GUI has no controls for rect (box x/y/z, extraction angle) | Add rect options to the Extract job and show rect box sizes | resolved: Extract job has rect controls; rescale handling checked; unit-tested |
| 9 | GPU, SYCL and `--cpu` are refused for rect | Keep as a clear error by design; document | resolved by design: refused with a clear message; tested (`--gpu`) |
| 10 | Mixed pixel size or box size in one dataset, and the fused extract-and-backproject step (design stage 7) | Fused path for virtual particles | mixed pixel or box size: refused with a clear message, tested. The fused extract-and-backproject path is implemented (`relion_reconstruct --fused_extract`, design section 7); remaining limits: virtual stacks only, no dust removal, no Zernike/magnification/multi-MTF optics, no ewald/s2/newbox/subtract/noise, not bitwise identical to the unfused path |
| 11 | `getPhaseCorrection(og, s)` uses the cropped size `angpix*s`, which looks wrong for cropped images (Jiang Lab code); Rect variants use full-box frequencies | Ask whether to change the square path; keep Rect as is | resolved: the square path used the wrong frequency scale for Fourier-cropped images; fixed with `max(s, box)` and covered by a unit test (not validated on real aberration data) |
| 12 | Remaining `ml_optimiser_mpi.cpp` sites with square assumptions (`PPref` setup near 3905, debug output near 3916) | Audit and fix | resolved: audited |
| 13 | Square-case equality with the baseline build is statistical only (RELION runs are not repeatable) | Add a deterministic unit test that a square box through the Rect code gives the same arrays as the square code | resolved: forced-rect unit tests compare a square box through the Rect code with the square code |
| 14 | Rect 480x512 (z shorter than x, y) on 10019 gave a poor low-resolution FSC with the EMDB map in one run (seed 1) | Unit tests for short-z projector and back-projector; repeat with another seed and a square control | resolved as run-to-run variation, not a rect defect: short-z unit tests pass; with seed 2 the square 512 control also had a poor FSC at 0.5 (29.6 A; 4.3 A at 0.143), and the rect run came out with the opposite hand (the hand-made start has no handedness). Use a start map with the right hand for 10019 |
| 15 | Every other program that reads particle images, maps or optics tables was never checked with rectangular input | Audit each program with rect optics (rlnImageSizeX != rlnImageSizeY), a rect `.mrcs` and a cuboid map (nx = ny < nz); each must work or stop with a message naming "rectangular particle images" and the program | resolved, except the items listed as not audited below: see the table under item 15 |

## Item 15: per-program audit

Method: each program was run on a small synthetic data set (rect optics, a 32 x 20 stack, a 20 x 20 x 32 cuboid) and its output checked. "Refused" means a clear error naming "rectangular particle images" and the program. The refusals share the wording of `refuseRectangularImages()` in `src/rect_refusal.h`. Tests: `tests/integration/test_rect_program_audit.py`.

| Program | Before | After |
|---|---|---|
| `relion_refine` (Class2D, Class3D, Refine3D) | works | works; `--solvent_correct_fsc` was silently using cubic shells, now refused |
| `relion_refine` with VDAM, GPU, SYCL, `--cpu`, multi-body, Blush, Ewald, local symmetry | refused | refused |
| `relion_reconstruct`, `relion_preprocess` | works | works |
| `relion_ctf_toolbox`, `relion_star_handler`, `relion_stack_create`, `relion_convert_star`, `relion_particle_symmetry_expand` | works | works |
| `relion_image_handler`, flips, stats, thresholds, `--sym`, 2D rescale of a rect stack | works | works |
| `relion_image_handler`, `--lowpass`, `--highpass`, `--bfactor`, `--LoG`, `--rescale_angpix`, `--new_box`, `--fsc`, power and Guinier options on a cuboid | silent wrong output | refused |
| `relion_image_handler --rescale_angpix/--new_box` on a rect STAR file (optics would keep the old rlnImageSizeX/Y) | wrong optics written | refused |
| `lowPassFilterMap`, `highPassFilterMap`, `applyBFactorToMap`, `LoGFilterMap`, `directionalFilterMap` (library) on a non-cubic 3D map | silent wrong output | refused |
| `relion_postprocess` | silent wrong output | refused for cuboid half maps |
| `relion_project` | wrong projections | refused for a cuboid map |
| `relion_align_symmetry` | crash ("incompatible shaped") | refused for a cuboid map |
| `relion_autopick` | crash or silent garbage | refused for non-square 2D references and non-cubic 3D references |
| `relion_particle_subtract`, `relion_particle_reposition`, `relion_class_ranker` | crash, or wrong images | refused |
| `relion_ctf_refine`, `relion_motion_refine`, `relion_class2d_consensus`, `relion_double_reconstruct` | refused, message did not say "rectangular particle images" | refused, wording fixed; a non-cubic reference map is refused too |
| `relion_localsym` | non-cubic input map refused without naming rectangular images | refused, wording fixed (not run end to end on a cuboid) |
| `relion_demodulate`, `relion_import`, `relion_mrc2vtk`, `relion_helix_toolbox` (map options other than `--impose_helical_symmetry`), `relion_helix_inimodel2d`, `relion_movie_reconstruct`, `relion_tomo_*` | not audited | not audited: the probes failed for reasons unrelated to rectangular input (missing inputs, old STAR format); they were not completed |

