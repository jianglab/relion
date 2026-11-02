# Several reference maps for Class3D

`relion_refine --ref maps.star` accepts a STAR file that lists several maps, and
starts each class from its own map. This was always possible but hidden: the GUI
only took one map. The Class3D job window now accepts several maps.

## What `relion_refine` does with a STAR file (checked with real runs)

- The file must have a table `model_classes`, or else a first table with the
  column `_rlnReferenceImage`. One row is one class. `--K` is ignored and the
  class count is the number of rows. `run_itNNN_model.star` of an earlier run
  works as is.
- Each class starts from its own map. Seeds are not generated, so the classes
  differ from the first iteration on.
- All maps must have the same box size, and the same pixel size (difference
  below 0.001 A, otherwise an error unless `--ref_angpix` is given). The box must
  equal the particle box unless `--trust_ref_size` is given. Mixed boxes are not
  safe: the last map read decides the box.
- Relative map paths are read from the working directory, i.e. the project
  directory.
- With `--continue`, `--ref` is not used; the classes come from the optimiser file.

## GUI

Field **Reference map(s)** on the I/O tab of Class3D:

- **Browse** opens a chooser that selects several files (Ctrl or Shift click).
  The names go into the field, separated by commas.
- **+** (at the end of the row) adds the selected files to the ones already
  listed. A map is listed once, however often it is added.
- The field can also be typed by hand: `a.mrc,b.mrc,c.mrc`. One path (or a STAR
  file) works as before.
- With two or more maps, **Number of classes** is set to the number of maps and
  greyed out. A tooltip on the field lists the maps.

## What the job does

Code: `src/reference_list.{h,cpp}` (namespace `reflist`, no FLTK), used by
`RelionJob::getCommandsClass3DJob()` and by the GUI entry.

- With two or more maps, the job writes `References/refs_<hash>.star` (the hash
  is of the list, so the same list gives the same file, and the file is only
  rewritten when its content changes) and passes it as `--ref`.
- `--K` and the output nodes use the number of maps, so the pipeline and the
  command agree.
- Each map is an input node of the job; the generated STAR file is not.
- The file and the map checks (existence, same box size, same pixel size) are
  done only when the job is actually started (`do_makedir`), because a Scheduler
  job may name maps that a previous job has not written yet.
- `None` inside a list, or a STAR file inside a list, is an error.

Not covered: Refine3D, MultiBody, Class2D (each has its own reference field).

Tests: `tests/unit/test_reference_list.cpp`, the greying and layout test in
`tests/unit/test_gui_jobwindows.cpp`.

## Per-class settings

A reference STAR file may also carry settings for each class. A row's value
replaces the job value for that class only; an empty or missing column means
"as the job".

| Column | Table heading | Meaning |
|---|---|---|
| `rlnSymmetryGroup` | Symmetry | point group, applied at reconstruction (needs job symmetry C1, or the same group) |
| `rlnSolventMaskName` | Mask | `None`, `auto` (automask this class) or a mask file |
| `rlnHelicalTwist`, `rlnHelicalRise` | Twist, Rise | start values of this class |
| `rlnHelicalTwistMin/Max`, `rlnHelicalRiseMin/Max` | Twist min ... | local symmetry search window of this class |
| `rlnHelicalMaskTubeInnerDiameter`, `...OuterDiameter` | Tube inner, Tube outer | helical tube of this class |

- The settings are saved in `run_itNNN_model.star` (table
  `model_class_settings`) and kept by `--continue`.
- `auto` turns on automasking only at iteration 0, so `--dont_auto_mask` still
  works on a continued run.
- The job tube is the widest of the classes (inner: the narrowest, none if any
  class has none); each class is masked and symmetrised with its own tube.

GUI: **Per-class settings...** on the I/O tab of Class3D (greyed out until
reference maps are given) opens a table with one row per map. It starts from the
job values (and from the STAR file, if one was given), offers the twist/rise
range columns only when local symmetry search is on (or the STAR file has
them), and writes `References/refs_<hash>.star`. Errors name the table headings
("class 1: Twist lies outside Twist min to Twist max").

**What helped** (a test mix of three helical filament types, 14,000 segments each,
`--keep_full_filaments`): starting each class from its own twist raised plain
purity from about 0.91-0.94 (shared twist) to about 0.95. Per-class point groups
helped only once the rows matched the types: seed round 2 from the class maps
of a first run (twist from that run's model, point group measured from the map).
That two-round protocol gave purity of at least 0.95 in 16 of 16 seeds without
using labels. Searching twist per class (+-0.1 degree) made it worse (classes
drift).

Tests: `[class_settings]` and `[reference_list]` unit tests, the GUI layout test,
`tests/integration/test_class_settings.py`.

## Two rounds

GUI "Two rounds?" (Class3D, Reference tab; opt-in) runs the classification
twice in one job. Round 1 is the job as set, written as `run_round1_*`. Round 2
is the same job started from round 1's last model:
`--ref run_round1_itNNN_model.star --ref_lowpass 20`, and `--ini_high 20` when
the job's initial low-pass filter is coarser than that (or not set).

- A `model.star` as `--ref` gives one class per row of `model_classes`, each
  starting with that row's helical twist and rise (the refined values), and
  also takes the run's saved per-class settings (`model_class_settings`: ranges,
  tube diameters, masks, point groups).
- `--ref_lowpass` filters only the starting maps (not again after the first
  cross-correlation iteration): round 1's maps keep what tells the types apart
  but lose the detail fitted to round 1's assignment, so round 2 sorts the
  segments again from scratch.
- Only round 2 carries `--pipeline_control`, so the job ends with round 2; a
  failed round 1 stops the job before round 2.

Why it helps: one round depends on its start. Early mistakes reinforce
themselves (a class that takes in wrong segments looks like them and keeps
them), and more iterations do not undo them. Round 2 starts with a
type-specific map per class, so the segments are sorted by real differences
from the first iteration.

Measured (3-type mix, 42,000 segments, K = 3, 25 + 25 iterations,
`--keep_full_filaments`, 4 seeds):

| start | round 1 | round 2 |
|---|---|---|
| single reference, random seeds | 0.9303-0.9462 (mean 0.937) | 0.9479-0.9502 (mean 0.949) |
| one reference per class with its own twist | 0.9444-0.9538 (8 seeds, mean 0.950) | 0.9516-0.9542 (mean 0.953; from one round-1 run) |

The built-in round 2 gives the same as the earlier manual protocol (class maps
filtered with `relion_image_handler --lowpass 20`, 0.9520-0.9567, mean 0.954).
Seeding round 1 from Class2D helical types (each segment's first-iteration
class = its `rlnHelicalType`) was tried and removed: 0.936 after round 1 and
0.949 after round 2, the same as random seeds. It costs twice the time. The remaining errors are mostly short filaments
(10 segments or fewer), which no variant gets much better than 75% right.
