# Several reference maps for Class3D

`relion_refine --ref maps.star` accepts a STAR file that lists several maps, and
starts each class from its own map. This was always possible but hidden: the GUI
only took one map. The Class3D job window now has a mode with one reference map
per class, chosen in a table.

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

The I/O tab of Class3D starts with **Reference mode** and **Number of classes (K)**
(moved here from the Optimisation tab):

- **One reference, K classes** (the default): the usual 3D classification, with
  one map in **Reference map**.
- **One reference per class**: the reference field gives way to a table right
  under **Number of classes**, with one column per class (Class 1, Class 2, ...)
  and one row per setting; the fields below move down to make room. The table is
  as tall as its settings need, up to what the tab leaves; it scrolls up and down
  (settings) and sideways (classes), with its scroll bars or the mouse wheel.
  Changing K adds or removes columns at once; classes taken away are remembered
  while the window is open, so making K larger again brings them back. A new
  class has no map and the settings of the class before it (the first class: the
  job-wide fields, and the map of the reference field).
  - Rows: **Map**, **Symmetry**, **Mask** (`None`, `auto` or a mask file); with
    helical reconstruction also **Twist**, **Rise**, **Tube inner**, **Tube
    outer**; with local searches of helical symmetry also **Twist min/max/step**
    and **Rise min/max/step** (a step of 0 lets RELION choose it). Each map and
    mask cell has a **...** Browse button; the map's chooser takes several maps,
    which fill this class and the next ones (K grows if needed).
  - The job-wide fields that the table holds per class are greyed out:
    Reference mask and Use automatic masks (I/O), Symmetry (Reference), tube
    diameters, initial twist and rise, and the twist and rise search ranges and
    steps (Helix).
    "Automatic masks from (A)" stays active for the classes marked `auto`.
  - The table is kept in job option `ref_table` as text
    (`rlnReferenceImage|<label>|...;<map>|<cell>|...`), so it is saved with the
    job and comes back when the job is loaded.
- A job of an earlier version with several maps (or a STAR file) in the
  reference field is shown in the second mode when loaded.

## What the job does

Code: `src/reference_list.{h,cpp}` (namespace `reflist`, no FLTK), used by
`RelionJob::getCommandsClass3DJob()`; the table widget is
`src/gui_reference_table.{h,cpp}`.

- One reference per class: K must equal the number of rows, and every row needs
  a map. A column that is the same for every class becomes the usual option
  (`--sym`, `--solvent_mask`/`--auto_mask`, `--helical_twist_initial`, ...) and is
  left out of the file; a column that differs stays in the file, and the
  job-wide option gets a value that covers all classes (`--sym C1`, the widest
  tube and search range, the finest search step, the first class's twist and rise). The job writes
  `References/refs_<hash>.star` (the hash is of the content, so the same table
  gives the same file, and the file is only rewritten when its content changes)
  and passes it as `--ref`.
- `--K` and the output nodes use the number of classes, so the pipeline and the
  command agree.
- Each map is an input node of the job (once, even if it starts several
  classes), and so is each mask file of the table; the generated STAR file is not.
- The file and the map checks (existence, same box size, same pixel size) are
  done only when the job is actually started (`do_makedir`), because a Scheduler
  job may name maps that a previous job has not written yet.
- The reference field still accepts a comma-separated list or a STAR file of
  maps (one class per map, whatever K says), so that jobs and Schedules saved by
  earlier versions keep running; `None` or a STAR file inside a list is an error.

Not covered: Refine3D, MultiBody, Class2D (each has its own reference field).

Tests: `tests/unit/test_reference_list.cpp`, the table, greying and layout test in
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
| `rlnHelicalTwistInitialStep`, `rlnHelicalRiseInitialStep` | Twist step, Rise step | initial step of that search (0: the job's) |
| `rlnHelicalMaskTubeInnerDiameter`, `...OuterDiameter` | Tube inner, Tube outer | helical tube of this class |
| `rlnReferenceSeedGroup` | (none) | classes with the same number (> 0) are seeded together (see below); found automatically for rows that hold identical maps |

- The settings are saved in `run_itNNN_model.star` (table
  `model_class_settings`) and kept by `--continue`.
- `auto` turns on automasking only at iteration 0, so `--dont_auto_mask` still
  works on a continued run.
- The job tube is the widest of the classes (inner: the narrowest, none if any
  class has none); each class is masked and symmetrised with its own tube.

**One map in several rows** (for example one map with different twists and
rises, to test symmetry hypotheses). Such classes start out identical: each
class's twist and rise are first applied after the first E-step. Without help,
`--firstiter_cc` (reference not on absolute greyscale) then puts every particle
in the first of those rows, and the others stay empty. Two ways to start them
apart:

- **Seeding within the group (default).** Rows that hold identical maps form a
  group, like the classes of a single `--ref` map. The job prints "Classes 1 and
  2 start from the same map ...".
  - In the first iteration (the second one with `--firstiter_cc`), each particle
    goes to one random class of its group. It still competes normally against
    classes with other maps. Each class is then rebuilt from its own random share
    and given its own twist and rise.
  - With `--firstiter_cc`, the CC iteration uses only the first class of each
    group. Afterwards its map is copied to the other classes of the group, and
    each of them gets its own helical symmetry.
  - The random class depends only on `--random_seed` and the particle. So
    serial, MPI and GPU runs, and a run continued after the CC iteration, give
    the same classes.
  - The groups are saved in model.star (`rlnReferenceSeedGroup`) until they have
    been seeded. You can also set this column in the `--ref` STAR file; a group
    then starts from the map of its first class.
  - `--no_shared_ref_seeds` switches this off.
- **`--helical_symmetrise_refs` (optional).** Before the first iteration, each
  class's starting map is symmetrised with its own twist and rise, so the first
  E-step already compares every particle with every hypothesis. Mind the
  reference: a map from a job with twist A, re-symmetrised with twist B, is a
  blurred A map rather than a B structure. That favours A unless the map is
  low-passed below where the lattices differ.

Rows that start from identical maps and are not seeded (`--no_shared_ref_seeds`)
can never separate if they also share the same symmetry: the job warns about
them ("classes 1 and 2 start from identical maps ...").

GUI: the table of **One reference per class** (above) holds these columns.
Errors name the table headings ("class 1: Twist lies outside Twist min to Twist max").

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
