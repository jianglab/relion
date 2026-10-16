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
