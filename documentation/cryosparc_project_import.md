# Importing a CryoSPARC project into RELION

`relion_import_cryosparc` takes a CryoSPARC project directory and produces a
RELION project that reuses its work: the movies and micrographs it imported,
the motion correction it computed, the CTF fits, the picks, the extracted
particles and the 2D/3D results. Nothing is recomputed, and nothing large is
copied — image files are symlinked, and only metadata is converted.

It is also reachable from the GUI as **Project → Import CryoSPARC project...**,
which decides the scope from the directory you pick: a project root imports the
whole project, a single `JNN` job directory imports that job and the jobs it
depends on, and anything else is refused with a message rather than guessed at.

## Usage

```bash
relion_import_cryosparc --i /path/to/CS-my-project --o /path/to/relion-project
relion_import_cryosparc --i /path/to/CS-my-project --o out --job J48
relion_import_cryosparc --i /path/to/CS-my-project --dry_run
```

| Option | Meaning |
|---|---|
| `--i` | CryoSPARC project directory (the one containing `J1`, `J2`, ...) |
| `--o` | Destination RELION project directory, created if needed |
| `--job` | Import only this job and its ancestry, instead of every completed job |
| `--dry_run` | Print what would be imported and write nothing |
| `--patches_x`, `--patches_y` | Patch grid used when re-expressing CryoSPARC's local motion as RELION's polynomial model |
| `--dose_per_frame`, `--pre_exposure` | Dose values for the MotionCorr metadata, which CryoSPARC does not always store |
| `--amplitude_contrast` | Amplitude contrast written into the optics groups |

`--dry_run` is worth running first: it lists every job, the RELION directory it
would become, and which of its outputs would be read, and it names the completed
jobs that have no RELION equivalent rather than dropping them silently.

## What gets imported

Jobs are walked in dependency order and only completed jobs are considered.
A CryoSPARC job type with no meaningful RELION counterpart (interactive
inspection, particle-set arithmetic, and so on) is reported and skipped.

Each imported job becomes a RELION job directory of the matching type —
`Import`, `MotionCorr`, `CtfFind`, `AutoPick`, `Extract`, `Select`, `Class2D`,
`InitialModel`, `Class3D`, `Refine3D`, `CtfRefine` — holding:

- the main STAR file for that job type, converted from the job's `.cs` tables
  with its passthrough files merged in on `uid`;
- symlinks to the image files the table refers to, under short in-project names,
  because RELION mirrors out-of-project absolute paths under every downstream
  job and a deep CryoSPARC tree can overflow external tools' filename buffers;
- for refinements, the maps under the names RELION uses (`run_class001.mrc`,
  `run_half1_class001_unfil.mrc`, and so on), and for 2D classification the
  class average stack;
- a `job.star` recording the job type, so that the GUI shows the right parameter
  panel and knows which imported job the input came from;
- `RELION_JOB_EXIT_SUCCESS`, so the job looks finished, as it is.

**Only the final round of an iterative job is imported.** CryoSPARC records
every round in a job's `output_results[]` with parallel `metafiles[]` and
`versions[]` arrays; the final one is the entry with the highest version. This
was checked against the `.csg` files, which reference only the final iteration
and agree with the rule, so the importer does not need to parse them.

The project as a whole gets `default_pipeline.star` — with the CryoSPARC parent
relationships turned into RELION pipeline edges — plus the `.gui_projectdir`
sentinel and `.TMP_runfiles/` that the GUI requires before it will open a
directory as a project. Jobs are numbered from a single project-wide counter, as
RELION numbers them, so `job001` appears once across all job directories.

## Motion correction

A CryoSPARC patch-motion job stores its result as a trajectory: a cubic
B-spline over (frame, x, y) sampled on a patch grid. RELION's MotionCorr
metadata instead wants a third-order polynomial in the same variables, plus the
global per-frame shifts. `cryosparc_motion_model.h` fits that polynomial to the
spline by least squares, which is what makes the imported job usable by
downstream RELION jobs such as Polishing.

Two details of CryoSPARC's aligned averages matter and are handled here:

- the average is stored **flipped in Y** relative to RELION's convention.
  Getting this wrong is silent — the data still processes, it just resolves to
  about 10 Å where it should reach 3.7 Å.
- averages are written as float16 (MRC mode 12) and are converted to float32,
  because CTFFIND4 cannot read mode 12.

The port was validated against the `cs2relion` Python implementation on the same
job: identical global shifts, accumulated motion agreeing exactly, local shifts
within 5e-6 px and polynomial coefficients within 5e-6 relative.

## Known limitations

- The imported `job.star` files hold RELION's defaults with the input wired up,
  not a translation of the CryoSPARC parameters, which do not correspond
  one-to-one. A job can be browsed and continued, but "Continue" starts from
  RELION defaults rather than reproducing the original CryoSPARC run.
- A CryoSPARC project whose own inputs have been deleted imports fine but leaves
  dangling symlinks; the importer warns and carries on rather than failing.
