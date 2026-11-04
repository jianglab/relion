# Helical types in Class2D (`--helical_context`)

In 2D, one kind of filament gives several classes (its views along the twist).
`--helical_context` groups the classes into **helical types** with a sticky
hierarchical (type, class) hidden Markov model along each filament, uses the
types of the other segments of a filament as the class prior of a segment, and
writes the type of each segment (`rlnHelicalType`) and one particle and class
file per type (`run_itNNN_typeK_data.star`, `..._classes.mrcs`). The class table
of `run_itNNN_model.star` (`model_classes`) gets the type of each class
(`rlnHelicalType`: the most common type of its segments, 0 if it has none) and
the fraction of its segments of that type (`rlnHelicalTypeProbability`).
Code: `src/helix_context.{h,cpp}`.

## Choosing the number of types

`--helical_context_types 2 --helical_context_types_max 10` searches 2 to 10
types: first in the iteration before the prior is used, then every
`--helical_context_types_every` (5) iterations near the current number (one
fewer to two more).

Each filament is taken as its counts of segments per class (each segment's
most likely class). For each number of types, a mixture of multinomials (one
class distribution per type, 6 starts) is fitted three times, each time to two
thirds of the filaments, and the three resulting filament types are compared
(mean adjusted Rand index). Real types are found again from other filaments;
splits of real types are not. The folds are a random split of the filaments
(filaments come in micrograph order, which can follow the types), and the
agreement and the held-out log-likelihood are averaged over 4 such splits.

- **Rule (default):** the most types whose agreement is at least
  `--helical_context_types_agree` (0.75), but at most the fewest types whose
  held-out log-likelihood per segment is within `--helical_context_types_tol`
  (0.01) of the best: the folds can also agree on splits of real types. If none
  reaches it: the best agreement at the first choice, the current number
  afterwards.
- **The agreement overstates reproducibility.** The folds share one
  classification. On a real data set (two Class2D runs of the same picks with
  different boxes) the runs chose 6 and 7 types; their per-filament
  types agreed only 0.16-0.23 between the runs (the 2-type runs: 0.42). The
  count also moved between checks (5, 6, 5, 5, 6), with the agreement for the
  same number of types between 0.64 and 0.82. Parallel replicas and the
  consensus job (below) test the types against a new classification.
- **Junk classes have no type.** Junk: classes whose resolution is worse than
  `--sort_classes_junk` (1.5) times the median class, classes wider than 80% of
  the box (the width is measured as for `--helical_auto_diameter`, below), and
  classes that do not continue the helix (`helixtypes::classContinuity`). Two
  segments of one filament, d apart along the track, show the same helix
  shifted by d, so their class averages should overlap at a shift of d along
  the axis. Each segment is paired with the one about 15% of the box further
  along its filament; a class's continuity is the median overlap of its class
  average with its partners' at the predicted shift (low-pass 25 A, normalised
  by the energies in the overlap, best over +-2 pixels across the axis; the
  direction of the track in the class frame is calibrated from the data, pairs
  of opposite polarity are left out). Below 0.6 is junk. On a real data set:
  the 11 blob and carbon-edge classes of a replica -0.27 to 0.50, then a gap to
  0.71; the middle 90% of the other classes 0.78 and more (the profile across
  the axis, tried first, missed one of the blobs). Junk is found again at every
  update (about a second) and left out of the counts; its segments give the
  chain model no evidence about the type (missing data), the context prior
  leaves junk classes alone (factor 1), and their segments get type 0
  (`rlnHelicalType`), as do the classes in `model_classes`. `--sort_classes`
  puts them last. Before, junk was only left out of the choice: the chain model
  then put it into a real type, whose filaments did not reproduce between
  replicas.
- `--helical_context_types_agree 0` gives the oldest rule: the fewest types
  whose held-out log-likelihood per segment is within
  `--helical_context_types_tol` of the best. Those log-likelihoods differ by
  only 0.001-0.01 between numbers of types, so it often took 2 types and merged
  two real ones.
- Before the mixture, the agreement was that of the chain model itself, fitted
  per fold (threshold 0.9). On a real data set with an estimated 5-10 types its
  agreement was 0.55-0.73 even for 2 types, so it always stayed at 2; the
  mixture, fitted to the final classes of that run, agreed 0.96 (2 types), 0.84
  (5), 0.78 (6), 0.72 (7), 0.55 (10).
- The type of each segment still comes from the chain model with the chosen
  number of types; it allows type switches within a filament (switch
  probability learnt per step).
- Each choice prints the log-likelihoods, the agreements, the number of junk
  classes left out and the reason.

## Measurements

A test mix of three helical filament types (42,000 segments, whole filaments),
K = 50, EM, 25 iterations, 3 MPI processes:

| | class purity | type purity | types right |
|---|---|---|---|
| filament context off | 0.908-0.931 | - | - |
| helical types, old rule | 0.907-0.931 | 0.65 (merged) or 0.94 with splits | 2 of 10 |
| helical types, agreement rule | 0.920-0.931 | 0.944-0.956 | 6 of 6 (13 of 15 with an earlier variant, see above) |
| fixed 3 types (for comparison) | 0.905-0.926 | 0.945-0.951 | - |

- Class purity is the same with and without (same seeds: 0.928 vs 0.925).
- `--helical_polarity` made type purity slightly worse (0.944-0.948).
- Cost: 20-30% more wall time than off (a type choice takes 20-40 s).
- **VDAM does not work:** each mini-batch sees only part of each filament; the
  types came out wrong (6 or 2) and runs took 5-10x longer. `relion_refine`
  refuses `--helical_context` with `--grad`.
- `--helical_context` and `--filament_context chain|filament` both turn the
  other segments of a filament into a class prior; together they would count
  that evidence twice, so they are refused together.

## GUI

Class2D, Helix tab, **Filament context: Helical types** (EM only; with VDAM the
job is refused). It emits `--filament_context off --helical_context
--helical_context_types 2 --helical_context_types_max 10` (at most K types; no
helical context with K < 2). Setting **Classify 2D helical segments?** to Yes
picks EM and Helical types; setting it back to No restores VDAM and Off unless
either was changed in between. Loading a job changes nothing.

The measurements above are from the chain-model agreement rule (threshold
0.9); the mixture rule has unit tests (2, 4 and 6 synthetic types, a junk class)
but has not been re-measured on the 3-type mix. Not yet tested: fewer EM
iterations.

## Helical types across replicas (Class2D consensus, `--helical_types`)

A Class2D job with **Number of parallel runs** > 1 runs replicas with
different seeds; the **Class2D consensus** job combines their classes. With
**Consensus helical types?** (`relion_class2d_consensus --helical_types`; off
by default, `RELION_CONSENSUS_HELICAL_TYPES=1` turns it on) it also finds the
helical types again across the replicas:

- Each replica's filaments are taken as counts over that replica's classes
  (`rlnClassNumber`), with its junk classes left out as above (resolution from
  its model; width only if it used `--helical_auto_diameter`). The replicas do
  not need `--helical_context`, only `rlnMicrographName` and `rlnHelicalTubeID`.
- For each number of types (2 to `--types_max`, 10) the mixture is fitted to
  each replica, and the filament types of the replicas are compared in pairs
  (mean adjusted Rand index; filaments with no counted segment in a replica are
  left out of that pair).
- The number of types: the most of which every type of every replica is found
  again in the other replicas with a Jaccard index (filaments) of at least
  `--types_jaccard` (0.6; mean over the other replicas, best-matching type), at
  most the fewest within `--types_tol` (0.01) of the best held-out
  log-likelihood (mean over the replicas, 3 folds, `--types_splits` 4 random
  splits). This is cluster-wise stability (Hennig 2007: 0.6 and more is a
  pattern). One overall agreement (adjusted Rand index; first version, 0.75) was
  pulled down by types that some replicas split in two and by junk, and took 2
  types where the class averages showed more.
- The consensus type of each filament comes from the replicas' filament types
  with the categorical EM of the class consensus. Types are numbered by
  segments (1 = most); filaments with segments only in junk classes get type 0.
  `consensus_data.star` gets `rlnHelicalType` and `rlnHelicalTypeProbability`
  (the probability of the filament's type), which the frozen-class refinement
  carries through.
- The output lists, per number of types, the held-out log-likelihood, the
  agreement within the replicas (folds) and between them, and the agreement of
  each replica with the consensus.
- On a real data set (3 replicas, about half a million segments; 2 minutes,
  1.3 GB) the overall agreement between the replicas was 0.77 (2 types), 0.74
  (3), 0.61 (4), 0.46 (6), while the replicas themselves chose 6, 5 and 7 types
  (fold agreement 0.70 at 6); the first version took 2. With junk found in the
  run (the continuity test, 11-13 junk classes per replica, by eye nearly all
  blobs, ice and carbon edges), 3 new replicas chose 7, 5 and 5 types; the
  consensus took 4 (lowest Jaccard index of a type 0.87, 0.75, 0.62 for 2-4
  types, 0.29 for 5; replicas against the consensus 0.76-0.81). The replicas
  split a 5th type differently (one thin type in two, or the ribbons further).
- Tried and not used as evidence for the types (Python prototypes): the
  similarity of the class averages along a filament, and the helical repeat
  (each type a ring of views; the order of the classes along the filaments).
  Both explain the sequences within a type well (the repeats were found again
  in every replica) but made the types less reproducible between replicas
  (lowest Jaccard index 0.45 -> 0.22).
- Without it, the job leaves `rlnHelicalType` and `rlnHelicalTypeProbability`
  out. Before, `consensus_data.star` carried the types of one replica (the
  anchor), which are not consensus types.

## Per-class tube diameter (`--helical_auto_diameter`)

Types can differ a lot in width (3-4 times in one data set), so one tube diameter either cuts the wide ones or lets noise into the
narrow ones. With `--helical_auto_diameter` (2D only, needs the cylindrical
mask and no `--solvent_mask`) each class's reference is masked with its own
tube, measured on the class average after it is turned and centred:

- the profile across the axis (mean over the central 70% along it, smoothed
  with a Gaussian of 0.7 pixels), less a straight background fitted to the
  outer 15% on each side (other filaments left out, 3 MADs);
- from the peak (central 60%) out to where the profile stays below 10% of the
  peak for 15 A (dark stripes inside a filament do not end it);
- diameter: twice the farther edge, plus 10% on each side; between 6 pixels
  and the job's tube. A class's tube narrows to at most half per iteration and
  widens at once; nothing changes before iteration 2.

The job's tube diameter is then only the upper limit, so it is set to the mask
diameter (it also masks the images). The tubes are saved per class in
`model_class_settings` and kept by `--continue`. The output prints the median
and range. Opt-in: `--helical_auto_diameter` / `--dont_helical_auto_diameter`
(both parse paths), `RELION_HELICAL_AUTO_DIAMETER=1` as the default; GUI
"Measure tube diameter per class?" on the Class2D Helix tab.
Code: `helixalign::classDiameter()`, `MlOptimiser::updateHelicalClassDiameters()`.
Classes whose average is off the axis or bent get a wider (symmetric) tube.

## Sorting the classes (`--sort_classes`)

After the last iteration, Class2D can renumber its classes (any Class2D, helical
or not): with helical types, the type with the most segments first, each type's
classes by size; otherwise by size. Junk comes last: empty classes and classes
whose resolution is worse than `--sort_classes_junk` (1.5) times the median
class, and with helical types the junk classes of the types (above). The helical types are renumbered too, by abundance (type 1 has the most
segments), so type numbers mean the same in every run. Class images,
`model_classes`, each particle's `rlnClassNumber` and `rlnHelicalType`, and the
per-type files are renumbered together; the output prints the rule, the
threshold and the new and old number of each class and type. Opt-in:
`--sort_classes` / `--dont_sort_classes` (both parse paths),
`RELION_SORT_CLASSES=1` as the default; GUI "Sort the classes?" on the Class2D
I/O tab. Code: `MlModel::sortedClassOrder()`, `MlModel::reorderClasses()`,
`MlOptimiser::sortClasses()` (MPI: rank 1's class sizes are broadcast, so every
rank renumbers the same way).

Checked on the 3-type mix (K = 50, MPI): types in contiguous blocks
(11 + 15 + 15 classes), 9 junk classes last, each class's type the majority of
its particles. A sorted and an unsorted single-process run with the same seed
differ only by the renumbering (class images, model rows, particles).
Not tested: `--continue` from a sorted last iteration.

Tests: Class2D filament context tests in `tests/unit/test_pipeline_jobs.cpp` and
`tests/unit/test_gui_jobwindows.cpp`; `[helix_context_types]` and
`[helical_class_align]` unit tests.
