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
job is refused). When **Helical types** is selected, the Helix tab switches the
context parameter block to the helical-context controls:
- **Helical types (min, max)** (`--helical_context_types`, `--helical_context_types_max`, default `2` to `10`, capped at `K`; set `min == max` to fix the type count)
- **Hierarchical typing?** (`--helical_context_hierarchical` / `--dont_helical_context_hierarchical`, default Yes unless `RELION_HELICAL_HIERARCHICAL=0`: refines types on the final iteration by recursive hierarchical tree splitting + twist-rate refinement)
- **Start iter, min fold ARI** (`--helical_context_start` default `4`, `--helical_context_types_agree` default `0.75`)
- **Context power, exclude (A)** (`--helical_context_temper` default `-1`, `--helical_context_exclude` default `-1`)
- **Smooth tracks, helical polarity** (`--helical_track_smooth` / `--dont_helical_track_smooth`, default Yes unless `RELION_HELICAL_TRACK_SMOOTH=0`; `--helical_polarity` / `--dont_helical_polarity`, default Yes unless `RELION_HELICAL_POLARITY=0`, greyed out unless image alignment and bimodal psi searches are enabled)

Setting **Classify 2D helical segments?** to Yes selects the optimal combined
default configuration: EM algorithm, **Helical types** (`--helical_context`),
**Use cylindrical mask?** = Yes, **Measure tube diameter per class?** = Yes
(default unless `RELION_HELICAL_AUTO_DIAMETER=0`), **Use automatic masks?** = Yes
from `15` A (`--auto_mask --auto_mask_start_res 15`), hierarchical typing, track
smoothing, polarity propagation, and **Sort the classes?** = Yes (most abundant
type first, junk last); setting it back to No restores VDAM, Off, `No` auto-masks
(reverting `15` -> `12` A), and unsorted classes unless another context mode was
chosen in between. Loading an existing job or reading an older `job.star`
missing `helical_auto_diameter` or `helctx_hierarchical` keeps `No` for
backwards compatibility.

In **3D classification**, **Filament context** defaults to **Per filament**
(`--filament_context filament`) and picking **Do helical reconstruction?** = Yes
enables **Use automatic masks?** = Yes from `15` A (`--auto_mask
--auto_mask_start_res 15`, unless a manual `Reference mask` is set), while
`"Two rounds?"` (`do_two_rounds`) remains off by default. In **HelixSubClassify**, Stage-1 defaults
to `2..10` types with `--hierarchical` = Yes and Stage-2 enables `--auto_mask
--auto_mask_start_res 15` by default.

The **Class2D consensus** Optimisation tab defaults **Consensus helical types?**
to Yes (unless `RELION_CONSENSUS_HELICAL_TYPES=0`; older `job.star` files missing
the key keep `No`) and exposes **Max types, min Jaccard** (`--types_max` default
`10`, `--types_jaccard` default `0.6`), greyed out when Consensus helical types
is No.

The measurements above are from the chain-model agreement rule (threshold
0.9); the mixture rule has unit tests (2, 4 and 6 synthetic types, a junk class)
but has not been re-measured on the 3-type mix. Not yet tested: fewer EM
iterations.

## Helical types across replicas (Class2D consensus, `--helical_types`)

A Class2D job with **Number of parallel runs** > 1 runs replicas with
different seeds; the **Class2D consensus** job combines their classes. With
**Consensus helical types?** (`relion_class2d_consensus --helical_types`; GUI
defaults to Yes unless `RELION_CONSENSUS_HELICAL_TYPES=0`, CLI opt-in unless
`RELION_CONSENSUS_HELICAL_TYPES=1`) it also finds the
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

With decoupled particle masking, the job's `--helical_outer_diameter` is kept
on experimental particle images for tight alignment masking, while each class
reference's adaptive tube covers up to `max(helical_tube_outer_diameter,
particle_diameter)` (saved per class in `model_class_settings` and kept by
`--continue`). The output prints the median and range. Flags:
`--helical_auto_diameter` / `--dont_helical_auto_diameter` (both parse paths;
CLI opt-in unless `RELION_HELICAL_AUTO_DIAMETER=1`; GUI "Measure tube diameter
per class?" on the Class2D Helix tab defaults to Yes unless
`RELION_HELICAL_AUTO_DIAMETER=0`, is active when "Use cylindrical mask?" is Yes,
leaves "Tube diameter (A)" active for particle alignment masking, and is
forwarded by Consensus Class2D).
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

## Upgraded multi-scale filament context

The upgraded Class2D helical context couples intra-filament pose, shift, polarity,
and polymorph type inference across iterations (`src/helix_context.{h,cpp}`,
`src/ml_optimiser.cpp`, `src/exp_model.cpp`, `src/ml_optimiser_mpi.cpp`,
`src/acc/acc_ml_optimiser_impl.h`):

1. **Multi-scale intra-filament Markov continuity & annealing**:
   - Models transitions along ordered chain sections (`makeSections()`) and couples
     gapped chain fragments per filament across `HelicalContext::update()` and
     `makeSections()` so broken or gapped filaments vote cohesively.
   - Progressively anneals the context prior (`--helical_context_start`, default `4`;
     `--helical_context_temper`, default `-1`) and smooths intra-filament tracks
     (`smoothFilamentTracks()`, guarded with `iter < nr_iter` in
     `updateHelicalContext()`).
2. **Filament polarity tracking (`updateHelicalPolarity()`)**:
   - Propagates coherent filament polarity along picking order (`--helical_polarity_start`,
     default `3`; disabled with `--dont_helical_polarity`), annealing `pol_prior_max`
     up to `0.985` by `iter >= start + 5`.
3. **Adaptive continuity junk filtering & chimeric class splitting**:
   - `helixtypes::junkClasses` combines class occupancy and sequential class
     continuity (`0.55 * med_c` threshold) to isolate disordered/edge junk classes.
   - `MlOptimiser::splitChimericHelicalClasses()` splits mixed-polymorph classes at
     `iter == 18` and reclaims duplicate-view minor donor classes (`dup_minor`).
4. **Mixture-model polymorph count & assignment (`selectNrTypesByMixture`)**:
   - `fitChainMixtureEM()` uses sharpened k-means++ initial responsibilities
     (`0.80` / `0.20 / (T - 1)`).
   - `helixtypes::scanChains()` rejects empty or degenerate CV folds; `scan_tmin = min(1, tmin)`
     and `eff_tol = min(tol, 0.0025)` enable a reliable `T = 1` homogeneous fallback
     even when `--helical_context_types >= 2`.
   - `initialiseModel()` aligns `Wn` column permutations to the previous `tpost` via
     `overlap`, preserves shared-crossover classes (`Aff < 0.80 * aff_sum` and
     `>= 0.12 * aff_sum`), and enforces `min_excl = max(1, min(2, n_good_cls / (3 * T)))`
     exclusive anchor classes per type.

### Fixed upstream & binned-extraction bugs

- Enabled GPU `--bimodal_psi` (`gpuBimodalPsiEnabled()`).
- Fixed `gpuFusedWeights()` skipping coarse polarity storage (`storeCoarsePolarity`).
- Fixed `AsyncFileWriter` race with rank-0 `writeHelicalTypeFiles()`.
- Fixed `classDiameter()` failing on negative-contrast class averages.
- Fixed `do_helical_refine` rotating 2D offset priors only when `ref_dim == 3`.
- Fixed `helixtypes::adjustedRand()` returning `1.` on single-cluster collapse (`0.` instead).
- Fixed `smoothFilamentTracks()` multiplying unbinned micrograph `rlnCoordinateX/Y` by
  binned `rlnImagePixelSize` (`angpix`); it now scales by per-optics-group
  `rlnMicrographOriginalPixelSize` and scales Huber `delta = max(4.0, 0.8 * angpix)`.
- Guarded `helical_track_sigma_A` on coarse/binned grids via
  `eff_track_sigma = max(helical_track_sigma_A, 0.75 * offset_step * pixel_size)`.
- Replaced dense `C x C` `ChainSuffStat` with sparse sorted unigram/transition vectors,
  cached `log_mu`/`log_B` in `ChainMixtureModel`, and parallelised `buildChainStats()`
  and `scanChains()` with OpenMP (`28,697` filaments x `160` states in `43.3 s`).
- Marked zero-occupancy classes (`pdf[k] <= 0`) as junk in `helixtypes::junkClasses()`.

### CLI options and environment switches

- CLI (`parseInitial` and `parseContinue` in `src/ml_optimiser.cpp`):
  - Fresh helical 2D EM runs automatically enable `--helical_context` (with track smoothing when multi-segment tracks are present), `--helical_context_types 2 --helical_context_types_max 10`, `--helical_context_hierarchical`, `--helical_auto_diameter`, `--sort_classes`, and (when flattening solvent) `--auto_mask --auto_mask_start_res 15`; fresh helical 3D Class3D runs automatically enable `--filament_context filament` (with track smoothing) and `--keep_full_filaments` when multi-segment tracks are present, plus `--auto_mask --auto_mask_start_res 15` when flattening solvent without `--solvent_mask`/per-class masks (override any with explicit `--dont_*` flags or `--filament_context off`).
  - `--helical_context_hierarchical` / `--dont_helical_context_hierarchical` (default on; `RELION_HELICAL_HIERARCHICAL=0` disables)
  - `--helical_track_smooth` / `--dont_helical_track_smooth`
  - `--helical_track_sigma_A` (default `3.5`)
  - `--dont_helical_polarity`
  - `--helical_context_start` (default `4`), `--helical_context_temper` (default `-1`),
    `--helical_polarity_start` (default `3`)
- Post-hoc tool (`relion_helix_filament_context` in `src/apps/helix_filament_context.cpp`):
  - Defaults `--types 2 --types_max 10`, `--hierarchical` (disable with `--dont_hierarchical` or `RELION_HELICAL_HIERARCHICAL=0`), `--temper -1` (auto) and `--types_agree 0.75`; adds `--model`, `--outer_diameter`,
    `--sort_classes_junk` (`1.5`), `--dont_smooth_flicker`, and `--j`; auto-detects
    `_classes.mrcs` / `_model.star` for `classContinuity`, `junkClasses`, `computeClassEquivalence`,
    and final surgical mismatch/flicker smoothing.
- Environment switches (`=0` disables each component for ablation):
  - `RELION_GPU_BIMODAL_PSI=0`
  - `RELION_HELICAL_CLASS_PDF=0`
  - `RELION_HELICAL_HIERARCHICAL=0`
  - `RELION_HELICAL_TRACK_SMOOTH=0`
  - `RELION_HELICAL_POLARITY=0`
  - `RELION_HELICAL_SECTION_SEED=0`
  - `RELION_WHOLE_FILAMENT_SHUFFLE=0`
  - `RELION_HELICAL_SPLIT_CHIMERIC=0`
  - `RELION_HELICAL_FINAL_SMOOTH=0`

### Benchmark summary (GPU Class2D, 25 iterations)

| Dataset / Run | Types | Filaments / Segs | Fil Type ARI (Acc) | Class Purity (Chimeric) | Neighbor Stay | Flicker | Pol Cons | 4.75 A Merid Ratio |
|---|---|---|---|---|---|---|---|---|
| **Synth 3-poly + 12 junk** (`17,597` segs, `300` fils, `K=24`) | | | | | | | | |
| Baseline independent (seed 1) | 0 | 300 / 17,597 | 0.0000 (38.54%) | 0.6724 (17) | 0.3935 | 0.1301 | 1.000 | 0.145 |
| Upgraded context (seed 1) | 3 | 300 / 17,597 | 0.9891 (99.65%) | 0.8363 (10) | 0.7463 | 0.0187 | 0.817 | 0.164 |
| Upgraded context (seed 2) | 3 | 300 / 17,597 | 0.9571 (98.61%) | 0.8753 (7) | 0.7701 | 0.0188 | 0.883 | 0.193 |
| Upgraded context (seed 3) | 4 | 300 / 17,597 | 0.7800 (82.64%) | 0.8661 (9) | 0.7738 | 0.0173 | 0.871 | 0.153 |
| **EMPIAR-10944 tau PHF vs SF** (`10,531` segs, `233` fils, `K=24`) | | | | | | | | |
| Baseline independent (seed 1) | 0 | 233 / 10,531 | 0.0000 (56.65%) | 0.7854 (17) | 0.4555 | 0.0209 | 0.610 | 4.330 |
| Upgraded context (seed 1) | 2 | 233 / 10,531 | 0.7431 (93.13%) | 0.9348 (3) | 0.5483 | 0.0142 | 0.875 | 4.449 |
| Upgraded context (seed 2) | 2 | 233 / 10,531 | 0.7136 (92.27%) | 0.9211 (4) | 0.5419 | 0.0157 | 0.894 | 4.656 |
| **Single-polymorph amyloid control A** (`4,963` segs, `104` fils, `K=20`) | | | | | | | | |
| Old context baseline | 2 (spurious 23/82 fils) | 104 / 4,963 | — | 17 non-junk | 0.3118 | 0.0795 | 1.000 | 1.873 |
| Upgraded context (seed 1) | 1 (`type_prob=1.000`) | 104 / 4,963 | cross s1/s2 1.0000 | 20 non-junk | 0.5182 | 0.0023 | 0.816 | 2.002 |
| Upgraded context (seed 2) | 1 (`type_prob=1.000`) | 104 / 4,963 | cross s1/s2 1.0000 | 20 non-junk | 0.5277 | 0.0004 | 0.813 | 1.619 |
| **Single-polymorph amyloid control B** (`31,811` segs, `555` fils, `K=25`) | | | | | | | | |
| Earlier baseline run | 2 (spurious 6/550 fils) | 555 / 31,811 | — | 8 non-junk | 0.2950 | 0.0536 | 1.000 | 1.830 |
| Upgraded context (seed 1) | 1 (`type_prob=1.000`) | 555 / 31,811 | cross s1/s2 1.0000 | 22 non-junk | 0.5065 | 0.0044 | 0.689 | 2.304 |
| Upgraded context (seed 2) | 1 (`type_prob=1.000`) | 555 / 31,811 | cross s1/s2 1.0000 | 21 non-junk | 0.4962 | 0.0081 | 0.710 | 4.197 |
| Upgraded context (cyl mask, seed 1) | 1 (`type_prob=1.000`) | 555 / 31,811 | Res 5.13/4.48 A | 22 non-junk | 0.5092 | 0.0075 | 0.715 | 0.761 |
| **Single-polymorph amyloid control C** (`32,474` segs, `568` fils, `K=25`, `--particle_diameter 190`) | | | | | | | | |
| Upgraded context (cyl mask, seed 1) | 1 (`type_prob=1.000`) | 568 / 32,474 | Res 5.01/4.38 A | 22 non-junk | 0.5015 | 0.0058 | 0.731 | 0.800 |
| **Multi-polymorph amyloid dataset A (full)** (`1,652,995` segs, `28,697` fils, `8x` binned `7.72 A/px`, `K=200`) | | | | | | | | |
| Independent baseline | 0 | 28,697 / 1,652,995 | 0.0000 (42.33% vs 3D) | 0.7084 (117) | 0.5684 | 0.0559 | 1.000 | — |
| Post-hoc `relion_helix_filament_context` | 8 (polarity-paired) | 28,697 / 1,652,995 | 0.2707 (78.77% vs 3D) | 0.7802 (107) | 0.7277 | 0.0170 | 1.000 | — |
| **Multi-polymorph amyloid dataset A (450-filament subset)** (`29,071` segs, `450` fils, `8x` binned `7.72 A/px`, `K=36`) | | | | | | | | |
| Baseline independent | 0 | 450 / 29,071 | 0.0000 (44.93% vs 3D) | 0.6813 (18) | 0.6995 | 0.0561 | 0.632 | — |
| Upgraded context (seed 1) | 4 | 450 / 29,071 | 0.4874 (70.29% vs 3D) | 0.7638 (26) | 0.6837 | 0.0325 | 0.714 | — |
| Upgraded context (seed 2) | 5 | 450 / 29,071 | 0.4054 (64.49% vs 3D) | 0.7420 (26) | 0.7499 | 0.0210 | 0.733 | — |

---

## Helical context for Class3D

`--filament_context` (`chain` / `filament`), `--keep_full_filaments`, helical track smoothing, and filament polarity propagation have been extended and debugged for 3D helical classification (`Class3D`), with matching Class3D GUI exposure on the `Helix` tab (`tab6`).

### Upstream & GPU/GUI bugs fixed

1. **Bimodal `psi` prior bug in 3D helical sampling (`src/healpix_sampling.cpp`)**:
   - `bool is_psi_flipped` was declared outside `for (long int ipsi = ...)` inside `selectOrientationsWithNonZeroPriorProbabilityFor3DHelicalReconstruction()`. When `prior_psi_flip_ratio != 0.5`, hitting a flipped `psi` latched `is_psi_flipped = true` for subsequent orientations in the same direction. Moving `bool is_psi_flipped = false;` inside the `ipsi` loop restores exact bimodal `rot/psi` prior weights (`prior_rot_flip_ratio`).
2. **GPU/CPU parity & GPU Class-Batching (`src/acc/acc_ml_optimiser_impl.h`)**:
   - Passed `op.prior_rot_flip_ratio` to `selectOrientationsWithNonZeroPriorProbabilityFor3DHelicalReconstruction` on the GPU path to match CPU.
   - Restricted GPU `storeCoarsePolarity()` to `mymodel.ref_dim == 2` so 3D Class3D does not clobber 3D polarity arrays with 2D coarse sums.
   - Guarded `RELION_GPU_GROUP_SHARED` deferred group `storeFilamentScores()` on `needFilamentLoglik() && useCoarseFilamentScores()`.
3. **`optimiser.star` persistence & dual-path parsing (`src/metadata_label.h`, `src/ml_optimiser.cpp`)**:
   - Added `EMDL_OPTIMISER_HELICAL_KEEP_FULL_FILAMENTS` (`rlnHelicalKeepFullFilaments`), saved in `optimiser.star` and parsed symmetrically across both `parseInitial` and `parseContinue` (`--keep_full_filaments` / `--dont_keep_full_filaments`), along with `--filament_clip`, `--filament_exclude`, `--filament_start`, `--filament_phase_weight`, `--helical_track_smooth` / `--dont_helical_track_smooth`, and `--helical_polarity` / `--dont_helical_polarity`.
4. **Final-iteration hard consistency in 3D (`src/ml_optimiser.{h,cpp}`)**:
   - Overwriting per-particle `ORIGIN_X/Y_ANGSTROM` and `PSI` mid-run via hard `enforceFilamentConsistency()` desynchronised `alignClasses()` class-frame offsets during early `--coarse_early` iterations. Soft Bayesian filament priors now guide intermediate E-steps continuously, while hard filament consistency (`shouldEnforceFilamentConsistencyThisIter()`) runs on the final iteration (`iter >= nr_iter`, or at every post-seed iteration with `RELION_FILAMENT_ENFORCE_EVERY_ITER=1`).
5. **FLTK `place2` widget overlap (`src/gui_entries.cpp`)**:
   - Fixed `place2()` creating its second `Fl_Choice` at `XCOL2` instead of the computed `x` offset, preventing side-by-side dropdown stacking in the Class3D `Helix` tab.

### 3D helical context architecture

- **MPI aggregation (`src/ml_optimiser_mpi.cpp`)**: Follower E-step ranks reduce `filctx_loglik` (`MPI_SUM`) and `filctx_visited` (`MPI_MAX`) onto the leader, which computes `filctx_logprior` in `updateFilamentContext()` and broadcasts it back before the next M/E cycle.
- **Debiased segment scoring & HMM inference (`src/filament_context.{h,cpp}`, `src/ml_optimiser.cpp`)**:
  - Uses Pass-2 best-pose profile weight (`class_best_weight / used_prior`) in 3D helical alignment (overridable via `RELION_FILAMENT_SCORE=coarse|marginal`), keeps seed/`--firstiter_cc` iterations neutral (`0.f`), uses uniform active-class `pi[k]` when `PRIOR_ROTTILT_PSI` omits class priors, clamps per-segment log-likelihood contrasts with Huber `--filament_clip` (`12` nats), supports physical leave-out windows (`--filament_exclude`), distance-aware Markov step powers (`T^m`), central-Z `eff_box` tempering (`max(0.35, helical_z_percentage) * box_length`), and a `0.55` ramp during `--coarse_early` reduced-Healpix iterations.
- **3D helical azimuthal phase coherence (`filctx::computeHelicalPhaseBonus`)**:
  - Evaluates wrap-aware circular phase consistency `cos(n_sym * ((rot_i - tw * track_i / rise) - rot_j))` within each filament once resolution reaches `<= 16 A` (`--filament_phase_weight 1`), converting length-invariant mean resultant growth (`excess_r2`) into an additive cavity log-score without penalising shorter fragments.
- **Tangent-only 3D track smoothing & 3D polarity propagation (`src/helix_context.{h,cpp}`, `src/ml_optimiser.cpp`)**:
  - `smoothFilamentTracks(..., update_origin_priors = false)` smooths `rlnAnglePsiPrior` (`±8°`) along IRLS Huber splines of filament coordinates without overwriting 3D `ORIGIN_X/Y_PRIOR_ANGSTROM` (preserving `alignClasses()` axial Z-registrations and bimodal `psi` offsets).
  - `updateHelicalPolarity3D()` runs bidirectional 1D Markov message passing along track order (`sw <= 0.005`, `pol_unit = 3.5 * pol_temper`) through `iter == nr_iter`, annealing `rlnAnglePsiFlipRatio` in `[0.05, 0.95]` (greyed out automatically in the GUI and disabled at runtime under unimodal local searches).

### Class3D GUI exposure (`src/pipeline_jobs.cpp`, `src/gui_jobwindow.cpp`)

- On the Class3D `Helix` tab (`tab6`): toggling `"Do helical reconstruction?"` automatically sets `"Share class info along filaments"` to **`Per filament`** (`Yes`) or **`Off`** (`No`) alongside `"Keep full filaments?"`.
- Exposes `"Smooth filament tracks?"` (`helctx_track_smooth`, default Yes) and `"Propagate filament polarity?"` (`helctx_polarity`, default Yes, inactive unless alignment is enabled with bimodal angular searches and positive psi range), with `helical_nr_asu` and `helical_z_percentage` arranged side-by-side via `place2`.

### Benchmark summary (GPU/MPI Class3D, 20 iterations, `alignClasses` on)

- **Exp1 (`4x` `3.86 A/px`, `120` fils / `9,533` segs, `60` Type A + `60` Type B, Healpix `3` with `--coarse_early`)** & **Exp2 (`2x` `1.93 A/px`, `100` fils / `6,489` segs, `61` Type A + `39` Type C, Healpix `4` with `--coarse_early`)**:

| Run | Seg ARI (Acc) | Fil ARI (Acc) | Fil Purity | Pol Switches (Log Sw) | Mean Phase R | Class Res [C1, C2] (A) |
|---|---|---|---|---|---|---|
| `Exp1 2Ref Base (off)` | 0.6035 (88.87%) | 0.9667 (99.17%) | 0.8807 | 1761 (—) | 0.3979 | [19.8, 19.0] |
| `Exp1 2Ref Vote (count)` | 0.9437 (98.57%) | 0.9667 (99.17%) | 1.0000 | 1794 (—) | 0.4136 | [19.8, 19.0] |
| `Exp1 2Ref Context (s1)` | **0.9556 (98.88%)** | 0.9339 (98.33%) | 1.0000 | **1406 (0)** | 0.4457 | **[18.3, 18.3]** |
| `Exp1 2Ref Context (s2)` | 0.8386 (95.79%) | 0.8389 (95.83%) | 1.0000 | **1348 (0)** | **0.4490** | [19.0, 18.3] |
| `Exp1 1Ref DeNovo Base` | 0.6552 (90.49%) | 0.9667 (99.17%) | 0.9044 | 2217 (—) | 0.3330 | [19.8, 19.0] |
| `Exp1 1Ref DeNovo Context` | **0.9437 (98.57%)** | **0.9667 (99.17%)** | 1.0000 | **1508 (1)** | 0.3321 | **[18.3, 14.1]** |
| `Exp2 2x Base (off)` | 0.2007 (72.45%) | 0.4298 (83.00%) | 0.8104 | 1982 (—) | 0.3240 | [8.0, 7.3] |
| `Exp2 2x Vote (count)` | **0.4226 (82.51%)** | **0.5137 (86.00%)** | 1.0000 | 1918 (—) | 0.3219 | [8.0, 7.3] |
| `Exp2 2x Context (s1)` | 0.3254 (78.53%) | 0.4850 (85.00%) | 1.0000 | **1338 (0)** | 0.3455 | **[7.7, 6.9]** |
| `Exp2 2x Context (s2)` | 0.3161 (78.13%) | 0.4572 (84.00%) | 1.0000 | **1322 (0)** | **0.3558** | **[7.7, 6.9]** |

---

## Next-Generation Class2D & Sub-Classification Enhancements

1. **DC-free circular crossover phase & per-filament twist-rate splitting (`src/helix_context.{h,cpp}`, `src/ml_optimiser.cpp`, `src/apps/helix_filament_context.cpp`)**:
   - **Zero-mean phasor period estimation (`fitTypePeriodAndPhasors`, `fitPeriods`)**: Subtracts global occupancy-weighted and per-filament DC phasor means before scoring (`2 A` sub-grid refinement, `[0.35 * Ct, 2.80 * Ct]` capped at `1.6 * span` per filament using global reconstructed positions and decoded polarity), eliminating the finite-span `C = 2000 A` constant-phase collapse. `_filaments.star` records per-filament `_rlnHelicalCrossoverPeriod` and `_rlnHelicalCrossoverFigureOfMerit`.
   - **Leave-one-out circular phase ring priors (`applyPhaseRingPriors()`, `RELION_HELICAL_PHASE_RING=0` disables)**: Couples intermediate Class2D segment class priors (`iter + 1 >= helical_context_start_iter + 2`) with circular crossover phase predictions from the same filament's remaining segments (`cos/sin` phasor prediction beyond `exclude_angst`).
   - **CV-gated twist-rate type splitting (`splitTypesByTwistRate()`, `RELION_HELICAL_TWIST_SPLIT=0` disables)**: Splits shared-view types whose per-filament crossover periods exhibit a reproducible bimodal mixture in `log(period)` (weighted 1D 2-Gaussian EM: Ashman `D >= 2.0`, `>= 11%` separation, `>= 8` & `8%` minor filaments, `3x3`-fold CV weighted LL gain `> 0.08`, fold ARI `>= 0.75`).
   - Fixed Class3D resolution gate unit comparison (`1. / mymodel.current_resolution <= 16.`) in `MlOptimiser::updateFilamentContext()`.
2. **Decoupled alignment tube mask & 2D contour reference auto-masking (`src/ml_optimiser.cpp`, `src/auto_mask.cpp`, `src/pipeline_jobs.cpp`, `src/gui_jobwindow.cpp`)**:
   - Keeps `--helical_outer_diameter` active on experimental particles (and editable in GUI even when `--helical_auto_diameter` is Yes) while class references adapt up to `max(helical_tube_outer_diameter, particle_diameter)`.
   - Extends `--auto_mask` (`autoMaskFromMap`, `MlOptimiser`) to `ref_dim == 2` (`eff_start = max(auto_mask_start_res, 2.5 * pixel_size)`, clamped positive references, scaled `nr_seeds`, and `< 1%` empty-mask identity fallback), exposed on the Class2D `I/O` tab and forwarded by `Class2D consensus`.
3. **Multi-stage transverse-mode chimeric class splitting (`MlOptimiser::splitChimericHelicalClasses`)**:
   - Runs at `iter == 8, 12, 16, 20` (`iter + 4 <= nr_iter`), ranks chimeric candidates with mixing-balance and transverse-broadening bonuses (`w2 * (1 + bonus)`), breaks symmetry with a `2%` RMS transverse cosine perturbation inside the tube, copies parent `loglik` to the donor slot before `helical_context.update()`, and applies `tpost` prior routing afterwards so `update()` cannot clobber the routed priors.
4. **Coarse-to-unbinned zero-copy virtual sub-classification (`src/apps/helix_subclassify.cpp` -> `relion_helix_subclassify`)**:
   - Groups Stage-1 segments by `(rlnMicrographName, rlnHelicalTubeID)`, runs `HelicalContext` typing + twist-rate split if `rlnHelicalType` is absent (or `--retype`), resolves coherent filament polarity, circularly smooths `rlnAnglePsiPrior`, smooths transverse Stage-1 centerlines, and re-centers integer micrograph coordinates (`rlnCoordinateX/Y`, residual `rlnOriginX/YAngst`).
   - Emits per-type zero-copy `.vstack` descriptors (`vparticles::writeDescriptor` with `RELION_VPARTICLE_CACHE` inside `--o/vcache`) referencing resolved original micrographs (auto CTF premultiplication), runs focused unimodal Stage-2 `relion_refine[_mpi]` per polymorph, and merges outputs into `merged_particles.star` and `subclassification_summary.star`.
   - **GUI / Pipeliner integration (`"2D helix sub-classify"`, `HelixSubClassify/jobNNN/`, `PROC_HELIX_SUBCLASSIFY`)**: Exposed in the left job browser directly under `"2D consensus"` with **I/O**, **Extraction**, **Classification**, **Compute**, and **Running** tabs (`src/pipeline_jobs.{h,cpp}`, `src/gui_jobwindow.{h,cpp}`, `src/gui_mainwindow.cpp`). Accepts Stage-1 `_data.star` or `_optimiser.star`, supports Slurm queue submission (reserving `--nodes=1` and `nr_mpi - 1` worker tasks without `relion_launch` or an outer `mpirun`), forwards `--auto_mask*` and `--keep_all_iterations` / `--dont_keep_all_iterations`, checks `RELION_JOB_ABORT_NOW` between Stage-2 types, and writes `RELION_OUTPUT_NODES.star` registering `merged_particles.star` plus each type's `type_N/run_itXXX_{optimiser,data}.star` (or `type_N/particles.star` when `--prepare_only`). Pipeline cleanup (`PipeLine::cleanupJob`) moves `vcache` and Stage-2 per-type `run.{out,err}` logs to `Trash/`.

### Next-generation real-data GPU benchmark results (RTX 2080 Ti)

| Benchmark | Conditions / Notes | Fil / Segs | Fil Acc (2-way / Core) | Purity (2-way / Core) | Stay / Flicker / Pol | Res / Key Outcome |
|---|---|---|---|---|---|---|
| Multi-polymorph dataset B (full) | Post-hoc `relion_helix_filament_context --types_max 8` | 7,249 / 544,757 | **96.74%** / 90.55% | **0.9435** / 0.8996 | 0.758 / 0.021 / 1.000 | Resolved `328 A`, `1032 A`, `1814 A` twist modes |
| Single-polymorph control A (neg ctrl) | `helical_context 1..6`, cyl+auto_diam | 104 / 4,963 | `nr_types=1`, cross `ARI=1.0` | 20/20 non-junk | 0.558 / 0.001 / 0.822 | **`tube_SNR` 2.08 -> 2.43**, `yperp_d2_rms` **7.18 -> 5.29 A** |
| Multi-polymorph dataset B (400-fil Stage 1) | Decoupled mask + phase ring + twist split + split at 8..20 | 400 / 31,777 | **98.11%** / 86.36% | 0.9341 / 0.8421 | **0.825 / 0.004** / 0.727 | 6 chimeric + 2 twist splits (`313 vs 521 A`, `1010 vs 1974 A`) |
| Multi-polymorph dataset B (auto-mask) | Same + `--auto_mask_start_res 15` | 400 / 31,777 | 97.35% / 85.23% | **0.9541** / 0.8553 | 0.785 / 0.015 / 0.719 | Best `seg_ARI=0.7888`, `chimeric 7/32` |
| Multi-polymorph dataset A (450-fil Stage 1) | Decoupled mask + context `2..8` | 450 / 29,071 | — / 66.67% | — / 0.7519 | 0.702 / 0.025 / 0.701 | **`tube_SNR` 4.07 -> 4.42**, `yperp_d2` **20.5 -> 19.4 A** |
| Dataset B Stage 2 (`256 px @ 1.93 A/px`) | `relion_helix_subclassify` on Stage-1 types (`1..7`) | 400 / 31,483 | — | **0.9866** / 0.8906 | 0.71–0.92 / 0.00–0.03 / **1.000** | **`7.27 A`** (T5), **`10.51 A`** (T4), **`11.23 A`** (T3), `4.75A` peak `4.1–8.6x` |

---

## Robustness, `--continue` Persistence, Rare-Type Survival, & CTF Premultiplication Tracking

Additional hardening covered by `[helix_context_types]`, `[pipeline]`, `[gui][jobwindow][extract]`, and `tests/integration/test_helical_class2d_context.py`:

1. **Unified 2D class diameter + late auto-mask gating**:
   - `helixalign::classDiameter` and `autoMask2DClass` share unified axial-averaging / Gaussian shape-filtering logic; `strtod`-safe STAR parsing handles `"inf"` / `"nan"` cleanly.
   - 2D `--auto_mask` (`autoMaskThisIteration()`) and `--helical_auto_diameter` are late-gated (`iter >= 6`; override diameter gate via `RELION_HELICAL_AUTO_DIAMETER_START`), with `eff_start = std::max(auto_mask_start_res, 2.5 * pixel_size)` preventing premature masking at coarse pixel sizes.
   - `junkClasses()` marks non-positive resolution (`res <= 0`) and tiny orphan classes (`< 5` assigned segments, preventing single-particle leakage) as junk.
   - Helical 2D disables random-Gaussian blob initialisation automatically unless `--init_blobs` is given (`--fast_subsets` also warns and enforces a `300`-segment subset floor).
2. **Rare-type survival in hierarchical typing & `writeTypeFiles`**:
   - In `HelicalContext::runHierarchicalTyping()`, active leaf types (`Aff > 0`) that receive `n_allow == 0` under the dominant-class/12%-share `class_allowed` rules fall back to up to 3 top lift-ranked non-junk classes (`cls_tot_k[k] >= min_own_cnt`).
   - In `selectNrTypesByMixture(tmin)`, distributions falling below `tmin` fall back cleanly without crashing or collapsing below requested bounds.
   - In `HelicalContext::writeTypeFiles()`, populated types (`type_cnt > 0`) whose classes fail strict `min_count` / `min_frac` thresholds fall back in two stages (enriched classes `enrich >= 1.25` with `cnt >= min(2, type_cnt)`, else top score) using total assigned count `total_assigned` so every populated type exports `>= 1` class in `*_typeN_classes.star/.mrcs`.
3. **Full `--continue` persistence & explicit GUI flags**:
   - `parseContinue()` accepts `--no_init_blobs` / `--init_blobs` as harmless no-ops; GUI builders emit `--no_init_blobs` only when `!is_continue` and always emit explicit positive/negative flags (`--center_classes` / `--dont_center_classes`, `--bimodal_psi` / `--dont_bimodal_psi`, `--dont_helical_context` when Class2D context is Off / `K < 2` / Chain / Filament).
   - `optimiser.star` persists primary and secondary helical options (`rlnDoFixGpuBimodalPsi`, `rlnHelicalContextTypesTol`, `rlnHelicalContextTypesAgree`, `rlnHelicalContextTypesEvery`, `rlnHelicalContextWarm`, `rlnHelicalTrackSigmaAngstrom`) with backward-compatible defaults on legacy STAR files.
   - Re-passing unchanged `--helical_context_types` / `--helical_context_types_max` on `--continue` preserves auto-selected `nr_types` and `rlnHelicalContextTypesSelected`; changing min/max resets selection cleanly.
4. **CTF-premultiplied `--ctf_intact_first_peak` & multi-optics provenance**:
   - New optics metadata label `rlnCtfDataAreCtfPremultipliedIntactFirstPeak` (`ObservationModel`) recorded by `relion_preprocess` (`Extract`) and `relion_helix_subclassify`, cleared/propagated by `relion_motion_refine` (`Polish`), and normalized across combined STAR files by `relion_star_handler --combine` (`JoinStar`).
   - `relion_helix_subclassify` handles multiple optics groups (per-optics original/binned pixel size and box size, Slurm `--bind-to none`, and dynamic merged class ID stride when `K > 100`).
   - Exposed in the Extract GUI alongside `Premultiply with the CTF?` (automatically deactivated when premultiplication is Off, rectangular, rotated helix, or `Continue`).
   - `MlOptimiser::initialiseGeneral()` warns when input premultiplied optics disagree with `--ctf_intact_first_peak` at refinement time (or when the label is absent while `--ctf_intact_first_peak` is requested).



