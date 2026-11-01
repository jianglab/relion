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

`--helical_context_types 2 --helical_context_types_max 6` searches 2 to 6
types: first in the iteration before the prior is used, then every
`--helical_context_types_every` (5) iterations near the current number (one
fewer to two more).

For each number of types, the model is learnt three times, each time from two
thirds of the filaments, and the three resulting segment types are compared
(mean adjusted Rand index). Real types are found again from other filaments;
splits and merges of real types are not.

- **Rule (default):** the most types whose agreement is at least
  `--helical_context_types_agree` (0.9). If none reaches it: the best agreement
  at the first choice, the current number afterwards.
- `--helical_context_types_agree 0` gives the old rule: the fewest types whose
  held-out log-likelihood per segment is within `--helical_context_types_tol`
  of the best. Those log-likelihoods differ by only 0.001-0.01 between numbers
  of types, so it often took 2 types and merged two real ones.
- There is no full-range choice at the last iteration: in an earlier variant it
  jumped to a split in 2 of 9 runs (all other choices were right).
- Each choice prints the log-likelihoods, the agreements and the reason.

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
--helical_context_types 2 --helical_context_types_max 6` (at most K types; no
helical context with K < 2). Setting **Classify 2D helical segments?** to Yes
picks EM and Helical types; setting it back to No restores VDAM and Off unless
either was changed in between. Loading a job changes nothing.

Not yet tested: other data (junk filaments, more types), fewer EM iterations.

Tests: Class2D filament context tests in `tests/unit/test_pipeline_jobs.cpp` and
`tests/unit/test_gui_jobwindows.cpp`.
