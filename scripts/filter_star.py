#!/usr/bin/env python
"""
filter_star.py
--------------
Filter a RELION run_data.star to keep only particles from filaments
that belong to a target co-occurrence cluster.

The keep-set is read from filament_pop_scores.txt (produced by
filament_pop_score.py or select_populations_gui.py):
  # micrograph  tubeID  n_pop1  n_pop2  ...  f_pop1  f_pop2  ...  dominant

Usage
-----
python filter_star.py \\
    --data_star  Refine3D/job045/run_data.star \\
    --pop_scores class_triage/filament_pop_scores.txt \\
    --population 1 \\
    --out        class_triage/particles_pop1.star

or supply a plain keep-list (two columns: micrograph tubeID):
python filter_star.py \\
    --data_star  Refine3D/job045/run_data.star \\
    --keep       class_triage/keep_pop1.txt \\
    --out        class_triage/particles_pop1.star

Output is a valid RELION run_data.star suitable for use as
--reextract_data_star in the next RELION Extract job.
"""

import argparse
import os
import sys


# ── Star file parser/writer ───────────────────────────────────────────────────

def parse_star_particles(path):
    """
    Parse a RELION star file.  Returns:
      preamble   : list of lines up to and including the last column header
      data_rows  : list of raw data-row lines (particles section only)
      suffix     : list of lines after the particles data block
      col        : dict mapping _rlnXxx -> 0-based column index in each data row
    """
    with open(path) as fh:
        lines = fh.readlines()

    # Strip trailing newlines but keep them for faithful output
    BEFORE, IN_SECTION, IN_LOOP_HDR, IN_DATA = range(4)
    state = BEFORE

    preamble = []
    data_rows = []
    suffix = []
    col = {}
    n_cols = 0
    in_particles = False

    for raw in lines:
        stripped = raw.strip()

        # Detect section headers
        if stripped.startswith('data_'):
            if stripped == 'data_particles':
                in_particles = True
                state = IN_SECTION
            else:
                if state == IN_DATA:
                    # New section after particles data — put in suffix
                    state = BEFORE
                    suffix.append(raw)
                    in_particles = False
                    continue
                in_particles = False
                state = BEFORE
            preamble.append(raw)
            continue

        if state == IN_DATA:
            if stripped.startswith('data_'):
                suffix.append(raw)
                state = BEFORE
            elif stripped and not stripped.startswith('#'):
                data_rows.append(raw)
            else:
                # blank line or comment — treat as end of data block
                suffix.append(raw)
                state = BEFORE
            continue

        if state == IN_SECTION and stripped == 'loop_':
            state = IN_LOOP_HDR
            preamble.append(raw)
            continue

        if state == IN_LOOP_HDR:
            if stripped.startswith('_rln'):
                name = stripped.split()[0]
                col[name] = n_cols
                n_cols += 1
                preamble.append(raw)
                continue
            elif stripped and not stripped.startswith('#'):
                # First data row
                state = IN_DATA
                if in_particles:
                    data_rows.append(raw)
                else:
                    preamble.append(raw)
                continue

        preamble.append(raw)

    return preamble, data_rows, suffix, col


def write_star(path, preamble, data_rows, suffix):
    with open(path, 'w') as fh:
        fh.writelines(preamble)
        fh.writelines(data_rows)
        fh.writelines(suffix)


# ── Keep-set builders ──��──────────────────────────────────────────────────────

def keep_set_from_pop_scores(pop_scores_path, population, min_purity=0.0):
    """
    Read filament_pop_scores.txt and return a set of (micrograph, tubeID)
    strings for filaments whose dominant population == `population` (1-based)
    and whose dominant-population fraction >= min_purity.
    """
    keep = set()
    target = population - 1  # file stores 0-based dominant

    # Parse header to find column positions
    with open(pop_scores_path) as fh:
        lines = fh.readlines()

    header = None
    for line in lines:
        if line.startswith('#'):
            header = line.lstrip('# ').strip().split()
            break

    if header is None:
        raise ValueError(f"No header line found in {pop_scores_path}")

    # dominant is always last column; fractions follow counts
    # Header: micrograph tubeID n_pop1 ... n_popK f_pop1 ... f_popK dominant
    dominant_col = len(header) - 1
    mic_col = 0
    tube_col = 1

    # Find the fraction column for the target population
    # header looks like: ['micrograph', 'tubeID', 'n_pop1', ..., 'f_pop1', ..., 'dominant']
    frac_cols = [i for i, h in enumerate(header) if h.startswith('f_pop')]
    frac_col = frac_cols[target] if target < len(frac_cols) else None

    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) <= dominant_col:
            continue
        dom = int(parts[dominant_col])
        if dom != target:
            continue
        if frac_col is not None and min_purity > 0.0:
            frac = float(parts[frac_col])
            if frac < min_purity:
                continue
        mic = parts[mic_col]
        tube = parts[tube_col]
        keep.add((mic, tube))

    return keep


def keep_set_from_file(keep_path):
    """Read a plain two-column keep-list: micrograph  tubeID."""
    keep = set()
    with open(keep_path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                keep.add((parts[0], parts[1]))
    return keep


# ── Core filter ───────────────────────────────────────────────────────────────

def filter_data_star(in_star, keep_set, out_star):
    """
    Filter in_star, keeping only particles where
    (_rlnMicrographName, _rlnHelicalTubeID) is in keep_set.
    Writes the result to out_star.
    Returns (n_kept, n_total).
    """
    preamble, data_rows, suffix, col = parse_star_particles(in_star)

    if '_rlnMicrographName' not in col:
        raise KeyError("_rlnMicrographName not found in star file")
    if '_rlnHelicalTubeID' not in col:
        raise KeyError("_rlnHelicalTubeID not found in star file")

    c_mic  = col['_rlnMicrographName']
    c_tube = col['_rlnHelicalTubeID']

    kept = []
    for raw in data_rows:
        parts = raw.split()
        if len(parts) <= max(c_mic, c_tube):
            continue
        key = (parts[c_mic], parts[c_tube])
        if key in keep_set:
            kept.append(raw)

    write_star(out_star, preamble, kept, suffix)
    return len(kept), len(data_rows)


# ── CLI ───────────────────────────────────���───────────────────────────────��───

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument('--data_star',   required=True,
                    help='Refine3D run_data.star to filter')

    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument('--pop_scores',
                     help='filament_pop_scores.txt from filament_pop_score.py')
    grp.add_argument('--keep',
                     help='plain two-column keep-list: micrograph tubeID')

    ap.add_argument('--population', type=int, default=1,
                    help='1-based population index to keep (used with --pop_scores)')
    ap.add_argument('--min_purity', type=float, default=0.0,
                    help='minimum dominant-population fraction [0-1] (default 0.0)')
    ap.add_argument('--out', required=True,
                    help='output filtered star file path')

    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.pop_scores:
        keep = keep_set_from_pop_scores(args.pop_scores, args.population,
                                        args.min_purity)
        print(f"Keep-set: {len(keep)} filaments from population {args.population}")
    else:
        keep = keep_set_from_file(args.keep)
        print(f"Keep-set: {len(keep)} filaments from {args.keep}")

    n_kept, n_total = filter_data_star(args.data_star, keep, args.out)
    pct = 100 * n_kept / n_total if n_total else 0
    print(f"Filtered: {n_kept} / {n_total} particles kept ({pct:.1f}%)")
    print(f"Written → {args.out}")


if __name__ == '__main__':
    main()
