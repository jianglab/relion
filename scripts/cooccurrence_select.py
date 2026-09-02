#!/usr/bin/env python
"""
cooccurrence_select.py
-----------------------
Top-level pipeline script for co-occurrence population selection.
This is the script invoked by the RELION External job.

Steps
-----
1. Check for precomputed Jaccard matrix; compute it if absent.
2. Launch the interactive population selector GUI.
3. Under RELION pipeline control, write RELION_JOB_EXIT_SUCCESS (or _FAILURE)
   on GUI exit.

RELION External job arguments (set in the GUI parameter panel)
--------------------------------------------------------------
  --i            Class2D run_it025_data.star
  --mrcs         Class2D run_it025_classes.mrcs
  --good_txt     class_triage/good_classes.txt   (from triage step)
  --refine3d_star Refine3D/jobXXX/run_data.star
  --o            output directory
  --min_segs     minimum segments per filament (default 20)
  --pipeline_control RELION pipeline-control output directory

If --good_txt does not exist, all classes present in --i are used.

Usage (standalone)
------------------
python cooccurrence_select.py \\
    --i            Class2D/job_XX/run_it025_data.star \\
    --mrcs         Class2D/job_XX/run_it025_classes.mrcs \\
    --good_txt     class_triage/good_classes.txt \\
    --refine3d_star Refine3D/job045/run_data.star \\
    --o            class_triage/populations \\
    --min_segs     20
"""

import argparse
import os
import sys
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def relion_exit(pipeline_control, success):
    """Write a RELION pipeline exit marker when pipeline control is active."""
    if not pipeline_control:
        return

    tag = 'RELION_JOB_EXIT_SUCCESS' if success else 'RELION_JOB_EXIT_FAILURE'
    path = os.path.join(pipeline_control, tag)
    open(path, 'w').close()


def all_classes_from_star(data_star):
    """Return sorted list of all class numbers present in data_star."""
    classes = set()
    with open(data_star) as fh:
        in_particles = in_loop = False
        headers = []
        col = {}
        for line in fh:
            s = line.strip()
            if s == 'data_particles':
                in_particles = True
                continue
            if s.startswith('data_') and in_particles:
                break
            if not in_particles:
                continue
            if s == 'loop_':
                in_loop = True
                continue
            if in_loop and s.startswith('_rln'):
                name = s.split()[0]
                col[name] = len(headers)
                headers.append(name)
                continue
            if in_loop and s and not s.startswith('#'):
                parts = s.split()
                if '_rlnClassNumber' in col:
                    c_cls = col['_rlnClassNumber']
                    if len(parts) > c_cls:
                        classes.add(int(parts[c_cls]))
    return sorted(classes)


def parse_apix_from_star(data_star):
    """Read _rlnImagePixelSize from the data_optics block. Returns float or None."""
    try:
        with open(data_star) as fh:
            in_optics = in_loop = False
            headers = []
            for line in fh:
                s = line.strip()
                if s == 'data_optics':
                    in_optics = True; continue
                if s.startswith('data_') and in_optics:
                    break
                if not in_optics:
                    continue
                if s == 'loop_':
                    in_loop = True; continue
                if in_loop and s.startswith('_rln'):
                    headers.append(s.split()[0]); continue
                if in_loop and s and not s.startswith('#'):
                    parts = s.split()
                    for i, h in enumerate(headers):
                        if h == '_rlnImagePixelSize' and i < len(parts):
                            return float(parts[i])
    except Exception:
        pass
    return None


def run_triage(data_star, out_dir):
    """Auto-triage Class2D output to produce good_classes.txt.

    Derives relion_dir and iteration from the data_star path, reads pixel size
    from the optics block, and runs triage_2d_classes.py as a subprocess so
    its matplotlib.use("Agg") call stays isolated from the GUI process.

    Returns the path to the generated good_classes.txt.
    """
    import subprocess, re

    relion_dir = os.path.dirname(os.path.abspath(data_star))
    basename   = os.path.basename(data_star)
    m = re.search(r'run_it(\d+)', basename)
    iteration  = m.group(1) if m else '025'

    apix = parse_apix_from_star(data_star)
    triage_dir = os.path.join(out_dir, 'auto_triage')
    os.makedirs(triage_dir, exist_ok=True)

    script = os.path.join(os.path.dirname(__file__), 'triage_2d_classes.py')
    cmd = [
        sys.executable, script,
        '--relion_dir', relion_dir,
        '--iter',       iteration,
        '--out_dir',    triage_dir,
    ]
    if apix is not None:
        cmd += ['--apix', str(apix)]
    print(f'Auto-triaging classes (pixel size {apix} Å/px) …')
    subprocess.run(cmd, check=True)

    good_txt = os.path.join(triage_dir, 'good_classes.txt')
    if not os.path.exists(good_txt):
        raise RuntimeError(f'Triage did not produce {good_txt}')
    return good_txt


def compute_jaccard(data_star, good_idx, out_dir):
    """Run cooccurrence_2d_classes.py in a subprocess and return path to jaccard.npy.

    Using a subprocess keeps cooccurrence_2d_classes.py's matplotlib.use("Agg")
    call isolated so it cannot poison the interactive backend needed by the GUI.
    """
    import subprocess

    good_txt = os.path.join(out_dir, 'good_classes_auto.txt')
    with open(good_txt, 'w') as fh:
        fh.write('# auto-generated: all classes\n')
        for cls in good_idx:
            fh.write(f'{cls}\n')

    script = os.path.join(os.path.dirname(__file__), 'cooccurrence_2d_classes.py')
    cmd = [
        sys.executable, script,
        '--data_star', data_star,
        '--good_txt',  good_txt,
        '--out_dir',   out_dir,
    ]
    result = subprocess.run(cmd, check=True)

    jaccard_path = os.path.join(out_dir, 'cooccurrence_jaccard.npy')
    return jaccard_path, good_txt


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument('--i',             required=True,
                    help='Class2D run_data.star (with _rlnClassNumber)')
    ap.add_argument('--mrcs',          required=True,
                    help='Class2D class images .mrcs stack')
    ap.add_argument('--good_txt',      default=None,
                    help='good_classes.txt from triage (optional; auto-triages if absent)')
    ap.add_argument('--refine3d_star', required=True,
                    help='Refine3D run_data.star to filter')
    ap.add_argument('--o',             required=True,
                    help='output directory')
    ap.add_argument('--min_segs',      type=int, default=20,
                    help='minimum segments per filament (default 20)')
    ap.add_argument('--jaccard',       default=None,
                    help='precomputed cooccurrence_jaccard.npy (skip recompute)')
    ap.add_argument('--pipeline_control', default='',
                    help='RELION pipeline-control output directory')

    args = ap.parse_args()

    os.makedirs(args.o, exist_ok=True)

    try:
        # ── Step 1: resolve good classes ─────────────────────────────────────
        if args.good_txt and os.path.exists(args.good_txt):
            good_txt = args.good_txt
            with open(good_txt) as fh:
                good_idx = [int(l.strip())
                            for l in fh
                            if l.strip() and not l.startswith('#')]
            print(f'Good classes: {len(good_idx)} from {good_txt}')
        else:
            if args.good_txt:
                print(f'WARNING: {args.good_txt} not found — running auto-triage')
            else:
                print('No good_classes.txt provided — running auto-triage …')
            good_txt = run_triage(args.i, args.o)
            with open(good_txt) as fh:
                good_idx = [int(l.strip())
                            for l in fh
                            if l.strip() and not l.startswith('#')]
            print(f'  Auto-triage: {len(good_idx)} good classes → {good_txt}')

        # ── Step 2: Jaccard matrix ────────────────────────────────────────────
        if args.jaccard and os.path.exists(args.jaccard):
            jaccard_path = args.jaccard
            print(f'Using precomputed Jaccard matrix: {jaccard_path}')
        else:
            default_jaccard = os.path.join(args.o, 'cooccurrence_jaccard.npy')
            if os.path.exists(default_jaccard):
                jaccard_path = default_jaccard
                print(f'Using existing Jaccard matrix: {jaccard_path}')
            else:
                print('Computing Jaccard co-occurrence matrix …')
                jaccard_path, good_txt = compute_jaccard(args.i, good_idx, args.o)
                print(f'  Saved → {jaccard_path}')

        J = np.load(jaccard_path)

        # Re-read good_idx from the file used for Jaccard (may have been rewritten)
        with open(good_txt) as fh:
            good_idx = [int(l.strip())
                        for l in fh
                        if l.strip() and not l.startswith('#')]

        if J.shape != (len(good_idx), len(good_idx)):
            raise ValueError(
                f'Jaccard matrix shape {J.shape} does not match '
                f'{len(good_idx)} good classes in {good_txt}'
            )

        # ── Step 3: launch GUI ────────────────────────────────────────────────
        import mrcfile
        from select_populations_gui import PopulationSelectorGUI

        print('Loading class images …')
        with mrcfile.open(args.mrcs, mode='r', permissive=True) as mrc:
            images = mrc.data.copy()

        print('Launching interactive population selector …')
        print('  Use the K slider to choose number of clusters.')
        print('  Navigate clusters with ◀ / ▶ buttons.')
        print('  Type a label for each cluster and press Enter.')
        print('  Click "Export ★" to write filtered star files.')
        print('  Close the window when done.\n')

        gui = PopulationSelectorGUI(
            J             = J,
            good_idx      = good_idx,
            images        = images,
            class2d_star  = args.i,
            refine3d_star = args.refine3d_star,
            out_dir       = args.o,
            min_segs      = args.min_segs,
        )
        gui.run()

        # ── Step 4: RELION exit marker ────────────────────────────────────────
        relion_exit(args.pipeline_control, success=True)
        print(f'Done.  Output in {args.o}')

    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        relion_exit(args.pipeline_control, success=False)
        sys.exit(1)


if __name__ == '__main__':
    main()
