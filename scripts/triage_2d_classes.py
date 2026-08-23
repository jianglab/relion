#!/usr/bin/env python
"""
triage_2d_classes.py
--------------------
Remove obvious garbage from a RELION Class2D result.

Usage:
    python triage_2d_classes.py [options]

Reads:
    run_it025_classes.mrcs  -- class-average images
    run_it025_model.star    -- per-class statistics (ClassDistribution, EstimatedResolution)
    run_it025_data.star     -- per-particle class assignments

Writes to OUT_DIR:
    good_classes.txt        -- 1-based indices of retained classes
    bad_classes.txt         -- 1-based indices + rejection reason
    gallery_good.png        -- montage of good classes
    gallery_bad.png         -- montage of garbage classes
    triage_summary.txt      -- human-readable report
    good_particles.star     -- particles in good classes only (RELION-ready)
"""

import argparse
import sys
import os
import re
import numpy as np
import mrcfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

# ── Defaults ──────────────────────────────────────────────────────────────────
RELION_DIR = ("/net/jiang/home/nfahim/scratch/AB_APOE_IU/"
              "096_AB_APOE/relion_096/Class2D/job017")
ITER       = "025"
APIX       = 5.79          # Å/px
MIN_WIDTH  = 20.0          # Å  — narrower → noise
MAX_WIDTH  = 140.0         # Å  — wider   → not a filament
MIN_FRAC   = 0.0005        # ClassDistribution minimum; below this → too few particles
GAUSS_AMP_SNR = 1.5        # Gaussian amplitude must exceed this × noise-floor std
OUT_DIR    = "class_triage"
NCOLS_GALLERY = 20         # images per row in gallery


# ── Star-file helpers ─────────────────────────────────────────────────────────

def parse_model_star(path):
    """Return list of dicts, one per class, from data_model_classes block."""
    classes = []
    in_block = False
    in_loop  = False
    headers  = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith("data_model_classes"):
                in_block = True
                in_loop  = False
                headers  = []
                continue
            if not in_block:
                continue
            if line.startswith("data_"):  # new block — stop
                break
            if line.startswith("loop_"):
                in_loop = True
                continue
            if in_loop and line.startswith("_rln"):
                headers.append(line.split()[0])
                continue
            if in_loop and line and not line.startswith("#"):
                vals = line.split()
                if len(vals) == len(headers):
                    classes.append(dict(zip(headers, vals)))
    return classes


def parse_data_star(path):
    """
    Return (optics_block_lines, particles_header_lines, particle_rows,
            class_col_idx).
    optics_block_lines : verbatim lines for the data_optics block (to copy)
    particles_header_lines : verbatim header/loop lines for data_particles
    particle_rows : list of raw data lines (one per particle, as-is)
    class_col_idx : 0-based index of _rlnClassNumber in particle rows
    """
    optics_lines     = []
    part_hdr_lines   = []
    particle_rows    = []
    class_col_idx    = None

    section = None   # 'optics' | 'particles' | None
    in_loop = False
    col_idx = 0

    with open(path) as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            s = stripped.strip()

            if s.startswith("data_optics"):
                section  = "optics"
                in_loop  = False
                optics_lines.append(stripped)
                continue
            if s.startswith("data_particles"):
                section  = "particles"
                in_loop  = False
                col_idx  = 0
                part_hdr_lines.append(stripped)
                continue

            if section == "optics":
                optics_lines.append(stripped)
                continue

            if section == "particles":
                if s.startswith("loop_"):
                    in_loop = True
                    part_hdr_lines.append(stripped)
                    continue
                if in_loop and s.startswith("_rln"):
                    if "_rlnClassNumber" in s:
                        class_col_idx = col_idx
                    col_idx += 1
                    part_hdr_lines.append(stripped)
                    continue
                if in_loop and s and not s.startswith("#"):
                    particle_rows.append(stripped)
                    continue
                if not s or s.startswith("#"):
                    if not in_loop:
                        part_hdr_lines.append(stripped)
                    continue

    return optics_lines, part_hdr_lines, particle_rows, class_col_idx


def write_filtered_star(path, optics_lines, part_hdr_lines,
                        particle_rows, class_col_idx, good_set):
    """Write a new star file keeping only particles in good_set (1-based)."""
    with open(path, "w") as fh:
        fh.write("# version 50001\n\n")
        for line in optics_lines:
            fh.write(line + "\n")
        fh.write("\n# version 50001\n\n")
        for line in part_hdr_lines:
            fh.write(line + "\n")
        fh.write(" \n")
        kept = 0
        for row in particle_rows:
            cols = row.split()
            cls  = int(cols[class_col_idx])
            if cls in good_set:
                fh.write(row + "\n")
                kept += 1
    return kept


# ── Image analysis ────────────────────────────────────────────────────────────

def _gauss1d(x, amp, mu, sigma, base):
    return base + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def detect_filament_axis(img):
    img_pos  = img - img.min()
    std_cols = np.std(img_pos.mean(axis=0))   # variance of column means → high if axis=Y
    std_rows = np.std(img_pos.mean(axis=1))   # variance of row means    → high if axis=X
    return "vertical" if std_cols > std_rows else "horizontal"


def orient_horizontal(img):
    if detect_filament_axis(img) == "vertical":
        return img.T
    return img


def measure_width_angstrom(img, apix):
    """
    Fit a Gaussian to the perpendicular marginal profile.
    Returns (fwhm_angstrom, amplitude_snr) or (None, None) on failure.
    """
    img_h   = orient_horizontal(img)
    profile = img_h.mean(axis=1).astype(float)   # collapse along filament axis
    n       = len(profile)
    x       = np.arange(n)

    # noise estimate from outer 20% of profile
    edge_n  = max(2, n // 5)
    noise   = np.std(np.concatenate([profile[:edge_n], profile[-edge_n:]]))
    if noise == 0:
        noise = 1e-9

    mu0     = np.argmax(profile)
    amp0    = profile[mu0] - np.median(profile)
    sigma0  = n / 8.0
    base0   = np.median(profile)
    try:
        popt, _ = curve_fit(_gauss1d, x, profile,
                            p0=[amp0, mu0, sigma0, base0],
                            bounds=([-np.inf, 0, 1, -np.inf],
                                    [np.inf, n, n, np.inf]),
                            maxfev=4000)
        amp, mu, sigma, base = popt
        fwhm_px  = 2.3548 * abs(sigma)
        fwhm_ang = fwhm_px * apix
        snr      = abs(amp) / noise
        return fwhm_ang, snr
    except Exception:
        return None, None


def is_blank(img, min_std_percentile=5.0):
    """True if image looks like pure noise / DC (very low contrast)."""
    return img.std() < min_std_percentile * 1e-3 * abs(img).max()


# ── Gallery ───────────────────────────────────────────────────────────────────

def make_gallery(indices, images, labels, title, path, ncols=NCOLS_GALLERY,
                 cmap="gray"):
    if not indices:
        return
    n     = len(indices)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 0.9, nrows * 0.9 + 0.5))
    axes = np.array(axes).reshape(nrows, ncols)
    for k, (idx, lbl) in enumerate(zip(indices, labels)):
        r, c = divmod(k, ncols)
        ax   = axes[r, c]
        img  = images[idx - 1]
        vmin, vmax = np.percentile(img, [2, 98])
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(lbl, fontsize=4, pad=1)
        ax.axis("off")
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")
    fig.suptitle(title, fontsize=8, y=1.0)
    plt.tight_layout(pad=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relion_dir",  default=RELION_DIR)
    ap.add_argument("--iter",        default=ITER)
    ap.add_argument("--apix",        type=float, default=APIX)
    ap.add_argument("--min_frac",    type=float, default=MIN_FRAC,
                    help="min ClassDistribution; below → too few particles")
    ap.add_argument("--min_width",   type=float, default=MIN_WIDTH,
                    help="min filament width (Å)")
    ap.add_argument("--max_width",   type=float, default=MAX_WIDTH,
                    help="max filament width (Å)")
    ap.add_argument("--min_snr",     type=float, default=GAUSS_AMP_SNR,
                    help="min Gaussian amplitude SNR for valid class")
    ap.add_argument("--out_dir",     default=OUT_DIR)
    args = ap.parse_args()

    base  = os.path.join(args.relion_dir, f"run_it{args.iter}")
    mrcs_path  = base + "_classes.mrcs"
    model_path = base + "_model.star"
    data_path  = base + "_data.star"

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load images ──────────────────────────────────────────────────────────
    with mrcfile.open(mrcs_path, mode="r", permissive=True) as mrc:
        images = mrc.data.copy().astype(float)
    n_classes = len(images)
    print(f"Loaded {n_classes} class averages  ({images[0].shape[0]}×{images[0].shape[1]} px, "
          f"{args.apix} Å/px)")

    # ── Load model.star ───────────────────────────────────────────────────────
    model_classes = parse_model_star(model_path)
    assert len(model_classes) == n_classes, \
        f"model.star has {len(model_classes)} classes but mrcs has {n_classes}"

    total_particles_approx = 0
    frac_vals = []
    resol_vals = []
    for mc in model_classes:
        frac  = float(mc.get("_rlnClassDistribution", 0))
        resol = float(mc.get("_rlnEstimatedResolution", 999))
        frac_vals.append(frac)
        resol_vals.append(resol)

    # ── Measure image features ────────────────────────────────────────────────
    print("Measuring class features …")
    widths    = []
    snrs      = []
    for i, img in enumerate(images):
        w, snr = measure_width_angstrom(img, args.apix)
        widths.append(w)
        snrs.append(snr)
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{n_classes}")

    # ── Triage ───────────────────────────────────────────────────────────────
    good_idx   = []   # 1-based
    bad_idx    = []
    bad_reason = []

    for i in range(n_classes):
        cls_num = i + 1
        frac    = frac_vals[i]
        w       = widths[i]
        snr     = snrs[i]

        reasons = []
        if frac == 0.0:
            reasons.append("empty class (no particles)")
        elif frac < args.min_frac:
            reasons.append(f"too few particles (frac={frac:.5f} < {args.min_frac})")
        if w is None or snr is None:
            reasons.append("Gaussian fit failed")
        else:
            if snr < args.min_snr:
                reasons.append(f"low signal (SNR={snr:.1f})")
            elif w < args.min_width:
                reasons.append(f"too narrow ({w:.0f} Å < {args.min_width:.0f} Å)")
            elif w > args.max_width:
                reasons.append(f"too wide ({w:.0f} Å > {args.max_width:.0f} Å)")

        if reasons:
            bad_idx.append(cls_num)
            bad_reason.append("; ".join(reasons))
        else:
            good_idx.append(cls_num)

    good_set = set(good_idx)

    print(f"\n── Triage results ───────────────────────────────────────────────")
    print(f"  Total classes : {n_classes}")
    print(f"  Good          : {len(good_idx)}")
    print(f"  Garbage       : {len(bad_idx)}")

    # ── Write class lists ────────────────────────────────────────────────────
    good_txt = os.path.join(args.out_dir, "good_classes.txt")
    bad_txt  = os.path.join(args.out_dir, "bad_classes.txt")

    with open(good_txt, "w") as fh:
        for idx in good_idx:
            fh.write(f"{idx}\n")

    with open(bad_txt, "w") as fh:
        fh.write("# class_index  reason\n")
        for idx, reason in zip(bad_idx, bad_reason):
            fh.write(f"{idx:4d}  {reason}\n")

    # ── Galleries ────────────────────────────────────────────────────────────
    good_labels = []
    for idx in good_idx:
        w   = widths[idx - 1]
        f   = frac_vals[idx - 1]
        good_labels.append(f"#{idx}\n{w:.0f}Å  {f:.4f}")

    bad_labels = []
    for idx, reason in zip(bad_idx, bad_reason):
        short = reason.split(";")[0][:20]
        bad_labels.append(f"#{idx}\n{short}")

    make_gallery(good_idx, images, good_labels,
                 f"Good classes  (n={len(good_idx)})",
                 os.path.join(args.out_dir, "gallery_good.png"))
    make_gallery(bad_idx, images, bad_labels,
                 f"Garbage classes  (n={len(bad_idx)})",
                 os.path.join(args.out_dir, "gallery_bad.png"))

    # ── Filter data.star ─────────────────────────────────────────────────────
    print("Filtering data.star …")
    optics_lines, part_hdr_lines, particle_rows, class_col_idx = \
        parse_data_star(data_path)

    if class_col_idx is None:
        print("ERROR: _rlnClassNumber not found in data.star — skipping star output")
    else:
        out_star = os.path.join(args.out_dir, "good_particles.star")
        kept = write_filtered_star(out_star, optics_lines, part_hdr_lines,
                                   particle_rows, class_col_idx, good_set)
        print(f"  Total particles in data.star : {len(particle_rows)}")
        print(f"  Particles in good classes    : {kept}")
        print(f"  Saved → {out_star}")

    # ── Summary report ───────────────────────────────────────────────────────
    summary_path = os.path.join(args.out_dir, "triage_summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("2D Class Triage Summary\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"Source  : {mrcs_path}\n")
        fh.write(f"Apix    : {args.apix} Å/px\n")
        fh.write(f"Criteria: width {args.min_width}–{args.max_width} Å, "
                 f"SNR>{args.min_snr}, min_frac>{args.min_frac}\n\n")
        fh.write(f"Total classes : {n_classes}\n")
        fh.write(f"Good          : {len(good_idx)}\n")
        fh.write(f"Garbage       : {len(bad_idx)}\n\n")

        # breakdown of rejection reasons
        from collections import Counter
        reason_keys = []
        for r in bad_reason:
            if "empty" in r:
                reason_keys.append("empty class")
            elif "too few" in r:
                reason_keys.append("too few particles")
            elif "too wide" in r:
                reason_keys.append("too wide")
            elif "too narrow" in r:
                reason_keys.append("too narrow")
            elif "low signal" in r:
                reason_keys.append("low signal")
            elif "Gaussian fit failed" in r:
                reason_keys.append("Gaussian fit failed")
            else:
                reason_keys.append("other")
        for k, v in Counter(reason_keys).most_common():
            fh.write(f"  {k:<30s}: {v}\n")

        fh.write("\nGood class indices (1-based):\n")
        fh.write("  " + " ".join(str(i) for i in good_idx) + "\n")

        fh.write("\nGarbage class indices with reason:\n")
        for idx, reason in zip(bad_idx, bad_reason):
            fh.write(f"  {idx:4d}  {reason}\n")

    print(f"  Saved → {summary_path}")
    print("\nAll outputs in:", args.out_dir)


if __name__ == "__main__":
    main()
