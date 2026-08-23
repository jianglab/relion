#!/usr/bin/env python
"""
cooccurrence_2d_classes.py
--------------------------
Build a class co-occurrence matrix from RELION 2D classification output.

Each filament (micrograph + rlnHelicalTubeID) is treated as a "document."
Each 2D class is a "word."  Classes that co-occur on the same filaments
are structurally related.

Steps:
  1. Parse data.star → filament × class count matrix  F[filament, class]
  2. Compute co-occurrence  C = F^T F  (weighted by segment counts)
  3. Normalize to Jaccard similarity
  4. Cluster with hierarchical (Ward) on 1 - Jaccard
  5. Plot: heatmap of Jaccard matrix (reordered) + dendrogram

Outputs in OUT_DIR:
  cooccurrence_raw.npy        -- raw F^T F  (n_good_classes × n_good_classes)
  cooccurrence_jaccard.npy    -- Jaccard similarity matrix
  cooccurrence_heatmap.png    -- clustered heatmap
  cooccurrence_dendrogram.png -- dendrogram only
  class_clusters.txt          -- class → cluster assignment
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_STAR  = ("/net/jiang/home/nfahim/scratch/AB_APOE_IU/"
              "096_AB_APOE/relion_096/Class2D/job017/run_it025_data.star")
GOOD_TXT   = "class_triage/good_classes.txt"
OUT_DIR    = "class_triage"
N_CLUSTERS = 6   # number of meta-classes to cut the dendrogram into

# Nahian selections (1-based)
NAHIAN_JOB029 = {
    1,2,5,6,7,8,9,10,11,12,13,14,16,18,19,20,21,24,25,26,
    27,28,30,32,34,37,38,39,41,42,43,44,45,46,48,52,53,54,
    55,56,57,58,59,60,62,64,65,66,67,68,69,70,71,72,73,74,
    76,77,78,79,80,81,82,84,85,86,88,89,90,91
}
NAHIAN_JOB082 = {
    31,33,35,40,47,49,51,61,63,83,92,93,94,95,96,97,98,99,
    100,101,102,103,104,105,107,108,110,111,112,113,115,116,
    120,121,122,123,124,125,126,128,129,130,131,132,133,134,
    135,136,137,138,139,140,141,142,143,145,146,147,149,150,
    153,154,155,156
}


# ── Star parser ───────────────────────────────────────────────────────────────

def parse_particles(data_star, good_set):
    """
    Returns list of (filament_key, class_number) for particles in good_set.
    filament_key = (micrograph_name, helical_tube_id)
    """
    records = []
    in_particles = False
    in_loop      = False
    headers      = []
    col          = {}

    with open(data_star) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("data_particles"):
                in_particles = True
                in_loop      = False
                headers      = []
                col          = {}
                continue
            if s.startswith("data_") and in_particles:
                break
            if not in_particles:
                continue
            if s.startswith("loop_"):
                in_loop = True
                continue
            if in_loop and s.startswith("_rln"):
                name = s.split()[0]
                headers.append(name)
                col[name] = len(headers) - 1
                continue
            if in_loop and s and not s.startswith("#"):
                vals = s.split()
                if len(vals) < len(headers):
                    continue
                cls = int(vals[col["_rlnClassNumber"]])
                if cls not in good_set:
                    continue
                mic   = vals[col["_rlnMicrographName"]]
                tube  = vals[col["_rlnHelicalTubeID"]]
                records.append(((mic, tube), cls))
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_star",   default=DATA_STAR)
    ap.add_argument("--good_txt",    default=GOOD_TXT)
    ap.add_argument("--out_dir",     default=OUT_DIR)
    ap.add_argument("--n_clusters",  type=int, default=N_CLUSTERS)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load good class list
    with open(args.good_txt) as fh:
        good_idx = [int(l.strip()) for l in fh if l.strip() and not l.startswith("#")]
    good_set  = set(good_idx)
    n_classes = len(good_idx)
    cls2col   = {c: i for i, c in enumerate(good_idx)}   # class → matrix column
    print(f"Good classes: {n_classes}")

    # parse data.star
    print("Parsing data.star …")
    records = parse_particles(args.data_star, good_set)
    print(f"  Segments in good classes: {len(records)}")

    # collect unique filaments
    filaments = sorted(set(r[0] for r in records))
    n_fil     = len(filaments)
    fil2row   = {f: i for i, f in enumerate(filaments)}
    print(f"  Unique filaments: {n_fil}")

    # build filament × class count matrix  F[fil, cls]
    F = np.zeros((n_fil, n_classes), dtype=np.float32)
    for (fil, cls) in records:
        F[fil2row[fil], cls2col[cls]] += 1

    # co-occurrence  C = F^T F   shape (n_classes, n_classes)
    C = F.T @ F
    np.save(os.path.join(args.out_dir, "cooccurrence_raw.npy"), C)
    print(f"Co-occurrence matrix: {C.shape}")

    # Jaccard similarity
    # J(a,b) = C[a,b] / (C[a,a] + C[b,b] - C[a,b])
    diag  = np.diag(C)
    denom = diag[:, None] + diag[None, :] - C
    denom = np.where(denom == 0, 1e-9, denom)
    J     = C / denom
    np.fill_diagonal(J, 1.0)
    np.save(os.path.join(args.out_dir, "cooccurrence_jaccard.npy"), J)

    # hierarchical clustering on distance = 1 - J
    dist_vec  = squareform(1.0 - J, checks=False)
    dist_vec  = np.clip(dist_vec, 0, None)
    Z         = linkage(dist_vec, method="ward")
    order     = dendrogram(Z, no_plot=True)["leaves"]
    labels    = fcluster(Z, t=args.n_clusters, criterion="maxclust")

    # ── Heatmap ──────────────────────────────────────────────────────────────
    J_ord = J[np.ix_(order, order)]
    tick_labels = [str(good_idx[i]) for i in order]

    # color markers per class (Nahian labels)
    def label_color(idx):
        if idx in NAHIAN_JOB029: return "orange"
        if idx in NAHIAN_JOB082: return "steelblue"
        return "grey"

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(J_ord, cmap="hot", vmin=0, vmax=1, aspect="auto",
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.03, label="Jaccard similarity")

    # tick labels — only show every Nth to avoid clutter
    step = max(1, n_classes // 40)
    shown = list(range(0, n_classes, step))
    ax.set_xticks(shown)
    ax.set_xticklabels([tick_labels[i] for i in shown], rotation=90, fontsize=4)
    ax.set_yticks(shown)
    ax.set_yticklabels([tick_labels[i] for i in shown], fontsize=4)

    # draw cluster boundaries
    boundaries = []
    prev = labels[order[0]]
    for k, i in enumerate(order):
        if labels[i] != prev:
            boundaries.append(k - 0.5)
            prev = labels[i]
    for b in boundaries:
        ax.axhline(b, color="cyan", lw=0.8)
        ax.axvline(b, color="cyan", lw=0.8)

    ax.set_title(
        f"Class co-occurrence (Jaccard)  —  {n_classes} good classes, {n_fil} filaments\n"
        f"Reordered by Ward clustering, {args.n_clusters} meta-classes (cyan lines)\n"
        "orange = ThinT1.1  blue = MidThick  grey = unselected",
        fontsize=9
    )
    plt.tight_layout()
    hmap_path = os.path.join(args.out_dir, "cooccurrence_heatmap.png")
    fig.savefig(hmap_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {hmap_path}")

    # ── Dendrogram ───────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(16, 5))
    ddata = dendrogram(Z, ax=ax2, labels=[str(c) for c in good_idx],
                       color_threshold=Z[-(args.n_clusters - 1), 2],
                       leaf_font_size=4, above_threshold_color="grey")
    # color x-tick labels by Nahian selection
    for lbl in ax2.get_xmajorticklabels():
        idx = int(lbl.get_text())
        lbl.set_color(label_color(idx))
    ax2.set_title(
        f"Dendrogram of class co-occurrence  ({n_classes} good classes)\n"
        "orange = ThinT1.1  blue = MidThick  grey = unselected",
        fontsize=9
    )
    ax2.set_ylabel("Ward distance")
    plt.tight_layout()
    dend_path = os.path.join(args.out_dir, "cooccurrence_dendrogram.png")
    fig2.savefig(dend_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved → {dend_path}")

    # ── Class cluster assignments ─────────────────────────────────────────────
    cluster_path = os.path.join(args.out_dir, "class_clusters.txt")
    with open(cluster_path, "w") as fh:
        fh.write("# class_idx  cluster  nahian_label  n_segments\n")
        for i, cls in enumerate(good_idx):
            seg_count = int(F[:, i].sum())
            lbl = ("ThinT1.1" if cls in NAHIAN_JOB029
                   else "MidThick" if cls in NAHIAN_JOB082
                   else "unselected")
            fh.write(f"{cls:4d}  {labels[i]:2d}  {lbl:<12s}  {seg_count}\n")

    # summary per cluster
    print("\n── Cluster summary ──────────────────────────────────────────────")
    for k in range(1, args.n_clusters + 1):
        members = [good_idx[i] for i in range(n_classes) if labels[i] == k]
        thin = sum(1 for c in members if c in NAHIAN_JOB029)
        mid  = sum(1 for c in members if c in NAHIAN_JOB082)
        other= len(members) - thin - mid
        print(f"  Cluster {k}: n={len(members):3d}  ThinT1.1={thin}  MidThick={mid}  other={other}")
        print(f"    classes: {members}")

    print(f"\nSaved → {cluster_path}")
    print("All outputs in:", args.out_dir)


if __name__ == "__main__":
    main()
