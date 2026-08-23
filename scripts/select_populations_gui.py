#!/usr/bin/env python
"""
select_populations_gui.py
--------------------------
Interactive two-panel GUI for co-occurrence population selection.

Left panel  : Jaccard co-occurrence heatmap with Ward dendrogram.
              A red dashed cut-line shows where K clusters are formed.
Right panel : Gallery of 2D class images belonging to the currently
              selected cluster.

Controls (bottom strip):
  K slider        — choose number of clusters (2–10)
  < Cluster / Cluster > buttons — navigate between clusters
  Label textbox   — type a name for the current cluster, press Enter
  Export button   — write one filtered run_data.star per labelled cluster

Usage
-----
python select_populations_gui.py \\
    --jaccard   class_triage/cooccurrence_jaccard.npy \\
    --good_txt  class_triage/good_classes.txt \\
    --mrcs      Class2D/job_XX/run_it025_classes.mrcs \\
    --class2d_star  Class2D/job_XX/run_it025_data.star \\
    --refine3d_star Refine3D/job045/run_data.star \\
    --out_dir   class_triage/populations \\
    --min_segs  20
"""

import argparse
import os
import sys
import numpy as np
import mrcfile

# Must set an interactive backend before importing pyplot.
# Try TkAgg first (most portable), fall back to Qt5Agg.
# Fail early with a clear message if no display is available —
# this job must be run locally, not submitted to the cluster.
if not os.environ.get('DISPLAY') and sys.platform != 'darwin':
    print('ERROR: No DISPLAY environment variable found.\n'
          'The Co-occurrence select job must be run locally (not submitted\n'
          'to the cluster queue) because it opens an interactive window.\n'
          'In RELION: set "Submit to queue?" to No before clicking Run.',
          file=sys.stderr)
    sys.exit(1)

import matplotlib
for _backend in ('TkAgg', 'Qt5Agg', 'GTK3Agg'):
    try:
        matplotlib.use(_backend)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.close()
        break
    except Exception:
        continue
else:
    print('ERROR: Could not find a working interactive matplotlib backend.\n'
          'Install python3-tk (for TkAgg) or PyQt5 (for Qt5Agg).',
          file=sys.stderr)
    sys.exit(1)

import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, TextBox
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

from filter_star import filter_data_star


CLUSTER_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
    '#469990', '#9a6324',
]


# ── Filament membership ────────────────────────────────────────────────────────

def compute_filament_memberships(class2d_star, cls_to_cluster, min_segs):
    """
    Parse Class2D data.star.  For each filament (micrograph, tubeID)
    count how many of its particles fall into each cluster.
    Returns:
      filaments   : list of (mic, tube) tuples
      count_mat   : (n_fil, K) array of segment counts per cluster
      dominant    : (n_fil,) array of dominant cluster index (0-based)
      K           : number of clusters
    """
    K = max(cls_to_cluster.values()) + 1  # 0-based max + 1

    fil_counts = {}
    with open(class2d_star) as fh:
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
                if len(parts) <= max(col.get('_rlnMicrographName', 0),
                                     col.get('_rlnHelicalTubeID', 0),
                                     col.get('_rlnClassNumber', 0)):
                    continue
                cls  = int(parts[col['_rlnClassNumber']])
                mic  = parts[col['_rlnMicrographName']]
                tube = parts[col['_rlnHelicalTubeID']]
                if cls not in cls_to_cluster:
                    continue
                cluster = cls_to_cluster[cls]
                key = (mic, tube)
                if key not in fil_counts:
                    fil_counts[key] = np.zeros(K, dtype=int)
                fil_counts[key][cluster] += 1

    filaments = []
    count_mat = []
    for fil, cnts in fil_counts.items():
        if cnts.sum() >= min_segs:
            filaments.append(fil)
            count_mat.append(cnts)

    if not filaments:
        return [], np.zeros((0, K), dtype=int), np.zeros(0, dtype=int), K

    count_mat = np.array(count_mat)
    dominant  = count_mat.argmax(axis=1)
    return filaments, count_mat, dominant, K


# ── GUI class ─────────────────────────────────────────────────────────────────

class PopulationSelectorGUI:

    def __init__(self, J, good_idx, images, class2d_star, refine3d_star,
                 out_dir, min_segs=20):
        self.J              = J
        self.good_idx       = good_idx        # list of 1-based class numbers
        self.images         = images          # (n_total_classes, H, W)
        self.class2d_star   = class2d_star
        self.refine3d_star  = refine3d_star
        self.out_dir        = out_dir
        self.min_segs       = min_segs

        os.makedirs(out_dir, exist_ok=True)

        # Compute Ward linkage once
        dist = squareform(np.clip(1.0 - J, 0, None), checks=False)
        self.Z = linkage(dist, method='ward')

        # State
        self.K               = 4
        self.current_cluster = 1        # 1-based
        self.cluster_names   = {}       # cluster_id (1-based) -> name string
        self.cluster_labels  = None     # (n_good,) array, 1-based
        self.dend_order      = None     # leaf order from dendrogram

        # Filament data — parsed once, recomputed in memory on K change
        print('Pre-loading particle assignments …')
        self._raw_particles = self._parse_raw_particles()
        print(f'  {len(self._raw_particles)} particles in good classes')
        self._filaments = []
        self._count_mat = np.zeros((0, self.K), dtype=int)
        self._dominant  = np.zeros(0, dtype=int)
        self._fil_K     = None   # K for which _filaments/_count_mat are valid

        self._recut()
        self._build_figure()
        self._redraw_all()

    # ── Clustering ────────────────────────────────────────────────────────────

    def _recut(self):
        self.cluster_labels = fcluster(self.Z, t=self.K, criterion='maxclust')
        dend = dendrogram(self.Z, no_plot=True)
        self.dend_order = dend['leaves']

    def _cut_height(self):
        """Y-axis height of the horizontal cut for K clusters."""
        heights = sorted(self.Z[:, 2])
        if self.K >= len(heights):
            return heights[0] * 0.5
        return (heights[-self.K] + heights[-self.K + 1]) / 2.0

    def _parse_raw_particles(self):
        """Parse class2d_star once → list of (cls, mic, tube) for good classes."""
        good_set = set(self.good_idx)
        result = []
        with open(self.class2d_star) as fh:
            in_particles = in_loop = False
            headers = []
            col = {}
            for line in fh:
                s = line.strip()
                if s == 'data_particles':
                    in_particles = True; continue
                if s.startswith('data_') and in_particles:
                    break
                if not in_particles:
                    continue
                if s == 'loop_':
                    in_loop = True; continue
                if in_loop and s.startswith('_rln'):
                    name = s.split()[0]
                    col[name] = len(headers)
                    headers.append(name)
                    continue
                if in_loop and s and not s.startswith('#'):
                    parts = s.split()
                    try:
                        cls  = int(parts[col['_rlnClassNumber']])
                        mic  =     parts[col['_rlnMicrographName']]
                        tube =     parts[col['_rlnHelicalTubeID']]
                    except (KeyError, IndexError, ValueError):
                        continue
                    if cls in good_set:
                        result.append((cls, mic, tube))
        return result

    def _ensure_fil_data(self):
        """Recompute filament membership from in-memory particles if K changed."""
        if self._fil_K == self.K:
            return
        cls_to_cluster = {
            self.good_idx[i]: int(self.cluster_labels[i]) - 1
            for i in range(len(self.good_idx))
        }
        K = self.K
        fil_counts = {}
        for cls, mic, tube in self._raw_particles:
            cluster = cls_to_cluster.get(cls)
            if cluster is None:
                continue
            key = (mic, tube)
            if key not in fil_counts:
                fil_counts[key] = np.zeros(K, dtype=int)
            fil_counts[key][cluster] += 1

        filaments, count_mat = [], []
        for fil, cnts in fil_counts.items():
            if cnts.sum() >= self.min_segs:
                filaments.append(fil)
                count_mat.append(cnts)

        if filaments:
            self._count_mat = np.array(count_mat)
            self._filaments = filaments
            self._dominant  = self._count_mat.argmax(axis=1)
        else:
            self._filaments = []
            self._count_mat = np.zeros((0, K), dtype=int)
            self._dominant  = np.zeros(0, dtype=int)
        self._fil_K = K

    def _classes_in_cluster(self, k):
        """Return list of 1-based class numbers in cluster k."""
        return [self.good_idx[i]
                for i, lbl in enumerate(self.cluster_labels)
                if lbl == k]

    # ── Figure construction ───────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(18, 12))
        self.fig.patch.set_facecolor('#1e1e1e')

        # ── Main grid: left (heatmap) | right (gallery) | bottom (widgets) ──
        outer = gridspec.GridSpec(
            2, 2,
            height_ratios=[10, 2],
            width_ratios=[1, 1],
            hspace=0.06, wspace=0.08,
            left=0.04, right=0.97,
            top=0.91, bottom=0.03,
        )

        # Left: dendrogram (top) + heatmap (bottom)
        left_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[0, 0],
            height_ratios=[1, 5], hspace=0.03,
        )
        self.ax_dend  = self.fig.add_subplot(left_gs[0])
        self.ax_heat  = self.fig.add_subplot(left_gs[1])

        # Right: class gallery
        self.ax_gal   = self.fig.add_subplot(outer[0, 1])

        # ── Bottom widgets: two rows ──────────────────────────────────────────
        bot = outer[1, :]
        pos = bot.get_position(self.fig)
        bx, by, bw, bh = pos.x0, pos.y0, pos.width, pos.height
        row_h = bh * 0.42
        row1_y = by + bh * 0.54   # top row
        row0_y = by + bh * 0.04   # bottom row
        pad = 0.003

        # Row 1: K slider | cut-height slider | prev | next | label | export
        max_h = float(self.Z[:, 2].max())
        init_h = self._cut_height()

        ax_k      = self.fig.add_axes([bx,              row1_y, bw * 0.21, row_h - pad])
        ax_height = self.fig.add_axes([bx + bw * 0.24,  row1_y, bw * 0.21, row_h - pad])
        ax_prev   = self.fig.add_axes([bx + bw * 0.47,  row1_y, bw * 0.09, row_h - pad])
        ax_next   = self.fig.add_axes([bx + bw * 0.57,  row1_y, bw * 0.09, row_h - pad])
        ax_label  = self.fig.add_axes([bx + bw * 0.68,  row1_y, bw * 0.19, row_h - pad])
        ax_exp    = self.fig.add_axes([bx + bw * 0.89,  row1_y, bw * 0.09, row_h - pad])

        # Row 2: contrast (vmin) slider | contrast (vmax) slider
        ax_vmin   = self.fig.add_axes([bx,              row0_y, bw * 0.44, row_h - pad])
        ax_vmax   = self.fig.add_axes([bx + bw * 0.46,  row0_y, bw * 0.44, row_h - pad])

        self.slider_K      = Slider(ax_k,      'K',   2,    20,
                                    valinit=self.K, valstep=1, color='#4a90d9')
        self.slider_height = Slider(ax_height, 'Ht', 0.0, max_h,
                                    valinit=init_h, color='#e07030')
        self.btn_prev      = Button(ax_prev,  '◀',
                                    color='#555555', hovercolor='#777777')
        self.btn_next      = Button(ax_next,  '▶',
                                    color='#555555', hovercolor='#777777')
        self.textbox       = TextBox(ax_label, '', initial='',
                                     color='#666666', hovercolor='#888888')
        ax_label.set_title('Label', color='#87ceeb', fontsize=8, pad=2)
        self.btn_export    = Button(ax_exp,   'Export ★',
                                    color='#2a5a2a', hovercolor='#3a7a3a')
        self.slider_vmin   = Slider(ax_vmin, 'J lo', 0.0, 0.9,
                                    valinit=0.0, color='#555555')
        self.slider_vmax   = Slider(ax_vmax, 'J hi', 0.1, 1.0,
                                    valinit=1.0, color='#888888')

        for ax_w in [ax_k, ax_height, ax_prev, ax_next, ax_label, ax_exp,
                     ax_vmin, ax_vmax]:
            ax_w.tick_params(left=False, bottom=False,
                             labelleft=False, labelbottom=False)

        self.btn_export.label.set_color('#87ceeb')
        for sl in (self.slider_K, self.slider_height,
                   self.slider_vmin, self.slider_vmax):
            sl.vline.set_visible(False)
            sl.label.set_color('#87ceeb')
        self.slider_K.on_changed(self._on_k_changed)
        self.slider_height.on_changed(self._on_height_changed)
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.textbox.on_submit(self._on_label_submit)
        self.btn_export.on_clicked(self._on_export)
        self.slider_vmin.on_changed(self._on_contrast_changed)
        self.slider_vmax.on_changed(self._on_contrast_changed)
        self.fig.canvas.mpl_connect('button_press_event', self._on_heatmap_click)

        self.fig.suptitle(
            'Co-occurrence Selector'
            '  ·  click heatmap to jump  ·  K / Height to split  ·  J lo/hi for contrast',
            fontsize=10, fontweight='bold', color='white', y=0.97)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _style_ax(self, ax):
        ax.set_facecolor('#1e1e1e')
        for sp in ax.spines.values():
            sp.set_color('#555555')
        ax.tick_params(colors='#aaaaaa')

    def _redraw_all(self):
        self._draw_dendrogram()
        self._draw_heatmap()
        self._draw_gallery()
        self.fig.canvas.draw_idle()

    def _draw_dendrogram(self):
        ax = self.ax_dend
        ax.clear()
        self._style_ax(ax)

        dendrogram(
            self.Z, ax=ax,
            link_color_func=lambda _: '#888888',
            no_labels=True,
            above_threshold_color='#888888',
        )
        cut_h = self._cut_height()
        ax.axhline(cut_h, color='#ff4444', lw=1.5, ls='--', alpha=0.9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Ward dendrogram  (K = {self.K})',
                     fontsize=9, color='#cccccc', pad=2)

    def _draw_heatmap(self):
        ax = self.ax_heat
        ax.clear()
        self._style_ax(ax)

        order = self.dend_order
        labels_ord = self.cluster_labels[order]
        J_ord = self.J[np.ix_(order, order)]

        vmin = float(self.slider_vmin.val)
        vmax = float(self.slider_vmax.val)
        if vmax <= vmin:
            vmax = vmin + 0.01
        ax.imshow(J_ord, aspect='auto', cmap='inferno',
                  vmin=vmin, vmax=vmax, interpolation='nearest')

        # Draw cluster boundary boxes
        pos = 0
        for k in range(1, self.K + 1):
            cnt = int((labels_ord == k).sum())
            if cnt == 0:
                continue
            color = CLUSTER_COLORS[(k - 1) % len(CLUSTER_COLORS)]
            lw = 3.0 if k == self.current_cluster else 1.5
            alpha = 1.0 if k == self.current_cluster else 0.6
            rect = plt.Rectangle(
                (pos - 0.5, pos - 0.5), cnt, cnt,
                fill=(k == self.current_cluster),
                facecolor=color, alpha=0.12 if k == self.current_cluster else 0,
                edgecolor=color, lw=lw,
            )
            ax.add_patch(rect)

            mid = pos + cnt / 2
            name = self.cluster_names.get(k, f'C{k}')
            # mean internal Jaccard
            mask = (labels_ord == k)
            sub = J_ord[np.ix_(mask, mask)]
            off = sub[~np.eye(sub.shape[0], dtype=bool)]
            mean_j = off.mean() if len(off) else 1.0
            label_txt = f'{name} J={mean_j:.2f}'
            # rotate vertically for narrow clusters to avoid overlap
            if cnt < 15:
                ax.text(mid, -0.8, label_txt,
                        ha='center', va='bottom', fontsize=6,
                        color=color, fontweight='bold', clip_on=False,
                        rotation=90)
            else:
                ax.text(mid, -1.5, label_txt,
                        ha='center', va='bottom', fontsize=7,
                        color=color, fontweight='bold', clip_on=False)
            pos += cnt

        n = len(self.good_idx)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Jaccard co-occurrence matrix', fontsize=9,
                     color='#cccccc', pad=2)

    def _draw_gallery(self):
        ax = self.ax_gal
        ax.clear()
        self._style_ax(ax)

        self._ensure_fil_data()
        cluster_classes = self._classes_in_cluster(self.current_cluster)
        n = len(cluster_classes)
        name = self.cluster_names.get(self.current_cluster,
                                      f'Cluster {self.current_cluster}')
        color = CLUSTER_COLORS[(self.current_cluster - 1) % len(CLUSTER_COLORS)]

        k0 = self.current_cluster - 1
        if len(self._filaments):
            mask  = self._dominant == k0
            n_fil = int(mask.sum())
            n_seg = int(self._count_mat[mask].sum())
            stats = f'  ·  {n_fil} filaments  ·  {n_seg} segs'
        else:
            stats = ''

        ax.set_title(
            f'Cluster {self.current_cluster} / {self.K}  —  {name}  '
            f'({n} classes{stats})',
            fontsize=10, color=color, fontweight='bold', pad=4,
        )

        if n == 0:
            ax.text(0.5, 0.5, 'No classes in this cluster',
                    ha='center', va='center', color='#888888',
                    transform=ax.transAxes)
            ax.axis('off')
            return

        ncols = min(8, int(np.ceil(np.sqrt(n * 1.5))))
        nrows = int(np.ceil(n / ncols))
        H, W  = self.images.shape[1], self.images.shape[2]

        mosaic = np.full((nrows * H, ncols * W), np.nan)
        for idx, cls in enumerate(cluster_classes):
            r, c   = divmod(idx, ncols)
            img    = self.images[cls - 1].astype(float)
            mu, sg = img.mean(), img.std() + 1e-9
            img    = np.clip((img - mu) / sg, -3, 3)
            mosaic[r * H:(r + 1) * H, c * W:(c + 1) * W] = img

        ax.imshow(mosaic, cmap='gray', aspect='equal',
                  interpolation='nearest',
                  vmin=np.nanmin(mosaic), vmax=np.nanmax(mosaic))

        for idx, cls in enumerate(cluster_classes):
            r, c = divmod(idx, ncols)
            ax.text(c * W + 2, r * H + 4, str(cls),
                    color='yellow', fontsize=5.5, va='top',
                    fontweight='bold')

        ax.axis('off')

    # ── Widget callbacks ──────────────────────────────────────────────────────

    def _on_k_changed(self, val):
        self.K = int(round(val))
        self.current_cluster = min(self.current_cluster, self.K)
        self._recut()
        # Sync height slider without triggering its callback
        self.slider_height.eventson = False
        self.slider_height.set_val(self._cut_height())
        self.slider_height.eventson = True
        self.textbox.set_val(self.cluster_names.get(self.current_cluster, ''))
        self._redraw_all()

    def _on_height_changed(self, val):
        from scipy.cluster.hierarchy import fcluster
        h = float(val)
        labels = fcluster(self.Z, t=max(h, 1e-9), criterion='distance')
        new_K = int(labels.max())
        if new_K != self.K:
            self.K = new_K
            self.cluster_labels = labels
            dend = dendrogram(self.Z, no_plot=True)
            self.dend_order = dend['leaves']
            self.current_cluster = min(self.current_cluster, self.K)
            # Sync K slider without triggering its callback
            self.slider_K.eventson = False
            self.slider_K.set_val(self.K)
            self.slider_K.eventson = True
            self.textbox.set_val(self.cluster_names.get(self.current_cluster, ''))
            self._redraw_all()

    def _on_contrast_changed(self, _val):
        self._draw_heatmap()
        self.fig.canvas.draw_idle()

    def _on_prev(self, _event):
        self.current_cluster = max(1, self.current_cluster - 1)
        self.textbox.set_val(self.cluster_names.get(self.current_cluster, ''))
        self._draw_heatmap()
        self._draw_gallery()
        self.fig.canvas.draw_idle()

    def _on_next(self, _event):
        self.current_cluster = min(self.K, self.current_cluster + 1)
        self.textbox.set_val(self.cluster_names.get(self.current_cluster, ''))
        self._draw_heatmap()
        self._draw_gallery()
        self.fig.canvas.draw_idle()

    def _on_label_submit(self, text):
        self.cluster_names[self.current_cluster] = text.strip()
        self._draw_heatmap()
        self._draw_gallery()
        self.fig.canvas.draw_idle()

    def _on_heatmap_click(self, event):
        if event.inaxes is not self.ax_heat:
            return
        col = int(round(event.xdata))
        n = len(self.good_idx)
        if not (0 <= col < n):
            return
        clicked_cluster = int(self.cluster_labels[self.dend_order[col]])
        if clicked_cluster != self.current_cluster:
            self.current_cluster = clicked_cluster
            self.textbox.set_val(self.cluster_names.get(self.current_cluster, ''))
            self._draw_heatmap()
            self._draw_gallery()
            self.fig.canvas.draw_idle()

    def _on_export(self, _event):
        k    = self.current_cluster
        name = self.cluster_names.get(k, f'cluster{k}')
        if k not in self.cluster_names:
            print(f'  (no label set — writing as "{name}")')
        print(f'\nExporting cluster {k} ({name}) to {self.out_dir} …')
        self._ensure_fil_data()
        self._write_pop_classes()
        self._export_cluster(k, name)
        print('Export complete.\n')

    # ── Export logic ──────────────────────────────────────────────────────────

    def _write_pop_classes(self):
        path = os.path.join(self.out_dir, 'pop_classes.txt')
        with open(path, 'w') as fh:
            fh.write(f'# class  cluster  label  (K={self.K})\n')
            for i, cls in enumerate(self.good_idx):
                k    = int(self.cluster_labels[i])
                name = self.cluster_names.get(k, f'cluster{k}')
                fh.write(f'{cls:4d}  {k}  {name}\n')
        print(f'  Saved class assignments → {path}')

    def _export_cluster(self, k, name):
        """Write particles_{name}.star for cluster k (1-based), current view only."""
        if not self._filaments:
            print('  WARNING: no filaments found — check class2d_star and min_segs')
            return
        safe_name = name.replace(' ', '_').replace('/', '_')
        mask      = self._dominant == (k - 1)
        keep_set  = {self._filaments[i] for i in np.where(mask)[0]}
        out_star  = os.path.join(self.out_dir, f'particles_{safe_name}.star')
        n_kept, n_total = filter_data_star(self.refine3d_star, keep_set, out_star)
        pct = 100 * n_kept / n_total if n_total else 0
        print(f'  Cluster {k} ({name}): '
              f'{int(mask.sum())} filaments, '
              f'{n_kept}/{n_total} particles ({pct:.1f}%) '
              f'→ {out_star}')

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument('--jaccard',      required=True,
                    help='cooccurrence_jaccard.npy from cooccurrence_2d_classes.py')
    ap.add_argument('--good_txt',     required=True,
                    help='good_classes.txt from triage_2d_classes.py')
    ap.add_argument('--mrcs',         required=True,
                    help='Class2D run_it025_classes.mrcs image stack')
    ap.add_argument('--class2d_star', required=True,
                    help='Class2D run_it025_data.star (for filament memberships)')
    ap.add_argument('--refine3d_star', required=True,
                    help='Refine3D run_data.star to filter')
    ap.add_argument('--out_dir',      required=True,
                    help='output directory for filtered star files')
    ap.add_argument('--min_segs',     type=int, default=20,
                    help='minimum segments per filament (default 20)')

    args = ap.parse_args()

    # Load inputs
    print('Loading Jaccard matrix …')
    J = np.load(args.jaccard)

    with open(args.good_txt) as fh:
        good_idx = [int(l.strip())
                    for l in fh
                    if l.strip() and not l.startswith('#')]
    print(f'  {len(good_idx)} good classes')

    print('Loading class images …')
    with mrcfile.open(args.mrcs, mode='r', permissive=True) as mrc:
        images = mrc.data.copy()
    print(f'  Stack shape: {images.shape}')

    assert J.shape == (len(good_idx), len(good_idx)), \
        f'Jaccard shape {J.shape} does not match {len(good_idx)} good classes'

    gui = PopulationSelectorGUI(
        J            = J,
        good_idx     = good_idx,
        images       = images,
        class2d_star = args.class2d_star,
        refine3d_star= args.refine3d_star,
        out_dir      = args.out_dir,
        min_segs     = args.min_segs,
    )
    gui.run()


if __name__ == '__main__':
    main()
