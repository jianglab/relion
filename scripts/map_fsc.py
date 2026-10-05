#!/usr/bin/env python3
"""
FSC between two maps that differ in box size, pixel size, position and in-plane
orientation, for example a RELION reconstruction and a map downloaded from EMDB.

Both maps are resampled onto one common grid (the finer of the two pixel sizes,
a box covering the region both maps share). The second map is first aligned to
the first (see --helical), then the Fourier shell correlation is computed inside a
soft cylindrical mask.

--helical (default): both maps have their helical axis along Z. The search covers
the rotation about Z, a shift along Z, an XY shift and, with --hand auto, the
handedness (a Z flip of the second map). One rotation + one Z shift is
ambiguous by the helical symmetry; any of the equivalent answers gives the same FSC.
--no_search: skip the alignment (the maps are already aligned).

Usage:
  map_fsc.py mine.mrc reference.mrc --radius 170 --out fsc
  (pixel sizes come from the MRC headers; override with --apix_a / --apix_b)

Writes <out>.txt (resolution, FSC) and optionally <out>_aligned.mrc (the second
map on the common grid) and prints the 0.5 and 0.143 resolutions.
"""

import argparse
import sys

import mrcfile
import numpy as np
from scipy import ndimage


def read_map(fn, apix_override):
    with mrcfile.open(fn, permissive=True) as m:
        data = np.array(m.data, dtype=np.float32)
        apix = float(m.voxel_size.x) if apix_override is None else apix_override
    if data.ndim != 3:
        raise SystemExit(f"{fn}: not a 3D map")
    if apix <= 0:
        raise SystemExit(f"{fn}: no pixel size in the header; pass --apix_a / --apix_b")
    return data, apix


def lowpass(vol, apix, res):
    f = np.fft.rfftn(vol)
    kz = np.fft.fftfreq(vol.shape[0], apix)[:, None, None]
    ky = np.fft.fftfreq(vol.shape[1], apix)[None, :, None]
    kx = np.fft.rfftfreq(vol.shape[2], apix)[None, None, :]
    k = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    f *= np.exp(-0.5 * (k * res / 2.0) ** 2) if res else 1.0
    return np.fft.irfftn(f, s=vol.shape).astype(np.float32)


def resample(vol, apix, new_apix, shape, shift_a=(0, 0, 0), angle=0.0, flip=False):
    """Sample vol on a grid of `shape` (z,y,x) voxels of size new_apix, centred on the
    box centre of vol. Rotation (degrees, about Z) and a shift (Angstrom, z,y,x) are
    applied to the sampled object. The map is low-passed first when it is coarsened."""
    if new_apix > apix * 1.001:
        vol = lowpass(vol, apix, 2 * new_apix)
    nz, ny, nx = shape
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32)
    z = (z - nz // 2) * new_apix - shift_a[0]
    y = (y - ny // 2) * new_apix - shift_a[1]
    x = (x - nx // 2) * new_apix - shift_a[2]
    t = np.deg2rad(angle)
    c, s = np.cos(t), np.sin(t)
    xr, yr = c * x + s * y, -s * x + c * y
    if int(flip) & 2:        # opposite filament polarity: a 180 degree turn about X
        y, z = -y, -z
    if int(flip) & 1:        # opposite hand: a mirror in Z
        z = -z
    cz, cy, cx = (np.array(vol.shape) // 2)
    coords = np.array([z / apix + cz, yr / apix + cy, xr / apix + cx])
    return ndimage.map_coordinates(vol, coords, order=3 if new_apix <= apix * 1.5 else 1,
                                   mode="constant", cval=0.0)


def coarsen(vol, apix, new_apix):
    """The whole map on a coarser grid (done once, so that the many resamplings of the
    alignment search are cheap)."""
    if new_apix <= apix * 1.001:
        return vol, apix
    shape = tuple(int(np.ceil(n * apix / new_apix)) // 2 * 2 + 2 for n in vol.shape)
    return resample(vol, apix, new_apix, shape), new_apix


def cylinder_mask(shape, apix, radius, length, soft=10.0):
    nz, ny, nx = shape
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32)
    r = np.sqrt(((x - nx // 2) * apix) ** 2 + ((y - ny // 2) * apix) ** 2)
    zz = np.abs((z - nz // 2) * apix)

    def edge(d, limit):
        return np.clip(0.5 + 0.5 * np.cos(np.pi * np.clip((d - limit) / soft, 0, 1)), 0, 1)

    return edge(r, radius) * edge(zz, length / 2.0)


def correlation(a, b, mask):
    a = a * mask
    b = b * mask
    n = mask.sum()
    ma, mb = a.sum() / n, b.sum() / n
    a = (a - ma * mask)
    b = (b - mb * mask)
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def best_shift(a, b, max_shift_px):
    """Peak of the cross-correlation of a and b within +-max_shift_px (z,y,x)."""
    fa, fb = np.fft.rfftn(a), np.fft.rfftn(b)
    cc = np.fft.irfftn(fa * np.conj(fb), s=a.shape)
    cc = np.fft.fftshift(cc)
    centre = np.array(a.shape) // 2
    sl = tuple(slice(max(0, c - m), c + m + 1) for c, m in zip(centre, max_shift_px))
    sub = cc[sl]
    idx = np.unravel_index(np.argmax(sub), sub.shape)
    off = np.array([i + s.start - c for i, s, c in zip(idx, sl, centre)])
    norm = np.sqrt((a * a).sum() * (b * b).sum() + 1e-30)
    return off, float(sub[idx] / norm)


def align(a, apix_a, b, apix_b, radius, hands, search_res, max_shift):
    hands_all = list(hands)
    """Return (angle, hand_flip, shift_zyx in A) that best overlays b on a."""
    a0, a0_apix, b0, b0_apix = a, apix_a, b, apix_b
    p = max(search_res / 2.5, a0_apix, b0_apix)
    a, apix_a = coarsen(a, apix_a, p)
    b, apix_b = coarsen(b, apix_b, p)
    n_xy = int(2 * (radius + 3 * search_res) / p) // 2 * 2 + 2
    n_z = int(min(a.shape[0] * apix_a, b.shape[0] * apix_b) * 1.0 / p) // 2 * 2 + 2
    shape = (n_z, n_xy, n_xy)
    ga = resample(a, apix_a, p, shape)
    mask = cylinder_mask(shape, p, radius, n_z * p * 0.9, soft=2 * p)
    ga = lowpass(ga, p, search_res) * mask
    ms = tuple(int(max_shift / p) for _ in range(3))
    best = (-2.0, 0.0, False, (0, 0, 0))
    for flip in hands:
        for angle in np.arange(0, 360, 3.0):
            gb = resample(b, apix_b, p, shape, angle=angle, flip=flip)
            gb = lowpass(gb, p, search_res) * mask
            off, score = best_shift(ga, gb, ms)
            if score > best[0]:
                best = (score, angle, flip, tuple(off * p))
        print(f"  hand/polarity code {flip}: best score so far {best[0]:.3f}", file=sys.stderr)
    score, angle, flip, shift = best

    # The coarse search cannot see features finer than search_res, but a helix with a short
    # repeat (tau: 4.8 A) is registered along Z only by those. Rotation and Z shift are also
    # coupled through the twist (and a twist near 180 degrees repeats every half turn), so
    # rescan the rotation near the coarse answer and its opposite, with the Z shift
    # found by cross-correlation, at a resolution that resolves the repeat.
    pix = max(1.2, a0_apix, b0_apix)
    res = max(5.0, 2.2 * pix)
    a, apix_a = coarsen(a0, a0_apix, pix)
    b, apix_b = coarsen(b0, b0_apix, pix)
    n_xy = int(2 * (radius + 3 * res) / pix) // 2 * 2 + 2
    n_z = int(min(a.shape[0] * apix_a, b.shape[0] * apix_b) / pix) // 2 * 2 + 2
    shape = (n_z, n_xy, n_xy)
    mask = cylinder_mask(shape, pix, radius, n_z * pix * 0.9, soft=2 * pix)
    ga = lowpass(resample(a, apix_a, pix, shape), pix, res) * mask
    ms = tuple(int(max_shift / pix) for _ in range(3))
    top = (-2.0, None)
    for code in hands_all:
        for base in (angle, angle + 180.0):
            for da in np.arange(-6.0, 6.01, 1.0):
                gb = lowpass(resample(b, apix_b, pix, shape, angle=base + da, flip=code), pix, res) * mask
                off, sc = best_shift(ga, gb, ms)
                if sc > top[0]:
                    top = (sc, (base + da, code, tuple(off * pix)))
        print(f"  fine scan, code {code}: best {top[0]:.3f}", file=sys.stderr)
    angle, flip, shift = top[1]

    # refine with finer local searches: first at search_res/4, then at the native pixel size
    from scipy.optimize import minimize
    x0 = np.array([angle, *shift])
    for pix, res, xtol in ((search_res / 4.0, search_res, 0.2), (max(a0_apix, b0_apix), min(search_res, 8.0), 0.03)):
        pix = max(pix, a0_apix, b0_apix)
        a, apix_a = coarsen(a0, a0_apix, pix)
        b, apix_b = coarsen(b0, b0_apix, pix)
        n_xy = int(2 * (radius + 3 * res) / pix) // 2 * 2 + 2
        n_z = int(min(a.shape[0] * apix_a, b.shape[0] * apix_b) / pix) // 2 * 2 + 2
        shape = (n_z, n_xy, n_xy)
        mask = cylinder_mask(shape, pix, radius, n_z * pix * 0.9, soft=2 * pix)
        ga = lowpass(resample(a, apix_a, pix, shape), pix, res) * mask

        def cost(v):
            gb = lowpass(resample(b, apix_b, pix, shape, shift_a=v[1:], angle=v[0], flip=flip), pix, res)
            return -correlation(ga, gb, mask)

        r = minimize(cost, x0, method="Powell", options=dict(xtol=xtol, ftol=1e-5, maxiter=6))
        x0, score = r.x, -float(r.fun)
    return float(x0[0]), flip, tuple(x0[1:]), score


def fsc(a, b, apix):
    fa, fb = np.fft.rfftn(a), np.fft.rfftn(b)
    nz, ny, nx = a.shape
    kz = np.fft.fftfreq(nz, apix)[:, None, None]
    ky = np.fft.fftfreq(ny, apix)[None, :, None]
    kx = np.fft.rfftfreq(nx, apix)[None, None, :]
    k = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    dk = 1.0 / (max(ny, nx) * apix)
    shell = np.rint(k / dk).astype(int)
    nshell = int(0.5 / apix / dk) + 1
    num = np.bincount(shell.ravel(), (fa * np.conj(fb)).real.ravel(), minlength=nshell + 1)[:nshell]
    da = np.bincount(shell.ravel(), (np.abs(fa) ** 2).ravel(), minlength=nshell + 1)[:nshell]
    db = np.bincount(shell.ravel(), (np.abs(fb) ** 2).ravel(), minlength=nshell + 1)[:nshell]
    curve = num / np.sqrt(da * db + 1e-30)
    freq = np.arange(nshell) * dk
    return freq, curve


def crossing(freq, curve, level):
    for i in range(2, len(curve)):  # shell 1 holds only a few Fourier terms; its value is noise
        if curve[i] < level <= curve[i - 1]:
            f = freq[i - 1] + (curve[i - 1] - level) / (curve[i - 1] - curve[i]) * (freq[i] - freq[i - 1])
            return 1.0 / f
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_a")
    ap.add_argument("map_b")
    ap.add_argument("--apix_a", type=float)
    ap.add_argument("--apix_b", type=float)
    ap.add_argument("--radius", type=float, required=True, help="mask radius around the helical axis (A)")
    ap.add_argument("--length", type=float, help="mask length along Z (A); default: 90%% of the shorter map")
    ap.add_argument("--soft", type=float, default=10.0, help="mask edge width (A)")
    ap.add_argument("--hand", choices=["auto", "same", "flip"], default="auto")
    ap.add_argument("--search_res", type=float, default=12.0, help="resolution used for the alignment (A)")
    ap.add_argument("--max_shift", type=float, default=40.0, help="largest shift searched (A)")
    ap.add_argument("--no_search", action="store_true")
    ap.add_argument("--mask_from_b", action="store_true",
                    help="also restrict the FSC to the density of map B (thresholded, dilated, soft edge); "
                         "removes solvent noise that a plain cylinder lets in")
    ap.add_argument("--pose", help="skip the search: rotation_deg,dz,dy,dx(A),code")
    ap.add_argument("--out", default="map_fsc")
    ap.add_argument("--write_aligned", action="store_true")
    args = ap.parse_args()

    a, apix_a = read_map(args.map_a, args.apix_a)
    b, apix_b = read_map(args.map_b, args.apix_b)
    a = a - np.median(a)
    b = b - np.median(b)
    print(f"A: {a.shape[::-1]} px at {apix_a:.4f} A;  B: {b.shape[::-1]} px at {apix_b:.4f} A")

    angle, flip, shift = 0.0, int(args.hand == "flip"), (0.0, 0.0, 0.0)
    if args.pose:
        v = [float(t) for t in args.pose.split(",")]
        angle, shift, flip = v[0], tuple(v[1:4]), int(v[4])
    elif not args.no_search:
        hands = {"auto": [0, 1, 2, 3], "same": [0], "flip": [1]}[args.hand]
        angle, flip, shift, score = align(a, apix_a, b, apix_b, args.radius, hands, args.search_res, args.max_shift)
        print(f"alignment: rotation about Z {angle:.1f} deg, shift (z,y,x) = "
              f"({shift[0]:.1f}, {shift[1]:.1f}, {shift[2]:.1f}) A, hand/polarity code {flip} (1 = mirrored, 2 = filament turned over, 3 = both), correlation {score:.3f}")

    p = min(apix_a, apix_b)
    length = args.length or 0.9 * min(a.shape[0] * apix_a, b.shape[0] * apix_b)
    n_xy = int(2 * (args.radius + 2 * args.soft) / p) // 2 * 2 + 2
    n_z = int((length + 2 * args.soft) / p) // 2 * 2 + 2
    shape = (n_z, n_xy, n_xy)
    ga = resample(a, apix_a, p, shape)
    gb = resample(b, apix_b, p, shape, shift_a=shift, angle=angle, flip=flip)
    mask = cylinder_mask(shape, p, args.radius, length, soft=args.soft)
    if args.mask_from_b:
        den = lowpass(gb, p, 15.0)
        thr = 0.25 * np.percentile(np.abs(den[mask > 0.5]), 99.5)
        core = (np.abs(den) > thr) & (mask > 0.5)
        grow = int(round(8.0 / p))
        core = ndimage.binary_dilation(core, iterations=max(grow, 1))
        soft = ndimage.gaussian_filter(core.astype(np.float32), 4.0 / p)
        mask = mask * soft
        print(f"mask from B: {100.0 * (mask > 0.5).sum() / (shape[0] * shape[1] * shape[2]):.1f}% of the box")
    freq, curve = fsc(ga * mask, gb * mask, p)

    with open(args.out + ".txt", "w") as f:
        f.write("# resolution_A  fsc\n")
        for fr, c in zip(freq[1:], curve[1:]):
            f.write(f"{1.0 / fr:.3f} {c:.4f}\n")
    for level in (0.5, 0.143):
        r = crossing(freq, curve, level)
        print(f"FSC = {level}: " + (f"{r:.2f} A" if r else f"not crossed (above {level} up to {1.0 / freq[-1]:.2f} A)"))
    if args.write_aligned:
        with mrcfile.new(args.out + "_aligned.mrc", overwrite=True) as m:
            m.set_data((gb).astype(np.float32))
            m.voxel_size = p


if __name__ == "__main__":
    main()
