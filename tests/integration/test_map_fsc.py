#!/usr/bin/env python3
"""scripts/map_fsc.py: FSC of two maps with different box size, pixel size, position and orientation."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "map_fsc.py"


def blobs():
    pts = [(60 * np.cos(np.deg2rad(29.4 * k)), 60 * np.sin(np.deg2rad(29.4 * k)), 21.8 * k) for k in range(-4, 5)]
    pts += [(25.0, -10.0, 8.0)]                                    # makes the object asymmetric
    return np.array(pts)


def render(n, apix, angle=0.0, shift=(0.0, 0.0, 0.0), sigma=6.0, flip=False):
    """Object seen on an n^3 grid; the object is rotated by `angle` about Z and shifted (x,y,z in A)."""
    p = blobs().copy()
    if flip:
        p[:, 2] *= -1
    t = np.deg2rad(angle)
    p = np.stack([np.cos(t) * p[:, 0] - np.sin(t) * p[:, 1], np.sin(t) * p[:, 0] + np.cos(t) * p[:, 1], p[:, 2]], 1)
    p += np.array(shift)
    z, y, x = np.mgrid[0:n, 0:n, 0:n].astype(np.float32)
    x, y, z = (x - n // 2) * apix, (y - n // 2) * apix, (z - n // 2) * apix
    v = np.zeros((n, n, n), np.float32)
    for px, py, pz in p:
        v += np.exp(-((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) / (2 * sigma ** 2))
    return v


def write(fn, vol, apix):
    with mrcfile.new(fn, overwrite=True) as m:
        m.set_data(vol)
        m.voxel_size = apix


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True, timeout=1200)


def resolution_at(text, level):
    for line in text.splitlines():
        if line.startswith(f"FSC = {level}:"):
            word = line.split(":")[1].split()[0]
            return 0.0 if word == "not" else float(word)      # 0 = never crossed
    raise AssertionError(text)


@pytest.mark.integration
@pytest.mark.parametrize("flip", [False, True], ids=["same_hand", "flipped_hand"])
def test_fsc_between_maps_of_different_size_pixel_size_and_pose(tmp_path, flip):
    write(tmp_path / "a.mrc", render(112, 2.5), 2.5)                                   # 280 A box
    write(tmp_path / "b.mrc", render(90, 3.0, angle=41.0, shift=(6.0, -4.0, 9.0), flip=flip), 3.0)  # 270 A box
    r = run(tmp_path / "a.mrc", tmp_path / "b.mrc", "--radius", 90, "--length", 150,
            "--search_res", 15, "--out", tmp_path / "fsc")
    assert r.returncode == 0, r.stdout + r.stderr
    assert resolution_at(r.stdout, 0.143) < 14.0, r.stdout
    assert (tmp_path / "fsc.txt").exists()


@pytest.mark.integration
def test_unaligned_maps_give_a_poor_fsc_when_the_search_is_off(tmp_path):
    write(tmp_path / "a.mrc", render(112, 2.5), 2.5)
    write(tmp_path / "b.mrc", render(90, 3.0, angle=41.0), 3.0)
    r = run(tmp_path / "a.mrc", tmp_path / "b.mrc", "--radius", 90, "--length", 150, "--no_search",
            "--out", tmp_path / "fsc")
    assert r.returncode == 0, r.stdout + r.stderr
    assert resolution_at(r.stdout, 0.5) > 25.0
