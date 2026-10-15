#!/usr/bin/env python3
"""
Integration test for RELION_GPU_BP_PACK (packed atomic additions in the GPU backprojection; on by
default, =0 turns it off).

The packed path adds the same terms to the model as the default path, only in a different order,
so a refinement with it must agree with the default one up to the noise of atomic additions. When
the path cannot be used for a launch (planar model layout, RELION_GPU_BP_TILE set) the program must fall back, say so once
on stderr, and still give the same result. Skipped when no GPU is visible.
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fixtures import generate_test_dataset, read_mrc


def _gpu_available(relion_bin):
    if shutil.which("nvidia-smi") is None:
        return False
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


def _run(relion_bin, paths, out, env_extra, extra_args=()):
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(relion_bin / "relion_refine"),
        "--i", str(paths["star_file"]),
        "--o", str(out / "run"),
        "--ref", str(paths["initial_model"]),
        "--K", "1", "--iter", "2", "--ini_high", "30",
        "--ctf", "--tau2_fudge", "4", "--particle_diameter", "25",
        "--flatten_solvent", "--zero_mask", "--oversampling", "1",
        "--healpix_order", "1", "--offset_range", "5", "--offset_step", "2",
        "--sym", "c1", "--norm", "--scale", "--j", "2", "--gpu", "0",
        "--dont_combine_weights_via_disc", "--random_seed", "1", "--keep_all_iterations",
        "--pool", "5",
    ] + list(extra_args)
    env = dict(os.environ)
    for k in ("RELION_GPU_BP_PACK", "RELION_GPU_BP_LAYOUT", "RELION_GPU_BP_TILE"):
        env.pop(k, None)
    env.update(env_extra)
    res = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=900)
    assert res.returncode == 0, f"relion_refine failed:\n{res.stderr[-3000:]}"
    maps = sorted(out.glob("run_it002_class001.mrc"))
    assert maps, "no output map"
    notes = [l for l in res.stderr.splitlines() if "RELION_GPU_BP_PACK" in l]
    return np.asarray(read_mrc(str(maps[0])), dtype=np.float64), notes


def _close(a, b):
    c = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    rel = np.sqrt(((a - b) ** 2).mean()) / np.sqrt((a ** 2).mean())
    return c, rel


@pytest.mark.integration
class TestBpPack:

    @pytest.fixture
    def setup(self, test_data_dir, relion_bin, relion_runner):
        if not _gpu_available(relion_bin):
            pytest.skip("no GPU visible")
        paths = generate_test_dataset(str(test_data_dir / "data"), relion_bin, relion_runner,
                                      num_particles=60, size=32, angpix=1.0)
        ref, notes = _run(relion_bin, paths, test_data_dir / "off", {"RELION_GPU_BP_PACK": "0"})
        assert notes == [], "RELION_GPU_BP_PACK=0 must not report anything"
        # the noise floor: the unpacked path run again
        ref2, _ = _run(relion_bin, paths, test_data_dir / "off2", {"RELION_GPU_BP_PACK": "0"})
        return paths, ref, _close(ref, ref2)

    def _check(self, setup, test_data_dir, relion_bin, tag, env, expect, extra_args=()):
        paths, ref, floor = setup
        got, notes = _run(relion_bin, paths, test_data_dir / tag, env, extra_args)
        assert notes, f"{tag}: no RELION_GPU_BP_PACK line on stderr"
        assert any(expect in n for n in notes), f"{tag}: expected '{expect}' in {notes}"
        corr, rel = _close(ref, got)
        # atomic additions are never bit-reproducible: allow a few times the run-to-run noise
        assert corr > 0.99999 and rel < max(10 * floor[1], 1e-4), (
            f"{tag}: corr {corr}, rel {rel}, noise floor {floor}")

    def test_pack_on_by_default(self, setup, test_data_dir, relion_bin):
        self._check(setup, test_data_dir, relion_bin, "pack", {}, "on (default")

    def test_fallback_planar_layout(self, setup, test_data_dir, relion_bin):
        self._check(setup, test_data_dir, relion_bin, "planar", {"RELION_GPU_BP_LAYOUT": "planar"}, "not used: planar model layout")

    def test_fallback_tile_mode(self, setup, test_data_dir, relion_bin):
        self._check(setup, test_data_dir, relion_bin, "tile", {"RELION_GPU_BP_TILE": "1"}, "not used: RELION_GPU_BP_TILE")
