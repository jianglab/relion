#!/usr/bin/env python3
"""
Integration test for RELION_GPU_COARSE_CACHE (the coarse pass of a 3D refinement scores the particles
against cached reference projections; on by default, =0 turns it off, =check runs both ways and reports
how far they differ).

The cache evaluates the same sum of squared differences in another order, so a refinement with it must
agree with the default (cross-term kernel) one up to float rounding; =check must report a small relative
difference; and a budget too small for the cache (RELION_GPU_COARSE_CACHE_MB=1) must fall back, say so,
and still give the same result. Skipped when no GPU is visible.
"""

import os
import re
import shutil
import subprocess

import numpy as np
import pytest

from fixtures import generate_test_dataset, gpu_double_precision, read_mrc


def _gpu_available(relion_bin):
    if shutil.which("nvidia-smi") is None:
        return False
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


def _run(relion_bin, paths, out, env_extra, extra=()):
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
    ] + list(extra)
    env = dict(os.environ)
    for k in ("RELION_GPU_COARSE_CACHE", "RELION_GPU_COARSE_CACHE_MB", "RELION_GPU_COARSE_CACHE_TM"):
        env.pop(k, None)
    env.update(env_extra)
    res = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=900)
    assert res.returncode == 0, f"relion_refine failed:\n{res.stderr[-3000:]}"
    maps = sorted(out.glob("run_it002_class001.mrc"))
    assert maps, "no output map"
    return np.asarray(read_mrc(str(maps[0])), dtype=np.float64), res.stdout, res.stderr


def _close(a, b):
    c = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    rel = np.sqrt(((a - b) ** 2).mean()) / np.sqrt((a ** 2).mean())
    return c, rel


@pytest.mark.integration
class TestCoarseCache:

    @pytest.fixture
    def setup(self, test_data_dir, relion_bin, relion_runner):
        if not _gpu_available(relion_bin):
            pytest.skip("no GPU visible")
        if gpu_double_precision(relion_bin):
            pytest.skip("the projection cache and cross-term kernel are single-precision only")
        paths = generate_test_dataset(str(test_data_dir / "data"), relion_bin, relion_runner,
                                      num_particles=60, size=32, angpix=1.0)
        ref, out, _ = _run(relion_bin, paths, test_data_dir / "off", {"RELION_GPU_COARSE_CACHE": "0"})
        assert "projection cache" not in out, "RELION_GPU_COARSE_CACHE=0 must not build the cache"
        return paths, ref

    def test_on_by_default_and_same_result(self, setup, test_data_dir, relion_bin):
        paths, ref = setup
        got, out, _ = _run(relion_bin, paths, test_data_dir / "on", {})
        assert "projection cache of" in out, "the cache was not used by default"
        corr, rel = _close(ref, got)
        assert corr > 0.9999 and rel < 1e-3, f"corr {corr}, rel {rel}"

    def test_check_reports_small_difference(self, setup, test_data_dir, relion_bin):
        paths, ref = setup
        got, out, err = _run(relion_bin, paths, test_data_dir / "check", {"RELION_GPU_COARSE_CACHE": "check"})
        lines = [l for l in err.splitlines() if "COARSE_CACHE_CHECK" in l]
        assert lines, "no COARSE_CACHE_CHECK line"
        for l in lines:
            m = re.search(r"max relative ([0-9.eE+-]+)", l)
            assert m and float(m.group(1)) < 1e-4, l

    def test_fallback_when_budget_too_small(self, setup, test_data_dir, relion_bin):
        paths, _ = setup
        big = ["--healpix_order", "3", "--ini_high", "8"]
        ref, _, _ = _run(relion_bin, paths, test_data_dir / "off_big", {"RELION_GPU_COARSE_CACHE": "0"}, extra=big)
        got, out, _ = _run(relion_bin, paths, test_data_dir / "small", {"RELION_GPU_COARSE_CACHE_MB": "1"}, extra=big)
        assert "using the cross-term kernel" in out
        corr, rel = _close(ref, got)
        assert corr > 0.99999 and rel < 1e-4, f"corr {corr}, rel {rel}"
