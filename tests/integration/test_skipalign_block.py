#!/usr/bin/env python3
"""
Integration test for RELION_GPU_SKIPALIGN_BLOCK (one-block significance pass of --skip_align on the
GPU; on by default, =0 turns it off).

With --skip_align every particle has one orientation and a few weights, and the one-block pass
picks the same significant weights as the general path, so a Class3D --skip_align run with it must
agree with a run without it up to the noise of atomic additions. Skipped when no GPU is visible.
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

from fixtures import generate_test_dataset, read_mrc


def _gpu_available(relion_bin):
    if shutil.which("nvidia-smi") is None:
        return False
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


def _run(relion_bin, paths, out, env_extra):
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(relion_bin / "relion_refine"),
        "--i", str(paths["star_file"]), "--o", str(out / "run"),
        "--ref", str(paths["initial_model"]),
        "--K", "2", "--iter", "2", "--ini_high", "30", "--skip_align",
        "--ctf", "--tau2_fudge", "4", "--particle_diameter", "25",
        "--flatten_solvent", "--zero_mask", "--sym", "c1", "--norm", "--scale",
        "--j", "2", "--gpu", "0", "--dont_combine_weights_via_disc",
        "--random_seed", "1", "--keep_all_iterations", "--pool", "5",
    ]
    env = dict(os.environ)
    env.pop("RELION_GPU_SKIPALIGN_BLOCK", None)
    env.update(env_extra)
    res = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=900)
    assert res.returncode == 0, f"relion_refine failed:\n{res.stderr[-3000:]}"
    maps = [np.asarray(read_mrc(str(out / f"run_it002_class00{k}.mrc")), dtype=np.float64)
            for k in (1, 2)]
    return maps, res.stdout


def _close(a, b):
    c = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    rel = np.sqrt(((a - b) ** 2).mean()) / np.sqrt((a ** 2).mean())
    return c, rel


@pytest.mark.integration
class TestSkipAlignBlock:

    def test_block_matches_general_path(self, test_data_dir, relion_bin, relion_runner):
        if not _gpu_available(relion_bin):
            pytest.skip("no GPU visible")
        paths = generate_test_dataset(str(test_data_dir / "data"), relion_bin, relion_runner,
                                      num_particles=60, size=32, angpix=1.0)
        off, out_off = _run(relion_bin, paths, test_data_dir / "off", {"RELION_GPU_SKIPALIGN_BLOCK": "0"})
        off2, _ = _run(relion_bin, paths, test_data_dir / "off2", {"RELION_GPU_SKIPALIGN_BLOCK": "0"})
        on, out_on = _run(relion_bin, paths, test_data_dir / "on", {})
        assert "switched off:" not in out_on or "RELION_GPU_SKIPALIGN_BLOCK" not in out_on.split("switched off:")[1].split("(VAR")[0]
        assert "switched off: " in out_off and "RELION_GPU_SKIPALIGN_BLOCK" in out_off.split("switched off:")[1].split("(VAR")[0]
        for k in range(2):
            floor = _close(off[k], off2[k])
            corr, rel = _close(off[k], on[k])
            assert corr > 0.99999 and rel < max(10 * floor[1], 1e-4), (
                f"class {k + 1}: corr {corr}, rel {rel}, noise floor {floor}")
