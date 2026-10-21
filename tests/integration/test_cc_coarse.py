#!/usr/bin/env python3
"""
Integration test for RELION_GPU_CC_COARSE (coarse pass of the cross-correlation first iteration
with several translations per GPU block; on by default, =0 turns it off).

The kernel forms every sum in the same order as the older one, so a --firstiter_cc Class3D
iteration must give the same orientations, offsets and classes, and maps that agree up to the noise
of atomic additions. Two offset ranges cover both block widths: 5 translations (8 per block) and
49 translations (16 per block, the last block partly filled). Skipped when no GPU is visible.
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

from fixtures import generate_test_dataset, read_mrc, read_star_particles

COLUMNS = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi", "rlnOriginXAngst", "rlnOriginYAngst",
           "rlnClassNumber"]


def _gpu_available(relion_bin):
    if shutil.which("nvidia-smi") is None:
        return False
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


def _run(relion_bin, paths, out, offset_range, offset_step, env_extra):
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(relion_bin / "relion_refine"),
        "--i", str(paths["star_file"]), "--o", str(out / "run"),
        "--ref", str(paths["initial_model"]),
        "--K", "2", "--iter", "1", "--ini_high", "30", "--firstiter_cc",
        "--offset_range", str(offset_range), "--offset_step", str(offset_step), "--healpix_order", "2",
        "--ctf", "--tau2_fudge", "4", "--particle_diameter", "25",
        "--flatten_solvent", "--zero_mask", "--sym", "c1", "--norm", "--scale",
        "--j", "2", "--gpu", "0", "--dont_combine_weights_via_disc",
        "--random_seed", "1", "--keep_all_iterations", "--pool", "5",
    ]
    env = dict(os.environ)
    env.pop("RELION_GPU_CC_COARSE", None)
    env.update(env_extra)
    res = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=900)
    assert res.returncode == 0, f"relion_refine failed:\n{res.stderr[-3000:]}"
    maps = [np.asarray(read_mrc(str(out / f"run_it001_class00{k}.mrc")), dtype=np.float64)
            for k in (1, 2)]
    return maps, read_star_particles(str(out / "run_it001_data.star"), COLUMNS)


def _close(a, b):
    c = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    rel = np.sqrt(((a - b) ** 2).mean()) / np.sqrt((a ** 2).mean())
    return c, rel


@pytest.mark.integration
class TestCcCoarse:

    @pytest.mark.parametrize("offset_range,offset_step", [(2, 2), (4, 1)])
    def test_matches_older_kernel(self, test_data_dir, relion_bin, relion_runner, offset_range, offset_step):
        if not _gpu_available(relion_bin):
            pytest.skip("no GPU visible")
        paths = generate_test_dataset(str(test_data_dir / "data"), relion_bin, relion_runner,
                                      num_particles=60, size=32, angpix=1.0)
        old, star_old = _run(relion_bin, paths, test_data_dir / "old", offset_range, offset_step,
                             {"RELION_GPU_CC_COARSE": "0"})
        old2, _ = _run(relion_bin, paths, test_data_dir / "old2", offset_range, offset_step,
                       {"RELION_GPU_CC_COARSE": "0"})
        new, star_new = _run(relion_bin, paths, test_data_dir / "new", offset_range, offset_step, {})
        for col in COLUMNS:
            assert np.array_equal(star_old[col], star_new[col]), f"{col} differs"
        for k in range(2):
            floor = _close(old[k], old2[k])
            corr, rel = _close(old[k], new[k])
            assert corr > 0.99999 and rel < max(10 * floor[1], 1e-4), (
                f"class {k + 1}: corr {corr}, rel {rel}, noise floor {floor}")
