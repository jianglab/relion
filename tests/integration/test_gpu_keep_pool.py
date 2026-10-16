#!/usr/bin/env python3
"""
GPU memory chunk kept during the GPU reconstruction (RELION_GPU_KEEP_POOL).

Skipped without a CUDA GPU or a build that has the switch.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _have_cuda_gpu(relion_bin, marker="GPU_KEEP_POOL"):
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        if subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=30).returncode != 0:
            return False
    except Exception:
        return False
    exe = Path(relion_bin) / "relion_refine"
    if not exe.exists():
        return False
    out = subprocess.run(["strings", str(exe)], capture_output=True, text=True).stdout
    return marker in out


def _log_likelihood(model_star):
    for line in Path(model_star).read_text().splitlines():
        if line.startswith("_rlnLogLikelihood"):
            return float(line.split()[1])
    raise AssertionError("no _rlnLogLikelihood in " + str(model_star))


def test_keep_pool_for_reconstruct(test_data_dir, relion_bin, relion_runner):
    """RELION_GPU_KEEP_POOL=1 keeps the E-step's GPU memory chunk during the GPU reconstruction
    (it is given back only when the reconstruction does not fit next to it): the maps are
    still made on the GPU and the likelihood agrees with the run that frees the chunk."""
    if not _have_cuda_gpu(relion_bin, "GPU_KEEP_POOL"):
        pytest.skip("needs a CUDA GPU and a build with RELION_GPU_KEEP_POOL")
    from fixtures import generate_test_dataset

    paths = generate_test_dataset(test_data_dir / "data_kp", relion_bin, relion_runner,
                                  num_particles=40, size=32, angpix=1.0)

    def run(tag, extra_env):
        out = test_data_dir / tag
        out.mkdir()
        cmd = [str(relion_bin / "relion_refine"),
               "--i", paths["star_file"], "--o", str(out / "run"),
               "--ref", paths["initial_model"], "--K", "2", "--iter", "3",
               "--ini_high", "50", "--ctf", "--tau2_fudge", "4", "--particle_diameter", "25",
               "--flatten_solvent", "--zero_mask", "--oversampling", "1",
               "--healpix_order", "1", "--offset_range", "5", "--offset_step", "2",
               "--sym", "c1", "--norm", "--scale", "--j", "2", "--random_seed", "7",
               "--dont_align_classes", "--gpu", "0"]
        env = dict(os.environ)
        env.update(extra_env)
        r = relion_runner(cmd, timeout=900, env=env)
        assert r.returncode == 0, f"{tag} failed:\n{r.stderr[-3000:]}"
        return r, out

    r_plain, d_plain = run("kp_plain", {})
    r_kp, d_kp = run("kp_on", {"RELION_GPU_KEEP_POOL": "1"})
    assert "Reconstructing the maps on the GPU" in r_plain.stdout
    assert "Reconstructing the maps on the GPU" in r_kp.stdout
    assert "on the CPU" not in r_kp.stdout
    ll_plain = _log_likelihood(d_plain / "run_it003_model.star")
    ll_kp = _log_likelihood(d_kp / "run_it003_model.star")
    assert abs(ll_kp - ll_plain) <= 1e-5 * abs(ll_plain), (ll_kp, ll_plain)
