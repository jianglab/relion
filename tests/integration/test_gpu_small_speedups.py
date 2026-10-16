#!/usr/bin/env python3
"""
Small GPU speed-ups that are on by default (pooled pinned copy buffers, kernel variants of the
weights and weighted sums): a run with all of them on exits cleanly (the pinned pool once crashed
the process at exit) and agrees with the run that has every one of them switched off with VAR=0.

Skipped without a CUDA GPU or a build that has the switches.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

# the default-on switches of this feature set (keep in step with gpuswitch::defaultOnNames)
SWITCHES = [
    "RELION_GPU_PINNED_POOL",
    "RELION_GPU_PINNED_BIG_COPY",
    "RELION_GPU_FUSED_WEIGHTS",
]


def _have_cuda_gpu(relion_bin, marker):
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


def test_small_speedups_default_and_off_agree(test_data_dir, relion_bin, relion_runner):
    if not _have_cuda_gpu(relion_bin, SWITCHES[0]):
        pytest.skip("needs a CUDA GPU and a build with the small GPU speed-ups")
    from fixtures import generate_test_dataset

    paths = generate_test_dataset(test_data_dir / "data_ss", relion_bin, relion_runner,
                                  num_particles=60, size=32, angpix=1.0)

    def run(tag, extra_env):
        out = test_data_dir / tag
        out.mkdir()
        cmd = [str(relion_bin / "relion_refine"),
               "--i", paths["star_file"], "--o", str(out / "run"),
               "--ref", paths["initial_model"], "--K", "2", "--iter", "3",
               "--ini_high", "50", "--ctf", "--tau2_fudge", "4", "--particle_diameter", "25",
               "--flatten_solvent", "--zero_mask", "--oversampling", "1",
               "--healpix_order", "1", "--offset_range", "5", "--offset_step", "2",
               "--sym", "c1", "--norm", "--scale", "--j", "4", "--random_seed", "7",
               "--dont_align_classes", "--gpu", "0"]
        env = dict(os.environ)
        for name in SWITCHES:
            env.pop(name, None)
        env.update(extra_env)
        r = relion_runner(cmd, timeout=900, env=env)
        assert r.returncode == 0, f"{tag} failed (exit {r.returncode}):\n{r.stderr[-3000:]}"
        assert "double free" not in r.stderr and "corruption" not in r.stderr
        return r, out

    r_def, d_def = run("ss_default", {})
    r_off, d_off = run("ss_off", {name: "0" for name in SWITCHES})
    assert "none is switched off" in r_def.stdout
    for name in SWITCHES:
        assert name in r_off.stdout          # named in the "switched off:" list
    ll_def = _log_likelihood(d_def / "run_it003_model.star")
    ll_off = _log_likelihood(d_off / "run_it003_model.star")
    assert abs(ll_def - ll_off) <= 1e-3 * abs(ll_off), (ll_def, ll_off)
