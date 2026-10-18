#!/usr/bin/env python3
"""
The radix selection of the significance threshold as one cooperative launch (RELION_GPU_SELECT_FUSED,
on by default, =0 for the separate launches): a 3D classification with enough orientations that the
radix selection runs (more than 65536 weights per particle) exits cleanly and agrees with the run that
has the switch off: same log-likelihood to rounding, same class assignment of the particles.

Skipped without a CUDA GPU or a build that has the switch.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


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


def _class_numbers(data_star):
    cols, classes, in_loop = [], [], False
    for line in Path(data_star).read_text().splitlines():
        s = line.strip()
        if s.startswith("data_"):
            in_loop = False
            cols = []
            continue
        if s.startswith("loop_"):
            in_loop = True
            continue
        if in_loop and s.startswith("_"):
            cols.append(s.split()[0])
            continue
        if in_loop and s and not s.startswith("#") and "_rlnClassNumber" in cols:
            classes.append(int(s.split()[cols.index("_rlnClassNumber")]))
    return classes


def test_fused_selection_default_and_off_agree(test_data_dir, relion_bin, relion_runner):
    if not _have_cuda_gpu(relion_bin, "RELION_GPU_SELECT_FUSED"):
        pytest.skip("needs a CUDA GPU and a build with RELION_GPU_SELECT_FUSED")
    from fixtures import generate_test_dataset

    paths = generate_test_dataset(test_data_dir / "data_sf", relion_bin, relion_runner,
                                  num_particles=60, size=32, angpix=1.0)

    def run(tag, extra_env):
        out = test_data_dir / tag
        out.mkdir()
        cmd = [str(relion_bin / "relion_refine"),
               "--i", paths["star_file"], "--o", str(out / "run"),
               "--ref", paths["initial_model"], "--K", "2", "--iter", "2",
               "--ini_high", "8", "--ctf", "--tau2_fudge", "4", "--particle_diameter", "25",
               "--flatten_solvent", "--zero_mask", "--oversampling", "1",
               "--healpix_order", "3", "--offset_range", "6", "--offset_step", "2",
               "--sym", "c1", "--norm", "--scale", "--j", "4", "--random_seed", "7",
               "--dont_align_classes", "--gpu", "0"]
        env = dict(os.environ)
        env.pop("RELION_GPU_SELECT_FUSED", None)
        env.update(extra_env)
        r = relion_runner(cmd, timeout=1800, env=env)
        assert r.returncode == 0, f"{tag} failed (exit {r.returncode}):\n{r.stderr[-3000:]}"
        assert "double free" not in r.stderr and "corruption" not in r.stderr
        return r, out

    r_def, d_def = run("sf_default", {})
    r_off, d_off = run("sf_off", {"RELION_GPU_SELECT_FUSED": "0"})
    assert "none is switched off" in r_def.stdout
    assert "RELION_GPU_SELECT_FUSED" in r_off.stdout          # named in the "switched off:" list
    ll_def = _log_likelihood(d_def / "run_it002_model.star")
    ll_off = _log_likelihood(d_off / "run_it002_model.star")
    assert abs(ll_def - ll_off) <= 1e-3 * abs(ll_off), (ll_def, ll_off)
    c_def = _class_numbers(d_def / "run_it002_data.star")
    c_off = _class_numbers(d_off / "run_it002_data.star")
    assert len(c_def) == len(c_off) > 0
    same = sum(a == b for a, b in zip(c_def, c_off)) / len(c_def)
    assert same >= 0.95, same
