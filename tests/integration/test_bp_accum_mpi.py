#!/usr/bin/env python3
"""
The GPU back-projection per orientation (RELION_GPU_BP_ACCUM, on by default) adds the particles'
sums into one buffer per orientation and spreads them into the volumes at the end of the iteration.
relion_refine_mpi once read the volumes back without that last step, so a 3D run with MPI lost
nearly all of its back-projected data (maps stayed at the start, class resolution stuck).
The maps must agree with those of RELION_GPU_BP_ACCUM=0 up to rounding. Skipped when no GPU is visible.
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

from fixtures import generate_test_dataset, read_mrc


def _gpu_available():
    return shutil.which("nvidia-smi") is not None and subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


def _run(relion_bin, paths, out, env_extra):
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["mpirun", "--oversubscribe", "-n", "3", str(relion_bin / "relion_refine_mpi"),
           "--i", str(paths["star_file"]), "--o", str(out / "run"), "--ref", str(paths["initial_model"]),
           "--K", "1", "--iter", "2", "--ini_high", "30", "--ctf", "--tau2_fudge", "4", "--particle_diameter", "25",
           "--flatten_solvent", "--zero_mask", "--oversampling", "1", "--healpix_order", "1",
           "--offset_range", "5", "--offset_step", "2", "--sym", "c1", "--norm", "--scale", "--j", "2", "--gpu", "0",
           "--dont_combine_weights_via_disc", "--random_seed", "1", "--keep_all_iterations", "--pool", "5"]
    env = dict(os.environ)
    env.pop("RELION_GPU_BP_ACCUM", None)
    env.update(env_extra)
    res = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=900)
    assert res.returncode == 0, res.stderr[-3000:]
    return np.asarray(read_mrc(str(out / "run_it002_class001.mrc")), dtype=np.float64)


@pytest.mark.integration
def test_mpi_run_matches_run_without_accumulation(test_data_dir, relion_bin, relion_runner):
    if not _gpu_available():
        pytest.skip("no GPU visible")
    paths = generate_test_dataset(str(test_data_dir / "data"), relion_bin, relion_runner,
                                  num_particles=60, size=32, angpix=1.0)
    ref = _run(relion_bin, paths, test_data_dir / "off", {"RELION_GPU_BP_ACCUM": "0"})
    got = _run(relion_bin, paths, test_data_dir / "on", {})
    assert np.corrcoef(ref.ravel(), got.ravel())[0, 1] > 0.99
    assert abs(got.std() / ref.std() - 1) < 0.05
