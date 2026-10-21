#!/usr/bin/env python3
"""
Integration test for --coarse_early (default from RELION_COARSE_EARLY; the Class3D GUI emits it by
default): a Class3D run with alignment searches one healpix order coarser than asked for, then the
asked-for order in the last --coarse_early_final_iters iterations. Checks the order each iteration
recorded, serial (switched on by the environment) and with 3 MPI ranks (on the command line; every
rank must make the same choice), that --dont_coarse_early wins over the environment, and that nothing
changes by default. Skipped when no GPU is
visible (the runs use --gpu, as real Class3D jobs do).
"""

import os
import shutil
import subprocess

import pytest

from fixtures import generate_test_dataset


def _gpu_available():
    if shutil.which("nvidia-smi") is None:
        return False
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


def _orders(out, n_iter):
    orders = []
    for it in range(1, n_iter + 1):
        with open(out / f"run_it{it:03d}_sampling.star") as f:
            for line in f:
                if line.startswith("_rlnHealpixOrder "):
                    orders.append(int(line.split()[1]))
    return orders


def _run(relion_bin, paths, out, env_extra, mpi, extra=()):
    out.mkdir(parents=True, exist_ok=True)
    prog = "relion_refine_mpi" if mpi else "relion_refine"
    cmd = (["mpirun", "--oversubscribe", "-n", "3"] if mpi else []) + [
        str(relion_bin / prog),
        "--i", str(paths["star_file"]), "--o", str(out / "run"),
        "--ref", str(paths["initial_model"]),
        "--K", "2", "--iter", "4", "--ini_high", "30", "--healpix_order", "2",
        "--offset_range", "2", "--offset_step", "1",
        "--ctf", "--tau2_fudge", "4", "--particle_diameter", "25",
        "--flatten_solvent", "--zero_mask", "--sym", "c1", "--norm", "--scale",
        "--j", "1", "--gpu", "0", "--dont_combine_weights_via_disc",
        "--random_seed", "1", "--keep_all_iterations", "--pool", "5",
    ] + list(extra)
    env = dict(os.environ)
    for k in ("RELION_COARSE_EARLY", "RELION_COARSE_EARLY_FINAL_ITERS"):
        env.pop(k, None)
    env.update(env_extra)
    res = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=900)
    assert res.returncode == 0, f"{prog} failed:\n{res.stderr[-3000:]}"
    return _orders(out, 4), res.stdout


@pytest.mark.integration
class TestCoarseEarly:

    @pytest.fixture
    def paths(self, test_data_dir, relion_bin, relion_runner):
        if not _gpu_available():
            pytest.skip("no GPU visible")
        return generate_test_dataset(str(test_data_dir / "data"), relion_bin, relion_runner,
                                     num_particles=60, size=32, angpix=1.0)

    @pytest.mark.parametrize("mpi", [False, True])
    def test_schedule(self, test_data_dir, relion_bin, paths, mpi):
        if mpi and (shutil.which("mpirun") is None or not (relion_bin / "relion_refine_mpi").exists()):
            pytest.skip("no MPI build")
        if mpi:
            orders, out = _run(relion_bin, paths, test_data_dir / "on_mpi", {}, True,
                               ["--coarse_early", "--coarse_early_final_iters", "2"])
        else:
            orders, out = _run(relion_bin, paths, test_data_dir / "on", {"RELION_COARSE_EARLY": "1",
                               "RELION_COARSE_EARLY_FINAL_ITERS": "2"}, False)
        assert orders == [1, 1, 2, 2], orders
        assert "Coarse early (--coarse_early): healpix order 1" in out

    def test_command_line_wins(self, test_data_dir, relion_bin, paths):
        orders, out = _run(relion_bin, paths, test_data_dir / "dont", {"RELION_COARSE_EARLY": "1"}, False,
                           ["--dont_coarse_early"])
        assert orders == [2, 2, 2, 2], orders

    def test_off_by_default(self, test_data_dir, relion_bin, paths):
        orders, out = _run(relion_bin, paths, test_data_dir / "off", {}, False)
        assert orders == [2, 2, 2, 2], orders
        assert "Coarse early" not in out
