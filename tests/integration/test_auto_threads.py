#!/usr/bin/env python3
"""
Without --j, relion_refine runs a thread per CPU core the process may use.

One process takes all the cores it may run on; MPI workers on a node that may
run on the same cores share them, and every process gets the same number. --j
still sets the number.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from fixtures import generate_test_dataset


def cores(n):
    """The first n cores this test may run on, as a taskset list (None if fewer)."""
    allowed = sorted(os.sched_getaffinity(0))
    return ",".join(str(c) for c in allowed[:n]) if len(allowed) >= n else None


def refine(relion_bin, outdir: Path, paths, core_list, mpi=0, extra=()):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = ["taskset", "-c", core_list]
    if mpi:
        cmd += ["mpirun", "--bind-to", "none", "--oversubscribe", "-n", str(mpi), str(relion_bin / "relion_refine_mpi")]
    else:
        cmd += [str(relion_bin / "relion_refine")]
    cmd += ["--i", str(paths["star_file"]), "--o", str(outdir / "run"), "--K", "2", "--iter", "1",
            "--tau2_fudge", "2", "--particle_diameter", "25", "--oversampling", "1",
            "--psi_step", "12", "--offset_range", "3", "--offset_step", "2",
            "--random_seed", "7", "--dont_combine_weights_via_disc", "--pool", "10", *extra]
    r = subprocess.run(cmd, cwd=outdir, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


@pytest.fixture
def dataset(test_data_dir, relion_bin, relion_runner):
    return generate_test_dataset(str(test_data_dir / "data"), relion_bin, relion_runner,
                                 num_particles=40, size=32)


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("taskset") is None, reason="needs taskset")
class TestAutoThreads:
    def test_one_process_takes_its_cores(self, test_data_dir, relion_bin, dataset):
        core_list = cores(3)
        if core_list is None:
            pytest.skip("needs 3 cores")
        out = refine(relion_bin, test_data_dir / "auto_j", dataset, core_list)
        assert "Using 3 threads: --j was not given" in out

    def test_given_j_is_kept(self, test_data_dir, relion_bin, dataset):
        core_list = cores(3)
        if core_list is None:
            pytest.skip("needs 3 cores")
        out = refine(relion_bin, test_data_dir / "given_j", dataset, core_list, extra=("--j", "2"))
        assert "--j was not given" not in out

    def test_mpi_workers_share_their_cores(self, test_data_dir, relion_bin, dataset):
        if shutil.which("mpirun") is None:
            pytest.skip("needs mpirun")
        core_list = cores(4)
        if core_list is None:
            pytest.skip("needs 4 cores")
        # a leader and 2 workers, all allowed on the same 4 cores: 2 threads each
        out = refine(relion_bin, test_data_dir / "auto_j_mpi", dataset, core_list, mpi=3)
        assert re.search(r"Using 2 threads per process: --j was not given", out), out[-2000:]
