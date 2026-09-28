#!/usr/bin/env python3
"""
relion_refine keeps only the files it needs from earlier iterations.

By default each iteration's files (<root>_itNNN_*) are deleted once the next
iteration has been written: a run that starts from an initial reference
(Class3D, Refine3D) keeps it000 and the last iteration, one without (Class2D,
InitialModel) only the last. --keep_all_iterations, or
RELION_KEEP_ALL_ITERATIONS=1 as the default, keeps everything;
--dont_keep_all_iterations overrides the environment.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

from fixtures import generate_test_dataset


def iterations(outdir: Path, root="run"):
    """Iteration numbers that still have files, for one output root."""
    found = set()
    for p in outdir.iterdir():
        m = re.match(rf"{re.escape(root)}_it(\d{{3,}})[_.]", p.name)
        if m:
            found.add(int(m.group(1)))
    return sorted(found)


def refine(relion_bin, outdir: Path, args, mpi=0, env_extra=None):
    env = dict(os.environ)
    env.pop("RELION_KEEP_ALL_ITERATIONS", None)
    if env_extra:
        env.update(env_extra)
    outdir.mkdir(parents=True, exist_ok=True)
    if mpi:
        cmd = ["mpirun", "--oversubscribe", "-n", str(mpi), str(relion_bin / "relion_refine_mpi")]
    else:
        cmd = [str(relion_bin / "relion_refine")]
    # relion_refine wants a directory in --o
    args = [str(outdir / args[i]) if i > 0 and args[i - 1] == "--o" else a for i, a in enumerate(args)]
    cmd += args + ["--j", "1", "--random_seed", "7", "--dont_combine_weights_via_disc", "--pool", "10"]
    r = subprocess.run(cmd, cwd=outdir, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


@pytest.fixture
def dataset(test_data_dir, relion_bin, relion_runner):
    return generate_test_dataset(str(test_data_dir / "data"), relion_bin, relion_runner,
                                 num_particles=40, size=32)


def class3d_args(paths, iters=3, root="run"):
    return ["--i", str(paths["star_file"]), "--o", root, "--ref", str(paths["initial_model"]),
            "--K", "2", "--iter", str(iters), "--ini_high", "20", "--tau2_fudge", "4",
            "--particle_diameter", "25", "--oversampling", "1", "--healpix_order", "1",
            "--offset_range", "3", "--offset_step", "2", "--sym", "C1"]


def class2d_args(paths, iters=3):
    return ["--i", str(paths["star_file"]), "--o", "run", "--K", "2", "--iter", str(iters),
            "--tau2_fudge", "2", "--particle_diameter", "25", "--oversampling", "1",
            "--psi_step", "12", "--offset_range", "3", "--offset_step", "2"]


@pytest.mark.integration
class TestKeepIterations:

    def test_class3d_keeps_first_and_last(self, test_data_dir, relion_bin, dataset):
        out = test_data_dir / "c3d"
        refine(relion_bin, out, class3d_args(dataset))
        assert iterations(out) == [0, 3]
        # The last iteration is complete: it can be continued from
        for suffix in ("_optimiser.star", "_model.star", "_data.star", "_sampling.star", "_class001.mrc"):
            assert (out / f"run_it003{suffix}").exists(), suffix

    def test_class2d_keeps_only_the_last(self, test_data_dir, relion_bin, dataset):
        out = test_data_dir / "c2d"
        refine(relion_bin, out, class2d_args(dataset))
        assert iterations(out) == [3]
        assert (out / "run_it003_classes.mrcs").exists()

    def test_keep_all_iterations(self, test_data_dir, relion_bin, dataset):
        out = test_data_dir / "all"
        refine(relion_bin, out, class3d_args(dataset) + ["--keep_all_iterations"])
        assert iterations(out) == [0, 1, 2, 3]

    def test_environment_default_and_its_override(self, test_data_dir, relion_bin, dataset):
        env = {"RELION_KEEP_ALL_ITERATIONS": "1"}
        out = test_data_dir / "env"
        refine(relion_bin, out, class2d_args(dataset, iters=2), env_extra=env)
        assert iterations(out) == [0, 1, 2]
        out = test_data_dir / "env_override"
        refine(relion_bin, out, class2d_args(dataset, iters=2) + ["--dont_keep_all_iterations"], env_extra=env)
        assert iterations(out) == [2]

    def test_mpi(self, test_data_dir, relion_bin, dataset):
        out = test_data_dir / "mpi"
        refine(relion_bin, out, class3d_args(dataset), mpi=3)
        assert iterations(out) == [0, 3]

    def test_continue_keeps_what_it_continued_from(self, test_data_dir, relion_bin, dataset):
        out = test_data_dir / "cont"
        refine(relion_bin, out, class3d_args(dataset))
        assert iterations(out) == [0, 3]
        refine(relion_bin, out, ["--continue", "run_it003_optimiser.star", "--o", "run_ct3", "--iter", "5"])
        # The original run is left as it was; the continuation keeps its last iteration
        assert iterations(out) == [0, 3]
        assert iterations(out, "run_ct3") == [5]
