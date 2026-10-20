#!/usr/bin/env python3
"""
RELION_CACHE_LAZY=1 fills the --cache_dir copy in the background.

The job starts at once and reads each stack from its original place until
the copy of that stack is complete. Results must not depend on how the cache
was filled, so a lazy run is compared with an eager run, MPI with one and two followers (relion_refine without MPI does not use --cache_dir).
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from fixtures import generate_test_dataset


def refine(relion_bin, outdir: Path, args, cache_dir=None, lazy=False, mpi=2, cwd=None):
    env = dict(os.environ)
    env.pop("RELION_CACHE_LAZY", None)
    if lazy:
        env["RELION_CACHE_LAZY"] = "1"
    outdir.mkdir(parents=True, exist_ok=True)
    if mpi:
        cmd = ["mpirun", "--oversubscribe", "-n", str(mpi), str(relion_bin / "relion_refine_mpi")]
    else:
        cmd = [str(relion_bin / "relion_refine")]
    cmd += args + ["--o", str(outdir / "run"), "--j", "1", "--random_seed", "7",
                   "--dont_combine_weights_via_disc", "--pool", "10", "--keep_all_iterations"]
    if cache_dir is not None:
        cmd += ["--cache_dir", str(cache_dir)]
    r = subprocess.run(cmd, cwd=cwd or outdir, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout + r.stderr


@pytest.fixture
def dataset(test_data_dir, relion_bin, relion_runner):
    return generate_test_dataset(str(test_data_dir / "data"), relion_bin, relion_runner,
                                 num_particles=40, size=32)


def args_for(paths, iters=2):
    return ["--i", str(paths["star_file"]), "--ref", str(paths["initial_model"]),
            "--K", "2", "--iter", str(iters), "--ini_high", "20", "--tau2_fudge", "4",
            "--particle_diameter", "25", "--oversampling", "1", "--healpix_order", "1",
            "--offset_range", "3", "--offset_step", "2", "--sym", "C1"]


def cached_stacks(cache_dir: Path):
    return sorted(p for p in cache_dir.rglob("*.mrcs"))


def leftovers(cache_dir: Path):
    return sorted(p for p in cache_dir.rglob("*") if p.suffix in (".lazytmp", ".lazylock"))


def model_text(outdir: Path, it=2):
    return (outdir / f"run_it{it:03d}_model.star").read_text().replace(str(outdir), "OUT")


@pytest.mark.integration
class TestLazyCache:

    @pytest.mark.parametrize("mpi", [2, 3])
    def test_lazy_matches_eager(self, test_data_dir, relion_bin, dataset, mpi):
        args = args_for(dataset)
        eager_cache = test_data_dir / "cache_eager"
        lazy_cache = test_data_dir / "cache_lazy"

        refine(relion_bin, test_data_dir / "eager", args, cache_dir=eager_cache, mpi=mpi)
        out = refine(relion_bin, test_data_dir / "lazy", args, cache_dir=lazy_cache, lazy=True, mpi=mpi)

        assert "Lazy cache" in out, "the lazy fill was not announced"
        assert model_text(test_data_dir / "lazy") == model_text(test_data_dir / "eager")

        lazy_files = cached_stacks(lazy_cache)
        assert lazy_files, "the lazy run left nothing in the cache"
        assert [p.stat().st_size for p in lazy_files] == [p.stat().st_size for p in cached_stacks(eager_cache)]
        assert not leftovers(lazy_cache)
        for p in lazy_files:
            assert p.read_bytes() == Path(dataset["particles_mrc"]).read_bytes()

    def test_second_job_reuses_cache(self, test_data_dir, relion_bin, dataset):
        args = args_for(dataset, iters=1)
        cache = test_data_dir / "cache"
        proj = test_data_dir / "proj"
        proj.mkdir()
        refine(relion_bin, test_data_dir / "first", args, cache_dir=cache, lazy=True, mpi=2, cwd=proj)
        before = {p: p.stat().st_ino for p in cached_stacks(cache)}
        assert before
        out = refine(relion_bin, test_data_dir / "second", args, cache_dir=cache, lazy=True, mpi=2, cwd=proj)
        after = {p: p.stat().st_ino for p in cached_stacks(cache)}
        assert after == before, f"a complete cache was copied again\n{before}\n{after}\n{out[-1500:]}"
        assert "Lazy cache" not in out

    def test_stale_cache_is_replaced(self, test_data_dir, relion_bin, dataset):
        args = args_for(dataset, iters=1)
        cache = test_data_dir / "cache"
        proj = test_data_dir / "proj"
        proj.mkdir()
        refine(relion_bin, test_data_dir / "first", args, cache_dir=cache, lazy=True, mpi=2, cwd=proj)
        stacks = cached_stacks(cache)
        assert stacks
        for p in stacks:
            p.write_bytes(b"garbage")
        refine(relion_bin, test_data_dir / "second", args, cache_dir=cache, lazy=True, mpi=2, cwd=proj)
        for p in cached_stacks(cache):
            assert p.read_bytes() == Path(dataset["particles_mrc"]).read_bytes()
        assert model_text(test_data_dir / "second", 1) == model_text(test_data_dir / "first", 1)
