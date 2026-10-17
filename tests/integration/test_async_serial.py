#!/usr/bin/env python3
"""
RELION_ASYNC_SERIAL: the half-map alignment of auto-refine runs in the background beside the
E-step (and, on a GPU, the weighted sums are zeroed in the background). The files of the run
must be byte for byte those of the run without the switch.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _have_mpi(relion_bin):
    return shutil.which("mpirun") is not None and (Path(relion_bin) / "relion_refine_mpi").exists()


def _have_cuda_gpu(relion_bin):
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        if subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=30).returncode != 0:
            return False
    except Exception:
        return False
    exe = Path(relion_bin) / "relion_refine_mpi"
    if not exe.exists():
        return False
    return "ASYNC_SERIAL" in subprocess.run(["strings", str(exe)], capture_output=True, text=True).stdout


def _have_switch(relion_bin):
    exe = Path(relion_bin) / "relion_refine_mpi"
    if not exe.exists():
        return False
    return "ASYNC_SERIAL" in subprocess.run(["strings", str(exe)], capture_output=True, text=True).stdout


def _refine(relion_bin, relion_runner, data_dir, out_dir, extra_env, extra_args):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["mpirun", "-n", "3", "--oversubscribe", str(Path(relion_bin) / "relion_refine_mpi"),
           "--i", str(data_dir / "particles.star"), "--o", str(out_dir / "run"),
           "--ref", str(data_dir / "initial_model.mrc"),
           "--auto_refine", "--split_random_halves", "--iter", "2", "--keep_all_iterations",
           "--ini_high", "40", "--ctf", "--particle_diameter", "25", "--flatten_solvent", "--zero_mask",
           "--oversampling", "1", "--healpix_order", "1", "--offset_range", "5", "--offset_step", "2",
           "--sym", "c1", "--norm", "--scale", "--j", "1", "--random_seed", "11",
           "--spatial_frequency_mode", "s", "--pad", "1"] + extra_args
    env = dict(os.environ)
    env.pop("RELION_ASYNC_SERIAL", None)
    env.update(extra_env)
    r = relion_runner(cmd, timeout=1200, env=env)
    assert r.returncode == 0, f"run failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}"
    return r


def _same_files(a, b):
    """Compare the outputs of two runs: STAR files without the name of the run directory, maps
    without the header (it carries the file name). One thread per rank (--j 1) is used because
    with more threads two identical runs already differ in the last bits of some half maps."""
    names = []
    for pat in ("run_it*_data.star", "run_it*_half?_model.star", "run_it*_half?_class001.mrc",
                "run_data.star", "run_model.star", "run_class001.mrc", "run_half?_class001_unfil.mrc"):
        names += [p.name for p in sorted(a.glob(pat))]
    assert any("_data.star" in n for n in names), "no output found in " + str(a)
    assert any("half2_class001" in n for n in names), "no half map found in " + str(a)
    for n in names:
        assert (b / n).exists(), f"{n} missing from the second run"
        if n.endswith(".mrc"):
            assert (a / n).read_bytes()[1024:] == (b / n).read_bytes()[1024:], f"{n} differs"
        else:
            ta = (a / n).read_text().replace(str(a) + "/", "")
            tb = (b / n).read_text().replace(str(b) + "/", "")
            assert ta == tb, f"{n} differs"


@pytest.mark.integration
def test_async_halves_alignment_is_bit_identical(test_data_dir, relion_bin, fixtures_module, relion_runner):
    """Half-map alignment in the background (CPU E-step): same files as the sequential order."""
    if not _have_mpi(relion_bin) or not _have_switch(relion_bin):
        pytest.skip("needs relion_refine_mpi, mpirun and a build with RELION_ASYNC_SERIAL")
    fixtures_module["generate_test_dataset"](str(test_data_dir), relion_bin, relion_runner,
                                             num_particles=60, size=32, angpix=1.0)
    base = _refine(relion_bin, relion_runner, test_data_dir, test_data_dir / "seq", {}, [])
    asy = _refine(relion_bin, relion_runner, test_data_dir, test_data_dir / "asy",
                  {"RELION_ASYNC_SERIAL": "1"}, [])
    assert "runs in the background during the E-step" in asy.stdout, asy.stdout[-3000:]
    assert "runs in the background" not in base.stdout
    _same_files(test_data_dir / "seq", test_data_dir / "asy")


@pytest.mark.integration
def test_async_serial_off_value_keeps_the_sequential_order(test_data_dir, relion_bin, fixtures_module, relion_runner):
    """RELION_ASYNC_SERIAL=0 is the same as unset."""
    if not _have_mpi(relion_bin) or not _have_switch(relion_bin):
        pytest.skip("needs relion_refine_mpi, mpirun and a build with RELION_ASYNC_SERIAL")
    fixtures_module["generate_test_dataset"](str(test_data_dir), relion_bin, relion_runner,
                                             num_particles=60, size=32, angpix=1.0)
    off = _refine(relion_bin, relion_runner, test_data_dir, test_data_dir / "off",
                  {"RELION_ASYNC_SERIAL": "0"}, [])
    assert "ASYNC_SERIAL" not in off.stdout


def _angles(star):
    """The orientation columns of the particle table of a data.star, as strings."""
    blocks, cols, rows = [], [], []
    for line in star.read_text().splitlines():
        if line.startswith("data_"):
            blocks.append((cols, rows))
            cols, rows = [], []
        elif line.startswith("_rln"):
            cols.append(line.split()[0])
        elif cols and line.strip() and not line.startswith(("loop_", "#")):
            rows.append(line.split())
    blocks.append((cols, rows))
    cols, rows = max(blocks, key=lambda b: len(b[1]))
    idx = [cols.index(c) for c in ("_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi")]
    return [tuple(r[i] for i in idx) for r in rows]


def _log_likelihood(model_star):
    for line in model_star.read_text().splitlines():
        if line.startswith("_rlnLogLikelihood"):
            return float(line.split()[1])
    raise AssertionError("no _rlnLogLikelihood in " + str(model_star))


@pytest.mark.integration
def test_async_serial_on_gpu_stays_within_run_to_run_spread(test_data_dir, relion_bin, fixtures_module, relion_runner):
    """GPU runs are not byte-reproducible (atomic additions), so a run with the switch is compared with
    two runs without it: same orientations after the first iteration, and a likelihood that differs
    from the first run by no more than a few times what the two plain runs differ by."""
    if not _have_mpi(relion_bin) or not _have_cuda_gpu(relion_bin):
        pytest.skip("needs a CUDA GPU, relion_refine_mpi, mpirun and a build with RELION_ASYNC_SERIAL")
    fixtures_module["generate_test_dataset"](str(test_data_dir), relion_bin, relion_runner,
                                             num_particles=60, size=32, angpix=1.0)
    gpu = ["--gpu", "0"]
    for tag in ("gseq", "gseq2"):
        _refine(relion_bin, relion_runner, test_data_dir, test_data_dir / tag, {}, gpu)
    asy = _refine(relion_bin, relion_runner, test_data_dir, test_data_dir / "gasy",
                  {"RELION_ASYNC_SERIAL": "1"}, gpu)
    assert "weighted sums are zeroed in the background" in asy.stdout, asy.stdout[-3000:]
    assert _angles(test_data_dir / "gasy" / "run_it001_data.star") == \
        _angles(test_data_dir / "gseq" / "run_it001_data.star")
    ll = {t: _log_likelihood(test_data_dir / t / "run_it001_half1_model.star") for t in ("gseq", "gseq2", "gasy")}
    spread = abs(ll["gseq"] - ll["gseq2"])
    assert abs(ll["gasy"] - ll["gseq"]) <= max(10 * spread, 1e-5 * abs(ll["gseq"])), ll
