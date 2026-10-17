#!/usr/bin/env python3
"""
RELION_PREP_AHEAD: the CTF images of the accelerated expectation step are memoised per distinct set
of CTF values. Particles that share CTF values must give exactly the files of the plain run:
data.star, model.star and the maps after two iterations. Needs a CUDA GPU (skipped without one).
Two plain runs are compared first: where the GPU makes plain runs differ (atomics), the identity
check cannot be bitwise and falls back to the likelihood.
"""

import filecmp
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _have_cuda_gpu(relion_bin):
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
    return "RELION_PREP_AHEAD" in subprocess.run(["strings", str(exe)], capture_output=True, text=True).stdout


def _share_ctf(star_in, star_out, n_sets=4):
    """Give the particles n_sets distinct (defocus U, V, angle) values, the way micrographs do."""
    lines = Path(star_in).read_text().splitlines()
    out, in_particles, cols = [], False, None
    for l in lines:
        if l.startswith("data_particles"):
            in_particles = True
        if in_particles and l.startswith("_rln") and "#" in l:
            cols = (cols or []) + [l.split()[0]]
        if in_particles and cols and l and not l.startswith(("_", "#", "data_", "loop_")):
            f = l.split("\t")
            k = hash_index(f[0]) % n_sets
            f[cols.index("_rlnDefocusU")] = f"{12000 + 1500 * k:.2f}"
            f[cols.index("_rlnDefocusV")] = f"{12500 + 1500 * k:.2f}"
            f[cols.index("_rlnDefocusAngle")] = f"{20 + 30 * k:.2f}"
            l = "\t".join(f)
        out.append(l)
    Path(star_out).write_text("\n".join(out) + "\n")


def hash_index(image_name):
    return int(image_name.split("@")[0])


def _log_likelihood(model_star):
    for line in Path(model_star).read_text().splitlines():
        if line.startswith("_rlnLogLikelihood"):
            return float(line.split()[1])
    raise AssertionError("no _rlnLogLikelihood in " + str(model_star))


def _text(path, out):
    t = Path(path).read_text().replace(str(out), "OUT")
    return "\n".join(l for l in t.splitlines() if not (l.startswith("# ") and "--o" in l))


def test_prep_ahead_identical_results(test_data_dir, relion_bin, relion_runner):
    if not _have_cuda_gpu(relion_bin):
        pytest.skip("needs a CUDA GPU and a build with RELION_PREP_AHEAD")
    from fixtures import generate_test_dataset

    paths = generate_test_dataset(test_data_dir / "data_pa", relion_bin, relion_runner,
                                  num_particles=60, size=32, angpix=1.0)
    star = str(test_data_dir / "shared_ctf.star")
    _share_ctf(paths["star_file"], star)

    def run(tag, prep):
        out = test_data_dir / tag
        out.mkdir()
        cmd = [str(relion_bin / "relion_refine"),
               "--i", star, "--o", str(out / "run"),
               "--ref", paths["initial_model"], "--K", "2", "--iter", "2",
               "--ini_high", "50", "--ctf", "--tau2_fudge", "4", "--particle_diameter", "25",
               "--flatten_solvent", "--zero_mask", "--oversampling", "1",
               "--healpix_order", "1", "--offset_range", "5", "--offset_step", "2",
               "--sym", "c1", "--norm", "--scale", "--j", "1", "--random_seed", "7",
               "--dont_align_classes", "--keep_all_iterations", "--gpu", "0"]
        env = dict(os.environ)
        env.pop("RELION_PREP_AHEAD", None)
        env["RELION_ITER_TIMING"] = "1"
        if prep is not None:
            env["RELION_PREP_AHEAD"] = prep
        r = relion_runner(cmd, timeout=900, env=env)
        assert r.returncode == 0, f"{tag} failed:\n{r.stderr[-3000:]}"
        return r, out

    r_off, d_off = run("off", None)
    r_off2, d_off2 = run("off2", "0")
    r_on, d_on = run("on", "1")
    assert "RELION_PREP_AHEAD is on" in r_on.stdout
    assert "PREP_AHEAD CTF memo" in r_on.stderr, r_on.stderr[-1500:]
    assert "RELION_PREP_AHEAD is on" not in r_off.stdout

    # "check" computes every memoised image again and compares it bit for bit: this proves identity
    # even where two plain GPU runs differ from each other
    r_chk, d_chk = run("chk", "check")
    last = [l for l in r_chk.stderr.splitlines() if l.startswith("PREP_AHEAD CTF memo") and "check:" in l]
    assert last, r_chk.stderr[-1500:]
    assert " 0 DIFFERENT" in last[-1] and " 0 identical" not in last[-1], last[-1]

    names = ["run_it002_data.star", "run_it002_model.star", "run_it002_class001.mrc", "run_it002_class002.mrc"]
    plain_identical = all(
        (_text(d_off / n, d_off) == _text(d_off2 / n, d_off2)) if n.endswith(".star")
        else filecmp.cmp(d_off / n, d_off2 / n, shallow=False) for n in names)
    print(f"plain runs identical: {plain_identical}")
    if plain_identical:
        for n in names:
            if n.endswith(".star"):
                assert _text(d_off / n, d_off) == _text(d_on / n, d_on), f"{n} differs with RELION_PREP_AHEAD"
            else:
                assert filecmp.cmp(d_off / n, d_on / n, shallow=False), f"{n} differs with RELION_PREP_AHEAD"
    else:
        ll_off = _log_likelihood(d_off / "run_it002_model.star")
        ll_on = _log_likelihood(d_on / "run_it002_model.star")
        assert abs(ll_on - ll_off) <= 1e-5 * abs(ll_off), (ll_on, ll_off)
