#!/usr/bin/env python3
"""
RELION_ASYNC_STAR_WRITE: the STAR files of an iteration (data, model, optimiser) are formatted
from a snapshot and written by the background thread. The files must be byte for byte the files
the plain run writes, the clean-up of old iterations must still keep only the last one, and the
"=check" mode (which compares each file with the text the calling thread would have written)
must report no difference. A CPU run with one thread is used, so two runs are identical.
"""

import os
import re
from pathlib import Path


def _run(relion_bin, relion_runner, paths, out, extra_env):
    out.mkdir()
    cmd = [str(relion_bin / "relion_refine"),
           "--i", paths["star_file"], "--o", str(out / "run"),
           "--ref", paths["initial_model"], "--K", "2", "--iter", "3",
           "--ini_high", "50", "--ctf", "--tau2_fudge", "4", "--particle_diameter", "25",
           "--flatten_solvent", "--zero_mask", "--oversampling", "1",
           "--healpix_order", "1", "--offset_range", "5", "--offset_step", "2",
           "--sym", "c1", "--norm", "--scale", "--j", "1", "--random_seed", "7",
           "--dont_align_classes", "--pad", "1"]
    env = dict(os.environ)
    env.pop("RELION_ASYNC_STAR_WRITE", None)
    env.update(extra_env)
    r = relion_runner(cmd, timeout=1800, env=env)
    assert r.returncode == 0, f"run failed:\n{r.stderr[-3000:]}"
    return r


def _text(path, out):
    # the output directory and the command line differ from run to run
    t = Path(path).read_text().replace(str(out), "OUT")
    return "\n".join(l for l in t.splitlines() if not (l.startswith("# ") and "--o" in l))


def test_async_star_write_identical_files(test_data_dir, relion_bin, relion_runner):
    from fixtures import generate_test_dataset

    paths = generate_test_dataset(test_data_dir / "data", relion_bin, relion_runner,
                                  num_particles=40, size=32, angpix=1.0)
    plain = test_data_dir / "plain"
    on = test_data_dir / "on"
    chk = test_data_dir / "chk"
    _run(relion_bin, relion_runner, paths, plain, {})
    _run(relion_bin, relion_runner, paths, on, {"RELION_ASYNC_STAR_WRITE": "1"})
    r = _run(relion_bin, relion_runner, paths, chk, {"RELION_ASYNC_STAR_WRITE": "check"})

    lines = [l for l in r.stderr.splitlines() if l.startswith("ASYNC_STAR_WRITE check")]
    for suffix in ("_data.star", "_model.star", "_optimiser.star"):
        assert any(l.endswith(suffix + ": identical") or (suffix in l and l.endswith(": identical")) for l in lines), \
            f"no check of {suffix}: {lines}\n{r.stderr[-1500:]}"
    assert not any("DIFFERENT" in l for l in lines)

    # Only the last iteration is kept (the clean-up joins the writer's queue), in all modes
    for d in (plain, on, chk):
        its = sorted({m.group(1) for f in d.iterdir() if (m := re.match(r"run_it(\d+)_", f.name))})
        assert its == ["000", "003"] or its == ["003"], f"{d.name}: iterations left {its}"

    for d in (on, chk):
        for suffix in ("_it003_data.star", "_it003_model.star", "_it003_optimiser.star"):
            a = _text(plain / ("run" + suffix), plain)
            b = _text(d / ("run" + suffix), d)
            assert a == b, f"{d.name} {suffix} differs from the plain run"
