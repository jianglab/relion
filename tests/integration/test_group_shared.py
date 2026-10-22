#!/usr/bin/env python3
"""
Integration test for RELION_GPU_GROUP_SHARED (particle groups on the GPU whose coarse pass projects
each orientation that the particles share once) and RELION_PARTICLE_ORDER=filament.

The synthetic particles get helical tube IDs (10 filaments, interleaved in the STAR file) and every
segment of a filament the angles of its first one, so with local searches the segments of a filament
search the same orientations. In filament order a group holds segments of one filament and shares
most of its projections. RELION_GPU_GROUP_CHECK compares every particle's coarse differences with
those of its own single-particle pass: at a fixed number of pixel chunks they must be the same bit
for bit. Skipped when no GPU is visible.
"""

import os
import re
import shutil
import subprocess

import pytest

from fixtures import generate_test_dataset

N_FIL = 10


def _gpu_available():
    if shutil.which("nvidia-smi") is None:
        return False
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


def _make_filaments(src, dst):
    """Tube ID i % N_FIL and the angles of the filament's first segment; track length along it."""
    with open(src) as f:
        lines = f.read().split("\n")
    out, cols, in_part, first = [], [], False, {}
    k = 0
    for line in lines:
        s = line.strip()
        if s.startswith("data_"):
            in_part = "particles" in s
        if in_part and s.startswith("_rln"):
            cols.append(s.split()[0])
            out.append(line)
            continue
        if in_part and cols and s and not s.startswith(("_", "loop_", "#")):
            if k == 0:
                out.append(f"_rlnHelicalTubeID #{len(cols) + 1}")
                out.append(f"_rlnHelicalTrackLengthAngst #{len(cols) + 2}")
            v = s.split()
            tube = k % N_FIL + 1
            if tube not in first:
                first[tube] = v
            for c in ("_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"):
                i = cols.index(c)
                v[i] = first[tube][cols.index(c)]
            out.append(" ".join(v + [str(tube), str(10.0 * (k // N_FIL))]))
            k += 1
            continue
        out.append(line)
    with open(dst, "w") as f:
        f.write("\n".join(out))


def _run(relion_bin, star, ref, out, env_extra):
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(relion_bin / "relion_refine"),
        "--i", str(star), "--o", str(out / "run"), "--ref", str(ref),
        "--K", "2", "--iter", "2", "--ini_high", "30",
        "--offset_range", "2", "--offset_step", "1", "--healpix_order", "2", "--sigma_ang", "8",
        "--ctf", "--tau2_fudge", "4", "--particle_diameter", "25",
        "--flatten_solvent", "--zero_mask", "--sym", "c1", "--norm", "--scale",
        "--j", "2", "--gpu", "0", "--dont_combine_weights_via_disc",
        "--random_seed", "1", "--pool", "30",
    ]
    env = dict(os.environ)
    env.update({"RELION_GPU_PARTICLE_GROUP": "6", "RELION_GPU_GROUP_SHARED": "1",
                "RELION_GPU_GROUP_CHECK": "1", "RELION_GPU_COARSE_SPLIT": "2"})
    env.update(env_extra)
    res = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=900)
    assert res.returncode == 0, f"relion_refine failed:\n{res.stderr[-3000:]}"
    checks = [l for l in res.stderr.splitlines() if l.startswith("GROUP_CHECK")]
    assert checks, "no coarse pass was compared"
    assert not any("DIFFER" in l for l in checks), "\n".join(checks[:5])
    share = re.findall(r"particle groups share their projections .*?: ([0-9.]+) particles per set, ([0-9.]+)x fewer",
                       res.stdout)
    assert share, "the shared coarse pass did not run"
    return float(share[0][1])


@pytest.mark.integration
class TestGroupShared:

    def test_shared_projections_are_exact(self, test_data_dir, relion_bin, relion_runner):
        if not _gpu_available():
            pytest.skip("no GPU visible")
        paths = generate_test_dataset(str(test_data_dir / "data"), relion_bin, relion_runner,
                                      num_particles=60, size=32, angpix=1.0)
        star = test_data_dir / "filaments.star"
        _make_filaments(paths["star_file"], star)
        interleaved = _run(relion_bin, star, paths["initial_model"], test_data_dir / "file_order",
                           {"RELION_PARTICLE_ORDER": "micrograph"})
        by_filament = _run(relion_bin, star, paths["initial_model"], test_data_dir / "filament_order",
                           {"RELION_PARTICLE_ORDER": "filament"})
        # neighbours in file order are other filaments; in filament order they share their orientations
        assert by_filament > 1.2 and by_filament > interleaved, (by_filament, interleaved)
