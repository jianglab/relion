#!/usr/bin/env python3
"""
Integration test for the half-map alignment of a 3D auto-refine under MPI (alignHalves, on by default).

alignHalves does not move any map: it finds how half 2 has to be moved to match half 1 and applies the
inverse to the orientations and offsets of the half-2 particles, so that the next reconstruction of
half 2 is in the frame of half 1. For 2D particle images the image-plane offset that goes with a 3D shift
t is the first two components of A * t, A being the particle's orientation matrix; subtracting t, as was
done before, is only right for particles that look down the map's z axis.

The half-2 particles are given poses in a rotated and shifted frame and the search is kept local, so
without alignment the halves stay in different frames. With the alignment, the half maps reconstructed
from the written data.star (half 1 and half 2 separately) must agree much better than without it.
"""

import os
import subprocess
import shutil

import numpy as np
import pytest

from fixtures import generate_test_dataset, read_mrc


def _parse(path):
    with open(path) as f:
        lines = f.read().split("\n")
    i = [k for k, l in enumerate(lines) if l.startswith("data_particles")][0]
    cols, k = {}, i
    while not lines[k].startswith("loop_"):
        k += 1
    k += 1
    while lines[k].startswith("_"):
        name, num = lines[k].split()[:2]
        cols[name[1:]] = int(num[1:]) - 1
        k += 1
    start, rows = k, []
    while k < len(lines) and lines[k].strip():
        rows.append(lines[k].split())
        k += 1
    return lines, cols, start, rows


def _euler(a, b, g):
    a, b, g = np.radians([a, b, g])
    ca, cb, cg, sa, sb, sg = np.cos(a), np.cos(b), np.cos(g), np.sin(a), np.sin(b), np.sin(g)
    cc, cs, sc, ss = cb * ca, cb * sa, sb * ca, sb * sa
    return np.array([[cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb],
                     [-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb],
                     [sc, ss, cb]])


def _move_half2(star_in, star_out, rot_shift=3.0, shift=(2.5, -2.0, 1.5)):
    """Express the half-2 poses in a frame that is rotated about z and shifted in 3D."""
    lines, c, start, rows = _parse(star_in)
    t = np.array(shift)
    for r in rows:
        if int(r[c["rlnRandomSubset"]]) != 2:
            continue
        rot = float(r[c["rlnAngleRot"]]) + rot_shift
        a = _euler(rot, float(r[c["rlnAngleTilt"]]), float(r[c["rlnAnglePsi"]]))
        r[c["rlnAngleRot"]] = repr(rot)
        # float(): with NumPy 2, repr() of a numpy float is "np.float64(...)", which RELION cannot read
        r[c["rlnOriginXAngst"]] = repr(float(r[c["rlnOriginXAngst"]]) + float((a @ t)[0]))
        r[c["rlnOriginYAngst"]] = repr(float(r[c["rlnOriginYAngst"]]) + float((a @ t)[1]))
    with open(star_out, "w") as f:
        f.write("\n".join(lines[:start] + [" ".join(r) for r in rows] + lines[start + len(rows):]))


def _refine(relion_bin, star, ref, out, extra=()):
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["mpirun", "--oversubscribe", "-n", "3", str(relion_bin / "relion_refine_mpi"),
           "--auto_refine", "--split_random_halves", "--i", str(star), "--o", str(out / "run"),
           "--ref", str(ref), "--ini_high", "15", "--particle_diameter", "25", "--ctf",
           "--flatten_solvent", "--zero_mask", "--oversampling", "1", "--healpix_order", "4",
           "--auto_local_healpix_order", "4", "--sigma_ang", "0.5", "--offset_range", "1",
           "--offset_step", "0.5", "--sym", "c1", "--norm", "--scale", "--j", "1",
           "--dont_combine_weights_via_disc", "--random_seed", "7", "--pool", "5"] + list(extra)
    res = subprocess.run(cmd, text=True, capture_output=True, timeout=7200,
                         env=dict(os.environ, RELION_ALIGN_TIMING="1"))
    assert res.returncode == 0, res.stderr[-3000:]
    return res.stdout


def _half_correlation(relion_bin, out):
    maps = []
    for half in (1, 2):
        m = out / f"rec{half}.mrc"
        res = subprocess.run([str(relion_bin / "relion_reconstruct"), "--i", str(out / "run_data.star"),
                              "--o", str(m), "--ctf", "--subset", str(half), "--sym", "c1"],
                             text=True, capture_output=True, timeout=600)
        assert res.returncode == 0, res.stderr[-2000:]
        maps.append(np.asarray(read_mrc(str(m)), dtype=np.float64))
    return np.corrcoef(maps[0].ravel(), maps[1].ravel())[0, 1]


@pytest.mark.integration
class TestMpiAlignHalves:

    def test_half2_metadata_follows_the_alignment(self, test_data_dir, relion_bin, relion_runner):
        if shutil.which("mpirun") is None or not (relion_bin / "relion_refine_mpi").exists():
            pytest.skip("no MPI build")
        paths = generate_test_dataset(str(test_data_dir / "data"), relion_bin, relion_runner,
                                      num_particles=200, size=32, angpix=1.0)
        moved = test_data_dir / "moved.star"
        _move_half2(paths["star_file"], moved)

        log_on = _refine(relion_bin, moved, paths["initial_model"], test_data_dir / "on")
        assert "Aligning half2 to half1" in log_on, "the half alignment did not run"
        _refine(relion_bin, moved, paths["initial_model"], test_data_dir / "off", ["--dont_align_halves"])

        c_on = _half_correlation(relion_bin, test_data_dir / "on")
        c_off = _half_correlation(relion_bin, test_data_dir / "off")
        assert c_on > c_off + 0.1, f"half-map agreement with alignment {c_on:.3f}, without {c_off:.3f}"
        assert c_on > 0.8, c_on
