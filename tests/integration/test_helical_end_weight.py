#!/usr/bin/env python3
"""
Integration test for --helical_end_weight (Class2D, Class3D and Refine3D): helical segments whose box
reaches beyond the start or end of their filament contribute to the reconstruction with this weight.

A segment is an end segment when its rlnHelicalTrackLengthAngst is less than half the box size
from 0 or from its rlnHelicalFilamentLengthAngst. The synthetic filaments below have six segments
8 A apart (tracks 0 .. 40 A, length 40 A) in 32 A boxes: the two segments at 16 and 24 A are whole,
the other four of each filament are end segments.
"""

import math
import os
import struct
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pytest

from test_filament_class3d import parse_star_particles

NUM_FILAMENTS = 4
SEGMENTS_PER_FILAMENT = 6
STEP_A = 8.0
SIZE = 32
ANGPIX = 1.0


def write_star(path, particles_mrc, filament_length=None, track_offset=0., with_lengths=True, seed=7):
    """Particle STAR file with helical tracks (from track_offset on); filament_length=None: the span of the tracks."""
    rng = np.random.default_rng(seed)
    length = (SEGMENTS_PER_FILAMENT - 1) * STEP_A if filament_length is None else filament_length
    cols = ["rlnImageName", "rlnMicrographName", "rlnOpticsGroup", "rlnHelicalTubeID",
            "rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle", "rlnAngleRot", "rlnAngleTilt",
            "rlnAnglePsi", "rlnOriginXAngst", "rlnOriginYAngst", "rlnRandomSubset"]
    if with_lengths:
        cols += ["rlnHelicalTrackLengthAngst", "rlnHelicalFilamentLengthAngst"]
    lines = ["", "# version 30001", "", "data_optics", "", "loop_"]
    lines += ["_rlnOpticsGroupName #1", "_rlnOpticsGroup #2", "_rlnMicrographOriginalPixelSize #3",
              "_rlnVoltage #4", "_rlnSphericalAberration #5", "_rlnAmplitudeContrast #6",
              "_rlnImagePixelSize #7", "_rlnImageSize #8", "_rlnImageDimensionality #9"]
    lines.append(f"opticsGroup1 1 {ANGPIX:.6f} 300.0 2.7 0.1 {ANGPIX:.6f} {SIZE} 2")
    lines += ["", "# version 30001", "", "data_particles", "", "loop_"]
    lines += [f"_{c} #{i + 1}" for i, c in enumerate(cols)]
    n = 0
    for f in range(NUM_FILAMENTS):
        for s in range(SEGMENTS_PER_FILAMENT):
            d = rng.uniform(12000, 18000)
            row = [f"{n + 1:06d}@{particles_mrc}", f"mic_{f:03d}.mrc", "1", str(f + 1),
                   f"{d:.2f}", f"{d:.2f}", "0.0",
                   f"{rng.uniform(0, 360):.3f}", f"{math.degrees(math.acos(rng.uniform(-1, 1))):.3f}",
                   f"{rng.uniform(0, 360):.3f}", "0.0", "0.0", str(n % 2 + 1)]
            if with_lengths:
                row += [f"{track_offset + s * STEP_A:.6f}", f"{length:.6f}"]
            lines.append(" ".join(row))
            n += 1
    Path(path).write_text("\n".join(lines) + "\n")
    return n


@pytest.fixture
def dataset(test_data_dir, relion_bin, relion_runner):
    from fixtures import generate_test_dataset
    d = test_data_dir / "data"
    d.mkdir(parents=True, exist_ok=True)
    paths = generate_test_dataset(str(d), relion_bin, relion_runner,
                                  num_particles=NUM_FILAMENTS * SEGMENTS_PER_FILAMENT, size=SIZE, angpix=ANGPIX)
    return {"dir": d, "particles_mrc": paths["particles_mrc"], "initial_model": paths["initial_model"]}


def class2d_cmd(relion_bin, star, out, extra, gpu=False):
    cmd = [str(relion_bin / "relion_refine"), "--i", str(star), "--o", str(out / "run"),
           "--K", "2", "--iter", "2", "--tau2_fudge", "2", "--particle_diameter", "25", "--ctf",
           "--zero_mask", "--oversampling", "1", "--psi_step", "10", "--offset_range", "3",
           "--offset_step", "1", "--norm", "--scale", "--j", "1", "--random_seed", "1",
           "--dont_combine_weights_via_disc", "--pool", "3"]
    if gpu:
        cmd += ["--gpu", "0"]
    return cmd + extra


def refine3d_cmd(relion_bin, star, ref, out, extra):
    return ["mpirun", "-n", "3", str(relion_bin / "relion_refine_mpi"), "--i", str(star),
            "--o", str(out / "run"), "--ref", str(ref), "--auto_refine", "--split_random_halves",
            "--ini_high", "30", "--ctf", "--particle_diameter", "25", "--flatten_solvent", "--zero_mask",
            "--oversampling", "1", "--healpix_order", "1", "--auto_local_healpix_order", "2",
            "--offset_range", "3", "--offset_step", "1", "--sym", "c1", "--norm", "--scale",
            "--j", "1", "--random_seed", "1", "--dont_combine_weights_via_disc"] + extra


def read_mrc(path):
    with open(path, "rb") as fh:
        hdr = fh.read(1024)
        nx, ny, nz, mode = struct.unpack("4i", hdr[:16])
        next_ = struct.unpack("i", hdr[92:96])[0]
        assert mode == 2
        fh.seek(1024 + next_)
        return np.frombuffer(fh.read(), dtype=np.float32).reshape(nz, ny, nx)


def run_ok(relion_runner, cmd, timeout=900):
    result = relion_runner(cmd, timeout=timeout)
    assert result.returncode == 0, f"run failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    return result.stdout or ""


def have_cuda_gpu(relion_bin):
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        if subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=30).returncode != 0:
            return False
    except Exception:
        return False
    exe = Path(relion_bin) / "relion_refine"
    out = subprocess.run(["strings", str(exe)], capture_output=True, text=True).stdout
    return "cudaGetDeviceCount" in out or "libcudart" in out


@pytest.mark.integration
class TestHelicalEndWeight:

    def test_class2d_counts_end_segments(self, test_data_dir, dataset, relion_bin, relion_runner):
        star = dataset["dir"] / "segments.star"
        n = write_star(star, dataset["particles_mrc"])
        out = test_data_dir / "c2d_half"
        out.mkdir()
        text = run_ok(relion_runner, class2d_cmd(relion_bin, star, out, ["--helical_end_weight", "0.5"]))
        nr_end = NUM_FILAMENTS * (SEGMENTS_PER_FILAMENT - 2)
        assert f"Helical end segments: {nr_end} of {n} segments" in text, text[-3000:]
        data = parse_star_particles(str(out / "run_it002_data.star"), ["rlnClassNumber"])
        assert len(data["rlnClassNumber"]) == n

    def test_off_without_end_segments_is_unchanged(self, test_data_dir, dataset, relion_bin, relion_runner):
        # The segments in the middle of filaments much longer than the box: no end segments, so the
        # weight must change nothing
        star = dataset["dir"] / "long.star"
        write_star(star, dataset["particles_mrc"], filament_length=1000., track_offset=500.)
        outs = []
        for name, extra in (("off", []), ("zero", ["--helical_end_weight", "0"])):
            out = test_data_dir / ("c2d_long_" + name)
            out.mkdir()
            text = run_ok(relion_runner, class2d_cmd(relion_bin, star, out, extra))
            if extra:
                assert "Helical end segments: 0 of" in text
            outs.append(read_mrc(out / "run_it002_classes.mrcs"))
        assert np.array_equal(outs[0], outs[1])

    def test_weight_changes_the_averages(self, test_data_dir, dataset, relion_bin, relion_runner):
        star = dataset["dir"] / "segments.star"
        write_star(star, dataset["particles_mrc"])
        avgs = {}
        for name, extra in (("off", []), ("zero", ["--helical_end_weight", "0"]),
                            ("half", ["--helical_end_weight", "0.5"])):
            out = test_data_dir / ("c2d_" + name)
            out.mkdir()
            run_ok(relion_runner, class2d_cmd(relion_bin, star, out, extra + ["--iter", "1"]))
            avgs[name] = read_mrc(out / "run_it001_classes.mrcs")
            assert np.all(np.isfinite(avgs[name]))
        assert not np.array_equal(avgs["off"], avgs["zero"])
        assert not np.array_equal(avgs["off"], avgs["half"])
        assert not np.array_equal(avgs["zero"], avgs["half"])

    def test_missing_filament_length_is_an_error(self, test_data_dir, dataset, relion_bin, relion_runner):
        star = dataset["dir"] / "nolength.star"
        write_star(star, dataset["particles_mrc"], with_lengths=False)
        out = test_data_dir / "c2d_nolength"
        out.mkdir()
        result = relion_runner(class2d_cmd(relion_bin, star, out, ["--helical_end_weight", "0.5"]), timeout=600)
        assert result.returncode != 0
        assert "rlnHelicalFilamentLengthAngst" in (result.stdout or "") + (result.stderr or "")

    def test_bad_weight_is_an_error(self, test_data_dir, dataset, relion_bin, relion_runner):
        star = dataset["dir"] / "segments.star"
        write_star(star, dataset["particles_mrc"])
        out = test_data_dir / "c2d_bad"
        out.mkdir()
        result = relion_runner(class2d_cmd(relion_bin, star, out, ["--helical_end_weight", "1.5"]), timeout=600)
        assert result.returncode != 0
        assert "between 0 and 1" in (result.stdout or "") + (result.stderr or "")

    def test_refine3d_runs(self, test_data_dir, dataset, relion_bin, relion_runner):
        star = dataset["dir"] / "segments.star"
        n = write_star(star, dataset["particles_mrc"])
        out = test_data_dir / "r3d"
        out.mkdir()
        text = run_ok(relion_runner, refine3d_cmd(relion_bin, star, dataset["initial_model"], out,
                                                  ["--helical_end_weight", "0.3"]), timeout=1800)
        assert f"Helical end segments: {NUM_FILAMENTS * (SEGMENTS_PER_FILAMENT - 2)} of {n}" in text
        assert np.all(np.isfinite(read_mrc(out / "run_class001.mrc")))

    def test_gpu_counts_and_weights(self, test_data_dir, dataset, relion_bin, relion_runner):
        if not have_cuda_gpu(relion_bin):
            pytest.skip("no CUDA build or GPU")
        star = dataset["dir"] / "segments.star"
        n = write_star(star, dataset["particles_mrc"])
        avgs = {}
        for name, extra in (("off", []), ("zero", ["--helical_end_weight", "0"]),
                            ("half", ["--helical_end_weight", "0.5"])):
            out = test_data_dir / ("gpu_" + name)
            out.mkdir()
            text = run_ok(relion_runner, class2d_cmd(relion_bin, star, out, extra + ["--iter", "1"], gpu=True))
            if extra:
                assert f"Helical end segments: {NUM_FILAMENTS * (SEGMENTS_PER_FILAMENT - 2)} of {n}" in text
            avgs[name] = read_mrc(out / "run_it001_classes.mrcs")
            assert np.all(np.isfinite(avgs[name]))
        assert not np.allclose(avgs["off"], avgs["zero"])
        assert not np.allclose(avgs["off"], avgs["half"])
