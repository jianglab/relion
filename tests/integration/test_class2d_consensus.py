"""Fresh fixed-class Class2D with synthetic particles; no upstream reference images.

Run with RELION_BIN_DIR pointing to a built RELION bin directory. MPI and GPU
comparisons additionally require RELION_TEST_MPI=1 or RELION_TEST_GPU=1.
"""
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess

import numpy as np
import pytest


@pytest.fixture
def consensus_binaries():
    directory = Path(os.environ.get("RELION_BIN_DIR", Path(__file__).resolve().parents[2] / "build/bin"))
    binaries = {}
    for name in ("relion_class2d_consensus", "relion_refine"):
        candidate = directory / name
        binary = str(candidate) if candidate.exists() else shutil.which(name)
        if not binary:
            pytest.skip(f"Build {name} and set RELION_BIN_DIR")
        binaries[name] = binary
    binaries["directory"] = directory
    return binaries


def _run(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _loop(path, block):
    """Read the one loop in a named block of the STAR fixtures/results."""
    active, looping, labels, rows = False, False, [], []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("data_"):
            if active:
                break
            active = line == "data_" + block
        elif active and line == "loop_":
            looping = True
        elif active and looping and line.startswith("_"):
            labels.append(line.split()[0][1:])
        elif active and looping:
            values = shlex.split(line)
            assert len(values) == len(labels)
            rows.append(dict(zip(labels, values)))
    return rows


def _read_mrc(path):
    with Path(path).open("rb") as handle:
        header = handle.read(1024)
        nx, ny, nz, mode = struct.unpack_from("<4i", header)
        assert mode == 2
        handle.seek(1024 + struct.unpack_from("<i", header, 92)[0])
        return np.frombuffer(handle.read(nx * ny * nz * 4), dtype="<f4").reshape(nz, ny, nx).copy()


def _write_stack(path, count=64):
    rng = np.random.default_rng(19)
    y, x = np.mgrid[-16:16, -16:16]
    data = rng.normal(0, 1, (count, 32, 32))
    for i in range(count):
        data[i] += 3 * np.exp(-((x - (i % 4 - 2)) ** 2 + (y + 2) ** 2) / 12)
    header = bytearray(1024)
    struct.pack_into("<4i", header, 0, 32, 32, count, 2)
    struct.pack_into("<3i", header, 28, 32, 32, count)
    struct.pack_into("<6f", header, 40, 32, 32, count, 90, 90, 90)
    struct.pack_into("<3i", header, 64, 1, 2, 3)
    header[208:212], header[212:216] = b"MAP ", b"DD\0\0"
    path.write_bytes(header + data.astype("<f4").tobytes())


def _sources(directory, source_classes=4, pose=37):
    _write_stack(directory / "particles.mrcs")
    optics = """# version 30001
data_optics
loop_
_rlnOpticsGroupName #1
_rlnOpticsGroup #2
_rlnVoltage #3
_rlnSphericalAberration #4
_rlnAmplitudeContrast #5
_rlnImagePixelSize #6
_rlnImageSize #7
_rlnImageDimensionality #8
optics1 1 300 2.7 0.1 1 32 2

data_particles
loop_
_rlnImageName #1
_rlnOpticsGroup #2
_rlnClassNumber #3
_rlnAnglePsi #4
_rlnOriginXAngst #5
_rlnOriginYAngst #6
_rlnDefocusU #7
_rlnDefocusV #8
_rlnDefocusAngle #9
_rlnNormCorrection #10
"""
    patterns = [[0, 1, 2, 3], [0, 0, 2, 2], [1, 1, 3, 3]]
    if source_classes == 2:
        patterns = [[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for run in range(3):
        prefix = f"run{run + 1:03}_it025"
        (directory / f"{prefix}_optimiser.star").write_text(
            # rlnExperimentalDataStarFile is what relion_refine actually writes
            # (EMDL_OPTIMISER_DATA_STARFILE); rlnDataStarFile is not a registered
            # label, so RELION silently ignored it and the read below failed.
            f"data_optimiser_general\n"
            f"_rlnExperimentalDataStarFile {prefix}_data.star\n"
            f"_rlnModelStarFile {prefix}_model.star\n")
        # Deliberately omit model_classes and all upstream average images.
        (directory / f"{prefix}_model.star").write_text(
            f"data_model_general\n_rlnNrClasses {source_classes}\n_rlnReferenceDimensionality 2\n")
        rows = [f"{i+1:06}@particles.mrcs 1 {patterns[run][i//16]+1} {pose} 2 -3 15000 15000 0 1"
                for i in range(64)]
        if run == 1:
            rows.reverse()
        (directory / f"{prefix}_data.star").write_text(optics + "\n".join(rows) + "\n")


def _prepare(binaries, directory, classes, output="consensus"):
    _run([binaries["relion_class2d_consensus"], "--i", "run001_it025_optimiser.star",
          "--nr_runs", "3", "--K", str(classes), "--o", output], directory)
    assert not (directory / f"{output}_references.star").exists()
    rows = _loop(directory / f"{output}_data.star", "particles")
    assert len(rows) == 64
    assert all(float(row["rlnAnglePsi"]) == 0 for row in rows)
    return {row["rlnImageName"]: row for row in rows}


def _refine(binaries, directory, classes, output="run", data="consensus_data.star", prefix=None):
    command = prefix or [binaries["relion_refine"]]
    # relion_refine takes the directory part of --o and requires it to exist
    # (ml_optimiser.cpp: fn_out.beforeLastOf("/")), so a bare prefix is rejected.
    _run(command + ["--i", data, "--o", f"./{output}", "--K", str(classes),
          "--fix_classes", "--iter", "1", "--random_seed", "1", "--ctf",
          "--particle_diameter", "24", "--ini_high", "10", "--psi_step", "90",
          "--offset_range", "1", "--offset_step", "1", "--oversampling", "0",
          "--j", "1", "--dont_check_norm"], directory)


@pytest.mark.parametrize("source_classes,classes", [(4, 2), (4, 4), (2, 5)])
def test_fresh_consensus_and_continuation(tmp_path, consensus_binaries, source_classes, classes):
    _sources(tmp_path, source_classes)
    expected = _prepare(consensus_binaries, tmp_path, classes)
    _refine(consensus_binaries, tmp_path, classes)
    for row in _loop(tmp_path / "run_it001_data.star", "particles"):
        original = expected[row["rlnImageName"]]
        for label in ("rlnClassNumber", "rlnClass2DConsensusProbability", "rlnClass2DConsensusEntropy"):
            assert float(row[label]) == pytest.approx(float(original[label]), abs=1e-5)
    initial = _read_mrc(tmp_path / "run_it000_classes.mrcs")
    refined = _read_mrc(tmp_path / "run_it001_classes.mrcs")
    assert initial.shape == refined.shape == (classes, 32, 32)
    assert np.isfinite(initial).all() and np.isfinite(refined).all()
    assert not np.allclose(initial, refined)
    occupied = {int(row["rlnClassNumber"]) for row in expected.values()}
    for k in range(classes):
        if k + 1 not in occupied:
            assert np.count_nonzero(initial[k]) == 0
            assert np.count_nonzero(refined[k]) == 0

    # Fresh images are reproducible and independent of upstream fitted poses.
    _sources(tmp_path, source_classes, pose=149)
    _prepare(consensus_binaries, tmp_path, classes, "repeat")
    _refine(consensus_binaries, tmp_path, classes, "again", "repeat_data.star")
    assert np.allclose(initial, _read_mrc(tmp_path / "again_it000_classes.mrcs"), atol=1e-6)

    _run([consensus_binaries["relion_refine"], "--continue", "run_it001_optimiser.star",
          "--o", "./continued", "--iter", "2", "--j", "1"], tmp_path)
    assert not (tmp_path / "continued_it000_classes.mrcs").exists()
    for row in _loop(tmp_path / "continued_it002_data.star", "particles"):
        assert row["rlnClassNumber"] == expected[row["rlnImageName"]]["rlnClassNumber"]


@pytest.mark.parametrize("backend", ["mpi", "gpu"])
def test_fresh_initialization_backends(tmp_path, consensus_binaries, backend):
    if os.environ.get("RELION_TEST_" + backend.upper()) != "1":
        pytest.skip(f"Set RELION_TEST_{backend.upper()}=1 to exercise this backend")
    _sources(tmp_path)
    expected = _prepare(consensus_binaries, tmp_path, 3)
    _refine(consensus_binaries, tmp_path, 3)
    if backend == "mpi":
        prefix = ["mpirun", "-n", "2", str(consensus_binaries["directory"] / "relion_refine_mpi")]
    else:
        prefix = [consensus_binaries["relion_refine"], "--gpu", "0"]
    _refine(consensus_binaries, tmp_path, 3, backend, prefix=prefix)
    assert np.allclose(_read_mrc(tmp_path / "run_it000_classes.mrcs"),
                       _read_mrc(tmp_path / f"{backend}_it000_classes.mrcs"), atol=1e-5)
    for row in _loop(tmp_path / f"{backend}_it001_data.star", "particles"):
        assert row["rlnClassNumber"] == expected[row["rlnImageName"]]["rlnClassNumber"]
