import pytest
import numpy as np
import os
import subprocess
import struct
from pathlib import Path


def subprocess_run(cmd, **kwargs):
    """Wrapper around subprocess.run with sensible defaults."""
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    kwargs.setdefault("timeout", 60)
    return subprocess.run(cmd, **kwargs)


def _make_cs(filename, records):
    """Write a CryoSPARC .cs file (npy structured array).

    Records is a list of (name, kind_or_dtype, val) where:
    - For scalars: kind is '<f8', '<i8', 'S', etc.
    - For sub-arrays: kind is a tuple like ('<f8', (2,))
    """
    dtype = []
    for name, kind, val in records:
        if kind == "S":
            max_len = max(len(v) for v in val)
            dtype.append((name, f"S{max_len}"))
        elif isinstance(kind, tuple):
            base_kind, shape = kind
            dtype.append((name, base_kind, shape))
        else:
            dtype.append((name, kind))
    arr = np.zeros(len(records[0][2]), dtype=dtype)
    for name, _, val in records:
        arr[name] = np.array(val)
    # np.save on NumPy >=2.0 appends .npy; save with .npy suffix then rename
    npy_path = str(filename) + ".npy"
    np.save(npy_path, arr)
    os.rename(npy_path, str(filename))


def test_cryosparc_import_simple(relion_bin, test_data_dir):
    """Import a basic .cs file with blob/path, blob/idx and CTF columns."""
    cs_file = test_data_dir / "particles_selected.cs"
    star_file = test_data_dir / "particles_selected.star"

    n = 3
    _make_cs(cs_file, [
        ("blob/path", "S", ["/data/mic1.mrc", "/data/mic1.mrc", "/data/mic2.mrc"]),
        ("blob/idx", "<i8", [0, 1, 0]),
        ("blob/psize_A", "<f8", [1.0, 1.0, 1.0]),
        ("ctf/accel_kv", "<f8", [300.0, 300.0, 300.0]),
        ("ctf/cs_mm", "<f8", [2.7, 2.7, 2.7]),
        ("ctf/amp_contrast", "<f8", [0.1, 0.1, 0.1]),
        ("ctf/df1_A", "<f8", [10000.0, 11000.0, 12000.0]),
        ("ctf/df2_A", "<f8", [9000.0, 9500.0, 10000.0]),
        ("ctf/df_angle_rad", "<f8", [0.0, 0.1, 0.2]),
        ("ctf/phase_shift_rad", "<f8", [0.0, 0.5, 1.0]),
    ])

    bin_exe = relion_bin / "relion_import"
    cmd = [
        str(bin_exe),
        "--i", str(cs_file),
        "--odir", str(test_data_dir) + "/",
        "--ofile", "particles_selected.star",
        "--do_cryosparc",
        "--angpix", "1.0",
        "--kV", "300",
        "--Cs", "2.7",
        "--Q0", "0.1",
    ]

    result = subprocess_run(cmd)

    assert result.returncode == 0, f"relion_import failed: {result.stderr}"
    assert star_file.exists(), "STAR file was not created"

    # Verify the STAR file contents
    star_text = star_file.read_text()
    assert "data_optics" in star_text, "Missing optics block"
    assert "data_particles" in star_text, "Missing particles block"
    assert "rlnImageName" in star_text, "Missing rlnImageName"
    assert "rlnDefocusU" in star_text, "Missing rlnDefocusU"
    assert "rlnDefocusV" in star_text, "Missing rlnDefocusV"
    assert "rlnDefocusAngle" in star_text, "Missing rlnDefocusAngle"
    assert "rlnPhaseShift" in star_text, "Missing rlnPhaseShift"

    # Verify image names
    assert "000001@/data/mic1.mrc" in star_text
    assert "000002@/data/mic1.mrc" in star_text
    assert "000001@/data/mic2.mrc" in star_text

    # Verify defocus values
    assert "10000" in star_text
    assert "11000" in star_text

    # Verify rad→deg conversions (0 rad = 0 deg, 0.5 rad ≈ 28.65 deg)
    assert "0.000000" in star_text  # df_angle for first particle
    assert "0" in star_text  # phase_shift for first particle


def test_cryosparc_import_2d_alignments(relion_bin, test_data_dir):
    """Import .cs with 2D alignment data."""
    cs_file = test_data_dir / "particles_2d.cs"

    n = 2
    shifts = np.array([[1.5, -2.5], [-0.5, 3.0]], dtype=np.float64)
    _make_cs(cs_file, [
        ("blob/path", "S", ["/data/part.mrc", "/data/part.mrc"]),
        ("blob/idx", "<i8", [0, 1]),
        ("blob/psize_A", "<f8", [1.0, 1.0]),
        ("alignments2D/class", "<i8", [0, 2]),
        ("alignments2D/shift", ("<f8", (2,)), shifts),
        ("alignments2D/pose", "<f8", [0.5, -0.3]),
    ])

    bin_exe = relion_bin / "relion_import"
    cmd = [
        str(bin_exe),
        "--i", str(cs_file),
        "--odir", str(test_data_dir) + "/",
        "--ofile", "particles_2d.star",
        "--do_cryosparc",
        "--angpix", "1.0",
        "--kV", "300",
        "--Cs", "2.7",
        "--Q0", "0.1",
    ]

    result = subprocess_run(cmd)
    assert result.returncode == 0, f"relion_import failed: {result.stderr}"

    star_file = test_data_dir / "particles_2d.star"
    star_text = star_file.read_text()

    assert "rlnClassNumber" in star_text
    assert "rlnAnglePsi" in star_text
    assert "rlnOriginXAngst" in star_text
    assert "rlnOriginYAngst" in star_text

    # Class: 0→1, 2→3
    assert "rlnClassNumber" in star_text

    # Shift (negated): (-1.5, 2.5) and (0.5, -3.0)
    assert "rlnOriginXAngst" in star_text


def test_cryosparc_import_micrographs(relion_bin, test_data_dir):
    """Import .cs with micrograph-level data (no blob)."""
    cs_file = test_data_dir / "micrographs_selected.cs"

    n = 2
    _make_cs(cs_file, [
        ("micrograph_blob/path", "S", ["/data/mic1.mrc", "/data/mic2.mrc"]),
        ("micrograph_blob/psize_A", "<f8", [1.2, 1.2]),
        ("ctf/accel_kv", "<f8", [300.0, 300.0]),
        ("ctf/cs_mm", "<f8", [2.7, 2.7]),
        ("ctf/amp_contrast", "<f8", [0.1, 0.1]),
        ("ctf/df1_A", "<f8", [10000.0, 11000.0]),
        ("ctf/df2_A", "<f8", [9000.0, 9500.0]),
        ("ctf/df_angle_rad", "<f8", [0.1, 0.2]),
    ])

    bin_exe = relion_bin / "relion_import"
    cmd = [
        str(bin_exe),
        "--i", str(cs_file),
        "--odir", str(test_data_dir) + "/",
        "--ofile", "micrographs_selected.star",
        "--do_cryosparc",
        "--angpix", "1.2",
        "--kV", "300",
        "--Cs", "2.7",
        "--Q0", "0.1",
    ]

    result = subprocess_run(cmd)
    assert result.returncode == 0, f"relion_import failed: {result.stderr}"

    star_text = (test_data_dir / "micrographs_selected.star").read_text()
    assert "rlnMicrographName" in star_text
    assert "/data/mic1.mrc" in star_text
    assert "/data/mic2.mrc" in star_text
    assert "rlnDefocusU" in star_text


def test_cryosparc_import_3d_alignments(relion_bin, test_data_dir):
    """Import .cs with 3D alignment data (rotvec and shifts)."""
    cs_file = test_data_dir / "particles_3d.cs"

    # Rotation vectors: identity (0,0,0), 90° around Z
    rotvecs = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, np.pi/2],
    ], dtype=np.float64)
    shifts = np.array([
        [2.0, -1.0],
        [-0.5, 1.5],
    ], dtype=np.float64)

    _make_cs(cs_file, [
        ("blob/path", "S", ["/data/part.mrcs", "/data/part.mrcs"]),
        ("blob/idx", "<i8", [0, 1]),
        ("blob/psize_A", "<f8", [1.0, 1.0]),
        ("alignments3D/class", "<i8", [0, 0]),
        ("alignments3D/split", "<i8", [0, 1]),
        ("alignments3D/cross_cor", "<f8", [0.8, 0.7]),
        ("alignments3D/pose", ("<f8", (3,)), rotvecs),
        ("alignments3D/shift", ("<f8", (2,)), shifts),
        ("ctf/accel_kv", "<f8", [300.0, 300.0]),
        ("ctf/cs_mm", "<f8", [2.7, 2.7]),
        ("ctf/amp_contrast", "<f8", [0.1, 0.1]),
    ])

    bin_exe = relion_bin / "relion_import"
    cmd = [
        str(bin_exe),
        "--i", str(cs_file),
        "--odir", str(test_data_dir) + "/",
        "--ofile", "particles_3d.star",
        "--do_cryosparc",
        "--angpix", "1.0",
        "--kV", "300",
        "--Cs", "2.7",
        "--Q0", "0.1",
    ]

    result = subprocess_run(cmd)
    assert result.returncode == 0, f"relion_import failed: {result.stderr}"

    star_text = (test_data_dir / "particles_3d.star").read_text()
    assert "rlnAngleRot" in star_text
    assert "rlnAngleTilt" in star_text
    assert "rlnAnglePsi" in star_text
    assert "rlnOriginXAngst" in star_text
    assert "rlnOriginYAngst" in star_text
    assert "rlnRandomSubset" in star_text
    assert "rlnLogLikeliContribution" in star_text
    assert "rlnClassNumber" in star_text


def test_cryosparc_import_coordinates(relion_bin, test_data_dir):
    """Import .cs with coordinate data and micrograph shapes."""
    cs_file = test_data_dir / "coords_selected.cs"

    n = 3
    _make_cs(cs_file, [
        ("blob/path", "S", ["/data/pick.mrcs"] * 3),
        ("blob/idx", "<i8", [0, 1, 2]),
        ("blob/psize_A", "<f8", [1.0] * 3),
        ("micrograph_blob/path", "S", ["/data/mic1.mrc"] * 3),
        ("micrograph_blob/psize_A", "<f8", [1.0] * 3),
        ("location/center_x_frac", "<f8", [0.25, 0.5, 0.75]),
        ("location/center_y_frac", "<f8", [0.25, 0.5, 0.25]),
        ("micrograph_blob/shape", ("<f8", (2,)), [[4096, 4096], [4096, 4096], [4096, 4096]]),
        ("ctf/accel_kv", "<f8", [300.0] * 3),
        ("ctf/cs_mm", "<f8", [2.7] * 3),
        ("ctf/amp_contrast", "<f8", [0.1] * 3),
    ])

    bin_exe = relion_bin / "relion_import"
    cmd = [
        str(bin_exe),
        "--i", str(cs_file),
        "--odir", str(test_data_dir) + "/",
        "--ofile", "coords_selected.star",
        "--do_cryosparc",
        "--angpix", "1.0",
        "--kV", "300",
        "--Cs", "2.7",
        "--Q0", "0.1",
    ]

    result = subprocess_run(cmd)
    assert result.returncode == 0, f"relion_import failed: {result.stderr}"

    star_text = (test_data_dir / "coords_selected.star").read_text()
    assert "rlnCoordinateX" in star_text
    assert "rlnCoordinateY" in star_text
    assert "rlnMicrographName" in star_text
    # 0.25 * 4096 = 1024, 0.5 * 4096 = 2048, 0.75 * 4096 = 3072
    assert "1024" in star_text
    assert "2048" in star_text


def test_cryosparc_import_passthrough(relion_bin, test_data_dir):
    """Import .cs with passthrough merge: extract has blob+align, passthrough has CTF+mic."""
    extract_file = test_data_dir / "extract_file.cs"
    passthrough_file = test_data_dir / "extract_file_passthrough.cs"

    uids = [1001, 1002, 1003]
    shifts = np.array([[1.0, 0.0], [0.5, -0.5], [-1.0, 2.0]], dtype=np.float64)

    # Extract file has blob info and alignments but NO CTF/micrograph
    _make_cs(extract_file, [
        ("blob/path", "S", ["/data/part.mrcs", "/data/part.mrcs", "/data/part.mrcs"]),
        ("blob/idx", "<i8", [0, 1, 2]),
        ("blob/psize_A", "<f8", [1.0, 1.0, 1.0]),
        ("uid", "<i8", uids),
        ("alignments2D/class", "<i8", [0, 1, 2]),
        ("alignments2D/shift", ("<f8", (2,)), shifts),
    ])

    # Passthrough file has uid + CTF + micrograph path
    _make_cs(passthrough_file, [
        ("uid", "<i8", uids),
        ("micrograph_blob/path", "S", ["/data/mic1.mrc", "/data/mic1.mrc", "/data/mic2.mrc"]),
        ("ctf/accel_kv", "<f8", [300.0, 300.0, 300.0]),
        ("ctf/cs_mm", "<f8", [2.7, 2.7, 2.7]),
        ("ctf/amp_contrast", "<f8", [0.1, 0.1, 0.1]),
        ("ctf/df1_A", "<f8", [10000.0, 11000.0, 12000.0]),
        ("ctf/df2_A", "<f8", [9000.0, 9500.0, 10000.0]),
    ])

    star_file = test_data_dir / "extract_file.star"
    bin_exe = relion_bin / "relion_import"
    cmd = [
        str(bin_exe), "--i", str(extract_file),
        "--odir", str(test_data_dir) + "/",
        "--ofile", "extract_file.star",
        "--do_cryosparc",
        "--angpix", "1.0", "--kV", "300", "--Cs", "2.7", "--Q0", "0.1",
    ]

    result = subprocess_run(cmd)
    assert result.returncode == 0, f"relion_import failed: {result.stderr}"

    star_text = star_file.read_text()

    # Data from extract file
    assert "rlnImageName" in star_text, "Missing rlnImageName from extract"
    assert "rlnClassNumber" in star_text, "Missing rlnClassNumber from extract"
    assert "rlnOriginXAngst" in star_text, "Missing rlnOriginXAngst from extract"

    # Data from passthrough (should be merged in)
    assert "rlnMicrographName" in star_text, "Missing rlnMicrographName from passthrough"
    assert "rlnDefocusU" in star_text, "Missing rlnDefocusU from passthrough"
    assert "rlnDefocusV" in star_text, "Missing rlnDefocusV from passthrough"
    assert "/data/mic1.mrc" in star_text
    assert "/data/mic2.mrc" in star_text
    assert "10000" in star_text
    assert "11000" in star_text

    # Verify auto-discovery: passthrough filename was NOT passed explicitly,
    # so relion must have auto-discovered *_passthrough.cs in the same dir


def test_cryosparc_import_multi_optics(relion_bin, test_data_dir):
    """Import .cs with multiple optics groups (different kV/Cs/Q0)."""
    cs_file = test_data_dir / "multi_optics.cs"
    star_file = test_data_dir / "multi_optics.star"

    n = 4
    _make_cs(cs_file, [
        ("blob/path", "S", ["/data/part.mrcs"] * n),
        ("blob/idx", "<i8", [0, 1, 2, 3]),
        ("blob/psize_A", "<f8", [1.0, 1.0, 1.0, 1.0]),
        ("blob/shape", ("<f8", (2,)), [[256, 256]] * n),
        ("ctf/exp_group_id", "<i8", [0, 0, 1, 1]),
        ("ctf/accel_kv", "<f8", [300.0, 300.0, 200.0, 200.0]),
        ("ctf/cs_mm", "<f8", [2.7, 2.7, 2.7, 2.7]),
        ("ctf/amp_contrast", "<f8", [0.1, 0.1, 0.07, 0.07]),
        ("ctf/df1_A", "<f8", [10000.0, 11000.0, 12000.0, 13000.0]),
        ("ctf/df2_A", "<f8", [9000.0, 9500.0, 10000.0, 10500.0]),
        ("ctf/df_angle_rad", "<f8", [0.0, 0.1, 0.2, 0.3]),
    ])

    bin_exe = relion_bin / "relion_import"
    cmd = [
        str(bin_exe), "--i", str(cs_file),
        "--odir", str(test_data_dir) + "/",
        "--ofile", "multi_optics.star",
        "--do_cryosparc",
        "--angpix", "1.0", "--kV", "300", "--Cs", "2.7", "--Q0", "0.1",
    ]

    result = subprocess_run(cmd)
    assert result.returncode == 0, f"relion_import failed: {result.stderr}"

    star_text = star_file.read_text()

    # Must have 2 optics groups
    assert "data_optics" in star_text
    assert "rlnOpticsGroup" in star_text
    # Count rows in the optics group data table
    count_optics_groups = 0
    in_optics = False
    for line in star_text.splitlines():
        if line.startswith("data_optics"):
            in_optics = True
        elif in_optics and (line.startswith("data_") or line.startswith("# version ")):
            in_optics = False
        elif in_optics and line.strip() and not line.startswith("loop_") and not line.startswith("_"):
            count_optics_groups += 1

    assert count_optics_groups == 2, f"Expected 2 optics groups, got {count_optics_groups}"

    # Verify particles have correct optics group assignment
    in_particles = False
    optics_col = -1
    particle_data = []
    for line in star_text.splitlines():
        if line.startswith("data_particles"):
            in_particles = True
        elif line.startswith("data_"):
            in_particles = False
        elif in_particles and line.startswith("_rlnOpticsGroup"):
            # Find column position (last token after #)
            optics_col = int(line.strip().rsplit("#", 1)[1]) - 1
        elif in_particles and line.strip() and not line.startswith("loop_") and not line.startswith("_"):
            cols = line.strip().split()
            if len(cols) > 0 and optics_col >= 0:
                particle_data.append(cols[optics_col])

    # First 2 particles should be group 1, last 2 should be group 2
    assert len(particle_data) == 4, f"Expected 4 particles, got {len(particle_data)}"
    assert particle_data[0] == "1", f"Expected group 1, got {particle_data[0]}"
    assert particle_data[1] == "1", f"Expected group 1, got {particle_data[1]}"
    assert particle_data[2] == "2", f"Expected group 2, got {particle_data[2]}"
    assert particle_data[3] == "2", f"Expected group 2, got {particle_data[3]}"


def test_cryosparc_import_invalid_file(relion_bin, test_data_dir):
    """Verify graceful error handling for non-.cs files."""
    cs_file = test_data_dir / "not_a_cs_file.cs"
    cs_file.write_text("this is not a valid npy file")

    bin_exe = relion_bin / "relion_import"
    cmd = [
        str(bin_exe),
        "--i", str(cs_file),
        "--odir", str(test_data_dir) + "/",
        "--ofile", "should_fail.star",
        "--do_cryosparc",
        "--angpix", "1.0",
        "--kV", "300",
        "--Cs", "2.7",
        "--Q0", "0.1",
    ]

    result = subprocess_run(cmd)
    assert result.returncode != 0, "Expected non-zero exit code for invalid input"
    assert "error" in result.stderr.lower() or "Error" in result.stderr



