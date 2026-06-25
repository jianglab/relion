#!/usr/bin/env python3
"""
Integration test for --keep_full_filaments in Class3D.

Verifies that all helical segments from the same filament (same micrograph
+ same helical tube ID) are assigned to the same 3D class in the output
STAR file after Class3D with --keep_full_filaments.
"""

import os
import struct
import numpy as np
from pathlib import Path

import pytest


def write_helical_star(output_path, particles_file, particles_dir,
                       num_filaments=6, segments_per_filament=5,
                       size=32, angpix=1.0, seed=42):
    """
    Write a RELION5 STAR file with helical filament metadata.
    Follows the exact format of write_custom_star from fixtures.py.
    """
    rng = np.random.default_rng(seed)

    optics_block = """\

# version 30001

data_optics

loop_
_rlnOpticsGroupName #1
_rlnOpticsGroup #2
_rlnMicrographOriginalPixelSize #3
_rlnVoltage #4
_rlnSphericalAberration #5
_rlnAmplitudeContrast #6
_rlnImagePixelSize #7
_rlnImageSize #8
_rlnImageDimensionality #9
opticsGroup1\t1\t{angpix:.6f}\t300.000000\t2.700000\t0.100000\t{angpix:.6f}\t{size}\t2
""".format(angpix=angpix, size=size)

    particles_header = """\

# version 30001

data_particles

loop_
_rlnImageName #1
_rlnMicrographName #2
_rlnOpticsGroup #3
_rlnHelicalTubeID #4
_rlnDefocusU #5
_rlnDefocusV #6
_rlnDefocusAngle #7
_rlnCtfFigureOfMerit #8
_rlnAngleRot #9
_rlnAngleTilt #10
_rlnAnglePsi #11
_rlnOriginXAngst #12
_rlnOriginYAngst #13
_rlnClassNumber #14
_rlnNormCorrection #15
_rlnRandomSubset #16
_rlnGroupNumber #17
"""

    rows = []
    total_particles = num_filaments * segments_per_filament
    particle_idx = 0

    for fil_id in range(num_filaments):
        mic_name = f"micrograph_{fil_id:03d}.mrc"
        tube_id = fil_id + 1  # 1-indexed

        for seg in range(segments_per_filament):
            d_u = rng.uniform(12000, 18000)
            d_v = rng.uniform(12000, 18000)
            rot = rng.uniform(0, 360)
            tilt = np.arccos(rng.uniform(-1, 1)) * 180.0 / np.pi
            psi = rng.uniform(0, 360)
            ox = rng.uniform(-2, 2)
            oy = rng.uniform(-2, 2)
            random_subset = (particle_idx % 2) + 1

            row = (
                f"{particle_idx + 1:06d}@{particles_file}\t"
                f"{mic_name}\t"
                f"1\t"
                f"{tube_id}\t"
                f"{d_u:.2f}\t"
                f"{d_v:.2f}\t"
                f"0.00\t"
                f"0.100000\t"
                f"{rot:.6f}\t"
                f"{tilt:.6f}\t"
                f"{psi:.6f}\t"
                f"{ox:.6f}\t"
                f"{oy:.6f}\t"
                f"{(particle_idx % 3) + 1}\t"
                f"1.000000\t"
                f"{random_subset}\t"
                f"1"
            )
            rows.append(row)
            particle_idx += 1

    with open(output_path, 'w') as f:
        f.write(optics_block)
        f.write(particles_header)
        f.write("\n".join(rows) + "\n")

    return total_particles


def generate_helical_dataset(test_dir, relion_bin, relion_runner,
                             num_filaments=6, segments_per_filament=5,
                             size=32, angpix=1.0):
    """
    Generate a synthetic helical dataset for testing --keep_full_filaments.

    Uses generate_test_dataset to create particle images, then writes a
    STAR file with helical metadata (rlnMicrographName, rlnHelicalTubeID).

    Returns dict of paths and metadata.
    """
    os.makedirs(test_dir, exist_ok=True)

    from fixtures import generate_test_dataset

    total_particles = num_filaments * segments_per_filament

    # Generate base dataset (particles + initial model)
    paths = generate_test_dataset(
        str(test_dir), relion_bin, relion_runner,
        num_particles=total_particles,
        size=size,
        angpix=angpix,
    )

    particles_mrc = paths["particles_mrc"]
    initial_model = paths["initial_model"]

    # Write helical STAR file with filament metadata
    helical_star = test_dir / "particles.star"
    write_helical_star(
        str(helical_star), str(particles_mrc), str(test_dir),
        num_filaments=num_filaments,
        segments_per_filament=segments_per_filament,
        size=size, angpix=angpix,
    )

    return {
        "helical_star": helical_star,
        "initial_model": Path(initial_model),
        "total_particles": total_particles,
        "num_filaments": num_filaments,
        "segments_per_filament": segments_per_filament,
    }


def parse_star_particles(star_path, columns):
    """Parse a RELION STAR file and extract named columns from data_particles."""
    with open(star_path) as f:
        lines = f.readlines()

    in_particles = False
    col_index = {}
    data_rows = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("data_") and "particles" in stripped:
            in_particles = True
            continue
        if not in_particles:
            continue
        if stripped.startswith("loop_"):
            col_index = {}
            continue
        if stripped.startswith("_"):
            parts = stripped.split()
            if len(parts) >= 2:
                col_name = parts[0].lstrip("_")
                col_num = int(parts[1].lstrip("#")) - 1
                col_index[col_name] = col_num
            continue
        if stripped:
            parts = stripped.split()
            if parts and len(parts) >= len(col_index):
                data_rows.append(parts)

    result = {}
    for col in columns:
        if col not in col_index:
            raise KeyError(f"Column '{col}' not found. Available: {list(col_index.keys())}")
        idx = col_index[col]
        result[col] = [row[idx] for row in data_rows]
    return result


@pytest.mark.integration
class TestFilamentClass3D:
    """Test that --keep_full_filaments groups segments by filament."""

    def test_filament_consistency_after_class3d(self, test_data_dir, relion_bin,
                                                relion_runner):
        """Run Class3D with --keep_full_filaments and verify all segments
        from the same filament end up in the same class."""
        paths = generate_helical_dataset(
            test_data_dir / "helical_data",
            relion_bin, relion_runner,
            num_filaments=4,
            segments_per_filament=6,
            size=32,
            angpix=1.0,
        )

        output_dir = test_data_dir / "class3d_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run Class3D with --keep_full_filaments
        cmd = [
            "mpirun", "-n", "2",
            str(relion_bin / "relion_refine_mpi"),
            "--i", str(paths["helical_star"]),
            "--o", str(output_dir / "run"),
            "--ref", str(paths["initial_model"]),
            "--K", "3",
            "--iter", "5",
            "--ini_high", "50",
            "--ctf",
            "--tau2_fudge", "4",
            "--particle_diameter", "25",
            "--flatten_solvent",
            "--zero_mask",
            "--oversampling", "1",
            "--healpix_order", "1",
            "--offset_range", "5",
            "--offset_step", "2",
            "--sym", "c1",
            "--norm",
            "--scale",
            "--j", "2",
            "--spatial_frequency_mode", "s",
            "--keep_full_filaments",
        ]

        result = relion_runner(cmd, timeout=600)
        assert result.returncode == 0, f"Class3D failed:\n{result.stderr}"

        # Find the last iteration data.star
        data_stars = sorted(output_dir.glob("run_it*_data.star"))
        assert len(data_stars) > 0, "No data.star output found"
        last_data_star = data_stars[-1]

        # Parse the output STAR file
        data = parse_star_particles(str(last_data_star),
                                    ["rlnMicrographName", "rlnHelicalTubeID",
                                     "rlnClassNumber"])

        # Group particles by filament (micrograph + tube ID)
        filaments = {}
        for i in range(len(data["rlnClassNumber"])):
            mic = data["rlnMicrographName"][i]
            tube = data["rlnHelicalTubeID"][i]
            cls = data["rlnClassNumber"][i]
            key = (mic, tube)
            if key not in filaments:
                filaments[key] = []
            filaments[key].append(cls)

        # Verify: every filament has all particles in the same class
        inconsistent = 0
        for key, classes in filaments.items():
            if len(set(classes)) > 1:
                inconsistent += 1

        assert inconsistent == 0, (
            f"{inconsistent} filaments have mixed classes! "
            f"Total filaments: {len(filaments)}, "
            f"Total particles: {len(data['rlnClassNumber'])}"
        )
