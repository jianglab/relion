#!/usr/bin/env python3
"""
Integration test for the filament context prior in Class3D, with the options the Class3D job window
writes (--filament_context off / chain / filament, --filament_temper, --filament_gap,
--filament_floor, --filament_alpha).

The run must finish, report the mode and its filaments, switch the prior on after the first
iteration, and write a data.star with finite orientations and a class for every segment.
"""

import math

import numpy as np
import pytest

from fixtures import read_mrc
from test_filament_class3d import generate_helical_dataset, parse_star_particles


def class3d_cmd(relion_bin, paths, out, extra):
    return [
        "mpirun", "-n", "2", str(relion_bin / "relion_refine_mpi"),
        "--i", str(paths["helical_star"]), "--o", str(out / "run"),
        "--ref", str(paths["initial_model"]), "--K", "3", "--iter", "2",
        "--ini_high", "50", "--ctf", "--tau2_fudge", "4", "--particle_diameter", "25",
        "--flatten_solvent", "--zero_mask", "--oversampling", "1", "--healpix_order", "1",
        "--offset_range", "5", "--offset_step", "2", "--sym", "c1", "--norm", "--scale",
        "--j", "2", "--keep_all_iterations",
    ] + extra


@pytest.mark.integration
class TestFilamentContextClass3D:

    @pytest.mark.parametrize("mode,extra", [
        ("off", ["--filament_context", "off"]),
        ("chain", ["--filament_context", "chain", "--filament_temper", "0.5", "--filament_gap", "2",
                   "--filament_floor", "0.05"]),
        ("filament", ["--filament_context", "filament", "--filament_temper", "0.5", "--filament_gap", "1",
                      "--filament_floor", "0.02", "--filament_alpha", "2"]),
    ])
    def test_modes_run(self, test_data_dir, relion_bin, relion_runner, mode, extra):
        paths = generate_helical_dataset(test_data_dir / "helical_data", relion_bin, relion_runner,
                                         num_filaments=4, segments_per_filament=6, size=32, angpix=1.0)
        out = test_data_dir / ("class3d_filctx_" + mode)
        out.mkdir(parents=True, exist_ok=True)

        result = relion_runner(class3d_cmd(relion_bin, paths, out, extra), timeout=900)
        assert result.returncode == 0, f"Class3D failed:\n{result.stderr}"
        text = result.stdout or ""
        if mode == "off":
            assert "Filament context: " not in text
        else:
            assert f"Filament context: {mode}" in text, text[-2000:]
            assert "4 filaments" in text
            if mode == "chain":
                assert "probability that the next segment keeps its class" in text

        data = parse_star_particles(str(sorted(out.glob("run_it002_data.star"))[-1]),
                                    ["rlnClassNumber", "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"])
        assert len(data["rlnClassNumber"]) == paths["total_particles"]
        assert all(1 <= int(c) <= 3 for c in data["rlnClassNumber"])
        for col in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"):
            assert all(math.isfinite(float(v)) for v in data[col])

    def test_helical_class3d_auto_mask_delayed_gating(self, test_data_dir, relion_bin, relion_runner):
        paths = generate_helical_dataset(
            test_data_dir / "helical_data",
            relion_bin,
            relion_runner,
            num_filaments=4,
            segments_per_filament=6,
            size=32,
            angpix=1.0,
        )
        out = test_data_dir / "class3d_automask"
        out.mkdir(parents=True, exist_ok=True)

        cmd = [
            "mpirun", "-n", "2", str(relion_bin / "relion_refine_mpi"),
            "--i", str(paths["helical_star"]),
            "--o", str(out / "run"),
            "--ref", str(paths["initial_model"]),
            "--K", "2",
            "--iter", "7",
            "--ini_high", "15",
            "--ctf",
            "--tau2_fudge", "4",
            "--particle_diameter", "28",
            "--helix",
            "--helical_outer_diameter", "20",
            "--helical_twist_initial", "-1.1",
            "--helical_rise_initial", "4.75",
            "--helical_z_percentage", "0.3",
            "--auto_mask",
            "--auto_mask_start_res", "20",
            "--flatten_solvent",
            "--zero_mask",
            "--oversampling", "1",
            "--healpix_order", "1",
            "--offset_range", "3",
            "--offset_step", "2",
            "--sym", "c1",
            "--norm",
            "--scale",
            "--j", "2",
            "--pool", "5",
            "--random_seed", "7",
            "--keep_all_iterations",
        ]
        result = relion_runner(cmd, timeout=900)
        out_txt = (result.stdout or "") + "\n" + (result.stderr or "")
        assert result.returncode == 0, f"Class3D failed:\n{out_txt[-3000:]}"

        for it in range(1, 6):
            assert not list(out.glob(f"run_it{it:03d}_*_automask.mrc")), (
                f"Class3D auto_mask triggered too early at it{it:03d}"
            )

        late_masks = sorted(
            list(out.glob("run_it006_*_automask.mrc")) + list(out.glob("run_it007_*_automask.mrc"))
        )
        assert late_masks, f"Expected Class3D automask at it006/it007; stdout tail:\n{out_txt[-3000:]}"
        for p in late_masks:
            arr = np.asarray(read_mrc(str(p)), dtype=np.float64)
            assert arr.shape == (32, 32, 32)
            assert np.all(np.isfinite(arr))
            assert arr.min() >= -1e-6
            assert arr.max() <= 1.0 + 1e-6
            assert arr.max() > 0.5
