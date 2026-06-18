#!/usr/bin/env python3
"""
Regression test for OMP data-race / buffer-overflow crashes in
relion_reconstruct with dense s2 grids.

Triggered by --s2_ctf_oversampling_min 2 with multi-threaded (-j 2)
reconstruction.  The test verifies that both the Ewald and non-Ewald
paths complete without heap corruption or ASan violations.
"""

import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.integration
class TestReconstructCrash:
    """Regression tests for reconstruct crashes with dense s2 grids."""

    def _run_reconstruct(
        self, test_data_dir, relion_bin, relion_runner,
        fixtures_module, extra_args,
    ):
        """Shared helper: generate data and run relion_reconstruct."""
        paths = fixtures_module["generate_test_dataset"](
            str(test_data_dir), relion_bin, relion_runner,
            num_particles=64,
            size=32,
            angpix=1.0,
        )

        assert Path(paths["particles_mrc"]).exists()
        assert Path(paths["star_file"]).exists()

        output_dir = test_data_dir / "reconstruction"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "mpirun", "-np", "2",
            str(relion_bin / "relion_reconstruct_mpi"),
            "--i", str(test_data_dir / "particles.star"),
            "--o", str(output_dir / "recon_"),
            "--spatial_frequency_mode", "s2",
            "--angpix", "1.0",
            "--ctf",
            "--s2_ctf_oversampling_min", "2",
            "--j", "2",
        ] + extra_args

        env = os.environ.copy()
        env["ASAN_OPTIONS"] = "detect_leaks=0"
        result = relion_runner(cmd, timeout=300, env=env)
        assert result.returncode == 0, (
            f"relion_reconstruct failed (code {result.returncode}):\n{result.stderr}"
        )

        output_map = output_dir / "recon_.spi"
        assert output_map.exists(), "Output map not found"
        return output_map

    def _check_no_nan(self, spi_path):
        """Verify output map contains no NaN values."""
        with open(spi_path, "rb") as f:
            raw = f.read()
        head = np.frombuffer(raw[:1024], dtype=np.float32)
        nx, ny, nz = int(head[0]), int(head[1]), int(head[11])
        offset = int(head[21])
        volume = np.frombuffer(raw[offset:], dtype=np.float32).reshape((nx, ny, nz))
        assert not np.any(np.isnan(volume)), "Output map contains NaN values"

    def test_s2_dense_ewald(
        self, test_data_dir, relion_bin, fixtures_module, relion_runner,
    ):
        """Dense s2 grid with Ewald sphere correction (--ewald)."""
        output = self._run_reconstruct(
            test_data_dir, relion_bin, relion_runner,
            fixtures_module,
            extra_args=["--mask_diameter", "220", "--ewald", "--reverse_curvature"],
        )
        self._check_no_nan(output)

    def test_s2_dense_no_ewald(
        self, test_data_dir, relion_bin, fixtures_module, relion_runner,
    ):
        """Dense s2 grid without Ewald sphere correction."""
        self._run_reconstruct(
            test_data_dir, relion_bin, relion_runner,
            fixtures_module,
            extra_args=["--mask_diameter", "220"],
        )
