#!/usr/bin/env python3
"""
Integration test for MPI asynchronous image prefetching.

The ``MpiAsyncImagePrefetcher`` overlaps inter-job I/O with computation in
the MPI follower loop. This test runs ``relion_refine_mpi`` with synthetic
data and verifies that the prefetch-enabled pipeline completes correctly.
"""

from pathlib import Path

import pytest


_COMMON_ARGS = [
    "--ini_high", "40",
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
]


@pytest.mark.integration
class TestMpiPrefetch:
    """Test suite for MPI image prefetching."""

    def test_refine_with_prefetch(self, test_data_dir, relion_bin, fixtures_module, relion_runner):
        """Run relion_refine_mpi with 2 MPI processes and verify the prefetch path."""
        paths = fixtures_module["generate_test_dataset"](
            str(test_data_dir), relion_bin, relion_runner,
            num_particles=12,
            size=32,
            angpix=1.0,
        )

        assert Path(paths["particles_mrc"]).exists()
        assert Path(paths["star_file"]).exists()

        output_dir = test_data_dir / "prefetch"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "mpirun", "-n", "2",
            str(relion_bin / "relion_refine_mpi"),
            "--i", str(test_data_dir / "particles.star"),
            "--o", str(output_dir / "run"),
            "--ref", str(test_data_dir / "initial_model.mrc"),
            "--K", "1",
            "--iter", "2",
        ] + _COMMON_ARGS

        result = relion_runner(cmd, timeout=600)
        assert result.returncode == 0, f"MPI prefetch test failed:\n{result.stderr}"

        assert (output_dir / "run_it002_model.star").exists(), \
            "Model STAR file from iteration 2 not found"

    def test_refine_with_prefetch_more_particles(self, test_data_dir, relion_bin, fixtures_module, relion_runner):
        """Run with more particles to exercise multiple prefetch cycles."""
        paths = fixtures_module["generate_test_dataset"](
            str(test_data_dir), relion_bin, relion_runner,
            num_particles=24,
            size=32,
            angpix=1.0,
        )

        assert Path(paths["particles_mrc"]).exists()

        output_dir = test_data_dir / "prefetch_more"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "mpirun", "-n", "3",
            str(relion_bin / "relion_refine_mpi"),
            "--i", str(test_data_dir / "particles.star"),
            "--o", str(output_dir / "run"),
            "--ref", str(test_data_dir / "initial_model.mrc"),
            "--K", "1",
            "--iter", "3",
        ] + _COMMON_ARGS

        result = relion_runner(cmd, timeout=600)
        assert result.returncode == 0, f"MPI prefetch (24 particles) test failed:\n{result.stderr}"

        assert (output_dir / "run_it003_model.star").exists()
        assert (output_dir / "run_it003_optimiser.star").exists()
