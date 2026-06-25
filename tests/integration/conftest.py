#!/usr/bin/env python3
"""
Pytest configuration and fixtures for RELION integration tests.

Provides shared fixtures and configuration for testing RELION programs
with s2 spatial frequency mode.
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture
def test_data_dir():
    """Create a temporary directory for test data."""
    tmpdir = tempfile.mkdtemp(prefix="relion_s2_test_")
    yield Path(tmpdir)
    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def relion_bin():
    """Get path to RELION binaries."""
    if "RELION_BIN_DIR" in os.environ:
        relion_dir = Path(os.environ["RELION_BIN_DIR"])
    else:
        # Default fallback locations relative to this script
        repo_root = Path(__file__).resolve().parent.parent.parent
        print(f"repo_root: {repo_root}")
        candidates = [
            repo_root.parent / "relion-build" / "bin",
            repo_root.parent / f"{repo_root.name}-build" / "bin",
            repo_root / "build" / "bin",
        ]
        
        relion_dir = None
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir() and any(f.stat().st_mode & 0o111 for f in candidate.iterdir() if f.is_file()):
                relion_dir = candidate
                break

    assert relion_dir is not None and relion_dir.exists(), (
        f"RELION binary directory not found in any of: {[str(c) for c in candidates]}\n"
        "You can specify it by setting the RELION_BIN_DIR environment variable."
    )
    return relion_dir


@pytest.fixture
def fixtures_module():
    """Import fixtures module for test data generation."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures import (
        generate_test_dataset,
        write_postprocess_star,
    )
    return {
        "generate_test_dataset": generate_test_dataset,
        "write_postprocess_star": write_postprocess_star,
    }


@pytest.fixture
def validate_module():
    """Import validation module for output checking."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from validate import (
        load_star_file,
        validate_reconstruction_output,
        validate_class3d_output,
        validate_ctf_refine_output
    )
    return {
        "load_star_file": load_star_file,
        "validate_reconstruction_output": validate_reconstruction_output,
        "validate_class3d_output": validate_class3d_output,
        "validate_ctf_refine_output": validate_ctf_refine_output
    }


def run_relion_program(cmd, timeout=300, capture=True, env=None):
    """
    Execute a RELION program.

    Args:
        cmd: List of command arguments (first element should be binary name)
        timeout: Maximum execution time in seconds
        capture: Whether to capture stdout/stderr
        env: Optional environment dict (defaults to inherited)

    Returns:
        CompletedProcess with returncode, stdout, stderr
    """
    kwargs = {}
    if capture:
        kwargs["capture_output"] = True
    if env is not None:
        kwargs["env"] = env
    result = subprocess.run(cmd, text=True, timeout=timeout, **kwargs)
    return result


def run_relion_program_mpi(cmd, timeout=300, capture=True, np=4):
    """
    Execute a RELION program via mpirun.

    Looks for an ``_mpi`` variant of the binary; if not found, runs the
    non-MPI binary directly with ``mpirun -np 1`` (single-process MPI).

    Args:
        cmd: List of command arguments (first element should be binary name)
        timeout: Maximum execution time in seconds
        capture: Whether to capture stdout/stderr
        np: Number of MPI processes (default 4)

    Returns:
        CompletedProcess with returncode, stdout, stderr
    """
    exe = Path(cmd[0])
    mpi_exe = exe.parent / f"{exe.stem}_mpi"
    if not mpi_exe.exists():
        mpi_exe = exe
    mpi_cmd = ["mpirun", "-np", str(np), str(mpi_exe)] + cmd[1:]
    if capture:
        result = subprocess.run(mpi_cmd, capture_output=True, text=True, timeout=timeout)
    else:
        result = subprocess.run(mpi_cmd, timeout=timeout)
    return result


@pytest.fixture
def relion_runner():
    """Fixture providing RELION program runner (single-process)."""
    return run_relion_program


@pytest.fixture
def relion_runner_mpi():
    """Fixture providing RELION program runner via mpirun (4 processes by default)."""
    return run_relion_program_mpi
