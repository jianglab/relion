#!/usr/bin/env python3
"""
--ini_high without a reference. In a reference-free classification (2D, or 3D de novo) the first
reconstruction is made from the particles themselves, in a box set by --ini_high. A resolution lower than the first
Fourier shell of the box (here 50 A in a 32 pixel box of 1 A) once gave a box of 0 pixels, and the
reconstruction crashed (a segmentation fault in 2D, an error exit in 3D de novo). It must run, and say what it did.

CPU only, so it runs on any machine.
"""

import pytest


def _run_class2d(test_data_dir, relion_bin, relion_runner, tag, ini_high):
    from fixtures import generate_test_dataset

    paths = generate_test_dataset(test_data_dir / "data_ih", relion_bin, relion_runner,
                                  num_particles=60, size=32, angpix=1.0)
    out = test_data_dir / tag
    out.mkdir()
    cmd = [str(relion_bin / "relion_refine"),
           "--i", paths["star_file"], "--o", str(out / "run"),
           "--K", "3", "--iter", "2", "--ctf", "--tau2_fudge", "2",
           "--particle_diameter", "25", "--flatten_solvent", "--zero_mask", "--oversampling", "1",
           "--psi_step", "20", "--offset_range", "5", "--offset_step", "2",
           "--norm", "--scale", "--j", "2", "--random_seed", "7", "--dont_check_norm",
           "--pool", "10", "--keep_all_iterations"]
    if ini_high is not None:
        cmd += ["--ini_high", str(ini_high)]
    r = relion_runner(cmd, timeout=900)
    return r, out


@pytest.mark.parametrize("ini_high", [50, 1000])
def test_class2d_ini_high_below_first_shell_does_not_crash(test_data_dir, relion_bin, relion_runner, ini_high):
    r, out = _run_class2d(test_data_dir, relion_bin, relion_runner, f"ih_{ini_high}", ini_high)
    assert r.returncode == 0, f"exit {r.returncode}:\n{r.stderr[-3000:]}"
    assert "--ini_high" in r.stderr and "WARNING" in r.stderr
    assert (out / "run_it002_model.star").exists()


def test_class2d_ini_high_usable_value_is_unchanged(test_data_dir, relion_bin, relion_runner):
    """A value the box can hold is used as given: no warning."""
    r, out = _run_class2d(test_data_dir, relion_bin, relion_runner, "ih_10", 10)
    assert r.returncode == 0, f"exit {r.returncode}:\n{r.stderr[-3000:]}"
    assert "--ini_high" not in r.stderr
    assert (out / "run_it002_model.star").exists()


@pytest.mark.parametrize("ini_high", [50, 1000])
def test_denovo_3d_ini_high_below_first_shell_does_not_crash(test_data_dir, relion_bin, relion_runner, ini_high):
    from fixtures import generate_test_dataset

    paths = generate_test_dataset(test_data_dir / "data_ih", relion_bin, relion_runner,
                                  num_particles=60, size=32, angpix=1.0)
    out = test_data_dir / f"ih3d_{ini_high}"
    out.mkdir()
    cmd = [str(relion_bin / "relion_refine"),
           "--i", paths["star_file"], "--o", str(out / "run"),
           "--denovo_3dref", "--grad", "--K", "1", "--iter", "2", "--ctf", "--particle_diameter", "25",
           "--flatten_solvent", "--zero_mask", "--sym", "C1", "--healpix_order", "1",
           "--offset_range", "3", "--offset_step", "2", "--norm", "--scale", "--j", "2", "--pool", "10",
           "--random_seed", "3", "--dont_check_norm", "--ini_high", str(ini_high)]
    r = relion_runner(cmd, timeout=900)
    assert r.returncode == 0, f"exit {r.returncode}:\n{r.stderr[-3000:]}"
    assert "--ini_high" in r.stderr and "WARNING" in r.stderr
