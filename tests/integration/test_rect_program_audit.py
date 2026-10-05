#!/usr/bin/env python3
"""
Programs that do not support rectangular particle images (documentation/rectangular_particles_limits.md,
item 15) must stop with a clear message naming "rectangular particle images" and the program, and
must not crash or write silently wrong output.

The inputs are a rectangular optics group (rlnImageSizeX != rlnImageSizeY), a rectangular .mrcs
stack and a cuboid map (nx = ny < nz).
"""

import shutil

import numpy as np
import pytest

from fixtures import read_mrc, write_mrc, write_volume_mrc, write_custom_star

N, NY_RECT, NPART = 32, 20, 6


@pytest.fixture
def audit(test_data_dir):
    d = test_data_dir
    rng = np.random.default_rng(3)
    cube = rng.normal(size=(N, N, N)).astype(np.float32)
    cuboid = cube[:, 6:6 + NY_RECT, 6:6 + NY_RECT].copy()    # long along z
    write_volume_mrc(cube, str(d / "cube.mrc"))
    write_volume_mrc((cube > 0).astype(np.float32), str(d / "cube_mask.mrc"))
    write_volume_mrc(cuboid, str(d / "cuboid.mrc"))
    write_volume_mrc((cuboid > 0).astype(np.float32), str(d / "cuboid_mask.mrc"))

    angles = (rng.uniform(0, 360, NPART), rng.uniform(0, 180, NPART), rng.uniform(0, 360, NPART))
    write_custom_star(NPART, str(d / "square.star"), "x.mrcs", *angles, size=N, angpix=1.0)
    write_mrc(rng.normal(size=(NPART, NY_RECT, N)).astype(np.float32), str(d / "rect.mrcs"))
    text = (d / "square.star").read_text()
    text = text.replace("_rlnImageSize #8\n_rlnImageDimensionality #9",
                        "_rlnImageSizeX #8\n_rlnImageSizeY #9\n_rlnImageDimensionality #10")
    text = text.replace(f"\t{N}\t2\n", f"\t{N}\t{NY_RECT}\t2\n", 1)
    text = text.replace("@x.mrcs", f"@{d / 'rect.mrcs'}")
    (d / "rect.star").write_text(text)
    return d


def refused(result, program):
    out = result.stdout + result.stderr
    assert result.returncode != 0, out
    assert "rectangular particle images" in out, out
    assert program in out, out
    assert "Segmentation" not in out and "core dumped" not in out, out


def run(relion_bin, relion_runner, program, *args):
    return relion_runner([str(relion_bin / program)] + [str(a) for a in args], timeout=300)


@pytest.mark.integration
@pytest.mark.parametrize("option", [["--lowpass", "5"], ["--highpass", "20"], ["--bfactor", "-50"], ["--LoG", "10"],
                                    ["--rescale_angpix", "2"], ["--new_box", "16"], ["--power_image"], ["--guinier"]])
def test_image_handler_refuses_fourier_operations_on_a_cuboid(audit, relion_bin, relion_runner, option):
    r = run(relion_bin, relion_runner, "relion_image_handler", "--i", audit / "cuboid.mrc",
            "--o", audit / "out.mrc", "--angpix", "1", *option)
    refused(r, "relion_image_handler")


@pytest.mark.integration
def test_image_handler_still_flips_and_symmetrises_a_cuboid(audit, relion_bin, relion_runner):
    r = run(relion_bin, relion_runner, "relion_image_handler", "--i", audit / "cuboid.mrc",
            "--o", audit / "flipped.mrc", "--flipZ")
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.integration
def test_image_handler_refuses_to_resize_a_rectangular_star_file(audit, relion_bin, relion_runner):
    r = run(relion_bin, relion_runner, "relion_image_handler", "--i", audit / "rect.star",
            "--o", "resized", "--new_box", "16")
    refused(r, "relion_image_handler")


@pytest.mark.integration
def test_postprocess_refuses_cuboid_half_maps(audit, relion_bin, relion_runner):
    (audit / "pp").mkdir(exist_ok=True)
    r = run(relion_bin, relion_runner, "relion_postprocess", "--i", audit / "cuboid.mrc", "--i2", audit / "cuboid.mrc",
            "--o", str(audit / "pp") + "/", "--angpix", "1")
    refused(r, "relion_postprocess")


@pytest.mark.integration
def test_project_refuses_a_cuboid_map(audit, relion_bin, relion_runner):
    r = run(relion_bin, relion_runner, "relion_project", "--i", audit / "cuboid.mrc", "--ang", audit / "rect.star",
            "--o", audit / "proj", "--angpix", "1")
    refused(r, "relion_project")


@pytest.mark.integration
def test_align_symmetry_refuses_a_cuboid_map(audit, relion_bin, relion_runner):
    r = run(relion_bin, relion_runner, "relion_align_symmetry", "--i", audit / "cuboid.mrc", "--sym", "C2",
            "--o", audit / "aligned.mrc", "--angpix", "1")
    refused(r, "relion_align_symmetry")


@pytest.mark.integration
def test_autopick_refuses_rectangular_references(audit, relion_bin, relion_runner):
    write_mrc(np.random.default_rng(1).normal(size=(256, 256)).astype(np.float32), str(audit / "mic.mrc"))
    (audit / "mics.star").write_text(
        "data_optics\n\nloop_\n_rlnOpticsGroup #1\n_rlnOpticsGroupName #2\n_rlnMicrographPixelSize #3\n"
        "_rlnVoltage #4\n_rlnSphericalAberration #5\n_rlnAmplitudeContrast #6\n1\topticsGroup1\t1.0\t300\t2.7\t0.1\n\n\n"
        f"data_micrographs\n\nloop_\n_rlnMicrographName #1\n_rlnOpticsGroup #2\n{audit / 'mic.mrc'}\t1\n")
    (audit / "ap").mkdir(exist_ok=True)
    r = run(relion_bin, relion_runner, "relion_autopick", "--i", audit / "mics.star", "--odir", str(audit / "ap") + "/",
            "--ref", audit / "rect.mrcs", "--angpix", "1", "--particle_diameter", "18")
    refused(r, "relion_autopick")


@pytest.mark.integration
def test_ctf_refine_refuses_rectangular_optics(audit, relion_bin, relion_runner):
    cube = read_mrc(audit / "cube.mrc")
    noise = np.random.default_rng(8).normal(size=cube.shape)
    write_volume_mrc(cube + noise, str(audit / "r_half1_class001.mrc"))
    write_volume_mrc(cube - noise, str(audit / "r_half2_class001.mrc"))
    (audit / "pp").mkdir(exist_ok=True)
    r = run(relion_bin, relion_runner, "relion_postprocess", "--i", audit / "r_half1_class001.mrc",
            "--mask", audit / "cube_mask.mrc", "--o", audit / "pp" / "post", "--angpix", "1")
    assert r.returncode == 0, r.stdout + r.stderr
    (audit / "cr").mkdir(exist_ok=True)
    r = run(relion_bin, relion_runner, "relion_ctf_refine", "--i", audit / "rect.star", "--f", audit / "pp" / "post.star",
            "--o", str(audit / "cr") + "/", "--fit_defocus")
    refused(r, "this program")


@pytest.mark.integration
def test_refine_refuses_solvent_corrected_fsc(audit, relion_bin, relion_runner):
    if shutil.which("mpirun") is None or not (relion_bin / "relion_refine_mpi").exists():
        pytest.skip("auto-refinement needs relion_refine_mpi and mpirun")
    (audit / "rs").mkdir(exist_ok=True)
    r = relion_runner(["mpirun", "-n", "3", "--oversubscribe", str(relion_bin / "relion_refine_mpi"),
                       "--o", str(audit / "rs" / "r"), "--i", str(audit / "rect.star"), "--ref", str(audit / "cuboid.mrc"),
                       "--auto_refine", "--split_random_halves", "--solvent_correct_fsc", "--particle_diameter", "18",
                       "--angpix", "1", "--ctf", "--j", "1", "--sym", "C1", "--flatten_solvent", "--zero_mask"],
                      timeout=300)
    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "rectangular images are not supported together with --solvent_correct_fsc" in out, out
