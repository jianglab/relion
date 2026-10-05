#!/usr/bin/env python3
"""
relion_reconstruct with rectangular particle images (documentation/rectangular_particles.md).

A compact synthetic object is projected into square images, which are then cropped
to a rectangle. As the object lies well inside the crop, the rectangular images are
exact projections, so the cuboid reconstruction has to match the central slab of the
ground-truth cube. Everything the rectangular path does not support must be refused
with a clear message.
"""

from pathlib import Path

import numpy as np
import pytest

from fixtures import read_mrc, write_mrc, write_volume_mrc, write_custom_star

N, NY_RECT, NPART = 32, 20, 90


def ground_truth():
    z, y, x = np.mgrid[0:N, 0:N, 0:N] - N // 2
    rng = np.random.default_rng(5)
    vol = np.zeros((N, N, N))
    for cx, cy, cz in rng.uniform(-5, 5, size=(6, 3)):
        vol += np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * 1.8 ** 2))
    return vol


@pytest.fixture
def rect_dataset(test_data_dir, relion_bin, relion_runner):
    rng = np.random.default_rng(11)
    gt = ground_truth()
    write_volume_mrc(gt, str(test_data_dir / "gt.mrc"))

    angles = (rng.uniform(0, 360, NPART), np.degrees(np.arccos(rng.uniform(-1, 1, NPART))),
              rng.uniform(0, 360, NPART))
    write_custom_star(NPART, str(test_data_dir / "square.star"), "x.mrcs", *angles, size=N, angpix=1.0)
    r = relion_runner([str(relion_bin / "relion_project"), "--i", str(test_data_dir / "gt.mrc"),
                       "--ang", str(test_data_dir / "square.star"), "--o", str(test_data_dir / "proj"),
                       "--angpix", "1.0"], timeout=600)
    assert r.returncode == 0, r.stderr
    stack = read_mrc(test_data_dir / "proj.mrcs")
    assert stack.shape == (NPART, N, N)
    lo = (N - NY_RECT) // 2
    write_mrc(stack[:, lo:lo + NY_RECT, :], str(test_data_dir / "rect.mrcs"))

    text = (test_data_dir / "square.star").read_text()
    text = text.replace("_rlnImageSize #8\n_rlnImageDimensionality #9",
                        "_rlnImageSizeX #8\n_rlnImageSizeY #9\n_rlnImageDimensionality #10")
    text = text.replace(f"\t{N}\t2\n", f"\t{N}\t{NY_RECT}\t2\n", 1)
    text = text.replace("@x.mrcs", f"@{test_data_dir / 'rect.mrcs'}")
    (test_data_dir / "rect.star").write_text(text)
    return test_data_dir, gt, lo


def reconstruct(relion_bin, relion_runner, d, extra=(), out="rect_recon.mrc"):
    return relion_runner([str(relion_bin / "relion_reconstruct"), "--i", str(d / "rect.star"),
                          "--o", str(d / out), "--angpix", "1.0"] + list(extra), timeout=600)


@pytest.mark.integration
def test_reconstruction_from_rectangular_images(rect_dataset, relion_bin, relion_runner):
    d, gt, lo = rect_dataset
    r = reconstruct(relion_bin, relion_runner, d)
    assert r.returncode == 0, r.stdout + r.stderr
    vol = read_mrc(d / "rect_recon.mrc")
    assert vol.shape == (N, NY_RECT, NY_RECT)           # nz = image width, nx = ny = image height

    ref = gt[:, lo:lo + NY_RECT, lo:lo + NY_RECT]
    c = np.corrcoef(vol.ravel(), ref.ravel())[0, 1]
    assert c > 0.9, c


@pytest.mark.integration
def test_box_z_sets_the_cuboid_depth(rect_dataset, relion_bin, relion_runner):
    d, gt, lo = rect_dataset
    r = reconstruct(relion_bin, relion_runner, d, ["--box_z", "24"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert read_mrc(d / "rect_recon.mrc").shape == (24, NY_RECT, NY_RECT)


@pytest.mark.integration
@pytest.mark.parametrize("extra,message", [
    (["--ewald", "--mask_diameter", "20"], "ewald"),
    (["--spatial_frequency_mode", "s2"], "s2"),
    (["--newbox", "16"], "newbox"),
    (["--reconstruct_ctf", "32"], "reconstruct_ctf"),
], ids=["ewald", "s2", "newbox", "ctf"])
def test_unsupported_options_are_refused(rect_dataset, relion_bin, relion_runner, extra, message):
    d, _, _ = rect_dataset
    r = reconstruct(relion_bin, relion_runner, d, extra, out="refused.mrc")
    assert r.returncode != 0
    text = r.stdout + r.stderr
    assert "rectangular" in text and message in text
    assert not Path(d / "refused.mrc").exists()


def refine(relion_bin, relion_runner, d, out, extra):
    low = d / "ref_cuboid.mrc"
    if not low.exists():
        gt = ground_truth()
        lo = (N - NY_RECT) // 2
        write_volume_mrc(gt[:, lo:lo + NY_RECT, lo:lo + NY_RECT], str(low))
    (d / out).mkdir(exist_ok=True)
    return relion_runner([str(relion_bin / "relion_refine"), "--i", str(d / "rect.star"),
                          "--ref", str(low), "--ini_high", "12", "--o", str(d / out / "run"),
                          "--particle_diameter", "24", "--sym", "C1",
                          "--oversampling", "0", "--healpix_order", "2", "--offset_range", "3",
                          "--offset_step", "2", "--j", "2", "--pool", "10", "--norm", "--scale",
                          "--flatten_solvent", "--zero_mask", "--ctf"] + list(extra), timeout=1800)


@pytest.mark.integration
def test_rectangular_class3d_runs_and_writes_cuboid_maps(rect_dataset, relion_bin, relion_runner):
    d, gt, lo = rect_dataset
    r = refine(relion_bin, relion_runner, d, "c3d", ["--iter", "2", "--K", "2", "--tau2_fudge", "4"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "padded references and back-projectors" in r.stdout
    maps = sorted((d / "c3d").glob("run_it002_class00*.mrc"))
    assert len(maps) == 2
    for m in maps:
        assert read_mrc(m).shape == (N, NY_RECT, NY_RECT)


@pytest.mark.integration
@pytest.mark.parametrize("extra,message", [
    (["--grad", "--iter", "2", "--K", "1"], "VDAM"),
    (["--gpu", "--iter", "1", "--K", "1"], "--gpu"),
], ids=["vdam", "gpu"])
def test_rectangular_refine_refuses_unsupported_modes(rect_dataset, relion_bin, relion_runner, extra, message):
    d, _, _ = rect_dataset
    r = refine(relion_bin, relion_runner, d, "refused_run", extra)
    assert r.returncode != 0
    text = r.stdout + r.stderr
    assert "rectangular" in text and message in text


@pytest.mark.integration
def test_rectangular_refine_refuses_mixed_optics_groups(rect_dataset, relion_bin, relion_runner):
    d, _, _ = rect_dataset
    lines = (d / "rect.star").read_text().splitlines()
    out, in_particles, n = [], False, 0
    for line in lines:
        if line.startswith("data_particles"):
            in_particles = True
        parts = line.split("\t")
        if line.startswith("opticsGroup1\t"):
            out.append(line)
            second = list(parts)
            second[0], second[1], second[6] = "opticsGroup2", "2", "1.200000"
            out.append("\t".join(second))
            continue
        if in_particles and len(parts) > 3 and parts[2] == "1":
            n += 1
            if n % 2 == 0:
                parts[2] = "2"
                line = "\t".join(parts)
        out.append(line)
    (d / "rect.star").write_text("\n".join(out) + "\n")
    r = refine(relion_bin, relion_runner, d, "mixed_run", ["--iter", "1", "--K", "1"])
    assert r.returncode != 0
    text = r.stdout + r.stderr
    assert "rectangular" in text and "same box size and pixel size" in text
