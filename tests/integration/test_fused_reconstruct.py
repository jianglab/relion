"""relion_reconstruct --fused_extract: particles computed straight from the micrographs.

The fused path takes virtual stacks (relion_preprocess --virtual) and, instead of
reading each particle image and Fourier transforming it, evaluates its Fourier
transform directly from the micrograph on the output grid. That lets the output
pixel size and box size differ from the stored particles. These tests check that,
at the stored pixel size and extent, the fused map agrees with the ordinary map
from the same stack, that the output size options are honoured, and that
unsupported inputs are refused.

Run with RELION_BIN_DIR pointing at a built RELION bin directory.
"""
import numpy as np
import pytest

from test_virtual_particles import (HELICAL, NX, NY, extract, filaments, finufft_available,  # noqa: F401
                                    project, read_particles, run)

RECT = ["--coord_list", "coords.star", "--extract_size_x", "96", "--extract_size_y", "48",
        "--norm", "--bg_radius", "18", "--invert_contrast"]
SQUARE = ["--coord_list", "coords.star", "--extract_size", "64", "--norm", "--bg_radius", "18"]


def read_map(path):
    with open(path, "rb") as f:
        raw = f.read()
    h = np.frombuffer(raw[:1024], dtype=np.int32)
    nx, ny, nz, nsymbt = int(h[0]), int(h[1]), int(h[2]), int(h[23])
    hf = np.frombuffer(raw[:1024], dtype=np.float32)
    vol = np.frombuffer(raw[1024 + nsymbt:], dtype=np.float32).reshape(nz, ny, nx)
    return vol, float(hf[10]) / nx


def add_orientations(star):
    """Give every particle in a STAR file a different orientation and no shift."""
    text = star.read_text()
    head, _, body = text.partition("data_particles")
    lines = body.split("\n")
    out, k = [], 0
    nlab = sum(1 for s in lines if s.startswith("_rln"))
    rng = np.random.default_rng(3)
    for s in lines:
        if s.startswith("_rln"):
            out.append(s)
            if int(s.split("#")[1]) == nlab:
                out += [f"_rlnAngleRot #{nlab + 1}", f"_rlnAngleTilt #{nlab + 2}", f"_rlnAnglePsi #{nlab + 3}",
                        f"_rlnOriginXAngst #{nlab + 4}", f"_rlnOriginYAngst #{nlab + 5}"]
        elif s.strip() and not s.startswith(("#", "loop_")) and len(s.split()) >= nlab:
            a = rng.uniform([-180, 0, -180], [180, 180, 180])
            out.append(s + f" {a[0]:.3f} {a[1]:.3f} {a[2]:.3f} 0.0 0.0")
        else:
            out.append(s)
    star.write_text(head + "data_particles" + "\n".join(out))


@pytest.fixture
def virtual_rect(relion_bin, project):  # noqa: F811
    if not finufft_available(relion_bin, project):
        pytest.skip("RELION was built without FINUFFT")
    star = extract(relion_bin, project, "Extract/v", RECT, virtual=True)
    add_orientations(star)
    return project, star


def reconstruct(relion_bin, project, star, out, extra=()):
    return run(relion_bin, project, ["relion_reconstruct", "--i", str(star), "--o", out, "--ctf",
                                     "--angpix", "1.5"] + list(extra))


def test_fused_matches_the_unfused_map(relion_bin, virtual_rect):
    project, star = virtual_rect
    r = reconstruct(relion_bin, project, star, "plain.mrc")
    assert r.returncode == 0, r.stdout + r.stderr
    r = reconstruct(relion_bin, project, star, "fused.mrc", ["--fused_extract"])
    assert r.returncode == 0, r.stdout + r.stderr
    a, _ = read_map(project / "plain.mrc")
    b, _ = read_map(project / "fused.mrc")
    assert a.shape == b.shape == (96, 48, 48)
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.99


def test_output_pixel_size_and_box_are_free(relion_bin, virtual_rect):
    project, star = virtual_rect
    # Coarser pixels, smaller box: 3 A/pixel, 48 x 24 pixels = the same physical extent
    r = reconstruct(relion_bin, project, star, "coarse.mrc", ["--fused_extract", "--angpix", "3.0"])
    assert r.returncode == 0, r.stdout + r.stderr
    vol, apix = read_map(project / "coarse.mrc")
    assert vol.shape == (48, 24, 24) and abs(apix - 3.0) < 1e-3

    # A different box at the stored pixel size
    r = reconstruct(relion_bin, project, star, "box.mrc",
                    ["--fused_extract", "--box_x", "80", "--box_y", "40", "--box_z", "64"])
    assert r.returncode == 0, r.stdout + r.stderr
    vol, apix = read_map(project / "box.mrc")
    assert vol.shape == (64, 40, 40) and abs(apix - 1.5) < 1e-3


def test_odd_box_is_refused(relion_bin, virtual_rect):
    project, star = virtual_rect
    r = reconstruct(relion_bin, project, star, "odd.mrc", ["--fused_extract", "--box_x", "81"])
    assert r.returncode != 0
    assert "even" in r.stdout + r.stderr


def test_square_particles_work_too(relion_bin, project):  # noqa: F811
    if not finufft_available(relion_bin, project):
        pytest.skip("RELION was built without FINUFFT")
    star = extract(relion_bin, project, "Extract/sq", SQUARE, virtual=True)
    add_orientations(star)
    assert reconstruct(relion_bin, project, star, "p.mrc").returncode == 0
    r = reconstruct(relion_bin, project, star, "f.mrc", ["--fused_extract"])
    assert r.returncode == 0, r.stdout + r.stderr
    a, _ = read_map(project / "p.mrc")
    b, _ = read_map(project / "f.mrc")
    assert a.shape == b.shape == (64, 64, 64)
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.99


def test_stored_particles_are_read_from_their_coordinates(relion_bin, project):  # noqa: F811
    if not finufft_available(relion_bin, project):
        pytest.skip("RELION was built without FINUFFT")
    star = extract(relion_bin, project, "Extract/real", SQUARE, virtual=False)
    add_orientations(star)
    # Not virtual: the coordinates are used, which then need the output geometry
    r = reconstruct(relion_bin, project, star, "x.mrc", ["--fused_extract"])
    assert r.returncode != 0
    assert "--box_x" in r.stdout + r.stderr


def test_incompatible_options_are_refused(relion_bin, virtual_rect):
    project, star = virtual_rect
    r = reconstruct(relion_bin, project, star, "x.mrc", ["--fused_extract", "--ewald", "--mask_diameter", "100"])
    assert r.returncode != 0
    assert "--fused_extract" in r.stdout + r.stderr


def drop_image_names(star, out):
    """A particle STAR file as a picker or Select job leaves it: no rlnImageName, only micrograph and coordinates."""
    text = star.read_text()
    head, _, body = text.partition("data_particles")
    lines = body.split("\n")
    col = next(int(s.split("#")[1]) - 1 for s in lines if s.startswith("_rlnImageName"))
    res = []
    for s in lines:
        f = s.split()
        if s.startswith("_rln"):
            if int(f[1][1:]) - 1 == col:
                continue
            n = int(f[1][1:])
            res.append(f"{f[0]} #{n - 1 if n - 1 > col else n}")
        elif s.strip() and not s.startswith(("#", "loop_")):
            res.append(" ".join(f[:col] + f[col + 1:]))
        else:
            res.append(s)
    out.write_text(head + "data_particles" + "\n".join(res))


def test_coordinates_without_virtual_stacks(relion_bin, virtual_rect):
    project, star = virtual_rect
    drop_image_names(star, project / "coords_only.star")
    opts = ["--fused_extract", "--box_x", "96", "--box_y", "48", "--fused_norm", "--fused_bg_radius", "18",
            "--fused_invert_contrast"]
    r = reconstruct(relion_bin, project, star, "from_vstack.mrc", ["--fused_extract"])
    assert r.returncode == 0, r.stdout + r.stderr
    r = reconstruct(relion_bin, project, project / "coords_only.star", "from_coords.mrc", opts)
    assert r.returncode == 0, r.stdout + r.stderr
    a, _ = read_map(project / "from_vstack.mrc")
    b, _ = read_map(project / "from_coords.mrc")
    assert a.shape == b.shape
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.999


def test_coordinates_need_the_geometry_options(relion_bin, virtual_rect):
    project, star = virtual_rect
    drop_image_names(star, project / "coords_only.star")
    r = reconstruct(relion_bin, project, project / "coords_only.star", "x.mrc", ["--fused_extract"])
    assert r.returncode != 0
    assert "--box_x" in r.stdout + r.stderr


def test_turned_helical_segments_from_coordinates(relion_bin, filaments):  # noqa: F811
    project = filaments
    if not finufft_available(relion_bin, project):
        pytest.skip("RELION was built without FINUFFT")
    opts = HELICAL + ["--extract_size_x", "96", "--extract_size_y", "48", "--rotate_to_horizontal",
                      "--interpolation", "cubic", "--norm", "--bg_radius", "18", "--invert_contrast"]
    star = extract(relion_bin, project, "Extract/h", opts, virtual=True)
    add_orientations(star)
    r = reconstruct(relion_bin, project, star, "h_vstack.mrc", ["--fused_extract"])
    assert r.returncode == 0, r.stdout + r.stderr
    drop_image_names(star, project / "h_coords.star")
    r = reconstruct(relion_bin, project, project / "h_coords.star", "h_coords.mrc",
                    ["--fused_extract", "--box_x", "96", "--box_y", "48", "--fused_norm", "--fused_bg_radius", "18",
                     "--fused_invert_contrast", "--fused_helical_diameter", "60", "--fused_rotate_to_horizontal"])
    assert r.returncode == 0, r.stdout + r.stderr
    a, _ = read_map(project / "h_vstack.mrc")
    b, _ = read_map(project / "h_coords.mrc")
    assert a.shape == b.shape
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.999
