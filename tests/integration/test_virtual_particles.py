"""Virtual particles are bitwise identical to extracted ones.

relion_preprocess --virtual writes .vstack descriptors instead of particle
images, and the particles are computed from the micrographs when read. The
contract that makes this safe to adopt is that a virtual particle is exactly
the particle relion_preprocess would have written with the same options. These
tests extract the same synthetic micrographs both ways and compare every pixel,
reading both sets through relion_stack_create (which goes through the ordinary
image reader, as every other program does).

Run with RELION_BIN_DIR pointing at a built RELION bin directory.
"""
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

NX, NY = 360, 300


# ---------------------------------------------------------------------------
# Small file helpers (no external STAR/MRC libraries needed)
# ---------------------------------------------------------------------------

def write_mrc(path, data, mode=2):
    """Minimal MRC writer: float32 (mode 2) or float16 (mode 12), no extended header."""
    data = np.asarray(data, dtype=np.float32 if mode == 2 else np.float16)
    ny, nx = data.shape
    header = np.zeros(256, dtype=np.int32)
    header[0:3] = (nx, ny, 1)
    header[3] = mode
    header[7:10] = (nx, ny, 1)
    hf = header.view(np.float32)
    hf[10:13] = (nx * 1.5, ny * 1.5, 1.5)         # cell: pixel size 1.5 A
    hf[13:16] = 90.0
    header[16:19] = (1, 2, 3)
    header[52] = int.from_bytes(b"MAP ", "little")
    header[53] = int.from_bytes(bytes([0x44, 0x44, 0, 0]), "little")
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(data.tobytes())


def read_mrc_stack(path):
    with open(path, "rb") as f:
        raw = f.read()
    h = np.frombuffer(raw[:1024], dtype=np.int32)
    nx, ny, nz, mode, nsymbt = h[0], h[1], h[2], h[3], h[23]
    dtype = {2: np.float32, 12: np.float16}[int(mode)]
    return np.frombuffer(raw[1024 + nsymbt:], dtype=dtype).reshape(nz, ny, nx).astype(np.float64)


def micrograph(seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:NY, 0:NX]
    blobs = sum(np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 90.0)
                for cx, cy in rng.uniform(0, [NX, NY], size=(25, 2)))
    return 5.0 * blobs + 0.02 * y + rng.normal(0, 1, (NY, NX))


def write_loop(path, blocks):
    """blocks: list of (name, [labels], [rows])"""
    with open(path, "w") as f:
        for name, labels, rows in blocks:
            f.write(f"\n# version 50001\n\ndata_{name}\n\nloop_\n")
            for i, lab in enumerate(labels):
                f.write(f"_rln{lab} #{i + 1}\n")
            for row in rows:
                f.write(" ".join(str(v) for v in row) + "\n")
            f.write("\n")


def read_particles(path):
    """Labels and rows of the data_particles block of a STAR file."""
    labels, rows, inside = [], [], False
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        s = line.strip()
        if s.startswith("data_"):
            inside = (s == "data_particles")
            continue
        if not inside or not s or s == "loop_" or s.startswith("#"):
            continue
        if s.startswith("_rln"):
            labels.append(s.split()[0][4:])
        else:
            rows.append(s.split())
    return labels, rows


# ---------------------------------------------------------------------------
# A tiny project: two micrographs with CTF, coordinates including edge cases
# ---------------------------------------------------------------------------

@pytest.fixture
def project(tmp_path):
    (tmp_path / "Micrographs").mkdir()
    (tmp_path / "Coords").mkdir()

    coords = {
        # centre, near and hanging off each edge, and fractional coordinates
        "mic001": [(180, 150), (60.4, 70.7), (8, 9), (NX - 4, 150), (180, NY - 6.2), (250.9, 99.5)],
        "mic002": [(100, 100), (300.5, 210.2), (3.7, NY - 2), (200, 12), (NX - 30, 40)],
    }
    mic_rows, coord_rows = [], []
    for k, (name, cs) in enumerate(coords.items()):
        write_mrc(tmp_path / "Micrographs" / f"{name}.mrc", micrograph(k + 1),
                  mode=12 if k == 1 else 2)   # one float16, one float32 micrograph
        write_loop(tmp_path / "Coords" / f"{name}.star",
                   [("", ["CoordinateX", "CoordinateY"], cs)])
        mic_rows.append([f"Micrographs/{name}.mrc", 1, 12000 + 500 * k, 11800 + 500 * k, 30.0])
        coord_rows.append([f"Micrographs/{name}.mrc", f"Coords/{name}.star"])

    write_loop(tmp_path / "micrographs_ctf.star", [
        ("optics", ["OpticsGroupName", "OpticsGroup", "MicrographPixelSize", "Voltage",
                    "SphericalAberration", "AmplitudeContrast"],
         [["opticsGroup1", 1, 1.5, 300, 2.7, 0.1]]),
        ("micrographs", ["MicrographName", "OpticsGroup", "DefocusU", "DefocusV", "DefocusAngle"],
         mic_rows),
    ])
    write_loop(tmp_path / "coords.star",
               [("coordinate_files", ["MicrographName", "MicrographCoordinates"], coord_rows)])
    return tmp_path


def run(relion_bin, project, args, env_extra=None):
    env = dict(os.environ)
    env["RELION_VPARTICLE_CACHE"] = str(project / "vcache")
    env.pop("RELION_VIRTUAL_PARTICLES", None)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run([str(relion_bin / args[0])] + args[1:], cwd=project, env=env,
                       capture_output=True, text=True, timeout=600)
    return r


def extract(relion_bin, project, outdir, options, virtual):
    (project / outdir).mkdir(parents=True, exist_ok=True)
    args = ["relion_preprocess", "--i", "micrographs_ctf.star",
            "--part_star", f"{outdir}/particles.star", "--part_dir", f"{outdir}/",
            "--extract"] + options + (["--virtual"] if virtual else ["--no_virtual"])
    r = run(relion_bin, project, args)
    assert r.returncode == 0, r.stdout + r.stderr
    return project / outdir / "particles.star"


def materialise(relion_bin, project, star, out):
    r = run(relion_bin, project, ["relion_stack_create", "--i", str(star), "--o", out])
    assert r.returncode == 0, r.stdout + r.stderr
    return read_mrc_stack(project / f"{out}.mrcs")


def assert_same_particles(relion_bin, project, real_star, virtual_star):
    a = materialise(relion_bin, project, real_star, "cmp_real")
    b = materialise(relion_bin, project, virtual_star, "cmp_virtual")
    assert a.shape == b.shape
    assert np.array_equal(a, b), f"max |diff| {np.abs(a - b).max()}"

    # And the metadata is the same apart from the stack file each name points at
    la, ra = read_particles(real_star)
    lb, rb = read_particles(virtual_star)
    assert la == lb
    i = la.index("ImageName")
    assert [r[:i] + r[i + 1:] for r in ra] == [r[:i] + r[i + 1:] for r in rb]
    assert all(r[i].endswith(".mrcs") for r in ra)
    assert all(r[i].endswith(".vstack") for r in rb)
    assert [r[i].split("@")[0] for r in ra] == [r[i].split("@")[0] for r in rb]
    return a


# ---------------------------------------------------------------------------

@pytest.mark.parametrize("options", [
    # binned, normalised, inverted, float16: the usual first extraction
    ["--extract_size", "64", "--scale", "48", "--norm", "--bg_radius", "18",
     "--white_dust", "-1", "--black_dust", "-1", "--invert_contrast", "--float16"],
    # full size, float32, plain mean subtraction
    ["--extract_size", "56", "--norm", "--bg_radius", "20", "--no_ramp"],
    # re-windowed after rescaling, no normalisation
    ["--extract_size", "80", "--scale", "60", "--window", "48"],
], ids=["binned_float16", "fullsize_float32", "rewindowed"])
def test_virtual_particles_match_extracted(relion_bin, project, options):
    opts = ["--coord_list", "coords.star"] + options
    real = extract(relion_bin, project, "Extract/real", opts, virtual=False)
    virt = extract(relion_bin, project, "Extract/virtual", opts, virtual=True)

    # No particle pixels written for the virtual set
    assert not list((project / "Extract/virtual").rglob("*.mrcs"))
    assert list((project / "Extract/virtual").rglob("*.vstack"))

    particles = assert_same_particles(relion_bin, project, real, virt)
    assert particles.shape[0] == 11


def test_recentred_reextraction_matches(relion_bin, project):
    """Re-extraction moves the boxes by the refined offsets: the path most
    likely to disagree, since the coordinates are computed rather than read."""
    opts = ["--coord_list", "coords.star", "--extract_size", "64", "--norm", "--bg_radius", "20"]
    first = extract(relion_bin, project, "Extract/first", opts, virtual=False)

    # Pretend a refinement found sub-pixel and multi-pixel offsets
    labels, rows = read_particles(first)
    rng = np.random.default_rng(7)
    labels2 = labels + ["OriginXAngst", "OriginYAngst"]
    rows2 = [r + [f"{x:.4f}", f"{y:.4f}"] for r, (x, y) in zip(rows, rng.uniform(-9, 9, (len(rows), 2)))]
    optics = Path(first).read_text().split("data_particles")[0]
    write_loop(project / "refined_particles.tmp", [("particles", labels2, rows2)])
    (project / "refined.star").write_text(optics + (project / "refined_particles.tmp").read_text())

    reopts = ["--reextract_data_star", "refined.star", "--recenter", "--recenter_x", "0",
              "--recenter_y", "0", "--recenter_z", "0", "--extract_size", "72", "--scale", "48",
              "--norm", "--bg_radius", "16", "--invert_contrast", "--float16"]
    real = extract(relion_bin, project, "Extract/re_real", reopts, virtual=False)
    virt = extract(relion_bin, project, "Extract/re_virtual", reopts, virtual=True)
    assert_same_particles(relion_bin, project, real, virt)


def test_environment_variable_sets_the_default(relion_bin, project):
    opts = ["--coord_list", "coords.star", "--extract_size", "48"]
    (project / "Extract/env").mkdir(parents=True)
    args = ["relion_preprocess", "--i", "micrographs_ctf.star", "--part_star", "Extract/env/particles.star",
            "--part_dir", "Extract/env/", "--extract"] + opts
    r = run(relion_bin, project, args, {"RELION_VIRTUAL_PARTICLES": "1"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert list((project / "Extract/env").rglob("*.vstack"))
    assert not list((project / "Extract/env").rglob("*.mrcs"))

    # --no_virtual wins over the environment
    (project / "Extract/env2").mkdir(parents=True)
    args2 = [a.replace("Extract/env/", "Extract/env2/") for a in args] + ["--no_virtual"]
    r = run(relion_bin, project, args2, {"RELION_VIRTUAL_PARTICLES": "1"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert list((project / "Extract/env2").rglob("*.mrcs"))


@pytest.mark.parametrize("option,message", [
    # CTF phase flipping cannot be expressed yet
    (["--phase_flip"], "phase flipping"),
    # dust removal fills pixels with random numbers: never the same particle twice
    (["--norm", "--bg_radius", "20", "--white_dust", "5"], "dust removal"),
], ids=["phase_flip", "dust"])
def test_unsupported_recipes_are_refused(relion_bin, project, option, message):
    """Refuse rather than approximate."""
    (project / "Extract/bad").mkdir(parents=True)
    r = run(relion_bin, project, ["relion_preprocess", "--i", "micrographs_ctf.star",
            "--coord_list", "coords.star", "--part_star", "Extract/bad/particles.star",
            "--part_dir", "Extract/bad/", "--extract", "--extract_size", "64",
            "--virtual"] + option)
    assert r.returncode != 0
    assert message in (r.stdout + r.stderr)


def test_particles_survive_deleting_the_cache(relion_bin, project):
    opts = ["--coord_list", "coords.star", "--extract_size", "64", "--scale", "48",
            "--norm", "--bg_radius", "18", "--float16"]
    virt = extract(relion_bin, project, "Extract/v", opts, virtual=True)
    first = materialise(relion_bin, project, virt, "cmp_first")
    assert list((project / "vcache").rglob("*.vpc"))

    subprocess.run(["rm", "-rf", str(project / "vcache")], check=True)
    again = materialise(relion_bin, project, virt, "cmp_again")
    assert np.array_equal(first, again)


# ---------------------------------------------------------------------------
# relion_stacks_to_virtual: converting existing Extract jobs in place
# ---------------------------------------------------------------------------

def extract_job(relion_bin, project, job, options):
    """A real (stack) Extract job laid out as the pipeline does, note.txt included."""
    outdir = f"Extract/{job}"
    star = extract(relion_bin, project, outdir, options, virtual=False)
    args = " ".join(["--i", "micrographs_ctf.star", "--part_star", f"{outdir}/particles.star",
                     "--part_dir", f"{outdir}/", "--extract"] + options)
    (project / outdir / "note.txt").write_text(
        f" ++++ Executing new job on Sat Sep 26 2026\n ++++ with the following command(s): \n"
        f"`which relion_preprocess` {args}   --pipeline_control {outdir}/\n ++++ \n")
    return star


def downstream_subset(project, star, out):
    """Every other particle, as a Select job would leave them: names untouched."""
    text = Path(star).read_text()
    head, body = text.split("data_particles")
    lines = body.splitlines()
    first_row = next(i for i, l in enumerate(lines) if l.strip() and not l.strip().startswith(("_", "loop_", "#")))
    rows = [l for l in lines[first_row:] if l.strip()]
    (project / out).write_text(head + "data_particles" + "\n".join(lines[:first_row] + rows[::2]) + "\n")
    return project / out


def stack_files(project, job):
    return sorted((project / "Extract" / job).rglob("*.mrcs"))


def is_descriptor(path):
    with open(path, "rb") as f:
        return f.read(31) == b"# RELION virtual particle stack"


def convert(relion_bin, project, *extra):
    r = run(relion_bin, project, ["relion_stacks_to_virtual", "--j", "4"] + list(extra))
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_converting_a_project_keeps_every_particle(relion_bin, project):
    extract_job(relion_bin, project, "job010", ["--coord_list", "coords.star", "--extract_size", "64",
                "--scale", "48", "--norm", "--bg_radius", "18", "--invert_contrast", "--float16"])
    extract_job(relion_bin, project, "job011", ["--coord_list", "coords.star", "--extract_size", "56",
                "--norm", "--bg_radius", "20", "--no_ramp"])
    # Dust removal fills pixels with random values: must be left exactly as it is
    extract_job(relion_bin, project, "job013", ["--coord_list", "coords.star", "--extract_size", "56",
                "--norm", "--bg_radius", "20", "--white_dust", "5", "--black_dust", "5"])
    dusty = {p: p.read_bytes() for p in stack_files(project, "job013")}
    # Many real projects have the per-micrograph *_extract.star files cleaned
    # away; the particle lists then come from the job's particles.star
    for p in (project / "Extract/job011").rglob("*_extract.star"):
        p.unlink()
    selected = downstream_subset(project, project / "Extract/job010/particles.star", "selected.star")

    before = {j: materialise(relion_bin, project, project / f"Extract/{j}/particles.star", f"before_{j}")
              for j in ("job010", "job011")}
    before_sel = materialise(relion_bin, project, selected, "before_sel")
    stars = {p: p.read_bytes() for p in project.rglob("*.star")}
    sizes = {p: p.stat().st_size for j in ("job010", "job011") for p in stack_files(project, j)}

    # A dry run writes nothing
    out = convert(relion_bin, project)
    assert "dry run" in out and out.count(" identical") >= 2
    assert {p: p.stat().st_size for p in sizes} == sizes

    out = convert(relion_bin, project, "--convert")
    assert "replaced" in out
    for p in sizes:
        assert is_descriptor(p), p
        assert p.stat().st_size < sizes[p]

    # No STAR file changed, and every particle - through the Extract jobs'
    # own STAR files and through a downstream selection - is the same
    assert {p: p.read_bytes() for p in stars} == stars
    for j in ("job010", "job011"):
        assert np.array_equal(before[j], materialise(relion_bin, project,
                              project / f"Extract/{j}/particles.star", f"after_{j}"))
    assert np.array_equal(before_sel, materialise(relion_bin, project, selected, "after_sel"))

    # The dust-removal job was skipped and is byte for byte as it was
    assert "dust removal" in out
    assert {p: p.read_bytes() for p in dusty} == dusty

    # Converting again finds nothing left to do
    assert "already virtual" in convert(relion_bin, project, "--convert")


def test_converter_keeps_stacks_it_cannot_reproduce(relion_bin, project):
    extract_job(relion_bin, project, "job012", ["--coord_list", "coords.star", "--extract_size", "64",
                "--norm", "--bg_radius", "20"])
    stacks = stack_files(project, "job012")
    assert len(stacks) == 2

    # Damage one stack, as if it had been written by some other program
    with open(stacks[0], "r+b") as f:
        f.seek(1024 + 4 * 1000)
        f.write(np.float32(123.0).tobytes())

    out = convert(relion_bin, project, "--convert")
    assert "DIFFERENT" in out
    assert not is_descriptor(stacks[0])     # kept, untouched
    assert is_descriptor(stacks[1])         # the good one was converted


# ---------------------------------------------------------------------------
# Helical segments (amyloid, tau, ...): normalised against a tube-shaped
# background oriented by each segment's psi
# ---------------------------------------------------------------------------

@pytest.fixture
def filaments(project):
    """Start-end coordinates of two filaments per micrograph, at awkward angles."""
    tubes = {
        "mic001": [((40, 60), (300, 210)), ((330, 40), (120, 280))],
        "mic002": [((50, 250), (310, 70)), ((180, 20), (185, 285))],
    }
    rows = []
    for name, ts in tubes.items():
        pts = [p for t in ts for p in t]
        write_loop(project / "Coords" / f"{name}_tubes.star", [("", ["CoordinateX", "CoordinateY"], pts)])
        rows.append([f"Micrographs/{name}.mrc", f"Coords/{name}_tubes.star"])
    write_loop(project / "tubes.star", [("coordinate_files", ["MicrographName", "MicrographCoordinates"], rows)])
    return project


HELICAL = ["--coord_list", "tubes.star", "--helix", "--from_startend", "--helical_outer_diameter", "60",
           "--helical_nr_asu", "1", "--helical_rise", "24", "--helical_cut_into_segments"]


@pytest.mark.parametrize("options", [
    ["--extract_size", "64", "--norm", "--bg_radius", "18"],
    # downscaled: relion_preprocess's tube radius becomes 0 (int / int), and
    # virtual particles have to reproduce exactly that
    ["--extract_size", "64", "--scale", "48", "--norm", "--bg_radius", "14", "--invert_contrast", "--float16"],
], ids=["fullsize", "downscaled_float16"])
def test_virtual_helical_segments_match_extracted(relion_bin, filaments, options):
    project = filaments
    real = extract(relion_bin, project, "Extract/hreal", HELICAL + options, virtual=False)
    virt = extract(relion_bin, project, "Extract/hvirtual", HELICAL + options, virtual=True)
    particles = assert_same_particles(relion_bin, project, real, virt)
    assert particles.shape[0] > 20          # the tubes really were cut into segments
    labels, _ = read_particles(virt)
    assert "AnglePsiPrior" in labels and "HelicalTubeID" in labels


def test_converting_helical_segments(relion_bin, filaments):
    project = filaments
    extract_job(relion_bin, project, "job020", HELICAL + ["--extract_size", "64", "--norm", "--bg_radius", "18"])
    star = project / "Extract/job020/particles.star"
    before = materialise(relion_bin, project, star, "before_helical")
    stacks = stack_files(project, "job020")

    out = convert(relion_bin, project, "--convert")
    # psi comes back from the _extract.star text (six decimals), not from the
    # value relion_preprocess held in memory, so a stack may legitimately fail
    # verification; what must never happen is a stack converted with differences
    converted = [p for p in stacks if is_descriptor(p)]
    assert converted, out
    assert np.array_equal(before, materialise(relion_bin, project, star, "after_helical"))


# ---------------------------------------------------------------------------
# Rectangular and rotated boxes
# ---------------------------------------------------------------------------

def finufft_available(relion_bin, project):
    (project / "x").mkdir(exist_ok=True)
    r = run(relion_bin, project, ["relion_preprocess", "--i", "micrographs_ctf.star", "--coord_list", "coords.star",
            "--part_star", "x/particles.star", "--part_dir", "x/", "--extract", "--extract_size_x", "64",
            "--extract_size_y", "32", "--interpolation", "nufft"])
    return "FINUFFT" not in (r.stdout + r.stderr)


RECT = ["--extract_size_x", "96", "--extract_size_y", "48", "--norm", "--bg_radius", "18", "--invert_contrast",
        "--float16"]


def test_rectangular_virtual_matches_extracted(relion_bin, project):
    opts = ["--coord_list", "coords.star"] + RECT
    real = extract(relion_bin, project, "Extract/rreal", opts, virtual=False)
    virt = extract(relion_bin, project, "Extract/rvirtual", opts, virtual=True)
    particles = assert_same_particles(relion_bin, project, real, virt)
    assert particles.shape == (11, 48, 96)

    # The optics table gives the two sizes, not a single rlnImageSize
    text = Path(real).read_text().split("data_particles")[0]
    assert "_rlnImageSizeX" in text and "_rlnImageSizeY" in text
    assert "_rlnImageSize " not in text


def test_rectangular_scale_keeps_the_aspect_ratio(relion_bin, project):
    opts = ["--coord_list", "coords.star", "--extract_size_x", "96", "--extract_size_y", "48", "--scale", "48",
            "--norm", "--bg_radius", "9", "--float16"]
    real = extract(relion_bin, project, "Extract/sreal", opts, virtual=False)
    virt = extract(relion_bin, project, "Extract/svirtual", opts, virtual=True)
    particles = assert_same_particles(relion_bin, project, real, virt)
    assert particles.shape == (11, 24, 48)
    text = Path(real).read_text().split("data_particles")[0]
    assert "_rlnImageSizeX" in text and "_rlnImageSizeY" in text


def test_rotating_a_box_with_aberrations_is_refused(relion_bin, filaments):
    project = filaments
    text = (project / "micrographs_ctf.star").read_text()
    text = text.replace("_rlnAmplitudeContrast #6\n", "_rlnAmplitudeContrast #6\n_rlnBeamTiltX #7\n_rlnBeamTiltY #8\n", 1)
    text = text.replace("1.5 300 2.7 0.1\n", "1.5 300 2.7 0.1 0.2 -0.1\n", 1)
    (project / "micrographs_aberr.star").write_text(text)
    (project / "Extract/ab").mkdir(parents=True)
    args = ["relion_preprocess", "--i", "micrographs_aberr.star", "--part_star", "Extract/ab/particles.star",
            "--part_dir", "Extract/ab/", "--extract"] + HELICAL + RECT + ["--rotate_to_horizontal", "--no_virtual"]
    r = run(relion_bin, project, args)
    assert r.returncode != 0 and "allow_unrotated_aberrations" in r.stdout + r.stderr
    r = run(relion_bin, project, args + ["--allow_unrotated_aberrations"])
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("method", ["linear", "cubic", "nufft"])
def test_rotated_helical_virtual_matches_extracted(relion_bin, filaments, method):
    project = filaments
    if method == "nufft" and not finufft_available(relion_bin, project):
        pytest.skip("this build has no FINUFFT")
    opts = HELICAL + RECT + ["--rotate_to_horizontal", "--interpolation", method]
    real = extract(relion_bin, project, "Extract/rotreal", opts, virtual=False)
    virt = extract(relion_bin, project, "Extract/rotvirtual", opts, virtual=True)
    particles = assert_same_particles(relion_bin, project, real, virt)
    assert particles.shape[0] > 10 and particles.shape[1:] == (48, 96)

    labels, rows = read_particles(real)
    psi = labels.index("AnglePsiPrior")
    ext = labels.index("ParticleExtractionAngle")
    assert all(abs(float(r[psi])) < 1e-4 for r in rows)          # residual after rotating
    assert max(abs(float(r[ext])) for r in rows) > 10            # the tubes are not horizontal


def test_rotation_follows_the_tube(relion_bin, project):
    """A bright line along a tube becomes horizontal; the opposite sign would not."""
    from math import atan2, degrees, cos, sin
    y, x = np.mgrid[0:NY, 0:NX]
    (x0, y0), (x1, y1) = (60.0, 70.0), (300.0, 230.0)
    ang = atan2(y1 - y0, x1 - x0)
    dist = np.abs((x - x0) * sin(ang) - (y - y0) * cos(ang))
    mic = 10 * np.exp(-dist ** 2 / (2 * 4.0 ** 2)) + np.random.default_rng(3).normal(0, 0.05, (NY, NX))
    write_mrc(project / "Micrographs" / "mic001.mrc", mic)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    psi = -degrees(ang)
    write_loop(project / "mic1.star", [
        ("optics", ["OpticsGroupName", "OpticsGroup", "MicrographPixelSize", "Voltage",
                    "SphericalAberration", "AmplitudeContrast"], [["opticsGroup1", 1, 1.5, 300, 2.7, 0.1]]),
        ("micrographs", ["MicrographName", "OpticsGroup", "DefocusU", "DefocusV", "DefocusAngle"],
         [["Micrographs/mic001.mrc", 1, 12000, 11800, 30.0]]),
    ])

    def contrast(tag, psi_prior):
        write_loop(project / f"parts_{tag}.star", [
            ("optics", ["OpticsGroupName", "OpticsGroup", "MicrographOriginalPixelSize", "Voltage",
                        "SphericalAberration", "AmplitudeContrast", "ImagePixelSize", "ImageSize",
                        "ImageDimensionality"],
             [["opticsGroup1", 1, 1.5, 300, 2.7, 0.1, 1.5, 64, 2]]),
            ("particles", ["MicrographName", "OpticsGroup", "CoordinateX", "CoordinateY", "AnglePsiPrior",
                           "AngleTiltPrior", "HelicalTubeID", "DefocusU", "DefocusV", "DefocusAngle"],
             [["Micrographs/mic001.mrc", 1, cx, cy, f"{psi_prior:.6f}", 90, 1, 12000, 11800, 30.0]]),
        ])
        (project / f"Extract/{tag}").mkdir(parents=True)
        r = run(relion_bin, project, ["relion_preprocess", "--i", "mic1.star", "--reextract_data_star",
                f"parts_{tag}.star", "--part_star", f"Extract/{tag}/particles.star", "--part_dir", f"Extract/{tag}/",
                "--extract", "--extract_size_x", "128", "--extract_size_y", "64", "--helix",
                "--helical_outer_diameter", "20", "--rotate_to_horizontal", "--interpolation", "cubic",
                "--no_virtual"])
        assert r.returncode == 0, r.stdout + r.stderr
        img = materialise(relion_bin, project, project / f"Extract/{tag}/particles.star", f"cmp_{tag}")[0]
        return img.mean(axis=1).var() / (img.mean(axis=0).var() + 1e-12)

    assert contrast("good", psi) > 20 * contrast("flipped", -psi)


@pytest.mark.parametrize("option,message", [
    (["--extract_size", "64", "--rotate_to_horizontal"], "helix"),           # rotating needs helical priors
    (["--extract_size_x", "64"], "extract_size_y"),                          # both or neither
    (["--extract_size_x", "64", "--extract_size_y", "32", "--phase_flip"], "phase"),
    (["--extract_size_x", "64", "--extract_size_y", "32", "--window", "32"], "window"),
    (["--extract_size_x", "96", "--extract_size_y", "50", "--scale", "48"], "scale"),     # 50*48/96 = 25 rows, odd
], ids=["rotate_without_helix", "one_side_only", "phase_flip", "window", "scale_gives_odd_height"])
def test_unsupported_rectangular_options_are_refused(relion_bin, project, option, message):
    (project / "Extract/bad").mkdir(parents=True)
    r = run(relion_bin, project, ["relion_preprocess", "--i", "micrographs_ctf.star", "--coord_list", "coords.star",
            "--part_star", "Extract/bad/particles.star", "--part_dir", "Extract/bad/", "--extract"] + option)
    assert r.returncode != 0
    assert message in (r.stdout + r.stderr)
