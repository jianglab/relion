"""Polish (relion_motion_refine) with CTF-premultiplied particles.

Polish fits the movie frames, which are never premultiplied, so particles that come in
premultiplied (as virtual premultiplied particles do) must be fitted exactly like plain
ones; --ctf_multiply --window --crop writes them premultiplied in the bigger box, cut to
the crop size, and marks them so. With --virtual Polish writes a descriptor per micrograph
instead of the stack, and reading it recombines the particles from the movie, bit for bit
what Polish writes.

Run with RELION_BIN_DIR pointing at a built RELION bin directory.
"""
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import test_virtual_movie_averages as vma
from fixtures import write_postprocess_star, write_volume_mrc
from test_virtual_particles import read_mrc_stack, read_particles, write_loop


def run(relion_bin, d, args, env_extra=None):
    env = vma._env()
    env.pop("RELION_VIRTUAL_POLISH", None)
    env["RELION_VPARTICLE_CACHE"] = str(d / "vcache")
    env.update(env_extra or {})
    r = subprocess.run([str(relion_bin / args[0])] + args[1:], cwd=d, env=env, capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r


@pytest.fixture
def polish_project(tmp_path, relion_bin):
    vma._write_movies(tmp_path)
    vma._motioncorr(relion_bin, tmp_path, "MotionCorr/job", [], threads=2)
    mics = sorted(str(p.relative_to(tmp_path)) for p in (tmp_path / "MotionCorr/job/Movies").glob("*.mrc")
                  if "_PS" not in p.name)
    z, y, x = np.mgrid[-32:32, -32:32, -32:32]
    rng = np.random.default_rng(3)
    vol = sum(np.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) / 10.0)
              for c in rng.uniform(-12, 12, size=(30, 3)))
    write_volume_mrc(vol.astype(np.float32), tmp_path / "ref.mrc", angpix=vma.ANGPIX)
    write_volume_mrc(np.ones_like(vol, dtype=np.float32), tmp_path / "mask.mrc", angpix=vma.ANGPIX)
    write_postprocess_star(tmp_path / "pp.star", "ref.mrc", "ref.mrc", size=64, angpix=vma.ANGPIX)
    rows = []
    for m, mic in enumerate(mics):
        for k, (cx, cy) in enumerate([(120, 110), (260, 240), (390, 150), (200, 360), (420, 380)]):
            rows.append([f"{k + 1:06d}@Extract/{Path(mic).stem}.mrcs", mic, cx, cy, 15000 + 300 * m + 50 * k, 14700 + 300 * m + 50 * k, 20.0, 1,
                         17 * k + 5 * m, 31 * k, 53 * k + 7 * m, 0, 0, 1 + (k + m) % 2])
    labels = ["ImageName", "MicrographName", "CoordinateX", "CoordinateY", "DefocusU", "DefocusV", "DefocusAngle", "OpticsGroup",
              "AngleRot", "AngleTilt", "AnglePsi", "OriginXAngst", "OriginYAngst", "RandomSubset"]
    for flag in (0, 1):
        write_loop(tmp_path / f"parts{flag}.star", [
            ("optics", ["OpticsGroupName", "OpticsGroup", "MicrographOriginalPixelSize", "Voltage",
                        "SphericalAberration", "AmplitudeContrast", "ImagePixelSize", "ImageSize",
                        "ImageDimensionality", "CtfDataAreCtfPremultiplied"],
             [["opticsGroup1", 1, vma.ANGPIX, 300, 2.7, 0.1, vma.ANGPIX, 64, 2, flag]]),
            ("particles", labels, rows)])
    return tmp_path


def with_aberrations(star, out):
    """`star` with an anisotropic magnification and even Zernike coefficients in its optics group."""
    head, body = Path(star).read_text().split("data_particles")
    lines = head.splitlines()
    n = sum(1 for l in lines if l.startswith("_rln"))
    i = max(k for k, l in enumerate(lines) if l.startswith("_rln"))
    labels = ["MagMat00", "MagMat01", "MagMat10", "MagMat11", "EvenZernike"]
    lines[i + 1:i + 1] = [f"_rln{lab} #{n + 1 + k}" for k, lab in enumerate(labels)]
    lines = [l + " 1.01 0.004 -0.003 0.995 [0.3,-0.2,2.5,0.1,-0.4,1.8]" if l.split()[:1] == ["opticsGroup1"] else l
             for l in lines]
    Path(out).write_text("\n".join(lines) + "\ndata_particles" + body)
    return out


def polish(relion_bin, d, star, out, extra=(), env_extra=None):
    polish.stdout = run(relion_bin, d, ["relion_motion_refine", "--i", star, "--f", "pp.star", "--m1", "ref.mrc", "--m2", "ref.mrc",
                        "--mask", "mask.mrc", "--corr_mic", "MotionCorr/job/corrected_micrographs.star",
                        "--first_frame", "1", "--last_frame", "-1", "--o", f"{out}/", "--s_vel", "0.2", "--s_div", "5000",
                        "--s_acc", "2", "--combine_frames", "--bfac_minfreq", "20", "--j", "1"] + list(extra), env_extra).stdout
    return d / out


def shiny(d):
    return np.concatenate([read_mrc_stack(p) for p in sorted(d.rglob("*_shiny.mrcs"))])


def optics(star):
    lines = [l.split() for l in Path(star).read_text().split("data_particles")[0].splitlines()]
    labels = [l[0][4:] for l in lines if l and l[0].startswith("_rln")]
    return dict(zip(labels, next(l for l in lines if l and l[0] == "opticsGroup1")))


def test_premultiplied_particles_are_polished_as_plain(relion_bin, polish_project):
    d = polish_project
    plain = polish(relion_bin, d, "parts0.star", "Polish/plain")
    premult = polish(relion_bin, d, "parts1.star", "Polish/premult")
    a, b = shiny(plain), shiny(premult)
    assert a.shape[0] == 10 and np.array_equal(a, b)
    tracks = sorted(plain.rglob("*_tracks.star"))
    assert tracks and all(t.read_text() == (premult / t.relative_to(plain)).read_text() for t in tracks)
    # without --ctf_multiply the shiny particles are plain
    assert optics(premult / "shiny.star")["CtfDataAreCtfPremultiplied"] == "0"


def test_ctf_multiply_in_a_bigger_box(relion_bin, polish_project):
    d = polish_project
    plain = polish(relion_bin, d, "parts1.star", "Polish/plain")
    big = polish(relion_bin, d, "parts1.star", "Polish/big", ["--ctf_multiply", "--window", "128", "--crop", "64"])
    o = optics(big / "shiny.star")
    assert o["CtfDataAreCtfPremultiplied"] == "1" and o["ImageSize"] == "64"
    a, b = shiny(plain), shiny(big)
    assert a.shape == b.shape and not np.array_equal(a, b)
    labels, rows = read_particles(big / "shiny.star")
    assert len(rows) == 10


def materialise(relion_bin, d, star, out, env_extra=None):
    run(relion_bin, d, ["relion_stack_create", "--i", str(star), "--o", out], env_extra)
    return read_mrc_stack(d / f"{out}.mrcs")


@pytest.mark.parametrize("extra,aberrations", [
    ([], False),
    (["--float16"], False),
    (["--ctf_multiply", "--window", "128", "--crop", "64"], False),
    (["--ctf_multiply", "--window", "128", "--crop", "64"], True),
], ids=["plain", "float16", "premultiplied_crop", "premultiplied_aberrations"])
def test_virtual_polished_particles_match(relion_bin, polish_project, extra, aberrations):
    d = polish_project
    star = str(with_aberrations(d / "parts1.star", d / "parts_ab.star")) if aberrations else "parts1.star"
    # premultiplied: real and virtual Polish both use the per-particle windows (the default)
    real = polish(relion_bin, d, star, "Polish/real", extra)
    virt = polish(relion_bin, d, star, "Polish/virtual", extra + ["--virtual"])
    assert not list(virt.rglob("*_shiny.mrcs")) and len(list(virt.rglob("*_shiny.vstack"))) == 2
    la, ra = read_particles(real / "shiny.star")
    lb, rb = read_particles(virt / "shiny.star")
    i = la.index("ImageName")
    assert la == lb and [r[:i] + r[i + 1:] for r in ra] == [r[:i] + r[i + 1:] for r in rb]
    assert [r[i].replace("real", "virtual").replace(".mrcs", ".vstack") for r in ra] == [r[i] for r in rb]
    assert optics(real / "shiny.star") == optics(virt / "shiny.star")
    a = materialise(relion_bin, d, real / "shiny.star", "cmp_real")
    b = materialise(relion_bin, d, virt / "shiny.star", "cmp_virtual")
    # the frame sums are rounded to 16 bits before the CTF (so they can be cached): close, not bitwise
    assert a.shape == b.shape and np.abs(a - b).max() <= 2e-3 * np.abs(a).max(), \
        f"max |diff| {np.abs(a - b).max()} of max |a| {np.abs(a).max()}"
    # bitwise the same through the cache (written by the read above) and without it
    assert np.array_equal(b, materialise(relion_bin, d, virt / "shiny.star", "cmp_cached"))
    assert np.array_equal(b, materialise(relion_bin, d, virt / "shiny.star", "cmp_nocache",
                                         {"RELION_VPARTICLE_CACHE": "off"}))
    # the environment variable makes --virtual the default
    env = polish(relion_bin, d, "parts1.star", "Polish/env", extra, {"RELION_VIRTUAL_POLISH": "1"})
    assert len(list(env.rglob("*_shiny.vstack"))) == 2


def test_ctf_multiply_chooses_its_window(relion_bin, polish_project):
    d = polish_project
    plain = polish(relion_bin, d, "parts1.star", "Polish/plain")
    auto = polish(relion_bin, d, "parts1.star", "Polish/auto", ["--ctf_multiply"])
    assert "window of" in polish.stdout and "cropped to the particle box" in polish.stdout
    o = optics(auto / "shiny.star")
    assert o["CtfDataAreCtfPremultiplied"] == "1" and o["ImageSize"] == optics(plain / "shiny.star")["ImageSize"]
    assert shiny(auto).shape == shiny(plain).shape


def test_per_particle_windows(relion_bin, polish_project):
    d = polish_project
    # a per-particle window is at most 4x the final box (256 here): the maximum window must be larger to differ
    extra = ["--ctf_multiply", "--window", "320", "--crop", "64", "--virtual"]
    win = polish(relion_bin, d, "parts1.star", "Polish/windows", extra)
    assert "Per-particle windows" in polish.stdout, polish.stdout[-2000:]
    max_window = {"RELION_VPOLISH_PARTICLE_WINDOWS": "0"}
    mx = polish(relion_bin, d, "parts1.star", "Polish/max_window", extra, max_window)
    a = materialise(relion_bin, d, win / "shiny.star", "win")
    b = materialise(relion_bin, d, mx / "shiny.star", "max")
    # smaller windows for the lower defoci: close to the maximum window, not the same (0.965 on these noise-only
    # 64 px particles, where the weak CTF tails reach different noise; 0.999 on EMPIAR-10940 at 384 px)
    assert a.shape == b.shape and not np.array_equal(a, b)
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.95
    # real Polish (stacks) follows the same switch: within the 16-bit rounding of the sums either way
    for env, v in ((None, a), (max_window, b)):
        r = shiny(polish(relion_bin, d, "parts1.star", "Polish/real_" + ("max" if env else "win"), extra[:-1], env))
        assert r.shape == v.shape and np.abs(r - v).max() <= 2e-3 * np.abs(r).max()
    # bitwise the same through the cache, without it, and with the frame sums made from the movie each time
    assert np.array_equal(a, materialise(relion_bin, d, win / "shiny.star", "win_cached"))
    assert np.array_equal(a, materialise(relion_bin, d, win / "shiny.star", "win_nocache",
                                         {"RELION_VPARTICLE_CACHE": "off"}))
    assert np.array_equal(a, materialise(relion_bin, d, win / "shiny.star", "win_nosums",
                                         {"RELION_VPARTICLE_CACHE": str(d / "vcache_nosums"), "RELION_VPOLISH_SUMS": "0"}))


def test_virtual_polished_particles_follow_the_ctf(relion_bin, polish_project):
    d = polish_project
    virt = polish(relion_bin, d, "parts1.star", "Polish/virtual", ["--ctf_multiply", "--virtual"])
    before = materialise(relion_bin, d, virt / "shiny.star", "before")
    text = (virt / "shiny.star").read_text()
    labels, _ = read_particles(virt / "shiny.star")
    k = labels.index("DefocusU")
    # 2000 A more defocus on every particle: the premultiplied particles change with it
    out_lines = []
    for line in text.splitlines():
        f = line.split()
        if len(f) == len(labels) and "@" in line:
            f[k] = str(float(f[k]) + 2000)
            line = " ".join(f)
        out_lines.append(line)
    (d / "changed.star").write_text("\n".join(out_lines) + "\n")
    after = materialise(relion_bin, d, d / "changed.star", "after")
    assert after.shape == before.shape and not np.array_equal(after, before)
    # and the recorded CTF still gives the original particles
    assert np.array_equal(before, materialise(relion_bin, d, virt / "shiny.star", "again"))


@pytest.mark.parametrize("what", ["record", "gain"])
def test_virtual_polished_particles_check_the_movie_and_gain(relion_bin, polish_project, what):
    d = polish_project
    virt = polish(relion_bin, d, "parts1.star", "Polish/virtual", ["--virtual"])
    materialise(relion_bin, d, virt / "shiny.star", "ok")
    # a changed motion record (re-run motion correction) or gain reference is refused, not misread
    files = sorted(d.rglob("MotionCorr/job/Movies/*.star")) if what == "record" else [d / "Movies" / "gain.mrc"]
    assert files
    for f in files:
        f.write_bytes(f.read_bytes() + (b"\n# changed\n" if what == "record" else b"\0\0\0\0"))
    env = dict(vma._env(), RELION_VPARTICLE_CACHE="off")
    r = subprocess.run([str(relion_bin / "relion_stack_create"), "--i", str(virt / "shiny.star"), "--o", "bad"],
                       cwd=d, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "has changed" in r.stdout + r.stderr


def shifted_coordinates(star, out, dx):
    """`star` with `dx` added to every particle's CoordinateX and CoordinateY."""
    labels, _ = read_particles(star)
    kx, ky = labels.index("CoordinateX"), labels.index("CoordinateY")
    lines = []
    for line in Path(star).read_text().splitlines():
        f = line.split()
        if len(f) == len(labels) and "@" in line:
            f[kx], f[ky] = str(float(f[kx]) + dx), str(float(f[ky]) + dx)
            line = " ".join(f)
        lines.append(line)
    Path(out).write_text("\n".join(lines) + "\n")
    return out


@pytest.mark.parametrize("centre", ["round", "trunc"])
def test_fractional_coordinates_use_the_centre_their_offsets_refer_to(relion_bin, polish_project, centre):
    """A particle at x + 0.6 was cut around x + 1 by this RELION's Extract (round) and around x by upstream's
    (trunc); Polish must cut the same box, so it gives the particle at that whole pixel, with the same offsets."""
    d = polish_project
    whole = shifted_coordinates(d / "parts0.star", d / "whole.star", 1.0 if centre == "round" else 0.0)
    frac = shifted_coordinates(d / "parts0.star", d / "frac.star", 0.6)
    ref = polish(relion_bin, d, "whole.star", "Polish/whole")
    env = {"RELION_COORD_CENTRE": "trunc"} if centre == "trunc" else None
    out = polish(relion_bin, d, "frac.star", "Polish/frac", env_extra=env)
    stdout = polish.stdout
    assert np.array_equal(shiny(ref), shiny(out))
    la, ra = read_particles(ref / "shiny.star")
    lb, rb = read_particles(out / "shiny.star")
    i = la.index("ImageName")
    assert la == lb and [r[:i] + r[i + 1:] for r in ra] == [r[:i] + r[i + 1:] for r in rb]
    assert "10 particles have fractional coordinates" in stdout, stdout[-2000:]
