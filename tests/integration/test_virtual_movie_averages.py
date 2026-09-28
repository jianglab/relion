#!/usr/bin/env python3
"""
Virtual movie averages (RELION_VIRTUAL_MOVIE_AVERAGES).

=real: relion_run_motioncorr --use_own writes a micrograph that
relion_regenerate_micrograph must reproduce bit for bit from the movie and the
motion record written next to it, whatever the thread count; a record written
without the switch must be refused.

=1: MotionCorr writes a descriptor under the micrograph's name instead; reading
it (Image::read, and the virtual-particle micrograph reader) must give exactly
the pixels of the real micrograph, through the cache or without it, and
particles extracted from it - real or virtual - must be those extracted from
the real micrograph.

Synthetic movies: a drifting textured specimen with Poisson noise, a gain reference
with one dead pixel and a few hot pixels, so that alignment, the local model,
dose weighting and the bad-pixel fill are all exercised.
"""

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fixtures import write_mrc

NX, NY, NFRAMES = 512, 480, 16
ANGPIX = 1.0


def _write_movies(d: Path, n_movies=2):
    rng = np.random.default_rng(7)
    (d / "Movies").mkdir()
    # Specimen: smooth random texture, large enough for the drift to wrap nothing important
    spec = rng.normal(size=(NY + 64, NX + 64))
    f = np.fft.rfft2(spec)
    ky = np.fft.fftfreq(spec.shape[0])[:, None]
    kx = np.fft.rfftfreq(spec.shape[1])[None, :]
    spec = np.fft.irfft2(f * np.exp(-(kx ** 2 + ky ** 2) / (2 * 0.05 ** 2)), s=spec.shape)
    spec = 2.0 + spec / spec.std()
    gain = rng.uniform(0.9, 1.1, size=(NY, NX)).astype(np.float32)
    gain[100, 200] = 0.0                                   # a dead pixel
    write_mrc(1.0 / np.where(gain > 0, gain, 1.0) * (gain > 0), d / "Movies" / "gain.mrc", angpix=ANGPIX)
    names = []
    for m in range(n_movies):
        frames = []
        for i in range(NFRAMES):
            dx, dy = 0.7 * i + 0.3 * m, 0.4 * i                # whole-frame drift, in pixels
            x0, y0 = 32 + int(round(dx)), 32 + int(round(dy))
            clean = spec[y0:y0 + NY, x0:x0 + NX].clip(0.05) * gain
            frame = rng.poisson(clean).astype(np.float32)
            frame[50 + m, 60] = 500.0                           # a hot pixel
            frames.append(frame)
        name = f"Movies/movie{m:02d}.mrcs"
        write_mrc(np.stack(frames), d / name, angpix=ANGPIX)
        names.append(name)
    with open(d / "movies.star", "w") as fh:
        fh.write("\ndata_optics\n\nloop_\n_rlnOpticsGroupName #1\n_rlnOpticsGroup #2\n"
                 "_rlnMicrographOriginalPixelSize #3\n_rlnVoltage #4\n_rlnSphericalAberration #5\n"
                 "_rlnAmplitudeContrast #6\n")
        fh.write(f"opticsGroup1 1 {ANGPIX:.6f} 300.000000 2.700000 0.100000\n\n")
        fh.write("\ndata_movies\n\nloop_\n_rlnMicrographMovieName #1\n_rlnOpticsGroup #2\n")
        for n in names:
            fh.write(f"{n} 1\n")
    return names


def _env(mode=None, cache=None):
    env = dict(os.environ)
    for k in ("RELION_VIRTUAL_MOVIE_AVERAGES", "RELION_VMOVIE_AVERAGE_CACHE",
              "RELION_VIRTUAL_PARTICLES", "RELION_VPARTICLE_CACHE"):
        env.pop(k, None)
    if mode is not None:
        env["RELION_VIRTUAL_MOVIE_AVERAGES"] = mode
    if cache is not None:
        env["RELION_VMOVIE_AVERAGE_CACHE"] = cache
    return env


def _motioncorr(relion_bin, d: Path, out: str, extra, mode="real", threads=4):
    env = _env(mode)
    cmd = [str(relion_bin / "relion_run_motioncorr"), "--i", "movies.star", "--o", out + "/",
           "--use_own", "--j", str(threads), "--first_frame_sum", "1", "--last_frame_sum", "-1",
           "--bfactor", "150", "--dose_per_frame", "1.3", "--preexposure", "0",
           "--patch_x", "3", "--patch_y", "3", "--gainref", "Movies/gain.mrc",
           "--gain_rot", "0", "--gain_flip", "0", "--grouping_for_ps", "3", "--ps_size", "128"] + extra
    r = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def _regenerate(relion_bin, d: Path, star: Path, mic: Path, threads, float16):
    out = d / f"regen_j{threads}.mrc"
    cmd = [str(relion_bin / "relion_regenerate_micrograph"), "--i", str(star), "--o", str(out),
           "--j", str(threads), "--compare", str(mic)] + (["--float16"] if float16 else [])
    r = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
    out.unlink(missing_ok=True)
    return r


CASES = {
    "dose_weighted_float16": ["--dose_weighting", "--float16"],
    "dose_weighted_float32": ["--dose_weighting"],
    "not_dose_weighted": [],
    "binned_early": ["--dose_weighting", "--float16", "--bin_factor", "2"],
    "binned_late": ["--dose_weighting", "--bin_factor", "2", "--no_early_binning"],
}


@pytest.mark.integration
class TestReproducibleMotionCorr:

    @pytest.mark.parametrize("case", sorted(CASES))
    def test_regenerated_micrograph_is_identical(self, test_data_dir, relion_bin, case):
        d = test_data_dir
        names = _write_movies(d)
        extra = CASES[case]
        _motioncorr(relion_bin, d, "MC", extra, threads=3)
        for n in names:
            stem = d / "MC" / Path(n).with_suffix("")
            star, mic = stem.with_suffix(".star"), stem.with_suffix(".mrc")
            assert "rlnMicrographSumRecipe" in star.read_text()
            for j in (1, 4):
                r = _regenerate(relion_bin, d, star, mic, j, "--float16" in extra)
                assert r.returncode == 0 and "IDENTICAL" in r.stdout, \
                    f"{case} {n} --j {j}: {r.stdout[-500:]} {r.stderr[-500:]}"

    def test_thread_count_of_motioncorr_does_not_matter(self, test_data_dir, relion_bin):
        d = test_data_dir
        names = _write_movies(d, n_movies=1)
        _motioncorr(relion_bin, d, "MC1", ["--dose_weighting"], threads=1)
        _motioncorr(relion_bin, d, "MC5", ["--dose_weighting"], threads=5)
        stem = Path(names[0]).with_suffix("")
        # Pixel data only: the header label carries the time of writing
        a = (d / "MC1" / stem).with_suffix(".mrc").read_bytes()[1024:]
        b = (d / "MC5" / stem).with_suffix(".mrc").read_bytes()[1024:]
        assert a == b

    def test_legacy_record_is_refused(self, test_data_dir, relion_bin):
        d = test_data_dir
        names = _write_movies(d, n_movies=1)
        _motioncorr(relion_bin, d, "MC", ["--dose_weighting"], mode=None)
        star = (d / "MC" / Path(names[0]).with_suffix("")).with_suffix(".star")
        text = star.read_text()
        assert "rlnMicrographSumRecipe" not in text
        # The legacy record is written exactly as before: at most 6 decimals
        assert re.search(r"\d\.\d{7,}", text) is None
        r = _regenerate(relion_bin, d, star, star.with_suffix(".mrc"), 1, False)
        assert r.returncode != 0
        assert "without a sum recipe" in (r.stdout + r.stderr)


# ---------------------------------------------------------------------------
# =1: virtual movie averages

MAGIC = b"# RELION virtual movie average\n"


def _read_mrc(path):
    raw = Path(path).read_bytes()
    h = np.frombuffer(raw[:1024], dtype=np.int32)
    nx, ny, nz, mode, nsymbt = h[0], h[1], h[2], h[3], h[23]
    dtype = {2: np.float32, 12: np.float16}[int(mode)]
    return np.frombuffer(raw[1024 + nsymbt:], dtype=dtype)[:nx * ny * nz].reshape(nz, ny, nx).astype(np.float64)


def _run(relion_bin, d, args, env):
    r = subprocess.run([str(relion_bin / args[0])] + args[1:], cwd=d, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r


def _read_through_relion(relion_bin, d, mic, env, out="read_back.mrc"):
    """The micrograph as RELION programs see it (Image::read), written as float32."""
    _run(relion_bin, d, ["relion_image_handler", "--i", str(mic), "--o", out, "--multiply_constant", "1"], env)
    a = _read_mrc(d / out)
    (d / out).unlink()
    return a


def _cache_entries(path):
    return sorted(p for p in Path(path).rglob("*.mrc")) if Path(path).exists() else []


VIRTUAL_CASES = ["dose_weighted_float16", "dose_weighted_float32", "binned_late"]


@pytest.mark.integration
class TestVirtualMovieAverages:

    @pytest.mark.parametrize("case", VIRTUAL_CASES)
    def test_descriptor_reads_as_the_real_micrograph(self, test_data_dir, relion_bin, case):
        d = test_data_dir
        names = _write_movies(d)
        _motioncorr(relion_bin, d, "MCR", CASES[case], mode="real")
        _motioncorr(relion_bin, d, "MCV", CASES[case], mode="1")
        for n in names:
            stem = Path(n).with_suffix("")
            real = (d / "MCR" / stem).with_suffix(".mrc")
            desc = (d / "MCV" / stem).with_suffix(".mrc")
            assert desc.read_bytes().startswith(MAGIC)
            assert desc.stat().st_size < real.stat().st_size / 10
            expected = _read_mrc(real)

            # Default cache: Cache/virtual_movie_averages/ in the project
            got = _read_through_relion(relion_bin, d, desc.relative_to(d), _env())
            assert np.array_equal(got, expected), f"{case} {n}: max |diff| {np.abs(got - expected).max()}"
        entries = _cache_entries(d / "Cache/virtual_movie_averages")
        assert len(entries) == len(names)

        # From the cache, after deleting it, and with the cache off
        n = names[0]
        desc = (d / "MCV" / Path(n).with_suffix("")).with_suffix(".mrc").relative_to(d)
        expected = _read_mrc((d / "MCR" / Path(n).with_suffix("")).with_suffix(".mrc"))
        assert np.array_equal(_read_through_relion(relion_bin, d, desc, _env()), expected)
        subprocess.run(["rm", "-rf", str(d / "Cache")], check=True)
        assert np.array_equal(_read_through_relion(relion_bin, d, desc, _env()), expected)
        subprocess.run(["rm", "-rf", str(d / "Cache")], check=True)
        assert np.array_equal(_read_through_relion(relion_bin, d, desc, _env(cache="off")), expected)
        assert not (d / "Cache").exists()

    def test_particles_from_virtual_micrographs(self, test_data_dir, relion_bin):
        d = test_data_dir
        names = _write_movies(d)
        extra = CASES["dose_weighted_float16"]
        _motioncorr(relion_bin, d, "MCR", extra, mode="real")
        _motioncorr(relion_bin, d, "MCV", extra, mode="1")
        (d / "Coords").mkdir()
        centres = [(100, 100), (250, 240), (400, 300), (60, 420), (500, 20)]
        for mc in ("MCR", "MCV"):
            rows = []
            for n in names:
                stem = Path(n).stem
                cf = d / "Coords" / f"{mc}_{stem}.star"
                cf.write_text("\ndata_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n"
                              + "".join(f"{x} {y}\n" for x, y in centres))
                rows.append(f"{mc}/Movies/{stem}.mrc Coords/{cf.name}\n")
            (d / f"coords_{mc}.star").write_text(
                "\ndata_coordinate_files\n\nloop_\n_rlnMicrographName #1\n_rlnMicrographCoordinates #2\n"
                + "".join(rows))

        opts = ["--extract", "--extract_size", "64", "--scale", "48", "--norm", "--bg_radius", "18",
                "--invert_contrast", "--float16"]

        def extract(mc, out, virtual, env):
            (d / out).mkdir(parents=True)
            _run(relion_bin, d, ["relion_preprocess", "--i", f"{mc}/corrected_micrographs.star",
                                 "--coord_list", f"coords_{mc}.star", "--part_star", f"{out}/particles.star",
                                 "--part_dir", f"{out}/"] + opts + (["--virtual"] if virtual else ["--no_virtual"]), env)
            _run(relion_bin, d, ["relion_stack_create", "--i", f"{out}/particles.star", "--o", f"{out}/all"], env)
            return _read_mrc(d / out / "all.mrcs")

        env = _env()
        env["RELION_VPARTICLE_CACHE"] = str(d / "vpcache")
        ref = extract("MCR", "Extract/real_from_real", False, env)
        assert ref.shape[0] == len(names) * len(centres)

        # Virtual particles need only the header of a virtual micrograph to be written
        (d / "Extract/virtual_from_virtual").mkdir(parents=True)
        _run(relion_bin, d, ["relion_preprocess", "--i", "MCV/corrected_micrographs.star",
                             "--coord_list", "coords_MCV.star", "--part_star", "Extract/virtual_from_virtual/particles.star",
                             "--part_dir", "Extract/virtual_from_virtual/"] + opts + ["--virtual"], env)
        assert _cache_entries(d / "Cache/virtual_movie_averages") == []
        subprocess.run(["rm", "-rf", str(d / "Extract/virtual_from_virtual")], check=True)

        for out, virtual in (("Extract/real_from_virtual", False), ("Extract/virtual_from_virtual", True)):
            got = extract("MCV", out, virtual, env)
            assert np.array_equal(got, ref), f"{out}: max |diff| {np.abs(got - ref).max()}"
