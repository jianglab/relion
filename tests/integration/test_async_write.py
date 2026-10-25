"""Image outputs written by the background writer are byte for byte what the synchronous code writes.

relion_preprocess (Extract), relion_particle_subtract, relion_motion_refine (Polish) and relion_run_motioncorr hand
their image files to the background writer (RELION_ASYNC_WRITE=0 makes them synchronous). Extract and Subtract also
write a whole stack in one pass instead of opening, rewriting the header of and closing the file for every particle.
Each test runs the program with the default, with RELION_ASYNC_WRITE=0 and with a tiny chunk size
(RELION_ASYNC_STACK_CHUNK_KB=1: every particle is flushed alone), in copies of one project, and compares every file.
MRC headers carry a time stamp in the first label, which is masked.

When RELION_BASELINE_BIN_DIR names the bin directory of a build without the background writer, its output is
compared as well: that is the proof that the files did not change.

Run with RELION_BIN_DIR pointing at a built RELION bin directory.
"""
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from test_virtual_particles import micrograph, write_loop, write_mrc as write_plain_mrc, NX, NY
import test_virtual_movie_averages as vma

LABEL = slice(224, 224 + 80)   # the first MRC label: "Relion <version>  <date time>"


def _bytes(path):
    raw = bytearray(Path(path).read_bytes())
    if Path(path).suffix in (".mrc", ".mrcs"):
        raw[LABEL] = bytes(80)
    return bytes(raw)


def _files(root):
    return sorted(p.relative_to(root) for p in Path(root).rglob("*")
                  if p.is_file() and p.suffix in (".mrc", ".mrcs", ".star") and not p.name.startswith(".tmp_"))


def assert_same_tree(a, b, what=""):
    fa, fb = _files(a), _files(b)
    assert fa == fb, f"{what}: different files: {set(fa) ^ set(fb)}"
    assert fa, f"{what}: no files written"
    for f in fa:
        assert _bytes(Path(a) / f) == _bytes(Path(b) / f), f"{what}: {f} differs"


def baseline_bin():
    d = os.environ.get("RELION_BASELINE_BIN_DIR")
    return Path(d) if d else None


def run_program(bin_dir, cwd, cmd, env_extra):
    env = dict(os.environ)
    for k in ("RELION_ASYNC_WRITE", "RELION_ASYNC_STACK_CHUNK_KB", "RELION_VIRTUAL_PARTICLES", "RELION_VIRTUAL_MOVIE_AVERAGES"):
        env.pop(k, None)
    env["RELION_VPARTICLE_CACHE"] = str(Path(cwd) / "vcache")
    env.update(env_extra)
    return subprocess.run([str(bin_dir / cmd[0])] + cmd[1:], cwd=cwd, env=env, capture_output=True, text=True, timeout=900)


VARIANTS = {
    "default": {},
    "synchronous": {"RELION_ASYNC_WRITE": "0"},
    "tiny_chunks": {"RELION_ASYNC_STACK_CHUNK_KB": "1"},
}


def run_variants(relion_bin, project, cmd, outdir):
    """Runs cmd in a copy of project for each variant (and the baseline); returns {name: copy of project}"""
    runs = {}
    variants = dict(VARIANTS)
    if baseline_bin():
        variants["baseline"] = {}
    for name, env in variants.items():
        d = Path(str(project) + "_" + name)
        shutil.copytree(project, d)
        r = run_program(baseline_bin() if name == "baseline" else relion_bin, d, cmd, env)
        assert r.returncode == 0, f"{name}: {r.stdout[-2000:]}{r.stderr[-2000:]}"
        runs[name] = d / outdir
    return runs


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

@pytest.fixture
def extract_project(tmp_path):
    (tmp_path / "Micrographs").mkdir()
    (tmp_path / "Coords").mkdir()
    # 6, 1, 2 and 5 particles: stacks of many images, of one, of two (first and last only) and with edge boxes
    coords = {
        "mic001": [(180, 150), (60.4, 70.7), (8, 9), (NX - 4, 150), (180, NY - 6.2), (250.9, 99.5)],
        "mic002": [(100, 100)],
        "mic003": [(300.5, 210.2), (3.7, NY - 2)],
        "mic004": [(200, 12), (NX - 30, 40), (120, 130), (130, 140), (70, 220)],
    }
    mic_rows, coord_rows = [], []
    for k, (name, cs) in enumerate(coords.items()):
        write_plain_mrc(tmp_path / "Micrographs" / f"{name}.mrc", micrograph(k + 1), mode=12 if k == 1 else 2)
        write_loop(tmp_path / "Coords" / f"{name}.star", [("", ["CoordinateX", "CoordinateY"], cs)])
        mic_rows.append([f"Micrographs/{name}.mrc", 1, 12000 + 500 * k, 11800 + 500 * k, 30.0])
        coord_rows.append([f"Micrographs/{name}.mrc", f"Coords/{name}.star"])
    write_loop(tmp_path / "micrographs_ctf.star", [
        ("optics", ["OpticsGroupName", "OpticsGroup", "MicrographPixelSize", "Voltage", "SphericalAberration", "AmplitudeContrast"],
         [["opticsGroup1", 1, 1.5, 300, 2.7, 0.1]]),
        ("micrographs", ["MicrographName", "OpticsGroup", "DefocusU", "DefocusV", "DefocusAngle"], mic_rows)])
    write_loop(tmp_path / "coords.star", [("coordinate_files", ["MicrographName", "MicrographCoordinates"], coord_rows)])
    (tmp_path / "Extract").mkdir()
    return tmp_path


EXTRACT_OPTIONS = {
    "binned_float16": ["--extract_size", "64", "--scale", "48", "--norm", "--bg_radius", "18",
                       "--white_dust", "-1", "--black_dust", "-1", "--invert_contrast", "--float16"],
    "fullsize_float32": ["--extract_size", "56", "--norm", "--bg_radius", "20", "--no_ramp"],
    "ctf_flipped": ["--extract_size", "64", "--phase_flip", "--norm", "--bg_radius", "20"],
}


@pytest.mark.parametrize("case", list(EXTRACT_OPTIONS))
def test_extract_stacks_are_identical(relion_bin, extract_project, case):
    cmd = ["relion_preprocess", "--i", "micrographs_ctf.star", "--part_star", "Extract/job/particles.star",
           "--part_dir", "Extract/job/", "--extract", "--coord_list", "coords.star", "--no_virtual"] + EXTRACT_OPTIONS[case]
    runs = run_variants(relion_bin, extract_project, cmd, "Extract")
    assert len([f for f in _files(runs["default"]) if f.suffix == ".mrcs"]) == 4
    for name, d in runs.items():
        if name != "default":
            assert_same_tree(runs["default"], d, f"extract {case}: default vs {name}")


@pytest.mark.parametrize("ranks", [2])
def test_extract_mpi_is_identical(relion_bin, extract_project, ranks):
    if shutil.which("mpirun") is None or not (relion_bin / "relion_preprocess_mpi").exists():
        pytest.skip("no MPI build")
    cmd = ["relion_preprocess", "--i", "micrographs_ctf.star", "--part_star", "Extract/job/particles.star",
           "--part_dir", "Extract/job/", "--extract", "--coord_list", "coords.star", "--no_virtual",
           "--extract_size", "48", "--norm", "--bg_radius", "16"]
    outs = {}
    for name, env in (("default", {}), ("synchronous", {"RELION_ASYNC_WRITE": "0"})):
        d = Path(str(extract_project) + "_mpi_" + name)
        shutil.copytree(extract_project, d)
        mpi_cmd = ["mpirun", "--oversubscribe", "-n", str(ranks + 1), str(relion_bin / "relion_preprocess_mpi")] + cmd[1:]
        e = dict(os.environ, OMPI_MCA_rmaps_base_oversubscribe="1", **env)
        e.pop("RELION_ASYNC_WRITE", None) if not env else None
        r = subprocess.run(mpi_cmd, cwd=d, env=e, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        outs[name] = d / "Extract"
    assert_same_tree(outs["default"], outs["synchronous"], "extract mpi")


def test_extract_write_failure_fails_the_job(relion_bin, extract_project):
    """A stack that cannot be written (its name is taken by a directory) must end the program with an error naming
    the file, not just print a message from the writer thread"""
    cmd = ["relion_preprocess", "--i", "micrographs_ctf.star", "--part_star", "Extract/job/particles.star",
           "--part_dir", "Extract/job/", "--extract", "--coord_list", "coords.star", "--no_virtual", "--extract_size", "48"]
    for env in ({}, {"RELION_ASYNC_WRITE": "0"}):
        d = Path(str(extract_project) + "_fail" + ("_sync" if env else ""))
        shutil.copytree(extract_project, d)
        blocker = d / "Extract/job/Micrographs/mic002.mrcs"
        blocker.mkdir(parents=True)
        (blocker / "keep").write_text("x")
        r = run_program(relion_bin, d, cmd, env)
        assert r.returncode != 0, "the job reported success although a stack could not be written"
        assert "mic002.mrcs" in r.stderr + r.stdout
        assert not (d / "Extract/job/particles.star").exists()
        assert not list((d / "Extract/job").rglob(".tmp_*")), "temporary file left behind"


# ---------------------------------------------------------------------------
# MotionCorr
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("extra", [["--dose_weighting", "--float16"], ["--dose_weighting", "--even_odd_split"], []],
                         ids=["dose_weighted_float16", "even_odd", "plain"])
def test_motioncorr_outputs_are_identical(relion_bin, tmp_path, extra):
    vma._write_movies(tmp_path)
    cmd = ["relion_run_motioncorr", "--i", "movies.star", "--o", "MotionCorr/job/", "--use_own", "--j", "1",
           "--first_frame_sum", "1", "--last_frame_sum", "-1", "--bfactor", "150", "--dose_per_frame", "1.3",
           "--preexposure", "0", "--patch_x", "3", "--patch_y", "3", "--gainref", "Movies/gain.mrc", "--gain_rot", "0",
           "--gain_flip", "0", "--grouping_for_ps", "3", "--ps_size", "128"] + extra
    runs = run_variants(relion_bin, tmp_path, cmd, "MotionCorr")
    files = _files(runs["default"])
    assert any(f.name.endswith(".mrc") for f in files) and any("PS" in f.name for f in files)
    for name, d in runs.items():
        if name != "default":
            assert_same_tree(runs["default"], d, f"motioncorr: default vs {name}")


# ---------------------------------------------------------------------------
# Subtract
# ---------------------------------------------------------------------------

@pytest.fixture
def subtract_project(test_data_dir, relion_bin, relion_runner):
    """A small Class3D run (optimiser.star) and a mask of its central part"""
    from fixtures import generate_test_dataset, write_volume_mrc
    paths = generate_test_dataset(str(test_data_dir / "data"), relion_bin, relion_runner, num_particles=40, size=32)
    out = test_data_dir / "c3d"
    out.mkdir()
    cmd = [str(relion_bin / "relion_refine"), "--i", str(paths["star_file"]), "--o", str(out / "run"),
           "--ref", str(paths["initial_model"]), "--K", "2", "--iter", "1", "--ini_high", "20", "--tau2_fudge", "4",
           "--particle_diameter", "25", "--oversampling", "1", "--healpix_order", "1", "--offset_range", "3",
           "--offset_step", "2", "--sym", "C1", "--j", "1", "--random_seed", "7", "--dont_combine_weights_via_disc",
           "--pool", "10"]
    r = subprocess.run(cmd, cwd=out, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    g = np.mgrid[0:32, 0:32, 0:32] - 16
    mask = (np.sqrt((g ** 2).sum(axis=0)) < 8).astype(np.float32)
    write_volume_mrc(mask, str(out / "mask.mrc"))
    return out


@pytest.mark.parametrize("extra", [[], ["--float16"], ["--new_box", "24"]], ids=["float32", "float16", "rebox"])
@pytest.mark.parametrize("ranks", [0, 3])
def test_subtract_stacks_are_identical(relion_bin, subtract_project, extra, ranks):
    if ranks and (shutil.which("mpirun") is None or not (relion_bin / "relion_particle_subtract_mpi").exists()):
        pytest.skip("no MPI build")
    program = "relion_particle_subtract" + ("_mpi" if ranks else "")
    cmd = [program, "--i", "run_it001_optimiser.star", "--o", "Subtract/job/", "--mask", "mask.mrc"] + extra
    if ranks:
        cmd = ["mpirun", "--oversubscribe", "-n", str(ranks)] + [str(relion_bin / cmd[0])] + cmd[1:]
    runs = {}
    variants = dict(VARIANTS)
    if baseline_bin() and (baseline_bin() / program).exists():
        variants["baseline"] = {}
    for name, env_extra in variants.items():
        d = Path(str(subtract_project) + "_" + name)
        shutil.copytree(subtract_project, d)
        c = list(cmd)
        if ranks:
            c = c[:4] + [str((baseline_bin() if name == "baseline" else relion_bin) / program)] + c[5:]
            env = dict(os.environ, OMPI_MCA_rmaps_base_oversubscribe="1", RELION_INTERPOLATION="linear")
            for k in ("RELION_ASYNC_WRITE", "RELION_ASYNC_STACK_CHUNK_KB"):
                env.pop(k, None)
            env.update(env_extra)
            r = subprocess.run(c, cwd=d, env=env, capture_output=True, text=True, timeout=900)
        else:
            r = run_program(baseline_bin() if name == "baseline" else relion_bin, d, c, dict(env_extra, RELION_INTERPOLATION="linear"))
        assert r.returncode == 0, f"{name}: {r.stdout[-2000:]}{r.stderr[-2000:]}"
        runs[name] = d / "Subtract"
    stacks = [f for f in _files(runs["default"]) if f.suffix == ".mrcs"]
    assert stacks
    for name, d in runs.items():
        if name != "default":
            assert_same_tree(runs["default"], d, f"subtract {ranks} ranks {extra}: default vs {name}")


def _blocked(project, name, rel):
    """A copy of project in which the file rel is taken by a directory, so that it cannot be written"""
    d = Path(str(project) + "_" + name)
    shutil.copytree(project, d)
    (d / rel).mkdir(parents=True)
    ((d / rel) / "keep").write_text("x")
    return d


@pytest.mark.parametrize("env", [{}, {"RELION_ASYNC_WRITE": "0"}], ids=["async", "sync"])
def test_subtract_write_failure_fails_the_job(relion_bin, subtract_project, env):
    d = _blocked(subtract_project, "blocked", "Subtract/job/Particles/subtracted_opticsgroup1.mrcs")
    r = run_program(relion_bin, d, ["relion_particle_subtract", "--i", "run_it001_optimiser.star", "--o", "Subtract/job/",
                                    "--mask", "mask.mrc"], dict(env, RELION_INTERPOLATION="linear"))
    assert r.returncode != 0, "the job reported success although a stack could not be written"
    assert "subtracted_opticsgroup1.mrcs" in r.stderr + r.stdout
    assert not (d / "Subtract/job/particles_subtracted.star").exists()


@pytest.mark.parametrize("env", [{}, {"RELION_ASYNC_WRITE": "0"}], ids=["async", "sync"])
def test_motioncorr_write_failure_fails_the_job(relion_bin, tmp_path, env):
    vma._write_movies(tmp_path)
    d = _blocked(tmp_path, "blocked", "MotionCorr/job/Movies/movie00.mrc")
    r = run_program(relion_bin, d, ["relion_run_motioncorr", "--i", "movies.star", "--o", "MotionCorr/job/", "--use_own", "--j", "1",
                                    "--first_frame_sum", "1", "--last_frame_sum", "-1", "--bfactor", "150", "--dose_per_frame", "1.3",
                                    "--preexposure", "0", "--patch_x", "3", "--patch_y", "3", "--gainref", "Movies/gain.mrc",
                                    "--gain_rot", "0", "--gain_flip", "0", "--grouping_for_ps", "3", "--ps_size", "128"], env)
    assert r.returncode != 0, "the job reported success although a micrograph could not be written"
    assert "movie00.mrc" in r.stderr + r.stdout
