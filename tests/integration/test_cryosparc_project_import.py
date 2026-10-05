"""Importing a whole CryoSPARC project as a RELION project.

These build a miniature CryoSPARC project on disk - job.json files with the
same shape CryoSPARC writes, plus .cs tables - and check that
relion_import_cryosparc walks it correctly. The point is the project-level
behaviour that unit tests cannot reach: which jobs get imported, in what order,
which round of an iterative job is treated as final, and what the emitted
pipeline says.

Run with RELION_BIN_DIR pointing at a built RELION bin directory.
"""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def importer():
    directory = Path(os.environ.get("RELION_BIN_DIR",
                                    Path(__file__).resolve().parents[2] / "build/bin"))
    binary = directory / "relion_import_cryosparc"
    if not binary.exists():
        pytest.skip("Build relion_import_cryosparc and set RELION_BIN_DIR")
    return str(binary)


def _write_cs(path, rows, extra_fields=()):
    """A minimal particle-ish .cs table (a NumPy structured array)."""
    dtype = [("uid", "<u8"),
             ("blob/path", "S96"),
             ("blob/idx", "<u4"),
             ("blob/psize_A", "<f4"),
             ("ctf/df1_A", "<f4"),
             ("ctf/df2_A", "<f4"),
             ("ctf/df_angle_rad", "<f4"),
             ("ctf/phase_shift_rad", "<f4"),
             ("ctf/accel_kv", "<f4"),
             ("ctf/cs_mm", "<f4"),
             ("ctf/amp_contrast", "<f4")]
    dtype += list(extra_fields)
    a = np.zeros(rows, dtype=dtype)
    for i in range(rows):
        a[i]["uid"] = i + 1
        a[i]["blob/path"] = f"J1/stack_{i:03d}.mrc".encode()
        a[i]["blob/idx"] = i
        a[i]["blob/psize_A"] = 1.0
        a[i]["ctf/df1_A"] = 10000.0 + i
        a[i]["ctf/df2_A"] = 10100.0 + i
        a[i]["ctf/accel_kv"] = 300.0
        a[i]["ctf/cs_mm"] = 2.7
        a[i]["ctf/amp_contrast"] = 0.1
    # np.save appends .npy; write then rename, matching the other cs tests
    np.save(str(path) + ".npy", a, allow_pickle=False)
    os.replace(str(path) + ".npy", str(path))


def _job_json(path, uid, job_type, status, parents, results):
    """`results` is a list of (group, name, type, [metafiles], [versions]),
    optionally followed by a passthrough flag."""
    groups = {}
    out_results = []
    for res in results:
        group, name, rtype, metafiles, versions = res[:5]
        passthrough = res[5] if len(res) > 5 else False
        groups.setdefault(group, {"name": group, "type": rtype.split(".")[0],
                                  "title": group, "num_items": 1})
        out_results.append({
            "group_name": group, "name": name, "type": rtype,
            "metafiles": metafiles, "versions": versions,
            "passthrough": passthrough, "num_items": [1] * len(metafiles),
        })
    doc = {
        "uid": uid, "job_type": job_type, "type": job_type, "status": status,
        "title": f"Job {uid}", "parents": parents,
        "output_result_groups": list(groups.values()),
        "output_results": out_results,
    }
    path.write_text(json.dumps(doc))


@pytest.fixture
def mini_project(tmp_path):
    """An import job, a 2D classification with several rounds, and jobs that
    must be skipped."""
    proj = tmp_path / "CS-mini"
    for uid in ("J1", "J2", "J3", "J4"):
        (proj / uid).mkdir(parents=True)

    # J1: import, 4 particles
    _write_cs(proj / "J1" / "imported_particles.cs", 4)
    _job_json(proj / "J1" / "job.json", "J1", "import_particles", "completed", [],
              [("imported_particles", "blob", "particle.blob",
                ["J1/imported_particles.cs"], [0])])

    # J2: three rounds of 2D classification; only the last is final
    for it in (0, 1, 2):
        _write_cs(proj / "J2" / f"J2_{it:03d}_particles.cs", 4)
    _job_json(proj / "J2" / "job.json", "J2", "class_2D_new", "completed", ["J1"],
              [("particles", "blob", "particle.blob",
                [f"J2/J2_{it:03d}_particles.cs" for it in (0, 1, 2)], [0, 1, 2])])

    # J3: failed, must be skipped
    _job_json(proj / "J3" / "job.json", "J3", "class_2D_new", "failed", ["J1"], [])

    # J4: completed but has no RELION equivalent, must be reported as skipped
    _job_json(proj / "J4" / "job.json", "J4", "check_corrupt_particles", "completed",
              ["J2"], [])

    return proj


def test_dry_run_lists_only_completed_supported_jobs(importer, mini_project):
    r = subprocess.run([importer, "--i", str(mini_project), "--dry_run"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout

    assert "J1" in out and "J2" in out
    assert "J3" not in out.split("no RELION equivalent")[0]   # failed: never listed
    assert "2 job(s) to import" in out
    # J4 completed but unsupported: reported rather than silently dropped
    assert "check_corrupt_particles" in out
    assert "--dry_run: nothing was written." in out


def test_final_round_is_the_one_imported(importer, mini_project):
    r = subprocess.run([importer, "--i", str(mini_project), "--dry_run"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    # The last round, not the first, and not every round
    assert "J2_002_particles.cs" in r.stdout
    assert "J2_000_particles.cs" not in r.stdout


def test_import_writes_jobs_and_pipeline(importer, mini_project, tmp_path):
    out = tmp_path / "relion"
    r = subprocess.run([importer, "--i", str(mini_project), "--o", str(out)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr

    # Each job keeps its CryoSPARC number: J1 -> job001, J2 -> job002
    assert (out / "Import" / "job001" / "particles.star").exists()
    assert (out / "Class2D" / "job002" / "run_it025_data.star").exists()

    pipeline = (out / "default_pipeline.star").read_text()
    assert "Import/job001/" in pipeline
    assert "Class2D/job002/" in pipeline
    assert "relion.class2d" in pipeline
    # The CryoSPARC parent relationship became a RELION pipeline edge
    assert "Import/job001/particles.star Class2D/job002/" in pipeline
    # The next job the GUI creates must not reuse a number
    assert "_rlnPipeLineJobCounter                     3" in pipeline


def test_job_numbers_follow_cryosparc(importer, tmp_path):
    """J5 becomes job005 even when J2-J4 do not exist or are not imported, so a
    job can be found under the same number in both projects."""
    proj = tmp_path / "CS-gap"
    for uid in ("J1", "J5"):
        (proj / uid).mkdir(parents=True)
    _write_cs(proj / "J1" / "imported_particles.cs", 4)
    _job_json(proj / "J1" / "job.json", "J1", "import_particles", "completed", [],
              [("imported_particles", "blob", "particle.blob",
                ["J1/imported_particles.cs"], [0])])
    _write_cs(proj / "J5" / "J5_000_particles.cs", 4)
    _job_json(proj / "J5" / "job.json", "J5", "class_2D_new", "completed", ["J1"],
              [("particles", "blob", "particle.blob", ["J5/J5_000_particles.cs"], [0])])

    out = tmp_path / "relion"
    r = subprocess.run([importer, "--i", str(proj), "--o", str(out)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr

    assert (out / "Import" / "job001" / "particles.star").exists()
    assert (out / "Class2D" / "job005" / "run_it025_data.star").exists()
    assert not (out / "Class2D" / "job002").exists()

    pipeline = (out / "default_pipeline.star").read_text()
    assert "Import/job001/particles.star Class2D/job005/" in pipeline
    # One past the highest CryoSPARC number, not the number of jobs imported
    assert "_rlnPipeLineJobCounter                     6" in pipeline


_LOCATION = [("location/micrograph_path", "S32"),
             ("location/center_x_frac", "<f4"),
             ("location/center_y_frac", "<f4"),
             ("location/micrograph_shape", "<u4", (2,))]
_FILAMENT = [("filament/filament_uid", "<u8"),
             ("filament/filament_pose", "<f4"),
             ("filament/position_A", "<f4")]


def _set_location(a):
    for i in range(len(a)):
        a[i]["location/micrograph_path"] = b"J1/mic1.mrc"
        a[i]["location/center_x_frac"] = 0.1 * (i + 1)
        a[i]["location/center_y_frac"] = 0.1 * (i + 1)
        a[i]["location/micrograph_shape"] = (4092, 5760)


def test_filament_columns_survive_2d_classification(importer, tmp_path):
    """CryoSPARC's 2D classification passes through location/* but not
    filament/*, so the helical columns must come from an ancestor, joined on
    particle uid."""
    proj = tmp_path / "CS-helix"
    for uid in ("J1", "J2"):
        (proj / uid).mkdir(parents=True)

    # J1: particles with filament/* (as an extraction from tracer picks has)
    _write_cs(proj / "J1" / "imported_particles.cs", 4, _LOCATION + _FILAMENT)
    a = np.load(proj / "J1" / "imported_particles.cs")
    _set_location(a)
    a["filament/filament_uid"] = [900001, 900001, 900002, 900002]
    a["filament/filament_pose"] = np.pi / 4
    a["filament/position_A"] = [0.0, 28.0, 0.0, 28.0]
    with open(proj / "J1" / "imported_particles.cs", "wb") as f:
        np.save(f, a, allow_pickle=False)
    _job_json(proj / "J1" / "job.json", "J1", "import_particles", "completed", [],
              [("imported_particles", "blob", "particle.blob",
                ["J1/imported_particles.cs"], [0])])

    # J2: 2D classification keeping 3 of the 4 particles, in a different order;
    # its passthrough has location/* and no filament/*
    _write_cs(proj / "J2" / "J2_000_particles.cs", 3)
    m = np.load(proj / "J2" / "J2_000_particles.cs")
    m["uid"] = [4, 1, 2]
    with open(proj / "J2" / "J2_000_particles.cs", "wb") as f:
        np.save(f, m, allow_pickle=False)
    pt = np.zeros(3, dtype=[("uid", "<u8")] + _LOCATION)
    _set_location(pt)
    pt["uid"] = [4, 1, 2]
    with open(proj / "J2" / "J2_passthrough_particles.cs", "wb") as f:
        np.save(f, pt, allow_pickle=False)
    _job_json(proj / "J2" / "job.json", "J2", "class_2D_new", "completed", ["J1"],
              [("particles", "blob", "particle.blob", ["J2/J2_000_particles.cs"], [0]),
               ("particles", "location", "particle.location",
                ["J2/J2_passthrough_particles.cs"], [0], True)])

    out = tmp_path / "relion"
    r = subprocess.run([importer, "--i", str(proj), "--o", str(out)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    # The import log says where the helical columns came from
    assert "filament columns from" in r.stdout
    assert "imported_particles.cs" in r.stdout.split("filament columns from", 1)[1]

    text = (out / "Class2D" / "job002" / "run_it025_data.star").read_text()
    labels, rows, inside = [], [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("data_"):
            inside = (s == "data_particles")
        elif not inside or not s or s.startswith(("loop_", "#")):
            continue
        elif s.startswith("_"):
            labels.append(s.split()[0][1:])
        else:
            rows.append(dict(zip(labels, s.split())))

    assert len(rows) == 3
    # Rows follow J2's order (uids 4, 1, 2); J1's filaments are uid 1,2 -> 900001
    # and uid 3,4 -> 900002, renumbered per micrograph in order of appearance
    assert [int(r["rlnHelicalTubeID"]) for r in rows] == [1, 2, 2]
    assert [float(r["rlnHelicalTrackLengthAngst"]) for r in rows] == [28.0, 0.0, 28.0]
    assert all(float(r["rlnAnglePsiPrior"]) == pytest.approx(-45.0, abs=1e-3) for r in rows)
    # The job's own passthrough is still merged too
    assert "rlnCoordinateX" in labels


def test_import_looks_like_a_relion_project(importer, mini_project, tmp_path):
    """The GUI refuses to open a directory without .gui_projectdir, and stages
    job.star files in .TMP_runfiles, so a command-line import has to make both."""
    out = tmp_path / "relion"
    r = subprocess.run([importer, "--i", str(mini_project), "--o", str(out)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr

    assert (out / ".gui_projectdir").exists()
    assert (out / ".TMP_runfiles").is_dir()
    # Each imported job is marked finished, as a job RELION ran would be
    assert (out / "Import" / "job001" / "RELION_JOB_EXIT_SUCCESS").exists()
    assert (out / "Class2D" / "job002" / "RELION_JOB_EXIT_SUCCESS").exists()


def test_each_job_has_a_readable_job_star(importer, mini_project, tmp_path):
    """Clicking a job in the GUI reads job.star; without one it can only warn
    about an unrecognised job type and leave the parameter panel empty."""
    out = tmp_path / "relion"
    r = subprocess.run([importer, "--i", str(mini_project), "--o", str(out)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr

    imp = (out / "Import" / "job001" / "job.star").read_text()
    assert "_rlnJobTypeLabel" in imp and "relion.import" in imp
    # The input wildcard is a CryoSPARC-side path, so it must not be invented
    assert "Micrographs/*.tif" not in imp

    cls = (out / "Class2D" / "job002" / "job.star").read_text()
    assert "relion.class2d" in cls
    # The job knows which imported job its particles came from
    assert "Import/job001/particles.star" in cls


def test_job_option_imports_only_that_lineage(importer, tmp_path, mini_project):
    # J2 depends on J1 only, so a J2-rooted import is both of them and nothing else
    r = subprocess.run([importer, "--i", str(mini_project), "--job", "J2", "--dry_run"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "2 job(s) to import" in r.stdout


def test_unknown_job_is_an_error(importer, mini_project):
    r = subprocess.run([importer, "--i", str(mini_project), "--job", "J99", "--dry_run"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode != 0
    assert "J99" in r.stdout + r.stderr


def test_missing_project_is_an_error(importer, tmp_path):
    r = subprocess.run([importer, "--i", str(tmp_path / "nope"), "--dry_run"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode != 0
