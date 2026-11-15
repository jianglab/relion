#!/usr/bin/env python3
"""
Fractional particle coordinates. Extract cuts the box around the rounded coordinate, Polish (like
upstream RELION) around the truncated one; for a pick at 100.6 they disagreed by a pixel. Extract now
writes the whole pixel it cut the box around (moving the difference into the offsets of particles that
have them), so every program finds the same centre. Offsets of older files with fractional coordinates refer to the
rounded centre (this RELION since 2026-10-05, the default) or the truncated one (upstream:
--coord_centre trunc or RELION_COORD_CENTRE=trunc).
"""

import math

import pytest

from test_virtual_particles import project, extract, read_particles, run, write_loop  # noqa: F401 (fixture)

ANGPIX = 1.5   # micrograph pixel size of the project fixture


def _centres(star):
    """(coordinate, centre) per particle and axis; centre = coordinate - offset, in micrograph pixels."""
    labels, rows = read_particles(star)
    out = []
    for r in rows:
        for c, o in (("CoordinateX", "OriginXAngst"), ("CoordinateY", "OriginYAngst")):
            coord = float(r[labels.index(c)])
            off = float(r[labels.index(o)]) / ANGPIX if o in labels else 0.
            out.append((coord, coord - off))
    return out


def test_extraction_writes_whole_pixel_coordinates(relion_bin, project):
    """Fractional picks: the whole pixel the box was cut around is written, and no offset columns are
    added (scripts that add their own must keep working)."""
    star = extract(relion_bin, project, "Extract/picks", ["--coord_list", "coords.star", "--extract_size", "64"],
                   virtual=False)
    picks = [tuple(float(v) for v in line.split())
             for f in sorted((project / "Coords").glob("*.star"))
             for line in f.read_text().splitlines()
             if line.strip() and not line.startswith(("_", "data_", "loop_", "#"))]
    labels, rows = read_particles(star)
    assert "OriginXAngst" not in labels
    got = [(float(r[labels.index("CoordinateX")]), float(r[labels.index("CoordinateY")])) for r in rows]
    assert any(x != math.floor(x) or y != math.floor(y) for x, y in picks), "the fixture has no fractional picks"
    for x, y in got:
        assert x == math.floor(x) and y == math.floor(y), (x, y)
        assert (x, y) in [(math.floor(px + 0.5), math.floor(py + 0.5)) for px, py in picks], (x, y)


@pytest.mark.parametrize("rule", ["default", "round", "trunc", "env_trunc"])
def test_offsets_of_older_files(relion_bin, project, rule):
    """Re-extraction of a file with fractional coordinates and offsets: the offsets refer to the rounded
    centre unless told otherwise, and the particle (centre - offset) stays where they put it."""
    xs = [(180.6, 150.2, 3.0, -1.5), (100.4, 120.5, -4.5, 6.0), (60.5, 70.7, 0.0, 0.0)]
    write_loop(project / "old.star", [
        ("optics", ["OpticsGroupName", "OpticsGroup", "MicrographOriginalPixelSize", "MicrographPixelSize",
                    "ImagePixelSize", "ImageDimensionality", "ImageSize", "Voltage", "SphericalAberration",
                    "AmplitudeContrast"],
         [["opticsGroup1", 1, ANGPIX, ANGPIX, ANGPIX, 2, 64, 300, 2.7, 0.1]]),
        ("particles", ["MicrographName", "CoordinateX", "CoordinateY", "OriginXAngst", "OriginYAngst",
                       "DefocusU", "DefocusV", "DefocusAngle", "OpticsGroup"],
         [["Micrographs/mic001.mrc", x, y, ox, oy, 12000, 11800, 30.0, 1] for x, y, ox, oy in xs]),
    ])
    opts = ["--reextract_data_star", "old.star", "--extract_size", "64"]
    if rule in ("round", "trunc"):
        opts += ["--coord_centre", rule]
    out = f"Extract/old_{rule}"
    (project / out).mkdir(parents=True)
    r = run(relion_bin, project, ["relion_preprocess", "--i", "micrographs_ctf.star", "--part_star",
                                  f"{out}/particles.star", "--part_dir", f"{out}/", "--extract", "--no_virtual"] + opts,
            env_extra={"RELION_COORD_CENTRE": "trunc"} if rule == "env_trunc" else None)
    assert r.returncode == 0, r.stdout + r.stderr
    star = project / out / "particles.star"
    rounded = rule in ("default", "round")
    want = []
    for x, y, ox, oy in xs:
        for c, o in ((x, ox), (y, oy)):
            base = math.floor(c + 0.5) if rounded else math.trunc(c)
            want.append(base - o / ANGPIX)
    got = _centres(star)
    assert all(c == math.floor(c) for c, _ in got)
    assert [round(centre, 4) for _, centre in got] == [round(w, 4) for w in want]
