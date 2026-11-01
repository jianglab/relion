#!/usr/bin/env python3
"""
Per-class settings in Class3D: columns beside the maps of a --ref STAR file give each
class its own helical twist/rise, tube diameter and point group. They are reported at
the start, saved in model.star (table model_class_settings) and kept by --continue.
"""

import shutil

import pytest

from test_filament_class3d import generate_helical_dataset


def read_table(star_path, table):
    """Rows of one table of a STAR file, as dicts keyed by label (without the leading _)."""
    labels, rows, inside = [], [], False
    with open(star_path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("data_"):
                if inside:
                    break
                inside = s == "data_" + table
                continue
            if not inside or not s or s.startswith("#") or s == "loop_":
                continue
            if s.startswith("_"):
                labels.append(s.split()[0][1:])
            else:
                rows.append(dict(zip(labels, s.split())))
    return rows


def refine_command(relion_bin, *args):
    return [str(relion_bin / "relion_refine"), "--j", "2", "--pool", "3", *args]


@pytest.mark.integration
def test_class_settings_reported_saved_and_continued(test_data_dir, relion_bin, relion_runner):
    paths = generate_helical_dataset(test_data_dir / "data", relion_bin, relion_runner,
                                     num_filaments=4, segments_per_filament=6, size=32, angpix=1.0)
    work = test_data_dir / "class_settings"
    work.mkdir()
    for name in ("a.mrc", "b.mrc"):
        shutil.copy(paths["initial_model"], work / name)
    refs = work / "refs.star"
    refs.write_text(
        "\ndata_model_classes\n\nloop_\n"
        "_rlnReferenceImage #1\n_rlnHelicalTwist #2\n_rlnHelicalRise #3\n"
        "_rlnHelicalMaskTubeOuterDiameter #4\n_rlnSymmetryGroup #5\n_rlnHelicalMaskTubeInnerDiameter #6\n"
        f"{work / 'a.mrc'} -1.1 4.75 20 C1 6\n"
        f"{work / 'b.mrc'} 2.0 4.8 24 C2 -1\n")

    common = ["--ini_high", "15", "--ctf", "--tau2_fudge", "4", "--particle_diameter", "28",
              "--flatten_solvent", "--zero_mask", "--oversampling", "1", "--healpix_order", "1",
              "--offset_range", "3", "--offset_step", "2", "--random_seed", "1"]
    result = relion_runner(refine_command(
        relion_bin, "--i", str(paths["helical_star"]), "--o", str(work / "run"), "--ref", str(refs),
        "--iter", "1", "--sym", "C1", "--helix", "--helical_outer_diameter", "26",
        "--helical_twist_initial", "-1.1", "--helical_rise_initial", "4.75",
        "--helical_z_percentage", "0.3", *common), timeout=600)
    out = result.stdout + result.stderr
    assert result.returncode == 0, out[-3000:]

    # The classes start from their own row, and the images are masked with the widest tube
    assert "Class 2 starts with helical twist" in out, out[-3000:]
    # (class 2 has no inner diameter, so the images get none either)
    assert "Particle images masked with a tube of up to 26 A diameter" in out, out[-3000:]
    assert "Class 1 reference masked with a tube of 6 to 20 A diameter" in out, out[-3000:]
    assert "Class 2 is reconstructed with point group C2" in out, out[-3000:]

    model = work / "run_it001_model.star"
    settings = read_table(model, "model_class_settings")
    assert len(settings) == 2
    assert settings[1]["rlnSymmetryGroup"] == "C2"
    assert float(settings[0]["rlnHelicalMaskTubeOuterDiameter"]) == pytest.approx(20)
    assert float(settings[1]["rlnHelicalRise"]) == pytest.approx(4.8)

    # A continued run reads the table back and saves it again unchanged
    result = relion_runner(refine_command(
        relion_bin, "--continue", str(work / "run_it001_optimiser.star"), "--o", str(work / "ct"),
        "--iter", "2"), timeout=600)
    out = result.stdout + result.stderr
    assert result.returncode == 0, out[-3000:]
    assert read_table(work / "ct_it002_model.star", "model_class_settings") == settings
    # ... and still masks each class with its own tube
    assert "Class 2 reference masked with a tube of up to 24 A diameter" in out, out[-3000:]
