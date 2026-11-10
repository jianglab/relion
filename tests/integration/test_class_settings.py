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


def class_counts(data_star, nr_classes):
    rows = read_table(data_star, "particles")
    counts = [0] * nr_classes
    for row in rows:
        counts[int(row["rlnClassNumber"]) - 1] += 1
    return counts


COMMON_HELIX = ["--sym", "C1", "--helix", "--helical_outer_diameter", "26", "--helical_twist_initial", "-1.1",
                "--helical_rise_initial", "4.75", "--helical_z_percentage", "0.3", "--ini_high", "6", "--ctf",
                "--tau2_fudge", "4", "--particle_diameter", "28", "--flatten_solvent", "--zero_mask", "--oversampling", "1",
                "--healpix_order", "1", "--offset_range", "3", "--offset_step", "2", "--random_seed", "1"]


def shared_map_setup(test_data_dir, relion_bin, relion_runner):
    paths = generate_helical_dataset(test_data_dir / "data", relion_bin, relion_runner,
                                     num_filaments=4, segments_per_filament=6, size=32, angpix=1.0)
    work = test_data_dir / "same_map"
    work.mkdir()

    def run(name, rows, *args):
        """rows: (map, twist, rise) per class"""
        refs = work / f"{name}.star"
        refs.write_text("\ndata_model_classes\n\nloop_\n_rlnReferenceImage #1\n_rlnHelicalTwist #2\n_rlnHelicalRise #3\n"
                        + "".join(f"{m} {twist} {rise}\n" for m, twist, rise in rows))
        result = relion_runner(refine_command(relion_bin, "--i", str(paths["helical_star"]), "--o", str(work / name),
                                              "--ref", str(refs), *COMMON_HELIX, *args), timeout=600)
        out = result.stdout + result.stderr
        assert result.returncode == 0, out[-3000:]
        return out

    return paths, work, run


@pytest.mark.integration
def test_same_map_seeded_within_its_classes(test_data_dir, relion_bin, relion_runner):
    """Rows that list one map are seeded among themselves: in the first iteration each particle goes to one
    random class of the group, while a class with a map of its own competes as usual."""
    paths, work, run = shared_map_setup(test_data_dir, relion_bin, relion_runner)
    a = paths["initial_model"]
    b = work / "other.mrc"   # (inverted contrast: a map of its own that attracts few particles)
    result = relion_runner([str(relion_bin / "relion_image_handler"), "--i", str(a), "--o", str(b),
                            "--multiply_constant", "-1"])
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    n = paths["total_particles"]

    out = run("em", [(a, -1.1, 4.75), (a, 30.0, 4.75), (b, -1.1, 4.75)], "--iter", "1")
    assert "Classes 1 and 2 start from the same map: in iteration 1 each particle goes to one random class among them" in out, out[-3000:]
    assert "start from identical maps" not in out, out[-3000:]
    counts = class_counts(work / "em_it001_data.star", 3)
    assert sum(counts) == n and counts[0] > 0 and counts[1] > 0, counts
    settings = read_table(work / "em_it000_model.star", "model_class_settings")
    assert [int(r["rlnReferenceSeedGroup"]) for r in settings] == [1, 1, 3]
    # once seeded, the groups are gone (a continued run does not seed again)
    assert "rlnReferenceSeedGroup" not in (work / "em_it001_model.star").read_text()

    # With --firstiter_cc the first iteration uses only the first class of the group, the second one seeds
    out = run("cc2", [(a, -1.1, 4.75), (a, 30.0, 4.75)], "--iter", "2", "--firstiter_cc")
    assert "in iteration 2 each particle goes to one random class among them" in out, out[-3000:]
    seeded = [r["rlnClassNumber"] for r in read_table(work / "cc2_it002_data.star", "particles")]
    assert "1" in seeded and "2" in seeded, seeded
    # ... and so does a run continued after the first iteration: the same random classes
    run("cc", [(a, -1.1, 4.75), (a, 30.0, 4.75)], "--iter", "1", "--firstiter_cc")
    assert class_counts(work / "cc_it001_data.star", 2) == [n, 0]
    result = relion_runner(refine_command(relion_bin, "--continue", str(work / "cc_it001_optimiser.star"),
                                          "--o", str(work / "cc"), "--iter", "2"), timeout=600)
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    assert [r["rlnClassNumber"] for r in read_table(work / "cc_it002_data.star", "particles")] == seeded

    # Without the seeding, --firstiter_cc puts every particle in the first of the identical classes
    out = run("noseed", [(a, -1.1, 4.75), (a, -1.1, 4.75)], "--iter", "1", "--firstiter_cc", "--no_shared_ref_seeds")
    assert "classes 1 and 2 start from identical maps with the same helical twist and rise" in out, out[-3000:]
    assert class_counts(work / "noseed_it001_data.star", 2) == [n, 0]


@pytest.mark.integration
def test_helical_symmetrise_refs(test_data_dir, relion_bin, relion_runner):
    """--helical_symmetrise_refs gives each class's starting map its own twist and rise before the first E-step"""
    paths, work, run = shared_map_setup(test_data_dir, relion_bin, relion_runner)
    a = paths["initial_model"]
    out = run("sym", [(a, -1.1, 4.75), (a, 30.0, 4.75)], "--iter", "1", "--firstiter_cc",
              "--helical_symmetrise_refs", "--no_shared_ref_seeds")
    assert "Each class's starting map has been given its own helical twist and rise" in out, out[-3000:]
    assert "start from identical maps" not in out, out[-3000:]
    assert class_counts(work / "sym_it001_data.star", 2)[1] > 0
