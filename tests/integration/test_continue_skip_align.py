#!/usr/bin/env python3
"""
Continuing a --skip_align run must not crash while writing the angular-distribution
bild files: such a model holds one direction per class, not the sampling's directions.
"""

import subprocess

import pytest

from fixtures import generate_test_dataset


def refine(relion_bin, outdir, args):
    cmd = [str(relion_bin / "relion_refine")] + args + [
        "--j", "1", "--random_seed", "7", "--dont_combine_weights_via_disc", "--pool", "10"]
    r = subprocess.run(cmd, cwd=outdir, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


@pytest.mark.integration
@pytest.mark.parametrize("with_flag", [True, False])
def test_continue_from_skip_align(test_data_dir, relion_bin, relion_runner, with_flag):
    paths = generate_test_dataset(str(test_data_dir / "data"), relion_bin, relion_runner,
                                  num_particles=40, size=32)
    out = test_data_dir / ("cont_flag" if with_flag else "cont_noflag")
    out.mkdir()
    common = ["--ref", str(paths["initial_model"]), "--K", "2", "--ini_high", "20", "--tau2_fudge", "4",
              "--particle_diameter", "25", "--oversampling", "1", "--healpix_order", "1",
              "--offset_range", "3", "--offset_step", "2", "--sym", "C1"]
    refine(relion_bin, out, ["--i", str(paths["star_file"]), "--o", str(out / "run"), "--iter", "2",
                             "--skip_align", "--keep_all_iterations"] + common)
    refine(relion_bin, out, ["--continue", str(out / "run_it002_optimiser.star"), "--o", str(out / "run_ct2"), "--iter", "3"]
           + (["--skip_align"] if with_flag else []))
    assert (out / "run_ct2_it003_model.star").exists()
