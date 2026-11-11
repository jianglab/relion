#!/usr/bin/env python3
"""
Integration test for --write_unmasked_maps in 3D classification and 3D auto-refinement.

With --flatten_solvent the maps of every iteration (rootname_itNNN[_halfN]_classNNN.mrc) are multiplied by the
solvent mask (here the soft sphere of --particle_diameter) before they are written. --write_unmasked_maps also
writes them as they were before the mask, as rootname_itNNN[_halfN]_classNNN_unmasked.mrc, so that:
  - inside the sphere the masked and unmasked maps are the same,
  - outside the soft edge the masked map is zero and the unmasked one is not,
  - without the option no such files are written,
  - a converged auto-refinement (whose final maps are not masked) writes no extra final copy.
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

from fixtures import generate_test_dataset, read_mrc

SIZE, ANGPIX, DIAMETER, EDGE = 32, 1.0, 25.0, 5   # EDGE: --width_mask_edge default (pixels)


def _radius_grid(n):
    c = np.arange(n) - n // 2
    z, y, x = np.meshgrid(c, c, c, indexing="ij")
    return np.sqrt(x * x + y * y + z * z)


def _check_pair(masked_path, unmasked_path):
    m = np.asarray(read_mrc(str(masked_path)), dtype=np.float64)
    u = np.asarray(read_mrc(str(unmasked_path)), dtype=np.float64)
    assert m.shape == u.shape == (SIZE, SIZE, SIZE)
    r = _radius_grid(SIZE)
    radius = DIAMETER / (2 * ANGPIX)
    inside, outside = r < radius - 0.5, r > radius + EDGE + 0.5
    assert np.allclose(m[inside], u[inside], atol=1e-5 * (np.abs(u).max() + 1e-12)), f"{masked_path}: differs inside the mask"
    assert np.abs(m[outside]).max() < 1e-6 * (np.abs(u).max() + 1e-12), f"{masked_path}: not masked outside"
    assert u[outside].std() > 1e-3 * u[inside].std(), f"{unmasked_path}: looks masked outside the sphere"


@pytest.fixture
def dataset(test_data_dir, relion_bin, relion_runner):
    return generate_test_dataset(str(test_data_dir / "data"), relion_bin, relion_runner,
                                 num_particles=200, size=SIZE, angpix=ANGPIX)


COMMON = ["--particle_diameter", str(DIAMETER), "--ctf", "--flatten_solvent", "--zero_mask", "--oversampling", "1",
          "--sym", "c1", "--norm", "--scale", "--j", "1", "--random_seed", "7", "--pool", "5",
          "--keep_all_iterations"]


def _run(cmd, cwd):
    res = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=3600)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-3000:]
    return res.stdout


@pytest.mark.integration
class TestWriteUnmaskedMaps:

    def test_class3d(self, dataset, test_data_dir, relion_bin):
        for name, extra in (("on", ["--write_unmasked_maps"]), ("off", [])):
            out = test_data_dir / name
            out.mkdir()
            _run([str(relion_bin / "relion_refine"), "--i", dataset["star_file"], "--ref", dataset["initial_model"],
                  "--o", str(out / "run"), "--K", "2", "--iter", "2", "--ini_high", "15", "--tau2_fudge", "4",
                  "--healpix_order", "2", "--offset_range", "2", "--offset_step", "1"] + COMMON + extra, out)
            unmasked = sorted(p.name for p in out.glob("*_unmasked.mrc"))
            if name == "off":
                assert unmasked == [], unmasked
                continue
            assert unmasked == [f"run_it00{i}_class00{k}_unmasked.mrc" for i in (1, 2) for k in (1, 2)], unmasked
            for i in (1, 2):
                for k in (1, 2):
                    _check_pair(out / f"run_it00{i}_class00{k}.mrc", out / f"run_it00{i}_class00{k}_unmasked.mrc")

    def test_auto_refine_mpi(self, dataset, test_data_dir, relion_bin):
        if shutil.which("mpirun") is None or not (relion_bin / "relion_refine_mpi").exists():
            pytest.skip("no MPI build")
        out = test_data_dir / "refine"
        out.mkdir()
        _run(["mpirun", "--oversubscribe", "-n", "3", str(relion_bin / "relion_refine_mpi"),
              "--auto_refine", "--split_random_halves", "--i", dataset["star_file"], "--ref", dataset["initial_model"],
              "--o", str(out / "run"), "--ini_high", "15", "--healpix_order", "2", "--auto_local_healpix_order", "4",
              "--offset_range", "2", "--offset_step", "1", "--dont_combine_weights_via_disc",
              "--write_unmasked_maps"] + COMMON, out)
        assert (out / "run_class001.mrc").exists(), "the refinement did not converge"
        assert not list(out.glob("run_class*_unmasked.mrc")), "a converged (unmasked) final map was written twice"
        its = sorted({p.name[len("run_it"):len("run_it") + 3] for p in out.glob("run_it*_half1_class001_unmasked.mrc")})
        assert its, "no unmasked half maps written"
        for it in its:
            for half in (1, 2):
                _check_pair(out / f"run_it{it}_half{half}_class001.mrc", out / f"run_it{it}_half{half}_class001_unmasked.mrc")
        # Every masked half map of an iteration with masking has its unmasked copy
        masked = {p.name[len("run_it"):len("run_it") + 3] for p in out.glob("run_it*_half1_class001.mrc") if p.name[6:9] != "000"}
        last = max(masked)   # the converged iteration is not masked
        assert masked - {last} == set(its) - {last}, (sorted(masked), its)
