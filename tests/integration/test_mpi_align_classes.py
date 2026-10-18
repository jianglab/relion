#!/usr/bin/env python3
"""
Integration test for --align_classes under MPI (relion_refine_mpi, Class3D).

At the last iteration every class map is aligned to the largest class and the inverse of each
transformation is applied to the orientations and offsets of that class's particles, so that the
written class maps and the written data.star describe the same thing. This used to be done only
by the serial relion_refine; under MPI the option silently did nothing.

The test makes two classes from the same density, one of them a shifted and rotated copy, and runs
three ranks with and without --dont_align_classes. Everything up to the last iteration is the same
in both runs, so:
  * the class maps correlate much better with alignment than without it;
  * a map reconstructed from the aligned data.star agrees with the aligned class map and not with
    the unaligned one (the particle metadata and the maps are mutually consistent);
  * the serial program finds the same alignment as the MPI one.
"""

import re
import shutil

import numpy as np
import pytest

from fixtures import generate_test_dataset, read_mrc, read_star_particles, write_volume_mrc

ALIGN_LINE = re.compile(r"[Cc]lass\s+(\d+)\s+to class\s+(\d+):\s+rot=(\S+) tilt=(\S+) psi=(\S+) dx=(\S+) dy=(\S+) dz=(\S+)")
SERIAL_LINE = re.compile(r"Best: rot=(\S+) tilt=(\S+) psi=(\S+) dx=(\S+) dy=(\S+) dz=(\S+)")


def _shifted_rotated(vol, shift, angle):
    ndimage = pytest.importorskip("scipy.ndimage")
    out = ndimage.rotate(vol, angle, axes=(1, 2), reshape=False, order=3, mode="constant")
    return ndimage.shift(out, shift, order=3, mode="constant")


def _corr(a, b):
    return np.corrcoef(np.ravel(a), np.ravel(b))[0, 1]


def _refine(cmd_prefix, relion_bin, program, paths, refs_star, out, run, extra=(), threads=1):
    import subprocess
    out.mkdir(parents=True, exist_ok=True)
    cmd = list(cmd_prefix) + [
        str(relion_bin / program),
        "--i", str(paths["star_file"]), "--o", str(out / "run"),
        "--ref", str(refs_star), "--K", "2", "--iter", "3", "--ini_high", "20",
        "--ctf", "--tau2_fudge", "4", "--particle_diameter", "25",
        "--flatten_solvent", "--zero_mask", "--oversampling", "1",
        "--healpix_order", "2", "--offset_range", "5", "--offset_step", "2",
        "--sym", "c1", "--norm", "--scale", "--j", str(threads),
        "--dont_combine_weights_via_disc", "--random_seed", "7", "--keep_all_iterations",
        "--pool", "5", "--dont_check_norm",
    ] + list(extra)
    res = subprocess.run(cmd, text=True, capture_output=True, timeout=3600)
    assert res.returncode == 0, f"{program} failed:\n{res.stderr[-3000:]}"
    return res.stdout


def _maps(out):
    return [np.asarray(read_mrc(str(out / f"run_it003_class00{k}.mrc")), dtype=np.float64) for k in (1, 2)]


@pytest.mark.integration
class TestMpiAlignClasses:

    @pytest.fixture
    def setup(self, test_data_dir, relion_bin, relion_runner):
        if shutil.which("mpirun") is None or not (relion_bin / "relion_refine_mpi").exists():
            pytest.skip("no MPI build")
        size = 32
        paths = generate_test_dataset(str(test_data_dir / "data"), relion_bin, relion_runner,
                                      num_particles=150, size=size, angpix=1.0)
        truth = np.asarray(read_mrc(paths["initial_model"]), dtype=np.float64)
        moved = _shifted_rotated(truth, (1.0, 2.0, 1.0), 3.0)
        d = test_data_dir / "refs"
        d.mkdir()
        write_volume_mrc(truth, str(d / "a.mrc"))
        write_volume_mrc(moved, str(d / "b.mrc"))
        star = d / "refs.star"
        star.write_text("data_\n\nloop_\n_rlnReferenceImage\n%s\n%s\n" % (d / "a.mrc", d / "b.mrc"))
        return paths, star

    def test_mpi_aligns_and_is_consistent(self, setup, test_data_dir, relion_bin):
        paths, star = setup
        mpi = ["mpirun", "--oversubscribe", "-n", "3"]
        out_on = test_data_dir / "on"
        out_off = test_data_dir / "off"
        log_on = _refine(mpi, relion_bin, "relion_refine_mpi", paths, star, out_on, "on")
        _refine(mpi, relion_bin, "relion_refine_mpi", paths, star, out_off, "off", ["--dont_align_classes"])

        found = ALIGN_LINE.findall(log_on)
        assert found, "the MPI run did not report any class alignment"
        assert "Aligning 2 classes" in log_on

        on_a, on_b = _maps(out_on)
        off_a, off_b = _maps(out_off)
        assert np.allclose(on_a, off_a, atol=1e-6) or _corr(on_a, off_a) > 0.9999, \
            "the reference class must not be moved"
        c_on, c_off = _corr(on_a, on_b), _corr(off_a, off_b)
        assert c_on > c_off + 0.02, f"alignment did not improve the class correlation: {c_off:.4f} -> {c_on:.4f}"

        # Consistency of data.star with the maps: particles of class 2, reconstructed with the written
        # orientations and offsets, must give the aligned class-2 map and not the unaligned one.
        import subprocess
        recon = {}
        for name, out in (("on", out_on), ("off", out_off)):
            mrc = out / "recon_b.mrc"
            res = subprocess.run([str(relion_bin / "relion_reconstruct"), "--i", str(out / "run_it003_data.star"),
                                  "--o", str(mrc), "--ctf", "--class", "2", "--sym", "c1"],
                                 text=True, capture_output=True, timeout=600)
            assert res.returncode == 0, res.stderr[-2000:]
            recon[name] = np.asarray(read_mrc(str(mrc)), dtype=np.float64)
        assert _corr(recon["on"], on_b) > _corr(recon["on"], off_b), \
            "data.star of the aligned run does not match its aligned class map"
        # ... and about as well as the unaligned data.star matches the unaligned map
        assert _corr(recon["on"], on_b) > _corr(recon["off"], off_b) - 0.1, \
            (_corr(recon["on"], on_b), _corr(recon["off"], off_b))

        # Only particles of the moved class change; the offsets of class 1 are untouched.
        s_on = read_star_particles(str(out_on / "run_it003_data.star"),
                                   ["rlnClassNumber", "rlnOriginXAngst"])
        s_off = read_star_particles(str(out_off / "run_it003_data.star"),
                                    ["rlnClassNumber", "rlnOriginXAngst"])
        same_class = s_on["rlnClassNumber"] == s_off["rlnClassNumber"]
        c1 = same_class & (s_on["rlnClassNumber"] == 1)
        assert np.allclose(s_on["rlnOriginXAngst"][c1], s_off["rlnOriginXAngst"][c1])

    def test_serial_finds_the_same_alignment(self, setup, test_data_dir, relion_bin):
        paths, star = setup
        mpi = ["mpirun", "--oversubscribe", "-n", "3"]
        log_on = _refine(mpi, relion_bin, "relion_refine_mpi", paths, star, test_data_dir / "on", "on")
        found = ALIGN_LINE.findall(log_on)
        assert found, "the MPI run did not report any class alignment"
        # The serial program finds the same transformation (within a search step or two)
        serial_log = _refine([], relion_bin, "relion_refine", paths, star, test_data_dir / "ser", "ser", threads=4)
        serial = SERIAL_LINE.findall(serial_log)
        assert serial, "the serial run did not report an alignment"
        mpi_best = [float(v) for v in found[-1][2:]]
        ser_best = [float(v) for v in serial[-1]]
        assert np.allclose(mpi_best[3:], ser_best[3:], atol=2.5), (mpi_best, ser_best)
        assert np.allclose(mpi_best[:3], ser_best[:3], atol=5.0), (mpi_best, ser_best)

    def test_dont_align_is_parsed_under_mpi_and_continue(self, setup, test_data_dir, relion_bin):
        paths, star = setup
        mpi = ["mpirun", "--oversubscribe", "-n", "3"]
        out = test_data_dir / "off1"
        log = _refine(mpi, relion_bin, "relion_refine_mpi", paths, star, out, "off", ["--dont_align_classes"])
        assert not ALIGN_LINE.search(log) and "Aligning 2 classes" not in log
