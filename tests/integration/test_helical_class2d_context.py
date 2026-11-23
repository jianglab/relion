#!/usr/bin/env python3
"""
Integration test for helical Class2D context (--helical_context), late 2D
--auto_mask / --helical_auto_diameter gating, --continue option preservation
and negative-flag overrides, and --ctf_intact_first_peak optics tracking.
"""

import os
import struct
from pathlib import Path

import numpy as np
import pytest


def write_mrc_stack(path: Path, data: np.ndarray, apix: float) -> None:
    arr = np.ascontiguousarray(data.astype(np.float32))
    nz, ny, nx = arr.shape
    hdr = bytearray(1024)
    struct.pack_into("<10i", hdr, 0, nx, ny, nz, 2, 0, 0, 0, nx, ny, nz)
    struct.pack_into("<3f", hdr, 40, nx * apix, ny * apix, nz * apix)
    struct.pack_into("<3f", hdr, 52, 90.0, 90.0, 90.0)
    struct.pack_into("<3i", hdr, 64, 1, 2, 3)
    struct.pack_into(
        "<3f", hdr, 76, float(arr.min()), float(arr.max()), float(arr.mean())
    )
    hdr[208:212] = b"MAP "
    struct.pack_into("<I", hdr, 212, 0x44410000)
    struct.pack_into("<f", hdr, 216, float(arr.std()))
    path.write_bytes(bytes(hdr) + arr.tobytes())


def parse_star_tables(star_path: Path):
    tables = {}
    lines = star_path.read_text().splitlines()
    cur_name = None
    cols = []
    rows = []
    in_loop = False

    def flush():
        if cur_name is not None:
            tables[cur_name] = (cols[:], rows[:])

    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("data_"):
            flush()
            cur_name = s[5:]
            cols = []
            rows = []
            in_loop = False
        elif s == "loop_":
            in_loop = True
            cols = []
        elif s.startswith("_rln"):
            parts = s.split()
            key = parts[0].lstrip("_")
            if in_loop:
                cols.append(key)
            else:
                if key not in cols:
                    cols.append(key)
                if not rows:
                    rows.append({})
                if len(parts) > 1:
                    rows[0][key] = parts[1]
        elif in_loop and cols:
            vals = s.split()
            if len(vals) >= len(cols):
                rows.append(dict(zip(cols, vals)))
    flush()
    return tables


def generate_synthetic_helical_2d(work_dir: Path, apix: float = 2.0, box: int = 48):
    work_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)
    n_fils = 30
    segs_per_fil = 6
    n_tot = n_fils * segs_per_fil

    yy, xx = np.mgrid[0:box, 0:box]
    c = (box - 1) / 2.0
    dy = yy - c
    dx = xx - c
    # Wide vs narrow continuous horizontal filaments with axial helical periodicity
    wide = np.exp(-0.5 * (dy / 3.5) ** 2) * (1.0 + 0.35 * np.cos(2.0 * np.pi * dx / 10.0))
    narrow = np.exp(-0.5 * (dy / 1.5) ** 2) * (1.0 + 0.35 * np.cos(2.0 * np.pi * dx / 16.0))
    bg_mask = np.abs(dy) >= (box * 0.32)

    stack = np.zeros((n_tot, box, box), dtype=np.float32)
    rows = []
    idx = 0
    for f in range(n_fils):
        tpl = wide if f < (n_fils // 2) else narrow
        for s in range(segs_per_fil):
            img = 2.5 * tpl + rng.normal(0.0, 0.35, size=(box, box))
            bg = img[bg_mask]
            img = (img - bg.mean()) / (bg.std() + 1e-6)
            stack[idx] = img.astype(np.float32)
            defocus = 14000.0 + 300.0 * (f % 4)
            rows.append(
                f"{idx + 1:06d}@{work_dir / 'particles.mrcs'} mic_{f // 5:02d}.mrc 1 {f + 1} "
                f"{s * 24.0:.2f} {(segs_per_fil - 1) * 24.0:.2f} "
                f"0.0 90.0 0.0 0.0 0.0 0.0 {defocus:.1f} {defocus:.1f} 0.0 {idx % 2 + 1}"
            )
            idx += 1

    stack_path = work_dir / "particles.mrcs"
    write_mrc_stack(stack_path, stack, apix)

    star_lines = [
        "",
        "# version 30001",
        "",
        "data_optics",
        "",
        "loop_",
        "_rlnOpticsGroupName #1",
        "_rlnOpticsGroup #2",
        "_rlnMicrographOriginalPixelSize #3",
        "_rlnVoltage #4",
        "_rlnSphericalAberration #5",
        "_rlnAmplitudeContrast #6",
        "_rlnImagePixelSize #7",
        "_rlnImageSize #8",
        "_rlnImageDimensionality #9",
        "_rlnCtfDataAreCtfPremultiplied #10",
        f"opticsGroup1 1 {apix:.6f} 300.0 2.7 0.1 {apix:.6f} {box} 2 0",
        "",
        "# version 30001",
        "",
        "data_particles",
        "",
        "loop_",
        "_rlnImageName #1",
        "_rlnMicrographName #2",
        "_rlnOpticsGroup #3",
        "_rlnHelicalTubeID #4",
        "_rlnHelicalTrackLengthAngst #5",
        "_rlnHelicalFilamentLengthAngst #6",
        "_rlnAngleRot #7",
        "_rlnAngleTilt #8",
        "_rlnAnglePsi #9",
        "_rlnAnglePsiPrior #10",
        "_rlnOriginXAngst #11",
        "_rlnOriginYAngst #12",
        "_rlnDefocusU #13",
        "_rlnDefocusV #14",
        "_rlnDefocusAngle #15",
        "_rlnRandomSubset #16",
    ] + rows
    star_path = work_dir / "particles.star"
    star_path.write_text("\n".join(star_lines) + "\n")
    return star_path


def _common_refine_flags():
    return [
        "--K", "4",
        "--tau2_fudge", "2",
        "--particle_diameter", "84",
        "--helix",
        "--helical_outer_diameter", "56",
        "--helical_cylindrical_mask",
        "--helical_auto_diameter",
        "--auto_mask",
        "--auto_mask_start_res", "20",
        "--helical_context",
        "--helical_context_types", "2",
        "--helical_context_types_max", "3",
        "--helical_context_types_tol", "0.02",
        "--helical_context_types_agree", "0.65",
        "--helical_context_types_every", "4",
        "--helical_context_start", "4",
        "--helical_context_warm",
        "--helical_track_sigma_A", "2.5",
        "--center_classes",
        "--bimodal_psi",
        "--dont_fix_gpu_bimodal_psi",
        "--flatten_solvent",
        "--zero_mask",
        "--oversampling", "1",
        "--psi_step", "10",
        "--offset_range", "2",
        "--offset_step", "1",
        "--norm",
        "--scale",
        "--j", "2",
        "--random_seed", "7",
        "--dont_combine_weights_via_disc",
        "--pool", "10",
        "--keep_all_iterations",
    ]


def test_helical_class2d_context_auto_mask_and_continue(test_data_dir, relion_bin, relion_runner):
    work_dir = test_data_dir / "helical_ctx_2d"
    star_path = generate_synthetic_helical_2d(work_dir)
    out_root = work_dir / "run"

    cmd = [
        str(relion_bin / "relion_refine"),
        "--i", str(star_path),
        "--o", str(out_root),
        "--iter", "7",
    ] + _common_refine_flags()

    res = relion_runner(cmd, timeout=240)
    assert res.returncode == 0, f"relion_refine failed:\n{res.stderr}\n{res.stdout}"

    # Late auto-mask gate: no automask files or reports for iterations 1..5
    for it in range(1, 6):
        assert not list(work_dir.glob(f"run_it{it:03d}_*_automask.mrc")), (
            f"auto_mask triggered too early at it{it:03d}"
        )

    assert (work_dir / "run_it007_data.star").exists()
    assert (work_dir / "run_it007_model.star").exists()
    assert (work_dir / "run_it007_optimiser.star").exists()
    assert (work_dir / "run_it007_classes.mrcs").exists()

    # Check optimiser persistence labels
    opt_tables = parse_star_tables(work_dir / "run_it007_optimiser.star")
    opt_row = opt_tables["optimiser_general"][1][0]
    assert opt_row["rlnDoHelicalContext"] == "1"
    assert opt_row["rlnDoCenterClasses"] == "1"
    assert opt_row["rlnDoBimodalPsi"] == "1"
    assert opt_row["rlnDoFixGpuBimodalPsi"] == "0"
    assert opt_row["rlnHelicalContextWarm"] == "1"
    assert float(opt_row["rlnHelicalContextTypesTol"]) == pytest.approx(0.02, abs=1e-4)
    assert float(opt_row["rlnHelicalContextTypesAgree"]) == pytest.approx(0.65, abs=1e-4)
    assert int(opt_row["rlnHelicalContextTypesEvery"]) == 4
    assert float(opt_row["rlnHelicalTrackSigmaAngstrom"]) == pytest.approx(2.5, abs=1e-4)

    # Every populated type in run_it007_data.star has >= 1 class written
    data_tables = parse_star_tables(work_dir / "run_it007_data.star")
    types_present = {
        int(r["rlnHelicalType"])
        for r in data_tables["particles"][1]
        if int(r.get("rlnHelicalType", "0")) >= 1
    }
    assert len(types_present) >= 1
    for t in sorted(types_present):
        t_cls_star = work_dir / f"run_it007_type{t}_classes.star"
        assert t_cls_star.exists()
        t_cls_tbl = parse_star_tables(t_cls_star)
        assert len(t_cls_tbl["model_classes"][1]) >= 1

    # Bare --continue + harmless --no_init_blobs preserves saved context & options
    cmd_bare = [
        str(relion_bin / "relion_refine"),
        "--continue", str(work_dir / "run_it007_optimiser.star"),
        "--o", str(work_dir / "cont_bare"),
        "--iter", "8",
        "--no_init_blobs",
        "--j", "2",
        "--dont_combine_weights_via_disc",
    ]
    res_bare = relion_runner(cmd_bare, timeout=120)
    assert res_bare.returncode == 0, f"bare continue failed:\n{res_bare.stderr}\n{res_bare.stdout}"
    bare_row = parse_star_tables(work_dir / "cont_bare_it008_optimiser.star")["optimiser_general"][1][0]
    assert bare_row["rlnDoHelicalContext"] == "1"
    assert bare_row["rlnDoCenterClasses"] == "1"
    assert bare_row["rlnDoBimodalPsi"] == "1"
    assert bare_row["rlnHelicalContextWarm"] == "1"
    assert bare_row["rlnHelicalContextTypes"] == opt_row["rlnHelicalContextTypes"]

    # Explicit negative flags on --continue override persisted booleans cleanly
    cmd_off = [
        str(relion_bin / "relion_refine"),
        "--continue", str(work_dir / "run_it007_optimiser.star"),
        "--o", str(work_dir / "cont_off"),
        "--iter", "8",
        "--dont_helical_context",
        "--dont_center_classes",
        "--dont_bimodal_psi",
        "--dont_helical_context_warm",
        "--j", "2",
        "--dont_combine_weights_via_disc",
    ]
    res_off = relion_runner(cmd_off, timeout=120)
    assert res_off.returncode == 0, f"negative-flag continue failed:\n{res_off.stderr}\n{res_off.stdout}"
    off_row = parse_star_tables(work_dir / "cont_off_it008_optimiser.star")["optimiser_general"][1][0]
    assert off_row["rlnDoHelicalContext"] == "0"
    assert off_row["rlnDoCenterClasses"] == "0"
    assert off_row["rlnDoBimodalPsi"] == "0"
    assert off_row["rlnHelicalContextWarm"] == "0"


def test_helix_subclassify_prepare_only_intact_first_peak_warning(test_data_dir, relion_bin, relion_runner):
    work_dir = test_data_dir / "subclassify_intact"
    work_dir.mkdir(parents=True, exist_ok=True)
    box_mic = 128
    apix = 2.0
    yy, xx = np.mgrid[0:box_mic, 0:box_mic]
    mic = np.exp(-0.5 * ((yy - 64.0) / 3.0) ** 2)
    rng = np.random.default_rng(99)
    mic = mic + rng.normal(0.0, 0.1, size=(box_mic, box_mic))
    mic_path = work_dir / "mic_sub.mrc"
    write_mrc_stack(mic_path, mic[np.newaxis, :, :], apix)

    # Stage-1 model & classes stub + Stage-1 data STAR with rlnHelicalType=1
    s1_classes = work_dir / "stage1_classes.mrcs"
    write_mrc_stack(s1_classes, np.zeros((2, 32, 32), dtype=np.float32), apix)

    s1_model = [
        "",
        "data_model_general",
        "_rlnReferenceDimensionality 2",
        "_rlnOriginalImageSize 32",
        f"_rlnPixelSize {apix:.6f}",
        "_rlnNrClasses 2",
        "",
        "data_model_classes",
        "loop_",
        "_rlnReferenceImage #1",
        "_rlnClassDistribution #2",
        "_rlnEstimatedResolution #3",
        f"000001@{s1_classes} 0.5 10.0",
        f"000002@{s1_classes} 0.5 10.0",
    ]
    (work_dir / "stage1_model.star").write_text("\n".join(s1_model) + "\n")

    s1_rows = []
    for s in range(8):
        x = 32.0 + s * 9.0
        s1_rows.append(
            f"00000{s + 1}@dummy.mrcs {mic_path} 1 1 1 {s * 18.0:.1f} 126.0 "
            f"{x:.2f} 64.0 0.0 90.0 0.0 0.0 0.0 0.0 1 15000.0 15000.0 0.0 {s % 2 + 1}"
        )
    s1_star = [
        "",
        "# version 30001",
        "",
        "data_optics",
        "",
        "loop_",
        "_rlnOpticsGroupName #1",
        "_rlnOpticsGroup #2",
        "_rlnMicrographOriginalPixelSize #3",
        "_rlnVoltage #4",
        "_rlnSphericalAberration #5",
        "_rlnAmplitudeContrast #6",
        "_rlnImagePixelSize #7",
        "_rlnImageSize #8",
        "_rlnImageDimensionality #9",
        f"opticsGroup1 1 {apix:.6f} 300.0 2.7 0.1 {apix:.6f} 32 2",
        "",
        "# version 30001",
        "",
        "data_particles",
        "",
        "loop_",
        "_rlnImageName #1",
        "_rlnMicrographName #2",
        "_rlnOpticsGroup #3",
        "_rlnHelicalTubeID #4",
        "_rlnHelicalType #5",
        "_rlnHelicalTrackLengthAngst #6",
        "_rlnHelicalFilamentLengthAngst #7",
        "_rlnCoordinateX #8",
        "_rlnCoordinateY #9",
        "_rlnAngleRot #10",
        "_rlnAngleTilt #11",
        "_rlnAnglePsi #12",
        "_rlnAnglePsiPrior #13",
        "_rlnOriginXAngst #14",
        "_rlnOriginYAngst #15",
        "_rlnClassNumber #16",
        "_rlnDefocusU #17",
        "_rlnDefocusV #18",
        "_rlnDefocusAngle #19",
        "_rlnRandomSubset #20",
    ] + s1_rows
    s1_path = work_dir / "stage1_data.star"
    s1_path.write_text("\n".join(s1_star) + "\n")

    prep_out = work_dir / "sub"
    res_prep = relion_runner(
        [
            str(relion_bin / "relion_helix_subclassify"),
            "--i", str(s1_path),
            "--model", str(work_dir / "stage1_model.star"),
            "--classes", str(s1_classes),
            "--o", str(prep_out),
            "--extract_size", "32",
            "--scale", "32",
            "--K", "2",
            "--min_segs", "2",
            "--premultiply_ctf",
            "--ctf_intact_first_peak",
            "--prepare_only",
        ],
        timeout=120,
    )
    assert res_prep.returncode == 0, f"helix_subclassify failed:\n{res_prep.stderr}\n{res_prep.stdout}"

    prepared_star = work_dir / "sub" / "type1" / "particles.star"
    if not prepared_star.exists():
        matches = list(work_dir.glob("**/particles.star"))
        assert matches, f"No prepared particles.star found under {work_dir}"
        prepared_star = matches[0]
    prep_optics = parse_star_tables(prepared_star)["optics"][1][0]
    assert prep_optics["rlnCtfDataAreCtfPremultiplied"] == "1"
    assert prep_optics["rlnCtfDataAreCtfPremultipliedIntactFirstPeak"] == "1"

    # Matching --ctf_intact_first_peak emits no mismatch warning; omitting it warns clearly
    res_match = relion_runner(
        [
            str(relion_bin / "relion_refine"),
            "--i", str(prepared_star),
            "--o", str(work_dir / "chk_match"),
            "--K", "2", "--iter", "1",
            "--particle_diameter", "52", "--ctf", "--ctf_intact_first_peak",
            "--oversampling", "1", "--psi_step", "15", "--offset_range", "2", "--offset_step", "1",
            "--j", "1", "--dont_combine_weights_via_disc",
        ],
        timeout=120,
    )
    assert res_match.returncode == 0, res_match.stderr
    assert "rlnCtfDataAreCtfPremultipliedIntactFirstPeak" not in res_match.stderr

    res_mismatch = relion_runner(
        [
            str(relion_bin / "relion_refine"),
            "--i", str(prepared_star),
            "--o", str(work_dir / "chk_mismatch"),
            "--K", "2", "--iter", "1",
            "--particle_diameter", "52", "--ctf",
            "--oversampling", "1", "--psi_step", "15", "--offset_range", "2", "--offset_step", "1",
            "--j", "1", "--dont_combine_weights_via_disc",
        ],
        timeout=120,
    )
    assert res_mismatch.returncode == 0, res_mismatch.stderr
    assert "WARNING:" in res_mismatch.stderr
    assert "rlnCtfDataAreCtfPremultipliedIntactFirstPeak" in res_mismatch.stderr


def test_helical_class2d_context_mpi_persists_type_selection(test_data_dir, relion_bin, relion_runner):
    mpi_bin = relion_bin / "relion_refine_mpi"
    if not mpi_bin.exists():
        pytest.skip("relion_refine_mpi not built")

    work_dir = test_data_dir / "helical_ctx_2d_mpi"
    star_path = generate_synthetic_helical_2d(work_dir)
    out_root = work_dir / "run"

    env = os.environ.copy()
    env["OMPI_MCA_rmaps_base_oversubscribe"] = "1"

    cmd = [
        "mpirun", "-np", "2", str(mpi_bin),
        "--i", str(star_path),
        "--o", str(out_root),
        "--iter", "5",
        "--K", "4",
        "--tau2_fudge", "2",
        "--particle_diameter", "84",
        "--helix",
        "--helical_outer_diameter", "56",
        "--helical_cylindrical_mask",
        "--helical_context",
        "--helical_context_types", "2",
        "--helical_context_types_max", "3",
        "--helical_context_types_agree", "0.6",
        "--helical_context_types_every", "5",
        "--helical_context_start", "4",
        "--center_classes",
        "--bimodal_psi",
        "--dont_fix_gpu_bimodal_psi",
        "--flatten_solvent",
        "--zero_mask",
        "--oversampling", "1",
        "--psi_step", "10",
        "--offset_range", "2",
        "--offset_step", "1",
        "--norm",
        "--scale",
        "--j", "1",
        "--random_seed", "7",
        "--dont_combine_weights_via_disc",
        "--pool", "10",
    ]
    res = relion_runner(cmd, timeout=240, env=env)
    assert res.returncode == 0, f"relion_refine_mpi failed:\n{res.stderr}\n{res.stdout}"

    opt_row = parse_star_tables(work_dir / "run_it005_optimiser.star")["optimiser_general"][1][0]
    assert opt_row["rlnHelicalContextTypesSelected"] == "1"
    assert int(opt_row["rlnHelicalContextTypes"]) >= 2
    assert int(opt_row["rlnHelicalContextTypesLastIter"]) >= 1

    cmd_cont = [
        "mpirun", "-np", "2", str(mpi_bin),
        "--continue", str(work_dir / "run_it005_optimiser.star"),
        "--o", str(work_dir / "cont"),
        "--iter", "6",
        "--j", "1",
        "--dont_combine_weights_via_disc",
    ]
    res_cont = relion_runner(cmd_cont, timeout=120, env=env)
    assert res_cont.returncode == 0, f"MPI bare continue failed:\n{res_cont.stderr}\n{res_cont.stdout}"
    assert "choosing the number of types" not in res_cont.stdout
    cont_row = parse_star_tables(work_dir / "cont_it006_optimiser.star")["optimiser_general"][1][0]
    assert cont_row["rlnHelicalContextTypesSelected"] == "1"
    assert cont_row["rlnHelicalContextTypesLastIter"] == opt_row["rlnHelicalContextTypesLastIter"]
    assert cont_row["rlnHelicalContextTypes"] == opt_row["rlnHelicalContextTypes"]


def test_helix_subclassify_multi_optics_and_join_star_intact_first_peak(
    test_data_dir, relion_bin, relion_runner
):
    work_dir = test_data_dir / "subclassify_multi_optics"
    work_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)

    box_mic = 128
    yy, xx = np.mgrid[0:box_mic, 0:box_mic]
    fil_img = np.exp(-0.5 * ((yy - 64.0) / 3.0) ** 2)
    mic1_path = work_dir / "mic_og1.mrc"
    mic2_path = work_dir / "mic_og2.mrc"
    write_mrc_stack(
        mic1_path,
        (fil_img + rng.normal(0.0, 0.1, size=(box_mic, box_mic)))[np.newaxis, :, :],
        1.0,
    )
    write_mrc_stack(
        mic2_path,
        (fil_img + rng.normal(0.0, 0.1, size=(box_mic, box_mic)))[np.newaxis, :, :],
        1.5,
    )

    rows = []
    for s in range(4):
        x = 32.0 + s * 16.0
        rows.append(
            f"{s + 1:06d}@dummy.mrcs {mic1_path} 1 1 1 {s * 16.0:.1f} 64.0 "
            f"{x:.2f} 64.0 0.0 90.0 0.0 0.0 0.0 3.0 15000.0 15000.0 0.0 {s % 2 + 1}"
        )
    for s in range(4):
        x = 32.0 + s * 14.0
        rows.append(
            f"{s + 5:06d}@dummy.mrcs {mic2_path} 2 2 1 {s * 21.0:.1f} 84.0 "
            f"{x:.2f} 64.0 0.0 90.0 0.0 0.0 0.0 3.0 16000.0 16000.0 0.0 {s % 2 + 1}"
        )

    s1_star = [
        "",
        "# version 30001",
        "",
        "data_optics",
        "",
        "loop_",
        "_rlnOpticsGroupName #1",
        "_rlnOpticsGroup #2",
        "_rlnMicrographOriginalPixelSize #3",
        "_rlnVoltage #4",
        "_rlnSphericalAberration #5",
        "_rlnAmplitudeContrast #6",
        "_rlnImagePixelSize #7",
        "_rlnImageSize #8",
        "_rlnImageDimensionality #9",
        "opticsGroup1 1 1.000000 300.0 2.7 0.1 2.000000 16 2",
        "opticsGroup2 2 1.500000 300.0 2.7 0.1 3.000000 14 2",
        "",
        "# version 30001",
        "",
        "data_particles",
        "",
        "loop_",
        "_rlnImageName #1",
        "_rlnMicrographName #2",
        "_rlnOpticsGroup #3",
        "_rlnHelicalTubeID #4",
        "_rlnHelicalType #5",
        "_rlnHelicalTrackLengthAngst #6",
        "_rlnHelicalFilamentLengthAngst #7",
        "_rlnCoordinateX #8",
        "_rlnCoordinateY #9",
        "_rlnAngleRot #10",
        "_rlnAngleTilt #11",
        "_rlnAnglePsi #12",
        "_rlnAnglePsiPrior #13",
        "_rlnOriginXAngst #14",
        "_rlnOriginYAngst #15",
        "_rlnDefocusU #16",
        "_rlnDefocusV #17",
        "_rlnDefocusAngle #18",
        "_rlnRandomSubset #19",
    ] + rows
    s1_path = work_dir / "stage1_multi.star"
    s1_path.write_text("\n".join(s1_star) + "\n")

    prep_out = work_dir / "sub"
    env = os.environ.copy()
    env.pop("RELION_MPIRUN", None)
    res_prep = relion_runner(
        [
            str(relion_bin / "relion_helix_subclassify"),
            "--i", str(s1_path),
            "--o", str(prep_out),
            "--extract_size", "-1",
            "--scale", "32",
            "--K", "2",
            "--min_segs", "2",
            "--premultiply_ctf",
            "--ctf_intact_first_peak",
            "--mpi", "2",
            "--prepare_only",
        ],
        timeout=120,
        env=env,
    )
    assert res_prep.returncode == 0, f"helix_subclassify failed:\n{res_prep.stderr}\n{res_prep.stdout}"

    stage2_sh = (prep_out / "stage2_commands.sh").read_text()
    assert "--bind-to none -n 2" in stage2_sh

    prepared_star = prep_out / "type_1" / "particles.star"
    assert prepared_star.exists()
    prep_optics = parse_star_tables(prepared_star)["optics"][1]
    assert len(prep_optics) == 2
    assert float(prep_optics[0]["rlnMicrographOriginalPixelSize"]) == pytest.approx(1.0, abs=1e-5)
    assert float(prep_optics[0]["rlnImagePixelSize"]) == pytest.approx(1.0, abs=1e-5)
    assert int(prep_optics[0]["rlnImageSize"]) == 32
    assert prep_optics[0]["rlnCtfDataAreCtfPremultiplied"] == "1"
    assert prep_optics[0]["rlnCtfDataAreCtfPremultipliedIntactFirstPeak"] == "1"

    # og2 inferred extract_size = even_round(14 * 3.0 / 1.5) = 28 -> rescaled to 32 -> 1.5 * 28 / 32 = 1.3125
    assert float(prep_optics[1]["rlnMicrographOriginalPixelSize"]) == pytest.approx(1.5, abs=1e-5)
    assert float(prep_optics[1]["rlnImagePixelSize"]) == pytest.approx(1.3125, abs=1e-5)
    assert int(prep_optics[1]["rlnImageSize"]) == 32
    assert prep_optics[1]["rlnCtfDataAreCtfPremultiplied"] == "1"
    assert prep_optics[1]["rlnCtfDataAreCtfPremultipliedIntactFirstPeak"] == "1"

    # Both optics groups' .vstack descriptors are readable by relion_refine without warnings
    res_ref = relion_runner(
        [
            str(relion_bin / "relion_refine"),
            "--i", str(prepared_star),
            "--o", str(work_dir / "chk_multi"),
            "--K", "2", "--iter", "1",
            "--particle_diameter", "28", "--ctf", "--ctf_intact_first_peak",
            "--oversampling", "1", "--psi_step", "15", "--offset_range", "2", "--offset_step", "1",
            "--j", "1", "--dont_combine_weights_via_disc",
        ],
        timeout=120,
    )
    assert res_ref.returncode == 0, f"multi-optics refine failed:\n{res_ref.stderr}\n{res_ref.stdout}"
    assert "rlnCtfDataAreCtfPremultipliedIntactFirstPeak" not in res_ref.stderr

    # JoinStar (--combine) normalizes rlnCtfDataAreCtfPremultipliedIntactFirstPeak when missing in one input
    legacy_star = work_dir / "legacy.star"
    legacy_star.write_text(
        "\n".join(
            [
                "",
                "# version 30001",
                "",
                "data_optics",
                "",
                "loop_",
                "_rlnOpticsGroupName #1",
                "_rlnOpticsGroup #2",
                "_rlnMicrographOriginalPixelSize #3",
                "_rlnVoltage #4",
                "_rlnSphericalAberration #5",
                "_rlnAmplitudeContrast #6",
                "_rlnImagePixelSize #7",
                "_rlnImageSize #8",
                "_rlnImageDimensionality #9",
                "_rlnCtfDataAreCtfPremultiplied #10",
                "opticsGroupLegacy 1 1.000000 300.0 2.7 0.1 1.000000 32 2 0",
                "",
                "# version 30001",
                "",
                "data_particles",
                "",
                "loop_",
                "_rlnImageName #1",
                "_rlnMicrographName #2",
                "_rlnOpticsGroup #3",
                f"000001@dummy.mrcs {mic1_path} 1",
            ]
        )
        + "\n"
    )
    joined_star = work_dir / "joined.star"
    res_join = relion_runner(
        [
            str(relion_bin / "relion_star_handler"),
            "--combine",
            "--i", f"{prepared_star} {legacy_star}",
            "--o", str(joined_star),
        ],
        timeout=60,
    )
    assert res_join.returncode == 0, f"star_handler --combine failed:\n{res_join.stderr}\n{res_join.stdout}"
    joined_optics = parse_star_tables(joined_star)["optics"][1]
    assert len(joined_optics) == 3
    assert [row["rlnCtfDataAreCtfPremultipliedIntactFirstPeak"] for row in joined_optics] == ["1", "1", "0"]

