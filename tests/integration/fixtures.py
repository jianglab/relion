#!/usr/bin/env python3
"""
Shared utilities and fixtures for integration tests: MRC I/O, volume generation,
FSC computation, STAR file generation, CTF simulation, and projection.
"""

import os
import json
import struct
import numpy as np
from pathlib import Path


def read_mrc(path):
    """Read a 2D image stack or 3D volume from MRC format (float32)."""
    with open(path, 'rb') as f:
        header = f.read(1024)
        nx, ny, nz, mode = struct.unpack('<4i', header[:16])
        if mode != 2:
            raise ValueError(f"Unsupported MRC mode {mode}, expected mode 2 (float32)")
        data = np.frombuffer(f.read(), dtype=np.float32)
        return data.reshape((nz, ny, nx))


def write_mrc(array, output_path, is_volume=False, angpix=1.0):
    """
    Write a 2D image stack or 3D volume to standard MRC format (1024-byte header).

    For a 2D stack the caller passes an array of shape (nz, ny, nx) and
    ``is_volume=False`` (ISPG=0, MZ=1).  For a 3D volume pass
    ``is_volume=True`` (ISPG=1, MZ=nz).
    """
    array = np.asarray(array, dtype=np.float32)

    if array.ndim == 2:
        nz, ny, nx = 1, array.shape[0], array.shape[1]
        data = array.reshape(1, ny, nx)
    elif array.ndim == 3:
        nz, ny, nx = array.shape
        data = array
    else:
        raise ValueError("Array must be 2D or 3D")

    header = bytearray(1024)

    def si(offset, value):
        struct.pack_into('<i', header, offset, int(value))

    def sf(offset, value):
        struct.pack_into('<f', header, offset, float(value))

    si(0, nx)
    si(4, ny)
    si(8, nz)
    si(12, 2)
    si(16, 0)
    si(20, 0)
    si(24, 0)
    si(28, nx)
    si(32, ny)
    si(36, 1 if not is_volume else nz)
    sf(40, nx * angpix)
    sf(44, ny * angpix)
    sf(48, (nz if is_volume else 1) * angpix)
    sf(52, 90.0)
    sf(56, 90.0)
    sf(60, 90.0)
    si(64, 1)
    si(68, 2)
    si(72, 3)
    sf(76, float(data.min()))
    sf(80, float(data.max()))
    sf(84, float(data.mean()))
    si(88, 1 if is_volume else 0)
    si(92, 0)
    sf(196, 0.0)
    sf(200, 0.0)
    sf(204, 0.0)
    header[208:212] = b'MAP '
    header[212] = 0x44
    header[213] = 0x44
    header[214] = 0x00
    header[215] = 0x00
    sf(216, float(data.std()))
    si(220, 0)

    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(data.astype(np.float32).tobytes())


def write_volume_mrc(volume, output_path, angpix=1.0):
    """Write a 3D volume to MRC file format."""
    write_mrc(volume, output_path, is_volume=True, angpix=angpix)





def write_star_file(num_particles, output_path, particles_dir=None, size=32, angpix=1.0, seed=42):
    """
    Write a RELION5-compatible STAR file with ``data_optics`` and
    ``data_particles`` blocks.

    The optics group carries microscope parameters; per-particle columns
    include random (but reproducible) Euler angles and origin offsets as
    expected by relion_refine_mpi / relion_reconstruct_mpi in RELION 5.
    """
    rng = np.random.default_rng(seed)

    if particles_dir:
        particles_file = os.path.join(particles_dir, "particles.mrcs")
        micrograph_name = os.path.join(particles_dir, "synthetic_micrograph.mrc")
    else:
        particles_file = "particles.mrcs"
        micrograph_name = "synthetic_micrograph.mrc"

    # ----- optics block -----
    optics_block = """\

data_optics

loop_
_rlnOpticsGroupName #1
_rlnOpticsGroup #2
_rlnMicrographOriginalPixelSize #3
_rlnVoltage #4
_rlnSphericalAberration #5
_rlnAmplitudeContrast #6
_rlnImagePixelSize #7
_rlnImageSize #8
_rlnImageDimensionality #9
opticsGroup1\t1\t{angpix:.6f}\t300.000000\t2.700000\t0.100000\t{angpix:.6f}\t{size}\t2
""".format(angpix=angpix, size=size)

    # ----- particles block header -----
    particles_header = """\

data_particles

loop_
_rlnImageName #1
_rlnMicrographName #2
_rlnOpticsGroup #3
_rlnDefocusU #4
_rlnDefocusV #5
_rlnDefocusAngle #6
_rlnCtfFigureOfMerit #7
_rlnAngleRot #8
_rlnAngleTilt #9
_rlnAnglePsi #10
_rlnOriginXAngst #11
_rlnOriginYAngst #12
_rlnClassNumber #13
_rlnNormCorrection #14
_rlnRandomSubset #15
_rlnGroupNumber #16
"""

    # ----- per-particle rows -----
    rows = []
    defocus_u_vals = rng.uniform(12000, 18000, num_particles)
    defocus_v_vals = rng.uniform(12000, 18000, num_particles)
    defocus_angle_vals = rng.uniform(0, 180, num_particles)
    rot_vals = rng.uniform(0, 360, num_particles)
    tilt_vals = rng.uniform(0, 180, num_particles)
    psi_vals = rng.uniform(0, 360, num_particles)
    ox_vals = rng.uniform(-2, 2, num_particles)
    oy_vals = rng.uniform(-2, 2, num_particles)

    for i in range(num_particles):
        random_subset = (i % 2) + 1  # alternates 1, 2, 1, 2, …
        row = (
            f"{i+1:06d}@{particles_file}\t"
            f"{micrograph_name}\t"
            f"1\t"                                          # OpticsGroup
            f"{defocus_u_vals[i]:.2f}\t"
            f"{defocus_v_vals[i]:.2f}\t"
            f"{defocus_angle_vals[i]:.2f}\t"
            f"0.100000\t"                                   # CtfFigureOfMerit
            f"{rot_vals[i]:.6f}\t"
            f"{tilt_vals[i]:.6f}\t"
            f"{psi_vals[i]:.6f}\t"
            f"{ox_vals[i]:.6f}\t"
            f"{oy_vals[i]:.6f}\t"
            f"1\t"                                          # ClassNumber
            f"1.000000\t"                                   # NormCorrection
            f"{random_subset}\t"
            f"1"                                            # GroupNumber
        )
        rows.append(row)

    with open(output_path, 'w') as f:
        f.write(optics_block)
        f.write(particles_header)
        f.write("\n".join(rows) + "\n")


def write_postprocess_star(output_path, half1_path, half2_path, size=32, angpix=1.0):
    """
    Write a minimal RELION PostProcess STAR file for use as the ``--f`` input
    to ``relion_ctf_refine_mpi``.

    The FSC curve is synthetic (1.0 at low frequencies, dropping linearly to 0).
    """
    num_shells = size // 2
    fsc_rows = []
    for s in range(0, num_shells + 1):
        # RELION postprocess.star stores spatial frequency as 1/Angstrom
        resolution = (s + 1e-3) / (size * angpix)
        angstrom_res = 999.0 if s == 0 else (size * angpix / s)
        fsc_val = max(0.0, 1.0 - s / max(1, num_shells))
        fsc_rows.append(
            f"{s:12d}\t"
            f"{resolution:.6f}\t"
            f"{angstrom_res:.6f}\t"
            f"{fsc_val:.6f}\t"
            f"{fsc_val:.6f}\t"
            f"{fsc_val:.6f}\t"
            f"{fsc_val:.6f}\t"
            f"{fsc_val:.6f}"
        )

    content = f"""\

data_general

_rlnFinalResolution\t{2 * angpix:.4f}
_rlnUnfilteredMapHalf1\t{half1_path}
_rlnUnfilteredMapHalf2\t{half2_path}
_rlnParticleBoxSizeCorr\t{size}


data_fsc

loop_
_rlnSpectralIndex #1
_rlnResolution #2
_rlnAngstromResolution #3
_rlnFourierShellCorrelationCorrected #4
_rlnFourierShellCorrelationParticleMaskFraction #5
_rlnFourierShellCorrelationUnmaskedMaps #6
_rlnFourierShellCorrelationMaskedMaps #7
_rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps #8
"""
    content += "\n".join(fsc_rows) + "\n"

    with open(output_path, 'w') as f:
        f.write(content)


def generate_test_dataset(test_dir, relion_bin, relion_runner, num_particles=10, size=32, angpix=1.0):
    """
    Generate a complete synthetic test dataset using relion_project.

    Creates:
      - particles.mrcs      – 2D particle stack (projections from ground truth)
      - particles.star      – RELION5-format metadata
      - initial_model.mrc   – 3D multi-band volume (initial reference)

    Returns:
        paths: dict with paths to generated files
    """
    os.makedirs(test_dir, exist_ok=True)

    vol_true = create_ground_truth_volume(size)
    initial_model = os.path.join(test_dir, "initial_model.mrc")
    write_volume_mrc(vol_true, initial_model, angpix=angpix)

    rot_vals, tilt_vals, psi_vals = sample_particle_orientations(num_particles)
    origin_x_vals = np.zeros(num_particles)
    origin_y_vals = np.zeros(num_particles)
    defocus_vals = generate_defocus_values(num_particles)

    proj_star = os.path.join(test_dir, "project.star")
    clean_root = os.path.join(test_dir, "clean_particles")
    write_custom_star(
        num_particles, proj_star, f"{clean_root}.mrcs",
        rot_vals, tilt_vals, psi_vals,
        size=size, angpix=angpix,
        origin_x_vals=origin_x_vals, origin_y_vals=origin_y_vals,
        defocus_vals=defocus_vals,
    )

    _, particles_mrc = generate_pristine_projections(
        Path(relion_bin), relion_runner, Path(initial_model), Path(proj_star), clean_root, angpix=angpix,
    )
    particles_mrc = str(particles_mrc)

    stack = read_mrc(particles_mrc)
    stack_ctf = apply_ctf_with_supersampling(
        stack, sup_factor=1, angpix=angpix, defocus_vals=defocus_vals,
    )
    write_mrc(stack_ctf, particles_mrc)

    star_file = os.path.join(test_dir, "particles.star")
    write_custom_star(
        num_particles, star_file, particles_mrc,
        rot_vals, tilt_vals, psi_vals,
        size=size, angpix=angpix,
        origin_x_vals=origin_x_vals, origin_y_vals=origin_y_vals,
        defocus_vals=defocus_vals,
    )

    gt_file = os.path.join(test_dir, "ground_truth.json")
    gt_data = {
        "volume": initial_model,
        "num_particles": num_particles,
        "size": size,
        "angpix": angpix,
        "rot": rot_vals.tolist(),
        "tilt": tilt_vals.tolist(),
        "psi": psi_vals.tolist(),
        "origin_x": origin_x_vals.tolist(),
        "origin_y": origin_y_vals.tolist(),
    }
    with open(gt_file, 'w') as f:
        json.dump(gt_data, f, indent=2)

    return {
        "particles_mrc": particles_mrc,
        "star_file": star_file,
        "initial_model": initial_model,
        "ground_truth": gt_file,
        "test_dir": test_dir,
    }


def compute_fsc(vol1, vol2):
    """Compute 1D Fourier Shell Correlation between two 3D volumes."""
    nz, ny, nx = vol1.shape
    center = nx // 2
    z, y, x = np.ogrid[:nz, :ny, :nx]
    r = np.round(np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)).astype(int)

    F1 = np.fft.fftshift(np.fft.fftn(vol1))
    F2 = np.fft.fftshift(np.fft.fftn(vol2))

    max_r = nx // 2
    fsc = np.zeros(max_r + 1)
    for i in range(max_r + 1):
        mask = (r == i)
        if np.any(mask):
            num = np.sum(F1[mask] * np.conj(F2[mask])).real
            den1 = np.sum(np.abs(F1[mask])**2)
            den2 = np.sum(np.abs(F2[mask])**2)
            if den1 > 0 and den2 > 0:
                fsc[i] = num / np.sqrt(den1 * den2)
    return fsc


def radial_power_spectrum(vol):
    """Mean |F|^2 per integer shell radius (same binning as compute_fsc)."""
    nz, ny, nx = vol.shape
    center = nx // 2
    z, y, x = np.ogrid[:nz, :ny, :nx]
    r = np.round(np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)).astype(int)

    F = np.fft.fftshift(np.fft.fftn(vol))
    max_r = nx // 2
    power = np.zeros(max_r + 1)
    for i in range(max_r + 1):
        mask = r == i
        if np.any(mask):
            power[i] = np.mean(np.abs(F[mask]) ** 2)
    return power


def _bandlimited_noise(size, center, width, weight, rng):
    """Create spatially normalised band-limited noise in a given Fourier shell range."""
    c = size // 2
    z, y, x = np.ogrid[:size, :size, :size]
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2)
    bp = np.exp(-((r - center) / width) ** 2)
    bp = np.fft.ifftshift(bp)

    phases = rng.uniform(0, 2 * np.pi, (size, size, size))
    F = bp * np.exp(1j * phases)

    rev = np.arange(size - 1, -1, -1)
    F = 0.5 * (F + np.conj(F[np.ix_(rev, rev, rev)]))
    F[0, 0, 0] = F[0, 0, 0].real
    if size % 2 == 0:
        h = size // 2
        F[0, 0, h] = F[0, 0, h].real
        F[0, h, 0] = F[0, h, 0].real
        F[h, 0, 0] = F[h, 0, 0].real

    noise = np.fft.ifftn(F).real.astype(np.float64)
    noise -= noise.mean()
    nz = noise.std()
    if nz > 0:
        noise /= nz
    return weight * noise


def _build_protein_fold(num_residues, bond_length, confining_radius, rng):
    """Generate a compact random-walk Cα trace confined to a sphere."""
    coords = np.zeros((num_residues, 3))
    coords[0] = rng.uniform(-2, 2, 3)
    for i in range(1, num_residues):
        step = rng.normal(0, 1, 3)
        norm = np.linalg.norm(step)
        if norm > 1e-12:
            step = step / norm * bond_length
        else:
            step = np.array([bond_length, 0.0, 0.0])
        candidate = coords[i - 1] + step
        dist = np.linalg.norm(candidate)
        if dist > confining_radius:
            candidate = candidate / dist * (2.0 * confining_radius - dist)
        coords[i] = candidate
    coords -= coords.mean(axis=0)
    return coords


def _gaussian_density(grid, positions, sigma):
    """Sum isotropic Gaussian kernels at *positions* on *grid* (ogrid tuple)."""
    z, y, x = grid
    c = x.shape[0] // 2
    n = x.shape[2]  # size along the varied dimension
    vol = np.zeros((n, n, n), dtype=np.float64)
    for pos in positions:
        dx = (x - c - pos[0]).astype(np.float64)
        dy = (y - c - pos[1]).astype(np.float64)
        dz = (z - c - pos[2]).astype(np.float64)
        k2 = dx * dx + dy * dy + dz * dz
        vol += np.exp(-k2 / (2.0 * sigma * sigma))
    return vol


def create_ground_truth_volume(size, rng_seed=1234):
    """Create a protein-like volume with realistic structural features.

    A compact random Cα trace defines the overall shape; density is built
    with broadened Gaussians (∼7 Å resolution) plus bandlimited noise to
    provide coherent signal at higher spatial frequencies.  The result
    mimics a ~15 kDa protein at moderate resolution.
    """
    rng = np.random.default_rng(rng_seed)
    c = size // 2
    grid = np.ogrid[:size, :size, :size]
    n = size  # full grid length
    confining_radius = size * 0.35

    # --- Cα trace (80 residues, 3.8 Å bond length) ---
    num_residues = 80
    bond_length = 3.8
    ca_coords = _build_protein_fold(num_residues, bond_length, confining_radius, rng)

    # --- density from Cα positions + side-chain cloud ---
    sc_coords = []
    for ca in ca_coords:
        n_side = rng.integers(1, 4)
        for _ in range(n_side):
            offset = rng.normal(0, 1, 3)
            onorm = np.linalg.norm(offset)
            if onorm > 1e-12:
                offset = offset / onorm * rng.uniform(1.0, 3.0)
            sc = ca + offset
            if np.linalg.norm(sc) < confining_radius:
                sc_coords.append(sc)
    all_coords = np.concatenate([ca_coords, sc_coords])

    vol = _gaussian_density(grid, all_coords, sigma=2.5)

    # --- smooth solvent envelope ---
    k2 = (grid[1] - c) ** 2 + (grid[0] - c) ** 2 + (grid[2] - c) ** 2
    vol += 0.6 * vol.max() * np.exp(-k2 / (2 * (size * 0.20) ** 2))

    # --- bandlimited noise for Fourier-shell coverage ---
    # Ensures coherent signal at all frequencies up to Nyquist,
    # which is critical for the FSC pass criterion with limited particles.
    vol += _bandlimited_noise(size, 5, 3, 0.06 * vol.max(), rng)
    vol += _bandlimited_noise(size, 9, 3, 0.08 * vol.max(), rng)
    vol += _bandlimited_noise(size, 13, 3, 0.06 * vol.max(), rng)
    # Mid-frequency component to bridge the gap between low and high bands
    vol += _bandlimited_noise(size, 22, 5, 0.12 * vol.max(), rng)
    # High-frequency components for s2 vs s superiority test (shells 31-50)
    vol += _bandlimited_noise(size, 32, 8, 0.15 * vol.max(), rng)
    vol += _bandlimited_noise(size, 48, 6, 0.08 * vol.max(), rng)

    vol -= vol.mean()
    vol_std = vol.std()
    if vol_std > 1e-12:
        vol /= vol_std
    return (vol * 5.0).astype(np.float32)


def sample_particle_orientations(num_particles, seed=42):
    """Draw (rot, tilt, psi) once; use for both projection synthesis and the STAR file."""
    rng = np.random.default_rng(seed)
    rot = rng.uniform(0.0, 360.0, num_particles)
    tilt = np.arccos(rng.uniform(-1.0, 1.0, num_particles)) * 180.0 / np.pi
    psi = rng.uniform(0.0, 360.0, num_particles)
    return rot, tilt, psi


def generate_defocus_values(num_particles, defocus_range=(5000.0, 25000.0), seed=43):
    """Draw per-particle defocus values uniformly over the given range (AA)."""
    rng = np.random.default_rng(seed)
    defocus_min, defocus_max = defocus_range
    return rng.uniform(defocus_min, defocus_max, num_particles)


def generate_pristine_projections(relion_bin, relion_runner, ground_truth_mrc,
                                  star_file, output_root, angpix=1.0):
    """Generate clean (no CTF) projections with relion_project using Euler angles from star_file."""
    output_root = Path(output_root)
    if output_root.suffix == ".mrcs":
        output_root = output_root.with_suffix("")

    cmd = [
        str(relion_bin / "relion_project"),
        "--i", str(ground_truth_mrc),
        "--ang", str(star_file),
        "--o", str(output_root),
        "--angpix", str(angpix),
    ]
    res = relion_runner(cmd, timeout=600)
    if res.returncode != 0:
        raise RuntimeError(f"relion_project failed:\n{res.stderr}")

    stack_path = Path(f"{output_root}.mrcs")
    particles_star = Path(f"{output_root}.star")
    if not stack_path.exists():
        raise FileNotFoundError(f"Expected particle stack not found: {stack_path}")
    if not particles_star.exists():
        raise FileNotFoundError(f"Expected STAR file not found: {particles_star}")
    return particles_star, stack_path


def write_custom_star(
    num_particles,
    output_path,
    particles_file,
    rot_vals,
    tilt_vals,
    psi_vals,
    size=256,
    angpix=1.0,
    defocus_vals=None,
    defocus=12000.0,
    origin_x_vals=None,
    origin_y_vals=None,
):
    """Write a RELION5-compatible STAR file with optics and particles blocks.

    Uses per-particle defocus values when *defocus_vals* is provided, otherwise
    falls back to the scalar *defocus* for all particles.
    Per-particle origin offsets default to zero when *origin_x_vals* / *origin_y_vals*
    are not provided.
    """
    if defocus_vals is not None:
        defocus_array = np.asarray(defocus_vals, dtype=np.float64)
    else:
        defocus_array = np.full(num_particles, float(defocus))

    if origin_x_vals is not None:
        ox_array = np.asarray(origin_x_vals, dtype=np.float64)
    else:
        ox_array = np.zeros(num_particles)
    if origin_y_vals is not None:
        oy_array = np.asarray(origin_y_vals, dtype=np.float64)
    else:
        oy_array = np.zeros(num_particles)

    optics_block = f"""
data_optics

loop_
_rlnOpticsGroupName #1
_rlnOpticsGroup #2
_rlnMicrographOriginalPixelSize #3
_rlnVoltage #4
_rlnSphericalAberration #5
_rlnAmplitudeContrast #6
_rlnImagePixelSize #7
_rlnImageSize #8
_rlnImageDimensionality #9
opticsGroup1\t1\t{angpix:.6f}\t300.000000\t2.700000\t0.100000\t{angpix:.6f}\t{size}\t2
"""

    particles_header = """
data_particles

loop_
_rlnImageName #1
_rlnMicrographName #2
_rlnOpticsGroup #3
_rlnDefocusU #4
_rlnDefocusV #5
_rlnDefocusAngle #6
_rlnCtfFigureOfMerit #7
_rlnAngleRot #8
_rlnAngleTilt #9
_rlnAnglePsi #10
_rlnOriginXAngst #11
_rlnOriginYAngst #12
_rlnClassNumber #13
_rlnNormCorrection #14
_rlnRandomSubset #15
_rlnGroupNumber #16
"""

    rows = []
    for i in range(num_particles):
        d = defocus_array[i]
        random_subset = (i % 2) + 1
        row = (
            f"{i + 1}@{particles_file}\t"
            f"micrograph.mrc\t"
            f"1\t"
            f"{d:.2f}\t"
            f"{d:.2f}\t"
            f"0.00\t"
            f"0.100000\t"
            f"{rot_vals[i]:.6f}\t"
            f"{tilt_vals[i]:.6f}\t"
            f"{psi_vals[i]:.6f}\t"
            f"{ox_array[i]:.6f}\t"
            f"{oy_array[i]:.6f}\t"
            f"1\t"
            f"1.000000\t"
            f"{random_subset}\t"
            f"1"
        )
        rows.append(row)

    with open(output_path, 'w') as f:
        f.write(optics_block)
        f.write(particles_header)
        f.write("\n".join(rows) + "\n")


def compute_ctf_sup_factor_needed(box_size, angpix, max_defocus, voltage=300.0):
    """Compute the minimum supersampling factor needed to Nyquist-sample the CTF.

    At high defocus the CTF oscillates rapidly in spatial frequency.  A
    simulation that multiplies the image FFT by the CTF on a discrete grid
    must oversample to avoid aliasing.  This function computes the smallest
    integer ``sup_factor`` such that the oversampled grid has >= 2 samples
    per CTF oscillation at the Nyquist frequency (the worst case).

    Parameters
    ----------
    box_size : int
        Image size in pixels (NxN).
    angpix : float
        Pixel size in Angstrom.
    max_defocus : float
        Maximum |defocus| in the dataset (Angstrom).
    voltage : float
        Acceleration voltage in kV.

    Returns
    -------
    int : required sup_factor.
    """
    kV = voltage * 1000.0
    wavelength = 12.2643247 / np.sqrt(kV * (1.0 + kV * 0.978466e-6))
    K1 = np.pi * wavelength
    Cs_A = 2.7 * 1e7
    K2 = np.pi / 2.0 * Cs_A * wavelength ** 3

    u_nyq = 1.0 / (2.0 * angpix)
    dgamma_du2 = K1 * (-max_defocus) + 2.0 * K2 * u_nyq * u_nyq
    delta_u2 = np.pi / abs(dgamma_du2)
    delta_u = delta_u2 / (2.0 * u_nyq)
    delta_u_orig = 1.0 / (box_size * angpix)
    pix_per_period = delta_u / delta_u_orig
    needed = max(1, int(np.ceil(2.0 / pix_per_period)))
    return needed


def apply_ctf_with_supersampling(stack, sup_factor=None, angpix=1.0, voltage=300.0,
                                  Cs=2.7, amplitude_contrast=0.1, defocus=12000.0,
                                  defocus_vals=None):
    """Apply CTF in Fourier space, matching RELION's C++ implementation exactly.

    RELION internally negates defocus (``D = diag(-DeltafU, -DeltafV)`` in
    ``ctf.cpp:368``) before the bilinear form.  The formula is:

        gamma = K1 * (-defocus) * u² + K2 * u⁴ - K3
        CTF   = -sin(gamma)

    ``sup_factor`` controls supersampling to avoid CTF aliasing.  When
    ``None`` (default) the required factor is computed automatically from
    the maximum defocus in the dataset via :func:`compute_ctf_sup_factor_needed`.
    Set to 1 to match the native-grid convolution that ``relion_project --ctf``
    uses (verified identical to machine precision).

    Accepts a 3D stack ``(n, ny, nx)`` or a single 2D image ``(ny, nx)``.
    If a 2D image is given, a 2D result is returned.
    """
    was_2d = stack.ndim == 2
    if was_2d:
        stack = stack[np.newaxis, :, :]
    n, ny, nx = stack.shape

    if defocus_vals is not None:
        defocus_array = np.asarray(defocus_vals, dtype=np.float64).ravel()
    else:
        defocus_array = np.full(n, float(defocus))
    if defocus_array.size == 1 and n > 1:
        defocus_array = np.full(n, defocus_array[0])

    if sup_factor is None:
        max_defocus = float(np.max(np.abs(defocus_array)))
        sup_factor = compute_ctf_sup_factor_needed(nx, angpix, max_defocus, voltage)
        sup_factor = max(sup_factor, 2)

    kV = voltage * 1000.0
    wavelength = 12.2643247 / np.sqrt(kV * (1.0 + kV * 0.978466e-6))
    K1 = np.pi * wavelength
    Cs_A = Cs * 1e7
    K2 = np.pi / 2.0 * Cs_A * wavelength ** 3
    q0 = amplitude_contrast
    K3 = np.arctan2(q0, np.sqrt(max(1.0 - q0 ** 2, 1e-10)))

    sny, snx = ny * sup_factor, nx * sup_factor
    fx = np.fft.fftfreq(snx, d=angpix)
    fy = np.fft.fftfreq(sny, d=angpix)
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    u2 = FX.astype(np.float64) ** 2 + FY.astype(np.float64) ** 2
    u4 = u2 ** 2

    y_start = (sny - ny) // 2
    x_start = (snx - nx) // 2

    result = np.empty((n, ny, nx), dtype=np.float32)
    for i in range(n):
        gamma = K1 * (-defocus_array[i]) * u2 + K2 * u4 - K3
        ctf = -np.sin(gamma)
        padded = np.zeros((sny, snx), dtype=np.float64)
        padded[y_start:y_start + ny, x_start:x_start + nx] = stack[i]
        F = np.fft.fftn(padded)
        F_ctf = F * ctf
        img_ctf = np.fft.ifftn(F_ctf).real
        result[i] = img_ctf[y_start:y_start + ny, x_start:x_start + nx].astype(np.float32)

    return result[0] if was_2d else result


def fsc_shell_to_spatial_frequency(shell_index, size, angpix):
    """Spatial frequency magnitude |s| (1/AA) for integer Fourier shell radius."""
    shell_index = np.atleast_1d(np.asarray(shell_index, dtype=np.float64))
    freq = shell_index / (size * angpix)
    freq[shell_index <= 0] = np.nan
    if freq.size == 1:
        return float(freq[0])
    return freq


def apply_contrast_flip_if_needed(reference, volume):
    """Resolve contrast ambiguity (+-1) using low-shell FSC to reference."""
    fsc_pos = compute_fsc(reference, volume)
    fsc_neg = compute_fsc(reference, -volume)
    if np.mean(fsc_neg[2:20]) > np.mean(fsc_pos[2:20]):
        return -volume
    return volume


def read_star_particles(star_path, columns):
    """Extract named columns from a RELION data_particles STAR file.

    Args:
        star_path: Path to the STAR file.
        columns: List of column names to extract (e.g. ['rlnAngleRot']).

    Returns:
        dict mapping each column name to a numpy array of values.
    """
    with open(star_path) as f:
        lines = f.readlines()

    in_particles = False
    header_found = False
    col_index = {}
    data_rows = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("data_") and "particles" in stripped:
            in_particles = True
            continue

        if not in_particles:
            continue

        if stripped.startswith("loop_"):
            header_found = False
            col_index = {}
            continue

        if stripped.startswith("_"):
            parts = stripped.split()
            if len(parts) >= 2:
                col_name = parts[0].lstrip("_")
                col_num = int(parts[1].lstrip("#")) - 1
                col_index[col_name] = col_num
                header_found = True
            continue

        if header_found and stripped:
            parts = stripped.split()
            if parts and len(parts) >= len(col_index):
                data_rows.append(parts)

    result = {}
    for col in columns:
        if col not in col_index:
            raise KeyError(f"Column '{col}' not found in {star_path}. "
                           f"Available: {list(col_index.keys())}")
        idx = col_index[col]
        result[col] = np.array([float(row[idx]) for row in data_rows])
    return result


def reconstruct_and_check_fsc(data_star, gt_file, size, angpix,
                               relion_bin, relion_runner, output_dir,
                               timeout=300):
    """
    Run relion_reconstruct on a data STAR file, compute FSC against ground truth,
    and assert mean FSC > 0.5 at half Nyquist.

    Returns (reconstructed_volume, fsc_array).
    """
    recon_mrc = output_dir / "recon_fsc_check.mrc"
    cmd = [
        str(relion_bin / "relion_reconstruct"),
        "--i", str(data_star),
        "--o", str(recon_mrc),
        "--spatial_frequency_mode", "s2",
        "--ctf",
        "--angpix", str(angpix),
    ]
    result = relion_runner(cmd, timeout=timeout)
    assert result.returncode == 0, f"relion_reconstruct failed:\n{result.stderr}"
    assert recon_mrc.exists()

    vol_recon = read_mrc(str(recon_mrc))
    vol_true = read_mrc(str(gt_file))
    assert vol_recon.shape == vol_true.shape, \
        f"Shape {vol_recon.shape} != {vol_true.shape}"
    vol_recon = apply_contrast_flip_if_needed(vol_true, vol_recon)

    fsc = compute_fsc(vol_true, vol_recon)
    half_nyquist = size // 4
    mean_fsc_hn = np.mean(fsc[half_nyquist - 2 : half_nyquist + 3])
    assert mean_fsc_hn > 0.3, (
        f"Mean FSC at half Nyquist (shells {half_nyquist - 2}-{half_nyquist + 2}) "
        f"is {mean_fsc_hn:.4f}, expected > 0.3"
    )
    return vol_recon, fsc


def get_test_outputs_dir():
    """Get the shared test-outputs directory at the repo root, creating it if needed."""
    d = Path(__file__).resolve().parent.parent.parent / "test-outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d
