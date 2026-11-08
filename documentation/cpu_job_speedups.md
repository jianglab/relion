# Faster CPU jobs: MaskCreate, LocalRes, Reconstruct, AutoPick (template), Polish training

Every program here gives the same output as before (bit for bit, except where
noted), whatever the number of threads or MPI processes. Measured on EMPIAR-10940
(8 CPUs, one node, cold cache unless noted).

| Job | Before | After |
|---|---|---|
| MaskCreate (`relion_mask_create`, extend + soft edge) | 44 s | 6 s |
| LocalRes (`relion_postprocess --locres`, sampling 50 A), one process | 716 s | 113 s |
| LocalRes, 8 MPI processes | 1206 s | 808 s |
| Reconstruct (15k particles, `--j 8`, cold cache) | 241 s | 145 s |
| Reconstruct, warm cache, `--j 8` (`--j 1`: 232 s) | 156 s | 57 s |
| Reconstruct MPI, 2 processes x 4 threads, warm cache | 151 s (2 x 1) | 60 s |
| AutoPick template, 8 MPI processes (24 micrographs) | 275 s | 167 s |
| AutoPick template, one process (3 micrographs) | 170 s | 100 s |
| Polish training (`--params3`) | 598 s | 564 s |

## MaskCreate

`autoMask` (also used for the helical Fourier mask of `relion_refine`) extended,
shrank and softened the edge by searching the neighbourhood of every voxel. It now
uses an exact squared Euclidean distance transform (`autoMaskSquaredDistance`) and
the same rules on the integer distances. Test: `[mask_edge]`.

## LocalRes

`--j` gives the threads (default: from the cores per node and the free memory; the
choice and its reason are printed). Each sampling point has its own work arrays and
only touches the cube its mask covers; results are added to the maps in the original
order, so the maps and `*_fscs.star` do not depend on `--j`.

## Reconstruct

With `--j` above 1 the threads first prepare a batch of images (read, FFT, CTF,
shift), then add them to the 3D map together: each thread owns ranges of z planes
(`BackProjector::backproject2Dto3D(..., z_lo, z_hi)`), and a row of the image is only
visited where it can reach those planes. Chunk boundaries are balanced per batch from
the viewing angles; while the current batch is added, the next one is prepared.
Every voxel receives its terms in particle order, so the map equals the old `--j 1`
map. Not used for Ewald sphere, s-squared mode, subtraction, 2D data or rectangular
boxes (they keep the old path), nor with `--j 1`. `RELION_RECONSTRUCT_PLANES=0`
switches it off. Test: `[bp_planes]`, `relion_reconstruct_mpi` check by hand.

## AutoPick (template)

Each MPI process used to prepare all references (a padded micrograph-size FFT each),
and the first-psi statistics resized one array per reference. Now every process
makes every n-th reference, with threads inside, and broadcasts them; the statistics
use their own full-size array and transformer. With the NUFFT interpolator the
projectors hold more than `data`, so each process prepares all references as before.
Picks are identical. `RELION_AUTOPICK_PREP_THREADS` sets the threads.

## Polish training

Only the reference prediction, which does not depend on the frame, is no longer
repeated per frame (stage "prepare alignment"). The track fitting (about 450 s of
564 s) was already using about 7.5 cores and is unchanged. Stage times:
`RELION_POLISH_TIMING=1`.
