# Down-weighting segments at the ends of filaments

A helical segment near the start or end of its filament is extracted in a box
that reaches beyond the end of the filament: part of the box holds no
filament. Such segments blur the ends of 2D class averages and add a
partly empty density to 3D reconstructions.

`--helical_end_weight <w>` (relion_refine: Class2D, Class3D and Refine3D)
multiplies the contribution of these end segments to the **reconstruction**
(the class averages in 2D) by `w`, between 0 and 1:

- `1` (the default): off.
- `0`: end segments are left out of the reconstruction altogether.

## Which segments are end segments

Helical extraction writes, for every segment, its position along the filament
(`rlnHelicalTrackLengthAngst`) and the length of its filament
(`rlnHelicalFilamentLengthAngst`). The filament runs from 0 to its length; the
segment's box covers `track +/- box/2` along it, where `box` is the image size
along x (`rlnImageSize`, or its x size for rectangular boxes) times the image
pixel size. A segment is an end segment when

    track - box/2 < 0   or   track + box/2 > filament length

with half a pixel of tolerance. Both columns are needed; without
`rlnHelicalFilamentLengthAngst` (particles extracted before it was added) the
run stops with an error, and re-extracting the segments adds it.

For filaments picked by their start and end points (or as lines) the filament
length is the picked length. Otherwise (segment coordinates) it is the
distance between the first and last segment, so the first and last segments
always count as end segments there.

The run reports the count at the start, for example

    Helical end segments: 1520 of 9800 segments reach beyond an end of their filament; their weight in the reconstruction is 0.3

## What changes

Only the back-projection (both the data and the CTF^2 weight terms) of these
segments is scaled. The alignment, the posterior probabilities, the
orientations and classes written to `_data.star`, and the sums for the noise
spectra, the class and direction distributions and the offsets all keep the
full weights, as with `--helical_tilt_weight_sigma`.

- Not for subtomograms or multi-body refinement.
- Not stored in the optimiser STAR file: it is read from the command line
  again on `--continue`, so it can be switched on, changed or switched off
  between iterations. The GUI passes it on continuation too.
- GUI: Helix tab of Class2D, Class3D and Refine3D, "Weight of filament-end
  segments" (only used when the job is helical).
- CPU and GPU (CUDA, HIP, SYCL, ALTCPU). On the CPU the summed image and
  CTF^2 weight are scaled before they are back-projected. On the GPU the
  back-projection kernels divide each weight by the particle's sum of
  weights; that sum is divided by `w`. With `w = 0` the back-projection of
  the segment is skipped.
