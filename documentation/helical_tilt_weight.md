# Down-weighting out-of-plane tilts in helical reconstructions

In helical 3D classification and refinement the tilt distribution often drifts
away from 90 degrees and piles up at both edges of the tilt search range. Once
those orientations are in the reconstruction they make the map look more like
a tilted filament, which pulls more segments to high tilts in the next
iteration.

`--helical_tilt_weight_sigma <deg>` (relion_refine, Class3D and Refine3D) multiplies
the contribution of every orientation to the **reconstruction** by

    w(tilt) = exp( -(tilt - 90)^2 / (2 sigma^2) )

Nothing else changes: the alignment, the posterior probabilities, the
orientations written to `_data.star`, and the sums for the noise spectra,
the class and direction distributions and the offsets all use the full
weights. Only the back-projection (both the data and the CTF^2 weight terms)
is scaled, so a segment whose orientations are all at high tilt still counts
for the noise model but adds less to the map.

- Off by default (`-1`, or any value <= 0).
- Applies to 3D references only, and not to multi-body refinement.
- Not stored in the optimiser STAR file: it is read from the command line
  again on `--continue`, so it can be switched on, changed or switched off
  between iterations. The GUI passes it on continuation too.
- GUI: Helix tab of Class3D and Refine3D, "Tilt down-weighting sigma (deg)"
  (only used when "Do helical reconstruction?" is Yes).
- CPU and GPU (CUDA, HIP, SYCL, ALTCPU) paths. With CUDA the weight map is
  scaled on the device (`cuda_kernel_scale_orientation_weights`; only one
  factor per orientation is uploaded). With HIP and SYCL the map goes to the
  host and back once more per particle when the option is on.

For orientation, the weight at the edge of a tilt search range of +/- R degrees
is 0.61 for sigma = R, 0.14 for sigma = R/2 and 0.011 for sigma = R/3.
