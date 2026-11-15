# Fractional particle coordinates and box centres

Particle coordinates (`rlnCoordinateX/Y`) need not be whole pixels: cryoSPARC imports, many pickers and
helical segment placement give fractions. A box has to be cut around a whole pixel, and the particle's
offsets (`rlnOriginX/YAngst`) are measured from that pixel. Programs that cut boxes from micrographs or
movies must therefore agree on which pixel it is:

| Program | Box centre for x = 100.6 |
|---|---|
| Extract, this RELION since 2026-10-05 | 101 (rounded) |
| Extract, upstream RELION (and this RELION before) | 100 (truncated) |
| Polish (`relion_motion_refine`), upstream and here | 100 (truncated) |

So for a particle extracted here with a fraction of 0.5 or more, Polish worked on a box one pixel away
from the one its offsets belong to.

In practice Polish hid this: it fits each particle's track against the reference, so a whole-pixel box error
became a constant one-pixel shift in the track and was removed. On EMPIAR-10940 (69912 helical segments, all
fractional) old and new polished particles correlate 0.999, and the maps are the same (PostProcess 2.637 A for
both). The fix makes the programs agree instead of relying on the track fit, and keeps the tracks free of a
fake one-pixel jump.

## What happens now

* **Extract writes whole-pixel coordinates**: the pixel it cut the box around. Particles that have
  offsets (re-extraction) keep their place: the difference goes into the offsets. Plain picks just get
  the rounded coordinate; no offset columns are added (the fraction of a pick is below its accuracy, and
  scripts that add their own offset columns keep working). Every program then finds the same centre.
* **Older files with fractional coordinates**: their offsets are taken to refer to the rounded centre
  (files extracted here since 2026-10-05) unless told otherwise:
  `--coord_centre trunc` (Extract) or `RELION_COORD_CENTRE=trunc` (all programs) for files extracted by
  upstream RELION.
  * Re-extraction moves the coordinate to the whole pixel and corrects the offsets.
  * Polish replaces each fractional coordinate by the whole pixel its offsets refer to before it cuts
    boxes, and prints how many particles that concerned. Virtual polished particles go through the same
    code, so their descriptors record whole-pixel centres.
  * `relion_reconstruct --fused_extract` from coordinates corrects the shift the same way.

Test: `tests/integration/test_coord_centre.py`.
