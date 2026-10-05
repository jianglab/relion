/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef RESAMPLE_ROTATE_H
#define RESAMPLE_ROTATE_H

/* Cuts a rotated rectangle out of a 2D image (for example a long helical
 * segment, turned to lie horizontally, out of a micrograph).
 *
 * Output pixel (x, y), 0 <= x < nx, 0 <= y < ny, sits at offset
 * (dx, dy) = (x - nx/2, y - ny/2) from the output centre, and samples the
 * source at
 *
 *     centre + dx * (cos a, sin a) + dy * (-sin a, cos a)
 *
 * in source pixel coordinates (x along columns, y along rows, pixel centres at
 * integer positions). So the output's horizontal axis follows the direction at
 * angle a in the source. Samples that fall outside the source are set to 0.
 *
 * This is the ONE place that does this resampling; extraction, virtual
 * particles and the fused reconstruction must all call it so that they agree.
 *
 * Methods:
 *   LINEAR  bilinear; fastest, smooths high frequencies.
 *   CUBIC   Catmull-Rom 4x4; sharper, mild ringing.
 *   NUFFT   band-limited (exact for a band-limited image) via a type-2 NUFFT of
 *           the source's Fourier transform. Treats the source as periodic, so
 *           it rings near the source edges. Needs a build with
 *           RELION_USE_FINUFFT; otherwise it is an error, never a silent
 *           fallback to another method.
 */

#include <string>

#include "src/multidim_array.h"

enum ResampleMethod { RESAMPLE_LINEAR, RESAMPLE_CUBIC, RESAMPLE_NUFFT };

/// "linear", "cubic" or "nufft"; anything else is an error.
ResampleMethod parseResampleMethod(const std::string &name);
std::string resampleMethodName(ResampleMethod method);

void resampleRotatedRectangle(const MultidimArray<RFLOAT> &src, MultidimArray<RFLOAT> &dest,
                              int nx, int ny, double centre_x, double centre_y,
                              double angle_degrees, ResampleMethod method);

#endif
