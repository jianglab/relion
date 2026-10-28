/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef HELICAL_CLASS_ALIGN_H_
#define HELICAL_CLASS_ALIGN_H_

/*
 * Turning and centring 2D class averages of helical segments, so that their helical axis
 * lies along X through the centre of the box (MlOptimiser::centerClasses() for helical
 * Class2D, instead of the centre of mass in 2D, which along the axis means nothing).
 *
 *  - Angle: from the segments, not the image. Each segment's psi prior comes from picking;
 *    the median over a class's segments of psi minus the prior is the angle of the class's
 *    axis (picking errors in psi are small and average out). On the class image, the strands
 *    of a twisted filament cross the axis at several degrees and can be taken for it.
 *  - Across the axis: the class's own centre of mass. Not the segments' offsets from their
 *    picked lines: those lines can be 20 A and more off the axis, the same along a whole
 *    filament, so they do not average out.
 *
 * Coordinates: x along the columns, y along the rows of the image, origin at its centre; an
 * angle turns the X axis towards the Y axis.
 */

#include "src/multidim_array.h"

#include <vector>

namespace helixalign
{

/// The angle (degrees) of a class's helical axis from its segments' psi - psi prior (any range; a
/// segment of opposite polarity, psi = prior + 180, counts the same). NaN with fewer than min_segments.
double classAngle(std::vector<double> psi_minus_prior, size_t min_segments = 20);

/// Turn the image so that an axis at angle_deg lies along X, then shift it by -offset across the axis
/// (an axis at y = offset comes to y = 0). One bilinear resampling; pixels from outside the image are 0.
void apply(MultidimArray<RFLOAT> &img, double angle_deg, double offset);

/// The centre of mass of the image across an axis at angle_deg (pixels from the centre), which apply()
/// moves to 0: the centre of mass in 2D, turned into the frame of the axis.
double offsetAcross(const MultidimArray<RFLOAT> &img, double angle_deg);

}

#endif /* HELICAL_CLASS_ALIGN_H_ */
