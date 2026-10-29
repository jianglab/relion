#ifndef HELIX_REPORT_H_
#define HELIX_REPORT_H_

#include <functional>
#include <string>
#include <vector>

#include "src/metadata_table.h"

/*
 * Read-only report on the symmetry that the poses of a helical refinement show. It changes nothing in the run.
 *
 * (1) How coherent the unwound view angles of a filament's segments are modulo 360/n (n = 1..6). Segments that sit on
 *     n copies of the structure give a high value at n and low values below it.
 * (2) Two copies 180 degrees apart are either a C2 axis (no axial shift between them) or a 2_1 screw (a shift of half
 *     a repeat). The refined axial position of the two copies, modulo the repeat, tells which, if the refinement did not
 *     impose the 2-fold and segments landed on both copies. The refined centre is coordinate - origin, projected on a
 *     line fitted through the filament's refined centres. A population is a mixture, so the report gives the share of
 *     C2-like, 2_1-like and other filaments (with an interval, and the same for shuffled labels as a control) and one row
 *     per filament for focused follow-up.
 *
 * A hint, not proof: the map and the resolution under each choice decide. Off with RELION_HELIX_REPORT=0.
 */
namespace helixreport
{

struct Segment
{
	std::string tube;                 // filament key (micrograph and tube id)
	double rot, tilt, psi;            // degrees
	double track;                     // Angstrom along the picked filament
	double cx, cy;                    // refined centre in the micrograph, Angstrom (coordinate - origin)
};

struct FilamentRow
{
	std::string tube;
	int n_segments = 0, n_good = 0, n_switches = 0;   // segments, on the majority polarity, copy A <-> B changes along the filament
	double r1 = 0, r2 = 0;                            // coherence of the unwound view angles modulo 360 and 180
	double copy_b_share = 0;
	double tilt_sd = 0;                               // spread of the tilt angle (polarity folded), degrees
	double shift = -1;                                // |axial shift| of copy B against A in repeats (0..0.5); -1 when not measured
	double sigma = 0;                                 // its uncertainty, radians of the repeat
	double p_c2 = 0, p_21 = 0, p_other = 0;           // posterior of C2-like, 2_1-like, neither
	std::string call;                                 // c2, 2_1, other, unclear, or why it was not measured
};

struct Result
{
	std::vector<FilamentRow> rows;
	double w[3] = {0, 0, 0}, w_lo[3] = {0, 0, 0}, w_hi[3] = {0, 0, 0};   // mixture of C2-like, 2_1-like, other filaments, with a 90% interval
	double ctrl_w[3] = {0, 0, 0};                     // the same for shuffled copy labels (what chance gives)
	int n_filaments = 0;              // filaments with enough segments
	int n_two_copy = 0;               // of those, filaments with two clear copies and a tight axial phase
	double tilt_edge = 0, tilt_range = 0;             // share of segments in the outer 15% of the tilt range seen; that range, degrees from 90
	double rn[6] = {0, 0, 0, 0, 0, 0};
	double copy_b_share = 0;
	double median = 0, near0 = 0, near_half = 0;                 // |axial shift| in repeats: median, share < 1/8, share > 3/8
	double ctrl_median = 0, ctrl_near0 = 0, ctrl_near_half = 0;  // the same with shuffled copy labels
	std::string verdict;              // one line
	std::string text;                 // the whole report
};

Result analyse(const std::vector<Segment> &segments, double twist_deg, double rise_A);

// From a particle table (rlnHelicalTubeID, angles, track length, coordinates and origin shifts); pixel_size(optics group, 1-based)
// gives the pixel size. Only the most populated class is used. Returns an empty text when the table lacks what it needs.
Result analyse(MetaDataTable &MD, const std::function<double(int)> &pixel_size, double twist_deg, double rise_A);

std::string csv(const Result &r);   // one row per filament

bool enabled();

}

#endif
