#include "src/helix_report.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <sstream>

namespace helixreport
{

namespace
{

typedef std::complex<double> cx_t;

double wrapRad(double a)
{
	a = std::fmod(a + PI, 2. * PI);
	if (a < 0.)
		a += 2. * PI;
	return a - PI;
}

double quantile(std::vector<double> v, double q)
{
	if (v.empty())
		return 0.;
	std::sort(v.begin(), v.end());
	return v[std::min(v.size() - 1, (size_t) (q * v.size()))];
}

// Position along a line fitted through the refined centres of one filament (Cauchy-weighted total least squares,
// scale 10 A), oriented to grow with the track length
std::vector<double> axialPositions(const std::vector<const Segment *> &s)
{
	const size_t n = s.size();
	std::vector<double> w(n, 1.), x(n);
	double vx = 1., vy = 0., cxm = 0., cym = 0.;
	for (int it = 0; it < 8; it++)
	{
		double sw = 0., sx = 0., sy = 0.;
		for (size_t j = 0; j < n; j++) { sw += w[j]; sx += w[j] * s[j]->cx; sy += w[j] * s[j]->cy; }
		cxm = sx / sw; cym = sy / sw;
		double a = 0., b = 0., c = 0.;
		for (size_t j = 0; j < n; j++)
		{
			const double dx = s[j]->cx - cxm, dy = s[j]->cy - cym;
			a += w[j] * dx * dx; b += w[j] * dx * dy; c += w[j] * dy * dy;
		}
		const double th = 0.5 * std::atan2(2. * b, a - c);          // principal direction of the 2x2 scatter
		vx = std::cos(th); vy = std::sin(th);
		for (size_t j = 0; j < n; j++)
		{
			const double r = -(s[j]->cx - cxm) * vy + (s[j]->cy - cym) * vx;
			w[j] = 1. / (1. + (r / 10.) * (r / 10.));
		}
	}
	double mt = 0., mx = 0.;
	for (size_t j = 0; j < n; j++)
	{
		x[j] = (s[j]->cx - cxm) * vx + (s[j]->cy - cym) * vy;
		mt += s[j]->track; mx += x[j];
	}
	mt /= n; mx /= n;
	double cov = 0.;
	for (size_t j = 0; j < n; j++)
		cov += (x[j] - mx) * (s[j]->track - mt);
	if (cov < 0.)
		for (double &v : x)
			v = -v;
	return x;
}

// The helical axis (volume z) in the image frame, for the polarity of each segment
void axisOf(const Segment &s, double d[3])
{
	const double g = s.psi * PI / 180., b = s.tilt * PI / 180.;
	d[0] = -std::cos(g) * std::sin(b); d[1] = std::sin(g) * std::sin(b); d[2] = std::cos(b);
}

// Indices (into s) of the segments that share the polarity of the majority
std::vector<int> goodPolarity(const std::vector<const Segment *> &s)
{
	double M[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
	std::vector<std::array<double, 3> > d(s.size());
	for (size_t j = 0; j < s.size(); j++)
	{
		axisOf(*s[j], d[j].data());
		for (int a = 0; a < 3; a++)
			for (int b = 0; b < 3; b++)
				M[a][b] += d[j][a] * d[j][b];
	}
	double v[3] = {d[0][0], d[0][1], d[0][2]};
	for (int it = 0; it < 100; it++)                                // power iteration: principal axis of all segments
	{
		double u[3] = {0, 0, 0}, nn = 0.;
		for (int a = 0; a < 3; a++)
			for (int b = 0; b < 3; b++)
				u[a] += M[a][b] * v[b];
		nn = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
		if (nn <= 0.) break;
		for (int a = 0; a < 3; a++) v[a] = u[a] / nn;
	}
	double sum = 0.;
	for (size_t j = 0; j < s.size(); j++)
		sum += d[j][0] * v[0] + d[j][1] * v[1] + d[j][2] * v[2];
	if (sum < 0.)
		for (int a = 0; a < 3; a++) v[a] = -v[a];
	std::vector<int> good;
	for (size_t j = 0; j < s.size(); j++)
		if (d[j][0] * v[0] + d[j][1] * v[1] + d[j][2] * v[2] > 0.5)
			good.push_back((int) j);
	return good;
}

// |shift| of copy B against copy A along the axis, in repeats (0 .. 0.5)
double copyShift(const std::vector<double> &th, const std::vector<char> &B)
{
	cx_t a(0, 0), b(0, 0);
	for (size_t j = 0; j < th.size(); j++)
		(B[j] ? b : a) += std::polar(1., th[j]);
	return std::fabs(wrapRad(std::arg(b) - std::arg(a))) / (2. * PI);
}

std::string fmt(const char *f, double a, double b = 0., double c = 0.)
{
	char buf[512];
	snprintf(buf, sizeof(buf), f, a, b, c);
	return buf;
}

} // namespace

bool enabled()
{
	const char *e = getenv("RELION_HELIX_REPORT");
	return !(e && std::string(e) == "0");
}

Result analyse(const std::vector<Segment> &segments, double twist, double rise)
{
	Result r;
	std::ostringstream o;
	o << "Helical symmetry report (read-only; twist " << twist << " deg, rise " << rise << " A; RELION_HELIX_REPORT=0 turns it off)\n";
	if (rise <= 0.)
	{
		r.verdict = "no helical rise known: nothing to report";
		r.text = o.str() + " " + r.verdict + "\n";
		return r;
	}
	const double omega = twist / rise;                               // degrees of view angle per Angstrom
	const bool screw_imposed = std::fabs(twist) >= 90.;
	std::map<std::string, std::vector<const Segment *> > fil;
	for (const Segment &s : segments)
		fil[s.tube].push_back(&s);
	// Tilt: a healthy refinement gives a bell around 90 degrees. A pile-up at the edge of the range searched (a flat spread puts 15% there) is the sign of a poor one.
	std::vector<double> dev;
	for (const Segment &s : segments)
		dev.push_back(std::fabs(90. - s.tilt));
	std::sort(dev.begin(), dev.end());
	if (!dev.empty())
	{
		r.tilt_range = dev[(size_t) (0.999 * (dev.size() - 1))];
		if (r.tilt_range > 1.)
		{
			size_t ne = 0;
			for (double d : dev) if (d >= 0.85 * r.tilt_range) ne++;
			r.tilt_edge = (double) ne / dev.size();
			o << " Tilt: " << fmt("%.0f", 100. * r.tilt_edge) << "% of the segments lie in the outer 15% of the tilt range seen (" << fmt("%.1f", r.tilt_range)
			  << " deg from 90; a flat spread gives 15%, a bell much less)";
			if (r.tilt_edge > 0.3)
				o << ". WARNING: the tilt piles up at the edge of the search range, a sign of an unconverged or poor refinement";
			o << ".\n";
		}
	}
	double rn_sum[6] = {0, 0, 0, 0, 0, 0};
	int rn_count = 0;
	std::vector<double> shifts, ctrl_shifts, shareB;
	std::vector<double> dsig;                                          // signed phase difference (radians) and its sigma, for the filaments measured
	std::vector<double> dctrl;
	std::vector<size_t> measured;                                      // row index of each
	std::mt19937 rng(12345);
	const double s0 = 0.9;                                             // positional floor, radians of the repeat (about 0.7 A of 4.75 A)
	std::vector<double> dd, sg, dc;
	for (auto &kv : fil)
	{
		std::vector<const Segment *> s = kv.second;
		if (s.size() < 20)
			continue;
		std::sort(s.begin(), s.end(), [](const Segment *a, const Segment *b) { return a->track < b->track; });
		r.n_filaments++;
		FilamentRow row;
		row.tube = kv.first;
		row.n_segments = (int) s.size();
		const std::vector<double> xall = axialPositions(s);
		const std::vector<int> good = goodPolarity(s);
		row.n_good = (int) good.size();
		if (good.size() < 12) { row.call = "too_few_on_majority_polarity"; r.rows.push_back(row); continue; }
		std::vector<double> x(good.size()), rot(good.size()), trk(good.size());
		for (size_t j = 0; j < good.size(); j++) { x[j] = xall[good[j]]; rot[j] = s[good[j]]->rot; trk[j] = s[good[j]]->track; }
		// A fast twist (a screw written as a helix) turns a position error of 1 A into tens of degrees: the picked track positions are the coordinate then
		const std::vector<double> &xr = std::fabs(omega) > 1. ? trk : x;
		// (1) coherence modulo 360/n, with the twist rate scanned a little
		for (int n = 1; n <= 6; n++)
		{
			double best = 0.;
			for (int k = 0; k < 61; k++)
			{
				const double w = std::fabs(omega) > 1. ? omega * (0.97 + 0.06 * k / 60.) : omega - 0.03 + 0.06 * k / 60.;
				cx_t m(0, 0);
				for (size_t j = 0; j < x.size(); j++)
					m += std::polar(1., n * (rot[j] - w * xr[j]) * PI / 180.);
				best = std::max(best, std::abs(m) / x.size());
			}
			rn_sum[n - 1] += best;
		}
		rn_count++;
		// (2) the two copies
		if (screw_imposed) { row.call = "screw_imposed"; r.rows.push_back(row); continue; }
		if (good.size() < 20) { row.call = "too_few_on_majority_polarity"; r.rows.push_back(row); continue; }
		std::vector<double> ph(x.size());
		cx_t m1(0, 0), m2(0, 0);
		for (size_t j = 0; j < x.size(); j++)
		{
			ph[j] = (rot[j] - omega * x[j]) * PI / 180.;
			m1 += std::polar(1., ph[j]); m2 += std::polar(1., 2. * ph[j]);
		}
		row.r1 = std::abs(m1) / x.size(); row.r2 = std::abs(m2) / x.size();
		if (row.r2 < 0.6 || row.r1 > row.r2 - 0.15) { row.call = "no_two_clear_copies"; r.rows.push_back(row); continue; }
		const double phi0 = 0.5 * std::arg(m2);
		std::vector<char> B(x.size());
		std::vector<double> th(x.size());
		int nb = 0;
		for (size_t j = 0; j < x.size(); j++)
		{
			B[j] = std::fabs(wrapRad(ph[j] - phi0)) > PI / 2.;
			nb += B[j];
			th[j] = 2. * PI * x[j] / rise;
			if (j > 0 && B[j] != B[j - 1]) row.n_switches++;
		}
		row.copy_b_share = (double) nb / x.size();
		if (nb < 5 || (int) x.size() - nb < 5) { row.call = "one_copy_dominates"; r.rows.push_back(row); continue; }
		cx_t a(0, 0), b(0, 0);
		for (size_t j = 0; j < x.size(); j++)
			(B[j] ? b : a) += std::polar(1., th[j]);
		const double RA = std::abs(a) / (x.size() - nb), RB = std::abs(b) / nb;
		if (RA < 0.3 || RB < 0.3) { row.call = "loose_axial_phase"; r.rows.push_back(row); continue; }
		const double d = wrapRad(std::arg(b) - std::arg(a));
		row.shift = std::fabs(d) / (2. * PI);
		row.sigma = std::sqrt(1. / (RA * RA * (x.size() - nb)) + 1. / (RB * RB * nb) + s0 * s0);
		shifts.push_back(row.shift);
		shareB.push_back(row.copy_b_share);
		dd.push_back(d); sg.push_back(row.sigma);
		std::vector<char> Bp = B;
		std::shuffle(Bp.begin(), Bp.end(), rng);
		cx_t ac(0, 0), bc(0, 0);
		for (size_t j = 0; j < x.size(); j++)
			(Bp[j] ? bc : ac) += std::polar(1., th[j]);
		dc.push_back(wrapRad(std::arg(bc) - std::arg(ac)));
		ctrl_shifts.push_back(std::fabs(dc.back()) / (2. * PI));
		measured.push_back(r.rows.size());
		r.rows.push_back(row);
	}
	if (rn_count > 0)
		for (int n = 0; n < 6; n++)
			r.rn[n] = rn_sum[n] / rn_count;
	r.n_two_copy = (int) shifts.size();
	// Mixture of C2-like (shift 0), 2_1-like (shift half a repeat) and other (anywhere) filaments, by EM on the signed phase difference
	auto lik = [&](double d, double sg, int c) {
		if (c == 2) return 1. / (2. * PI);
		const double e = wrapRad(c == 0 ? d : d - PI);
		return std::exp(-0.5 * e * e / (sg * sg)) / (std::sqrt(2. * PI) * sg);
	};
	auto em = [&](const std::vector<double> &d, const std::vector<double> &sg, double *w) {
		w[0] = w[1] = 0.4; w[2] = 0.2;
		for (int it = 0; it < 200; it++)
		{
			double acc[3] = {0, 0, 0};
			for (size_t i = 0; i < d.size(); i++)
			{
				double p[3], z = 0.;
				for (int c = 0; c < 3; c++) { p[c] = w[c] * lik(d[i], sg[i], c); z += p[c]; }
				for (int c = 0; c < 3; c++) acc[c] += p[c] / std::max(z, 1e-300);
			}
			for (int c = 0; c < 3; c++) w[c] = acc[c] / std::max<size_t>(d.size(), 1);
		}
	};
	if (!dd.empty())
	{
		em(dd, sg, r.w);
		em(dc, sg, r.ctrl_w);
		std::vector<std::vector<double> > bs(3);
		std::mt19937 rb(777);
		for (int rep = 0; rep < 200; rep++)
		{
			std::vector<double> d2, s2;
			for (size_t i = 0; i < dd.size(); i++) { size_t k = rb() % dd.size(); d2.push_back(dd[k]); s2.push_back(sg[k]); }
			double w[3];
			em(d2, s2, w);
			for (int c = 0; c < 3; c++) bs[c].push_back(w[c]);
		}
		for (int c = 0; c < 3; c++) { r.w_lo[c] = quantile(bs[c], 0.05); r.w_hi[c] = quantile(bs[c], 0.95); }
		for (size_t i = 0; i < dd.size(); i++)
		{
			FilamentRow &row = r.rows[measured[i]];
			double p[3], z = 0.;
			for (int c = 0; c < 3; c++) { p[c] = r.w[c] * lik(dd[i], sg[i], c); z += p[c]; }
			row.p_c2 = p[0] / z; row.p_21 = p[1] / z; row.p_other = p[2] / z;
			row.call = row.p_21 > 0.8 ? "2_1" : row.p_c2 > 0.8 ? "c2" : row.p_other > 0.8 ? "other" : "unclear";
		}
		for (double v : shifts) { r.near0 += v < 0.125; r.near_half += v > 0.375; }
		for (double v : ctrl_shifts) { r.ctrl_near0 += v < 0.125; r.ctrl_near_half += v > 0.375; }
		r.near0 /= shifts.size(); r.near_half /= shifts.size();
		r.ctrl_near0 /= ctrl_shifts.size(); r.ctrl_near_half /= ctrl_shifts.size();
		r.median = quantile(shifts, 0.5); r.ctrl_median = quantile(ctrl_shifts, 0.5);
		double sb = 0.;
		for (double v : shareB) sb += v;
		r.copy_b_share = sb / shareB.size();
	}
	o << " Filaments with 20+ segments: " << r.n_filaments << "; segments on the majority polarity are used.\n";
	if (rn_count > 0)
	{
		o << " Coherence of the view angles modulo 360/n, n = 1..6:";
		for (int n = 0; n < 6; n++)
			o << " " << fmt("%.2f", r.rn[n]);
		o << "\n";
		int best = 0;
		for (int n = 1; n < 6; n++)
			if (r.rn[n] > r.rn[best])
				best = n;
		if (best > 0 && r.rn[best] > r.rn[0] + 0.2)
			o << " -> the segments sit on " << (best + 1) << " copies related by " << fmt("%.0f", 360. / (best + 1))
			  << " degrees: check that the symmetry or the helical parameters account for it.\n";
	}
	const std::string nfil = std::to_string(r.n_two_copy);
	if (screw_imposed)
		r.verdict = fmt("twist %.1f deg: a screw is already part of the imposed helical symmetry, so C2 against 2_1 does not apply", twist);
	else if (shifts.empty())
		r.verdict = "no filament has two clear 180-degree copies, so C2 cannot be told from a 2_1 screw (the 2-fold may already be imposed, or all segments sit on one copy)";
	else if (r.n_two_copy < 20)
		r.verdict = "only " + nfil + " filaments show two clear 180-degree copies (20 are needed): no verdict";
	else
	{
		const char *names[3] = {"C2-like", "2_1-like", "other"};
		int top = r.w[1] > r.w[0] ? 1 : 0, sec = 1 - top;
		std::string mix = std::string(names[top]) + " " + fmt("%.2f", r.w[top]) + " [" + fmt("%.2f", r.w_lo[top]) + "-" + fmt("%.2f", r.w_hi[top]) + "], "
				+ names[sec] + " " + fmt("%.2f", r.w[sec]) + " [" + fmt("%.2f", r.w_lo[sec]) + "-" + fmt("%.2f", r.w_hi[sec]) + "], other "
				+ fmt("%.2f", r.w[2]) + " [" + fmt("%.2f", r.w_lo[2]) + "-" + fmt("%.2f", r.w_hi[2]) + "]";
		const double dom = std::max(r.w[0], r.w[1]), oth = std::min(r.w[0], r.w[1]);
		const std::string lead = (r.w[1] > r.w[0]) ? "2_1 screw" : "C2 axis";
		if (dom + oth < 0.5)
			r.verdict = "most filaments fit neither a C2 axis nor a 2_1 screw (" + mix + "; " + nfil + " filaments)";
		else if (oth > 0.35 * (dom + oth))
			r.verdict = "mixture, " + std::string(r.w[1] > r.w[0] ? "2_1" : "C2") + " somewhat more common: " + mix + " (" + nfil + " filaments)";
		else
			r.verdict = "data suggest mostly a " + lead + ": " + mix + " (" + nfil + " filaments)";
		o << " Two clear copies in " << r.n_two_copy << " filaments (copy B share " << fmt("%.0f", 100. * r.copy_b_share) << "%).\n"
		  << " Axial shift of copy B against A, in repeats: median " << fmt("%.2f", r.median) << " (C2 expects 0, 2_1 expects 0.5), share below 1/8 "
		  << fmt("%.0f", 100. * r.near0) << "%, share above 3/8 " << fmt("%.0f", 100. * r.near_half) << "%\n"
		  << " Shuffled copy labels (control): median " << fmt("%.2f", r.ctrl_median) << ", mixture C2-like " << fmt("%.2f", r.ctrl_w[0])
		  << ", 2_1-like " << fmt("%.2f", r.ctrl_w[1]) << ", other " << fmt("%.2f", r.ctrl_w[2]) << "\n"
		  << " Per-filament table with the posterior of each class: see the _helical_symmetry_filaments.csv file next to the outputs.\n";
	}
	o << " " << r.verdict << ".\n A hint, not proof: the map and the resolution under each choice decide.\n";
	r.text = o.str();
	return r;
}

std::string csv(const Result &r)
{
	std::ostringstream o;
	o << "filament,segments,on_majority_polarity,coherence_360,coherence_180,copy_B_share,copy_switches,tilt_sd_deg,shift_repeats,sigma_rad,p_C2,p_2_1,p_other,call\n";
	for (const FilamentRow &w : r.rows)
		o << w.tube << "," << w.n_segments << "," << w.n_good << "," << fmt("%.3f", w.r1) << "," << fmt("%.3f", w.r2) << "," << fmt("%.3f", w.copy_b_share) << ","
		  << w.n_switches << "," << fmt("%.2f", w.tilt_sd) << "," << (w.shift < 0 ? std::string("") : fmt("%.3f", w.shift)) << "," << fmt("%.3f", w.sigma) << "," << fmt("%.3f", w.p_c2) << ","
		  << fmt("%.3f", w.p_21) << "," << fmt("%.3f", w.p_other) << "," << w.call << "\n";
	return o.str();
}

Result analyse(MetaDataTable &MD, const std::function<double(int)> &pixel_size, double twist, double rise)
{
	Result empty;
	const EMDLabel need[] = {EMDL_PARTICLE_HELICAL_TUBE_ID, EMDL_MICROGRAPH_NAME, EMDL_ORIENT_ROT, EMDL_ORIENT_TILT, EMDL_ORIENT_PSI,
			EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM, EMDL_IMAGE_COORD_X, EMDL_IMAGE_COORD_Y, EMDL_ORIENT_ORIGIN_X_ANGSTROM, EMDL_ORIENT_ORIGIN_Y_ANGSTROM};
	for (EMDLabel l : need)
		if (!MD.containsLabel(l))
			return empty;
	std::map<int, long> per_class;
	if (MD.containsLabel(EMDL_PARTICLE_CLASS))
		FOR_ALL_OBJECTS_IN_METADATA_TABLE(MD)
		{
			int c = 1;
			MD.getValue(EMDL_PARTICLE_CLASS, c);
			per_class[c]++;
		}
	int use = 0;
	long most = -1;
	for (auto &kv : per_class)
		if (kv.second > most) { most = kv.second; use = kv.first; }
	std::vector<Segment> seg;
	seg.reserve(MD.numberOfObjects());
	FOR_ALL_OBJECTS_IN_METADATA_TABLE(MD)
	{
		int c = 1, tube = 0, og = 1;
		if (MD.containsLabel(EMDL_PARTICLE_CLASS))
			MD.getValue(EMDL_PARTICLE_CLASS, c);
		if (!per_class.empty() && c != use)
			continue;
		RFLOAT rot, tilt, psi, track, x, y, ox, oy;
		std::string mic;
		MD.getValue(EMDL_PARTICLE_HELICAL_TUBE_ID, tube);
		MD.getValue(EMDL_MICROGRAPH_NAME, mic);
		MD.getValue(EMDL_ORIENT_ROT, rot); MD.getValue(EMDL_ORIENT_TILT, tilt); MD.getValue(EMDL_ORIENT_PSI, psi);
		MD.getValue(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM, track);
		MD.getValue(EMDL_IMAGE_COORD_X, x); MD.getValue(EMDL_IMAGE_COORD_Y, y);
		MD.getValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, ox); MD.getValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, oy);
		if (MD.containsLabel(EMDL_IMAGE_OPTICS_GROUP))
			MD.getValue(EMDL_IMAGE_OPTICS_GROUP, og);
		const double ang = pixel_size(og);
		seg.push_back(Segment{mic + "#" + std::to_string(tube), (double) rot, (double) tilt, (double) psi, (double) track,
				x * ang - ox, y * ang - oy});
	}
	return analyse(seg, twist, rise);
}

} // namespace helixreport
