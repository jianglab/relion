/*
 * tests/unit/test_helix_report.cpp
 *
 * The read-only report on the symmetry that the poses of a helical refinement show: copies 180 degrees apart that share the
 * axial position (C2), copies that are shifted by half a repeat (2_1 screw), a mixture of both, and filaments with one copy.
 */

#include <catch2/catch.hpp>

#include <cmath>
#include <random>
#include <vector>

#include "src/helix_report.h"

namespace
{
// n_fil straight filaments along x, 14.2 A between segments, segments on copy B (rot + 180) with probability pB; copy B centres
// shifted by `shift` repeats. Polarity: psi 0, tilt 90.
std::vector<helixreport::Segment> make(int n_fil, int per, double twist, double rise, double pB, double shift_frac, unsigned seed, double noise_A = 0.5)
{
	std::mt19937 g(seed);
	std::normal_distribution<double> nrm(0., 1.);
	std::uniform_real_distribution<double> uni(0., 1.);
	std::vector<helixreport::Segment> out;
	for (int f = 0; f < n_fil; f++)
	{
		const double y0 = 100. * f, phase = 360. * uni(g);
		for (int j = 0; j < per; j++)
		{
			const double s = 14.2 * j;
			const bool B = uni(g) < pB;
			helixreport::Segment seg;
			seg.tube = "mic#" + std::to_string(f);
			seg.tilt = 90.; seg.psi = 0.;
			seg.track = s;
			// the refined axial position: the segment is centred on a repeat boundary (nearest to the picked position), copy B half a repeat away if shifted
			const double z = std::round(s / rise) * rise + (B ? shift_frac * rise : 0.) + noise_A * nrm(g);
			seg.cx = z; seg.cy = y0 + noise_A * nrm(g);          // the axial coordinate grows with the track length
			seg.rot = phase + (twist / rise) * z + (B ? 180. : 0.) + 3. * nrm(g);
			out.push_back(seg);
		}
	}
	return out;
}
}

TEST_CASE("helix report: C2 copies share the axial position", "[helix_report]")
{
	auto s = make(60, 40, -1.1, 4.75, 0.5, 0.0, 1);
	auto r = helixreport::analyse(s, -1.1, 4.75);
	CHECK(r.n_two_copy >= 40);
	CHECK(r.w[0] > 0.6);
	CHECK(r.w[1] < 0.2);
	CHECK(r.verdict.find("C2") != std::string::npos);
}

TEST_CASE("helix report: copies shifted by half a repeat are a 2_1 screw", "[helix_report]")
{
	auto s = make(60, 40, -1.1, 4.75, 0.5, 0.5, 2);
	auto r = helixreport::analyse(s, -1.1, 4.75);
	CHECK(r.n_two_copy >= 40);
	CHECK(r.w[1] > 0.6);
	CHECK(r.w[0] < 0.2);
	CHECK(r.verdict.find("2_1") != std::string::npos);
}

TEST_CASE("helix report: a mixture is reported as one", "[helix_report]")
{
	auto a = make(40, 40, -1.1, 4.75, 0.5, 0.5, 3), b = make(20, 40, -1.1, 4.75, 0.5, 0.0, 4);
	for (auto &x : b) x.tube += "_c2";
	a.insert(a.end(), b.begin(), b.end());
	auto r = helixreport::analyse(a, -1.1, 4.75);
	CHECK(r.w[1] > r.w[0]);
	CHECK(r.w[0] > 0.15);
	CHECK(r.w[1] < 0.85);
	int c2 = 0, s21 = 0;
	for (const auto &w : r.rows) { c2 += w.call == "c2"; s21 += w.call == "2_1"; }
	CHECK(s21 > c2);
	CHECK(c2 > 5);
	CHECK(helixreport::csv(r).find("filament,segments") == 0);
}

TEST_CASE("helix report: one copy only gives no verdict and a twist of 180 degrees says the screw is imposed", "[helix_report]")
{
	auto one = helixreport::analyse(make(30, 40, -1.1, 4.75, 0.0, 0.0, 5), -1.1, 4.75);
	CHECK(one.n_two_copy == 0);
	CHECK(one.verdict.find("no filament") != std::string::npos);
	auto screw = helixreport::analyse(make(30, 40, 179.4, 2.38, 0.5, 0.0, 6), 179.4, 2.38);
	CHECK(screw.verdict.find("already part of the imposed") != std::string::npos);
}

TEST_CASE("helix report: tilt piled at the edge of the range is flagged, a bell is not", "[helix_report]")
{
	std::mt19937 g(7);
	std::normal_distribution<double> nrm(0., 1.);
	auto s = make(30, 40, -1.1, 4.75, 0.0, 0.0, 3);
	for (auto &x : s) x.tilt = 90. + 3. * nrm(g);
	auto bell = helixreport::analyse(s, -1.1, 4.75);
	CHECK(bell.tilt_edge < 0.1);
	CHECK(bell.text.find("WARNING") == std::string::npos);
	for (auto &x : s) x.tilt = 90. + (nrm(g) > 0 ? 15. : -15.) + 0.5 * nrm(g);
	auto edge = helixreport::analyse(s, -1.1, 4.75);
	CHECK(edge.tilt_edge > 0.8);
	CHECK(edge.text.find("WARNING") != std::string::npos);
}
