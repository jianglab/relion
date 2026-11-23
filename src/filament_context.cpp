/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/filament_context.h"
#include "src/metadata_table.h"
#include "src/error.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>

namespace filctx
{

std::string modeName(Mode m)
{
	switch (m)
	{
	case CHAIN: return "chain";
	case FILAMENT: return "filament";
	default: return "off";
	}
}

Mode modeFromString(const std::string &s, const std::string &what)
{
	if (s == "off" || s == "0") return OFF;
	if (s == "chain" || s == "hmm") return CHAIN;
	if (s == "filament") return FILAMENT;
	REPORT_ERROR("ERROR: " + what + " should be off, chain or filament, not '" + s + "'.");
	return OFF;
}

static bool envDouble(const char *name, double &v)
{
	const char *s = getenv(name);
	if (s == NULL || *s == '\0') return false;
	char *end;
	double x = strtod(s, &end);
	if (*end != '\0')
		REPORT_ERROR(std::string("ERROR: ") + name + " should be a number, not '" + s + "'.");
	v = x;
	return true;
}

Options optionsFromEnvironment()
{
	Options o;
	const char *m = getenv("RELION_FILAMENT_CONTEXT");
	if (m != NULL && *m != '\0')
		o.mode = modeFromString(m, "RELION_FILAMENT_CONTEXT");
	double v;
	if (envDouble("RELION_FILAMENT_TEMPER", v)) o.temper = v;
	if (envDouble("RELION_FILAMENT_GAP", v))
	{
		if (v < 1.) REPORT_ERROR("ERROR: RELION_FILAMENT_GAP should be at least 1.");
		o.gap = (int)v;
	}
	if (envDouble("RELION_FILAMENT_FLOOR", v))
	{
		if (v < 0. || v > 1.) REPORT_ERROR("ERROR: RELION_FILAMENT_FLOOR should be between 0 and 1.");
		o.floor = v;
	}
	if (envDouble("RELION_FILAMENT_ALPHA", v)) o.alpha = v;
	if (envDouble("RELION_FILAMENT_CLIP", v)) o.clip = v;
	if (envDouble("RELION_FILAMENT_EXCLUDE", v)) o.exclude_angst = v;
	if (envDouble("RELION_FILAMENT_START", v))
	{
		if (v < 1.) REPORT_ERROR("ERROR: RELION_FILAMENT_START should be at least 1.");
		o.start_iter = (int)v;
	}
	if (envDouble("RELION_FILAMENT_PHASE_WEIGHT", v)) o.phase_weight = v;
	return o;
}

std::vector<std::vector<long int> > groupFilaments(const MetaDataTable &MD, double &step)
{
	std::vector<std::vector<double> > tracks;
	return groupFilamentsWithTracks(MD, step, tracks);
}

std::vector<std::vector<long int> > groupFilamentsWithTracks(const MetaDataTable &MD, double &step,
                                                             std::vector<std::vector<double> > &tracks)
{
	std::vector<std::vector<long int> > out;
	tracks.clear();
	step = -1.;
	if (!MD.containsLabel(EMDL_PARTICLE_HELICAL_TUBE_ID))
		return out;
	const bool has_mic = MD.containsLabel(EMDL_MICROGRAPH_NAME);
	const bool has_tla = MD.containsLabel(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM);
	const bool has_tl = MD.containsLabel(EMDL_PARTICLE_HELICAL_TRACK_LENGTH);

	std::map<std::pair<std::string, int>, std::vector<std::pair<double, long int> > > groups;
	const long int n = MD.numberOfObjects();
	for (long int i = 0; i < n; i++)
	{
		int tube = -1;
		MD.getValue(EMDL_PARTICLE_HELICAL_TUBE_ID, tube, i);
		if (tube < 1) continue;
		std::string mic;
		if (has_mic) MD.getValue(EMDL_MICROGRAPH_NAME, mic, i);
		double t = (double)i;
		if (has_tla) MD.getValue(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM, t, i);
		else if (has_tl) MD.getValue(EMDL_PARTICLE_HELICAL_TRACK_LENGTH, t, i);
		groups[std::make_pair(mic, tube)].push_back(std::make_pair(t, i));
	}

	std::vector<double> steps;
	for (auto &g : groups)
	{
		std::sort(g.second.begin(), g.second.end());
		std::vector<long int> f;
		std::vector<double> tr;
		for (size_t k = 0; k < g.second.size(); k++)
		{
			f.push_back(g.second[k].second);
			if (has_tla || has_tl)
			{
				tr.push_back(g.second[k].first);
				if (k > 0)
				{
					const double d = g.second[k].first - g.second[k - 1].first;
					if (d > 0.) steps.push_back(d);
				}
			}
		}
		out.push_back(f);
		if (has_tla || has_tl)
			tracks.push_back(tr);
	}
	if (!steps.empty())
	{
		std::nth_element(steps.begin(), steps.begin() + steps.size() / 2, steps.end());
		step = steps[steps.size() / 2];
	}
	return out;
}

namespace
{

// y = x T (row vector times matrix)
void leftMul(const double *x, const std::vector<double> &T, int K, double *y)
{
	for (int k = 0; k < K; k++) y[k] = 0.;
	for (int j = 0; j < K; j++)
	{
		const double xj = x[j];
		if (xj == 0.) continue;
		const double *row = &T[(size_t)j * K];
		for (int k = 0; k < K; k++) y[k] += xj * row[k];
	}
}

// y = T x (matrix times column vector)
void rightMul(const std::vector<double> &T, const double *x, int K, double *y)
{
	for (int j = 0; j < K; j++)
	{
		const double *row = &T[(size_t)j * K];
		double s = 0.;
		for (int k = 0; k < K; k++) s += row[k] * x[k];
		y[j] = s;
	}
}

double normalise(double *x, int K)
{
	double s = 0.;
	for (int k = 0; k < K; k++) s += x[k];
	if (s > 0.)
		for (int k = 0; k < K; k++) x[k] /= s;
	return s;
}

std::vector<double> matMul(const std::vector<double> &A, const std::vector<double> &B, int K)
{
	std::vector<double> C((size_t)K * K, 0.);
	for (int i = 0; i < K; i++)
		for (int j = 0; j < K; j++)
		{
			const double a = A[(size_t)i * K + j];
			if (a == 0.) continue;
			for (int k = 0; k < K; k++) C[(size_t)i * K + k] += a * B[(size_t)j * K + k];
		}
	return C;
}

std::vector<double> matPow(const std::vector<double> &T, int K, int p)
{
	std::vector<double> R((size_t)K * K, 0.), B = T;
	for (int k = 0; k < K; k++) R[(size_t)k * K + k] = 1.;
	while (p > 0)
	{
		if (p & 1) R = matMul(R, B, K);
		p >>= 1;
		if (p > 0) B = matMul(B, B, K);
	}
	return R;
}

// Tempered emissions of one segment, scaled to a maximum of 1 (all 1 when unknown),
// with optional Huber clipping on the per-segment contrast (in nats before temper).
void emissions(const float *ll, int K, double temper, double clip, const std::vector<double> &pi, double *e)
{
	bool known = false;
	double mx = -1e300;
	for (int k = 0; k < K; k++)
	{
		if (ll[k] != 0.f) known = true;
		if (pi[k] > 0. && ll[k] > mx) mx = ll[k];
	}
	for (int k = 0; k < K; k++)
	{
		if (!known) { e[k] = 1.; continue; }
		if (pi[k] <= 0.) { e[k] = 0.; continue; }
		double d = (double)ll[k] - mx;
		if (clip > 0. && d < -clip) d = -clip;
		e[k] = std::exp(temper * d);
	}
}

void finishPrior(double *p, const std::vector<double> &pi, int K, double floor,
                 const float *extra, float *out)
{
	if (normalise(p, K) <= 0.)
		for (int k = 0; k < K; k++) p[k] = pi[k];
	if (extra != NULL)
	{
		double mx = -1e300;
		for (int k = 0; k < K; k++)
			if (p[k] > 0.)
				mx = std::max(mx, std::max(-20., std::min(20., (double)extra[k])));
		for (int k = 0; k < K; k++)
		{
			if (p[k] <= 0.) continue;
			const double b = std::max(-20., std::min(20., (double)extra[k])) - mx;
			p[k] *= std::exp(b);
		}
		if (normalise(p, K) <= 0.)
			for (int k = 0; k < K; k++) p[k] = pi[k];
	}
	for (int k = 0; k < K; k++)
	{
		const double v = (1. - floor) * p[k] + floor * pi[k];
		out[k] = (v > 0.) ? (float)std::log(v) : -1e30f;
	}
}

void leaveOneOutWindow(const std::vector<double> *track, int n, int t, int gap, double exclude_angst,
                       int &lo, int &hi)
{
	lo = std::max(0, t - gap + 1);
	hi = std::min(n, t + gap);
	if (track != NULL && track->size() == (size_t)n && exclude_angst > 0.)
	{
		const double st = (*track)[t];
		while (lo > 0 && std::fabs(st - (*track)[lo - 1]) < exclude_angst) lo--;
		while (hi < n && std::fabs((*track)[hi] - st) < exclude_angst) hi++;
	}
}

int trackMultiplicity(const std::vector<double> *track, double step, int i0, int i1, int max_mult, int fallback)
{
	if (track == NULL || step <= 0. || i0 < 0 || i1 >= (int)track->size())
		return fallback;
	const long int m = std::lround(std::fabs((*track)[i1] - (*track)[i0]) / step);
	return (int)std::max(1L, std::min((long int)max_mult, m));
}

// Forward-backward along one filament with distance-aware step multiplicities:
// scaled messages a (forward, includes the emission) and b (backward, excludes it);
// optionally accumulates the expected transition counts per effective step.
void forwardBackward(const std::vector<double> &e, int n, int K, const std::vector<double> &pi,
                     const std::vector<std::vector<double> > &Tpow, const std::vector<int> *edge_mult,
                     std::vector<double> &a, std::vector<double> &b, std::vector<double> *counts)
{
	a.assign((size_t)n * K, 0.);
	b.assign((size_t)n * K, 1.);
	std::vector<double> tmp(K);
	for (int k = 0; k < K; k++) a[k] = pi[k] * e[k];
	normalise(&a[0], K);
	for (int t = 1; t < n; t++)
	{
		const int m = (edge_mult != NULL) ? (*edge_mult)[t - 1] : 1;
		const std::vector<double> &Tm = Tpow[m];
		leftMul(&a[(size_t)(t - 1) * K], Tm, K, &a[(size_t)t * K]);
		for (int k = 0; k < K; k++) a[(size_t)t * K + k] *= e[(size_t)t * K + k];
		if (normalise(&a[(size_t)t * K], K) <= 0.)
			leftMul(&a[(size_t)(t - 1) * K], Tm, K, &a[(size_t)t * K]), normalise(&a[(size_t)t * K], K);
	}
	for (int t = n - 2; t >= 0; t--)
	{
		const int m = (edge_mult != NULL) ? (*edge_mult)[t] : 1;
		const std::vector<double> &Tm = Tpow[m];
		for (int k = 0; k < K; k++) tmp[k] = e[(size_t)(t + 1) * K + k] * b[(size_t)(t + 1) * K + k];
		rightMul(Tm, &tmp[0], K, &b[(size_t)t * K]);
		if (normalise(&b[(size_t)t * K], K) <= 0.)
			for (int k = 0; k < K; k++) b[(size_t)t * K + k] = 1.;
	}
	if (counts == NULL) return;
	for (int t = 0; t + 1 < n; t++)
	{
		// Down-weight multi-step gapped jumps in transition learning (m == 1 gives direct neighbour counts)
		const int m = (edge_mult != NULL) ? (*edge_mult)[t] : 1;
		if (m > 2) continue;
		const std::vector<double> &Tm = Tpow[m];
		const double *at = &a[(size_t)t * K];
		for (int k = 0; k < K; k++) tmp[k] = e[(size_t)(t + 1) * K + k] * b[(size_t)(t + 1) * K + k];
		double z = 0.;
		for (int j = 0; j < K; j++)
		{
			if (at[j] == 0.) continue;
			const double *row = &Tm[(size_t)j * K];
			for (int k = 0; k < K; k++) z += at[j] * row[k] * tmp[k];
		}
		if (z <= 0.) continue;
		const double w = 1. / m;
		for (int j = 0; j < K; j++)
		{
			if (at[j] == 0.) continue;
			const double *row = &Tm[(size_t)j * K];
			double *c = &(*counts)[(size_t)j * K];
			const double f = w * at[j] / z;
			for (int k = 0; k < K; k++) c[k] += f * row[k] * tmp[k];
		}
	}
}

} // namespace

void computeHelicalPhaseBonus(const std::vector<std::vector<long int> > &filaments,
                              const std::vector<std::vector<double> > &tracks,
                              long int nr_parts, int K,
                              const std::vector<double> &rot_deg,
                              const std::vector<double> &psi_deg,
                              const std::vector<double> &dx_angst,
                              const std::vector<double> &dy_angst,
                              const std::vector<char> &valid_pose,
                              const std::vector<double> &psi_prior_deg,
                              const std::vector<ClassHelixParams> &helix_params,
                              int gap, double exclude_angst, double phase_weight,
                              std::vector<float> &phase_bonus)
{
	phase_bonus.assign((size_t)nr_parts * K, 0.f);
	if (phase_weight <= 0. || K <= 0 || filaments.empty() || tracks.size() != filaments.size())
		return;
	if ((long int)helix_params.size() != K)
		return;

	const double pi_rad = 3.14159265358979323846;
	const int eff_gap = std::max(1, gap);

	for (int k = 0; k < K; k++)
	{
		if (helix_params[k].rise_angst <= 1e-3 || std::fabs(helix_params[k].twist_deg) <= 1e-4)
			continue;
		const int sym = std::max(1, helix_params[k].sym_order);
		const double period_deg = 360. / sym;
		const double omega_deg_per_A = helix_params[k].twist_deg / helix_params[k].rise_angst;
		// High-rate screw helices (~180 deg / 4.7 A) are sensitive to 1 A coordinate jitter unless step is small;
		// damp weights gently above 25 deg/A so only reliable low/moderate twist progressions dominate
		const double rate = std::fabs(omega_deg_per_A) * sym;
		const double rate_scale = (rate <= 25.) ? 1. : (25. / rate);
		const double scale = (2. * pi_rad / period_deg);

		#pragma omp parallel for schedule(dynamic, 16)
		for (long int f = 0; f < (long int)filaments.size(); f++)
		{
			const std::vector<long int> &fil = filaments[f];
			const int n = (int)fil.size();
			if (n < 3 || tracks[f].size() != (size_t)n)
				continue;

			// 4 accumulators: sgn_idx (0: -1, 1: +1) x pol (0, 1)
			std::vector<double> pref_c(4 * (n + 1), 0.), pref_s(4 * (n + 1), 0.);
			std::vector<int> pref_cnt(4 * (n + 1), 0);

			for (int t = 0; t < n; t++)
			{
				for (int m = 0; m < 4; m++)
				{
					pref_c[m * (n + 1) + t + 1] = pref_c[m * (n + 1) + t];
					pref_s[m * (n + 1) + t + 1] = pref_s[m * (n + 1) + t];
					pref_cnt[m * (n + 1) + t + 1] = pref_cnt[m * (n + 1) + t];
				}
				const long int p = fil[t];
				if (p < 0 || p >= nr_parts) continue;
				const size_t pk = (size_t)p * K + k;
				if (pk >= valid_pose.size() || !valid_pose[pk]) continue;

				const double psi = psi_deg[pk];
				const double psi_rad = psi * (pi_rad / 180.);
				const double x_hel = dx_angst[pk] * std::cos(psi_rad) - dy_angst[pk] * std::sin(psi_rad);
				const double u = tracks[f][t] - x_hel;

				int pol = 0;
				if ((size_t)p < psi_prior_deg.size() && std::fabs(psi_prior_deg[p]) <= 361.)
				{
					const double d = std::fabs(std::remainder(psi - psi_prior_deg[p], 360.));
					if (d > 90.) pol = 1;
				}

				for (int si = 0; si < 2; si++)
				{
					const double sgn = (si == 0) ? -1. : 1.;
					const double theta = scale * (rot_deg[pk] + sgn * omega_deg_per_A * u);
					const int m = si * 2 + pol;
					pref_c[m * (n + 1) + t + 1] += std::cos(theta);
					pref_s[m * (n + 1) + t + 1] += std::sin(theta);
					pref_cnt[m * (n + 1) + t + 1] += 1;
				}
			}

			for (int t = 0; t < n; t++)
			{
				const long int p = fil[t];
				if (p < 0 || p >= nr_parts) continue;
				int lo = 0, hi = 0;
				leaveOneOutWindow(&tracks[f], n, t, eff_gap, exclude_angst, lo, hi);
				double best_r2 = 0.;
				for (int m = 0; m < 4; m++)
				{
					const int off = m * (n + 1);
					const int cnt = pref_cnt[off + n] - (pref_cnt[off + hi] - pref_cnt[off + lo]);
					if (cnt < 2) continue;
					const double c = pref_c[off + n] - (pref_c[off + hi] - pref_c[off + lo]);
					const double s = pref_s[off + n] - (pref_s[off + hi] - pref_s[off + lo]);
					const double r2_norm = (c * c + s * s) / ((double)cnt * (double)cnt);
					best_r2 = std::max(best_r2, r2_norm - 1. / (double)cnt);
				}
				phase_bonus[(size_t)p * K + k] = (float)(phase_weight * rate_scale * best_r2);
			}
		}
	}
}

void computePriors(const std::vector<std::vector<long int> > &filaments,
                   const std::vector<float> &loglik, int K,
                   const std::vector<double> &pi, const Options &opt, double temper,
                   std::vector<double> &T, std::vector<float> &logprior)
{
	std::vector<std::vector<double> > no_tracks;
	computePriors(filaments, no_tracks, -1., loglik, K, pi, opt, temper, T, logprior, NULL);
}

void computePriors(const std::vector<std::vector<long int> > &filaments,
                   const std::vector<std::vector<double> > &tracks, double step,
                   const std::vector<float> &loglik, int K,
                   const std::vector<double> &pi, const Options &opt, double temper,
                   std::vector<double> &T, std::vector<float> &logprior,
                   const std::vector<float> *extra_cavity_logscore)
{
	const long int nr_parts = (long int)(loglik.size() / K);
	logprior.resize(loglik.size());
	{
		std::vector<double> p(pi);
		std::vector<float> row(K);
		finishPrior(&p[0], pi, K, 0., NULL, &row[0]);
		for (long int i = 0; i < nr_parts; i++)
			std::copy(row.begin(), row.end(), logprior.begin() + (size_t)i * K);
	}
	if (opt.mode == OFF) return;
	const int gap = std::max(1, opt.gap);
	const bool have_tracks = (!tracks.empty() && tracks.size() == filaments.size());
	const bool have_step = (have_tracks && step > 0.);
	const bool have_extra = (extra_cavity_logscore != NULL && extra_cavity_logscore->size() == loglik.size());

	if (opt.mode == FILAMENT)
	{
		#pragma omp parallel for schedule(dynamic, 16)
		for (long int f = 0; f < (long int)filaments.size(); f++)
		{
			const std::vector<long int> &fil = filaments[f];
			const std::vector<double> *tr = have_tracks ? &tracks[f] : NULL;
			const int n = (int)fil.size();
			// tempered posteriors of each segment, and their running sum along the filament
			std::vector<double> csum((size_t)(n + 1) * K, 0.), e(K), p(K);
			for (int t = 0; t < n; t++)
			{
				emissions(&loglik[(size_t)fil[t] * K], K, 1., opt.clip, pi, &e[0]);
				for (int k = 0; k < K; k++) e[k] *= pi[k];
				normalise(&e[0], K);
				for (int k = 0; k < K; k++)
					csum[(size_t)(t + 1) * K + k] = csum[(size_t)t * K + k] + temper * e[k];
			}
			for (int t = 0; t < n; t++)
			{
				int lo = 0, hi = 0;
				leaveOneOutWindow(tr, n, t, gap, opt.exclude_angst, lo, hi);
				for (int k = 0; k < K; k++)
					p[k] = opt.alpha * pi[k] + csum[(size_t)n * K + k] - (csum[(size_t)hi * K + k] - csum[(size_t)lo * K + k]);
				const float *extra = have_extra ? &(*extra_cavity_logscore)[(size_t)fil[t] * K] : NULL;
				finishPrior(&p[0], pi, K, opt.floor, extra, &logprior[(size_t)fil[t] * K]);
			}
		}
		return;
	}

	// chain mode
	if (T.size() != (size_t)K * K)
	{
		T.assign((size_t)K * K, 0.);
		for (int j = 0; j < K; j++)
			for (int k = 0; k < K; k++)
				T[(size_t)j * K + k] = (1. - opt.stay) * pi[k] + ((j == k) ? opt.stay : 0.);
	}

	std::vector<std::vector<double> > E(filaments.size());
	std::vector<std::vector<int> > edge_mult(filaments.size());
	int max_edge_mult = 1;
	#pragma omp parallel for schedule(dynamic, 16) reduction(max : max_edge_mult)
	for (long int f = 0; f < (long int)filaments.size(); f++)
	{
		const std::vector<long int> &fil = filaments[f];
		const std::vector<double> *tr = have_tracks ? &tracks[f] : NULL;
		E[f].resize(fil.size() * K);
		for (size_t t = 0; t < fil.size(); t++)
			emissions(&loglik[(size_t)fil[t] * K], K, temper, opt.clip, pi, &E[f][t * K]);
		if (fil.size() > 1)
		{
			edge_mult[f].assign(fil.size() - 1, 1);
			if (have_step && tr != NULL && tr->size() == fil.size())
			{
				for (size_t t = 0; t + 1 < fil.size(); t++)
				{
					const int m = trackMultiplicity(tr, step, (int)t, (int)(t + 1), 16, 1);
					edge_mult[f][t] = m;
					if (m > max_edge_mult) max_edge_mult = m;
				}
			}
		}
	}

	const int max_pow = std::max(max_edge_mult, std::max(gap, have_step ? 32 : gap));
	auto buildTpow = [&](const std::vector<double> &Tbase)
	{
		std::vector<std::vector<double> > Tp(max_pow + 1, std::vector<double>((size_t)K * K, 0.));
		for (int k = 0; k < K; k++) Tp[0][(size_t)k * K + k] = 1.;
		if (max_pow >= 1) Tp[1] = Tbase;
		for (int m = 2; m <= max_pow; m++) Tp[m] = matMul(Tp[m - 1], Tbase, K);
		return Tp;
	};

	// Baum-Welch updates of the transitions; counts are symmetrised because a
	// filament may have been traced in either direction
	for (int it = 0; it < opt.bw_iters; it++)
	{
		const std::vector<std::vector<double> > Tpow = buildTpow(T);
		std::vector<double> counts((size_t)K * K, 0.);
		#pragma omp parallel
		{
			std::vector<double> my((size_t)K * K, 0.), a, b;
			#pragma omp for schedule(dynamic, 16) nowait
			for (long int f = 0; f < (long int)filaments.size(); f++)
				if (filaments[f].size() > 1)
					forwardBackward(E[f], (int)filaments[f].size(), K, pi, Tpow, &edge_mult[f], a, b, &my);
			#pragma omp critical(filctx_counts)
			for (size_t i = 0; i < counts.size(); i++) counts[i] += my[i];
		}
		for (int j = 0; j < K; j++)
		{
			double s = 0.;
			for (int k = 0; k < K; k++)
			{
				const double c = 0.5 * (counts[(size_t)j * K + k] + counts[(size_t)k * K + j]) + pi[k];
				T[(size_t)j * K + k] = c;
				s += c;
			}
			for (int k = 0; k < K; k++) T[(size_t)j * K + k] = (s > 0.) ? T[(size_t)j * K + k] / s : pi[k];
		}
	}

	const std::vector<std::vector<double> > Tpow = buildTpow(T);
	// pi T^m for segments near the start, where the left context is empty
	std::vector<std::vector<double> > piT(max_pow + 1, std::vector<double>(K));
	piT[0] = pi;
	for (int j = 1; j <= max_pow; j++) leftMul(&piT[j - 1][0], T, K, &piT[j][0]);

	#pragma omp parallel for schedule(dynamic, 16)
	for (long int f = 0; f < (long int)filaments.size(); f++)
	{
		const std::vector<long int> &fil = filaments[f];
		const std::vector<double> *tr = have_tracks ? &tracks[f] : NULL;
		const int n = (int)fil.size();
		std::vector<double> a, b, L(K), R(K), tmp(K);
		forwardBackward(E[f], n, K, pi, Tpow, (n > 1) ? &edge_mult[f] : NULL, a, b, NULL);
		for (int t = 0; t < n; t++)
		{
			int lo = 0, hi = 0;
			leaveOneOutWindow(tr, n, t, gap, opt.exclude_angst, lo, hi);
			if (lo - 1 >= 0)
			{
				const int ml = have_step ? trackMultiplicity(tr, step, lo - 1, t, max_pow, t - (lo - 1)) : (t - (lo - 1));
				leftMul(&a[(size_t)(lo - 1) * K], Tpow[std::min(ml, max_pow)], K, &L[0]);
			}
			else
			{
				const int m0 = have_step ? std::min(max_pow, (int)std::lround(std::fabs((*tr)[t] - (*tr)[0]) / step)) : std::min(max_pow, t);
				L = piT[std::max(0, m0)];
			}
			if (hi < n)
			{
				const int mr = have_step ? trackMultiplicity(tr, step, t, hi, max_pow, hi - t) : (hi - t);
				for (int k = 0; k < K; k++) tmp[k] = E[f][(size_t)hi * K + k] * b[(size_t)hi * K + k];
				rightMul(Tpow[std::min(mr, max_pow)], &tmp[0], K, &R[0]);
			}
			else
				std::fill(R.begin(), R.end(), 1.);
			for (int k = 0; k < K; k++) L[k] *= R[k];
			const float *extra = have_extra ? &(*extra_cavity_logscore)[(size_t)fil[t] * K] : NULL;
			finishPrior(&L[0], pi, K, opt.floor, extra, &logprior[(size_t)fil[t] * K]);
		}
	}
}

}
