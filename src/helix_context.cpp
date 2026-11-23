#include "src/helix_context.h"
#include "src/Eigen/Dense"
#include "src/error.h"
#include "src/macros.h"
#include "src/image.h"
#include "src/fftw.h"
#include <sstream>
#include <map>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <queue>
#include <random>
#include <complex>
#ifdef _OPENMP
#include <omp.h>
#endif

void HelicalContext::initialise(MetaDataTable &MD, const std::vector<long int> &seg_rows, int nr_classes, RFLOAT diameter_angst)
{
	K = nr_classes;
	N = seg_rows.size();
	if (!MD.containsLabel(EMDL_PARTICLE_HELICAL_TUBE_ID) || !MD.containsLabel(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM))
		REPORT_ERROR("HelicalContext: the particles need rlnHelicalTubeID and rlnHelicalTrackLengthAngst");

	std::vector<RFLOAT> track(N);
	std::vector<long int> fil(N);
	std::map<std::string, long int> fil_id;
	for (long int i = 0; i < N; i++)
	{
		FileName mic;
		int tube;
		RFLOAT t;
		MD.getValue(EMDL_MICROGRAPH_NAME, mic, seg_rows[i]);
		MD.getValue(EMDL_PARTICLE_HELICAL_TUBE_ID, tube, seg_rows[i]);
		MD.getValue(EMDL_PARTICLE_HELICAL_TRACK_LENGTH_ANGSTROM, t, seg_rows[i]);
		std::string key = mic + ":" + integerToString(tube);
		auto it = fil_id.find(key);
		if (it == fil_id.end())
		{
			it = fil_id.insert(std::make_pair(key, (long int)filament_name.size())).first;
			filament_name.push_back(key);
		}
		fil[i] = it->second;
		track[i] = t;
	}

	// segments of each filament in track order
	std::vector<std::vector<long int> > members(filament_name.size());
	for (long int i = 0; i < N; i++)
		members[fil[i]].push_back(i);
	std::vector<RFLOAT> diffs;
	for (auto &m : members)
	{
		std::sort(m.begin(), m.end(), [&](long int a, long int b) { return track[a] < track[b]; });
		for (size_t j = 1; j < m.size(); j++)
			if (track[m[j]] - track[m[j-1]] > 0.)
				diffs.push_back(track[m[j]] - track[m[j-1]]);
	}
	if (diffs.empty())
		REPORT_ERROR("HelicalContext: no filament has more than one segment");
	std::nth_element(diffs.begin(), diffs.begin() + diffs.size() / 2, diffs.end());
	step = diffs[diffs.size() / 2];

	diameter = diameter_angst;
	if (temper <= 0.)
		temper = std::max((RFLOAT)0.12, std::min((RFLOAT)1.0, step / ((RFLOAT)0.65 * std::max((RFLOAT)1., diameter_angst))));
	if (exclude_angst < 0.)
		exclude_steps = 1;
	else
		exclude_steps = std::max(1, (int)ROUND(exclude_angst / step));

	// chains on a regular grid of track positions; large gaps start a new chain
	seg_chain.assign(N, -1);
	seg_pos.assign(N, -1);
	chains.clear();
	filament_of_chain.clear();
	for (long int f = 0; f < (long int)members.size(); f++)
	{
		const std::vector<long int> &m = members[f];
		size_t j0 = 0;
		while (j0 < m.size())
		{
			size_t j1 = j0 + 1;
			while (j1 < m.size() && (track[m[j1]] - track[m[j1-1]]) / step <= max_gap_steps)
				j1++;
			long int npos = (long int)ROUND((track[m[j1-1]] - track[m[j0]]) / step) + 1;
			std::vector<long int> grid(npos, -1);
			long int c = chains.size();
			for (size_t j = j0; j < j1; j++)
			{
				long int p = (long int)ROUND((track[m[j]] - track[m[j0]]) / step);
				if (grid[p] < 0)
				{
					grid[p] = m[j];
					seg_chain[m[j]] = c;
					seg_pos[m[j]] = p;
				}
			}
			chains.push_back(grid);
			filament_of_chain.push_back(f);
			j0 = j1;
		}
	}

	// polarity: all segments of a filament in track order, positions in steps
	if (pol_temper <= 0.)
		pol_temper = step / diameter_angst;
	if (pol_exclude_angst < 0.)
		pol_exclude_angst = diameter_angst / 2.;
	pol_exclude_steps = std::max(1, (int)ROUND(pol_exclude_angst / step));
	pol_members = members;
	pol_pos.resize(members.size());
	for (size_t f = 0; f < members.size(); f++)
	{
		pol_pos[f].resize(members[f].size());
		for (size_t j = 0; j < members[f].size(); j++)
			pol_pos[f][j] = (long int)ROUND((track[members[f][j]] - track[members[f][0]]) / step);
	}
	chain_reversed.assign(chains.size(), false);
	pol_llr.assign(N, std::numeric_limits<float>::quiet_NaN());
	pol_prior.assign(N, 0.5f);
	pol_post.assign(N, 0.5f);

	loglik.assign(N * K, std::numeric_limits<float>::quiet_NaN());
	prior.assign(N * K, 1.f);
	post.assign(N * K, 0.f);
	tpost.assign(N * nr_types, 0.f);
	orig_psi_prior.clear();
	class_owner.clear();
	class_allowed.clear();
	split_lock_owner.clear();
	split_lock_until = 0;
	class_equiv.clear();
	class_phase_cos.clear();
	class_phase_sin.clear();
	class_phase_r.clear();
	fil_period.clear();
	fil_period_R.clear();
	fil_type_posterior.clear();
	fil_chain_post.clear();
	initialised_model = have_prior = false;
	if (const char *ep = getenv("RELION_HELICAL_PHASE_RING"))
		use_phase_ring = !(ep[0] == '\0' || std::string(ep) == "0" || std::string(ep) == "no" || std::string(ep) == "off" || std::string(ep) == "false");
	if (const char *et = getenv("RELION_HELICAL_TWIST_SPLIT"))
		use_twist_rate_split = !(et[0] == '\0' || std::string(et) == "0" || std::string(et) == "no" || std::string(et) == "off" || std::string(et) == "false");

	if (verb > 0)
		std::cout << " Filament context: " << N << " segments, " << filament_name.size() << " filaments, "
		          << chains.size() << " chains; step " << step << " A, " << nr_types << " types, temper " << temper
		          << ", prior from segments at least " << exclude_steps << " steps (" << exclude_steps * step << " A) away" << std::endl;
}

void HelicalContext::storeLogLik(long int i, const std::vector<RFLOAT> &ll)
{
	RFLOAT mx = -std::numeric_limits<RFLOAT>::max();
	for (int k = 0; k < K; k++)
		mx = std::max(mx, ll[k]);
	for (int k = 0; k < K; k++)
		loglik[i * K + k] = (float)std::max(ll_floor, ll[k] - mx);
}

void HelicalContext::storePosterior(long int i, const std::vector<RFLOAT> &class_post, const std::vector<RFLOAT> *class_prior_used)
{
	RFLOAT sum = 0.;
	for (int k = 0; k < K; k++)
		sum += class_post[k];
	if (!(sum > 0.))
		return;
	std::vector<RFLOAT> ll(K);
	for (int k = 0; k < K; k++)
	{
		RFLOAT p = std::max((RFLOAT)1e-30, class_post[k] / sum);
		RFLOAT pr = (class_prior_used == NULL) ? 1. : (*class_prior_used)[k];
		ll[k] = log(p) - ((pr > 0.) ? log(pr) : 0.);
	}
	storeLogLik(i, ll);
}

void HelicalContext::matPower(const std::vector<double> &M, int p, std::vector<double> &R) const
{
	std::vector<double> B(M), T(K * K);
	R.assign(K * K, 0.);
	for (int k = 0; k < K; k++)
		R[k * K + k] = 1.;
	while (p > 0)
	{
		if (p & 1)
		{
			std::fill(T.begin(), T.end(), 0.);
			for (int i = 0; i < K; i++)
				for (int l = 0; l < K; l++)
				{
					double r = R[i * K + l];
					if (r == 0.) continue;
					for (int j = 0; j < K; j++)
						T[i * K + j] += r * B[l * K + j];
				}
			R.swap(T);
		}
		p >>= 1;
		if (p > 0)
		{
			std::fill(T.begin(), T.end(), 0.);
			for (int i = 0; i < K; i++)
				for (int l = 0; l < K; l++)
				{
					double b = B[i * K + l];
					if (b == 0.) continue;
					for (int j = 0; j < K; j++)
						T[i * K + j] += b * B[l * K + j];
				}
			B.swap(T);
		}
	}
}

// v A (row vector times matrix)
static inline void vecMat(const double *v, const double *A, double *out, int K)
{
	for (int j = 0; j < K; j++) out[j] = 0.;
	for (int i = 0; i < K; i++)
	{
		double vi = v[i];
		if (vi == 0.) continue;
		const double *a = A + (size_t)i * K;
		for (int j = 0; j < K; j++)
			out[j] += vi * a[j];
	}
}

// A v (matrix times column vector)
static inline void matVec(const double *A, const double *v, double *out, int K)
{
	for (int i = 0; i < K; i++)
	{
		const double *a = A + (size_t)i * K;
		double s = 0.;
		for (int j = 0; j < K; j++)
			s += a[j] * v[j];
		out[i] = s;
	}
}

static inline void normalise(double *v, int K)
{
	double s = 0.;
	for (int k = 0; k < K; k++) s += v[k];
	if (s > 0.)
		for (int k = 0; k < K; k++) v[k] /= s;
	else
		for (int k = 0; k < K; k++) v[k] = 1. / K;
}

namespace
{
std::vector<long int> buildGlobalPositions(long int N, int F,
                                           const std::vector<std::vector<long int> > &pol_members,
                                           const std::vector<std::vector<long int> > &pol_pos,
                                           const std::vector<long int> &seg_pos)
{
	std::vector<long int> gpos(N, -1);
	if ((int)pol_members.size() == F && (int)pol_pos.size() == F)
	{
		for (int f = 0; f < F; f++)
			for (size_t j = 0; j < pol_members[f].size() && j < pol_pos[f].size(); j++)
			{
				const long int i = pol_members[f][j];
				if (i >= 0 && i < N) gpos[i] = pol_pos[f][j];
			}
	}
	for (long int i = 0; i < N; i++)
		if (gpos[i] < 0 && i < (long int)seg_pos.size())
			gpos[i] = seg_pos[i];
	return gpos;
}

void fitTypePeriodAndPhasors(const std::vector<int> &ch_id, const std::vector<int> &cls_id,
                             const std::vector<double> &x_signed, int F_sel, int K,
                             RFLOAT min_angst, RFLOAT max_angst, RFLOAT step_angst,
                             double &bestC_out, double &bestR_out,
                             std::vector<double> &z_cos, std::vector<double> &z_sin, std::vector<double> &z_r)
{
	bestC_out = 0.;
	bestR_out = 0.;
	z_cos.assign(K, 0.);
	z_sin.assign(K, 0.);
	z_r.assign(K, 0.);
	const long int nsel = (long int)ch_id.size();
	if (nsel <= 0 || F_sel <= 0 || K <= 0 || max_angst < min_angst || step_angst <= 0.)
		return;

	std::vector<int> cnt_k(K, 0);
	std::vector<std::vector<long int> > fil_elems(F_sel);
	for (long int n = 0; n < nsel; n++)
	{
		cnt_k[cls_id[n]]++;
		fil_elems[ch_id[n]].push_back(n);
	}
	int active_k = 0;
	for (int k = 0; k < K; k++)
		if (cnt_k[k] > 0) active_k++;
	if (active_k < 2)
		return;

	auto unit = [](std::complex<double> c) {
		const double a = std::abs(c);
		return (a > 0.) ? (c / a) : std::complex<double>(1., 0.);
	};
	auto evalPeriod = [&](double C, std::vector<std::complex<double> > *pf_out = NULL) -> double {
		std::mt19937 rng(1);
		std::uniform_real_distribution<double> U(0., 2. * PI);
		std::vector<std::complex<double> > en(nsel), pf(F_sel), tk(K, 0.), tk0(K, 0.), z(K), zf(F_sel);
		for (int f = 0; f < F_sel; f++) pf[f] = std::polar(1., U(rng));
		for (long int n = 0; n < nsel; n++) en[n] = std::polar(1., 2. * PI * x_signed[n] / C);
		for (int it = 0; it < 12; it++)
		{
			std::fill(z.begin(), z.end(), std::complex<double>(0., 0.));
			std::complex<double> sum_all(0., 0.);
			for (long int n = 0; n < nsel; n++)
			{
				const std::complex<double> v = en[n] * pf[ch_id[n]];
				z[cls_id[n]] += v;
				sum_all += v;
			}
			const std::complex<double> dc = sum_all / (double)nsel;
			std::complex<double> mu_tk(0., 0.);
			for (int k = 0; k < K; k++)
			{
				if (cnt_k[k] > 0)
				{
					tk[k] = unit(z[k] / (double)cnt_k[k] - dc);
					mu_tk += (double)cnt_k[k] * tk[k];
				}
				else
					tk[k] = 0.;
			}
			mu_tk /= (double)nsel;
			for (int k = 0; k < K; k++)
				tk0[k] = (cnt_k[k] > 0) ? (tk[k] - mu_tk) : std::complex<double>(0., 0.);
			std::fill(zf.begin(), zf.end(), std::complex<double>(0., 0.));
			for (long int n = 0; n < nsel; n++)
				zf[ch_id[n]] += tk0[cls_id[n]] * std::conj(en[n]);
			for (int f = 0; f < F_sel; f++)
				pf[f] = unit(zf[f]);
		}
		double val_sum = 0.;
		for (int f = 0; f < F_sel; f++)
		{
			const std::vector<long int> &fe = fil_elems[f];
			if (fe.empty()) continue;
			std::complex<double> mf(0., 0.);
			for (long int n : fe)
				mf += tk0[cls_id[n]];
			mf /= (double)fe.size();
			std::complex<double> rf(0., 0.);
			for (long int n : fe)
				rf += (tk0[cls_id[n]] - mf) * std::conj(en[n]);
			val_sum += std::abs(rf);
		}
		if (pf_out) *pf_out = std::move(pf);
		return val_sum / (double)nsel;
	};

	const int nC = (int)std::floor((max_angst - min_angst) / step_angst + 1e-6) + 1;
	std::vector<double> Rs(nC, -1.);
	#pragma omp parallel for schedule(dynamic)
	for (int ic = 0; ic < nC; ic++)
		Rs[ic] = evalPeriod(min_angst + ic * step_angst);

	double bestR = -1., bestC = min_angst;
	for (int ic = 0; ic < nC; ic++)
		if (Rs[ic] > bestR) { bestR = Rs[ic]; bestC = min_angst + ic * step_angst; }

	// Refine bestC on a 2 A sub-grid around the coarse peak
	const double rlo = std::max((double)min_angst, bestC - 16.);
	const double rhi = std::min((double)max_angst, bestC + 16.);
	const int nRef = (int)std::floor((rhi - rlo) / 2.0 + 1e-6) + 1;
	if (nRef > 1)
	{
		std::vector<double> Rref(nRef, -1.);
		#pragma omp parallel for schedule(dynamic)
		for (int ir = 0; ir < nRef; ir++)
			Rref[ir] = evalPeriod(rlo + ir * 2.0);
		for (int ir = 0; ir < nRef; ir++)
			if (Rref[ir] > bestR) { bestR = Rref[ir]; bestC = rlo + ir * 2.0; }
	}

	bestC_out = bestC;
	bestR_out = std::max(0., bestR);
	std::vector<std::complex<double> > pf;
	evalPeriod(bestC, &pf);
	std::vector<std::complex<double> > sum_k(K, 0.), raw_z(K, 0.);
	for (long int n = 0; n < nsel; n++)
		sum_k[cls_id[n]] += std::polar(1., 2. * PI * x_signed[n] / bestC) * pf[ch_id[n]];
	std::complex<double> dc_out(0., 0.);
	for (int k = 0; k < K; k++)
	{
		if (cnt_k[k] > 0)
		{
			raw_z[k] = sum_k[k] / (double)cnt_k[k];
			dc_out += (double)cnt_k[k] * raw_z[k];
		}
	}
	dc_out /= (double)nsel;
	for (int k = 0; k < K; k++)
	{
		if (cnt_k[k] > 0)
		{
			const std::complex<double> zk = raw_z[k] - dc_out;
			z_cos[k] = zk.real();
			z_sin[k] = zk.imag();
			z_r[k] = std::min(1.0, std::abs(zk));
		}
	}
}

struct Gmm1D2
{
	double pi = 0.5, m0 = 0., m1 = 0., v0 = 1., v1 = 1.;
	double ashmanD() const
	{
		return std::fabs(m1 - m0) / std::sqrt(0.5 * (v0 + v1));
	}
	double llPoint(double u) const
	{
		const double d0 = u - m0, d1 = u - m1;
		const double p0 = pi * std::exp(-0.5 * d0 * d0 / v0) / std::sqrt(2. * PI * v0);
		const double p1 = (1. - pi) * std::exp(-0.5 * d1 * d1 / v1) / std::sqrt(2. * PI * v1);
		return std::log(std::max(1e-300, p0 + p1));
	}
	double resp0(double u) const
	{
		const double d0 = u - m0, d1 = u - m1;
		const double l0 = std::log(std::max(1e-12, pi)) - 0.5 * std::log(2. * PI * v0) - 0.5 * d0 * d0 / v0;
		const double l1 = std::log(std::max(1e-12, 1. - pi)) - 0.5 * std::log(2. * PI * v1) - 0.5 * d1 * d1 / v1;
		const double mx = std::max(l0, l1);
		const double e0 = std::exp(l0 - mx), e1 = std::exp(l1 - mx);
		return e0 / (e0 + e1);
	}
};

bool fitWeightedGmm2(const std::vector<double> &u, const std::vector<double> &w, const std::vector<int> &idx,
                     double min_var, Gmm1D2 &out)
{
	if (idx.size() < 6) return false;
	std::vector<double> sorted_u;
	sorted_u.reserve(idx.size());
	double wtot = 0.;
	for (int id : idx) { sorted_u.push_back(u[id]); wtot += w[id]; }
	if (wtot <= 0.) return false;
	std::sort(sorted_u.begin(), sorted_u.end());
	out.m0 = sorted_u[sorted_u.size() / 4];
	out.m1 = sorted_u[(3 * sorted_u.size()) / 4];
	if (std::fabs(out.m1 - out.m0) < 1e-4)
	{
		out.m0 = sorted_u.front();
		out.m1 = sorted_u.back();
	}
	if (std::fabs(out.m1 - out.m0) < 1e-6) return false;
	double mean_all = 0., var_all = 0.;
	for (int id : idx) mean_all += w[id] * u[id];
	mean_all /= wtot;
	for (int id : idx) var_all += w[id] * (u[id] - mean_all) * (u[id] - mean_all);
	var_all = std::max(min_var, var_all / wtot);
	out.v0 = out.v1 = std::max(min_var, 0.25 * var_all);
	out.pi = 0.5;

	for (int it = 0; it < 25; it++)
	{
		double n0 = 0., n1 = 0., s0 = 0., s1 = 0., q0 = 0., q1 = 0.;
		for (int id : idx)
		{
			const double g0 = out.resp0(u[id]), g1 = 1. - g0;
			n0 += w[id] * g0;
			n1 += w[id] * g1;
			s0 += w[id] * g0 * u[id];
			s1 += w[id] * g1 * u[id];
		}
		if (n0 < 1e-6 || n1 < 1e-6) return false;
		const double nm0 = s0 / n0, nm1 = s1 / n1;
		for (int id : idx)
		{
			const double g0 = out.resp0(u[id]), g1 = 1. - g0;
			q0 += w[id] * g0 * (u[id] - nm0) * (u[id] - nm0);
			q1 += w[id] * g1 * (u[id] - nm1) * (u[id] - nm1);
		}
		out.m0 = nm0;
		out.m1 = nm1;
		out.v0 = std::max(min_var, q0 / n0);
		out.v1 = std::max(min_var, q1 / n1);
		out.pi = std::max(0.02, std::min(0.98, n0 / (n0 + n1)));
	}
	if (out.m0 > out.m1)
	{
		std::swap(out.m0, out.m1);
		std::swap(out.v0, out.v1);
		out.pi = 1. - out.pi;
	}
	return true;
}

void computeFilamentPosteriors(long int N, int K, int T, int F,
                               const std::vector<long int> &seg_chain,
                               const std::vector<long int> &filament_of_chain,
                               const std::vector<char> &junk_class,
                               const std::vector<float> &loglik,
                               const std::vector<float> &tpost,
                               const std::vector<int> &seg_type,
                               const std::vector<double> &fil_chain_post,
                               std::vector<float> &fil_type_posterior)
{
	fil_type_posterior.assign(F, 0.f);
	if (F <= 0 || T <= 0 || N <= 0) return;
	std::vector<double> tsum((size_t)F * T, 0.);
	std::vector<long int> tcnt((size_t)F * T, 0);
	std::vector<long int> ngood(F, 0);
	for (long int i = 0; i < N; i++)
	{
		if (seg_chain[i] < 0 || seg_chain[i] >= (long int)filament_of_chain.size()) continue;
		const long int f = filament_of_chain[seg_chain[i]];
		if (f < 0 || f >= F) continue;
		if ((long int)seg_type.size() != N || seg_type[i] < 0 || seg_type[i] >= T) continue;
		if (!junk_class.empty() && K > 0 && (long int)loglik.size() == N * (long int)K && !std::isnan(loglik[(size_t)i * K]))
		{
			int best = 0;
			for (int k = 1; k < K; k++)
				if (loglik[(size_t)i * K + k] > loglik[(size_t)i * K + best]) best = k;
			if (junk_class[best]) continue;
		}
		ngood[f]++;
		tcnt[(size_t)f * T + seg_type[i]]++;
		if ((long int)tpost.size() == N * (long int)T)
			for (int t = 0; t < T; t++)
				tsum[(size_t)f * T + t] += tpost[(size_t)i * T + t];
	}
	const bool have_cp = ((long int)fil_chain_post.size() == (long int)F * T);
	for (int f = 0; f < F; f++)
	{
		if (ngood[f] <= 0) continue;
		int bf = 0;
		for (int t = 1; t < T; t++)
			if (tsum[(size_t)f * T + t] > tsum[(size_t)f * T + bf] ||
			    (tsum[(size_t)f * T + t] == tsum[(size_t)f * T + bf] && tcnt[(size_t)f * T + t] > tcnt[(size_t)f * T + bf]))
				bf = t;
		const double mean_tp = ((long int)tpost.size() == N * (long int)T)
			? (tsum[(size_t)f * T + bf] / (double)ngood[f])
			: ((double)tcnt[(size_t)f * T + bf] / (double)ngood[f]);
		const double purity = (double)tcnt[(size_t)f * T + bf] / (double)ngood[f];
		double p = mean_tp * purity;
		if (have_cp)
			p = std::min(fil_chain_post[(size_t)f * T + bf], mean_tp) * purity;
		fil_type_posterior[f] = (float)std::max(0.0, std::min(1.0, p));
	}
}
} // namespace

void HelicalContext::fitPeriods(const std::vector<RFLOAT> &psi, const std::vector<RFLOAT> &psi_prior,
                                RFLOAT min_angst, RFLOAT max_angst, RFLOAT step_angst)
{
	const int T = std::max(1, nr_types);
	const int F = (int)filament_name.size();
	type_period.assign(T, 0.);
	type_period_R.assign(T, 0.);
	class_phase_cos.assign((size_t)T * K, 0.);
	class_phase_sin.assign((size_t)T * K, 0.);
	class_phase_r.assign((size_t)T * K, 0.);
	fil_period.assign(F, 0.);
	fil_period_R.assign(F, 0.);
	if (N <= 0 || K <= 0 || step <= 0.) return;

	const std::vector<long int> gpos = buildGlobalPositions(N, F, pol_members, pol_pos, seg_pos);
	auto segSignedCoord = [&](long int i) -> double {
		const long int p = (gpos[i] >= 0) ? gpos[i] : seg_pos[i];
		double sg = 1.;
		if ((long int)pol_post.size() == N && std::isfinite(pol_post[i]) && std::fabs(pol_post[i] - 0.5f) > 1e-5f)
			sg = (pol_post[i] > 0.5f) ? -1. : 1.;
		else if (i < (long int)psi.size() && i < (long int)psi_prior.size())
			sg = (cos(DEG2RAD(psi[i] - psi_prior[i])) >= 0.) ? 1. : -1.;
		return (double)p * (double)step * sg;
	};
	auto segBestClass = [&](long int i) -> int {
		int best = -1;
		double mx = -1e30;
		const bool have_p = ((long int)post.size() == N * (long int)K);
		for (int k = 0; k < K; k++)
		{
			if (!junk_class.empty() && k < (int)junk_class.size() && junk_class[k]) continue;
			const double v = have_p ? (double)post[(size_t)i * K + k] : (double)loglik[(size_t)i * K + k];
			if (v > mx) { mx = v; best = k; }
		}
		return best;
	};

	std::vector<int> seg_fil(N, -1), seg_cls(N, -1);
	for (long int i = 0; i < N; i++)
	{
		if (seg_chain[i] >= 0 && seg_chain[i] < (long int)filament_of_chain.size())
			seg_fil[i] = (int)filament_of_chain[seg_chain[i]];
		seg_cls[i] = segBestClass(i);
	}

	for (int t = 0; t < T; t++)
	{
		std::map<int, int> per_fil_cnt;
		std::map<int, std::pair<double, double> > per_fil_span;
		for (long int i = 0; i < N; i++)
		{
			const int f = seg_fil[i];
			if (f < 0 || seg_cls[i] < 0) continue;
			if ((long int)seg_type.size() == N && seg_type[i] != t) continue;
			if ((long int)seg_type_prob.size() == N && seg_type_prob[i] < 0.75f) continue;
			per_fil_cnt[f]++;
			const double posA = (double)((gpos[i] >= 0) ? gpos[i] : seg_pos[i]) * (double)step;
			auto it = per_fil_span.find(f);
			if (it == per_fil_span.end()) per_fil_span[f] = {posA, posA};
			else { it->second.first = std::min(it->second.first, posA); it->second.second = std::max(it->second.second, posA); }
		}
		std::vector<int> fil_idx_map(F, -1);
		std::vector<int> sel_fil, sel_cls;
		std::vector<double> sel_x;
		int F_sel = 0;
		for (long int i = 0; i < N; i++)
		{
			const int f = seg_fil[i];
			if (f < 0 || seg_cls[i] < 0) continue;
			if ((long int)seg_type.size() == N && seg_type[i] != t) continue;
			if ((long int)seg_type_prob.size() == N && seg_type_prob[i] < 0.75f) continue;
			if (per_fil_cnt[f] < 8) continue;
			const auto sp = per_fil_span[f];
			if (sp.second - sp.first < 150.0) continue;
			if (fil_idx_map[f] < 0) fil_idx_map[f] = F_sel++;
			sel_fil.push_back(fil_idx_map[f]);
			sel_cls.push_back(seg_cls[i]);
			sel_x.push_back(segSignedCoord(i));
		}
		if (sel_fil.size() < 80 || F_sel < 3) continue;

		std::vector<double> zc, zs, zr;
		fitTypePeriodAndPhasors(sel_fil, sel_cls, sel_x, F_sel, K, min_angst, max_angst, step_angst,
		                        type_period[t], type_period_R[t], zc, zs, zr);
		for (int k = 0; k < K; k++)
		{
			class_phase_cos[(size_t)t * K + k] = zc[k];
			class_phase_sin[(size_t)t * K + k] = zs[k];
			class_phase_r[(size_t)t * K + k] = zr[k];
		}
	}

	// Per-filament crossover period from the type's empirical class phasors
	std::vector<std::vector<long int> > fil_segs(F);
	for (long int i = 0; i < N; i++)
		if (seg_fil[i] >= 0 && seg_fil[i] < F && seg_cls[i] >= 0)
			fil_segs[seg_fil[i]].push_back(i);

	#pragma omp parallel for schedule(dynamic, 16)
	for (int f = 0; f < F; f++)
	{
		const std::vector<long int> &fs = fil_segs[f];
		if (fs.size() < 8) continue;
		// Dominant type on filament f
		std::vector<int> tcnt(T, 0);
		double pmin = 1e30, pmax = -1e30;
		for (long int i : fs)
		{
			const int t = ((long int)seg_type.size() == N && seg_type[i] >= 0 && seg_type[i] < T) ? seg_type[i] : 0;
			tcnt[t]++;
			const double posA = (double)((gpos[i] >= 0) ? gpos[i] : seg_pos[i]) * (double)step;
			if (posA < pmin) pmin = posA;
			if (posA > pmax) pmax = posA;
		}
		if (pmax - pmin < 150.0) continue;
		const int tf = (int)(std::max_element(tcnt.begin(), tcnt.end()) - tcnt.begin());
		const double Ct = type_period[tf];
		if (!(type_period_R[tf] >= 0.25 && Ct > 0.)) continue;

		std::vector<double> sx, z_re, z_im, mag;
		double mag_sum = 0.;
		for (long int i : fs)
		{
			const int k = seg_cls[i];
			const double zr = class_phase_r[(size_t)tf * K + k];
			if (zr <= 0.) continue;
			sx.push_back(segSignedCoord(i));
			z_re.push_back(class_phase_cos[(size_t)tf * K + k]);
			z_im.push_back(class_phase_sin[(size_t)tf * K + k]);
			mag_sum += zr;
		}
		if (sx.size() < 8 || mag_sum <= 1e-6) continue;

		double mean_zre = 0., mean_zim = 0.;
		for (size_t j = 0; j < sx.size(); j++)
		{
			mean_zre += z_re[j];
			mean_zim += z_im[j];
		}
		mean_zre /= (double)sx.size();
		mean_zim /= (double)sx.size();
		std::vector<double> zc_re(sx.size()), zc_im(sx.size());
		double znorm2 = 0.;
		for (size_t j = 0; j < sx.size(); j++)
		{
			zc_re[j] = z_re[j] - mean_zre;
			zc_im[j] = z_im[j] - mean_zim;
			znorm2 += zc_re[j] * zc_re[j] + zc_im[j] * zc_im[j];
		}
		const double znorm = std::sqrt(znorm2);
		if (znorm < 1e-6) continue;

		auto scoreP = [&](double P) -> double {
			const double w = 2. * PI / P;
			const size_t M = sx.size();
			std::vector<double> ex_re(M), ex_im(M);
			double me_re = 0., me_im = 0.;
			for (size_t j = 0; j < M; j++)
			{
				const double ang = w * sx[j];
				ex_re[j] = std::cos(ang);
				ex_im[j] = std::sin(ang);
				me_re += ex_re[j];
				me_im += ex_im[j];
			}
			me_re /= (double)M;
			me_im /= (double)M;
			double re = 0., im = 0., enorm2 = 0.;
			for (size_t j = 0; j < M; j++)
			{
				const double er = ex_re[j] - me_re, ei = ex_im[j] - me_im;
				enorm2 += er * er + ei * ei;
				re += zc_re[j] * er + zc_im[j] * ei;
				im += zc_im[j] * er - zc_re[j] * ei;
			}
			const double denom = znorm * std::sqrt(enorm2);
			return (denom > 1e-12) ? (std::sqrt(re * re + im * im) / denom) : 0.;
		};

		const double plo = std::max((double)min_angst, 0.35 * Ct);
		double phi = std::min((double)std::max((double)max_angst, 2400.), 2.80 * Ct);
		phi = std::min(phi, std::max(plo + 20.0, 1.6 * (pmax - pmin)));
		if (phi <= plo) continue;
		double bestP = Ct, bestRf = -1.;
		for (double P = plo; P <= phi + 1e-6; P += 4.0)
		{
			const double r = scoreP(P);
			if (r > bestRf) { bestRf = r; bestP = P; }
		}
		const double r0 = std::max(plo, bestP - 10.0), r1 = std::min(phi, bestP + 10.0);
		for (double P = r0; P <= r1 + 1e-6; P += 1.0)
		{
			const double r = scoreP(P);
			if (r > bestRf) { bestRf = r; bestP = P; }
		}
		fil_period[f] = bestP;
		fil_period_R[f] = std::min(1.0, std::max(0.0, bestRf));
	}
}

void HelicalContext::applyPhaseRingPriors(const std::vector<RFLOAT> &psi, const std::vector<RFLOAT> &psi_prior)
{
	const int T = std::max(1, nr_types);
	const int F = (int)filament_name.size();
	if (!use_phase_ring || !have_prior || N <= 0 || K <= 1 || step <= 0.) return;
	if (prior.size() != (size_t)N * K || class_phase_r.size() != (size_t)T * K) return;

	const std::vector<long int> gpos = buildGlobalPositions(N, F, pol_members, pol_pos, seg_pos);
	auto segSignedCoord = [&](long int i) -> double {
		const long int p = (gpos[i] >= 0) ? gpos[i] : seg_pos[i];
		double sg = 1.;
		if ((long int)pol_post.size() == N && std::isfinite(pol_post[i]) && std::fabs(pol_post[i] - 0.5f) > 1e-5f)
			sg = (pol_post[i] > 0.5f) ? -1. : 1.;
		else if (i < (long int)psi.size() && i < (long int)psi_prior.size())
			sg = (cos(DEG2RAD(psi[i] - psi_prior[i])) >= 0.) ? 1. : -1.;
		return (double)p * (double)step * sg;
	};
	auto segBestClass = [&](long int i) -> int {
		int best = -1;
		double mx = -1e30;
		const bool have_p = ((long int)post.size() == N * (long int)K);
		for (int k = 0; k < K; k++)
		{
			if (!junk_class.empty() && k < (int)junk_class.size() && junk_class[k]) continue;
			const double v = have_p ? (double)post[(size_t)i * K + k] : (double)loglik[(size_t)i * K + k];
			if (v > mx) { mx = v; best = k; }
		}
		return best;
	};

	std::vector<std::vector<long int> > fil_segs(F);
	for (long int i = 0; i < N; i++)
		if (seg_chain[i] >= 0 && seg_chain[i] < (long int)filament_of_chain.size())
		{
			const long int f = filament_of_chain[seg_chain[i]];
			if (f >= 0 && f < F) fil_segs[f].push_back(i);
		}
	const double excl_A = (exclude_angst >= 0.) ? (double)exclude_angst : (double)(exclude_steps * step);

	#pragma omp parallel for schedule(dynamic, 16)
	for (int f = 0; f < F; f++)
	{
		const std::vector<long int> &fs = fil_segs[f];
		if (fs.size() < 4) continue;
		std::vector<int> tcnt(T, 0);
		for (long int i : fs)
		{
			const int t = ((long int)seg_type.size() == N && seg_type[i] >= 0 && seg_type[i] < T) ? seg_type[i] : 0;
			tcnt[t]++;
		}
		const int tf = (int)(std::max_element(tcnt.begin(), tcnt.end()) - tcnt.begin());
		if ((int)type_period_R.size() != T || type_period_R[tf] < 0.28 || type_period[tf] <= 0.)
			continue;

		const double Pf = (f < (int)fil_period.size() && f < (int)fil_period_R.size() &&
		                   fil_period_R[f] >= 0.45 && fil_period[f] > 0.)
		                  ? fil_period[f] : type_period[tf];
		const double omega = 2. * PI / Pf;
		const double max_dist = 2.5 * Pf;

		std::vector<double> x(fs.size());
		std::vector<int> cls_f(fs.size());
		for (size_t j = 0; j < fs.size(); j++)
		{
			x[j] = segSignedCoord(fs[j]);
			cls_f[j] = segBestClass(fs[j]);
		}

		for (size_t idx_i = 0; idx_i < fs.size(); idx_i++)
		{
			const long int i = fs[idx_i];
			double ur = 0., ui = 0., wsum = 0.;
			for (size_t idx_j = 0; idx_j < fs.size(); idx_j++)
			{
				const double dist = std::fabs(x[idx_j] - x[idx_i]);
				if (dist < excl_A || dist > max_dist) continue;
				const int kj = cls_f[idx_j];
				if (kj < 0) continue;
				const double zr = class_phase_r[(size_t)tf * K + kj];
				if (zr <= 0.) continue;
				const double zre = class_phase_cos[(size_t)tf * K + kj];
				const double zim = class_phase_sin[(size_t)tf * K + kj];
				const double ang = omega * (x[idx_i] - x[idx_j]);
				const double ca = std::cos(ang), sa = std::sin(ang);
				ur += zre * ca - zim * sa;
				ui += zre * sa + zim * ca;
				wsum += zr;
			}
			if (wsum <= 0.) continue;
			const double denom = wsum + 2.0;
			ur /= denom;
			ui /= denom;

			double mx_good = 0.;
			std::vector<double> tmp(K, 1.0);
			for (int k = 0; k < K; k++)
			{
				if (!junk_class.empty() && k < (int)junk_class.size() && junk_class[k])
				{
					tmp[k] = 1.0;
					continue;
				}
				const bool allow = (class_allowed.size() != (size_t)K * T) || class_allowed[(size_t)k * T + tf];
				const double cos_match = allow ? (ur * class_phase_cos[(size_t)tf * K + k] + ui * class_phase_sin[(size_t)tf * K + k]) : 0.;
				const double v = (double)prior[(size_t)i * K + k] * std::exp(0.85 * cos_match);
				tmp[k] = v;
				if (v > mx_good) mx_good = v;
			}
			for (int k = 0; k < K; k++)
				prior[(size_t)i * K + k] = (mx_good > 0. && (junk_class.empty() || !junk_class[k]))
				                           ? (float)std::max((double)prior_floor, tmp[k] / mx_good) : 1.f;
		}
	}
}

int HelicalContext::splitTypesByTwistRate(int tmax)
{
	const int F = (int)filament_name.size();
	if (!use_twist_rate_split || nr_types < 1 || nr_types >= tmax || F <= 0)
		return nr_types;
	if ((int)fil_period.size() != F || (int)fil_period_R.size() != F || (long int)tpost.size() != N * (long int)nr_types)
		return nr_types;

	const std::vector<long int> gpos = buildGlobalPositions(N, F, pol_members, pol_pos, seg_pos);
	std::vector<std::vector<long int> > fil_segs(F);
	for (long int i = 0; i < N; i++)
		if (seg_chain[i] >= 0 && seg_chain[i] < (long int)filament_of_chain.size())
		{
			const long int f = filament_of_chain[seg_chain[i]];
			if (f >= 0 && f < F) fil_segs[f].push_back(i);
		}

	std::vector<int> fil_best_t(F, -1), fil_ngood(F, 0);
	std::vector<double> fil_span(F, 0.);
	for (int f = 0; f < F; f++)
	{
		if (fil_segs[f].empty()) continue;
		std::vector<double> tsum(nr_types, 0.);
		double pmin = 1e30, pmax = -1e30;
		for (long int i : fil_segs[f])
		{
			const double posA = (double)((gpos[i] >= 0) ? gpos[i] : seg_pos[i]) * (double)step;
			pmin = std::min(pmin, posA);
			pmax = std::max(pmax, posA);
			if ((long int)seg_type.size() == N && seg_type[i] >= 0 && seg_type[i] < nr_types)
			{
				fil_ngood[f]++;
				for (int t = 0; t < nr_types; t++)
					tsum[t] += tpost[(size_t)i * nr_types + t];
			}
		}
		fil_span[f] = std::max(0., pmax - pmin);
		fil_best_t[f] = (int)(std::max_element(tsum.begin(), tsum.end()) - tsum.begin());
	}

	const int T_orig = nr_types;
	const double min_var = 0.03 * 0.03;
	for (int t = 0; t < T_orig && nr_types < tmax; t++)
	{
		if ((int)type_period_R.size() <= t || type_period_R[t] < 0.30) continue;
		std::vector<int> cand_f;
		for (int f = 0; f < F; f++)
		{
			if (fil_best_t[f] != t) continue;
			if (fil_period[f] > 0. && fil_period_R[f] >= 0.40 && fil_span[f] >= 220.)
				cand_f.push_back(f);
		}
		const int Ft = (int)cand_f.size();
		if (Ft < 24) continue;

		std::vector<double> u(Ft), w(Ft);
		std::vector<int> all_idx(Ft);
		for (int j = 0; j < Ft; j++)
		{
			const int f = cand_f[j];
			u[j] = std::log(fil_period[f]);
			w[j] = std::max(1.0, (double)fil_ngood[f]) * fil_period_R[f] * fil_period_R[f];
			all_idx[j] = j;
		}
		Gmm1D2 full;
		if (!fitWeightedGmm2(u, w, all_idx, min_var, full)) continue;

		const double P0 = std::exp(full.m0), P1 = std::exp(full.m1);
		const double rel_sep = std::fabs(P1 - P0) / std::exp(0.5 * (full.m0 + full.m1));
		const double D = full.ashmanD();
		if (rel_sep < 0.11 || D < 2.0) continue;

		int c0 = 0, c1 = 0;
		for (int j = 0; j < Ft; j++)
		{
			if (full.resp0(u[j]) >= 0.5) c0++;
			else c1++;
		}
		const int min_sub = std::max(8, (int)std::ceil(0.08 * Ft));
		if (c0 < min_sub || c1 < min_sub) continue;

		// 3-fold x 3-split cross-validation over candidate filaments
		const int nfold = 3, nsplit = 3;
		double cv_ll_gain = 0., cv_ari = 0.;
		int cv_valid = 0;
		for (int split = 0; split < nsplit; split++)
		{
			std::vector<int> order = all_idx;
			std::mt19937 rng(17u + (unsigned)split);
			std::shuffle(order.begin(), order.end(), rng);
			std::vector<int> fold_of(Ft, 0);
			for (int j = 0; j < Ft; j++) fold_of[order[j]] = j % nfold;
			std::vector<std::vector<int> > fold_pred(nfold, std::vector<int>(Ft, 0));
			bool ok_split = true;
			double split_gain = 0.;
			for (int fold = 0; fold < nfold; fold++)
			{
				std::vector<int> tr_idx, te_idx;
				for (int j = 0; j < Ft; j++)
				{
					if (fold_of[j] == fold) te_idx.push_back(j);
					else tr_idx.push_back(j);
				}
				Gmm1D2 gf;
				if (!fitWeightedGmm2(u, w, tr_idx, min_var, gf)) { ok_split = false; break; }
				double wtr = 0., mtr = 0., vtr = 0.;
				for (int j : tr_idx) { wtr += w[j]; mtr += w[j] * u[j]; }
				mtr /= std::max(1e-12, wtr);
				for (int j : tr_idx) vtr += w[j] * (u[j] - mtr) * (u[j] - mtr);
				vtr = std::max(min_var, vtr / std::max(1e-12, wtr));
				double wte = 0., gte = 0.;
				for (int j : te_idx)
				{
					const double d1 = u[j] - mtr;
					const double ll1 = -0.5 * std::log(2. * PI * vtr) - 0.5 * d1 * d1 / vtr;
					const double ll2 = gf.llPoint(u[j]);
					gte += w[j] * (ll2 - ll1);
					wte += w[j];
				}
				split_gain += (wte > 0.) ? (gte / wte) / nfold : 0.;
				for (int j = 0; j < Ft; j++)
					fold_pred[fold][j] = (gf.resp0(u[j]) >= 0.5) ? 0 : 1;
			}
			if (!ok_split) continue;
			double a = 0.;
			int npair = 0;
			for (int f1 = 0; f1 < nfold; f1++)
				for (int f2 = f1 + 1; f2 < nfold; f2++, npair++)
					a += helixtypes::adjustedRand(fold_pred[f1], fold_pred[f2]);
			cv_ll_gain += split_gain;
			cv_ari += (npair > 0) ? a / npair : 0.;
			cv_valid++;
		}
		if (cv_valid == 0) continue;
		cv_ll_gain /= cv_valid;
		cv_ari /= cv_valid;
		if (cv_ll_gain <= 0.08 || cv_ari < 0.75) continue;

		// Split accepted: create t_new = nr_types and transfer shared-view classes and segments
		const int T_old = nr_types;
		const int t_new = T_old;
		const int T_new = T_old + 1;
		nr_types = T_new;

		if (class_allowed.size() == (size_t)K * T_old)
		{
			std::vector<char> new_allowed((size_t)K * T_new, 0);
			for (int k = 0; k < K; k++)
			{
				for (int u_t = 0; u_t < T_old; u_t++)
					new_allowed[(size_t)k * T_new + u_t] = class_allowed[(size_t)k * T_old + u_t];
				new_allowed[(size_t)k * T_new + t_new] = class_allowed[(size_t)k * T_old + t];
			}
			class_allowed.swap(new_allowed);
		}
		if (q.size() == (size_t)T_old * K)
		{
			q.resize((size_t)T_new * K);
			std::copy(q.begin() + (size_t)t * K, q.begin() + (size_t)(t + 1) * K, q.begin() + (size_t)t_new * K);
		}
		if (pi0.size() == (size_t)T_old * K)
		{
			pi0.resize((size_t)T_new * K);
			for (int k = 0; k < K; k++)
			{
				const double pv = pi0[(size_t)t * K + k];
				pi0[(size_t)t * K + k] = pv * full.pi;
				pi0[(size_t)t_new * K + k] = pv * (1. - full.pi);
			}
		}
		if (A.size() == (size_t)T_old * K * K)
		{
			A.resize((size_t)T_new * K * K);
			std::copy(A.begin() + (size_t)t * K * K, A.begin() + (size_t)(t + 1) * K * K, A.begin() + (size_t)t_new * K * K);
		}

		std::vector<double> gamma0(F, full.pi);
		for (int f = 0; f < F; f++)
			if (fil_best_t[f] == t && fil_period[f] > 0. && fil_period_R[f] >= 0.30)
				gamma0[f] = full.resp0(std::log(fil_period[f]));

		if ((long int)fil_chain_post.size() == (long int)F * T_old)
		{
			std::vector<double> new_cp((size_t)F * T_new, 0.);
			for (int f = 0; f < F; f++)
			{
				for (int u_t = 0; u_t < T_old; u_t++)
					new_cp[(size_t)f * T_new + u_t] = fil_chain_post[(size_t)f * T_old + u_t];
				const double old_c = fil_chain_post[(size_t)f * T_old + t];
				new_cp[(size_t)f * T_new + t] = old_c * gamma0[f];
				new_cp[(size_t)f * T_new + t_new] = old_c * (1. - gamma0[f]);
			}
			fil_chain_post.swap(new_cp);
		}

		std::vector<float> new_tpost((size_t)N * T_new, 0.f);
		type_mass.assign(T_new, 0.);
		for (long int i = 0; i < N; i++)
		{
			const int f = (seg_chain[i] >= 0 && seg_chain[i] < (long int)filament_of_chain.size()) ? (int)filament_of_chain[seg_chain[i]] : -1;
			const double g0 = (f >= 0 && f < F) ? gamma0[f] : full.pi;
			for (int u_t = 0; u_t < T_old; u_t++)
				new_tpost[(size_t)i * T_new + u_t] = tpost[(size_t)i * T_old + u_t];
			const float old_t = tpost[(size_t)i * T_old + t];
			new_tpost[(size_t)i * T_new + t] = (float)(old_t * g0);
			new_tpost[(size_t)i * T_new + t_new] = (float)(old_t * (1. - g0));
			if (seg_chain[i] < 0 || ((long int)seg_type.size() == N && seg_type[i] < 0)) continue;
			int best = 0;
			for (int u_t = 0; u_t < T_new; u_t++)
			{
				type_mass[u_t] += new_tpost[(size_t)i * T_new + u_t];
				if (new_tpost[(size_t)i * T_new + u_t] > new_tpost[(size_t)i * T_new + best]) best = u_t;
			}
			seg_type[i] = best;
			seg_type_prob[i] = new_tpost[(size_t)i * T_new + best];
			if (fil_best_t[f] == t && best == t_new) fil_best_t[f] = t_new;
		}
		tpost.swap(new_tpost);

		if (verb > 0)
			std::cout << " Filament context: split type " << t + 1 << " by twist rate into types " << t + 1
			          << " (period " << ROUND(P0) << " A, n=" << c0 << ") and " << t_new + 1
			          << " (period " << ROUND(P1) << " A, n=" << c1 << "; Ashman D=" << std::setprecision(3) << D
			          << ", CV dLL=" << cv_ll_gain << ", fold ARI=" << cv_ari << ")" << std::endl;
	}

	if (nr_types != T_orig)
	{
		makeSections();
		computeFilamentPosteriors(N, K, nr_types, F, seg_chain, filament_of_chain,
		                          junk_class, loglik, tpost, seg_type, fil_chain_post, fil_type_posterior);
		// Recompute per-type crossover periods and phasors across all active types
		const std::vector<double> saved_fp = fil_period;
		const std::vector<double> saved_fr = fil_period_R;
		fitPeriods(std::vector<RFLOAT>(), std::vector<RFLOAT>());
		if (saved_fp.size() == (size_t)F) fil_period = saved_fp;
		if (saved_fr.size() == (size_t)F) fil_period_R = saved_fr;
	}
	return nr_types;
}

void HelicalContext::writeLabels(MetaDataTable &MD, const std::vector<long int> &seg_rows, const std::vector<int> *assigned_class) const
{
	const long int F = (long int)filament_name.size();
	for (long int i = 0; i < N; i++)
	{
		long int r = seg_rows[i];
		// a segment in a junk class has no type (0)
		const bool junk = assigned_class != NULL && !junk_class.empty() && (*assigned_class)[i] >= 0 && (*assigned_class)[i] < K &&
		                  junk_class[(*assigned_class)[i]];
		MD.setValue(EMDL_PARTICLE_HELICAL_TYPE, junk ? 0 : seg_type[i] + 1, r);
		MD.setValue(EMDL_PARTICLE_HELICAL_TYPE_PROBABILITY, junk ? (RFLOAT)0. : (RFLOAT)seg_type_prob[i], r);
		RFLOAT fil_post = 0.;
		if (seg_chain[i] >= 0 && seg_chain[i] < (long int)filament_of_chain.size())
		{
			const long int f = filament_of_chain[seg_chain[i]];
			if (f >= 0 && f < F && f < (long int)fil_type_posterior.size())
				fil_post = (RFLOAT)fil_type_posterior[f];
		}
		MD.setValue(EMDL_PARTICLE_HELICAL_TYPE_POSTERIOR, fil_post, r);
		MD.setValue(EMDL_PARTICLE_HELICAL_SECTION_ID, seg_section[i] + 1, r);
		RFLOAT p = 0.;
		if (assigned_class != NULL && (*assigned_class)[i] >= 0 && (*assigned_class)[i] < K && seg_chain[i] >= 0)
		{
			double s = 0.;
			for (int k = 0; k < K; k++)
				s += prior[i * K + k];
			p = (s > 0.) ? prior[i * K + (*assigned_class)[i]] / s : 0.;
		}
		MD.setValue(EMDL_PARTICLE_HELICAL_CONTEXT_PROBABILITY, p, r);
	}
}

static inline double normaliseSum(double *v, int n)
{
	double s = 0.;
	for (int k = 0; k < n; k++) s += v[k];
	if (s > 0.)
		for (int k = 0; k < n; k++) v[k] /= s;
	else
		for (int k = 0; k < n; k++) v[k] = 1. / n;
	return s;
}

// Build class_equiv Union-Find clusters of visually equivalent non-junk classes (low-passed inside tube mask)
void HelicalContext::computeClassEquivalence(const std::vector<MultidimArray<RFLOAT> > &refs, RFLOAT apix, RFLOAT outer_diam)
{
	class_equiv.assign(K, 0);
	for (int k = 0; k < K; k++) class_equiv[k] = k;
	if ((int)refs.size() != K || K < 2 || apix <= 0. || YSIZE(refs[0]) < 8 || ZSIZE(refs[0]) != 1)
		return;
	const int ny = YSIZE(refs[0]), nx = XSIZE(refs[0]);
	const int dy_max = 1;
	const int sx_max = std::max(1, (int)ROUND(3.0 / apix));
	const double r2max = 0.45 * 0.45 * nx * nx;
	const double hy_max = (outer_diam > 0.) ? (0.60 * outer_diam / apix) : 1e9;
	std::vector<char> inside((size_t)ny * nx, 0);
	for (int i = 0; i < ny; i++)
		for (int j = 0; j < nx; j++)
		{
			const double y = i - ny / 2, x = j - nx / 2;
			inside[(size_t)i * nx + j] = (x * x + y * y < r2max) && (std::fabs(y) <= hy_max);
		}
	std::vector<std::vector<double> > img(K);
	std::vector<double> norm2(K, 0.);
	const RFLOAT lp = std::max((RFLOAT)9., (RFLOAT)(2.2 * apix));
	for (int k = 0; k < K; k++)
	{
		if (!junk_class.empty() && junk_class[k]) continue;
		if (MULTIDIM_SIZE(refs[k]) == 0) continue;
		MultidimArray<RFLOAT> f = refs[k];
		lowPassFilterMap(f, lp, apix);
		double sum = 0.;
		long int n = 0;
		for (int p = 0; p < ny * nx; p++)
			if (inside[p]) sum += DIRECT_MULTIDIM_ELEM(f, p), n++;
		if (n == 0) continue;
		img[k].assign((size_t)ny * nx, 0.);
		double n2 = 0.;
		for (int p = 0; p < ny * nx; p++)
			if (inside[p])
			{
				const double v = DIRECT_MULTIDIM_ELEM(f, p) - sum / n;
				img[k][p] = v;
				n2 += v * v;
			}
		norm2[k] = n2;
	}
	std::vector<int> parent(K);
	for (int k = 0; k < K; k++) parent[k] = k;
	auto find_rep = [&](int x) { while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
	for (int a = 0; a < K; a++)
	{
		if (norm2[a] <= 0.) continue;
		for (int b = a + 1; b < K; b++)
		{
			if (norm2[b] <= 0.) continue;
			if (find_rep(a) == find_rep(b)) continue;
			double best = -1.;
			for (int dy = -dy_max; dy <= dy_max; dy++)
				for (int sx = -sx_max; sx <= sx_max; sx++)
				{
					double ab = 0., aa = 0., bb = 0.;
					for (int i = std::max(0, -dy); i < std::min(ny, ny - dy); i++)
						for (int j = std::max(0, -sx); j < std::min(nx, nx - sx); j++)
						{
							const int p = i * nx + j, q = (i + dy) * nx + j + sx;
							if (!inside[p] || !inside[q]) continue;
							const double va = img[a][p], vb = img[b][q];
							ab += va * vb; aa += va * va; bb += vb * vb;
						}
					if (aa > 0. && bb > 0.)
						best = std::max(best, ab / std::sqrt(aa * bb));
				}
			if (best >= 0.975)
			{
				int ra = find_rep(a), rb = find_rep(b);
				if (ra != rb) parent[std::max(ra, rb)] = std::min(ra, rb);
			}
		}
	}
	for (int k = 0; k < K; k++)
		class_equiv[k] = find_rep(k);
}

helixtypes::FilamentChains HelicalContext::buildFilamentChains(std::vector<int> *state_to_first_cls) const
{
	helixtypes::FilamentChains FC;
	std::vector<int> eq(K);
	for (int k = 0; k < K; k++)
		eq[k] = (int)class_equiv.size() == K ? class_equiv[k] : k;

	std::map<int, int> rep_to_state;
	std::vector<int> col(K, -1);
	if (state_to_first_cls) state_to_first_cls->clear();
	for (int k = 0; k < K; k++)
	{
		if (!junk_class.empty() && junk_class[k]) continue;
		const int r = eq[k];
		auto it = rep_to_state.find(r);
		if (it == rep_to_state.end())
		{
			const int sid = FC.C++;
			rep_to_state[r] = sid;
			if (state_to_first_cls) state_to_first_cls->push_back(k);
			col[k] = sid;
		}
		else
			col[k] = it->second;
	}

	const int nfil = (int)filament_name.size();
	FC.seqs.resize(nfil);
	FC.mics.resize(nfil);
	for (int f = 0; f < nfil; f++)
	{
		const std::string &nm = filament_name[f];
		const size_t colon = nm.rfind(':');
		FC.mics[f] = (colon != std::string::npos) ? nm.substr(0, colon) : std::string();
	}
	FC.kappa = (step > 0. && diameter > 0.) ? (double)temper : 1.0;

	if (!chains.empty())
	{
		for (size_t c = 0; c < chains.size(); c++)
		{
			const int f = (int)filament_of_chain[c];
			const std::vector<long int> &grid = chains[c];
			const size_t G = grid.size();
			for (size_t p = 0; p < G; p++)
			{
				const long int i = grid[p];
				if (i < 0 || std::isnan(loglik[(size_t)i * K])) continue;
				const long int ip = (p > 0 && grid[p - 1] >= 0 && !std::isnan(loglik[(size_t)grid[p - 1] * K])) ? grid[p - 1] : -1;
				const long int in = (p + 1 < G && grid[p + 1] >= 0 && !std::isnan(loglik[(size_t)grid[p + 1] * K])) ? grid[p + 1] : -1;
				int b = 0;
				double bval = -1e300;
				double wsum = 0.50 + (ip >= 0 ? 0.25 : 0.) + (in >= 0 ? 0.25 : 0.);
				for (int k = 0; k < K; k++)
				{
					double v = 0.50 * (double)loglik[(size_t)i * K + k];
					if (ip >= 0) v += 0.25 * (double)loglik[(size_t)ip * K + k];
					if (in >= 0) v += 0.25 * (double)loglik[(size_t)in * K + k];
					v /= wsum;
					if (v > bval) { bval = v; b = k; }
				}
				if (col[b] >= 0)
					FC.seqs[f].push_back(col[b]);
			}
		}
	}
	else
	{
		std::vector<int> best_cls(N, -1);
		for (long int i = 0; i < N; i++)
		{
			if (seg_chain[i] < 0 || std::isnan(loglik[i * K])) continue;
			int b = 0;
			for (int k = 1; k < K; k++)
				if (loglik[i * K + k] > loglik[i * K + b]) b = k;
			best_cls[i] = col[b];
		}
		for (long int i = 0; i < N; i++)
		{
			if (seg_chain[i] < 0 || best_cls[i] < 0) continue;
			const int f = (int)filament_of_chain[seg_chain[i]];
			if (f >= 0 && f < nfil)
				FC.seqs[f].push_back(best_cls[i]);
		}
	}
	return FC;
}

// Types from a tempered first-order Markov chain mixture over the non-junk classes of filaments
// (falls back cleanly when chains are unavailable), followed by exclusive class-to-type partitioning.
void HelicalContext::initialiseModel()
{
	const int T = nr_types, F = filament_name.size();
	std::vector<int> best(N, -1);
	std::vector<char> is_junk(K, 0);
	for (int k = 0; k < K; k++)
		is_junk[k] = (!junk_class.empty() && junk_class[k]) ? 1 : 0;

	for (long int i = 0; i < N; i++)
		if (!std::isnan(loglik[i * K]))
			best[i] = std::max_element(loglik.begin() + i * K, loglik.begin() + (i + 1) * K) - (loglik.begin() + i * K);

	std::vector<double> Wn(F * T, 0.);
	helixtypes::FilamentChains FC = buildFilamentChains();
	std::vector<double> chain_post;
	bool used_chain_mix = false;
	if (FC.C >= 1 && F > 0)
	{
		int n_nonempty = 0;
		for (const auto &s : FC.seqs) if (!s.empty()) n_nonempty++;
		if (n_nonempty >= T)
		{
			helixtypes::fitChainTypes(FC, T, 1u, &chain_post);
			if ((int)chain_post.size() == F * T)
			{
				Wn = chain_post;
				used_chain_mix = true;
			}
		}
	}

	std::vector<double> fil_cnt(F * K, 0.);
	std::vector<double> cls_tot(K, 0.);
	if (!chains.empty())
	{
		for (size_t c = 0; c < chains.size(); c++)
			if (chain_test.empty() || !chain_test[c])
				for (long int i : chains[c])
					if (i >= 0 && best[i] >= 0)
						fil_cnt[filament_of_chain[c] * K + best[i]] += 1., cls_tot[best[i]] += 1.;
	}
	else
	{
		for (long int i = 0; i < N; i++)
			if (seg_chain[i] >= 0 && best[i] >= 0)
				fil_cnt[filament_of_chain[seg_chain[i]] * K + best[i]] += 1., cls_tot[best[i]] += 1.;
	}

	if (!used_chain_mix)
	{
		// Fallback KL-NMF for degenerate/uninitialized inputs
		std::vector<double> W(F * T), H(T * K), R(F * K);
		std::mt19937 rng(42);
		std::uniform_real_distribution<double> U(0.5, 1.5);
		for (auto &w : W) w = U(rng);
		for (auto &h : H) h = U(rng);
		for (int it = 0; it < 200; it++)
		{
			for (int f = 0; f < F; f++)
				for (int k = 0; k < K; k++)
				{
					double s = 0.;
					for (int t = 0; t < T; t++) s += W[f * T + t] * H[t * K + k];
					R[f * K + k] = (s > 0.) ? fil_cnt[f * K + k] / s : 0.;
				}
			for (int t = 0; t < T; t++)
			{
				double wsum = 0.;
				for (int f = 0; f < F; f++) wsum += W[f * T + t];
				for (int k = 0; k < K; k++)
				{
					double s = 0.;
					for (int f = 0; f < F; f++) s += W[f * T + t] * R[f * K + k];
					H[t * K + k] *= (wsum > 0.) ? s / wsum : 0.;
				}
			}
			for (int f = 0; f < F; f++)
				for (int k = 0; k < K; k++)
				{
					double s = 0.;
					for (int t = 0; t < T; t++) s += W[f * T + t] * H[t * K + k];
					R[f * K + k] = (s > 0.) ? fil_cnt[f * K + k] / s : 0.;
				}
			for (int t = 0; t < T; t++)
			{
				double hsum = 0.;
				for (int k = 0; k < K; k++) hsum += H[t * K + k];
				for (int f = 0; f < F; f++)
				{
					double s = 0.;
					for (int k = 0; k < K; k++) s += H[t * K + k] * R[f * K + k];
					W[f * T + t] *= (hsum > 0.) ? s / hsum : 0.;
				}
			}
		}
		for (int f = 0; f < F; f++)
		{
			for (int t = 0; t < T; t++)
			{
				double h = 0.;
				for (int k = 0; k < K; k++) h += H[t * K + k];
				Wn[f * T + t] = W[f * T + t] * h + 1e-9;
			}
			normaliseSum(&Wn[f * T], T);
		}
	}

	// Align newly fitted filament type indices to the previous tpost to avoid arbitrary label permutations across iterations
	if (T > 1 && (long int)tpost.size() == N * (long int)T)
	{
		std::vector<double> overlap((size_t)T * T, 0.);
		double tot_prev = 0.;
		for (long int i = 0; i < N; i++)
		{
			if (seg_chain[i] < 0) continue;
			const long int f = filament_of_chain[seg_chain[i]];
			if (f < 0 || f >= F) continue;
			for (int t_old = 0; t_old < T; t_old++)
			{
				const double tp = (double)tpost[(size_t)i * T + t_old];
				tot_prev += tp;
				for (int t_new = 0; t_new < T; t_new++)
					overlap[(size_t)t_old * T + t_new] += tp * Wn[(size_t)f * T + t_new];
			}
		}
		if (tot_prev > 1.0)
		{
			std::vector<int> perm(T);
			for (int t = 0; t < T; t++) perm[t] = t;
			if (T <= 7)
			{
				std::vector<int> p = perm;
				double best_score = -1.;
				do
				{
					double sc = 0.;
					for (int t_old = 0; t_old < T; t_old++)
						sc += overlap[(size_t)t_old * T + p[t_old]];
					if (sc > best_score) { best_score = sc; perm = p; }
				} while (std::next_permutation(p.begin(), p.end()));
			}
			else
			{
				std::vector<char> used_old(T, 0), used_new(T, 0);
				for (int st_idx = 0; st_idx < T; st_idx++)
				{
					int bo = -1, bn = -1;
					double bval = -1.;
					for (int t_old = 0; t_old < T; t_old++)
					{
						if (used_old[t_old]) continue;
						for (int t_new = 0; t_new < T; t_new++)
							if (!used_new[t_new] && overlap[(size_t)t_old * T + t_new] > bval)
							{
								bval = overlap[(size_t)t_old * T + t_new];
								bo = t_old; bn = t_new;
							}
					}
					if (bo >= 0 && bn >= 0) { perm[bo] = bn; used_old[bo] = 1; used_new[bn] = 1; }
				}
			}
			std::vector<double> tmp_row(T);
			for (int f = 0; f < F; f++)
			{
				for (int t_old = 0; t_old < T; t_old++)
					tmp_row[t_old] = Wn[(size_t)f * T + perm[t_old]];
				for (int t_old = 0; t_old < T; t_old++)
					Wn[(size_t)f * T + t_old] = tmp_row[t_old];
			}
		}
	}

	fil_chain_post = Wn;
	std::vector<double> Aff(K * T, 0.);
	std::vector<double> type_fil_mass(T, 0.);
	for (int f = 0; f < F; f++)
		for (int t = 0; t < T; t++)
		{
			type_fil_mass[t] += Wn[f * T + t];
			for (int k = 0; k < K; k++)
				Aff[k * T + t] += Wn[f * T + t] * fil_cnt[f * K + k];
		}

	// Hybrid exclusive/shared class-to-type allowance:
	//   - Split chimeric classes stay locked to split_lock_owner while active.
	//   - Unambiguous views (>= 68% affinity in one type) stay exclusive; shared crossovers (>= 22% in multiple types) allow those types.
	class_owner.assign(K, -1);
	class_allowed.assign((size_t)K * T, 0);
	int n_good_cls = 0;
	const double min_own_cnt = (N >= 2000) ? 5.0 : 0.0;
	for (int k = 0; k < K; k++)
	{
		if (is_junk[k] || cls_tot[k] <= 0. || cls_tot[k] < min_own_cnt) continue;
		n_good_cls++;
		int bt = 0;
		double aff_sum = 0.;
		for (int t = 0; t < T; t++)
		{
			aff_sum += Aff[k * T + t];
			if (Aff[k * T + t] > Aff[k * T + bt]) bt = t;
		}
		if (k < (int)split_lock_owner.size() && split_lock_owner[k] >= 0 && split_lock_owner[k] < T)
		{
			bt = split_lock_owner[k];
			class_owner[k] = bt;
			class_allowed[(size_t)k * T + bt] = 1;
		}
		else
		{
			class_owner[k] = bt;
			class_allowed[(size_t)k * T + bt] = 1;
			if (aff_sum > 0. && Aff[k * T + bt] < 0.80 * aff_sum)
			{
				for (int t = 0; t < T; t++)
					if (Aff[k * T + t] >= 0.12 * aff_sum)
						class_allowed[(size_t)k * T + t] = 1;
			}
		}
	}
	// Guarantee each active type owns and allows at least min_own classes
	const int min_own = (T > 0 && n_good_cls >= 2 * T) ? 2 : 1;
	for (int round = 0; round < min_own && T > 1; round++)
	{
		std::vector<int> owned(T, 0);
		for (int k = 0; k < K; k++)
			if (class_owner[k] >= 0) owned[class_owner[k]]++;
		for (int t = 0; t < T; t++)
		{
			if (type_fil_mass[t] < 1.0 || owned[t] > round) continue;
			int best_k = -1;
			double best_score = -1.;
			for (int k = 0; k < K; k++)
			{
				const int cur = class_owner[k];
				if (cur < 0 || cur == t || owned[cur] <= round + 1) continue;
				if (k < (int)split_lock_owner.size() && split_lock_owner[k] >= 0 && split_lock_owner[k] < T) continue;
				const double sc = Aff[k * T + t] / std::max(1e-9, Aff[k * T + cur]);
				if (sc > best_score) { best_score = sc; best_k = k; }
			}
			if (best_k >= 0)
			{
				owned[class_owner[best_k]]--;
				owned[t]++;
				class_owner[best_k] = t;
				class_allowed[(size_t)best_k * T + t] = 1;
			}
		}
	}
	// Guarantee each active type keeps non-shared exclusive anchor classes
	if (T > 1 && n_good_cls >= T)
	{
		const int min_excl = std::max(1, std::min(2, n_good_cls / (3 * T)));
		for (int t = 0; t < T; t++)
		{
			if (type_fil_mass[t] < 1.0) continue;
			int n_excl = 0;
			for (int k = 0; k < K; k++)
			{
				if (class_owner[k] != t) continue;
				int n_allow = 0;
				for (int u = 0; u < T; u++) n_allow += (int)class_allowed[(size_t)k * T + u];
				if (n_allow == 1) n_excl++;
			}
			while (n_excl < min_excl)
			{
				int best_k = -1;
				double best_frac = -1.;
				for (int k = 0; k < K; k++)
				{
					if (class_owner[k] != t) continue;
					if (k < (int)split_lock_owner.size() && split_lock_owner[k] >= 0 && split_lock_owner[k] < T) continue;
					int n_allow = 0;
					for (int u = 0; u < T; u++) n_allow += (int)class_allowed[(size_t)k * T + u];
					if (n_allow <= 1) continue;
					double aff_sum = 0.;
					for (int u = 0; u < T; u++) aff_sum += Aff[k * T + u];
					const double frac = (aff_sum > 0.) ? Aff[k * T + t] / aff_sum : 0.;
					if (frac > best_frac) { best_frac = frac; best_k = k; }
				}
				if (best_k < 0) break;
				for (int u = 0; u < T; u++)
					class_allowed[(size_t)best_k * T + u] = (u == t) ? 1 : 0;
				n_excl++;
			}
		}
	}

	// Stationary q_t & sparse transition matrices A_t restricted to class_allowed[k, t]
	q.assign(T * K, 0.);
	for (int t = 0; t < T; t++)
	{
		double s = 0.;
		for (int k = 0; k < K; k++)
			if (class_allowed[(size_t)k * T + t])
			{
				q[t * K + k] = Aff[k * T + t] + 0.05;
				s += q[t * K + k];
			}
		if (s > 0.)
			for (int k = 0; k < K; k++) q[t * K + k] /= s;
		else
		{
			for (int k = 0; k < K; k++) q[t * K + k] = is_junk[k] ? 0. : 1.;
			normaliseSum(&q[t * K], K);
		}
	}

	std::vector<double> cnt(T * K * K, 0.), mass(T, 0.);
	for (size_t c = 0; c < chains.size(); c++)
	{
		if (!chain_test.empty() && chain_test[c]) continue;
		const long int f = filament_of_chain[c];
		const std::vector<long int> &g = chains[c];
		for (size_t s = 0; s < g.size(); s++)
		{
			const int ks = (g[s] >= 0) ? best[g[s]] : -1;
			if (ks < 0 || is_junk[ks]) continue;
			const int kn = (s + 1 < g.size() && g[s + 1] >= 0) ? best[g[s + 1]] : -1;
			for (int t = 0; t < T; t++)
			{
				if (!class_allowed[(size_t)ks * T + t]) continue;
				mass[t] += Wn[f * T + t];
				if (kn >= 0 && !is_junk[kn] && class_allowed[(size_t)kn * T + t])
					cnt[(t * K + ks) * K + kn] += Wn[f * T + t];
			}
		}
	}
	A.assign(T * K * K, 0.);
	for (int t = 0; t < T; t++)
		for (int j = 0; j < K; j++)
		{
			double *a = &A[(t * K + j) * K];
			if (!class_allowed[(size_t)j * T + t])
				continue;
			for (int k = 0; k < K; k++)
				if (class_allowed[(size_t)k * T + t])
					a[k] = cnt[(t * K + j) * K + k] + ((j == k) ? 0.5 : 0.0) + 0.02 * q[t * K + k];
			normaliseSum(a, K);
		}
	for (int t = 0; t < T; t++)
		if (mass[t] <= 0.) mass[t] = type_fil_mass[t] + 1e-6;
	normaliseSum(&mass[0], T);
	pi0.assign(T * K, 0.);
	for (int t = 0; t < T; t++)
		for (int k = 0; k < K; k++)
			pi0[t * K + k] = mass[t] * q[t * K + k];
	rho = (T > 1) ? 1e-6 : 0.;
	initialised_model = true;
	if (verb > 0)
	{
		std::cout << " Filament context: initial type fractions (" << (used_chain_mix ? "Markov chain" : "NMF") << "):";
		for (int t = 0; t < T; t++) std::cout << " " << mass[t];
		std::cout << std::endl;
	}
}

void HelicalContext::update()
{
	const int T = nr_types, TK = T * K;
	const int nch = chains.size();
	const int F = (int)filament_name.size();

	std::vector<long int> seg_global_pos(N, -1);
	if ((int)pol_members.size() == F && (int)pol_pos.size() == F)
	{
		for (int f = 0; f < F; f++)
			for (size_t j = 0; j < pol_members[f].size() && j < pol_pos[f].size(); j++)
			{
				const long int i = pol_members[f][j];
				if (i >= 0 && i < N) seg_global_pos[i] = pol_pos[f][j];
			}
	}

	std::vector<std::vector<int> > fil_chains(F);
	for (int c = 0; c < nch; c++)
	{
		const long int f = (c < (int)filament_of_chain.size()) ? filament_of_chain[c] : -1;
		if (f >= 0 && f < F && !chains[c].empty())
			fil_chains[f].push_back(c);
	}
	if ((int)chain_reversed.size() != nch) chain_reversed.assign(nch, false);

	// polarity order: all chain fragments of a flipped-polarity filament run backwards in reversed fragment order
	if (use_polarity_order)
	{
		for (int f = 0; f < F; f++)
		{
			if (fil_chains[f].empty()) continue;
			double sum = 0.;
			long int n = 0;
			if (f < (int)pol_members.size() && !pol_members[f].empty())
			{
				for (long int i : pol_members[f])
					if (i >= 0 && i < N && (long int)pol_post.size() == N)
						sum += pol_post[i], n++;
			}
			else
			{
				for (int c : fil_chains[f])
					for (long int i : chains[c])
						if (i >= 0 && i < N && (long int)pol_post.size() == N)
							sum += pol_post[i], n++;
			}
			const bool rev = (n > 0 && sum / n > 0.5);
			for (int c : fil_chains[f])
				if (rev != chain_reversed[c])
				{
					std::reverse(chains[c].begin(), chains[c].end());
					chain_reversed[c] = rev;
				}
			if (rev)
				std::reverse(fil_chains[f].begin(), fil_chains[f].end());
		}
	}
	else
	{
		for (int f = 0; f < F; f++)
			if (!fil_chains[f].empty() && chain_reversed[fil_chains[f][0]])
				std::reverse(fil_chains[f].begin(), fil_chains[f].end());
	}

	if (tpost.size() != (size_t)(N * T))
		tpost.assign(N * T, 0.f);
	// Segments whose most likely class is junk say nothing about the type (missing data in the chain) and get no type
	std::vector<char> seg_junk(N, 0);
	if (!junk_class.empty())
		for (long int i = 0; i < N; i++)
		{
			if (std::isnan(loglik[i * K])) continue;
			int best = 0;
			for (int k = 1; k < K; k++)
				if (loglik[i * K + k] > loglik[i * K + best]) best = k;
			seg_junk[i] = junk_class[best];
		}
	if (!initialised_model)
		initialiseModel();

	auto gapStaySwitch = [&](long int gap_steps, double &stay_g, double &sw_g)
	{
		if (T <= 1) { stay_g = 1.; sw_g = 0.; return; }
		const double r_eff = rho * (double)T / (double)(T - 1);
		stay_g = (1.0 / (double)T) + (1.0 - 1.0 / (double)T) * std::pow(std::max(0.0, 1.0 - r_eff), (double)std::max(1L, gap_steps));
		sw_g = (1.0 - stay_g) / (double)(T - 1);
	};

	std::vector<double> Aw;
	for (int bw = 0; bw <= nr_bw_iter; bw++)
	{
		const bool last = (bw == nr_bw_iter);
		std::vector<std::vector<double> > Awt;
		double stayw = 1., switchw = 0.;
		if (last)
		{
			Awt.resize(T);
			for (int t = 0; t < T; t++)
			{
				std::vector<double> At(A.begin() + t * K * K, A.begin() + (t + 1) * K * K);
				matPower(At, exclude_steps, Awt[t]);
			}
			stayw = pow(1. - rho, exclude_steps);
			switchw = (T > 1) ? (1. - stayw) / (T - 1) : 0.;
		}
		const double sw = (T > 1) ? rho / (T - 1) : 0.;

		int nthr = 1;
#ifdef _OPENMP
		nthr = omp_get_max_threads();
#endif
		std::vector<std::vector<double> > xi_t(nthr), g_t(nthr, std::vector<double>(TK, 0.));
		std::vector<double> nsw_t(nthr, 0.), ntr_t(nthr, 0.), ll_t(nthr, 0.), lltest_t(nthr, 0.);
		if (!last)
			for (auto &x : xi_t) x.assign(T * K * K, 0.);

		#pragma omp parallel for schedule(dynamic, 2)
		for (int f = 0; f < F; f++)
		{
			const std::vector<int> &fc = fil_chains[f];
			const int M = (int)fc.size();
			if (M == 0) continue;

			int tid = 0;
#ifdef _OPENMP
			tid = omp_get_thread_num();
#endif
			bool held_out = false;
			if (!chain_test.empty())
				for (int c : fc)
					if (chain_test[c]) { held_out = true; break; }
			double &ll_c = held_out ? lltest_t[tid] : ll_t[tid];

			std::vector<long int> gaps(M, 1);
			for (int m = 1; m < M; m++)
			{
				const std::vector<long int> &gp = chains[fc[m - 1]];
				const std::vector<long int> &gc = chains[fc[m]];
				long int p_end = (long int)gp.size() - 1;
				while (p_end >= 0 && gp[p_end] < 0) p_end--;
				long int p_start = 0;
				while (p_start < (long int)gc.size() && gc[p_start] < 0) p_start++;
				long int g = std::max(1L, (long int)std::ceil((double)max_gap_steps) + 1L);
				if (p_end >= 0 && p_start < (long int)gc.size())
				{
					const long int i_end = gp[p_end], i_start = gc[p_start];
					if (seg_global_pos[i_end] >= 0 && seg_global_pos[i_start] >= 0)
					{
						const long int pad = ((long int)gp.size() - 1 - p_end) + p_start;
						g = std::max(1L, std::labs(seg_global_pos[i_start] - seg_global_pos[i_end]) - pad);
					}
				}
				gaps[m] = g;
			}

			std::vector<std::vector<double> > e_f(M), al_f(M), be_f(M), in_prior_f(M);
			std::vector<double> v(TK), tmp(K), S(T), Q(T), left(TK), right(TK);

			// Forward across all fragments m = 0 .. M-1 of filament f
			for (int m = 0; m < M; m++)
			{
				const std::vector<long int> &grid = chains[fc[m]];
				const long int L = grid.size();
				e_f[m].resize(L * K);
				al_f[m].resize(L * TK);
				be_f[m].resize(L * TK);
				in_prior_f[m].resize(TK);
				for (long int s = 0; s < L; s++)
				{
					const long int i = grid[s];
					for (int k = 0; k < K; k++)
					{
						const float l = (i >= 0 && !seg_junk[i]) ? loglik[i * K + k] : std::numeric_limits<float>::quiet_NaN();
						e_f[m][s * K + k] = std::isnan(l) ? 1. : exp(temper * l);
					}
				}
				if (m == 0)
				{
					for (int tk = 0; tk < TK; tk++) in_prior_f[0][tk] = pi0[tk];
				}
				else
				{
					const long int Lp = chains[fc[m - 1]].size();
					const double *ap_end = &al_f[m - 1][(Lp - 1) * TK];
					double Stot = 0.;
					for (int t = 0; t < T; t++)
					{
						S[t] = 0.;
						for (int k = 0; k < K; k++) S[t] += ap_end[t * K + k];
						Stot += S[t];
					}
					double stay_g = 1., sw_g = 0.;
					gapStaySwitch(gaps[m], stay_g, sw_g);
					for (int t = 0; t < T; t++)
					{
						const double ft = stay_g * S[t] + sw_g * (Stot - S[t]);
						for (int k = 0; k < K; k++)
							in_prior_f[m][t * K + k] = ft * q[t * K + k];
					}
				}
				double *al = &al_f[m][0];
				for (int t = 0; t < T; t++)
					for (int k = 0; k < K; k++)
						al[t * K + k] = in_prior_f[m][t * K + k] * e_f[m][k];
				ll_c += log(normaliseSum(al, TK));
				for (long int s = 1; s < L; s++)
				{
					const double *ap = &al[(s - 1) * TK];
					double Stot = 0.;
					for (int t = 0; t < T; t++)
					{
						S[t] = 0.;
						for (int k = 0; k < K; k++) S[t] += ap[t * K + k];
						Stot += S[t];
					}
					double *an = &al[s * TK];
					for (int t = 0; t < T; t++)
					{
						vecMat(ap + t * K, &A[t * K * K], &tmp[0], K);
						const double sin = sw * (Stot - S[t]);
						for (int k = 0; k < K; k++)
							an[t * K + k] = e_f[m][s * K + k] * ((1. - rho) * tmp[k] + sin * q[t * K + k]);
					}
					ll_c += log(normaliseSum(an, TK));
				}
			}

			// Backward across all fragments m = M-1 .. 0 of filament f
			for (int m = M - 1; m >= 0; m--)
			{
				const long int L = chains[fc[m]].size();
				double *al = &al_f[m][0];
				double *be = &be_f[m][0];
				if (m == M - 1)
				{
					for (int tk = 0; tk < TK; tk++)
						be[(L - 1) * TK + tk] = 1.;
				}
				else
				{
					double stay_g = 1., sw_g = 0.;
					gapStaySwitch(gaps[m + 1], stay_g, sw_g);
					double Qtot = 0.;
					for (int t = 0; t < T; t++)
					{
						Q[t] = 0.;
						for (int k = 0; k < K; k++)
						{
							v[t * K + k] = e_f[m + 1][k] * be_f[m + 1][t * K + k];
							Q[t] += q[t * K + k] * v[t * K + k];
						}
						Qtot += Q[t];
					}
					double *bn_end = &be[(L - 1) * TK];
					for (int t = 0; t < T; t++)
					{
						const double bt = stay_g * Q[t] + sw_g * (Qtot - Q[t]);
						for (int k = 0; k < K; k++)
							bn_end[t * K + k] = ((int)class_allowed.size() != K * T || class_allowed[(size_t)k * T + t]) ? bt : 0.;
					}
					if (!last && !held_out)
					{
						const double *as_end = &al[(L - 1) * TK];
						double Z = 0.;
						for (int tk = 0; tk < TK; tk++) Z += as_end[tk] * bn_end[tk];
						if (Z > 0.)
						{
							double cross = 0.;
							for (int t = 0; t < T; t++)
							{
								double St = 0.;
								for (int k = 0; k < K; k++)
									if ((int)class_allowed.size() != K * T || class_allowed[(size_t)k * T + t])
										St += as_end[t * K + k];
								cross += St * sw_g * (Qtot - Q[t]);
							}
							nsw_t[tid] += cross / Z;
							ntr_t[tid] += (double)gaps[m + 1];
						}
					}
					normaliseSum(bn_end, TK);
				}

				for (long int s = L - 2; s >= 0; s--)
				{
					double Qtot = 0.;
					for (int t = 0; t < T; t++)
					{
						Q[t] = 0.;
						for (int k = 0; k < K; k++)
						{
							v[t * K + k] = e_f[m][(s + 1) * K + k] * be[(s + 1) * TK + t * K + k];
							Q[t] += q[t * K + k] * v[t * K + k];
						}
						Qtot += Q[t];
					}
					double *bn = &be[s * TK];
					for (int t = 0; t < T; t++)
					{
						matVec(&A[t * K * K], &v[t * K], &tmp[0], K);
						const double sout = sw * (Qtot - Q[t]);
						for (int k = 0; k < K; k++)
							bn[t * K + k] = ((int)class_allowed.size() != K * T || class_allowed[(size_t)k * T + t]) ? ((1. - rho) * tmp[k] + sout) : 0.;
					}
					if (!last && !held_out)
					{
						const double *as = &al[s * TK];
						double Z = 0.;
						for (int tk = 0; tk < TK; tk++) Z += as[tk] * bn[tk];
						if (Z > 0.)
						{
							for (int t = 0; t < T; t++)
							{
								double St = 0.;
								for (int j = 0; j < K; j++)
								{
									const double a = as[t * K + j] * (1. - rho) / Z;
									St += as[t * K + j];
									if (a == 0.) continue;
									const double *Ar = &A[(t * K + j) * K];
									double *xr = &xi_t[tid][(t * K + j) * K];
									for (int k = 0; k < K; k++)
										xr[k] += a * Ar[k] * v[t * K + k];
								}
								nsw_t[tid] += St * sw * (Qtot - Q[t]) / Z;
							}
							ntr_t[tid] += 1.;
						}
					}
					normaliseSum(bn, TK);
				}
			}

			// Posteriors & cavity priors across all fragments m = 0 .. M-1
			for (int m = 0; m < M; m++)
			{
				const std::vector<long int> &grid = chains[fc[m]];
				const long int L = grid.size();
				const double *al = &al_f[m][0];
				const double *be = &be_f[m][0];
				for (long int s = 0; s < L; s++)
				{
					for (int tk = 0; tk < TK; tk++)
						v[tk] = al[s * TK + tk] * be[s * TK + tk];
					normaliseSum(&v[0], TK);
					if (!held_out)
						for (int tk = 0; tk < TK; tk++)
							g_t[tid][tk] += v[tk];
					const long int i = grid[s];
					if (!last || i < 0) continue;
					for (int k = 0; k < K; k++)
					{
						double p = 0.;
						for (int t = 0; t < T; t++) p += v[t * K + k];
						post[i * K + k] = p;
					}
					for (int t = 0; t < T; t++)
					{
						double p = 0.;
						for (int k = 0; k < K; k++) p += v[t * K + k];
						tpost[i * T + t] = p;
					}
				}
				if (last)
				{
					const long int w = exclude_steps;
					for (long int s = 0; s < L; s++)
					{
						const long int i = grid[s];
						if (i < 0) continue;
						if (s - w >= 0)
						{
							const double *ap = &al[(s - w) * TK];
							double Stot = 0.;
							for (int t = 0; t < T; t++)
							{
								S[t] = 0.;
								for (int k = 0; k < K; k++) S[t] += ap[t * K + k];
								Stot += S[t];
							}
							for (int t = 0; t < T; t++)
							{
								vecMat(ap + t * K, &Awt[t][0], &tmp[0], K);
								for (int k = 0; k < K; k++)
									left[t * K + k] = stayw * tmp[k] + switchw * (Stot - S[t]) * q[t * K + k];
							}
						}
						else
							for (int tk = 0; tk < TK; tk++) left[tk] = in_prior_f[m][tk];
						if (s + w < L)
						{
							double Qtot = 0.;
							for (int t = 0; t < T; t++)
							{
								Q[t] = 0.;
								for (int k = 0; k < K; k++)
								{
									v[t * K + k] = e_f[m][(s + w) * K + k] * be[(s + w) * TK + t * K + k];
									Q[t] += q[t * K + k] * v[t * K + k];
								}
								Qtot += Q[t];
							}
							for (int t = 0; t < T; t++)
							{
								matVec(&Awt[t][0], &v[t * K], &tmp[0], K);
								for (int k = 0; k < K; k++)
									right[t * K + k] = ((int)class_allowed.size() != K * T || class_allowed[(size_t)k * T + t]) ? (stayw * tmp[k] + switchw * (Qtot - Q[t])) : 0.;
							}
						}
						else
							for (int tk = 0; tk < TK; tk++) right[tk] = be[(L - 1) * TK + tk];
						double mx = 0.;
						for (int k = 0; k < K; k++)
						{
							double p = 0.;
							for (int t = 0; t < T; t++) p += left[t * K + k] * right[t * K + k];
							tmp[k] = p;
							mx = std::max(mx, p);
						}
						double mx_good = 0.;
						for (int k = 0; k < K; k++)
							if (junk_class.empty() || !junk_class[k])
								mx_good = std::max(mx_good, tmp[k]);
						for (int k = 0; k < K; k++)
							prior[i * K + k] = (mx_good > 0. && (junk_class.empty() || !junk_class[k])) ? (float)std::max(prior_floor, tmp[k] / mx_good) : 1.f;
					}
				}
			}
		}

		std::vector<double> gsum(TK, 0.);
		double nsw = 0., ntr = 0.;
		loglik_total = 0.;
		loglik_test = 0.;
		for (int th = 0; th < nthr; th++)
		{
			loglik_test += lltest_t[th];
			for (int tk = 0; tk < TK; tk++) gsum[tk] += g_t[th][tk];
			nsw += nsw_t[th];
			ntr += ntr_t[th];
			loglik_total += ll_t[th];
		}
		if (verb > 1)
			std::cout << "  Baum-Welch " << bw << ": log-likelihood " << loglik_total << ", rho " << rho << std::endl;
		if (last)
			break;

		// M-step: keep class ownership block-diagonal and transitions sparse (no uniform teleportation flattening)
		rho = (T > 1 && ntr > 0.) ? std::min(1e-4, std::max(1e-7, nsw / ntr)) : 0.;
		for (int t = 0; t < T; t++)
		{
			double qs = 0.;
			for (int k = 0; k < K; k++)
				if ((int)class_allowed.size() != K * T || class_allowed[(size_t)k * T + t])
				{
					q[t * K + k] = gsum[t * K + k] + 1e-4;
					qs += q[t * K + k];
				}
				else
					q[t * K + k] = 0.;
			if (qs > 0.)
				for (int k = 0; k < K; k++) q[t * K + k] /= qs;
			else
			{
				for (int k = 0; k < K; k++) q[t * K + k] = (junk_class.empty() || !junk_class[k]) ? 1. : 0.;
				normaliseSum(&q[t * K], K);
			}
		}
		std::vector<double> xi(T * K * K, 0.);
		for (int th = 0; th < nthr; th++)
			for (size_t n = 0; n < xi.size(); n++) xi[n] += xi_t[th][n];
		for (int t = 0; t < T; t++)
			for (int j = 0; j < K; j++)
			{
				double *a = &A[(t * K + j) * K];
				if ((int)class_allowed.size() == K * T && !class_allowed[(size_t)j * T + t])
				{
					std::fill(a, a + K, 0.);
					continue;
				}
				for (int k = 0; k < K; k++)
					if ((int)class_allowed.size() != K * T || class_allowed[(size_t)k * T + t])
						a[k] = xi[(t * K + j) * K + k] + ((j == k) ? 0.5 : 0.0) + 0.02 * q[t * K + k];
					else
						a[k] = 0.;
				normaliseSum(a, K);
			}
		for (int t = 0; t < T; t++)
			for (int k = 0; k < K; k++)
				pi0[t * K + k] = ((int)class_allowed.size() != K * T || class_allowed[(size_t)k * T + t]) ? (gsum[t * K + k] + 1e-4) : 0.;
		normaliseSum(&pi0[0], TK);
	}
	have_prior = true;

	// types and sections
	seg_type.assign(N, -1);
	seg_type_prob.assign(N, 0.f);
	type_mass.assign(T, 0.);
	for (long int i = 0; i < N; i++)
	{
		if (seg_chain[i] < 0 || seg_junk[i]) continue;
		int b = 0;
		for (int t = 0; t < T; t++)
		{
			type_mass[t] += tpost[i * T + t];
			if (tpost[i * T + t] > tpost[i * T + b]) b = t;
		}
		seg_type[i] = b;
		seg_type_prob[i] = tpost[i * T + b];
	}
	// Refresh the per-filament type posteriors from the converged Baum-Welch posteriors, so that their type
	// columns are those of tpost (the initial mixture's columns are permuted by the Baum-Welch alignment)
	{
		fil_chain_post.assign((size_t)F * T, 0.);
		for (long int i = 0; i < N; i++)
		{
			if (seg_chain[i] < 0 || seg_junk[i]) continue;
			const long int f = filament_of_chain[seg_chain[i]];
			if (f < 0 || f >= F) continue;
			for (int t = 0; t < T; t++)
				fil_chain_post[(size_t)f * T + t] += tpost[i * T + t];
		}
		for (int f = 0; f < F; f++)
		{
			double s = 0.;
			for (int t = 0; t < T; t++) s += fil_chain_post[(size_t)f * T + t];
			if (s > 0.)
				for (int t = 0; t < T; t++) fil_chain_post[(size_t)f * T + t] /= s;
		}
	}
	makeSections();
	computeFilamentPosteriors(N, K, T, F, seg_chain, filament_of_chain,
	                          junk_class, loglik, tpost, seg_type, fil_chain_post, fil_type_posterior);
	type_period.assign(T, 0.);
	type_period_R.assign(T, 0.);
}

void HelicalContext::makeSections()
{
	// runs of the same type along each filament (across all its chain fragments);
	// runs of fewer than 3 segments between two runs of the same type are taken into those
	const int T = nr_types;
	const int F = (int)filament_name.size();
	seg_section.assign(N, -1);
	type_nr_filaments.assign(T, 0);
	type_nr_sections.assign(T, 0);
	type_section_angst.assign(T, 0.);
	std::vector<long int> seg_global_pos(N, -1);
	if ((int)pol_members.size() == F && (int)pol_pos.size() == F)
	{
		for (int f = 0; f < F; f++)
			for (size_t j = 0; j < pol_members[f].size() && j < pol_pos[f].size(); j++)
			{
				const long int i = pol_members[f][j];
				if (i >= 0 && i < N) seg_global_pos[i] = pol_pos[f][j];
			}
	}
	std::vector<std::vector<int> > fil_chains(F);
	for (size_t c = 0; c < chains.size(); c++)
	{
		const long int f = (c < filament_of_chain.size()) ? filament_of_chain[c] : -1;
		if (f >= 0 && f < F && !chains[c].empty())
			fil_chains[f].push_back((int)c);
	}
	std::vector<std::vector<bool> > fil_has(F, std::vector<bool>(T, false));
	int section = 0;
	for (int f = 0; f < F; f++)
	{
		std::vector<long int> segs;
		const bool rev = !fil_chains[f].empty() && fil_chains[f][0] < (int)chain_reversed.size() && chain_reversed[fil_chains[f][0]];
		if (rev)
		{
			for (int idx = (int)fil_chains[f].size() - 1; idx >= 0; idx--)
				for (long int i : chains[fil_chains[f][idx]])
					if (i >= 0 && seg_type[i] >= 0) segs.push_back(i);
		}
		else
		{
			for (int c : fil_chains[f])
				for (long int i : chains[c])
					if (i >= 0 && seg_type[i] >= 0) segs.push_back(i);
		}
		if (segs.empty()) continue;
		std::vector<std::pair<int, int> > runs; // (type, length)
		for (long int i : segs)
			if (runs.empty() || runs.back().first != seg_type[i])
				runs.push_back(std::make_pair(seg_type[i], 1));
			else
				runs.back().second++;
		bool changed = true;
		while (changed)
		{
			changed = false;
			for (size_t r = 1; r + 1 < runs.size(); r++)
				if (runs[r].second < 3 && runs[r-1].first == runs[r+1].first)
				{
					runs[r-1].second += runs[r].second + runs[r+1].second;
					runs.erase(runs.begin() + r, runs.begin() + r + 2);
					changed = true;
					break;
				}
		}
		size_t j = 0;
		for (auto &r : runs)
		{
			const long int i_first = segs[j];
			for (int n = 0; n < r.second; n++, j++)
			{
				seg_section[segs[j]] = section;
				seg_type[segs[j]] = r.first;
			}
			const long int i_last = segs[j - 1];
			long int span_steps = r.second;
			if (seg_global_pos[i_first] >= 0 && seg_global_pos[i_last] >= 0)
				span_steps = std::labs(seg_global_pos[i_last] - seg_global_pos[i_first]) + 1;
			else if (seg_chain[i_first] >= 0 && seg_chain[i_first] == seg_chain[i_last])
				span_steps = std::labs(seg_pos[i_last] - seg_pos[i_first]) + 1;
			type_nr_sections[r.first]++;
			type_section_angst[r.first] += span_steps * step;
			fil_has[f][r.first] = true;
			section++;
		}
	}
	for (int t = 0; t < T; t++)
		if (type_nr_sections[t] > 0)
			type_section_angst[t] /= type_nr_sections[t];
	for (auto &fh : fil_has)
		for (int t = 0; t < T; t++)
			if (fh[t]) type_nr_filaments[t]++;
}

void HelicalContext::writeStar(const FileName &fn) const
{
	std::ofstream out(fn.c_str());
	if (!out)
		REPORT_ERROR("HelicalContext: cannot write " + fn);
	out << std::setprecision(6);
	out << "\n# Filament context (helical 2D classification)\n\n";
	out << "data_helical_context_general\n\n";
	out << "_rlnHelicalContextStepAngst " << step << "\n";
	out << "_rlnHelicalContextTemper " << temper << "\n";
	out << "_rlnHelicalContextExcludeAngst " << exclude_steps * step << "\n";
	out << "_rlnHelicalContextNrTypes " << nr_types << "\n";
	out << "_rlnHelicalContextSwitchPerAngst " << rho / step << "\n";
	out << "_rlnHelicalContextLogLikelihood " << loglik_total << "\n\n";

	out << "data_helical_context_types\n\nloop_\n";
	out << "_rlnHelicalType #1\n_rlnHelicalTypeNrSegments #2\n_rlnHelicalTypeNrFilaments #3\n"
	       "_rlnHelicalTypeNrSections #4\n_rlnHelicalTypeMeanSectionAngst #5\n_rlnHelicalTypePeriodAngst #6\n"
	       "_rlnHelicalTypePeriodScore #7\n_rlnHelicalTypeMainClasses #8\n";
	for (int t = 0; t < nr_types; t++)
	{
		// classes making up 90% of the type
		std::vector<int> o(K);
		for (int k = 0; k < K; k++) o[k] = k;
		std::sort(o.begin(), o.end(), [&](int a, int b) { return q[t * K + a] > q[t * K + b]; });
		std::string cls;
		double cum = 0.;
		for (int k : o)
		{
			if (cum >= 0.9) break;
			cum += q[t * K + k];
			cls += (cls.empty() ? "" : ",") + integerToString(k + 1);
		}
		out << std::setw(4) << t + 1 << " " << std::setw(10) << ROUND(type_mass[t]) << " " << std::setw(6) << type_nr_filaments[t]
		    << " " << std::setw(6) << type_nr_sections[t] << " " << std::setw(8) << type_section_angst[t]
		    << " " << std::setw(8) << type_period[t] << " " << std::setw(8) << type_period_R[t] << " " << cls << "\n";
	}

	out << "\ndata_helical_context_filaments\n\nloop_\n";
	out << "_rlnMicrographName #1\n_rlnHelicalTubeID #2\n_rlnHelicalFilamentNrSegments #3\n"
	       "_rlnHelicalFilamentNrSections #4\n_rlnHelicalFilamentMeanTypeProbability #5\n_rlnHelicalFilamentSections #6\n"
	       "_rlnHelicalCrossoverPeriod #7\n_rlnHelicalCrossoverFigureOfMerit #8\n";
	std::vector<std::vector<long int> > segs(filament_name.size());
	for (size_t c = 0; c < chains.size(); c++)
		for (long int i : chains[c])
			if (i >= 0)
				segs[filament_of_chain[c]].push_back(i);
	for (size_t f = 0; f < filament_name.size(); f++)
	{
		const std::string &name = filament_name[f];
		size_t colon = name.rfind(':');
		std::string sec;
		int nsec = 0, last_sec = -2, run = 0, run_type = -1;
		double ps = 0.;
		int np = 0;
		for (long int i : segs[f])
		{
			if (seg_section[i] < 0) continue;
			ps += seg_type_prob[i];
			np++;
			if (seg_section[i] != last_sec)
			{
				if (run > 0)
					sec += (sec.empty() ? "" : ",") + integerToString(run_type + 1) + "x" + integerToString(run);
				last_sec = seg_section[i];
				run_type = seg_type[i];
				run = 0;
				nsec++;
			}
			run++;
		}
		if (run > 0)
			sec += (sec.empty() ? "" : ",") + integerToString(run_type + 1) + "x" + integerToString(run);
		if (sec.empty()) sec = "-";
		const double fp = (f < fil_period.size()) ? fil_period[f] : 0.;
		const double fr = (f < fil_period_R.size()) ? fil_period_R[f] : 0.;
		out << name.substr(0, colon) << " " << name.substr(colon + 1) << " " << segs[f].size() << " " << nsec
		    << " " << ((np > 0) ? ps / np : 0.) << " " << sec << " " << fp << " " << fr << "\n";
	}
	out << "\n";
}

void HelicalContext::storePolarity(long int i, double w_noflip, double w_flip, double f_noflip, double f_flip)
{
	double l0 = (f_noflip > 0.) ? w_noflip / f_noflip : 0.;
	double l1 = (f_flip > 0.) ? w_flip / f_flip : 0.;
	double llr;
	if (l0 <= 0. && l1 <= 0.)
		llr = 0.;
	else if (l0 <= 0.)
		llr = pol_clamp;
	else if (l1 <= 0.)
		llr = -pol_clamp;
	else
		llr = std::max(-pol_clamp, std::min(pol_clamp, (RFLOAT)log(l1 / l0)));
	pol_llr[i] = llr;
}

// log-odds of the polarity after d steps of the symmetric two-state chain, from log-odds x
static inline double polPropagate(double x, long int d, double rho)
{
	if (d <= 0)
		return x;
	double stay = 0.5 * (1. + pow(1. - 2. * rho, (double)d));
	double p = 1. / (1. + exp(-x));
	double q = stay * p + (1. - stay) * (1. - p);
	q = std::max(1e-300, std::min(1. - 1e-16, q));
	return log(q / (1. - q));
}

void HelicalContext::updatePolarity()
{
	long int nr_fil = pol_members.size();
	std::vector<long int> fil_switches(nr_fil, 0);
	std::vector<long int> fil_flipped(nr_fil, 0);
	double lmax = log(pol_prior_max / (1. - pol_prior_max));

	#pragma omp parallel for schedule(dynamic)
	for (long int f = 0; f < nr_fil; f++)
	{
		const std::vector<long int> &m = pol_members[f];
		const std::vector<long int> &pos = pol_pos[f];
		long int n = m.size();
		std::vector<double> e(n), a(n), b(n);
		for (long int j = 0; j < n; j++)
			e[j] = std::isnan(pol_llr[m[j]]) ? 0. : pol_temper * pol_llr[m[j]];
		// a[j]: log-odds of P(flipped at j | evidence up to and including j)
		// b[j]: log-likelihood ratio of the evidence from j onwards, given the polarity at j
		for (long int j = 0; j < n; j++)
			a[j] = e[j] + ((j > 0) ? polPropagate(a[j-1], pos[j] - pos[j-1], pol_switch) : 0.);
		for (long int j = n - 1; j >= 0; j--)
			b[j] = e[j] + ((j < n - 1) ? polPropagate(b[j+1], pos[j+1] - pos[j], pol_switch) : 0.);
		long int jl = -1, jr = 0;
		int last = -1;
		for (long int j = 0; j < n; j++)
		{
			double x = a[j] + ((j < n - 1) ? polPropagate(b[j+1], pos[j+1] - pos[j], pol_switch) : 0.);
			pol_post[m[j]] = 1. / (1. + exp(-x));
			// prior from the segments at least pol_exclude_steps away on either side
			while (jl + 1 < n && pos[jl + 1] <= pos[j] - pol_exclude_steps)
				jl++;
			while (jr < n && pos[jr] < pos[j] + pol_exclude_steps)
				jr++;
			double c = 0.;
			if (jl >= 0)
				c += polPropagate(a[jl], pos[j] - pos[jl], pol_switch);
			if (jr < n)
				c += polPropagate(b[jr], pos[jr] - pos[j], pol_switch);
			c = std::max(-lmax, std::min(lmax, c));
			pol_prior[m[j]] = 1. / (1. + exp(-c));
			int now = (x > 0.);
			fil_flipped[f] += now;
			if (last >= 0 && now != last)
				fil_switches[f]++;
			last = now;
		}
	}

	pol_nr_switches = pol_nr_consistent = 0;
	long int nr_flipped = 0;
	for (long int f = 0; f < nr_fil; f++)
	{
		pol_nr_switches += fil_switches[f];
		pol_nr_consistent += (fil_switches[f] == 0);
		nr_flipped += fil_flipped[f];
	}
	pol_frac_flipped = (N > 0) ? (double)nr_flipped / N : 0.;
}

// Solve symmetric pentadiagonal system A y = b with diagonals d (n), off-1 e1 (n-1), off-2 e2 (n-2)
static void solvePentadiagonalSPD(std::vector<double> d, std::vector<double> e1, std::vector<double> e2,
                                  const std::vector<double> &rhs, std::vector<double> &y)
{
	const int n = (int)d.size();
	std::vector<double> l1(std::max(0, n - 1), 0.), l2(std::max(0, n - 2), 0.), z(n, 0.);
	for (int i = 0; i < n; i++)
	{
		double piv = d[i];
		if (i >= 1) piv -= l1[i - 1] * l1[i - 1] * d[i - 1];
		if (i >= 2) piv -= l2[i - 2] * l2[i - 2] * d[i - 2];
		d[i] = std::max(1e-12, piv);
		if (i + 1 < n)
		{
			double v = e1[i];
			if (i >= 1) v -= l1[i - 1] * l2[i - 1] * d[i - 1];
			l1[i] = v / d[i];
		}
		if (i + 2 < n)
			l2[i] = e2[i] / d[i];
	}
	for (int i = 0; i < n; i++)
	{
		double s = rhs[i];
		if (i >= 1) s -= l1[i - 1] * z[i - 1];
		if (i >= 2) s -= l2[i - 2] * z[i - 2];
		z[i] = s;
	}
	y.assign(n, 0.);
	for (int i = n - 1; i >= 0; i--)
	{
		double s = z[i] / d[i];
		if (i + 1 < n) s -= l1[i] * y[i + 1];
		if (i + 2 < n) s -= l2[i] * y[i + 2];
		y[i] = s;
	}
}

void HelicalContext::smoothCenterline1D(const std::vector<double> &y_obs, const std::vector<double> &init_w,
                                          double step_A, double huber_delta, double alpha_pick,
                                          const std::vector<double> &y_pick, std::vector<double> &y_hat)
{
	const int L = (int)y_obs.size();
	if (L <= 0)
	{
		y_hat.clear();
		return;
	}
	if (L < 4)
	{
		y_hat = y_obs;
		return;
	}
	const double step_eff = std::max(4.0, step_A);
	const double lam2 = 12.0 * std::pow(14.25 / step_eff, 3.0);
	const double delta = std::max(1e-3, huber_delta);
	std::vector<double> base_w(L, 1.0);
	if ((int)init_w.size() == L)
		for (int j = 0; j < L; j++)
			base_w[j] = std::max(1e-4, init_w[j]);
	std::vector<double> w = base_w;
	y_hat = ((int)y_pick.size() == L) ? y_pick : y_obs;

	for (int irls = 0; irls < 3; irls++)
	{
		std::vector<double> d(L, 0.), e1(L - 1, 0.), e2(L - 2, 0.), rhs(L, 0.);
		for (int j = 0; j < L; j++)
		{
			const double wi = w[j];
			const double yp = ((int)y_pick.size() == L) ? y_pick[j] : 0.0;
			d[j] = std::max(1e-9, wi + alpha_pick);
			rhs[j] = wi * y_obs[j] + alpha_pick * yp;
		}
		for (int r = 0; r + 2 < L; r++)
		{
			d[r] += lam2;
			d[r + 1] += 4.0 * lam2;
			d[r + 2] += lam2;
			e1[r] -= 2.0 * lam2;
			e1[r + 1] -= 2.0 * lam2;
			e2[r] += lam2;
		}
		solvePentadiagonalSPD(d, e1, e2, rhs, y_hat);
		for (int j = 0; j < L; j++)
		{
			const double res = std::fabs(y_obs[j] - y_hat[j]);
			w[j] = base_w[j] * ((res <= delta) ? 1.0 : (delta / res));
		}
	}
}

namespace
{
void resolveCoordApixMap(const MetaDataTable &MD, const MetaDataTable *MDopt, RFLOAT default_angpix,
                         RFLOAT &default_coord_apix, std::map<int, RFLOAT> &og_coord_apix)
{
	default_coord_apix = default_angpix;
	og_coord_apix.clear();
	if (MDopt != NULL && MDopt->containsLabel(EMDL_MICROGRAPH_ORIGINAL_PIXEL_SIZE) && MDopt->numberOfObjects() > 0)
	{
		RFLOAT p0 = 0.;
		if (MDopt->getValue(EMDL_MICROGRAPH_ORIGINAL_PIXEL_SIZE, p0, 0) && p0 > 0.)
			default_coord_apix = p0;
		if (MDopt->containsLabel(EMDL_IMAGE_OPTICS_GROUP) && MD.containsLabel(EMDL_IMAGE_OPTICS_GROUP))
		{
			for (long int oi = 0; oi < (long int)MDopt->numberOfObjects(); oi++)
			{
				int og = 0;
				RFLOAT p = 0.;
				if (MDopt->getValue(EMDL_IMAGE_OPTICS_GROUP, og, oi) &&
				    MDopt->getValue(EMDL_MICROGRAPH_ORIGINAL_PIXEL_SIZE, p, oi) && p > 0.)
					og_coord_apix[og] = p;
			}
		}
	}
}
} // namespace

void HelicalContext::smoothFilamentTracks(MetaDataTable &MD, const std::vector<long int> &seg_rows,
                                          RFLOAT angpix, RFLOAT max_offset_A, int iter, const MetaDataTable *MDopt, bool update_origin_priors)
{
	if (N <= 0 || pol_members.empty() || angpix <= 0.)
		return;
	if (!MD.containsLabel(EMDL_IMAGE_COORD_X) || !MD.containsLabel(EMDL_IMAGE_COORD_Y) ||
	    !MD.containsLabel(EMDL_ORIENT_PSI_PRIOR))
		return;

	// rlnCoordinateX/Y are in unbinned micrograph pixels (rlnMicrographOriginalPixelSize), whereas
	// rlnOriginX/YAngst are in Angstroms: scale coordinates by original micrograph pixel size when available.
	RFLOAT default_coord_apix = angpix;
	std::map<int, RFLOAT> og_coord_apix;
	resolveCoordApixMap(MD, MDopt, angpix, default_coord_apix, og_coord_apix);

	if ((long int)orig_psi_prior.size() != N)
	{
		orig_psi_prior.assign(N, 0.);
		for (long int i = 0; i < N; i++)
			MD.getValue(EMDL_ORIENT_PSI_PRIOR, orig_psi_prior[i], seg_rows[i]);
	}

	const bool has_ox = MD.containsLabel(EMDL_ORIENT_ORIGIN_X_ANGSTROM);
	const bool has_oy = MD.containsLabel(EMDL_ORIENT_ORIGIN_Y_ANGSTROM);
	if (update_origin_priors)
	{
		if (!MD.containsLabel(EMDL_ORIENT_ORIGIN_X_PRIOR_ANGSTROM))
			MD.addLabel(EMDL_ORIENT_ORIGIN_X_PRIOR_ANGSTROM);
		if (!MD.containsLabel(EMDL_ORIENT_ORIGIN_Y_PRIOR_ANGSTROM))
			MD.addLabel(EMDL_ORIENT_ORIGIN_Y_PRIOR_ANGSTROM);
	}

	const double alpha_pick = (iter <= 1) ? 0.50 : 0.04;
	const double huber_delta = std::max(4.0, 0.8 * (double)angpix);
	const double max_shift = (max_offset_A > 0.) ? (double)max_offset_A * 0.85 : 12.0;

	for (size_t f = 0; f < pol_members.size(); f++)
	{
		const std::vector<long int> &m = pol_members[f];
		const int L = (int)m.size();
		if (L < 4)
			continue;

		std::vector<double> px_pick(L), py_pick(L), px_obs(L), py_obs(L);
		for (int j = 0; j < L; j++)
		{
			const long int r = seg_rows[m[j]];
			RFLOAT cx = 0., cy = 0., ox = 0., oy = 0.;
			MD.getValue(EMDL_IMAGE_COORD_X, cx, r);
			MD.getValue(EMDL_IMAGE_COORD_Y, cy, r);
			if (has_ox) MD.getValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, ox, r);
			if (has_oy) MD.getValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, oy, r);
			RFLOAT coord_apix = default_coord_apix;
			if (!og_coord_apix.empty())
			{
				int og = 0;
				if (MD.getValue(EMDL_IMAGE_OPTICS_GROUP, og, r))
				{
					auto it_og = og_coord_apix.find(og);
					if (it_og != og_coord_apix.end()) coord_apix = it_og->second;
				}
			}
			px_pick[j] = cx * coord_apix;
			py_pick[j] = cy * coord_apix;
			px_obs[j] = px_pick[j] - ox;
			py_obs[j] = py_pick[j] - oy;
		}

		// Robust chord from first quartile to last quartile of picked points
		const int qcnt = std::max(1, L / 4);
		double x0 = 0., y0 = 0., x1 = 0., y1 = 0.;
		for (int j = 0; j < qcnt; j++) { x0 += px_pick[j]; y0 += py_pick[j]; }
		for (int j = L - qcnt; j < L; j++) { x1 += px_pick[j]; y1 += py_pick[j]; }
		double ch_x = x1 / qcnt - x0 / qcnt, ch_y = y1 / qcnt - y0 / qcnt;
		double ch_len = std::sqrt(ch_x * ch_x + ch_y * ch_y);
		if (ch_len < 1e-3)
		{
			ch_x = px_pick[L - 1] - px_pick[0];
			ch_y = py_pick[L - 1] - py_pick[0];
			ch_len = std::sqrt(ch_x * ch_x + ch_y * ch_y);
		}
		if (ch_len < 1e-3)
			continue;
		const double nx_v = -ch_y / ch_len, ny_v = ch_x / ch_len;

		std::vector<double> y_pick(L), y_obs(L), y_hat(L), init_w(L, 1.0);
		for (int j = 0; j < L; j++)
		{
			y_pick[j] = (px_pick[j] - px_pick[0]) * nx_v + (py_pick[j] - py_pick[0]) * ny_v;
			y_obs[j] = (px_obs[j] - px_pick[0]) * nx_v + (py_obs[j] - py_pick[0]) * ny_v;
		}
		smoothCenterline1D(y_obs, init_w, (double)step, huber_delta, alpha_pick, y_pick, y_hat);

		std::vector<double> sx(L), sy(L);
		for (int j = 0; j < L; j++)
		{
			double dy_clamp = y_hat[j] - y_pick[j];
			if (dy_clamp > max_shift) dy_clamp = max_shift;
			if (dy_clamp < -max_shift) dy_clamp = -max_shift;
			sx[j] = px_pick[j] + dy_clamp * nx_v;
			sy[j] = py_pick[j] + dy_clamp * ny_v;

			if (update_origin_priors)
			{
				const long int r = seg_rows[m[j]];
				MD.setValue(EMDL_ORIENT_ORIGIN_X_PRIOR_ANGSTROM, (RFLOAT)(px_pick[j] - sx[j]), r);
				MD.setValue(EMDL_ORIENT_ORIGIN_Y_PRIOR_ANGSTROM, (RFLOAT)(py_pick[j] - sy[j]), r);
			}
		}
		for (int j = 0; j < L; j++)
		{
			const int j0 = std::max(0, j - 1), j1 = std::min(L - 1, j + 1);
			const double tdx = sx[j1] - sx[j0], tdy = sy[j1] - sy[j0];
			if (tdx * tdx + tdy * tdy < 1e-6)
				continue;
			double psi_tan = -RAD2DEG(std::atan2(tdy, tdx));
			const double orig = orig_psi_prior[m[j]];
			if (std::fabs(orig) > 361.0) continue;
			double diff = std::remainder(psi_tan - orig, 360.0);
			if (std::fabs(diff) > 90.0)
			{
				psi_tan += 180.0;
				diff = std::remainder(psi_tan - orig, 360.0);
			}
			if (diff > 8.0) diff = 8.0;
			if (diff < -8.0) diff = -8.0;
			double psi_out = std::remainder(orig + diff, 360.0);
			MD.setValue(EMDL_ORIENT_PSI_PRIOR, (RFLOAT)psi_out, seg_rows[m[j]]);
		}
	}
}

void HelicalContext::smoothAndRecenterCoordinates(MetaDataTable &MD, const std::vector<long int> &seg_rows,
                                                  const std::vector<int> *cls, RFLOAT img_angpix, RFLOAT max_offset_A,
                                                  const MetaDataTable *MDopt, bool resolve_polarity, bool update_psi_also,
                                                  bool smooth_coords)
{
	if (N <= 0 || pol_members.empty() || img_angpix <= 0.)
		return;
	if (!MD.containsLabel(EMDL_IMAGE_COORD_X) || !MD.containsLabel(EMDL_IMAGE_COORD_Y))
		return;

	RFLOAT default_coord_apix = img_angpix;
	std::map<int, RFLOAT> og_coord_apix;
	resolveCoordApixMap(MD, MDopt, img_angpix, default_coord_apix, og_coord_apix);

	const bool has_ox = MD.containsLabel(EMDL_ORIENT_ORIGIN_X_ANGSTROM);
	const bool has_oy = MD.containsLabel(EMDL_ORIENT_ORIGIN_Y_ANGSTROM);
	const bool has_oxp = MD.containsLabel(EMDL_ORIENT_ORIGIN_X_PRIOR_ANGSTROM);
	const bool has_oyp = MD.containsLabel(EMDL_ORIENT_ORIGIN_Y_PRIOR_ANGSTROM);
	const bool has_psi = MD.containsLabel(EMDL_ORIENT_PSI);
	const bool has_psip = MD.containsLabel(EMDL_ORIENT_PSI_PRIOR);
	const bool has_pmax = MD.containsLabel(EMDL_PARTICLE_PMAX);

	if (resolve_polarity && !has_psip && has_psi)
		MD.addLabel(EMDL_ORIENT_PSI_PRIOR);

	const double huber_delta = std::max(4.0, 0.8 * (double)img_angpix);
	const double max_shift = (max_offset_A > 0.) ? (double)max_offset_A * 0.85 : ((diameter > 0.) ? 0.45 * (double)diameter : 20.0);

	for (size_t f = 0; f < pol_members.size(); f++)
	{
		const std::vector<long int> &m = pol_members[f];
		const int L = (int)m.size();
		if (L <= 0) continue;

		std::vector<double> px_pick(L), py_pick(L), px_obs(L), py_obs(L), c_apix(L, (double)default_coord_apix);
		std::vector<double> psi_v(L, 0.), psip_v(L, 0.), pmax_v(L, 1.0), init_w(L, 1.0);
		std::vector<char> is_junk_seg(L, 0);
		int n_good = 0;
		for (int j = 0; j < L; j++)
		{
			const long int idx = m[j];
			const long int r = seg_rows[idx];
			RFLOAT cx = 0., cy = 0., ox = 0., oy = 0., p = 0., pp = 0., pm = 1.0;
			MD.getValue(EMDL_IMAGE_COORD_X, cx, r);
			MD.getValue(EMDL_IMAGE_COORD_Y, cy, r);
			if (has_ox) MD.getValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, ox, r);
			if (has_oy) MD.getValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, oy, r);
			if (has_psi) MD.getValue(EMDL_ORIENT_PSI, p, r);
			if (has_psip) MD.getValue(EMDL_ORIENT_PSI_PRIOR, pp, r);
			else pp = p;
			if (!has_psi) p = pp;
			if (has_pmax) MD.getValue(EMDL_PARTICLE_PMAX, pm, r);
			if (!og_coord_apix.empty())
			{
				int og = 0;
				if (MD.getValue(EMDL_IMAGE_OPTICS_GROUP, og, r))
				{
					auto it_og = og_coord_apix.find(og);
					if (it_og != og_coord_apix.end()) c_apix[j] = (double)it_og->second;
				}
			}
			px_pick[j] = (double)cx * c_apix[j];
			py_pick[j] = (double)cy * c_apix[j];
			px_obs[j] = px_pick[j] - (double)ox;
			py_obs[j] = py_pick[j] - (double)oy;
			psi_v[j] = (double)p;
			psip_v[j] = (double)pp;
			pmax_v[j] = std::max(0.05, (double)pm);
			const bool jseg = (cls != NULL && idx >= 0 && idx < (long int)cls->size() &&
			                   (*cls)[idx] >= 0 && (*cls)[idx] < K &&
			                   !junk_class.empty() && junk_class[(*cls)[idx]]);
			is_junk_seg[j] = jseg ? 1 : 0;
			if (!jseg) n_good++;
			init_w[j] = jseg ? 0.1 : 1.0;
		}

		// 1. Centerline smoothing along transverse filament normal
		std::vector<double> sx = px_pick, sy = py_pick;
		if (smooth_coords && L >= 2)
		{
			const int qcnt = std::max(1, L / 4);
			double x0 = 0., y0 = 0., x1 = 0., y1 = 0.;
			for (int j = 0; j < qcnt; j++) { x0 += px_pick[j]; y0 += py_pick[j]; }
			for (int j = L - qcnt; j < L; j++) { x1 += px_pick[j]; y1 += py_pick[j]; }
			double ch_x = x1 / qcnt - x0 / qcnt, ch_y = y1 / qcnt - y0 / qcnt;
			double ch_len = std::sqrt(ch_x * ch_x + ch_y * ch_y);
			if (ch_len < 1e-3)
			{
				ch_x = px_pick[L - 1] - px_pick[0];
				ch_y = py_pick[L - 1] - py_pick[0];
				ch_len = std::sqrt(ch_x * ch_x + ch_y * ch_y);
			}
			if (ch_len >= 1e-3)
			{
				const double nx_v = -ch_y / ch_len, ny_v = ch_x / ch_len;
				std::vector<double> y_pick(L), y_obs(L), y_hat(L);
				for (int j = 0; j < L; j++)
				{
					y_pick[j] = (px_pick[j] - px_pick[0]) * nx_v + (py_pick[j] - py_pick[0]) * ny_v;
					y_obs[j] = (px_obs[j] - px_pick[0]) * nx_v + (py_obs[j] - py_pick[0]) * ny_v;
				}
				if (L >= 4)
					smoothCenterline1D(y_obs, init_w, (double)step, huber_delta, 0.0, y_pick, y_hat);
				else
				{
					for (int j = 0; j < L; j++)
					{
						double num = 0., den = 0.;
						for (int q = std::max(0, j - 1); q <= std::min(L - 1, j + 1); q++)
						{
							const double wq = ((q == j) ? 0.5 : 0.25) * init_w[q];
							num += wq * y_obs[q];
							den += wq;
						}
						y_hat[j] = (den > 0.) ? (num / den) : y_obs[j];
					}
				}
				for (int j = 0; j < L; j++)
				{
					double dy_clamp = y_hat[j] - y_pick[j];
					if (dy_clamp > max_shift) dy_clamp = max_shift;
					if (dy_clamp < -max_shift) dy_clamp = -max_shift;
					sx[j] = px_pick[j] + dy_clamp * nx_v;
					sy[j] = py_pick[j] + dy_clamp * ny_v;
				}
			}
		}

		if (smooth_coords)
		{
			for (int j = 0; j < L; j++)
			{
				const long int r = seg_rows[m[j]];
				if (c_apix[j] > 0.)
				{
					MD.setValue(EMDL_IMAGE_COORD_X, (RFLOAT)(sx[j] / c_apix[j]), r);
					MD.setValue(EMDL_IMAGE_COORD_Y, (RFLOAT)(sy[j] / c_apix[j]), r);
				}
				if (has_ox) MD.setValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, (RFLOAT)0., r);
				if (has_oy) MD.setValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, (RFLOAT)0., r);
				if (has_oxp) MD.setValue(EMDL_ORIENT_ORIGIN_X_PRIOR_ANGSTROM, (RFLOAT)0., r);
				if (has_oyp) MD.setValue(EMDL_ORIENT_ORIGIN_Y_PRIOR_ANGSTROM, (RFLOAT)0., r);
			}
		}

		// 2. Coherent filament polarity and smoothed tangent rlnAnglePsiPrior
		if (resolve_polarity && (has_psi || has_psip))
		{
			double keep_votes = 0., flip_votes = 0.;
			for (int j = 0; j < L; j++)
			{
				if (n_good > 0 && is_junk_seg[j]) continue;
				const double d = std::fabs(std::remainder(psi_v[j] - psip_v[j], 360.0));
				if (d > 90.0) flip_votes += pmax_v[j];
				else keep_votes += pmax_v[j];
			}
			const double flip_shift = (flip_votes > keep_votes) ? 180.0 : 0.0;
			for (int j = 0; j < L; j++)
			{
				const long int r = seg_rows[m[j]];
				double base_psi = std::remainder(psip_v[j] + flip_shift, 360.0);
				if (std::fabs(psip_v[j]) > 361.0) continue;
				if (L >= 2)
				{
					const int j0 = std::max(0, j - 1), j1 = std::min(L - 1, j + 1);
					const double tdx = sx[j1] - sx[j0], tdy = sy[j1] - sy[j0];
					if (tdx * tdx + tdy * tdy >= 1e-6)
					{
						double psi_tan = -RAD2DEG(std::atan2(tdy, tdx));
						double diff = std::remainder(psi_tan - base_psi, 360.0);
						if (std::fabs(diff) > 90.0)
						{
							psi_tan += 180.0;
							diff = std::remainder(psi_tan - base_psi, 360.0);
						}
						if (diff > 12.0) diff = 12.0;
						if (diff < -12.0) diff = -12.0;
						base_psi = std::remainder(base_psi + diff, 360.0);
					}
				}
				MD.setValue(EMDL_ORIENT_PSI_PRIOR, (RFLOAT)base_psi, r);
				if (update_psi_also && has_psi)
					MD.setValue(EMDL_ORIENT_PSI, (RFLOAT)base_psi, r);
			}
		}
	}
}

void HelicalContext::writePolarityLabels(MetaDataTable &MD, const std::vector<long int> &seg_rows) const
{
	for (long int i = 0; i < N; i++)
		MD.setValue(EMDL_PARTICLE_HELICAL_POLARITY_FLIP_PROBABILITY, (RFLOAT)pol_post[i], seg_rows[i]);
}

void HelicalContext::resetEvidence()
{
	std::fill(loglik.begin(), loglik.end(), std::numeric_limits<float>::quiet_NaN());
	std::fill(pol_llr.begin(), pol_llr.end(), std::numeric_limits<float>::quiet_NaN());
}

namespace
{
struct ChainSuffStat
{
	int c0 = -1;
	long int len = 0;
	std::vector<std::pair<int, double> > unigram;
	std::vector<std::pair<int, double> > trans;
};

struct ChainMixtureModel
{
	int T = 1;
	int C = 0;
	double kappa = 1.0;
	std::vector<double> logw;   // T
	std::vector<double> mu;     // T x C (probabilities)
	std::vector<double> B;      // T x C x C (row-stochastic probabilities)
	std::vector<double> log_mu; // T x C (cached log max(1e-12, mu))
	std::vector<double> log_B;  // T x C x C (cached log max(1e-12, B))
};

static double chainLogSumExp(const double *a, int T)
{
	double m = a[0];
	for (int t = 1; t < T; t++) m = std::max(m, a[t]);
	double s = 0.;
	for (int t = 0; t < T; t++) s += std::exp(a[t] - m);
	return m + std::log(std::max(1e-300, s));
}

static void buildChainStats(const helixtypes::FilamentChains &X, std::vector<ChainSuffStat> &stats)
{
	const int C = X.C;
	const long int F = (long int)X.seqs.size();
	stats.resize(F);
	#pragma omp parallel for schedule(dynamic, 64)
	for (long int f = 0; f < F; f++)
	{
		ChainSuffStat &st = stats[f];
		const std::vector<int> &seq = X.seqs[f];
		st.len = (long int)seq.size();
		st.unigram.clear();
		st.trans.clear();
		if (st.len == 0) continue;
		st.c0 = seq[0];
		std::vector<int> u_ids, t_ids;
		u_ids.reserve(st.len);
		t_ids.reserve(st.len);
		for (long int m = 0; m < st.len; m++)
		{
			const int c = seq[m];
			if (c >= 0 && c < C)
			{
				u_ids.push_back(c);
				if (m > 0 && seq[m - 1] >= 0 && seq[m - 1] < C)
					t_ids.push_back(seq[m - 1] * C + c);
			}
		}
		std::sort(u_ids.begin(), u_ids.end());
		for (size_t i = 0; i < u_ids.size(); )
		{
			size_t j = i + 1;
			while (j < u_ids.size() && u_ids[j] == u_ids[i]) j++;
			st.unigram.push_back({u_ids[i], (double)(j - i)});
			i = j;
		}
		std::sort(t_ids.begin(), t_ids.end());
		for (size_t i = 0; i < t_ids.size(); )
		{
			size_t j = i + 1;
			while (j < t_ids.size() && t_ids[j] == t_ids[i]) j++;
			st.trans.push_back({t_ids[i], (double)(j - i)});
			i = j;
		}
	}
}

static void evalChainLogLik(const ChainMixtureModel &M, const ChainSuffStat &st, double *L)
{
	const int C = M.C, T = M.T;
	for (int t = 0; t < T; t++)
	{
		if (st.len == 0 || st.c0 < 0 || st.c0 >= C) { L[t] = M.logw[t]; continue; }
		double ll_uni = 0.;
		const double *lmu = &M.log_mu[(size_t)t * C];
		for (const auto &kv : st.unigram)
			ll_uni += kv.second * lmu[kv.first];
		double ll_tr = lmu[st.c0];
		const double *lBt = &M.log_B[(size_t)t * C * C];
		for (const auto &kv : st.trans)
			ll_tr += kv.second * lBt[kv.first];
		L[t] = M.logw[t] + M.kappa * (0.80 * ll_uni + 0.20 * ll_tr);
	}
}

struct ChainSpectralBasis
{
	int C = 0;
	int n_top_edges = 0;
	int D = 0;
	int max_m_comb = 0;
	int max_m_uni = 0;
	std::vector<int> edge_col;
	std::vector<double> feat;       // N_tr x D (normalized sqrt unigram + top-edge)
	std::vector<double> len_w;      // N_tr (segment counts)
	std::vector<double> z_comb_all; // N_tr x max_m_comb (scaled by 1/sqrt(max(eig, 0.01*eig0)))
	std::vector<double> z_uni_all;  // N_tr x max_m_uni
};

static void buildChainSpectralBasis(const std::vector<ChainSuffStat> &stats, int C,
                                    const std::vector<long int> &train, int max_T,
                                    ChainSpectralBasis &spec)
{
	spec = ChainSpectralBasis();
	spec.C = C;
	const size_t N_tr = train.size();
	if (N_tr == 0 || C <= 0) return;

	// Select top-variance transitions on train
	std::vector<double> s1((size_t)C * C, 0.), s2((size_t)C * C, 0.);
	for (long int f : train)
	{
		const double inv_d = 1.0 / std::max(1.0, (double)stats[f].len - 1.0);
		for (const auto &kv : stats[f].trans)
		{
			const double v = kv.second * inv_d;
			s1[kv.first] += v;
			s2[kv.first] += v * v;
		}
	}
	std::vector<std::pair<double, int> > edge_var((size_t)C * C);
	const double inv_nt = 1.0 / (double)N_tr;
	for (int p = 0; p < C * C; p++)
	{
		const double m = s1[p] * inv_nt;
		edge_var[p] = {s2[p] * inv_nt - m * m, p};
	}
	spec.n_top_edges = std::min((int)edge_var.size(), 64);
	std::partial_sort(edge_var.begin(), edge_var.begin() + spec.n_top_edges, edge_var.end(),
	                  [](const std::pair<double, int> &a, const std::pair<double, int> &b) { return a.first > b.first; });
	spec.edge_col.assign((size_t)C * C, -1);
	for (int e = 0; e < spec.n_top_edges; e++)
		spec.edge_col[edge_var[e].second] = e;

	spec.D = C + spec.n_top_edges;
	const int D = spec.D;
	spec.feat.assign(N_tr * D, 0.);
	spec.len_w.assign(N_tr, 1.0);

	// Build sparse normalized row entries for combined (D) and unigram-only (C) co-occurrence graphs
	std::vector<std::vector<std::pair<int, double> > > rows_comb(N_tr), rows_uni(N_tr);
	std::vector<double> deg_comb(D, 0.), deg_uni(C, 0.);
	for (size_t idx = 0; idx < N_tr; idx++)
	{
		const ChainSuffStat &st = stats[train[idx]];
		spec.len_w[idx] = std::max(1.0, (double)st.len);
		const double du = std::max(1.0, (double)st.len);
		const double dt = std::max(1.0, (double)st.len - 1.0);

		double n2 = 0., r_comb = 0., r_uni = 0.;
		for (const auto &kv : st.unigram)
		{
			const double sf = std::sqrt(kv.second / du);
			spec.feat[idx * D + kv.first] = sf;
			n2 += sf * sf;

			const double vf = kv.second / du;
			if (vf > 0.)
			{
				rows_uni[idx].push_back({kv.first, vf});
				rows_comb[idx].push_back({kv.first, vf});
				r_uni += vf;
				r_comb += vf;
			}
		}
		for (const auto &kv : st.trans)
		{
			const int e = spec.edge_col[kv.first];
			if (e >= 0)
			{
				const double sf = 0.7 * std::sqrt(kv.second / dt);
				spec.feat[idx * D + C + e] = sf;
				n2 += sf * sf;

				const double vf = 0.7 * kv.second / dt;
				if (vf > 0.)
				{
					rows_comb[idx].push_back({C + e, vf});
					r_comb += vf;
				}
			}
		}
		const double inv = (n2 > 0.) ? (1.0 / std::sqrt(n2)) : 1.0;
		for (int d = 0; d < D; d++) spec.feat[idx * D + d] *= inv;

		if (r_uni > 0.)
			for (auto &kv : rows_uni[idx]) { kv.second /= r_uni; deg_uni[kv.first] += kv.second; }
		if (r_comb > 0.)
			for (auto &kv : rows_comb[idx]) { kv.second /= r_comb; deg_comb[kv.first] += kv.second; }
	}

	auto solveCoocSpectrum = [&](int dim, const std::vector<std::vector<std::pair<int, double> > > &rows,
	                             const std::vector<double> &deg, int max_m,
	                             std::vector<double> &z_out, int &m_out)
	{
		m_out = std::min(dim, std::max(1, max_m));
		z_out.assign(N_tr * m_out, 0.);
		if (dim <= 0) return;

		double dsum = 0.;
		int dact = 0;
		for (int a = 0; a < dim; a++)
			if (deg[a] > 0.) { dsum += deg[a]; dact++; }
		if (dact == 0) return;
		const double tau = std::max(1e-6, 0.05 * (dsum / dact));
		std::vector<double> wcol(dim, 0.);
		for (int a = 0; a < dim; a++)
			if (deg[a] > 0.) wcol[a] = 1.0 / std::sqrt(deg[a] + tau);

		Eigen::MatrixXd S = Eigen::MatrixXd::Zero(dim, dim);
		for (size_t idx = 0; idx < N_tr; idx++)
		{
			const auto &r = rows[idx];
			for (size_t i = 0; i < r.size(); i++)
			{
				const int ai = r[i].first;
				const double vi = wcol[ai] * r[i].second;
				S(ai, ai) += vi * vi;
				for (size_t j = i + 1; j < r.size(); j++)
				{
					const int aj = r[j].first;
					const double vv = vi * wcol[aj] * r[j].second;
					S(ai, aj) += vv;
					S(aj, ai) += vv;
				}
			}
		}

		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
		if (es.info() != Eigen::Success) return;
		const auto &evals = es.eigenvalues();
		const auto &evecs = es.eigenvectors();
		const double lam0 = std::max(1e-8, evals(dim - 1));
		const double lam_floor = std::max(1e-8, 0.01 * lam0);

		std::vector<double> inv_sig(m_out), sgn(m_out, 1.0);
		for (int k = 0; k < m_out; k++)
		{
			const int col = dim - 1 - k;
			inv_sig[k] = 1.0 / std::sqrt(std::max(lam_floor, evals(col)));
			double mx = 0., val_mx = 0.;
			for (int a = 0; a < dim; a++)
			{
				const double av = std::fabs(evecs(a, col));
				if (av > mx) { mx = av; val_mx = evecs(a, col); }
			}
			sgn[k] = (val_mx < 0.) ? -1.0 : 1.0;
		}

		for (size_t idx = 0; idx < N_tr; idx++)
		{
			const auto &r = rows[idx];
			for (int k = 0; k < m_out; k++)
			{
				const int col = dim - 1 - k;
				double dot = 0.;
				for (const auto &kv : r)
					dot += (wcol[kv.first] * kv.second) * evecs(kv.first, col);
				z_out[idx * m_out + k] = sgn[k] * inv_sig[k] * dot;
			}
		}
	};

	solveCoocSpectrum(D, rows_comb, deg_comb, max_T, spec.z_comb_all, spec.max_m_comb);
	solveCoocSpectrum(C, rows_uni, deg_uni, max_T, spec.z_uni_all, spec.max_m_uni);
}

static void extractNormalizedSlice(const std::vector<double> &z_all, size_t N_tr, int max_m, int m,
                                   std::vector<double> &Z_out)
{
	Z_out.assign(N_tr * m, 0.);
	for (size_t idx = 0; idx < N_tr; idx++)
	{
		double n2 = 0.;
		for (int k = 0; k < m; k++)
		{
			const double v = z_all[idx * max_m + k];
			Z_out[idx * m + k] = v;
			n2 += v * v;
		}
		if (n2 > 1e-24)
		{
			const double inv = 1.0 / std::sqrt(n2);
			for (int k = 0; k < m; k++) Z_out[idx * m + k] *= inv;
		}
		else
			Z_out[idx * m] = 1.0;
	}
}

static void sphericalKmeansRefine(const std::vector<double> &Z, size_t N_tr, int m, int T,
                                  std::vector<int> &labels, int max_iter)
{
	if (N_tr == 0 || m <= 0 || T <= 1) return;
	std::vector<double> cent((size_t)T * m, 0.);
	std::vector<int> cnt(T, 0);

	for (int it = 0; it < max_iter; it++)
	{
		std::fill(cent.begin(), cent.end(), 0.);
		std::fill(cnt.begin(), cnt.end(), 0);
		for (size_t idx = 0; idx < N_tr; idx++)
		{
			const int t = labels[idx];
			if (t < 0 || t >= T) continue;
			cnt[t]++;
			for (int d = 0; d < m; d++)
				cent[(size_t)t * m + d] += Z[idx * m + d];
		}
		// Rescue any empty cluster deterministically from largest-SSE cluster
		for (int t = 0; t < T; t++)
		{
			if (cnt[t] > 0)
			{
				double n2 = 0.;
				for (int d = 0; d < m; d++) n2 += cent[(size_t)t * m + d] * cent[(size_t)t * m + d];
				const double inv = (n2 > 1e-24) ? (1.0 / std::sqrt(n2)) : 1.0;
				for (int d = 0; d < m; d++) cent[(size_t)t * m + d] *= inv;
			}
		}
		for (int t = 0; t < T; t++)
		{
			if (cnt[t] > 0) continue;
			size_t far_idx = 0;
			double worst_dot = 1e30;
			for (size_t idx = 0; idx < N_tr; idx++)
			{
				const int ct = labels[idx];
				if (ct < 0 || ct >= T || cnt[ct] < 2) continue;
				double dot = 0.;
				for (int d = 0; d < m; d++) dot += Z[idx * m + d] * cent[(size_t)ct * m + d];
				if (dot < worst_dot) { worst_dot = dot; far_idx = idx; }
			}
			for (int d = 0; d < m; d++) cent[(size_t)t * m + d] = Z[far_idx * m + d];
			cnt[labels[far_idx]]--;
			labels[far_idx] = t;
			cnt[t] = 1;
		}

		bool changed = false;
		for (size_t idx = 0; idx < N_tr; idx++)
		{
			int best_t = 0;
			double best_dot = -1e30;
			for (int t = 0; t < T; t++)
			{
				double dot = 0.;
				for (int d = 0; d < m; d++) dot += Z[idx * m + d] * cent[(size_t)t * m + d];
				if (dot > best_dot) { best_dot = dot; best_t = t; }
			}
			if (labels[idx] != best_t) { labels[idx] = best_t; changed = true; }
		}
		if (!changed) break;
	}
}

static void divisiveSpectralKmeans(const std::vector<double> &Z, size_t N_tr, int m, int T,
                                   std::vector<int> &labels)
{
	labels.assign(N_tr, 0);
	if (N_tr == 0 || m <= 0 || T <= 1) return;

	int k_cur = 1;
	std::vector<double> mu((size_t)T * m, 0.), pdir(m, 0.), cov((size_t)m * m, 0.), tmp_v(m, 0.);
	std::vector<double> sse(T, 0.);
	std::vector<int> cnt(T, 0);

	while (k_cur < T)
	{
		std::fill(mu.begin(), mu.begin() + (size_t)k_cur * m, 0.);
		std::fill(cnt.begin(), cnt.begin() + k_cur, 0);
		for (size_t idx = 0; idx < N_tr; idx++)
		{
			const int c = labels[idx];
			cnt[c]++;
			for (int d = 0; d < m; d++) mu[(size_t)c * m + d] += Z[idx * m + d];
		}
		for (int c = 0; c < k_cur; c++)
		{
			if (cnt[c] > 0)
				for (int d = 0; d < m; d++) mu[(size_t)c * m + d] /= (double)cnt[c];
		}
		std::fill(sse.begin(), sse.begin() + k_cur, 0.);
		for (size_t idx = 0; idx < N_tr; idx++)
		{
			const int c = labels[idx];
			double d2 = 0.;
			for (int d = 0; d < m; d++)
			{
				const double diff = Z[idx * m + d] - mu[(size_t)c * m + d];
				d2 += diff * diff;
			}
			sse[c] += d2;
		}
		int split_c = -1;
		double best_sse = 0.;
		for (int c = 0; c < k_cur; c++)
			if (cnt[c] >= 2 && sse[c] > best_sse) { best_sse = sse[c]; split_c = c; }
		if (split_c < 0) break;

		// Deterministic 1st principal component inside cluster split_c via power iteration
		size_t far_idx = 0;
		double far_d2 = -1.;
		std::fill(cov.begin(), cov.end(), 0.);
		for (size_t idx = 0; idx < N_tr; idx++)
		{
			if (labels[idx] != split_c) continue;
			double d2 = 0.;
			for (int d = 0; d < m; d++)
			{
				const double diff = Z[idx * m + d] - mu[(size_t)split_c * m + d];
				tmp_v[d] = diff;
				d2 += diff * diff;
			}
			if (d2 > far_d2) { far_d2 = d2; far_idx = idx; }
			for (int d1 = 0; d1 < m; d1++)
				for (int d2_i = d1; d2_i < m; d2_i++)
					cov[(size_t)d1 * m + d2_i] += tmp_v[d1] * tmp_v[d2_i];
		}
		for (int d1 = 0; d1 < m; d1++)
			for (int d2_i = 0; d2_i < d1; d2_i++)
				cov[(size_t)d1 * m + d2_i] = cov[(size_t)d2_i * m + d1];

		double pn2 = 0.;
		for (int d = 0; d < m; d++)
		{
			pdir[d] = Z[far_idx * m + d] - mu[(size_t)split_c * m + d];
			pn2 += pdir[d] * pdir[d];
		}
		if (pn2 > 1e-24)
		{
			const double inv = 1.0 / std::sqrt(pn2);
			for (int d = 0; d < m; d++) pdir[d] *= inv;
		}
		else
			pdir[0] = 1.0;

		for (int pit = 0; pit < 12; pit++)
		{
			double vn2 = 0.;
			for (int d1 = 0; d1 < m; d1++)
			{
				double s = 0.;
				for (int d2_i = 0; d2_i < m; d2_i++) s += cov[(size_t)d1 * m + d2_i] * pdir[d2_i];
				tmp_v[d1] = s;
				vn2 += s * s;
			}
			if (vn2 <= 1e-24) break;
			const double inv = 1.0 / std::sqrt(vn2);
			for (int d = 0; d < m; d++) pdir[d] = tmp_v[d] * inv;
		}

		int n_left = 0, n_right = 0;
		for (size_t idx = 0; idx < N_tr; idx++)
		{
			if (labels[idx] != split_c) continue;
			double proj = 0.;
			for (int d = 0; d < m; d++)
				proj += (Z[idx * m + d] - mu[(size_t)split_c * m + d]) * pdir[d];
			if (proj < 0.) { labels[idx] = k_cur; n_right++; }
			else n_left++;
		}
		if (n_left == 0 || n_right == 0)
		{
			int toggle = 0;
			for (size_t idx = 0; idx < N_tr; idx++)
				if (labels[idx] == split_c || labels[idx] == k_cur)
					labels[idx] = (toggle++ % 2 == 0) ? split_c : k_cur;
		}
		k_cur++;
		sphericalKmeansRefine(Z, N_tr, m, k_cur, labels, 5);
	}
	sphericalKmeansRefine(Z, N_tr, m, T, labels, 12);
}

static void deterministicFarthestSpectral(const std::vector<double> &Z, const std::vector<double> &len_w,
                                          size_t N_tr, int m, int T, std::vector<int> &labels)
{
	labels.assign(N_tr, 0);
	if (N_tr == 0 || m <= 0 || T <= 1) return;

	std::vector<double> sl = len_w;
	std::nth_element(sl.begin(), sl.begin() + sl.size() / 2, sl.end());
	const double min_len = std::max(2.0, 0.25 * sl[sl.size() / 2]);

	std::vector<double> gmean(m, 0.);
	for (size_t idx = 0; idx < N_tr; idx++)
		for (int d = 0; d < m; d++) gmean[d] += Z[idx * m + d];
	double gn2 = 0.;
	for (int d = 0; d < m; d++) gn2 += gmean[d] * gmean[d];
	if (gn2 > 1e-24)
	{
		const double inv = 1.0 / std::sqrt(gn2);
		for (int d = 0; d < m; d++) gmean[d] *= inv;
	}

	int c0 = 0;
	double best_dot = -1e30;
	bool any_long = false;
	for (size_t idx = 0; idx < N_tr; idx++) if (len_w[idx] >= min_len) { any_long = true; break; }
	for (size_t idx = 0; idx < N_tr; idx++)
	{
		if (any_long && len_w[idx] < min_len) continue;
		double dot = 0.;
		for (int d = 0; d < m; d++) dot += Z[idx * m + d] * gmean[d];
		if (dot > best_dot) { best_dot = dot; c0 = (int)idx; }
	}
	std::vector<int> centers = {c0};
	std::vector<double> min_d2(N_tr, 1e30);
	for (int t = 1; t < T; t++)
	{
		const int lc = centers.back();
		int chosen = 0;
		double far_d2 = -1.;
		for (size_t idx = 0; idx < N_tr; idx++)
		{
			double dot = 0.;
			for (int d = 0; d < m; d++) dot += Z[idx * m + d] * Z[(size_t)lc * m + d];
			const double d2 = std::max(0., 2.0 * (1.0 - dot));
			if (d2 < min_d2[idx]) min_d2[idx] = d2;
			if (any_long && len_w[idx] < min_len) continue;
			if (min_d2[idx] > far_d2) { far_d2 = min_d2[idx]; chosen = (int)idx; }
		}
		centers.push_back(chosen);
	}
	for (size_t idx = 0; idx < N_tr; idx++)
	{
		int best_t = 0;
		double bd = -1e30;
		for (int t = 0; t < T; t++)
		{
			double dot = 0.;
			for (int d = 0; d < m; d++) dot += Z[idx * m + d] * Z[(size_t)centers[t] * m + d];
			if (dot > bd) { bd = dot; best_t = t; }
		}
		labels[idx] = best_t;
	}
	sphericalKmeansRefine(Z, N_tr, m, T, labels, 12);
}

static void seededKmeansPlusPlus(const std::vector<double> &Z, size_t N_tr, int m, int T,
                                 std::mt19937 &rng, std::vector<int> &labels, bool refine)
{
	labels.assign(N_tr, 0);
	if (N_tr == 0 || m <= 0 || T <= 1) return;
	std::vector<int> centers;
	std::uniform_int_distribution<size_t> pick(0, N_tr - 1);
	centers.push_back((int)pick(rng));
	std::vector<double> min_d2(N_tr, 1e30);
	for (int t = 1; t < T; t++)
	{
		const int lc = centers.back();
		double sum_d2 = 0.;
		for (size_t idx = 0; idx < N_tr; idx++)
		{
			double dot = 0.;
			for (int d = 0; d < m; d++) dot += Z[idx * m + d] * Z[(size_t)lc * m + d];
			const double dist2 = std::max(0., 2.0 * (1.0 - dot));
			if (dist2 < min_d2[idx]) min_d2[idx] = dist2;
			sum_d2 += min_d2[idx];
		}
		if (sum_d2 > 0.)
		{
			std::uniform_real_distribution<double> u01(0., sum_d2);
			double r = u01(rng), acc = 0.;
			int chosen = (int)pick(rng);
			for (size_t idx = 0; idx < N_tr; idx++)
			{
				acc += min_d2[idx];
				if (acc >= r) { chosen = (int)idx; break; }
			}
			centers.push_back(chosen);
		}
		else
			centers.push_back((int)pick(rng));
	}
	for (size_t idx = 0; idx < N_tr; idx++)
	{
		int best_t = 0;
		double best_dot = -1e30;
		for (int t = 0; t < T; t++)
		{
			double dot = 0.;
			for (int d = 0; d < m; d++) dot += Z[idx * m + d] * Z[(size_t)centers[t] * m + d];
			if (dot > best_dot) { best_dot = dot; best_t = t; }
		}
		labels[idx] = best_t;
	}
	if (refine)
		sphericalKmeansRefine(Z, N_tr, m, T, labels, 10);
}

static ChainMixtureModel fitChainMixtureEM(const std::vector<ChainSuffStat> &stats, int C, double kappa,
                                          const std::vector<long int> &train, int T, unsigned seed,
                                          const ChainSpectralBasis *precomp_spec = NULL)
{
	ChainMixtureModel best;
	best.T = T; best.C = C; best.kappa = kappa;
	best.logw.assign(T, -std::log((double)std::max(1, T)));
	best.mu.assign((size_t)T * C, 1.0 / std::max(1, C));
	best.B.assign((size_t)T * C * C, 1.0 / std::max(1, C));
	const double log_unif = std::log(std::max(1e-12, 1.0 / std::max(1, C)));
	best.log_mu.assign((size_t)T * C, log_unif);
	best.log_B.assign((size_t)T * C * C, log_unif);
	if (train.empty() || C <= 0) return best;

	auto mStepFromGamma = [&](const std::vector<double> &gamma, ChainMixtureModel &M)
	{
		std::vector<double> w(T, 0.);
		std::vector<double> uni((size_t)T * C, 0.), tr((size_t)T * C * C, 0.);
		std::vector<char> row_active((size_t)T * C, 0);
		for (size_t idx = 0; idx < train.size(); idx++)
		{
			const ChainSuffStat &st = stats[train[idx]];
			for (int t = 0; t < T; t++)
			{
				const double g = gamma[idx * T + t];
				w[t] += g;
				double *ut = &uni[(size_t)t * C];
				for (const auto &kv : st.unigram) ut[kv.first] += g * kv.second;
				if (g > 0.)
				{
					double *trt = &tr[(size_t)t * C * C];
					for (const auto &kv : st.trans)
					{
						trt[kv.first] += g * kv.second;
						row_active[(size_t)t * C + kv.first / C] = 1;
					}
				}
			}
		}
		double wtot = 0.;
		for (int t = 0; t < T; t++) wtot += (w[t] + 0.5);
		for (int t = 0; t < T; t++)
		{
			M.logw[t] = std::log((w[t] + 0.5) / wtot);
			double us = 0.;
			const double eps_mu = 1e-2 / C;
			for (int c = 0; c < C; c++) { M.mu[(size_t)t * C + c] = uni[(size_t)t * C + c] + eps_mu; us += M.mu[(size_t)t * C + c]; }
			for (int c = 0; c < C; c++)
			{
				M.mu[(size_t)t * C + c] /= us;
				M.log_mu[(size_t)t * C + c] = std::log(std::max(1e-12, M.mu[(size_t)t * C + c]));
			}
			const double alpha0 = std::max(1.0, 0.15 * C);
			const double *mu_t = &M.mu[(size_t)t * C];
			const double *lmu_t = &M.log_mu[(size_t)t * C];
			for (int i = 0; i < C; i++)
			{
				double *row = &M.B[((size_t)t * C + i) * C];
				double *lrow = &M.log_B[((size_t)t * C + i) * C];
				if (!row_active[(size_t)t * C + i])
				{
					std::copy(mu_t, mu_t + C, row);
					std::copy(lmu_t, lmu_t + C, lrow);
					continue;
				}
				double rs = 0.;
				for (int j = 0; j < C; j++)
				{
					row[j] = tr[((size_t)t * C + i) * C + j] + alpha0 * mu_t[j];
					rs += row[j];
				}
				for (int j = 0; j < C; j++)
				{
					row[j] /= rs;
					lrow[j] = std::log(std::max(1e-12, row[j]));
				}
			}
		}
	};

	if (T == 1)
	{
		std::vector<double> gamma(train.size(), 1.0);
		mStepFromGamma(gamma, best);
		return best;
	}

	ChainSpectralBasis local_spec;
	const ChainSpectralBasis *spec_ptr = precomp_spec;
	if (spec_ptr == NULL)
	{
		buildChainSpectralBasis(stats, C, train, T, local_spec);
		spec_ptr = &local_spec;
	}
	const ChainSpectralBasis &spec = *spec_ptr;
	const size_t N_tr = train.size();
	const int m_comb = std::min(spec.max_m_comb, T);
	const int m_uni = std::min(spec.max_m_uni, T);
	std::vector<double> Z_comb, Z_uni;
	extractNormalizedSlice(spec.z_comb_all, N_tr, spec.max_m_comb, m_comb, Z_comb);
	extractNormalizedSlice(spec.z_uni_all, N_tr, spec.max_m_uni, m_uni, Z_uni);

	double best_ll = -1e300;
	std::mt19937 rng(seed);
	const int n_starts = 8;
	std::vector<double> L(T), gamma(N_tr * T);
	std::vector<std::vector<int> > seen_inits;

	for (int st_idx = 0; st_idx < n_starts; st_idx++)
	{
		std::vector<int> labels_init;
		if (st_idx == 0)
			divisiveSpectralKmeans(Z_comb, N_tr, m_comb, T, labels_init);
		else if (st_idx == 1)
			divisiveSpectralKmeans(Z_uni, N_tr, m_uni, T, labels_init);
		else if (st_idx == 2)
			deterministicFarthestSpectral(Z_comb, spec.len_w, N_tr, m_comb, T, labels_init);
		else if (st_idx <= 6)
			seededKmeansPlusPlus(Z_comb, N_tr, m_comb, T, rng, labels_init, true);
		else
			seededKmeansPlusPlus(spec.feat, N_tr, spec.D, T, rng, labels_init, false);

		// Skip duplicate initializations (common when spectral restarts agree identically)
		bool dup = false;
		for (const auto &prev : seen_inits)
			if (prev == labels_init) { dup = true; break; }
		if (dup && st_idx < n_starts - 1) continue;
		seen_inits.push_back(labels_init);

		const double off_w = (T > 1) ? (0.15 / (double)(T - 1)) : 0.;
		for (size_t idx = 0; idx < N_tr; idx++)
			for (int t = 0; t < T; t++)
				gamma[idx * T + t] = (t == labels_init[idx]) ? 0.85 : off_w;

		ChainMixtureModel M;
		M.T = T; M.C = C; M.kappa = kappa;
		M.logw.assign(T, 0.); M.mu.assign((size_t)T * C, 0.); M.B.assign((size_t)T * C * C, 0.);
		M.log_mu.assign((size_t)T * C, 0.); M.log_B.assign((size_t)T * C * C, 0.);
		double cur_ll = -1e300;
		for (int em = 0; em < 30; em++)
		{
			mStepFromGamma(gamma, M);
			double tot_ll = 0., max_chg = 0.;
			for (size_t idx = 0; idx < N_tr; idx++)
			{
				evalChainLogLik(M, stats[train[idx]], L.data());
				const double lz = chainLogSumExp(L.data(), T);
				tot_ll += lz;
				for (int t = 0; t < T; t++)
				{
					const double gnew = std::exp(L[t] - lz);
					max_chg = std::max(max_chg, std::fabs(gnew - gamma[idx * T + t]));
					gamma[idx * T + t] = gnew;
				}
			}
			cur_ll = tot_ll;
			if (em >= 4 && max_chg < 1e-4) break;
		}
		if (cur_ll > best_ll) { best_ll = cur_ll; best = M; }
	}
	return best;
}
} // anonymous namespace

namespace helixtypes
{
// Adjusted Rand index of two labellings (labels >= 0; entries with a negative label in either are left out)
double adjustedRand(const std::vector<int> &a, const std::vector<int> &b)
{
	std::map<std::pair<int, int>, double> nab;
	std::map<int, double> na, nb;
	double n = 0.;
	for (size_t i = 0; i < a.size(); i++)
		if (a[i] >= 0 && b[i] >= 0)
			nab[{a[i], b[i]}] += 1., na[a[i]] += 1., nb[b[i]] += 1., n += 1.;
	auto pairs = [](double x) { return 0.5 * x * (x - 1.); };
	double sab = 0., sa = 0., sb = 0.;
	for (auto &x : nab) sab += pairs(x.second);
	for (auto &x : na) sa += pairs(x.second);
	for (auto &x : nb) sb += pairs(x.second);
	const double expected = (n > 1.) ? sa * sb / pairs(n) : 0., top = 0.5 * (sa + sb);
	return (top > expected + 1e-12) ? (sab - expected) / (top - expected) : 0.;
}

// Mixture of multinomials over the class counts of filaments (sparse rows), for choosing the number of types
struct Mixture
{
	std::vector<double> logp;  // T x C
	std::vector<double> logw;  // T
};

// log sum exp of T values
static double lse(const double *a, int T)
{
	double m = a[0];
	for (int t = 1; t < T; t++) m = std::max(m, a[t]);
	double s = 0.;
	for (int t = 0; t < T; t++) s += exp(a[t] - m);
	return m + log(s);
}

static void mixtureLogLik(const FilamentCounts &X, const Mixture &M, int T, long int f, double *L)
{
	for (int t = 0; t < T; t++)
	{
		double l = M.logw[t];
		for (const auto &e : X.rows[f]) l += e.second * M.logp[t * X.C + e.first];
		L[t] = l;
	}
}

// EM from `restarts` random starts (each a random training filament plus a pseudo-count); the best by log-likelihood
static Mixture fitMixture(const FilamentCounts &X, const std::vector<long int> &train, int T, unsigned seed, int restarts = 6, int max_iter = 300)
{
	std::vector<Mixture> fits(restarts);
	std::vector<double> fit_ll(restarts, -std::numeric_limits<double>::infinity());
	#pragma omp parallel for schedule(dynamic)
	for (int r = 0; r < restarts; r++)
	{
		std::mt19937 rng(seed * 7919u + (unsigned)r * 104729u + (unsigned)T);
		std::uniform_int_distribution<long int> pick(0, (long int)train.size() - 1);
		Mixture M;
		M.logp.assign((size_t)T * X.C, 0.);
		M.logw.assign(T, -log((double)T));
		for (int t = 0; t < T; t++)
		{
			std::vector<double> p(X.C, 1.);
			for (const auto &e : X.rows[train[pick(rng)]]) p[e.first] += e.second;
			double s = 0.;
			for (double v : p) s += v;
			for (int c = 0; c < X.C; c++) M.logp[t * X.C + c] = log(p[c] / s);
		}
		std::vector<double> L(T), num((size_t)T * X.C), wsum(T);
		double last = -std::numeric_limits<double>::infinity(), ll = last;
		for (int it = 0; it < max_iter; it++)
		{
			std::fill(num.begin(), num.end(), 0.1);
			std::fill(wsum.begin(), wsum.end(), 0.);
			ll = 0.;
			for (long int f : train)
			{
				mixtureLogLik(X, M, T, f, L.data());
				const double z = lse(L.data(), T);
				ll += z;
				for (int t = 0; t < T; t++)
				{
					const double g = exp(L[t] - z);
					wsum[t] += g;
					for (const auto &e : X.rows[f]) num[t * X.C + e.first] += g * e.second;
				}
			}
			for (int t = 0; t < T; t++)
			{
				double s = 0.;
				for (int c = 0; c < X.C; c++) s += num[t * X.C + c];
				for (int c = 0; c < X.C; c++) M.logp[t * X.C + c] = log(num[t * X.C + c] / s);
				M.logw[t] = log(std::max(1e-12, wsum[t] / train.size()));
			}
			if (ll - last < 1e-7 * std::fabs(ll))
				break;
			last = ll;
		}
		fits[r] = M;
		fit_ll[r] = ll;
	}
	int best = 0;
	for (int r = 1; r < restarts; r++)
		if (fit_ll[r] > fit_ll[best]) best = r;
	return fits[best];
}

void scan(const FilamentCounts &X, int tmin, int tmax, int nfold, int nsplit, std::vector<double> &score, std::vector<double> &agree)
{
	score.assign(tmax + 1, 0.);
	agree.assign(tmax + 1, 0.);
	std::vector<long int> used;
	for (long int f = 0; f < (long int)X.rows.size(); f++)
		if (!X.rows[f].empty()) used.push_back(f);
	const long int F = used.size();
	if (F < 3 * nfold || X.C < 2 || nsplit < 1)
		return;
	for (int split = 0; split < nsplit; split++)
	{
		// folds by a shuffle, not by f % nfold: filaments come in micrograph order, which can follow the types
		std::vector<int> fold_of(F);
		for (long int i = 0; i < F; i++) fold_of[i] = i % nfold;
		std::shuffle(fold_of.begin(), fold_of.end(), std::mt19937(12345u + 7919u * split));
		for (int T = tmin; T <= tmax; T++)
		{
			std::vector<std::vector<int> > labels(nfold, std::vector<int>(F));
			double test_ll = 0., test_segments = 0.;
			for (int fold = 0; fold < nfold; fold++)
			{
				std::vector<long int> train;
				for (long int i = 0; i < F; i++)
					if (fold_of[i] != fold) train.push_back(used[i]);
				const Mixture M = fitMixture(X, train, T, 1u + fold + nfold * split);
				std::vector<double> L(T);
				for (long int i = 0; i < F; i++)
				{
					mixtureLogLik(X, M, T, used[i], L.data());
					labels[fold][i] = std::max_element(L.begin(), L.end()) - L.begin();
					if (fold_of[i] == fold)
						test_ll += lse(L.data(), T), test_segments += X.total[used[i]];
				}
			}
			score[T] += test_ll / std::max(1., test_segments) / nsplit;
			double a = 0.;
			int npair = 0;
			for (int f1 = 0; f1 < nfold; f1++)
				for (int f2 = f1 + 1; f2 < nfold; f2++, npair++)
					a += adjustedRand(labels[f1], labels[f2]);
			agree[T] += a / std::max(1, npair) / nsplit;
		}
	}
}

std::vector<int> fitTypes(const FilamentCounts &X, int T, unsigned seed)
{
	std::vector<long int> train;
	for (long int f = 0; f < (long int)X.rows.size(); f++)
		if (!X.rows[f].empty()) train.push_back(f);
	std::vector<int> type(X.rows.size(), 0);
	if (train.empty() || T < 2)
		return type;
	const Mixture M = fitMixture(X, train, T, seed);
	std::vector<double> L(T);
	for (long int f = 0; f < (long int)X.rows.size(); f++)
	{
		mixtureLogLik(X, M, T, f, L.data());
		type[f] = std::max_element(L.begin(), L.end()) - L.begin();
	}
	return type;
}

int choose(const std::vector<double> &score, const std::vector<double> &agree, int tmin, int tmax, double tol,
           double agree_min, int current, std::string &why)
{
	// More types than the held-out log-likelihood asks for only split real types (the folds can still agree on
	// such splits): at most the fewest types within tol (nats per segment) of the best
	int top = tmin, tll = tmin;
	for (int T = tmin; T <= tmax; T++)
		if (score[T] > score[top]) top = T;
	for (int T = tmax; T >= tmin; T--)
		if (score[T] >= score[top] - tol) tll = T;
	int best = -1;
	for (int T = tmin; T <= tll; T++)
		if (agree[T] >= agree_min)
			best = T;
	if (best < 0 && current >= tmin && current <= tmax)
		best = current, why = "none agrees to " + floatToString(agree_min) + ": keeping the current number";
	else if (best < 0)
	{
		best = tmin;
		for (int T = tmin; T <= tll; T++)
			if (agree[T] > agree[best]) best = T;
		why = "none agrees to " + floatToString(agree_min) + ": the best agreement";
	}
	else
		why = "the most with agreement of at least " + floatToString(agree_min);
	why += "; at most " + integerToString(tll) + ", the fewest within " + floatToString(tol) + " of the best log-likelihood";
	return best;
}

std::vector<double> typeJaccard(const std::vector<std::vector<int> > &labels, int T)
{
	const int R = labels.size();
	std::vector<double> out;
	for (int a = 0; a < R; a++)
		for (int t = 0; t < T; t++)
		{
			double sum = 0.;
			bool any = false;
			for (int b = 0; b < R; b++)
			{
				if (b == a) continue;
				// filaments labelled in both: of type t in a, of each type in b, and in both
				std::vector<long int> nb(T, 0), both(T, 0);
				long int na = 0;
				for (size_t f = 0; f < labels[a].size(); f++)
				{
					const int la = labels[a][f], lb = labels[b][f];
					if (la < 0 || lb < 0 || lb >= T) continue;
					nb[lb]++;
					if (la == t) na++, both[lb]++;
				}
				if (na == 0) continue;
				double best = 0.;
				for (int u = 0; u < T; u++)
					best = std::max(best, (double)both[u] / (na + nb[u] - both[u]));
				sum += best;
				any = true;
			}
			if (any) out.push_back(sum / (R - 1));
		}
	return out;
}

void typeJaccardStats(const std::vector<std::vector<int> > &labels, int T,
                      double &min_jacc, double &second_min_jacc, double &weighted_jacc)
{
	const int R = (int)labels.size();
	min_jacc = 1.0;
	second_min_jacc = 1.0;
	weighted_jacc = 0.0;
	if (R <= 1 || T <= 0) { min_jacc = 0.; second_min_jacc = 0.; return; }

	bool any_all = false;
	double w_num = 0., w_den = 0., sec_sum = 0.;
	int sec_cnt = 0;
	for (int a = 0; a < R; a++)
	{
		std::vector<double> fold_jaccs;
		for (int t = 0; t < T; t++)
		{
			double sum = 0.;
			long int na_ref = 0;
			int nb_other = 0;
			for (int b = 0; b < R; b++)
			{
				if (b == a) continue;
				std::vector<long int> nb(T, 0), both(T, 0);
				long int na = 0;
				for (size_t f = 0; f < labels[a].size(); f++)
				{
					const int la = labels[a][f], lb = labels[b][f];
					if (la < 0 || lb < 0 || lb >= T) continue;
					nb[lb]++;
					if (la == t) na++, both[lb]++;
				}
				if (na == 0) continue;
				double best = 0.;
				for (int u = 0; u < T; u++)
					best = std::max(best, (double)both[u] / (na + nb[u] - both[u]));
				sum += best;
				na_ref = std::max(na_ref, na);
				nb_other++;
			}
			if (nb_other > 0)
			{
				const double j = sum / nb_other;
				fold_jaccs.push_back(j);
				min_jacc = std::min(min_jacc, j);
				w_num += (double)na_ref * j;
				w_den += (double)na_ref;
				any_all = true;
			}
		}
		if (!fold_jaccs.empty())
		{
			std::sort(fold_jaccs.begin(), fold_jaccs.end());
			sec_sum += (fold_jaccs.size() >= 2) ? fold_jaccs[1] : fold_jaccs[0];
			sec_cnt++;
		}
	}
	if (!any_all)
	{
		min_jacc = 0.;
		second_min_jacc = 0.;
		weighted_jacc = 0.;
		return;
	}
	second_min_jacc = (sec_cnt > 0) ? (sec_sum / sec_cnt) : min_jacc;
	weighted_jacc = (w_den > 0.) ? (w_num / w_den) : min_jacc;
}

std::vector<int> fitChainTypes(const FilamentChains &X, int T, unsigned seed, std::vector<double> *post_out)
{
	const int F = (int)X.seqs.size();
	std::vector<int> type(F, 0);
	if (post_out) post_out->assign((size_t)F * std::max(1, T), 0.);
	if (F == 0 || X.C <= 0) return type;

	std::vector<long int> train;
	for (long int f = 0; f < F; f++)
		if (!X.seqs[f].empty()) train.push_back(f);
	if (train.empty()) return type;

	std::vector<ChainSuffStat> stats;
	buildChainStats(X, stats);
	const ChainMixtureModel M = fitChainMixtureEM(stats, X.C, X.kappa, train, std::max(1, T), seed);
	std::vector<double> L(std::max(1, T));
	for (long int f = 0; f < F; f++)
	{
		evalChainLogLik(M, stats[f], L.data());
		type[f] = (int)(std::max_element(L.begin(), L.end()) - L.begin());
		if (post_out)
		{
			const double lz = chainLogSumExp(L.data(), std::max(1, T));
			for (int t = 0; t < std::max(1, T); t++)
				(*post_out)[(size_t)f * std::max(1, T) + t] = std::exp(L[t] - lz);
		}
	}
	return type;
}

void scanChains(const FilamentChains &X, int tmin, int tmax, int nfold, int nsplit,
                std::vector<double> &score, std::vector<double> &agree,
                std::vector<double> *min_jacc,
                std::vector<double> *weighted_jacc,
                std::vector<double> *second_min_jacc)
{
	score.assign(tmax + 1, 0.);
	agree.assign(tmax + 1, 0.);
	if (min_jacc) min_jacc->assign(tmax + 1, 0.);
	if (weighted_jacc) weighted_jacc->assign(tmax + 1, 0.);
	if (second_min_jacc) second_min_jacc->assign(tmax + 1, 0.);

	std::vector<long int> nonempty;
	for (long int f = 0; f < (long int)X.seqs.size(); f++)
		if (!X.seqs[f].empty()) nonempty.push_back(f);
	const int nf = (int)nonempty.size();
	if (nf < nfold || X.C <= 0) return;

	std::vector<ChainSuffStat> stats;
	buildChainStats(X, stats);

	// Check whether micrograph stratification is valid (avoid unit tests with 1 shared prefix like "mic:")
	std::map<std::string, std::vector<long int> > mic_groups;
	bool use_mic_strat = ((int)X.mics.size() == (int)X.seqs.size());
	if (use_mic_strat)
	{
		for (long int f : nonempty)
		{
			if (X.mics[f].empty()) { use_mic_strat = false; break; }
			mic_groups[X.mics[f]].push_back(f);
		}
		if ((int)mic_groups.size() < 3 * nfold) use_mic_strat = false;
		if (use_mic_strat)
			for (const auto &kv : mic_groups)
				if ((int)kv.second.size() > 0.35 * nf) { use_mic_strat = false; break; }
	}

	std::vector<std::string> mic_keys;
	if (use_mic_strat)
		for (const auto &kv : mic_groups) mic_keys.push_back(kv.first);

	std::vector<std::vector<int> > fold_of_f_all(nsplit, std::vector<int>(X.seqs.size(), -1));
	for (int split = 0; split < nsplit; split++)
	{
		std::mt19937 rng(100u + (unsigned)split);
		if (use_mic_strat)
		{
			std::vector<std::string> mk = mic_keys;
			std::shuffle(mk.begin(), mk.end(), rng);
			for (size_t i = 0; i < mk.size(); i++)
				for (long int f : mic_groups[mk[i]])
					fold_of_f_all[split][f] = (int)(i % nfold);
		}
		else
		{
			std::vector<long int> perm = nonempty;
			std::shuffle(perm.begin(), perm.end(), rng);
			for (int i = 0; i < nf; i++) fold_of_f_all[split][perm[i]] = i % nfold;
		}
	}

	// Precompute (split, fold) training subsets and normalized co-occurrence spectral bases once across all T
	std::vector<std::vector<long int> > train_cache((size_t)nsplit * nfold);
	std::vector<std::vector<long int> > test_cache((size_t)nsplit * nfold);
	std::vector<ChainSpectralBasis> spec_cache((size_t)nsplit * nfold);
	#pragma omp parallel for collapse(2) schedule(dynamic, 1)
	for (int split = 0; split < nsplit; split++)
	{
		for (int fold = 0; fold < nfold; fold++)
		{
			const size_t sf = (size_t)split * nfold + fold;
			const std::vector<int> &fold_of_f = fold_of_f_all[split];
			for (long int f : nonempty)
			{
				if (fold_of_f[f] == fold) test_cache[sf].push_back(f);
				else train_cache[sf].push_back(f);
			}
			if (tmax >= 2)
				buildChainSpectralBasis(stats, X.C, train_cache[sf], tmax, spec_cache[sf]);
		}
	}

	const int span_T = tmax - tmin + 1;
	std::vector<double> score_s((size_t)nsplit * span_T, 0.);
	std::vector<double> agree_s((size_t)nsplit * span_T, 0.);
	std::vector<double> jacc_s((size_t)nsplit * span_T, 0.);
	std::vector<double> wjacc_s((size_t)nsplit * span_T, 0.);
	std::vector<double> jacc2_s((size_t)nsplit * span_T, 0.);

	#pragma omp parallel for collapse(2) schedule(dynamic, 1)
	for (int split = 0; split < nsplit; split++)
	{
		for (int T = tmin; T <= tmax; T++)
		{
			const size_t cell = (size_t)split * span_T + (T - tmin);
			std::vector<std::vector<int> > labels(nfold, std::vector<int>(nf, -1));
			double test_ll = 0., test_segs = 0.;
			std::vector<double> L(T);
			for (int fold = 0; fold < nfold; fold++)
			{
				const size_t sf = (size_t)split * nfold + fold;
				const std::vector<long int> &train = train_cache[sf];
				const std::vector<long int> &test = test_cache[sf];
				const ChainSpectralBasis *sptr = (T >= 2) ? &spec_cache[sf] : NULL;
				const ChainMixtureModel M = fitChainMixtureEM(stats, X.C, X.kappa, train, T, 1000u + 17u * split + (unsigned)fold, sptr);
				for (long int f : test)
				{
					evalChainLogLik(M, stats[f], L.data());
					test_ll += chainLogSumExp(L.data(), T);
					test_segs += (double)stats[f].len;
				}
				for (int idx = 0; idx < nf; idx++)
				{
					evalChainLogLik(M, stats[nonempty[idx]], L.data());
					labels[fold][idx] = (int)(std::max_element(L.begin(), L.end()) - L.begin());
				}
			}
			score_s[cell] = test_ll / std::max(1.0, X.kappa * test_segs) / nsplit;
			if (T == 1)
			{
				agree_s[cell] = 0.;
				if (min_jacc) jacc_s[cell] = 1.0 / nsplit;
				if (weighted_jacc) wjacc_s[cell] = 1.0 / nsplit;
				if (second_min_jacc) jacc2_s[cell] = 1.0 / nsplit;
			}
			else
			{
				// Rare-type-safe degeneracy check: allows legitimate 1-2% polymorphs while rejecting singleton splits
				const int min_cnt = (nf >= 10 * T) ? std::max(2, std::min(30, nf / (30 * T)))
				                                  : ((nf >= 4 * T) ? 2 : 1);
				bool degen = false;
				for (int fold = 0; fold < nfold && !degen; fold++)
				{
					std::vector<int> cnt(T, 0);
					for (int idx = 0; idx < nf; idx++)
						if (labels[fold][idx] >= 0 && labels[fold][idx] < T)
							cnt[labels[fold][idx]]++;
					for (int t = 0; t < T; t++)
						if (cnt[t] < min_cnt) { degen = true; break; }
				}
				if (!degen)
				{
					double a = 0.;
					int npair = 0;
					for (int f1 = 0; f1 < nfold; f1++)
						for (int f2 = f1 + 1; f2 < nfold; f2++, npair++)
							a += adjustedRand(labels[f1], labels[f2]);
					agree_s[cell] = a / std::max(1, npair) / nsplit;
					if (min_jacc || weighted_jacc || second_min_jacc)
					{
						double mj = 0., j2 = 0., wj = 0.;
						typeJaccardStats(labels, T, mj, j2, wj);
						jacc_s[cell] = mj / nsplit;
						wjacc_s[cell] = wj / nsplit;
						jacc2_s[cell] = j2 / nsplit;
					}
				}
			}
		}
	}

	for (int split = 0; split < nsplit; split++)
		for (int T = tmin; T <= tmax; T++)
		{
			const size_t cell = (size_t)split * span_T + (T - tmin);
			score[T] += score_s[cell];
			agree[T] += agree_s[cell];
			if (min_jacc) (*min_jacc)[T] += jacc_s[cell];
			if (weighted_jacc) (*weighted_jacc)[T] += wjacc_s[cell];
			if (second_min_jacc) (*second_min_jacc)[T] += jacc2_s[cell];
		}
}

std::vector<RFLOAT> classContinuity(const std::vector<MultidimArray<RFLOAT> > &refs, RFLOAT apix, const std::vector<long int> &filament,
                                    const std::vector<RFLOAT> &track_angst, const std::vector<int> &cls,
                                    const std::vector<char> &flipped, std::string &note, RFLOAT outer_diam)
{
	const int K = refs.size();
	std::vector<RFLOAT> cont(K, std::numeric_limits<RFLOAT>::quiet_NaN());
	note = "no pairs of segments";
	if (K == 0 || YSIZE(refs[0]) < 8 || ZSIZE(refs[0]) != 1) return cont;
	const int ny = YSIZE(refs[0]), nx = XSIZE(refs[0]), max_shift = (int)(0.4 * nx);
	const int dy_max = std::max(2, (int)ROUND(12. / std::max((RFLOAT)0.5, apix)));
	auto flip = [&](long int i) { return !flipped.empty() && flipped[i]; };

	// segments in track order per filament; each paired with the one lag segments further (about 15% of the box)
	std::vector<long int> seg;
	for (size_t i = 0; i < cls.size(); i++)
		if (filament[i] >= 0 && cls[i] >= 0 && cls[i] < K) seg.push_back(i);
	std::sort(seg.begin(), seg.end(), [&](long int a, long int b)
	          { return filament[a] != filament[b] ? filament[a] < filament[b] : track_angst[a] < track_angst[b]; });
	std::vector<RFLOAT> steps;
	for (size_t j = 1; j < seg.size(); j++)
		if (filament[seg[j]] == filament[seg[j - 1]] && track_angst[seg[j]] > track_angst[seg[j - 1]])
			steps.push_back(track_angst[seg[j]] - track_angst[seg[j - 1]]);
	if (steps.empty()) return cont;
	std::nth_element(steps.begin(), steps.begin() + steps.size() / 2, steps.end());
	const RFLOAT step = steps[steps.size() / 2];
	const int lag = std::max(1, (int)ROUND(0.15 * nx * apix / step));
	struct Pair { int a, b; RFLOAT d; };  // classes of the earlier and the later segment, distance (pixels, sign of the polarity)
	std::vector<Pair> pairs;
	for (size_t j = 0; j + lag < seg.size(); j++)
	{
		const long int s = seg[j], t = seg[j + lag];
		if (filament[s] == filament[t] && flip(s) == flip(t))
			pairs.push_back({cls[s], cls[t], (flip(s) ? -1. : 1.) * (track_angst[t] - track_angst[s]) / apix});
	}
	if (pairs.empty()) return cont;

	// class averages: low-passed (>=22 A to suppress 4.75A aliasing), zero mean inside tube/circle mask
	const double r2max = 0.45 * 0.45 * nx * nx;
	const double hy_max = (outer_diam > 0.) ? (0.60 * outer_diam / apix) : 1e9;
	std::vector<char> inside(ny * nx);
	for (int i = 0; i < ny; i++)
		for (int j = 0; j < nx; j++)
		{
			const double dy_c = i - ny / 2, dx_c = j - nx / 2;
			inside[i * nx + j] = (dy_c * dy_c + dx_c * dx_c < r2max) && (std::fabs(dy_c) <= hy_max);
		}
	std::vector<std::vector<double> > img(K, std::vector<double>(ny * nx, 0.));
	const RFLOAT lp_res = std::max((RFLOAT)22., (RFLOAT)(2.5 * apix));
	for (int k = 0; k < K; k++)
	{
		MultidimArray<RFLOAT> f = refs[k];
		lowPassFilterMap(f, lp_res, apix);
		double sum = 0.;
		long int n = 0;
		for (int p = 0; p < ny * nx; p++)
			if (inside[p]) sum += DIRECT_MULTIDIM_ELEM(f, p), n++;
		for (int p = 0; p < ny * nx; p++)
			img[k][p] = (inside[p] && n > 0) ? DIRECT_MULTIDIM_ELEM(f, p) - sum / n : 0.;
	}
	// overlap of a and b shifted by sx: sum a(r) b(r + s) / sqrt(energies in the overlap), best over dy
	auto overlap = [&](int a, int b, int sx)
	{
		double best = -1.;
		for (int dy = -dy_max; dy <= dy_max; dy++)
		{
			double ab = 0., aa = 0., bb = 0.;
			for (int i = std::max(0, -dy); i < std::min(ny, ny - dy); i++)
				for (int j = std::max(0, -sx); j < std::min(nx, nx - sx); j++)
				{
					const int p = i * nx + j, q = (i + dy) * nx + j + sx;
					if (!inside[p] || !inside[q]) continue;
					const double va = img[a][p], vb = img[b][q];
					ab += va * vb, aa += va * va, bb += vb * vb;
				}
			if (aa > 0. && bb > 0.) best = std::max(best, ab / sqrt(aa * bb));
		}
		return best;
	};
	const long long W = 2 * max_shift + 1;
	std::map<long long, double> cache;
	auto shiftOf = [&](const Pair &pr, int sign) { return (int)ROUND(sign * pr.d); };
	auto keyOf = [&](const Pair &pr, int s) { return ((long long)pr.a * K + pr.b) * W + s + max_shift; };
	auto fill = [&](const std::vector<long long> &keys)
	{
		std::vector<double> val(keys.size());
		#pragma omp parallel for schedule(dynamic, 16)
		for (long int n = 0; n < (long int)keys.size(); n++)
		{
			const long long kk = keys[n];
			val[n] = overlap((int)(kk / W / K), (int)(kk / W % K), (int)(kk % W) - max_shift);
		}
		for (size_t n = 0; n < keys.size(); n++) cache[keys[n]] = val[n];
	};
	auto need = [&](const std::vector<size_t> &which, int sign)
	{
		std::vector<long long> keys;
		for (size_t n : which)
		{
			const int s = shiftOf(pairs[n], sign);
			if (abs(s) <= max_shift && !cache.count(keyOf(pairs[n], s))) keys.push_back(keyOf(pairs[n], s));
		}
		std::sort(keys.begin(), keys.end());
		keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
		fill(keys);
	};
	auto median = [](std::vector<double> v)
	{
		if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
		std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
		return v[v.size() / 2];
	};
	// the direction of the track in the class frame: the one with the higher median overlap, on a sample of pairs
	std::vector<size_t> sample;
	for (size_t n = 0; n < pairs.size(); n += std::max((size_t)1, pairs.size() / 3000)) sample.push_back(n);
	double med[2];
	for (int si = 0; si < 2; si++)
	{
		need(sample, si ? 1 : -1);
		std::vector<double> v;
		for (size_t n : sample)
		{
			const int s = shiftOf(pairs[n], si ? 1 : -1);
			if (abs(s) <= max_shift) v.push_back(cache[keyOf(pairs[n], s)]);
		}
		med[si] = median(v);
	}
	const int sign = (med[1] > med[0]) ? 1 : -1;
	std::vector<size_t> all(pairs.size());
	for (size_t n = 0; n < pairs.size(); n++) all[n] = n;
	need(all, sign);
	std::vector<std::vector<double> > per(K);
	for (const Pair &pr : pairs)
	{
		const int s = shiftOf(pr, sign);
		if (abs(s) > max_shift) continue;
		const double v = cache[keyOf(pr, s)];
		per[pr.a].push_back(v);
		if (pr.b != pr.a) per[pr.b].push_back(v);
	}
	for (int k = 0; k < K; k++)
		if (per[k].size() >= 20) cont[k] = median(per[k]);
	std::ostringstream os;
	os << pairs.size() << " pairs of segments " << lag << " apart (" << ROUND(lag * step) << " A), the track along "
	   << (sign > 0 ? "+x" : "-x") << " in the class frame (median overlap " << std::setprecision(3) << std::max(med[0], med[1])
	   << ", the other way " << std::min(med[0], med[1]) << ")";
	note = os.str();
	return cont;
}

std::vector<char> junkClasses(const std::vector<RFLOAT> &res, const std::vector<RFLOAT> &diam, RFLOAT box_angst, RFLOAT res_factor,
                              const std::vector<RFLOAT> &cont, const std::vector<RFLOAT> &pdf)
{
	const int K = (int)res.size();
	RFLOAT med_res = 0.;
	if ((int)pdf.size() == K)
	{
		std::vector<std::pair<RFLOAT, RFLOAT> > witems;
		double wtot = 0.;
		for (int k = 0; k < K; k++)
			if (pdf[k] > 1e-5 && res[k] > 0. && std::isfinite(res[k]))
			{
				witems.push_back({res[k], pdf[k]});
				wtot += pdf[k];
			}
		if (!witems.empty() && wtot > 0.)
		{
			std::sort(witems.begin(), witems.end());
			double acc = 0.;
			for (size_t idx = 0; idx < witems.size(); idx++)
			{
				acc += witems[idx].second;
				if (acc >= 0.5 * wtot) { med_res = witems[idx].first; break; }
			}
		}
	}
	if (!(med_res > 0.))
	{
		std::vector<RFLOAT> ok;
		for (RFLOAT r : res)
			if (r > 0. && std::isfinite(r)) ok.push_back(r);
		std::sort(ok.begin(), ok.end());
		if (!ok.empty()) med_res = ok[ok.size() / 2];
	}
	const RFLOAT junk_res = (med_res > 0.) ? res_factor * med_res : 0.;
	std::vector<char> base_junk(K, 0);
	int n_valid = 0, n_cont_bad = 0;
	std::vector<RFLOAT> valid_cont;
	for (int k = 0; k < K; k++)
	{
		base_junk[k] = (junk_res > 0. && !(res[k] > 0. && std::isfinite(res[k]) && res[k] <= junk_res))
		             || (k < (int)diam.size() && diam[k] > 0.8 * box_angst)
		             || ((int)pdf.size() == K && !(pdf[k] > 0.));
		if (!base_junk[k])
		{
			n_valid++;
			if (k < (int)cont.size() && std::isfinite(cont[k]))
			{
				valid_cont.push_back(cont[k]);
				if (cont[k] < 0.6) n_cont_bad++;
			}
		}
	}
	RFLOAT cont_cut = 0.6;
	const int min_survive = std::max(4, (n_valid + 1) / 2);
	if (K >= 8 && !valid_cont.empty())
	{
		std::vector<RFLOAT> sc = valid_cont;
		std::sort(sc.begin(), sc.end());
		const RFLOAT med_c = sc[sc.size() / 2];
		cont_cut = std::min((RFLOAT)0.6, std::max((RFLOAT)0.30, (RFLOAT)0.55 * med_c));
		// Also guarantee at least min_survive classes remain by capping cutoff at the min_survive-th highest continuity
		if ((int)sc.size() >= min_survive)
			cont_cut = std::min(cont_cut, sc[sc.size() - min_survive]);
	}
	std::vector<char> junk(K, 0);
	for (int k = 0; k < K; k++)
		junk[k] = base_junk[k] || (k < (int)cont.size() && std::isfinite(cont[k]) && cont[k] < cont_cut);
	return junk;
}
}
using helixtypes::adjustedRand;

int HelicalContext::selectNrTypes(int tmin, int tmax, int nfold, RFLOAT tol, RFLOAT agree_min, bool keep)
{
	if (agree_min > 0.)
		return selectNrTypesByMixture(tmin, tmax, nfold, tol, agree_min, keep);
	const int current = nr_types;
	const int nch = chains.size();
	int verb_save = verb;
	verb = 0;
	std::vector<double> score(tmax + 1, 0.), agree(tmax + 1, 0.);
	for (int T = tmin; T <= tmax; T++)
	{
		nr_types = T;
		std::vector<std::vector<int> > fold_types(nfold);
		for (int fold = 0; fold < nfold; fold++)
		{
			chain_test.assign(nch, 0);
			for (int c = 0; c < nch; c++)
				chain_test[c] = (filament_of_chain[c] % nfold == fold);
			initialised_model = false;
			update();
			score[T] += loglik_test;
			fold_types[fold] = seg_type;
		}
		// how well the types learnt from different filaments agree on all segments: real types are found
		// again, splits of real types are not
		int npair = 0;
		for (int a = 0; a < nfold; a++)
			for (int b = a + 1; b < nfold; b++, npair++)
				agree[T] += adjustedRand(fold_types[a], fold_types[b]);
		if (npair > 0) agree[T] /= npair;
	}
	chain_test.clear();
	int top = tmin;
	for (int T = tmin; T <= tmax; T++)
		if (score[T] > score[top])
			top = T;
	int best = top;
	std::string why;
	if (agree_min > 0.)
	{
		// the most types that are found again from other filaments (the held-out log-likelihood hardly differs
		// between numbers of types); none early on, while the classes are unsettled: the best agreement, later: no change
		best = -1;
		for (int T = tmin; T <= tmax; T++)
			if (agree[T] >= agree_min)
				best = T;
		if (best < 0 && keep && current >= tmin && current <= tmax)
		{
			best = current;
			why = "none agrees to " + floatToString(agree_min) + ": keeping the current number";
		}
		else if (best < 0)
		{
			best = tmin;
			for (int T = tmin; T <= tmax; T++)
				if (agree[T] > agree[best])
					best = T;
			why = "none agrees to " + floatToString(agree_min) + ": the best agreement";
		}
		else
			why = "the most with agreement of at least " + floatToString(agree_min);
	}
	else
	{
		// the smallest number of types that is (nearly) as good: more types split real ones less reproducibly
		for (int T = tmin; T <= tmax; T++)
			if (score[T] / N >= score[top] / N - tol)
			{
				best = T;
				break;
			}
		why = "the fewest within " + floatToString(tol) + " of the best log-likelihood, " + integerToString(top);
	}
	verb = verb_save;
	if (verb > 0)
	{
		std::cout << " Filament context: held-out log-likelihood per segment by number of types (" << nfold << "-fold over filaments):";
		for (int T = tmin; T <= tmax; T++)
			std::cout << " " << T << ": " << std::setprecision(5) << score[T] / N;
		std::cout << std::endl << " Filament context: agreement of the types between the folds (adjusted Rand index) by number of types:";
		for (int T = tmin; T <= tmax; T++)
			std::cout << " " << T << ": " << std::setprecision(4) << agree[T];
		std::cout << std::endl << " Filament context: using " << best << " types (" << why << ")" << std::endl;
	}
	nr_types = best;
	initialised_model = false;
	return best;
}

int HelicalContext::selectNrTypesByMixture(int tmin, int tmax, int nfold, RFLOAT tol, RFLOAT agree_min, bool keep)
{
	// Filaments as ordered chains of their segments' non-junk equivalence classes: a mixture of tempered
	// first-order Markov chains separates polymorphs with shared crossover views and resists defocus/drift splitting.
	const helixtypes::FilamentChains FC = buildFilamentChains();
	int n_nonempty = 0;
	for (const auto &s : FC.seqs) if (!s.empty()) n_nonempty++;

	const int nsplit = 4;
	const int scan_tmin = std::min(1, tmin);
	const double t0 = omp_get_wtime();
	std::vector<double> score, agree, min_jacc, w_jacc, second_jacc;
	helixtypes::scanChains(FC, scan_tmin, tmax, nfold, nsplit, score, agree, &min_jacc, &w_jacc, &second_jacc);
	std::vector<double> eff_agree = agree;
	std::vector<double> gate_jacc(tmax + 1, 0.);
	if (scan_tmin <= 1 && (int)gate_jacc.size() > 1) gate_jacc[1] = 1.0;
	for (int T = 2; T <= tmax; T++)
	{
		const double mj = (T < (int)min_jacc.size()) ? min_jacc[T] : 0.;
		const double wj = (T < (int)w_jacc.size()) ? w_jacc[T] : mj;
		const double j2 = (T < (int)second_jacc.size()) ? second_jacc[T] : mj;
		// Abundance-weighted + second-minimum Jaccard gating prevents 1 rare polymorph from hiding valid T >= 4
		// while preserving T = 1 specificity via strict ARI + w_jacc + T=2/3 min_jacc checks.
		double gj = (T >= 4 && mj >= 0.15) ? std::min(wj, j2) : mj;
		if (wj < 0.55) gj = std::min(gj, wj);
		gate_jacc[T] = gj;
		if (gj < 0.40)
			eff_agree[T] = std::min(agree[T], gj);
	}
	const double eff_tol = std::min((double)tol, 0.0005);
	std::string why;
	int best = helixtypes::choose(score, eff_agree, tmin, tmax, eff_tol, agree_min, keep ? nr_types : 0, why);
	for (int T = std::max(2, best + 1); T <= tmax; T++)
	{
		const double wj = (T < (int)w_jacc.size()) ? w_jacc[T] : 0.;
		if (score[T] > score[best] + 0.0002 &&
		    agree[T] >= std::max((double)agree_min, 0.75) &&
		    gate_jacc[T] >= 0.55 && wj >= 0.75)
		{
			best = T;
			why = "strictly higher held-out LL with high fold agreement/Jaccard";
		}
		else if (score[T] < score[best] - 0.001)
			break;
	}
	if (tmax > tmin && best >= 2 &&
	    (score[best] <= score[1] + 0.001 || eff_agree[best] < std::min((double)agree_min, 0.50)))
	{
		int fallback = -1;
		for (int T = std::max(2, scan_tmin); T < tmin && T <= tmax; T++)
			if (score[T] > score[1] + 0.001 && eff_agree[T] >= (double)agree_min)
				fallback = T;
		if (fallback < 0)
		{
			double best_ag = std::min((double)agree_min, 0.50);
			for (int T = std::max(2, scan_tmin); T < tmin && T <= tmax; T++)
				if (score[T] > score[1] + 0.001 && eff_agree[T] >= best_ag)
				{
					best_ag = eff_agree[T];
					fallback = T;
				}
		}
		if (fallback >= 2)
		{
			best = fallback;
			why = "best reproducible split below tmin";
		}
		else if (keep && nr_types >= 2 && nr_types <= tmax && eff_agree[nr_types] >= 0.50 && score[nr_types] > score[1])
		{
			best = nr_types;
			why = "keeping current reproducible split";
		}
		else
		{
			best = 1;
			why = "single homogeneous type (no T >= 2 beats 1 type reproducibly across folds)";
		}
	}
	if (verb > 0)
	{
		long int njunk = 0;
		for (int k = 0; k < K; k++) njunk += (!junk_class.empty() && junk_class[k]);
		std::cout << " Filament context: choosing the number of types from " << n_nonempty << " filaments as chains over " << FC.C
		          << " states (" << njunk << " junk classes left out), Markov mixture fitted to " << nfold << " folds, "
		          << nsplit << " splits (" << std::setprecision(3) << omp_get_wtime() - t0 << " s)" << std::endl;
		std::cout << " Filament context: held-out log-likelihood per segment by number of types:";
		for (int T = scan_tmin; T <= tmax; T++)
			std::cout << " " << T << ": " << std::setprecision(5) << score[T];
		std::cout << std::endl << " Filament context: agreement (ARI / min-Jacc / w-Jacc / 2nd-Jacc) by number of types:";
		for (int T = scan_tmin; T <= tmax; T++)
			std::cout << " " << T << ": " << std::setprecision(4) << agree[T]
			          << "/" << (T < (int)min_jacc.size() ? min_jacc[T] : 0.)
			          << "/" << (T < (int)w_jacc.size() ? w_jacc[T] : 0.)
			          << "/" << (T < (int)second_jacc.size() ? second_jacc[T] : 0.);
		std::cout << std::endl << " Filament context: using " << best << " types (" << why << ")" << std::endl;
	}
	nr_types = best;
	initialised_model = false;
	return best;
}

int HelicalContext::runHierarchicalTyping(const std::vector<RFLOAT> &psi, const std::vector<RFLOAT> &psi_prior,
                                          int tmin, int tmax_per_node, int max_types_total, int min_type_filaments,
                                          int nfold, RFLOAT tol, RFLOAT agree_min, bool do_twist_split)
{
	const int F = (int)filament_name.size();
	if (N <= 0 || K <= 0 || F <= 0)
		return std::max(1, nr_types);
	const int max_tot = std::max(1, max_types_total);
	const int min_fil = std::max(2, min_type_filaments);
	const int tmax_node_cap = std::max(2, tmax_per_node);

	// Group chains and segments by global filament ID
	std::vector<std::vector<int> > fil_chains(F);
	for (size_t c = 0; c < chains.size(); c++)
	{
		const long int f = (c < filament_of_chain.size()) ? filament_of_chain[c] : -1;
		if (f >= 0 && f < F && !chains[c].empty())
			fil_chains[f].push_back((int)c);
	}
	std::vector<std::vector<long int> > fil_segs(F);
	for (long int i = 0; i < N; i++)
	{
		if (seg_chain[i] >= 0 && seg_chain[i] < (long int)filament_of_chain.size())
		{
			const long int f = filament_of_chain[seg_chain[i]];
			if (f >= 0 && f < F) fil_segs[f].push_back(i);
		}
	}

	// Best class of each segment to build active-class mask per child node
	std::vector<int> seg_best_cls(N, -1);
	for (long int i = 0; i < N; i++)
	{
		if (std::isnan(loglik[(size_t)i * K])) continue;
		int b = 0;
		for (int k = 1; k < K; k++)
			if (loglik[(size_t)i * K + k] > loglik[(size_t)i * K + b]) b = k;
		seg_best_cls[i] = b;
	}

	auto buildSubContext = [&](const std::vector<int> &sub_fils, HelicalContext &sub,
	                           std::vector<long int> &sub_to_global_seg,
	                           std::vector<RFLOAT> &sub_psi, std::vector<RFLOAT> &sub_psip)
	{
		const int Fs = (int)sub_fils.size();
		sub = HelicalContext();
		sub.K = K;
		sub.step = step;
		sub.diameter = diameter;
		sub.temper = temper;
		sub.exclude_angst = exclude_angst;
		sub.exclude_steps = exclude_steps;
		sub.max_gap_steps = max_gap_steps;
		sub.prior_floor = prior_floor;
		sub.ll_floor = ll_floor;
		sub.nr_bw_iter = nr_bw_iter;
		sub.verb = (verb > 1) ? (verb - 1) : 0;
		sub.pol_switch = pol_switch;
		sub.pol_temper = pol_temper;
		sub.pol_clamp = pol_clamp;
		sub.pol_exclude_angst = pol_exclude_angst;
		sub.pol_exclude_steps = pol_exclude_steps;
		sub.pol_prior_max = pol_prior_max;
		sub.use_polarity_order = use_polarity_order;
		sub.use_phase_ring = use_phase_ring;
		sub.use_twist_rate_split = do_twist_split;
		sub.nr_types = 1;
		sub.class_equiv = class_equiv;

		sub_to_global_seg.clear();
		for (int sf = 0; sf < Fs; sf++)
		{
			const int gf = sub_fils[sf];
			for (long int gi : fil_segs[gf])
				sub_to_global_seg.push_back(gi);
		}
		const long int Ns = (long int)sub_to_global_seg.size();
		sub.N = Ns;
		std::map<long int, long int> g2s;
		for (long int si = 0; si < Ns; si++) g2s[sub_to_global_seg[si]] = si;

		sub.filament_name.resize(Fs);
		sub.pol_members.resize(Fs);
		sub.pol_pos.resize(Fs);
		for (int sf = 0; sf < Fs; sf++)
		{
			const int gf = sub_fils[sf];
			sub.filament_name[sf] = filament_name[gf];
			for (int gc : fil_chains[gf])
			{
				const long int sc = (long int)sub.chains.size();
				std::vector<long int> grid = chains[gc];
				for (size_t p = 0; p < grid.size(); p++)
				{
					if (grid[p] >= 0)
					{
						auto it = g2s.find(grid[p]);
						grid[p] = (it != g2s.end()) ? it->second : -1;
					}
				}
				sub.chains.push_back(grid);
				sub.filament_of_chain.push_back(sf);
				sub.chain_reversed.push_back((gc < (int)chain_reversed.size()) ? chain_reversed[gc] : false);
				for (size_t p = 0; p < grid.size(); p++)
					if (grid[p] >= 0)
					{
						if ((long int)sub.seg_chain.size() < Ns) { sub.seg_chain.assign(Ns, -1); sub.seg_pos.assign(Ns, -1); }
						sub.seg_chain[grid[p]] = sc;
						sub.seg_pos[grid[p]] = (long int)p;
					}
			}
			if (sub.seg_chain.size() != (size_t)Ns) { sub.seg_chain.assign(Ns, -1); sub.seg_pos.assign(Ns, -1); }
			if (gf < (int)pol_members.size())
			{
				for (size_t j = 0; j < pol_members[gf].size(); j++)
				{
					auto it = g2s.find(pol_members[gf][j]);
					if (it != g2s.end())
					{
						sub.pol_members[sf].push_back(it->second);
						sub.pol_pos[sf].push_back((j < pol_pos[gf].size()) ? pol_pos[gf][j] : (long int)j);
					}
				}
			}
		}
		if (sub.seg_chain.size() != (size_t)Ns) { sub.seg_chain.assign(Ns, -1); sub.seg_pos.assign(Ns, -1); }

		sub.loglik.resize((size_t)Ns * K);
		sub.prior.assign((size_t)Ns * K, 1.f);
		sub.post.assign((size_t)Ns * K, 0.f);
		sub.tpost.assign((size_t)Ns, 0.f);
		sub.pol_llr.assign(Ns, std::numeric_limits<float>::quiet_NaN());
		sub.pol_prior.assign(Ns, 0.5f);
		sub.pol_post.assign(Ns, 0.5f);
		sub_psi.assign(Ns, 0.);
		sub_psip.assign(Ns, 0.);

		std::vector<long int> cls_cnt_sub(K, 0);
		for (long int si = 0; si < Ns; si++)
		{
			const long int gi = sub_to_global_seg[si];
			std::copy(loglik.begin() + (size_t)gi * K, loglik.begin() + (size_t)(gi + 1) * K,
			          sub.loglik.begin() + (size_t)si * K);
			if ((long int)pol_post.size() == N) sub.pol_post[si] = pol_post[gi];
			if ((long int)psi.size() == N) sub_psi[si] = psi[gi];
			if ((long int)psi_prior.size() == N) sub_psip[si] = psi_prior[gi];
			if (seg_best_cls[gi] >= 0 && seg_best_cls[gi] < K)
				cls_cnt_sub[seg_best_cls[gi]]++;
		}
		sub.junk_class.assign(K, 0);
		for (int k = 0; k < K; k++)
			sub.junk_class[k] = ((!junk_class.empty() && junk_class[k]) || cls_cnt_sub[k] == 0) ? 1 : 0;
	};

	struct LeafState
	{
		int leaf_id = 0;
		int depth = 0;
		std::vector<int> fils;
	};
	struct SplitStep
	{
		int parent_leaf = 0;
		std::vector<int> child_leaves;
		std::vector<long int> seg_ids;
		std::vector<float> node_tpost;
		std::vector<int> fil_ids;
		std::vector<float> node_fil_post;
	};

	std::vector<LeafState> leaves;
	leaves.reserve(max_tot);
	{
		LeafState root;
		root.leaf_id = 0;
		root.depth = 0;
		root.fils.reserve(F);
		for (int f = 0; f < F; f++)
			if (!fil_segs[f].empty()) root.fils.push_back(f);
		leaves.push_back(root);
	}
	int nr_leaves = 1;

	// Store per-leaf final class prior/post/period results
	std::vector<std::vector<long int> > leaf_seg_ids(max_tot);
	std::vector<std::vector<float> > leaf_seg_post(max_tot), leaf_seg_prior(max_tot);
	std::vector<double> leaf_loglik(max_tot, 0.);
	std::vector<bool> leaf_done(max_tot, false);
	// converged single-type chain model (class distribution q, transitions A) of every leaf, for the global calibration
	std::vector<std::vector<double> > leaf_q(max_tot), leaf_A(max_tot);
	std::vector<double> leaf_mass(max_tot, 0.);

	struct HeapItem
	{
		size_t nfil;
		int depth;
		int leaf_id;
		bool operator<(const HeapItem &o) const
		{
			if (nfil != o.nfil) return nfil < o.nfil;
			return leaf_id > o.leaf_id;
		}
	};
	std::priority_queue<HeapItem> pq;
	pq.push({leaves[0].fils.size(), 0, 0});

	std::vector<SplitStep> split_history;
	const int max_depth = 6;

	while (!pq.empty())
	{
		const HeapItem cur = pq.top();
		pq.pop();
		const int lid = cur.leaf_id;
		const std::vector<int> node_fils = leaves[lid].fils;
		const int Fs = (int)node_fils.size();

		HelicalContext sub;
		std::vector<long int> sub_to_global;
		std::vector<RFLOAT> sub_psi, sub_psip;
		buildSubContext(node_fils, sub, sub_to_global, sub_psi, sub_psip);

		const int budget_tmax = std::min(tmax_node_cap, max_tot - nr_leaves + 1);
		const bool can_split = (Fs >= min_fil) && (cur.depth < max_depth) && (budget_tmax >= 2) && (nr_leaves < max_tot);

		int T_node = 1;
		if (can_split)
		{
			const int node_tmin = (cur.depth == 0) ? std::min(1, tmin) : 1;
			sub.selectNrTypes(node_tmin, budget_tmax, nfold, tol, agree_min, false);
			sub.update();
			sub.fitPeriods(sub_psi, sub_psip);
			if (do_twist_split && sub.nr_types < budget_tmax)
				sub.splitTypesByTwistRate(budget_tmax);
			T_node = std::max(1, sub.nr_types);
		}
		else
		{
			sub.nr_types = 1;
			sub.update();
			sub.fitPeriods(sub_psi, sub_psip);
		}

		// Partition sub filaments into branches by dominant tpost mass
		std::vector<std::vector<int> > branch_fils(T_node);
		std::vector<int> fil_branch(Fs, 0);
		std::vector<float> node_fil_post((size_t)Fs * T_node, 0.f);
		if (T_node > 1)
		{
			std::vector<std::vector<long int> > sub_fil_segs(Fs);
			for (long int si = 0; si < sub.N; si++)
			{
				const int sf = (sub.seg_chain[si] >= 0) ? (int)sub.filament_of_chain[sub.seg_chain[si]] : -1;
				if (sf >= 0 && sf < Fs) sub_fil_segs[sf].push_back(si);
			}
			for (int sf = 0; sf < Fs; sf++)
			{
				std::vector<double> tsum(T_node, 0.);
				int ng = 0;
				for (long int si : sub_fil_segs[sf])
				{
					if (sub.seg_type[si] < 0) continue;
					ng++;
					for (int t = 0; t < T_node; t++)
						tsum[t] += sub.tpost[(size_t)si * T_node + t];
				}
				int bf = 0;
				double tot = 0.;
				for (int t = 0; t < T_node; t++)
				{
					tot += tsum[t];
					if (tsum[t] > tsum[bf]) bf = t;
				}
				if (tot > 0.)
				{
					for (int t = 0; t < T_node; t++)
						node_fil_post[(size_t)sf * T_node + t] = (float)(tsum[t] / tot);
				}
				else
					node_fil_post[(size_t)sf * T_node + 0] = 1.f;
				fil_branch[sf] = bf;
				branch_fils[bf].push_back(node_fils[sf]);
			}

			// Collapse any singleton/empty branches (< 2 filaments) into their next-best populated branch
			// a child must hold enough filaments: noise shards must not consume leaf slots
			const int min_branch = (Fs >= 2 * min_fil) ? std::max(10, min_fil / 2) : 2;
			int valid_branches = 0;
			for (int t = 0; t < T_node; t++)
				if ((int)branch_fils[t].size() >= min_branch) valid_branches++;
			if (valid_branches < 2)
				T_node = 1;
			else
			{
				for (int t = 0; t < T_node; t++)
				{
					if (branch_fils[t].empty() || (int)branch_fils[t].size() >= min_branch) continue;
					for (int gf : branch_fils[t])
					{
						int sf_loc = -1;
						for (int sf = 0; sf < Fs; sf++) if (node_fils[sf] == gf) { sf_loc = sf; break; }
						int best_u = -1;
						double best_p = -1.;
						for (int u = 0; u < T_node; u++)
						{
							if ((int)branch_fils[u].size() < min_branch) continue;
							const double p = (sf_loc >= 0) ? node_fil_post[(size_t)sf_loc * T_node + u] : 0.;
							if (p > best_p) { best_p = p; best_u = u; }
						}
						if (best_u >= 0) branch_fils[best_u].push_back(gf);
					}
					branch_fils[t].clear();
				}
			}
		}

		auto captureLeaf = [&]()
		{
			if (sub.nr_types != 1)
			{
				sub.nr_types = 1;
				sub.initialised_model = false;
				sub.update();
				sub.fitPeriods(sub_psi, sub_psip);
			}
			leaf_done[lid] = true;
			leaf_seg_ids[lid] = sub_to_global;
			leaf_seg_post[lid] = sub.post;
			leaf_seg_prior[lid] = sub.prior;
			leaf_loglik[lid] = sub.loglik_total;
			leaf_q[lid] = sub.q;
			leaf_A[lid] = sub.A;
			leaf_mass[lid] = (double)Fs;
		};

		if (T_node <= 1)
		{
			captureLeaf();
			if (verb > 0)
				std::cout << " Hierarchical tree leaf " << lid + 1 << " (depth " << cur.depth << "): "
				          << Fs << " filaments, " << sub.N << " segments, period "
				          << (sub.type_period.empty() ? 0. : sub.type_period[0]) << " A" << std::endl;
			continue;
		}

		// Collect non-empty child branches
		std::vector<int> active_b;
		for (int t = 0; t < T_node; t++)
			if (!branch_fils[t].empty()) active_b.push_back(t);
		const int T_act = (int)active_b.size();
		if (T_act <= 1)
		{
			captureLeaf();
			continue;
		}

		if (verb > 0)
		{
			std::cout << " Hierarchical tree split node " << lid + 1 << " (depth " << cur.depth << ", "
			          << Fs << " filaments) -> " << T_act << " branches:";
			for (int t : active_b) std::cout << " " << branch_fils[t].size();
			std::cout << std::endl;
		}

		SplitStep step_rec;
		step_rec.parent_leaf = lid;
		step_rec.seg_ids = sub_to_global;
		step_rec.fil_ids = node_fils;
		step_rec.node_tpost.assign((size_t)sub.N * T_act, 0.f);
		step_rec.node_fil_post.assign((size_t)Fs * T_act, 0.f);
		for (long int si = 0; si < sub.N; si++)
		{
			float s = 0.f;
			for (int a = 0; a < T_act; a++)
			{
				const float v = sub.tpost[(size_t)si * T_node + active_b[a]];
				step_rec.node_tpost[(size_t)si * T_act + a] = v;
				s += v;
			}
			if (s > 0.f)
				for (int a = 0; a < T_act; a++) step_rec.node_tpost[(size_t)si * T_act + a] /= s;
			else
				step_rec.node_tpost[(size_t)si * T_act + 0] = 1.f;
		}
		for (int sf = 0; sf < Fs; sf++)
		{
			float s = 0.f;
			for (int a = 0; a < T_act; a++)
			{
				const float v = node_fil_post[(size_t)sf * T_node + active_b[a]];
				step_rec.node_fil_post[(size_t)sf * T_act + a] = v;
				s += v;
			}
			if (s > 0.f)
				for (int a = 0; a < T_act; a++) step_rec.node_fil_post[(size_t)sf * T_act + a] /= s;
			else
				step_rec.node_fil_post[(size_t)sf * T_act + 0] = 1.f;
		}

		step_rec.child_leaves.resize(T_act);
		for (int a = 0; a < T_act; a++)
		{
			const int cid = (a == 0) ? lid : (int)leaves.size();
			step_rec.child_leaves[a] = cid;
			LeafState ch;
			ch.leaf_id = cid;
			ch.depth = cur.depth + 1;
			ch.fils = std::move(branch_fils[active_b[a]]);
			if (a == 0) leaves[cid] = std::move(ch);
			else leaves.push_back(std::move(ch));
			pq.push({leaves[cid].fils.size(), cur.depth + 1, cid});
		}
		nr_leaves += (T_act - 1);
		split_history.push_back(std::move(step_rec));
	}

	// Replay splits to assemble exact N x L segment posteriors and F x L filament posteriors
	const int L = (int)leaves.size();
	std::vector<float> raw_tpost((size_t)N * L, 0.f);
	std::vector<double> raw_fpost((size_t)F * L, 0.);
	for (long int i = 0; i < N; i++) raw_tpost[(size_t)i * L + 0] = 1.f;
	for (int f = 0; f < F; f++) raw_fpost[(size_t)f * L + 0] = 1.0;

	for (const SplitStep &st : split_history)
	{
		const int T_act = (int)st.child_leaves.size();
		for (size_t si = 0; si < st.seg_ids.size(); si++)
		{
			const long int gi = st.seg_ids[si];
			const float p_par = raw_tpost[(size_t)gi * L + st.parent_leaf];
			raw_tpost[(size_t)gi * L + st.parent_leaf] = 0.f;
			for (int a = 0; a < T_act; a++)
				raw_tpost[(size_t)gi * L + st.child_leaves[a]] = p_par * st.node_tpost[si * T_act + a];
		}
		for (size_t sf = 0; sf < st.fil_ids.size(); sf++)
		{
			const int gf = st.fil_ids[sf];
			const double p_par = raw_fpost[(size_t)gf * L + st.parent_leaf];
			raw_fpost[(size_t)gf * L + st.parent_leaf] = 0.;
			for (int a = 0; a < T_act; a++)
				raw_fpost[(size_t)gf * L + st.child_leaves[a]] = p_par * (double)st.node_fil_post[sf * T_act + a];
		}
	}

	// Copy leaf-level post and prior into global context
	post.assign((size_t)N * K, 0.f);
	prior.assign((size_t)N * K, 1.f);
	loglik_total = 0.;
	for (int lid = 0; lid < L; lid++)
	{
		if (!leaf_done[lid]) continue;
		loglik_total += leaf_loglik[lid];
		const std::vector<long int> &sids = leaf_seg_ids[lid];
		if (leaf_seg_post[lid].size() == sids.size() * (size_t)K)
			for (size_t si = 0; si < sids.size(); si++)
				std::copy(leaf_seg_post[lid].begin() + si * K, leaf_seg_post[lid].begin() + (si + 1) * K,
				          post.begin() + (size_t)sids[si] * K);
		if (leaf_seg_prior[lid].size() == sids.size() * (size_t)K)
			for (size_t si = 0; si < sids.size(); si++)
				std::copy(leaf_seg_prior[lid].begin() + si * K, leaf_seg_prior[lid].begin() + (si + 1) * K,
				          prior.begin() + (size_t)sids[si] * K);
	}

	// Order final leaves by segment count descending (type 1 = most abundant)
	std::vector<int> fil_leaf(F, 0);
	std::vector<long int> leaf_seg_cnt(L, 0);
	for (int lid = 0; lid < L; lid++)
	{
		for (int gf : leaves[lid].fils)
		{
			fil_leaf[gf] = lid;
			for (long int gi : fil_segs[gf])
				if (seg_best_cls[gi] >= 0 && (junk_class.empty() || !junk_class[seg_best_cls[gi]]))
					leaf_seg_cnt[lid]++;
		}
	}
	std::vector<int> order(L), inv_order(L);
	for (int lid = 0; lid < L; lid++) order[lid] = lid;
	std::sort(order.begin(), order.end(), [&](int a, int b) {
		if (leaf_seg_cnt[a] != leaf_seg_cnt[b]) return leaf_seg_cnt[a] > leaf_seg_cnt[b];
		return a < b;
	});
	for (int new_t = 0; new_t < L; new_t++) inv_order[order[new_t]] = new_t;

	nr_types = L;
	tpost.assign((size_t)N * L, 0.f);
	fil_chain_post.assign((size_t)F * L, 0.);
	for (long int i = 0; i < N; i++)
		for (int new_t = 0; new_t < L; new_t++)
			tpost[(size_t)i * L + new_t] = raw_tpost[(size_t)i * L + order[new_t]];
	for (int f = 0; f < F; f++)
		for (int new_t = 0; new_t < L; new_t++)
			fil_chain_post[(size_t)f * L + new_t] = raw_fpost[(size_t)f * L + order[new_t]];

	// Depth-independent calibration: all L leaf chain models (class distribution q_l, transitions A_l, each with a
	// small floor so that shared crossover views are allowed everywhere) compete for every segment in one locked
	// forward-backward pass, so tpost / fil_chain_post / rlnHelicalTypePosterior are Bayes-calibrated across the
	// leaves whatever their depth in the tree (no products of the posteriors of successive splits)
	bool leaf_models_ok = (L >= 2);
	for (int lid = 0; lid < L && leaf_models_ok; lid++)
		leaf_models_ok = ((int)leaf_q[lid].size() == K && (int)leaf_A[lid].size() == K * K);
	if (leaf_models_ok)
	{
		const double eps_q = 1e-4;
		q.assign((size_t)L * K, 0.);
		A.assign((size_t)L * K * K, 0.);
		pi0.assign((size_t)L * K, 0.);
		class_allowed.assign((size_t)K * L, 0);
		class_owner.assign(K, -1);
		std::vector<double> mass_t(L, 0.);
		for (int new_t = 0; new_t < L; new_t++)
		{
			const int lid = order[new_t];
			double qs = 0.;
			for (int k = 0; k < K; k++)
			{
				const bool good = junk_class.empty() || !junk_class[k];
				q[(size_t)new_t * K + k] = good ? (leaf_q[lid][k] + eps_q) : 0.;
				qs += q[(size_t)new_t * K + k];
			}
			if (qs > 0.)
				for (int k = 0; k < K; k++) q[(size_t)new_t * K + k] /= qs;
			for (int j = 0; j < K; j++)
			{
				if (!junk_class.empty() && junk_class[j]) continue;
				class_allowed[(size_t)j * L + new_t] = 1;
				double *a = &A[((size_t)new_t * K + j) * K];
				for (int k = 0; k < K; k++)
					a[k] = (!junk_class.empty() && junk_class[k]) ? 0. : (leaf_A[lid][(size_t)j * K + k] + 0.02 * eps_q + 0.02 * q[(size_t)new_t * K + k]);
				normaliseSum(a, K);
			}
			mass_t[new_t] = std::max(1.0, leaf_mass[lid]);
		}
		normaliseSum(&mass_t[0], L);
		for (int new_t = 0; new_t < L; new_t++)
			for (int k = 0; k < K; k++)
				pi0[(size_t)new_t * K + k] = mass_t[new_t] * q[(size_t)new_t * K + k];
		for (int k = 0; k < K; k++)
		{
			int bt = 0;
			for (int t = 1; t < L; t++)
				if (q[(size_t)t * K + k] > q[(size_t)bt * K + k]) bt = t;
			class_owner[k] = bt;
		}
		rho = 1e-6;
		have_prior = true;
		initialised_model = true;
		const int bw_save = nr_bw_iter;
		nr_bw_iter = 0;
		const int verb_save = verb;
		verb = 0;
		chain_test.clear();
		update();
		verb = verb_save;
		nr_bw_iter = bw_save;
	}

	seg_type.assign(N, -1);
	seg_type_prob.assign(N, 0.f);
	type_mass.assign(L, 0.);
	std::vector<double> Aff((size_t)K * L, 0.), cls_tot_k(K, 0.);
	for (long int i = 0; i < N; i++)
	{
		if (seg_chain[i] < 0) continue;
		const int bk = seg_best_cls[i];
		if (bk < 0 || (!junk_class.empty() && junk_class[bk])) continue;
		const long int f = filament_of_chain[seg_chain[i]];
		if (f < 0 || f >= F) continue;
		const int t_assigned = inv_order[fil_leaf[f]];
		seg_type[i] = t_assigned;
		seg_type_prob[i] = tpost[(size_t)i * L + t_assigned];
		for (int t = 0; t < L; t++)
		{
			type_mass[t] += tpost[(size_t)i * L + t];
			Aff[(size_t)bk * L + t] += tpost[(size_t)i * L + t];
		}
		cls_tot_k[bk] += 1.0;
	}

	class_owner.assign(K, -1);
	class_allowed.assign((size_t)K * L, 0);
	q.assign((size_t)L * K, 0.);
	std::vector<double> qsum(L, 0.);
	const double min_own_cnt = (N >= 2000) ? 5.0 : 1.0;
	for (int k = 0; k < K; k++)
	{
		if ((!junk_class.empty() && junk_class[k]) || cls_tot_k[k] < min_own_cnt) continue;
		int bt = 0;
		double aff_sum = 0.;
		for (int t = 0; t < L; t++)
		{
			aff_sum += Aff[(size_t)k * L + t];
			if (Aff[(size_t)k * L + t] > Aff[(size_t)k * L + bt]) bt = t;
		}
		class_owner[k] = bt;
		class_allowed[(size_t)k * L + bt] = 1;
		if (aff_sum > 0. && Aff[(size_t)k * L + bt] < 0.80 * aff_sum)
			for (int t = 0; t < L; t++)
				if (Aff[(size_t)k * L + t] >= 0.12 * aff_sum)
					class_allowed[(size_t)k * L + t] = 1;
		for (int t = 0; t < L; t++)
			if (class_allowed[(size_t)k * L + t])
			{
				q[(size_t)t * K + k] = Aff[(size_t)k * L + t] + 0.05;
				qsum[t] += q[(size_t)t * K + k];
			}
	}
	{
		std::vector<double> aff_by_cls(K, 0.), aff_by_type(L, 0.);
		double tot_aff_all = 0.;
		for (int k = 0; k < K; k++)
		{
			if ((!junk_class.empty() && junk_class[k]) || cls_tot_k[k] < min_own_cnt) continue;
			for (int t = 0; t < L; t++)
			{
				const double a = Aff[(size_t)k * L + t];
				if (a > 0.)
				{
					aff_by_cls[k] += a;
					aff_by_type[t] += a;
					tot_aff_all += a;
				}
			}
		}
		for (int t = 0; t < L; t++)
		{
			if (aff_by_type[t] <= 0. || tot_aff_all <= 0.) continue;
			int n_allow = 0;
			for (int k = 0; k < K; k++)
				if (class_allowed[(size_t)k * L + t]) n_allow++;
			if (n_allow > 0) continue;

			const double prior_t = aff_by_type[t] / tot_aff_all;
			struct CandClass { int k; double lift; double aff; };
			std::vector<CandClass> cands;
			for (int k = 0; k < K; k++)
			{
				if ((!junk_class.empty() && junk_class[k]) || cls_tot_k[k] < min_own_cnt) continue;
				const double a = Aff[(size_t)k * L + t];
				if (a <= 0. || aff_by_cls[k] <= 0.) continue;
				const double lift = (a / aff_by_cls[k]) / prior_t;
				cands.push_back({k, lift, a});
			}
			std::sort(cands.begin(), cands.end(), [](const CandClass &a, const CandClass &b) {
				if (a.lift != b.lift) return a.lift > b.lift;
				if (a.aff != b.aff) return a.aff > b.aff;
				return a.k < b.k;
			});
			const int keep = std::min<int>(3, (int)cands.size());
			for (int i = 0; i < keep; i++)
			{
				const int k = cands[i].k;
				class_allowed[(size_t)k * L + t] = 1;
				q[(size_t)t * K + k] = Aff[(size_t)k * L + t] + 0.05;
				qsum[t] += q[(size_t)t * K + k];
			}
		}
	}
	for (int t = 0; t < L; t++)
	{
		if (qsum[t] > 0.)
			for (int k = 0; k < K; k++) q[(size_t)t * K + k] /= qsum[t];
		else
		{
			for (int k = 0; k < K; k++) q[(size_t)t * K + k] = (!junk_class.empty() && junk_class[k]) ? 0. : 1.;
			normaliseSum(&q[(size_t)t * K], K);
		}
	}

	makeSections();
	computeFilamentPosteriors(N, K, L, F, seg_chain, filament_of_chain,
	                          junk_class, loglik, tpost, seg_type, fil_chain_post, fil_type_posterior);
	fitPeriods(psi, psi_prior);
	rho = (L > 1) ? 1e-6 : 0.;
	have_prior = true;
	initialised_model = true;
	return L;
}

void HelicalContext::writeTypeFiles(const FileName &fn_root, MetaDataTable &MDopt, MetaDataTable &MDimg, int nr_types,
                                    const FileName &fn_classes, RFLOAT min_frac, long int min_count, int verb)
{
	if (!MDimg.containsLabel(EMDL_PARTICLE_HELICAL_TYPE) || !MDimg.containsLabel(EMDL_PARTICLE_CLASS))
		return;
	int K = 0;
	std::vector<int> type(MDimg.numberOfObjects()), cls(MDimg.numberOfObjects());
	for (long int i = 0; i < (long int)MDimg.numberOfObjects(); i++)
	{
		MDimg.getValue(EMDL_PARTICLE_HELICAL_TYPE, type[i], i);
		MDimg.getValue(EMDL_PARTICLE_CLASS, cls[i], i);
		K = std::max(K, cls[i]);
	}
	std::vector<long int> class_total(K + 1, 0);
	std::vector<std::vector<long int> > count(nr_types + 1, std::vector<long int>(K + 1, 0));
	for (size_t i = 0; i < type.size(); i++)
		if (cls[i] >= 1)
		{
			class_total[cls[i]]++;
			if (type[i] >= 1 && type[i] <= nr_types)
				count[type[i]][cls[i]]++;
		}
	bool have_stack = exists(fn_classes);
	MDopt.setName("optics");
	for (int t = 1; t <= nr_types; t++)
	{
		FileName fn_t = fn_root + "_type" + integerToString(t);
		MetaDataTable MDt;
		MDt.setName("particles");
		long int n = 0;
		for (long int i = 0; i < (long int)type.size(); i++)
			if (type[i] == t)
			{
				MDt.addObject(MDimg.getObject(i));
				n++;
			}
		std::ofstream fh((fn_t + "_data.star").c_str());
		MDopt.write(fh);
		MDt.write(fh);
		fh.close();

		std::vector<int> sel;
		long type_cnt = 0, total_assigned = 0;
		for (int c = 1; c <= K; c++)
		{
			type_cnt += count[t][c];
			total_assigned += class_total[c];
			if (count[t][c] >= min_count && count[t][c] >= min_frac * class_total[c])
				sel.push_back(c);
		}
		if (sel.empty() && type_cnt > 0)
		{
			struct TypeClassScore { int c; double score; double enrich; long cnt; };
			std::vector<TypeClassScore> stage1, stage2;
			const double prior = (total_assigned > 0) ? (double)type_cnt / (double)total_assigned : 0.;
			for (int c = 1; c <= K; c++)
			{
				if (count[t][c] <= 0 || class_total[c] <= 0) continue;
				const double frac = (double)count[t][c] / (double)class_total[c];
				const double enrich = (prior > 0.) ? (frac / prior) : frac;
				const double score = (double)count[t][c] * enrich;
				stage2.push_back({c, score, enrich, (long)count[t][c]});
				if (enrich >= 1.25 && (double)count[t][c] >= std::min(2.0, (double)type_cnt))
					stage1.push_back({c, score, enrich, (long)count[t][c]});
			}
			auto cmp = [](const TypeClassScore &a, const TypeClassScore &b) {
				if (a.score != b.score) return a.score > b.score;
				if (a.cnt != b.cnt) return a.cnt > b.cnt;
				return a.c < b.c;
			};
			if (!stage1.empty())
			{
				std::sort(stage1.begin(), stage1.end(), cmp);
				long cum = 0;
				const int cap = std::min<int>(5, (int)stage1.size());
				for (int i = 0; i < cap; i++)
				{
					sel.push_back(stage1[i].c);
					cum += stage1[i].cnt;
					if (cum * 2 >= type_cnt && sel.size() >= 1) break;
				}
			}
			else if (!stage2.empty())
			{
				std::sort(stage2.begin(), stage2.end(), cmp);
				sel.push_back(stage2[0].c);
			}
		}
		std::sort(sel.begin(), sel.end(), [&](int a, int b) { return count[t][a] > count[t][b]; });
		MetaDataTable MDc;
		MDc.setName("model_classes");
		Image<RFLOAT> stack;
		for (size_t j = 0; j < sel.size(); j++)
		{
			int c = sel[j];
			MDc.addObject();
			FileName fn_img;
			fn_img.compose(j + 1, fn_t + "_classes.mrcs");
			MDc.setValue(EMDL_MLMODEL_REF_IMAGE, fn_img);
			MDc.setValue(EMDL_PARTICLE_CLASS, c);
			MDc.setValue(EMDL_MLMODEL_PDF_CLASS, (n > 0) ? (RFLOAT)count[t][c] / n : 0.);
			MDc.setValue(EMDL_PARTICLE_HELICAL_TYPE_PROBABILITY, (RFLOAT)count[t][c] / class_total[c]);
			if (have_stack)
			{
				FileName fn_src;
				fn_src.compose(c, fn_classes);
				Image<RFLOAT> img;
				img.read(fn_src);
				if (j == 0)
					stack().resize(sel.size(), 1, YSIZE(img()), XSIZE(img()));
				stack().setImage(j, img());
				if (j == 0)
					stack.MDMainHeader = img.MDMainHeader;
			}
		}
		MDc.write(fn_t + "_classes.star");
		if (have_stack && !sel.empty())
			stack.write(fn_t + "_classes.mrcs");
		if (verb > 0)
			std::cout << "   type " << t << ": " << n << " segments in " << fn_t << "_data.star, " << sel.size()
			          << " classes in " << fn_t << "_classes.star" << (have_stack ? "/.mrcs" : "") << std::endl;
	}
}
