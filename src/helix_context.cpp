#include "src/helix_context.h"
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

	if (temper <= 0.)
		temper = 1.;
	if (exclude_angst < 0.)
		exclude_angst = diameter_angst;
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
	initialised_model = have_prior = false;

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

void HelicalContext::fitPeriods(const std::vector<RFLOAT> &psi, const std::vector<RFLOAT> &psi_prior,
                                RFLOAT min_angst, RFLOAT max_angst, RFLOAT step_angst)
{
	type_period.assign(nr_types, 0.);
	type_period_R.assign(nr_types, 0.);
	for (int t = 0; t < nr_types; t++)
	{
		// confident segments of this type, in chains with at least 20 of them
		std::vector<long int> sel;
		std::map<long int, int> per_chain;
		for (long int i = 0; i < N; i++)
			if (seg_type[i] == t && seg_type_prob[i] >= 0.9)
				per_chain[seg_chain[i]]++;
		for (long int i = 0; i < N; i++)
			if (seg_type[i] == t && seg_type_prob[i] >= 0.9 && per_chain[seg_chain[i]] >= 20)
				sel.push_back(i);
		if (sel.size() < 500) continue;
		std::map<long int, int> chain_idx;
		std::vector<int> ch(sel.size()), cl(sel.size());
		std::vector<double> x(sel.size()), sg(sel.size());
		for (size_t n = 0; n < sel.size(); n++)
		{
			long int i = sel[n];
			auto it = chain_idx.find(seg_chain[i]);
			if (it == chain_idx.end())
				it = chain_idx.insert(std::make_pair(seg_chain[i], (int)chain_idx.size())).first;
			ch[n] = it->second;
			cl[n] = std::max_element(post.begin() + i * K, post.begin() + (i + 1) * K) - (post.begin() + i * K);
			x[n] = seg_pos[i] * step;
			sg[n] = (cos(DEG2RAD(psi[i] - psi_prior[i])) >= 0.) ? 1. : -1.;
		}
		const int F = chain_idx.size();
		const long int nsel = sel.size();
		const int nC = (int)floor((max_angst - min_angst) / step_angst + 1e-6) + 1;
		std::vector<double> Rs(nC, -1.);
		// For each trial period C: unit phase factors e_n = exp(i 2 pi x_n sg_n / C) once, then alternate class phases
		// theta_k and chain phases phi_f by complex products (same as the angles, 8 rounds, chain phases from seed 1)
		#pragma omp parallel for schedule(dynamic)
		for (int ic = 0; ic < nC; ic++)
		{
			const double C = min_angst + ic * step_angst;
			std::mt19937 rng(1);
			std::uniform_real_distribution<double> U(0., 2. * PI);
			std::vector<std::complex<double> > en(nsel), pf(F), tk(K), z(K), zf(F);
			for (int f = 0; f < F; f++) pf[f] = std::polar(1., U(rng));
			for (long int n = 0; n < nsel; n++) en[n] = std::polar(1., 2. * PI * x[n] / C * sg[n]);
			auto unit = [](std::complex<double> c) { double a = std::abs(c); return (a > 0.) ? c / a : std::complex<double>(1., 0.); };
			for (int it = 0; it < 8; it++)
			{
				std::fill(z.begin(), z.end(), 0.);
				for (long int n = 0; n < nsel; n++) z[cl[n]] += en[n] * pf[ch[n]];
				for (int k = 0; k < K; k++) tk[k] = unit(z[k]);
				std::fill(zf.begin(), zf.end(), 0.);
				for (long int n = 0; n < nsel; n++) zf[ch[n]] += tk[cl[n]] * std::conj(en[n]);
				for (int f = 0; f < F; f++) pf[f] = unit(zf[f]);
			}
			std::complex<double> r(0.);
			for (long int n = 0; n < nsel; n++) r += tk[cl[n]] * std::conj(en[n] * pf[ch[n]]);
			Rs[ic] = std::abs(r) / nsel;
		}
		double bestR = -1., bestC = 0.;
		for (int ic = 0; ic < nC; ic++)
			if (Rs[ic] > bestR) { bestR = Rs[ic]; bestC = min_angst + ic * step_angst; }
		type_period[t] = bestC;
		type_period_R[t] = bestR;
	}
}

void HelicalContext::writeLabels(MetaDataTable &MD, const std::vector<long int> &seg_rows, const std::vector<int> *assigned_class) const
{
	for (long int i = 0; i < N; i++)
	{
		long int r = seg_rows[i];
		// a segment in a junk class has no type (0)
		const bool junk = assigned_class != NULL && !junk_class.empty() && (*assigned_class)[i] >= 0 && (*assigned_class)[i] < K &&
		                  junk_class[(*assigned_class)[i]];
		MD.setValue(EMDL_PARTICLE_HELICAL_TYPE, junk ? 0 : seg_type[i] + 1, r);
		MD.setValue(EMDL_PARTICLE_HELICAL_TYPE_PROBABILITY, junk ? (RFLOAT)0. : (RFLOAT)seg_type_prob[i], r);
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

// Types from a topic model of the filaments: NMF (Kullback-Leibler) of the counts of the
// most likely classes per filament, X ~ W H with T topics; q_t = H_t, the type weights of a
// filament = W_f, A_t from the class transitions of the filaments weighted by W_f(t)
void HelicalContext::initialiseModel()
{
	const int T = nr_types, F = filament_name.size();
	std::vector<int> best(N, -1);
	for (long int i = 0; i < N; i++)
		if (!std::isnan(loglik[i * K]))
			best[i] = std::max_element(loglik.begin() + i * K, loglik.begin() + (i + 1) * K) - (loglik.begin() + i * K);

	std::vector<double> X(F * K, 0.), W(F * T), H(T * K), WH(F * K), R(F * K);
	for (size_t c = 0; c < chains.size(); c++)
		if (chain_test.empty() || !chain_test[c])
		for (long int i : chains[c])
			if (i >= 0 && best[i] >= 0)
				X[filament_of_chain[c] * K + best[i]] += 1.;
	std::mt19937 rng(42);
	std::uniform_real_distribution<double> U(0.5, 1.5);
	for (auto &w : W) w = U(rng);
	for (auto &h : H) h = U(rng);
	for (int it = 0; it < 300; it++)
	{
		for (int f = 0; f < F; f++)
			for (int k = 0; k < K; k++)
			{
				double s = 0.;
				for (int t = 0; t < T; t++) s += W[f * T + t] * H[t * K + k];
				R[f * K + k] = (s > 0.) ? X[f * K + k] / s : 0.;
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
				R[f * K + k] = (s > 0.) ? X[f * K + k] / s : 0.;
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

	q.assign(T * K, 0.);
	for (int t = 0; t < T; t++)
	{
		for (int k = 0; k < K; k++) q[t * K + k] = H[t * K + k] + 1e-6;
		normaliseSum(&q[t * K], K);
	}
	// filament type weights (fractions of its segments)
	std::vector<double> Wn(F * T);
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
	std::vector<double> cnt(T * K * K, 0.), mass(T, 0.);
	for (size_t c = 0; c < chains.size(); c++)
	{
		if (!chain_test.empty() && chain_test[c]) continue;
		const long int f = filament_of_chain[c];
		const std::vector<long int> &g = chains[c];
		for (size_t s = 0; s < g.size(); s++)
		{
			if (g[s] < 0 || best[g[s]] < 0) continue;
			for (int t = 0; t < T; t++)
				mass[t] += Wn[f * T + t];
			if (s + 1 < g.size() && g[s+1] >= 0 && best[g[s+1]] >= 0)
				for (int t = 0; t < T; t++)
					cnt[(t * K + best[g[s]]) * K + best[g[s+1]]] += Wn[f * T + t];
		}
	}
	A.assign(T * K * K, 0.);
	for (int t = 0; t < T; t++)
		for (int j = 0; j < K; j++)
		{
			double *a = &A[(t * K + j) * K];
			for (int k = 0; k < K; k++)
				a[k] = cnt[(t * K + j) * K + k] + q[t * K + k];
			normaliseSum(a, K);
		}
	normaliseSum(&mass[0], T);
	pi0.assign(T * K, 0.);
	for (int t = 0; t < T; t++)
		for (int k = 0; k < K; k++)
			pi0[t * K + k] = mass[t] * q[t * K + k];
	rho = 1e-3;
	initialised_model = true;
	if (verb > 0)
	{
		std::cout << " Filament context: initial type fractions (NMF of the filaments):";
		for (int t = 0; t < T; t++) std::cout << " " << mass[t];
		std::cout << std::endl;
	}
}

void HelicalContext::update()
{
	const int T = nr_types, TK = T * K;
	const int nch = chains.size();
	// polarity order: a chain of a filament with flipped polarity runs backwards along the track
	if (use_polarity_order)
		for (int c = 0; c < nch; c++)
		{
			double sum = 0.;
			long int n = 0;
			for (long int i : chains[c])
				if (i >= 0)
				{
					sum += pol_post[i];
					n++;
				}
			bool rev = (n > 0 && sum / n > 0.5);
			if (rev != chain_reversed[c])
			{
				std::reverse(chains[c].begin(), chains[c].end());
				chain_reversed[c] = rev;
			}
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

		#pragma omp parallel for schedule(dynamic, 4)
		for (int c = 0; c < nch; c++)
		{
			int tid = 0;
#ifdef _OPENMP
			tid = omp_get_thread_num();
#endif
			const bool held_out = !chain_test.empty() && chain_test[c];
			double &ll_c = held_out ? lltest_t[tid] : ll_t[tid];
			const std::vector<long int> &grid = chains[c];
			const long int L = grid.size();
			std::vector<double> e(L * K), al(L * TK), be(L * TK), v(TK), tmp(K), S(T), Q(T), left(TK), right(TK);
			for (long int s = 0; s < L; s++)
			{
				long int i = grid[s];
				for (int k = 0; k < K; k++)
				{
					float l = (i >= 0 && !seg_junk[i]) ? loglik[i * K + k] : std::numeric_limits<float>::quiet_NaN();
					e[s * K + k] = std::isnan(l) ? 1. : exp(temper * l);
				}
			}
			// forward
			for (int t = 0; t < T; t++)
				for (int k = 0; k < K; k++)
					al[t * K + k] = pi0[t * K + k] * e[k];
			ll_c += log(normaliseSum(&al[0], TK));
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
						an[t * K + k] = e[s * K + k] * ((1. - rho) * tmp[k] + sin * q[t * K + k]);
				}
				ll_c += log(normaliseSum(an, TK));
			}
			// backward (with the expected transition counts)
			for (int tk = 0; tk < TK; tk++)
				be[(L - 1) * TK + tk] = 1.;
			for (long int s = L - 2; s >= 0; s--)
			{
				double Qtot = 0.;
				for (int t = 0; t < T; t++)
				{
					Q[t] = 0.;
					for (int k = 0; k < K; k++)
					{
						v[t * K + k] = e[(s + 1) * K + k] * be[(s + 1) * TK + t * K + k];
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
						bn[t * K + k] = (1. - rho) * tmp[k] + sout;
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
			// posteriors
			for (long int s = 0; s < L; s++)
			{
				for (int tk = 0; tk < TK; tk++)
					v[tk] = al[s * TK + tk] * be[s * TK + tk];
				normaliseSum(&v[0], TK);
				if (!held_out)
					for (int tk = 0; tk < TK; tk++)
						g_t[tid][tk] += v[tk];
				long int i = grid[s];
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
			// priors from the segments at least w steps away
			if (last)
			{
				const long int w = exclude_steps;
				for (long int s = 0; s < L; s++)
				{
					long int i = grid[s];
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
						for (int tk = 0; tk < TK; tk++) left[tk] = pi0[tk];
					if (s + w < L)
					{
						double Qtot = 0.;
						for (int t = 0; t < T; t++)
						{
							Q[t] = 0.;
							for (int k = 0; k < K; k++)
							{
								v[t * K + k] = e[(s + w) * K + k] * be[(s + w) * TK + t * K + k];
								Q[t] += q[t * K + k] * v[t * K + k];
							}
							Qtot += Q[t];
						}
						for (int t = 0; t < T; t++)
						{
							matVec(&Awt[t][0], &v[t * K], &tmp[0], K);
							for (int k = 0; k < K; k++)
								right[t * K + k] = stayw * tmp[k] + switchw * (Qtot - Q[t]);
						}
					}
					else
						for (int tk = 0; tk < TK; tk++) right[tk] = 1.;
					double mx = 0.;
					for (int k = 0; k < K; k++)
					{
						double p = 0.;
						for (int t = 0; t < T; t++) p += left[t * K + k] * right[t * K + k];
						tmp[k] = p;
						mx = std::max(mx, p);
					}
					// the context neither favours nor penalises junk classes: their likelihood decides
					for (int k = 0; k < K; k++)
						prior[i * K + k] = (mx > 0. && (junk_class.empty() || !junk_class[k])) ? (float)std::max(prior_floor, tmp[k] / mx) : 1.f;
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

		// M-step
		rho = (ntr > 0.) ? std::min(0.05, std::max(1e-6, nsw / ntr)) : rho;
		for (int t = 0; t < T; t++)
		{
			for (int k = 0; k < K; k++) q[t * K + k] = gsum[t * K + k] + 1e-6;
			normaliseSum(&q[t * K], K);
		}
		std::vector<double> xi(T * K * K, 0.);
		for (int th = 0; th < nthr; th++)
			for (size_t n = 0; n < xi.size(); n++) xi[n] += xi_t[th][n];
		for (int t = 0; t < T; t++)
			for (int j = 0; j < K; j++)
			{
				double *a = &A[(t * K + j) * K];
				for (int k = 0; k < K; k++)
					a[k] = xi[(t * K + j) * K + k] + q[t * K + k];
				normaliseSum(a, K);
			}
		pi0 = gsum;
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
	makeSections();
	type_period.assign(T, 0.);
	type_period_R.assign(T, 0.);
}

void HelicalContext::makeSections()
{
	// runs of the same type along each chain; runs of fewer than 3 segments between two runs
	// of the same type are taken into those
	const int T = nr_types;
	seg_section.assign(N, -1);
	type_nr_filaments.assign(T, 0);
	type_nr_sections.assign(T, 0);
	type_section_angst.assign(T, 0.);
	std::vector<std::vector<bool> > fil_has(filament_name.size(), std::vector<bool>(T, false));
	int section = 0;
	for (size_t c = 0; c < chains.size(); c++)
	{
		std::vector<long int> segs;
		for (long int i : chains[c])
			if (i >= 0 && seg_type[i] >= 0)
				segs.push_back(i);
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
			long int p0 = seg_pos[segs[j]];
			for (int n = 0; n < r.second; n++, j++)
			{
				seg_section[segs[j]] = section;
				seg_type[segs[j]] = r.first;
			}
			type_nr_sections[r.first]++;
			type_section_angst[r.first] += (labs(seg_pos[segs[j-1]] - p0) + 1) * step;
			fil_has[filament_of_chain[c]][r.first] = true;
			section++;
		}
	}
	for (int t = 0; t < T; t++)
		if (type_nr_sections[t] > 0)
			type_section_angst[t] /= type_nr_sections[t];
	for (auto &f : fil_has)
		for (int t = 0; t < T; t++)
			if (f[t]) type_nr_filaments[t]++;
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
	       "_rlnHelicalFilamentNrSections #4\n_rlnHelicalFilamentMeanTypeProbability #5\n_rlnHelicalFilamentSections #6\n";
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
		out << name.substr(0, colon) << " " << name.substr(colon + 1) << " " << segs[f].size() << " " << nsec
		    << " " << ((np > 0) ? ps / np : 0.) << " " << sec << "\n";
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
	return (top > expected) ? (sab - expected) / (top - expected) : 1.;
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

std::vector<RFLOAT> classContinuity(const std::vector<MultidimArray<RFLOAT> > &refs, RFLOAT apix, const std::vector<long int> &filament,
                                    const std::vector<RFLOAT> &track_angst, const std::vector<int> &cls,
                                    const std::vector<char> &flipped, std::string &note)
{
	const int K = refs.size();
	std::vector<RFLOAT> cont(K, std::numeric_limits<RFLOAT>::quiet_NaN());
	note = "no pairs of segments";
	if (K == 0 || YSIZE(refs[0]) < 8 || ZSIZE(refs[0]) != 1) return cont;
	const int ny = YSIZE(refs[0]), nx = XSIZE(refs[0]), max_shift = (int)(0.4 * nx), dy_max = 2;
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

	// class averages: low-passed, zero mean inside the circle, zero outside
	const double r2max = 0.45 * 0.45 * nx * nx;
	std::vector<char> inside(ny * nx);
	for (int i = 0; i < ny; i++)
		for (int j = 0; j < nx; j++)
			inside[i * nx + j] = (i - ny / 2) * (i - ny / 2) + (j - nx / 2) * (j - nx / 2) < r2max;
	std::vector<std::vector<double> > img(K, std::vector<double>(ny * nx, 0.));
	for (int k = 0; k < K; k++)
	{
		MultidimArray<RFLOAT> f = refs[k];
		lowPassFilterMap(f, 25., apix);
		double sum = 0.;
		long int n = 0;
		for (int p = 0; p < ny * nx; p++)
			if (inside[p]) sum += DIRECT_MULTIDIM_ELEM(f, p), n++;
		for (int p = 0; p < ny * nx; p++)
			img[k][p] = inside[p] ? DIRECT_MULTIDIM_ELEM(f, p) - sum / n : 0.;
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
                              const std::vector<RFLOAT> &cont)
{
	std::vector<RFLOAT> ok;
	for (RFLOAT r : res)
		if (r > 0. && std::isfinite(r)) ok.push_back(r);
	std::sort(ok.begin(), ok.end());
	const RFLOAT junk_res = ok.empty() ? 0. : res_factor * ok[ok.size() / 2];
	std::vector<char> junk(res.size(), 0);
	for (size_t k = 0; k < res.size(); k++)
		junk[k] = (junk_res > 0. && !(res[k] <= junk_res)) || (k < diam.size() && diam[k] > 0.8 * box_angst) ||
		          (k < cont.size() && cont[k] < 0.6);  // NaN (too few pairs): not decided by continuity
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
	// Filaments as counts of their segments' most likely classes (junk classes left out). The chain model fitted per
	// fold (the other rule) starts once from NMF and its segment types agreed only about 0.6-0.7 between folds even
	// for 2 types on real data; this mixture, from several starts, finds the same types again (about 0.95).
	const int nfil = filament_name.size();
	std::vector<int> col(K, -1);
	helixtypes::FilamentCounts X;
	for (int k = 0; k < K; k++)
		if (junk_class.empty() || !junk_class[k])
			col[k] = X.C++;
	std::vector<std::map<int, double> > cnt(nfil);
	for (long int i = 0; i < N; i++)
	{
		if (seg_chain[i] < 0 || std::isnan(loglik[i * K]))
			continue;
		int best = 0;
		for (int k = 1; k < K; k++)
			if (loglik[i * K + k] > loglik[i * K + best]) best = k;
		if (col[best] >= 0)
			cnt[filament_of_chain[seg_chain[i]]][col[best]] += 1.;
	}
	for (int f = 0; f < nfil; f++)
	{
		if (cnt[f].empty())
			continue;
		X.rows.emplace_back(cnt[f].begin(), cnt[f].end());
		double n = 0.;
		for (auto &e : cnt[f]) n += e.second;
		X.total.push_back(n);
	}
	// one split's fold agreement for the same number of types went from 0.64 to 0.82 between checks on real data
	const int nsplit = 4;
	const double t0 = omp_get_wtime();
	std::vector<double> score, agree;
	helixtypes::scan(X, tmin, tmax, nfold, nsplit, score, agree);
	std::string why;
	const int best = helixtypes::choose(score, agree, tmin, tmax, tol, agree_min, keep ? nr_types : 0, why);
	if (verb > 0)
	{
		long int njunk = 0;
		for (int k = 0; k < K; k++) njunk += (col[k] < 0);
		std::cout << " Filament context: choosing the number of types from " << X.rows.size() << " filaments as counts over " << X.C
		          << " classes (" << njunk << " junk classes left out), a mixture fitted to " << nfold << " folds, "
		          << nsplit << " random splits (" << std::setprecision(3) << omp_get_wtime() - t0 << " s)" << std::endl;
		std::cout << " Filament context: held-out log-likelihood per segment by number of types:";
		for (int T = tmin; T <= tmax; T++)
			std::cout << " " << T << ": " << std::setprecision(5) << score[T];
		std::cout << std::endl << " Filament context: agreement of the types between the folds (adjusted Rand index) by number of types:";
		for (int T = tmin; T <= tmax; T++)
			std::cout << " " << T << ": " << std::setprecision(4) << agree[T];
		std::cout << std::endl << " Filament context: using " << best << " types (" << why << ")" << std::endl;
	}
	nr_types = best;
	initialised_model = false;
	return best;
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
		for (int c = 1; c <= K; c++)
			if (count[t][c] >= min_count && count[t][c] >= min_frac * class_total[c])
				sel.push_back(c);
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
