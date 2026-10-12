#include "src/helix_context.h"
#include "src/error.h"
#include "src/macros.h"
#include "src/image.h"
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
		MD.setValue(EMDL_PARTICLE_HELICAL_TYPE, seg_type[i] + 1, r);
		MD.setValue(EMDL_PARTICLE_HELICAL_TYPE_PROBABILITY, (RFLOAT)seg_type_prob[i], r);
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
					float l = (i >= 0) ? loglik[i * K + k] : std::numeric_limits<float>::quiet_NaN();
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
					for (int k = 0; k < K; k++)
						prior[i * K + k] = (mx > 0.) ? (float)std::max(prior_floor, tmp[k] / mx) : 1.f;
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
		if (seg_chain[i] < 0) continue;
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

int HelicalContext::selectNrTypes(int tmin, int tmax, int nfold, RFLOAT tol)
{
	const int nch = chains.size();
	int verb_save = verb;
	verb = 0;
	std::vector<double> score(tmax + 1, 0.);
	for (int T = tmin; T <= tmax; T++)
	{
		nr_types = T;
		for (int fold = 0; fold < nfold; fold++)
		{
			chain_test.assign(nch, 0);
			for (int c = 0; c < nch; c++)
				chain_test[c] = (filament_of_chain[c] % nfold == fold);
			initialised_model = false;
			update();
			score[T] += loglik_test;
		}
	}
	chain_test.clear();
	int top = tmin;
	for (int T = tmin; T <= tmax; T++)
		if (score[T] > score[top])
			top = T;
	// the smallest number of types that is (nearly) as good: more types split real ones less reproducibly
	int best = top;
	for (int T = tmin; T <= tmax; T++)
		if (score[T] / N >= score[top] / N - tol)
		{
			best = T;
			break;
		}
	verb = verb_save;
	if (verb > 0)
	{
		std::cout << " Filament context: held-out log-likelihood per segment by number of types (" << nfold << "-fold over filaments):";
		for (int T = tmin; T <= tmax; T++)
			std::cout << " " << T << ": " << std::setprecision(5) << score[T] / N;
		std::cout << std::endl << " Filament context: using " << best << " types (the fewest within " << tol << " of the best, " << top << ")" << std::endl;
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
