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
	return o;
}

std::vector<std::vector<long int> > groupFilaments(const MetaDataTable &MD, double &step)
{
	std::vector<std::vector<long int> > out;
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
		for (size_t k = 0; k < g.second.size(); k++)
		{
			f.push_back(g.second[k].second);
			if (k > 0 && (has_tla || has_tl))
				steps.push_back(g.second[k].first - g.second[k - 1].first);
		}
		out.push_back(f);
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

// Tempered emissions of one segment, scaled to a maximum of 1 (all 1 when unknown)
void emissions(const float *ll, int K, double temper, const std::vector<double> &pi, double *e)
{
	bool known = false;
	double mx = -1e300;
	for (int k = 0; k < K; k++)
	{
		if (ll[k] != 0.f) known = true;
		if (pi[k] > 0. && ll[k] > mx) mx = ll[k];
	}
	for (int k = 0; k < K; k++)
		e[k] = (!known) ? 1. : (pi[k] > 0. ? std::exp(temper * ((double)ll[k] - mx)) : 0.);
}

void finishPrior(double *p, const std::vector<double> &pi, int K, double floor, float *out)
{
	if (normalise(p, K) <= 0.)
		for (int k = 0; k < K; k++) p[k] = pi[k];
	for (int k = 0; k < K; k++)
	{
		const double v = (1. - floor) * p[k] + floor * pi[k];
		out[k] = (v > 0.) ? (float)std::log(v) : -1e30f;
	}
}

// Forward-backward along one filament: scaled messages a (forward, includes the
// emission) and b (backward, excludes it); optionally accumulates the expected
// transition counts.
void forwardBackward(const std::vector<double> &e, int n, int K, const std::vector<double> &pi,
                     const std::vector<double> &T, std::vector<double> &a, std::vector<double> &b,
                     std::vector<double> *counts)
{
	a.assign((size_t)n * K, 0.);
	b.assign((size_t)n * K, 1.);
	std::vector<double> tmp(K), tmp2(K);
	for (int k = 0; k < K; k++) a[k] = pi[k] * e[k];
	normalise(&a[0], K);
	for (int t = 1; t < n; t++)
	{
		leftMul(&a[(size_t)(t - 1) * K], T, K, &a[(size_t)t * K]);
		for (int k = 0; k < K; k++) a[(size_t)t * K + k] *= e[(size_t)t * K + k];
		if (normalise(&a[(size_t)t * K], K) <= 0.)
			leftMul(&a[(size_t)(t - 1) * K], T, K, &a[(size_t)t * K]), normalise(&a[(size_t)t * K], K);
	}
	for (int t = n - 2; t >= 0; t--)
	{
		for (int k = 0; k < K; k++) tmp[k] = e[(size_t)(t + 1) * K + k] * b[(size_t)(t + 1) * K + k];
		rightMul(T, &tmp[0], K, &b[(size_t)t * K]);
		if (normalise(&b[(size_t)t * K], K) <= 0.)
			for (int k = 0; k < K; k++) b[(size_t)t * K + k] = 1.;
	}
	if (counts == NULL) return;
	for (int t = 0; t + 1 < n; t++)
	{
		const double *at = &a[(size_t)t * K];
		for (int k = 0; k < K; k++) tmp[k] = e[(size_t)(t + 1) * K + k] * b[(size_t)(t + 1) * K + k];
		double z = 0.;
		for (int j = 0; j < K; j++)
		{
			if (at[j] == 0.) continue;
			const double *row = &T[(size_t)j * K];
			for (int k = 0; k < K; k++) z += at[j] * row[k] * tmp[k];
		}
		if (z <= 0.) continue;
		for (int j = 0; j < K; j++)
		{
			if (at[j] == 0.) continue;
			const double *row = &T[(size_t)j * K];
			double *c = &(*counts)[(size_t)j * K];
			const double f = at[j] / z;
			for (int k = 0; k < K; k++) c[k] += f * row[k] * tmp[k];
		}
	}
}

} // namespace

void computePriors(const std::vector<std::vector<long int> > &filaments,
                   const std::vector<float> &loglik, int K,
                   const std::vector<double> &pi, const Options &opt, double temper,
                   std::vector<double> &T, std::vector<float> &logprior)
{
	const long int nr_parts = (long int)(loglik.size() / K);
	logprior.resize(loglik.size());
	{
		std::vector<double> p(pi);
		std::vector<float> row(K);
		finishPrior(&p[0], pi, K, 0., &row[0]);
		for (long int i = 0; i < nr_parts; i++)
			std::copy(row.begin(), row.end(), logprior.begin() + (size_t)i * K);
	}
	if (opt.mode == OFF) return;
	const int gap = std::max(1, opt.gap);

	if (opt.mode == FILAMENT)
	{
		#pragma omp parallel for schedule(dynamic, 16)
		for (long int f = 0; f < (long int)filaments.size(); f++)
		{
			const std::vector<long int> &fil = filaments[f];
			const int n = (int)fil.size();
			// tempered posteriors of each segment, and their running sum along the filament
			std::vector<double> post((size_t)n * K), csum((size_t)(n + 1) * K, 0.), e(K), p(K);
			for (int t = 0; t < n; t++)
			{
				emissions(&loglik[(size_t)fil[t] * K], K, 1., pi, &e[0]);
				for (int k = 0; k < K; k++) e[k] *= pi[k];
				normalise(&e[0], K);
				for (int k = 0; k < K; k++)
				{
					post[(size_t)t * K + k] = temper * e[k];
					csum[(size_t)(t + 1) * K + k] = csum[(size_t)t * K + k] + temper * e[k];
				}
			}
			for (int t = 0; t < n; t++)
			{
				const int lo = std::max(0, t - gap + 1), hi = std::min(n, t + gap); // left out: [lo, hi)
				for (int k = 0; k < K; k++)
					p[k] = opt.alpha * pi[k] + csum[(size_t)n * K + k] - (csum[(size_t)hi * K + k] - csum[(size_t)lo * K + k]);
				finishPrior(&p[0], pi, K, opt.floor, &logprior[(size_t)fil[t] * K]);
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
	#pragma omp parallel for schedule(dynamic, 16)
	for (long int f = 0; f < (long int)filaments.size(); f++)
	{
		const std::vector<long int> &fil = filaments[f];
		E[f].resize(fil.size() * K);
		for (size_t t = 0; t < fil.size(); t++)
			emissions(&loglik[(size_t)fil[t] * K], K, temper, pi, &E[f][t * K]);
	}

	// Baum-Welch updates of the transitions; counts are symmetrised because a
	// filament may have been traced in either direction
	for (int it = 0; it < opt.bw_iters; it++)
	{
		std::vector<double> counts((size_t)K * K, 0.);
		#pragma omp parallel
		{
			std::vector<double> my((size_t)K * K, 0.), a, b;
			#pragma omp for schedule(dynamic, 16) nowait
			for (long int f = 0; f < (long int)filaments.size(); f++)
				if (filaments[f].size() > 1)
					forwardBackward(E[f], (int)filaments[f].size(), K, pi, T, a, b, &my);
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

	const std::vector<double> Tg = (gap == 1) ? T : matPow(T, K, gap);
	// pi T^t for segments near the start, where the left context is shorter than the gap
	std::vector<std::vector<double> > piT(gap, std::vector<double>(K));
	piT[0] = pi;
	for (int j = 1; j < gap; j++) leftMul(&piT[j - 1][0], T, K, &piT[j][0]);

	#pragma omp parallel for schedule(dynamic, 16)
	for (long int f = 0; f < (long int)filaments.size(); f++)
	{
		const std::vector<long int> &fil = filaments[f];
		const int n = (int)fil.size();
		std::vector<double> a, b, L(K), R(K), tmp(K);
		forwardBackward(E[f], n, K, pi, T, a, b, NULL);
		for (int t = 0; t < n; t++)
		{
			if (t - gap >= 0)
				leftMul(&a[(size_t)(t - gap) * K], Tg, K, &L[0]);
			else
				L = piT[t];
			if (t + gap < n)
			{
				for (int k = 0; k < K; k++) tmp[k] = E[f][(size_t)(t + gap) * K + k] * b[(size_t)(t + gap) * K + k];
				rightMul(Tg, &tmp[0], K, &R[0]);
			}
			else
				std::fill(R.begin(), R.end(), 1.);
			for (int k = 0; k < K; k++) L[k] *= R[k];
			finishPrior(&L[0], pi, K, opt.floor, &logprior[(size_t)fil[t] * K]);
		}
	}
}

}
