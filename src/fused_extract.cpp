/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/fused_extract.h"

#include <complex>
#include <map>

#include "src/error.h"
#include "src/euler.h"
#include "src/funcs.h"
#include "src/matrix2d.h"
#include "src/relion_finufft.h"

namespace fusedextract {

void normaliseSamples(Samples& s, int nx, int ny, const Normalisation& n)
{
	if (!n.normalise)
	{
		if (n.invert) for (size_t i = 0; i < s.value.size(); i++) s.value[i] = -s.value[i];
		return;
	}

	const size_t N = s.value.size();
	std::vector<char> bg(N, 0);
	size_t nbg = 0;

	if (n.helical)
	{
		Matrix2D<RFLOAT> A(3, 3);
		Euler_angles2matrix(0., 0., n.psi, A, false);
		A = A.transpose();
		// y component of A * (x, y, 0)
		const double a10 = MAT_ELEM(A, 1, 0), a11 = MAT_ELEM(A, 1, 1);
		for (size_t i = 0; i < N; i++)
			if (std::fabs(a10 * s.dx[i] + a11 * s.dy[i]) > n.helical_radius) { bg[i] = 1; nbg++; }
	}
	else
	{
		const double shortest = std::min(nx, ny);
		const double ax = n.bg_radius * nx / shortest, ay = n.bg_radius * ny / shortest;
		for (size_t i = 0; i < N; i++)
			if (s.dx[i] * s.dx[i] / (ax * ax) + s.dy[i] * s.dy[i] / (ay * ay) > 1.) { bg[i] = 1; nbg++; }
	}
	if (nbg < 5)
		REPORT_ERROR("fused extraction: fewer than 5 background pixels; the background radius is too large for the box.");

	if (n.ramp)
	{
		std::vector<fit_point3D> pts;
		pts.reserve(nbg);
		for (size_t i = 0; i < N; i++)
			if (bg[i])
			{
				fit_point3D p;
				p.x = s.dx[i]; p.y = s.dy[i]; p.z = s.value[i]; p.w = 1.;
				pts.push_back(p);
			}
		RFLOAT pA, pB, pC;
		fitLeastSquaresPlane(pts, pA, pB, pC);
		for (size_t i = 0; i < N; i++) s.value[i] -= pA * s.dx[i] + pB * s.dy[i] + pC;
	}

	double sum = 0., sum2 = 0.;
	for (size_t i = 0; i < N; i++)
		if (bg[i]) { sum += s.value[i]; sum2 += s.value[i] * s.value[i]; }
	const double avg = sum / nbg;
	const double sd = std::sqrt(std::max(0., sum2 / nbg - avg * avg));
	if (sd >= 1e-10)
		for (size_t i = 0; i < N; i++) s.value[i] = (s.value[i] - avg) / sd;
	if (n.invert)
		for (size_t i = 0; i < N; i++) s.value[i] = -s.value[i];
}

#ifdef RELION_USE_FINUFFT

namespace {

struct PlanKey {
	int nx, ny;
	bool operator<(const PlanKey& o) const { return nx != o.nx ? nx < o.nx : ny < o.ny; }
};

}

void fourierTransform(const Samples& s, const Geometry& g, MultidimArray<Complex>& F2D)
{
	if (g.nx % 2 != 0 || g.ny % 2 != 0 || g.nx < 2 || g.ny < 2)
		REPORT_ERROR("fused extraction: the output box must have even sides.");

	// One more mode on each side so that the Nyquist column kx = nx/2 exists
	const int64_t N1 = g.nx + 2, N2 = g.ny + 2;
	const int64_t nj = (int64_t)s.value.size();

	// Plans are kept per thread: they are reused for every particle of that box size
	thread_local std::map<PlanKey, FinufftPlanGuard> plans;
	thread_local std::vector<RFLOAT> xj, yj;
	thread_local std::vector<std::complex<RFLOAT> > cj, fk;

	xj.resize(nj); yj.resize(nj); cj.resize(nj);
	for (int64_t j = 0; j < nj; j++)
	{
		xj[j] = (RFLOAT)(2.0 * M_PI * s.dx[j] / g.nx);
		yj[j] = (RFLOAT)(2.0 * M_PI * s.dy[j] / g.ny);
		cj[j] = std::complex<RFLOAT>((RFLOAT)s.value[j], 0.);
	}
	fk.resize((size_t)(N1 * N2));

	PlanKey key = {g.nx, g.ny};
	std::map<PlanKey, FinufftPlanGuard>::iterator it = plans.find(key);
	if (it == plans.end())
	{
		FinufftPlanGuard guard;
		finufft_opts opts;
		finufft_default_opts(&opts);
		opts.modeord = 0;
		opts.nthreads = 1;
		int64_t modes[2] = {N1, N2};
		const double tol = (sizeof(RFLOAT) == 4) ? 1e-5 : 1e-9;
		int ier = FINUFFT_MAKEPLAN(1, 2, modes, -1, 1, tol, &guard.plan, &opts);
		if (ier > 1) handleFinufftError(ier, "finufft_makeplan type1 (fused extraction)");
		it = plans.insert(std::make_pair(key, std::move(guard))).first;
	}

	int ier = FINUFFT_SETPTS(it->second.plan, nj, xj.data(), yj.data(), NULL, 0, NULL, NULL, NULL);
	if (ier > 1) handleFinufftError(ier, "finufft_setpts type1 (fused extraction)");
	ier = FINUFFT_EXECUTE(it->second.plan, cj.data(), fk.data());
	if (ier > 1) handleFinufftError(ier, "finufft_execute type1 (fused extraction)");

	// Each micrograph pixel stands for step^-2 output pixels of area
	const RFLOAT scale = (RFLOAT)(1.0 / ((double)g.nx * g.ny * g.step * g.step));
	// A pixel size finer than the micrograph's has no signal above its Nyquist frequency
	const double fmax2 = (g.step < 1.) ? 0.25 * g.step * g.step : 1e30;

	F2D.resize(g.ny, g.nx / 2 + 1);
	for (int i = 0; i < g.ny; i++)
	{
		const int ky = (i < g.ny / 2) ? i : i - g.ny;
		for (int j = 0; j <= g.nx / 2; j++)
		{
			const double fx = (double)j / g.nx, fy = (double)ky / g.ny;
			if (fx * fx + fy * fy > fmax2)
			{
				DIRECT_A2D_ELEM(F2D, i, j) = Complex(0., 0.);
				continue;
			}
			const std::complex<RFLOAT>& v = fk[(size_t)((ky + N2 / 2) * N1 + (j + N1 / 2))];
			DIRECT_A2D_ELEM(F2D, i, j) = Complex(v.real() * scale, v.imag() * scale);
		}
	}
}

#else

void fourierTransform(const Samples&, const Geometry&, MultidimArray<Complex>&)
{
	REPORT_ERROR("fused extraction needs a build with -DRELION_USE_FINUFFT=ON.");
}

#endif

} // namespace fusedextract
