/*
 * tests/unit/test_exact_map3d.cpp
 *
 * ExactMap3D (RELION_MAP_INTERPOLATION=nufft): the value of a map's periodic band-limited interpolant anywhere,
 * checked against a direct evaluation of the Fourier series, on even and odd sizes.
 */

#include <catch2/catch.hpp>

#ifdef RELION_USE_FINUFFT

#include <cmath>
#include <random>

#include "src/exact_map3d.h"
#include "src/euler.h"
#include "src/transformations.h"

namespace
{

// The real Dirichlet kernel of an n-point grid at offset t: sum of e^{2 pi i k t / n} over the n frequencies, with
// the Nyquist term (even n) as cos, which keeps the interpolant real
double dirichlet(int n, double t)
{
	double s = 1;
	for (int k = 1; 2 * k < n; k++) s += 2 * std::cos(2 * PI * k * t / n);
	if (n % 2 == 0) s += std::cos(PI * t);
	return s / n;
}

double seriesAt(const MultidimArray<RFLOAT> &m, double x, double y, double z)
{
	double s = 0;
	FOR_ALL_ELEMENTS_IN_ARRAY3D(m)
		s += A3D_ELEM(m, k, i, j) * dirichlet(XSIZE(m), x - j) * dirichlet(YSIZE(m), y - i) * dirichlet(ZSIZE(m), z - k);
	return s;
}

}

TEST_CASE("ExactMap3D matches the Fourier series of the map", "[exact_map3d]")
{
	for (int odd = 0; odd < 2; odd++)
	{
		const int nx = 12 + odd, ny = 10, nz = 14 + odd;
		std::mt19937 rng(11 + odd);
		std::normal_distribution<double> gauss;
		std::uniform_real_distribution<double> uni(-8, 8);
		MultidimArray<RFLOAT> m(nz, ny, nx);
		m.setXmippOrigin();
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(m) DIRECT_MULTIDIM_ELEM(m, n) = gauss(rng);

		const ExactMap3D em(m, 2);
		double max_vox = 0, max_off = 0;
		FOR_ALL_ELEMENTS_IN_ARRAY3D(m)
			max_vox = std::max(max_vox, std::fabs(em.value(j, i, k) - A3D_ELEM(m, k, i, j)));
		for (int t = 0; t < 200; t++)
		{
			const double x = uni(rng), y = uni(rng), z = uni(rng);   // also outside the box: periodic
			max_off = std::max(max_off, std::fabs(em.value(x, y, z) - seriesAt(m, x, y, z)));
		}
		INFO("size " << nx << "x" << ny << "x" << nz << ": max error at voxels " << max_vox << ", off the grid " << max_off);
		CHECK(max_vox < 1e-3);
		CHECK(max_off < 1e-3);
	}
}

TEST_CASE("ExactMap3D::applyGeometry follows applyGeometry's conventions", "[exact_map3d]")
{
	// a 90 degree turn and a whole-voxel shift land on voxels, where trilinear interpolation is exact too
	const int size = 12;
	std::mt19937 rng(5);
	std::normal_distribution<double> gauss;
	MultidimArray<RFLOAT> m(size, size, size), lin, ex;
	m.setXmippOrigin();
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(m) DIRECT_MULTIDIM_ELEM(m, n) = gauss(rng);
	const ExactMap3D em(m, 2);
	for (int inv = 0; inv < 2; inv++)
	{
		Matrix2D<RFLOAT> R, T;
		Euler_rotation3DMatrix(90., 90., 0., R);
		translation3DMatrix(2., -1., 3., T);
		const Matrix2D<RFLOAT> A = T * R;
		applyGeometry(m, lin, A, inv, DONT_WRAP);
		em.applyGeometry(ex, A, inv, 2);
		double max_err = 0;
		int nonzero = 0;
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(lin)
		{
			max_err = std::max(max_err, std::fabs(DIRECT_MULTIDIM_ELEM(ex, n) - DIRECT_MULTIDIM_ELEM(lin, n)));
			nonzero += DIRECT_MULTIDIM_ELEM(lin, n) != 0;
		}
		INFO("inv " << inv << ": max difference " << max_err << ", " << nonzero << " voxels inside");
		CHECK(nonzero > 500);
		CHECK(max_err < 1e-3);
		CHECK(STARTINGX(ex) == STARTINGX(m));
	}
}

#endif // RELION_USE_FINUFFT
