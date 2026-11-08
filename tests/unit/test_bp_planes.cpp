/*
 * tests/unit/test_bp_planes.cpp
 *
 * relion_reconstruct adds images to the map on several threads, each owning a range of z planes
 * (BackProjector::backproject2Dto3D with z_lo, z_hi). Adding every image to each range in turn must give
 * the map of adding them whole, bit for bit.
 */

#include <catch2/catch.hpp>

#include "src/backprojector.h"
#include "src/euler.h"

#include <random>

TEST_CASE("backproject2Dto3D: plane ranges add up to the whole map bit for bit", "[bp_planes]")
{
	const int ori = 24;
	std::mt19937 rng(3);
	std::uniform_real_distribution<RFLOAT> u(-1., 1.);
	std::vector<MultidimArray<Complex> > imgs(5);
	std::vector<MultidimArray<RFLOAT> > ctfs(5);
	std::vector<Matrix2D<RFLOAT> > As(5);
	for (int k = 0; k < 5; k++)
	{
		imgs[k].resize(ori, ori / 2 + 1);
		ctfs[k].resize(ori, ori / 2 + 1);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(imgs[k])
		{
			DIRECT_MULTIDIM_ELEM(imgs[k], n) = Complex(u(rng), u(rng));
			DIRECT_MULTIDIM_ELEM(ctfs[k], n) = u(rng) * u(rng);
		}
		Euler_angles2matrix(180. * u(rng), 90. * u(rng) + 90., 180. * u(rng), As[k]);
	}

	for (int interpolator : {TRILINEAR, NEAREST_NEIGHBOUR})
	{
	for (bool single_planes : {false, true})
	{
		BackProjector whole(ori, 3, "c1", interpolator, 2, 10, 0, 1.9, 15, 2, false);
		BackProjector split(ori, 3, "c1", interpolator, 2, 10, 0, 1.9, 15, 2, false);
		whole.initZeros();
		split.initZeros();
		const int zdim = ZSIZE(whole.data);
		std::vector<int> lo = {0, zdim / 3, zdim / 2 + 1}, hi = {zdim / 3 - 1, zdim / 2, zdim - 1};
		if (single_planes)
		{
			lo.clear();
			hi.clear();
			for (int z = 0; z < zdim; z++)
			{
				lo.push_back(z);
				hi.push_back(z);
			}
		}
		for (int k = 0; k < 5; k++)
			whole.backproject2Dto3D(imgs[k], As[k], &ctfs[k]);
		for (int r = (int)lo.size() - 1; r >= 0; r--) // the ranges in any order
			for (int k = 0; k < 5; k++)
				split.backproject2Dto3D(imgs[k], As[k], &ctfs[k], -1., true, 0, lo[r], hi[r]);
		INFO("interpolator " << interpolator << " single planes " << single_planes);
		long differ = 0;
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(whole.data)
			if (DIRECT_MULTIDIM_ELEM(whole.data, n).real != DIRECT_MULTIDIM_ELEM(split.data, n).real
			    || DIRECT_MULTIDIM_ELEM(whole.data, n).imag != DIRECT_MULTIDIM_ELEM(split.data, n).imag
			    || DIRECT_MULTIDIM_ELEM(whole.weight, n) != DIRECT_MULTIDIM_ELEM(split.weight, n))
				differ++;
		REQUIRE(differ == 0);
		REQUIRE(whole.weight.sum() > 0.);
	}
	}
}
