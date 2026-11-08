/*
 * tests/unit/test_mask_edge.cpp
 *
 * autoMask (relion_mask_create, the helical Fourier mask of relion_refine): the extension, shrinking and
 * soft edge use an exact distance transform instead of searching every voxel's neighbours. The result
 * must be bit for bit what the neighbour search gave; the search is copied here as the reference.
 */

#include <catch2/catch.hpp>

#include "src/mask.h"

#include <random>

namespace
{

// The neighbour search autoMask used before (single-threaded, no progress bar)
void referenceAutoMask(const MultidimArray<RFLOAT> &img, MultidimArray<RFLOAT> &msk, RFLOAT thr, RFLOAT ext, RFLOAT soft)
{
	msk.resize(img);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img)
		DIRECT_MULTIDIM_ELEM(msk, n) = (DIRECT_MULTIDIM_ELEM(img, n) >= thr) ? 1. : 0.;
	auto inside = [](const MultidimArray<RFLOAT> &m, long k, long i, long j)
	{
		return k >= STARTINGZ(m) && k <= FINISHINGZ(m) && i >= STARTINGY(m) && i <= FINISHINGY(m) &&
		       j >= STARTINGX(m) && j <= FINISHINGX(m);
	};
	if (ext != 0.)
	{
		const int e = ABS(CEIL(ext));
		const RFLOAT e2 = ext * ext;
		MultidimArray<RFLOAT> cp = msk;
		FOR_ALL_ELEMENTS_IN_ARRAY3D(cp)
		{
			const bool candidate = (ext > 0.) ? A3D_ELEM(cp, k, i, j) < 0.001 : A3D_ELEM(cp, k, i, j) > 0.999;
			if (!candidate)
				continue;
			for (long kp = k - e; kp <= k + e; kp++)
				for (long ip = i - e; ip <= i + e; ip++)
					for (long jp = j - e; jp <= j + e; jp++)
						if (inside(cp, kp, ip, jp))
						{
							const bool source = (ext > 0.) ? A3D_ELEM(cp, kp, ip, jp) > 0.999 : A3D_ELEM(cp, kp, ip, jp) < 0.001;
							const RFLOAT r2 = (RFLOAT)((kp-k)*(kp-k) + (ip-i)*(ip-i) + (jp-j)*(jp-j));
							if (source && r2 < e2)
								A3D_ELEM(msk, k, i, j) = (ext > 0.) ? 1. : 0.;
						}
		}
	}
	if (soft > 0.)
	{
		MultidimArray<RFLOAT> cp = msk;
		const int e = CEIL(soft);
		const RFLOAT s2 = soft * soft;
		FOR_ALL_ELEMENTS_IN_ARRAY3D(cp)
		{
			if (A3D_ELEM(cp, k, i, j) >= 0.001)
				continue;
			RFLOAT min_r2 = 9999.;
			for (long kp = k - e; kp <= k + e; kp++)
				for (long ip = i - e; ip <= i + e; ip++)
					for (long jp = j - e; jp <= j + e; jp++)
						if (inside(cp, kp, ip, jp) && A3D_ELEM(cp, kp, ip, jp) > 0.999)
							min_r2 = std::min(min_r2, (RFLOAT)((kp-k)*(kp-k) + (ip-i)*(ip-i) + (jp-j)*(jp-j)));
			if (min_r2 < s2)
				A3D_ELEM(msk, k, i, j) = 0.5 + 0.5 * cos(PI * sqrt(min_r2) / soft);
		}
	}
}

}

TEST_CASE("autoMask: distance transform gives the neighbour search's mask bit for bit", "[mask_edge]")
{
	// A box that is not a cube, with a few blobs (one touching the edge) and noise
	MultidimArray<RFLOAT> img(30, 26, 22);
	img.setXmippOrigin();
	std::mt19937 rng(7);
	std::uniform_real_distribution<RFLOAT> u(0., 1.);
	const RFLOAT centres[4][3] = {{0, 0, 0}, {-8, 5, 6}, {10, -9, -4}, {-14, 0, 10}};
	FOR_ALL_ELEMENTS_IN_ARRAY3D(img)
	{
		RFLOAT v = 0.3 * u(rng);
		for (auto &c : centres)
			v += exp(-((k - c[0]) * (k - c[0]) + (i - c[1]) * (i - c[1]) + (j - c[2]) * (j - c[2])) / 12.);
		A3D_ELEM(img, k, i, j) = v;
	}

	const RFLOAT exts[] = {0., 2., 3.5, -1.5, -2.};
	const RFLOAT softs[] = {0., 3., 4.5, 6.};
	for (RFLOAT ext : exts)
		for (RFLOAT soft : softs)
			for (int threads : {1, 3})
			{
				MultidimArray<RFLOAT> ref, got;
				referenceAutoMask(img, ref, 0.6, ext, soft);
				autoMask(img, got, 0.6, ext, soft, false, threads);
				INFO("extend " << ext << " soft edge " << soft << " threads " << threads);
				long differ = 0;
				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(ref)
					if (DIRECT_MULTIDIM_ELEM(ref, n) != DIRECT_MULTIDIM_ELEM(got, n))
						differ++;
				REQUIRE(differ == 0);
			}
}
