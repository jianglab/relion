/*
 * tests/unit/test_ctf_memo.cpp
 *
 * RELION_PREP_AHEAD memo of CTF images: a copy from the memo equals what CTF::getFftwImage writes,
 * bit for bit; keys that differ in any value do not collide; the memory cap and the per-iteration
 * clear work.
 */

#include <catch2/catch.hpp>

#include "src/ctf_memo.h"
#include "src/ctf.h"

#include <cstring>

namespace
{
MultidimArray<RFLOAT> makeCtf(double defU, double defV, double angle, int size, RFLOAT angpix)
{
	CTF ctf;
	ctf.setValues(defU, defV, angle, 300., 2.7, 0.1, 0.07, 1., 0.);
	MultidimArray<RFLOAT> img(size, size / 2 + 1);
	ctf.getFftwImage(img, size, size, angpix, false, false, true, true, false);
	return img;
}

bool sameBits(const MultidimArray<RFLOAT> &a, const MultidimArray<RFLOAT> &b)
{
	return XSIZE(a) == XSIZE(b) && YSIZE(a) == YSIZE(b)
		&& memcmp(MULTIDIM_ARRAY(a), MULTIDIM_ARRAY(b), sizeof(RFLOAT) * NZYXSIZE(a)) == 0;
}
}

TEST_CASE("CTF memo returns the exact image and separates different keys", "[ctf_memo]")
{
	CtfMemo memo(1 << 20);
	const CtfMemo::Key k1 = CtfMemo::makeKey(15000., 15500., 30., 0., 1., 0., 1.2, 1, 17, 32, 32, 5u);
	CHECK(!memo.find(k1));

	MultidimArray<RFLOAT> a = makeCtf(15000., 15500., 30., 32, 1.2);
	CHECK(memo.insert(k1, a));
	CtfMemo::Ptr hit = memo.find(k1);
	REQUIRE(hit);
	MultidimArray<RFLOAT> copy = *hit;
	CHECK(sameBits(copy, a));
	CHECK(sameBits(copy, makeCtf(15000., 15500., 30., 32, 1.2)));

	// a change in any one value of the key is a miss
	CtfMemo::Key k = k1;
	for (int i = 0; i < 7; i++)
	{
		k = k1;
		k.v[i] = std::nextafter(k.v[i], 1e9);
		CHECK(!memo.find(k));
	}
	k = k1; k.optics_group = 2; CHECK(!memo.find(k));
	k = k1; k.xdim = 18; CHECK(!memo.find(k));
	k = k1; k.ydim = 33; CHECK(!memo.find(k));
	k = k1; k.full_size = 64; CHECK(!memo.find(k));
	k = k1; k.flags = 4u; CHECK(!memo.find(k));
	CHECK(memo.entries() == 1);
	CHECK(memo.hits() == 1);
}

TEST_CASE("CTF memo keeps to its memory cap and clears when the image size changes", "[ctf_memo]")
{
	const size_t one = (size_t)32 * 17 * sizeof(RFLOAT);
	CtfMemo memo(2 * one + one / 2);
	MultidimArray<RFLOAT> a = makeCtf(15000., 15000., 0., 32, 1.);
	CHECK(memo.insert(CtfMemo::makeKey(1, 0, 0, 0, 0, 0, 1, 0, 17, 32, 32, 0), a));
	CHECK(memo.insert(CtfMemo::makeKey(2, 0, 0, 0, 0, 0, 1, 0, 17, 32, 32, 0), a));
	CHECK(!memo.insert(CtfMemo::makeKey(3, 0, 0, 0, 0, 0, 1, 0, 17, 32, 32, 0), a));
	CHECK(memo.entries() == 2);
	CHECK(memo.bytes() == 2 * one);
	CHECK(memo.skipped() == 1);
	// inserting an existing key is not a second copy
	CHECK(memo.insert(CtfMemo::makeKey(1, 0, 0, 0, 0, 0, 1, 0, 17, 32, 32, 0), a));
	CHECK(memo.bytes() == 2 * one);

	memo.beginIteration("a");
	CHECK(memo.entries() == 0);
	CHECK(memo.insert(CtfMemo::makeKey(1, 0, 0, 0, 0, 0, 1, 0, 17, 32, 32, 0), a));
	memo.beginIteration("a");       // same sizes: the entries stay
	CHECK(memo.entries() == 1);
	memo.beginIteration("b");
	CHECK(memo.entries() == 0);
	CHECK(memo.bytes() == 0);
}
