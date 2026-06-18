/*
 * tests/unit/test_metadata_table.cpp
 *
 * Unit tests for MetaDataTable: add/set/get rows, label types,
 * write/read round-trip via string stream.
 */

#include <catch2/catch.hpp>

#include "src/metadata_table.h"
#include "src/metadata_label.h"

#include <cstdio>  // std::remove

// ---------------------------------------------------------------------------
// Construction and basic CRUD
// ---------------------------------------------------------------------------

TEST_CASE("MDT: newly constructed table has zero objects", "[metadata]")
{
    MetaDataTable mdt;
    REQUIRE(mdt.numberOfObjects() == 0);
}

TEST_CASE("MDT: addObject increases count by one", "[metadata]")
{
    MetaDataTable mdt;
    mdt.addObject();
    REQUIRE(mdt.numberOfObjects() == 1);
    mdt.addObject();
    REQUIRE(mdt.numberOfObjects() == 2);
}

TEST_CASE("MDT: setValue / getValue round-trip for RFLOAT", "[metadata]")
{
    MetaDataTable mdt;
    mdt.addObject();
    const RFLOAT val = 1.2345;
    mdt.setValue(EMDL_CTF_DEFOCUSU, val, 0);
    RFLOAT got = 0.0;
    bool ok = mdt.getValue(EMDL_CTF_DEFOCUSU, got, 0);
    REQUIRE(ok);
    REQUIRE(got == Approx(val).epsilon(1e-10));
}

TEST_CASE("MDT: setValue / getValue round-trip for int", "[metadata]")
{
    MetaDataTable mdt;
    mdt.addObject();
    const int val = 42;
    mdt.setValue(EMDL_IMAGE_ID, val, 0);
    int got = 0;
    bool ok = mdt.getValue(EMDL_IMAGE_ID, got, 0);
    REQUIRE(ok);
    REQUIRE(got == val);
}

TEST_CASE("MDT: setValue / getValue round-trip for string", "[metadata]")
{
    MetaDataTable mdt;
    mdt.addObject();
    const std::string val = "Particles/job001/particles.mrcs";
    mdt.setValue(EMDL_IMAGE_NAME, val, 0);
    std::string got;
    bool ok = mdt.getValue(EMDL_IMAGE_NAME, got, 0);
    REQUIRE(ok);
    REQUIRE(got == val);
}

TEST_CASE("MDT: getValue returns false for absent label", "[metadata]")
{
    MetaDataTable mdt;
    mdt.addObject();
    RFLOAT dummy = 0.0;
    bool ok = mdt.getValue(EMDL_CTF_DEFOCUSU, dummy, 0);
    REQUIRE_FALSE(ok);
}

TEST_CASE("MDT: multiple objects are independent", "[metadata]")
{
    MetaDataTable mdt;
    for (int i = 0; i < 5; i++)
    {
        mdt.addObject();
        mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)(i * 1000.0), i);
    }
    REQUIRE(mdt.numberOfObjects() == 5);
    for (int i = 0; i < 5; i++)
    {
        RFLOAT got;
        mdt.getValue(EMDL_CTF_DEFOCUSU, got, i);
        REQUIRE(got == Approx((RFLOAT)(i * 1000.0)).epsilon(1e-8));
    }
}

// ---------------------------------------------------------------------------
// STAR file write / read round-trip
// ---------------------------------------------------------------------------

TEST_CASE("MDT: write and read STAR round-trip preserves data", "[metadata]")
{
    MetaDataTable mdt;
    mdt.setName("particles");
    for (int i = 0; i < 3; i++)
    {
        mdt.addObject();
        mdt.setValue(EMDL_CTF_DEFOCUSU,  (RFLOAT)(10000.0 + i * 500.0), i);
        mdt.setValue(EMDL_CTF_DEFOCUSV,  (RFLOAT)(10200.0 + i * 500.0), i);
        mdt.setValue(EMDL_CTF_DEFOCUS_ANGLE, (RFLOAT)(i * 30.0),         i);
        mdt.setValue(EMDL_IMAGE_ID,      i,                               i);
    }

    // Write to a temporary file
    const FileName tmpfn = "/tmp/relion_mdt_roundtrip_test.star";
    mdt.write(tmpfn);

    // Read back
    MetaDataTable mdt2;
    mdt2.read(tmpfn);

    REQUIRE(mdt2.numberOfObjects() == 3);
    for (int i = 0; i < 3; i++)
    {
        RFLOAT defU, defV, defAng;
        int id;
        mdt2.getValue(EMDL_CTF_DEFOCUSU,     defU,   i);
        mdt2.getValue(EMDL_CTF_DEFOCUSV,     defV,   i);
        mdt2.getValue(EMDL_CTF_DEFOCUS_ANGLE, defAng, i);
        mdt2.getValue(EMDL_IMAGE_ID,          id,     i);
        REQUIRE(defU   == Approx(10000.0 + i * 500.0).epsilon(1e-6));
        REQUIRE(defV   == Approx(10200.0 + i * 500.0).epsilon(1e-6));
        REQUIRE(defAng == Approx((RFLOAT)(i * 30.0)).epsilon(1e-6));
        REQUIRE(id == i);
    }

    // Clean up
    std::remove(tmpfn.c_str());
}

// ---------------------------------------------------------------------------
// Label presence
// ---------------------------------------------------------------------------

TEST_CASE("MDT: containsLabel is false before any setValue", "[metadata]")
{
    MetaDataTable mdt;
    mdt.addObject();
    REQUIRE_FALSE(mdt.containsLabel(EMDL_CTF_DEFOCUSU));
}

TEST_CASE("MDT: containsLabel is true after setValue", "[metadata]")
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)10000.0, 0);
    REQUIRE(mdt.containsLabel(EMDL_CTF_DEFOCUSU));
}
