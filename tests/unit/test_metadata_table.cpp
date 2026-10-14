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
#include <cstdlib>
#include <fstream>
#include <sstream>

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

// ---------------------------------------------------------------------------
// The parallel loop reader must give exactly what the line-by-line one gives
// ---------------------------------------------------------------------------

namespace
{
std::string writeTable(const MetaDataTable &t)
{
    std::ostringstream o;
    t.write(o);
    return o.str();
}
}

TEST_CASE("MDT: threaded and one-line-at-a-time STAR readers agree", "[metadata][star_parse]")
{
    const std::string fn = "relion_star_parse_test.star";
    const int n = 40000;
    {
        std::ofstream o(fn);
        o << "# version 30001\n\ndata_optics\n\nloop_\n_rlnOpticsGroupName #1\n_rlnOpticsGroup #2\nog1 1\n\n";
        o << "data_particles\n\nloop_\n_rlnCoordinateX #1\n_rlnImageName #2\n_rlnGroupNumber #3\n"
             "_rlnDefocusU #4\n_rlnMicrographName #5\n_rlnCtfFigureOfMerit #6\n";
        for (int i = 0; i < n; i++)
        {
            switch (i % 7)
            {
            case 0: o << i * 0.37 << " " << i << "@stack" << i / 100 << ".mrcs " << i % 13 << " 12345.678901234567 mic" << i << ".mrc 0.5\n"; break;
            case 1: o << "-" << i << ".5e-3\t'a quoted  name'  " << "+" << i << " 1e5 \"mic with space.mrc\" 1e-320\n"; break;
            case 2: o << "  " << i << "   x@y.mrcs   0x10   nan   mic.mrc   inf\n"; break;
            case 3: o << ".5 s " << i << ".0 -.25 m 3.\n"; break;
            case 4: o << i << " s 99999999999999999999 1e999 m 0.1234567890123456789\n"; break;
            case 5: o << "1.5abc s 7 5 m 2\n"; break;
            default: o << i << " s " << i << " 0.1 m 0.30000000000000004\n";
            }
        }
        o << "\ndata_after\n\nloop_\n_rlnGroupNumber #1\n3\n4\n";
    }

    auto readWith = [&](const char *threads, std::string &particles, std::string &after, std::string &optics)
    {
        if (threads) setenv("RELION_STARTUP_THREADS", threads, 1);
        else unsetenv("RELION_STARTUP_THREADS");
        MetaDataTable a, b, c;
        REQUIRE(a.read(fn, "particles") == n);
        particles = writeTable(a);
        b.read(fn, "after");
        after = writeTable(b);
        c.read(fn, "optics");
        optics = writeTable(c);
    };

    std::string p0, a0, o0, p1, a1, o1, p4, a4, o4;
    readWith("0", p0, a0, o0);
    readWith(nullptr, p1, a1, o1);
    readWith("4", p4, a4, o4);
    unsetenv("RELION_STARTUP_THREADS");
    std::remove(fn.c_str());

    REQUIRE(p1 == p0);
    REQUIRE(p4 == p0);
    REQUIRE(a1 == a0);
    REQUIRE(a4 == a0);
    REQUIRE(o1 == o0);
    REQUIRE(o4 == o0);
    REQUIRE(a0.find("3") != std::string::npos);
}

TEST_CASE("MDT: a bad row is reported by the threaded reader as by the serial one", "[metadata][star_parse]")
{
    const std::string fn = "relion_star_parse_bad.star";
    {
        std::ofstream o(fn);
        o << "data_x\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnGroupNumber #3\n";
        for (int i = 0; i < 20000; i++)
            o << i << " " << i << (i == 12345 ? "\n" : " 1\n");
    }
    std::string msg0, msg4;
    for (const char *t : {"0", "4"})
    {
        setenv("RELION_STARTUP_THREADS", t, 1);
        MetaDataTable m;
        try { m.read(fn, "x"); FAIL("no error"); }
        catch (const RelionError &e) { (std::string(t) == "0" ? msg0 : msg4) = e.msg; }
    }
    unsetenv("RELION_STARTUP_THREADS");
    std::remove(fn.c_str());
    REQUIRE(!msg0.empty());
    REQUIRE(msg4 == msg0);
}
