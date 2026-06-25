/*
 * tests/unit/test_filament_consistency.cpp
 *
 * Unit tests for enforceFilamentConsistency: majority-vote class assignment
 * per helical filament across multiple scenarios.
 */

#include <catch2/catch.hpp>

#include "src/metadata_table.h"
#include "src/ml_optimiser.h"

// ---------------------------------------------------------------------------
// Helper: build a MetaDataTable with a set of particles
// ---------------------------------------------------------------------------
static MetaDataTable makeParticles(
    const std::vector<std::string> &micrographs,
    const std::vector<int>         &tube_ids,
    const std::vector<int>         &classes)
{
    REQUIRE(micrographs.size() == tube_ids.size());
    REQUIRE(tube_ids.size() == classes.size());

    MetaDataTable mdt;
    for (size_t i = 0; i < micrographs.size(); i++)
    {
        mdt.addObject();
        mdt.setValue(EMDL_MICROGRAPH_NAME,          micrographs[i], i);
        mdt.setValue(EMDL_PARTICLE_HELICAL_TUBE_ID, tube_ids[i],    i);
        mdt.setValue(EMDL_PARTICLE_CLASS,           classes[i],     i);
    }
    return mdt;
}

// ---------------------------------------------------------------------------
// Helper: read back class values
// ---------------------------------------------------------------------------
static std::vector<int> getClasses(const MetaDataTable &mdt)
{
    std::vector<int> out(mdt.numberOfObjects());
    for (long int i = 0; i < mdt.numberOfObjects(); i++)
    {
        int c = -1;
        mdt.getValue(EMDL_PARTICLE_CLASS, c, i);
        out[i] = c;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("filament: single filament, all same class", "[filament]")
{
    // 4 particles, all class 1
    auto mdt = makeParticles(
        {"mic1","mic1","mic1","mic1"},
        {1,1,1,1},
        {1,1,1,1});

    enforceFilamentConsistency(mdt, 3, 0);

    auto cls = getClasses(mdt);
    for (auto c : cls)
        REQUIRE(c == 1);
}

TEST_CASE("filament: single filament, majority wins", "[filament]")
{
    // 5 particles: 3 in class 1, 1 in class 2, 1 in class 3
    auto mdt = makeParticles(
        {"mic1","mic1","mic1","mic1","mic1"},
        {1,1,1,1,1},
        {1,1,1,2,3});

    enforceFilamentConsistency(mdt, 3, 0);

    auto cls = getClasses(mdt);
    for (auto c : cls)
        REQUIRE(c == 1);
}

TEST_CASE("filament: two filaments, independent", "[filament]")
{
    // Filament A (mic1/tube1): 2x class1, 2x class2  → tie → first majority = 1
    // Filament B (mic1/tube2): 2x class2, 1x class3  → class 2 wins
    auto mdt = makeParticles(
        {"mic1","mic1","mic1","mic1","mic1","mic1","mic1"},
        {1,     1,     1,     1,     2,     2,     2},
        {1,     1,     2,     2,     2,     2,     3});

    enforceFilamentConsistency(mdt, 3, 0);

    auto cls = getClasses(mdt);
    // Filament A (indices 0-3): all class 1
    for (int i = 0; i < 4; i++)
        REQUIRE(cls[i] == 1);
    // Filament B (indices 4-6): all class 2
    for (int i = 4; i < 7; i++)
        REQUIRE(cls[i] == 2);
}

TEST_CASE("filament: single-particle filament unchanged", "[filament]")
{
    // Filament A: 1 particle class 1 (should keep)
    // Filament B: 4 particles, 3 class 2 → all become class 2
    auto mdt = makeParticles(
        {"mic1","mic2","mic2","mic2","mic2"},
        {1,     1,     1,     1,     1},
        {1,     1,     2,     3,     2});

    enforceFilamentConsistency(mdt, 3, 0);

    auto cls = getClasses(mdt);
    REQUIRE(cls[0] == 1);  // single particle unchanged
    for (int i = 1; i < 5; i++)
        REQUIRE(cls[i] == 2);  // B becomes class 2
}

TEST_CASE("filament: tie broken to lower class number", "[filament]")
{
    // 2 particles class 1, 2 particles class 2 → tie → majority = 1 (lower)
    auto mdt = makeParticles(
        {"mic1","mic1","mic1","mic1"},
        {1,1,1,1},
        {2,2,1,1});

    enforceFilamentConsistency(mdt, 3, 0);

    auto cls = getClasses(mdt);
    for (auto c : cls)
        REQUIRE(c == 1);
}

TEST_CASE("filament: single particle unchanged", "[filament]")
{
    // A single particle cannot form a filament group, so it stays unchanged
    auto mdt = makeParticles(
        {"mic1"},
        {1},
        {3});

    enforceFilamentConsistency(mdt, 3, 0);

    auto cls = getClasses(mdt);
    REQUIRE(cls[0] == 3);
}

TEST_CASE("filament: nr_classes <= 1 returns early", "[filament]")
{
    auto mdt = makeParticles(
        {"mic1","mic1","mic1"},
        {1,1,1},
        {1,2,3});

    enforceFilamentConsistency(mdt, 1, 0);

    auto cls = getClasses(mdt);
    REQUIRE(cls[0] == 1);
    REQUIRE(cls[1] == 2);
    REQUIRE(cls[2] == 3);
}

TEST_CASE("filament: three filaments with scattered indices", "[filament]")
{
    // Filaments interleaved in the table:
    // idx 0: mic1/tube1 class 1
    // idx 1: mic1/tube2 class 3
    // idx 2: mic1/tube1 class 2  (same filament as idx 0)
    // idx 3: mic1/tube2 class 2
    // idx 4: mic1/tube1 class 1  (same filament as idx 0,2)
    // Filament A (mic1/tube1): classes {1,2,1} → majority 1
    // Filament B (mic1/tube2): classes {3,2} → tie → majority 3 (lower/higher?)
    //   std::map keys: 2→1, 3→1; iteration order: 2 first, max_count=1, majority=2;
    //   3: count=1, max=1, not >, so stays 2
    // Actually wait: filament B has 2 particles: class 3 at idx 1, class 2 at idx 3
    //   class_counts = {2:1, 3:1}
    //   Iteration: key=2, count=1 > max_count(0) → max_count=1, majority=2
    //              key=3, count=1 > max_count(1)? No → majority stays 2
    // So all of filament B becomes class 2

    auto mdt = makeParticles(
        {"mic1","mic1","mic1","mic1","mic1"},
        {1,     2,     1,     2,     1},
        {1,     3,     2,     2,     1});

    enforceFilamentConsistency(mdt, 3, 0);

    auto cls = getClasses(mdt);
    // Filament A (idx 0,2,4): all class 1
    REQUIRE(cls[0] == 1);
    REQUIRE(cls[2] == 1);
    REQUIRE(cls[4] == 1);
    // Filament B (idx 1,3): tie, first majority = 2
    REQUIRE(cls[1] == 2);
    REQUIRE(cls[3] == 2);
}
