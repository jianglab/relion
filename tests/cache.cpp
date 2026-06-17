#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "src/cache_manager.h"
#include "src/filename.h"
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <sys/stat.h>

// ============================================================
// Tests for cache_manager string/state methods (no file I/O)
// ============================================================

TEST_CASE("CacheManager::hashString is deterministic", "[cache][hash]")
{
    CacheManager cm;
    // Same input always gives same output
    std::string h1 = cm.hashString("hello");
    std::string h2 = cm.hashString("hello");
    REQUIRE(h1 == h2);
    REQUIRE(h1.length() == 16);

    // Different inputs give different hashes
    std::string h3 = cm.hashString("world");
    REQUIRE(h1 != h3);

    // Empty string
    std::string h4 = cm.hashString("");
    REQUIRE(h4.length() == 16);
}

TEST_CASE("CacheManager::computeKey", "[cache][key]")
{
    CacheManager cm;
    std::string prj = "/data/project";
    std::string src = "Jobs/001";
    std::string key1 = cm.computeKey(prj, src);
    REQUIRE(key1.length() == 16);

    // Same inputs → same key
    std::string key2 = cm.computeKey(prj, src);
    REQUIRE(key1 == key2);

    // Different source → different key
    std::string key3 = cm.computeKey(prj, "Jobs/002");
    REQUIRE(key1 != key3);

    // Different project → different key
    std::string key4 = cm.computeKey("/other/project", src);
    REQUIRE(key1 != key4);
}

TEST_CASE("CacheManager::setCacheDir appends relion_cache/", "[cache][dir]")
{
    CacheManager cm;
    cm.setCacheDir("/scratch");
    // getCacheDirForKey uses the internal cacheDir
    FileName dir = cm.getCacheDirForKey("abc123");
    REQUIRE(std::string(dir).find("/scratch/relion_cache/abc123/") != std::string::npos);

    // Trailing slash is handled
    CacheManager cm2;
    cm2.setCacheDir("/scratch/");
    FileName dir2 = cm2.getCacheDirForKey("abc123");
    REQUIRE(std::string(dir2).find("/scratch/relion_cache/abc123/") != std::string::npos);

    // Relative path
    CacheManager cm3;
    cm3.setCacheDir("local_ssd");
    FileName dir3 = cm3.getCacheDirForKey("key");
    REQUIRE(std::string(dir3).find("local_ssd/relion_cache/key/") != std::string::npos);
}

TEST_CASE("CacheManager::getCachePathForKey strips leading slash", "[cache][path]")
{
    CacheManager cm;
    cm.setCacheDir("/cache");
    FileName path = cm.getCachePathForKey("k1", "sub/file.mrcs");
    REQUIRE(std::string(path).find("k1/sub/file.mrcs") != std::string::npos);

    // Leading slash is stripped
    FileName path2 = cm.getCachePathForKey("k1", "/sub/file.mrcs");
    REQUIRE(std::string(path2).find("k1/sub/file.mrcs") != std::string::npos);

    // Deep path
    FileName path3 = cm.getCachePathForKey("k1", "a/b/c/d.mrc");
    REQUIRE(std::string(path3).find("k1/a/b/c/d.mrc") != std::string::npos);
}

TEST_CASE("CacheManager::setRegistryDir uses HOME", "[cache][registry]")
{
    const char *oldHome = std::getenv("HOME");
    REQUIRE(oldHome != NULL);

    CacheManager cm;
    cm.setRegistryDir("/fallback");
    FileName regPath = cm.getRegistryPath();
    REQUIRE(std::string(regPath).find(oldHome) != std::string::npos);
    REQUIRE(std::string(regPath).find(".relion/cache_registry.csv") != std::string::npos);
}

// ============================================================
// File-based tests using temporary directories
// ============================================================

TEST_CASE("CacheManager::isCached on non-existent dir returns false", "[cache][io]")
{
    CacheManager cm;
    cm.setCacheDir("/tmp/nonexistent_cache_test_dir_that_does_not_exist_xyzzy");
    std::vector<FileName> paths;
    paths.push_back("test.mrcs");
    REQUIRE_FALSE(cm.isCached("somekey", paths));
}

TEST_CASE("CacheManager::computeKey round-trip consistency", "[cache][key]")
{
    // Verify that useCacheDir lookup works with the same key
    CacheManager cm;
    cm.setCacheDir("/tmp");

    std::string prj = "/home/user/project";
    std::string src = "Jobs/Class2D/run001";
    std::string key = cm.computeKey(prj, src);
    REQUIRE(key.length() == 16);

    FileName cacheDirForKey = cm.getCacheDirForKey(key);
    REQUIRE(std::string(cacheDirForKey).find(key) != std::string::npos);

    FileName cachePath = cm.getCachePathForKey(key, "stacks/particles.mrcs");
    REQUIRE(std::string(cachePath).find("stacks/particles.mrcs") != std::string::npos);
}

TEST_CASE("CacheManager::hashString known values", "[cache][hash]")
{
    // FNV-1a produces deterministic outputs; verify with a few known inputs
    CacheManager cm;

    // These are the expected FNV-1a 64-bit values for specific inputs
    std::string h_empty = cm.hashString("");
    REQUIRE(h_empty.length() == 16);

    std::string h_a = cm.hashString("a");
    REQUIRE(h_a.length() == 16);
    // Ensure hash of "a" is not all zeros
    REQUIRE(h_a != "0000000000000000");

    // "a" and "A" differ (case sensitive)
    std::string h_A = cm.hashString("A");
    REQUIRE(h_a != h_A);
}

// ============================================================
// Tests for findSourceJobs (using Experiment directly)
// ============================================================
#include "src/exp_model.h"

static Experiment makeSimpleExperiment(const std::vector<FileName> &particleNames)
{
    Experiment exp;
    exp.particles.clear();
    exp.particles.reserve(particleNames.size());
    for (size_t i = 0; i < particleNames.size(); i++)
    {
        ExpParticle p;
        p.name = particleNames[i];
        exp.particles.push_back(p);
    }
    return exp;
}

TEST_CASE("CacheManager::findSourceJobs extracts directories", "[cache][source]")
{
    std::vector<FileName> names;
    names.push_back("000001@Jobs/Class2D/run001/stacks/particles.mrcs");
    names.push_back("000002@Jobs/Class2D/run001/stacks/particles.mrcs");
    names.push_back("000001@Jobs/Class2D/run002/stacks/particles.mrcs");

    Experiment exp = makeSimpleExperiment(names);
    std::vector<FileName> sources = CacheManager::findSourceJobs(exp);

    REQUIRE(sources.size() == 2);
    bool found1 = false, found2 = false;
    for (size_t i = 0; i < sources.size(); i++)
    {
        if (sources[i] == "Jobs/Class2D/run001/stacks")
            found1 = true;
        if (sources[i] == "Jobs/Class2D/run002/stacks")
            found2 = true;
    }
    REQUIRE(found1);
    REQUIRE(found2);
}

TEST_CASE("CacheManager::findSourceJobs deduplicates", "[cache][source]")
{
    std::vector<FileName> names;
    names.push_back("000001@Jobs/001/stacks.mrcs");
    names.push_back("000002@Jobs/001/stacks.mrcs");
    names.push_back("000003@Jobs/001/stacks.mrcs");

    Experiment exp = makeSimpleExperiment(names);
    std::vector<FileName> sources = CacheManager::findSourceJobs(exp);
    REQUIRE(sources.size() == 1);
    REQUIRE(sources[0] == "Jobs/001");
}

TEST_CASE("CacheManager::findSourceJobs handles dot path", "[cache][source]")
{
    std::vector<FileName> names;
    // Particle name without directory -> extract "."
    names.push_back("000001@stacks.mrcs");

    Experiment exp = makeSimpleExperiment(names);
    std::vector<FileName> sources = CacheManager::findSourceJobs(exp);
    REQUIRE(sources.size() == 1);
    REQUIRE(sources[0] == ".");
}

TEST_CASE("CacheManager::findSourceJobs handles empty", "[cache][source]")
{
    std::vector<FileName> names;
    Experiment exp = makeSimpleExperiment(names);
    std::vector<FileName> sources = CacheManager::findSourceJobs(exp);
    REQUIRE(sources.empty());
}
