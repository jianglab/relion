/***************************************************************************
 *
 * Author: "Wen Jiang"
 * Pennsylvania State University
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef CACHE_MANAGER_H_
#define CACHE_MANAGER_H_

#include <string>
#include <vector>
#include <ctime>
#include "src/filename.h"

class Experiment;

class CacheManager
{
public:
    static const int LOCK_TIMEOUT_SEC = 120;

    CacheManager() : capacityGb(0) {}

    void setCacheDir(const FileName &dir);
    void setRegistryDir(const FileName &dir);
    void setCapacity(RFLOAT gb) { capacityGb = gb; }

    std::string computeKey(const std::string &projectRoot, const std::string &sourceJobDir) const;

    FileName getCacheDirForKey(const std::string &key) const;
    FileName getCachePathForKey(const std::string &key, const FileName &relativePath) const;

    bool isCached(const std::string &key, const std::vector<FileName> &relativePaths) const;

    static std::vector<FileName> findSourceJobs(Experiment &exp);

    bool populateCache(const std::string &key,
                       const std::vector<FileName> &relativeStackPaths,
                       const FileName &projectRoot,
                       int verb, int cacheCopyThreads);

    FileName getRegistryPath() const;

    struct RegistryEntry
    {
        std::string key;
        std::string sourceJobDir;
        std::string nodeName;
        std::string cachePath;
        time_t created;
        time_t lastAccess;
        long sizeBytes;
        int nrParticles;
        int nrOpticsGroups;
        std::string projectRoot;
    };

    void touchRegistry(const std::string &key,
                       const std::string &sourceJobDir,
                       int nrParticles_ = -1,
                       int nrOpticsGroups_ = -1,
                       const std::string &projectRoot = "");
    void readRegistry(std::vector<RegistryEntry> &entries) const;
    int cleanup(int maxAgeDays, RFLOAT maxSizeGb, const std::string &nodeName = "");

    bool evict(const std::string &key, const std::string &nodeName = "");
    bool evictProject(const FileName &projectRoot, bool dryRun = false);
    void evictIfFull();

    bool acquireLock(const std::string &key, int timeoutSecs = LOCK_TIMEOUT_SEC);
    void releaseLock(const std::string &key);

    static std::string hashString(const std::string &s);

private:
    FileName cacheDir;
    FileName registryDir;
    RFLOAT capacityGb;

    void ensureCacheDirExists() const;
    void writeRegistry(const std::vector<RegistryEntry> &entries) const;
    void cleanOrphanedTmpFiles(const FileName &regPath) const;
    std::string csvEscape(const std::string &s) const;
    long dirSize(const FileName &path) const;

    FileName getLockDirForKey(const std::string &key) const;
    bool tryAcquireLock(const FileName &lockDir, int timeoutSecs);
    void breakStaleLock(const FileName &lockDir);

    bool acquireRegistryLock(int timeoutSecs = LOCK_TIMEOUT_SEC);
    void releaseRegistryLock();
};

#endif
