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

#include "src/cache_manager.h"
#include "src/exp_model.h"
#include "src/funcs.h"
#include <cstdlib>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <climits>
#include <iomanip>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <set>
#include <thread>
#include <mutex>
#include <atomic>
#include <unistd.h>
#include <cerrno>
#include <signal.h>

static size_t fnv1a(const std::string &s)
{
    size_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < s.size(); i++)
    {
        hash ^= static_cast<size_t>(static_cast<unsigned char>(s[i]));
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string CacheManager::hashString(const std::string &s)
{
    size_t h = fnv1a(s);
    char hex[17];
    int ret = std::snprintf(hex, sizeof(hex), "%016zx", h);
    if (ret < 0 || ret >= (int)sizeof(hex))
        return "0000000000000000";
    return std::string(hex);
}

void CacheManager::setCacheDir(const FileName &dir)
{
    cacheDir = dir;
    if (cacheDir.length() > 0 && cacheDir[cacheDir.length() - 1] != '/')
        cacheDir += '/';
    cacheDir = cacheDir + "relion_cache/";
}

void CacheManager::setRegistryDir(const FileName &dir)
{
    const char *home = std::getenv("HOME");
    if (home != NULL)
    {
        registryDir = std::string(home) + "/.relion/";
    }
    else
    {
        registryDir = dir;
        if (registryDir.length() > 0 && registryDir[registryDir.length() - 1] != '/')
            registryDir += '/';
        registryDir += ".relion/";
    }
}

FileName CacheManager::getRegistryPath() const
{
    if (registryDir != "")
        return registryDir + "cache_registry.csv";
    return FileName("cache_registry.csv");
}

FileName CacheManager::getCacheDirForKey(const std::string &key) const
{
    return cacheDir + key + "/";
}

FileName CacheManager::getCachePathForKey(const std::string &key, const FileName &relativePath) const
{
    FileName clean = relativePath;
    if (clean[0] == '/')
        clean = clean.substr(1);
    return getCacheDirForKey(key) + clean;
}

FileName CacheManager::getLockDirForKey(const std::string &key) const
{
    return cacheDir + "locks/" + key + ".lock/";
}

void CacheManager::ensureCacheDirExists() const
{
    if (!exists(cacheDir))
        mktree(cacheDir, 0777);
}

std::vector<FileName> CacheManager::findSourceJobs(Experiment &exp)
{
    std::set<FileName> sources;
    for (long int p = 0; p < exp.numberOfParticles(); p++)
    {
        long int imgno;
        FileName fn_stack;
        exp.particles[p].name.decompose(imgno, fn_stack);
        // Extract the directory part of the stack path (relative to project root)
        FileName dirPart = fn_stack.beforeLastOf("/");
        if (dirPart == fn_stack)
            dirPart = ".";
        sources.insert(dirPart);
    }
    std::vector<FileName> result(sources.begin(), sources.end());
    return result;
}

std::string CacheManager::computeKey(const std::string &projectRoot, const std::string &sourceJobDir) const
{
    return hashString(projectRoot + "|" + sourceJobDir);
}

bool CacheManager::isCached(const std::string &key, const std::vector<FileName> &relativePaths) const
{
    if (relativePaths.empty())
        return false;

    FileName keyDir = getCacheDirForKey(key);
    if (!exists(keyDir))
        return false;

    for (size_t i = 0; i < relativePaths.size(); i++)
    {
        FileName cached = getCachePathForKey(key, relativePaths[i]);
        if (!exists(cached))
            return false;

        // Verify size matches original
        FileName src = relativePaths[i];
        if (exists(src))
        {
            struct stat srcStat, cachedStat;
            if (stat(src.c_str(), &srcStat) != 0)
                return false;
            if (stat(cached.c_str(), &cachedStat) != 0)
                return false;
            if (cachedStat.st_size != srcStat.st_size)
                return false;
            if (cachedStat.st_mtime != srcStat.st_mtime)
                return false;
        }
    }
    return true;
}

bool CacheManager::populateCache(const std::string &key,
                                  const std::vector<FileName> &relativeStackPaths,
                                  const FileName &projectRoot,
                                  int verb, int cacheCopyThreads)
{
    FileName keyDir = getCacheDirForKey(key);
    ensureCacheDirExists();
    if (!exists(keyDir))
        mktree(keyDir, 0777);

    // Determine which stacks need copying (skip if size+mtime match)
    std::vector<FileName> toCopy;
    std::vector<long int> sizesToCopy;
    long int totalBytes = 0;
    long int nrSkipped = 0;
    for (size_t i = 0; i < relativeStackPaths.size(); i++)
    {
        FileName srcPath = (projectRoot != "" && relativeStackPaths[i][0] != '/') ? FileName(projectRoot + "/" + relativeStackPaths[i]) : relativeStackPaths[i];
        FileName cachePath = getCachePathForKey(key, relativeStackPaths[i]);
        if (exists(cachePath))
        {
            struct stat srcStat, cachedStat;
            if (stat(srcPath.c_str(), &srcStat) == 0 && stat(cachePath.c_str(), &cachedStat) == 0)
            {
                if (cachedStat.st_size == srcStat.st_size && cachedStat.st_mtime == srcStat.st_mtime)
                {
                    nrSkipped++;
                    continue;
                }
            }
        }
        toCopy.push_back(relativeStackPaths[i]);
        long int sz = (long int)srcPath.getFileSize();
        sizesToCopy.push_back(sz);
        totalBytes += sz;
    }

    if (toCopy.empty())
    {
        if (verb > 0)
            std::cout << " All stacks already cached for this source job. Nothing to copy." << std::endl;
        return true;
    }

        if (verb > 0)
        {
            double gb = totalBytes / (1024.0 * 1024.0 * 1024.0);
            if (nrSkipped > 0)
                std::cout << " " << nrSkipped << " of " << relativeStackPaths.size()
                          << " image stacks already cached, copying " << toCopy.size()
                          << " remaining (" << std::fixed << std::setprecision(1) << gb << " GB)." << std::endl;
            else
                std::cout << " Copying " << toCopy.size() << " image stacks ("
                          << std::fixed << std::setprecision(1) << gb
                          << " GB) to cache directory: " << keyDir << std::endl;
        }

    long int nr_to_copy = toCopy.size();
    if (cacheCopyThreads < 1) cacheCopyThreads = 1;
    if (cacheCopyThreads > nr_to_copy) cacheCopyThreads = (int)nr_to_copy;

    bool allOK = true;

    if (cacheCopyThreads < 2)
    {
        if (verb > 0)
            init_progress_bar(nr_to_copy);
        int barstep = XMIPP_MAX(1, nr_to_copy / 60);
        long int bytesCopied = 0;
        time_t start = time(NULL);
        for (long int i = 0; i < nr_to_copy; i++)
        {
            FileName srcPath = (projectRoot != "" && toCopy[i][0] != '/') ? projectRoot + "/" + toCopy[i] : toCopy[i];
            FileName cachePath = getCachePathForKey(key, toCopy[i]);
            FileName parent = cachePath.beforeLastOf("/");
            if (parent != cachePath && !exists(parent))
                mktree(parent, 0777);
            std::string cmd = "cp -f --preserve=timestamps \"" + srcPath + "\" \"" + cachePath + "\"";
            if (system(cmd.c_str()) != 0)
            {
                std::cerr << " ERROR: failed to copy " << srcPath << " to " << cachePath << std::endl;
                allOK = false;
            }
            bytesCopied += sizesToCopy[i];
            if (verb > 0 && (i + 1) % barstep == 0)
            {
                time_t now = time(NULL);
                long int elapsed = now - start;
                double mbPerSec = (elapsed > 0) ? (double)bytesCopied / (1024.0 * 1024.0) / elapsed : 0.0;
                progress_bar(i + 1);
                fprintf(stdout, " [%ld/%ld  %.0f MB/s  1 thread]", i + 1, nr_to_copy, mbPerSec);
                fflush(stdout);
            }
        }
        if (verb > 0)
        {
            progress_bar(nr_to_copy);
            fflush(stdout);
        }
    }
    else
    {
        std::mutex mtx;
        std::atomic<long int> filesCopied(0);
        std::atomic<long int> bytesCopied(0);
        long int nextIdx = 0;

        auto worker = [&]()
        {
            while (true)
            {
                long int idx;
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    if (nextIdx >= nr_to_copy) break;
                    idx = nextIdx++;
                }
                FileName srcPath = (projectRoot != "" && toCopy[idx][0] != '/') ? projectRoot + "/" + toCopy[idx] : toCopy[idx];
                FileName cachePath = getCachePathForKey(key, toCopy[idx]);
                FileName parent = cachePath.beforeLastOf("/");
                if (parent != cachePath && !exists(parent))
                    mktree(parent, 0777);
                std::string cmd = "cp -f --preserve=timestamps \"" + srcPath + "\" \"" + cachePath + "\"";
                if (system(cmd.c_str()) != 0)
                {
                    std::cerr << " ERROR: failed to copy " << srcPath << " to " << cachePath << std::endl;
                    allOK = false;
                }
                bytesCopied += sizesToCopy[idx];
                filesCopied++;
            }
        };

        if (verb > 0)
            init_progress_bar(nr_to_copy);

        std::vector<std::thread> workers;
        for (int t = 0; t < cacheCopyThreads; t++)
            workers.emplace_back(worker);

        if (verb > 0)
        {
            time_t start = time(NULL);
            while (true)
            {
                sleep(1);
                long int done = filesCopied;
                if (done >= nr_to_copy) break;
                time_t now = time(NULL);
                long int elapsed = now - start;
                if (elapsed < 1) continue;
                double mbPerSec = (double)bytesCopied / (1024.0 * 1024.0) / elapsed;
                long int eta = (done > 0) ? (long int)(elapsed * (double)(nr_to_copy - done) / done) : 0;
                progress_bar(done);
                fprintf(stdout, " [%ld/%ld  %.0f MB/s  %d threads  ETA %lds]",
                        done, nr_to_copy, mbPerSec, cacheCopyThreads, eta);
                fflush(stdout);
            }
        }

        for (auto &t : workers)
            t.join();

        if (verb > 0)
        {
            progress_bar(nr_to_copy);
            fflush(stdout);
        }
    }

    // Set permissions
    std::string permCmd = "chmod -R 777 " + keyDir;
    if (system(permCmd.c_str()) != 0)
        std::cerr << " WARNING: failed to set permissions on " << keyDir << std::endl;

    return allOK;
}

void CacheManager::cleanOrphanedTmpFiles(const FileName &regPath) const
{
    FileName pattern = regPath + ".tmp.*";
    std::vector<FileName> orphans;
    pattern.globFiles(orphans);
    for (size_t i = 0; i < orphans.size(); i++)
        std::remove(orphans[i].c_str());
}

static std::string csvUnescape(const std::string &field)
{
    if (field.size() < 2 || field[0] != '"')
        return field;
    std::string inner = field.substr(1, field.size() - 2);
    std::string result;
    for (size_t i = 0; i < inner.size(); i++)
    {
        if (inner[i] == '"' && i + 1 < inner.size() && inner[i + 1] == '"')
        {
            result += '"';
            i++;
        }
        else
        {
            result += inner[i];
        }
    }
    return result;
}

std::string CacheManager::csvEscape(const std::string &s) const
{
    if (s.find(',') != std::string::npos || s.find('"') != std::string::npos || s.find(' ') != std::string::npos)
    {
        std::string e = s;
        size_t p = 0;
        while ((p = e.find('"', p)) != std::string::npos)
        {
            e.insert(p, "\"");
            p += 2;
        }
        return "\"" + e + "\"";
    }
    return s;
}

void CacheManager::readRegistry(std::vector<RegistryEntry> &entries) const
{
    entries.clear();
    FileName regPath = getRegistryPath();
    cleanOrphanedTmpFiles(regPath);

    if (!exists(regPath))
        return;

    struct stat regStat;
    if (stat(regPath.c_str(), &regStat) == 0 && S_ISDIR(regStat.st_mode))
    {
        ::rmdir(regPath.c_str());
        return;
    }

    long fileSize = regPath.getFileSize();
    if (fileSize < 1)
        return;

    std::ifstream in(regPath.c_str());
    if (!in.is_open())
        return;

    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#')
            continue;

        std::vector<std::string> fields;
        std::string field;
        bool inQuotes = false;
        for (size_t i = 0; i < line.size(); i++)
        {
            char c = line[i];
            if (c == '"')
                inQuotes = !inQuotes;
            else if (c == ',' && !inQuotes)
            {
                fields.push_back(field);
                field.clear();
            }
            else
            {
                field += c;
            }
        }
        fields.push_back(field);

        if (fields.size() < 6)
            continue;

        RegistryEntry e;
        e.projectRoot = "";
        e.sourceJobDir = "";
        if (fields.size() >= 10)
        {
            e.key = csvUnescape(fields[0]);
            e.sourceJobDir = csvUnescape(fields[1]);
            e.nodeName = csvUnescape(fields[2]);
            e.cachePath = csvUnescape(fields[3]);
            e.created = (time_t)std::atol(fields[4].c_str());
            e.lastAccess = (time_t)std::atol(fields[5].c_str());
            e.sizeBytes = std::atol(fields[6].c_str());
            e.nrParticles = std::atoi(fields[7].c_str());
            e.nrOpticsGroups = std::atoi(fields[8].c_str());
            e.projectRoot = csvUnescape(fields[9]);
        }
        else if (fields.size() >= 9)
        {
            // Old format (9 fields): key,nodeName,cachePath,created,lastAccess,size,nrParticles,nrOpticsGroups,projectRoot
            e.key = csvUnescape(fields[0]);
            e.nodeName = csvUnescape(fields[1]);
            e.cachePath = csvUnescape(fields[2]);
            e.created = (time_t)std::atol(fields[3].c_str());
            e.lastAccess = (time_t)std::atol(fields[4].c_str());
            e.sizeBytes = std::atol(fields[5].c_str());
            e.nrParticles = std::atoi(fields[6].c_str());
            e.nrOpticsGroups = std::atoi(fields[7].c_str());
            e.projectRoot = csvUnescape(fields[8]);
        }
        else
        {
            e.key = csvUnescape(fields[0]);
            e.nodeName = "";
            e.cachePath = "";
            e.created = (time_t)std::atol(fields[1].c_str());
            e.lastAccess = (time_t)std::atol(fields[2].c_str());
            e.sizeBytes = std::atol(fields[3].c_str());
            e.nrParticles = std::atoi(fields[4].c_str());
            e.nrOpticsGroups = std::atoi(fields[5].c_str());
        }
        entries.push_back(e);
    }
}

void CacheManager::writeRegistry(const std::vector<RegistryEntry> &entries) const
{
    FileName regPath = getRegistryPath();
    FileName regDir = "";
    if (regPath.rfind("/") != std::string::npos)
        regDir = regPath.beforeLastOf("/");
    if (regDir != "" && !exists(regDir))
        mktree(regDir);
    cleanOrphanedTmpFiles(regPath);

    FileName tmpPath = regPath + ".tmp." + integerToString(getpid());

    std::ofstream out(tmpPath.c_str());
    if (!out.is_open())
    {
        std::cout << "+++ WARNING: cache_manager: cannot write registry to "
                  << tmpPath << std::endl;
        return;
    }

    out << "# key,source_job_dir,node_name,cache_path,created_epoch,last_access_epoch,"
        << "size_bytes,nr_particles,nr_optics_groups,project_root\n";

    for (size_t i = 0; i < entries.size(); i++)
    {
        out << entries[i].key << ","
            << csvEscape(entries[i].sourceJobDir) << ","
            << csvEscape(entries[i].nodeName) << ","
            << csvEscape(entries[i].cachePath) << ","
            << (long)entries[i].created << ","
            << (long)entries[i].lastAccess << ","
            << entries[i].sizeBytes << ","
            << entries[i].nrParticles << ","
            << entries[i].nrOpticsGroups << ","
            << csvEscape(entries[i].projectRoot) << "\n";
    }
    out.close();

    if (out.fail())
    {
        std::cout << "+++ WARNING: cache_manager: error writing registry to "
                  << tmpPath << "; removing" << std::endl;
        std::remove(tmpPath.c_str());
        return;
    }

    struct stat regStat;
    if (stat(regPath.c_str(), &regStat) == 0 && S_ISDIR(regStat.st_mode))
        ::rmdir(regPath.c_str());

    if (std::rename(tmpPath.c_str(), regPath.c_str()) != 0)
    {
        std::cout << "+++ WARNING: cache_manager: failed to rename registry "
                  << tmpPath << " -> " << regPath << std::endl;
    }
}

void CacheManager::touchRegistry(const std::string &key,
                                  const std::string &sourceJobDir,
                                  int nrParticles_, int nrOpticsGroups_,
                                  const std::string &projectRoot_)
{
    acquireRegistryLock();

    char host[256];
    host[0] = '\0';
    if (gethostname(host, sizeof(host)) != 0)
        host[0] = '\0';
    host[sizeof(host) - 1] = '\0';
    std::string nodeName(host);

    std::string cachePathStr = getCacheDirForKey(key).c_str();
    long sizeBytes = dirSize(cachePathStr);

    std::vector<RegistryEntry> entries;
    readRegistry(entries);

    time_t now = std::time(NULL);
    bool found = false;
    for (size_t i = 0; i < entries.size(); i++)
    {
        if (entries[i].key == key && entries[i].nodeName == nodeName)
        {
            entries[i].lastAccess = now;
            if (entries[i].cachePath == "")
                entries[i].cachePath = cachePathStr;
            entries[i].sizeBytes = sizeBytes;
            if (entries[i].sourceJobDir == "" && sourceJobDir != "")
                entries[i].sourceJobDir = sourceJobDir;
            if (entries[i].nrParticles == 0 && nrParticles_ >= 0)
                entries[i].nrParticles = nrParticles_;
            if (entries[i].nrOpticsGroups == 0 && nrOpticsGroups_ >= 0)
                entries[i].nrOpticsGroups = nrOpticsGroups_;
            if (entries[i].projectRoot == "" && projectRoot_ != "")
                entries[i].projectRoot = projectRoot_;
            found = true;
            break;
        }
    }

    if (!found)
    {
        RegistryEntry e;
        e.key = key;
        e.sourceJobDir = sourceJobDir;
        e.nodeName = nodeName;
        e.cachePath = cachePathStr;
        e.created = now;
        e.lastAccess = now;
        e.sizeBytes = sizeBytes;
        e.nrParticles = (nrParticles_ >= 0) ? nrParticles_ : 0;
        e.nrOpticsGroups = (nrOpticsGroups_ >= 0) ? nrOpticsGroups_ : 0;
        e.projectRoot = projectRoot_;
        entries.push_back(e);
    }

    writeRegistry(entries);

    releaseRegistryLock();
}

long CacheManager::dirSize(const FileName &path) const
{
    long total = 0;
    std::string cmd = "du -sb \"" + path + "\" 2>/dev/null";
    FILE *fp = popen(cmd.c_str(), "r");
    if (!fp)
        return 0;
    char buf[128];
    if (fgets(buf, sizeof(buf), fp))
        total = atol(buf);
    pclose(fp);
    return total;
}

bool CacheManager::evict(const std::string &key, const std::string &nodeName)
{
    acquireLock(key);

    std::vector<RegistryEntry> entries;
    readRegistry(entries);
    std::vector<RegistryEntry> remaining;
    bool allSuccess = true;

    for (size_t i = 0; i < entries.size(); i++)
    {
        bool match = (entries[i].key == key);
        if (nodeName != "")
            match = match && (entries[i].nodeName == nodeName);

        if (!match)
        {
            remaining.push_back(entries[i]);
        }
        else
        {
            bool removalOK = true;

            if (entries[i].cachePath != "" && entries[i].nodeName != "")
            {
                char host[256] = "";
                if (gethostname(host, sizeof(host)) == 0)
                {
                    host[sizeof(host) - 1] = '\0';
                    if (entries[i].nodeName != host)
                    {
                        std::string cmd = "ssh " + entries[i].nodeName
                                        + " 'rm -rf \"" + entries[i].cachePath + "\"' 2>/dev/null";
                        int ret = std::system(cmd.c_str());
                        if (ret != 0)
                        {
                            std::cerr << " WARNING: failed to remove cache on "
                                      << entries[i].nodeName << ": "
                                      << entries[i].cachePath << std::endl;
                            removalOK = false;
                        }
                    }
                }
            }

            FileName dir(entries[i].cachePath);
            if (dir != "" && exists(dir))
            {
                std::string cmd = "rm -rf \"" + dir + "\"";
                int ret = std::system(cmd.c_str());
                if (ret != 0)
                {
                    std::cerr << " WARNING: failed to remove local cache: "
                              << dir << std::endl;
                    removalOK = false;
                }
            }

            if (!removalOK)
            {
                remaining.push_back(entries[i]);
                allSuccess = false;
            }
        }
    }

    if (remaining.size() < entries.size())
        writeRegistry(remaining);

    releaseLock(key);
    return allSuccess;
}

bool CacheManager::evictProject(const FileName &projectRoot, bool dryRun)
{
    std::string projPrefix = projectRoot;
    if (projPrefix.length() > 0 && projPrefix[projPrefix.length() - 1] != '/')
        projPrefix += "/";

    // Read registry from the project directory
    setRegistryDir(projectRoot);
    std::vector<RegistryEntry> entries;
    readRegistry(entries);

    if (entries.empty())
    {
        std::cout << " No cache entries found for project " << projectRoot << std::endl;
        return true;
    }

    std::vector<RegistryEntry> remaining;
    int total = 0, succeeded = 0;

    for (size_t i = 0; i < entries.size(); i++)
    {
        if (entries[i].projectRoot != projPrefix)
        {
            remaining.push_back(entries[i]);
            continue;
        }

        total++;

        if (dryRun)
        {
            std::cout << " Would evict key=" << entries[i].key
                      << " source=" << entries[i].sourceJobDir
                      << " node=" << entries[i].nodeName
                      << " path=" << entries[i].cachePath
                      << std::endl;
            remaining.push_back(entries[i]);
            continue;
        }

        bool removalOK = true;

        if (entries[i].cachePath != "" && entries[i].nodeName != "")
        {
            char host[256] = "";
            if (gethostname(host, sizeof(host)) == 0)
            {
                host[sizeof(host) - 1] = '\0';
                if (entries[i].nodeName != host)
                {
                    std::string cmd = "ssh " + entries[i].nodeName
                                    + " 'rm -rf \"" + entries[i].cachePath + "\"' 2>/dev/null";
                    int ret = std::system(cmd.c_str());
                    if (ret != 0)
                    {
                        std::cerr << " WARNING: failed to remove cache on "
                                  << entries[i].nodeName << ": "
                                  << entries[i].cachePath
                                  << " (permission denied or unreachable)"
                                  << std::endl;
                        removalOK = false;
                    }
                }
            }
        }

        FileName dir(entries[i].cachePath);
        if (dir != "" && exists(dir))
        {
            std::string cmd = "rm -rf \"" + dir + "\"";
            int ret = std::system(cmd.c_str());
            if (ret != 0)
            {
                std::cerr << " WARNING: failed to remove local cache: "
                          << dir << " (permission denied)" << std::endl;
                removalOK = false;
            }
        }

        if (removalOK)
            succeeded++;
        else
            remaining.push_back(entries[i]);
    }

    writeRegistry(remaining);

    if (total == 0)
    {
        std::cout << " No cache entries found for project " << projectRoot << std::endl;
    }
    else if (dryRun)
    {
        std::cout << " Dry run: would evict " << total << " entr"
                  << (total == 1 ? "y" : "ies")
                  << " for project " << projectRoot << std::endl;
    }
    else
    {
        std::cout << " Evicted " << succeeded << " of " << total << " entr"
                  << (total == 1 ? "y" : "ies")
                  << " for project " << projectRoot << std::endl;
    }

    return succeeded == total;
}

void CacheManager::evictIfFull()
{
    if (capacityGb <= 0)
        return;

    struct statvfs vfs;
    if (statvfs(cacheDir.c_str(), &vfs) != 0)
        return;

    RFLOAT usedGb = (RFLOAT)(vfs.f_blocks - vfs.f_bfree) * (RFLOAT)vfs.f_bsize
                    / (1024.0 * 1024.0 * 1024.0);

    if (usedGb <= capacityGb)
        return;

    std::vector<RegistryEntry> entries;
    readRegistry(entries);

    std::sort(entries.begin(), entries.end(),
              [](const RegistryEntry &a, const RegistryEntry &b)
              { return a.lastAccess < b.lastAccess; });

    for (size_t i = 0; i < entries.size(); i++)
    {
        evict(entries[i].key);

        if (statvfs(cacheDir.c_str(), &vfs) == 0)
        {
            usedGb = (RFLOAT)(vfs.f_blocks - vfs.f_bfree) * (RFLOAT)vfs.f_bsize
                     / (1024.0 * 1024.0 * 1024.0);
            if (usedGb <= capacityGb)
                break;
        }
    }
}

int CacheManager::cleanup(int maxAgeDays, RFLOAT maxSizeGb, const std::string &nodeName)
{
    std::vector<RegistryEntry> allEntries;
    readRegistry(allEntries);
    if (allEntries.empty())
        return 0;

    std::vector<RegistryEntry> candidates;
    if (nodeName == "")
    {
        candidates = allEntries;
    }
    else
    {
        for (size_t i = 0; i < allEntries.size(); i++)
            if (allEntries[i].nodeName == nodeName)
                candidates.push_back(allEntries[i]);
        if (candidates.empty())
            return 0;
    }

    time_t now = std::time(NULL);
    time_t maxAgeSec = maxAgeDays * 86400;

    std::sort(candidates.begin(), candidates.end(),
              [](const RegistryEntry &a, const RegistryEntry &b)
              { return a.lastAccess < b.lastAccess; });

    int removed = 0;

    for (size_t i = 0; i < candidates.size(); i++)
    {
        double ageSec = difftime(now, candidates[i].lastAccess);
        bool removeByAge = (maxAgeDays > 0 && ageSec > maxAgeSec);
        bool removeBySize = false;

        if (maxSizeGb > 0 && !removeByAge)
        {
            long totalBytes = 0;
            for (size_t j = i; j < candidates.size(); j++)
            {
                FileName dir(candidates[j].cachePath);
                if (dir != "" && exists(dir))
                    totalBytes += dirSize(dir);
            }
            RFLOAT totalGb = (RFLOAT)totalBytes / (1024.0 * 1024.0 * 1024.0);
            if (totalGb > maxSizeGb)
                removeBySize = true;
        }

        if (removeByAge || removeBySize)
        {
            evict(candidates[i].key, candidates[i].nodeName);
            removed++;
        }
    }

    return removed;
}

// Registry lock uses the shared filesystem (registryDir) so all nodes can coordinate
bool CacheManager::acquireRegistryLock(int timeoutSecs)
{
    FileName regPath = getRegistryPath();
    FileName lockDir = regPath + ".lock/";

    FileName lockDir_parent = lockDir.beforeLastOf("/");
    if (!exists(lockDir_parent))
        mktree(lockDir_parent);

    time_t deadline = std::time(NULL) + timeoutSecs;
    while (std::time(NULL) < deadline)
    {
        if (mkdir(lockDir.c_str(), 0700) == 0)
        {
            std::string pidPath = lockDir + "pid";
            std::ofstream pf(pidPath.c_str());
            if (pf.is_open())
            {
                pf << getpid() << std::endl;
                pf.close();
            }
            std::string tsPath = lockDir + "timestamp";
            std::ofstream tf(tsPath.c_str());
            if (tf.is_open())
            {
                tf << std::time(NULL) << std::endl;
                tf.close();
            }
            return true;
        }

        if (errno != EEXIST)
            return false;

        // Check if stale
        std::string tsPath = lockDir + "timestamp";
        std::ifstream tf(tsPath.c_str());
        if (tf.is_open())
        {
            time_t stamp = 0;
            tf >> stamp;
            if (std::time(NULL) - stamp < LOCK_TIMEOUT_SEC)
            {
                std::string pidPath = lockDir + "pid";
                std::ifstream pf(pidPath.c_str());
                if (pf.is_open())
                {
                    pid_t pid = 0;
                    pf >> pid;
                    if (pid > 0 && kill(pid, 0) == 0)
                    {
                        struct timespec ts = {0, 200000000};
                        nanosleep(&ts, NULL);
                        continue;
                    }
                }
            }
        }

        // Break stale lock
        (void)std::remove((lockDir + "pid").c_str());
        (void)std::remove((lockDir + "timestamp").c_str());
        (void)::rmdir(lockDir.c_str());
    }

    return false;
}

void CacheManager::releaseRegistryLock()
{
    FileName regPath = getRegistryPath();
    FileName lockDir = regPath + ".lock/";
    if (!exists(lockDir))
        return;
    (void)std::remove((lockDir + "pid").c_str());
    (void)std::remove((lockDir + "timestamp").c_str());
    (void)::rmdir(lockDir.c_str());
}

bool CacheManager::acquireLock(const std::string &key, int timeoutSecs)
{
    ensureCacheDirExists();

    FileName locksDir = cacheDir + "locks/";
    if (!exists(locksDir))
        mktree(locksDir, 0777);

    FileName lockDir = getLockDirForKey(key);

    time_t deadline = std::time(NULL) + timeoutSecs;
    while (std::time(NULL) < deadline)
    {
        if (tryAcquireLock(lockDir, timeoutSecs))
            return true;

        struct timespec ts = {0, 200000000};
        nanosleep(&ts, NULL);
    }

    return false;
}

void CacheManager::releaseLock(const std::string &key)
{
    FileName lockDir = getLockDirForKey(key);
    if (!exists(lockDir))
        return;

    (void)std::remove((lockDir + "pid").c_str());
    (void)std::remove((lockDir + "timestamp").c_str());
    (void)::rmdir(lockDir.c_str());
}

bool CacheManager::tryAcquireLock(const FileName &lockDir, int timeoutSecs)
{
    if (mkdir(lockDir.c_str(), 0700) == 0)
    {
        std::string pidPath = lockDir + "pid";
        std::ofstream pf(pidPath.c_str());
        if (pf.is_open())
        {
            pf << getpid() << std::endl;
            pf.close();
        }

        std::string tsPath = lockDir + "timestamp";
        std::ofstream tf(tsPath.c_str());
        if (tf.is_open())
        {
            tf << std::time(NULL) << std::endl;
            tf.close();
        }

        return true;
    }

    if (errno != EEXIST)
        return false;

    std::string tsPath = lockDir + "timestamp";
    std::ifstream tf(tsPath.c_str());
    if (tf.is_open())
    {
        time_t stamp = 0;
        tf >> stamp;
        time_t now = std::time(NULL);
        if (now - stamp < LOCK_TIMEOUT_SEC)
        {
            std::string pidPath = lockDir + "pid";
            std::ifstream pf(pidPath.c_str());
            if (pf.is_open())
            {
                pid_t pid = 0;
                pf >> pid;
                if (pid > 0 && kill(pid, 0) == 0)
                    return false;
            }
        }
    }

    breakStaleLock(lockDir);
    return false;
}

void CacheManager::breakStaleLock(const FileName &lockDir)
{
    if (!exists(lockDir))
        return;

    (void)std::remove((lockDir + "pid").c_str());
    (void)std::remove((lockDir + "timestamp").c_str());
    (void)::rmdir(lockDir.c_str());
}
