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

#include <iostream>
#include <ctime>
#include <unistd.h>
#include "src/args.h"
#include "src/cache_manager.h"

static FileName deriveCacheBaseDir(const FileName &cachePath, const std::string &key)
{
    std::string p = cachePath;
    std::string suffix = "relion_cache/" + key + "/";
    size_t pos = p.rfind(suffix);
    if (pos != std::string::npos)
        p = p.substr(0, pos);
    if (!p.empty() && p[p.size() - 1] == '/')
        p.erase(p.size() - 1);
    return p;
}

class CacheCleanupApp
{
public:
    IOParser parser;

    void read(int argc, char **argv)
    {
        parser.setCommandLine(argc, argv);
        int gen_section = parser.addSection("Cache maintenance options");

        fn_registry = parser.getOption("--registry", "Registry file (default: cache_registry.csv in cwd)", "");
        do_list = parser.checkOption("--list", "List cache entries");
        do_evict_current = parser.checkOption("--evict_current", "Evict all cache entries for the current project (cwd-based)");
        evict_project_path = parser.getOption("--evict_project", "Evict all cache entries for a specific project directory", "");
        do_cleanup = parser.checkOption("--cleanup", "Age/size-based cleanup (default mode)");
        node_filter = parser.getOption("--node", "Filter by node name", "");
        max_age = textToInteger(parser.getOption("--max_age", "Max age in days for cleanup (0=no limit)", "30"));
        max_size = textToFloat(parser.getOption("--max_size", "Max cache size in GB for cleanup (0=no limit)", "0"));
        fn_execute = parser.checkOption("--execute", "Actually perform cleanup (default: dry-run only)");
        dry_run = !fn_execute;
    }

    void run()
    {
        CacheManager cm;
        if (fn_registry != "")
            cm.setRegistryDir(fn_registry.beforeLastOf("/"));

        std::vector<CacheManager::RegistryEntry> entries;
        cm.readRegistry(entries);

        // Derive cache base dir from first entry for evictIfFull
        if (!entries.empty())
        {
            FileName base = deriveCacheBaseDir(entries[0].cachePath, entries[0].key);
            cm.setCacheDir(base);
        }

        // Helper to filter entries by node
        auto filterByNode = [&](const std::vector<CacheManager::RegistryEntry> &all)
        {
            if (node_filter == "")
                return all;
            std::vector<CacheManager::RegistryEntry> filtered;
            for (size_t i = 0; i < all.size(); i++)
                if (all[i].nodeName == node_filter)
                    filtered.push_back(all[i]);
            return filtered;
        };

        bool mode_list = do_list;
        bool mode_evict_current = do_evict_current;
        bool mode_evict_project = (evict_project_path != "");
        bool mode_cleanup = do_cleanup || (!mode_list && !mode_evict_current && !mode_evict_project);

        if (mode_list)
        {
            std::vector<CacheManager::RegistryEntry> shown = filterByNode(entries);
            if (shown.empty())
            {
                std::cout << " No matching entries in registry." << std::endl;
                return;
            }
            std::cout << " Source job                              Node          Created          Last Access      Size      Particles  OG  Key" << std::endl;
            std::cout << " " << std::string(140, '-') << std::endl;
            for (size_t i = 0; i < shown.size(); i++)
            {
                char created[32], accessed[32];
                struct tm *tc = gmtime(&shown[i].created);
                struct tm *ta = gmtime(&shown[i].lastAccess);
                std::strftime(created, sizeof(created), "%Y-%m-%d %H:%M:%S", tc);
                std::strftime(accessed, sizeof(accessed), "%Y-%m-%d %H:%M:%S", ta);

                char sizeStr[32];
                double sizeMb = shown[i].sizeBytes / (1024.0 * 1024.0);
                if (sizeMb > 1024)
                    std::snprintf(sizeStr, sizeof(sizeStr), "%.1f GB", sizeMb / 1024.0);
                else
                    std::snprintf(sizeStr, sizeof(sizeStr), "%.0f MB", sizeMb);

                std::string node = shown[i].nodeName.empty() ? "?" : shown[i].nodeName;
                std::string sourceJob = shown[i].sourceJobDir.empty() ? "?" : shown[i].sourceJobDir;
                std::string proj = shown[i].projectRoot.empty() ? "" : shown[i].projectRoot;

                std::string sourceDisplay;
                if (!proj.empty() && !shown[i].sourceJobDir.empty())
                    sourceDisplay = proj + "/" + shown[i].sourceJobDir;
                else if (!shown[i].sourceJobDir.empty())
                    sourceDisplay = shown[i].sourceJobDir;
                else
                    sourceDisplay = proj;
                if (sourceDisplay.length() > 38)
                    sourceDisplay = "..." + sourceDisplay.substr(sourceDisplay.length() - 35);

                std::cout << " " << sourceDisplay.substr(0, 38)
                          << "  " << node.substr(0, 12)
                          << "  " << created
                          << "  " << accessed
                          << "  " << sizeStr
                          << "  " << shown[i].nrParticles
                          << "  " << shown[i].nrOpticsGroups
                          << "  " << shown[i].key.substr(0, 16)
                          << std::endl;
            }
        }
        else if (mode_evict_current)
        {
            if (dry_run)
                std::cout << " DRY-RUN mode: no changes will be made. Pass --execute to proceed." << std::endl;
            else
                std::cout << " LIVE mode: changes will be applied." << std::endl;

            FileName cwd(".");
            char buf[4096];
            if (getcwd(buf, sizeof(buf)))
                cwd = buf;

            if (dry_run)
            {
                std::cout << " DRY RUN: would evict cache for current project: " << cwd << std::endl;
            }
            else
            {
                std::cout << " Evicting cache for current project: " << cwd << " ..." << std::endl;
                cm.evictProject(cwd);
                std::cout << " Done." << std::endl;
            }
        }
        else if (mode_evict_project)
        {
            if (dry_run)
                std::cout << " DRY-RUN mode: no changes will be made. Pass --execute to proceed." << std::endl;
            else
                std::cout << " LIVE mode: changes will be applied." << std::endl;

            if (dry_run)
            {
                std::cout << " DRY RUN: would evict cache for project: " << evict_project_path << std::endl;
            }
            else
            {
                std::cout << " Evicting cache for project: " << evict_project_path << " ..." << std::endl;
                cm.evictProject(evict_project_path);
                std::cout << " Done." << std::endl;
            }
        }
        else if (mode_cleanup)
        {
            if (dry_run)
                std::cout << " DRY-RUN mode: no changes will be made. Pass --execute to proceed." << std::endl;
            else
                std::cout << " LIVE mode: changes will be applied." << std::endl;
            std::vector<CacheManager::RegistryEntry> toClean = filterByNode(entries);
            if (toClean.empty())
            {
                std::cout << " No matching entries in registry." << std::endl;
                return;
            }

            std::cout << " Max age: " << max_age << " days" << std::endl;
            std::cout << " Max size: " << max_size << " GB" << std::endl;
            if (node_filter != "")
                std::cout << " Node filter: " << node_filter << std::endl;

            if (dry_run)
            {
                std::cout << " DRY RUN: no files will be removed" << std::endl;

                time_t now = std::time(NULL);
                int wouldRemove = 0;
                long totalBytes = 0;

                for (size_t i = 0; i < toClean.size(); i++)
                {
                    double ageDays = std::difftime(now, toClean[i].lastAccess) / 86400.0;
                    bool remove = (max_age > 0 && ageDays > max_age);

                    std::string src = toClean[i].sourceJobDir.empty() ? toClean[i].key : toClean[i].sourceJobDir;
                    std::cout << "   " << src.substr(0, 40)
                              << "  (" << toClean[i].nodeName << ")"
                              << "  age=" << (int)ageDays << "d"
                              << "  size=" << (toClean[i].sizeBytes / (1024*1024)) << "MB"
                              << (remove ? "  [WOULD REMOVE]" : "")
                              << std::endl;

                    if (remove)
                        wouldRemove++;
                    totalBytes += toClean[i].sizeBytes;
                }

                RFLOAT totalGb = (RFLOAT)totalBytes / (1024.0 * 1024.0 * 1024.0);
                RFLOAT wouldFreeGb = max_size > 0 && totalGb > max_size ? totalGb - max_size : 0;

                std::cout << " Total: " << toClean.size() << " entries, "
                          << totalGb << " GB" << std::endl;
                std::cout << " Would remove " << wouldRemove << " entries by age";

                if (max_size > 0 && wouldFreeGb > 0)
                    std::cout << ", plus ~" << wouldFreeGb << " GB to stay under limit";
                std::cout << std::endl;
            }
            else
            {
                int removed = cm.cleanup(max_age, max_size, node_filter);
                std::cout << " Removed " << removed << " stale cache entries." << std::endl;
            }
        }
    }

private:
    FileName fn_registry;
    std::string evict_project_path;
    std::string node_filter;
    int max_age;
    RFLOAT max_size;
    bool fn_execute, dry_run, do_list, do_evict_current, do_cleanup;
};

int main(int argc, char *argv[])
{
    CacheCleanupApp app;
    try
    {
        app.read(argc, argv);
        app.run();
    }
    catch (RelionError XE)
    {
        std::cerr << XE;
        return RELION_EXIT_FAILURE;
    }
    return RELION_EXIT_SUCCESS;
}
