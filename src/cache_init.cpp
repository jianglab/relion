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

#include "src/cache_init.h"
#include "src/cache_manager.h"
#include "src/mpi.h"
#include <unistd.h>
#include <sys/statvfs.h>
#include <iomanip>
#include <map>
#include <set>

bool CacheInitializer::initializeCache(const FileName &fn_cache,
                                       int cache_copy_threads,
                                       MetaDataTable &MDimg,
                                       int verb,
                                       ObservationModel *obsModel,
                                       bool do_register)
{
	if (fn_cache == "")
		return true;

	// Determine project root directory
	char cwdBuf[4096];
	std::string projectRoot;
	if (::getcwd(cwdBuf, sizeof(cwdBuf)) != NULL)
		projectRoot = cwdBuf;
	else
		projectRoot = ".";

	CacheManager cm;
	cm.setCacheDir(fn_cache);
	cm.setRegistryDir(projectRoot);

	if (verb > 0)
		std::cout << " Using cache directory: " << fn_cache << std::endl;

	// Validate cache directory
	{
		if (!exists(fn_cache))
		{
			if (verb > 0)
				std::cout << " Creating cache directory: " << fn_cache << std::endl;
			mktree(fn_cache, 0777);
		}
		if (!exists(fn_cache))
			REPORT_ERROR("Cache directory does not exist and could not be created: " + fn_cache);
		if (access(fn_cache.c_str(), W_OK) != 0)
			REPORT_ERROR("Cache directory is not writable: " + fn_cache);
	}

	// Group stacks by source job directory
	std::map<FileName, std::vector<FileName> > stacksBySource;
	for (long int p = 0; p < MDimg.numberOfObjects(); p++)
	{
		FileName fn_img_name, fn_stack;
		long int imgno;
		MDimg.getValue(EMDL_IMAGE_NAME, fn_img_name, p);
		fn_img_name.decompose(imgno, fn_stack);
		FileName srcDir = fn_stack.beforeLastOf("/");
		if (srcDir == fn_stack)
			srcDir = ".";
		stacksBySource[srcDir].push_back(fn_stack);
	}

	// De-duplicate stacks per source job
	std::map<FileName, std::vector<FileName> > uniqueStacksBySource;
	for (std::map<FileName, std::vector<FileName> >::const_iterator it = stacksBySource.begin();
	     it != stacksBySource.end(); ++it)
	{
		std::set<FileName> uniq(it->second.begin(), it->second.end());
		uniqueStacksBySource[it->first] = std::vector<FileName>(uniq.begin(), uniq.end());
	}

	// Pre-scan: sum total bytes needed for cache misses, then check free space
	{
		long int totalNeededBytes = 0;
		for (std::map<FileName, std::vector<FileName> >::const_iterator it = uniqueStacksBySource.begin();
		     it != uniqueStacksBySource.end(); ++it)
		{
			const FileName &srcDir = it->first;
			const std::vector<FileName> &relPaths = it->second;
			std::string cacheKey = cm.computeKey(projectRoot, srcDir);
			if (cm.isCached(cacheKey, relPaths))
				continue;
			for (size_t i = 0; i < relPaths.size(); i++)
			{
				std::string fullPath = (projectRoot != "" && relPaths[i][0] != '/') ? projectRoot + "/" + relPaths[i] : (std::string)relPaths[i];
				FileName src(fullPath);
				totalNeededBytes += (long int)src.getFileSize();
			}
		}
		if (totalNeededBytes > 0)
		{
			struct statvfs vfs;
			if (statvfs(fn_cache.c_str(), &vfs) == 0)
			{
				long int freeBytes = (long int)vfs.f_bsize * (long int)vfs.f_bavail;
				if (freeBytes < totalNeededBytes)
				{
					double needGb = totalNeededBytes / (1024.0 * 1024.0 * 1024.0);
					double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
					std::cerr << " ERROR: Not enough free space on cache filesystem." << std::endl;
					std::cerr << "   Cache directory:  " << fn_cache << std::endl;
					std::cerr << "   Space needed:     " << std::fixed << std::setprecision(1) << needGb << " GB" << std::endl;
					std::cerr << "   Space available:  " << std::fixed << std::setprecision(1) << freeGb << " GB" << std::endl;
					REPORT_ERROR("Insufficient free space on cache filesystem for " + fn_cache);
				}
				else if (verb > 0)
				{
					double needGb = totalNeededBytes / (1024.0 * 1024.0 * 1024.0);
					double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
					std::cout << " Cache space check: need " << std::fixed << std::setprecision(1)
					          << needGb << " GB, " << freeGb << " GB available" << std::endl;
				}
			}
		}
	}

	// Process each source job: ensure cached, then register
	std::map<FileName, std::string> keyForSource;
	for (std::map<FileName, std::vector<FileName> >::const_iterator it = uniqueStacksBySource.begin();
	     it != uniqueStacksBySource.end(); ++it)
	{
		const FileName &srcDir = it->first;
		const std::vector<FileName> &relPaths = it->second;

		std::string cacheKey = cm.computeKey(projectRoot, srcDir);
		keyForSource[srcDir] = cacheKey;

		if (verb > 0)
			std::cout << " Source job: " << srcDir << "  key=" << cacheKey << std::endl;

		if (cm.isCached(cacheKey, relPaths))
		{
			if (verb > 0)
				std::cout << "  Reusing cache " << cacheKey
				          << " (" << cm.getCacheDirForKey(cacheKey) << ")" << std::endl;
			int nrOpticsGroups = (obsModel != NULL) ? obsModel->numberOfOpticsGroups() : 0;
			if (do_register)
				cm.touchRegistry(cacheKey, srcDir, (int)MDimg.numberOfObjects(), nrOpticsGroups, projectRoot);
			continue;
		}

		// Cache miss: acquire lock and double-check
		cm.acquireLock(cacheKey);
		if (cm.isCached(cacheKey, relPaths))
		{
			if (verb > 0)
				std::cout << "  Reusing cache " << cacheKey
				          << " (" << cm.getCacheDirForKey(cacheKey)
				          << ") — cached by another process" << std::endl;
			cm.releaseLock(cacheKey);
			int nrOpticsGroups = (obsModel != NULL) ? obsModel->numberOfOpticsGroups() : 0;
			if (do_register)
				cm.touchRegistry(cacheKey, srcDir, (int)MDimg.numberOfObjects(), nrOpticsGroups, projectRoot);
			continue;
		}

		if (verb > 0)
		{
			long int totalBytes = 0;
			for (size_t i = 0; i < relPaths.size(); i++)
			{
				std::string fullPath = (projectRoot != "" && relPaths[i][0] != '/') ? projectRoot + "/" + relPaths[i] : (std::string)relPaths[i];
				FileName src(fullPath);
				totalBytes += (long int)src.getFileSize();
			}
			double gb = totalBytes / (1024.0 * 1024.0 * 1024.0);
			std::cout << "  Cache MISS for " << srcDir << " — copying " << relPaths.size()
			          << " image stacks (" << std::fixed << std::setprecision(1) << gb << " GB) to cache "
			          << cacheKey << std::endl;
		}

		bool popOK = cm.populateCache(cacheKey, relPaths, projectRoot, verb, cache_copy_threads);
		if (!popOK)
			REPORT_ERROR("Cache population failed for " + srcDir + " — copy command returned an error.");
		// Post-populate verification: all cached files must exist with correct sizes
		if (!cm.isCached(cacheKey, relPaths))
		{
			std::cerr << " Cache population reported success but verification failed for key="
			          << cacheKey << " — some cached files may be missing or corrupt." << std::endl;
			std::cerr << " Check that the cache directory has sufficient space and is not on a full or failing filesystem." << std::endl;
			REPORT_ERROR("Cache population verification failed for " + srcDir);
		}
		int nrOpticsGroups = (obsModel != NULL) ? obsModel->numberOfOpticsGroups() : 0;
		if (do_register)
			cm.touchRegistry(cacheKey, srcDir, (int)MDimg.numberOfObjects(), nrOpticsGroups, projectRoot);
		cm.evictIfFull();
		cm.releaseLock(cacheKey);
	}

	// Rewrite all particle names to point to cache
	for (long int p = 0; p < MDimg.numberOfObjects(); p++)
	{
		FileName fn_img_name, fn_stack, fn_cached;
		long int imgno;
		MDimg.getValue(EMDL_IMAGE_NAME, fn_img_name, p);
		fn_img_name.decompose(imgno, fn_stack);
		FileName srcDir = fn_stack.beforeLastOf("/");
		if (srcDir == fn_stack)
			srcDir = ".";
		std::string cacheKey = keyForSource[srcDir];
		fn_cached = cm.getCachePathForKey(cacheKey, fn_stack);
		FileName fn_rewritten;
		fn_rewritten.compose(imgno, fn_cached);
		MDimg.setValue(EMDL_IMAGE_NAME, fn_rewritten, p);
	}

    if (verb > 0)
        std::cout << " Rewrote " << MDimg.numberOfObjects()
                  << " particle image names to point to cache." << std::endl;

    return true;
}

bool CacheInitializer::initializeCacheMpi(const FileName &fn_cache,
    int cache_copy_threads, MetaDataTable &MDimg, int verb,
    ObservationModel *obsModel, bool isLeader, intptr_t mpi_comm_handle)
{
    if (fn_cache == "")
        return true;

    MPI_Comm comm;
    memcpy(&comm, &mpi_comm_handle, sizeof(comm));

    if (isLeader)
        initializeCache(fn_cache, cache_copy_threads, MDimg, verb, obsModel, true);

    MPI_Barrier(comm);

    if (!isLeader)
        initializeCache(fn_cache, cache_copy_threads, MDimg, verb, obsModel, false);

    return true;
}
