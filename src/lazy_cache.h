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

#ifndef LAZY_CACHE_H_
#define LAZY_CACHE_H_

#include <string>

// Fills the particle-stack cache (--cache_dir) in the background while the job runs, instead of
// copying everything before the first iteration (opt-in: RELION_CACHE_LAZY=1).
//
// The cache layout, the registry and the "is this stack cached" test (same size and modification
// time as the source) are those of CacheManager, so a later job, lazy or not, reuses what was
// copied. Until a stack has been copied, reading its cache path falls back to the original file
// (fImageHandler::openFile asks resolve()). A stack appears in the cache only when complete: it is
// copied to a temporary name and renamed, so a reader never sees a partial file.
//
// Several processes may run this against one cache directory (MPI ranks, other jobs): a stack is
// copied by whoever holds its flock() lock, the others skip it.
// Header kept free of heavy includes: image.h includes it.
namespace lazycache
{

/// RELION_CACHE_LAZY is set to a non-zero value
bool enabled();

/// Root of the cache directory tree (<cache_dir>/relion_cache/, with a trailing slash). Lets a
/// process that does not copy itself still find the original of a cache path.
void setRoot(const std::string &cache_root);

/// Queue one stack for copying. A stale cache file (other size or time than the source) is removed
/// so that it is read from the source until the new copy is in place.
void add(const std::string &cache_path, const std::string &source_path);

/// Start (or keep running) the background copier: one thread at first, more while the copy rate keeps
/// growing, up to max_threads. Does nothing when nothing was queued.
void start(int max_threads, int verb);

/// For a cache path that does not exist yet: the file to read instead, or "" if there is none
std::string resolve(const std::string &cache_path);

/// Stop copying (a stack being copied is dropped) and print the summary. Also runs at exit.
void stop();

}

#endif
