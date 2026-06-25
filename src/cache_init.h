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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/
#ifndef CACHE_INIT_H_
#define CACHE_INIT_H_

#include "src/metadata_table.h"
#include "src/exp_model.h"

class CacheInitializer
{
public:
    static bool initializeCache(const FileName &fn_cache,
        int cache_copy_threads, MetaDataTable &MDimg, int verb,
        ObservationModel *obsModel = NULL, bool do_register = true,
        const std::string &node_name = "");

    static bool initializeCacheMpi(const FileName &fn_cache,
        int cache_copy_threads, MetaDataTable &MDimg, int verb,
        ObservationModel *obsModel, bool isLeader, intptr_t mpi_comm_handle);

    static bool initializeCacheMpi(const FileName &fn_cache,
        int cache_copy_threads, Experiment &exp, int verb,
        ObservationModel *obsModel, bool isLeader, intptr_t mpi_comm_handle);
};

#endif /* CACHE_INIT_H_ */
