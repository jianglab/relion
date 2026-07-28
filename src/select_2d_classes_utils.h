/***************************************************************************
 *
 * Author: Jiang Lab
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 ***************************************************************************/

#ifndef SELECT_2D_CLASSES_UTILS_H_
#define SELECT_2D_CLASSES_UTILS_H_

#include <string>
#include <vector>

#include "src/macros.h"

namespace Select2DClasses
{

const int DEFAULT_NR_TYPES = 1;
const int MAX_NR_TYPES = 20;

// Return junk (zero) for assignments that are not valid for nr_types.
int validTypeOrJunk(int type_id, int nr_types);

// Build a deterministic nearest-neighbour correlation chain, starting at the
// most populated class. Feature vectors are expected to be normalised.
std::vector<int> similarityOrder(
		const std::vector<std::vector<RFLOAT> > &features,
		const std::vector<RFLOAT> &populations);

// Junk is index zero and wins exact ties. As a consequence, exact ties among
// non-junk types are resolved in favour of the lowest-numbered type.
int winningType(const std::vector<long int> &counts);

struct FilamentVoteResult
{
	std::vector<int> particle_types;
	std::vector<long int> filament_counts;
};

// Vote independently for each (micrograph, tube ID) pair and return the
// winning type for every input particle, plus per-type filament totals.
FilamentVoteResult voteFilaments(
		const std::vector<std::string> &micrographs,
		const std::vector<int> &tube_ids,
		const std::vector<int> &assigned_types,
		int nr_types);

} // namespace Select2DClasses

#endif // SELECT_2D_CLASSES_UTILS_H_
