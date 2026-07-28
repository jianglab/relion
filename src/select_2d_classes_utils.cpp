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

#include "src/select_2d_classes_utils.h"

#include <limits>
#include <map>
#include <utility>

namespace Select2DClasses
{

int validTypeOrJunk(int type_id, int nr_types)
{
	return type_id >= 1 && type_id <= nr_types ? type_id : 0;
}

std::vector<int> similarityOrder(
		const std::vector<std::vector<RFLOAT> > &features,
		const std::vector<RFLOAT> &populations)
{
	std::vector<int> order;
	if (features.empty())
		return order;
	if (features.size() != populations.size())
		REPORT_ERROR("Select 2D classes: feature and population counts differ.");

	const size_t feature_size = features[0].size();
	for (size_t index = 1; index < features.size(); index++)
		if (features[index].size() != feature_size)
			REPORT_ERROR("Select 2D classes: class feature sizes differ.");

	int first = 0;
	for (size_t index = 1; index < populations.size(); index++)
		if (populations[index] > populations[first])
			first = static_cast<int>(index);

	std::vector<bool> used(features.size(), false);
	order.reserve(features.size());
	order.push_back(first);
	used[first] = true;

	while (order.size() < features.size())
	{
		const int previous = order.back();
		int best = -1;
		RFLOAT best_correlation = -std::numeric_limits<RFLOAT>::max();
		for (size_t candidate = 0; candidate < features.size(); candidate++)
		{
			if (used[candidate])
				continue;

			RFLOAT correlation = 0.;
			for (size_t element = 0; element < feature_size; element++)
				correlation += features[previous][element] * features[candidate][element];

			if (correlation > best_correlation)
			{
				best_correlation = correlation;
				best = static_cast<int>(candidate);
			}
		}
		order.push_back(best);
		used[best] = true;
	}

	return order;
}

int winningType(const std::vector<long int> &counts)
{
	if (counts.empty())
		REPORT_ERROR("Select 2D classes: cannot vote without type counts.");

	int winner = 0;
	for (size_t type_id = 1; type_id < counts.size(); type_id++)
		if (counts[type_id] > counts[winner])
			winner = static_cast<int>(type_id);
	return winner;
}

FilamentVoteResult voteFilaments(
		const std::vector<std::string> &micrographs,
		const std::vector<int> &tube_ids,
		const std::vector<int> &assigned_types,
		int nr_types)
{
	if (nr_types < 1)
		REPORT_ERROR("Select 2D classes: the number of types must be positive.");
	if (micrographs.size() != tube_ids.size() ||
		micrographs.size() != assigned_types.size())
		REPORT_ERROR("Select 2D classes: particle voting inputs have different sizes.");

	typedef std::pair<std::string, int> FilamentKey;
	std::map<FilamentKey, std::vector<long int> > counts;
	for (size_t particle = 0; particle < micrographs.size(); particle++)
	{
		const FilamentKey key(micrographs[particle], tube_ids[particle]);
		if (counts.find(key) == counts.end())
			counts[key] = std::vector<long int>(nr_types + 1, 0);
		const int type_id = validTypeOrJunk(assigned_types[particle], nr_types);
		counts[key][type_id]++;
	}

	std::map<FilamentKey, int> winners;
	FilamentVoteResult result;
	result.filament_counts.assign(nr_types + 1, 0);
	for (std::map<FilamentKey, std::vector<long int> >::const_iterator it = counts.begin();
			it != counts.end(); ++it)
	{
		const int winner = winningType(it->second);
		winners[it->first] = winner;
		result.filament_counts[winner]++;
	}

	result.particle_types.reserve(micrographs.size());
	for (size_t particle = 0; particle < micrographs.size(); particle++)
		result.particle_types.push_back(
				winners[FilamentKey(micrographs[particle], tube_ids[particle])]);

	return result;
}

} // namespace Select2DClasses
