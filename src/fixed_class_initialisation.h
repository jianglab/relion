#ifndef RELION_FIXED_CLASS_INITIALISATION_H
#define RELION_FIXED_CLASS_INITIALISATION_H

#include <cstdint>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

// Stable across processes, platforms and STAR row order (unlike std::hash).
inline std::uint64_t fixedClassSeedKey(const std::string &name, int seed, bool orientation = false)
{
	std::uint64_t value = 14695981039346656037ULL;
	for (unsigned char c : name) { value ^= c; value *= 1099511628211ULL; }
	value ^= (std::uint64_t)(unsigned int)seed + (orientation ? 0x9e3779b97f4a7c15ULL : 0);
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
	value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
	return value ^ (value >> 31);
}

inline double fixedClassInitialPsi(const std::string &name, int seed)
{
	return (fixedClassSeedKey(name, seed, true) >> 11) * (360. / 9007199254740992.);
}

// Streaming reservoir by deterministic random priority: O(100 * C) storage.
// Particle indices only identify selected rows; names break hash collisions.
class FixedClassSeedSelection
{
	typedef std::tuple<std::uint64_t, std::string, long int> Candidate;
	std::vector<std::priority_queue<Candidate> > candidates;
	int seed;
public:
	explicit FixedClassSeedSelection(int nr_classes, int seed) : seed(seed)
	{
		if (nr_classes < 2 || seed < 1) throw std::invalid_argument("Invalid fixed-class seed parameters");
		candidates.resize(nr_classes);
	}

	void add(const std::string &name, long int index, int one_based_class)
	{
		if (one_based_class < 1 || (size_t)one_based_class > candidates.size() || index < 0)
			throw std::invalid_argument("Invalid particle in fixed-class seed selection");
		auto &heap = candidates[one_based_class - 1];
		Candidate item(fixedClassSeedKey(name, seed), name, index);
		if (heap.size() < 100) heap.push(item);
		else if (item < heap.top()) { heap.pop(); heap.push(item); }
	}

	std::vector<long int> takeIndices()
	{
		std::vector<long int> indices;
		for (auto &heap : candidates)
			while (!heap.empty()) { indices.push_back(std::get<2>(heap.top())); heap.pop(); }
		return indices;
	}
};

#endif
