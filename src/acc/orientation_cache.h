#ifndef ACC_ORIENTATION_CACHE_H_
#define ACC_ORIENTATION_CACHE_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>
#include "src/acc/settings.h"
#include "src/macros.h"

// The oversampled orientations of the fine pass, and their Euler matrices, depend on the
// sampling only (not on the particle or the class), so they are computed once per iteration and
// shared by all particles, classes and threads. Entries are filled on first use by whichever
// thread gets there first; a reader that finds an entry not ready computes it itself.
struct OrientationCache
{
	size_t entries;        // nr_dir * nr_psi, indexed like iorientclasses % entries
	size_t nover;          // oversampled orientations per entry
	std::vector<RFLOAT> angles;   // 3 values (rot, tilt, psi) per orientation
	std::vector<XFLOAT> matrices; // 9 values per orientation (inverse Euler matrix, no body or optics factors)
	std::unique_ptr<std::atomic<uint8_t>[]> state; // 0 empty, 1 being filled, 2 ready

	OrientationCache(size_t entries_, size_t nover_):
		entries(entries_), nover(nover_),
		angles(3 * entries_ * nover_), matrices(9 * entries_ * nover_),
		state(new std::atomic<uint8_t>[entries_])
	{
		for (size_t i = 0; i < entries; i++)
			state[i].store(0, std::memory_order_relaxed);
	}

	bool ready(size_t e) const { return state[e].load(std::memory_order_acquire) == 2; }
	bool claim(size_t e)
	{
		uint8_t expected = 0;
		return state[e].compare_exchange_strong(expected, 1, std::memory_order_acq_rel);
	}
	void publish(size_t e) { state[e].store(2, std::memory_order_release); }
};

#endif
