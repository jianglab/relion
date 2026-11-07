#ifndef CTF_ORDERED_SUM_H
#define CTF_ORDERED_SUM_H

#include <algorithm>

// Sums per-particle images pixel by pixel with the image rows shared among the threads, each row adding up
// the particles in their order: the sums are the same whatever the number of threads (and the same as the
// per-thread sums gave with one thread). prepare(pp, slot) first makes per-particle data (such as a CTF image)
// for a chunk of particles, in parallel, into slot "slot" of a buffer of orderedsum::chunk entries;
// addRow(pp, slot, y) then adds row y of particle pp.
namespace orderedsum
{
	const int chunk = 64;

	template <class Prepare, class AddRow>
	void run(int particles, int rows, int threads, Prepare prepare, AddRow addRow)
	{
		for (int p0 = 0; p0 < particles; p0 += chunk)
		{
			const int p1 = std::min(particles, p0 + chunk);

			#pragma omp parallel for num_threads(threads)
			for (int pp = p0; pp < p1; pp++)
			{
				prepare(pp, pp - p0);
			}

			#pragma omp parallel for num_threads(threads)
			for (int y = 0; y < rows; y++)
			for (int pp = p0; pp < p1; pp++)
			{
				addRow(pp, pp - p0, y);
			}
		}
	}
}

#endif
