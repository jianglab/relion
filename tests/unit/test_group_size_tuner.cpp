/*
 * tests/unit/test_group_size_tuner.cpp
 *
 * The run-time choice of the GPU particle-group size (RELION_GPU_PARTICLE_GROUP=auto) from the measured
 * speed of blocks of real work: the smallest size within 3% of the fastest wins, a clearly faster
 * neighbour replaces it at the re-check, noise and pauses do not move it.
 */

#include <catch2/catch.hpp>

#include "src/acc/group_size_tuner.h"

#include <functional>
#include <random>

namespace
{
// Plays the E-step: `threads` threads take groups of the asked size; a group of p particles takes
// p * ms(p) thread-milliseconds, so the wall clock advances by that over the number of threads.
struct Sim
{
	GroupSizeTuner t;
	double now = 0.;
	std::mt19937 rng;
	double noise;
	Sim(double noise_, unsigned seed, long recheck = 150000): t(true, 512, 4, recheck), rng(seed), noise(noise_) {}

	int run(std::function<double(int)> ms_per_particle, long particles, int threads = 8, int pause_every = 0)
	{
		std::uniform_real_distribution<double> u(1. - noise, 1. + noise);
		long done = 0, groups = 0;
		while (done < particles)
		{
			const int p = t.next();
			now += p * ms_per_particle(p) * u(rng) / 1000. / threads;
			t.report(p, p, now);
			done += p;
			if (pause_every && ++groups % pause_every == 0)
				now += 2.; // end of an iteration
		}
		return t.chosen();
	}
};
}

TEST_CASE("GroupSizeTuner picks the fastest size, the smallest within 3%", "[group_size_tuner]")
{
	// time per particle falls with the size and flattens out: 8 and 16 are within 3% of each other, 4 is not
	auto flat = [](int p) { return p == 1 ? 10. : p == 2 ? 7. : p == 4 ? 5.5 : p == 8 ? 5.0 : 4.9; };
	Sim a(0.03, 1);
	REQUIRE(a.run(flat, 60000) == 8);

	// bigger groups slower than small ones (memory pressure, latency): not chosen
	auto hump = [](int p) { return p == 1 ? 6. : p == 2 ? 5. : p == 4 ? 7. : p == 8 ? 9. : 12.; };
	Sim h(0.03, 2);
	REQUIRE(h.run(hump, 60000) == 2);

	// no gain at all: stay with one particle at a time
	Sim n(0.03, 3);
	REQUIRE(n.run([](int) { return 5.; }, 60000) == 1);
}

TEST_CASE("GroupSizeTuner is not fooled by pauses between iterations", "[group_size_tuner]")
{
	auto hump = [](int p) { return p == 1 ? 6. : p == 2 ? 5. : p == 4 ? 7. : p == 8 ? 9. : 12.; };
	Sim s(0.03, 4);
	REQUIRE(s.run(hump, 80000, 8, 300) == 2);
}

TEST_CASE("GroupSizeTuner follows a change of the best size at the re-check", "[group_size_tuner]")
{
	Sim s(0.02, 5, 20000);
	auto early = [](int p) { return p == 1 ? 10. : p == 2 ? 6. : p == 4 ? 7. : p == 8 ? 9. : 12.; };
	REQUIRE(s.run(early, 30000) == 2);
	// the work changes (for example, fewer orientations): now 4 is clearly better than 2
	auto late = [](int p) { return p == 1 ? 10. : p == 2 ? 6. : p == 4 ? 4.5 : p == 8 ? 9. : 12.; };
	REQUIRE(s.run(late, 60000) == 4);
}

TEST_CASE("GroupSizeTuner only changes its choice for a clear gain", "[group_size_tuner]")
{
	Sim s(0.01, 6, 20000);
	auto a = [](int p) { return p == 1 ? 10. : p == 2 ? 6. : p == 4 ? 7. : p == 8 ? 9. : 12.; };
	REQUIRE(s.run(a, 30000) == 2);
	// 4 is 2% better than 2: not worth switching
	auto b = [](int p) { return p == 1 ? 10. : p == 2 ? 6. : p == 4 ? 5.88 : p == 8 ? 9. : 12.; };
	REQUIRE(s.run(b, 80000) == 2);
}
