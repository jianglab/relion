#ifndef ACC_GROUP_SIZE_TUNER_H_
#define ACC_GROUP_SIZE_TUNER_H_

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <vector>

// RELION_GPU_PARTICLE_GROUP=auto: how many particles a thread takes through the E-step together.
// The best size depends on the GPU (SMs, memory, speed against the host) and on the data (orientations
// per particle), so it is measured on real work, with no extra work done: the sizes 1, 2, 4, 8, 16 are
// used by all threads in turn, one block of a few hundred particles each, in several rounds (the first
// round only warms up), and the speed of a block is its particles per second of wall time (what the whole
// GPU delivers, not what one thread sees). The smallest size within 3% of the best median speed wins: a
// bigger group is worth its memory and its longer latency only when it is clearly faster. After
// `recheck_particles` more particles the winner and its two neighbours are measured again and the choice
// moves only for a clear (4%) gain. A pause (the end of an iteration) is left out of the time.
// The choices and the speeds are printed. RELION_GPU_PARTICLE_GROUP=n fixes the size instead.
class GroupSizeTuner
{
public:
	GroupSizeTuner(bool quiet = false, long block_particles = 512, int rounds = 4, long recheck_particles = 150000):
		cands_{1, 2, 4, 8, 16}, rate_(5), quiet_(quiet), block_(block_particles), rounds_(std::max(2, rounds)),
		recheck_(recheck_particles)
	{
		for (size_t i = 0; i < cands_.size(); i++)
			pool_.push_back((int) i);
		startPhase(true);
	}

	static GroupSizeTuner &instance()
	{
		static GroupSizeTuner t;
		return t;
	}

	// The size to use for the next group
	int next()
	{
		std::lock_guard<std::mutex> lock(mutex_);
		return cands_[phaseIsRun_ ? chosen_ : pool_[step_ % pool_.size()]];
	}

	// A full group of `size` particles is done; `now` is the time in seconds (any origin)
	void report(int size, int particles, double now)
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (phaseIsRun_)
		{
			if ((since_ += particles) >= recheck_)
				startPhase(false);
			return;
		}
		if (size != cands_[pool_[step_ % pool_.size()]])
			return; // started under the previous block's size
		if (!started_)
		{
			started_ = true;
			t_first_ = now;
			count_ = 0;
		}
		else
		{
			if (now - t_last_ > 0.5)
				t_first_ += now - t_last_; // a pause (end of an iteration) is not the speed of this size
			count_ += particles; // the first group began before the clock started: not counted
		}
		t_last_ = now;
		if (count_ < block_ || t_last_ <= t_first_)
			return;
		const int c = pool_[step_ % pool_.size()];
		if ((int) (step_ / pool_.size()) >= 1)
			rate_[c].push_back(count_ / (t_last_ - t_first_));
		started_ = false;
		if (++step_ >= (long) (rounds_ * pool_.size()))
			decide();
	}

	// 0 while still measuring for the first time
	int chosen() const { return chosen_ < 0 ? 0 : cands_[chosen_]; }

private:
	void startPhase(bool first)
	{
		phaseIsRun_ = false;
		started_ = false;
		step_ = 0;
		first_ = first;
		if (!first)
		{
			pool_.clear();
			for (int c = chosen_ - 1; c <= chosen_ + 1; c++)
				if (c >= 0 && c < (int) cands_.size())
					pool_.push_back(c);
		}
		for (auto &r : rate_)
			r.clear();
	}

	static double median(std::vector<double> v)
	{
		if (v.empty())
			return 0.;
		std::sort(v.begin(), v.end());
		return v[v.size() / 2];
	}

	void decide()
	{
		double best = 0.;
		for (int c : pool_)
			best = std::max(best, median(rate_[c]));
		int pick = -1;
		for (int c : pool_)
			if (median(rate_[c]) >= 0.97 * best) { pick = c; break; }
		if (pick < 0)
			pick = pool_[0];
		const char *how = "measured";
		if (!first_)
		{
			how = "re-measured";
			if (pick == chosen_ || median(rate_[pick]) < 1.04 * median(rate_[chosen_]))
				pick = chosen_;
		}
		const bool changed = pick != chosen_;
		chosen_ = pick;
		if (!quiet_ && (first_ || changed))
		{
			fprintf(stderr, "GPU particle group (auto): %s, using %d particles at a time; particles per second:", how, cands_[chosen_]);
			for (int c : pool_)
				fprintf(stderr, " P=%d: %.0f", cands_[c], median(rate_[c]));
			fprintf(stderr, " (smallest size within 3%% of the fastest)\n");
		}
		phaseIsRun_ = true;
		since_ = 0;
	}

	std::mutex mutex_;
	std::vector<int> cands_;
	std::vector<int> pool_;
	std::vector<std::vector<double> > rate_;
	bool quiet_;
	long block_;
	int rounds_;
	long recheck_;
	int chosen_ = -1;
	bool phaseIsRun_ = false, first_ = true, started_ = false;
	long step_ = 0, count_ = 0, since_ = 0;
	double t_first_ = 0., t_last_ = 0.;
};

#endif
