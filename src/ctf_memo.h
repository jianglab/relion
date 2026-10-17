#ifndef CTF_MEMO_H_
#define CTF_MEMO_H_

#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include "src/multidim_array.h"

/*
 * RELION_PREP_AHEAD: exact memo of the CTF images the expectation step builds on the host.
 * A CTF image is a pure function of the six per-particle CTF values, the optics group, the pixel
 * size, the image size and a few flags, and many particles share them (one set per micrograph), so
 * the image is computed once per distinct key and copied afterwards. The copy equals what
 * CTF::getFftwImage would have written, bit for bit. The memory is capped and an insert that would
 * exceed the cap is skipped (the caller then just computes).
 */
class CtfMemo
{
public:
	typedef std::shared_ptr<const MultidimArray<RFLOAT> > Ptr;

	struct Key
	{
		double v[7];            // defU, defV, angle, bfactor, kfactor, phase shift, pixel size
		int optics_group, xdim, ydim, full_size;
		unsigned flags;
		std::string str() const { return std::string((const char *)this, sizeof(Key)); }
	};

	explicit CtfMemo(size_t max_bytes, bool check = false) : max_bytes_(max_bytes), check_(check) {}

	// RELION_PREP_AHEAD=check: a hit is also computed again and compared bit for bit
	bool checking() const { return check_; }
	void recordCheck(bool same)
	{
		std::lock_guard<std::mutex> lock(mutex_);
		(same ? checked_same_ : checked_diff_)++;
	}
	size_t checkedSame() const { std::lock_guard<std::mutex> lock(mutex_); return checked_same_; }
	size_t checkedDifferent() const { std::lock_guard<std::mutex> lock(mutex_); return checked_diff_; }

	static Key makeKey(double defU, double defV, double angle, double bfactor, double kfactor, double phase_shift,
	                   double pixel_size, int optics_group, int xdim, int ydim, int full_size, unsigned flags)
	{
		Key k;
		memset(&k, 0, sizeof(k));
		k.v[0] = defU; k.v[1] = defV; k.v[2] = angle; k.v[3] = bfactor; k.v[4] = kfactor;
		k.v[5] = phase_shift; k.v[6] = pixel_size;
		k.optics_group = optics_group; k.xdim = xdim; k.ydim = ydim; k.full_size = full_size; k.flags = flags;
		return k;
	}

	Ptr find(const Key &k)
	{
		std::lock_guard<std::mutex> lock(mutex_);
		auto it = map_.find(k.str());
		if (it == map_.end()) { misses_++; return Ptr(); }
		hits_++;
		return it->second;
	}

	// Returns false when the cap would be exceeded (the entry is not stored)
	bool insert(const Key &k, const MultidimArray<RFLOAT> &img)
	{
		const size_t bytes = (size_t)NZYXSIZE(img) * sizeof(RFLOAT);
		std::lock_guard<std::mutex> lock(mutex_);
		if (map_.count(k.str())) return true;
		if (bytes_ + bytes > max_bytes_) { skipped_++; return false; }
		map_[k.str()] = Ptr(new MultidimArray<RFLOAT>(img));
		bytes_ += bytes;
		return true;
	}

	// Entries of an earlier image size are of no use: call at the start of each iteration
	void beginIteration(const std::string &signature)
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (signature != signature_)
		{
			map_.clear();
			bytes_ = 0;
			signature_ = signature;
		}
	}

	size_t entries() const { std::lock_guard<std::mutex> lock(mutex_); return map_.size(); }
	size_t bytes() const { std::lock_guard<std::mutex> lock(mutex_); return bytes_; }
	size_t hits() const { std::lock_guard<std::mutex> lock(mutex_); return hits_; }
	size_t misses() const { std::lock_guard<std::mutex> lock(mutex_); return misses_; }
	size_t skipped() const { std::lock_guard<std::mutex> lock(mutex_); return skipped_; }

private:
	mutable std::mutex mutex_;
	std::unordered_map<std::string, Ptr> map_;
	std::string signature_;
	size_t max_bytes_;
	bool check_;
	size_t bytes_ = 0, hits_ = 0, misses_ = 0, skipped_ = 0, checked_same_ = 0, checked_diff_ = 0;
};

#endif /* CTF_MEMO_H_ */
