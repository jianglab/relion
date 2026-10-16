#ifndef ACC_PAIR_LIST_H_
#define ACC_PAIR_LIST_H_

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include "src/gpu_switch.h"

// Host-side construction of the fine-pass (orientation, translation) pair list and the
// jobs that group its translations. Plain pointers, so it can be tested on its own.
//
// Row i is one fine orientation. Its translations are j = t * nov + o, with t the
// coarse translation (0 .. nt-1) and o its oversampled translation (0 .. nov-1). A pair
// is kept when sig[iorientclasses[i] * nr_trans + t] is set. Kept pairs of a row that
// are neighbours form a job, split every `chunk` pairs; a job never spans two rows.
namespace pairlist
{
// RELION_GPU_PAIR_LIST: on by default; =0 or =reference selects the plain loop
inline bool fastEnabled()
{
	static const bool on = []() {
		const char *e = getenv("RELION_GPU_PAIR_LIST");
		return !gpuswitch::isOffValue(e) && !(e != NULL && strcmp(e, "reference") == 0);
	}();
	return on;
}

struct Input
{
	size_t orientation_num, nt, nov, nor, nr_trans, rot_mod, chunk;
	const size_t *iorientclasses, *iover_rots;
	const bool *sig;
};
struct Output
{
	size_t *rot_id, *rot_idx, *trans_idx, *ihidden_overs; // written from index 0
	size_t *job_origin, *job_extent;                      // room for orientation_num * nt * nov + 1 jobs
	size_t pairs, jobs;
};

// The straightforward loop over all pairs: the definition of the result
inline void buildReference(const Input &in, Output &out)
{
	size_t w = 0, k = 0;
	out.job_origin[0] = 0;
	for (size_t i = 0; i < in.orientation_num; i++)
	{
		out.job_extent[k] = 0;
		size_t tk = 0;
		for (size_t j = 0; j < in.nt * in.nov; j++)
		{
			const size_t t = j / in.nov, o = j % in.nov;
			const size_t ihidden = in.iorientclasses[i] * in.nr_trans + t;
			if (in.sig[ihidden])
			{
				out.rot_id[w] = in.iorientclasses[i] % in.rot_mod;
				out.rot_idx[w] = i;
				out.trans_idx[w] = j;
				out.ihidden_overs[w] = (ihidden * in.nor + in.iover_rots[i]) * in.nov + o;
				if (tk >= in.chunk)
				{
					tk = 0;
					k++;
					out.job_origin[k] = w;
					out.job_extent[k] = 0;
				}
				tk++;
				out.job_extent[k]++;
				w++;
			}
			else if (tk != 0)
			{
				tk = 0;
				k++;
				out.job_origin[k] = w;
				out.job_extent[k] = 0;
			}
		}
		if (tk > 0)
		{
			k++;
			out.job_origin[k] = w;
			out.job_extent[k] = 0;
		}
	}
	if (out.job_extent[k] != 0)
		k += 1;
	out.pairs = w;
	out.jobs = k;
}

// Same result as buildReference, but with everything that depends on the row hoisted,
// one test per coarse translation instead of one per pair, and no division per pair
inline void build(const Input &in, Output &out)
{
	size_t w = 0, k = 0;
	out.job_origin[0] = 0;
	const size_t nov = in.nov, chunk = in.chunk;
	size_t *const rot_id = out.rot_id, *const rot_idx = out.rot_idx, *const trans_idx = out.trans_idx,
			*const ihidden_overs = out.ihidden_overs, *const job_origin = out.job_origin, *const job_extent = out.job_extent;
	for (size_t i = 0; i < in.orientation_num; i++)
	{
		job_extent[k] = 0;
		size_t tk = 0;
		const size_t oc = in.iorientclasses[i];
		const bool *const row = in.sig + oc * in.nr_trans;
		const size_t my_rot_id = oc % in.rot_mod;
		const size_t over_rot = in.iover_rots[i];
		for (size_t t = 0; t < in.nt; t++)
		{
			if (!row[t])
			{
				if (tk != 0)
				{
					tk = 0;
					k++;
					job_origin[k] = w;
					job_extent[k] = 0;
				}
				continue;
			}
			const size_t hidden_base = ((oc * in.nr_trans + t) * in.nor + over_rot) * nov;
			size_t o = 0;
			while (o < nov)
			{
				if (tk >= chunk)
				{
					tk = 0;
					k++;
					job_origin[k] = w;
					job_extent[k] = 0;
				}
				size_t n = nov - o;
				if (n > chunk - tk)
					n = chunk - tk;
				for (size_t q = 0; q < n; q++)
				{
					rot_id[w + q] = my_rot_id;
					rot_idx[w + q] = i;
					trans_idx[w + q] = t * nov + o + q;
					ihidden_overs[w + q] = hidden_base + o + q;
				}
				tk += n;
				job_extent[k] += n;
				w += n;
				o += n;
			}
		}
		if (tk > 0)
		{
			k++;
			job_origin[k] = w;
			job_extent[k] = 0;
		}
	}
	if (job_extent[k] != 0)
		k += 1;
	out.pairs = w;
	out.jobs = k;
}
}

#endif
