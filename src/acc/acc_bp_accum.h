#ifndef ACC_BP_ACCUM_H_
#define ACC_BP_ACCUM_H_

#include "src/acc/settings.h"

// Per-orientation accumulation for the back-projection of 3D references (see
// cuda_kernel_backproject3D): the weighted image sums of every particle are added into one
// image-plane buffer per (class, orientation) slot, and the slots are spread into the volumes
// once at the end of the iteration (cuda_kernel_backproject3D_flush). The volume position of a
// pixel depends on the orientation only, so the sums are the same up to the order of the
// additions, with far fewer (and coalesced) atomic additions.
struct BPAccumView
{
	int *counter;        // buffers handed out so far
	int *slot_idx;       // per slot: buffer index, -1 free, -2 being claimed, -3 no room
	unsigned *idx_slot;  // per buffer: its slot
	XFLOAT *euler;       // per buffer: 9 values
	XFLOAT *acc;         // per buffer: real, imaginary and weight sums (pixels each)
	int cap;             // buffers available (0: accumulation off)
	unsigned pixels;     // image pixels per buffer
};

struct BPModelPtrs
{
	XFLOAT *real;
	XFLOAT *imag;
	XFLOAT *weight;
};

#endif
