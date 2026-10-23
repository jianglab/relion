#include <algorithm>
#include <atomic>
#include <vector>
#include <map>
#include <array>
#include <mutex>
#include <string>
#include "src/gpu_switch.h"
#include "src/acc/pair_list.h"
/*
#undef ALTCPU
#ifdef _CUDA_ENABLED
#include <cuda_runtime.h>
#include "src/acc/cuda/cuda_settings.h"
#include "src/acc/cuda/cuda_kernels/BP.cuh"
#elif _HIP_ENABLED
#include <hip/hip_runtime.h>
#include "src/acc/hip/hip_settings.h"
#include "src/acc/hip/hip_kernels/BP.h"
#endif
#include "src/macros.h"
#include "src/error.h"
*/
#ifdef _CUDA_ENABLED
	using deviceStream_t = cudaStream_t;
#elif _HIP_ENABLED
	using deviceStream_t = hipStream_t;
#elif _SYCL_ENABLED
	#include "src/acc/sycl/sycl_virtual_dev.h"
	using deviceStream_t = virtualSYCL*;
#else
	using deviceStream_t = float;
#endif

long int makeJobsForDiff2Fine(
		OptimisationParamters &op,  SamplingParameters &sp,
		long int orientation_num, long int translation_num,
		ProjectionParams &FineProjectionData,
		std::vector< long unsigned > &iover_transes,
		std::vector< long unsigned > &ihiddens,
		long int nr_over_orient, long int nr_over_trans,
		IndexedDataArray &FPW, // FPW=FinePassWeights
		IndexedDataArrayMask &dataMask,
		int chunk)
{
	long unsigned w_base = dataMask.firstPos, w(0), k(0);
	// be on the safe side with the jobArrays: make them as large as they could possibly be
	// (this will be reduced at exit of this function). The loop below always opens the
	// element after the last job, so one more than the number of jobs: with one
	// orientation and one translation (--skip_align on the GPU) it wrote past the end.
	dataMask.setNumberOfJobs(orientation_num*translation_num + 1);
	dataMask.setNumberOfWeights(orientation_num*translation_num);
	dataMask.jobOrigin.freeHostIfSet();
	dataMask.jobExtent.freeHostIfSet();
	dataMask.jobOrigin.hostAlloc();
	dataMask.jobExtent.hostAlloc();

	// RELION_GPU_PAIR_LIST=0 (or "reference") uses the plain loop below
	const bool plainLoop = !pairlist::fastEnabled();
	bool regular = !plainLoop && nr_over_trans > 0 && translation_num > 0 && (long unsigned) translation_num == iover_transes.size() &&
			iover_transes.size() == ihiddens.size() && translation_num % nr_over_trans == 0 && chunk >= 1;
	for (long unsigned j = 0; regular && j < (long unsigned) translation_num; j++)
		regular = ihiddens[j] == j / nr_over_trans && iover_transes[j] == j % nr_over_trans;
	if (regular)
	{
		pairlist::Input in;
		in.orientation_num = orientation_num;
		in.nt = translation_num / nr_over_trans;
		in.nov = nr_over_trans;
		in.nor = nr_over_orient;
		in.nr_trans = sp.nr_trans;
		in.rot_mod = sp.nr_dir * sp.nr_psi;
		in.chunk = chunk;
		in.iorientclasses = &FineProjectionData.iorientclasses[0];
		in.iover_rots = &FineProjectionData.iover_rots[0];
		in.sig = op.Mcoarse_significant.data;
		pairlist::Output out;
		out.rot_id = &FPW.rot_id[w_base];
		out.rot_idx = &FPW.rot_idx[w_base];
		out.trans_idx = &FPW.trans_idx[w_base];
		out.ihidden_overs = &FPW.ihidden_overs[w_base];
		out.job_origin = &dataMask.jobOrigin[0];
		out.job_extent = &dataMask.jobExtent[0];
		pairlist::build(in, out);
		dataMask.setNumberOfJobs(out.jobs);
		dataMask.setNumberOfWeights(out.pairs);
		return (long int) out.pairs;
	}

	dataMask.jobOrigin[k]=0;
	for (long unsigned i = 0; i < orientation_num; i++)
	{
		dataMask.jobExtent[k]=0;
		long int tk=0;
		long int iover_rot = FineProjectionData.iover_rots[i];
		for (long unsigned j = 0; j < translation_num; j++)
		{
			long int iover_trans = iover_transes[j];
			long int ihidden = FineProjectionData.iorientclasses[i] * sp.nr_trans + ihiddens[j];

			if(DIRECT_A1D_ELEM(op.Mcoarse_significant, ihidden)==1)
			{
				FPW.rot_id[w_base+w] = FineProjectionData.iorientclasses[i] % (sp.nr_dir*sp.nr_psi); 	// where to look for priors etc
				FPW.rot_idx[w_base+w] = i;					// which rot for this significant task
				FPW.trans_idx[w_base+w] = j;					// which trans       - || -
				FPW.ihidden_overs[w_base+w]= (ihidden * nr_over_orient + iover_rot) * nr_over_trans + iover_trans;

				if(tk>=chunk)
				{
					tk=0;             // reset counter
					k++;              // use new element
					dataMask.jobOrigin[k]=w;
					dataMask.jobExtent[k]=0;   // prepare next element for ++ incrementing
				}
				tk++;                 		   // increment limit-checker
				dataMask.jobExtent[k]++;       // increment number of transes this job
				w++;
			}
			else if(tk!=0) 		  // start a new one with the same rotidx - we expect transes to be sequential.
			{
				tk=0;             // reset counter
				k++;              // use new element
				dataMask.jobOrigin[k]=w;
				dataMask.jobExtent[k]=0;   // prepare next element for ++ incrementing
			}
		}
		if(tk>0) // use new element (if tk==0) then we are currently on an element with no signif, so we should continue using this element
		{
			k++;
			dataMask.jobOrigin[k]=w;
			dataMask.jobExtent[k]=0;
		}
	}
	if(dataMask.jobExtent[k]!=0) // if we started putting somehting in last element, then the count is one higher than the index
		k+=1;

	dataMask.setNumberOfJobs(k);
	dataMask.setNumberOfWeights(w);
//	if(dataMask.weightNum>0)
//	{
//		dataMask.jobOrigin.device_alloc();
//		dataMask.jobExtent.device_alloc();
//	}
	return(w);
}

long int  makeJobsForCollect(IndexedDataArray &FPW,
    IndexedDataArrayMask &dataMask, unsigned long NewJobNum) // FPW=FinePassWeights
{
	// reset the old (diff2Fine) job-definitions
//	dataMask.jobOrigin.free_host();
//    dataMask.jobOrigin.free_device();
//    dataMask.jobExtent.free_host();
//    dataMask.jobExtent.free_device();
    dataMask.setNumberOfJobs(NewJobNum);
//    dataMask.jobOrigin.hostAlloc();
//    dataMask.jobExtent.hostAlloc();

	long int jobid=0;
	dataMask.jobOrigin[jobid]=0;
	dataMask.jobExtent[jobid]=1;
    long int crot =FPW.rot_idx[jobid]; // set current rot
	for(long int n=1; n<FPW.rot_idx.getSize(); n++)
	{
        if(FPW.rot_idx[n]==crot)
		{
            dataMask.jobExtent[jobid]++;
		}
		else
		{
			jobid++;
            dataMask.jobExtent[jobid]=1;
			dataMask.jobOrigin[jobid]=n;
			crot=FPW.rot_idx[n];
		}
	}
	dataMask.setNumberOfJobs(jobid+1); // because max index is one less than size
//	dataMask.jobOrigin.putOnDevice();
//	dataMask.jobExtent.putOnDevice();

	return (jobid+1);
}

/*
 * Maps weights to a decoupled indexing of translations and orientations
 */
void mapWeights(
		unsigned long orientation_start,
		XFLOAT *mapped_weights,
		unsigned long orientation_num,
		unsigned long idxArr_start,
		unsigned long idxArr_end,
		unsigned long translation_num,
		XFLOAT *weights,
		long unsigned *rot_idx,
		long unsigned *trans_idx,
		unsigned long current_oversampling)
{

	for (long unsigned i = 0; i < orientation_num*translation_num; i++)
		mapped_weights[i] = std::numeric_limits<XFLOAT>::lowest();

	for (long unsigned i = idxArr_start; i < idxArr_end; i++)
		mapped_weights[ (rot_idx[i]-orientation_start) * translation_num + trans_idx[i] ]= weights[i];
}

void buildCorrImage(MlOptimiser *baseMLO,
		OptimisationParamters &op,
		AccPtr<XFLOAT> &corr_img,
		int img_id,
		long int group_id)
{

	// CC or not
	if((baseMLO->iter == 1 && baseMLO->do_firstiter_cc) || baseMLO->do_always_cc)
		for(size_t i = 0; i < corr_img.getSize(); i++)
			corr_img[i] = 1. / (op.local_sqrtXi2[img_id]*op.local_sqrtXi2[img_id]);
	else
		for(size_t i = 0; i < corr_img.getSize(); i++)
			corr_img[i] = *(op.local_Minvsigma2.data + i );

	// ctf-correction or not ( NOTE this is not were the difference metric is ctf-corrected, but
	// rather where we apply the additional correction to make the GPU-specific arithmetic equal
	// to the CPU method)
	if (baseMLO->do_ctf_correction)
	{
		if (baseMLO->refs_are_ctf_corrected)
			for(size_t i = 0; i < corr_img.getSize(); i++)
				corr_img[i] *= DIRECT_MULTIDIM_ELEM(op.local_Fctf[img_id], i)*DIRECT_MULTIDIM_ELEM(op.local_Fctf[img_id], i);
	}

	// scale-correction or not ( NOTE this is not were the difference metric is scale-corrected, but
	// rather where we apply the additional correction to make the GPU-specific arithmetic equal
	// to the CPU method)
	XFLOAT myscale = baseMLO->mymodel.scale_correction[group_id];
	if (baseMLO->do_scale_correction)
		for(size_t i = 0; i < corr_img.getSize(); i++)
			corr_img[i] *= myscale * myscale;
}

// One Euler matrix from three angles in degrees, the same arithmetic wherever it is evaluated.
static inline void eulerMatrixOne(
		RFLOAT rot, RFLOAT tilt, RFLOAT psi,
		bool inverse, bool doL, Matrix2D<RFLOAT> &L, bool doR, Matrix2D<RFLOAT> &R,
		Matrix2D<RFLOAT> &A, XFLOAT *out)
{
	RFLOAT alpha, beta, gamma;
	RFLOAT ca, sa, cb, sb, cg, sg;
	RFLOAT cc, cs, sc, ss;

	//TODO In a sense we're doing RAD2DEG just to do DEG2RAD here.
	//The only place the degree value is actually used is in the metadata assignment.
	alpha = DEG2RAD(rot);
	beta  = DEG2RAD(tilt);
	gamma = DEG2RAD(psi);

#ifdef RELION_SINGLE_PRECISION
	sincosf(alpha, &sa, &ca);
	sincosf(beta,  &sb, &cb);
	sincosf(gamma, &sg, &cg);
#else
	sincos(alpha, &sa, &ca);
	sincos(beta,  &sb, &cb);
	sincos(gamma, &sg, &cg);
#endif

	cc = cb * ca;
	cs = cb * sa;
	sc = sb * ca;
	ss = sb * sa;

	A(0, 0) =  cg * cc - sg * sa;
	A(0, 1) =  cg * cs + sg * ca;
	A(0, 2) = -cg * sb;
	A(1, 0) = -sg * cc - cg * sa;
	A(1, 1) = -sg * cs + cg * ca;
	A(1, 2) = sg * sb;
	A(2, 0) = sc;
	A(2, 1) = ss;
	A(2, 2) = cb;

	if (doL)
		A = L * A;

	if (doR)
		A = A * R;

	if(inverse)
		A = A.inv();

	for (int m = 0; m < 3; m ++)
		for (int n = 0; n < 3; n ++)
			out[m*3 + n] = A(m, n);
}

void generateEulerMatrices(
		ProjectionParams &ProjectionData,
		XFLOAT *eulers,
		bool inverse,
		Matrix2D<RFLOAT> &L,
		Matrix2D<RFLOAT> &R,
		const OrientationCache *cache)
{
	Matrix2D<RFLOAT> A(3,3);

	bool doL = (L.mdimx == 3 && L.mdimy == 3);
	bool doR = (R.mdimx == 3 && R.mdimy == 3);
	const bool use_cache = cache && inverse && !doL && !doR;

	for (long int i = 0; i < ProjectionData.rots.size(); i++)
	{
		if (use_cache)
		{
			const size_t e = ProjectionData.iorientclasses[i] % cache->entries;
			if (cache->ready(e))
			{
				const XFLOAT *src = &cache->matrices[9 * (e * cache->nover + ProjectionData.iover_rots[i])];
				for (int k = 0; k < 9; k++)
					eulers[9 * i + k] = src[k];
				continue;
			}
		}
		eulerMatrixOne(ProjectionData.rots[i], ProjectionData.tilts[i], ProjectionData.psis[i],
				inverse, doL, L, doR, R, A, &eulers[9 * i]);
	}
}

// The orientation cache of this iteration for the fine pass, or null when the orientations
// cannot be tabulated by (direction, psi) alone: priors, other than a full search range,
// bodies, or a table that would be too large. RELION_GPU_ORIENT_CACHE=0 turns it off.
static std::atomic<long> orientCacheHits(0), orientCacheMisses(0);

std::shared_ptr<OrientationCache> getOrientationCache(
		MlOptimiser *baseMLO, SamplingParameters &sp, OptimisationParamters &op)
{
	static const bool enabled = []() {
		const char *e = getenv("RELION_GPU_ORIENT_CACHE");
		return !(e && atoi(e) == 0);
	}();
	static const bool verbose = getenv("RELION_GPU_ORIENT_CACHE_VERBOSE") != nullptr;
	const char *why = nullptr;
	if (!enabled) why = "disabled";
	else if (sp.current_oversampling == 0) why = "oversampling 0";
	else if (baseMLO->mymodel.nr_bodies != 1) why = "bodies";
	else if (!op.pointer_dir_nonzeroprior.empty() || !op.pointer_psi_nonzeroprior.empty()) why = "prior pointers";
	else if (sp.idir_min != 0 || sp.ipsi_min != 0 || sp.idir_max != sp.nr_dir - 1 || sp.ipsi_max != sp.nr_psi - 1) why = "partial range";
	else if (baseMLO->do_skip_align || baseMLO->do_skip_rotate || baseMLO->do_only_sample_tilt) why = "skip";
	const size_t entries = (size_t) sp.nr_dir * sp.nr_psi;
	const size_t nover = sp.nr_oversampled_rot;
	const size_t bytes = entries * nover * (3 * sizeof(RFLOAT) + 9 * sizeof(XFLOAT));
	if (!why && bytes > ((size_t) 1 << 29)) why = "too large";
	if (why)
	{
		static std::atomic<int> reported(0);
		if (verbose && enabled && reported.fetch_add(1) < 3)
			fprintf(stderr, "ORIENT_CACHE off: %s\n", why);
		return nullptr;
	}

	struct Signature
	{
		MlOptimiser *who; int iter; int over; size_t entries, nover;
		RFLOAT perturb, psi_step; int order;
		bool operator==(const Signature &o) const
		{
			return who == o.who && iter == o.iter && over == o.over && entries == o.entries
				&& nover == o.nover && perturb == o.perturb && psi_step == o.psi_step && order == o.order;
		}
	};
	const Signature sig = { baseMLO, baseMLO->iter, (int) sp.current_oversampling, entries, nover,
			baseMLO->sampling.random_perturbation, baseMLO->sampling.psi_step, baseMLO->sampling.healpix_order };

	static std::mutex mtx;
	static Signature cur_sig = { nullptr, -1, -1, 0, 0, 0, 0, 0 };
	static std::shared_ptr<OrientationCache> cur;
	std::lock_guard<std::mutex> lock(mtx);
	if (!cur || !(cur_sig == sig))
	{
		if (verbose)
			fprintf(stderr, "ORIENT_CACHE new table iter %d: %zu entries x %zu, previous hits %ld misses %ld\n",
					sig.iter, entries, nover, orientCacheHits.exchange(0), orientCacheMisses.exchange(0));
		cur.reset(new OrientationCache(entries, nover));
		cur_sig = sig;
	}
	return cur;
}

long unsigned generateProjectionSetupFine(
		OptimisationParamters &op,
		SamplingParameters &sp,
		MlOptimiser *baseMLO,
		unsigned iclass,
		ProjectionParams &ProjectionData)
// FIXME : For coarse iteration this is **SLOW**    HERE ARE SOME NOTES FOR PARALLELIZING IT (GPU OFFLOAD):
/*
 *    Since it is based on push_back, parallelizing sould be fine given som atomic opreation appends,
 *    what takes time is looping through all this. The job-splitting in collect2jobs-preproccesing and
 *    divideOrientationsIntoBlockjobs() relies on chunks of shared orientations being adjacent in
 *    ProjectionData.rot_id (and thus also .rot_idx), but does not care which order those chunks appear
 *    in. So as long as a parallelilsm and "atomic push_back" is organised to use an orientation as a
 *    minimum unit, the job-splitting should be fine with the output.
 */
{
	//Local variables
	std::vector< RFLOAT > oversampled_rot, oversampled_tilt, oversampled_psi;
	long int orientation_num = 0;
	std::shared_ptr<OrientationCache> cache = getOrientationCache(baseMLO, sp, op);
	Matrix2D<RFLOAT> A(3,3), noL, noR;

	for (long int idir = sp.idir_min, iorient = 0; idir <= sp.idir_max; idir++)
	{
		for (long int ipsi = sp.ipsi_min; ipsi <= sp.ipsi_max; ipsi++, iorient++)
		{
			long int iorientclass = iclass * sp.nr_dir * sp.nr_psi + iorient;

			if (baseMLO->isSignificantAnyImageAnyTranslation(iorientclass, sp.itrans_min, sp.itrans_max, op.Mcoarse_significant))
			{
				if (cache && cache->ready(iorient))
				{
					orientCacheHits.fetch_add(1, std::memory_order_relaxed);
					const RFLOAT *ang = &cache->angles[3 * iorient * cache->nover];
					for (long int iover_rot = 0; iover_rot < sp.nr_oversampled_rot; iover_rot++)
					{
						ProjectionData.pushBackAll(	(long unsigned)iclass,
													ang[3 * iover_rot], ang[3 * iover_rot + 1], ang[3 * iover_rot + 2],
													iorientclass,
													iover_rot 					);
						orientation_num ++;
					}
					continue;
				}

				// Now get the oversampled (rot, tilt, psi) triplets
				// This will be only the original (rot,tilt,psi) triplet in the first pass (sp.current_oversampling==0)
				baseMLO->sampling.getOrientations(idir, ipsi, sp.current_oversampling, oversampled_rot, oversampled_tilt, oversampled_psi,
						op.pointer_dir_nonzeroprior, op.directions_prior, op.pointer_psi_nonzeroprior, op.psi_prior);

				orientCacheMisses.fetch_add(1, std::memory_order_relaxed);
				if (cache && cache->claim(iorient))
				{
					for (long int iover_rot = 0; iover_rot < sp.nr_oversampled_rot; iover_rot++)
					{
						RFLOAT *ang = &cache->angles[3 * (iorient * cache->nover + iover_rot)];
						ang[0] = oversampled_rot[iover_rot];
						ang[1] = oversampled_tilt[iover_rot];
						ang[2] = oversampled_psi[iover_rot];
						eulerMatrixOne(ang[0], ang[1], ang[2], true, false, noL, false, noR, A,
								&cache->matrices[9 * (iorient * cache->nover + iover_rot)]);
					}
					cache->publish(iorient);
				}

				// Loop over all oversampled orientations (only a single one in the first pass)
				for (long int iover_rot = 0; iover_rot < sp.nr_oversampled_rot; iover_rot++)
				{
					ProjectionData.pushBackAll(	(long unsigned)iclass,
												oversampled_rot[iover_rot],
											    oversampled_tilt[iover_rot],
											    oversampled_psi[iover_rot],
											    iorientclass,
											    iover_rot 					);
					orientation_num ++;
				}
			}
		}
	}
	ProjectionData.orientation_num[iclass]=orientation_num;
	return orientation_num;
}

void runWavgKernel(
		AccProjectorKernel &projector,
		XFLOAT *eulers,
		XFLOAT *Fimg_real,
		XFLOAT *Fimg_imag,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT *sorted_weights,
		XFLOAT *ctfs,
		XFLOAT *wdiff2s_parts,
		XFLOAT *wdiff2s_AA,
		XFLOAT *wdiff2s_XA,
		OptimisationParamters &op,
		long unsigned orientation_num,
		long unsigned translation_num,
		unsigned long image_size,
		int img_id,
		int group_id,
		int exp_iclass,
		XFLOAT part_scale,
		bool refs_are_ctf_corrected,
		bool ctf_premultiplied,
		bool data_is_3D,
		deviceStream_t stream)
{
	//cudaFuncSetCacheConfig(cuda_kernel_wavg_fast, cudaFuncCachePreferShared);

	if (refs_are_ctf_corrected)
	{
		if(data_is_3D)
			AccUtilities::kernel_wavg<true,true,true,WAVG_BLOCK_SIZE_DATA3D>(
				eulers,
				projector,
				image_size,
				orientation_num,
				Fimg_real,
				Fimg_imag,
				trans_x,
				trans_y,
				trans_z,
				sorted_weights,
				ctfs,
				wdiff2s_parts,
				wdiff2s_AA,
				wdiff2s_XA,
				translation_num,
				(XFLOAT) op.sum_weight,
				(XFLOAT) op.significant_weight,
				part_scale,
				stream
				);
		else if (projector.mdlZ!=0)
#if defined ALTCPU || defined _SYCL_ENABLED
			AccUtilities::kernel_wavg<true,true,false,WAVG_BLOCK_SIZE_REF3D>(
#else
			AccUtilities::kernel_wavg<true,true,false,WAVG_BLOCK_SIZE>(
#endif
				eulers,
				projector,
				image_size,
				orientation_num,
				Fimg_real,
				Fimg_imag,
				trans_x,
				trans_y,
				trans_z,
				sorted_weights,
				ctfs,
				wdiff2s_parts,
				wdiff2s_AA,
				wdiff2s_XA,
				translation_num,
				(XFLOAT) op.sum_weight,
				(XFLOAT) op.significant_weight,
				part_scale,
				stream
				);
		else
#if defined ALTCPU || defined _SYCL_ENABLED
			AccUtilities::kernel_wavg<true,false,false,WAVG_BLOCK_SIZE_2D>(
#else
			AccUtilities::kernel_wavg<true,false,false,WAVG_BLOCK_SIZE>(
#endif
				eulers,
				projector,
				image_size,
				orientation_num,
				Fimg_real,
				Fimg_imag,
				trans_x,
				trans_y,
				trans_z,
				sorted_weights,
				ctfs,
				wdiff2s_parts,
				wdiff2s_AA,
				wdiff2s_XA,
				translation_num,
				(XFLOAT) op.sum_weight,
				(XFLOAT) op.significant_weight,
				part_scale,
				stream
				);
	}
	else
	{
		if(data_is_3D)
			AccUtilities::kernel_wavg<false,true,true,WAVG_BLOCK_SIZE_DATA3D>(
				eulers,
				projector,
				image_size,
				orientation_num,
				Fimg_real,
				Fimg_imag,
				trans_x,
				trans_y,
				trans_z,
				sorted_weights,
				ctfs,
				wdiff2s_parts,
				wdiff2s_AA,
				wdiff2s_XA,
				translation_num,
				(XFLOAT) op.sum_weight,
				(XFLOAT) op.significant_weight,
				part_scale,
				stream
				);
		else if (projector.mdlZ!=0)
#if defined ALTCPU || defined _SYCL_ENABLED
			AccUtilities::kernel_wavg<false,true,false,WAVG_BLOCK_SIZE_REF3D>(
#else
			AccUtilities::kernel_wavg<false,true,false,WAVG_BLOCK_SIZE>(
#endif
				eulers,
				projector,
				image_size,
				orientation_num,
				Fimg_real,
				Fimg_imag,
				trans_x,
				trans_y,
				trans_z,
				sorted_weights,
				ctfs,
				wdiff2s_parts,
				wdiff2s_AA,
				wdiff2s_XA,
				translation_num,
				(XFLOAT) op.sum_weight,
				(XFLOAT) op.significant_weight,
				part_scale,
				stream
				);
		else
#if defined ALTCPU || defined _SYCL_ENABLED
			AccUtilities::kernel_wavg<false,false,false,WAVG_BLOCK_SIZE_2D>(
#else
			AccUtilities::kernel_wavg<false,false,false,WAVG_BLOCK_SIZE>(
#endif
				eulers,
				projector,
				image_size,
				orientation_num,
				Fimg_real,
				Fimg_imag,
				trans_x,
				trans_y,
				trans_z,
				sorted_weights,
				ctfs,
				wdiff2s_parts,
				wdiff2s_AA,
				wdiff2s_XA,
				translation_num,
				(XFLOAT) op.sum_weight,
				(XFLOAT) op.significant_weight,
				part_scale,
				stream
				);
	}
#ifdef _CUDA_ENABLED
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
#elif _HIP_ENABLED
	LAUNCH_HANDLE_ERROR(hipGetLastError());
#endif
}

void runBackProjectKernel(
		AccBackprojector &BP,
		AccProjectorKernel &projector,
		XFLOAT *d_img_real,
		XFLOAT *d_img_imag,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT* d_weights,
		XFLOAT* d_Minvsigma2s,
		XFLOAT* d_ctfs,
		unsigned long translation_num,
		XFLOAT significant_weight,
		XFLOAT weight_norm,
		XFLOAT *d_eulers,
		int imgX,
		int imgY,
		int imgZ,
		unsigned long imageCount,
		bool data_is_3D,
		bool do_grad,
		bool ctf_premultiplied,
		deviceStream_t optStream)
{

	if(BP.mdlZ==1)
	{
#ifdef _CUDA_ENABLED
		if(do_grad)
			if(ctf_premultiplied)
				cuda_kernel_backproject2D_SGD<true><<<imageCount, BP_2D_BLOCK_SIZE, 0, optStream>>>(
						projector,
                                                d_img_real, d_img_imag,
						trans_x, trans_y,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, BP.padding_factor,
						imgX, imgY, imgX*imgY,
						BP.mdlX, BP.mdlInitY);
			else
				cuda_kernel_backproject2D_SGD<false><<<imageCount, BP_2D_BLOCK_SIZE, 0, optStream>>>(
						projector,
                                                d_img_real, d_img_imag,
						trans_x, trans_y,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, BP.padding_factor,
						imgX, imgY, imgX*imgY,
						BP.mdlX, BP.mdlInitY);
		else
			if(ctf_premultiplied)
				cuda_kernel_backproject2D<true><<<imageCount,BP_2D_BLOCK_SIZE,0,optStream>>>(
					d_img_real, d_img_imag,
					trans_x, trans_y,
					d_weights, d_Minvsigma2s, d_ctfs,
					translation_num, significant_weight, weight_norm, d_eulers,
					BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
					BP.maxR, BP.maxR2, BP.padding_factor,
					imgX, imgY, imgX*imgY,
					BP.mdlX, BP.mdlInitY);
			else
				cuda_kernel_backproject2D<false><<<imageCount,BP_2D_BLOCK_SIZE,0,optStream>>>(
					d_img_real, d_img_imag,
					trans_x, trans_y,
					d_weights, d_Minvsigma2s, d_ctfs,
					translation_num, significant_weight, weight_norm, d_eulers,
					BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
					BP.maxR, BP.maxR2, BP.padding_factor,
					imgX, imgY, imgX*imgY,
					BP.mdlX, BP.mdlInitY);
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
#elif  _HIP_ENABLED
		if(do_grad)
			if(ctf_premultiplied)
				hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_backproject2D_SGD<true>), dim3(imageCount), dim3(BP_2D_BLOCK_SIZE), 0, optStream,
						projector,
                        d_img_real, d_img_imag,
						trans_x, trans_y,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, BP.padding_factor,
						imgX, imgY, imgX*imgY,
						BP.mdlX, BP.mdlInitY);
			else
				hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_backproject2D_SGD<false>), dim3(imageCount), dim3(BP_2D_BLOCK_SIZE), 0, optStream,
						projector,
                        d_img_real, d_img_imag,
						trans_x, trans_y,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, BP.padding_factor,
						imgX, imgY, imgX*imgY,
						BP.mdlX, BP.mdlInitY);
		else
			if(ctf_premultiplied)
				hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_backproject2D<true>), dim3(imageCount), dim3(BP_2D_BLOCK_SIZE), 0, optStream,
					d_img_real, d_img_imag,
					trans_x, trans_y,
					d_weights, d_Minvsigma2s, d_ctfs,
					translation_num, significant_weight, weight_norm, d_eulers,
					BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
					BP.maxR, BP.maxR2, BP.padding_factor,
					imgX, imgY, imgX*imgY,
					BP.mdlX, BP.mdlInitY);
			else
				hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_backproject2D<false>), dim3(imageCount), dim3(BP_2D_BLOCK_SIZE), 0, optStream,
					d_img_real, d_img_imag,
					trans_x, trans_y,
					d_weights, d_Minvsigma2s, d_ctfs,
					translation_num, significant_weight, weight_norm, d_eulers,
					BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
					BP.maxR, BP.maxR2, BP.padding_factor,
					imgX, imgY, imgX*imgY,
					BP.mdlX, BP.mdlInitY);
		LAUNCH_HANDLE_ERROR(hipGetLastError());
#elif _SYCL_ENABLED
		if(do_grad)
			if(ctf_premultiplied)
				syclKernels::backproject2D<true, true>(
						imageCount, BP_2D_BLOCK_SIZE,
						projector,
						d_img_real, d_img_imag,
						trans_x, trans_y,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, (XFLOAT)BP.padding_factor,
						imgX, imgY, imgX*imgY,
						BP.mdlX, BP.mdlInitY, BP.mdlY, optStream);
			else
				syclKernels::backproject2D<false, true>(
						imageCount, BP_2D_BLOCK_SIZE,
						projector,
						d_img_real, d_img_imag,
						trans_x, trans_y,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, (XFLOAT)BP.padding_factor,
						imgX, imgY, imgX*imgY,
						BP.mdlX, BP.mdlInitY, BP.mdlY, optStream);
		else
			if(ctf_premultiplied)
				syclKernels::backproject2D<true, false>(
						imageCount, BP_2D_BLOCK_SIZE,
						projector,
						d_img_real, d_img_imag,
						trans_x, trans_y,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, (XFLOAT)BP.padding_factor,
						imgX, imgY, imgX*imgY,
						BP.mdlX, BP.mdlInitY, BP.mdlY, optStream);
			else
				syclKernels::backproject2D<false, false>(
						imageCount, BP_2D_BLOCK_SIZE,
						projector,
						d_img_real, d_img_imag,
						trans_x, trans_y,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, (XFLOAT)BP.padding_factor,
						imgX, imgY, imgX*imgY,
						BP.mdlX, BP.mdlInitY, BP.mdlY, optStream);
#else
		if(do_grad)
			if(ctf_premultiplied)
				CpuKernels::backproject2D_SGD<true>(
						imageCount, BP_2D_BLOCK_SIZE,
						projector,
                        d_img_real, d_img_imag,
                        trans_x, trans_y,
                        d_weights, d_Minvsigma2s, d_ctfs,
                        translation_num, significant_weight, weight_norm, d_eulers,
                        BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
                        BP.maxR, BP.maxR2, (XFLOAT)BP.padding_factor,
                        (unsigned)imgX, (unsigned)imgY, (unsigned)imgX*imgY,
                        (unsigned)BP.mdlX, BP.mdlInitY, BP.mutexes);
			else
				CpuKernels::backproject2D_SGD<false>(
						imageCount, BP_2D_BLOCK_SIZE,
						projector,
						d_img_real, d_img_imag,
						trans_x, trans_y,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, (XFLOAT)BP.padding_factor,
						(unsigned)imgX, (unsigned)imgY, (unsigned)imgX*imgY,
						(unsigned)BP.mdlX, BP.mdlInitY, BP.mutexes);
		else
			if(ctf_premultiplied)
				CpuKernels::backproject2D<true>(
						imageCount, BP_2D_BLOCK_SIZE,
						d_img_real, d_img_imag,
						trans_x, trans_y,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, (XFLOAT)BP.padding_factor,
						(unsigned)imgX, (unsigned)imgY, (unsigned)imgX*imgY,
						(unsigned)BP.mdlX, BP.mdlInitY, BP.mutexes);
			else
				CpuKernels::backproject2D<false>(
						imageCount, BP_2D_BLOCK_SIZE,
						d_img_real, d_img_imag,
						trans_x, trans_y,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, (XFLOAT)BP.padding_factor,
						(unsigned)imgX, (unsigned)imgY, (unsigned)imgX*imgY,
						(unsigned)BP.mdlX, BP.mdlInitY, BP.mutexes);
#endif
	}
	else
	{
		if(do_grad)
		{
			if(data_is_3D)
#ifdef _CUDA_ENABLED
				if(ctf_premultiplied)
                    cuda_kernel_backproject3D_SGD<true, true><<<imageCount, BP_DATA3D_BLOCK_SIZE, 0, optStream>>>(
                            projector, d_img_real, d_img_imag,
                            trans_x, trans_y, trans_z,
                            d_weights, d_Minvsigma2s, d_ctfs,
                            translation_num, significant_weight, weight_norm, d_eulers,
                            BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
                            BP.maxR, BP.maxR2, BP.padding_factor,
                            imgX, imgY, imgZ, imgX * imgY * imgZ,
                            BP.mdlX, BP.mdlY, BP.mdlInitY, BP.mdlInitZ);
				else
                    cuda_kernel_backproject3D_SGD<true, false><<<imageCount, BP_DATA3D_BLOCK_SIZE, 0, optStream>>>(
                            projector, d_img_real, d_img_imag,
                            trans_x, trans_y, trans_z,
                            d_weights, d_Minvsigma2s, d_ctfs,
                            translation_num, significant_weight, weight_norm, d_eulers,
                            BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
                            BP.maxR, BP.maxR2, BP.padding_factor,
                            imgX, imgY, imgZ, imgX * imgY * imgZ,
                            BP.mdlX, BP.mdlY, BP.mdlInitY, BP.mdlInitZ);
#elif _HIP_ENABLED
				if(ctf_premultiplied)
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_backproject3D_SGD<true, true>),dim3(imageCount), dim3(BP_DATA3D_BLOCK_SIZE), 0, optStream,
                            projector, d_img_real, d_img_imag,
                            trans_x, trans_y, trans_z,
                            d_weights, d_Minvsigma2s, d_ctfs,
                            translation_num, significant_weight, weight_norm, d_eulers,
                            BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
                            BP.maxR, BP.maxR2, BP.padding_factor,
                            imgX, imgY, imgZ, imgX * imgY * imgZ,
                            BP.mdlX, BP.mdlY, BP.mdlInitY, BP.mdlInitZ);
				else
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_backproject3D_SGD<true, false>),  dim3(imageCount), dim3(BP_DATA3D_BLOCK_SIZE), 0, optStream,
                            projector, d_img_real, d_img_imag,
                            trans_x, trans_y, trans_z,
                            d_weights, d_Minvsigma2s, d_ctfs,
                            translation_num, significant_weight, weight_norm, d_eulers,
                            BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
                            BP.maxR, BP.maxR2, BP.padding_factor,
                            imgX, imgY, imgZ, imgX * imgY * imgZ,
                            BP.mdlX, BP.mdlY, BP.mdlInitY, BP.mdlInitZ);
#elif _SYCL_ENABLED
				if(ctf_premultiplied)
					syclKernels::backproject3D<true, true, true>(imageCount, BP_DATA3D_BLOCK_SIZE,
						projector, d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, BP.padding_factor,
						imgX, imgY, imgZ, imgX*imgY*imgZ,
						BP.mdlX, BP.mdlY, BP.mdlInitY,  BP.mdlInitZ, BP.mdlXYZ, optStream);
				else
					syclKernels::backproject3D<true, false, true>(imageCount, BP_DATA3D_BLOCK_SIZE,
						projector, d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, BP.padding_factor,
						imgX, imgY, imgZ, imgX*imgY*imgZ,
						BP.mdlX, BP.mdlY, BP.mdlInitY,  BP.mdlInitZ, BP.mdlXYZ, optStream);
#else
				if(ctf_premultiplied)
					CpuKernels::backproject3D_SGD<true, true>(imageCount, BP_DATA3D_BLOCK_SIZE,
						projector, d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, BP.padding_factor,
						imgX, imgY, imgZ, (size_t)imgX*(size_t)imgY*(size_t)imgZ,
						BP.mdlX, BP.mdlY, BP.mdlInitY, 	BP.mdlInitZ, BP.mutexes);
				else
					CpuKernels::backproject3D_SGD<true, false>(imageCount, BP_DATA3D_BLOCK_SIZE,
						projector, d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, BP.padding_factor,
						imgX, imgY, imgZ, (size_t)imgX*(size_t)imgY*(size_t)imgZ,
						BP.mdlX, BP.mdlY, BP.mdlInitY, 	BP.mdlInitZ, BP.mutexes);

#endif
			else
#ifdef _CUDA_ENABLED
		        if(ctf_premultiplied)
				cuda_kernel_backproject3D_SGD<false, true><<<imageCount, BP_REF3D_BLOCK_SIZE, 0, optStream>>>(
			                projector, d_img_real, d_img_imag,
			                trans_x, trans_y, trans_z,
			                d_weights, d_Minvsigma2s, d_ctfs,
			                translation_num, significant_weight, weight_norm, d_eulers,
			                BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
			                BP.maxR, BP.maxR2, BP.padding_factor,
			                imgX, imgY, imgZ, imgX * imgY * imgZ,
			                BP.mdlX, BP.mdlY, BP.mdlInitY, BP.mdlInitZ);
			    else
			        cuda_kernel_backproject3D_SGD<false, false><<<imageCount, BP_REF3D_BLOCK_SIZE, 0, optStream>>>(
			                projector, d_img_real, d_img_imag,
			                trans_x, trans_y, trans_z,
			                d_weights, d_Minvsigma2s, d_ctfs,
			                translation_num, significant_weight, weight_norm, d_eulers,
			                BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
			                BP.maxR, BP.maxR2, BP.padding_factor,
			                imgX, imgY, imgZ, imgX * imgY * imgZ,
			                BP.mdlX, BP.mdlY, BP.mdlInitY, BP.mdlInitZ);
#elif _HIP_ENABLED
		        if(ctf_premultiplied)
					hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_backproject3D_SGD<false, true>), dim3(imageCount), dim3(BP_REF3D_BLOCK_SIZE), 0, optStream,
			                projector, d_img_real, d_img_imag,
			                trans_x, trans_y, trans_z,
			                d_weights, d_Minvsigma2s, d_ctfs,
			                translation_num, significant_weight, weight_norm, d_eulers,
			                BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
			                BP.maxR, BP.maxR2, BP.padding_factor,
			                imgX, imgY, imgZ, imgX * imgY * imgZ,
			                BP.mdlX, BP.mdlY, BP.mdlInitY, BP.mdlInitZ);
			    else
			        hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_backproject3D_SGD<false, false>), dim3(imageCount), dim3(BP_REF3D_BLOCK_SIZE), 0, optStream,
			                projector, d_img_real, d_img_imag,
			                trans_x, trans_y, trans_z,
			                d_weights, d_Minvsigma2s, d_ctfs,
			                translation_num, significant_weight, weight_norm, d_eulers,
			                BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
			                BP.maxR, BP.maxR2, BP.padding_factor,
			                imgX, imgY, imgZ, imgX * imgY * imgZ,
			                BP.mdlX, BP.mdlY, BP.mdlInitY, BP.mdlInitZ);
#elif _SYCL_ENABLED
				if(ctf_premultiplied)
					syclKernels::backproject3D<false, true, true>(imageCount, BP_REF3D_BLOCK_SIZE,
						projector, d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, (XFLOAT)BP.padding_factor,
						imgX, imgY, imgZ, imgX*imgY*imgZ,
						BP.mdlX, BP.mdlY, BP.mdlInitY,  BP.mdlInitZ, BP.mdlXYZ, optStream);
				else
					syclKernels::backproject3D<false, false, true>(imageCount, BP_REF3D_BLOCK_SIZE,
						projector, d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, (XFLOAT)BP.padding_factor,
						imgX, imgY, imgZ, imgX*imgY*imgZ,
						BP.mdlX, BP.mdlY, BP.mdlInitY, BP.mdlInitZ, BP.mdlXYZ, optStream);
#else
				if(ctf_premultiplied)
					CpuKernels::backproject3D_SGD<false, true>(imageCount, BP_REF3D_BLOCK_SIZE,
						projector, d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, (XFLOAT)BP.padding_factor,
						(unsigned)imgX, (unsigned)imgY, (unsigned)imgZ, (size_t)imgX*(size_t)imgY*(size_t)imgZ,
						(unsigned)BP.mdlX, (unsigned)BP.mdlY, BP.mdlInitY, 	BP.mdlInitZ, BP.mutexes);
				else
					CpuKernels::backproject3D_SGD<false, false>(imageCount, BP_REF3D_BLOCK_SIZE,
						projector, d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, (XFLOAT)BP.padding_factor,
						(unsigned)imgX, (unsigned)imgY, (unsigned)imgZ, (size_t)imgX*(size_t)imgY*(size_t)imgZ,
						(unsigned)BP.mdlX, (unsigned)BP.mdlY, BP.mdlInitY, 	BP.mdlInitZ, BP.mutexes);
#endif
		}
		else
		{
			if(data_is_3D)
#ifdef _CUDA_ENABLED
				if(ctf_premultiplied)
					cuda_kernel_backproject3D<true, true><<<imageCount,BP_DATA3D_BLOCK_SIZE,0,optStream>>>(
						d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, BP.padding_factor,
						imgX, imgY, imgZ, imgX*imgY*imgZ,
						BP.mdlX, BP.mdlY, BP.mdlInitY, 	BP.mdlInitZ);
				else
					cuda_kernel_backproject3D<true, false><<<imageCount,BP_DATA3D_BLOCK_SIZE,0,optStream>>>(
						d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, BP.padding_factor,
						imgX, imgY, imgZ, imgX*imgY*imgZ,
						BP.mdlX, BP.mdlY, BP.mdlInitY, 	BP.mdlInitZ);
#elif _HIP_ENABLED
				if(ctf_premultiplied)
					hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_backproject3D<true, true>), dim3(imageCount), dim3(BP_DATA3D_BLOCK_SIZE), 0, optStream,
						d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, BP.padding_factor,
						imgX, imgY, imgZ, imgX*imgY*imgZ,
						BP.mdlX, BP.mdlY, BP.mdlInitY, 	BP.mdlInitZ);
				else
					hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_backproject3D<true, false>), dim3(imageCount), dim3(BP_DATA3D_BLOCK_SIZE), 0, optStream,
						d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, BP.padding_factor,
						imgX, imgY, imgZ, imgX*imgY*imgZ,
						BP.mdlX, BP.mdlY, BP.mdlInitY, 	BP.mdlInitZ);
#elif _SYCL_ENABLED
				if(ctf_premultiplied)
					syclKernels::backproject3D<true, true, false>(imageCount,BP_DATA3D_BLOCK_SIZE,
						projector,
						d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, (XFLOAT)BP.padding_factor,
						imgX, imgY, imgZ, imgX*imgY*imgZ,
						BP.mdlX, BP.mdlY, BP.mdlInitY, BP.mdlInitZ, BP.mdlXYZ, optStream);
				else
					syclKernels::backproject3D<true, false, false>(imageCount,BP_DATA3D_BLOCK_SIZE,
						projector,
						d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, (XFLOAT)BP.padding_factor,
						imgX, imgY, imgZ, imgX*imgY*imgZ,
						BP.mdlX, BP.mdlY, BP.mdlInitY, BP.mdlInitZ, BP.mdlXYZ, optStream);
#else
			    if(ctf_premultiplied)
					CpuKernels::backproject3D<true, true>(imageCount,BP_DATA3D_BLOCK_SIZE,
						d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, (XFLOAT)BP.padding_factor,
						(unsigned)imgX, (unsigned)imgY, (unsigned)imgZ, (size_t)imgX*(size_t)imgY*(size_t)imgZ,
						(unsigned)BP.mdlX, (unsigned)BP.mdlY, BP.mdlInitY, 	BP.mdlInitZ, BP.mutexes);
			    else
					CpuKernels::backproject3D<true, false>(imageCount,BP_DATA3D_BLOCK_SIZE,
						d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, (XFLOAT)BP.padding_factor,
						(unsigned)imgX, (unsigned)imgY, (unsigned)imgZ, (size_t)imgX*(size_t)imgY*(size_t)imgZ,
						(unsigned)BP.mdlX, (unsigned)BP.mdlY, BP.mdlInitY, 	BP.mdlInitZ, BP.mutexes);

#endif
			else
#ifdef _CUDA_ENABLED
			    if(ctf_premultiplied)
					cuda_kernel_backproject3D<false, true><<<imageCount,BP_REF3D_BLOCK_SIZE,0,optStream>>>(
						d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, BP.padding_factor,
						imgX, imgY, imgZ, imgX*imgY*imgZ,
						BP.mdlX, BP.mdlY, BP.mdlInitY, 	BP.mdlInitZ);
			    else
					cuda_kernel_backproject3D<false, false><<<imageCount,BP_REF3D_BLOCK_SIZE,0,optStream>>>(
						d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, BP.padding_factor,
						imgX, imgY, imgZ, imgX*imgY*imgZ,
						BP.mdlX, BP.mdlY, BP.mdlInitY, 	BP.mdlInitZ);
#elif _HIP_ENABLED
			    if(ctf_premultiplied)
					hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_backproject3D<false, true>), dim3(imageCount), dim3(BP_REF3D_BLOCK_SIZE), 0, optStream,
						d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, BP.padding_factor,
						imgX, imgY, imgZ, imgX*imgY*imgZ,
						BP.mdlX, BP.mdlY, BP.mdlInitY, 	BP.mdlInitZ);
			    else
					hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_backproject3D<false, false>), dim3(imageCount), dim3(BP_REF3D_BLOCK_SIZE), 0, optStream,
						d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, BP.padding_factor,
						imgX, imgY, imgZ, imgX*imgY*imgZ,
						BP.mdlX, BP.mdlY, BP.mdlInitY, 	BP.mdlInitZ);
#elif _SYCL_ENABLED
				if(ctf_premultiplied)
					syclKernels::backproject3D<false, true, false>(imageCount, BP_REF3D_BLOCK_SIZE,
						projector,
						d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, (XFLOAT)BP.padding_factor,
						imgX, imgY, imgZ, imgX*imgY*imgZ,
						BP.mdlX, BP.mdlY, BP.mdlInitY, BP.mdlInitZ, BP.mdlXYZ, optStream);
				else
					syclKernels::backproject3D<false, false, false>(imageCount, BP_REF3D_BLOCK_SIZE,
						projector,
						d_img_real, d_img_imag,
						trans_x, trans_y, trans_z,
						d_weights, d_Minvsigma2s, d_ctfs,
						translation_num, significant_weight, weight_norm, d_eulers,
						BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
						BP.maxR, BP.maxR2, (XFLOAT)BP.padding_factor,
						imgX, imgY, imgZ, imgX*imgY*imgZ,
						BP.mdlX, BP.mdlY, BP.mdlInitY, BP.mdlInitZ, BP.mdlXYZ, optStream);
#else
			if(ctf_premultiplied)
				CpuKernels::backprojectRef3D<true>(imageCount,
					d_img_real, d_img_imag,
					trans_x, trans_y,
					d_weights, d_Minvsigma2s, d_ctfs,
					translation_num, significant_weight, weight_norm, d_eulers,
					BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
					BP.maxR, BP.maxR2, (XFLOAT)BP.padding_factor,
					(unsigned)imgX, (unsigned)imgY, (unsigned)imgZ, (size_t)imgX*(size_t)imgY*(size_t)imgZ,
					(unsigned)BP.mdlX, (unsigned)BP.mdlY, BP.mdlInitY, 	BP.mdlInitZ, BP.mutexes);
			else
				CpuKernels::backprojectRef3D<false>(imageCount,
					d_img_real, d_img_imag,
					trans_x, trans_y,
					d_weights, d_Minvsigma2s, d_ctfs,
					translation_num, significant_weight, weight_norm, d_eulers,
					BP.d_mdlReal, BP.d_mdlImag, BP.d_mdlWeight,
					BP.maxR, BP.maxR2, (XFLOAT)BP.padding_factor,
					(unsigned)imgX, (unsigned)imgY, (unsigned)imgZ, (size_t)imgX*(size_t)imgY*(size_t)imgZ,
					(unsigned)BP.mdlX, (unsigned)BP.mdlY, BP.mdlInitY, 	BP.mdlInitZ, BP.mutexes);
#endif
		} // do_grad is false
	#ifdef _CUDA_ENABLED
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
	#elif _HIP_ENABLED
		LAUNCH_HANDLE_ERROR(hipGetLastError());
	#endif
	}
}

#define WEIGHT_MAP_BLOCK_SIZE 512

#ifdef _CUDA_ENABLED
// All classes of a particle in one launch (per-class launches cost a driver call each: with 100
// classes in 2D that was 100 tiny launches per particle)
void mapAllWeightsToMweightsBatched(
		const std::vector<WeightMapJob> &jobs,
		XFLOAT * d_allweights,
		XFLOAT * d_mweights,
		unsigned long translation_num,
		AccPtr<char> &table)
{
	if (jobs.empty())
		return;
	unsigned long max_orient = 0;
	for (const WeightMapJob &j : jobs)
		max_orient = std::max(max_orient, j.n_orient);
	table.setSize(jobs.size() * sizeof(WeightMapJob));
	table.allAlloc();
	memcpy(&table[0], jobs.data(), jobs.size() * sizeof(WeightMapJob));
	table.cpToDevice();
	dim3 grid((unsigned) ceil((float)(max_orient * translation_num) / (float) WEIGHT_MAP_BLOCK_SIZE), (unsigned) jobs.size());
	cuda_kernel_allweights_to_mweights_batched<<< grid, WEIGHT_MAP_BLOCK_SIZE, 0, table.getStream() >>>(
			(const WeightMapJob *) ~table, d_allweights, d_mweights, translation_num);
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
}
#endif

void mapAllWeightsToMweights(
		unsigned long * d_iorient, //projectorPlan.iorientclasses
		XFLOAT * d_allweights, //allWeights
		XFLOAT * d_mweights, //Mweight
		unsigned long orientation_num, //projectorPlan.orientation_num
		unsigned long translation_num, //translation_num
		deviceStream_t stream
		)
{
	size_t combinations = orientation_num*translation_num;
	int grid_size = ceil((float)(combinations)/(float)WEIGHT_MAP_BLOCK_SIZE);
#ifdef _CUDA_ENABLED
	cuda_kernel_allweights_to_mweights<<< grid_size, WEIGHT_MAP_BLOCK_SIZE, 0, stream >>>(
			d_iorient,
			d_allweights,
			d_mweights,
			orientation_num,
			translation_num,
			WEIGHT_MAP_BLOCK_SIZE);
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
#elif _HIP_ENABLED
	hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_allweights_to_mweights), dim3(grid_size), dim3(WEIGHT_MAP_BLOCK_SIZE), 0, stream,
			d_iorient,
			d_allweights,
			d_mweights,
			orientation_num,
			translation_num,
			WEIGHT_MAP_BLOCK_SIZE);
	LAUNCH_HANDLE_ERROR(hipGetLastError());
#elif defined(_SYCL_ENABLED) && defined(USE_ONEDPL)
	syclGpuKernels::sycl_kernel_allweights_to_mweights(
			d_iorient,
			d_allweights,
			d_mweights,
			orientation_num,
			translation_num,
			WEIGHT_MAP_BLOCK_SIZE, stream);
#else
	for (size_t i=0; i < combinations; i++)
		d_mweights[d_iorient[i/translation_num] * translation_num + i%translation_num] =
			d_allweights[i/translation_num * translation_num + i%translation_num];
			// TODO - isn't this just d_allweights[idx + idx%translation_num]?   Really?
#endif
}

#ifdef _CUDA_ENABLED

// Pixel chunks of the class-batched wavg and backprojection kernels: a particle has
// one block per orientation (a few dozen for local searches), so its kernels leave
// most SMs idle. Chunks of the pixel passes give n_chunks times as many blocks;
// chosen for about 3 blocks per SM, at least 2 passes per chunk. An env value N > 0
// sets the number (1: no chunks, as before).
static int envChunkOverride(const char *name)
{
	const char *e = getenv(name);
	return e ? atoi(e) : 0;
}

#ifndef ACC_DOUBLE_PRECISION
// Run-time choice of how many 16-translation tiles one coarse_xt block handles (RELION_GPU_XT_TILES).
// A bigger value reads each reference projection once for several tiles, which saves texture/L2
// traffic when a particle has more than 16 translations. Every value gives bit-identical results, so
// the choice can be timed on real launches: candidates are launched in turn, timed with events
// (read later, never waiting), and the median of the last few launches per candidate decides.
// A smaller value wins ties and anything under a 3% gain. One launch in 256 re-times the candidate
// with the oldest sample, so a changed workload or a busy GPU is noticed.
// On by default (automatic choice). RELION_GPU_XT_TILES=N fixes the value, =off (or 0) keeps 1 tile.
struct XtTileTuner
{
	static const int MAXC = 6, RING = 5;
	struct Pending { cudaEvent_t t0, t1; int cand; double work; };
	struct State
	{
		int ncand = 0, cands[MAXC] = {0};
		double samples[MAXC][RING];
		int count[MAXC] = {0};
		bool dead[MAXC] = {false};
		int best = 0;
		bool announced = false;
		long calls = 0;
		std::vector<Pending> pending;
	};
	std::mutex mtx;
	std::map<std::pair<int, int>, State> states;
	std::vector<std::pair<cudaEvent_t, cudaEvent_t> > free_events;

	static double median(const double *v, int n)
	{
		double t[RING];
		for (int i = 0; i < n; i++) t[i] = v[i];
		std::sort(t, t + n);
		return t[n / 2];
	}

	void collect(State &st)
	{
		for (size_t i = 0; i < st.pending.size(); )
		{
			Pending &pd = st.pending[i];
			if (cudaEventQuery(pd.t1) == cudaErrorNotReady) { i++; continue; }
			float ms = 0;
			if (cudaEventElapsedTime(&ms, pd.t0, pd.t1) == cudaSuccess && ms > 0)
			{
				const int c = pd.cand;
				st.samples[c][st.count[c] % RING] = 1e3 * ms / pd.work;
				st.count[c]++;
			}
			cudaGetLastError();
			free_events.push_back(std::make_pair(pd.t0, pd.t1));
			st.pending.erase(st.pending.begin() + i);
		}
	}

	void decide(State &st, int device, int ntt)
	{
		for (int c = 0; c < st.ncand; c++)
			if (!st.dead[c] && st.count[c] < RING)
				return;
		int best = st.best;
		double bm = 1e300;
		for (int c = 0; c < st.ncand; c++)
			if (!st.dead[c] && c == best) bm = median(st.samples[c], RING);
		if (bm == 1e300) best = 0;
		int pick = best;
		double pm = bm;
		for (int c = 0; c < st.ncand; c++)
		{
			if (st.dead[c] || c == best) continue;
			const double m = median(st.samples[c], RING);
			if (m < pm * 0.97 || (m < pm && cands_less(st, c, pick))) { pick = c; pm = m; }
		}
		if (pick != st.best || !st.announced)
		{
			fprintf(stderr, " GPU coarse_xt tiles: device %d, %d translation tiles -> %d per block (measured, per-launch median relative to 1 tile:",
					device, ntt, st.cands[pick]);
			const double base = median(st.samples[0], RING);
			for (int c = 0; c < st.ncand; c++)
				if (!st.dead[c]) fprintf(stderr, " %d:%.2f", st.cands[c], median(st.samples[c], RING) / base);
			fprintf(stderr, "); 3%% margin, re-checked every 256 launches)\n");
			st.best = pick;
			st.announced = true;
		}
	}
	static bool cands_less(const State &st, int a, int b) { return st.cands[a] < st.cands[b]; }

	// Returns the number of tiles to launch with; the caller reports the event pair with finish()

	int begin(int device, int ntt, cudaStream_t stream, int &cand_out, std::pair<cudaEvent_t, cudaEvent_t> &ev)
	{
		std::lock_guard<std::mutex> lock(mtx);
		State &st = states[std::make_pair(device, ntt)];
		if (st.ncand == 0)
		{
			static const int all[] = {1, 2, 3, 4, 6, 8};
			for (int k = 0; k < 6; k++)
				if (all[k] <= std::max(ntt, 1) && st.ncand < MAXC)
					st.cands[st.ncand++] = all[k];
			if (st.ncand == 0) { st.cands[0] = 1; st.ncand = 1; }
		}
		collect(st);
		decide(st, device, ntt);
		int cand = st.best;
		bool exploring = false;
		for (int c = 0; c < st.ncand; c++)
			if (!st.dead[c] && st.count[c] + (int) std::count_if(st.pending.begin(), st.pending.end(),
					[c](const Pending &p) { return p.cand == c; }) < RING)
			{
				cand = c;
				exploring = true;
				break;
			}
		if (!exploring && st.calls > 0 && st.calls % 256 == 0)
		{
			int oldest = -1;
			long oc = 1L << 60;
			for (int c = 0; c < st.ncand; c++)
				if (!st.dead[c] && c != st.best && st.count[c] < oc) { oldest = c; oc = st.count[c]; }
			if (oldest >= 0) cand = oldest;
		}
		st.calls++;
		cand_out = cand;
		ev = std::make_pair((cudaEvent_t) NULL, (cudaEvent_t) NULL);
		if ((int) st.pending.size() < 64)
		{
			if (!free_events.empty()) { ev = free_events.back(); free_events.pop_back(); }
			else
			{
				cudaEvent_t a = NULL, b = NULL;
				if (cudaEventCreate(&a) == cudaSuccess && cudaEventCreate(&b) == cudaSuccess)
					ev = std::make_pair(a, b);
				else
					cudaGetLastError();
			}
			if (ev.first != NULL)
				cudaEventRecord(ev.first, stream);
		}
		return st.cands[cand];
	}

	// Records the stop event after the launch and queues the pair to be read later
	void finish(int device, int ntt, std::pair<cudaEvent_t, cudaEvent_t> ev, int cand, double work, cudaStream_t stream)
	{
		if (ev.first == NULL) return;
		cudaEventRecord(ev.second, stream);
		std::lock_guard<std::mutex> lock(mtx);
		Pending pd = {ev.first, ev.second, cand, work};
		states[std::make_pair(device, ntt)].pending.push_back(pd);
	}

	void discard(std::pair<cudaEvent_t, cudaEvent_t> ev)
	{
		if (ev.first == NULL) return;
		std::lock_guard<std::mutex> lock(mtx);
		free_events.push_back(ev);
	}

	void mark_dead(int device, int ntt, int tiles)
	{
		std::lock_guard<std::mutex> lock(mtx);
		State &st = states[std::make_pair(device, ntt)];
		for (int c = 0; c < st.ncand; c++)
			if (st.cands[c] == tiles && c > 0) st.dead[c] = true;
		if (st.dead[st.best]) st.best = 0;
		for (size_t i = 0; i < st.pending.size(); )
			if (st.cands[st.pending[i].cand] == tiles) { free_events.push_back(std::make_pair(st.pending[i].t0, st.pending[i].t1)); st.pending.erase(st.pending.begin() + i); }
			else i++;
	}
};
static XtTileTuner &xtTileTuner() { static XtTileTuner t; return t; }
#endif

// Blocks per multiprocessor that pixel chunks aim for. RELION_GPU_BLOCKS_PER_SM=N overrides. By
// default one full wave of 128-thread blocks per multiprocessor (as many as the device keeps
// resident: 12 on sm_86, 16 on sm_80/90, 8 on sm_75), and the old 3 when the GPU has under 6 GiB
// in total, because the chunks need partial-sum buffers. The size test uses the total, not the
// free memory: by the time this runs the allocator has already reserved nearly all of the free
// memory, so a free-memory test always fell back to 3. Chosen once per device and said once.
static size_t gpuBlocksPerSM()
{
	static std::mutex mutex;
	static std::map<int, size_t> chosen;
	int dev = 0;
	DEBUG_HANDLE_ERROR(cudaGetDevice(&dev));
	std::lock_guard<std::mutex> lock(mutex);
	auto it = chosen.find(dev);
	if (it != chosen.end())
		return it->second;
	size_t b = 3;
	std::string why;
	const char *e = getenv("RELION_GPU_BLOCKS_PER_SM");
	const int v = (e != NULL) ? atoi(e) : 0;
	if (v > 0)
	{
		b = (size_t) v;
		why = "RELION_GPU_BLOCKS_PER_SM";
	}
	else
	{
		int threads = 0, sms = 0;
		size_t free_b = 0, total_b = 0;
		cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
		cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
		cudaMemGetInfo(&free_b, &total_b);
		if (threads > 0 && total_b >= ((size_t) 6 << 30))
		{
			b = (size_t) std::max(3, threads / 128);
			why = "device " + std::to_string(dev) + ": " + std::to_string(sms) + " multiprocessors, up to " +
			      std::to_string(threads) + " threads each";
		}
		else
			why = "device " + std::to_string(dev) + " has " + std::to_string(total_b >> 20) + " MiB in total, safe default";
	}
	chosen[dev] = b;
	std::cout << " GPU pixel chunks: " << b << " blocks per multiprocessor (" << why << "; RELION_GPU_BLOCKS_PER_SM=N overrides)." << std::endl;
	return b;
}

// Says once per process and step how many pixel chunks a GPU step uses, and why
static void reportPixelChunks(const char *step, const char *env, int n_chunks, size_t blocks, bool overridden)
{
	static std::mutex mutex;
	static std::set<std::string> reported;
	std::lock_guard<std::mutex> lock(mutex);
	if (!reported.insert(step).second)
		return;
	if (overridden)
		std::cout << " GPU " << step << ": " << n_chunks << " pixel chunks per particle (" << env << ")." << std::endl;
	else
		std::cout << " GPU " << step << ": " << n_chunks << " pixel chunks per particle, for about " << gpuBlocksPerSM() << " blocks per multiprocessor"
		          << " instead of " << blocks << " blocks in all (" << env << "=N to choose, 1 for none)." << std::endl;
}

static int gpuPixelChunks(size_t n_blocks, size_t n_pass, int requested)
{
	if (n_blocks == 0 || n_pass == 0)
		return 1;
	if (requested > 0)
		return (int) std::min<size_t>(requested, n_pass);
	int dev = 0, sms = 0;
	DEBUG_HANDLE_ERROR(cudaGetDevice(&dev));
	DEBUG_HANDLE_ERROR(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
	const size_t target = gpuBlocksPerSM() * (size_t) std::max(sms, 1);
	const size_t chunks = std::min((target + n_blocks - 1) / n_blocks, n_pass / 2);
	return (int) std::max<size_t>(chunks, 1);
}

// Upload the class table and the first block of each class (one block per
// orientation), in one copy; returns the number of blocks. A few dozen bytes per
// class, so batching never needs memory in proportion to the orientations.
template <class Job>
static size_t uploadClassTable(std::vector<Job> &jobs, std::vector<size_t> &nums,
		AccPtr<char> &table, const Job *&d_jobs, const int *&d_start,
		const std::vector<unsigned> *extra = NULL, const unsigned **d_extra = NULL)
{
	std::vector<int> start(jobs.size() + 1, 0);
	for (size_t c = 0; c < jobs.size(); c++)
		start[c + 1] = start[c] + (int) nums[c];
	const size_t class_bytes = ((jobs.size() * sizeof(Job) + 15) / 16) * 16;
	const size_t extra_pos = ((class_bytes + start.size() * sizeof(int) + 15) / 16) * 16;
	const size_t extra_bytes = extra ? extra->size() * sizeof(unsigned) : 0;
	table.setSize(extra_pos + extra_bytes);
	table.allAlloc();
	memcpy(&table[0], jobs.data(), jobs.size() * sizeof(Job));
	memcpy(&table[class_bytes], start.data(), start.size() * sizeof(int));
	if (extra_bytes)
		memcpy(&table[extra_pos], extra->data(), extra_bytes);
	table.cpToDevice();
	d_jobs = (const Job *) ~table;
	d_start = (const int *) &(~table)[class_bytes];
	if (d_extra)
		*d_extra = extra_bytes ? (const unsigned *) &(~table)[extra_pos] : NULL;
	return start.back();
}

void mapAllWeightsToMweightsBatched(
		std::vector<MapClassJob> &jobs,
		XFLOAT * d_mweights,
		unsigned long translation_num,
		AccPtr<char> &table)
{
	if (jobs.empty())
		return;
	std::vector<size_t> blocks(jobs.size());
	for (size_t c = 0; c < jobs.size(); c++)
		blocks[c] = (jobs[c].n * translation_num + WEIGHT_MAP_BLOCK_SIZE - 1) / WEIGHT_MAP_BLOCK_SIZE;
	const MapClassJob *d_jobs;
	const int *d_start;
	const size_t n = uploadClassTable(jobs, blocks, table, d_jobs, d_start);
	cuda_kernel_allweights_to_mweights_batched<<< n, WEIGHT_MAP_BLOCK_SIZE, 0, table.getStream() >>>(
			d_jobs, d_start, (int) jobs.size(), d_mweights, translation_num);
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
}

void runWavgKernelBatched(
		std::vector<WavgClassJob> &jobs,
		std::vector<size_t> &orientation_nums,
		XFLOAT *Fimg_real,
		XFLOAT *Fimg_imag,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT *ctfs,
		XFLOAT *wdiff2s_parts,
		OptimisationParamters &op,
		long unsigned translation_num,
		unsigned long image_size,
		XFLOAT part_scale,
		bool refs_are_ctf_corrected,
		bool data_is_3D,
		AccPtr<char> &table)
{
	if (jobs.empty())
		return;
	const WavgClassJob *d_jobs;
	const int *d_start;
	const size_t n = uploadClassTable(jobs, orientation_nums, table, d_jobs, d_start);
	const int nc = jobs.size();
	deviceStream_t stream = table.getStream();
	AccProjectorKernel &projector = jobs[0].projector;
	// Pixel chunks (see cuda_kernel_wavg)
	const unsigned wavg_block = data_is_3D ? WAVG_BLOCK_SIZE_DATA3D : WAVG_BLOCK_SIZE;
	const int wavg_requested = envChunkOverride("RELION_GPU_WAVG_SPLIT");
	const int wavg_chunks = gpuPixelChunks(n, (image_size + wavg_block - 1) / wavg_block, wavg_requested);
	reportPixelChunks("weighted averaging", "RELION_GPU_WAVG_SPLIT", wavg_chunks, n, wavg_requested > 0);
#ifndef ACC_DOUBLE_PRECISION
	// RELION_GPU_WAVG_SUM=1: 2D data, up to 1024 translations, table up to 64 MB
	if (gpuWavgSum() && !data_is_3D && translation_num > 0 && translation_num <= 1024 &&
		image_size * translation_num * sizeof(float2) <= ((size_t) 64 << 20))
	{
		AccPtr<float2> shifts = table.make<float2>();
		shifts.setSize(translation_num * image_size);
		shifts.deviceAlloc();
		cuda_kernel_shift_table<<<dim3((image_size + 127) / 128, translation_num), 128, 0, stream>>>(
				trans_x, trans_y, ~shifts, image_size, projector.imgX, projector.imgY, projector.maxR);
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
		const size_t smem = (16 + 2 * translation_num) * sizeof(XFLOAT);
#define WAVG_SUM(RC, R3) cuda_kernel_wavg_sum<RC, R3, WAVG_BLOCK_SIZE><<<dim3(n, wavg_chunks), WAVG_BLOCK_SIZE, smem, stream>>>( \
		image_size, Fimg_real, Fimg_imag, ctfs, wdiff2s_parts, translation_num, (XFLOAT) op.sum_weight, \
		(XFLOAT) op.significant_weight, part_scale, ~shifts, d_jobs, d_start, nc, wavg_chunks)
		if (refs_are_ctf_corrected) { if (projector.mdlZ != 0) WAVG_SUM(true, true);  else WAVG_SUM(true, false); }
		else                        { if (projector.mdlZ != 0) WAVG_SUM(false, true); else WAVG_SUM(false, false); }
#undef WAVG_SUM
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
		return;
	}
#endif
	// The same kernel variants as runWavgKernel
	// Room for the list of significant translations in shared memory (see cuda_kernel_wavg)
	const bool sig_list = translation_num * sizeof(int) <= 16384;
	const size_t sig_bytes = sig_list ? translation_num * sizeof(int) : 0;
#define WAVG_BATCH(RC, R3, D3, BS) cuda_kernel_wavg<RC, R3, D3, BS><<<dim3(n, wavg_chunks), BS, (3*BS+9)*sizeof(XFLOAT) + sig_bytes, stream>>>( \
		NULL, projector, image_size, 0, Fimg_real, Fimg_imag, trans_x, trans_y, trans_z, NULL, ctfs, \
		wdiff2s_parts, NULL, NULL, translation_num, (XFLOAT) op.sum_weight, (XFLOAT) op.significant_weight, \
		part_scale, d_jobs, d_start, nc, wavg_chunks, sig_list)
	if (refs_are_ctf_corrected)
	{
		if (data_is_3D)                WAVG_BATCH(true, true, true, WAVG_BLOCK_SIZE_DATA3D);
		else if (projector.mdlZ != 0)  WAVG_BATCH(true, true, false, WAVG_BLOCK_SIZE);
		else                           WAVG_BATCH(true, false, false, WAVG_BLOCK_SIZE);
	}
	else
	{
		if (data_is_3D)                WAVG_BATCH(false, true, true, WAVG_BLOCK_SIZE_DATA3D);
		else if (projector.mdlZ != 0)  WAVG_BATCH(false, true, false, WAVG_BLOCK_SIZE);
		else                           WAVG_BATCH(false, false, false, WAVG_BLOCK_SIZE);
	}
#undef WAVG_BATCH
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
}

void runDiff2KernelFineBatched(
		std::vector<FineClassJob> &jobs,
		std::vector<size_t> &job_nums,
		XFLOAT *corr_img,
		XFLOAT *Fimgs_real,
		XFLOAT *Fimgs_imag,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		unsigned long image_size,
		XFLOAT sum_init,
		long unsigned translation_num,
		bool data_is_3D,
		AccPtr<char> &table)
{
	if (jobs.empty())
		return;
	const FineClassJob *d_jobs;
	const int *d_start;
	const size_t n = uploadClassTable(jobs, job_nums, table, d_jobs, d_start);
	const int nc = jobs.size();
	if (n == 0)
		return;
	deviceStream_t stream = table.getStream();
	AccProjectorKernel &projector = jobs[0].projector;
	// The same kernel variants as runDiff2KernelFine (not do_CC)
	// 2D references: pixel chunks, like the coarse pass (RELION_GPU_FINE_SPLIT=auto chooses the number, =N sets it; off unless
	// asked for). Each chunk's partial sums go to a buffer and a second kernel adds them in chunk order, so the result is the same
	// from run to run (but not the order of the additions of the unchunked kernel: last-digit differences against it).
	static const char *fine_env = getenv("RELION_GPU_FINE_SPLIT");
	const int fine_requested = envChunkOverride("RELION_GPU_FINE_SPLIT");
	const bool fine_on = fine_env != NULL && (fine_requested > 1 || !strcmp(fine_env, "auto"));
	const int fine_chunks = (!fine_on || data_is_3D || projector.mdlZ != 0) ? 1 : gpuPixelChunks(n, (image_size + D2F_BLOCK_SIZE_2D - 1) / D2F_BLOCK_SIZE_2D, fine_requested);
	AccPtr<XFLOAT> fine_partial = table.make<XFLOAT>();
	if (fine_chunks > 1)
	{
		fine_partial.setSize((size_t) fine_chunks * n * D2F_CHUNK_2D);
		fine_partial.deviceAlloc();
	}
#define D2F_BATCH(R3, D3, BS, CH) cuda_kernel_diff2_fine<R3, D3, BS, CH><<<dim3(n, (R3 || D3) ? 1 : fine_chunks), BS, 0, stream>>>( \
		NULL, Fimgs_real, Fimgs_imag, trans_x, trans_y, trans_z, projector, corr_img, NULL, \
		image_size, sum_init, 0, translation_num, 0, NULL, NULL, NULL, NULL, d_jobs, d_start, nc, (R3 || D3) ? 1 : fine_chunks, \
		fine_chunks > 1 ? ~fine_partial : (XFLOAT *) NULL)
	if (data_is_3D)               D2F_BATCH(true, true, D2F_BLOCK_SIZE_DATA3D, D2F_CHUNK_DATA3D);
	else if (projector.mdlZ != 0) D2F_BATCH(true, false, D2F_BLOCK_SIZE_REF3D, D2F_CHUNK_REF3D);
	else                          D2F_BATCH(false, false, D2F_BLOCK_SIZE_2D, D2F_CHUNK_2D);
#undef D2F_BATCH
	if (fine_chunks > 1)
		cuda_kernel_diff2_fine_sum_chunks<D2F_CHUNK_2D><<<(unsigned) n, 32, 0, stream>>>(d_jobs, d_start, nc, ~fine_partial, fine_chunks, sum_init);
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
}

void runCollect2jobsBatched(
		std::vector<CollectClassJob> &jobs,
		std::vector<size_t> &block_nums,
		XFLOAT significant_weight,
		XFLOAT sum_weight,
		unsigned long nr_trans,
		unsigned long nr_oversampled_trans,
		unsigned long nr_oversampled_rot,
		unsigned long oversamples,
		bool skip_rots,
		XFLOAT * p_weights,
		XFLOAT * p_thr_wsum_prior_offsetx_class,
		XFLOAT * p_thr_wsum_prior_offsety_class,
		XFLOAT * p_thr_wsum_prior_offsetz_class,
		XFLOAT * p_thr_wsum_sigma2_offset,
		bool data_is_3D,
		AccPtr<char> &table)
{
	if (jobs.empty())
		return;
	const CollectClassJob *d_jobs;
	const int *d_start;
	const size_t n = uploadClassTable(jobs, block_nums, table, d_jobs, d_start);
	const int nc = jobs.size();
	if (n == 0)
		return;
	size_t shared_buffer = sizeof(XFLOAT)*SUMW_BLOCK_SIZE*5; // x+y+z+myp+weights
#define C2J_BATCH(D3) cuda_kernel_collect2jobs<D3><<<n, SUMW_BLOCK_SIZE, shared_buffer, table.getStream()>>>( \
		NULL, NULL, NULL, NULL, NULL, significant_weight, sum_weight, nr_trans, nr_oversampled_trans, \
		nr_oversampled_rot, oversamples, skip_rots, p_weights, p_thr_wsum_prior_offsetx_class, \
		p_thr_wsum_prior_offsety_class, p_thr_wsum_prior_offsetz_class, p_thr_wsum_sigma2_offset, \
		NULL, NULL, NULL, NULL, d_jobs, d_start, nc)
	if (data_is_3D) C2J_BATCH(true);
	else            C2J_BATCH(false);
#undef C2J_BATCH
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
}

// Gradient (VDAM) 2D backprojection of all classes in one launch: the kernel of
// runBackProjectKernel with do_grad, one block per class and orientation. A block does
// what it did in a per-class launch.
void runBackProjectSgdKernelBatched(
		AccBackprojector &BP,
		std::vector<BPSgdClassJob> &jobs,
		std::vector<size_t> &orientation_nums,
		XFLOAT *d_img_real,
		XFLOAT *d_img_imag,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT* d_Minvsigma2s,
		XFLOAT* d_ctfs,
		unsigned long translation_num,
		XFLOAT significant_weight,
		int imgX,
		int imgY,
		bool ctf_premultiplied,
		AccPtr<char> &table)
{
	if (jobs.empty())
		return;
	const BPSgdClassJob *d_jobs;
	const int *d_start;
	const size_t n = uploadClassTable(jobs, orientation_nums, table, d_jobs, d_start);
	const int nc = jobs.size();
	deviceStream_t stream = table.getStream();
#ifndef ACC_DOUBLE_PRECISION
	// RELION_GPU_WAVG_SUM=1 also for the back-projection: see cuda_kernel_backproject2D_SGD_sum
	if (gpuWavgSum() && (imgY % 2) == 0 && BP.maxR == imgY / 2 && translation_num > 0 && translation_num <= 1024 &&
		(size_t) imgX * imgY * translation_num * sizeof(float2) <= ((size_t) 64 << 20))
	{
		AccPtr<float2> shifts = table.make<float2>();
		shifts.setSize(translation_num * imgX * imgY);
		shifts.deviceAlloc();
		cuda_kernel_shift_table<<<dim3((imgX * imgY + 127) / 128, translation_num), 128, 0, stream>>>(
				trans_x, trans_y, ~shifts, imgX * imgY, imgX, imgY, BP.maxR);
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
		const size_t smem = (16 + 2 * translation_num) * sizeof(XFLOAT);
#define BP2DSGD_SUM(CP) cuda_kernel_backproject2D_SGD_sum<CP><<<n, BP_2D_BLOCK_SIZE, smem, stream>>>( \
		jobs[0].projector, d_img_real, d_img_imag, d_Minvsigma2s, d_ctfs, translation_num, significant_weight, \
		BP.maxR2, BP.padding_factor, imgX, imgY, imgX*imgY, BP.mdlX, BP.mdlInitY, ~shifts, d_jobs, d_start, nc)
		if (ctf_premultiplied) BP2DSGD_SUM(true);
		else                   BP2DSGD_SUM(false);
#undef BP2DSGD_SUM
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
		return;
	}
#endif
#define BP2DSGD_BATCH(CP) cuda_kernel_backproject2D_SGD<CP><<<n, BP_2D_BLOCK_SIZE, 0, stream>>>( \
		jobs[0].projector, d_img_real, d_img_imag, trans_x, trans_y, NULL, d_Minvsigma2s, d_ctfs, \
		translation_num, significant_weight, (XFLOAT) 0, NULL, NULL, NULL, NULL, \
		BP.maxR, BP.maxR2, BP.padding_factor, imgX, imgY, imgX*imgY, BP.mdlX, BP.mdlInitY, \
		d_jobs, d_start, nc)
	if (ctf_premultiplied) BP2DSGD_BATCH(true);
	else                   BP2DSGD_BATCH(false);
#undef BP2DSGD_BATCH
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
}

void runBackProjectKernelBatched(
		AccBackprojector &BP,
		std::vector<BPClassJob> &jobs,
		std::vector<size_t> &orientation_nums,
		XFLOAT *d_img_real,
		XFLOAT *d_img_imag,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT* d_Minvsigma2s,
		XFLOAT* d_ctfs,
		unsigned long translation_num,
		XFLOAT significant_weight,
		int imgX,
		int imgY,
		int imgZ,
		bool data_is_3D,
		bool ctf_premultiplied,
		AccPtr<char> &table,
		const std::vector<unsigned> *slots,
		const BPAccumView *accum)
{
	if (jobs.empty())
		return;
	// The orientation stride of the scatter order, once per class on the host instead of a gcd loop in
	// every block (the same values as the kernel's own; RELION_GPU_BP_SCATTER_HOST=0 leaves it to the kernel)
	static const bool stride_on_host = gpuswitch::on("RELION_GPU_BP_SCATTER_HOST");
	if (stride_on_host)
		for (size_t c = 0; c < jobs.size() && c < orientation_nums.size(); c++)
		{
			const unsigned long long cnt = orientation_nums[c];
			unsigned long long stride = (unsigned long long) (cnt * 0.6180339887) | 1;
			auto gcd = [](unsigned long long a, unsigned long long b) { while (b) { unsigned long long t = a % b; a = b; b = t; } return a; };
			while (stride < cnt && gcd(stride, cnt) != 1)
				stride += 2;
			jobs[c].scatter_stride = (unsigned) stride;
		}
	const BPClassJob *d_jobs;
	const int *d_start;
	const unsigned *d_slots = NULL;
	const BPAccumView acc = (slots && accum && !data_is_3D && BP.mdlZ != 1) ? *accum : BPAccumView();
	const size_t n = uploadClassTable(jobs, orientation_nums, table, d_jobs, d_start,
			acc.cap > 0 ? slots : NULL, &d_slots);
	const int nc = jobs.size();
	deviceStream_t stream = table.getStream();
	// Pixel chunks for the 3D kernels (see cuda_kernel_backproject3D)
	const unsigned bp_block = data_is_3D ? BP_DATA3D_BLOCK_SIZE : BP_REF3D_BLOCK_SIZE;
	const int bp_requested = envChunkOverride("RELION_GPU_BP_SPLIT");
	const int bp_chunks = gpuPixelChunks(n, ((size_t) imgX * imgY * imgZ + bp_block - 1) / bp_block, bp_requested);
	if (BP.mdlZ != 1)
		reportPixelChunks("back-projection", "RELION_GPU_BP_SPLIT", bp_chunks, n, bp_requested > 0);
	// The same kernel variants as runBackProjectKernel (without do_grad); BP gives the
	// dimensions shared by all classes' models, jobs their arrays
	if (BP.mdlZ == 1)
	{
		// The significant translations listed once per block (see cuda_kernel_backproject2D)
		const size_t smem = translation_num * (sizeof(XFLOAT) + sizeof(int));
		const bool compact = smem <= 40960;
		// Pixel chunks for the 2D kernel too (RELION_GPU_BP_SPLIT=N to choose, 1 for none): its blocks are one per
		// significant orientation, a handful per particle, each looping over all the pixel passes
		const int bp2d_chunks = gpuPixelChunks(n, ((size_t) imgX * imgY + BP_2D_BLOCK_SIZE - 1) / BP_2D_BLOCK_SIZE, bp_requested);
#define BP2D_BATCH(CP) cuda_kernel_backproject2D<CP><<<dim3(n, bp2d_chunks), BP_2D_BLOCK_SIZE, compact ? smem : 0, stream>>>( \
			d_img_real, d_img_imag, trans_x, trans_y, NULL, d_Minvsigma2s, d_ctfs, \
			translation_num, significant_weight, (XFLOAT) 0, NULL, NULL, NULL, NULL, \
			BP.maxR, BP.maxR2, BP.padding_factor, imgX, imgY, imgX*imgY, BP.mdlX, BP.mdlInitY, \
			d_jobs, d_start, nc, compact, bp2d_chunks)
		if (ctf_premultiplied) BP2D_BATCH(true);
		else                   BP2D_BATCH(false);
#undef BP2D_BATCH
	}
	else
	{
		const size_t smem3 = translation_num * (sizeof(XFLOAT) + sizeof(int));
		const bool compact3 = smem3 <= 40960;
		static const bool bp_scatter = [] { const char *e = getenv("RELION_GPU_BP_SCATTER"); return !(e && atoi(e) == 0); }();
		// RELION_GPU_BP_TILE=1: a warp takes a tile of 16 x 2 pixels (2D data) so that its atomic additions
		// are denser; =check also runs the default path on the first RELION_GPU_BP_TILE_CHECKS (2) launches
		// and reports how far the two models differ (one class, no accumulation); off by default.
		// The check saves and restores the whole model, so it is only valid with --j 1 (threads share the model)
		static const int bp_tile_mode = [] {
			const char *e = getenv("RELION_GPU_BP_TILE");
			if (!e || !strcmp(e, "0") || !strcmp(e, "off")) return 0;
			return strcmp(e, "check") ? 1 : 2;
		}();
		// RELION_GPU_BP_PACK: the atomic additions of a warp are packed into runs of neighbouring floats
		// (see cuda_kernel_backproject3D); same terms, order of the additions only; on unless =0
		static const bool bp_pack = [] { const char *e = getenv("RELION_GPU_BP_PACK"); return !(e && atoi(e) == 0); }();
		// The staging buffer of the packed path follows the compacted translations list in shared memory (at most
		// 1024 translations are supported in single precision, so both fit in 48 KB; the check below is a safety net)
		// RELION_GPU_BP_PACK_RUNS=1|2|4: runs staged at a time by the packed path (default 4, as before); 2 halves the
		// shared memory per block, which lifts the occupancy
		static const int bp_pack_runs = [] { const char *e = getenv("RELION_GPU_BP_PACK_RUNS"); int v = e ? atoi(e) : BP_PACK_RUNS; return (v == 1 || v == 2) ? v : 4; }();
		const size_t pk_bytes = (size_t) (BP_REF3D_BLOCK_SIZE / 32) * (32 * 6 * bp_pack_runs * sizeof(XFLOAT) + 32 * bp_pack_runs * sizeof(size_t));
		int pack_state = 1;
		const size_t pk_off = compact3 ? ((smem3 + 15) & ~(size_t) 15) : 0;
		const bool pack_ok = bp_pack && !data_is_3D && BP.interleaved && bp_tile_mode == 0 && pk_off + pk_bytes <= 48 * 1024;
		if (bp_pack)
		{
			if (!pack_ok)
				pack_state = data_is_3D ? 2 : !BP.interleaved ? 3 : bp_tile_mode != 0 ? 4 : 5;
			static const char *const notes[] = {"",
				"on (default; RELION_GPU_BP_PACK=0 turns it off)",
				"not used: 3D data",
				"not used: planar model layout",
				"not used: RELION_GPU_BP_TILE is set",
				"not used: shared memory"};
			static std::atomic<unsigned> reported(0);
			const unsigned bit = 1u << pack_state;
			if (!(reported.fetch_or(bit) & bit))
				fprintf(stderr, "RELION_GPU_BP_PACK: %s\n", notes[pack_state]);
		}
		// RELION_GPU_BP_SHIFT=1: 2D data, up to 1024 translations, table up to 64 MB. The table numbers the rows beyond maxR
		// differently from this kernel (see cuda_kernel_shift_table), but those pixels lie outside the sphere of maxR2 = maxR^2
		// and add nothing, so any even-sized image with maxR <= imgY / 2 can use it
		AccPtr<float2> bp_shifts = table.make<float2>();
		bool use_shift = false;
#ifndef ACC_DOUBLE_PRECISION
		if (gpuBpShift() && !data_is_3D && (imgY % 2) == 0 && BP.maxR <= imgY / 2 && (int) BP.maxR2 == BP.maxR * BP.maxR && translation_num > 0 &&
			translation_num <= 1024 && (size_t) imgX * imgY * translation_num * sizeof(float2) <= ((size_t) 64 << 20))
		{
			bp_shifts.setSize(translation_num * imgX * imgY);
			bp_shifts.deviceAlloc();
			cuda_kernel_shift_table<<<dim3((imgX * imgY + 127) / 128, translation_num), 128, 0, stream>>>(
					trans_x, trans_y, ~bp_shifts, imgX * imgY, imgX, imgY, BP.maxR);
			LAUNCH_HANDLE_ERROR(cudaGetLastError());
			use_shift = true;
		}
#endif
		// RELION_GPU_BP_DIVHOIST=1: with the compacted translation list, weight / weight_norm is made once per translation
		// when the list is built (the same value as the division in the pixel loop); 2D data
		static const bool bp_divhoist = [] { const char *e = getenv("RELION_GPU_BP_DIVHOIST"); return e && atoi(e) != 0; }();
		const bool hoist_div = bp_divhoist && compact3 && !data_is_3D;
		auto launch_bp = [&](int tl) {
#define BP3D_BATCH(D3, CP, BS) BP3D_BATCH_S(D3, CP, false, BS)
#define BP3D_BATCH_S(D3, CP, SH, BS) BP3D_BATCH_H(D3, CP, SH, false, BS)
#define BP3D_BATCH_H(D3, CP, SH, HD, BS) cuda_kernel_backproject3D<D3, CP, SH, HD><<<dim3(n, bp_chunks), BS, (!D3 && tl == 2) ? pk_off + pk_bytes : (compact3 ? smem3 : 0), stream>>>( \
			d_img_real, d_img_imag, trans_x, trans_y, trans_z, NULL, d_Minvsigma2s, d_ctfs, \
			translation_num, significant_weight, (XFLOAT) 0, NULL, NULL, NULL, NULL, \
			BP.maxR, BP.maxR2, BP.padding_factor, imgX, imgY, imgZ, imgX*imgY*imgZ, \
			BP.mdlX, BP.mdlY, BP.mdlInitY, BP.mdlInitZ, d_jobs, d_start, nc, bp_chunks, compact3, bp_scatter, acc, d_slots, \
			data_is_3D ? 0 : tl, ~bp_shifts, bp_pack_runs)
		if (use_shift)
		{
			if (ctf_premultiplied) BP3D_BATCH_S(false, true, true, BP_REF3D_BLOCK_SIZE);
			else                   BP3D_BATCH_S(false, false, true, BP_REF3D_BLOCK_SIZE);
		}
		else if (hoist_div)
		{
			if (ctf_premultiplied) BP3D_BATCH_H(false, true, false, true, BP_REF3D_BLOCK_SIZE);
			else                   BP3D_BATCH_H(false, false, false, true, BP_REF3D_BLOCK_SIZE);
		}
		else if (data_is_3D)
		{
			if (ctf_premultiplied) BP3D_BATCH(true, true, BP_DATA3D_BLOCK_SIZE);
			else                   BP3D_BATCH(true, false, BP_DATA3D_BLOCK_SIZE);
		}
		else
		{
			if (ctf_premultiplied) BP3D_BATCH(false, true, BP_REF3D_BLOCK_SIZE);
			else                   BP3D_BATCH(false, false, BP_REF3D_BLOCK_SIZE);
		}
#undef BP3D_BATCH
#undef BP3D_BATCH_S
#undef BP3D_BATCH_H
		};
		static int tile_checks_left = [] { const char *e = getenv("RELION_GPU_BP_TILE_CHECKS"); return e ? atoi(e) : 2; }();
		if (bp_tile_mode == 2 && tile_checks_left > 0 && !data_is_3D && nc == 1 && acc.cap == 0 && BP.mdlXYZ > 0)
		{
			tile_checks_left--;
			// the model of the class: one interleaved region of 3 floats per voxel, or three planar ones
			XFLOAT *reg[3] = { jobs[0].model_real, jobs[0].model_imag, jobs[0].model_weight };
			const int nreg = BP.interleaved ? 1 : 3;
			const size_t reg_n = BP.interleaved ? 3 * BP.mdlXYZ : BP.mdlXYZ;
			const size_t total = nreg * reg_n;
			static float *d_save0 = NULL, *d_save1 = NULL;
			static size_t save_cap = 0;
			if (save_cap < total)
			{
				if (d_save0) { cudaFree(d_save0); cudaFree(d_save1); }
				cudaMalloc(&d_save0, total * sizeof(float));
				cudaMalloc(&d_save1, total * sizeof(float));
				save_cap = total;
			}
			static unsigned *d_acc = NULL;
			if (!d_acc) cudaMalloc(&d_acc, 4 * sizeof(double));
			cudaMemsetAsync(d_acc, 0, 4 * sizeof(double), stream);
			for (int r = 0; r < nreg; r++) cudaMemcpyAsync(d_save0 + r * reg_n, reg[r], reg_n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
			launch_bp(0);   // the default path
			for (int r = 0; r < nreg; r++) cudaMemcpyAsync(d_save1 + r * reg_n, reg[r], reg_n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
			for (int r = 0; r < nreg; r++) cudaMemcpyAsync(reg[r], d_save0 + r * reg_n, reg_n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
			launch_bp(1);   // the tiled path, from the same start (its result stays in the model)
			unsigned *d_max = d_acc;            // [0] max |d|, [1] max |r|
			double *d_sum = (double *) (d_acc + 2); // [0] sum d^2, [1] sum r^2
			for (int r = 0; r < nreg; r++)
				cuda_kernel_bp_check_diff<<<1024, 256, 0, stream>>>(d_save0 + r * reg_n, d_save1 + r * reg_n, reg[r], reg_n, d_max, d_max + 1, d_sum, d_sum + 1);
			unsigned h_max[2]; double h_sum[2];
			cudaMemcpyAsync(h_max, d_max, sizeof(h_max), cudaMemcpyDeviceToHost, stream);
			cudaMemcpyAsync(h_sum, d_sum, sizeof(h_sum), cudaMemcpyDeviceToHost, stream);
			cudaStreamSynchronize(stream);
			float md, mr;
			memcpy(&md, &h_max[0], 4); memcpy(&mr, &h_max[1], 4);
			fprintf(stderr, "BP_TILE_CHECK: model update of this launch, max |tiled - default| %.3g, max |update| %.3g (ratio %.2e), rms ratio %.2e\n",
					md, mr, mr > 0 ? md / mr : 0.f, h_sum[1] > 0 ? sqrt(h_sum[0] / h_sum[1]) : 0.0);
		}
		else
			launch_bp(pack_ok ? 2 : (bp_tile_mode == 1 ? 1 : 0));
	}
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
}

// Spreads the per-orientation accumulation into the class volumes (see BPAccumView)
void runBackProjectAccumFlush(
		AccBackprojector &BP,
		const BPAccumView &acc,
		const BPModelPtrs *d_models,
		unsigned slots_per_class,
		unsigned img_x,
		unsigned img_y,
		int used)
{
	cuda_kernel_backproject3D_flush<<<used, 128>>>(
			acc, d_models, slots_per_class, img_x, img_y, BP.maxR2, BP.padding_factor,
			BP.mdlX, BP.mdlY, BP.mdlInitY, BP.mdlInitZ);
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
}
#endif

#ifdef _CUDA_ENABLED
// Pixel chunks of the class-batched coarse pass (see runDiff2KernelCoarseBatched).
// RELION_GPU_COARSE_SPLIT=N sets the number of chunks (1: no chunks, as before);
// unset, it is chosen for about 3 blocks per SM of the current device.
static int coarseChunkOverride()
{
	static const int n = [] { const char *e = getenv("RELION_GPU_COARSE_SPLIT"); return e ? atoi(e) : 0; }();
	return n;
}

// RELION_GPU_COARSE_DETERMINISTIC=1: always sum the coarse differences through the chunk buffer
// (one chunk when the pass is not split), never with atomics, so the order of the sums is fixed
// and a particle comes out bit for bit the same alone or in a group (RELION_GPU_PARTICLE_GROUP)
static bool coarseDeterministic()
{
	static const bool d = [] { const char *e = getenv("RELION_GPU_COARSE_DETERMINISTIC"); return e && atoi(e) != 0; }();
	return d;
}

static size_t coarseChunkTargetBlocks()
{
	int dev = 0, sms = 0;
	DEBUG_HANDLE_ERROR(cudaGetDevice(&dev));
	DEBUG_HANDLE_ERROR(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
	return gpuBlocksPerSM() * (size_t) std::max(sms, 1);
}


// Class batching of the coarse difference kernel: the two launches (whole blocks of
// eulers_per_block orientations, then the rest one per block) for all classes at once
template<bool REF3D, bool DATA3D, int block_sz, int eulers_per_block, int prefetch_fraction>
static void launchDiff2CoarseBatched(
		size_t n_even, size_t n_rest, int n_classes,
		const CoarseClassJob *d_classes, const int *d_even, const int *d_rest,
		AccProjectorKernel &projector,
		XFLOAT *trans_x, XFLOAT *trans_y, XFLOAT *trans_z,
		XFLOAT *corr_img, XFLOAT *Fimg_real, XFLOAT *Fimg_imag,
		unsigned long translation_num, unsigned long image_size,
		deviceStream_t stream,
		XFLOAT *d_partial = NULL, int n_chunks = 1, size_t chunk_stride = 0, size_t max_orient = 0,
		const float2 *d_shift = NULL)
{
	if (d_partial != NULL && n_even && !n_rest)
	{
		// Pixel chunks (see cuda_kernel_diff2_coarse): n_even x n_chunks blocks, then
		// the chunks added in order, one row of blocks per class
		cuda_kernel_diff2_coarse<REF3D, DATA3D, block_sz, eulers_per_block, prefetch_fraction>
			<<<dim3(n_even, n_chunks), block_sz, 0, stream>>>(
				NULL, trans_x, trans_y, trans_z, Fimg_real, Fimg_imag, projector, corr_img, NULL,
				translation_num, image_size, d_classes, d_even, n_classes, false,
				d_partial, n_chunks, chunk_stride, d_shift);
		const size_t n = max_orient * translation_num;
		const unsigned sum_blocks = (unsigned) std::min<size_t>((n + 255) / 256, 1024);
		cuda_kernel_diff2_coarse_sum_chunks<<<dim3(sum_blocks, n_classes), 256, 0, stream>>>(
				d_classes, d_partial, n_chunks, chunk_stride, (int) translation_num);
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
		return;
	}
	if (n_even)
		AccUtilities::diff2_coarse<REF3D, DATA3D, block_sz, eulers_per_block, prefetch_fraction>(
			n_even, block_sz, NULL, trans_x, trans_y, trans_z, Fimg_real, Fimg_imag,
			projector, corr_img, NULL, translation_num, image_size, stream, d_classes, d_even, n_classes, false, d_shift);
	if (n_rest)
		AccUtilities::diff2_coarse<REF3D, DATA3D, block_sz, 1, prefetch_fraction>(
			n_rest, block_sz, NULL, trans_x, trans_y, trans_z, Fimg_real, Fimg_imag,
			projector, corr_img, NULL, translation_num, image_size, stream, d_classes, d_rest, n_classes, true);
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
}

#ifndef ACC_DOUBLE_PRECISION
#include "src/acc/cuda/cuda_kernels/coarse_cache.cuh"

static CoarseShapeTuner &coarseShapeTuner() { static CoarseShapeTuner t; return t; }

// The cache for the classes of this pass: built on the first call of the iteration, then reused. Returns whether it can
// serve this pass (same classes, orientations and pixel mask as it was built for).
static bool coarseCacheReady(
		CoarseCacheCtx &ctx,
		std::vector<CoarseClassJob> &classes,
		const std::vector<size_t> &nums,
		int image_size,
		unsigned long translation_num,
		deviceStream_t stream)
{
	CoarseProjectionCache &cc = *ctx.cache;
	{
		std::lock_guard<std::mutex> lock(cc.mtx);
		if (!cc.tried)
		{
			cc.tried = true;
			cc.allocator = ctx.allocator;
			std::vector<int> pix;
			cc.mask.assign(image_size, 0);
			for (int p = 0; p < image_size; p++)
				if (ctx.host_minvsigma2[p] != 0)
				{
					cc.mask[p] = 1;
					pix.push_back(p);
				}
			size_t O_total = 0;
			for (size_t n : nums)
				O_total += n;
			const size_t K = pix.size();
			size_t bytes = K * O_total * sizeof(float2) + K * sizeof(int) + 1024;
			size_t budget = std::min<size_t>((size_t) 4 << 30, ctx.allocator->getPotentialFreeSpace() / 4);
			const char *mb = getenv("RELION_GPU_COARSE_CACHE_MB");
			if (mb != NULL && atoi(mb) > 0)
				budget = (size_t) atoi(mb) << 20;
			if (K == 0 || O_total == 0 || O_total >= ((size_t) 1 << 31) || K >= ((size_t) 1 << 24) || bytes > budget)
			{
				if (ctx.verb > 0)
					std::cout << " GPU coarse pass: the projection cache would need " << (bytes >> 20) << " MB and may use " << (budget >> 20)
					          << " MB (RELION_GPU_COARSE_CACHE_MB), using the cross-term kernel." << std::endl;
				return false;
			}
			cc.image_size = image_size;
			cc.K = (int) K;
			cc.O_total = O_total;
			cc.bytes = bytes;
			cc.block = ctx.allocator->alloc(bytes);
			char *base = (char *) cc.block->getPtr();
			cc.d_pix = (int *) base;
			cc.d_R = (float2 *) (base + ((K * sizeof(int) + 255) / 256) * 256);
			DEBUG_HANDLE_ERROR(cudaMemcpyAsync(cc.d_pix, pix.data(), K * sizeof(int), cudaMemcpyHostToDevice, stream));
			size_t off = 0;
			for (size_t c = 0; c < classes.size(); c++)
			{
				cc.class_ids.push_back(ctx.class_ids[c]);
				cc.eulers.push_back(classes[c].eulers);
				cc.n_orient.push_back(nums[c]);
				cc.offset.push_back(off);
				cuda_kernel_cc_build<<<(unsigned) ((nums[c] + 127) / 128), 128, 0, stream>>>(
						classes[c].projector, classes[c].eulers, cc.d_pix, (int) K, (int) nums[c], O_total, off, cc.d_R);
				LAUNCH_HANDLE_ERROR(cudaGetLastError());
				off += nums[c];
			}
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
			cc.ok = true;
			if (ctx.verb > 0)
				std::cout << " GPU coarse pass: projection cache of " << K << " pixels x " << O_total << " orientations ("
				          << (bytes >> 20) << " MB), one pass over it per particle (RELION_GPU_COARSE_CACHE=0 for the cross-term kernel)." << std::endl;
		}
		if (!cc.ok)
			return false;
	}
	if (cc.image_size != image_size || cc.class_ids != ctx.class_ids || classes.size() != cc.eulers.size())
		return false;
	for (size_t c = 0; c < classes.size(); c++)
	{
		if (cc.eulers[c] != classes[c].eulers || cc.n_orient[c] != nums[c])
			return false;
		if (classes[c].diff2s != classes[0].diff2s + cc.offset[c] * translation_num)
			return false;
	}
	for (int p = 0; p < image_size; p++)
		if (ctx.host_minvsigma2[p] != 0 && !cc.mask[p])
			return false;
	return true;
}

// One particle against the cache (see cuda_kernel_cc_score). `diff2s` of the classes are consecutive in memory.
static void coarseCacheScore(
		CoarseCacheCtx &ctx,
		std::vector<CoarseClassJob> &classes,
		XFLOAT *trans_x, XFLOAT *trans_y,
		XFLOAT *corr_img, XFLOAT *Fimg_real, XFLOAT *Fimg_imag,
		unsigned long translation_num, unsigned long image_size,
		AccPtr<char> &table)
{
	CoarseProjectionCache &cc = *ctx.cache;
	deviceStream_t stream = table.getStream();
	const int T = (int) translation_num;
	const int tiles = (T + 31) / 32;
	const int TT = (((T + tiles - 1) / tiles + 3) / 4) * 4;
	const int tpad = tiles * TT;
	AccPtr<float2> A = table.make<float2>();
	A.setSize((size_t) cc.K * tpad);
	A.deviceAlloc();
	AccPtr<float> W = table.make<float>();
	W.setSize(cc.K);
	W.deviceAlloc();
	const size_t n_prep = (size_t) cc.K * tpad;
	cuda_kernel_cc_prepare<<<(unsigned) ((n_prep + 127) / 128), 128, 0, stream>>>(
			trans_x, trans_y, Fimg_real, Fimg_imag, corr_img, cc.d_pix, ~A, ~W,
			cc.K, T, tpad, TT, classes[0].projector.imgX, classes[0].projector.imgY, classes[0].projector.maxR);
	LAUNCH_HANDLE_ERROR(cudaGetLastError());

	static const int forced_tm = [] { const char *e = getenv("RELION_GPU_COARSE_CACHE_TM"); return e ? atoi(e) : 0; }();
	const int cands[2] = {2, 1};
	CoarseShapeTuner &tuner = coarseShapeTuner();
	std::pair<cudaEvent_t, cudaEvent_t> ev((cudaEvent_t) NULL, (cudaEvent_t) NULL);
	int cand = 0;
	int tm = forced_tm > 0 ? (forced_tm >= 2 ? 2 : 1) : 0;
	if (tm == 0)
	{
		cand = tuner.begin(ctx.device, TT, 2, cands, ev);
		tm = cands[cand];
	}
	if (ev.first != NULL)
		cudaEventRecord(ev.first, stream);
	const float2 *R = cc.d_R;
	float *out = classes[0].diff2s;
	const size_t O = cc.O_total;
#define CC_SCORE(TTV, TMV) cuda_kernel_cc_score<TTV, TMV><<<dim3((unsigned) ((O + (size_t) CC_SB * TMV - 1) / ((size_t) CC_SB * TMV)), (unsigned) tiles), CC_SB, 0, stream>>>( \
		R, cc.K, O, ~A, ~W, cc.d_pix, Fimg_real, Fimg_imag, T, out)
#define CC_TT_CASE(TTV) case TTV: if (tm == 2) CC_SCORE(TTV, 2); else CC_SCORE(TTV, 1); break
	switch (TT)
	{
		CC_TT_CASE(4);
		CC_TT_CASE(8);
		CC_TT_CASE(12);
		CC_TT_CASE(16);
		CC_TT_CASE(20);
		CC_TT_CASE(24);
		CC_TT_CASE(28);
		CC_TT_CASE(32);
		default: REPORT_ERROR("BUG: coarseCacheScore: unexpected translation tile");
	}
#undef CC_TT_CASE
#undef CC_SCORE
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		LAUNCH_HANDLE_ERROR(err);
	if (ev.first != NULL)
	{
		cudaEventRecord(ev.second, stream);
		tuner.finish(ctx.device, TT, ev, cand, 1e-3 * (double) O * (double) cc.K * (double) tiles);
	}
}
#endif

bool runDiff2KernelCoarseBatched(
		std::vector<CoarseClassJob> &classes,
		std::vector<size_t> &orientation_nums,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT *corr_img,
		XFLOAT *Fimg_real,
		XFLOAT *Fimg_imag,
		unsigned long translation_num,
		unsigned long image_size,
		bool data_is_3D,
		AccPtr<char> &table,
		CoarseDeferred *defer,
		CoarseCacheCtx *cc)
{
	if (classes.empty())
		return true;
	AccProjectorKernel &projector = classes[0].projector;
	const bool ref3D = projector.mdlZ != 0;

	// Per class: the orientations up to a multiple of `split` in blocks of `per_block`,
	// the rest one per block. runDiff2KernelCoarse splits 3D references at a multiple of
	// the block size (128), which sends up to 127 orientations per class one per block;
	// a block only needs whole blocks of per_block orientations, so split there instead.
	int per_block, split, mult = 1;
	if (ref3D)
	{
		const unsigned long blocks3D = (data_is_3D ? D2C_BLOCK_SIZE_DATA3D : D2C_BLOCK_SIZE_REF3D);
#ifdef ACC_DOUBLE_PRECISION
		const int max_mult = 4;
#else
		const int max_mult = 8;
#endif
		while (mult < max_mult && translation_num > blocks3D * mult)
			mult *= 2;
		if (translation_num > blocks3D * mult)
			CRITICAL(ERR_TRANSLIM);
		per_block = data_is_3D ? D2C_EULERS_PER_BLOCK_DATA3D : D2C_EULERS_PER_BLOCK_REF3D;
		split = per_block;
	}
	else
	{
		if (translation_num > D2C_BLOCK_SIZE_2D)
		{
			printf("Number of coarse translations larger than %d on the GPU not supported.\n", D2C_BLOCK_SIZE_2D);
			fflush(stdout);
			exit(1);
		}
		split = per_block = D2C_EULERS_PER_BLOCK_2D;
	}

	// Per class: its first block. The last block of a class takes the rest of its
	// orientations (fewer than per_block), so one launch covers everything: a block's
	// time hardly depends on its number of orientations (the loop over the image
	// pixels sets it), and a second launch for the rest one per block cost about as
	// much as half the first.
	const size_t nc = classes.size();
	std::vector<int> even(nc + 1, 0), rest(nc + 1, 0);
	size_t n_orient_all = 0, max_orient = 0;
	for (size_t c = 0; c < nc; c++)
	{
		const size_t n = orientation_nums[c];
		classes[c].n_even = (int) n;
		classes[c].n_orient = (int) n;
		classes[c].orient_offset = (int) n_orient_all;
		even[c + 1] = even[c] + (int) ((n + per_block - 1) / per_block);
		n_orient_all += n;
		max_orient = std::max(max_orient, n);
	}
	(void) split;

	// Pixel chunks: a particle's coarse pass has only even[nc] blocks (a few dozen
	// for local searches, against tens of SMs that each hold several blocks), so
	// split the pixel passes in chunks for about coarseChunkTargetBlocks() blocks.
	// Each chunk needs a pass of the block's orientations through the projector, so
	// keep at least 4 passes per chunk (3D references; 1 for 2D), and the partial sums under 32 MB.
	const int block_sz_used = !ref3D ? D2C_BLOCK_SIZE_2D :
			(data_is_3D ? D2C_BLOCK_SIZE_DATA3D : D2C_BLOCK_SIZE_REF3D) * mult;
	const size_t n_pass = (image_size + block_sz_used - 1) / block_sz_used;
	const size_t chunk_stride = n_orient_all * translation_num;
	int n_chunks = coarseChunkOverride();
	const bool overridden = n_chunks > 0;
	const size_t target = coarseChunkTargetBlocks();
	if (n_chunks <= 0)
	{
		// 2D references are projected by a cheap interpolation, so a chunk may be a single pass there
		n_chunks = (int) std::min<size_t>((target + even[nc] - 1) / std::max(even[nc], 1), ref3D ? n_pass / 4 : n_pass);
		while (n_chunks > 1 && n_chunks * chunk_stride * sizeof(XFLOAT) > ((size_t) 32 << 20))
			n_chunks--;
	}
	n_chunks = (int) std::max<size_t>(1, std::min<size_t>(n_chunks, n_pass));
	reportPixelChunks("coarse pass", "RELION_GPU_COARSE_SPLIT", n_chunks, even[nc], overridden);
	{
		static const bool stats = getenv("RELION_GPU_COARSE_STATS") != NULL;
		static long calls = 0;
		if (stats && (calls++ % 500) == 0)
			fprintf(stderr, "COARSE_STATS call %ld: classes %zu orientations %zu (blocks %d) translations %zu image_size %zu chunks %d block %d\n",
					calls, nc, n_orient_all, even[nc], (size_t) translation_num, (size_t) image_size, n_chunks, block_sz_used);
	}
#ifndef ACC_DOUBLE_PRECISION
	// RELION_GPU_COARSE_XT=1: cross-term form of the coarse differences (cuda_kernel_diff2_coarse_xt);
	// =check runs both and reports how far they differ
	static const int xt_mode = [] {
		return gpuswitch::mode("RELION_GPU_COARSE_XT");
	}();
	const size_t xt_tpad = ((translation_num + XT_TRANS - 1) / XT_TRANS) * XT_TRANS;
	const bool use_xt = xt_mode && ref3D && !data_is_3D && image_size * xt_tpad * sizeof(float2) <= ((size_t) 128 << 20);
	if (xt_mode)
	{
		static std::atomic<int> reported_xt(-1);
		const int now = use_xt ? 1 : 0;
		if (reported_xt.exchange(now) != now)
			std::cout << " GPU coarse pass: " << (use_xt ? "cross-term kernel (RELION_GPU_COARSE_XT=0 for the direct one)"
			          : "direct kernel (the cross-term kernel is for 3D references and 2D images with a table up to 128 MB)") << "." << std::endl;
	}
#else
	const bool use_xt = false;
#endif
#ifndef ACC_DOUBLE_PRECISION
	// RELION_GPU_COARSE_CACHE: cached reference projections (see cuda_kernels/coarse_cache.cuh); =check also runs the
	// cross-term kernel and reports how far the two differ
	if (cc != NULL && cc->cache != NULL && use_xt && defer == NULL && xt_mode == 1 && !data_is_3D && ref3D && translation_num > 0)
	{
		static const int cache_mode = gpuswitch::mode("RELION_GPU_COARSE_CACHE");
		if (cache_mode != 0 && coarseCacheReady(*cc, classes, orientation_nums, (int) image_size, translation_num, table.getStream()))
		{
			static std::atomic<int> reported(0);
			std::vector<std::vector<XFLOAT> > before(nc);
			if (cache_mode == 2)
			{
				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(table.getStream()));
				for (size_t c = 0; c < nc; c++)
				{
					before[c].resize(orientation_nums[c] * translation_num);
					DEBUG_HANDLE_ERROR(cudaMemcpy(before[c].data(), classes[c].diff2s, before[c].size() * sizeof(XFLOAT), cudaMemcpyDeviceToHost));
				}
			}
			coarseCacheScore(*cc, classes, trans_x, trans_y, corr_img, Fimg_real, Fimg_imag, translation_num, image_size, table);
			if (cache_mode == 2)
			{
				std::vector<std::vector<XFLOAT> > cached(nc);
				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(table.getStream()));
				for (size_t c = 0; c < nc; c++)
				{
					cached[c].resize(before[c].size());
					DEBUG_HANDLE_ERROR(cudaMemcpy(cached[c].data(), classes[c].diff2s, cached[c].size() * sizeof(XFLOAT), cudaMemcpyDeviceToHost));
					DEBUG_HANDLE_ERROR(cudaMemcpy(classes[c].diff2s, before[c].data(), before[c].size() * sizeof(XFLOAT), cudaMemcpyHostToDevice));
				}
				runDiff2KernelCoarseBatched(classes, orientation_nums, trans_x, trans_y, trans_z, corr_img, Fimg_real, Fimg_imag,
						translation_num, image_size, data_is_3D, table, NULL, NULL);
				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(table.getStream()));
				double max_abs = 0, max_rel = 0, sum_sq = 0;
				long n_all = 0;
				for (size_t c = 0; c < nc; c++)
				{
					std::vector<XFLOAT> xt(cached[c].size());
					DEBUG_HANDLE_ERROR(cudaMemcpy(xt.data(), classes[c].diff2s, xt.size() * sizeof(XFLOAT), cudaMemcpyDeviceToHost));
					for (size_t k = 0; k < xt.size(); k++)
					{
						const double d = fabs((double) xt[k] - cached[c][k]);
						max_abs = std::max(max_abs, d);
						max_rel = std::max(max_rel, d / std::max(1.0, fabs((double) xt[k])));
						sum_sq += d * d;
						n_all++;
					}
				}
				if (reported++ % 200 == 0)
					fprintf(stderr, "COARSE_CACHE_CHECK call %d: %ld values, max |diff| %.4g, max relative %.3g, rms %.4g\n",
							(int) reported, n_all, max_abs, max_rel, sqrt(sum_sq / std::max(1L, n_all)));
			}
			return true;
		}
	}
#endif
	// A deferred (particle group) launch always sums through the chunk buffer
	const bool use_partial = n_chunks > 1 || use_xt || defer != NULL || coarseDeterministic();
	if (defer != NULL)
		defer->partial.reset(new AccPtr<XFLOAT>(table.make<XFLOAT>()));
	AccPtr<XFLOAT> partial_local = table.make<XFLOAT>();
	AccPtr<XFLOAT> &partial = defer != NULL ? *defer->partial : partial_local;
	if (use_partial)
	{
		partial.setSize(n_chunks * chunk_stride);
		partial.deviceAlloc();
	}

	// One upload for the class table and both block tables
	const size_t class_bytes = ((nc * sizeof(CoarseClassJob) + 15) / 16) * 16;
	const size_t bytes = class_bytes + 2 * (nc + 1) * sizeof(int);
	table.setSize(bytes);
	table.allAlloc();
	memcpy(&table[0], classes.data(), nc * sizeof(CoarseClassJob));
	memcpy(&table[class_bytes], even.data(), (nc + 1) * sizeof(int));
	memcpy(&table[class_bytes + (nc + 1) * sizeof(int)], rest.data(), (nc + 1) * sizeof(int));
	table.cpToDevice();

	const CoarseClassJob *d_classes = (const CoarseClassJob *) ~table;
	const int *d_even = (const int *) &(~table)[class_bytes];
	const int *d_rest = d_even + nc + 1;
	deviceStream_t stream = table.getStream();

	// (cos, sin) of the shifts once per particle instead of a sincosf for every pixel,
	// translation and block (2D data, single precision, up to 64 MB)
	const float2 *d_shift = NULL;
#ifndef ACC_DOUBLE_PRECISION
	if (defer != NULL)
		defer->shift.reset(new AccPtr<float2>(table.make<float2>()));
	AccPtr<float2> shift_local = table.make<float2>();
	AccPtr<float2> &shift = defer != NULL ? *defer->shift : shift_local;
	if (gpuShiftTable() && !data_is_3D && !(use_xt && defer != NULL) && image_size * translation_num * sizeof(float2) <= ((size_t) 64 << 20))
	{
		shift.setSize(image_size * translation_num);
		shift.deviceAlloc();
		cuda_kernel_shift_table_pt<<<(image_size + 127) / 128, 128, 0, stream>>>(
				trans_x, trans_y, ~shift, (int) translation_num, (int) image_size,
				projector.imgX, projector.imgY, projector.maxR);
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
		d_shift = ~shift;
	}
#endif

	if (defer != NULL)
	{
		// The launch is made later for the whole group (runDiff2KernelCoarseGroup)
		if (use_xt)
		{
			defer->xt = true;
			defer->h_classes = classes;
		}
		else if (ref3D || data_is_3D || use_xt)
			REPORT_ERROR("BUG: runDiff2KernelCoarseBatched: a deferred launch is for 2D references and 2D data only");
		defer->projector.reset(new AccProjectorKernel(projector));
		defer->trans_x = trans_x; defer->trans_y = trans_y; defer->trans_z = trans_z;
		defer->corr_img = corr_img; defer->Fimg_real = Fimg_real; defer->Fimg_imag = Fimg_imag;
		defer->translation_num = translation_num; defer->image_size = image_size;
		defer->d_classes = d_classes; defer->d_even = d_even; defer->d_shift = d_shift;
		defer->n_blocks = even[nc]; defer->n_classes = (int) nc; defer->n_chunks = n_chunks;
		defer->chunk_stride = chunk_stride; defer->max_orient = max_orient;
		defer->d_partial = ~partial;
		defer->stream = stream;
		return true;
	}

#ifndef ACC_DOUBLE_PRECISION
	static const int xt_ablate = getenv("RELION_XT_ABLATE") ? atoi(getenv("RELION_XT_ABLATE")) : 0;
	auto run_xt = [&]()
	{
		const int ntt = (int) (xt_tpad / XT_TRANS);
		AccPtr<float2> xt_a = table.make<float2>();
		xt_a.setSize(image_size * xt_tpad);
		xt_a.deviceAlloc();
		AccPtr<float> xt_x2 = table.make<float>();
		xt_x2.setSize(image_size);
		xt_x2.deviceAlloc();
		cuda_kernel_coarse_xt_prepare<<<(image_size + 127) / 128, 128, 0, stream>>>(
				trans_x, trans_y, Fimg_real, Fimg_imag, corr_img, ~xt_a, ~xt_x2,
				(int) translation_num, (int) xt_tpad, (int) image_size, projector.imgX, projector.imgY, projector.maxR);
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
		static const char *xt_env = getenv("RELION_GPU_XT_TILES");
		static const bool xt_off = gpuswitch::isOffValue(xt_env);
		static const bool xt_auto = !xt_off && (xt_env == NULL || std::string(xt_env) == "auto" || std::string(xt_env) == "check" || atoi(xt_env) <= 0);
		static const int xt_fixed = (xt_env != NULL && !xt_auto && !xt_off) ? atoi(xt_env) : 1;
		int device = 0;
		int nt = std::max(1, std::min(xt_fixed, std::min(ntt, 8)));
		int cand = 0;
		std::pair<cudaEvent_t, cudaEvent_t> ev((cudaEvent_t) NULL, (cudaEvent_t) NULL);
		if (xt_auto && ntt > 1)
		{
			cudaGetDevice(&device);
			nt = xtTileTuner().begin(device, ntt, stream, cand, ev);
		}
#define XT_LAUNCH(NT) cuda_kernel_diff2_coarse_xt<NT><<<dim3(even[nc], n_chunks, (ntt + NT - 1) / NT), XT_BLOCK, 0, stream>>>( \
				d_classes, d_even, (int) nc, ~xt_a, ~xt_x2, corr_img, (int) translation_num, (int) xt_tpad, \
				(int) image_size, ~partial, n_chunks, chunk_stride, xt_ablate)
		auto launch = [&](int t)
		{
			if (t >= 8) XT_LAUNCH(8); else if (t >= 6) XT_LAUNCH(6); else if (t >= 4) XT_LAUNCH(4); else if (t == 3) XT_LAUNCH(3); else if (t == 2) XT_LAUNCH(2); else XT_LAUNCH(1);
			return cudaGetLastError();
		};
#undef XT_LAUNCH
		cudaError_t xt_err = launch(nt);
		if (xt_err != cudaSuccess && nt > 1)
		{
			// Too many resources for this tile count: say so, never use it again, and run the plain one
			fprintf(stderr, " GPU coarse_xt: %d tiles per block cannot be launched (%s), using 1\n", nt, cudaGetErrorString(xt_err));
			if (xt_auto) { xtTileTuner().discard(ev); xtTileTuner().mark_dead(device, ntt, nt); }
			ev = std::make_pair((cudaEvent_t) NULL, (cudaEvent_t) NULL);
			xt_err = launch(1);
		}
		LAUNCH_HANDLE_ERROR(xt_err);
		if (ev.first != NULL)
			xtTileTuner().finish(device, ntt, ev, cand, (double) even[nc] * (double) image_size * (double) ntt, stream);
		const size_t n = max_orient * translation_num;
		const unsigned sum_blocks = (unsigned) std::min<size_t>((n + 255) / 256, 1024);
		cuda_kernel_diff2_coarse_sum_chunks<<<dim3(sum_blocks, (unsigned) nc), 256, 0, stream>>>(
				d_classes, ~partial, n_chunks, chunk_stride, (int) translation_num);
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
	};
	if (use_xt && xt_mode == 1)
	{
		run_xt();
		return true;
	}
#endif

	std::vector<std::vector<XFLOAT> > xt_init(nc);
	if (use_xt && xt_mode == 2)
	{
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
		for (size_t c = 0; c < nc; c++)
		{
			xt_init[c].resize(orientation_nums[c] * translation_num);
			DEBUG_HANDLE_ERROR(cudaMemcpy(xt_init[c].data(), classes[c].diff2s, xt_init[c].size() * sizeof(XFLOAT), cudaMemcpyDeviceToHost));
		}
	}
#define D2C_BATCH(R3, D3, BS, EPB, PF) launchDiff2CoarseBatched<R3, D3, BS, EPB, PF>(even[nc], rest[nc], (int) nc, \
		d_classes, d_even, d_rest, projector, trans_x, trans_y, trans_z, corr_img, Fimg_real, Fimg_imag, \
		translation_num, image_size, stream, ~partial, n_chunks, chunk_stride, max_orient, d_shift)
	if (!ref3D)
	{
		if (data_is_3D) D2C_BATCH(false, true, D2C_BLOCK_SIZE_2D, D2C_EULERS_PER_BLOCK_2D, 2);
		else            D2C_BATCH(false, false, D2C_BLOCK_SIZE_2D, D2C_EULERS_PER_BLOCK_2D, 2);
	}
	else if (data_is_3D)
	{
		if      (mult == 1) D2C_BATCH(true, true, D2C_BLOCK_SIZE_DATA3D,   D2C_EULERS_PER_BLOCK_DATA3D, 4);
		else if (mult == 2) D2C_BATCH(true, true, D2C_BLOCK_SIZE_DATA3D*2, D2C_EULERS_PER_BLOCK_DATA3D, 4);
		else if (mult == 4) D2C_BATCH(true, true, D2C_BLOCK_SIZE_DATA3D*4, D2C_EULERS_PER_BLOCK_DATA3D, 4);
#ifndef ACC_DOUBLE_PRECISION
		else                D2C_BATCH(true, true, D2C_BLOCK_SIZE_DATA3D*8, D2C_EULERS_PER_BLOCK_DATA3D, 4);
#endif
	}
	else
	{
		if      (mult == 1) D2C_BATCH(true, false, D2C_BLOCK_SIZE_REF3D,   D2C_EULERS_PER_BLOCK_REF3D, 4);
		else if (mult == 2) D2C_BATCH(true, false, D2C_BLOCK_SIZE_REF3D*2, D2C_EULERS_PER_BLOCK_REF3D, 4);
		else if (mult == 4) D2C_BATCH(true, false, D2C_BLOCK_SIZE_REF3D*4, D2C_EULERS_PER_BLOCK_REF3D, 4);
#ifndef ACC_DOUBLE_PRECISION
		else                D2C_BATCH(true, false, D2C_BLOCK_SIZE_REF3D*8, D2C_EULERS_PER_BLOCK_REF3D, 4);
#endif
	}
#undef D2C_BATCH
#ifndef ACC_DOUBLE_PRECISION
	if (use_xt && xt_mode == 2)
	{
		// Debug: keep the direct differences, run the cross-term form into zeroed arrays and compare
		static long checks = 0;
		std::vector<std::vector<XFLOAT> > direct(nc), cross(nc);
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
		for (size_t c = 0; c < nc; c++)
		{
			const size_t n = orientation_nums[c] * translation_num;
			direct[c].resize(n);
			cross[c].resize(n);
			DEBUG_HANDLE_ERROR(cudaMemcpy(direct[c].data(), classes[c].diff2s, n * sizeof(XFLOAT), cudaMemcpyDeviceToHost));
			DEBUG_HANDLE_ERROR(cudaMemcpy(classes[c].diff2s, xt_init[c].data(), n * sizeof(XFLOAT), cudaMemcpyHostToDevice));
		}
		run_xt();
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
		double max_abs = 0, max_rel = 0, sum_sq = 0, init_max = 0;
		long n_all = 0;
		for (size_t c = 0; c < nc; c++)
		{
			const size_t n = direct[c].size();
			DEBUG_HANDLE_ERROR(cudaMemcpy(cross[c].data(), classes[c].diff2s, n * sizeof(XFLOAT), cudaMemcpyDeviceToHost));
			for (size_t k = 0; k < n; k++)
			{
				init_max = std::max(init_max, (double) fabs(xt_init[c][k]));
				const double d = fabs((double) direct[c][k] - cross[c][k]);
				max_abs = std::max(max_abs, d);
				max_rel = std::max(max_rel, d / std::max(1.0, fabs((double) direct[c][k])));
				sum_sq += d * d;
				n_all++;
			}
			DEBUG_HANDLE_ERROR(cudaMemcpy(classes[c].diff2s, direct[c].data(), n * sizeof(XFLOAT), cudaMemcpyHostToDevice));
		}
		if (checks++ % 200 == 0)
			fprintf(stderr, "COARSE_XT_CHECK call %ld: %ld values, max |diff| %.4g, max relative %.3g, rms %.4g, max initial %.4g\n",
					checks, n_all, max_abs, max_rel, sqrt(sum_sq / std::max(1L, n_all)), init_max);
	}
#endif
	return true;
}

// The coarse differences of several particles (2D references, 2D data) in two launches: the blocks of
// particle q are the z = q slice of the grid, its inputs come from a CoarseParticle record (the same
// kernels as for one particle, so a particle gets exactly the sums it gets alone with
// RELION_GPU_COARSE_DETERMINISTIC=1). Each item was prepared by runDiff2KernelCoarseBatched with `defer`.
// Items with different translation numbers or image sizes go in separate launches.
#ifndef ACC_DOUBLE_PRECISION
// Deferred cross-term passes (3D references) of several particles: per set of particles with the same classes,
// image and number of translations, at most XT_GMAX, the union of their orientations is projected once
// (cuda_kernel_diff2_coarse_xt_group). `tables` and `work` must live until the launches have run.
static void runDiff2KernelCoarseGroupXt(std::vector<CoarseDeferred *> &items, AccPtr<char> &tables, AccPtr<char> &work)
{
	// Sets of particles that can share: same classes, image and translations
	std::map<std::vector<long>, std::vector<CoarseDeferred *> > by_key;
	std::vector<std::vector<CoarseDeferred *> > sets;
	for (CoarseDeferred *d : items)
	{
		const AccProjectorKernel &pk = d->h_classes[0].projector;
		std::vector<long> key = {(long) d->image_size, (long) d->translation_num, pk.imgX, pk.imgY, pk.maxR, d->optics_group};
		key.insert(key.end(), d->iclass.begin(), d->iclass.end());
		std::vector<CoarseDeferred *> &v = by_key[key];
		v.push_back(d);
		if ((int) v.size() == XT_GMAX)
		{
			sets.push_back(v);
			v.clear();
		}
	}
	for (auto &kv : by_key)
		if (!kv.second.empty())
			sets.push_back(kv.second);

	// Per set: the union of the orientations of each class (by their angles), and where each particle has them
	struct Set
	{
		std::vector<CoarseClassJob> classes;
		std::vector<XtGroupBlock> blocks;
		std::vector<int> bcols;
		int max_tiles = 1;
		std::vector<const float *> eul;
		std::vector<int> map;
		std::vector<int2> cols;
		int n_chunks;
		size_t off_classes, off_blocks, off_bcols, off_eul, off_map, off_parts, off_cols; // in `tables`
		size_t off_A, off_x2, off_x2sum;                                       // in `work`
	};
	std::vector<Set> plan(sets.size());
	size_t tbytes = 0, wbytes = 0;
	auto take = [](size_t &total, size_t bytes) { const size_t at = total; total += ((bytes + 255) / 256) * 256; return at; };
	const int override_chunks = coarseChunkOverride();
	const size_t target = coarseChunkTargetBlocks();
	for (size_t si = 0; si < sets.size(); si++)
	{
		std::vector<CoarseDeferred *> &S = sets[si];
		Set &P = plan[si];
		const int G = (int) S.size(), nc = (int) S[0]->h_classes.size();
		const int T = (int) S[0]->translation_num;
		for (int j = 0; j < nc; j++)
		{
			std::map<std::array<XFLOAT, 3>, int> where;
			const size_t u0 = P.eul.size();
			for (int k = 0; k < G; k++)
			{
				const CoarseClassJob &cj = S[k]->h_classes[j];
				for (int i = 0; i < cj.n_orient; i++)
				{
					const XFLOAT *a = S[k]->angles + 3 * (size_t) (cj.orient_offset + i);
					auto ins = where.insert(std::make_pair(std::array<XFLOAT, 3>{a[0], a[1], a[2]}, (int) (P.eul.size() - u0)));
					if (ins.second)
					{
						P.eul.push_back(cj.eulers + 9 * (size_t) i);
						P.map.resize(P.map.size() + G, -1);
					}
					P.map[(u0 + ins.first->second) * G + k] = cj.orient_offset + i;
				}
			}
			// Orientations searched by the same particles next to each other (stable: first seen first), so that
			// a block of 16 holds the columns of few particles; then the blocks and their columns
			const int U = (int) (P.eul.size() - u0);
			std::vector<unsigned> mask(U, 0u);
			for (int u = 0; u < U; u++)
				for (int k = 0; k < G; k++)
					if (P.map[(u0 + u) * G + k] >= 0)
						mask[u] |= 1u << k;
			std::vector<int> order(U);
			for (int u = 0; u < U; u++)
				order[u] = u;
			std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return mask[a] < mask[b]; });
			std::vector<const float *> eul_sorted(U);
			std::vector<int> map_sorted((size_t) U * G);
			for (int u = 0; u < U; u++)
			{
				eul_sorted[u] = P.eul[u0 + order[u]];
				std::copy(P.map.begin() + (u0 + order[u]) * G, P.map.begin() + (u0 + order[u] + 1) * G, map_sorted.begin() + (size_t) u * G);
			}
			std::copy(eul_sorted.begin(), eul_sorted.end(), P.eul.begin() + u0);
			std::copy(map_sorted.begin(), map_sorted.end(), P.map.begin() + u0 * G);
			for (int b0 = 0; b0 < U; b0 += XT_ORIENT)
			{
				XtGroupBlock B = {j, b0, (int) P.bcols.size(), 0, 0u};
				for (int u = b0; u < std::min(U, b0 + XT_ORIENT); u++)
					B.mask |= mask[order[u]];
				for (int k = 0; k < G; k++)
					if ((B.mask >> k) & 1u)
						for (int t = 0; t < T; t++)
							P.bcols.push_back(k * T + t);
				B.n_cols = (int) P.bcols.size() - B.col_off;
				P.max_tiles = std::max(P.max_tiles, (B.n_cols + XT_TRANS - 1) / XT_TRANS);
				P.blocks.push_back(B);
			}
			CoarseClassJob uc = S[0]->h_classes[j];
			uc.eulers = NULL; uc.diff2s = NULL;
			uc.n_orient = uc.n_even = U;
			uc.orient_offset = (int) u0;
			P.classes.push_back(uc);
		}
		for (int k = 0; k < G; k++)
			for (int t = 0; t < T; t++)
				P.cols.push_back(make_int2(k, t));
		// Pixel chunks as the single passes choose them, for the union's blocks; never more than a particle's buffer holds
		int nch = S[0]->n_chunks;
		for (CoarseDeferred *d : S)
			nch = std::min(nch, d->n_chunks);
		if (override_chunks <= 0)
			nch = std::max(1, std::min(nch, (int) ((target + P.blocks.size() - 1) / std::max(P.blocks.size(), (size_t) 1))));
		P.n_chunks = nch;
		P.off_classes = take(tbytes, nc * sizeof(CoarseClassJob));
		P.off_blocks = take(tbytes, P.blocks.size() * sizeof(XtGroupBlock));
		P.off_bcols = take(tbytes, P.bcols.size() * sizeof(int));
		P.off_eul = take(tbytes, P.eul.size() * sizeof(float *));
		P.off_map = take(tbytes, P.map.size() * sizeof(int));
		P.off_parts = take(tbytes, G * sizeof(XtGroupParticle));
		P.off_cols = take(tbytes, P.cols.size() * sizeof(int2));
		P.off_A = take(wbytes, S[0]->image_size * P.cols.size() * sizeof(float2));
		P.off_x2 = take(wbytes, (size_t) G * S[0]->image_size * sizeof(float));
		P.off_x2sum = take(wbytes, (size_t) G * nch * sizeof(float));
	}
	{
		// Said once, then every 5000 sets with RELION_GPU_COARSE_STATS: how much the particles share
		static std::atomic<long> n_sets(0), n_parts(0), n_own(0), n_union(0), n_needed(0), n_scored(0);
		static const bool stats = getenv("RELION_GPU_COARSE_STATS") != NULL;
		long own = 0, uni = 0, parts = 0;
		for (size_t si = 0; si < sets.size(); si++)
		{
			for (CoarseDeferred *d : sets[si])
				for (const CoarseClassJob &cj : d->h_classes)
					own += cj.n_orient;
			uni += (long) plan[si].eul.size();
			parts += (long) sets[si].size();
			for (CoarseDeferred *d : sets[si])
				for (const CoarseClassJob &cj : d->h_classes)
					n_needed += (long) cj.n_orient * (long) d->translation_num;
			for (const XtGroupBlock &B : plan[si].blocks)
				n_scored += (long) std::min(XT_ORIENT, plan[si].classes[B.cls].n_orient - B.orient0) * B.n_cols;
		}
		const long k = n_sets.fetch_add((long) sets.size());
		n_parts += parts; n_own += own; n_union += uni;
		if (k == 0 || (stats && k / 5000 != (k + (long) sets.size()) / 5000))
			printf(" GPU coarse pass: particle groups share their projections (RELION_GPU_GROUP_SHARED): %.1f particles per set, "
			       "%.2fx fewer orientations projected, %.2fx the differences needed (%ld sets so far)\n",
			       (double) n_parts / std::max(1L, (long) n_sets), (double) n_own / std::max(1L, (long) n_union),
			       (double) n_scored / std::max(1L, (long) n_needed), (long) n_sets);
	}
	tables.setSize(tbytes);
	tables.allAlloc();
	work.setSize(wbytes);
	work.deviceAlloc();
	for (size_t si = 0; si < sets.size(); si++)
	{
		std::vector<CoarseDeferred *> &S = sets[si];
		Set &P = plan[si];
		const int G = (int) S.size(), nc = (int) P.classes.size();
		memcpy(&tables[P.off_classes], P.classes.data(), nc * sizeof(CoarseClassJob));
		memcpy(&tables[P.off_blocks], P.blocks.data(), P.blocks.size() * sizeof(XtGroupBlock));
		memcpy(&tables[P.off_bcols], P.bcols.data(), P.bcols.size() * sizeof(int));
		memcpy(&tables[P.off_eul], P.eul.data(), P.eul.size() * sizeof(float *));
		memcpy(&tables[P.off_map], P.map.data(), P.map.size() * sizeof(int));
		memcpy(&tables[P.off_cols], P.cols.data(), P.cols.size() * sizeof(int2));
		XtGroupParticle *parts = (XtGroupParticle *) &tables[P.off_parts];
		for (int k = 0; k < G; k++)
		{
			const CoarseDeferred &d = *S[k];
			parts[k] = XtGroupParticle{d.trans_x, d.trans_y, d.Fimg_real, d.Fimg_imag, d.corr_img,
					(float *) &(~work)[P.off_x2 + (size_t) k * d.image_size * sizeof(float)],
					d.d_partial, d.chunk_stride, (int) d.translation_num, k * (int) d.translation_num};
		}
	}
	tables.cpToDevice();

	deviceStream_t stream = tables.getStream();
	for (size_t si = 0; si < sets.size(); si++)
	{
		std::vector<CoarseDeferred *> &S = sets[si];
		Set &P = plan[si];
		const int G = (int) S.size(), nc = (int) P.classes.size(), n_cols = (int) P.cols.size();
		const int image_size = (int) S[0]->image_size;
		const AccProjectorKernel &pk = P.classes[0].projector;
		const XtGroupParticle *d_parts = (const XtGroupParticle *) &(~tables)[P.off_parts];
		float2 *d_A = (float2 *) &(~work)[P.off_A];
		float *d_x2sum = (float *) &(~work)[P.off_x2sum];
		cuda_kernel_coarse_xt_group_prepare<<<dim3((image_size + 127) / 128, G), 128, 0, stream>>>(
				d_parts, d_A, n_cols, image_size, pk.imgX, pk.imgY, pk.maxR);
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
		cuda_kernel_coarse_xt_group_x2<<<dim3(P.n_chunks, G), XT_BLOCK, 0, stream>>>(d_parts, d_x2sum, image_size, P.n_chunks);
		LAUNCH_HANDLE_ERROR(cudaGetLastError());

		const int ntt = P.max_tiles;
		int device = 0, cand = 0;
		cudaGetDevice(&device);
		std::pair<cudaEvent_t, cudaEvent_t> ev((cudaEvent_t) NULL, (cudaEvent_t) NULL);
		const int nt = ntt > 1 ? xtTileTuner().begin(device, ntt, stream, cand, ev) : 1;
#define XT_GROUP_LAUNCH(NT) cuda_kernel_diff2_coarse_xt_group<NT><<<dim3((unsigned) P.blocks.size(), P.n_chunks, (ntt + NT - 1) / NT), XT_BLOCK, 0, stream>>>( \
				(const CoarseClassJob *) &(~tables)[P.off_classes], (const XtGroupBlock *) &(~tables)[P.off_blocks], \
				(const int *) &(~tables)[P.off_bcols], \
				(const float *const *) &(~tables)[P.off_eul], (const int *) &(~tables)[P.off_map], d_parts, G, \
				(const int2 *) &(~tables)[P.off_cols], n_cols, d_A, d_x2sum, image_size, P.n_chunks)
		auto launch = [&](int t)
		{
			if (t >= 8) XT_GROUP_LAUNCH(8); else if (t >= 6) XT_GROUP_LAUNCH(6); else if (t >= 4) XT_GROUP_LAUNCH(4);
			else if (t == 3) XT_GROUP_LAUNCH(3); else if (t == 2) XT_GROUP_LAUNCH(2); else XT_GROUP_LAUNCH(1);
			return cudaGetLastError();
		};
#undef XT_GROUP_LAUNCH
		cudaError_t err = launch(nt);
		if (err != cudaSuccess && nt > 1)
		{
			fprintf(stderr, " GPU coarse_xt (particle group): %d tiles per block cannot be launched (%s), using 1\n", nt, cudaGetErrorString(err));
			xtTileTuner().discard(ev);
			xtTileTuner().mark_dead(device, ntt, nt);
			ev = std::make_pair((cudaEvent_t) NULL, (cudaEvent_t) NULL);
			err = launch(1);
		}
		LAUNCH_HANDLE_ERROR(err);
		if (ev.first != NULL)
			xtTileTuner().finish(device, ntt, ev, cand, (double) P.blocks.size() * (double) image_size * (double) ntt, stream);

		// Each particle's chunks into its differences (as cuda_kernel_diff2_coarse_sum_chunks)
		for (int k = 0; k < G; k++)
		{
			const CoarseDeferred &d = *S[k];
			const size_t n = d.max_orient * d.translation_num;
			const unsigned sum_blocks = (unsigned) std::min<size_t>((n + 255) / 256, 1024);
			cuda_kernel_diff2_coarse_sum_chunks<<<dim3(sum_blocks, (unsigned) d.n_classes), 256, 0, stream>>>(
					d.d_classes, d.d_partial, P.n_chunks, d.chunk_stride, (int) d.translation_num);
			LAUNCH_HANDLE_ERROR(cudaGetLastError());
		}
	}
}
#endif

bool runDiff2KernelCoarseGroup(std::vector<CoarseDeferred *> &items, AccPtr<char> &recs, AccPtr<char> &xt_tables, AccPtr<char> &xt_work)
{
#ifndef ACC_DOUBLE_PRECISION
	std::vector<CoarseDeferred *> xt_items;
	for (size_t i = 0; i < items.size(); )
		if (items[i]->xt) { xt_items.push_back(items[i]); items.erase(items.begin() + i); }
		else i++;
	if (!xt_items.empty())
		runDiff2KernelCoarseGroupXt(xt_items, xt_tables, xt_work);
#endif
	if (items.empty())
		return true;
	std::vector<size_t> order(items.size());
	for (size_t i = 0; i < order.size(); i++)
		order[i] = i;
	std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
		return std::make_pair(items[a]->translation_num, items[a]->image_size) <
				std::make_pair(items[b]->translation_num, items[b]->image_size); });

	// One upload of all records
	recs.setSize(items.size() * sizeof(CoarseParticle));
	recs.allAlloc();
	CoarseParticle *h = (CoarseParticle *) &recs[0];
	for (size_t k = 0; k < order.size(); k++)
	{
		const CoarseDeferred &d = *items[order[k]];
		h[k] = CoarseParticle{d.trans_x, d.trans_y, d.Fimg_real, d.Fimg_imag, d.corr_img, d.d_shift,
				d.d_classes, d.d_even, d.d_partial, d.n_blocks, d.n_classes, d.n_chunks, d.chunk_stride};
	}
	recs.cpToDevice();
	const CoarseParticle *d_recs = (const CoarseParticle *) ~recs;

	for (size_t first = 0; first < order.size(); )
	{
		size_t last = first + 1;
		const CoarseDeferred &d0 = *items[order[first]];
		int max_blocks = d0.n_blocks, max_chunks = d0.n_chunks, max_classes = d0.n_classes;
		size_t max_orient = d0.max_orient;
		while (last < order.size() && items[order[last]]->translation_num == d0.translation_num &&
				items[order[last]]->image_size == d0.image_size)
		{
			const CoarseDeferred &d = *items[order[last]];
			max_blocks = std::max(max_blocks, d.n_blocks);
			max_chunks = std::max(max_chunks, d.n_chunks);
			max_classes = std::max(max_classes, d.n_classes);
			max_orient = std::max(max_orient, d.max_orient);
			last++;
		}
		const unsigned n = (unsigned) (last - first);
		deviceStream_t stream = d0.stream;
		AccProjectorKernel projector = *d0.projector;
		for (size_t z0 = 0; z0 < n; z0 += 65535)
		{
			const unsigned nz = (unsigned) std::min<size_t>(65535, n - z0);
			cuda_kernel_diff2_coarse<false, false, D2C_BLOCK_SIZE_2D, D2C_EULERS_PER_BLOCK_2D, 2>
				<<<dim3(max_blocks, max_chunks, nz), D2C_BLOCK_SIZE_2D, 0, stream>>>(
					NULL, d0.trans_x, d0.trans_y, d0.trans_z, d0.Fimg_real, d0.Fimg_imag, projector, d0.corr_img, NULL,
					d0.translation_num, d0.image_size, NULL, NULL, 0, false,
					NULL, 1, 0, NULL, d_recs + first + z0);
			const size_t nel = max_orient * d0.translation_num;
			const unsigned sum_blocks = (unsigned) std::min<size_t>((nel + 255) / 256, 1024);
			cuda_kernel_diff2_coarse_sum_chunks_batched<<<dim3(sum_blocks, max_classes, nz), 256, 0, stream>>>(
					d_recs + first + z0, (int) d0.translation_num, false);
			LAUNCH_HANDLE_ERROR(cudaGetLastError());
		}
		first = last;
	}
	return true;
}

void minOnDeviceBatched(const std::vector<std::pair<const XFLOAT *, size_t> > &arrays, std::vector<XFLOAT> &mins,
		AccPtr<char> &scratch, deviceStream_t stream)
{
	const size_t n = arrays.size();
	mins.assign(n, 0);
	if (n == 0)
		return;
	const size_t job_bytes = ((n * sizeof(MinJob) + 15) / 16) * 16;
	scratch.setSize(job_bytes + n * sizeof(XFLOAT));
	scratch.setStream(stream);
	scratch.allAlloc();
	MinJob *h = (MinJob *) &scratch[0];
	for (size_t i = 0; i < n; i++)
		h[i] = MinJob{arrays[i].first, arrays[i].second};
	scratch.cpToDevice();
	cuda_kernel_min_batched<<<(unsigned) n, 256, 0, stream>>>((const MinJob *) ~scratch, (XFLOAT *) (&(~scratch)[job_bytes]));
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
	HANDLE_ERROR(cudaMemcpyAsync(&scratch[job_bytes], &(~scratch)[job_bytes], n * sizeof(XFLOAT), cudaMemcpyDeviceToHost, stream));
	DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
	memcpy(mins.data(), &scratch[job_bytes], n * sizeof(XFLOAT));
}
#endif

void runDiff2KernelCoarse(
		AccProjectorKernel &projector,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT *corr_img,
		XFLOAT *Fimg_real,
		XFLOAT *Fimg_imag,
		XFLOAT *d_eulers,
		XFLOAT *diff2s,
		XFLOAT local_sqrtXi2,
		long unsigned orientation_num,
		long unsigned translation_num,
		long unsigned image_size,
		deviceStream_t stream,
		bool do_CC,
		bool data_is_3D)
{
	const long unsigned blocks3D = (data_is_3D? D2C_BLOCK_SIZE_DATA3D : D2C_BLOCK_SIZE_REF3D);

	if(!do_CC)
	{
		if(projector.mdlZ!=0)
		{
#ifdef _SYCL_ENABLED
			assert(translation_num <= blocks3D);

			long unsigned rest = orientation_num % blocks3D;
			long unsigned even_orientation_num = orientation_num - rest;

			if (even_orientation_num)
			{
				if (data_is_3D)
				{
					unsigned long grid_size = (D2C_EULERS_PER_BLOCK_DATA3D == 1) ? orientation_num : even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_DATA3D;
					AccUtilities::diff2_coarse<true,true, D2C_BLOCK_SIZE_DATA3D, D2C_EULERS_PER_BLOCK_DATA3D, PREFETCH_FRACTION_3D>(
						grid_size,
						D2C_BLOCK_SIZE_DATA3D,
						d_eulers,
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						diff2s,
						translation_num,
						image_size,
						stream);
				}
				else
				{
					unsigned long grid_size = (D2C_EULERS_PER_BLOCK_REF3D == 1) ? orientation_num : even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_REF3D;
					AccUtilities::diff2_coarse<true,false, D2C_BLOCK_SIZE_REF3D, D2C_EULERS_PER_BLOCK_REF3D, PREFETCH_FRACTION_3D>(
						grid_size,
						D2C_BLOCK_SIZE_REF3D,
						d_eulers,
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						diff2s,
						translation_num,
						image_size,
						stream);
				}
			}

			if (rest)
			{
				if (data_is_3D && D2C_EULERS_PER_BLOCK_DATA3D != 1)
					AccUtilities::diff2_coarse<true,true, D2C_BLOCK_SIZE_DATA3D, 1, PREFETCH_FRACTION_3D>(
						rest,
						D2C_BLOCK_SIZE_DATA3D,
						&d_eulers[9*even_orientation_num],
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						&diff2s[translation_num*even_orientation_num],
						translation_num,
						image_size,
						stream);
				else if (!data_is_3D && D2C_EULERS_PER_BLOCK_REF3D != 1)
					AccUtilities::diff2_coarse<true,false, D2C_BLOCK_SIZE_REF3D, 1, PREFETCH_FRACTION_3D>(
						rest,
						D2C_BLOCK_SIZE_REF3D,
						&d_eulers[9*even_orientation_num],
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						&diff2s[translation_num*even_orientation_num],
						translation_num,
						image_size,
						stream);
			}
#else
#ifdef ACC_DOUBLE_PRECISION
			if (translation_num > blocks3D*4)
				CRITICAL(ERR_TRANSLIM);
#else
			if (translation_num > blocks3D*8)
				CRITICAL(ERR_TRANSLIM);
#endif

			long unsigned rest = orientation_num % blocks3D;
			long unsigned even_orientation_num = orientation_num - rest;
// TODO - find a more compact way to represent these combinations resulting in
// a single call to diff2_course?
			if (translation_num <= blocks3D)
			{
				if (even_orientation_num != 0)
				{
					if(data_is_3D)
						AccUtilities::diff2_coarse<true,true, D2C_BLOCK_SIZE_DATA3D, D2C_EULERS_PER_BLOCK_DATA3D, 4>(
							even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_DATA3D,
							D2C_BLOCK_SIZE_DATA3D,
							d_eulers,
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							diff2s,
							translation_num,
							image_size,
							stream);
					else
						AccUtilities::diff2_coarse<true,false, D2C_BLOCK_SIZE_REF3D, D2C_EULERS_PER_BLOCK_REF3D, 4>(
							even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_REF3D,
							D2C_BLOCK_SIZE_REF3D,
							d_eulers,
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							diff2s,
							translation_num,
							image_size,
							stream);
				}

				if (rest != 0)
				{
					if(data_is_3D)
						AccUtilities::diff2_coarse<true,true, D2C_BLOCK_SIZE_DATA3D, 1, 4>(
							rest,
							D2C_BLOCK_SIZE_DATA3D,
							&d_eulers[9*even_orientation_num],
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							&diff2s[translation_num*even_orientation_num],
							translation_num,
							image_size,
							stream);
					else
						AccUtilities::diff2_coarse<true,false, D2C_BLOCK_SIZE_REF3D, 1, 4>(
							rest,
							D2C_BLOCK_SIZE_REF3D,
							&d_eulers[9*even_orientation_num],
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							&diff2s[translation_num*even_orientation_num],
							translation_num,
							image_size,
							stream);
				}
			}
			else if (translation_num <= blocks3D*2)
			{
				if (even_orientation_num != 0)
				{
					if(data_is_3D)
						AccUtilities::diff2_coarse<true,true, D2C_BLOCK_SIZE_DATA3D*2, D2C_EULERS_PER_BLOCK_DATA3D, 4>(
							even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_DATA3D,
							D2C_BLOCK_SIZE_DATA3D*2,
							d_eulers,
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							diff2s,
							translation_num,
							image_size,
							stream);
					else
						AccUtilities::diff2_coarse<true,false, D2C_BLOCK_SIZE_REF3D*2, D2C_EULERS_PER_BLOCK_REF3D, 4>(
							even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_REF3D,
							D2C_BLOCK_SIZE_REF3D*2,
							d_eulers,
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							diff2s,
							translation_num,
							image_size,
							stream);

				}

				if (rest != 0)
				{
					if(data_is_3D)
						AccUtilities::diff2_coarse<true, true, D2C_BLOCK_SIZE_DATA3D*2, 1, 4>(
							rest,
							D2C_BLOCK_SIZE_DATA3D*2,
							&d_eulers[9*even_orientation_num],
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							&diff2s[translation_num*even_orientation_num],
							translation_num,
							image_size,
							stream);
					else
						AccUtilities::diff2_coarse<true,false, D2C_BLOCK_SIZE_REF3D*2, 1, 4>(
							rest,
							D2C_BLOCK_SIZE_REF3D*2,
							&d_eulers[9*even_orientation_num],
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							&diff2s[translation_num*even_orientation_num],
							translation_num,
							image_size,
							stream);
				}
			}
			else if (translation_num <= blocks3D*4)
			{
				if (even_orientation_num != 0)
				{
					if(data_is_3D)
						AccUtilities::diff2_coarse<true,true, D2C_BLOCK_SIZE_DATA3D*4, D2C_EULERS_PER_BLOCK_DATA3D, 4>(
							even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_DATA3D,
							D2C_BLOCK_SIZE_DATA3D*4,
							d_eulers,
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							diff2s,
							translation_num,
							image_size,
							stream);
					else
						AccUtilities::diff2_coarse<true,false, D2C_BLOCK_SIZE_REF3D*4, D2C_EULERS_PER_BLOCK_REF3D, 4>(
							even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_REF3D,
							D2C_BLOCK_SIZE_REF3D*4,
							d_eulers,
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							diff2s,
							translation_num,
							image_size,
							stream);
				}

				if (rest != 0)
				{
					if(data_is_3D)
						AccUtilities::diff2_coarse<true,true, D2C_BLOCK_SIZE_DATA3D*4, 1, 4>(
							rest,
							D2C_BLOCK_SIZE_DATA3D*4,
							&d_eulers[9*even_orientation_num],
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							&diff2s[translation_num*even_orientation_num],
							translation_num,
							image_size,
							stream);
					else
						AccUtilities::diff2_coarse<true,false, D2C_BLOCK_SIZE_REF3D*4, 1, 4>(
							rest,
							D2C_BLOCK_SIZE_REF3D*4,
							&d_eulers[9*even_orientation_num],
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							&diff2s[translation_num*even_orientation_num],
							translation_num,
							image_size,
							stream);
				}
			}
#ifndef ACC_DOUBLE_PRECISION
			else
			{
				if (even_orientation_num != 0)
				{
					if(data_is_3D)
						AccUtilities::diff2_coarse<true,true, D2C_BLOCK_SIZE_DATA3D*8, D2C_EULERS_PER_BLOCK_DATA3D, 4>(
							even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_DATA3D,
							D2C_BLOCK_SIZE_DATA3D*8,
							d_eulers,
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							diff2s,
							translation_num,
							image_size,
							stream);
					else
						AccUtilities::diff2_coarse<true,false, D2C_BLOCK_SIZE_REF3D*8, D2C_EULERS_PER_BLOCK_REF3D, 4>(
							even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_REF3D,
							D2C_BLOCK_SIZE_REF3D*8,
							d_eulers,
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							diff2s,
							translation_num,
							image_size,
							stream);
				}

				if (rest != 0)
				{
					if(data_is_3D)
						AccUtilities::diff2_coarse<true,true, D2C_BLOCK_SIZE_DATA3D*8, 1, 4>(
							rest,
							D2C_BLOCK_SIZE_DATA3D*8,
							&d_eulers[9*even_orientation_num],
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							&diff2s[translation_num*even_orientation_num],
							translation_num,
							image_size,
							stream);
					else
						AccUtilities::diff2_coarse<true,false, D2C_BLOCK_SIZE_REF3D*8, 1, 4>(
							rest,
							D2C_BLOCK_SIZE_REF3D*8,
							&d_eulers[9*even_orientation_num],
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							&diff2s[translation_num*even_orientation_num],
							translation_num,
							image_size,
							stream);
				}
			}
#endif
#endif  // End of non-SYCL code
		}  // projector.mdlZ!=0
		else
		{
#ifdef _SYCL_ENABLED
			assert(translation_num <= D2C_BLOCK_SIZE_2D);

			long unsigned rest = orientation_num % (unsigned long)D2C_EULERS_PER_BLOCK_2D;
			long unsigned even_orientation_num = orientation_num - rest;
			long unsigned grid_size = (D2C_EULERS_PER_BLOCK_2D == 1) ? orientation_num : even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_2D;

			if (even_orientation_num)
			{
				if(data_is_3D)
					AccUtilities::diff2_coarse<false,true, D2C_BLOCK_SIZE_2D, D2C_EULERS_PER_BLOCK_2D, PREFETCH_FRACTION_2D>(
						grid_size,
						D2C_BLOCK_SIZE_2D,
						d_eulers,
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						diff2s,
						translation_num,
						image_size,
						stream);
				else
					AccUtilities::diff2_coarse<false,false, D2C_BLOCK_SIZE_2D, D2C_EULERS_PER_BLOCK_2D, PREFETCH_FRACTION_2D>(
						grid_size,
						D2C_BLOCK_SIZE_2D,
						d_eulers,
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						diff2s,
						translation_num,
						image_size,
						stream);
			}

			if (rest && D2C_EULERS_PER_BLOCK_2D != 1)
			{
				if (data_is_3D)
					AccUtilities::diff2_coarse<false,true, D2C_BLOCK_SIZE_2D, 1, PREFETCH_FRACTION_2D>(
						rest,
						D2C_BLOCK_SIZE_2D,
						&d_eulers[9*even_orientation_num],
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						&diff2s[translation_num*even_orientation_num],
						translation_num,
						image_size,
						stream);
				else
					AccUtilities::diff2_coarse<false,false, D2C_BLOCK_SIZE_2D, 1, PREFETCH_FRACTION_2D>(
						rest,
						D2C_BLOCK_SIZE_2D,
						&d_eulers[9*even_orientation_num],
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						&diff2s[translation_num*even_orientation_num],
						translation_num,
						image_size,
						stream);
			}
#else
			if (translation_num > D2C_BLOCK_SIZE_2D)
			{
				printf("Number of coarse translations larger than %d on the GPU not supported.\n", D2C_BLOCK_SIZE_2D);
				fflush(stdout);
				exit(1);
			}


			long unsigned rest = orientation_num % (unsigned long)D2C_EULERS_PER_BLOCK_2D;
			long unsigned even_orientation_num = orientation_num - rest;

			if (even_orientation_num != 0)
			{
				if(data_is_3D)
					AccUtilities::diff2_coarse<false,true, D2C_BLOCK_SIZE_2D, D2C_EULERS_PER_BLOCK_2D, 2>(
						even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_2D,
						D2C_BLOCK_SIZE_2D,
						d_eulers,
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						diff2s,
						translation_num,
						image_size,
						stream);
				else
					AccUtilities::diff2_coarse<false,false, D2C_BLOCK_SIZE_2D, D2C_EULERS_PER_BLOCK_2D, 2>(
						even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_2D,
						D2C_BLOCK_SIZE_2D,
						d_eulers,
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						diff2s,
						translation_num,
						image_size,
						stream);
				#ifdef _CUDA_ENABLED
				LAUNCH_HANDLE_ERROR(cudaGetLastError());
				#elif _HIP_ENABLED
				LAUNCH_HANDLE_ERROR(hipGetLastError());
				#endif
			}

			if (rest != 0)
			{
				if(data_is_3D)
					AccUtilities::diff2_coarse<false,true, D2C_BLOCK_SIZE_2D, 1, 2>(
						rest,
						D2C_BLOCK_SIZE_2D,
						&d_eulers[9*even_orientation_num],
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						&diff2s[translation_num*even_orientation_num],
						translation_num,
						image_size,
						stream);
				else
					AccUtilities::diff2_coarse<false,false, D2C_BLOCK_SIZE_2D, 1, 2>(
						rest,
						D2C_BLOCK_SIZE_2D,
						&d_eulers[9*even_orientation_num],
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						&diff2s[translation_num*even_orientation_num],
						translation_num,
						image_size,
						stream);
				#ifdef _CUDA_ENABLED
				LAUNCH_HANDLE_ERROR(cudaGetLastError());
				#elif _HIP_ENABLED
				LAUNCH_HANDLE_ERROR(hipGetLastError());
				#endif
			}
#endif  // End of non-SYCL code
		}  // projector.mdlZ==0
	}  // !do_CC
	else
	{  // do_CC
// TODO - find a more compact way to represent these combinations resulting in
// a single call to diff2_CC_course?
		// dim3 CCblocks(orientation_num,translation_num);
		if(data_is_3D)
#if defined ALTCPU || defined _SYCL_ENABLED
			AccUtilities::diff2_CC_coarse<true,true,D2C_CC_BLOCK_SIZE_DATA3D>(
#else
			AccUtilities::diff2_CC_coarse<true,true,D2C_BLOCK_SIZE_DATA3D>(
#endif
				orientation_num,
				D2C_BLOCK_SIZE_DATA3D,
				d_eulers,
				Fimg_real,
				Fimg_imag,
				trans_x,
				trans_y,
				trans_z,
				projector,
				corr_img,
				diff2s,
				translation_num,
				image_size,
				local_sqrtXi2,
				stream);
		else if(projector.mdlZ!=0)
#if defined ALTCPU || defined _SYCL_ENABLED
			AccUtilities::diff2_CC_coarse<true,false,D2C_CC_BLOCK_SIZE_REF3D>(
#else
			AccUtilities::diff2_CC_coarse<true,false,D2C_BLOCK_SIZE_REF3D>(
#endif
				orientation_num,
				D2C_BLOCK_SIZE_REF3D,
				d_eulers,
				Fimg_real,
				Fimg_imag,
				trans_x,
				trans_y,
				trans_z,
				projector,
				corr_img,
				diff2s,
				translation_num,
				image_size,
				local_sqrtXi2,
				stream);
		else
#if defined ALTCPU || defined _SYCL_ENABLED
			AccUtilities::diff2_CC_coarse<false,false,D2C_CC_BLOCK_SIZE_2D>(
#else
			AccUtilities::diff2_CC_coarse<false,false,D2C_BLOCK_SIZE_2D>(
#endif
				orientation_num,
				D2C_BLOCK_SIZE_2D,
				d_eulers,
				Fimg_real,
				Fimg_imag,
				trans_x,
				trans_y,
				trans_z,
				projector,
				corr_img,
				diff2s,
				translation_num,
				image_size,
				local_sqrtXi2,
				stream);
		#ifdef _CUDA_ENABLED
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
		#elif _HIP_ENABLED
		LAUNCH_HANDLE_ERROR(hipGetLastError());
		#endif
	} // do_CC
}


void runDiff2KernelFine(
		AccProjectorKernel &projector,
		XFLOAT *corr_img,
		XFLOAT *Fimgs_real,
		XFLOAT *Fimgs_imag,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT *eulers,
		long unsigned *rot_id,
		long unsigned *rot_idx,
		long unsigned *trans_idx,
		long unsigned *job_idx,
		long unsigned *job_num,
		XFLOAT *diff2s,
		OptimisationParamters &op,
		MlOptimiser *baseMLO,
		long unsigned orientation_num,
		long unsigned translation_num,
		long unsigned significant_num,
		unsigned long image_size,
		int img_id,
		int exp_iclass,
		deviceStream_t stream,
		long unsigned job_num_count,
		bool do_CC,
		bool data_is_3D)
{
    long unsigned block_dim = job_num_count;

    if(!do_CC)
    {
// TODO - find a more compact way to represent these combinations resulting in
// a single call to diff2_fine?
			if(data_is_3D)
				AccUtilities::diff2_fine<true,true, D2F_BLOCK_SIZE_DATA3D, D2F_CHUNK_DATA3D>(
					block_dim,
					D2F_BLOCK_SIZE_DATA3D,
					eulers,
					Fimgs_real,
					Fimgs_imag,
					trans_x,
					trans_y,
					trans_z,
					projector,
					corr_img,    // in these non-CC kernels this is effectively an adjusted MinvSigma2
					diff2s,
					image_size,
					op.highres_Xi2_img[img_id] / 2.,
					orientation_num,
					translation_num,
					job_num_count, //significant_num,
					rot_idx,
					trans_idx,
					job_idx,
					job_num,
					stream);
			else if(projector.mdlZ!=0)
				AccUtilities::diff2_fine<true,false,D2F_BLOCK_SIZE_REF3D,D2F_CHUNK_REF3D>(
					block_dim,
					D2F_BLOCK_SIZE_REF3D,
					eulers,
					Fimgs_real,
					Fimgs_imag,
					trans_x,
					trans_y,
					trans_z,
					projector,
					corr_img,    // in these non-CC kernels this is effectively an adjusted MinvSigma2
					diff2s,
					image_size,
					op.highres_Xi2_img[img_id] / 2.,
					orientation_num,
					translation_num,
					job_num_count, //significant_num,
					rot_idx,
					trans_idx,
					job_idx,
					job_num,
					stream);
			else
				AccUtilities::diff2_fine<false,false,D2F_BLOCK_SIZE_2D,D2F_CHUNK_2D>(
					block_dim,
					D2F_BLOCK_SIZE_2D,
					eulers,
					Fimgs_real,
					Fimgs_imag,
					trans_x,
					trans_y,
					trans_z,
					projector,
					corr_img,    // in these non-CC kernels this is effectively an adjusted MinvSigma2
					diff2s,
					image_size,
					op.highres_Xi2_img[img_id] / 2.,
					orientation_num,
					translation_num,
					job_num_count, //significant_num,
					rot_idx,
					trans_idx,
					job_idx,
					job_num,
					stream);
		#ifdef _CUDA_ENABLED
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
		#elif _HIP_ENABLED
		LAUNCH_HANDLE_ERROR(hipGetLastError());
		#endif
    }
    else
    {
// TODO - find a more compact way to represent these combinations resulting in
// a single call to diff2_CC_fine?
    	if(data_is_3D)
			AccUtilities::diff2_CC_fine<true,true,D2F_BLOCK_SIZE_DATA3D,D2F_CHUNK_DATA3D>(
				block_dim,
				D2F_BLOCK_SIZE_DATA3D,
				eulers,
				Fimgs_real,
				Fimgs_imag,
				trans_x,
				trans_y,
				trans_z,
				projector,
				corr_img,
				diff2s,
				image_size,
				op.highres_Xi2_img[img_id] / 2.,
				(XFLOAT) op.local_sqrtXi2[img_id],
				orientation_num,
				translation_num,
				job_num_count, //significant_num,
				rot_idx,
				trans_idx,
				job_idx,
				job_num,
				stream);
    	else if(projector.mdlZ!=0)
			AccUtilities::diff2_CC_fine<true,false,D2F_BLOCK_SIZE_REF3D,D2F_CHUNK_REF3D>(
				block_dim,
				D2F_BLOCK_SIZE_REF3D,
				eulers,
				Fimgs_real,
				Fimgs_imag,
				trans_x,
				trans_y,
				trans_z,
				projector,
				corr_img,
				diff2s,
				image_size,
				op.highres_Xi2_img[img_id] / 2.,
				(XFLOAT) op.local_sqrtXi2[img_id],
				orientation_num,
				translation_num,
				job_num_count, //significant_num,
				rot_idx,
				trans_idx,
				job_idx,
				job_num,
				stream);
		else
			AccUtilities::diff2_CC_fine<false,false,D2F_BLOCK_SIZE_2D,D2F_CHUNK_2D>(
				block_dim,
				D2F_BLOCK_SIZE_2D,
				eulers,
				Fimgs_real,
				Fimgs_imag,
				trans_x,
				trans_y,
				trans_z,
				projector,
				corr_img,
				diff2s,
				image_size,
				op.highres_Xi2_img[img_id] / 2.,
				(XFLOAT) op.local_sqrtXi2[img_id],
				orientation_num,
				translation_num,
				job_num_count, //significant_num,
				rot_idx,
				trans_idx,
				job_idx,
				job_num,
				stream);
		#ifdef _CUDA_ENABLED
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
		#elif _HIP_ENABLED
		LAUNCH_HANDLE_ERROR(hipGetLastError());
		#endif
    }

}

void runCollect2jobs(	int grid_dim,
						XFLOAT * oo_otrans_x,          // otrans-size -> make const
						XFLOAT * oo_otrans_y,          // otrans-size -> make const
						XFLOAT * oo_otrans_z,          // otrans-size -> make const
						XFLOAT * myp_oo_otrans_x2y2z2, // otrans-size -> make const
						XFLOAT * weights,
						XFLOAT significant_weight,
						XFLOAT sum_weight,
						unsigned long nr_trans,
						unsigned long nr_oversampled_trans,
						unsigned long nr_oversampled_rot,
						unsigned long oversamples,
						bool skip_rots,
						XFLOAT * p_weights,
						XFLOAT * p_thr_wsum_prior_offsetx_class,
						XFLOAT * p_thr_wsum_prior_offsety_class,
						XFLOAT * p_thr_wsum_prior_offsetz_class,
						XFLOAT * p_thr_wsum_sigma2_offset,
						size_t * rot_idx,
						size_t * trans_idx,
						size_t * jobOrigin,
						size_t * jobExtent,
						bool data_is_3D,
						deviceStream_t stream)
{
	if (data_is_3D) {
#ifdef _CUDA_ENABLED
	dim3 numblocks(grid_dim);
	size_t shared_buffer = sizeof(XFLOAT)*SUMW_BLOCK_SIZE*5; // x+y+z+myp+weights
	cuda_kernel_collect2jobs<true><<<numblocks,SUMW_BLOCK_SIZE,shared_buffer,stream>>>(
			oo_otrans_x,          // otrans-size -> make const
			oo_otrans_y,          // otrans-size -> make const
			oo_otrans_z,          // otrans-size -> make const
			myp_oo_otrans_x2y2z2, // otrans-size -> make const
			weights,
			significant_weight,
			sum_weight,
			nr_trans,
			nr_oversampled_trans,
			nr_oversampled_rot,
			oversamples,
			skip_rots,
			p_weights,
			p_thr_wsum_prior_offsetx_class,
			p_thr_wsum_prior_offsety_class,
			p_thr_wsum_prior_offsetz_class,
			p_thr_wsum_sigma2_offset,
			rot_idx,
			trans_idx,
			jobOrigin,
			jobExtent);
#elif _HIP_ENABLED
	dim3 numblocks(grid_dim);
	size_t shared_buffer = sizeof(XFLOAT)*SUMW_BLOCK_SIZE*5; // x+y+z+myp+weights
	hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_collect2jobs<true>), numblocks, dim3(SUMW_BLOCK_SIZE), shared_buffer, stream,
			oo_otrans_x,          // otrans-size -> make const
			oo_otrans_y,          // otrans-size -> make const
			oo_otrans_z,          // otrans-size -> make const
			myp_oo_otrans_x2y2z2, // otrans-size -> make const
			weights,
			significant_weight,
			sum_weight,
			nr_trans,
			nr_oversampled_trans,
			nr_oversampled_rot,
			oversamples,
			skip_rots,
			p_weights,
			p_thr_wsum_prior_offsetx_class,
			p_thr_wsum_prior_offsety_class,
			p_thr_wsum_prior_offsetz_class,
			p_thr_wsum_sigma2_offset,
			rot_idx,
			trans_idx,
			jobOrigin,
			jobExtent);
#elif _SYCL_ENABLED
	syclKernels::collect2jobs<true>(grid_dim, SUMW_BLOCK_SIZE,
				oo_otrans_x,          // otrans-size -> make const
				oo_otrans_y,          // otrans-size -> make const
				oo_otrans_z,          // otrans-size -> make const
				myp_oo_otrans_x2y2z2, // otrans-size -> make const
				weights,
				significant_weight,
				sum_weight,
				nr_trans,
				nr_oversampled_trans,
				nr_oversampled_rot,
				oversamples,
				skip_rots,
				p_weights,
				p_thr_wsum_prior_offsetx_class,
				p_thr_wsum_prior_offsety_class,
				p_thr_wsum_prior_offsetz_class,
				p_thr_wsum_sigma2_offset,
				rot_idx,
				trans_idx,
				jobOrigin,
				jobExtent);
#else
		CpuKernels::collect2jobs<true>(grid_dim, SUMW_BLOCK_SIZE,
				oo_otrans_x,          // otrans-size -> make const
				oo_otrans_y,          // otrans-size -> make const
				oo_otrans_z,          // otrans-size -> make const
				myp_oo_otrans_x2y2z2, // otrans-size -> make const
				weights,
				significant_weight,
				sum_weight,
				nr_trans,
				nr_oversampled_trans,
				nr_oversampled_rot,
				oversamples,
				skip_rots,
				p_weights,
				p_thr_wsum_prior_offsetx_class,
				p_thr_wsum_prior_offsety_class,
				p_thr_wsum_prior_offsetz_class,
				p_thr_wsum_sigma2_offset,
				rot_idx,
				trans_idx,
				jobOrigin,
				jobExtent);
#endif
	}
	else
	{
#ifdef _CUDA_ENABLED
	dim3 numblocks(grid_dim);
	size_t shared_buffer = sizeof(XFLOAT)*SUMW_BLOCK_SIZE*4; // x+y+myp+weights
	cuda_kernel_collect2jobs<false><<<numblocks,SUMW_BLOCK_SIZE,shared_buffer,stream>>>(
			oo_otrans_x,          // otrans-size -> make const
			oo_otrans_y,          // otrans-size -> make const
			oo_otrans_z,          // otrans-size -> make const
			myp_oo_otrans_x2y2z2, // otrans-size -> make const
			weights,
			significant_weight,
			sum_weight,
			nr_trans,
			nr_oversampled_trans,
			nr_oversampled_rot,
			oversamples,
			skip_rots,
			p_weights,
			p_thr_wsum_prior_offsetx_class,
			p_thr_wsum_prior_offsety_class,
			p_thr_wsum_prior_offsetz_class,
			p_thr_wsum_sigma2_offset,
			rot_idx,
			trans_idx,
			jobOrigin,
			jobExtent);
#elif _HIP_ENABLED
	dim3 numblocks(grid_dim);
	size_t shared_buffer = sizeof(XFLOAT)*SUMW_BLOCK_SIZE*4; // x+y+myp+weights
	hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_collect2jobs<false>), numblocks, dim3(SUMW_BLOCK_SIZE), shared_buffer, stream,
			oo_otrans_x,          // otrans-size -> make const
			oo_otrans_y,          // otrans-size -> make const
			oo_otrans_z,          // otrans-size -> make const
			myp_oo_otrans_x2y2z2, // otrans-size -> make const
			weights,
			significant_weight,
			sum_weight,
			nr_trans,
			nr_oversampled_trans,
			nr_oversampled_rot,
			oversamples,
			skip_rots,
			p_weights,
			p_thr_wsum_prior_offsetx_class,
			p_thr_wsum_prior_offsety_class,
			p_thr_wsum_prior_offsetz_class,
			p_thr_wsum_sigma2_offset,
			rot_idx,
			trans_idx,
			jobOrigin,
			jobExtent);
#elif _SYCL_ENABLED
	syclKernels::collect2jobs<false>(grid_dim, SUMW_BLOCK_SIZE,
			oo_otrans_x,          // otrans-size -> make const
			oo_otrans_y,          // otrans-size -> make const
			oo_otrans_z,          // otrans-size -> make const
			myp_oo_otrans_x2y2z2, // otrans-size -> make const
			weights,
			significant_weight,
			sum_weight,
			nr_trans,
			nr_oversampled_trans,
			nr_oversampled_rot,
			oversamples,
			skip_rots,
			p_weights,
			p_thr_wsum_prior_offsetx_class,
			p_thr_wsum_prior_offsety_class,
			p_thr_wsum_prior_offsetz_class,
			p_thr_wsum_sigma2_offset,
			rot_idx,
			trans_idx,
			jobOrigin,
			jobExtent);
#else
	CpuKernels::collect2jobs<false>(grid_dim, SUMW_BLOCK_SIZE,
			oo_otrans_x,          // otrans-size -> make const
			oo_otrans_y,          // otrans-size -> make const
			oo_otrans_z,          // otrans-size -> make const
			myp_oo_otrans_x2y2z2, // otrans-size -> make const
			weights,
			significant_weight,
			sum_weight,
			nr_trans,
			nr_oversampled_trans,
			nr_oversampled_rot,
			oversamples,
			skip_rots,
			p_weights,
			p_thr_wsum_prior_offsetx_class,
			p_thr_wsum_prior_offsety_class,
			p_thr_wsum_prior_offsetz_class,
			p_thr_wsum_sigma2_offset,
			rot_idx,
			trans_idx,
			jobOrigin,
			jobExtent);
#endif
	}
}

#define WINDOW_FT_BLOCK_SIZE 128

void windowFourierTransform2(
		AccPtr<ACCCOMPLEX > &d_in,
		AccPtr<ACCCOMPLEX > &d_out,
		size_t iX, size_t iY, size_t iZ, //Input dimensions
		size_t oX, size_t oY, size_t oZ,  //Output dimensions
		size_t Npsi,
		size_t pos,
		deviceStream_t stream)
{
	if (iX > 1 && iY/2 + 1 != iX)
		REPORT_ERROR("windowFourierTransform ERROR: the Fourier transform should be of an image with equal sizes in all dimensions!");

//	if (oX == iX)
//		REPORT_ERROR("windowFourierTransform ERROR: there is a one-to-one map between input and output!");


#ifdef _CUDA_ENABLED
	// Only enlarging leaves output values unwritten (copying and shrinking write all)
	if (oX > iX)
#endif
	deviceInitComplexValue<ACCCOMPLEX>(d_out, (XFLOAT)0.);
	// CUDA: no wait after the clear, the copy or kernel below follows it on d_out's stream
#ifdef _HIP_ENABLED
	HANDLE_ERROR(hipStreamSynchronize(d_out.getStream()));
#endif
	if(oX==iX)
	{
#ifdef _CUDA_ENABLED
		if (d_in.getStream() != d_out.getStream())
			HANDLE_ERROR(cudaStreamSynchronize(d_in.getStream()));
		cudaCpyDeviceToDevice(&d_in(pos), ~d_out, oX*oY*oZ*Npsi, d_out.getStream() );
#elif _HIP_ENABLED
		HANDLE_ERROR(hipStreamSynchronize(d_in.getStream()));
		hipCpyDeviceToDevice(&d_in(pos), ~d_out, oX*oY*oZ*Npsi, d_out.getStream() );
#else
		memcpy(&d_out[0], &d_in[0], oX*oY*oZ*Npsi*sizeof(ACCCOMPLEX));
#endif
		return;
	}

	if (oX > iX)
	{
		long int max_r2 = (iX - 1) * (iX - 1);

#ifdef _CUDA_ENABLED
		dim3 grid_dim(ceil((float)(iX*iY*iZ) / (float) WINDOW_FT_BLOCK_SIZE),Npsi);
		cuda_kernel_window_fourier_transform<true><<< grid_dim, WINDOW_FT_BLOCK_SIZE, 0, d_out.getStream() >>>(
				&d_in(pos),
				~d_out,
				iX, iY, iZ, iX * iY, //Input dimensions
				oX, oY, oZ, oX * oY, //Output dimensions
				iX*iY*iZ,
				WINDOW_FT_BLOCK_SIZE,
				max_r2);
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
#elif _HIP_ENABLED
		dim3 grid_dim(ceil((float)(iX*iY*iZ) / (float) WINDOW_FT_BLOCK_SIZE),Npsi);
		hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_window_fourier_transform<true>), grid_dim, dim3(WINDOW_FT_BLOCK_SIZE), 0, d_out.getStream(),
				&d_in(pos),
				~d_out,
				iX, iY, iZ, iX * iY, //Input dimensions
				oX, oY, oZ, oX * oY, //Output dimensions
				iX*iY*iZ,
				WINDOW_FT_BLOCK_SIZE,
				max_r2);
		LAUNCH_HANDLE_ERROR(hipGetLastError());
#elif _SYCL_ENABLED
		size_t grid_dim = (size_t)( ceil((float)(iX*iY*iZ) / (float) WINDOW_FT_BLOCK_SIZE));
		syclKernels::window_fourier_transform<true>(
				grid_dim,
				Npsi,
				WINDOW_FT_BLOCK_SIZE,
				&d_in[pos],
				&d_out[0],
				iX, iY, iZ, iX * iY, //Input dimensions
				oX, oY, oZ, oX * oY, //Output dimensions
				iX*iY*iZ,
				max_r2);
#else
		size_t grid_dim = (size_t)( ceil((float)(iX*iY*iZ) / (float) WINDOW_FT_BLOCK_SIZE));
		CpuKernels::window_fourier_transform<true>(
				grid_dim,
				Npsi,
				WINDOW_FT_BLOCK_SIZE,
				&d_in[pos],
				&d_out[0],
				iX, iY, iZ, iX * iY, //Input dimensions
				oX, oY, oZ, oX * oY, //Output dimensions
				iX*iY*iZ,
				max_r2);
#endif
	}
	else
	{
#ifdef _CUDA_ENABLED
		dim3 grid_dim(ceil((float)(oX*oY*oZ) / (float) WINDOW_FT_BLOCK_SIZE),Npsi);
		cuda_kernel_window_fourier_transform<false><<< grid_dim, WINDOW_FT_BLOCK_SIZE, 0, d_out.getStream() >>>(
				&d_in(pos),
				~d_out,
				iX, iY, iZ, iX * iY, //Input dimensions
				oX, oY, oZ, oX * oY, //Output dimensions
				oX*oY*oZ,
				WINDOW_FT_BLOCK_SIZE);
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
#elif _HIP_ENABLED
		dim3 grid_dim(ceil((float)(oX*oY*oZ) / (float) WINDOW_FT_BLOCK_SIZE),Npsi);
		hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_window_fourier_transform<false>), grid_dim, dim3(WINDOW_FT_BLOCK_SIZE), 0, d_out.getStream(),
				&d_in(pos),
				~d_out,
				iX, iY, iZ, iX * iY, //Input dimensions
				oX, oY, oZ, oX * oY, //Output dimensions
				oX*oY*oZ,
				WINDOW_FT_BLOCK_SIZE);
		LAUNCH_HANDLE_ERROR(hipGetLastError());
#elif _SYCL_ENABLED
		int grid_dim = (int)( ceil((float)(oX*oY*oZ) / (float) WINDOW_FT_BLOCK_SIZE));
		syclKernels::window_fourier_transform<false>(
				grid_dim,
				Npsi,
				WINDOW_FT_BLOCK_SIZE,
				&d_in[pos],
				&d_out[0],
				iX, iY, iZ, iX * iY, //Input dimensions
				oX, oY, oZ, oX * oY, //Output dimensions
				oX*oY*oZ
				);
#else
		int grid_dim = (int)( ceil((float)(oX*oY*oZ) / (float) WINDOW_FT_BLOCK_SIZE));
		CpuKernels::window_fourier_transform<false>(
				grid_dim,
				Npsi,
				WINDOW_FT_BLOCK_SIZE,
				&d_in[pos],
				&d_out[0],
				iX, iY, iZ, iX * iY, //Input dimensions
				oX, oY, oZ, oX * oY, //Output dimensions
				oX*oY*oZ
				);
#endif
	}
}

void run_calcPowerSpectrum(Complex *dFaux, int padoridim, Complex *ddata, int data_sz, RFLOAT *dpower_spectrum, RFLOAT *dcounter,
											  int max_r2, int min_r2, RFLOAT normfft, RFLOAT padding_factor, RFLOAT weight,
											  RFLOAT *dfourier_mask, int fx, int fy, int fz, bool do_fourier_mask, bool if3D)
{
#ifdef _CUDA_ENABLED
	dim3 bs(32,4);
	dim3 gs(ceil((padoridim/2+1)/(float)bs.x), ceil(padoridim/(float)bs.y));
	if(if3D)
	{
		bs.z = 2;
		gs.z = ceil(padoridim/(float)bs.z);
	}
	if(sizeof(RFLOAT) == sizeof(double))
		cuda_kernel_calcPowerSpectrum<<<gs,bs>>>((double2*)dFaux,padoridim,(double2*)ddata,data_sz,dpower_spectrum,dcounter,
												  max_r2,min_r2,normfft,padding_factor,weight,dfourier_mask,fx,fy,fz,do_fourier_mask);
	else
		cuda_kernel_calcPowerSpectrum<<<gs,bs>>>((float2*)dFaux,padoridim,(float2*)ddata,data_sz,dpower_spectrum,dcounter,
												  max_r2,min_r2,normfft,padding_factor,weight,dfourier_mask,fx,fy,fz,do_fourier_mask);
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
#elif _HIP_ENABLED
	dim3 bs(32,4);
	dim3 gs(ceil((padoridim/2+1)/(float)bs.x), ceil(padoridim/(float)bs.y));
	if(if3D)
	{
		bs.z = 2;
		gs.z = ceil(padoridim/(float)bs.z);
	}
	if(sizeof(RFLOAT) == sizeof(double))
		hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_calcPowerSpectrum), gs, bs, 0, 0, (double2*)dFaux,padoridim,(double2*)ddata,data_sz,dpower_spectrum,dcounter,
												  max_r2,min_r2,normfft,padding_factor,weight,dfourier_mask,fx,fy,fz,do_fourier_mask);
	else
		hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_calcPowerSpectrum), gs, bs, 0, 0, (float2*)dFaux,padoridim,(float2*)ddata,data_sz,dpower_spectrum,dcounter,
												  max_r2,min_r2,normfft,padding_factor,weight,dfourier_mask,fx,fy,fz,do_fourier_mask);
	LAUNCH_HANDLE_ERROR(hipGetLastError());
#endif
}

void run_updatePowerSpectrum(RFLOAT *dcounter, int sz, RFLOAT *dpower_spectrum)
{
#ifdef _CUDA_ENABLED
	cuda_kernel_updatePowerSpectrum<<<ceil(sz/(float)256),256>>>(dcounter, dpower_spectrum, sz);
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
#elif _HIP_ENABLED
	hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_kernel_updatePowerSpectrum), dim3(ceil(sz/(float)256)), dim3(256), 0, 0, dcounter, dpower_spectrum, sz);
	LAUNCH_HANDLE_ERROR(hipGetLastError());
#endif
}

void scale(RFLOAT *img, size_t sz, RFLOAT val, deviceStream_t stream)
{
	int block_size = 256;
	int MultiBsize = ceil(sz/(float)block_size);
#if defined _CUDA_ENABLED || defined _HIP_ENABLED
	AccUtilities::multiply(MultiBsize,block_size, stream, img, val, (size_t)sz);
#endif
}

void selfApplyBeamTilt2(MultidimArray<Complex > &Fimg, RFLOAT beamtilt_x, RFLOAT beamtilt_y,
		RFLOAT wavelength, RFLOAT Cs, RFLOAT angpix, int ori_size)
{
	if (Fimg.getDim() != 2)
		REPORT_ERROR("applyBeamTilt can only be done on 2D Fourier Transforms!");

	RFLOAT boxsize = angpix * ori_size;
	RFLOAT factor = 0.360 * Cs * 10000000 * wavelength * wavelength / (boxsize * boxsize * boxsize);

	for (unsigned n = 0 ; n < Fimg.yxdim; n ++)
	{
		unsigned i = n / Fimg.xdim;
		unsigned j = n % Fimg.xdim;
		unsigned jp = j;
		int ip = i < Fimg.xdim ? i : i - Fimg.ydim;

		RFLOAT delta_phase = factor * (ip * ip + jp * jp) * (ip * beamtilt_y + jp * beamtilt_x);
		RFLOAT realval = Fimg.data[i*Fimg.xdim+j].real;
		RFLOAT imagval = Fimg.data[i*Fimg.xdim+j].imag;
		RFLOAT mag = sqrt(realval * realval + imagval * imagval);
		RFLOAT phas = atan2(imagval, realval) + DEG2RAD(delta_phase); // apply phase shift!
		realval = mag * cos(phas);
		imagval = mag * sin(phas);
		Fimg.data[i*Fimg.xdim+j] = Complex(realval, imagval);

	}

}
