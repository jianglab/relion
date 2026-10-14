#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include "src/gpu_switch.h"
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

// Blocks per multiprocessor that pixel chunks aim for. RELION_GPU_BLOCKS_PER_SM=N overrides. By
// default one full wave of 128-thread blocks per multiprocessor (as many as the device keeps
// resident: 12 on sm_86, 16 on sm_80/90, 8 on sm_75), and the old 3 when the GPU has under 2 GiB
// free, because the chunks need partial-sum buffers. Chosen once per device and said once.
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
		if (threads > 0 && free_b >= ((size_t) 2 << 30))
		{
			b = (size_t) std::max(3, threads / 128);
			why = "device " + std::to_string(dev) + ": " + std::to_string(sms) + " multiprocessors, up to " +
			      std::to_string(threads) + " threads each";
		}
		else
			why = "device " + std::to_string(dev) + " has " + std::to_string(free_b >> 20) + " MiB free, safe default";
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

// Orientations per block of cuda_kernel_shared_sums
#define SHSUM_G 8

bool runSharedSumsKernelBatched(
		std::vector<SharedSumsClassJob> &jobs,
		AccBackprojector &BP,
		XFLOAT *d_img_real,
		XFLOAT *d_img_imag,
		XFLOAT *d_img_real_nomask,
		XFLOAT *d_img_imag_nomask,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *d_ctfs,
		XFLOAT *d_Minvsigma2s,
		XFLOAT *wdiff2s_parts,
		unsigned long translation_num,
		XFLOAT weight_norm,
		XFLOAT significant_weight,
		XFLOAT part_scale,
		unsigned long image_size,
		int imgX,
		int imgY,
		bool refs_are_ctf_corrected,
		bool ctf_premultiplied,
		int parts,
		AccPtr<char> &table)
{
#ifdef ACC_DOUBLE_PRECISION
	return false;
#else
	if (jobs.empty())
		return false;
	const int max_r = imgX - 1; // the projector's maxR
	// The kernels agree on the pixel coordinates (and one table serves both) when the
	// image has an even size and maxR is half of it; the sums need shared memory
	if ((imgY % 2) != 0 || max_r != imgY / 2 || translation_num == 0 || translation_num > 1024)
		return false;
	std::vector<size_t> groups(jobs.size());
	for (size_t c = 0; c < jobs.size(); c++)
		groups[c] = (jobs[c].n_orient + SHSUM_G - 1) / SHSUM_G;
	const SharedSumsClassJob *d_jobs;
	const int *d_start;
	const size_t n = uploadClassTable(jobs, groups, table, d_jobs, d_start);
	deviceStream_t stream = table.getStream();

	// The shifts, once for all orientations
	AccPtr<float2> shifts = table.make<float2>();
	shifts.setSize(translation_num * image_size);
	shifts.deviceAlloc();
	cuda_kernel_shift_table<<<dim3((image_size + SHSUM_THREADS - 1) / SHSUM_THREADS, translation_num), SHSUM_THREADS, 0, stream>>>(
			trans_x, trans_y, ~shifts, image_size, imgX, imgY, max_r);

	const dim3 grid(n, (image_size + SHSUM_THREADS - 1) / SHSUM_THREADS);
	const size_t smem = (size_t) SHSUM_G * translation_num * sizeof(XFLOAT);
	const int bp_max_r2 = BP.maxR2;
#define SHSUM_LAUNCH(RC, CP) cuda_kernel_shared_sums<SHSUM_G, RC, CP><<<grid, SHSUM_THREADS, smem, stream>>>( \
		~shifts, d_img_real, d_img_imag, d_img_real_nomask, d_img_imag_nomask, d_ctfs, d_Minvsigma2s, wdiff2s_parts, \
		translation_num, weight_norm, significant_weight, part_scale, image_size, imgX, imgY, max_r, bp_max_r2, \
		BP.padding_factor, BP.mdlX, BP.mdlY, BP.mdlInitY, BP.mdlInitZ, d_jobs, d_start, (int) jobs.size(), parts)
	if (refs_are_ctf_corrected)
	{
		if (ctf_premultiplied) SHSUM_LAUNCH(true, true);
		else                   SHSUM_LAUNCH(true, false);
	}
	else
	{
		if (ctf_premultiplied) SHSUM_LAUNCH(false, true);
		else                   SHSUM_LAUNCH(false, false);
	}
#undef SHSUM_LAUNCH
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
	return true;
#endif
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
		auto launch_bp = [&](int tl) {
#define BP3D_BATCH(D3, CP, BS) cuda_kernel_backproject3D<D3, CP><<<dim3(n, bp_chunks), BS, compact3 ? smem3 : 0, stream>>>( \
			d_img_real, d_img_imag, trans_x, trans_y, trans_z, NULL, d_Minvsigma2s, d_ctfs, \
			translation_num, significant_weight, (XFLOAT) 0, NULL, NULL, NULL, NULL, \
			BP.maxR, BP.maxR2, BP.padding_factor, imgX, imgY, imgZ, imgX*imgY*imgZ, \
			BP.mdlX, BP.mdlY, BP.mdlInitY, BP.mdlInitZ, d_jobs, d_start, nc, bp_chunks, compact3, bp_scatter, acc, d_slots, \
			data_is_3D ? 0 : tl)
		if (data_is_3D)
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
			launch_bp(bp_tile_mode == 1 ? 1 : 0);
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
		CoarseDeferred *defer)
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
	if (gpuShiftTable() && !data_is_3D && image_size * translation_num * sizeof(float2) <= ((size_t) 64 << 20))
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
		if (ref3D || data_is_3D || use_xt)
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
		cuda_kernel_diff2_coarse_xt<<<dim3(even[nc], n_chunks, ntt), XT_BLOCK, 0, stream>>>(
				d_classes, d_even, (int) nc, ~xt_a, ~xt_x2, corr_img, (int) translation_num, (int) xt_tpad,
				(int) image_size, ~partial, n_chunks, chunk_stride, xt_ablate);
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
bool runDiff2KernelCoarseGroup(std::vector<CoarseDeferred *> &items, AccPtr<char> &recs)
{
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
