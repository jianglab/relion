#undef ALTCPU
#include <sys/time.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cuda_runtime.h>
#include <signal.h>

#include "src/ml_optimiser.h"
#include "src/virtual_movie_averages.h"
#include "src/async_file_writer.h"
#include "src/acc/acc_ptr.h"
#include "src/acc/acc_projector.h"
#include "src/acc/acc_backprojector.h"
#include "src/acc/acc_projector_plan.h"
#include "src/acc/cuda/cuda_kernels/helper.cuh"
#include "src/acc/cuda/cuda_mem_utils.h"
#include "src/acc/cuda/cuda_settings.h"
#include "src/acc/cuda/cuda_benchmark_utils.h"
#include "src/acc/cuda/cuda_fft.h"

#include "src/macros.h"
#include "src/error.h"

#ifdef CUDA_FORCESTL
#include "src/acc/cuda/cuda_utils_stl.cuh"
#else
#include "src/acc/cuda/cuda_utils_cub.cuh"
#endif

#include "src/acc/utilities.h"
#include "src/acc/acc_helper_functions.h"

#include "src/acc/cuda/cuda_autopicker.h"


namespace {

template <typename T>
static void fillOutsideWindowWithGaus(T *dst, int xsize, int ysize, int mic_size)
{
	const long int y0 = FIRST_XMIPP_INDEX(mic_size);
	const long int y1 = LAST_XMIPP_INDEX(mic_size);
	const long int x0 = FIRST_XMIPP_INDEX(mic_size);
	const long int x1 = LAST_XMIPP_INDEX(mic_size);
	const long int vy0 = FIRST_XMIPP_INDEX(ysize);
	const long int vy1 = LAST_XMIPP_INDEX(ysize);
	const long int vx0 = FIRST_XMIPP_INDEX(xsize);
	const long int vx1 = LAST_XMIPP_INDEX(xsize);

	for (long int i = y0; i < vy0; i++)
	{
		T *row = dst + (i - y0) * (long int)mic_size;
		for (long int j = x0; j <= x1; j++)
			row[j - x0] = (T)rnd_gaus(0., 1.);
	}
	for (long int i = vy0; i <= vy1; i++)
	{
		T *row = dst + (i - y0) * (long int)mic_size;
		for (long int j = x0; j < vx0; j++)
			row[j - x0] = (T)rnd_gaus(0., 1.);
		for (long int j = vx1 + 1; j <= x1; j++)
			row[j - x0] = (T)rnd_gaus(0., 1.);
	}
	for (long int i = vy1 + 1; i <= y1; i++)
	{
		T *row = dst + (i - y0) * (long int)mic_size;
		for (long int j = x0; j <= x1; j++)
			row[j - x0] = (T)rnd_gaus(0., 1.);
	}
}

struct PrefetchedMic
{
	long int imic = -1;
	bool has_error = false;
	std::string error_msg;
	std::vector<XFLOAT> h_reals;
	std::vector<XFLOAT> h_ctf;
};

static void loadMicToSlot(AutoPicker *pckr, long int imic, PrefetchedMic &out)
{
	out.imic = imic;
	out.has_error = false;
	out.error_msg.clear();

	const FileName &fn_mic = pckr->fn_micrographs[imic];
	Image<RFLOAT> Imic;
	Imic.read(fn_mic);
	Imic().setXmippOrigin();

	RFLOAT my_xsize = XSIZE(Imic());
	RFLOAT my_ysize = YSIZE(Imic());
	RFLOAT my_size = (my_xsize != my_ysize) ? XMIPP_MAX(my_xsize, my_ysize) : my_xsize;
	if (pckr->extra_padding > 0)
		my_size += 2 * pckr->extra_padding;

	if (my_size != pckr->micrograph_size || my_xsize != pckr->micrograph_xsize || my_ysize != pckr->micrograph_ysize)
	{
		std::ostringstream oss;
		oss << "AutoPicker::autoPickOneMicrograph ERROR: No differently sized micrographs are allowed in one run, sorry you will have to run separately for each size... "
		    << "(" << my_xsize << "x" << my_ysize << " vs " << pckr->micrograph_xsize << "x" << pckr->micrograph_ysize << ")";
		out.has_error = true;
		out.error_msg = oss.str();
		return;
	}

	RFLOAT avg0, stddev0, minval0, maxval0;
	Imic().computeStats(avg0, stddev0, minval0, maxval0);

	const int xsize = pckr->micrograph_xsize;
	const int ysize = pckr->micrograph_ysize;
	const int mic_size = pckr->micrograph_size;
	const RFLOAT zcut = pckr->outlier_removal_zscore;
	const RFLOAT *src = Imic().data;

	out.h_reals.resize((size_t)mic_size * (size_t)mic_size);
	XFLOAT *dst = out.h_reals.data();

	const long int off_x = FIRST_XMIPP_INDEX(xsize) - FIRST_XMIPP_INDEX(mic_size);
	const long int off_y = FIRST_XMIPP_INDEX(ysize) - FIRST_XMIPP_INDEX(mic_size);
	for (int sy = 0; sy < ysize; sy++)
	{
		const RFLOAT *srow = src + (size_t)sy * (size_t)xsize;
		XFLOAT *drow = dst + (size_t)(sy + off_y) * (size_t)mic_size + (size_t)off_x;
		for (int sx = 0; sx < xsize; sx++)
		{
			RFLOAT v = srow[sx];
			if (ABS(v - avg0) / stddev0 > zcut)
				v = avg0;
			drow[sx] = (XFLOAT)((v - avg0) / stddev0);
		}
	}

	if (pckr->do_ctf)
	{
		auto it = pckr->mic_to_md_idx.find(fn_mic);
		if (it == pckr->mic_to_md_idx.end())
		{
			out.has_error = true;
			out.error_msg = "Logic error: failed to find CTF information for " + fn_mic;
			return;
		}
		CTF ctf;
		ctf.readByGroup(pckr->MDmic, &pckr->obsModel, it->second);
		MultidimArray<RFLOAT> Fctf(pckr->workSize, pckr->workSize / 2 + 1);
		ctf.getFftwImage(Fctf, pckr->micrograph_size, pckr->micrograph_size, pckr->angpix, false, false, pckr->intact_ctf_first_peak, true);
		out.h_ctf.resize((size_t)Fctf.nzyxdim);
		for (size_t i = 0; i < (size_t)Fctf.nzyxdim; i++)
			out.h_ctf[i] = (XFLOAT)Fctf.data[i];
	}
}

class MicPrefetcher
{
public:
	explicit MicPrefetcher(AutoPicker *p) : pckr_(p)
	{
		worker_ = std::thread([this] { loop(); });
	}

	~MicPrefetcher()
	{
		{
			std::lock_guard<std::mutex> lk(mtx_);
			stop_ = true;
			cv_work_.notify_all();
		}
		if (worker_.joinable())
			worker_.join();
	}

	void start(long int imic, PrefetchedMic *dst)
	{
		std::unique_lock<std::mutex> lk(mtx_);
		cv_done_.wait(lk, [this] { return !in_flight_; });
		req_imic_ = imic;
		req_dst_ = dst;
		in_flight_ = true;
		cv_work_.notify_one();
	}

	void finish()
	{
		std::unique_lock<std::mutex> lk(mtx_);
		cv_done_.wait(lk, [this] { return !in_flight_; });
	}

private:
	void loop()
	{
		for (;;)
		{
			long int imic = -1;
			PrefetchedMic *dst = nullptr;
			{
				std::unique_lock<std::mutex> lk(mtx_);
				cv_work_.wait(lk, [this] { return stop_ || in_flight_; });
				if (stop_ && !in_flight_)
					return;
				imic = req_imic_;
				dst = req_dst_;
			}
			try
			{
				loadMicToSlot(pckr_, imic, *dst);
			}
			catch (const RelionError &e)
			{
				dst->has_error = true;
				dst->error_msg = e.msg;
			}
			catch (const std::exception &e)
			{
				dst->has_error = true;
				dst->error_msg = e.what();
			}
			{
				std::lock_guard<std::mutex> lk(mtx_);
				in_flight_ = false;
				cv_done_.notify_all();
			}
		}
	}

	AutoPicker *pckr_;
	std::mutex mtx_;
	std::condition_variable cv_work_, cv_done_;
	long int req_imic_ = -1;
	PrefetchedMic *req_dst_ = nullptr;
	bool stop_ = false;
	bool in_flight_ = false;
	std::thread worker_;
};

static void writeAutoPickStar(const FileName &fn_star, std::shared_ptr<MetaDataTable> md_ptr)
{
	const bool allow_async = AsyncFileWriter::starMode() > 0;
	AsyncFileWriter::putText(fn_star, [md_ptr]() -> std::shared_ptr<std::string> {
		std::ostringstream oss;
		md_ptr->write(oss);
		return std::make_shared<std::string>(oss.str());
	}, allow_async);
}

} // anonymous namespace


AutoPickerCuda::AutoPickerCuda(AutoPicker *basePicker, const char * timing_fnm) :
	node(NULL),
	basePckr(basePicker),
	allocator(new CudaCustomAllocator(0, 1)),
	micTransformer(0, allocator),
	cudaTransformer1(0, allocator),
#ifdef TIMING_FILES
	timer(timing_fnm),
#endif
	cudaTransformer2(0, allocator),
	d_Fmsk((cudaStream_t)0, allocator),
	d_Fmsk2((cudaStream_t)0, allocator),
	d_Ftmp((cudaStream_t)0, allocator),
	d_Fcov((cudaStream_t)0, allocator),
	d_Fmic((cudaStream_t)0, allocator),
	d_Mstddev((cudaStream_t)0, allocator),
	d_Mstddev_inv((cudaStream_t)0, allocator),
	d_Mstddev_tmp((cudaStream_t)0, allocator),
	d_Mmean((cudaStream_t)0, allocator),
	d_Mstddev2((cudaStream_t)0, allocator),
	d_Mavg((cudaStream_t)0, allocator),
	d_Mccf_best{AccPtr<XFLOAT>((cudaStream_t)0, allocator), AccPtr<XFLOAT>((cudaStream_t)0, allocator)},
	d_Mpsi_best{AccPtr<XFLOAT>((cudaStream_t)0, allocator), AccPtr<XFLOAT>((cudaStream_t)0, allocator)},
	d_Mccf_combined((cudaStream_t)0, allocator),
	d_Mpsi_combined((cudaStream_t)0, allocator),
	d_Mclass_combined((cudaStream_t)0, allocator),
	d_ctf((cudaStream_t)0, allocator),
	d_prep_patches((cudaStream_t)0, allocator)

{
	projectors.resize(basePckr->Mrefs.size());
	have_warned_batching=false;
	/*======================================================
	                    DEVICE SETTINGS
	======================================================*/
	device_id = basePicker->device_id;
	int devCount;
	HANDLE_ERROR(cudaGetDeviceCount(&devCount));

	if(device_id >= devCount)
	{
		//std::cerr << " using device_id=" << device_id << " (device no. " << device_id+1 << ") which is higher than the available number of devices=" << devCount << std::endl;
		CRITICAL(ERR_GPUID);
	}
	else
		HANDLE_ERROR(cudaSetDevice(device_id));
};

AutoPickerCuda::AutoPickerCuda(AutoPickerMpi *basePicker, const char * timing_fnm) :
	basePckr(basePicker),
	allocator(new CudaCustomAllocator(0, 1)),
	micTransformer(0, allocator),
	cudaTransformer1(0, allocator),
#ifdef TIMING_FILES
	timer(timing_fnm),
#endif
	cudaTransformer2(0, allocator),
	d_Fmsk((cudaStream_t)0, allocator),
	d_Fmsk2((cudaStream_t)0, allocator),
	d_Ftmp((cudaStream_t)0, allocator),
	d_Fcov((cudaStream_t)0, allocator),
	d_Fmic((cudaStream_t)0, allocator),
	d_Mstddev((cudaStream_t)0, allocator),
	d_Mstddev_inv((cudaStream_t)0, allocator),
	d_Mstddev_tmp((cudaStream_t)0, allocator),
	d_Mmean((cudaStream_t)0, allocator),
	d_Mstddev2((cudaStream_t)0, allocator),
	d_Mavg((cudaStream_t)0, allocator),
	d_Mccf_best{AccPtr<XFLOAT>((cudaStream_t)0, allocator), AccPtr<XFLOAT>((cudaStream_t)0, allocator)},
	d_Mpsi_best{AccPtr<XFLOAT>((cudaStream_t)0, allocator), AccPtr<XFLOAT>((cudaStream_t)0, allocator)},
	d_Mccf_combined((cudaStream_t)0, allocator),
	d_Mpsi_combined((cudaStream_t)0, allocator),
	d_Mclass_combined((cudaStream_t)0, allocator),
	d_ctf((cudaStream_t)0, allocator),
	d_prep_patches((cudaStream_t)0, allocator)

{
	node = basePicker->getNode();
	basePicker->verb = (node->isLeader()) ? 1 : 0;

	projectors.resize(basePckr->Mrefs.size());
	have_warned_batching=false;
	/*======================================================
	                    DEVICE SETTINGS
	======================================================*/
	device_id = basePckr->device_id;
	int devCount;
	HANDLE_ERROR(cudaGetDeviceCount(&devCount));

	if(device_id >= devCount)
	{
		//std::cerr << " using device_id=" << device_id << " (device no. " << device_id+1 << ") which is higher than the available number of devices=" << devCount << std::endl;
		CRITICAL(ERR_GPUID);
	}
	else
		HANDLE_ERROR(cudaSetDevice(device_id));
};

void AutoPickerCuda::initPersistentBuffers(int Npsi)
{
	if (persistent_init || basePckr->do_read_fom_maps)
		return;

	const size_t mic_size = (size_t)basePckr->micrograph_size;
	const size_t work_size = (size_t)basePckr->workSize;
	const size_t mic_f_sz = (mic_size / 2 + 1) * mic_size;
	const size_t work_f_sz = (work_size / 2 + 1) * work_size;
	const size_t work_sq = work_size * work_size;
	const int n_refs = (int)basePckr->Mrefs.size();
	const size_t P = (size_t)basePckr->particle_size;

	micTransformer.setSize(mic_size, mic_size, 1, 1, 0, false);
	cudaTransformer2.setSize(work_size, work_size, 1, 1, FFTW_BACKWARD, false);

	// Upload masks once across the whole run
	d_Fmsk.setSize(mic_f_sz);
	d_Fmsk.deviceAlloc();
	{
		std::vector<ACCCOMPLEX> h_msk(mic_f_sz);
		for (size_t i = 0; i < mic_f_sz; i++)
		{
			h_msk[i].x = (XFLOAT)basePckr->Finvmsk.data[i].real;
			h_msk[i].y = (XFLOAT)basePckr->Finvmsk.data[i].imag;
		}
		cudaCpyHostToDevice(h_msk.data(), ~d_Fmsk, mic_f_sz);
	}

	if (basePckr->autopick_helical_segments)
	{
		d_Fmsk2.setSize(mic_f_sz);
		d_Fmsk2.deviceAlloc();
		std::vector<ACCCOMPLEX> h_msk2(mic_f_sz);
		for (size_t i = 0; i < mic_f_sz; i++)
		{
			h_msk2[i].x = (XFLOAT)basePckr->Favgmsk.data[i].real;
			h_msk2[i].y = (XFLOAT)basePckr->Favgmsk.data[i].imag;
		}
		cudaCpyHostToDevice(h_msk2.data(), ~d_Fmsk2, mic_f_sz);
	}

	d_Ftmp.deviceAlloc(mic_f_sz);
	d_Fcov.deviceAlloc(mic_f_sz);
	d_Fmic.deviceAlloc(work_f_sz);

	d_Mstddev.deviceAlloc(work_sq);
	d_Mstddev_inv.deviceAlloc(work_sq);
	d_Mstddev_tmp.deviceAlloc(work_sq);
	d_Mmean.deviceAlloc(work_sq);

	if (basePckr->autopick_helical_segments)
	{
		d_Mstddev2.deviceAlloc(work_sq);
		d_Mavg.deviceAlloc(work_sq);
		d_Mccf_combined.deviceAlloc(work_sq);
		if (basePckr->do_amyloid)
			d_Mpsi_combined.deviceAlloc(work_sq);
		else
			d_Mclass_combined.deviceAlloc(work_sq);
	}

	d_Mccf_best[0].deviceAlloc(work_sq);
	d_Mpsi_best[0].deviceAlloc(work_sq);
	HANDLE_ERROR(cudaMemsetAsync(~d_Mpsi_best[0], 0, work_sq * sizeof(XFLOAT), 0));
	if (!basePckr->autopick_helical_segments)
	{
		d_Mccf_best[1].deviceAlloc(work_sq);
		d_Mpsi_best[1].deviceAlloc(work_sq);
		HANDLE_ERROR(cudaMemsetAsync(~d_Mpsi_best[1], 0, work_sq * sizeof(XFLOAT), 0));
	}

	if (basePckr->do_ctf)
		d_ctf.deviceAlloc(work_f_sz);

	const size_t patch_elems = P * P;
	prep_chunk_size = (n_refs > 0 && patch_elems > 0)
		? std::min(n_refs, std::max(1, (int)((64ULL << 20) / (patch_elems * sizeof(XFLOAT)))))
		: 0;
	if (prep_chunk_size > 0)
	{
		d_prep_patches.deviceAlloc((size_t)prep_chunk_size * patch_elems);
		HANDLE_ERROR(cudaMallocHost(&h_prep, (size_t)prep_chunk_size * patch_elems * sizeof(XFLOAT)));
	}

	// Size cudaTransformer1 AFTER all persistent device buffers so cudaMemGetInfo sees true headroom
	cudaTransformer1.setSize(work_size, work_size, 1, Npsi, FFTW_BACKWARD, false);
	if (cudaTransformer1.batchSize.size() > 1 && !have_warned_batching)
	{
		have_warned_batching = true;
		std::cerr << std::endl << "*-----------------------------WARNING------------------------------------------------*" << std::endl;
		std::cerr              << "With the current settings the GPU memory is imposing a soft limit on your performace," << std::endl;
		std::cerr              << "since one or more micrographs has to use (at least " << cudaTransformer1.batchSize.size() << ") batches of orientations to " << std::endl;
		std::cerr              << "achieve the total requested " << Npsi << " orientations. Consider using" << std::endl;
		std::cerr              << "\t higher --ang" << std::endl;
		std::cerr              << "\t harder --shrink" << std::endl;
		std::cerr              << "\t higher --lowpass with --shrink 0" << std::endl;
		std::cerr              << "*------------------------------------------------------------------------------------*" << std::endl;
	}

	need_h_stddev = basePckr->do_write_fom_maps || basePckr->max_stddev_noise > 0. || basePckr->autopick_helical_segments;
	need_h_mean   = basePckr->do_write_fom_maps || basePckr->min_avg_noise > -900. || basePckr->autopick_helical_segments;

	if (!basePckr->autopick_helical_segments)
	{
		for (int b = 0; b < 2; b++)
		{
			HANDLE_ERROR(cudaMallocHost(&h_ccf[b], work_sq * sizeof(XFLOAT)));
			HANDLE_ERROR(cudaMallocHost(&h_psi[b], work_sq * sizeof(XFLOAT)));
			HANDLE_ERROR(cudaEventCreateWithFlags(&ev_ref_done[b], cudaEventDisableTiming));
		}
		if (need_h_stddev)
			HANDLE_ERROR(cudaMallocHost(&h_Mstddev, work_sq * sizeof(XFLOAT)));
		if (need_h_mean)
			HANDLE_ERROR(cudaMallocHost(&h_Mmean, work_sq * sizeof(XFLOAT)));
		HANDLE_ERROR(cudaEventCreateWithFlags(&ev_stats_done, cudaEventDisableTiming));
	}
	else
	{
		HANDLE_ERROR(cudaMallocHost(&h_ccf[0], work_sq * sizeof(XFLOAT)));
		HANDLE_ERROR(cudaMallocHost(&h_Mstddev, work_sq * sizeof(XFLOAT)));
		HANDLE_ERROR(cudaMallocHost(&h_Mmean, work_sq * sizeof(XFLOAT)));
		if (basePckr->do_amyloid)
			HANDLE_ERROR(cudaMallocHost(&h_psi[0], work_sq * sizeof(XFLOAT)));
		else
			HANDLE_ERROR(cudaMallocHost(&h_class_combined, work_sq * sizeof(int)));
	}

	persistent_init = true;
}

void AutoPickerCuda::releasePersistentBuffers()
{
	if (!persistent_init)
		return;

	cudaStreamSynchronize(0);
	for (int b = 0; b < 2; b++)
	{
		if (ev_ref_done[b] != NULL) { cudaEventDestroy(ev_ref_done[b]); ev_ref_done[b] = NULL; }
		if (h_ccf[b] != NULL) { cudaFreeHost(h_ccf[b]); h_ccf[b] = NULL; }
		if (h_psi[b] != NULL) { cudaFreeHost(h_psi[b]); h_psi[b] = NULL; }
	}
	if (ev_stats_done != NULL) { cudaEventDestroy(ev_stats_done); ev_stats_done = NULL; }
	if (h_Mstddev != NULL) { cudaFreeHost(h_Mstddev); h_Mstddev = NULL; }
	if (h_Mmean != NULL) { cudaFreeHost(h_Mmean); h_Mmean = NULL; }
	if (h_class_combined != NULL) { cudaFreeHost(h_class_combined); h_class_combined = NULL; }
	if (h_prep != NULL) { cudaFreeHost(h_prep); h_prep = NULL; }

	d_Fmsk.freeIfSet();
	d_Fmsk2.freeIfSet();
	d_Ftmp.freeIfSet();
	d_Fcov.freeIfSet();
	d_Fmic.freeIfSet();
	d_Mstddev.freeIfSet();
	d_Mstddev_inv.freeIfSet();
	d_Mstddev_tmp.freeIfSet();
	d_Mmean.freeIfSet();
	d_Mstddev2.freeIfSet();
	d_Mavg.freeIfSet();
	d_Mccf_best[0].freeIfSet();
	d_Mccf_best[1].freeIfSet();
	d_Mpsi_best[0].freeIfSet();
	d_Mpsi_best[1].freeIfSet();
	d_Mccf_combined.freeIfSet();
	d_Mpsi_combined.freeIfSet();
	d_Mclass_combined.freeIfSet();
	d_ctf.freeIfSet();
	d_prep_patches.freeIfSet();

	micTransformer.clear();
	cudaTransformer1.clear();
	cudaTransformer2.clear();

	persistent_init = false;
}

void AutoPickerCuda::run()
{
	long int my_first_micrograph, my_last_micrograph, my_nr_micrographs;
	if(node!=NULL)
	{
		// Each node does part of the work
		divide_equally(basePckr->fn_micrographs.size(), node->size, node->rank, my_first_micrograph, my_last_micrograph);
	}
	else
	{
		my_first_micrograph = 0;
		my_last_micrograph = basePckr->fn_micrographs.size() - 1;
	}
	my_nr_micrographs = (basePckr->fn_micrographs.empty() || my_last_micrograph < my_first_micrograph)
		? 0
		: (my_last_micrograph - my_first_micrograph + 1);

	int barstep = 1;
	if (basePckr->verb > 0)
	{
		std::cout << " Autopicking ..." << std::endl;
		init_progress_bar(my_nr_micrographs);
		barstep = XMIPP_MAX(1, my_nr_micrographs / 60);
	}

	if (!basePckr->do_read_fom_maps && my_nr_micrographs > 0)
	{
		CTIC(timer,"setupProjectors");
		for (int iref = 0; iref < (basePckr->Mrefs.size()); iref++)
		{
			projectors[iref].setMdlDim(
							basePckr->PPref[iref].data.xdim,
							basePckr->PPref[iref].data.ydim,
							basePckr->PPref[iref].data.zdim,
							basePckr->PPref[iref].data.yinit,
							basePckr->PPref[iref].data.zinit,
							basePckr->PPref[iref].r_max,
							basePckr->PPref[iref].padding_factor);
			projectors[iref].initMdl(&(basePckr->PPref[iref].data.data[0]));
		}
		CTOC(timer,"setupProjectors");

		const int Npsi = 360 / basePckr->psi_sampling;
		initPersistentBuffers(Npsi);
	}

	FileName fn_olddir="";
	vmovies::ReadAhead read_ahead(-1);
	read_ahead.expectMicrographs(my_nr_micrographs);

	if (basePckr->do_read_fom_maps)
	{
		for (long int imic = my_first_micrograph; imic <= my_last_micrograph; imic++)
		{
			if (basePckr->verb > 0 && imic % barstep == 0)
				progress_bar(imic - my_first_micrograph);

			FileName fn_dir = basePckr->getOutputRootName(basePckr->fn_micrographs[imic]);
			fn_dir = fn_dir.beforeLastOf("/");
			if (fn_dir != fn_olddir)
			{
				mktree(fn_dir);
				fn_olddir = fn_dir;
			}
#ifdef TIMING
			basePckr->timer.tic(basePckr->TIMING_A5);
#endif
			autoPickOneMicrograph(basePckr->fn_micrographs[imic], imic);
#ifdef TIMING
			basePckr->timer.toc(basePckr->TIMING_A5);
#endif
		}
	}
	else if (my_nr_micrographs > 0)
	{
		PrefetchedMic slots[2];
		MicPrefetcher prefetcher(basePckr);

		if (read_ahead.enabled())
		{
			std::vector<std::string> next;
			for (long int k = my_first_micrograph; k <= my_last_micrograph && (int)next.size() < vmovies::ReadAhead::WINDOW; k++)
				next.push_back(basePckr->fn_micrographs[k]);
			read_ahead.upcoming(next);
		}

		int cur_slot = 0;
		prefetcher.start(my_first_micrograph, &slots[cur_slot]);

		for (long int imic = my_first_micrograph; imic <= my_last_micrograph; imic++)
		{
			if (basePckr->verb > 0 && imic % barstep == 0)
				progress_bar(imic - my_first_micrograph);

			FileName fn_dir = basePckr->getOutputRootName(basePckr->fn_micrographs[imic]);
			fn_dir = fn_dir.beforeLastOf("/");
			if (fn_dir != fn_olddir)
			{
				mktree(fn_dir);
				fn_olddir = fn_dir;
			}

			prefetcher.finish();
			if (slots[cur_slot].has_error)
				REPORT_ERROR(slots[cur_slot].error_msg);

			// RNG border fill stays on main thread in strict row-major order
			init_random_generator(basePckr->random_seed + imic);
			if (basePckr->micrograph_xsize != basePckr->micrograph_size || basePckr->micrograph_ysize != basePckr->micrograph_size)
			{
				fillOutsideWindowWithGaus(
					slots[cur_slot].h_reals.data(),
					basePckr->micrograph_xsize,
					basePckr->micrograph_ysize,
					basePckr->micrograph_size);
			}

			const XFLOAT *ctf_ptr = slots[cur_slot].h_ctf.empty() ? NULL : slots[cur_slot].h_ctf.data();

			// Upload current micrograph & CTF to device BEFORE launching prefetch for imic+1 on next slot
			cudaStream_t mic_stream = micTransformer.reals.getStream();
			cudaCpyHostToDevice(slots[cur_slot].h_reals.data(), ~micTransformer.reals, slots[cur_slot].h_reals.size(), mic_stream);
			if (basePckr->do_ctf && ctf_ptr != NULL)
			{
				cudaStream_t ctf_stream = d_ctf.getStream();
				cudaCpyHostToDevice(const_cast<XFLOAT*>(ctf_ptr), ~d_ctf, slots[cur_slot].h_ctf.size(), ctf_stream);
			}

			const int next_slot = cur_slot ^ 1;
			if (imic + 1 <= my_last_micrograph)
			{
				if (read_ahead.enabled())
				{
					std::vector<std::string> next;
					for (long int k = imic + 1; k <= my_last_micrograph && (int)next.size() < vmovies::ReadAhead::WINDOW; k++)
						next.push_back(basePckr->fn_micrographs[k]);
					read_ahead.upcoming(next);
				}
				prefetcher.start(imic + 1, &slots[next_slot]);
			}

#ifdef TIMING
			basePckr->timer.tic(basePckr->TIMING_A5);
#endif
			autoPickOneMicrographPrepared(basePckr->fn_micrographs[imic], imic, slots[cur_slot].h_reals.data(), ctf_ptr);
#ifdef TIMING
			basePckr->timer.toc(basePckr->TIMING_A5);
#endif
			cur_slot = next_slot;
		}
	}

	AsyncFileWriter::instance().waitOrThrow();

	if (basePckr->verb > 0)
		progress_bar(my_nr_micrographs);

	releasePersistentBuffers();
	projectors.clear();
}

void AutoPickerCuda::calculateStddevAndMeanUnderMask(
		AccPtr< ACCCOMPLEX > &d_Fmic_in,
		AccPtr< ACCCOMPLEX > &d_Fmic2_in,
		AccPtr< ACCCOMPLEX > &d_Fmsk_in,
		int nr_nonzero_pixels_mask,
		AccPtr< XFLOAT > &d_Mstddev_out,
		AccPtr< XFLOAT > &d_Mmean_out,
		XFLOAT *d_Mstddev_inv_ptr,
		size_t x, size_t y, size_t mic_size, size_t workSize)
{
	RFLOAT normfft = (RFLOAT)(mic_size * mic_size) / (RFLOAT)nr_nonzero_pixels_mask;
	const int W = (int)workSize;
	dim3 grid2d((unsigned)((W + 31) / 32), (unsigned)((W + 7) / 8));
	dim3 block2d(32, 8);

	int Bsize = (int)ceilf((float)d_Fmic_in.getSize() / (float)BLOCK_SIZE);
	cuda_kernel_convol_B<<<Bsize, BLOCK_SIZE>>>(
			~d_Fmic_in,
			~d_Fmsk_in,
			~d_Fcov,
			d_Fmic_in.getSize());
	LAUNCH_HANDLE_ERROR(cudaGetLastError());

	windowFourierTransform2(
			d_Fcov,
			cudaTransformer2.fouriers,
			x, y, 1,
			workSize/2+1, workSize, 1);

	cudaTransformer2.backward();

	cuda_kernel_centerScaleAndNegSq<<<grid2d, block2d>>>(
			~cudaTransformer2.reals,
			~d_Mmean_out,
			~d_Mstddev_tmp,
			W,
			(XFLOAT)normfft);
	LAUNCH_HANDLE_ERROR(cudaGetLastError());

	Bsize = (int)ceilf((float)d_Fmsk_in.getSize() / (float)BLOCK_SIZE);
	cuda_kernel_convol_A<<<Bsize, BLOCK_SIZE>>>(
			~d_Fmsk_in,
			~d_Fmic2_in,
			~d_Fcov,
			d_Fmsk_in.getSize());
	LAUNCH_HANDLE_ERROR(cudaGetLastError());

	windowFourierTransform2(
			d_Fcov,
			cudaTransformer2.fouriers,
			x, y, 1,
			workSize/2+1, workSize, 1);

	cudaTransformer2.backward();

	cuda_kernel_finalizeMstddev_centered<<<grid2d, block2d>>>(
			~d_Mstddev_tmp,
			~cudaTransformer2.reals,
			~d_Mstddev_out,
			d_Mstddev_inv_ptr,
			W,
			(XFLOAT)normfft);
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
}

void AutoPickerCuda::autoPickOneMicrograph(FileName &fn_mic, long int imic)
{
	if (basePckr->do_read_fom_maps)
	{
		MultidimArray<RFLOAT> Mstddev, Mmean, Mstddev2, Mavg, Mccf_best, Mpsi_best, Mccf_best_combined, Mpsi_best_combined;
		MultidimArray<int> Mclass_best_combined;
		int my_skip_side = basePckr->autopick_skip_side + basePckr->particle_size / 2;
		int min_distance_pix = ROUND(basePckr->min_particle_distance / basePckr->angpix);
		XFLOAT scale = (XFLOAT)basePckr->workSize / (XFLOAT)basePckr->micrograph_size;
		init_random_generator(basePckr->random_seed + imic);

		Mccf_best.resize(basePckr->workSize, basePckr->workSize);
		Mpsi_best.resize(basePckr->workSize, basePckr->workSize);

		FileName fn_tmp = basePckr->getOutputRootName(fn_mic) + "_" + basePckr->fn_out + "_stddevNoise.spi";
		Image<RFLOAT> It;
		It.read(fn_tmp);
		if (basePckr->autopick_helical_segments)
			Mstddev2 = It();
		else
			Mstddev = It();
		fn_tmp = basePckr->getOutputRootName(fn_mic) + "_" + basePckr->fn_out + "_avgNoise.spi";
		It.read(fn_tmp);
		if (basePckr->autopick_helical_segments)
			Mavg = It();
		else
			Mmean = It();

		std::vector<Peak> peaks;
		if (basePckr->autopick_helical_segments)
		{
			Image<RFLOAT> It_float;
			Image<int> It_int;
			fn_tmp = basePckr->getOutputRootName(fn_mic) + "_" + basePckr->fn_out + "_combinedCCF.spi";
			It_float.read(fn_tmp);
			Mccf_best_combined = It_float();
			if (basePckr->do_amyloid)
			{
				fn_tmp = basePckr->getOutputRootName(fn_mic) + "_" + basePckr->fn_out + "_combinedPSI.spi";
				It_float.read(fn_tmp);
				Mpsi_best_combined = It_float();
			}
			else
			{
				fn_tmp = basePckr->getOutputRootName(fn_mic) + "_" + basePckr->fn_out + "_combinedCLASS.spi";
				It_int.read(fn_tmp);
				Mclass_best_combined = It_int();
			}
		}

		for (int iref = 0; iref < (int)basePckr->Mrefs.size(); iref++)
		{
			RFLOAT expected_Pratio = 0.;
			if (!basePckr->autopick_helical_segments)
			{
				Image<RFLOAT> It_ref;
				fn_tmp.compose(basePckr->getOutputRootName(fn_mic) + "_" + basePckr->fn_out + "_ref", iref, "_bestCCF.spi");
				It_ref.read(fn_tmp);
				Mccf_best = It_ref();
				It_ref.MDMainHeader.getValue(EMDL_IMAGE_STATS_MAX, expected_Pratio);
				fn_tmp.compose(basePckr->getOutputRootName(fn_mic) + "_" + basePckr->fn_out + "_ref", iref, "_bestPSI.spi");
				It_ref.read(fn_tmp);
				Mpsi_best = It_ref();

				std::vector<Peak> my_ref_peaks;
				Mstddev.setXmippOrigin();
				Mmean.setXmippOrigin();
				Mccf_best.setXmippOrigin();
				Mpsi_best.setXmippOrigin();
				basePckr->peakSearch(Mccf_best, Mpsi_best, Mstddev, Mmean, iref, my_skip_side, my_ref_peaks, scale);
				basePckr->prunePeakClusters(my_ref_peaks, min_distance_pix, scale);
				peaks.insert(peaks.end(), my_ref_peaks.begin(), my_ref_peaks.end());
			}
		}

		if (basePckr->autopick_helical_segments)
		{
			RFLOAT thres = basePckr->min_fraction_expected_Pratio;
			int peak_r_min = 1;
			std::vector<ccfPeak> ccf_peak_list;
			std::vector<std::vector<ccfPeak> > tube_coord_list, tube_track_list;
			std::vector<RFLOAT> tube_len_list;
			MultidimArray<RFLOAT> Mccfplot;

			Mccf_best_combined.setXmippOrigin();
			Mpsi_best_combined.setXmippOrigin();
			Mstddev2.setXmippOrigin();
			Mavg.setXmippOrigin();
			Mclass_best_combined.setXmippOrigin();
			if (basePckr->do_amyloid)
			{
				basePckr->pickAmyloids(Mccf_best_combined, Mpsi_best_combined, Mstddev2, Mavg, thres, basePckr->amyloid_max_psidiff, fn_mic, basePckr->fn_out,
						(basePckr->helical_tube_diameter / basePckr->angpix), basePckr->autopick_skip_side, scale);
			}
			else
			{
				basePckr->pickCCFPeaks(Mccf_best_combined, Mstddev2, Mavg, Mclass_best_combined, thres, peak_r_min, (basePckr->particle_diameter / basePckr->angpix),
						ccf_peak_list, Mccfplot, my_skip_side, scale);
				basePckr->extractHelicalTubes(ccf_peak_list, tube_coord_list, tube_len_list, tube_track_list,
						(basePckr->particle_diameter / basePckr->angpix), basePckr->helical_tube_curvature_factor_max,
						(basePckr->min_particle_distance / basePckr->angpix), (basePckr->helical_tube_diameter / basePckr->angpix), scale);
				basePckr->exportHelicalTubes(Mccf_best_combined, Mccfplot, Mclass_best_combined,
						tube_coord_list, tube_track_list, tube_len_list,
						fn_mic, basePckr->fn_out,
						(basePckr->particle_diameter / basePckr->angpix),
						(basePckr->helical_tube_length_min / basePckr->angpix),
						my_skip_side, scale);
			}
			if (!basePckr->do_amyloid)
			{
				Image<RFLOAT> It_plot;
				It_plot() = Mccfplot;
				fn_tmp = basePckr->getOutputRootName(fn_mic) + "_" + basePckr->fn_out + "_combinedPLOT.spi";
				It_plot.write(fn_tmp);
			}
		}
		else
		{
			basePckr->prunePeakClusters(peaks, min_distance_pix, scale);
			basePckr->removeTooCloselyNeighbouringPeaks(peaks, min_distance_pix, scale);
			auto md_ptr = std::make_shared<MetaDataTable>();
			for (int ipeak = 0; ipeak < (int)peaks.size(); ipeak++)
			{
				md_ptr->addObject();
				md_ptr->setValue(EMDL_IMAGE_COORD_X, (RFLOAT)(peaks[ipeak].x) / scale);
				md_ptr->setValue(EMDL_IMAGE_COORD_Y, (RFLOAT)(peaks[ipeak].y) / scale);
				md_ptr->setValue(EMDL_PARTICLE_CLASS, peaks[ipeak].ref + 1);
				md_ptr->setValue(EMDL_PARTICLE_AUTOPICK_FOM, peaks[ipeak].fom);
				md_ptr->setValue(EMDL_ORIENT_PSI, peaks[ipeak].psi);
			}
			FileName fn_star = basePckr->getOutputRootName(fn_mic) + "_" + basePckr->fn_out + ".star";
			writeAutoPickStar(fn_star, md_ptr);
		}
		return;
	}

	const int Npsi = 360 / basePckr->psi_sampling;
	initPersistentBuffers(Npsi);

	PrefetchedMic slot;
	loadMicToSlot(basePckr, imic, slot);
	if (slot.has_error)
		REPORT_ERROR(slot.error_msg);

	init_random_generator(basePckr->random_seed + imic);
	if (basePckr->micrograph_xsize != basePckr->micrograph_size || basePckr->micrograph_ysize != basePckr->micrograph_size)
	{
		fillOutsideWindowWithGaus(
			slot.h_reals.data(),
			basePckr->micrograph_xsize,
			basePckr->micrograph_ysize,
			basePckr->micrograph_size);
	}

	const XFLOAT *ctf_ptr = slot.h_ctf.empty() ? NULL : slot.h_ctf.data();
	cudaStream_t mic_stream = micTransformer.reals.getStream();
	cudaCpyHostToDevice(slot.h_reals.data(), ~micTransformer.reals, slot.h_reals.size(), mic_stream);
	if (basePckr->do_ctf && ctf_ptr != NULL)
	{
		cudaStream_t ctf_stream = d_ctf.getStream();
		cudaCpyHostToDevice(const_cast<XFLOAT*>(ctf_ptr), ~d_ctf, slot.h_ctf.size(), ctf_stream);
	}

	autoPickOneMicrographPrepared(fn_mic, imic, slot.h_reals.data(), ctf_ptr);
}

void AutoPickerCuda::autoPickOneMicrographPrepared(
		const FileName &fn_mic,
		long int imic,
		const XFLOAT *h_mic_reals,
		const XFLOAT *h_ctf_data)
{
	(void)imic;
	(void)h_mic_reals;
	(void)h_ctf_data;

	const int mic_size = basePckr->micrograph_size;
	const int W = basePckr->workSize;
	const int P = basePckr->particle_size;
	const size_t work_sq = (size_t)W * (size_t)W;
	const int Npsi = 360 / basePckr->psi_sampling;
	const int n_refs = (int)basePckr->Mrefs.size();
	const int my_skip_side = basePckr->autopick_skip_side + basePckr->particle_size / 2;
	const int min_distance_pix = ROUND(basePckr->min_particle_distance / basePckr->angpix);
	const XFLOAT scale = (XFLOAT)W / (XFLOAT)mic_size;
	const RFLOAT normfft = (RFLOAT)(mic_size * mic_size) / (RFLOAT)basePckr->nr_pixels_circular_mask;

#ifdef TIMING
	basePckr->timer.tic(basePckr->TIMING_B1);
#endif
	runCenterFFT(micTransformer.reals, micTransformer.xSize, micTransformer.ySize, true, 1);

	micTransformer.forward();
	int FMultiBsize = (int)ceilf((float)micTransformer.fouriers.getSize() * 2 / (float)BLOCK_SIZE);
	CudaKernels::cuda_kernel_multi<XFLOAT><<<FMultiBsize, BLOCK_SIZE>>>(
			(XFLOAT*)~micTransformer.fouriers,
			(XFLOAT)1 / ((XFLOAT)(micTransformer.reals.getSize())),
			micTransformer.fouriers.getSize() * 2);
	LAUNCH_HANDLE_ERROR(cudaGetLastError());

	if (basePckr->highpass > 0.)
	{
		lowPassFilterMapGPU(
				micTransformer.fouriers,
				(size_t)1,
				micTransformer.yFSize,
				micTransformer.xFSize,
				mic_size,
				basePckr->lowpass,
				basePckr->highpass,
				basePckr->angpix,
				2,
				true);
		micTransformer.backward();
	}

	micTransformer.fouriers.cpOnAcc(d_Ftmp);

	int RMultiBsize = (int)ceilf((float)micTransformer.reals.getSize() / (float)BLOCK_SIZE);
	cuda_kernel_square<<<RMultiBsize, BLOCK_SIZE>>>(
			~micTransformer.reals,
			micTransformer.reals.getSize());
	LAUNCH_HANDLE_ERROR(cudaGetLastError());

	micTransformer.forward();
	CudaKernels::cuda_kernel_multi<XFLOAT><<<FMultiBsize, BLOCK_SIZE>>>(
			(XFLOAT*)~micTransformer.fouriers,
			(XFLOAT)1 / ((XFLOAT)(micTransformer.reals.getSize())),
			micTransformer.fouriers.getSize() * 2);
	LAUNCH_HANDLE_ERROR(cudaGetLastError());

	if (basePckr->autopick_helical_segments)
	{
		calculateStddevAndMeanUnderMask(
				d_Ftmp, micTransformer.fouriers, d_Fmsk2,
				basePckr->nr_pixels_avg_mask, d_Mstddev2, d_Mavg, NULL,
				micTransformer.xFSize, micTransformer.yFSize, mic_size, W);
		calculateStddevAndMeanUnderMask(
				d_Ftmp, micTransformer.fouriers, d_Fmsk,
				basePckr->nr_pixels_circular_invmask, d_Mstddev, d_Mmean, ~d_Mstddev_inv,
				micTransformer.xFSize, micTransformer.yFSize, mic_size, W);
	}
	else
	{
		calculateStddevAndMeanUnderMask(
				d_Ftmp, micTransformer.fouriers, d_Fmsk,
				basePckr->nr_pixels_circular_invmask, d_Mstddev, d_Mmean, ~d_Mstddev_inv,
				micTransformer.xFSize, micTransformer.yFSize, mic_size, W);
		if (need_h_stddev)
			HANDLE_ERROR(cudaMemcpyAsync(h_Mstddev, ~d_Mstddev, work_sq * sizeof(XFLOAT), cudaMemcpyDeviceToHost, 0));
		if (need_h_mean)
			HANDLE_ERROR(cudaMemcpyAsync(h_Mmean, ~d_Mmean, work_sq * sizeof(XFLOAT), cudaMemcpyDeviceToHost, 0));
		HANDLE_ERROR(cudaEventRecord(ev_stats_done, 0));
	}

	windowFourierTransform2(
			d_Ftmp,
			d_Fmic,
			(size_t)mic_size / 2 + 1, (size_t)mic_size, 1,
			(size_t)W / 2 + 1, (size_t)W, 1);

#ifdef TIMING
	basePckr->timer.toc(basePckr->TIMING_B1);
#endif

	// Batched PREP_CALCS for all references, preserving strict ascending rnd_gaus order
	std::vector<RFLOAT> sum_ref_vec(n_refs, 0.);
	std::vector<RFLOAT> sum_ref2_vec(n_refs, 0.);
	std::vector<RFLOAT> expected_Pratio_vec(n_refs, 0.);

	const int FauxStride = (W / 2 + 1) * W;
	const size_t patch_elems = (size_t)P * (size_t)P;
	const int p_first = (int)FIRST_XMIPP_INDEX(P);
	const bool mic_even = ((mic_size & 1) == 0);
	const dim3 rot_blocks((unsigned)ceilf((float)FauxStride / (float)BLOCK_SIZE), 1);
	const dim3 patch_blocks((unsigned)((P + 15) / 16), (unsigned)((P + 15) / 16));
	const dim3 patch_threads(16, 16);

	for (int c0 = 0; c0 < n_refs; c0 += prep_chunk_size)
	{
		const int c1 = std::min(n_refs, c0 + prep_chunk_size);
		for (int iref = c0; iref < c1; iref++)
		{
			AccProjectorKernel projKernel = AccProjectorKernel::makeKernel(
					projectors[iref],
					W / 2 + 1,
					W,
					1,
					W / 2);

			if (basePckr->do_ctf)
			{
				cuda_kernel_rotateAndCtf<<<rot_blocks, BLOCK_SIZE>>>(
						~cudaTransformer1.fouriers,
						~d_ctf,
						0,
						projKernel,
						0);
			}
			else
			{
				cuda_kernel_rotateOnly<<<rot_blocks, BLOCK_SIZE>>>(
						~cudaTransformer1.fouriers,
						0,
						projKernel,
						0);
			}
			LAUNCH_HANDLE_ERROR(cudaGetLastError());

			windowFourierTransform2(
					cudaTransformer1.fouriers,
					micTransformer.fouriers,
					(size_t)W / 2 + 1, (size_t)W, 1,
					(size_t)mic_size / 2 + 1, (size_t)mic_size, 1);

			micTransformer.backward();
			if (!mic_even)
			{
				runCenterFFT(
						micTransformer.reals,
						(int)micTransformer.xSize,
						(int)micTransformer.ySize,
						false,
						1);
			}

			cuda_kernel_extractPrepPatch<<<patch_blocks, patch_threads>>>(
					~micTransformer.reals,
					~d_prep_patches + (size_t)(iref - c0) * patch_elems,
					mic_size,
					P,
					p_first,
					mic_even);
			LAUNCH_HANDLE_ERROR(cudaGetLastError());
		}

		HANDLE_ERROR(cudaMemcpyAsync(
				h_prep,
				~d_prep_patches,
				(size_t)(c1 - c0) * patch_elems * sizeof(XFLOAT),
				cudaMemcpyDeviceToHost,
				0));
		HANDLE_ERROR(cudaStreamSynchronize(0));

		for (int iref = c0; iref < c1; iref++)
		{
			const XFLOAT *patch = h_prep + (size_t)(iref - c0) * patch_elems;
			RFLOAT sum_ref_under_circ_mask = 0.;
			RFLOAT sum_ref2_under_circ_mask = 0.;
			RFLOAT suma2 = 0.;
			RFLOAT sumn = 1.;
			for (int py = 0; py < P; py++)
			{
				const int i = py + p_first;
				const int i2 = i * i;
				const XFLOAT *prow = patch + (size_t)py * (size_t)P;
				for (int px = 0; px < P; px++)
				{
					const int j = px + p_first;
					if (i2 + j * j < basePckr->particle_radius2)
					{
						const RFLOAT val = (RFLOAT)prow[px];
						suma2 += val * val;
						suma2 += 2. * val * rnd_gaus(0., 1.);
						sum_ref_under_circ_mask += val;
						sum_ref2_under_circ_mask += val * val;
						sumn += 1.;
					}
				}
			}
			sum_ref_vec[iref] = sum_ref_under_circ_mask / sumn;
			sum_ref2_vec[iref] = sum_ref2_under_circ_mask / sumn;
			expected_Pratio_vec[iref] = exp(suma2 / (2. * sumn));
		}
	}

	MultidimArray<RFLOAT> Mstddev, Mmean, Mccf_best, Mpsi_best;
	std::vector<Peak> peaks;
	if (!basePckr->autopick_helical_segments)
	{
		Mccf_best.resize(W, W);
		Mccf_best.setXmippOrigin();
		Mpsi_best.resize(W, W);
		Mpsi_best.setXmippOrigin();
	}

	bool stats_waited = false;
	auto finishHostRef = [&](int prev)
	{
		const int prev_buf = prev & 1;
		if (!stats_waited)
		{
			HANDLE_ERROR(cudaEventSynchronize(ev_stats_done));
			if (need_h_stddev)
			{
				Mstddev.resizeNoCp(1, W, W);
				for (size_t k = 0; k < work_sq; k++)
					Mstddev.data[k] = (RFLOAT)h_Mstddev[k];
				Mstddev.setXmippOrigin();
			}
			if (need_h_mean)
			{
				Mmean.resizeNoCp(1, W, W);
				for (size_t k = 0; k < work_sq; k++)
					Mmean.data[k] = (RFLOAT)h_Mmean[k];
				Mmean.setXmippOrigin();
			}
			if (basePckr->do_write_fom_maps)
			{
				FileName fn_tmp = basePckr->getOutputRootName(fn_mic) + "_" + basePckr->fn_out + "_stddevNoise.spi";
				Image<RFLOAT> It;
				It() = Mstddev;
				It.write(fn_tmp);
				fn_tmp = basePckr->getOutputRootName(fn_mic) + "_" + basePckr->fn_out + "_avgNoise.spi";
				It() = Mmean;
				It.write(fn_tmp);
			}
			stats_waited = true;
		}
		HANDLE_ERROR(cudaEventSynchronize(ev_ref_done[prev_buf]));
		for (size_t k = 0; k < work_sq; k++)
		{
			Mccf_best.data[k] = (RFLOAT)h_ccf[prev_buf][k];
			Mpsi_best.data[k] = (RFLOAT)h_psi[prev_buf][k];
		}

		if (basePckr->do_write_fom_maps)
		{
			FileName fn_tmp;
			Image<RFLOAT> It;
			It() = Mccf_best;
			It.MDMainHeader.setValue(EMDL_IMAGE_STATS_MAX, expected_Pratio_vec[prev]);
			fn_tmp.compose(basePckr->getOutputRootName(fn_mic) + "_" + basePckr->fn_out + "_ref", prev, "_bestCCF.spi");
			It.write(fn_tmp);
			It() = Mpsi_best;
			fn_tmp.compose(basePckr->getOutputRootName(fn_mic) + "_" + basePckr->fn_out + "_ref", prev, "_bestPSI.spi");
			It.write(fn_tmp);
		}

		std::vector<Peak> my_ref_peaks;
		basePckr->peakSearch(Mccf_best, Mpsi_best, Mstddev, Mmean, prev, my_skip_side, my_ref_peaks, scale);
		basePckr->prunePeakClusters(my_ref_peaks, min_distance_pix, scale);
		peaks.insert(peaks.end(), my_ref_peaks.begin(), my_ref_peaks.end());
	};

	const dim3 rc_threads(16, 16);
	const int projX = W / 2 + 1;
	const int projY = W;
	const dim3 pr_threads(32, 8);
	const dim3 pr_blocks((unsigned)((W + 31) / 32), (unsigned)((W + 7) / 8));
	const XFLOAT dpsi_rad = DEG2RAD(basePckr->psi_sampling);
	const XFLOAT pr_normfft = (XFLOAT)(-2. * normfft);

	for (int iref = 0; iref < n_refs; iref++)
	{
		const int buf = basePckr->autopick_helical_segments ? 0 : (iref & 1);
		AccProjectorKernel projKernel = AccProjectorKernel::makeKernel(
				projectors[iref],
				projX,
				projY,
				1,
				W / 2);

		int startPsi = 0;
		for (int psiIter = 0; psiIter < (int)cudaTransformer1.batchIters; psiIter++)
		{
			const int npsi_batch = cudaTransformer1.batchSize[psiIter];
			dim3 rc_blocks(
					(unsigned)((projX + 15) / 16),
					(unsigned)((projY + 15) / 16),
					(unsigned)npsi_batch);
			if (basePckr->do_ctf)
			{
				cuda_kernel_rotateAndConvol<true><<<rc_blocks, rc_threads>>>(
						~cudaTransformer1.fouriers,
						~d_ctf,
						~d_Fmic,
						dpsi_rad,
						projKernel,
						startPsi);
			}
			else
			{
				cuda_kernel_rotateAndConvol<false><<<rc_blocks, rc_threads>>>(
						~cudaTransformer1.fouriers,
						NULL,
						~d_Fmic,
						dpsi_rad,
						projKernel,
						startPsi);
			}
			LAUNCH_HANDLE_ERROR(cudaGetLastError());

			cudaTransformer1.backward();

			cuda_kernel_probRatio_centered<<<pr_blocks, pr_threads>>>(
					~d_Mccf_best[buf],
					~d_Mpsi_best[buf],
					~cudaTransformer1.reals,
					~d_Mmean,
					~d_Mstddev_inv,
					W,
					pr_normfft,
					(XFLOAT)(2. * sum_ref_vec[iref]),
					(XFLOAT)sum_ref2_vec[iref],
					(XFLOAT)expected_Pratio_vec[iref],
					npsi_batch,
					startPsi,
					Npsi,
					(psiIter == 0));
			LAUNCH_HANDLE_ERROR(cudaGetLastError());
			startPsi += npsi_batch;
		}

		if (basePckr->autopick_helical_segments)
		{
			int comb_blocks = (int)ceilf((float)work_sq / (float)BLOCK_SIZE);
			cuda_kernel_combineHelicalBest<<<comb_blocks, BLOCK_SIZE>>>(
					~d_Mccf_best[0],
					~d_Mpsi_best[0],
					~d_Mccf_combined,
					basePckr->do_amyloid ? ~d_Mpsi_combined : NULL,
					basePckr->do_amyloid ? NULL : ~d_Mclass_combined,
					work_sq,
					iref,
					basePckr->do_amyloid);
			LAUNCH_HANDLE_ERROR(cudaGetLastError());
		}
		else
		{
			HANDLE_ERROR(cudaMemcpyAsync(h_ccf[buf], ~d_Mccf_best[buf], work_sq * sizeof(XFLOAT), cudaMemcpyDeviceToHost, 0));
			HANDLE_ERROR(cudaMemcpyAsync(h_psi[buf], ~d_Mpsi_best[buf], work_sq * sizeof(XFLOAT), cudaMemcpyDeviceToHost, 0));
			HANDLE_ERROR(cudaEventRecord(ev_ref_done[buf], 0));

			// Overlap host peak search/prune for previous ref with GPU execution of current ref
			if (iref > 0)
				finishHostRef(iref - 1);
		}
	}

	if (!basePckr->autopick_helical_segments && n_refs > 0)
		finishHostRef(n_refs - 1);

	if (basePckr->autopick_helical_segments)
	{
		MultidimArray<RFLOAT> Mstddev2, Mavg, Mccf_best_combined, Mpsi_best_combined;
		MultidimArray<int> Mclass_best_combined;

		HANDLE_ERROR(cudaMemcpyAsync(h_Mstddev, ~d_Mstddev2, work_sq * sizeof(XFLOAT), cudaMemcpyDeviceToHost, 0));
		HANDLE_ERROR(cudaMemcpyAsync(h_Mmean, ~d_Mavg, work_sq * sizeof(XFLOAT), cudaMemcpyDeviceToHost, 0));
		HANDLE_ERROR(cudaMemcpyAsync(h_ccf[0], ~d_Mccf_combined, work_sq * sizeof(XFLOAT), cudaMemcpyDeviceToHost, 0));
		if (basePckr->do_amyloid)
			HANDLE_ERROR(cudaMemcpyAsync(h_psi[0], ~d_Mpsi_combined, work_sq * sizeof(XFLOAT), cudaMemcpyDeviceToHost, 0));
		else
			HANDLE_ERROR(cudaMemcpyAsync(h_class_combined, ~d_Mclass_combined, work_sq * sizeof(int), cudaMemcpyDeviceToHost, 0));
		HANDLE_ERROR(cudaStreamSynchronize(0));

		Mstddev2.resizeNoCp(1, W, W);
		Mavg.resizeNoCp(1, W, W);
		Mccf_best_combined.resizeNoCp(1, W, W);
		if (basePckr->do_amyloid)
			Mpsi_best_combined.resizeNoCp(1, W, W);
		else
			Mclass_best_combined.resizeNoCp(1, W, W);

		for (size_t k = 0; k < work_sq; k++)
		{
			Mstddev2.data[k] = (RFLOAT)h_Mstddev[k];
			Mavg.data[k] = (RFLOAT)h_Mmean[k];
			Mccf_best_combined.data[k] = (RFLOAT)h_ccf[0][k];
			if (basePckr->do_amyloid)
				Mpsi_best_combined.data[k] = (RFLOAT)h_psi[0][k];
			else
				Mclass_best_combined.data[k] = h_class_combined[k];
		}

		if (basePckr->do_write_fom_maps)
		{
			FileName fn_tmp;
			Image<RFLOAT> It_float;
			Image<int> It_int;

			It_float() = Mstddev2;
			fn_tmp = basePckr->getOutputRootName(fn_mic) + "_" + basePckr->fn_out + "_stddevNoise.spi";
			It_float.write(fn_tmp);

			It_float() = Mavg;
			fn_tmp = basePckr->getOutputRootName(fn_mic) + "_" + basePckr->fn_out + "_avgNoise.spi";
			It_float.write(fn_tmp);

			It_float() = Mccf_best_combined;
			fn_tmp = basePckr->getOutputRootName(fn_mic) + "_" + basePckr->fn_out + "_combinedCCF.spi";
			It_float.write(fn_tmp);

			if (basePckr->do_amyloid)
			{
				It_float() = Mpsi_best_combined;
				fn_tmp = basePckr->getOutputRootName(fn_mic) + "_" + basePckr->fn_out + "_combinedPSI.spi";
				It_float.write(fn_tmp);
			}
			else
			{
				It_int() = Mclass_best_combined;
				fn_tmp = basePckr->getOutputRootName(fn_mic) + "_" + basePckr->fn_out + "_combinedCLASS.spi";
				It_int.write(fn_tmp);
			}
		}

		RFLOAT thres = basePckr->min_fraction_expected_Pratio;
		int peak_r_min = 1;
		std::vector<ccfPeak> ccf_peak_list;
		std::vector<std::vector<ccfPeak> > tube_coord_list, tube_track_list;
		std::vector<RFLOAT> tube_len_list;
		MultidimArray<RFLOAT> Mccfplot;

		Mccf_best_combined.setXmippOrigin();
		Mpsi_best_combined.setXmippOrigin();
		Mstddev2.setXmippOrigin();
		Mavg.setXmippOrigin();
		Mclass_best_combined.setXmippOrigin();
		FileName fn_mic_mut = fn_mic;
		if (basePckr->do_amyloid)
		{
			basePckr->pickAmyloids(Mccf_best_combined, Mpsi_best_combined, Mstddev2, Mavg, thres, basePckr->amyloid_max_psidiff, fn_mic_mut, basePckr->fn_out,
					(basePckr->helical_tube_diameter / basePckr->angpix), basePckr->autopick_skip_side, scale);
		}
		else
		{
			basePckr->pickCCFPeaks(Mccf_best_combined, Mstddev2, Mavg, Mclass_best_combined, thres, peak_r_min, (basePckr->particle_diameter / basePckr->angpix),
					ccf_peak_list, Mccfplot, my_skip_side, scale);
			basePckr->extractHelicalTubes(ccf_peak_list, tube_coord_list, tube_len_list, tube_track_list,
					(basePckr->particle_diameter / basePckr->angpix), basePckr->helical_tube_curvature_factor_max,
					(basePckr->min_particle_distance / basePckr->angpix), (basePckr->helical_tube_diameter / basePckr->angpix), scale);
			basePckr->exportHelicalTubes(Mccf_best_combined, Mccfplot, Mclass_best_combined,
					tube_coord_list, tube_track_list, tube_len_list,
					fn_mic_mut, basePckr->fn_out,
					(basePckr->particle_diameter / basePckr->angpix),
					(basePckr->helical_tube_length_min / basePckr->angpix),
					my_skip_side, scale);
		}

		if (basePckr->do_write_fom_maps && !basePckr->do_amyloid)
		{
			FileName fn_tmp;
			Image<RFLOAT> It;
			It() = Mccfplot;
			fn_tmp = basePckr->getOutputRootName(fn_mic) + "_" + basePckr->fn_out + "_combinedPLOT.spi";
			It.write(fn_tmp);
		}
	}
	else
	{
#ifdef TIMING
		basePckr->timer.tic(basePckr->TIMING_B9);
#endif
		basePckr->prunePeakClusters(peaks, min_distance_pix, scale);
		basePckr->removeTooCloselyNeighbouringPeaks(peaks, min_distance_pix, scale);

		auto md_ptr = std::make_shared<MetaDataTable>();
		for (int ipeak = 0; ipeak < (int)peaks.size(); ipeak++)
		{
			md_ptr->addObject();
			md_ptr->setValue(EMDL_IMAGE_COORD_X, (RFLOAT)(peaks[ipeak].x) / scale);
			md_ptr->setValue(EMDL_IMAGE_COORD_Y, (RFLOAT)(peaks[ipeak].y) / scale);
			md_ptr->setValue(EMDL_PARTICLE_CLASS, peaks[ipeak].ref + 1);
			md_ptr->setValue(EMDL_PARTICLE_AUTOPICK_FOM, peaks[ipeak].fom);
			md_ptr->setValue(EMDL_ORIENT_PSI, peaks[ipeak].psi);
		}
		FileName fn_tmp = basePckr->getOutputRootName(fn_mic) + "_" + basePckr->fn_out + ".star";
		writeAutoPickStar(fn_tmp, md_ptr);
#ifdef TIMING
		basePckr->timer.toc(basePckr->TIMING_B9);
#endif
	}
}
