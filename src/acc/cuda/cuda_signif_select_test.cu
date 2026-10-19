/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

// Host entry point for the unit tests of the significance selection: src/acc/cuda/cuda_signif_select_test.h

#include "src/acc/cuda/cuda_signif_select_test.h"
#include "src/acc/acc_ptr.h"
#include "src/acc/cuda/custom_allocator.cuh"
#include "src/acc/cuda/cuda_utils_cub.cuh"

#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>

namespace gpusel
{

bool runFusedWeightPass(const std::vector<float> &log_weights, double adaptive_fraction, long max_significants,
		bool radix, Outcome &out)
{
	int ndev = 0;
	if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1)
	{
		cudaGetLastError();
		return false;
	}
	typedef CudaKernels::SignificanceResult<float> SigRes;
	const size_t n = log_weights.size();
	CudaCustomAllocator pool(0, 512);
	pool.configureLean((size_t) 512 << 20, (size_t) 128 << 20);
	AccPtrFactory factory(&pool, cudaStreamPerThread);

	AccPtr<float> w = factory.make<float>(n);
	w.allAlloc();
	for (size_t i = 0; i < n; i++)
		w[i] = log_weights[i];
	w.cpToDevice();

	AccPtr<char> res = factory.make<char>(sizeof(SigRes) + n);
	res.allAlloc();

	CudaKernels::gpuSignifSelectOverride() = radix ? 1 : 0;
	CudaKernels::fusedWeightPass<float>(w, adaptive_fraction, max_significants, ~res, true);
	CudaKernels::gpuSignifSelectOverride() = -1;
	res.cpToHostWait();
	w.cpToHost();
	cudaStreamSynchronize(cudaStreamPerThread);

	SigRes r;
	memcpy(&r, &res[0], sizeof(SigRes));
	out.sum_weight = r.sum_weight;
	out.threshold = r.threshold;
	out.significant_weight = r.significant_weight;
	out.max_value = r.max.value;
	out.max_key = r.max.key;
	out.idx = r.idx;
	out.idx_capped = r.idx_capped;
	out.weights.assign(&w[0], &w[0] + n);
	out.mask.assign(&res[sizeof(SigRes)], &res[sizeof(SigRes)] + n);
	return true;
}


namespace
{
__global__ void stressSpinKernel(long long cycles, int *sink)
{
	const long long t0 = clock64();
	while (clock64() - t0 < cycles) { }
	if (threadIdx.x == 0 && blockIdx.x == 0 && cycles < 0)
		*sink = 1;
}
}

bool stressRadixSelect(int nthreads, double seconds, size_t n, bool compete, double stall_seconds, StressOutcome &out)
{
	int ndev = 0;
	if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1)
	{
		cudaGetLastError();
		return false;
	}
	int coop = 0, sms = 0;
	cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0);
	cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
	out.cooperative = coop != 0;

	typedef CudaKernels::SignificanceResult<float> SigRes;
	const long fused0 = CudaKernels::rsFusedCount(0).load(), fall0 = CudaKernels::rsFusedCount(1).load();
	std::atomic<long> progress(0), mism(0);
	std::atomic<bool> stop(false);
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds((long) (seconds * 1000));

	std::vector<std::thread> workers;
	for (int t = 0; t < nthreads; t++)
	{
		workers.emplace_back([&, t]() {
			cudaSetDevice(0);
			cudaStream_t stream;
			cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
			std::mt19937 rng(100 + t);
			std::vector<float> h(n);
			for (size_t i = 0; i < n; i++)
				h[i] = std::exp(-6.f * (float) (rng() % 100000) / 100000.f);
			float *d_w = NULL;
			void *d_scratch = NULL;
			SigRes *d_r = NULL;
			cudaMalloc(&d_w, n * sizeof(float));
			cudaMalloc(&d_scratch, CudaKernels::radixSelectScratchBytes());
			cudaMalloc(&d_r, sizeof(SigRes));
			cudaMemcpyAsync(d_w, h.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
			cudaStreamSynchronize(stream); // the stream is non-blocking: a plain cudaMemcpy would not be ordered before the first selection
			SigRes first;
			bool have_first = false;
			while (!stop.load() && std::chrono::steady_clock::now() < deadline)
			{
				CudaKernels::radixSelectSignificance(d_w, n, stream, d_scratch, 0.999, 0, d_r);
				SigRes r;
				cudaMemcpyAsync(&r, d_r, sizeof(SigRes), cudaMemcpyDeviceToHost, stream);
				cudaStreamSynchronize(stream);
				if (!have_first)
				{
					first = r;
					have_first = true;
				}
				else if (r.significant_weight != first.significant_weight || r.idx != first.idx || r.idx_capped != first.idx_capped)
				{
					if (mism++ < 5)
						fprintf(stderr, "  stress mismatch: thread %d selection %ld: idx %llu vs first %llu, idx_capped %llu vs %llu, significant_weight %.9g vs %.9g, sum_weight %.12g vs %.12g, threshold %.9g vs %.9g\n",
								t, progress.load(), (unsigned long long) r.idx, (unsigned long long) first.idx, (unsigned long long) r.idx_capped,
								(unsigned long long) first.idx_capped, (double) r.significant_weight, (double) first.significant_weight,
								(double) r.sum_weight, (double) first.sum_weight, (double) r.threshold, (double) first.threshold);
				}
				progress++;
			}
			cudaFree(d_w);
			cudaFree(d_scratch);
			cudaFree(d_r);
			cudaStreamDestroy(stream);
		});
	}

	std::thread competitor;
	if (compete)
	{
		competitor = std::thread([&]() {
			cudaSetDevice(0);
			cudaStream_t stream;
			cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
			int *sink = NULL;
			cudaMalloc(&sink, sizeof(int));
			while (!stop.load() && std::chrono::steady_clock::now() < deadline)
			{
				// about 20 ms of work on every multiprocessor (one 1024-thread block per SM, 2 per SM where they fit)
				stressSpinKernel<<<2 * sms, 512, 0, stream>>>(30000000LL, sink);
				cudaStreamSynchronize(stream);
			}
			cudaFree(sink);
			cudaStreamDestroy(stream);
		});
	}

	long last = -1;
	auto last_move = std::chrono::steady_clock::now();
	bool stalled = false;
	while (std::chrono::steady_clock::now() < deadline + std::chrono::seconds(2))
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
		const long p = progress.load();
		const auto now = std::chrono::steady_clock::now();
		if (p != last)
		{
			last = p;
			last_move = now;
		}
		else if (std::chrono::duration<double>(now - last_move).count() > stall_seconds)
		{
			stalled = true;
			break;
		}
		if (now > deadline && p == last)
			break;
	}
	if (stalled)
	{
		// the worker threads are stuck in a kernel that never ends: nothing can be joined or freed
		out.stalled = true;
		out.launches = progress.load();
		out.mismatches = mism.load();
		for (auto &w : workers)
			w.detach();
		if (competitor.joinable())
			competitor.detach();
		return true;
	}
	stop = true;
	for (auto &w : workers)
		w.join();
	if (competitor.joinable())
		competitor.join();
	out.launches = progress.load();
	out.mismatches = mism.load();
	out.fused_launches = CudaKernels::rsFusedCount(0).load() - fused0;
	out.fallbacks = CudaKernels::rsFusedCount(1).load() - fall0;
	return true;
}


bool radixSelectOnce(const std::vector<float> &weights, bool refuse, OnceOutcome &out)
{
	int ndev = 0;
	if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1 || cudaSetDevice(0) != cudaSuccess)
	{
		cudaGetLastError();
		return false;
	}
	typedef CudaKernels::SignificanceResult<float> SigRes;
	const size_t n = weights.size();
	cudaStream_t stream;
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	float *d_w = NULL;
	void *d_scratch = NULL;
	SigRes *d_r = NULL;
	cudaMalloc(&d_w, n * sizeof(float));
	cudaMalloc(&d_scratch, CudaKernels::radixSelectScratchBytes());
	cudaMalloc(&d_r, sizeof(SigRes));
	cudaMemcpyAsync(d_w, weights.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
	const long fused0 = CudaKernels::rsFusedCount(0).load(), fall0 = CudaKernels::rsFusedCount(1).load();
	const int old = CudaKernels::rsFusedRefuse().exchange(refuse ? 1 : 0);
	CudaKernels::radixSelectSignificance(d_w, n, stream, d_scratch, 0.999, 0, d_r);
	SigRes r;
	cudaMemcpyAsync(&r, d_r, sizeof(SigRes), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
	CudaKernels::rsFusedRefuse().store(old);
	out.sum_weight = r.sum_weight;
	out.threshold = r.threshold;
	out.significant_weight = r.significant_weight;
	out.idx = r.idx;
	out.idx_capped = r.idx_capped;
	out.fused_launches = CudaKernels::rsFusedCount(0).load() - fused0;
	out.fallbacks = CudaKernels::rsFusedCount(1).load() - fall0;
	cudaFree(d_w);
	cudaFree(d_scratch);
	cudaFree(d_r);
	cudaStreamDestroy(stream);
	return cudaGetLastError() == cudaSuccess;
}

}
