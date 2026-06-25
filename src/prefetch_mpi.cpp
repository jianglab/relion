/***************************************************************************
 *
 * Author: "Wen Jiang"
 * Pennsylvania State University
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/prefetch_mpi.h"
#include "src/funcs.h"

MpiAsyncImagePrefetcher::MpiAsyncImagePrefetcher(Experiment *mydata, MpiNode *node)
	:
	mydata_(mydata),
	node_(node),
	fill_idx_(0),
	ready_idx_(-1),
	has_work_(true),
	shutdown_(false)
{}

MpiAsyncImagePrefetcher::~MpiAsyncImagePrefetcher()
{
	shutdown_ = true;
	cv_fill_.notify_all();
	if (worker_thread_.joinable())
		worker_thread_.join();
}

void MpiAsyncImagePrefetcher::start()
{
	worker_thread_ = std::thread(&MpiAsyncImagePrefetcher::workerThread, this);
}

bool MpiAsyncImagePrefetcher::waitForJob(std::vector<MultidimArray<RFLOAT>>& images,
        MultidimArray<RFLOAT>& metadata,
        long int& job_first,
        long int& job_last)
{
	std::unique_lock<std::mutex> lock(mtx_);
	cv_ready_.wait(lock, [this]() { return ready_idx_ >= 0 || !has_work_; });

	if (!has_work_)
		return false;

	JobData &src = buffers_[ready_idx_];
	images.swap(src.images);
	metadata = src.metadata;
	job_first = src.job_first;
	job_last = src.job_last;

	// Release this buffer and immediately signal the worker to
	// start loading the next job (overlaps I/O with computation)
	ready_idx_ = -1;
	fill_idx_ = 1 - fill_idx_;
	lock.unlock();
	cv_fill_.notify_one();

	return true;
}

void MpiAsyncImagePrefetcher::stop()
{
	shutdown_ = true;
	cv_fill_.notify_one();
	if (worker_thread_.joinable())
		worker_thread_.join();
}

void MpiAsyncImagePrefetcher::workerThread()
{
	while (!shutdown_)
	{
		int buf_idx;
		{
			std::unique_lock<std::mutex> lock(mtx_);
			// Wait until this buffer is safe to fill (not the same as ready_idx_,
			// which means the main thread is still consuming it)
			cv_fill_.wait(lock, [this]() {
				return (fill_idx_ != ready_idx_ && fill_idx_ >= 0) || shutdown_;
			});
			if (shutdown_)
				break;
			buf_idx = fill_idx_;
		}

		// Send request to leader
		int dummy = 0;
		node_->relion_MPI_Send(&dummy, 1, MPI_INT, 0, MPITAG_PREFETCH_REQ, MPI_COMM_WORLD);

		// Receive job assignment from leader
		long int job_msg[6];
		MPI_Status status;
		node_->relion_MPI_Recv(job_msg, 6, MPI_LONG, 0, MPITAG_PREFETCH_REQ, MPI_COMM_WORLD, status);

		JobData &buf = buffers_[buf_idx];
		buf.job_first = job_msg[0];
		buf.job_last  = job_msg[1];
		buf.nimg      = job_msg[2];
		long int len_fn_img  = job_msg[3];
		long int len_fn_ctf  = job_msg[4];
		long int len_fn_recimg = job_msg[5];

		if (buf.job_first < 0)
		{
			// No more work
			std::lock_guard<std::mutex> lock(mtx_);
			has_work_ = false;
			ready_idx_ = buf_idx;
			cv_ready_.notify_one();
			return;
		}
		else
		{
		}

		// Receive number of metadata columns from leader
		long int num_meta_cols;
		node_->relion_MPI_Recv(&num_meta_cols, 1, MPI_LONG, 0, MPITAG_PREFETCH_REQ, MPI_COMM_WORLD, status);

		// Receive metadata (num_meta_cols values per image)
		buf.metadata.resize(buf.nimg, num_meta_cols);
		node_->relion_MPI_Recv(MULTIDIM_ARRAY(buf.metadata), MULTIDIM_SIZE(buf.metadata), MY_MPI_DOUBLE, 0, MPITAG_METADATA, MPI_COMM_WORLD, status);

		// Receive filename string for the images and read them from disk
		if (len_fn_img > 0)
		{
			char *rec_buf = (char *)malloc(len_fn_img);
			node_->relion_MPI_Recv(rec_buf, len_fn_img, MPI_CHAR, 0, MPITAG_METADATA, MPI_COMM_WORLD, status);
			std::string fn_img_str(rec_buf);
			free(rec_buf);

			std::istringstream split(fn_img_str);
			std::string fn;
			buf.images.clear();
			buf.images.reserve(buf.nimg);
			for (int i = 0; i < buf.nimg; i++)
			{
				getline(split, fn);
				Image<RFLOAT> img;
				img.read(fn);
				img().setXmippOrigin();
				buf.images.push_back(img());
			}
		}

		// Receive (and discard) CTF filename string if present
		if (len_fn_ctf > 1)
		{
			char *rec_buf = (char *)malloc(len_fn_ctf);
			node_->relion_MPI_Recv(rec_buf, len_fn_ctf, MPI_CHAR, 0, MPITAG_METADATA, MPI_COMM_WORLD, status);
			free(rec_buf);
		}

		// Receive (and discard) reconstruct filename string if present
		if (len_fn_recimg > 1)
		{
			char *rec_buf = (char *)malloc(len_fn_recimg);
			node_->relion_MPI_Recv(rec_buf, len_fn_recimg, MPI_CHAR, 0, MPITAG_METADATA, MPI_COMM_WORLD, status);
			free(rec_buf);
		}

		// Signal main thread that data is ready
		{
			std::lock_guard<std::mutex> lock(mtx_);
			ready_idx_ = buf_idx;
		}
		cv_ready_.notify_one();
	}
}
