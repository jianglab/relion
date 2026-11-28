#ifndef JAZ_MOVIE_LOADER_H
#define JAZ_MOVIE_LOADER_H

#include <src/image.h>
#include <src/mem_tiff.h>
#include <src/jaz/image/buffered_image.h>
#include <src/renderEER.h>
#include <cstdint>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <mutex>

class MovieLoader
{
	// An exception must not leave an OpenMP loop: keep the first one and throw it after the loop
	class ReadErrors
	{
		public:
			bool failed() { std::lock_guard<std::mutex> g(m); return has; }
			void keep(const RelionError &e) { std::lock_guard<std::mutex> g(m); if (!has) { first.reset(new RelionError(e)); has = true; } }
			void rethrow() { if (has) throw *first; }
		private:
			std::mutex m;
			bool has = false;
			std::unique_ptr<RelionError> first;
	};

	public:

		template <typename T>
		static BufferedImage<T> readDense(
				std::string movieFn,
				const RawImage<RFLOAT>* gainRef,
				const RawImage<bool>* defectivePixels,
				int frame0,
				int numFrames,
				RFLOAT hot,
				int num_threads);

		template <typename T>
		static BufferedImage<T> readEER(
				std::string movieFn,
				const RawImage<RFLOAT>* gainRef,
				const RawImage<bool>* defectivePixels,
				int frame0,
				int numFrames,
				int eer_upsampling,
				int eer_grouping,
				int num_threads);

		// Defective pixels get a good neighbour, or Gaussian noise, chosen by a hash of (frame, x, y)
		// instead of rand(), so the result is the same in every run, at any thread count
		static std::vector<std::pair<long int, long int>> collectDefects(
				const RawImage<bool>* defectivePixels,
				int num_threads);

		template <typename T>
		static void fixDefects(
				RawImage<T>& muGraphFrame,
				const RawImage<bool>* defectivePixels,
				const std::vector<std::pair<long int, long int>>& defects,
				int frame,
				int num_threads, bool isEER);

		template <typename T>
		static void fixDefects(
				RawImage<T>& muGraphFrame,
				const RawImage<bool>* defectivePixels,
				int frame,
				int num_threads, bool isEER);

		// splitmix64, as in MotioncorrRunner
		static uint64_t reproducibleHash(uint64_t x)
		{
			x += 0x9E3779B97F4A7C15ULL;
			x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
			x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
			return x ^ (x >> 31);
		}
};

template <typename T>
BufferedImage<T> MovieLoader::readDense(
			std::string movieFn,
			const RawImage<RFLOAT>* gainRef,
			const RawImage<bool>* defectivePixels,
			int frame0,
			int numFrames,
			RFLOAT hot,
			int num_threads)
{
	const bool isCompressedMRC = CompressedMRCReader::isCompressedMRC(movieFn);
	const bool isTiff = !isCompressedMRC && memtiff::TiffMovieBuffer::isTiffFilename(movieFn);

	CompressedMRCReader reader;
	memtiff::TiffMovieBuffer tiffBuf;
	bool useMemTiff = false;

	long int w0 = 0, h0 = 0, nTotal = 0, zDim = 1;
	if (isCompressedMRC)
	{
		reader.read(movieFn, num_threads);
		w0 = XSIZE(reader.Ihead());
		h0 = YSIZE(reader.Ihead());
		zDim = ZSIZE(reader.Ihead());
		nTotal = NSIZE(reader.Ihead());
	}
	else if (isTiff && tiffBuf.load(movieFn))
	{
		useMemTiff = true;
		w0 = tiffBuf.xDim;
		h0 = tiffBuf.yDim;
		nTotal = tiffBuf.nFrames;
	}
	else
	{
		Image<float> mgStack;
		mgStack.read(movieFn, false, -1, false, true); // final true means 2D movies, not 3D map
		w0 = mgStack.data.xdim;
		h0 = mgStack.data.ydim;
		zDim = mgStack.data.zdim;
		nTotal = mgStack.data.ndim;
	}

	const std::string tag = "MovieLoader::readDense: ";

	if (zDim > 1)
	{
		REPORT_ERROR_STR(tag << "the file " << movieFn << " looks like a 3D image and not a stack "
						 << "(the slices are along the Z dimension instead of N)");
	}

	const long int pixCt = w0 * h0;
	const int fc = numFrames;


	if (nTotal < frame0 + numFrames)
	{
		REPORT_ERROR_STR(
			tag << "insufficient number of frames in " << movieFn
			<< " (found: " << nTotal << ", expected: " << frame0 + numFrames << ")");
	}

	const bool useGain = gainRef != 0;
	if (useGain && (w0 != gainRef->xdim || h0 != gainRef->ydim))
	{
		REPORT_ERROR_STR(tag << "incompatible gain reference - size is different from " << movieFn);
	}

	const bool do_fixDefect = defectivePixels != 0;
	if (do_fixDefect && (w0 != defectivePixels->xdim || h0 != defectivePixels->ydim))
	{
		REPORT_ERROR_STR(tag << "incompatible defect mask - size is different from " << movieFn);
	}
	const std::vector<std::pair<long int, long int>> defectCoords =
		do_fixDefect ? collectDefects(defectivePixels, num_threads) : std::vector<std::pair<long int, long int>>();
	const bool run_fixDefect = do_fixDefect && !defectCoords.empty();

	BufferedImage<T> out(w0, h0, fc);
	ReadErrors errors;

	// Frames in parallel, each thread with its own file/memory handle (the compressed-MRC reader is shared, so one at a time)
	const bool par_frames = (fc > 1 && !isCompressedMRC);
	const int inner_threads = par_frames ? 1 : std::max(1, num_threads);
	#pragma omp parallel num_threads(num_threads) if(par_frames)
	{
		std::unique_ptr<memtiff::TiffMovieBuffer::ThreadHandle> tiffHandle;
		MultidimArray<float> memFrame;
		if (useMemTiff)
			tiffHandle.reset(new memtiff::TiffMovieBuffer::ThreadHandle(tiffBuf));

		#pragma omp for schedule(static)
		for (long f = 0; f < fc; f++)
		{
			if (errors.failed()) continue;
			try {
			Image<float> muGraphFrame_xmipp;
			const float* __restrict srcPtr = 0;

			if (useMemTiff)
			{
				tiffBuf.decodeFrame(*tiffHandle, frame0 + f, memFrame);
				srcPtr = MULTIDIM_ARRAY(memFrame);
			}
			else if (isCompressedMRC)
			{
				reader.readFrameInto(muGraphFrame_xmipp, frame0 + f);
				srcPtr = MULTIDIM_ARRAY(muGraphFrame_xmipp());
			}
			else
			{
				muGraphFrame_xmipp.read(movieFn, true, frame0 + f, false, true);
				srcPtr = MULTIDIM_ARRAY(muGraphFrame_xmipp());
			}

			RawImage<T> dst = out.getSliceRef(f);

			#pragma omp parallel for num_threads(inner_threads) if(inner_threads > 1)
			for (long int i = 0; i < pixCt; i++)
			{
				RFLOAT val = srcPtr[i];
				const RFLOAT gain = useGain? (*gainRef)[i] : 1;

				if (hot > 0.0 && val > hot) val = hot;

				dst[i] = (T)(-gain * val);
			}

			if (run_fixDefect)
			{
				fixDefects(dst, defectivePixels, defectCoords, frame0 + f, inner_threads, false);
			}
			} catch (RelionError &e) { errors.keep(e); }
		}
	}

	errors.rethrow();
	return out;
}

template <typename T>
BufferedImage<T> MovieLoader::readEER(
		std::string movieFn,
		const RawImage<RFLOAT>* gainRef,
		const RawImage<bool>* defectivePixels,
		int frame0,
		int numFrames,
		int eer_upsampling,
		int eer_grouping,
		int num_threads)
{
	EERRenderer renderer;
	renderer.read(movieFn, eer_upsampling);

	const long int w0 = renderer.getWidth();
	const long int h0 = renderer.getHeight();
	const long int pixCt = w0 * h0;
	const int fc = numFrames;
	const std::string tag = "MovieLoader::readDense: ";


	const bool useGain = gainRef != 0;
	if (useGain && (w0 != gainRef->xdim || h0 != gainRef->ydim))
	{
		REPORT_ERROR_STR(tag << "incompatible gain reference - size (x = " <<  gainRef->xdim <<
                                 ", y = " << gainRef->ydim << ") is different from " << movieFn <<
                                 " (x = " << w0 << ", y = " << h0 << ")");
	}

	const bool do_fixDefect = defectivePixels != 0;
	if (do_fixDefect && (w0 != defectivePixels->xdim || h0 != defectivePixels->ydim))
	{
		REPORT_ERROR_STR(tag << "incompatible defect mask - size (x = " << defectivePixels->xdim <<
                                 ", y = " << defectivePixels->ydim << ") is different from " << movieFn <<
                                 " (x = " << w0 << ", y = " << h0 << ")");
	}
	const std::vector<std::pair<long int, long int>> defectCoords =
		do_fixDefect ? collectDefects(defectivePixels, num_threads) : std::vector<std::pair<long int, long int>>();
	const bool run_fixDefect = do_fixDefect && !defectCoords.empty();

	BufferedImage<T> out(w0, h0, fc);
	ReadErrors errors;

	const bool par_frames = (fc > 1);
	const int inner_threads = par_frames ? 1 : std::max(1, num_threads);
	#pragma omp parallel num_threads(num_threads) if(par_frames)
	{
		MultidimArray<float> muGraphFrame_xmipp;

		#pragma omp for schedule(static)
		for (int f = 0; f < fc; f++)
		{
			if (errors.failed()) continue;
			try {
			// this takes 1-indexed frame numbers
			renderer.renderFrames(
						(frame0 + f) * eer_grouping + 1,
						(frame0 + f + 1) * eer_grouping,
						muGraphFrame_xmipp);

			RawImage<T> muGraphFrame(muGraphFrame_xmipp);
			RawImage<T> dst = out.getSliceRef(f);

			#pragma omp parallel for num_threads(inner_threads) if(inner_threads > 1)
			for (long int i = 0; i < pixCt; i++)
			{
				const RFLOAT val = muGraphFrame[i];
				const RFLOAT gain = useGain? (*gainRef)[i] : 1.0;

				dst[i] = (T)(-gain * val);
			}

			if (run_fixDefect)
			{
				fixDefects(dst, defectivePixels, defectCoords, frame0 + f, inner_threads, true);
			}
			} catch (RelionError &e) { errors.keep(e); }
		}
	}

	errors.rethrow();
	return out;
}

inline std::vector<std::pair<long int, long int>> MovieLoader::collectDefects(
		const RawImage<bool>* defectivePixels,
		int num_threads)
{
	std::vector<std::pair<long int, long int>> defects;
	if (defectivePixels == 0) return defects;

	const long int w0 = defectivePixels->xdim;
	const long int h0 = defectivePixels->ydim;
	if (w0 <= 0 || h0 <= 0) return defects;

	if (num_threads > 1 && w0 * h0 >= 65536 && h0 > 1)
	{
		std::vector<std::vector<long int>> per_row(h0);
		#pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads)
		for (long int y = 0; y < h0; y++)
		{
			for (long int x = 0; x < w0; x++)
			{
				if ((*defectivePixels)(x, y))
					per_row[y].push_back(x);
			}
		}
		for (long int y = 0; y < h0; y++)
		{
			for (long int x : per_row[y])
				defects.emplace_back(x, y);
		}
	}
	else
	{
		for (long int y = 0; y < h0; y++)
		for (long int x = 0; x < w0; x++)
		{
			if ((*defectivePixels)(x, y))
				defects.emplace_back(x, y);
		}
	}
	return defects;
}

template <typename T>
void MovieLoader::fixDefects(
		RawImage<T>& muGraphFrame,
		const RawImage<bool>* defectivePixels,
		int frame,
		int num_threads, bool isEER)
{
	std::vector<std::pair<long int, long int>> defects = collectDefects(defectivePixels, num_threads);
	if (defects.empty()) return;
	fixDefects(muGraphFrame, defectivePixels, defects, frame, num_threads, isEER);
}

template <typename T>
void MovieLoader::fixDefects(
		RawImage<T>& muGraphFrame,
		const RawImage<bool>* defectivePixels,
		const std::vector<std::pair<long int, long int>>& defects,
		int frame,
		int num_threads, bool isEER)
{
	if (defects.empty()) return;

	const long int w0 = muGraphFrame.xdim;
	const long int h0 = muGraphFrame.ydim;
	const long int pixCt = w0 * h0;

	const int min_num_ok = 6;
	const int d_max = isEER ? 4: 2;
	const int PBUF_SIZE = 100;

	std::vector<std::pair<long int, long int>> need_stats;
	std::vector<std::pair<long int, RFLOAT>> neighbor_fills(defects.size());
	std::vector<char> has_neighbor_fill(defects.size(), 0);

	#pragma omp parallel for num_threads(num_threads) if(defects.size() > 64)
	for (size_t k = 0; k < defects.size(); k++)
	{
		const long int x = defects[k].first;
		const long int y = defects[k].second;
		int n_ok = 0;
		RFLOAT pbuf[PBUF_SIZE];

		for (int dy = -d_max; dy <= d_max; dy++)
		for (int dx = -d_max; dx <= d_max; dx++)
		{
			const int yy = y + dy;
			const int xx = x + dx;

			if (xx < 0 || xx >= w0 || yy < 0 || yy >= h0 || (*defectivePixels)(xx,yy))
				continue;

			pbuf[n_ok] = muGraphFrame(xx,yy);
			n_ok++;
		}
		if (n_ok > min_num_ok)
		{
			const uint64_t hash = reproducibleHash(((uint64_t)frame << 42) ^ ((uint64_t)y << 21) ^ (uint64_t)x);
			has_neighbor_fill[k] = 1;
			neighbor_fills[k] = std::make_pair(y * w0 + x, pbuf[hash % n_ok]);
		}
	}

	for (size_t k = 0; k < defects.size(); k++)
	{
		if (has_neighbor_fill[k])
			muGraphFrame[neighbor_fills[k].first] = neighbor_fills[k].second;
		else
			need_stats.push_back(defects[k]);
	}

	if (need_stats.empty()) return;

	RFLOAT frame_mean = 0;
	long int n_valid = 0;

	#pragma omp parallel for reduction(+:frame_mean, n_valid) num_threads(num_threads)
	for (long int n = 0; n < pixCt; ++n)
	{
		if (!(*defectivePixels)[n])
		{
			frame_mean += muGraphFrame[n];
			n_valid ++;
		}
	}

	frame_mean /= n_valid;

	RFLOAT frame_var = 0;

	#pragma omp parallel for reduction(+:frame_var) num_threads(num_threads)
	for (long int n = 0; n < pixCt; ++n)
	{
		if (!(*defectivePixels)[n])
		{
			RFLOAT d = (muGraphFrame[n] - frame_mean);
			frame_var += d * d;
		}
	}

	const RFLOAT frame_std = std::sqrt(frame_var / (n_valid - 1));

	for (size_t k = 0; k < need_stats.size(); k++)
	{
		const long int x = need_stats[k].first;
		const long int y = need_stats[k].second;
		const uint64_t hash = reproducibleHash(((uint64_t)frame << 42) ^ ((uint64_t)y << 21) ^ (uint64_t)x);
		// Gaussian from two hash-derived uniforms (Box-Muller)
		const double u1 = ((hash >> 11) + 1.0) / 9007199254740993.0;
		const double u2 = (reproducibleHash(hash) >> 11) / 9007199254740992.0;
		muGraphFrame(x,y) = frame_mean + frame_std * std::sqrt(-2 * std::log(u1)) * std::cos(2 * PI * u2);
	}
}

#endif
