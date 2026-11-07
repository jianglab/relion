#ifndef JAZ_MOVIE_LOADER_H
#define JAZ_MOVIE_LOADER_H

#include <src/image.h>
#include <src/jaz/image/buffered_image.h>
#include <src/renderEER.h>
#include <cstdint>
#include <string>
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

	CompressedMRCReader reader;

	Image<float> mgStack;
	if (isCompressedMRC)
	{
		reader.read(movieFn, num_threads);
		// mgStack = bz2reader.Ihead() causes SEGV, because the array is not allocated.
		mgStack().copyShape(reader.Ihead());
	}
	else
		mgStack.read(movieFn, false, -1, false, true); // final true means 2D movies, not 3D map

	const std::string tag = "MovieLoader::readDense: ";

	if (mgStack.data.zdim > 1)
	{
		REPORT_ERROR_STR(tag << "the file " << movieFn << " looks like a 3D image and not a stack "
						 << "(the slices are along the Z dimension instead of N)");
	}

	const long int w0 = mgStack.data.xdim;
	const long int h0 = mgStack.data.ydim;
	const long int pixCt = w0 * h0;
	const int fc = numFrames;


	if (mgStack.data.ndim < frame0 + numFrames)
	{
		REPORT_ERROR_STR(
			tag << "insufficient number of frames in " << movieFn
			<< " (found: " << mgStack.data.ndim << ", expected: " << frame0 + numFrames << ")");
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


	BufferedImage<T> out(w0, h0, fc);
	ReadErrors errors;

	// Frames in parallel, each thread with its own file handle (the compressed-MRC reader is shared, so one at a time)
	#pragma omp parallel for num_threads(num_threads) if(fc > 1 && !isCompressedMRC)
	for (long f = 0; f < fc; f++)
	{
		if (errors.failed()) continue;
		try {
		Image<float> muGraphFrame_xmipp;

		if (isCompressedMRC)
			reader.readFrameInto(muGraphFrame_xmipp, frame0 + f);
		else
			muGraphFrame_xmipp.read(movieFn, true, frame0 + f, false, true);

		RawImage<T> muGraphFrame(muGraphFrame_xmipp);

		#pragma omp parallel for num_threads(num_threads)
		for (long int i = 0; i < pixCt; i++)
		{
			RFLOAT val = muGraphFrame[i];
			const RFLOAT gain = useGain? (*gainRef)[i] : 1;

			if (hot > 0.0 && val > hot) val = hot;

			 muGraphFrame[i] = -gain * val;
		}

		if (do_fixDefect)
		{
			fixDefects(muGraphFrame, defectivePixels, frame0 + f, num_threads, false);
		}

		out.getSliceRef(f).copyFrom(muGraphFrame);
		} catch (RelionError &e) { errors.keep(e); }
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


	BufferedImage<T> out(w0, h0, fc);
	ReadErrors errors;

	#pragma omp parallel for num_threads(num_threads) if(fc > 1)
	for (int f = 0; f < fc; f++)
	{
		if (errors.failed()) continue;
		try {
		MultidimArray<float> muGraphFrame_xmipp;
		// this takes 1-indexed frame numbers
		renderer.renderFrames(
					(frame0 + f) * eer_grouping + 1,
					(frame0 + f + 1) * eer_grouping,
					muGraphFrame_xmipp);

		RawImage<T> muGraphFrame(muGraphFrame_xmipp);

		#pragma omp parallel for num_threads(num_threads)
		for (long int i = 0; i < pixCt; i++)
		{
			const RFLOAT val = muGraphFrame[i];
			const RFLOAT gain = useGain? (*gainRef)[i] : 1.0;

			muGraphFrame[i] = -gain * val;
		}

		if (do_fixDefect)
		{
			fixDefects(muGraphFrame, defectivePixels, frame0 + f, num_threads, true);
		}

		out.getSliceRef(f).copyFrom(muGraphFrame);
		} catch (RelionError &e) { errors.keep(e); }
	}

	errors.rethrow();
	return out;
}

template <typename T>
void MovieLoader::fixDefects(
		RawImage<T>& muGraphFrame,
		const RawImage<bool>* defectivePixels,
		int frame,
		int num_threads, bool isEER)
{
	const long int w0 = muGraphFrame.xdim;
	const long int h0 = muGraphFrame.ydim;
	const long int pixCt = w0 * h0;

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

	const int min_num_ok = 6;
	const int d_max = isEER ? 4: 2;
	const int PBUF_SIZE = 100;

	#pragma omp parallel for num_threads(num_threads)
	for (long int y = 0; y < h0; y++)
	for (long int x = 0; x < w0; x++)
	{
		if (!(*defectivePixels)(x,y)) continue;

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
		const uint64_t hash = reproducibleHash(((uint64_t)frame << 42) ^ ((uint64_t)y << 21) ^ (uint64_t)x);

		if (n_ok > min_num_ok)
		{
			muGraphFrame(x,y) = pbuf[hash % n_ok];
		}
		else
		{
			// Gaussian from two hash-derived uniforms (Box-Muller)
			const double u1 = ((hash >> 11) + 1.0) / 9007199254740993.0;
			const double u2 = (reproducibleHash(hash) >> 11) / 9007199254740992.0;
			muGraphFrame(x,y) = frame_mean + frame_std * std::sqrt(-2 * std::log(u1)) * std::cos(2 * PI * u2);
		}
	}
}

#endif
