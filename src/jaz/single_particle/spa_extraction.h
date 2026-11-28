#ifndef SPA_EXTRACTION_H
#define SPA_EXTRACTION_H

#include <src/metadata_table.h>
#include <src/image.h>
#include <src/jaz/image/raw_image.h>
#include <src/jaz/gravis/t2Vector.h>
#include <src/jaz/single_particle/parallel_ft.h>
#include <src/jaz/single_particle/img_proc/filter_helper.h>
#include <vector>


class SpaExtraction
{
	public:
		
		// @TODO: replace output type by vector<BufferedImage>
		template <typename T>
		static std::vector<std::vector<Image<Complex>>> extractMovieStackFS(
				const MetaDataTable& mdt, 
				const RawImage<T>& movie,
				int boxSize,
				double outPs, double coordsPs, double moviePs, double dataPs, // @TODO: replace by data type
				const std::vector<std::vector<gravis::d2Vector>>* offsets_in,
				std::vector<std::vector<gravis::d2Vector>>* offsets_out,
				int num_threads);

		// The same frames, handed to consume(p, f, frame) one at a time instead of all kept in memory:
		// particles over threads, the frames of each particle in order. (*offsets_out)[p][f] is set first.
		template <typename T, typename Consume>
		static void consumeMovieStackFS(
				const MetaDataTable& mdt,
				const RawImage<T>& movie,
				int boxSize,
				double outPs, double coordsPs, double moviePs, double dataPs,
				const std::vector<std::vector<gravis::d2Vector>>* offsets_in,
				std::vector<std::vector<gravis::d2Vector>>* offsets_out,
				int num_threads,
				Consume consume);

	private:

		template <typename T>
		static void extractFrame(
				const MetaDataTable& mdt, const RawImage<T>& movie, long p, long f, int boxSize,
				double outPs, double coordsPs, double moviePs, double dataPs,
				const std::vector<std::vector<gravis::d2Vector>>* offsets_in,
				std::vector<std::vector<gravis::d2Vector>>* offsets_out,
				ParFourierTransformer& ft, Image<RFLOAT>& aux0, Image<Complex>& out);
};

template <typename T>
std::vector<std::vector<Image<Complex>>> SpaExtraction::extractMovieStackFS(
		const MetaDataTable& mdt, 
		const RawImage<T>& movie,
		int boxSize,
		double outPs, double coordsPs, double moviePs, double dataPs,
		const std::vector<std::vector<gravis::d2Vector>>* offsets_in,
		std::vector<std::vector<gravis::d2Vector>>* offsets_out,
		int num_threads)
{
	std::vector<std::vector<Image<Complex>>> out(mdt.numberOfObjects());
	const long pc = mdt.numberOfObjects();

	const int fc = movie.zdim;

	if (dataPs < 0)
	{
		dataPs = outPs;
	}

	for (long p = 0; p < pc; p++)
	{
		out[p] = std::vector<Image<Complex>>(fc);
	}

	const int sqMg = 2*(int)(0.5 * boxSize * outPs / moviePs + 0.5);

	std::vector<ParFourierTransformer> fts(num_threads);

	std::vector<Image<RFLOAT>> aux0(num_threads);

	for (int th = 0; th < num_threads; th++)
	{
		aux0[th] = Image<RFLOAT>(sqMg, sqMg);
	}

	// Frames and particles together, so that a single frame (Polish's frame combination) also uses every thread
	#pragma omp parallel for collapse(2) num_threads(num_threads)
	for (long int f = 0; f < fc; f++)
	for (long p = 0; p < pc; p++)
	{
		const int t = omp_get_thread_num();

		extractFrame(mdt, movie, p, f, boxSize, outPs, coordsPs, moviePs, dataPs,
			offsets_in, offsets_out, fts[t], aux0[t], out[p][f]);
	}

	return out;
}


template <typename T>
void SpaExtraction::extractFrame(
		const MetaDataTable& mdt, const RawImage<T>& movie, long p, long f, int boxSize,
		double outPs, double coordsPs, double moviePs, double dataPs,
		const std::vector<std::vector<gravis::d2Vector>>* offsets_in,
		std::vector<std::vector<gravis::d2Vector>>* offsets_out,
		ParFourierTransformer& ft, Image<RFLOAT>& aux0, Image<Complex>& out)
{
	const int w0 = movie.xdim;
	const int h0 = movie.ydim;

	const int sqMg = 2*(int)(0.5 * boxSize * outPs / moviePs + 0.5);

	double xpC, ypC;

	mdt.getValue(EMDL_IMAGE_COORD_X, xpC, p);
	mdt.getValue(EMDL_IMAGE_COORD_Y, ypC, p);

	const double xpO = (int)(coordsPs * xpC / dataPs);
	const double ypO = (int)(coordsPs * ypC / dataPs);

	int x0 = (int)round(xpO * dataPs / moviePs) - sqMg / 2;
	int y0 = (int)round(ypO * dataPs / moviePs) - sqMg / 2;

	if (offsets_in != 0 && offsets_out != 0)
	{
		double dxM = (*offsets_in)[p][f].x * outPs / moviePs;
		double dyM = (*offsets_in)[p][f].y * outPs / moviePs;

		int dxI = (int)round(dxM);
		int dyI = (int)round(dyM);

		x0 += dxI;
		y0 += dyI;

		double dxR = (dxM - dxI) * moviePs / outPs;
		double dyR = (dyM - dyI) * moviePs / outPs;

		(*offsets_out)[p][f] = gravis::d2Vector(dxR, dyR);
	}

	if (x0 >= 0 && x0 + sqMg <= w0 && y0 >= 0 && y0 + sqMg <= h0)
	{
		for (long int y = 0; y < sqMg; y++)
		{
			const T* __restrict src = &movie(x0, y0 + y, f);
			RFLOAT* __restrict dst = &DIRECT_NZYX_ELEM(aux0.data, 0, 0, y, 0);
			for (long int x = 0; x < sqMg; x++)
			{
				dst[x] = (RFLOAT)src[x];
			}
		}
	}
	else
	{
		for (long int y = 0; y < sqMg; y++)
		for (long int x = 0; x < sqMg; x++)
		{
			int xx = x0 + x;
			int yy = y0 + y;

			if (xx < 0) xx = 0;
			else if (xx >= w0) xx = w0 - 1;

			if (yy < 0) yy = 0;
			else if (yy >= h0) yy = h0 - 1;

			DIRECT_NZYX_ELEM(aux0.data, 0, 0, y, x) = movie(xx,yy,f);
		}
	}

	if (outPs == moviePs)
	{
		ft.FourierTransform(aux0(), out());
	}
	else
	{
		ft.setReal(aux0());
		ft.Transform(FFTW_FORWARD);
		const int w1 = XSIZE(ft.fFourier);
		const int h1 = YSIZE(ft.fFourier);
		const int w = boxSize / 2 + 1;
		const int h = boxSize;
		const int cx = std::min(w1, w);
		const int py = std::min(h1 / 2, h / 2);
		const int ny = std::min(h1 - h1 / 2, h / 2);
		if (XSIZE(out()) != w || YSIZE(out()) != h)
			out().initZeros(h, w);
		else if (w > w1 || py + ny < h)
			out().initZeros();
		for (int y = 0; y < py; y++)
		{
			const Complex* __restrict src = &DIRECT_A2D_ELEM(ft.fFourier, y, 0);
			Complex* __restrict dst = &DIRECT_A2D_ELEM(out(), y, 0);
			for (int x = 0; x < cx; x++) dst[x] = src[x];
		}
		for (int k = 0; k < ny; k++)
		{
			const Complex* __restrict src = &DIRECT_A2D_ELEM(ft.fFourier, h1 - ny + k, 0);
			Complex* __restrict dst = &DIRECT_A2D_ELEM(out(), h - ny + k, 0);
			for (int x = 0; x < cx; x++) dst[x] = src[x];
		}
	}

	out(0,0) = Complex(0.0,0.0);
}

template <typename T, typename Consume>
void SpaExtraction::consumeMovieStackFS(
		const MetaDataTable& mdt,
		const RawImage<T>& movie,
		int boxSize,
		double outPs, double coordsPs, double moviePs, double dataPs,
		const std::vector<std::vector<gravis::d2Vector>>* offsets_in,
		std::vector<std::vector<gravis::d2Vector>>* offsets_out,
		int num_threads,
		Consume consume)
{
	const long pc = mdt.numberOfObjects();
	const int fc = movie.zdim;

	if (dataPs < 0)
	{
		dataPs = outPs;
	}

	const int sqMg = 2*(int)(0.5 * boxSize * outPs / moviePs + 0.5);

	std::vector<ParFourierTransformer> fts(num_threads);

	std::vector<Image<RFLOAT>> aux0(num_threads);
	std::vector<Image<Complex>> frame(num_threads);

	for (int th = 0; th < num_threads; th++)
	{
		aux0[th] = Image<RFLOAT>(sqMg, sqMg);
	}

	#pragma omp parallel for num_threads(num_threads)
	for (long p = 0; p < pc; p++)
	{
		const int t = omp_get_thread_num();

		for (long int f = 0; f < fc; f++)
		{
			extractFrame(mdt, movie, p, f, boxSize, outPs, coordsPs, moviePs, dataPs,
				offsets_in, offsets_out, fts[t], aux0[t], frame[t]);

			consume(p, f, frame[t]);
		}
	}
}


#endif
