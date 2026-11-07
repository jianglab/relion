/***************************************************************************
 *
 * Author: "Jasenko Zivanov"
 * MRC Laboratory of Molecular Biology
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

#include <src/jaz/single_particle/stack_helper.h>
#include <src/jaz/single_particle/slice_helper.h>
#include <src/projector.h>
#include <src/jaz/single_particle/img_proc/filter_helper.h>
#include <src/jaz/single_particle/Fourier_helper.h>
#include <src/jaz/optimization/nelder_mead.h>
#include <src/jaz/gravis/t4Matrix.h>
#include <src/jaz/single_particle/image_log.h>
#include <src/fftw.h>
#include <src/micrograph_model.h>
#include <src/jaz/single_particle/resampling_helper.h>
#include <src/jaz/single_particle/parallel_ft.h>
#include <src/jaz/single_particle/new_ft.h>

#include <src/async_file_writer.h>
#include <map>
#include <omp.h>

using namespace gravis;


std::vector<MetaDataTable> StackHelper::splitByMicrographName(const MetaDataTable& mdt)
{
	std::vector<MetaDataTable> out(0);

	if (!mdt.containsLabel(EMDL_MICROGRAPH_NAME))
	{
		REPORT_ERROR("StackHelper::splitByMicrographName: "
					 + EMDL::label2Str(EMDL_MICROGRAPH_NAME)
					 + " missing from MetaDataTable.\n");
	}

	MetaDataTable md2(mdt);
	md2.newSort(EMDL_MICROGRAPH_NAME);

	const long lc = md2.numberOfObjects();
	std::string lastName = "", curName;
	long curInd = -1;

	for (int i = 0; i < lc; i++)
	{
		md2.getValue(EMDL_MICROGRAPH_NAME, curName, i);

		if (curName != lastName)
		{
			lastName = curName;
			curInd++;
			out.push_back(MetaDataTable());
		}

		out[curInd].addObject(md2.getObject(i));
	}

	for (int i = 0; i <= curInd; i++)
	{
		out[i].newSort(EMDL_IMAGE_NAME, false, false, true);
	}

	return out;
}

MetaDataTable StackHelper::merge(const std::vector<MetaDataTable> &mdts)
{
	MetaDataTable out;

	for (int i = 0; i < mdts.size(); i++)
	{
		out.append(mdts[i]);
	}

	return out;
}

std::vector<MetaDataTable> StackHelper::splitByStack(const MetaDataTable* mdt)
{
	std::vector<MetaDataTable> out(0);

	if (!mdt->containsLabel(EMDL_IMAGE_NAME))
	{
		REPORT_ERROR("StackHelper::splitByStack: "+EMDL::label2Str(EMDL_IMAGE_NAME)+" missing in meta_data_table.\n");
	}

	std::string testString;
	mdt->getValue(EMDL_IMAGE_NAME, testString, 0);

	if (testString.find("@") < 0)
	{
		REPORT_ERROR("StackHelper::splitByStack: "+EMDL::label2Str(EMDL_IMAGE_NAME)+" does not contain an '@'.\n");
	}

	MetaDataTable md2(*mdt);
	md2.newSort(EMDL_IMAGE_NAME, false, true);

	const long lc = md2.numberOfObjects();
	std::string lastName = "", curName, curFullName;
	long curInd = -1;

	for (int i = 0; i < lc; i++)
	{
		md2.getValue(EMDL_IMAGE_NAME, curFullName, i);

		curName = curFullName.substr(curFullName.find("@")+1);

		if (curName != lastName)
		{
			lastName = curName;
			curInd++;
			out.push_back(MetaDataTable());
		}

		out[curInd].addObject(md2.getObject(i));
	}

	for (int i = 0; i <= curInd; i++)
	{
		out[i].newSort(EMDL_IMAGE_NAME, false, false, true);
	}

	return out;
}

std::vector<Image<RFLOAT> > StackHelper::loadStack(const MetaDataTable* mdt, std::string path, int threads)
{
	const long ic = mdt->numberOfObjects();
	std::vector<Image<RFLOAT>> out(ic);

	// A stack file of which at least half the images are wanted is read in one pass, which on network
	// file systems is several times faster than reading one image at a time (the pixels are the same).
	// Not with a path override, and not when the stack would take more than a quarter of the free memory.
	std::map<std::string, Image<RFLOAT>> whole;

	if (path == "")
	{
		std::map<std::string, long> wanted;

		for (long i = 0; i < ic; i++)
		{
			long n;
			std::string fn;
			FileName(mdt->getString(EMDL_IMAGE_NAME, i)).decompose(n, fn);
			if (n > 0) wanted[fn]++;
		}

		for (const auto& w : wanted)
		{
			Image<RFLOAT> header;
			header.read(w.first, false, -1, false, true);

			const long n = NSIZE(header.data);
			const double bytes = (double) n * XSIZE(header.data) * YSIZE(header.data) * sizeof(RFLOAT);

			const long long avail = asyncwrite::memoryAvailableBytes();  // -1: unknown

			if (ZSIZE(header.data) == 1 && 2 * w.second >= n && (avail < 0 || bytes < 0.25 * avail))
			{
				whole[w.first].read(w.first, true, -1, false, true);
			}
		}
	}

	#pragma omp parallel for num_threads(threads)
	for (long i = 0; i < ic; i++)
	{
		std::string sliceName;
		mdt->getValue(EMDL_IMAGE_NAME, sliceName, i);

		long n;
		std::string fn;
		FileName(sliceName).decompose(n, fn);
		const auto stack = whole.find(fn);

		if (stack != whole.end() && n > 0 && n <= NSIZE(stack->second.data))
		{
			stack->second.data.getImage(n - 1, out[i].data);
		}
		else
		{
			out[i].read(sliceName, true, -1, false, true);
		}
	}

	return out;
}

std::vector<Image<Complex> > StackHelper::loadStackFS(
		const MetaDataTable& mdt, std::string path,
		int threads, bool centerParticle, ObservationModel* obs)
{
	std::vector<Image<RFLOAT>> images = loadStack(&mdt, path, threads);

	return transformStackFS(images, mdt, threads, centerParticle, obs);
}

std::vector<Image<Complex> > StackHelper::transformStackFS(
		std::vector<Image<RFLOAT>>& images, const MetaDataTable& mdt,
		int threads, bool centerParticle, ObservationModel* obs)
{
	std::vector<Image<Complex> > out(mdt.numberOfObjects());

	if (centerParticle && obs == 0)
	{
		REPORT_ERROR("StackHelper::loadStackFS: centering particles requires an observation model.");
	}

	const long ic = mdt.numberOfObjects();

	if (ic == 0) return out;

	const int s = images[0].data.xdim;

	NewFFTPlan<RFLOAT>::type plan(s,s,1);

	#pragma omp parallel for num_threads(threads)
	for (long i = 0; i < ic; i++)
	{
		Image<RFLOAT>& in = images[i];

		NewFFT::FourierTransform(in(), out[i](), plan);

		if (centerParticle)
		{
			int optGroup = obs->getOpticsGroup(mdt, i);
			double angpix = obs->getPixelSize(optGroup);

			const int s = in.data.ydim;

			double xoff, yoff;

			mdt.getValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, xoff, i);
			mdt.getValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, yoff, i);

			xoff /= angpix;
			yoff /= angpix;

			shiftImageInFourierTransform(out[i](), out[i](), s, xoff - s/2, yoff - s/2);
		}

		in.data.clear();
	}

	return out;
}

void StackHelper::saveStack(std::vector<Image<RFLOAT> > &stack, std::string fn)
{
	const int w = stack[0].data.xdim;
	const int h = stack[0].data.ydim;
	const int c = stack.size();

	Image<RFLOAT> img(w,h,1,c);

	for (int i = 0; i < c; i++)
	{
		SliceHelper::insertStackSlice(stack[i], img, i);
	}

	img.write(fn);
}

std::vector<std::vector<Image<RFLOAT> > > StackHelper::loadMovieStack(const MetaDataTable* mdt, std::string moviePath)
{
	std::vector<std::vector<Image<RFLOAT> > > out(mdt->numberOfObjects());
	const long pc = mdt->numberOfObjects();

	std::string name, fullName, movieName;
	mdt->getValue(EMDL_IMAGE_NAME, fullName, 0);
	mdt->getValue(EMDL_MICROGRAPH_NAME, movieName, 0);
	name = fullName.substr(fullName.find("@")+1);

	std::string finName;

	if (moviePath == "")
	{
		finName = name;
	}
	else
	{
		finName = moviePath + "/" + movieName.substr(movieName.find_last_of("/")+1);
	}

	std::cout << "loading real: " << finName << "\n";

	Image<RFLOAT> in;
	in.read(finName);

	std::cout << "size = " << in.data.xdim << "x" << in.data.ydim << "x" << in.data.zdim << "x" << in.data.ndim << "\n";
	std::cout << "pc = " << pc << "\n";

	const int fc = in.data.ndim / pc;

	const int w = in.data.xdim;
	const int h = in.data.ydim;

	for (long p = 0; p < pc; p++)
	{
		out[p] = std::vector<Image<RFLOAT> >(fc);

		for (long f = 0; f < fc; f++)
		{
			out[p][f] = Image<RFLOAT>(w,h);
			SliceHelper::extractStackSlice(in, out[p][f], f*pc + p);
		}
	}

	return out;
}

std::vector<Image<Complex> > StackHelper::FourierTransform(std::vector<Image<RFLOAT> >& stack)
{
	std::vector<Image<Complex> > out(stack.size());
	const long ic = stack.size();

	for (long i = 0; i < ic; i++)
	{
		FourierTransformer ft;
		ft.FourierTransform(stack[i].data, out[i].data);
	}

	return out;
}

std::vector<Image<RFLOAT> > StackHelper::inverseFourierTransform(std::vector<Image<Complex> >& stack)
{
	std::vector<Image<RFLOAT> > out(stack.size());
	const long ic = stack.size();

	const int h = stack[0].data.ydim;
	const int ww = stack[0].data.xdim;
	const int w = 2*(ww - 1);

	for (long i = 0; i < ic; i++)
	{
		out[i] = Image<RFLOAT>(w,h);

		FourierTransformer ft;
		ft.inverseFourierTransform(stack[i].data, out[i].data);
	}

	return out;
}

Image<RFLOAT> StackHelper::toSingleImage(const std::vector<Image<RFLOAT>> stack)
{
	const int s = stack.size();

	if (s < 1) return Image<RFLOAT>(0,0,0);

	const int w = stack[0].data.xdim;
	const int h = stack[0].data.ydim;

	Image<RFLOAT> out(w,h,1,s);

	for (int n = 0; n < s; n++)
	{
		for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			DIRECT_NZYX_ELEM(out(),n,0,y,x) = stack[n](y,x);
		}
	}

	return out;
}

void StackHelper::varianceNormalize(std::vector<Image<Complex>>& movie, bool circleCropped)
{
	const int fc = movie.size();
	const int w = movie[0].data.xdim;
	const int h = movie[0].data.ydim;
	const int wt = 2*(w-1);

	double var = 0.0;
	double cnt = 0.0;

	const double rr = (w-2)*(w-2);

	for (int f = 0; f < fc; f++)
	{
		for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			if (x == 0 && y == 0) continue;

			if (circleCropped)
			{
				const double yy = y < w? y : y - h;
				const double xx = x;

				if (xx*xx + yy*yy > rr) continue;
			}

			double scale = x > 0? 2.0 : 1.0;

			var += scale * movie[f](y,x).norm();
			cnt += scale;
		}
	}

	const double scale = sqrt( wt * h * var / (cnt * fc) );

	for (int f = 0; f < fc; f++)
	{
		for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			movie[f](y,x) /= scale;
		}
	}
}


RFLOAT StackHelper::computePower(const RawImage<Complex>& movie, bool circleCropped)
{
	const int w = movie.xdim;
	const int h = movie.ydim;
	const int fc = movie.zdim;

	double var = 0.0;
	double cnt = 0.0;

	const double rr = (w-2)*(w-2);

	for (int f = 0; f < fc; f++)
	{
		for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			if (x == 0 && y == 0) continue;

			if (circleCropped)
			{
				const double yy = y < w? y : y - h;
				const double xx = x;

				if (xx*xx + yy*yy > rr) continue;
			}

			double scale = x > 0? 2.0 : 1.0;

			var += scale * movie(x,y,f).norm();
			cnt += scale;
		}
	}

	return var / cnt;
}

RFLOAT StackHelper::computePower(std::vector<Image<Complex>>& movie, bool circleCropped)
{
	const int fc = movie.size();
	const int w = movie[0].data.xdim;
	const int h = movie[0].data.ydim;

	double var = 0.0;
	double cnt = 0.0;

	const double rr = (w-2)*(w-2);

	for (int f = 0; f < fc; f++)
	{
		for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			if (x == 0 && y == 0) continue;

			if (circleCropped)
			{
				const double yy = y < w? y : y - h;
				const double xx = x;

				if (xx*xx + yy*yy > rr) continue;
			}

			double scale = x > 0? 2.0 : 1.0;

			var += scale * movie[f](y,x).norm();
			cnt += scale;
		}
	}

	return var / cnt;
}

std::vector<double> StackHelper::powerSpectrum(const std::vector<std::vector<Image<Complex>>> &stack, int threads)
{
	const int ic = stack.size();
	const int fc = stack[0].size();
	const int w = stack[0][0].data.xdim;
	const int h = stack[0][0].data.ydim;

	// The radius of each pixel, once instead of in every image
	std::vector<int> radius(w*h);
	std::vector<double> out(w, 0.0), wgh(w, 0.0);

	for (int y = 0; y < h; y++)
	for (int x = 0; x < w; x++)
	{
		const double yy = y < w? y : y - h;
		const double xx = x;

		const int r = (int) sqrt(xx*xx + yy*yy);

		radius[y*w + x] = r;

		if (r < w) wgh[r] += (double) ic * fc;
	}

	// Summed per image, then over the images in order: the same at any thread count
	std::vector<std::vector<double>> sums(ic, std::vector<double>(w, 0.0));

	#pragma omp parallel for num_threads(threads)
	for (int i = 0; i < ic; i++)
	for (int f = 0; f < fc; f++)
	{
		for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			const int r = radius[y*w + x];

			if (r >= w) continue;

			sums[i][r] += DIRECT_A2D_ELEM(stack[i][f].data, y, x).norm();
		}
	}

	for (int i = 0; i < ic; i++)
	for (int r = 0; r < w; r++)
	{
		out[r] += sums[i][r];
	}

	for (int x = 0; x < w; x++)
	{
		if (wgh[x] > 0.0)
		{
			out[x] /= wgh[x];
		}
	}

	return out;
}

std::vector<double> StackHelper::varSpectrum(const std::vector<std::vector<Image<Complex>>> &stack)
{
	const int ic = stack.size();
	const int fc = stack[0].size();
	const int w = stack[0][0].data.xdim;
	const int h = stack[0][0].data.ydim;

	std::vector<double> out(w, 0.0), wgh(w, 0.0);
	std::vector<Complex> mean(w, 0.0);

	for (int i = 0; i < ic; i++)
	for (int f = 0; f < fc; f++)
	{
		for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			const Complex z = DIRECT_A2D_ELEM(stack[i][f].data, y, x);

			const double yy = y < w? y : y - h;
			const double xx = x;

			const int r = (int) sqrt(xx*xx + yy*yy);

			if (r >= w) continue;

			mean[r] += z;
			wgh[r] += 1.0;
		}
	}

	for (int x = 0; x < w; x++)
	{
		if (wgh[x] > 0.0)
		{
			mean[x] /= wgh[x];
		}
	}

	for (int i = 0; i < ic; i++)
	for (int f = 0; f < fc; f++)
	{
		for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			const Complex z = DIRECT_A2D_ELEM(stack[i][f].data, y, x);

			const double yy = y < w? y : y - h;
			const double xx = x;

			const int r = (int) sqrt(xx*xx + yy*yy);

			if (r >= w) continue;

			out[r] += (z - mean[r]).norm();
		}
	}

	for (int x = 0; x < w; x++)
	{
		if (wgh[x] > 1.0)
		{
			out[x] /= (wgh[x] - 1.0);
		}
	}

	return out;
}

std::vector<double> StackHelper::powerSpectrum(
		const std::vector<std::vector<Image<Complex>>>& obs,
		const std::vector<Image<Complex> >& signal)
{
	const int ic = obs.size();
	const int fc = obs[0].size();
	const int w = obs[0][0].data.xdim;
	const int h = obs[0][0].data.ydim;

	std::vector<double> out(w, 0.0), wgh(w, 0.0);

	for (int i = 0; i < ic; i++)
	for (int f = 0; f < fc; f++)
	{
		for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			const Complex z = DIRECT_A2D_ELEM(obs[i][f].data, y, x) - DIRECT_A2D_ELEM(signal[i].data, y, x);

			const double yy = y < w? y : y - h;
			const double xx = x;

			const int r = (int) sqrt(xx*xx + yy*yy);

			if (r >= w) continue;

			out[r] += z.norm();
			wgh[r] += 1.0;
		}
	}

	for (int x = 0; x < w; x++)
	{
		if (wgh[x] > 0.0)
		{
			out[x] /= wgh[x];
		}
	}

	return out;
}
