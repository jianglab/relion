/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/fftw_rect.h"
#include "src/fftw.h"

void windowFourierTransformRect(const MultidimArray<Complex> &in, MultidimArray<Complex> &out,
                                long int new_nx, long int new_ny, long int new_nz)
{
	const int dim = in.getDim();
	if (dim < 2 || dim > 3)
		REPORT_ERROR("windowFourierTransformRect ERROR: dimension should be 2 or 3!");
	if (dim == 2) new_nz = 1;

	const long int in_nx = 2 * (XSIZE(in) - 1), in_ny = YSIZE(in), in_nz = ZSIZE(in);
	const long int new_hx = new_nx / 2 + 1;

	if (new_nx == in_nx && new_ny == in_ny && new_nz == in_nz)
	{
		out = in;
		return;
	}

	MultidimArray<Complex> result;
	if (dim == 2)
		result.initZeros(new_ny, new_hx);
	else
		result.initZeros(new_nz, new_ny, new_hx);

	const bool grows = new_hx > XSIZE(in) || new_ny > in_ny || new_nz > in_nz;
	const double jmax = std::max<long int>(XSIZE(in) - 1, 1);
	const double imax = std::max<long int>(in_ny / 2, 1);
	const double kmax = std::max<long int>(in_nz / 2, 1);

	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM_RECT(result)
	{
		// the component must exist in the input: |kp|, |ip| within the input half-ranges
		if (jp >= XSIZE(in)) continue;
		if (ip > in_ny / 2 || ip < -((in_ny - 1) / 2)) continue;
		if (dim == 3 && (kp > in_nz / 2 || kp < -((in_nz - 1) / 2))) continue;
		if (grows)
		{
			const double a = jp / jmax, b = ip / imax, c = (dim == 3) ? kp / kmax : 0.;
			if (a*a + b*b + c*c > 1.0) continue;
		}
		FFTW_ELEM(result, kp, ip, jp) = FFTW_ELEM(in, kp, ip, jp);
	}
	out.moveFrom(result);
}

void resizeMapRect(MultidimArray<RFLOAT> &img, long int new_nx, long int new_ny)
{
	if (img.getDim() != 2)
		REPORT_ERROR("resizeMapRect ERROR: only 2D images are supported.");
	FourierTransformer transformer;
	MultidimArray<Complex> FT, FT2;
	transformer.FourierTransform(img, FT, false);
	windowFourierTransformRect(FT, FT2, new_nx, new_ny);
	img.resize(new_ny, new_nx);
	transformer.inverseFourierTransform(FT2, img);
}

void shiftImageInFourierTransformRect(const MultidimArray<Complex> &in, MultidimArray<Complex> &out,
                                      RFLOAT orix, RFLOAT oriy, RFLOAT oriz,
                                      RFLOAT xshift, RFLOAT yshift, RFLOAT zshift)
{
	const int dim = in.getDim();
	if (dim < 2 || dim > 3)
		REPORT_ERROR("shiftImageInFourierTransformRect ERROR: dimension should be 2 or 3!");

	MultidimArray<Complex> result;
	result.resize(in);
	xshift /= -orix;
	yshift /= -oriy;
	if (dim == 3) zshift /= -oriz;

	RFLOAT a, b, c, d, ac, bd, ab_cd;
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM_RECT(in)
	{
		const RFLOAT dotp = 2 * PI * (jp * xshift + ip * yshift + ((dim == 3) ? kp * zshift : 0.));
#ifdef RELION_SINGLE_PRECISION
		SINCOSF(dotp, &b, &a);
#else
		SINCOS(dotp, &b, &a);
#endif
		c = DIRECT_A3D_ELEM(in, k, i, j).real;
		d = DIRECT_A3D_ELEM(in, k, i, j).imag;
		ac = a * c;
		bd = b * d;
		ab_cd = (a + b) * (c + d);
		DIRECT_A3D_ELEM(result, k, i, j) = Complex(ac - bd, ab_cd - ac - bd);
	}
	out.moveFrom(result);
}

void getSpectrumRect(MultidimArray<RFLOAT> &Min, MultidimArray<RFLOAT> &spectrum, long int L, int spectrum_type)
{
	MultidimArray<Complex> F;
	FourierTransformer transformer;
	MultidimArray<RFLOAT> count(L / 2 + 1);
	spectrum.initZeros(L / 2 + 1);
	count.initZeros();
	transformer.FourierTransform(Min, F, false);

	const long int nx = XSIZE(Min), ny = YSIZE(Min), nz = ZSIZE(Min);
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM_RECT(F)
	{
		long int idx = shellIndexRect(kp, ip, jp, nx, ny, nz, L);
		if (idx > L / 2) continue; // beyond the nominal Nyquist (corners)
		if (spectrum_type == AMPLITUDE_SPECTRUM)
			spectrum(idx) += abs(dAkij(F, k, i, j));
		else
			spectrum(idx) += norm(dAkij(F, k, i, j));
		count(idx) += 1.;
	}
	for (long int i = 0; i < L / 2 + 1; i++)
		if (count(i) > 0.)
			spectrum(i) /= count(i);
}

void shiftImageInFourierTransformWithTabSincosRect(const MultidimArray<Complex> &in, MultidimArray<Complex> &out,
                                                   RFLOAT orix, RFLOAT oriy, long int new_nx, long int new_ny,
                                                   TabSine &tabsin, TabCosine &tabcos,
                                                   RFLOAT xshift, RFLOAT yshift)
{
	if (&in == &out)
		REPORT_ERROR("shiftImageInFourierTransformWithTabSincosRect: input and output must be different arrays.");
	if (in.getDim() != 2)
		REPORT_ERROR("shiftImageInFourierTransformWithTabSincosRect: only 2D images are supported.");
	if (YSIZE(in) != new_ny || XSIZE(in) != new_nx / 2 + 1)
		REPORT_ERROR("shiftImageInFourierTransformWithTabSincosRect: the output size must equal the input size.");

	out.clear();
	out.initZeros(new_ny, new_nx / 2 + 1);

	const RFLOAT twopi = 2. * PI;
	xshift /= -orix;
	yshift /= -oriy;
	if (ABS(xshift) < XMIPP_EQUAL_ACCURACY && ABS(yshift) < XMIPP_EQUAL_ACCURACY)
	{
		out = in;
		return;
	}
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM_RECT(out)
	{
		RFLOAT dotp = twopi * (jp * xshift + ip * yshift);
		RFLOAT a = tabcos(dotp);
		RFLOAT b = tabsin(dotp);
		RFLOAT c = DIRECT_A2D_ELEM(in, i, j).real;
		RFLOAT d = DIRECT_A2D_ELEM(in, i, j).imag;
		RFLOAT ac = a * c, bd = b * d, ab_cd = (a + b) * (c + d);
		DIRECT_A2D_ELEM(out, i, j) = Complex(ac - bd, ab_cd - ac - bd);
	}
}
