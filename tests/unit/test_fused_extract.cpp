/*
 * tests/unit/test_fused_extract.cpp
 *
 * Fused extraction: the Fourier transform of a particle evaluated straight from
 * its micrograph (fused_extract.h) must equal, to NUFFT accuracy, what is
 * obtained by cutting the particle and running FourierTransform() +
 * CenterFFTbySign() on it, wherever those two routes mean the same thing.
 */

#include <catch2/catch.hpp>

#include "src/virtual_particles.h"
#include "src/fused_extract.h"
#include "src/fftw.h"
#include "src/image.h"
#include "src/error.h"

#include <cstdlib>
#include <string>
#include <vector>

#ifdef RELION_USE_FINUFFT

namespace {

class TempDir {
public:
	TempDir()
	{
		char tmpl[] = "/tmp/relion_fused_XXXXXX";
		const char* p = mkdtemp(tmpl);
		REQUIRE(p != NULL);
		path_ = p;
	}
	~TempDir()
	{
		const std::string cmd = "rm -rf '" + path_ + "'";
		if (system(cmd.c_str()) != 0) { /* best effort */ }
	}
	const std::string& path() const { return path_; }
private:
	std::string path_;
};

const int NX = 400, NY = 360;

/// White noise on top of smooth structure, so that every frequency is occupied
void writeMicrograph(const std::string& fn, double noise_amp)
{
	Image<RFLOAT> mic(NX, NY);
	unsigned s = 7;
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(mic())
	{
		s = s * 1103515245u + 12345u;
		const double noise = ((s >> 8) & 0xffff) / 65536.0 - 0.5;
		DIRECT_A2D_ELEM(mic(), i, j) = 3.0 * sin(0.07 * i) * cos(0.05 * j) + 2.0 * cos(0.11 * (i + j))
		                               + noise_amp * noise + 0.01 * i;
	}
	mic.write(fn, -1, false, WRITE_OVERWRITE, Float);
}

vparticles::Recipe recipe(int w, int h, int scale, bool normalise)
{
	vparticles::Recipe r;
	r.extract_size = w;
	r.extract_size_y = (h == w) ? -1 : h;
	r.scale = scale;
	r.normalise = normalise;
	r.bg_radius = (std::min(w, h) / 2 - 6) * (scale > 0 ? (double)scale / w : 1.);
	r.ramp = normalise;
	r.invert_contrast = normalise;
	r.float16 = false;
	r.angpix = 1.5 * (scale > 0 ? (double)w / scale : 1.);
	return r;
}

/// FourierTransform + CenterFFTbySign of a stored particle
MultidimArray<Complex> standardTransform(const std::string& vs, long index)
{
	MultidimArray<RFLOAT> img;
	vparticles::readParticle(vs, index, img);
	img.setXmippOrigin();
	FourierTransformer transformer;
	MultidimArray<Complex> F;
	transformer.FourierTransform(img, F);
	CenterFFTbySign(F);
	return F;
}

/// Relative RMS difference over the frequencies below `fraction` of Nyquist
double relativeDifference(const MultidimArray<Complex>& a, const MultidimArray<Complex>& b, int nx, int ny, double fraction)
{
	REQUIRE(YSIZE(a) == YSIZE(b));
	REQUIRE(XSIZE(a) == XSIZE(b));
	double num = 0., den = 0.;
	for (int i = 0; i < (int)YSIZE(a); i++)
	{
		const int ky = (i < ny / 2) ? i : i - ny;
		for (int j = 0; j < (int)XSIZE(a); j++)
		{
			const double fx = (double)j / nx, fy = (double)ky / ny;
			if (2. * std::sqrt(fx * fx + fy * fy) > fraction) continue;
			const Complex d = DIRECT_A2D_ELEM(a, i, j) - DIRECT_A2D_ELEM(b, i, j);
			num += d.abs() * d.abs();
			den += DIRECT_A2D_ELEM(a, i, j).abs() * DIRECT_A2D_ELEM(a, i, j).abs();
		}
	}
	return std::sqrt(num / den);
}

} // namespace

TEST_CASE("fused transform equals the transform of the cut particle (same pixel size)", "[fused]")
{
	TempDir tmp;
	const std::string mic = tmp.path() + "/mic.mrc", vs = tmp.path() + "/mic.vstack";
	writeMicrograph(mic, 1.0);

	std::vector<std::pair<long, long> > c;
	c.push_back(std::make_pair(200L, 180L));
	c.push_back(std::make_pair(30L, 25L));          // hangs off the corner
	c.push_back(std::make_pair(NX - 10L, 170L));    // hangs off the right edge

	const int sizes[][2] = {{64, 64}, {96, 48}};
	for (int normalise = 0; normalise < 2; normalise++)
		for (const auto& sz : sizes)
		{
			INFO("normalise " << normalise << " box " << sz[0] << " x " << sz[1]);
			vparticles::writeDescriptor(vs, mic, recipe(sz[0], sz[1], -1, normalise == 1), c);
			for (size_t i = 0; i < c.size(); i++)
			{
				MultidimArray<Complex> fused;
				vparticles::readParticleFourier(vs, i, 1.5, sz[0], sz[1], fused);
				CHECK(relativeDifference(standardTransform(vs, i), fused, sz[0], sz[1], 1.0) < 1e-6);
			}
		}
}

TEST_CASE("fused transform at a coarser pixel size equals Fourier cropping of the finer box", "[fused]")
{
	TempDir tmp;
	const std::string mic = tmp.path() + "/mic.mrc", vs = tmp.path() + "/mic.vstack";
	writeMicrograph(mic, 1.0);
	std::vector<std::pair<long, long> > c;
	c.push_back(std::make_pair(200L, 180L));
	c.push_back(std::make_pair(150L, 170L));

	// The stored particle is binned by 2 with an exact Fourier crop; no normalisation, since
	// that would be done on different pixels in the two routes
	vparticles::writeDescriptor(vs, mic, recipe(96, 64, 48, false), c);
	for (size_t i = 0; i < c.size(); i++)
	{
		MultidimArray<Complex> fused;
		vparticles::readParticleFourier(vs, i, 3.0, 48, 32, fused);
		CHECK(relativeDifference(standardTransform(vs, i), fused, 48, 32, 0.9) < 1e-5);
	}
}

TEST_CASE("fused transform can use a box larger than the stored particle", "[fused]")
{
	TempDir tmp;
	const std::string mic = tmp.path() + "/mic.mrc";
	writeMicrograph(mic, 1.0);
	std::vector<std::pair<long, long> > c;
	c.push_back(std::make_pair(200L, 180L));

	const std::string small = tmp.path() + "/small.vstack", big = tmp.path() + "/big.vstack";
	vparticles::writeDescriptor(small, mic, recipe(64, 64, -1, false), c);
	vparticles::writeDescriptor(big, mic, recipe(128, 96, -1, false), c);

	int nx, ny;
	vparticles::sameExtentBox(small, 1.5, nx, ny);
	CHECK(nx == 64);
	CHECK(ny == 64);

	MultidimArray<Complex> fused;
	vparticles::readParticleFourier(small, 0, 1.5, 128, 96, fused);
	CHECK(relativeDifference(standardTransform(big, 0), fused, 128, 96, 1.0) < 1e-6);
}

TEST_CASE("fused transform of a turned box agrees with the stored particle at low resolution", "[fused]")
{
	TempDir tmp;
	const std::string mic = tmp.path() + "/mic.mrc", vs = tmp.path() + "/mic.vstack";
	writeMicrograph(mic, 0.0);   // smooth: interpolation cannot matter at low frequency

	std::vector<std::pair<long, long> > c;
	std::vector<double> cx, cy, ang, psi;
	for (int i = 0; i < 3; i++)
	{
		c.push_back(std::make_pair(200L + 5 * i, 180L));
		cx.push_back(200.37 + 5 * i);
		cy.push_back(180.21);
		ang.push_back(-50. + 40. * i);
		psi.push_back(0.);
	}
	vparticles::Recipe r = recipe(96, 48, -1, true);
	r.rotated = true;
	r.interpolation = RESAMPLE_CUBIC;
	vparticles::writeDescriptor(vs, mic, r, c, psi, cx, cy, ang);

	for (size_t i = 0; i < c.size(); i++)
	{
		MultidimArray<Complex> fused;
		vparticles::readParticleFourier(vs, i, 1.5, 96, 48, fused);
		// The stored particle is cut by interpolation at the edge of its box and the fused one by
		// whole micrograph pixels, so only the low frequencies are compared
		CHECK(relativeDifference(standardTransform(vs, i), fused, 96, 48, 0.25) < 0.05);
	}
}

TEST_CASE("fused transform of a finer pixel size has nothing above the micrograph's Nyquist frequency", "[fused]")
{
	TempDir tmp;
	const std::string mic = tmp.path() + "/mic.mrc", vs = tmp.path() + "/mic.vstack";
	writeMicrograph(mic, 1.0);
	std::vector<std::pair<long, long> > c(1, std::make_pair(200L, 180L));
	vparticles::writeDescriptor(vs, mic, recipe(64, 64, -1, false), c);

	MultidimArray<Complex> fused;
	vparticles::readParticleFourier(vs, 0, 0.75, 128, 128, fused);
	double inside = 0., outside = 0.;
	for (int i = 0; i < 128; i++)
	{
		const int ky = (i < 64) ? i : i - 128;
		for (int j = 0; j <= 64; j++)
		{
			const double f = std::sqrt((double)j * j + (double)ky * ky) / 128.;
			const double a = DIRECT_A2D_ELEM(fused, i, j).abs();
			if (f > 0.25) outside += a; else inside += a;
		}
	}
	CHECK(inside > 0.);
	CHECK(outside == 0.);
}

#endif
