/*
 * tests/unit/test_mem_tiff.cpp
 *
 * Tests in-memory O(1)-IFD parallel TIFF decoding (memtiff::TiffMovieBuffer) and whole-stack
 * Image<float>::read() TIFF integration across uint8/uint16/float32 and uncompressed/deflate strips.
 */

#include <catch2/catch.hpp>

#include <tiffio.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "src/image.h"
#include "src/mem_tiff.h"

namespace
{

struct TempDir
{
	std::string path;
	TempDir()
	{
		const char *base = getenv("TMPDIR");
		std::string tmpl = std::string(base ? base : "/tmp") + "/relion_memtiff_XXXXXX";
		std::vector<char> buf(tmpl.begin(), tmpl.end());
		buf.push_back('\0');
		REQUIRE(mkdtemp(buf.data()) != NULL);
		path = buf.data();
	}
	~TempDir()
	{
		std::string cmd = "rm -rf '" + path + "'";
		if (system(cmd.c_str()) != 0) {}
	}
};

template <typename SampleT>
void writeSynthTiff(const std::string &fn, int W, int H, int nFrames,
                    uint16_t bitsPerSample, uint16_t sampleFormat,
                    uint16_t compression, uint32_t rowsPerStrip,
                    std::vector<std::vector<float> > &ref_yflipped)
{
	TIFF *tif = TIFFOpen(fn.c_str(), "w");
	REQUIRE(tif != NULL);
	ref_yflipped.resize((size_t)nFrames);

	for (int f = 0; f < nFrames; f++)
	{
		TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, (uint32_t)W);
		TIFFSetField(tif, TIFFTAG_IMAGELENGTH, (uint32_t)H);
		TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, (uint16_t)1);
		TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bitsPerSample);
		TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, sampleFormat);
		TIFFSetField(tif, TIFFTAG_ORIENTATION, (uint16_t)ORIENTATION_TOPLEFT);
		TIFFSetField(tif, TIFFTAG_PLANARCONFIG, (uint16_t)PLANARCONFIG_CONTIG);
		TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, (uint16_t)PHOTOMETRIC_MINISBLACK);
		TIFFSetField(tif, TIFFTAG_COMPRESSION, compression);
		TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, rowsPerStrip);

		ref_yflipped[(size_t)f].resize((size_t)W * (size_t)H);
		std::vector<SampleT> row((size_t)W);
		for (int y = 0; y < H; y++)
		{
			const int flipped_y = H - 1 - y;
			for (int x = 0; x < W; x++)
			{
				SampleT v = (SampleT)0;
				const int code = (f * 37 + y * 13 + x * 5) % 211;
				if (sampleFormat == SAMPLEFORMAT_IEEEFP)
					v = (SampleT)((float)code * 0.25f - 17.5f);
				else
					v = (SampleT)code;
				row[(size_t)x] = v;
				ref_yflipped[(size_t)f][(size_t)flipped_y * (size_t)W + (size_t)x] = (float)v;
			}
			REQUIRE(TIFFWriteScanline(tif, row.data(), (uint32_t)y, 0) == 1);
		}
		REQUIRE(TIFFWriteDirectory(tif) == 1);
	}
	TIFFClose(tif);
}

} // namespace

TEST_CASE("memtiff::isTiffFilename detects tif/tiff extensions cleanly", "[mem_tiff]")
{
	CHECK(memtiff::TiffMovieBuffer::isTiffFilename("movie.tif"));
	CHECK(memtiff::TiffMovieBuffer::isTiffFilename("path/to/003@movie.TIFF:tiff"));
	CHECK_FALSE(memtiff::TiffMovieBuffer::isTiffFilename("stack.mrcs"));
	CHECK_FALSE(memtiff::TiffMovieBuffer::isTiffFilename("mic.mrc"));
}

TEST_CASE("memtiff::TiffMovieBuffer decodes multi-frame TIFFs accurately across formats and OMP threads", "[mem_tiff]")
{
	TempDir dir;
	const int W = 64, H = 48, nFrames = 6;
	struct Spec {
		std::string name;
		uint16_t bps, fmt, comp;
		uint32_t rps;
	};
	const std::vector<Spec> specs = {
		{"u8_none.tif",   8,  SAMPLEFORMAT_UINT,   COMPRESSION_NONE,          12},
		{"u16_defl.tif",  16, SAMPLEFORMAT_UINT,   COMPRESSION_ADOBE_DEFLATE, 16},
		{"f32_multi.tif", 32, SAMPLEFORMAT_IEEEFP, COMPRESSION_NONE,           7},
	};

	for (const auto &sp : specs)
	{
		INFO("format=" << sp.name);
		const std::string fn = dir.path + "/" + sp.name;
		std::vector<std::vector<float> > ref;
		if (sp.bps == 8)
			writeSynthTiff<unsigned char>(fn, W, H, nFrames, sp.bps, sp.fmt, sp.comp, sp.rps, ref);
		else if (sp.bps == 16)
			writeSynthTiff<unsigned short>(fn, W, H, nFrames, sp.bps, sp.fmt, sp.comp, sp.rps, ref);
		else
			writeSynthTiff<float>(fn, W, H, nFrames, sp.bps, sp.fmt, sp.comp, sp.rps, ref);

		memtiff::TiffMovieBuffer buf;
		REQUIRE(buf.load(fn));
		REQUIRE(buf.xDim == W);
		REQUIRE(buf.yDim == H);
		REQUIRE(buf.nFrames == nFrames);

		// Parallel per-frame decode
		std::vector<MultidimArray<float> > par((size_t)nFrames);
		#pragma omp parallel num_threads(4)
		{
			memtiff::TiffMovieBuffer::ThreadHandle th(buf);
			#pragma omp for schedule(dynamic, 1)
			for (long int f = 0; f < nFrames; f++)
				buf.decodeFrame(th, f, par[(size_t)f]);
		}
		for (int f = 0; f < nFrames; f++)
		{
			REQUIRE(XSIZE(par[(size_t)f]) == W);
			REQUIRE(YSIZE(par[(size_t)f]) == H);
			const float *p = MULTIDIM_ARRAY(par[(size_t)f]);
			for (size_t i = 0; i < (size_t)W * H; i++)
				REQUIRE(p[i] == ref[(size_t)f][i]);
		}

		// Whole-stack Image<float>::read() path (src/rwTIFF.h)
		Image<float> img;
		img.read(fn);
		REQUIRE(XSIZE(img()) == W);
		REQUIRE(YSIZE(img()) == H);
		REQUIRE(NSIZE(img()) == nFrames);
		for (int f = 0; f < nFrames; f++)
		{
			const float *p = &DIRECT_NZYX_ELEM(img(), f, 0, 0, 0);
			for (size_t i = 0; i < (size_t)W * H; i++)
				REQUIRE(p[i] == ref[(size_t)f][i]);
		}
	}
}
