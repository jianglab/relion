/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef MEM_TIFF_H_
#define MEM_TIFF_H_

#include <tiffio.h>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <typeinfo>
#include <vector>

#include "src/async_file_writer.h"
#include "src/error.h"
#include "src/filename.h"
#include "src/metadata_table.h"
#include "src/multidim_array.h"

namespace memtiff
{

namespace detail
{

// Mirror enum DataType values from src/image.h without creating a header cycle
enum SampleType
{
	ST_UChar  = 1,
	ST_SChar  = 2,
	ST_UShort = 3,
	ST_SShort = 4,
	ST_Float  = 8,
	ST_UHalf  = 11
};

struct MemCtx
{
	const unsigned char *data = NULL;
	tmsize_t size = 0;
	toff_t off = 0;
};

inline tmsize_t memRead(thandle_t h, void *dst, tmsize_t n)
{
	MemCtx *m = (MemCtx*)h;
	if (m->off >= (toff_t)m->size || n <= 0)
		return 0;
	const toff_t rem = (toff_t)m->size - m->off;
	if ((toff_t)n > rem)
		n = (tmsize_t)rem;
	memcpy(dst, m->data + m->off, (size_t)n);
	m->off += n;
	return n;
}

inline tmsize_t memWrite(thandle_t, void *, tmsize_t)
{
	return 0;
}

inline toff_t memSeek(thandle_t h, toff_t off, int whence)
{
	MemCtx *m = (MemCtx*)h;
	toff_t target = m->off;
	if (whence == SEEK_SET)
		target = off;
	else if (whence == SEEK_CUR)
		target = m->off + off;
	else if (whence == SEEK_END)
		target = (toff_t)m->size + off;
	else
		return (toff_t)-1;

	if (target > (toff_t)m->size)
		return (toff_t)-1;
	m->off = target;
	return target;
}

inline int memClose(thandle_t)
{
	return 0;
}

inline toff_t memSize(thandle_t h)
{
	return (toff_t)((MemCtx*)h)->size;
}

inline int memMap(thandle_t, void **, toff_t *)
{
	return 0;
}

inline void memUnmap(thandle_t, void *, toff_t)
{
}

inline TIFF* openMemTiffHandle(MemCtx &ctx, const unsigned char *buf, size_t len)
{
	ctx.data = buf;
	ctx.size = (tmsize_t)len;
	ctx.off = 0;
	return TIFFClientOpen("in-memory-tiff", "rm", (thandle_t)&ctx,
	                      memRead, memWrite, memSeek, memClose, memSize, memMap, memUnmap);
}

inline bool hasTiffMagicBytes(const unsigned char *b, size_t n)
{
	if (n < 4) return false;
	return (b[0] == 'I' && b[1] == 'I' && (b[2] == 42 || b[2] == 43) && b[3] == 0)
	    || (b[0] == 'M' && b[1] == 'M' && b[2] == 0 && (b[3] == 42 || b[3] == 43));
}

template <typename T>
inline void castSamples(const unsigned char *src, T * __restrict dst, int stype, size_t count)
{
	switch (stype)
	{
	case ST_UChar:
		if (typeid(T) == typeid(unsigned char))
			memcpy(dst, src, count * sizeof(T));
		else
			for (size_t i = 0; i < count; i++) dst[i] = (T)src[i];
		break;
	case ST_SChar:
		if (typeid(T) == typeid(signed char))
			memcpy(dst, src, count * sizeof(T));
		else
		{
			const signed char *p = (const signed char*)src;
			for (size_t i = 0; i < count; i++) dst[i] = (T)p[i];
		}
		break;
	case ST_UShort:
		if (typeid(T) == typeid(unsigned short))
			memcpy(dst, src, count * sizeof(T));
		else
		{
			const unsigned short *p = (const unsigned short*)src;
			for (size_t i = 0; i < count; i++) dst[i] = (T)p[i];
		}
		break;
	case ST_SShort:
		if (typeid(T) == typeid(short))
			memcpy(dst, src, count * sizeof(T));
		else
		{
			const short *p = (const short*)src;
			for (size_t i = 0; i < count; i++) dst[i] = (T)p[i];
		}
		break;
	case ST_Float:
		if (typeid(T) == typeid(float))
			memcpy(dst, src, count * sizeof(T));
		else
		{
			const float *p = (const float*)src;
			for (size_t i = 0; i < count; i++) dst[i] = (T)p[i];
		}
		break;
	case ST_UHalf:
	{
		if (count % 2 != 0)
			REPORT_ERROR("Logic error in memtiff::castSamples; for UHalf, count must be even.");
		const size_t half = count / 2;
		for (size_t i = 0; i < half; i++)
		{
			// IMOD assumes LSB2MSB fill order even for TIFF (matches src/image.h castPage2T)
			const unsigned char v = src[i];
			dst[i * 2]     = (T)(v & 15);
			dst[i * 2 + 1] = (T)((v >> 4) & 15);
		}
		break;
	}
	default:
		REPORT_ERROR("memtiff::castSamples: unsupported TIFF datatype.");
	}
}

} // namespace detail

/**
 * Loads a multi-frame TIFF movie once into memory and records per-IFD byte offsets
 * so multiple OpenMP threads can jump to and decode arbitrary frames in O(1) time
 * with zero per-frame file opens, directory scans, or Y-flip passes.
 */
class TiffMovieBuffer
{
public:
	long int xDim = 0;
	long int yDim = 0;
	long int nFrames = 0;
	uint32_t width = 0;
	uint32_t length = 0;
	uint16_t bitsPerSample = 0;
	uint16_t sampleFormat = 0;
	bool packed_4bit = false;
	int datatype = 0;
	bool hasResolution = false;
	double angpix = 0.0;
	FileName sourceName;

	struct ThreadHandle
	{
		detail::MemCtx ctx;
		TIFF *ftiff = NULL;
		tdata_t stripBuf = NULL;
		tsize_t stripBufCap = 0;

		explicit ThreadHandle(const TiffMovieBuffer &owner)
		{
			if (!owner.bytes.empty())
				ftiff = detail::openMemTiffHandle(ctx, owner.bytes.data(), owner.bytes.size());
		}

		~ThreadHandle()
		{
			if (stripBuf != NULL)
			{
				_TIFFfree(stripBuf);
				stripBuf = NULL;
				stripBufCap = 0;
			}
			if (ftiff != NULL)
			{
				TIFFClose(ftiff);
				ftiff = NULL;
			}
		}

		ThreadHandle(const ThreadHandle&) = delete;
		ThreadHandle& operator=(const ThreadHandle&) = delete;

		void ensureStripBuf(tsize_t sz)
		{
			if (sz > stripBufCap)
			{
				if (stripBuf != NULL) _TIFFfree(stripBuf);
				stripBuf = _TIFFmalloc(sz);
				if (stripBuf == NULL)
					REPORT_ERROR("memtiff::ThreadHandle: failed to allocate TIFF strip buffer.");
				stripBufCap = sz;
			}
		}
	};

	static bool isTiffFilename(const FileName &fn)
	{
		long int idx = 0;
		FileName path;
		fn.decompose(idx, path);
		path = path.removeFileFormat();
		std::string ext = path.getExtension();
		for (size_t i = 0; i < ext.size(); i++)
			ext[i] = (char)std::tolower((unsigned char)ext[i]);
		return ext == "tif" || ext == "tiff";
	}

	void clear()
	{
		std::vector<unsigned char>().swap(bytes);
		dirOffsets.clear();
		xDim = yDim = nFrames = 0;
		width = length = 0;
		bitsPerSample = sampleFormat = 0;
		packed_4bit = false;
		datatype = 0;
		hasResolution = false;
		angpix = 0.0;
		sourceName.clear();
	}

	bool empty() const { return bytes.empty() || nFrames <= 0; }

	bool load(const FileName &fn, size_t max_bytes = 0)
	{
		clear();
		long int idx = 0;
		FileName path;
		fn.decompose(idx, path);
		path = path.removeFileFormat();

		struct stat st;
		if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 4)
			return false;

		const size_t fsize = (size_t)st.st_size;
		size_t limit = max_bytes > 0 ? max_bytes : ((size_t)2 << 30);
		if (max_bytes == 0)
		{
			const long long avail = asyncwrite::memoryAvailableBytes();
			if (avail >= 0)
				limit = std::min(limit, (size_t)(avail / 4));
		}
		if (fsize > limit)
			return false;

		bytes.resize(fsize);
		FILE *fp = fopen(path.c_str(), "rb");
		if (fp == NULL)
		{
			clear();
			return false;
		}
		size_t total = 0;
		while (total < fsize)
		{
			const size_t got = fread(bytes.data() + total, 1, fsize - total, fp);
			if (got == 0) break;
			total += got;
		}
		fclose(fp);
		if (total != fsize || !detail::hasTiffMagicBytes(bytes.data(), bytes.size()))
		{
			clear();
			return false;
		}

		detail::MemCtx ctx;
		TIFF *ftiff = detail::openMemTiffHandle(ctx, bytes.data(), bytes.size());
		if (ftiff == NULL)
		{
			clear();
			return false;
		}
		if (TIFFIsTiled(ftiff) ||
		    TIFFGetField(ftiff, TIFFTAG_IMAGEWIDTH, &width) != 1 ||
		    TIFFGetField(ftiff, TIFFTAG_IMAGELENGTH, &length) != 1)
		{
			TIFFClose(ftiff);
			clear();
			return false;
		}

		TIFFGetFieldDefaulted(ftiff, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
		TIFFGetFieldDefaulted(ftiff, TIFFTAG_SAMPLEFORMAT, &sampleFormat);

		packed_4bit = (bitsPerSample == 8 &&
		    ((width == 5760 && length == 8184)  || (width == 8184  && length == 5760) ||
		     (width == 4092 && length == 11520) || (width == 11520 && length == 4092) ||
		     (width == 3710 && length == 7676)  || (width == 7676  && length == 3710) ||
		     (width == 3838 && length == 7420)  || (width == 7420  && length == 3838)));

		xDim = packed_4bit ? (long int)width * 2 : (long int)width;
		yDim = (long int)length;

		if (packed_4bit)
			datatype = detail::ST_UHalf;
		else if (bitsPerSample == 8 && sampleFormat == SAMPLEFORMAT_UINT)
			datatype = detail::ST_UChar;
		else if (bitsPerSample == 8 && sampleFormat == SAMPLEFORMAT_INT)
			datatype = detail::ST_SChar;
		else if (bitsPerSample == 16 && sampleFormat == SAMPLEFORMAT_UINT)
			datatype = detail::ST_UShort;
		else if (bitsPerSample == 16 && sampleFormat == SAMPLEFORMAT_INT)
			datatype = detail::ST_SShort;
		else if (bitsPerSample == 32 && sampleFormat == SAMPLEFORMAT_IEEEFP)
			datatype = detail::ST_Float;
		else
		{
			TIFFClose(ftiff);
			clear();
			return false;
		}

		uint16_t resolutionUnit = 0;
		float xResolution = 0.0f;
		if (TIFFGetField(ftiff, TIFFTAG_RESOLUTIONUNIT, &resolutionUnit) == 1 &&
		    TIFFGetField(ftiff, TIFFTAG_XRESOLUTION, &xResolution) == 1 && xResolution > 0.0f)
		{
			if (resolutionUnit == RESUNIT_INCH)
			{
				hasResolution = true;
				angpix = 2.54E8 / (double)xResolution;
			}
			else if (resolutionUnit == RESUNIT_CENTIMETER)
			{
				hasResolution = true;
				angpix = 1.00E8 / (double)xResolution;
			}
		}

		dirOffsets.clear();
		do
		{
			if (TIFFIsTiled(ftiff))
			{
				TIFFClose(ftiff);
				clear();
				return false;
			}
			dirOffsets.push_back(TIFFCurrentDirOffset(ftiff));
		} while (TIFFReadDirectory(ftiff) != 0);

		TIFFClose(ftiff);
		if (dirOffsets.empty())
		{
			clear();
			return false;
		}
		nFrames = (long int)dirOffsets.size();
		sourceName = path;
		return true;
	}

	template <typename T>
	void decodeFrameToPtr(ThreadHandle &th, long int frame_idx, T * __restrict dstPtr, MetaDataTable *mdHeader = NULL) const
	{
		if (frame_idx < 0 || frame_idx >= nFrames)
			REPORT_ERROR("memtiff: Image number " + integerToString(frame_idx + 1) + " exceeds stack size "
			             + integerToString(nFrames) + " of image " + sourceName);

		if (th.ftiff == NULL)
			REPORT_ERROR("memtiff: failed to open in-memory TIFF handle for " + sourceName);

		if (TIFFSetSubDirectory(th.ftiff, dirOffsets[frame_idx]) == 0)
			REPORT_ERROR("memtiff: failed to seek TIFF frame " + integerToString(frame_idx) + " in " + sourceName);

		uint32_t cur_width = 0, cur_length = 0;
		uint16_t cur_sampleFormat = 0, cur_bitsPerSample = 0;
		if (TIFFGetField(th.ftiff, TIFFTAG_IMAGEWIDTH, &cur_width) != 1 ||
		    TIFFGetField(th.ftiff, TIFFTAG_IMAGELENGTH, &cur_length) != 1)
			REPORT_ERROR(sourceName + ": The input TIFF file does not have the width or height field.");

		TIFFGetFieldDefaulted(th.ftiff, TIFFTAG_BITSPERSAMPLE, &cur_bitsPerSample);
		TIFFGetFieldDefaulted(th.ftiff, TIFFTAG_SAMPLEFORMAT, &cur_sampleFormat);
		if (cur_width != width || cur_length != length ||
		    cur_bitsPerSample != bitsPerSample || cur_sampleFormat != sampleFormat)
			REPORT_ERROR(sourceName + ": All frames in a TIFF should have same width, height and pixel format.\n");

		if (mdHeader != NULL)
		{
			mdHeader->setValue(EMDL_IMAGE_DATATYPE, datatype);
			if (hasResolution)
			{
				mdHeader->setValue(EMDL_IMAGE_SAMPLINGRATE_X, (RFLOAT)angpix);
				mdHeader->setValue(EMDL_IMAGE_SAMPLINGRATE_Y, (RFLOAT)angpix);
			}
		}

		const tsize_t stripSize = TIFFStripSize(th.ftiff);
		const tstrip_t numberOfStrips = TIFFNumberOfStrips(th.ftiff);
		if (stripSize <= 0)
			REPORT_ERROR("memtiff: invalid TIFF strip size in " + sourceName);
		th.ensureStripBuf(stripSize);

		const size_t totalPix = (size_t)xDim * (size_t)yDim;
		const size_t rowBytes = ((size_t)width * bitsPerSample) / 8;
		size_t haveread_n = 0;
		bool direct_flipped = (totalPix > 0 && rowBytes > 0 && (((size_t)width * bitsPerSample) % 8 == 0));

		for (tstrip_t strip = 0; strip < numberOfStrips; strip++)
		{
			const tsize_t actually_read = TIFFReadEncodedStrip(th.ftiff, strip, th.stripBuf, stripSize);
			if (actually_read == -1)
				REPORT_ERROR((std::string)"Failed to read an image data from " + sourceName);

			size_t actually_read_n = (size_t)actually_read * 8 / bitsPerSample;
			if (packed_4bit) actually_read_n *= 2;
			if (haveread_n + actually_read_n > totalPix)
				REPORT_ERROR(sourceName + ": TIFF strip data exceeds image dimensions.");

			const bool row_aligned = (haveread_n % (size_t)xDim == 0) && (actually_read_n % (size_t)xDim == 0);
			if (direct_flipped && row_aligned)
			{
				const long int row0 = (long int)(haveread_n / (size_t)xDim);
				const long int nrows = (long int)(actually_read_n / (size_t)xDim);
				const unsigned char *src = (const unsigned char*)th.stripBuf;
				for (long int r = 0; r < nrows; r++)
				{
					T * __restrict dstRow = dstPtr + (size_t)(yDim - 1 - (row0 + r)) * (size_t)xDim;
					detail::castSamples(src + (size_t)r * rowBytes, dstRow, datatype, (size_t)xDim);
				}
			}
			else
			{
				if (direct_flipped)
				{
					// Restore already-written flipped rows back to top-down before continuing linearly
					const long int rows_done = (long int)(haveread_n / (size_t)xDim);
					if (rows_done > 0)
					{
						std::vector<T> saved((size_t)rows_done * (size_t)xDim);
						for (long int r = 0; r < rows_done; r++)
						{
							const T *srcRow = dstPtr + (size_t)(yDim - 1 - r) * (size_t)xDim;
							std::copy(srcRow, srcRow + xDim, saved.data() + (size_t)r * (size_t)xDim);
						}
						std::copy(saved.begin(), saved.end(), dstPtr);
					}
					direct_flipped = false;
				}
				detail::castSamples((const unsigned char*)th.stripBuf, dstPtr + haveread_n, datatype, actually_read_n);
			}

			haveread_n += actually_read_n;
		}

		if (!direct_flipped)
		{
			const long int ylim = yDim / 2;
			for (long int y1 = 0; y1 < ylim; y1++)
			{
				const long int y2 = yDim - 1 - y1;
				T * __restrict r1 = dstPtr + (size_t)y1 * (size_t)xDim;
				T * __restrict r2 = dstPtr + (size_t)y2 * (size_t)xDim;
				std::swap_ranges(r1, r1 + xDim, r2);
			}
		}
	}

	template <typename T>
	void decodeFrame(ThreadHandle &th, long int frame_idx, MultidimArray<T> &dst, MetaDataTable *mdHeader = NULL) const
	{
		dst.setDimensions(xDim, yDim, 1, 1);
		dst.coreAllocateReuse();
		decodeFrameToPtr(th, frame_idx, MULTIDIM_ARRAY(dst), mdHeader);
	}

	template <typename ImageT>
	void decodeFrameImage(ThreadHandle &th, long int frame_idx, ImageT &img) const
	{
		img.MDMainHeader.clear();
		img.MDMainHeader.addObject();
		decodeFrame(th, frame_idx, img(), &img.MDMainHeader);
	}

private:
	std::vector<unsigned char> bytes;
	std::vector<uint64_t> dirOffsets;
};

} // namespace memtiff

#endif /* MEM_TIFF_H_ */
