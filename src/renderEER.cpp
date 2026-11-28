#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

#include <src/time.h>
#include <src/metadata_table.h>
#include <src/image.h>
#include <src/renderEER.h>

namespace
{
struct EerMemTiff
{
	const unsigned char *data;
	tmsize_t size;
	toff_t off;
};

tmsize_t eerMemRead(thandle_t h, void *dst, tmsize_t n)
{
	EerMemTiff *m = (EerMemTiff*)h;
	if ((toff_t)n > (toff_t)m->size - m->off)
		n = (tmsize_t)((toff_t)m->size - m->off);
	if (n > 0)
	{
		memcpy(dst, m->data + m->off, n);
		m->off += n;
	}
	return n;
}

tmsize_t eerMemWrite(thandle_t, void *, tmsize_t)
{
	return 0;
}

toff_t eerMemSeek(thandle_t h, toff_t off, int whence)
{
	EerMemTiff *m = (EerMemTiff*)h;
	toff_t target = m->off;
	if (whence == SEEK_SET) target = off;
	else if (whence == SEEK_CUR) target += off;
	else if (whence == SEEK_END) target = (toff_t)m->size + off;
	if (target > (toff_t)m->size) return (toff_t)-1;
	m->off = target;
	return target;
}

int eerMemClose(thandle_t)
{
	return 0;
}

toff_t eerMemSize(thandle_t h)
{
	return (toff_t)((EerMemTiff*)h)->size;
}

int eerMemMap(thandle_t, void **, toff_t *)
{
	return 0;
}

void eerMemUnmap(thandle_t, void *, toff_t)
{
}

bool hasTiffMagic(const unsigned char *b, long long n)
{
	if (n < 4) return false;
	return (b[0] == 'I' && b[1] == 'I' && (b[2] == 42 || b[2] == 43) && b[3] == 0)
	    || (b[0] == 'M' && b[1] == 'M' && b[2] == 0 && (b[3] == 42 || b[3] == 43));
}
}

//#define DEBUG_EER

//#define TIMING
#ifdef TIMING
	#define RCTIC(label) (EERtimer.tic(label))
	#define RCTOC(label) (EERtimer.toc(label))

	Timer EERtimer;
	int TIMING_READ_EER = EERtimer.setNew("read EER");
	int TIMING_BUILD_INDEX = EERtimer.setNew("build index");
	int TIMING_UNPACK_RLE = EERtimer.setNew("unpack RLE");
	int TIMING_RENDER_ELECTRONS = EERtimer.setNew("render electrons");
#else
	#define RCTIC(label)
	#define RCTOC(label)
#endif

const char EERRenderer::EER_FOOTER_OK[]  = "ThermoFisherECComprOK000";
const char EERRenderer::EER_FOOTER_ERR[] = "ThermoFisherECComprERR00";
const int EERRenderer::EER_4K= 4096;
const int EERRenderer::EER_2K = 2048;
const unsigned int EERRenderer::EER_LEN_FOOTER = 24;
const uint16_t EERRenderer::TIFF_COMPRESSION_EER8bit = 65000;
const uint16_t EERRenderer::TIFF_COMPRESSION_EER7bit = 65001;
const uint16_t EERRenderer::TIFF_COMPRESSION_EERDetailed = 65002;
const ttag_t EERRenderer::TIFFTAG_EER_RLE_DEPTH = 65007;
const ttag_t EERRenderer::TIFFTAG_EER_SUBPIXEL_H_DEPTH = 65008;
const ttag_t EERRenderer::TIFFTAG_EER_SUBPIXEL_V_DEPTH = 65009;

TIFFErrorHandler EERRenderer::prevTIFFWarningHandler = NULL;

void EERRenderer::TIFFWarningHandler(const char* module, const char* fmt, va_list ap)
{
	// Silence warnings for private tags
	if (strstr("Unknown field with tag", fmt) == 0 || strstr("encountered", fmt) == 0)
		return;

	if (prevTIFFWarningHandler != NULL)
		prevTIFFWarningHandler(module, fmt, ap);
}

void EERRenderer::silenceTIFFWarnings()
{
	if (prevTIFFWarningHandler == NULL)
	{
		// Thread safety issue:
		// Calling this simultaneously is safe but
		TIFFErrorHandler prev = TIFFSetWarningHandler(EERRenderer::TIFFWarningHandler);

		// we have to make sure prevTIFFWarningHandler does NOT become our own TIFFWarningHandler
		// to avoid an infinite loop.
		if (prev != EERRenderer::TIFFWarningHandler)
			prevTIFFWarningHandler = prev;
	}
}

template <typename T>
void EERRenderer::render4K_to_16K(MultidimArray<T> &image, std::vector<unsigned int> &positions, std::vector<unsigned char> &symbols, int n_electrons)
{
	T* __restrict dst = MULTIDIM_ARRAY(image);
	const unsigned int* __restrict pos = positions.data();
	const unsigned char* __restrict sym = symbols.data();
	for (int i = 0; i < n_electrons; i++)
	{
		const unsigned int p = pos[i];
		const unsigned char s = sym[i];
		const int x = ((p & 4095u) << 2) | (s & 3u);
		const int y = ((p >> 12) << 2) | ((s & 12u) >> 2);
		dst[(size_t)y * 16384u + x]++;
	}
}

template <typename T>
void EERRenderer::render4K_to_8K(MultidimArray<T> &image, std::vector<unsigned int> &positions, std::vector<unsigned char> &symbols, int n_electrons)
{
	T* __restrict dst = MULTIDIM_ARRAY(image);
	const unsigned int* __restrict pos = positions.data();
	const unsigned char* __restrict sym = symbols.data();
	for (int i = 0; i < n_electrons; i++)
	{
		const unsigned int p = pos[i];
		const unsigned char s = sym[i];
		const int x = ((p & 4095u) << 1) | ((s & 2u) >> 1);
		const int y = ((p >> 12) << 1) | ((s & 8u) >> 3);
		dst[(size_t)y * 8192u + x]++;
	}
}

template <typename T>
void EERRenderer::render4K_to_4K(MultidimArray<T> &image, std::vector<unsigned int> &positions, std::vector<unsigned char> &symbols, int n_electrons)
{
	T* __restrict dst = MULTIDIM_ARRAY(image);
	const unsigned int* __restrict pos = positions.data();
	for (int i = 0; i < n_electrons; i++)
	{
		dst[pos[i]]++;
	}
}

template <typename T>
void EERRenderer::render2K_to_4K(MultidimArray<T> &image, std::vector<unsigned int> &positions, std::vector<unsigned char> &symbols, int n_electrons)
{
	T* __restrict dst = MULTIDIM_ARRAY(image);
	const unsigned int* __restrict pos = positions.data();
	const unsigned char* __restrict sym = symbols.data();
	for (int i = 0; i < n_electrons; i++)
	{
		const unsigned int p = pos[i];
		const unsigned char s = sym[i];
		const int x = ((p & 2047u) << 1) | (s & 1u);
		const int y = ((p >> 11) << 1) | (s >> 1);
		dst[(size_t)y * 4096u + x]++;
	}
}

template <typename T>
void EERRenderer::render4K_to_2K(MultidimArray<T> &image, std::vector<unsigned int> &positions, std::vector<unsigned char> &symbols, int n_electrons)
{
	T* __restrict dst = MULTIDIM_ARRAY(image);
	const unsigned int* __restrict pos = positions.data();
	for (int i = 0; i < n_electrons; i++)
	{
		const unsigned int p = pos[i];
		const int x = (p & 4095u) >> 1;
		const int y = (p >> 12) >> 1;
		dst[(size_t)y * 2048u + x]++;
	}
}

template <typename T>
void EERRenderer::render2K_to_2K(MultidimArray<T> &image, std::vector<unsigned int> &positions, std::vector<unsigned char> &symbols, int n_electrons)
{
	T* __restrict dst = MULTIDIM_ARRAY(image);
	const unsigned int* __restrict pos = positions.data();
	for (int i = 0; i < n_electrons; i++)
	{
		dst[pos[i]]++;
	}
}

EERRenderer::EERRenderer()
{
	ready = false;
	read_data = false;
	buf = NULL;
	preread_start = -1;
	preread_end = -1;
	eer_upsampling = 1;
}

void EERRenderer::read(FileName _fn_movie, int eer_upsampling, bool header_only)
{
	if (ready)
		REPORT_ERROR("Logic error: you cannot recycle EERRenderer for multiple files (now)");

	if (eer_upsampling == -1 || eer_upsampling == 1 || eer_upsampling == 2 || eer_upsampling == 3)
		this->eer_upsampling = eer_upsampling;
	else
	{
		std::cerr << "EERRenderer::read: eer_upsampling = " << eer_upsampling << std::endl;
		REPORT_ERROR("EERRenderer::read: eer_upsampling must be -1, 1, 2 or 3.");
	}

	fn_movie = _fn_movie;

	FILE *fh = fopen(fn_movie.c_str(), "rb");
	if (fh == NULL)
		REPORT_ERROR("Failed to open " + fn_movie);

	fseek(fh, 0, SEEK_END);
	file_size = ftell(fh);
	fseek(fh, 0, SEEK_SET);

	silenceTIFFWarnings();

	TIFF *ftiff = NULL;
	EerMemTiff memctx{NULL, (tmsize_t)file_size, 0};
	if (header_only)
	{
		unsigned char hdr[4] = {0, 0, 0, 0};
		const size_t nread = fread(hdr, 1, sizeof(hdr), fh);
		if (hasTiffMagic(hdr, (long long)nread))
		{
			ftiff = TIFFOpen(fn_movie.c_str(), "rm");
			if (ftiff != NULL)
			{
				fclose(fh);
				fh = NULL;
			}
		}
		if (fh != NULL)
			fseek(fh, 0, SEEK_SET);
	}

	if (fh != NULL)
	{
		// Load whole EER file in one sequential read, then parse TIFF/legacy from memory
		buf = (unsigned char*)malloc((size_t)file_size + 16);
		if (buf == NULL)
		{
			fclose(fh);
			REPORT_ERROR("Failed to allocate the buffer for " + fn_movie);
		}
		RCTIC(TIMING_READ_EER);
		if ((long long)fread(buf, sizeof(char), file_size, fh) != file_size)
		{
			fclose(fh);
			REPORT_ERROR("EERRenderer::read: Failed to read the expected size from " + fn_movie);
		}
		RCTOC(TIMING_READ_EER);
		memset(buf + file_size, 0, 16);
		fclose(fh);

		memctx.data = buf;
		ftiff = hasTiffMagic(buf, file_size)
			? TIFFClientOpen(fn_movie.c_str(), "rm", (thandle_t)&memctx,
			                 eerMemRead, eerMemWrite, eerMemSeek, eerMemClose, eerMemSize, eerMemMap, eerMemUnmap)
			: NULL;
	}

	if (ftiff == NULL)
	{
		is_legacy = true;
		readLegacy(NULL);
	}
	else
	{
		is_legacy = false;

		// Check width & size
		uint16_t compression = 0;
		TIFFGetField(ftiff, TIFFTAG_IMAGEWIDTH, &width);
		TIFFGetField(ftiff, TIFFTAG_IMAGELENGTH, &height);
		TIFFGetField(ftiff, TIFFTAG_COMPRESSION, &compression);

#ifdef DEBUG_EER
		printf("EER in TIFF: %s size = %lld, width = %d, height = %d, compression = %d\n", fn_movie.c_str(), file_size, width, height, compression);
#endif

		// TIA can write an EER file whose first page is a sum and compressoin == 1.
		// This is not supported (yet). EPU never writes such movies.
		if (compression == EERRenderer::TIFF_COMPRESSION_EER8bit)
		{
			rle_bits = 8;
			subpixel_bits = 4;
		}
		else if (compression == EERRenderer::TIFF_COMPRESSION_EER7bit)
		{
			rle_bits = 7;
			subpixel_bits = 4;
		}
		else if (compression == EERRenderer::TIFF_COMPRESSION_EERDetailed)
		{
			// See https://stackoverflow.com/questions/33522589/how-to-read-custom-tiff-tags-w-o-tifffieldinfo
			// and "AUTOREGISTERED TAGS" in https://manpages.debian.org/testing/libtiff-dev/TIFFGetField.3tiff.en.html.
			uint32_t count;
			uint16_t *rle, *subpix_h, *subpix_v;

			TIFFGetField(ftiff, EERRenderer::TIFFTAG_EER_RLE_DEPTH, &count, (void*)&rle);
			rle_bits = *rle;
			TIFFGetField(ftiff, EERRenderer::TIFFTAG_EER_SUBPIXEL_H_DEPTH, &count, (void*)&subpix_h);
			// Prototype images apparently don't have TIFFTAG_EER_SUBPIXEL_V_DEPTH
			if (TIFFGetField(ftiff, EERRenderer::TIFFTAG_EER_SUBPIXEL_V_DEPTH, &count, (void*)&subpix_v) == 0)
				subpix_v = subpix_h;

			if (rle_bits != 7 || *subpix_h != *subpix_v || *subpix_h != 1)
			{
				TIFFClose(ftiff);
				REPORT_ERROR("Unsupported compression scheme: type = " + integerToString(compression) +
				             ", rle_bits = " + integerToString(rle_bits) + ", subpix_h = " + integerToString(*subpix_h) +
				             ", subpix_v = " + integerToString(*subpix_v));
			}
			subpixel_bits = 1 << *subpix_h;

			if (subpixel_bits == 2 && (eer_upsampling == -1 || eer_upsampling >= 3))
			{
				TIFFClose(ftiff);
				REPORT_ERROR("For subpixel_bits = 2, eer_upsamling must be 1 or 2.");
			}
		}
		else
		{
			TIFFClose(ftiff);
			REPORT_ERROR("Unknown compression scheme for EER: " + integerToString(compression));
		}

		if (width != height || (width != EER_4K && width != EER_2K))
		{
			TIFFClose(ftiff);
			REPORT_ERROR("Currently we support only 4096x4096 or 2048x2048 pixel EER movies.");
		}

		total_pixels = (long long)width * height;

		if (header_only && buf == NULL)
		{
			nframes = TIFFNumberOfDirectories(ftiff);
			read_data = false;
			TIFFClose(ftiff);
			ready = true;
			return;
		}

		// Walk IFDs once in memory and index single-strip frames directly into buf (zero copy)
		RCTIC(TIMING_BUILD_INDEX);
		bool zero_copy = true;
		frame_starts.clear();
		frame_sizes.clear();
		do
		{
			const uint32_t nstrips = TIFFNumberOfStrips(ftiff);
			uint64_t *offsets = NULL, *counts = NULL;
			if (nstrips != 1
			 || TIFFGetField(ftiff, TIFFTAG_STRIPOFFSETS, &offsets) != 1
			 || TIFFGetField(ftiff, TIFFTAG_STRIPBYTECOUNTS, &counts) != 1
			 || offsets == NULL || counts == NULL
			 || (long long)offsets[0] < 0
			 || (long long)(offsets[0] + counts[0]) > file_size)
			{
				zero_copy = false;
				break;
			}
			frame_starts.push_back((long long)offsets[0]);
			frame_sizes.push_back((long long)counts[0]);
		}
		while (TIFFReadDirectory(ftiff));

		if (!zero_copy)
		{
			TIFFSetDirectory(ftiff, 0);
			nframes = TIFFNumberOfDirectories(ftiff);
			frame_starts.assign(nframes, 0);
			frame_sizes.assign(nframes, 0);
			unsigned char *sbuf = (unsigned char*)malloc((size_t)file_size + 16);
			if (sbuf == NULL)
			{
				TIFFClose(ftiff);
				REPORT_ERROR("Failed to allocate the buffer for " + fn_movie);
			}
			memset(sbuf + file_size, 0, 16);
			long long pos = 0;
			for (int frame = 0; frame < nframes; frame++)
			{
				if (!(frame == 0 ? TIFFSetDirectory(ftiff, 0) : TIFFReadDirectory(ftiff)))
				{
					free(sbuf);
					TIFFClose(ftiff);
					REPORT_ERROR("EER: cannot read the directory of frame " + integerToString(frame + 1) + " in " + fn_movie);
				}
				const int nstrips = TIFFNumberOfStrips(ftiff);
				frame_starts[frame] = pos;
				for (int strip = 0; strip < nstrips; strip++)
				{
					const tmsize_t strip_size = TIFFRawStripSize(ftiff, strip);
					if (strip_size < 0 || pos + strip_size > file_size)
					{
						free(sbuf);
						TIFFClose(ftiff);
						REPORT_ERROR("EER: buffer overflow when reading raw strips.");
					}
					TIFFReadRawStrip(ftiff, strip, sbuf + pos, strip_size);
					pos += strip_size;
					frame_sizes[frame] += strip_size;
				}
			}
			free(buf);
			buf = sbuf;
		}
		RCTOC(TIMING_BUILD_INDEX);

		nframes = frame_starts.size();
		read_data = true;
		TIFFClose(ftiff);
#ifdef DEBUG_EER
		printf("EER in TIFF: %s rle_bits = %d, subpixel_bits = %d, nframes = %d\n", fn_movie.c_str(), rle_bits, subpixel_bits, nframes);
#endif
	}

	ready = true;
}

void EERRenderer::readLegacy(FILE *fh)
{
	if (buf == NULL)
	{
		RCTIC(TIMING_READ_EER);
		buf = (unsigned char*)malloc((size_t)file_size + 16);
		if (buf == NULL)
			REPORT_ERROR("Failed to allocate the buffer.");
		if ((long long)fread(buf, sizeof(char), file_size, fh) != file_size)
			REPORT_ERROR("EERRenderer::readLegacy: Failed to read the expected size from " + fn_movie);
		memset(buf + file_size, 0, 16);
		RCTOC(TIMING_READ_EER);
	}

	/* Build frame index */
	RCTIC(TIMING_BUILD_INDEX);
	long long pos = file_size;
	while (pos > 0)
	{
		pos -= EER_LEN_FOOTER;
		if (strncmp(EER_FOOTER_OK, (char*)buf + pos, EER_LEN_FOOTER) == 0)
		{
			pos -= 8;
			long long frame_size = *(long long*)(buf + pos) * sizeof(long long);
			pos -= frame_size;
			frame_starts.push_back(pos);
			frame_sizes.push_back(frame_size);
#ifdef DEBUG_EER
			printf("Frame: LAST-%5d Start at: %08d Frame size: %lld\n", frame_starts.size(), pos, frame_size);
#endif
		}
		else // if (strncmp(EER_FOOTER_ERR, (char*)buf + pos, EER_LEN_FOOTER) == 0)
		{
			REPORT_ERROR("Broken frame in file " + fn_movie);
		}
	}
	std::reverse(frame_starts.begin(), frame_starts.end());
	std::reverse(frame_sizes.begin(), frame_sizes.end());
	RCTOC(TIMING_BUILD_INDEX);

	nframes = frame_starts.size();
	read_data = true;
}

void EERRenderer::lazyReadFrames()
{
	// One lock per movie: a process-wide omp critical let only one movie be read at a time
	std::lock_guard<std::mutex> guard(read_mutex);
	if (read_data)
		return;

	TIFF *ftiff = TIFFOpen(fn_movie.c_str(), "r");

	frame_starts.resize(nframes, 0);
	frame_sizes.resize(nframes, 0);
	buf = (unsigned char*)malloc(file_size); // This is big enough
	if (buf == NULL)
		REPORT_ERROR("Failed to allocate the buffer for " + fn_movie);
	long long pos = 0;

	// Read everything
	const int first = (preread_start > 0) ? preread_start : 0;
	const int last = (preread_end > 0) ? std::min(preread_end, nframes - 1) : nframes - 1;
	for (int frame = first; frame <= last; frame++)
	{
		// Step to the next directory: TIFFSetDirectory(frame) walks the chain from the first one in
		// older libtiff, which is quadratic in the number of frames, in small reads
		if (!(frame == first ? TIFFSetDirectory(ftiff, frame) : TIFFReadDirectory(ftiff)))
			REPORT_ERROR("EER: cannot read the directory of frame " + integerToString(frame + 1) + " in " + fn_movie);
		const int nstrips = TIFFNumberOfStrips(ftiff);
		frame_starts[frame] = pos;

		for (int strip = 0; strip < nstrips; strip++)
		{
			const int strip_size = TIFFRawStripSize(ftiff, strip);
			if (pos + strip_size >= file_size)
				REPORT_ERROR("EER: buffer overflow when reading raw strips.");

			TIFFReadRawStrip(ftiff, strip, buf + pos, strip_size);
			pos += strip_size;
			frame_sizes[frame] += strip_size;
		}
#ifdef DEBUG_EER
		printf("EER in TIFF: Read frame %d from %s, nstrips = %d, current pos in buffer = %lld / %lld\n", frame, fn_movie.c_str(), nstrips, pos, file_size);
#endif
	}

	TIFFClose(ftiff);

	read_data = true;
}

EERRenderer::~EERRenderer()
{
	if (buf != NULL)
		free(buf);
}

int EERRenderer::getNFrames()
{
	if (!ready)
		REPORT_ERROR("EERRenderer::getNFrames called before ready.");

	return nframes;
}

int EERRenderer::getWidth()
{
	if (!ready)
		REPORT_ERROR("EERRenderer::getNFrames called before ready.");

	if (eer_upsampling == -1)
		return width >> 1;

	return width << (eer_upsampling - 1);
}

int EERRenderer::getHeight()
{
	if (!ready)
		REPORT_ERROR("EERRenderer::getNFrames called before ready.");

	if (eer_upsampling == -1)
		return height >> 1;

	return height << (eer_upsampling - 1);
}

template <typename T>
long long EERRenderer::renderFrames(int frame_start, int frame_end, MultidimArray<T> &image)
{
	if (!ready)
		REPORT_ERROR("EERRenderer::renderNFrames called before ready.");

	lazyReadFrames();

	if (frame_start <= 0 || frame_start > getNFrames() ||
	    frame_end < frame_start || frame_end > getNFrames())
	{
		std::cerr << "EERRenderer::renderFrames(frame_start = " << frame_start << ", frame_end = " << frame_end << "),  NFrames = " << getNFrames() << std::endl;
		REPORT_ERROR("Invalid frame range was requested.");
	}

	// Make this 0-indexed
	frame_start--;
	frame_end--;

	if (sizeof(T) > sizeof(unsigned short) && (frame_end - frame_start + 1) <= 16383)
	{
		MultidimArray<unsigned short> counts;
		const long long tot = renderFrames<unsigned short>(frame_start + 1, frame_end + 1, counts);
		image.resizeNoCp(1, 1, counts.ydim, counts.xdim);
		const size_t n = NZYXSIZE(counts);
		const unsigned short* __restrict src = MULTIDIM_ARRAY(counts);
		T* __restrict dst = MULTIDIM_ARRAY(image);
		for (size_t i = 0; i < n; i++) dst[i] = (T)src[i];
		return tot;
	}

	long long total_n_electron = 0;

	std::vector<unsigned int> positions;
	std::vector<unsigned char> symbols;
	image.initZeros(getHeight(), getWidth());

	for (int iframe = frame_start; iframe <= frame_end; iframe++)
	{
		RCTIC(TIMING_UNPACK_RLE);
		if ((preread_start > 0 && iframe < preread_start) ||
	  	    (preread_end > 0 && iframe > preread_end))
		{
			std::cerr << "EERRenderer::renderFrames(frame_start = " << frame_start + 1 << ", frame_end = " << frame_end + 1<< "),  NFrames = " << getNFrames() << " preread_start = " << preread_start + 1 << " prered_end = " << preread_end + 1<< std::endl;
			REPORT_ERROR("Tried to render frames outside pre-read region");
		}		

		long long pos = frame_starts[iframe];
		unsigned int n_pix = 0, n_electron = 0;
		const int max_electrons = frame_sizes[iframe] * 2; // at 4 bits per electron (very permissive bound!)
		if (positions.size() < max_electrons)
		{
			positions.resize(max_electrons);
			symbols.resize(max_electrons);
		}

		if (rle_bits == 7 && subpixel_bits == 4)
		{
			unsigned int bit_pos = 0; // 4 K * 4 K * 11 bit << 2 ** 32
			unsigned char p, s;

			while (true)
			{
				// Fetch 32 bits and unpack up to 2 chunks of 7 + 4 bits.
				// This is faster than unpack 7 and 4 bits sequentially.
				// Since the size of buf is larger than the actual size by the TIFF header size,
				// it is always safe to read ahead.

				long long first_byte = pos + (bit_pos >> 3);
				const unsigned int bit_offset_in_first_byte = bit_pos & 7; // 7 = 00000111 (same as % 8)
				const unsigned int chunk = (*(unsigned int*)(buf + first_byte)) >> bit_offset_in_first_byte;

				p = (unsigned char)(chunk & 127); // 127 = 01111111; 7 bits for RLE
				bit_pos += 7; // TODO: we can remove this for further speed.
				n_pix += p;
				if (n_pix >= total_pixels) break;
				if (p == 127) continue; // this should be rare.

				// 15 = 00001111; 4 bits for symbol. See the 8+4 bit section for 0x0A
				s = (unsigned char)((chunk >> 7) & 15) ^ 0x0A;
				bit_pos += 4;
				positions[n_electron] = n_pix;
				symbols[n_electron] = s;
				n_electron++;
				n_pix++;

				p = (unsigned char)((chunk >> 11) & 127);
				bit_pos += 7;
				n_pix += p;
				if (n_pix >= total_pixels) break;
				if (p == 127) continue;

				s = (unsigned char)((chunk >> 18) & 15) ^ 0x0A;
				bit_pos += 4;
				positions[n_electron] = n_pix;
				symbols[n_electron] = s;
				n_electron++;
				n_pix++;
			}
		}
		else if (rle_bits == 7 && subpixel_bits == 2)
		{
			unsigned int bit_pos = 0; // 4 K * 4 K * 11 bit << 2 ** 32
			unsigned char p, s;

			while (true)
			{
				// Fetch 32 bits and unpack up to 2 chunks of 7 + 2 bits.
				// (Note: one cannot unpack 3 chunks because bit_offset_in_first_byte can be 7;
				//  7 + 9 + 9 + 9 = 34 > 32 so the last subpixel symbol would be lost.)
				// This is faster than unpack 7 and 2 bits sequentially.
				// Since the size of buf is larger than the actual size by the TIFF header size,
				// it is always safe to read ahead.

				long long first_byte = pos + (bit_pos >> 3);
				const unsigned int bit_offset_in_first_byte = bit_pos & 7; // 7 = 00000111 (same as % 8)
				const unsigned int chunk = (*(unsigned int*)(buf + first_byte)) >> bit_offset_in_first_byte;

				p = (unsigned char)(chunk & 127); // 127 = 01111111; 7 bits for RLE
				bit_pos += 7; // TODO: we can remove this for further speed.
				n_pix += p;
				if (n_pix >= total_pixels) break;
				if (p == 127) continue; // this should be rare.

				// 3 = 00000011; 2 bits for symbol
				// Note that we have to flip bits (see below).
				s = (unsigned char)((chunk >> 7) & 3) ^ 3;
				bit_pos += 2;
				positions[n_electron] = n_pix;
				symbols[n_electron] = s;
				n_electron++;
				n_pix++;

				p = (unsigned char)((chunk >> 9) & 127);
				bit_pos += 7;
				n_pix += p;
				if (n_pix >= total_pixels) break;
				if (p == 127) continue;

				s = (unsigned char)((chunk >> 16) & 3) ^ 3;
				bit_pos += 2;
				positions[n_electron] = n_pix;
				symbols[n_electron] = s;
				n_electron++;
				n_pix++;
			}
		}
		else if (rle_bits == 8 && subpixel_bits == 4)
		{
			// unpack every two symbols = 12 bit * 2 = 24 bit = 3 byte
			// high <- |bbbbBBBB|BBBBaaaa|AAAAAAAA| -> low
			// With SIMD intrinsics at the SSSE3 level, we can unpack 10 symbols (120 bits) simultaneously.
			unsigned char p1, p2, s1, s2;

			const long long pos_limit = frame_starts[iframe] + frame_sizes[iframe];
			// Because there is a footer, it is safe to go beyond the limit by two bytes.
			while (pos < pos_limit)
			{
				// Symbol is bit tricky: 0000YyXx, where Y and X must be flipped.
				// In other words, the bits for shifts 0, 1, 2, 3 are 10, 11, 00, 01.
				// This can be considered as 'signed 2 bit' representation of -2, -1, 0, 1.
				// For 2 bit symbols (2K EER): 000000YX and Y and X must be flipped.
				// That is, shifts 0 and 1 correspond to bits 1 and 0.
				// This is "signed 1 bit" representation of -1 and 0..
				// ref: Lingbo Yu, TFS (Email to Takanori on 10-11 May 2023)
				p1 = buf[pos];
				s1 = (buf[pos + 1] & 0x0F) ^ 0x0A; // 0x0F = 00001111, 0x0A = 00001010

				p2 = (buf[pos + 1] >> 4) | (buf[pos + 2] << 4);
				s2 = (buf[pos + 2] >> 4) ^ 0x0A;

				// Note the order. Add p before checking the size and placing a new electron.
				n_pix += p1;
				if (n_pix >= total_pixels) break;
				if (p1 < 255)
				{
					positions[n_electron] = n_pix;
					symbols[n_electron] = s1;
					n_electron++;
					n_pix++;
				}

				n_pix += p2;
				if (n_pix >= total_pixels) break;
				if (p2 < 255)
				{
					positions[n_electron] = n_pix;
					symbols[n_electron] = s2;
					n_electron++;
					n_pix++;
				}
#ifdef DEBUG_EER_DETAIL
				printf("%d: %u %u, %u %u %d\n", pos, p1, s1, p2, s2, n_pix);
#endif
				pos += 3;
			}
		}

		if (n_pix != total_pixels)
		{
			std::cerr << "WARNING: The number of pixels is not right in " + fn_movie + " frame " + integerToString(iframe + 1) + ". Probably this frame is corrupted. This frame is skipped." << std::endl;
			continue;
		}

		RCTOC(TIMING_UNPACK_RLE);

		RCTIC(TIMING_RENDER_ELECTRONS);
		if (width == EER_4K)
		{
			if (eer_upsampling == 3)
				render4K_to_16K(image, positions, symbols, n_electron);
			else if (eer_upsampling == 2)
				render4K_to_8K(image, positions, symbols, n_electron);
			else if (eer_upsampling == 1)
				render4K_to_4K(image, positions, symbols, n_electron);
			else if (eer_upsampling == -1)
				render4K_to_2K(image, positions, symbols, n_electron);
			else
				REPORT_ERROR("Invalid EER upsamle for 4K images. This must be 3, 2, 1 or -1.");
		}
		else if (width == EER_2K)
		{
			if (eer_upsampling == 2)
				render2K_to_4K(image, positions, symbols, n_electron);
			else if (eer_upsampling == 1)
				render2K_to_2K(image, positions, symbols, n_electron);
			else
				REPORT_ERROR("Invalid EER upsamle for 2K images. This must be 2 or 1.");
		}
		else
			REPORT_ERROR("Logic error: an invalid EER size at EERRenderer::renderFrames().");

		RCTOC(TIMING_RENDER_ELECTRONS);

		total_n_electron += n_electron;
#ifdef DEBUG_EER
		printf("Decoded %lld electrons / %d pixels from frame %5d.\n", n_electron, n_pix, iframe);
#endif
	}
#ifdef DEBUG_EER
	printf("Decoded %lld electrons in total.\n", total_n_electron);
#endif

#ifdef TIMING
	EERtimer.printTimes(false);
#endif

	return total_n_electron;
}

// Instantiate for Polishing
template long long EERRenderer::renderFrames<float>(int frame_start, int frame_end, MultidimArray<float> &image);
template long long EERRenderer::renderFrames<short>(int frame_start, int frame_end, MultidimArray<short> &image);
template long long EERRenderer::renderFrames<unsigned short>(int frame_start, int frame_end, MultidimArray<unsigned short> &image);
template long long EERRenderer::renderFrames<char>(int frame_start, int frame_end, MultidimArray<char> &image);
template long long EERRenderer::renderFrames<signed char>(int frame_start, int frame_end, MultidimArray<signed char> &image);
template long long EERRenderer::renderFrames<unsigned char>(int frame_start, int frame_end, MultidimArray<unsigned char> &image);
