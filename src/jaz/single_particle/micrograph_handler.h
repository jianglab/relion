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

#ifndef MICROGRAPH_HANDLER_H
#define MICROGRAPH_HANDLER_H

#include <string>
#include <map>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>

#include <src/jaz/gravis/t2Vector.h>
#include <src/jaz/single_particle/parallel_ft.h>

#include <src/micrograph_model.h>
#include <deque>
#include <functional>
#include <src/image.h>
#include <src/jaz/image/buffered_image.h>

// A gain reference that cannot be read: fatal, unlike a single unreadable movie, which callers skip
class GainReferenceError : public RelionError
{
	public:
		GainReferenceError(const std::string& what, const std::string& file, long line)
		: RelionError(what, file, line) {}
};

class MicrographHandler
{
	public:

	MicrographHandler();
	~MicrographHandler();
	

		int nr_omp_threads, firstFrame, lastFrame;
		double movie_angpix, coords_angpix, data_angpix, hotCutoff;
		int eer_upsampling, eer_grouping;
	
		bool debug, saveMem, ready;

		// With keepForCombination (Polish combines the frames after estimating the motion), movies decoded for
		// motion estimation are kept for their frame combination, as many as fit in half of the memory that would be
		// free without them: never one in place of another, so the first ones of a pass are kept and then combined in
		// the same order. A movie is released once a different one is requested for frame combination or estimation
		// after its own combination; one asked for frame combination that is not kept is decoded whole (in parallel)
		// for its frames. Never with --sbs; RELION_POLISH_MOVIE_CACHE=0 turns it off.
		bool keepForCombination;

		// One line on how often decoded movies were kept, or "" if never asked
		std::string movieCacheReport() const;

		// Read-ahead: the micrographs that will be loaded after the next one, and after that (0: none). Once
		// loadMovie() has the next movie, it starts decoding these in the background, one after the other, so that
		// their own requests do not wait (two ahead, so that a micrograph with few particles leaves enough time).
		// As many reader threads as computing threads, at nice 19, so they use the CPUs the computation leaves
		// idle; only as many movies as fit in half of the free memory.
		// RELION_POLISH_READ_AHEAD=0 turns it off, RELION_POLISH_READ_AHEAD_THREADS sets the threads,
		// RELION_POLISH_READ_AHEAD_NICE their nice value (0: normal priority).
		void expectNext(const MetaDataTable* mdt, const MetaDataTable* afterIt = 0);
		// Start decoding these micrographs' movies in the background now (e.g. while the reference is loaded)
		void startReadAhead(const MetaDataTable& mdt, const MetaDataTable* afterIt = 0);

		// One line on what read-ahead did, or "" if it never ran
		std::string readAheadReport() const;
	
		std::string corrMicFn;
	
		gravis::t2Vector<int> micrograph_size;


	// initialise corrected/uncorrected micrograph dictionary, then
	// load first movie (or read corrected_micrographs.star) to obtain:
	//   fc, micrograph_xsize, micrograph_ysize, motionEstimator.dosePerFrame
	void init(
		// in:
		const std::vector<MetaDataTable>& mdts,
		bool verb,
		int nr_omp_threads,
		// out:
		int& fc,
		double& dosePerFrame,
		std::string& metaFn);

	void validatePixelSize(RFLOAT angpix) const;

	// remove movies from the list for which either the meta-star or the movie itself is missing
	std::vector<MetaDataTable> cullMissingMovies(
		const std::vector<MetaDataTable>& mdts, int verb);

	// find the greatest number of frames available in all micrographs
	void findLowestFrameCount(
		const std::vector<MetaDataTable>& mdts, int verb);

	// find all movies of sufficient length
	std::vector<MetaDataTable> findLongEnoughMovies(
		const std::vector<MetaDataTable>& mdts,
		int fc, int verb);

	// load a movie and extract all particles
	// returns a per-particle vector of per-frame images of size (s/2+1) x s
	std::vector<std::vector<Image<Complex>>> loadMovie(
		const MetaDataTable& mdt, int s, double angpix, 
		std::vector<ParFourierTransformer>& fts,
		const std::vector<std::vector<gravis::d2Vector>>* offsets_in = 0,
		std::vector<std::vector<gravis::d2Vector>>* offsets_out = 0,
		double data_angpix = -1,
		int single_frame_relative_index = -1); // if this isn't negative, return a single frame for all particles

	// Frame combination straight from the decoded movie when it is kept in memory: consume(p, f, frame) for every
	// particle (over threads) and frame (in order), without keeping the extracted frames; (*offsets_out)[p][f] is
	// set first. Returns false, doing nothing, if the movie is not kept: then loadMovie() one frame at a time.
	bool combineFromMemory(
		const MetaDataTable& mdt, int s, double angpix, int threads,
		const std::vector<std::vector<gravis::d2Vector>>* offsets_in,
		std::vector<std::vector<gravis::d2Vector>>* offsets_out,
		double data_angpix,
		const std::function<void(int p, int f, Image<Complex>& frame)>& consume);

	/* Write the initial tracks of particles at 'pos' into 'tracks_out' 
	   (by interpolating  the polynomial motionCor2 model).
	   If 'unregGlob' is set, also write the global component of motion into 'globalComponent_out'.*/
	void loadInitialTracks(
		const MetaDataTable& mdt, 
		double angpix, 
		const std::vector<gravis::d2Vector>& pos,
		std::vector<std::vector<gravis::d2Vector>>& tracks_out,
		bool unregGlob, 
		std::vector<gravis::d2Vector>& globalComponent_out);

	std::string getMicrographMetadataFilename(
			const MetaDataTable& mdt,
			bool die_on_error);

	Micrograph loadMicrographMetadata(
			const MetaDataTable& mdt,
			bool die_on_error);

	std::string getMovieFilename(
			const MetaDataTable& mdt, bool die_on_error = true);
	
	std::string getMetaName(
			std::string micName, bool die_on_error = true);

	protected:

		struct Decoded
		{
			BufferedImage<float> frames;
			int frame0;
			double seconds;
		};

		// Gain references by file (and EER upsampling), shared with the background reader
		std::mutex gainMutex;
		std::map<std::string, std::shared_ptr<Image<RFLOAT>>> gains;

		// Decoded movies kept for frame combination, in the order they were decoded
		std::deque<std::pair<std::string, std::shared_ptr<Decoded>>> kept;
		std::string combiningMetaFn, movieCacheWhy;
		int moviesFromMemory, moviesDecoded, moviesNotKept, mostKept;
		double decodedFrameBytes, movieGB;

		// Whether one more decoded movie of this size may be kept
		bool movieCacheFits(double bytes);
		double keptBytes() const;
		void release(const std::string& metaFn);
		// The kept movie with these frames, or none; for frame combination, those kept before it are dropped
		std::shared_ptr<Decoded> findKept(const std::string& metaFn, int frame0, int fc, bool combining);

		// Movies being decoded ahead, in the order they will be asked for; each waits for the one before it
		struct Ahead
		{
			std::string metaFn;
			std::shared_future<std::shared_ptr<Decoded>> movie;
			std::chrono::steady_clock::time_point start;
		};
		std::deque<Ahead> ahead;
		std::vector<std::string> nextMetaFns;
		int aheadThreads, aheadUsed, aheadNoMemory, mostAhead;
		int aheadNice;
		double aheadWait;

		std::shared_ptr<Image<RFLOAT>> gainReference(
				const std::string& gainFn, const std::string& movieFn, bool isEER, int upsampling);

		// Frames [frame0, frame0 + fc) of a movie, gain-corrected and with defects filled; reads only
		// settings and the gain references, so it also runs on the background reader
		std::shared_ptr<Decoded> decodeMovie(const std::string& metaFn, int frame0, int fc, int threads);

		// Starts decoding the movies of nextMetaFns that are neither in memory nor on their way, and clears it
		void prefetch();
		bool isAhead(const std::string& metaFn) const;
		std::shared_ptr<Decoded> takeAhead(const std::string& metaFn);
		void dropAhead();
	
		std::map<std::string, std::string> mic2meta;

	void loadInitial(
			const std::vector<MetaDataTable>& mdts, bool verb,
			int& fc, double& dosePerFrame, std::string& metaFn);

	int determineFrameCount(
			const MetaDataTable& mdt);

	bool isMoviePresent(
			const MetaDataTable& mdt, bool die_on_error = true);
};

#endif
