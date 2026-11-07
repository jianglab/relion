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

#include "micrograph_handler.h"
#include "motion/polish_timing.h"
#include <src/async_file_writer.h>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <src/jaz/single_particle/stack_helper.h>
#include <src/jaz/single_particle/movie_loader.h>
#include <src/jaz/single_particle/spa_extraction.h>


using namespace gravis;

MicrographHandler::MicrographHandler()
	: nr_omp_threads(1),
	  firstFrame(0),
	  lastFrame(-1),
	  hotCutoff(-1),
	  debug(false),
	  saveMem(false),
	  ready(false),
	  keepForCombination(false),
	  corrMicFn(""),
	  eer_upsampling(-1),
	  eer_grouping(-1)
{
	decodedFrameBytes = 0;
	moviesFromMemory = moviesDecoded = moviesNotKept = mostKept = 0;
	movieGB = 0;
	aheadThreads = 0;
	aheadNice = 0;
	aheadUsed = aheadNoMemory = mostAhead = 0;
	aheadWait = 0;
}

MicrographHandler::~MicrographHandler()
{
	dropAhead();
}

void MicrographHandler::init(
		// in:
		const std::vector<MetaDataTable>& mdts,
		bool verb,
		int nr_omp_threads,
		// out:
		int& fc,
		double& dosePerFrame,
		std::string& metaFn)
{
	this->nr_omp_threads = nr_omp_threads;

	MetaDataTable corrMic;		
	ObservationModel obsModel;

	// Don't die even if conversion failed. Polishing does not use obsModel from a motion correction STAR file
	ObservationModel::loadSafely(corrMicFn, obsModel, corrMic, "micrographs", verb, false);
	mic2meta.clear();

	std::string micName, metaName;

	if (!corrMic.containsLabel(EMDL_MICROGRAPH_NAME))
	{
		REPORT_ERROR(" The corrected_micrographs STAR file does not contain rlnMicrographName label.");
	}
	if (!corrMic.containsLabel(EMDL_MICROGRAPH_METADATA_NAME))
	{
		REPORT_ERROR(" The corrected_micrographs STAR file does not contain rlnMicrographMetadata label. Did you not run motion correction from the RELION-3.0 GUI?");
	}

	for (int i = 0; i < corrMic.numberOfObjects(); i++)
	{
		corrMic.getValueToString(EMDL_MICROGRAPH_NAME, micName, i);
		corrMic.getValueToString(EMDL_MICROGRAPH_METADATA_NAME, metaName, i);
		
		// remove the pipeline job prefix
		FileName fn_pre, fn_jobnr, fn_post;
		decomposePipelineFileName(micName, fn_pre, fn_jobnr, fn_post);

		mic2meta[fn_post] = metaName;
	}

	loadInitial(mdts, verb, fc, dosePerFrame, metaFn);

	ready = true;
}

std::vector<MetaDataTable> MicrographHandler::cullMissingMovies(
		const std::vector<MetaDataTable> &mdts, int verb)
{
	if (!ready)
	{
		REPORT_ERROR("ERROR: MicrographHandler::cullMissingMovies - MicrographHandler not initialized.");
	}

	std::vector<MetaDataTable> good(0);
	std::vector<std::string> bad(0);

	const int mc = mdts.size();

	for (int m = 0; m < mc; m++)
	{
		if (isMoviePresent(mdts[m], false))
		{
			good.push_back(mdts[m]);
		}
		else
		{
			FileName fn_movie;
			mdts[m].getValueToString(EMDL_MICROGRAPH_NAME, fn_movie, 0);
			bad.push_back(fn_movie);
		}
	}

	if (verb && bad.size() > 0)
	{
		if (bad.size() == 1)
		{
			std::cerr << " - The movie for the following micrograph is missing:\n";
		}
		else
		{
			std::cerr << " - Movies for the following micrographs are missing:\n";
		}

		for (int i = 0; i < bad.size(); i++)
		{
			std::cerr << "       " << bad[i] << "\n";
		}
	}

	return good;
}

void MicrographHandler::findLowestFrameCount(
		const std::vector<MetaDataTable> &mdts, int verb)
{
	if (!ready)
	{
		REPORT_ERROR("ERROR: MicrographHandler::findLowestFrameCount - MicrographHandler not initialized.");
	}

	int fcmin = std::numeric_limits<int>::max();
	const int mc = mdts.size();

	for (int m = 0; m < mc; m++)
	{
		int fcm = determineFrameCount(mdts[m]);

		if (fcm < fcmin)
		{
			fcmin = fcm;
		}
	}

	if (lastFrame >= fcmin)
	{
		std::cout << " - Warning: some movies contain only " << fcmin
				  << " frames. Unable to load frames " << (fcmin+1)
				  << ".." << (lastFrame+1) << " ( = --last_frame).\n";
	}
	else if (verb > 0)
	{
		std::cout << " + Max. frame number available in all movies: " << fcmin << "\n";
	}

	if (lastFrame < 0 || lastFrame > fcmin-1)
	{
		lastFrame = fcmin - 1;
	}
}

std::vector<MetaDataTable> MicrographHandler::findLongEnoughMovies(
		const std::vector<MetaDataTable> &mdts, int fc, int verb)
{
	if (!ready)
	{
		REPORT_ERROR("ERROR: MicrographHandler::findLongEnoughMovies - MicrographHandler not initialized.");
	}

	std::vector<MetaDataTable> good(0);
	std::vector<std::string> bad(0);

	const int mc = mdts.size();

	for (int m = 0; m < mc; m++)
	{
		int fcm = determineFrameCount(mdts[m]);

		if (fcm < fc)
		{
			bad.push_back(getMovieFilename(mdts[m]));
		}
		else
		{
			good.push_back(mdts[m]);
		}
	}

	if (good.size() == 0)
	{
		REPORT_ERROR_STR("ERROR: Not a single movie contains the requested number of frames ("
						 << fc << ")");
	}

	if (verb && bad.size() > 0)
	{
		if (bad.size() == 1)
		{
			std::cerr << " - The following micrograph does not contain "
					  << fc << " frames. Particles in it will be ignored:\n";
		}
		else
		{
			std::cerr << " - The following micrographs do not contain "
					  << fc << " frames. Particles in them will be ignored:\n";
		}

		for (int i = 0; i < bad.size(); i++)
		{
			std::cerr << "       " << bad[i] << "\n";
		}
	}

	return good;
}

// This reads pixel sizes from a single metadata star file.
// For multi optics group scenarios, we should process only micrographs
// in the given MotionCorr STAR file. Then we can safely assume all pixel sizes are the same.
// TODO: TAKANORI: make sure in MotionCorr runner and Polish
void MicrographHandler::loadInitial(
		const std::vector<MetaDataTable>& mdts, bool verb,
		int& fc, double& dosePerFrame, std::string& metaFn)
{
	std::string mgFn;
	FileName fn_pre, fn_jobnr, fn_post;
	
	const int mc = mdts.size();

	for (int m = 0; m < mc; m++)
	{
		mdts[m].getValueToString(EMDL_MICROGRAPH_NAME, mgFn, 0);

		// remove the pipeline job prefix
		decomposePipelineFileName(mgFn, fn_pre, fn_jobnr, fn_post);

		metaFn = getMetaName(fn_post, false);
		if (metaFn != "") break;
	}

	if (metaFn == "")
	{
		REPORT_ERROR("There is no movie metadata STAR file for any micrographs!");
	}

	if (debug)
	{
		std::cout << "first movie: " << fn_post << "\n";
		std::cout << "maps to: " << metaFn << "\n";
	}

	Micrograph micrograph = Micrograph(metaFn);

	if (movie_angpix <= 0)
	{
		movie_angpix = micrograph.angpix;

		if (verb > 0)
		{
			std::cout << " + Using movie pixel size from " << metaFn << ": "
					  << movie_angpix << " A\n";
		}
	}
	else
	{
		if (verb > 0)
		{
			std::cout << " + Using movie pixel size from command line: "
					  << movie_angpix << " A\n";
		}
	}

	if (coords_angpix <= 0)
	{
		coords_angpix = micrograph.angpix * micrograph.getBinningFactor();

		if (verb > 0)
		{
			std::cout << " + Using coord. pixel size from " << metaFn << ": "
					  << coords_angpix << " A\n";
		}
	}
	else
	{
		if (verb > 0)
		{
			std::cout << " + Using coord. pixel size from command line: "
					  << coords_angpix << " A\n";
		}
	}

	dosePerFrame = micrograph.dose_per_frame;

	micrograph_size.x = micrograph.getWidth();
	micrograph_size.y = micrograph.getHeight();

	if (lastFrame >= micrograph.getNframes())
	{
		REPORT_ERROR_STR("ERROR: There are only " << micrograph.getNframes()
						 << " frames in " << metaFn << " - " << lastFrame
						 << " have been requested using the --lastFrame option.");
	}

	if (lastFrame < 0)
	{
		fc = micrograph.getNframes() - firstFrame;
	}
	else
	{
		fc = lastFrame - firstFrame + 1;
	}
}

void MicrographHandler::validatePixelSize(RFLOAT angpix) const
{
  	if (angpix < coords_angpix - 1e-9)
	{
		std::cerr << "WARNING: pixel size (--angpix) is smaller than the AutoPick pixel size (--coords_angpix)\n";

		if (coords_angpix < angpix + 0.01)
		{
			std::cerr << "        This is probably a rounding error. It is recommended to set --angpix ("
					  << angpix << ") to at least " << coords_angpix << "\n";

		}
	}

	if (angpix < movie_angpix - 1e-9)
	{
		std::cerr << "WARNING: pixel size (--angpix) is smaller than the movie pixel size (--movie_angpix)\n";

		if (movie_angpix < angpix + 0.01)
		{			std::cerr << "        This is probably a rounding error. It is recommended to set --angpix ("
					  << angpix << ") to at least " << movie_angpix << "\n";

		}
	}
}

// dst = src without copying the pixels (BufferedImage has no move assignment); src is left empty
static void takeOver(BufferedImage<float> &dst, BufferedImage<float> &src)
{
	static_cast<RawImage<float>&>(dst) = src;
	dst.dataVec.swap(src.dataVec);
	dst.data = dst.dataVec.data();
	std::vector<float>().swap(src.dataVec);
	static_cast<RawImage<float>&>(src) = RawImage<float>();
}

bool MicrographHandler::movieCacheFits(double bytes)
{
	const char *env = getenv("RELION_POLISH_MOVIE_CACHE");
	const bool off = env && std::string(env) == "0";
	const long long avail = asyncwrite::memoryAvailableBytes();
	const double held = keptBytes();

	// at most half of what would be free without the kept movies
	const bool fits = !off && (avail < 0? kept.empty() : held + bytes <= 0.5 * (avail + held));

	if (!fits)
	{
		moviesNotKept++;
		movieCacheWhy = off? "RELION_POLISH_MOVIE_CACHE=0" : "no room in half of the free memory";
	}

	movieGB = bytes / 1e9;

	return fits;
}

double MicrographHandler::keptBytes() const
{
	double b = 0;
	for (const auto& k : kept) b += (double) k.second->frames.xdim * k.second->frames.ydim * k.second->frames.zdim * sizeof(float);
	return b;
}

void MicrographHandler::release(const std::string& metaFn)
{
	for (auto it = kept.begin(); it != kept.end(); ++it)
	{
		if (it->first == metaFn)
		{
			kept.erase(it);
			return;
		}
	}
}

std::string MicrographHandler::movieCacheReport() const
{
	if (moviesFromMemory + moviesNotKept == 0) return "";

	std::ostringstream os;
	os << " + Movie cache: " << moviesFromMemory << " movie loads needed no decoding (" << moviesDecoded << " decoded; up to "
	   << mostKept << " movies of " << std::setprecision(3) << movieGB << " GB kept at once";
	if (moviesNotKept > 0) os << "; " << moviesNotKept << " not kept: " << movieCacheWhy;
	os << "). RELION_POLISH_MOVIE_CACHE=0 turns it off.";
	return os.str();
}

std::shared_ptr<Image<RFLOAT>> MicrographHandler::gainReference(
		const std::string &gainFn, const std::string &movieFn, bool isEER, int upsampling)
{
	std::lock_guard<std::mutex> guard(gainMutex);

	const std::string key = isEER? gainFn + "#eer" + integerToString(upsampling) : gainFn;
	std::map<std::string, std::shared_ptr<Image<RFLOAT>>>::iterator it = gains.find(key);
	if (it != gains.end()) return it->second;

	// (e.msg starts with "ERROR: \n")
	auto gainError = [&](const RelionError &e)
	{
		return GainReferenceError("cannot read the gain reference " + gainFn + ": " + e.msg.substr(e.msg.find('\n') + 1), __FILE__, __LINE__);
	};

	std::shared_ptr<Image<RFLOAT>> gain(new Image<RFLOAT>());

	if (isEER) // TODO: Takanori: Remove this once we updated RelionCor
	{
		EERRenderer renderer;
		renderer.read(movieFn, upsampling);
		try { renderer.loadEERGain(gainFn, (*gain)()); }
		catch (RelionError &e) { throw gainError(e); }
	}
	else
	{
		try { gain->read(gainFn); }
		catch (RelionError &e) { throw gainError(e); }
	}

	gains[key] = gain;
	return gain;
}

std::shared_ptr<MicrographHandler::Decoded> MicrographHandler::decodeMovie(
		const std::string &metaFn, int frame0, int fc, int threads)
{
	polishtiming::Scope ts("load: decode frames");
	const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

	Micrograph micrograph = Micrograph(metaFn);

	const FileName movieFn = micrograph.getMovieFilename();
	const std::string gainFn = micrograph.getGainFilename();
	const bool mgHasGain = (gainFn != "");
	const bool hasDefect = (mgHasGain || micrograph.fnDefect != "" || micrograph.hotpixelX.size() != 0);
	const bool isEER = EERRenderer::isEER(movieFn);

	if (debug)
	{
		std::cout << "loading: " << "\n";
		std::cout << "-> meta: " << metaFn << "\n";
		std::cout << "-> data: " << movieFn << "\n";
		std::cout << "-> gain: " << gainFn << "\n";
		std::cout << "-> mask: " << micrograph.fnDefect << "\n";
		std::cout << "-> nhot: " << micrograph.hotpixelX.size() << "\n";
		std::cout << "-> hasdefect: " << (hasDefect ? 1 : 0) << std::endl;
	}

	MultidimArray<bool> defectMask;

	if (hasDefect)
	{
		micrograph.fillDefectAndHotpixels(defectMask);
	}

	std::shared_ptr<Image<RFLOAT>> gain;

	if (mgHasGain)
	{
		gain = gainReference(gainFn, movieFn, isEER,
			eer_upsampling < 0 && isEER? micrograph.getEERUpsampling() : eer_upsampling);

		// Mask pixels with zero gain
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(defectMask)
			if (DIRECT_MULTIDIM_ELEM((*gain)(), n) == 0)
				DIRECT_MULTIDIM_ELEM(defectMask, n) = true;
	}

	RawImage<RFLOAT> gainRef_new = mgHasGain? RawImage<RFLOAT>(*gain) : RawImage<RFLOAT>();
	RawImage<bool> defectMask_new(defectMask);

	RawImage<RFLOAT>* gainRefToUse = mgHasGain? &gainRef_new : 0;
	RawImage<bool>* defectMaskToUse = hasDefect? &defectMask_new : 0;

	BufferedImage<float> decoded = isEER?
		MovieLoader::readEER<float>(
			movieFn, gainRefToUse, defectMaskToUse,
			frame0, fc,
			eer_upsampling < 0? micrograph.getEERUpsampling() : eer_upsampling,
			eer_grouping < 0? micrograph.getEERGrouping() : eer_grouping,
			threads) :
		MovieLoader::readDense<float>(
			movieFn, gainRefToUse, defectMaskToUse,
			frame0, fc,
			hotCutoff,
			threads);

	std::shared_ptr<Decoded> out(new Decoded());
	takeOver(out->frames, decoded);
	out->frame0 = frame0;
	out->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

	return out;
}

void MicrographHandler::expectNext(const MetaDataTable* mdt, const MetaDataTable* afterIt)
{
	nextMetaFns.clear();

	for (const MetaDataTable* m : {mdt, afterIt})
	{
		if (m && ready && m->numberOfObjects() > 0) nextMetaFns.push_back(getMicrographMetadataFilename(*m, false));
	}
}

void MicrographHandler::startReadAhead(const MetaDataTable& mdt, const MetaDataTable* afterIt)
{
	expectNext(&mdt, afterIt);
	prefetch();
}

void MicrographHandler::prefetch()
{
	std::vector<std::string> wanted;
	wanted.swap(nextMetaFns);

	static const char *off = getenv("RELION_POLISH_READ_AHEAD");
	if (off && std::string(off) == "0") return;

	wanted.erase(std::remove_if(wanted.begin(), wanted.end(), [this](const std::string& fn)
	{
		for (const auto& k : kept) if (k.first == fn) return true;
		return false;
	}), wanted.end());

	// Nothing new expected (e.g. the next frame of the same movie): those on their way stay
	if (wanted.empty()) return;

	// Movies on their way that are no longer expected
	for (auto it = ahead.begin(); it != ahead.end();)
	{
		if (std::find(wanted.begin(), wanted.end(), it->metaFn) != wanted.end()) { ++it; continue; }

		try { it->movie.get(); } catch (...) {}  // a movie that was not asked for after all: its errors do not matter
		it = ahead.erase(it);
	}

	const int allFc = lastFrame - firstFrame + 1;
	if (allFc < 1) return;

	for (const std::string& metaFn : wanted)
	{
		if (isAhead(metaFn)) continue;

		// Before the first movie has been decoded, its size comes from the motion record
		double frameBytes = decodedFrameBytes;
		if (frameBytes <= 0)
		{
			try
			{
				Micrograph micrograph(metaFn);
				frameBytes = (double) micrograph.getWidth() * micrograph.getHeight() * sizeof(float);
				const int mgUp = micrograph.getEERUpsampling();
				if (EERRenderer::isEER(micrograph.getMovieFilename()) && eer_upsampling > 0 && mgUp > 0)
					frameBytes *= (double) (eer_upsampling * eer_upsampling) / (mgUp * mgUp);
			}
			catch (...) { return; }  // the movie is then read when it is needed, which reports the error
			if (frameBytes <= 0) return;
		}

		// All movies on their way in half of the free memory (those already decoded are counted twice: on the safe side)
		const long long avail = asyncwrite::memoryAvailableBytes();
		if (avail >= 0 && frameBytes * allFc * (ahead.size() + 1) > 0.5 * avail)
		{
			if (ahead.empty()) aheadNoMemory++;
			return;
		}

		// As many readers as computing threads, but at the lowest CPU priority (inherited by the OpenMP threads the
		// reader starts): they use only the CPUs the computation leaves idle, and all of them while it waits.
		// (Fewer readers at normal priority, adapted to the waits, made the computation wait twice as long.)
		if (aheadThreads <= 0)
		{
			const char *fixed = getenv("RELION_POLISH_READ_AHEAD_THREADS");
			const char *nice = getenv("RELION_POLISH_READ_AHEAD_NICE");
			aheadThreads = fixed && atoi(fixed) > 0? atoi(fixed) : std::max(1, nr_omp_threads);
			aheadNice = nice? std::max(0, std::min(19, atoi(nice))) : 19;
		}

		const int first = firstFrame, threads = aheadThreads, niceness = aheadNice;
		// one after the other, so that the movie needed first gets the idle CPUs first
		const std::shared_future<std::shared_ptr<Decoded>> before =
			ahead.empty()? std::shared_future<std::shared_ptr<Decoded>>() : ahead.back().movie;

		Ahead a;
		a.metaFn = metaFn;
		a.start = std::chrono::steady_clock::now();
		a.movie = std::async(std::launch::async, [this, metaFn, first, allFc, threads, niceness, before]()
		{
			if (niceness > 0) setpriority(PRIO_PROCESS, (id_t) syscall(SYS_gettid), niceness);
			if (before.valid()) before.wait();  // (its errors are its own)

			return decodeMovie(metaFn, first, allFc, threads);
		}).share();

		ahead.push_back(a);
		mostAhead = std::max(mostAhead, (int) ahead.size());
	}
}

bool MicrographHandler::isAhead(const std::string &metaFn) const
{
	for (const Ahead& a : ahead) if (a.metaFn == metaFn) return true;
	return false;
}

std::shared_ptr<MicrographHandler::Decoded> MicrographHandler::takeAhead(const std::string &metaFn)
{
	auto it = std::find_if(ahead.begin(), ahead.end(), [&](const Ahead& a) { return a.metaFn == metaFn; });
	const Ahead a = *it;
	ahead.erase(it);

	const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	const double busy = std::chrono::duration<double>(t0 - a.start).count();

	a.movie.wait();

	const double wait = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

	std::shared_ptr<Decoded> d = a.movie.get();  // throws what the reader threw

	aheadUsed++;
	aheadWait += wait;

	if (polishtiming::enabled())
	{
		fprintf(stderr, " Polish read-ahead: movie %d decoded in %.2f s, %.2f s after it was queued; waited %.2f s\n",
				aheadUsed, d->seconds, busy, wait);
	}

	return d;
}

void MicrographHandler::dropAhead()
{
	for (Ahead& a : ahead)
	{
		try { a.movie.get(); } catch (...) {}  // movies not asked for after all: their errors do not matter
	}

	ahead.clear();
}

std::string MicrographHandler::readAheadReport() const
{
	if (aheadUsed + aheadNoMemory == 0) return "";

	std::ostringstream os;
	os << " + Read-ahead: " << aheadUsed << " movies were decoded in the background while earlier ones were processed"
	   << " (up to " << mostAhead << " ahead, one after the other; "
	   << aheadThreads << (aheadThreads == 1? " reader thread" : " reader threads")
	   << (aheadNice > 0? " at nice " + std::to_string(aheadNice) + ", so they mostly use CPUs the computation leaves idle" : "")
	   << "; the computation waited " << std::setprecision(3) << aheadWait << " s in total)";
	if (aheadNoMemory > 0) os << "; " << aheadNoMemory << " were not, for lack of memory";
	os << ". RELION_POLISH_READ_AHEAD=0 turns it off, RELION_POLISH_READ_AHEAD_THREADS sets the threads,"
	   << " RELION_POLISH_READ_AHEAD_NICE their nice value (0: normal priority).";
	return os.str();
}

std::shared_ptr<MicrographHandler::Decoded> MicrographHandler::findKept(
		const std::string& metaFn, int frame0, int fc, bool combining)
{
	// The movie whose frames were being combined is done once another one is asked for
	if (combiningMetaFn != "" && combiningMetaFn != metaFn)
	{
		release(combiningMetaFn);
		combiningMetaFn = "";
	}

	for (auto it = kept.begin(); it != kept.end(); ++it)
	{
		const Decoded& d = *it->second;

		if (it->first == metaFn && frame0 >= d.frame0 && frame0 + fc <= d.frame0 + d.frames.zdim)
		{
			std::shared_ptr<Decoded> out = it->second;

			if (combining)
			{
				// those kept before it will not be combined any more (done, skipped, or another MPI rank's)
				if (combiningMetaFn != metaFn) moviesFromMemory++;
				kept.erase(kept.begin(), it);
				combiningMetaFn = metaFn;
			}

			return out;
		}
	}

	return std::shared_ptr<Decoded>();
}

bool MicrographHandler::combineFromMemory(
		const MetaDataTable& mdt, int s, double angpix, int threads,
		const std::vector<std::vector<gravis::d2Vector>>* offsets_in,
		std::vector<std::vector<gravis::d2Vector>>* offsets_out,
		double data_angpix,
		const std::function<void(int p, int f, Image<Complex>& frame)>& consume)
{
	if (!ready)
	{
		REPORT_ERROR("ERROR: MicrographHandler::combineFromMemory - MicrographHandler not initialized.");
	}

	const std::string metaFn = getMicrographMetadataFilename(mdt, true);
	const int allFc = lastFrame - firstFrame + 1;

	std::shared_ptr<Decoded> movieData = findKept(metaFn, firstFrame, allFc, true);

	if (!movieData) return false;

	RawImage<float> frames = movieData->frames.getSlabRef(firstFrame - movieData->frame0, allFc);

	prefetch();

	SpaExtraction::consumeMovieStackFS(
		mdt, frames, s,
		angpix, coords_angpix, movie_angpix, data_angpix,
		offsets_in, offsets_out,
		threads, consume);

	return true;
}

std::vector<std::vector<Image<Complex>>> MicrographHandler::loadMovie(
		const MetaDataTable &mdt, int s,
		double angpix, std::vector<ParFourierTransformer>& fts,
		const std::vector<std::vector<gravis::d2Vector>>* offsets_in,
		std::vector<std::vector<gravis::d2Vector>>* offsets_out,
		double data_angpix,
		int single_frame_relative_index)
{
	if (!ready)
	{
		REPORT_ERROR("ERROR: MicrographHandler::loadMovie - MicrographHandler not initialized.");
	}

	const int nr_omp_threads = fts.size();
	const std::string metaFn = getMicrographMetadataFilename(mdt, true);

	const bool returnSingleFrame = single_frame_relative_index >= 0;
	
	if (returnSingleFrame && offsets_in != 0 && (*offsets_in)[0].size() != 1)
	{
		REPORT_ERROR_STR("MicrographHandler::loadMovie: attempting to read one single frame "
						 << "while the initial trajectories contain more than one position");
	}

	const int allFc = lastFrame - firstFrame + 1;
	// (a single frame is counted from firstFrame, like the frames of the whole movie and the tracks)
	const int frame0 = returnSingleFrame? firstFrame + single_frame_relative_index : firstFrame;
	const int fc = returnSingleFrame? 1 : allFc;

	std::shared_ptr<Decoded> movieData = findKept(metaFn, frame0, fc, returnSingleFrame);

	if (!movieData)
	{
		const bool wantKeep = !saveMem && keepForCombination;

		if (isAhead(metaFn))
		{
			polishtiming::Scope tw("load: wait for read-ahead");
			movieData = takeAhead(metaFn);  // the whole movie
		}
		else
		{
			dropAhead();

			// Frame combination asks for one frame at a time: decode the whole movie at once, in parallel, and keep it
			const bool keepWhole = returnSingleFrame && wantKeep
			                       && decodedFrameBytes > 0 && movieCacheFits(decodedFrameBytes * allFc);

			movieData = keepWhole?
				decodeMovie(metaFn, firstFrame, allFc, nr_omp_threads) :
				decodeMovie(metaFn, frame0, fc, nr_omp_threads);
		}

		moviesDecoded++;

		const int dc = movieData->frames.zdim;
		decodedFrameBytes = (double) movieData->frames.xdim * movieData->frames.ydim * sizeof(float);

		// Keep it for the requests that follow (a whole movie decoded for one frame has already been checked)
		if (wantKeep && dc == allFc && (dc > fc || movieCacheFits(decodedFrameBytes * dc)))
		{
			kept.push_back(std::make_pair(metaFn, movieData));
			if (returnSingleFrame) combiningMetaFn = metaFn;
			mostKept = std::max(mostKept, (int) kept.size());
		}
	}

	RawImage<float> frames = movieData->frames.getSlabRef(frame0 - movieData->frame0, fc);

	// This movie is in memory: start reading the ones after it
	prefetch();

	std::vector<std::vector<Image<Complex>>> movie;
	{
		polishtiming::Scope te("load: extract + FFT");
		movie = SpaExtraction::extractMovieStackFS(
			mdt, frames, s,
			angpix, coords_angpix, movie_angpix, data_angpix,
			offsets_in, offsets_out, 
			nr_omp_threads);
	}
	
	const int pc = movie.size();

	if (!returnSingleFrame)
	{
		polishtiming::Scope tn("load: normalise");
		#pragma omp parallel for num_threads(nr_omp_threads)
		for (int p = 0; p < pc; p++)
		{
			RFLOAT scale2 = StackHelper::computePower(movie[p], false);
			
			for (int f = 0; f < fc; f++)
			{
				// NOTE: sqrt(fc) is a legacy scaling factor.
				// It probably shouldn't be there.
				//                   --JZ
				RawImage<Complex>(movie[p][f]) /= s * sqrt(scale2 / fc); 
			}
		}
	}

	return movie;
}

void MicrographHandler::loadInitialTracks(
		const MetaDataTable &mdt, double angpix,
		const std::vector<d2Vector>& pos,
		std::vector<std::vector<d2Vector>>& tracks_out,
		bool unregGlob, 
		std::vector<d2Vector>& globalComponent_out)
{	
	Micrograph micrograph = loadMicrographMetadata(mdt, true);
	
	const int fc0 = micrograph.getNframes();
	int fc;

	if (lastFrame >= 0)
	{
		fc = lastFrame - firstFrame + 1;
	}
	else
	{
		fc = fc0 - firstFrame;
	}

	const int pc = pos.size();

	const d2Vector inputScale(
				coords_angpix / (movie_angpix * micrograph.getWidth()),
				coords_angpix / (movie_angpix * micrograph.getHeight()));

	const double outputScale = movie_angpix / angpix;

	globalComponent_out = std::vector<d2Vector>(fc, d2Vector(0,0));

	if (unregGlob)
	{
		for (int f = 0; f < fc; f++)
		{
			RFLOAT sx, sy;
			micrograph.getShiftAt(firstFrame + f + 1, 0, 0, sx, sy, false);

			globalComponent_out[f] = -outputScale * d2Vector(sx, sy);
		}
	}

	tracks_out.resize(pc);

	for (int p = 0; p < pc; p++)
	{
		tracks_out[p] = std::vector<d2Vector>(fc);

		for (int f = 0; f < fc; f++)
		{
			d2Vector in(inputScale.x * pos[p].x - 0.5,
						inputScale.y * pos[p].y - 0.5);

			RFLOAT sx, sy;
			micrograph.getShiftAt(firstFrame + f + 1, in.x, in.y, sx, sy, true);

			tracks_out[p][f] = -outputScale * d2Vector(sx,sy) - globalComponent_out[f];
		}
	}
}

std::string MicrographHandler::getMetaName(std::string micName, bool die_on_error)
{
	std::map<std::string, std::string>::iterator it = mic2meta.find(micName);

	if (it == mic2meta.end())
	{
		if (die_on_error)
		{
			REPORT_ERROR("ERROR: MicrographHandler::getMetaName: no metadata star-file for "
			              +micName+" found in "+corrMicFn+".");
		}
		else
		{
			return "";
		}
	}
	else
	{
		return it->second;
	}
}

std::string MicrographHandler::getMicrographMetadataFilename(
		const MetaDataTable &mdt, 
		bool die_on_error)
{
	std::string mgFn;
	// all entries in the MetaDataTable mdt (particles) have the same micrograph name
	mdt.getValueToString(EMDL_MICROGRAPH_NAME, mgFn, 0);
	
	FileName fn_pre, fn_jobnr, fn_post;
	decomposePipelineFileName(mgFn, fn_pre, fn_jobnr, fn_post);	
	
	return getMetaName(fn_post, die_on_error);
}

Micrograph MicrographHandler::loadMicrographMetadata(
		const MetaDataTable &mdt,
		bool die_on_error)
{
	return Micrograph(getMicrographMetadataFilename(mdt, die_on_error));
}

// TODO: TAKANORI: This needs to handle changes in EER grouping
int MicrographHandler::determineFrameCount(const MetaDataTable &mdt)
{
	Micrograph micrograph = loadMicrographMetadata(mdt, true); // TODO: TAKANORI: shouldn't read hot pixels

	if (!exists(micrograph.getMovieFilename()))
	{
		return -1;
	}
	else
	{
		return micrograph.getNframes();
	}
}

bool MicrographHandler::isMoviePresent(const MetaDataTable &mdt, bool die_on_error)
{	
	std::string metaFn = getMicrographMetadataFilename(mdt, die_on_error);
	
	if (exists(metaFn))
	{
		Micrograph micrograph = Micrograph(metaFn);
		
		return exists(micrograph.getMovieFilename());
	}
	else
	{
		return false;
	}
}

std::string MicrographHandler::getMovieFilename(const MetaDataTable& mdt, bool die_on_error)
{
	std::string metaFn = getMicrographMetadataFilename(mdt, die_on_error);

	if (exists(metaFn))
	{
		Micrograph micrograph = Micrograph(metaFn);

		return micrograph.getMovieFilename();
	}
	else
	{
		// Does this make sense!?
		return metaFn;
	}
}
