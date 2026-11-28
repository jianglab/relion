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

#ifndef FRAME_RECOMBINER_H
#define FRAME_RECOMBINER_H

#include <src/image.h>
#include <src/virtual_particles.h>
#include <src/jaz/image/buffered_image.h>
#include <functional>
#include <map>
#include <utility>
#include <vector>
#include <string>

class IOParser;
class ObservationModel;
class MicrographHandler;
class ReferenceMap;

class FrameRecombiner
{
	public:

		FrameRecombiner();

		
		void read(IOParser& parser, int argc, char *argv[]);

		void init(const std::vector<MetaDataTable>& allMdts,
		          int verb, int s_ref, int fc, 
		          double maxFreq, double angpix_ref,
		          int nr_omp_threads,
		          std::string outPath, bool debug,
		          ReferenceMap* reference,
		          ObservationModel* obsModel,
		          MicrographHandler* micrographHandler);
		
		void process_old(const std::vector<MetaDataTable>& mdts, long g_start, long g_end);
		// afterLast: the micrograph whose movie will be loaded after g_end (for read-ahead), or 0
		void process(const std::vector<MetaDataTable>& mdts, long g_start, long g_end,
		             const MetaDataTable* afterLast = 0);
		// The particles of one micrograph (mdt: their coordinates) as Polish writes them: the frames, each shifted
		// by its track (priorShift, pixels at angpix) and weighted, summed; then times ctf(p, image) where that
		// returns true, normalised, transformed back and cut to the size of `out` (one slice per particle).
		// Polish and virtual polished particles are both made here, so that they are the same bit for bit.
		static void recombineMicrograph(
				const MetaDataTable& mdt, MicrographHandler& micrographHandler,
				const std::vector<std::vector<gravis::d2Vector>>& priorShift,
				int s, double angpix, double data_angpix,
				const std::vector<Image<RFLOAT>>& weights, int threads,
				const std::function<bool(int p, MultidimArray<RFLOAT>& ctf)>& ctf,
				RawImage<RFLOAT> out);

		// The two halves of recombineMicrograph: the weighted sum of the shifted frames of each particle (Fourier
		// space, one slice per particle; all that needs the movie), and the particles made from those sums.
		static BufferedImage<Complex> sumMicrograph(
				const MetaDataTable& mdt, MicrographHandler& micrographHandler,
				const std::vector<std::vector<gravis::d2Vector>>& priorShift,
				int s, double angpix, double data_angpix,
				const std::vector<Image<RFLOAT>>& weights, int threads);
		// Reusable per-thread FFTW c2r plan & scratch workspace for finishParticle (keyed by window size)
		struct FinishWorkspace
		{
			struct PlanSlot
			{
				BufferedImage<Complex> scratch;
				BufferedImage<RFLOAT> real;
				void* plan = 0;
			};

			int s = 0;
			std::map<int, PlanSlot> plans;
			PlanSlot* active = 0;
			MultidimArray<RFLOAT> ctfImg;

			FinishWorkspace() {}
			~FinishWorkspace();
			FinishWorkspace(const FinishWorkspace&) = delete;
			FinishWorkspace& operator=(const FinishWorkspace&) = delete;

			void prepare(int new_s);
		};

		static void finishParticles(
				BufferedImage<Complex>& sumStack, int threads,
				const std::function<bool(int p, MultidimArray<RFLOAT>& ctf)>& ctf,
				RawImage<RFLOAT> out);
		// One particle from its sum (changed in place), multiplied with `ctf` if given, into `out` (cut to its size)
		static void finishParticle(
				BufferedImage<Complex>& sum, MultidimArray<RFLOAT>* ctf, RawImage<RFLOAT> out);
		static void finishParticle(
				BufferedImage<Complex>& sum, MultidimArray<RFLOAT>* ctf, RawImage<RFLOAT> out,
				FinishWorkspace& ws);

		bool writesVirtual() const { return write_virtual; }
		// Virtual particles are only described, but their frame sums are cached if a cache is in use
		bool readsMovies() const { return !write_virtual || vparticles::polishedSumsWanted(); }

		void process_new(const std::vector<MetaDataTable>& mdts, long g_start, long g_end);

		bool doingRecombination();
		
		// has a max. freq. parameter been supplied?
		bool outerFreqKnown();

		std::vector<bool> findUnfinishedJobs(const std::vector<MetaDataTable>& mdts,
		                                              std::string path);
		
		double getOutputPixelSize(int opticsGroup);
		int getOutputBoxSize(int opticsGroup);
		std::string getOutputSuffix();
		bool isCtfMultiplied(int opticsGroup);
		
		int getVerbosity();
		void setVerbosity(int v);

		
	protected:
		// read from cmd. line:
		bool doCombineFrames, bfac_diag, do_ctf_multiply, do_recenter, write_float16, write_virtual;
		int k0, k1, box_arg, scale_arg, crop_arg;
		std::vector<int> crop_auto;   // per optics group: the particle box, when --ctf_multiply chose the window itself
		double k0a, k1a, recenter_x, recenter_y, recenter_z;
		std::string bfacFn, suffix;

		// set at init:
		int s_ref, sh_ref, fc;
		std::vector<int> s_mov, s_out, sh_out;
		std::vector<RFLOAT> data_angpix;
		int verb, nr_omp_threads;
		std::string outPath;
		bool debug;
		double angpix_ref, maxFreq;
		std::vector<double> angpix_out;

		ReferenceMap* reference;
		ObservationModel* obsModel;
		MicrographHandler* micrographHandler;

		// computed by weightsFromFCC or weightsFromBfacs:
		std::vector<std::vector<Image<RFLOAT>>> freqWeights;
		// and the per-frame (sigma, k) they were computed from, per optics group (for virtual particles)
		std::vector<std::vector<gravis::d2Vector>> bkFactors;
		// cached per-(ogmg, window_s) damage weights for per-particle premultiplied windows
		std::map<std::pair<int, int>, std::vector<Image<RFLOAT>>> windowWeightsCache_;

		std::vector<Image<RFLOAT>> weightsFromFCC(const std::vector<MetaDataTable>& allMdts,
		                                          int s, double angpix, std::string og_name);
		
		std::vector<Image<RFLOAT>> weightsFromBfacs(const std::vector<MetaDataTable>& allMdts,
		                                            int s, double angpix);

		bool isJobFinished(std::string filenameRoot);
		
		void recenterParticles(
				MetaDataTable& mdtOut,
				RFLOAT ref_angpix,
				RFLOAT coords_angpix);
};

#endif
