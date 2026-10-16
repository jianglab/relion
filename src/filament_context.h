/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef FILAMENT_CONTEXT_H_
#define FILAMENT_CONTEXT_H_

/*
 * Filament context for classifying helical segments.
 *
 * Segments of one filament are not independent: they share the polymorph, and
 * along a twisted filament the 2D classes (views) follow each other in a fixed
 * order. Here each segment gets a prior over the classes from the other
 * segments of its filament, computed from their per-class log-likelihoods of
 * the previous iteration. The segment's own likelihood is left out, so the
 * prior never counts its own data. Modes (RELION_FILAMENT_CONTEXT):
 *
 *   chain     a Markov chain along the filament: forward-backward with a
 *             class-to-class transition matrix learned from the data
 *             (Baum-Welch, symmetrised because filament polarity is unknown).
 *   filament  one class distribution per filament (Dirichlet-multinomial):
 *             the prior is the tempered class posteriors of the other segments.
 *
 * Overlapping boxes share pixels, so neighbouring likelihoods repeat the same
 * noise. Each likelihood is raised to the power `temper` (by default the step
 * between segments divided by the box length) so that every pixel of the
 * filament counts about once. Segments closer than `gap` steps are left out of
 * a segment's prior (gap = 1 leaves out only the segment itself).
 */

#include <string>
#include <vector>

class MetaDataTable;

namespace filctx
{

enum Mode { OFF = 0, CHAIN, FILAMENT };

struct Options
{
	Mode   mode = OFF;
	double temper = -1.;   ///< power on the likelihoods; <= 0: step / box length
	int    gap = 1;        ///< leave out segments closer than this many steps
	double floor = 0.02;   ///< mixed in with the global class distribution
	int    bw_iters = 3;   ///< Baum-Welch updates of the transitions per iteration
	double alpha = 1.;     ///< filament mode: weight of the global distribution
	double stay = 0.5;     ///< chain mode: initial probability of keeping the class
};

/// RELION_FILAMENT_CONTEXT (off|chain|filament), RELION_FILAMENT_TEMPER,
/// RELION_FILAMENT_GAP, RELION_FILAMENT_FLOOR, RELION_FILAMENT_ALPHA. The
/// command-line options --filament_context, --filament_temper, --filament_gap,
/// --filament_floor and --filament_alpha of relion_refine override these.
/// An unknown mode is an error.
Options optionsFromEnvironment();

std::string modeName(Mode m);

/// "off" (or 0), "chain" (or hmm), "filament"; anything else is an error naming `what`.
Mode modeFromString(const std::string &s, const std::string &what);

/// Segments grouped per filament (rlnMicrographName + rlnHelicalTubeID), each list in
/// track order (rlnHelicalTrackLengthAngst, else rlnHelicalTrackLength). Particles
/// without a tube ID are left out. `step` returns the median distance between
/// consecutive segments (A, or pixels for the deprecated label; -1 if unknown).
std::vector<std::vector<long int> > groupFilaments(const MetaDataTable &MD, double &step);

/// The priors. `loglik` holds nr_parts x K per-class log-likelihoods (a row of all
/// zeros means "no information"); `pi` the global class distribution (K, sums to 1).
/// `logprior` (nr_parts x K) receives normalised log-priors; rows of particles not in
/// any filament are log(pi). Chain mode reads and updates `T` (K x K, row-stochastic;
/// empty: initialised from opt.stay).
void computePriors(const std::vector<std::vector<long int> > &filaments,
                   const std::vector<float> &loglik, int K,
                   const std::vector<double> &pi, const Options &opt, double temper,
                   std::vector<double> &T, std::vector<float> &logprior);

}

#endif
