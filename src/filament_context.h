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
	double clip = 12.;     ///< Huber clamp (nats before temper) on per-segment contrast; <= 0: off
	double exclude_angst = -1.; ///< physical leave-out radius along track (A); <= 0: step gap only
	int    start_iter = 1; ///< first iteration after which priors are applied
	double phase_weight = 1.; ///< weight on 3D helical azimuthal phase coherence; <= 0: off
};

struct ClassHelixParams
{
	double twist_deg = 0.;
	double rise_angst = 0.;
	int    sym_order = 1;  ///< rotational symmetry along Z (Cn * helical_nstart)
};

/// RELION_FILAMENT_CONTEXT (off|chain|filament), RELION_FILAMENT_TEMPER,
/// RELION_FILAMENT_GAP, RELION_FILAMENT_FLOOR, RELION_FILAMENT_ALPHA,
/// RELION_FILAMENT_CLIP, RELION_FILAMENT_EXCLUDE, RELION_FILAMENT_START,
/// RELION_FILAMENT_PHASE_WEIGHT. The command-line options --filament_* override these.
/// An unknown mode is an error.
Options optionsFromEnvironment();

std::string modeName(Mode m);

/// "off" (or 0), "chain" (or hmm), "filament"; anything else is an error naming `what`.
Mode modeFromString(const std::string &s, const std::string &what);

/// Segments grouped per filament (rlnMicrographName + rlnHelicalTubeID), each list in
/// track order (rlnHelicalTrackLengthAngst, else rlnHelicalTrackLength). Particles
/// without a tube ID are left out. `step` returns the median positive distance between
/// consecutive segments (A, or pixels for the deprecated label; -1 if unknown).
std::vector<std::vector<long int> > groupFilaments(const MetaDataTable &MD, double &step);

/// Same as groupFilaments, also returning track coordinates (A) per filament when known
/// (empty if track-length labels are absent).
std::vector<std::vector<long int> > groupFilamentsWithTracks(const MetaDataTable &MD, double &step,
                                                             std::vector<std::vector<double> > &tracks);

/// Leave-one-out 3D helical azimuthal phase coherence log-bonus (nr_parts x K) for each
/// segment and class, from other segments of its filament outside the exclusion gap.
void computeHelicalPhaseBonus(const std::vector<std::vector<long int> > &filaments,
                              const std::vector<std::vector<double> > &tracks,
                              long int nr_parts, int K,
                              const std::vector<double> &rot_deg,
                              const std::vector<double> &psi_deg,
                              const std::vector<double> &dx_angst,
                              const std::vector<double> &dy_angst,
                              const std::vector<char> &valid_pose,
                              const std::vector<double> &psi_prior_deg,
                              const std::vector<ClassHelixParams> &helix_params,
                              int gap, double exclude_angst, double phase_weight,
                              std::vector<float> &phase_bonus);

/// The priors. `loglik` holds nr_parts x K per-class log-likelihoods (a row of all
/// zeros means "no information"); `pi` the global class distribution (K, sums to 1).
/// `logprior` (nr_parts x K) receives normalised log-priors; rows of particles not in
/// any filament are log(pi). Chain mode reads and updates `T` (K x K, row-stochastic;
/// empty: initialised from opt.stay).
void computePriors(const std::vector<std::vector<long int> > &filaments,
                   const std::vector<float> &loglik, int K,
                   const std::vector<double> &pi, const Options &opt, double temper,
                   std::vector<double> &T, std::vector<float> &logprior);

/// Distance-aware overload supporting gapped tracks, physical exclusion radius, and optional
/// leave-one-out per-segment class log-score bonus (`extra_cavity_logscore`, nr_parts x K).
void computePriors(const std::vector<std::vector<long int> > &filaments,
                   const std::vector<std::vector<double> > &tracks, double step,
                   const std::vector<float> &loglik, int K,
                   const std::vector<double> &pi, const Options &opt, double temper,
                   std::vector<double> &T, std::vector<float> &logprior,
                   const std::vector<float> *extra_cavity_logscore = NULL);

}

#endif
