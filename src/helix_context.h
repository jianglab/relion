/***************************************************************************
 *
 * Filament context for helical 2D classification
 *
 * Segments of one filament are not independent: neighbours show the same helical
 * type, and along one type the 2D class (the view of the helix, i.e. the position
 * within the crossover) changes in a regular way with the distance along the track.
 * The segments of a filament are a hidden Markov chain over a regular grid of track
 * positions (one step = the segment spacing) with a two-level (hierarchical, "sticky")
 * state (type t, class c):
 *
 *   type:  stays with probability 1 - rho per step, otherwise switches to another type
 *   class: within type t, P(c' | c) = A_t[c][c']; after a switch to t', c' ~ q_t'
 *
 * The emissions are the class likelihoods of the segments from the E-step. The
 * parameters (rho, A_t, q_t) are learnt by Baum-Welch; the types are initialised
 * from a topic model (NMF) of the class counts of the filaments. A first-order chain
 * over the classes alone cannot do this: noisy and junk classes connect the types, so
 * after a few hundred A the class no longer tells the type.
 *
 * Segments overlap a lot (box >> step), so neighbouring emissions share most of their
 * pixels. The prior of a segment for the next E-step ("cavity") therefore only uses the
 * segments at least exclude_steps (w) away:
 *     left_t'(c')  = (1-rho)^w sum_c alpha_{s-w}(t',c) A_t'^w[c][c'] + (1-(1-rho)^w) q_t'(c') P(other type)
 *     right_t(c)   = the same backwards from the evidence at s+w and beyond
 *     prior(c)     = sum_t left_t(c) right_t(c)
 * and the log-likelihoods of the emissions can be tempered (multiplied by temper).
 *
 * Polarity: the picking direction of a filament is arbitrary, so about half of the filaments
 * have psi = psi_prior + 180. All segments of one filament share its polarity. The polarity is a
 * second, two-state sticky chain along each filament (over all of its segments, also across gaps;
 * switch probability pol_switch per step, for picking errors). Its emissions are the log-likelihood
 * ratios flipped / not flipped of the segments (from the coarse pass, where both psi modes are
 * always evaluated), clamped to +-pol_clamp and tempered by pol_temper (default step / diameter,
 * as the segments overlap). The prior P(flip) of a segment comes from the segments at least
 * pol_exclude_angst away and is the flip ratio of its psi prior in the next E-step.
 *
 ***************************************************************************/

#ifndef HELIX_CONTEXT_H_
#define HELIX_CONTEXT_H_

#include <vector>
#include <string>
#include "src/metadata_table.h"

class HelicalContext
{
public:

	// Options
	int nr_types;           // number of helical types (including one for junk if there is any)
	RFLOAT temper;          // multiplies the log-likelihoods of the emissions
	RFLOAT exclude_angst;   // segments closer than this do not contribute to the prior of a segment
	RFLOAT max_gap_steps;   // a larger gap in the track breaks the chain
	RFLOAT prior_floor;     // prior factors are at least this times their maximum
	RFLOAT ll_floor;        // class log-likelihoods are at least this below their maximum
	int nr_bw_iter;         // Baum-Welch iterations per update
	int verb;
	RFLOAT pol_switch;        // polarity switch probability per step
	RFLOAT pol_temper;        // factor on the polarity log-likelihood ratios (<= 0: step / diameter)
	RFLOAT pol_clamp;         // per-segment polarity log-likelihood ratios are clamped to +- this
	RFLOAT pol_exclude_angst; // segments closer than this do not contribute to the polarity prior (< 0: diameter / 2)
	RFLOAT pol_prior_max;     // polarity priors are within [1 - pol_prior_max, pol_prior_max]
	bool use_polarity_order;  // run the class chain of a filament with flipped polarity backwards (the views follow the molecule, not the picking direction)

	// Data
	int K;                                  // number of classes
	long int N;                             // number of segments
	RFLOAT step;                            // segment spacing (A)
	int exclude_steps;
	std::vector<std::vector<long int> > chains;  // per chain: segment index per grid position (-1: none)
	std::vector<long int> filament_of_chain;
	std::vector<std::string> filament_name; // micrograph:tube
	std::vector<long int> seg_chain, seg_pos;

	std::vector<float> loglik;    // N x K, log-likelihood of each class (NaN row: not seen yet)
	std::vector<float> prior;     // N x K, class prior factors for the next E-step (max 1)
	std::vector<float> post;      // N x K, class posterior of the chain
	std::vector<float> tpost;     // N x T, type posterior of the chain

	// Polarity (per segment; NaN: not seen yet)
	std::vector<std::vector<long int> > pol_members; // per filament: segments in track order
	std::vector<std::vector<long int> > pol_pos;     // their track positions in steps
	int pol_exclude_steps;
	std::vector<bool> chain_reversed; // the grid of the chain is in reverse track order
	std::vector<float> pol_llr;    // log P(data | flipped) / P(data | not flipped)
	std::vector<float> pol_prior;  // P(flipped) from the rest of the filament, for the next E-step
	std::vector<float> pol_post;   // P(flipped) given the whole filament
	long int pol_nr_switches, pol_nr_consistent;  // filaments: switches of the decoded polarity, filaments without switches
	double pol_frac_flipped;       // fraction of segments decoded as flipped

	// Model
	bool initialised_model, have_prior;
	RFLOAT rho;                   // type switch probability per step
	std::vector<double> A;        // T x K x K
	std::vector<double> q;        // T x K
	std::vector<double> pi0;      // T x K

	// Types
	std::vector<int> seg_type, seg_section; // N (type 0-based, section global 0-based)
	std::vector<float> seg_type_prob;
	std::vector<double> type_period, type_period_R, type_mass, type_section_angst;
	std::vector<int> type_nr_filaments, type_nr_sections;
	double loglik_total;
	double loglik_test;              // log-likelihood of the held-out chains (cross-validation)
	std::vector<char> chain_test;    // 1: chain held out from training (empty: none)

	HelicalContext():
		nr_types(4), temper(1.), exclude_angst(-1.),
		max_gap_steps(3.), prior_floor(1e-3), ll_floor(-6.9), nr_bw_iter(3), verb(1),
		pol_switch(1e-4), pol_temper(-1.), pol_clamp(6.9), pol_exclude_angst(-1.), pol_prior_max(0.999), use_polarity_order(false),
		pol_exclude_steps(1), pol_nr_switches(0), pol_nr_consistent(0), pol_frac_flipped(0.),
		K(0), N(0), step(0.), exclude_steps(1), initialised_model(false), have_prior(false),
		rho(1e-3), loglik_total(0.), loglik_test(0.)
	{}

	// Build the chains from the micrograph names, tube ids and track lengths of the segments
	// (seg_rows[i] is the row in MD of segment i). diameter_angst is the default of exclude_angst.
	void initialise(MetaDataTable &MD, const std::vector<long int> &seg_rows, int nr_classes, RFLOAT diameter_angst);

	// Store the class likelihoods of segment i from its posterior over classes (sums of the
	// weights per class) and the class prior that was used for it (NULL: flat).
	void storePosterior(long int i, const std::vector<RFLOAT> &class_post, const std::vector<RFLOAT> *class_prior_used);

	// Store log-likelihoods directly
	void storeLogLik(long int i, const std::vector<RFLOAT> &ll);

	// Forget the evidence of the last E-step (with MPI, each rank only sees part of the segments)
	void resetEvidence();

	// With MPI: sum the evidence of all ranks (each segment is seen by one rank) and broadcast the priors
	// (implemented in ml_optimiser_mpi.cpp, which has the MPI node)

	// Learn the model, compute posteriors, priors, types and sections
	void update();

	// Choose the number of types (tmin..tmax) by the log-likelihood of held-out filaments (nfold-fold
	// cross-validation over the filaments): the smallest number within tol (nats per segment) of the best;
	// sets nr_types and returns it
	int selectNrTypes(int tmin, int tmax, int nfold = 3, RFLOAT tol = 0.01);

	// Prior factor for class k of segment i (1 before the first update)
	inline RFLOAT getPrior(long int i, int k) const
	{
		return have_prior ? prior[i * K + k] : 1.;
	}

	// Polarity evidence of segment i: sums of its coarse weights over the psi angles that are not
	// flipped / flipped with respect to its psi prior, and the polarity prior factors that were applied
	void storePolarity(long int i, double w_noflip, double w_flip, double f_noflip, double f_flip);

	// Polarity chain: posteriors and priors
	void updatePolarity();

	// P(flipped) for the next E-step (0.5 before the first update)
	inline RFLOAT getPolarityPrior(long int i) const
	{
		return pol_prior.empty() ? 0.5 : pol_prior[i];
	}

	// rlnHelicalPolarityFlipProbability into MD
	void writePolarityLabels(MetaDataTable &MD, const std::vector<long int> &seg_rows) const;

	// Periods of the types from the class sequences: psi and psi_prior (degrees) per segment
	void fitPeriods(const std::vector<RFLOAT> &psi, const std::vector<RFLOAT> &psi_prior,
	                RFLOAT min_angst = 100., RFLOAT max_angst = 2000., RFLOAT step_angst = 10.);

	// Per-segment labels into MD (rlnHelicalType, rlnHelicalTypeProbability, rlnHelicalSectionID,
	// rlnHelicalContextProbability), using seg_rows as in initialise
	void writeLabels(MetaDataTable &MD, const std::vector<long int> &seg_rows, const std::vector<int> *assigned_class = NULL) const;

	// Tables of types and filaments
	void writeStar(const FileName &fn) const;

	// One particle file and one class-average file per type, from the labels in MDimg (rlnHelicalType, rlnClassNumber):
	// <root>_type<t>_data.star (optics + the segments of type t), and <root>_type<t>_classes.star/.mrcs (the classes
	// with >= min_frac of their segments and >= min_count segments of type t, from the stack fn_classes, if it exists)
	static void writeTypeFiles(const FileName &fn_root, MetaDataTable &MDopt, MetaDataTable &MDimg, int nr_types,
	                           const FileName &fn_classes, RFLOAT min_frac = 0.1, long int min_count = 20, int verb = 1);

private:
	void initialiseModel();
	void matPower(const std::vector<double> &M, int p, std::vector<double> &R) const;
	void makeSections();
};

#endif
