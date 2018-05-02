#include "motion_param_estimator.h"
#include "motion_refiner.h"
#include "two_hyperparameter_fit.h"
#include "three_hyperparameter_fit.h"

#include <src/jaz/optimization/nelder_mead.h>
#include <src/jaz/index_sort.h>
#include <src/jaz/filter_helper.h>


using namespace gravis;

const double MotionParamEstimator::velScale = 1000.0;
const double MotionParamEstimator::divScale = 1.0;
const double MotionParamEstimator::accScale = 10000.0;

MotionParamEstimator::MotionParamEstimator()
:   paramsRead(false), ready(false)
{
}

int MotionParamEstimator::read(IOParser& parser, int argc, char *argv[])
{
    parser.addSection("Parameter estimation");

    estim2 = parser.checkOption("--params2", "Estimate 2 parameters instead of motion");
    estim3 = parser.checkOption("--params3", "Estimate 3 parameters instead of motion");
    k_cutoff = textToFloat(parser.getOption("--k_cut", "Freq. cutoff for parameter estimation [Pixels]", "-1.0"));
    k_cutoff_Angst = textToFloat(parser.getOption("--k_cut_A", "Freq. cutoff for parameter estimation [Angstrom]", "-1.0"));
    k_eval = textToFloat(parser.getOption("--k_eval", "Threshold freq. for parameter evaluation [Pixels]", "-1.0"));
    k_eval_Angst = textToFloat(parser.getOption("--k_eval_A", "Threshold freq. for parameter evaluation [Angstrom]", "-1.0"));

    minParticles = textToInteger(parser.getOption("--min_p", "Minimum number of particles on which to estimate the parameters", "1000"));
    sV = textToFloat(parser.getOption("--s_vel_0", "Initial s_vel", "0.6"));
    sD = textToFloat(parser.getOption("--s_div_0", "Initial s_div", "3000"));
    sA = textToFloat(parser.getOption("--s_acc_0", "Initial s_acc", "5"));
    iniStep = textToFloat(parser.getOption("--in_step", "Initial step size in s_div", "100"));
    conv = textToFloat(parser.getOption("--conv", "Abort when simplex diameter falls below this", "10"));
    maxIters = textToInteger(parser.getOption("--par_iters", "Max. number of iterations", "50"));
    maxRange = textToInteger(parser.getOption("--mot_range", "Limit allowed motion range [Px]", "50"));
    seed = textToInteger(parser.getOption("--seed", "Random seed for micrograph selection", "23"));

    paramsRead = true;
}

void MotionParamEstimator::init(
    int verb, int nr_omp_threads, bool debug,
    int s, int fc,
    const std::vector<MetaDataTable>& allMdts,
    MotionEstimator* motionEstimator,
    ReferenceMap* reference,
    ObservationModel* obsModel)
{
    if (!paramsRead)
    {
        REPORT_ERROR("ERROR: MotionParamEstimator::init: MotionParamEstimator has not read its cmd-line parameters.");
    }

    this->verb = verb;
    this->nr_omp_threads = nr_omp_threads;
    this->debug = debug;
    this->s = s;
    this->fc = fc;
    this->motionEstimator = motionEstimator;
    this->obsModel = obsModel;
    this->reference = reference;

    if (!motionEstimator->isReady())
    {
        REPORT_ERROR("ERROR: MotionParamEstimator initialized before MotionEstimator.");
    }

    if (k_cutoff_Angst > 0.0 && k_cutoff > 0.0)
    {
        REPORT_ERROR("ERROR: Cutoff frequency can only be provided in pixels (--k_cut) or Angstrom (--k_eval_A), not both.");
    }

    if (k_eval_Angst > 0.0 && k_eval > 0.0)
    {
        REPORT_ERROR("ERROR: Evaluation frequency can only be provided in pixels (--k_eval) or Angstrom (--k_cut_A), not both.");
    }

    if (k_cutoff_Angst > 0.0 && k_cutoff < 0)
    {
        k_cutoff = obsModel->angToPix(k_cutoff_Angst, s);
    }

    else if (k_cutoff > 0 && k_cutoff_Angst < 0.0)
    {
        k_cutoff_Angst = obsModel->angToPix(k_cutoff, s);
    }

    if ((estim2 || estim3) && k_cutoff < 0)
    {
        REPORT_ERROR("ERROR: Parameter estimation requires a freq. cutoff (--k_cut or --k_cut_A).");
    }

    if (estim2 && estim3)
    {
        REPORT_ERROR("ERROR: Only 2 or 3 parameters can be estimated (--params2 or --params3), not both.");
    }

    if (k_eval < 0 && k_eval_Angst > 0.0)
    {
        k_eval = obsModel->angToPix(k_eval_Angst, s);
    }
    else if (k_eval > 0 && k_eval_Angst < 0.0)
    {
        k_eval_Angst = obsModel->angToPix(k_eval, s);
    }
    else
    {
        k_eval = k_cutoff;
        k_eval_Angst = k_cutoff_Angst;
    }

    if (verb > 0)
    {
        std::cout << " + maximum frequency to consider for alignment: "
            << k_cutoff_Angst << " A (" << k_cutoff << " px)\n";

        std::cout << " + frequency range to consider for evaluation:  "
            << k_eval_Angst << " - " << obsModel->pixToAng(reference->k_out,s) << " A ("
            << k_eval << " - " << reference->k_out << " px)\n";

    }

    const long mc = allMdts.size();

    srand(seed);

    std::vector<double> randNums(mc);

    for (int m = 0; m < mc; m++)
    {
        randNums[m] = rand() / (double)RAND_MAX;
    }

    std::vector<int> order = IndexSort<double>::sortIndices(randNums);

    int pc = 0;
    mdts.clear();

    if (verb > 0)
    {
        std::cout << " + micrographs randomly selected for parameter optimization:\n";
    }

    for (int i = 0; i < order.size(); i++)
    {
        const int m = order[i];

        const int pcm = allMdts[m].numberOfObjects();

        // motion estimation does not work on one single particle
        if (pcm < 2) continue;

        mdts.push_back(allMdts[m]);
        pc += pcm;

        if (verb > 0)
        {
            std::string mn;
            allMdts[m].getValue(EMDL_MICROGRAPH_NAME, mn, 0);

            std::cout << "        " << m << ": " << mn << "\n";
        }

        if (pc >= minParticles)
        {
            if (verb > 0)
            {
                std::cout << "\n + " << pc << " particles found in "
                          << mdts.size() << " micrographs\n";
            }

            break;
        }
    }

    if (verb > 0 && pc < minParticles)
    {
        std::cout << "\n   - Warning: this dataset does not contain " << minParticles
                  << " particles (--min_p) in micrographs with at least 2 particles\n";
    }

    k_out = reference->k_out;

    ready = true;
}

void MotionParamEstimator::run()
{
    #ifdef TIMING
        timeSetup = paramTimer.setNew(" time_Setup ");
        timeOpt = paramTimer.setNew(" time_Opt ");
        timeEval = paramTimer.setNew(" time_Eval ");
    #endif

    if (!ready)
    {
        REPORT_ERROR("ERROR: MotionParamEstimator::run: MotionParamEstimator not initialized.");
    }

    if (!estim2 && !estim3) return;

    RCTIC(paramTimer, timeSetup);

    prepAlignment();

    RCTOC(paramTimer, timeSetup);

    d4Vector opt;

    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);

    if (estim2)
    {
        //estimateTwoParamsRec(sV, sD, sA, rV*sV, rD*sD, steps, recursions);
        opt = estimateTwoParamsNM(sV, sD, sA, iniStep, conv, maxIters);
    }

    if (estim3)
    {
        //estimateThreeParamsRec(sV, sD, sA, rV*sV, rD*sD, rA*sA, steps, recursions);
        opt = estimateThreeParamsNM(sV, sD, sA, iniStep, conv, maxIters);
    }

    std::cout.setf(std::ios::floatfield);

    d3Vector nrm(
        opt[0] * velScale,
        opt[1] * divScale,
        opt[2] * accScale);

    // round result to conv / 2 (the min. radius of the optimization simplex)
    d3Vector rnd(
        conv * 0.5 * ((int)(2.0*nrm[0]/conv + 0.5)) / velScale,
        conv * 0.5 * ((int)(2.0*nrm[1]/conv + 0.5)) / divScale,
        conv * 0.5 * ((int)(2.0*nrm[2]/conv + 0.5)) / accScale);

    if (estim2)
    {
        rnd[2] = sA;
    }

    if (opt[2] <= 0.0)
    {
        rnd[2] = -1;
    }

    std::cout << "\ngood parameters:"
              << " --s_vel " << rnd[0]
              << " --s_div " << rnd[1]
              << " --s_acc " << rnd[2] << "\n\n";

    #ifdef TIMING
        paramTimer.printTimes(true);
    #endif
}

bool MotionParamEstimator::anythingToDo()
{
    return estim2 || estim3;
}

d4Vector MotionParamEstimator::estimateTwoParamsNM(
        double sig_v_0, double sig_d_0, double sig_acc, double inStep, double conv, int maxIters)
{
    std::cout << "\nit: \t s_vel: \t s_div: \t s_acc: \t fsc:\n\n";

    TwoHyperParameterProblem thpp(*this, sig_acc);

    std::vector<double> initial = TwoHyperParameterProblem::motionToProblem(
            d2Vector(sig_v_0, sig_d_0));

    double minTsc;

    std::vector<double> final = NelderMead::optimize(
            initial, thpp, inStep, conv, maxIters,
            1.0, 2.0, 0.5, 0.5, false, &minTsc);

    d2Vector vd = TwoHyperParameterProblem::problemToMotion(final);

    return d4Vector(vd[0], vd[1], sig_acc, -minTsc);
}

d4Vector MotionParamEstimator::estimateThreeParamsNM(
        double sig_v_0, double sig_d_0, double sig_a_0, double inStep, double conv, int maxIters)
{
    std::cout << "\nit: \t s_vel: \t s_div: \t s_acc: \t fsc:\n\n";

    ThreeHyperParameterProblem thpp(*this);

    std::vector<double> initial = ThreeHyperParameterProblem::motionToProblem(
            d3Vector(sig_v_0, sig_d_0, sig_a_0));

    double minTsc;

    std::vector<double> final = NelderMead::optimize(
            initial, thpp, inStep, conv, maxIters,
            1.0, 2.0, 0.5, 0.5, false, &minTsc);

    d3Vector vd = ThreeHyperParameterProblem::problemToMotion(final);

    return d4Vector(vd[0], vd[1], vd[2], -minTsc);
}

void MotionParamEstimator::evaluateParams(
    const std::vector<d3Vector>& sig_vals,
    std::vector<double>& TSCs)
{
    const int paramCount = sig_vals.size();
    TSCs.resize(paramCount);

    std::vector<double> sig_v_vals_px(paramCount);
    std::vector<double> sig_d_vals_px(paramCount);
    std::vector<double> sig_a_vals_px(paramCount);

    for (int i = 0; i < paramCount; i++)
    {
        sig_v_vals_px[i] = motionEstimator->normalizeSigVel(sig_vals[i][0]);
        sig_d_vals_px[i] = motionEstimator->normalizeSigDiv(sig_vals[i][1]);
        sig_a_vals_px[i] = motionEstimator->normalizeSigAcc(sig_vals[i][2]);
    }

    int pctot = 0;
    std::vector<d3Vector> tscsAs(paramCount, d3Vector(0.0, 0.0, 0.0));

    const int gc = mdts.size();

    for (long g = 0; g < gc; g++)
    {
        const int pc = mdts[g].numberOfObjects();

        if (pc < 2) continue; // not really needed, mdts are pre-screened

        pctot += pc;

        if (debug)
        {
            std::cout << "    micrograph " << (g+1) << " / " << mdts.size() << ": "
                << pc << " particles [" << pctot << " total]\n";
        }

        for (int i = 0; i < paramCount; i++)
        {
            if (debug)
            {
                std::cout << "        evaluating: " << sig_vals[i] << "\n";
            }

            RCTIC(paramTimer,timeOpt);

            std::vector<std::vector<gravis::d2Vector>> tracks =
                motionEstimator->optimize(
                    alignmentSet.CCs[g],
                    alignmentSet.initialTracks[g],
                    sig_v_vals_px[i], sig_a_vals_px[i], sig_d_vals_px[i],
                    alignmentSet.positions[g], alignmentSet.globComp[g]);

            RCTOC(paramTimer,timeOpt);

            RCTIC(paramTimer,timeEval);

            tscsAs[i] += alignmentSet.updateTsc(tracks, g, nr_omp_threads);

            RCTOC(paramTimer,timeEval);
        }

    } // micrographs

    if (debug)
    {
        std::cout << "\n";
    }

    RCTIC(paramTimer,timeEval);

    // compute final TSC
    for (int i = 0; i < paramCount; i++)
    {
        double wg = tscsAs[i][1] * tscsAs[i][2];

        if (wg > 0.0)
        {
            TSCs[i] = tscsAs[i][0] / sqrt(wg);
        }
    }

    RCTOC(paramTimer,timeEval);
}

void MotionParamEstimator::prepAlignment()
{
    std::cout << " + preparing alignment data... \n";

    const std::vector<Image<RFLOAT>>& dmgWgh = motionEstimator->getDamageWeights();
    std::vector<Image<RFLOAT>> alignDmgWgh(fc);

    for (int f = 0; f < fc; f++)
    {
        alignDmgWgh[f] = FilterHelper::ButterworthEnvFreq2D(dmgWgh[f], k_cutoff-1, k_cutoff+1);
    }

    alignmentSet = AlignmentSet<float>(mdts, fc, s, k_eval+2, k_out, maxRange);

    for (int f = 0; f < fc; f++)
    {
        alignmentSet.accelerate(dmgWgh[f], alignmentSet.damage[f]);
    }

    std::vector<ParFourierTransformer> fts(nr_omp_threads);

    const int gc = mdts.size();

    int pctot = 0;

    for (long g = 0; g < gc; g++)
    {
        const int pc = mdts[g].numberOfObjects();

        if (pc < 2) continue;

        pctot += pc;

        std::cout << "        micrograph " << (g+1) << " / " << gc << ": "
            << pc << " particles [" << pctot << " total]\n";

        std::vector<std::vector<Image<Complex>>> movie;
        std::vector<std::vector<Image<RFLOAT>>> movieCC;

        try
        {
            motionEstimator->prepMicrograph(
                mdts[g], fts, alignDmgWgh,
                movie, movieCC,
                alignmentSet.positions[g],
                alignmentSet.initialTracks[g],
                alignmentSet.globComp[g]);
        }
        catch (RelionError XE)
        {
            std::cerr << "warning: unable to load micrograph #" << (g+1) << "\n";
            continue;
        }

        #pragma omp parallel for num_threads(nr_omp_threads)
        for (int p = 0; p < pc; p++)
        {
            for (int f = 0; f < fc; f++)
            {
                if (maxRange > 0)
                {
                    movieCC[p][f] = FilterHelper::cropCorner2D(movieCC[p][f], 2*maxRange, 2*maxRange);
                }

                //alignmentSet.CCs[g][p][f] = movieCC[p][f];
                alignmentSet.copyCC(g, p, f, movieCC[p][f]);

                Image<Complex> pred = reference->predict(
                    mdts[g], p, *obsModel, ReferenceMap::Opposite);

                alignmentSet.accelerate(movie[p][f], alignmentSet.obs[g][p][f]);
                alignmentSet.accelerate(pred, alignmentSet.pred[g][p]);
            }
        }

        std::vector<std::vector<d2Vector>> tracks =
            motionEstimator->optimize(
                alignmentSet.CCs[g],
                alignmentSet.initialTracks[g],
                motionEstimator->normalizeSigVel(sV),
                motionEstimator->normalizeSigAcc(sA),
                motionEstimator->normalizeSigDiv(sD),
                alignmentSet.positions[g],
                alignmentSet.globComp[g]);

        for (int p = 0; p < pc; p++)
        {
            for (int f = 0; f < fc; f++)
            {
                alignmentSet.initialTracks[g][p][f] = tracks[p][f];
            }
        }
    }

    // release all unneeded heap space back to the OS
    // (this can free tens of Gb)
    malloc_trim(0);

    std::cout << "   done\n";
}
