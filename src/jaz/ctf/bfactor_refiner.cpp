#include "bfactor_refiner.h"
#include "ctf_refiner.h"

#include <omp.h>

#include <src/ctf.h>
#include <src/time.h>

#include <src/jaz/obs_model.h>
#include <src/jaz/reference_map.h>
#include <src/jaz/fftw_helper.h>
#include <src/jaz/image_log.h>
#include <src/jaz/img_proc/filter_helper.h>
#include <src/jaz/gravis/t2Vector.h>
#include <src/jaz/img_proc/image_op.h>
#include <src/jaz/img_proc/radial_avg.h>


using namespace gravis;

BFactorRefiner::BFactorRefiner()
	:	ready(false)
{}

void BFactorRefiner::read(IOParser &parser, int argc, char *argv[])
{
	perMicrograph = parser.checkOption("--bfac_per_mg", 
		"Estimate B-factors per micrograph, instead of per particle");
	
	min_B = textToDouble(parser.getOption("--bfac_min_B",
		"Minimal allowed B-factor", "-30"));
	
	max_B = textToDouble(parser.getOption("--bfac_max_B",
		"Maximal allowed B-factor", "300"));
	
	min_scale = textToDouble(parser.getOption("--bfac_min_scale",
		"Minimal allowed scale-factor (essential for outlier rejection)", "0.2"));
	
	kmin = textToDouble(parser.getOption("--kmin_bfac",
		"Inner freq. threshold for B-factor estimation [Angst]", "30.0"));
}

void BFactorRefiner::init(
		int verb, int s, int nr_omp_threads,
		bool debug, bool diag, std::string outPath,
		ReferenceMap *reference, ObservationModel *obsModel)
{
	this->verb = verb;
	this->s = s;
	sh = s/2 + 1;
	this->nr_omp_threads = nr_omp_threads;
	
	this->debug = debug;
	this->diag = diag;
	this->outPath = outPath;
	
	this->reference = reference;
	this->obsModel = obsModel;
	
	angpix = obsModel->getPixelSize(0);
	
	double kmin_px = obsModel->angToPix(kmin, s, 0);
	freqWeight = reference->getHollowWeight(kmin_px);
	
	ready = true;
}

void BFactorRefiner::processMicrograph(
		long g, MetaDataTable& mdt,
		const std::vector<Image<Complex>>& obs,
		const std::vector<Image<Complex>>& pred)
{
	if (!ready)
	{
		REPORT_ERROR("ERROR: BFactorRefiner::processMicrograph: BFactorRefiner not initialized.");
	}
	
	long pc = obs.size();
	
	std::stringstream stsg;
	stsg << g;
	
	std::vector<std::vector<std::pair<int,d2Vector>>> valsPerPart(nr_omp_threads);
	
	const double as = s * angpix;
	const double min_B_px = min_B / (as*as);
	const double max_B_px = max_B / (as*as);
	
	// search recursively numIters times, scanning the range at stepsPerIter points each time:
	const int stepsPerIter = 20;
	const int numIters = 5;
	
	
	if (perMicrograph)
	{
		std::vector<std::vector<double>>  t_rad(nr_omp_threads), s_rad(nr_omp_threads);
		
		for (int t = 0; t < nr_omp_threads; t++)
		{
			t_rad[t] = std::vector<double>(sh, 0.0);
			s_rad[t] = std::vector<double>(sh, 0.0);
		}
		
		// Parallel loop over all particles in this micrograph
		#pragma omp parallel for num_threads(nr_omp_threads)
		for (long p = 0; p < pc; p++)
		{
			int t = omp_get_thread_num();
			
			CTF ctf;
			ctf.readByGroup(mdt, obsModel, p);
			Image<RFLOAT> ctfImg(sh,s);
			ctf.getFftwImage(ctfImg(), s, s, angpix, false, false, false, false);
			
			for (int y = 0; y < s;  y++)
			for (int x = 0; x < sh; x++)
			{
				const double xx = x;
				const double yy = (y + s/2) % s - s/2;
				
				const int ri = (int)(sqrt(xx*xx + yy*yy) + 0.5);
				
				if (ri < sh)
				{
					const Complex zobs = obs[p](y,x);
					const Complex zpred = ctfImg(y,x) * pred[p](y,x);
					const double wp = freqWeight(y,x);
					
					t_rad[t][ri] += wp * (zpred.real * zpred.real + zpred.imag * zpred.imag);
					s_rad[t][ri] += wp * (zpred.real * zobs.real + zpred.imag * zobs.imag);
				}
			}
		}
		
		for (int t = 1; t < nr_omp_threads; t++)
		{
			for (int r = 0; r < sh; r++)
			{
				t_rad[0][r] += t_rad[t][r];
				s_rad[0][r] += s_rad[t][r];
			}
		}
		
		d2Vector BK = BFactorRefiner::findBKRec1D(
			t_rad[0], s_rad[0], min_B_px, max_B_px, min_scale, stepsPerIter, numIters);
		
		for (int p = 0; p < pc; p++)
		{
			mdt.setValue(EMDL_CTF_BFACTOR, as*as*BK[0] - min_B, p);
			mdt.setValue(EMDL_CTF_SCALEFACTOR, BK[1], p);
		}
		
		writePerMicrographEPS(mdt, s_rad[0], t_rad[0]);
		
	}
	else
	{
		#pragma omp parallel for num_threads(nr_omp_threads)
		for (long p = 0; p < pc; p++)
		{
			CTF ctf;
			ctf.readByGroup(mdt, obsModel, p);
			Image<RFLOAT> ctfImg(sh,s);
			ctf.getFftwImage(ctfImg(), s, s, angpix, false, false, false, false);
			
			std::vector<double> t_rad(sh, 0.0), s_rad(sh, 0.0);
			
			for (int y = 0; y < s;  y++)
			for (int x = 0; x < sh; x++)
			{
				const double xx = x;
				const double yy = (y + s/2) % s - s/2;
				
				const int ri = (int)(sqrt(xx*xx + yy*yy) + 0.5);
				
				if (ri < sh)
				{
					const Complex zobs = obs[p](y,x);
					const Complex zpred = ctfImg(y,x) * pred[p](y,x);
					const double wp = freqWeight(y,x);
					
					t_rad[ri] += wp * (zpred.real * zpred.real + zpred.imag * zpred.imag);
					s_rad[ri] += wp * (zpred.real * zobs.real + zpred.imag * zobs.imag);
				}
			}
			
			// slower, but will be necessary for anisotropic B-factors:
			/*
				Image<Complex> predCTF;			
				ImageOp::multiply(ctfImg, pred[p], predCTF);

				d2Vector sigmaK = BFactorRefiner::findBKRec2D(
				obs[p], predCTF, freqWeight, min_B_px, max_B_px, min_scale, stepsPerIter, numIters);
			*/
			
			d2Vector BK = BFactorRefiner::findBKRec1D(
						t_rad, s_rad, min_B_px, max_B_px, min_scale, stepsPerIter, numIters);
			
			int threadnum = omp_get_thread_num();				
			valsPerPart[threadnum].push_back(std::make_pair(p, BK));
			
			if (diag) writePerParticleDiagEPS(mdt, BK, s_rad, t_rad, p);
		}

		for (int t = 0; t < nr_omp_threads; t++)
		{
			for (int i = 0; i < valsPerPart[t].size(); i++)
			{
				int p = valsPerPart[t][i].first;
				d2Vector BK = valsPerPart[t][i].second;
				
				if (debug)
				{
					std::cout << p << ": " << as*as*BK[0] << " \t " << BK[1] << "\n";
				}
				
				mdt.setValue(EMDL_CTF_BFACTOR, as*as*BK[0] - min_B, p);
				mdt.setValue(EMDL_CTF_SCALEFACTOR, BK[1], p);
			}
		}
		
		// Output a diagnostic Postscript file
		writePerParticleEPS(mdt);
		
		if (diag)
		{
			std::string outRoot = CtfRefiner::getOutputFilenameRoot(mdt, outPath);
			
			std::vector<FileName> diagFns;
			
			for (int p = 0; p < pc; p++)
			{
				std::stringstream sts;
				sts << p;
				
				FileName fn_eps = outRoot + "_diag_particle_" + sts.str() + ".eps";
				
				if (exists(fn_eps))
				{
					diagFns.push_back(fn_eps);
				}
			}
			
			if (diagFns.size() > 0)
			{
				joinMultipleEPSIntoSinglePDF(outRoot + "_bfactors_per-particle.pdf", diagFns);
			}
		}
	}
	
	// Now write out STAR file with optimised values for this micrograph
	std::string outRoot = CtfRefiner::getOutputFilenameRoot(mdt, outPath);
	mdt.write(outRoot + "_bfactor_fit.star");
}

void BFactorRefiner::writePerMicrographEPS(
		const MetaDataTable& mdt, 
		const std::vector<double>& s_rad,
		const std::vector<double>& t_rad)
{
	if (!ready)
	{
		REPORT_ERROR("ERROR: BFactorRefiner::writeEPS: BFactorRefiner not initialized.");
	}
	
	std::string outRoot = CtfRefiner::getOutputFilenameRoot(mdt, outPath);
	
	FileName fn_eps = outRoot + "_bfactor_fit.eps";
	
	CPlot2D plot2D(fn_eps);
	plot2D.SetXAxisSize(600);
	plot2D.SetYAxisSize(600);
	plot2D.SetDrawLegend(false);
	plot2D.SetFlipY(false);
	
	RFLOAT B, a;
	mdt.getValue(EMDL_CTF_BFACTOR, B, 0);
	mdt.getValue(EMDL_CTF_SCALEFACTOR, a, 0);
	
	CDataSet curve;	
	curve.SetDrawMarker(false);
	curve.SetDrawLine(true);
	curve.SetDatasetColor(0,0,0);	
	
	const double as = s * angpix;
	
	double tMax = 0.0;
	
	for (int r = 0; r < sh; r++)
	{
		if (t_rad[r] > tMax) tMax = t_rad[r];
	}
	
	for (int r = 0; r < sh; r++)
	{
		const double ra = r / as;
		double cval = a * exp(-(B+min_B) * ra*ra / 4.0);
		
		CDataPoint cp(r, cval);		
		curve.AddDataPoint(cp);
		
		if (t_rad[r] > 1e-10 /*&& std::abs(s_rad[r] / t_rad[r]) < 5*/)
		{
			double pval = s_rad[r] / t_rad[r];
			double ucert = 0.9*(1.0 - t_rad[r] / tMax);
			
			CDataSet dataPts;
			dataPts.SetDrawMarker(true);
			dataPts.SetDrawLine(false);
			dataPts.SetMarkerSize(10);
			dataPts.SetDatasetColor(ucert,ucert,ucert);
					
			CDataPoint dp(r, pval);		
			dataPts.AddDataPoint(dp);
			
			plot2D.AddDataSet(dataPts);
		}
	}
	
	plot2D.AddDataSet(curve);
	
	std::string title = "CTF amplitude and B/k-factor fit";
	plot2D.SetXAxisTitle(title);
	
	plot2D.OutputPostScriptPlot(fn_eps);
}

void BFactorRefiner::writePerParticleDiagEPS(
		const MetaDataTable& mdt,
		d2Vector BKpixels, 
		const std::vector<double> &s_rad, 
		const std::vector<double> &t_rad,
		int particle_index)
{
	std::string outRoot = CtfRefiner::getOutputFilenameRoot(mdt, outPath);
	
	std::stringstream sts;
	sts << particle_index;
	FileName fn_eps = outRoot + "_diag_particle_" + sts.str() + ".eps";
	
	CPlot2D plot2D(fn_eps);
	plot2D.SetXAxisSize(600);
	plot2D.SetYAxisSize(600);
	plot2D.SetDrawLegend(false);
	plot2D.SetFlipY(false);
		
	CDataSet curve;	
	curve.SetDrawMarker(false);
	curve.SetDrawLine(true);
	curve.SetDatasetColor(0,0,0);	
		
	double tMax = 0.0;
	
	for (int r = 0; r < sh; r++)
	{
		if (t_rad[r] > tMax) tMax = t_rad[r];
	}
	
	for (int r = 0; r < sh; r++)
	{
		double cval = BKpixels[1] * exp(-BKpixels[0] * r*r / 4.0);
		
		CDataPoint cp(r, cval);		
		curve.AddDataPoint(cp);
		
		if (t_rad[r] > 1e-10 /*&& std::abs(s_rad[r] / t_rad[r]) < 5*/)
		{
			double pval = s_rad[r] / t_rad[r];
			double ucert = 0.9*(1.0 - t_rad[r] / tMax);
			
			CDataSet dataPts;
			dataPts.SetDrawMarker(true);
			dataPts.SetDrawLine(false);
			dataPts.SetMarkerSize(10);
			dataPts.SetDatasetColor(ucert,ucert,ucert);
					
			CDataPoint dp(r, pval);		
			dataPts.AddDataPoint(dp);
			
			plot2D.AddDataSet(dataPts);
		}
	}
	
	plot2D.AddDataSet(curve);
	
	std::string title = "CTF amplitude and B/k-factor fit";
	plot2D.SetXAxisTitle(title);
	
	plot2D.OutputPostScriptPlot(fn_eps);
	
}


void BFactorRefiner::writePerParticleEPS(const MetaDataTable& mdt)
{
	if (!ready)
	{
		REPORT_ERROR("ERROR: BFactorRefiner::writeEPS: BFactorRefiner not initialized.");
	}
	
	std::string outRoot = CtfRefiner::getOutputFilenameRoot(mdt, outPath);
	
	FileName fn_eps = outRoot + "_bfactor_fit.eps";
	
	CPlot2D plot2D(fn_eps);
	plot2D.SetXAxisSize(600);
	plot2D.SetYAxisSize(600);
	plot2D.SetDrawLegend(false);
	plot2D.SetFlipY(true);
	
	const int pc = mdt.numberOfObjects();
	
	for (int p = 0; p < pc; p++)
	{
		RFLOAT B, a;
		RFLOAT xcoor, ycoor;
		
		mdt.getValue(EMDL_IMAGE_COORD_X, xcoor, p);
		mdt.getValue(EMDL_IMAGE_COORD_Y, ycoor, p);
		mdt.getValue(EMDL_CTF_BFACTOR, B, p);
		mdt.getValue(EMDL_CTF_SCALEFACTOR, a, p);
		
		RFLOAT aval = 1.0 - a/2.0;
		RFLOAT bval = 1.01 - (B - min_B) / (max_B - min_B);
		
		CDataSet dataSet;
		
		dataSet.SetDrawMarker(true);
		dataSet.SetDrawLine(false);
		dataSet.SetMarkerSize(50*bval);
		dataSet.SetDatasetColor(aval, aval, aval);
		
		CDataPoint point(xcoor, ycoor);
		
		dataSet.AddDataPoint(point);
		
		plot2D.AddDataSet(dataSet);
	}
	
	std::string title = "B-factor (size) and CTF-scale (intensity)";
	plot2D.SetXAxisTitle(title);
	
	plot2D.OutputPostScriptPlot(fn_eps);
}

bool BFactorRefiner::isFinished(const MetaDataTable &mdt)
{
	if (!ready)
	{
		REPORT_ERROR("ERROR: BFactorRefiner::isFinished: BFactorRefiner not initialized.");
	}
	
	std::string outRoot = CtfRefiner::getOutputFilenameRoot(mdt, outPath);
	
	return exists(outRoot + "_bfactor_fit.star");
}

d2Vector BFactorRefiner::findBKRec1D(
		const std::vector<double>& t_rad, 
		const std::vector<double>& s_rad, 
		double B0, double B1, double min_scale,
		int steps, int depth)
{
	double minErr = std::numeric_limits<double>::max();
	double bestB = B0;
	double bestA = 1.0;
	
	const double eps = 1e-10;
	
	const int sh = t_rad.size();
	
	std::vector<double> sigVals(sh);
	
	for (int st = 0; st < steps; st++)
	{
		const double B = B0 + st*(B1 - B0)/(steps-1);
		
		for (int r = 0; r < sh; r++)
		{
			sigVals[r] = exp(-B * r * r / 4.0);
		}
		
		// find optimal scale-factor for hypothetical B-factor
		double num = 0.0, denom = 0.0;
		
		for (int r = 0; r < sh; r++)
		{
			const double tr = t_rad[r];
			const double sr = s_rad[r];
			const double br = sigVals[r];
			
			num   += sr * br;
			denom += tr * br * br;
		}
		
		double a = denom > eps? num / denom : num / eps;
		
		if (a < min_scale) a = min_scale;
		
		double sum = 0.0;
		
		for (int r = 0; r < sh; r++)
		{
			const double tr = t_rad[r];
			const double sr = s_rad[r];
			const double br = sigVals[r];
			
			// avoid the division by tr in:
			
			// const double er = a * br - sr / tr;			
			// sum += tr * er * er;
			
			// by dropping the constant-over-br offset sr²/tr²:
			
			sum += tr * a * a * br * br - 2.0 * a * br * sr;
		}
		
		if (sum < minErr)
		{
			minErr = sum;
			bestB = B;
			bestA = a;
		}
	}
	
	if (depth > 0)
	{
		const double hrange = (B1 - B0) / (steps - 1.0);
		double Bnext0 = bestB - hrange;
		double Bnext1 = bestB + hrange;
		
		if (Bnext0 < B0) Bnext0 = B0;
		if (Bnext1 > B1) Bnext1 = B1;
		
		return findBKRec1D(
			t_rad, s_rad, Bnext0, Bnext1, min_scale, steps, depth - 1);
	}
	
	return d2Vector(bestB, bestA);
	
}

d2Vector BFactorRefiner::findBKRec2D(
		const Image<Complex> &obs, 
		const Image<Complex> &pred, 
		const Image<RFLOAT> &weight, 
		double B0, double B1, double min_scale,
		int steps, int depth)
{
	double minErr = std::numeric_limits<double>::max();
	double bestB = B0;
	double bestA = 1.0;
	
	const int s = obs.data.ydim;
	const int sh = s/2 + 1;
	
	std::vector<double> sigVals(sh);
	
	for (int st = 0; st < steps; st++)
	{
		const double B = B0 + st*(B1 - B0)/(steps-1);
		
		for (int r = 0; r < sh; r++)
		{
			sigVals[r] = exp(-B * r * r / 4.0);
		}
		
		// find optimal scale-factor for hypothetical B-factor
		double num = 0.0, denom = 0.0;
		
		for (long y = 0; y < s;  y++)
		for (long x = 0; x < sh; x++)
		{
			const int xx = x;
			const int yy = y < sh? y : y - s;
			const int r = (int) (sqrt(xx*xx + yy*yy) + 0.5);
			
			if (r >= sh) continue;
			
			Complex vx = DIRECT_A2D_ELEM(pred.data, y, x);
			const Complex vy = DIRECT_A2D_ELEM(obs.data, y, x);
			const double vw = DIRECT_A2D_ELEM(weight.data, y, x);
			const double vb = sigVals[r];
			
			num   += vw * vb * (vx.real * vy.real + vx.imag * vy.imag);
			denom += vw * vb * vb * (vx.real * vx.real + vx.imag * vx.imag);
		}
		
		const double eps = 1e-20;
		double a = denom > eps? num / denom : num / eps;
		
		if (a < min_scale) a = min_scale;
		
		double sum = 0.0;
		
		for (long y = 0; y < s;  y++)
		for (long x = 0; x < sh; x++)
		{
			const int xx = x;
			const int yy = y < sh? y : y - s;
			const int r = (int) (sqrt(xx*xx + yy*yy) + 0.5);
			
			if (r >= sh) continue;
			
			Complex vx = DIRECT_A2D_ELEM(pred.data, y, x);
			const Complex vy = DIRECT_A2D_ELEM(obs.data, y, x);
			const double vw = DIRECT_A2D_ELEM(weight.data, y, x);
			const double vb = sigVals[r];
			
			sum += vw * (vy - a * vb * vx).norm();
		}
		
		if (sum < minErr)
		{
			minErr = sum;
			bestB = B;
			bestA = a;
		}
	}
	
	if (depth > 0)
	{
		const double hrange = (B1 - B0) / (steps - 1.0);
		double Bnext0 = bestB - hrange;
		double Bnext1 = bestB + hrange;
		
		if (Bnext0 < B0) Bnext0 = B0;
		if (Bnext1 > B1) Bnext1 = B1;
		
		return findBKRec2D(
			obs, pred, weight, Bnext0, Bnext1, min_scale, steps, depth - 1);
	}
	
	return d2Vector(bestB, bestA);
}