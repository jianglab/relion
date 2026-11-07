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

#include "tilt_estimator.h"
#include "tilt_helper.h"
#include "ordered_sum.h"
#include "ctf_refiner.h"
#include <src/async_file_writer.h>

#include <src/jaz/single_particle/obs_model.h>
#include <src/jaz/single_particle/reference_map.h>
#include <src/jaz/single_particle/img_proc/image_op.h>
#include <src/jaz/single_particle/complex_io.h>
#include <src/jaz/single_particle/img_proc/filter_helper.h>
#include <src/jaz/single_particle/fftw_helper.h>
#include <src/jaz/single_particle/vtk_helper.h>
#include <src/jaz/single_particle/image_log.h>
#include <src/jaz/gravis/t2Vector.h>
#include <src/jaz/image/color_helper.h>

#include <src/args.h>
#include <src/ctf.h>

#include <set>

#include <omp.h>

using namespace gravis;


TiltEstimator::TiltEstimator()
	:	ready(false)
{}

void TiltEstimator::read(IOParser &parser, int argc, char *argv[])
{
	kmin = textToFloat(parser.getOption("--kmin_tilt",
		"Inner freq. threshold for beamtilt estimation [Å]", "20.0"));

	std::string aberrToken = "--odd_aberr_max_n";

	aberr_n_max  = textToInteger(parser.getOption(aberrToken,
		"Maximum degree of Zernike polynomials used to fit odd (i.e. antisymmetrical) aberrations", "0"));

	xring0 = textToDouble(parser.getOption("--xr0_t",
		"Exclusion ring start [Å] - use to exclude dominant frequency (e.g. for helices)", "-1"));

	xring1 = textToDouble(parser.getOption("--xr1_t",
		"Exclusion ring end [Å]", "-1"));

}

void TiltEstimator::init(
		int verb, int nr_omp_threads,
		bool debug, bool diag, std::string outPath,
		ReferenceMap* reference, ObservationModel* obsModel)
{
	this->verb = verb;
	this->nr_omp_threads = nr_omp_threads;

	this->debug = debug;
	this->diag = diag;
	this->outPath = outPath;

	this->reference = reference;
	this->obsModel = obsModel;

	angpix = obsModel->getPixelSizes();
	obsModel->getBoxSizes(s, sh);

	ready = true;
}

void TiltEstimator::processMicrograph(
		long g, MetaDataTable& mdt,
		const std::vector<Image<Complex>>& obs,
		const std::vector<Image<Complex>>& pred,
		bool do_ctf_padding)
{
	if (!ready)
	{
		REPORT_ERROR("ERROR: TiltEstimator::processMicrograph: TiltEstimator not initialized.");
	}

	std::vector<std::pair<int, std::vector<int>>> particlesByOpticsGroup
			= obsModel->splitParticlesByOpticsGroup(mdt);

	for (int pog = 0; pog < particlesByOpticsGroup.size(); pog++)
	{
		const int og = particlesByOpticsGroup[pog].first;
		const std::vector<int>& partIndices = particlesByOpticsGroup[pog].second;

		// TODO: SHWS 29mar2018: when data is CTF-premultiplied: do we need to change updateTiltShift??
		if (obsModel->getCtfPremultiplied(og))
			std::cerr << "TODO: check tilt estimation with CTF-premultiplied data!!" << std::endl;

		const int pc = partIndices.size();

		// Weighted sums over the particles, in their order whatever the number of threads
		Image<Complex> xyAccSum(sh[og], s[og]);
		Image<RFLOAT> wAccSum(sh[og], s[og]);

		xyAccSum.data.initZeros();
		wAccSum.data.initZeros();

		std::vector<Image<RFLOAT>> ctfImgs(orderedsum::chunk);

		orderedsum::run(pc, s[og], nr_omp_threads,
			[&](int pp, int slot)
			{
				CTF ctf;
				ctf.readByGroup(mdt, obsModel, partIndices[pp]);

				ctfImgs[slot] = Image<RFLOAT>(sh[og], s[og]);
				ctf.getFftwImage(ctfImgs[slot](), s[og], s[og], angpix[og], false, false, false, true, do_ctf_padding);
			},
			[&](int pp, int slot, int y)
			{
				const int p = partIndices[pp];
				TiltHelper::updateTiltShift(pred[p], obs[p], ctfImgs[slot], y, xyAccSum, wAccSum);
			});

		// Write out the intermediate results for this micrograph:

		std::string outRoot = CtfRefiner::getOutputFilenameRoot(mdt, outPath);

		std::stringstream sts;
		sts << (og+1);

		// By the background writer; CtfRefiner waits for it before the sums are read back
		auto xy = std::make_shared<Image<Complex>>(std::move(xyAccSum));
		auto wt = std::make_shared<Image<RFLOAT>>(std::move(wAccSum));
		const std::string root = outRoot, og_str = sts.str();

		AsyncFileWriter::instance().submit([xy, wt, root, og_str]
		{
			ComplexIO::write((*xy)(), root + "_xyAcc_optics-group_" + og_str, ".mrc");
			wt->write(root + "_wAcc_optics-group_" + og_str + ".mrc");
		}, 3 * wt->data.nzyxdim * sizeof(RFLOAT));
	}
}

void TiltEstimator::parametricFit(
		const std::vector<MetaDataTable>& mdts,
		MetaDataTable& optOut, std::vector <FileName> &fn_eps)
{
	if (!ready)
	{
		REPORT_ERROR("ERROR: TiltEstimator::parametricFit: TiltEstimator not initialized.");
	}

	if (verb > 0)
	{
		std::cout << " + Fitting beam tilt ..." << std::endl;
	}

	const int gc = mdts.size();
	const int ogc = obsModel->numberOfOpticsGroups();

	// write to std::vector<bool> is not thread safe even when writing to different elements (due to bit packing)
	// https://en.cppreference.com/cpp/container/vector_bool
	// https://github.com/3dem/relion/pull/1361
	std::vector<char> groupUsed(ogc, false);

	#pragma omp parallel for num_threads(nr_omp_threads)
	for (int og = 0; og < ogc; og++)
	{
		double Cs = obsModel->getSphericalAberration(og);
		double lambda = obsModel->getWavelength(og);

		std::stringstream sts;
		sts << (og+1);
		std::string ogstr = sts.str();

		Image<Complex> xyAccSum(sh[og], s[og]);
		Image<RFLOAT> wAccSum(sh[og], s[og]);

		xyAccSum.data.initZeros();
		wAccSum.data.initZeros();

		for (long g = 0; g < gc; g++)
		{
			std::string outRoot = CtfRefiner::getOutputFilenameRoot(mdts[g], outPath);

			if (   exists(outRoot+"_xyAcc_optics-group_"+ogstr+"_real.mrc")
				&& exists(outRoot+"_xyAcc_optics-group_"+ogstr+"_imag.mrc")
				&& exists(outRoot+ "_wAcc_optics-group_"+ogstr+".mrc"))
			{
				Image<Complex> xyAcc;
				Image<RFLOAT> wAcc;

				wAcc.read(outRoot+"_wAcc_optics-group_"+ogstr+".mrc");
				ComplexIO::read(xyAcc, outRoot+"_xyAcc_optics-group_"+ogstr, ".mrc");

				xyAccSum() += xyAcc();
				wAccSum()  +=  wAcc();

				groupUsed[og] = true;
			}
		}

		if (!groupUsed[og])
		{
			continue;
		}

		Image<RFLOAT> wgh, phase, fit, phaseFull, fitFull;

		FilterHelper::getPhase(xyAccSum, phase);

		Image<Complex> xyNrm(sh[og],s[og]);

		Image<RFLOAT> wgh0 = reference->getHollowWeight(kmin, s[og], angpix[og]);

		FilterHelper::multiply(wAccSum, wgh0, wgh);

		if (xring1 > 0.0)
		{
			for (int y = 0; y < s[og]; y++)
			for (int x = 0; x < sh[og]; x++)
			{
				double xx = x;
				double yy = y <= sh[og]? y : y - s[og];
				double rp = sqrt(xx*xx + yy*yy);
				double ra = s[og] * angpix[og] / rp;

				if (ra > xring0 && ra <= xring1)
				{
					wgh(y,x) = 0.0;
				}
			}
		}

		for (int y = 0; y < s[og]; y++)
		for (int x = 0; x < sh[og]; x++)
		{
			xyNrm(y,x) = wAccSum(y,x) > 0.0? xyAccSum(y,x)/wAccSum(y,x) : Complex(0.0, 0.0);
		}

		if (debug)
		{
			Image<RFLOAT> wghFull;
			FftwHelper::decenterDouble2D(wgh(), wghFull());

			ImageLog::write(wghFull, outPath + "beamtilt_weight-full_optics-group_"+ogstr);
		}

		FftwHelper::decenterUnflip2D(phase.data, phaseFull.data);

		ImageLog::write(phaseFull, outPath + "beamtilt_delta-phase_per-pixel_optics-group_"+ogstr);

		std::vector<Image<RFLOAT> > imgs_for_eps;
		std::vector<double> scales;
		std::vector<std::string> labels;

		imgs_for_eps.push_back(phaseFull);
		scales.push_back(1.);
		labels.push_back("Asymm. obs [-1, 1] " +obsModel->getGroupName(og));
		imgs_for_eps.push_back(phaseFull);
		scales.push_back(PI);
		labels.push_back("Asymm. obs [-pi, pi] " +obsModel->getGroupName(og));

		//ColorHelper::writeAngleToPNG(phaseFull,
		//	outPath + "beamtilt_delta-phase_per-pixel_optics-group_"+ogstr);
		//ColorHelper::writeAngleToEPS(phaseFull,
		//	outPath + "beamtilt_delta-phase_per-pixel_optics-group_"+ogstr);

		double shift_x(0), shift_y(0), tilt_x(0), tilt_y(0);

		if (aberr_n_max < 3)
		{
			TiltHelper::fitTiltShift(
				phase, wgh, Cs, lambda, angpix[og], obsModel->getMagMatrix(og),
				&shift_x, &shift_y, &tilt_x, &tilt_y, &fit);

			FftwHelper::decenterUnflip2D(fit.data, fitFull.data);

			ImageLog::write(fitFull, outPath + "beamtilt_delta-phase_lin-fit_optics-group_"+ogstr);

			TiltHelper::optimizeTilt(
					xyNrm, wgh, Cs, lambda,
					angpix[og], obsModel->getMagMatrix(og), false,
					shift_x, shift_y, tilt_x, tilt_y,
					&shift_x, &shift_y, &tilt_x, &tilt_y, &fit);

			FftwHelper::decenterUnflip2D(fit.data, fitFull.data);

			ImageLog::write(fitFull, outPath+"beamtilt_delta-phase_iter-fit_optics-group_"+ogstr);

			imgs_for_eps.push_back(fitFull);
			scales.push_back(1.);
			labels.push_back("Beamtilt-only fit [-1, 1] " +obsModel->getGroupName(og));
			imgs_for_eps.push_back(fitFull);
			scales.push_back(PI);
			labels.push_back("Beamtilt-only fit [-pi, pi] " +obsModel->getGroupName(og));

			#pragma omp critical
			{
				optOut.setValue(EMDL_IMAGE_BEAMTILT_X, tilt_x, og);
				optOut.setValue(EMDL_IMAGE_BEAMTILT_Y, tilt_y, og);
			}
		}
		else
		{
			Image<RFLOAT> one(sh[og],s[og]);
			one.data.initConstant(1);

			std::vector<double> Zernike_coeffs = TiltHelper::fitOddZernike(
				xyNrm, wgh, angpix[og], obsModel->getMagMatrix(og), aberr_n_max, &fit);

			FftwHelper::decenterUnflip2D(fit.data, fitFull.data);

			std::stringstream sts;
			sts << aberr_n_max;

			ImageLog::write(fitFull,
				outPath + "beamtilt_delta-phase_lin-fit_optics-group_"+ogstr+"_N-"+sts.str());

			{
				Image<RFLOAT> residual;
				residual.data = phaseFull.data - fitFull.data;

				ImageLog::write(residual,
					outPath + "beamtilt_delta-phase_lin-fit_optics-group_"
						+ogstr+"_N-"+sts.str()+"_residual");
			}

			std::vector<double> Zernike_coeffs_opt = TiltHelper::optimiseOddZernike(
				xyNrm, wgh, angpix[og], obsModel->getMagMatrix(og),
				aberr_n_max, Zernike_coeffs, &fit);

			FftwHelper::decenterUnflip2D(fit.data, fitFull.data);

			ImageLog::write(fitFull, outPath + "beamtilt_delta-phase_iter-fit_optics-group_"
				+ogstr+"_N-"+sts.str());

			imgs_for_eps.push_back(fitFull);
			scales.push_back(1.);
			labels.push_back("Asymm. fit (N="+sts.str()+") fit [-1, 1] " +obsModel->getGroupName(og));
			imgs_for_eps.push_back(fitFull);
			scales.push_back(PI);
			labels.push_back("Asymm. fit (N="+sts.str()+") fit [-pi, pi] " +obsModel->getGroupName(og));

			TiltHelper::extractTilt(Zernike_coeffs_opt, tilt_x, tilt_y, Cs, lambda);

			#pragma omp critical
			{
				optOut.setValue(EMDL_IMAGE_BEAMTILT_X, tilt_x, og);
				optOut.setValue(EMDL_IMAGE_BEAMTILT_Y, tilt_y, og);
				optOut.setValue(EMDL_IMAGE_ODD_ZERNIKE_COEFFS, Zernike_coeffs_opt, og);
			}
		}

		FileName fn_root = outPath + "asymmetric_aberrations_optics-group_"+ ogstr;
		ColorHelper::writeSignedToEPS(fn_root, 2, imgs_for_eps, scales, labels);
		fn_eps.push_back(fn_root+".eps");

	}
}

bool TiltEstimator::isFinished(const MetaDataTable &mdt)
{
	if (!ready)
	{
		REPORT_ERROR("ERROR: TiltEstimator::isFinished: TiltEstimator not initialized.");
	}

	std::string outRoot = CtfRefiner::getOutputFilenameRoot(mdt, outPath);

	bool allDone = true;

	std::vector<int> ogp = obsModel->getOptGroupsPresent_zeroBased(mdt);

	for (int i = 0; i < ogp.size(); i++)
	{
		std::stringstream sts;
		sts << (ogp[i]+1);
		std::string ogs = sts.str();

		if (   !exists(outRoot+"_xyAcc_optics-group_"+ogs+"_real.mrc")
			|| !exists(outRoot+"_xyAcc_optics-group_"+ogs+"_imag.mrc")
			|| !exists(outRoot+"_wAcc_optics-group_"+ogs+".mrc"))
		{
			allDone = false;
			break;
		}
	}

	return allDone;
}
