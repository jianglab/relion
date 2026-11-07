#include "aberration_estimator.h"
#include "tilt_helper.h"
#include "ctf_refiner.h"

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
#include "ordered_sum.h"
#include <src/async_file_writer.h>

using namespace gravis;


AberrationEstimator::AberrationEstimator()
	:	ready(false)
{}

void AberrationEstimator::read(IOParser &parser, int argc, char *argv[])
{
	kmin = textToFloat(parser.getOption("--kmin_aberr",
		"Inner freq. threshold for symmetrical aberration estimation [Å]", "20.0"));

	std::string aberrToken = "--even_aberr_max_n";

	aberr_n_max  = textToInteger(parser.getOption(aberrToken,
		"Maximum degree of Zernike polynomials used to fit even (i.e. symmetrical) aberrations", "4"));

	xring0 = textToDouble(parser.getOption("--xr0_a", "Exclusion ring start [Å]", "-1"));
	xring1 = textToDouble(parser.getOption("--xr1_a", "Exclusion ring end [Å]", "-1"));
}

void AberrationEstimator::init(
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

void AberrationEstimator::processMicrograph(
		long g, MetaDataTable& mdt,
		const std::vector<Image<Complex>>& obs,
		const std::vector<Image<Complex>>& pred)
{
	if (!ready)
	{
		REPORT_ERROR("ERROR: AberrationEstimator::processMicrograph: AberrationEstimator not initialized.");
	}

	std::vector<std::pair<int, std::vector<int>>> particlesByOpticsGroup
			= obsModel->splitParticlesByOpticsGroup(mdt);

	for (int pog = 0; pog < particlesByOpticsGroup.size(); pog++)
	{
		const int og = particlesByOpticsGroup[pog].first;
		const std::vector<int>& partIndices = particlesByOpticsGroup[pog].second;

		// TODO: SHWS 29mar2018: when data is CTF-premultiplied: do we need to change below??
		if (obsModel->getCtfPremultiplied(og))
			std::cerr << "TODO: check aberration estimation with CTF-premultiplied data!!" << std::endl;

		const int pc = partIndices.size();

		// Sums over the particles, in their order whatever the number of threads
		Image<RFLOAT>
			AxxSum(sh[og],s[og]), AxySum(sh[og],s[og]), AyySum(sh[og],s[og]),
			bxSum(sh[og],s[og]), bySum(sh[og],s[og]);

		const double as = (double)s[og] * angpix[og];

		std::vector<CTF> ctfs(orderedsum::chunk);

		orderedsum::run(pc, s[og], nr_omp_threads,
			[&](int pp, int slot)
			{
				ctfs[slot].readByGroup(mdt, obsModel, partIndices[pp]);
			},
			[&](int pp, int slot, int y)
			{
				const int p = partIndices[pp];
				const CTF& ctf = ctfs[slot];

				for (int x = 0; x < sh[og]; x++)
				{
					const double xf = x;
					const double yf = y < sh[og]? y : y - s[og];

					const double gamma_i = ctf.getLowOrderGamma(xf/as, yf/as);
					const double cg = cos(gamma_i);
					const double sg = sin(gamma_i);

					Complex zobs = obs[p](y,x);
					Complex zprd = pred[p](y,x);

					double zz = zobs.real * zprd.real + zobs.imag * zprd.imag;
					double nr = zprd.norm();

					AxxSum(y,x) += nr * sg * sg;
					AxySum(y,x) += nr * cg * sg;
					AyySum(y,x) += nr * cg * cg;

					bxSum(y,x) -= zz * sg;
					bySum(y,x) -= zz * cg;
				}
			});

		// Write out the intermediate results per-micrograph:

		std::string outRoot = CtfRefiner::getOutputFilenameRoot(mdt, outPath);

		std::stringstream sts;
		sts << (og+1);

		// By the background writer; CtfRefiner waits for it before the sums are read back
		auto sums = std::make_shared<std::vector<Image<RFLOAT>>>(5);
		(*sums)[0] = std::move(AxxSum); (*sums)[1] = std::move(AxySum); (*sums)[2] = std::move(AyySum);
		(*sums)[3] = std::move(bxSum); (*sums)[4] = std::move(bySum);
		const std::string root = outRoot + "_aberr-", tail = "_optics-group_" + sts.str() + ".mrc";

		AsyncFileWriter::instance().submit([sums, root, tail]
		{
			const char* names[5] = {"Axx", "Axy", "Ayy", "bx", "by"};
			for (int i = 0; i < 5; i++) (*sums)[i].write(root + names[i] + tail);
		}, 5 * (*sums)[0].data.nzyxdim * sizeof(RFLOAT));
	}
}

void AberrationEstimator::parametricFit(
		const std::vector<MetaDataTable>& mdts,
		MetaDataTable& optOut, std::vector <FileName> &fn_eps)
{
	if (!ready)
	{
		REPORT_ERROR("ERROR: AberrationEstimator::parametricFit: AberrationEstimator not initialized.");
	}

	if (verb > 0)
	{
		std::cout << " + Fitting symmetrical aberrations ..." << std::endl;
	}

	const int gc = mdts.size();
	const int ogc = obsModel->numberOfOpticsGroups();

	// write to std::vector<bool> is not thread safe even when writing to different elements (due to bit packing)
	// https://en.cppreference.com/cpp/container/vector_bool
	// https://github.com/3dem/relion/pull/1361
	std::vector<char> groupUsed(ogc,false);

	#pragma omp parallel for num_threads(nr_omp_threads)
	for (int og = 0; og < ogc; og++)
	{
		std::stringstream sts;
		sts << og+1;
		std::string ogstr = sts.str();

		Image<RFLOAT>
			AxxSum(sh[og],s[og]), AxySum(sh[og],s[og]), AyySum(sh[og],s[og]),
			bxSum(sh[og],s[og]), bySum(sh[og],s[og]);

		for (long g = 0; g < gc; g++)
		{
			std::string outRoot = CtfRefiner::getOutputFilenameRoot(mdts[g], outPath);

			if (   exists(outRoot+"_aberr-Axx_optics-group_" + ogstr + ".mrc")
				&& exists(outRoot+"_aberr-Axy_optics-group_" + ogstr + ".mrc")
				&& exists(outRoot+"_aberr-Ayy_optics-group_" + ogstr + ".mrc")
				&& exists(outRoot+"_aberr-bx_optics-group_" + ogstr + ".mrc")
				&& exists(outRoot+"_aberr-by_optics-group_" + ogstr + ".mrc"))
			{

				Image<RFLOAT>
					Axx(sh[og],s[og]), Axy(sh[og],s[og]), Ayy(sh[og],s[og]),
					bx(sh[og],s[og]), by(sh[og],s[og]);

				Axx.read(outRoot+"_aberr-Axx_optics-group_" + ogstr + ".mrc");
				Axy.read(outRoot+"_aberr-Axy_optics-group_" + ogstr + ".mrc");
				Ayy.read(outRoot+"_aberr-Ayy_optics-group_" + ogstr + ".mrc");

				bx.read(outRoot+"_aberr-bx_optics-group_" + ogstr + ".mrc");
				by.read(outRoot+"_aberr-by_optics-group_" + ogstr + ".mrc");

				AxxSum() += Axx();
				AxySum() += Axy();
				AyySum() += Ayy();

				bxSum() += bx();
				bySum() += by();

				groupUsed[og] = true;
			}
		}

		if (!groupUsed[og])
		{
			continue;
		}

		Image<RFLOAT> wgh0(sh[og],s[og]), wgh(sh[og],s[og]), phase(sh[og],s[og]);
		Image<Complex> optXY(sh[og],s[og]);

		wgh0 = reference->getHollowWeight(kmin, s[og], angpix[og]);

		for (int y = 0; y < s[og];  y++)
		for (int x = 0; x < sh[og]; x++)
		{
			d2Matrix A(
				AxxSum(y,x), AxySum(y,x),
				AxySum(y,x), AyySum(y,x));

			d2Vector b(bxSum(y,x), bySum(y,x));

			double det = A(0,0) * A(1,1) - A(1,0) * A(0,1);

			if (det != 0.0)
			{
				d2Matrix Ai = A;
				Ai.invert();

				d2Vector opt = Ai * b;

				optXY(y,x) = Complex(opt.x, opt.y);
				phase(y,x) = std::abs(opt.x) > 0.0? atan2(opt.y, opt.x) : 0.0;
				wgh(y,x) = wgh0(y,x) * sqrt(std::abs(det));
			}
			else
			{
				optXY(y,x) = 0.0;
				phase(y,x) = 0.0;
				wgh0(y,x) = 0.0;
				wgh(y,x) = 0.0;
			}
		}

		if (xring1 > 0.0)
		{
			for (int y = 0; y < s[og]; y++)
			for (int x = 0; x < sh[og]; x++)
			{
				double xx = x;
				double yy = y < sh[og]? y : y - s[og];
				double rp = sqrt(xx*xx + yy*yy);
				double ra = s[og] * angpix[og] / rp;

				if (ra > xring0 && ra <= xring1)
				{
					wgh0(y,x) = 0.0;
					wgh(y,x) = 0.0;
				}
			}
		}

		if (debug)
		{
			Image<RFLOAT> full;
			FftwHelper::decenterDouble2D(wgh(), full());
			ImageLog::write(full, outPath + "aberr_weight-full_optics-group_"+ogstr);
		}

		std::vector<Image<RFLOAT> > imgs_for_eps;
		std::vector<double> scales;
		std::vector<std::string> labels;

		Image<RFLOAT> fit, phaseFull, fitFull;
		FftwHelper::decenterDouble2D(phase.data, phaseFull.data);
		ImageLog::write(phaseFull, outPath + "aberr_delta-phase_per-pixel_optics-group_"+ogstr);

		imgs_for_eps.push_back(phaseFull);
		scales.push_back(1.);
		labels.push_back("Symm. obs [-1, 1] "+obsModel->getGroupName(og));
		imgs_for_eps.push_back(phaseFull);
		scales.push_back(PI);
		labels.push_back("Symm. obs [-pi, pi] "+obsModel->getGroupName(og));


		{
			std::vector<double> Zernike_coeffs = TiltHelper::fitEvenZernike(
				phase, wgh, angpix[og], obsModel->getMagMatrix(og), aberr_n_max, &fit);

			FftwHelper::decenterDouble2D(fit.data, fitFull.data);

			std::stringstream sts;
			sts << aberr_n_max;

			ImageLog::write(fitFull, outPath + "aberr_delta-phase_lin-fit_optics-group_"
							+ogstr+"_N-"+sts.str());

			{
				Image<RFLOAT> residual;
				residual.data = phaseFull.data - fitFull.data;

				ImageLog::write(residual, outPath + "aberr_delta-phase_lin-fit_optics-group_"
								+ogstr+"_N-"+sts.str()+"_residual");
			}

			std::vector<double> Zernike_coeffs_opt = TiltHelper::optimiseEvenZernike(
				optXY, wgh0, AxxSum, AxySum, AyySum,
				angpix[og], obsModel->getMagMatrix(og),
				aberr_n_max, Zernike_coeffs, &fit);

			FftwHelper::decenterDouble2D(fit.data, fitFull.data);

			ImageLog::write(fitFull, outPath + "aberr_delta-phase_iter-fit_optics-group_"
							+ogstr+"_N-"+sts.str());

			imgs_for_eps.push_back(fitFull);
			scales.push_back(1.);
			labels.push_back("Symm. (N="+sts.str()+") fit [-1, 1] "+obsModel->getGroupName(og));
			imgs_for_eps.push_back(fitFull);
			scales.push_back(PI);
			labels.push_back("Symm. (N="+sts.str()+") fit [-pi, pi] "+obsModel->getGroupName(og));


			// extract Q0, Cs, defocus and astigmatism?
			#pragma omp critical
			{
				optOut.setValue(EMDL_IMAGE_EVEN_ZERNIKE_COEFFS, Zernike_coeffs_opt, og);
			}
		}

		FileName fn_root = outPath + "symmetric_aberrations_optics-group_"+ ogstr;
		ColorHelper::writeSignedToEPS(fn_root, 2, imgs_for_eps, scales, labels);
		fn_eps.push_back(fn_root+".eps");

	}

}

bool AberrationEstimator::isFinished(const MetaDataTable &mdt)
{
	if (!ready)
	{
		REPORT_ERROR("ERROR: AberrationEstimator::isFinished: AberrationEstimator not initialized.");
	}

	std::string outRoot = CtfRefiner::getOutputFilenameRoot(mdt, outPath);

	bool allDone = true;

	std::vector<int> ogs = obsModel->getOptGroupsPresent_oneBased(mdt);

	for (int i = 0; i < ogs.size(); i++)
	{
		const int og = ogs[i];

		std::stringstream sts;
		sts << og;

		if (   !exists(outRoot+"_aberr-Axx_optics-group_" + sts.str() + ".mrc")
			|| !exists(outRoot+"_aberr-Axy_optics-group_" + sts.str() + ".mrc")
			|| !exists(outRoot+"_aberr-Ayy_optics-group_" + sts.str() + ".mrc")
			|| !exists(outRoot+"_aberr-bx_optics-group_"  + sts.str() + ".mrc")
			|| !exists(outRoot+"_aberr-by_optics-group_"  + sts.str() + ".mrc"))
		{
			allDone = false;
			break;
		}
	}

	return allDone;
}
