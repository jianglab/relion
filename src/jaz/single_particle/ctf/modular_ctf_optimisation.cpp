#include "modular_ctf_optimisation.h"
#include <src/jaz/optics/magnification_helper.h>
#include <src/ctf.h>
#include <omp.h>

using namespace gravis;


ModularCtfOptimisation::ModularCtfOptimisation(
	MetaDataTable &mdt, 
	ObservationModel* obsModel, 
	const std::vector<Image<Complex>>& obs, 
	const std::vector<Image<Complex>>& pred, 
	const std::vector<Image<RFLOAT>>& frqWghByGroup,
	std::string modeStr,
	int num_treads)
	:
	mdt(mdt),
	obsModel(obsModel),
	obs(obs),
	pred(pred),
	particle_count(mdt.numberOfObjects()),
	num_treads(num_treads),
	frqWghByGroup(frqWghByGroup)
{
	initialValues.resize(CtfParamCount * particle_count);
	
	for (int p = 0; p < particle_count; p++)
	{
		CTF ctf;
		ctf.readByGroup(mdt, obsModel, p);
		
		std::vector<double> K = ctf.getK();
		
		const double Axx = K[1] * ctf.getAxx();
		const double Axy = K[1] * ctf.getAxy();
		const double Ayy = K[1] * ctf.getAyy();
				
		const double avgDz = (Axx + Ayy) / 2.0;
		const double bfac = ctf.Bfac;
		const double kfac = ctf.scale;
				
		initialValues[CtfParamCount * p + Phase]               = -K[5] - K[3];
		initialValues[CtfParamCount * p + Defocus]             = avgDz;
		initialValues[CtfParamCount * p + Astigmatism1]        = Axx - avgDz;
		initialValues[CtfParamCount * p + Astigmatism2]        = Axy;
		initialValues[CtfParamCount * p + SphericalAberration] = K[2];
		initialValues[CtfParamCount * p + BFactor]             = bfac;
		initialValues[CtfParamCount * p + ScaleFactor]         = kfac;
	}
	
	modes = decodeModes(modeStr);
			
	int currOff = 0;
	
	for (int i = 0; i < CtfParamCount; i++)
	{
		switch (modes[i])
		{
			case PerParticle:
			{
				paramOffset[i] = currOff;
				paramParticleStep[i] = 1;
				
				currOff += particle_count;
				
				break;
			}
				
			case PerMicrograph:
			{
				paramOffset[i] = currOff;
				paramParticleStep[i] = 0;
				
				currOff += 1;
				
				break;
			}
				
			case Fixed:
			{
				paramOffset[i] = 0;
				paramParticleStep[i] = 0;
				
				break;
			}
		}
	}
	
	param_count = currOff;
	
	angpix = obsModel->getPixelSizes();
	
	if (obsModel->hasEvenZernike)
	{
		std::vector<int> ogPres = obsModel->getOptGroupsPresent_zeroBased(mdt);
		
		aberrationByGroup.resize(obsModel->numberOfOpticsGroups());
		
		for (int ogpi = 0; ogpi < ogPres.size(); ogpi++)
		{
			const int og = ogPres[ogpi];
			aberrationByGroup[og] = obsModel->getGammaOffset(og, obsModel->getBoxSize(og));
		}
	}

	pixelsByGroup.resize(obsModel->numberOfOpticsGroups());

	for (int og : obsModel->getOptGroupsPresent_zeroBased(mdt))
	{
		const int s = obsModel->getBoxSize(og);
		const int sh = s/2 + 1;
		const double as = s * angpix[og];

		for (int yi = 0; yi < s;  yi++)
		for (int xi = 0; xi < sh; xi++)
		{
			const double freqWgh = frqWghByGroup[og](yi,xi);

			if (freqWgh == 0.0) continue;

			const double wght = (xi == 0)? 1.0 : 2.0;

			const double xx = xi/as;
			const double yy = (yi <= s/2)? yi/as : (yi - s)/as;

			double xu, yu;

			if (obsModel->hasMagMatrices)
			{
				const Matrix2D<RFLOAT>& M = obsModel->getMagMatrix(og);

				xu = M(0,0) * xx + M(0,1) * yy;
				yu = M(1,0) * xx + M(1,1) * yy;
			}
			else
			{
				xu = xx;
				yu = yy;
			}

			GroupPixels& gp = pixelsByGroup[og];
			const double u2 = xu * xu + yu * yu;
			gp.index.push_back(yi * (long)sh + xi);
			gp.u2.push_back(u2);
			gp.u4.push_back(u2 * u2);
			gp.a1.push_back(xu * xu - yu * yu);
			gp.a2.push_back(2.0 * xu * yu);
			gp.bf.push_back(-u2 / 4.0);
			gp.wght.push_back(freqWgh * wght);
			gp.gammaOffset.push_back(obsModel->hasEvenZernike? aberrationByGroup[og](xi,yi) : 0.0);
		}
	}

	for (int p = 0; p < particle_count; p++)
	{
		const int s = obsModel->getBoxSize(obsModel->getOpticsGroup(mdt, p));

		if (obs[p].data.ydim != s || pred[p].data.ydim != s || obs[p].data.xdim != s/2 + 1 || pred[p].data.xdim != s/2 + 1)
		{
			REPORT_ERROR_STR("ModularCtfOptimisation: particle " << p << " is not " << s << " pixels in size.");
		}
	}

	particleOg.resize(particle_count);
	partA.resize(particle_count);
	partB.resize(particle_count);
	partC0.resize(particle_count);

	#pragma omp parallel for num_threads(num_treads)
	for (int p = 0; p < particle_count; p++)
	{
		const int og = obsModel->getOpticsGroup(mdt, p);
		particleOg[p] = og;

		const GroupPixels& gp = pixelsByGroup[og];
		const size_t n = gp.index.size();
		partA[p].resize(n);
		partB[p].resize(n);

		const long* __restrict idx = gp.index.data();
		const double* __restrict w = gp.wght.data();
		double* __restrict ap = partA[p].data();
		double* __restrict bp = partB[p].data();
		const Complex* __restrict zo = obs[p].data.data;
		const Complex* __restrict zp = pred[p].data.data;

		double c0 = 0.0;
		for (size_t i = 0; i < n; i++)
		{
			const long k = idx[i];
			const Complex z_obs = zo[k];
			const Complex z_pred = zp[k];
			const double wt = w[i];

			ap[i] = wt * (z_pred.real * z_pred.real + z_pred.imag * z_pred.imag);
			bp[i] = wt * (z_pred.real * z_obs.real + z_pred.imag * z_obs.imag);
			c0 += wt * (z_obs.real * z_obs.real + z_obs.imag * z_obs.imag);
		}
		partC0[p] = c0;
	}
}

double ModularCtfOptimisation::gradAndValue(
		const std::vector<double> &x,
		std::vector<double> &gradDest) const
{
	// Each particle's cost and gradient are summed on their own, then over the particles in their order,
	// so the result does not depend on the number of threads.
	std::vector<double> costByParticle(particle_count), gradByParticle(CtfParamCount * particle_count);

	bool fit[CtfParamCount];

	for (int i = 0; i < CtfParamCount; i++)
	{
		fit[i] = modes[i] != Fixed;
	}

	const bool fitPh = fit[Phase];
	const bool fitDz = fit[Defocus];
	const bool fitA1 = fit[Astigmatism1];
	const bool fitA2 = fit[Astigmatism2];
	const bool fitCs = fit[SphericalAberration];
	const bool fitBf = fit[BFactor];
	const bool fitKf = fit[ScaleFactor];
	const bool onlyDefocusAstig = !fitPh && !fitCs && !fitBf && !fitKf;

	#pragma omp parallel for num_threads(num_treads)
	for (int p = 0; p < particle_count; p++)
	{
		const double ph = readParam(Phase, x, p);
		const double dz = readParam(Defocus, x, p);
		const double a1 = readParam(Astigmatism1, x, p);
		const double a2 = readParam(Astigmatism2, x, p);
		const double cs = readParam(SphericalAberration, x, p);
		const double bf = readParam(BFactor, x, p);
		const double kf = readParam(ScaleFactor, x, p);

		const int og = particleOg[p];
		const GroupPixels& gp = pixelsByGroup[og];
		const size_t n = gp.index.size();
		const double* __restrict u2v = gp.u2.data();
		const double* __restrict u4v = gp.u4.data();
		const double* __restrict a1v = gp.a1.data();
		const double* __restrict a2v = gp.a2.data();
		const double* __restrict bfv = gp.bf.data();
		const double* __restrict gov = gp.gammaOffset.data();
		const double* __restrict Ap = partA[p].data();
		const double* __restrict Bp = partB[p].data();

		double cost = partC0[p], sum[CtfParamCount] = {};

		if (onlyDefocusAstig && bf == 0.0)
		{
			double sDz = 0.0, sA1 = 0.0, sA2 = 0.0;
			if (fitDz && fitA1 && fitA2)
			{
				for (size_t i = 0; i < n; i++)
				{
					const double u2 = u2v[i];
					const double a1q = a1v[i];
					const double a2q = a2v[i];
					const double gamma = ph + dz * u2 + a1 * a1q + a2 * a2q + cs * u4v[i] + gov[i];
					double sg, cg;
					SINCOS(gamma, &sg, &cg);
					const double ctfVal = -kf * sg;
					const double err = ctfVal * Ap[i] - Bp[i];
					cost += ctfVal * (err - Bp[i]);
					const double dE_dGamma = -2.0 * err * kf * cg;
					sDz += dE_dGamma * u2;
					sA1 += dE_dGamma * a1q;
					sA2 += dE_dGamma * a2q;
				}
			}
			else
			{
				for (size_t i = 0; i < n; i++)
				{
					const double u2 = u2v[i];
					const double a1q = a1v[i];
					const double a2q = a2v[i];
					const double gamma = ph + dz * u2 + a1 * a1q + a2 * a2q + cs * u4v[i] + gov[i];
					double sg, cg;
					SINCOS(gamma, &sg, &cg);
					const double ctfVal = -kf * sg;
					const double err = ctfVal * Ap[i] - Bp[i];
					cost += ctfVal * (err - Bp[i]);
					const double dE_dGamma = -2.0 * err * kf * cg;
					if (fitDz) sDz += dE_dGamma * u2;
					if (fitA1) sA1 += dE_dGamma * a1q;
					if (fitA2) sA2 += dE_dGamma * a2q;
				}
			}
			sum[Defocus] = sDz;
			sum[Astigmatism1] = sA1;
			sum[Astigmatism2] = sA2;
		}
		else
		{
			const bool hasBf = (bf != 0.0);
			for (size_t i = 0; i < n; i++)
			{
				const double u2 = u2v[i];
				const double u4 = u4v[i];
				const double a1q = a1v[i];
				const double a2q = a2v[i];
				const double bfq = bfv[i];
				const double gamma = ph + dz * u2 + a1 * a1q + a2 * a2q + cs * u4 + gov[i];

				const double env = hasBf ? exp(bf * bfq) : 1.0;
				double sg, cg;
				SINCOS(gamma, &sg, &cg);
				const double ew = env * (-sg);
				const double ctfVal = kf * ew;
				const double err = ctfVal * Ap[i] - Bp[i];
				cost += ctfVal * (err - Bp[i]);

				const double dE_dCtfVal = 2.0 * err;
				const double dE_dGamma = dE_dCtfVal * kf * env * (-cg);

				if (fitPh) sum[Phase] += dE_dGamma;
				if (fitDz) sum[Defocus] += dE_dGamma * u2;
				if (fitA1) sum[Astigmatism1] += dE_dGamma * a1q;
				if (fitA2) sum[Astigmatism2] += dE_dGamma * a2q;
				if (fitCs) sum[SphericalAberration] += dE_dGamma * u4;
				if (fitBf) sum[BFactor] += dE_dCtfVal * ctfVal * bfq;
				if (fitKf) sum[ScaleFactor] += dE_dCtfVal * ew;
			}
		}

		costByParticle[p] = cost;

		for (int i = 0; i < CtfParamCount; i++)
		{
			gradByParticle[CtfParamCount * p + i] = sum[i];
		}
	}

	double out = 0.0;

	for (int i = 0; i < param_count; i++)
	{
		gradDest[i] = 0.0;
	}

	for (int p = 0; p < particle_count; p++)
	{
		out += costByParticle[p];

		for (int i = 0; i < CtfParamCount; i++)
		{
			if (fit[i])
			{
				gradDest[paramOffset[i] + p * paramParticleStep[i]] += gradByParticle[CtfParamCount * p + i];
			}
		}
	}

	return out;
}

std::vector<double> ModularCtfOptimisation::encodeInitial()
{
	std::vector<double> out(param_count, 0.0);
	
	for (int cp = 0; cp < CtfParamCount; cp++)
	{
		switch (modes[cp])
		{
			case PerParticle:
			{
				for (int p = 0; p < particle_count; p++)
				{
					out[paramOffset[cp] + p] = initialValues[CtfParamCount * p + cp];
				}
				
				break;
			}
				
			case PerMicrograph:
			{
				out[paramOffset[cp]] = initialValues[cp]; // returning value of first particle
				
				break;
			}
				
			case Fixed:
			{				
				break;
			}
		}
	}
	
	return out;
}

void ModularCtfOptimisation::writeToTable(const std::vector<double> &x)
{
	for (int p = 0; p < particle_count; p++)
	{
		const double ph = readParam(Phase, x, p);
		const double dz = readParam(Defocus, x, p);
		const double a1 = readParam(Astigmatism1, x, p);
		const double a2 = readParam(Astigmatism2, x, p);
		const double cs = readParam(SphericalAberration, x, p);
		const double bf = readParam(BFactor, x, p);
		const double kf = readParam(ScaleFactor, x, p);
				
		CTF ctf;
		ctf.readByGroup(mdt, obsModel, p);
		
		std::vector<double> K = ctf.getK();		
		
		d2Matrix A(-(dz+a1)/ K[1],  -a2 / K[1],
				       -a2 / K[1],  -(dz-a1) / K[1]);
		
		RFLOAT defocusU, defocusV, angleDeg;		
		MagnificationHelper::matrixToPolar(A, defocusU, defocusV, angleDeg);
		
		RFLOAT local_kV = ctf.kV * 1e3;		
		const double lambda = 12.2643247 / sqrt(local_kV * (1. + local_kV * 0.978466e-6));
		
		// ph = -K[5] - K[3] = -DEG2RAD(phase_shift) - atan(Q0/sqrt(1-Q0*Q0));
		// => phase_shift = RAD2DEG(-ph - K[3])
		if (modes[Phase] != Fixed)	
			mdt.setValue(EMDL_CTF_PHASESHIFT, RAD2DEG(-ph - K[3]), p);	
		
		mdt.setValue(EMDL_CTF_DEFOCUSU, defocusU, p);
		mdt.setValue(EMDL_CTF_DEFOCUSV, defocusV, p);
		mdt.setValue(EMDL_CTF_DEFOCUS_ANGLE, angleDeg, p);
		
		// cs = K[2] = (PI / 2) * C_s * 1e7 * lambda^3
		// => C_s = cs * (2/PI) / (1e7 / lambda^3)	
		if (modes[SphericalAberration] != Fixed)
			mdt.setValue(EMDL_CTF_CS, 2 * cs / (1e7 * PI * lambda*lambda*lambda), p);

		if (modes[BFactor] != Fixed)
			mdt.setValue(EMDL_CTF_BFACTOR, bf, p);

		if (modes[ScaleFactor] != Fixed)
			mdt.setValue(EMDL_CTF_SCALEFACTOR, kf, p);
	}
}

bool ModularCtfOptimisation::validateModeString(std::string mode)
{
	if (mode.length() != 5) return false;
	
	for (int i = 0; i < 5; i++)
	{
		if (mode[i] != 'p' && mode[i] != 'm' && mode[i] != 'f')
		{
			return false;
		}
	}
	
	return true;
}

std::vector<ModularCtfOptimisation::Mode> ModularCtfOptimisation::decodeModes(std::string s)
{
	std::vector<Mode> out(CtfParamCount);
	
	std::vector<Mode> charToMode(256, Fixed);
	
	charToMode[(unsigned char)'p'] = PerParticle;
	charToMode[(unsigned char)'m'] = PerMicrograph;
	
	out[Phase]                = charToMode[(unsigned char)s[0]];
	out[Defocus]              = charToMode[(unsigned char)s[1]];
	out[Astigmatism1]         = charToMode[(unsigned char)s[2]];
	out[Astigmatism2]         = charToMode[(unsigned char)s[2]];
	out[SphericalAberration]  = charToMode[(unsigned char)s[3]];
	out[BFactor]              = charToMode[(unsigned char)s[4]];
	out[ScaleFactor]          = charToMode[(unsigned char)s[4]];
	
	return out;
}
