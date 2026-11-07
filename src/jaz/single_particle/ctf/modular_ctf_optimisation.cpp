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

			Pixel q;
			q.index = yi * (long)sh + xi;
			q.xu = xu;
			q.yu = yu;
			q.u2 = xu * xu + yu * yu;
			q.u4 = q.u2 * q.u2;
			q.a1 = xu * xu - yu * yu;
			q.bf = -q.u2/4.0;
			q.wght = freqWgh * wght;
			q.gammaOffset = obsModel->hasEvenZernike? aberrationByGroup[og](xi,yi) : 0.0;

			pixelsByGroup[og].push_back(q);
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

		const std::vector<Pixel>& pixels = pixelsByGroup[obsModel->getOpticsGroup(mdt, p)];
		const Complex* zo = obs[p].data.data;
		const Complex* zp = pred[p].data.data;

		double cost = 0.0, sum[CtfParamCount] = {};

		for (int i = 0; i < pixels.size(); i++)
		{
			const Pixel& q = pixels[i];
			const double xu = q.xu, yu = q.yu, u2 = q.u2;

			const double gamma = ph
					+ (dz + a1) * xu * xu + 2.0 * a2 * xu * yu + (dz - a1) * yu * yu
					+ cs * q.u4 + q.gammaOffset;

			// exp(-0.0) == 1
			const double env = bf == 0.0? 1.0 : exp(-bf * u2 / 4.0);
			const double wave = -sin(gamma);
			const double ctfVal = kf * env * wave;

			const Complex z_obs = zo[q.index];
			const Complex z_pred = zp[q.index];
			const Complex z_err = ctfVal * z_pred - z_obs;

			cost += q.wght * z_err.norm();

			const double dE_dCtfVal = 2.0 * q.wght * (
						z_err.real * z_pred.real + z_err.imag * z_pred.imag);

			const double dE_dGamma = dE_dCtfVal * kf * env * (-cos(gamma));

			if (fit[Phase])               sum[Phase] += dE_dGamma;
			if (fit[Defocus])             sum[Defocus] += dE_dGamma * u2;
			if (fit[Astigmatism1])        sum[Astigmatism1] += dE_dGamma * q.a1;
			if (fit[Astigmatism2])        sum[Astigmatism2] += dE_dGamma * 2.0 * xu * yu;
			if (fit[SphericalAberration]) sum[SphericalAberration] += dE_dGamma * q.u4;
			if (fit[BFactor])             sum[BFactor] += dE_dCtfVal * kf * wave * env * q.bf;
			if (fit[ScaleFactor])         sum[ScaleFactor] += dE_dCtfVal * env * wave;
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
