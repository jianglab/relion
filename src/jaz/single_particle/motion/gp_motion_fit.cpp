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

#include "gp_motion_fit.h"

#include <src/jaz/single_particle/interpolation.h>
#include <src/jaz/math/Gaussian_process.h>

#include <omp.h>
#include <functional>

using namespace gravis;

GpMotionFit::GpMotionFit(
		const std::vector<std::vector<Image<double>>>& correlation,
		double cc_pad,
		double sig_vel_px, double sig_div_px, double sig_acc_px,
		int maxDims,
		const std::vector<d2Vector>& positions,
		const std::vector<d2Vector>& perFrameOffsets,
		int threads, bool expKer,
		const GaussianProcess::Basis* unitBasis)
	:
	  expKer(expKer),
	  pc(correlation.size()),
	  fc(pc > 0 ? correlation[0].size() : 0),
	  threads(threads),
	  cc_pad(cc_pad),
	  sig_vel_px(sig_vel_px),
	  sig_div_px(sig_div_px),
	  sig_acc_px(sig_acc_px),
	  corrD(&correlation),
	  corrF(0),
	  positions(positions),
	  perFrameOffsets(perFrameOffsets)
{
	initBasis(maxDims, unitBasis);
}

GpMotionFit::GpMotionFit(
		const std::vector<std::vector<Image<float>>>& correlation,
		double cc_pad,
		double sig_vel_px, double sig_div_px, double sig_acc_px,
		int maxDims,
		const std::vector<d2Vector>& positions,
		const std::vector<d2Vector>& perFrameOffsets,
		int threads, bool expKer,
		const GaussianProcess::Basis* unitBasis)
	:
	  expKer(expKer),
	  pc(correlation.size()),
	  fc(pc > 0 ? correlation[0].size() : 0),
	  threads(threads),
	  cc_pad(cc_pad),
	  sig_vel_px(sig_vel_px),
	  sig_div_px(sig_div_px),
	  sig_acc_px(sig_acc_px),
	  corrD(0),
	  corrF(&correlation),
	  positions(positions),
	  perFrameOffsets(perFrameOffsets)
{
	initBasis(maxDims, unitBasis);
}

void GpMotionFit::initBasis(int maxDims, const GaussianProcess::Basis* unitBasis)
{
	const double sv2 = sig_vel_px * sig_vel_px;

	if (unitBasis != 0)
	{
		const int unit_dc = unitBasis->eigenvalues.size();
		int cap_dc = (maxDims < 0 || maxDims > unit_dc) ? unit_dc : maxDims;
		dc = 0;
		while (dc < cap_dc && sv2 * unitBasis->eigenvalues[dc] >= 1e-10)
			dc++;

		basisRow.resize((size_t)pc * dc);
		basisT.resize((size_t)dc * pc);
		eigenVals.resize(dc);

		for (int d = 0; d < dc; d++)
		{
			const double ev = sv2 * unitBasis->eigenvalues[d];
			const double l = sqrt(ev);
			eigenVals[d] = ev;
			for (int p = 0; p < pc; p++)
			{
				const double val = l * unitBasis->eigenvectors[(size_t)unit_dc * p + d];
				basisRow[(size_t)p * dc + d] = val;
				basisT[(size_t)d * pc + p] = val;
			}
		}
		return;
	}

	Matrix2D<double> A(pc,pc);

	const double sd2 = sig_div_px * sig_div_px;
	const double sd1 = sig_div_px;

	for (int i = 0; i < pc; i++)
	for (int j = i; j < pc; j++)
	{
		const double dd = (positions[i] - positions[j]).norm2();
		const double k = sv2 * (expKer? exp(-sqrt(dd/sd2)) : exp(-0.5*dd/sd1));
		A(i,j) = k;
		A(j,i) = k;
	}

	GaussianProcess::Basis defBasis = GaussianProcess::getBasis(A, maxDims, 1e-10);

	dc = defBasis.eigenvalues.size();

	basisRow.resize((size_t)pc * dc);
	basisT.resize((size_t)dc * pc);
	eigenVals.resize(dc);

	for (int d = 0; d < dc; d++)
	{
		for (int p = 0; p < pc; p++)
		{
			const double val = (RFLOAT) defBasis.eigenvectors[(size_t)dc * p + d];
			basisRow[(size_t)p * dc + d] = val;
			basisT[(size_t)d * pc + p] = val;
		}
		eigenVals[d] = (RFLOAT) defBasis.eigenvalues[d];
	}
}

double GpMotionFit::f(const std::vector<double> &x) const
{
	std::vector<d2Vector> pos_flat((size_t)pc * fc);
	std::vector<double> e_p(pc);

	return energy(x, pos_flat, e_p);
}

double GpMotionFit::f(const std::vector<double> &x, void* tempStorage) const
{
	if (tempStorage == 0) return f(x);

	TempStorage* ts = (TempStorage*) tempStorage;

	return energy(x, ts->pos_flat, ts->e_p);
}

void GpMotionFit::grad(const std::vector<double> &x,
					   std::vector<double> &gradDest) const
{
	const size_t fp_sz = (size_t)(fc > 1 ? fc - 1 : 0) * pc;
	std::vector<double> ccgx_fp(fp_sz), ccgy_fp(fp_sz);

	gradient(x, ccgx_fp, ccgy_fp, gradDest);
}

void GpMotionFit::grad(const std::vector<double> &x,
					   std::vector<double> &gradDest,
					   void* tempStorage) const
{
	if (tempStorage == 0) { grad(x, gradDest); return; }

	TempStorage* ts = (TempStorage*) tempStorage;

	gradient(x, ts->ccgx_fp, ts->ccgy_fp, gradDest);
}

double GpMotionFit::gradAndValue(const std::vector<double>& x,
                                 std::vector<double>& gradDest,
                                 void* tempStorage) const
{
	if (tempStorage != 0)
	{
		TempStorage* ts = (TempStorage*) tempStorage;
		return gradAndValueImpl(x, gradDest, ts->e_p, ts->ccgx_fp, ts->ccgy_fp);
	}
	const size_t fp_sz = (size_t)(fc > 1 ? fc - 1 : 0) * pc;
	std::vector<double> e_p(pc), ccgx_fp(fp_sz), ccgy_fp(fp_sz);
	return gradAndValueImpl(x, gradDest, e_p, ccgx_fp, ccgy_fp);
}

static inline void accumulateRegEnergyAndGrad(
		const std::vector<double>& x,
		int pc, int fc, int dc,
		double sig_acc_px,
		const std::vector<double>& eigenVals,
		double& e_tot,
		std::vector<double>* gradDest)
{
	for (int f = 0; f < fc - 1; f++)
	for (int d = 0; d < dc; d++)
	{
		const int i = 2 * (pc + dc * f + d);
		const double cx = x[i];
		const double cy = x[i + 1];
		e_tot += cx * cx + cy * cy;
	}

	if (sig_acc_px > 0.0)
	{
		const double inv_sa2 = 1.0 / (sig_acc_px * sig_acc_px);
		for (int f = 0; f < fc - 2; f++)
		for (int d = 0; d < dc; d++)
		{
			const double dcx = x[2 * (pc + dc * (f + 1) + d)    ] - x[2 * (pc + dc * f + d)    ];
			const double dcy = x[2 * (pc + dc * (f + 1) + d) + 1] - x[2 * (pc + dc * f + d) + 1];
			e_tot += eigenVals[d] * (dcx * dcx + dcy * dcy) * inv_sa2;
		}
	}
	(void)gradDest;
}

template<typename CC_T>
static double computeEnergyTyped(
		const std::vector<std::vector<Image<CC_T>>>& corr,
		const std::vector<double>& x,
		const std::vector<double>& basisRow,
		const std::vector<d2Vector>& perFrameOffsets,
		const std::vector<std::vector<i2Vector>>* centres,
		int window, int pc, int fc, int dc, int threads, double cc_pad,
		double sig_acc_px, const std::vector<double>& eigenVals,
		std::vector<char>& outside,
		std::vector<double>& e_p,
		const std::function<bool(int,int,double,double)>& inWin)
{
	#pragma omp parallel for num_threads(threads) if(threads > 1 && pc > 1)
	for (int p = 0; p < pc; p++)
	{
		d2Vector pp(x[2 * p], x[2 * p + 1]);
		const double* __restrict br = &basisRow[(size_t)p * dc];
		double e = 0.0;

		for (int f = 0; f < fc; f++)
		{
			const double px = cc_pad * (pp.x + perFrameOffsets[f].x);
			const double py = cc_pad * (pp.y + perFrameOffsets[f].y);

			if (centres && !inWin(p, f, px, py)) outside[p] = 1;

			e -= (double) Interpolation::cubicXY(corr[p][f], px, py, 0, 0, true);

			if (f < fc - 1)
			{
				double vx = 0.0, vy = 0.0;
				const double* __restrict xf = &x[2 * (pc + dc * f)];
				for (int d = 0; d < dc; d++)
				{
					const double b = br[d];
					vx += xf[2 * d    ] * b;
					vy += xf[2 * d + 1] * b;
				}
				pp.x += vx;
				pp.y += vy;
			}
		}

		e_p[p] = e;
	}

	double e_tot = 0.0;
	for (int p = 0; p < pc; p++)
		e_tot += e_p[p];

	accumulateRegEnergyAndGrad(x, pc, fc, dc, sig_acc_px, eigenVals, e_tot, 0);
	return e_tot;
}

template<typename CC_T>
static double computeGradAndValTyped(
		const std::vector<std::vector<Image<CC_T>>>& corr,
		const std::vector<double>& x,
		const std::vector<double>& basisRow,
		const std::vector<double>& basisT,
		const std::vector<d2Vector>& perFrameOffsets,
		const std::vector<std::vector<i2Vector>>* centres,
		int pc, int fc, int dc, int threads, double cc_pad,
		double sig_acc_px, const std::vector<double>& eigenVals,
		bool needEnergy,
		std::vector<char>& outside,
		std::vector<double>& e_p,
		std::vector<double>& ccgx_fp,
		std::vector<double>& ccgy_fp,
		std::vector<double>& gradDest,
		const std::function<bool(int,int,double,double)>& inWin)
{
	#pragma omp parallel num_threads(threads) if(threads > 1 && pc > 1)
	{
		std::vector<double> gx_f(fc), gy_f(fc);

		#pragma omp for schedule(static)
		for (int p = 0; p < pc; p++)
		{
			d2Vector pp(x[2 * p], x[2 * p + 1]);
			const double* __restrict br = &basisRow[(size_t)p * dc];
			double e = 0.0;
			double g0x = 0.0, g0y = 0.0;

			for (int f = 0; f < fc; f++)
			{
				const double px = cc_pad * (pp.x + perFrameOffsets[f].x);
				const double py = cc_pad * (pp.y + perFrameOffsets[f].y);

				if (centres && !inWin(p, f, px, py)) outside[p] = 1;

				double v = 0.0;
				d2Vector cg;
				Interpolation::cubicXYValAndGrad(corr[p][f], px, py, v, cg, 0, 0, true);

				e -= v;
				g0x -= cg.x;
				g0y -= cg.y;
				gx_f[f] = cg.x;
				gy_f[f] = cg.y;

				if (f < fc - 1)
				{
					double vx = 0.0, vy = 0.0;
					const double* __restrict xf = &x[2 * (pc + dc * f)];
					for (int d = 0; d < dc; d++)
					{
						const double b = br[d];
						vx += xf[2 * d    ] * b;
						vy += xf[2 * d + 1] * b;
					}
					pp.x += vx;
					pp.y += vy;
				}
			}

			if (needEnergy) e_p[p] = e;
			gradDest[2 * p    ] = g0x;
			gradDest[2 * p + 1] = g0y;

			double after_x = 0.0, after_y = 0.0;
			for (int f = fc - 1; f >= 1; f--)
			{
				after_x += gx_f[f];
				after_y += gy_f[f];
				ccgx_fp[(size_t)(f - 1) * pc + p] = after_x;
				ccgy_fp[(size_t)(f - 1) * pc + p] = after_y;
			}
		}
	}

	double e_tot = 0.0;
	if (needEnergy)
	{
		for (int p = 0; p < pc; p++)
			e_tot += e_p[p];
		accumulateRegEnergyAndGrad(x, pc, fc, dc, sig_acc_px, eigenVals, e_tot, 0);
	}

	const double sa2 = sig_acc_px * sig_acc_px;

	#pragma omp parallel for collapse(2) num_threads(threads) if(threads > 1 && (fc - 1) * dc > 1)
	for (int f = 0; f < fc - 1; f++)
	for (int d = 0; d < dc; d++)
	{
		const int i = 2 * (pc + dc * f + d);
		const double* __restrict b = &basisT[(size_t)d * pc];
		const double* __restrict cx = &ccgx_fp[(size_t)f * pc];
		const double* __restrict cy = &ccgy_fp[(size_t)f * pc];

		double gx = 0.0, gy = 0.0;
		for (int p = 0; p < pc; p++)
		{
			gx -= b[p] * cx[p];
			gy -= b[p] * cy[p];
		}

		gx += 2.0 * x[i];
		gy += 2.0 * x[i + 1];

		if (sig_acc_px > 0.0)
		{
			if (f < fc - 2)
			{
				gx -= 2.0 * eigenVals[d] * (x[i + 2 * dc    ] - x[i    ]) / sa2;
				gy -= 2.0 * eigenVals[d] * (x[i + 2 * dc + 1] - x[i + 1]) / sa2;
			}
			if (f > 0)
			{
				gx += 2.0 * eigenVals[d] * (x[i    ] - x[i - 2 * dc    ]) / sa2;
				gy += 2.0 * eigenVals[d] * (x[i + 1] - x[i - 2 * dc + 1]) / sa2;
			}
		}

		gradDest[i    ] = gx;
		gradDest[i + 1] = gy;
	}

	return e_tot;
}

// Every sum below has a single owner and a fixed order, so the result is the same at any thread count
double GpMotionFit::energy(
		const std::vector<double> &x,
		std::vector<d2Vector>& pos_flat,
		std::vector<double>& e_p) const
{
	(void)pos_flat;
	auto inWin = [this](int p, int f, double px, double py) { return inWindow(p, f, px, py); };
	if (corrD)
		return computeEnergyTyped(*corrD, x, basisRow, perFrameOffsets, centres, window, pc, fc, dc, threads, cc_pad, sig_acc_px, eigenVals, outside, e_p, inWin);
	return computeEnergyTyped(*corrF, x, basisRow, perFrameOffsets, centres, window, pc, fc, dc, threads, cc_pad, sig_acc_px, eigenVals, outside, e_p, inWin);
}

void GpMotionFit::gradient(
		const std::vector<double> &x,
		std::vector<double>& ccgx_fp,
		std::vector<double>& ccgy_fp,
		std::vector<double> &gradDest) const
{
	std::vector<double> dummy_ep;
	auto inWin = [this](int p, int f, double px, double py) { return inWindow(p, f, px, py); };
	if (corrD)
		computeGradAndValTyped(*corrD, x, basisRow, basisT, perFrameOffsets, centres, pc, fc, dc, threads, cc_pad, sig_acc_px, eigenVals, false, outside, dummy_ep, ccgx_fp, ccgy_fp, gradDest, inWin);
	else
		computeGradAndValTyped(*corrF, x, basisRow, basisT, perFrameOffsets, centres, pc, fc, dc, threads, cc_pad, sig_acc_px, eigenVals, false, outside, dummy_ep, ccgx_fp, ccgy_fp, gradDest, inWin);
}

double GpMotionFit::gradAndValueImpl(
		const std::vector<double> &x,
		std::vector<double> &gradDest,
		std::vector<double> &e_p,
		std::vector<double> &ccgx_fp,
		std::vector<double> &ccgy_fp) const
{
	auto inWin = [this](int p, int f, double px, double py) { return inWindow(p, f, px, py); };
	if (corrD)
		return computeGradAndValTyped(*corrD, x, basisRow, basisT, perFrameOffsets, centres, pc, fc, dc, threads, cc_pad, sig_acc_px, eigenVals, true, outside, e_p, ccgx_fp, ccgy_fp, gradDest, inWin);
	return computeGradAndValTyped(*corrF, x, basisRow, basisT, perFrameOffsets, centres, pc, fc, dc, threads, cc_pad, sig_acc_px, eigenVals, true, outside, e_p, ccgx_fp, ccgy_fp, gradDest, inWin);
}

void GpMotionFit::setWindow(const std::vector<std::vector<i2Vector>>* centres, int window)
{
	this->centres = centres;
	this->window = window;
	outside = std::vector<char>(pc, 0);
}

bool GpMotionFit::leftWindow() const
{
	for (int p = 0; p < outside.size(); p++)
	{
		if (outside[p]) return true;
	}

	return false;
}

bool GpMotionFit::inWindow(int p, int f, double x, double y) const
{
	// the same floor as Interpolation::cubicXY, which reads pixels floor-1 .. floor+2
	const int xi = (int)std::floor(x), yi = (int)std::floor(y);
	const i2Vector c = (*centres)[p][f];

	return xi - 1 >= c.x - window/2 && xi + 2 < c.x + window/2
	    && yi - 1 >= c.y - window/2 && yi + 2 < c.y + window/2;
}

void *GpMotionFit::allocateTempStorage() const
{
	TempStorage* ts = new TempStorage;

	const size_t fp_sz = (size_t)(fc > 1 ? fc - 1 : 0) * pc;
	ts->pos_flat.resize((size_t)pc * fc);
	ts->ccgx_fp.resize(fp_sz);
	ts->ccgy_fp.resize(fp_sz);
	ts->e_p.assign(pc, 0.0);

	return ts;
}

void GpMotionFit::deallocateTempStorage(void* ts) const
{
	if (ts) delete (TempStorage*) ts;
}

void GpMotionFit::paramsToPos(
		const std::vector<double>& x,
		std::vector<std::vector<d2Vector>>& pos) const
{
#pragma omp parallel for num_threads(threads) if(threads > 1 && pc > 1)
	for (int p = 0; p < pc; p++)
	{
		d2Vector pp(x[2*p], x[2*p+1]);
		const double* __restrict br = &basisRow[(size_t)p * dc];

		for (int f = 0; f < fc; f++)
		{
			pos[p][f] = pp;

			if (f < fc-1)
			{
				d2Vector vel(0.0, 0.0);
				const double* __restrict xf = &x[2*(pc + dc*f)];

				for (int d = 0; d < dc; d++)
				{
					const double b = br[d];
					vel.x += xf[2*d    ] * b;
					vel.y += xf[2*d + 1] * b;
				}

				pp += vel;
			}
		}
	}
}

void GpMotionFit::posToParams(
		const std::vector<std::vector<d2Vector>>& pos,
		std::vector<double>& x) const
{
	x.resize(2*(pc + dc*(fc-1)));

	for (int p = 0; p < pc; p++)
	{
		x[2*p]   = pos[p][0].x;
		x[2*p+1] = pos[p][0].y;
	}

	for (int f = 0; f < fc-1; f++)
		for (int d = 0; d < dc; d++)
		{
			d2Vector c(0.0, 0.0);
			const double* __restrict bt = &basisT[(size_t)d * pc];

			for (int p = 0; p < pc; p++)
			{
				d2Vector v = pos[p][f+1] - pos[p][f];

				c.x += v.x * bt[p];
				c.y += v.y * bt[p];
			}

			x[2*(pc + dc*f + d)  ] = c.x/eigenVals[d];
			x[2*(pc + dc*f + d)+1] = c.y/eigenVals[d];
		}
}
