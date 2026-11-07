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

using namespace gravis;

GpMotionFit::GpMotionFit(
		const std::vector<std::vector<Image<double>>>& correlation,
		double cc_pad,
		double sig_vel_px, double sig_div_px, double sig_acc_px,
		int maxDims,
		const std::vector<d2Vector>& positions,
		const std::vector<d2Vector>& perFrameOffsets,
		int threads, bool expKer)
	:
	  expKer(expKer),
	  pc(correlation.size()),
	  fc(correlation[0].size()),
	  threads(threads),
	  cc_pad(cc_pad),
	  sig_vel_px(sig_vel_px),
	  sig_div_px(sig_div_px),
	  sig_acc_px(sig_acc_px),
	  correlation(correlation),
	  positions(positions),
	  perFrameOffsets(perFrameOffsets)
{
	Matrix2D<double> A(pc,pc);

	const double sv2 = sig_vel_px * sig_vel_px;
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

	basis = Matrix2D<RFLOAT>(pc,dc);

	for (int d = 0; d < dc; d++)
	{
		for (int p = 0; p < pc; p++)
		{
			basis(p,d) = (RFLOAT) defBasis.eigenvectors[dc*p + d];
		}
	}

	eigenVals = std::vector<double>(dc);

	for (int d = 0; d < dc; d++)
	{
		eigenVals[d] = (RFLOAT) defBasis.eigenvalues[d];
	}
}


double GpMotionFit::f(const std::vector<double> &x) const
{
	std::vector<std::vector<d2Vector>> pos(pc, std::vector<d2Vector>(fc));
	std::vector<double> e_p(pc);

	return energy(x, pos, e_p);
}

double GpMotionFit::f(const std::vector<double> &x, void* tempStorage) const
{
	if (tempStorage == 0) return f(x);

	TempStorage* ts = (TempStorage*) tempStorage;

	return energy(x, ts->pos, ts->e_p);
}

void GpMotionFit::grad(const std::vector<double> &x,
					   std::vector<double> &gradDest) const
{
	std::vector<std::vector<d2Vector>> pos(pc, std::vector<d2Vector>(fc));
	std::vector<std::vector<d2Vector>> ccg_pf(pc, std::vector<d2Vector>(fc));

	gradient(x, pos, ccg_pf, gradDest);
}

void GpMotionFit::grad(const std::vector<double> &x,
					   std::vector<double> &gradDest,
					   void* tempStorage) const
{
	if (tempStorage == 0) return grad(x, gradDest);

	TempStorage* ts = (TempStorage*) tempStorage;

	gradient(x, ts->pos, ts->ccg_pf, gradDest);
}

// Every sum below has a single owner and a fixed order, so the result is the same at any thread count
double GpMotionFit::energy(
		const std::vector<double> &x,
		std::vector<std::vector<d2Vector>>& pos,
		std::vector<double>& e_p) const
{
	paramsToPos(x, pos);

#pragma omp parallel for num_threads(threads)
	for (int p = 0; p < pc; p++)
	{
		double e = 0.0;

		for (int f = 0; f < fc; f++)
		{
			const double x = cc_pad * (pos[p][f].x + perFrameOffsets[f].x);
			const double y = cc_pad * (pos[p][f].y + perFrameOffsets[f].y);

			if (centres && !inWindow(p, f, x, y)) outside[p] = 1;

			e -= Interpolation::cubicXY(correlation[p][f], x, y, 0, 0, true);
		}

		e_p[p] = e;
	}

	double e_tot = 0.0;

	for (int p = 0; p < pc; p++)
	{
		e_tot += e_p[p];
	}

	for (int f = 0; f < fc-1; f++)
	for (int d = 0; d < dc; d++)
	{
		const double cx = x[2*(pc + dc*f + d)    ];
		const double cy = x[2*(pc + dc*f + d) + 1];

		e_tot += cx*cx + cy*cy;
	}

	if (sig_acc_px > 0.0)
	{
		for (int f = 0; f < fc-2; f++)
		for (int d = 0; d < dc; d++)
		{
			const double dcx = x[2*(pc + dc*(f+1) + d)    ] - x[2*(pc + dc*f + d)    ];
			const double dcy = x[2*(pc + dc*(f+1) + d) + 1] - x[2*(pc + dc*f + d) + 1];

			e_tot += eigenVals[d]*(dcx*dcx + dcy*dcy) / (sig_acc_px*sig_acc_px);
		}
	}

	return e_tot;
}

void GpMotionFit::gradient(
		const std::vector<double> &x,
		std::vector<std::vector<d2Vector>>& pos,
		std::vector<std::vector<d2Vector>>& ccg_pf,
		std::vector<double> &gradDest) const
{
	paramsToPos(x, pos);

	// Per particle: the gradient of its start position, then ccg_pf[p][f] becomes the sum over the frames after f
#pragma omp parallel for num_threads(threads)
	for (int p = 0; p < pc; p++)
	{
		d2Vector g0(0.0, 0.0);

		for (int f = 0; f < fc; f++)
		{
			const double x = cc_pad * (pos[p][f].x + perFrameOffsets[f].x);
			const double y = cc_pad * (pos[p][f].y + perFrameOffsets[f].y);

			if (centres && !inWindow(p, f, x, y)) outside[p] = 1;

			ccg_pf[p][f] = Interpolation::cubicXYgrad(correlation[p][f], x, y, 0, 0, true);

			g0 -= ccg_pf[p][f];
		}

		gradDest[2*p  ] = g0.x;
		gradDest[2*p+1] = g0.y;

		d2Vector after(0.0, 0.0);

		for (int f = fc-1; f >= 0; f--)
		{
			const d2Vector g = ccg_pf[p][f];
			ccg_pf[p][f] = after;
			after += g;
		}
	}

	const double sa2 = sig_acc_px*sig_acc_px;

	// Per velocity coefficient, summed over the particles in order
#pragma omp parallel for collapse(2) num_threads(threads)
	for (int f = 0; f < fc-1; f++)
	for (int d = 0; d < dc; d++)
	{
		const int i = 2*(pc + dc*f + d);

		d2Vector g(0.0, 0.0);

		for (int p = 0; p < pc; p++)
		{
			g -= (double) basis(p,d) * ccg_pf[p][f];
		}

		g.x += 2.0 * x[i];
		g.y += 2.0 * x[i+1];

		if (sig_acc_px > 0.0)
		{
			if (f < fc-2)
			{
				g.x -= 2.0 * eigenVals[d] * (x[i + 2*dc]     - x[i])     / sa2;
				g.y -= 2.0 * eigenVals[d] * (x[i + 2*dc + 1] - x[i + 1]) / sa2;
			}

			if (f > 0)
			{
				g.x += 2.0 * eigenVals[d] * (x[i]     - x[i - 2*dc])     / sa2;
				g.y += 2.0 * eigenVals[d] * (x[i + 1] - x[i - 2*dc + 1]) / sa2;
			}
		}

		gradDest[i  ] = g.x;
		gradDest[i+1] = g.y;
	}
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

	const int pad = 512;

	ts->pad = pad;
	ts->pos = std::vector<std::vector<d2Vector>>(pc, std::vector<d2Vector>(fc + pad));
	ts->ccg_pf = std::vector<std::vector<d2Vector>>(pc, std::vector<d2Vector>(fc + pad));
	ts->e_p = std::vector<double>(pc, 0.0);

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
#pragma omp parallel for num_threads(threads)
	for (int p = 0; p < pc; p++)
	{
		d2Vector pp(x[2*p], x[2*p+1]);

		for (int f = 0; f < fc; f++)
		{
			pos[p][f] = pp;

			if (f < fc-1)
			{
				d2Vector vel(0.0, 0.0);

				for (int d = 0; d < dc; d++)
				{
					const double cx = x[2*(pc + dc*f + d)    ];
					const double cy = x[2*(pc + dc*f + d) + 1];

					vel.x += cx * basis(p,d);
					vel.y += cy * basis(p,d);
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

			for (int p = 0; p < pc; p++)
			{
				d2Vector v = pos[p][f+1] - pos[p][f];

				c.x += v.x * basis(p,d);
				c.y += v.y * basis(p,d);
			}

			x[2*(pc + dc*f + d)  ] = c.x/eigenVals[d];
			x[2*(pc + dc*f + d)+1] = c.y/eigenVals[d];
		}
}
