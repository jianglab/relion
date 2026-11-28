#include "Gaussian_process.h"

#include <src/Eigen/Dense>
#include <src/Eigen/Eigenvalues>
#include <cmath>

using namespace Eigen;


Matrix2D<double> GaussianProcess::computeCovariance(
		const std::vector<gravis::d3Vector> &points, 
		GpKernel *kernel)
{
	const int pc = points.size();
	
	Matrix2D<double> A(pc,pc);

    for (int i = 0; i < pc; i++)
    for (int j = i; j < pc; j++)
    {
		const double k = kernel->covariance(points[i], points[j]);
        A(i,j) = k;
        A(j,i) = k;
    }
	
	return A;
}

static GaussianProcess::Basis computeSymmetricBasis(
		const Matrix2D<double> &C, int maxDims, double eps, bool scaleSqrt)
{
	const int N = C.Xdim();

	int dc = (maxDims < 0 || maxDims > N)? N : maxDims;

	MatrixXd A0(N,N);

	for (int c = 0; c < N; c++)
	for (int r = 0; r < N; r++)
	{
		A0(r,c) = C(r,c);
	}

	SelfAdjointEigenSolver<MatrixXd> es(A0);

	// Eigenvalues from SelfAdjointEigenSolver are in ascending order
	for (int d = 0; d < dc; d++)
	{
		if (es.eigenvalues()[N - 1 - d] < eps)
		{
			dc = d;
			break;
		}
	}

	GaussianProcess::Basis out;
	out.eigenvalues.resize(dc);
	out.eigenvectors.resize((size_t)N * dc);

	for (int d = 0; d < dc; d++)
	{
		const int col = N - 1 - d;
		const double ev = es.eigenvalues()[col];

		// Canonicalize eigenvector sign for deterministic reproducibility
		int p_max = 0;
		double v_max = 0.0;
		for (int p = 0; p < N; p++)
		{
			const double av = std::abs(es.eigenvectors()(p, col));
			if (av > v_max)
			{
				v_max = av;
				p_max = p;
			}
		}
		const double sign = (es.eigenvectors()(p_max, col) < 0.0) ? -1.0 : 1.0;
		const double scale = sign * (scaleSqrt ? sqrt(ev) : 1.0);

		for (int p = 0; p < N; p++)
		{
			out.eigenvectors[(size_t)dc * p + d] = scale * es.eigenvectors()(p, col);
		}
		out.eigenvalues[d] = ev;
	}

	return out;
}

GaussianProcess::Basis GaussianProcess::getBasis(
		const Matrix2D<double> &C, int maxDims, double eps)
{
	return computeSymmetricBasis(C, maxDims, eps, true);
}

GaussianProcess::Basis GaussianProcess::getUnitBasis(
		const Matrix2D<double> &K_unit, int maxDims, double eps)
{
	return computeSymmetricBasis(K_unit, maxDims, eps, false);
}

SquareExponentialKernel::SquareExponentialKernel(double sig_vel, double sig_div)
:	sig_vel(sig_vel),
	sig_div(sig_div)
{}

double SquareExponentialKernel::covariance(gravis::d3Vector a, gravis::d3Vector b)
{
	double d2 = (a - b).norm2() / (sig_div * sig_div);
	
	return sig_vel * sig_vel * exp(-d2);
}

ExponentialKernel::ExponentialKernel(double sig_vel, double sig_div)
:	sig_vel(sig_vel),
	sig_div(sig_div)
{}

double ExponentialKernel::covariance(gravis::d3Vector a, gravis::d3Vector b)
{
	double d = (a - b).length() / sig_div;
	
	return sig_vel * sig_vel * exp(-d);
}
