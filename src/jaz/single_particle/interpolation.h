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

#ifndef JAZ_INTERPOLATION_H
#define JAZ_INTERPOLATION_H

#include <src/image.h>
#include <src/fftw.h>
#include <src/jaz/gravis/t2Vector.h>
#include <src/jaz/gravis/t4Matrix.h>

#define INTERPOL_WRAP(i,n) ((i) >= 0 ? (i) % (n) : -(i) % (n) ? (n) - (-(i) % (n)) : 0)

class Interpolation
{
    public:

        static bool isInSlice(const Image<RFLOAT>& img, double x, double y);
        static double getTaperWeight(const Image<RFLOAT>& img, double x, double y, double rx, double ry);

        static double linearXY(const Image<RFLOAT>& img, double x, double y, int n);
        static Complex linear3D(const Image<Complex>& img, double x, double y, double z);
        static Complex linearFFTW3D(const Image<Complex>& img, double x, double y, double z);
        static Complex linearFFTW2D(const Image<Complex>& img, double x, double y);

        template<typename T>
        static gravis::d2Vector quadraticMaxXY(const Image<T>& img, double eps = 1e-25)
        {
            const int w = img.data.xdim;
            const int h = img.data.ydim;

            int xmax = -1, ymax = -1;
            double vmax = 0.0;

            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                T v = DIRECT_A2D_ELEM(img.data, y, x);

                if (xmax < 0 || v > vmax)
                {
                    vmax = v;
                    xmax = x;
                    ymax = y;
                }
            }

            gravis::d2Vector p(xmax, ymax);

            if (xmax > 0 && xmax < w-1)
            {
                const T vp = DIRECT_A2D_ELEM(img.data, ymax, xmax+1);
                const T vn = DIRECT_A2D_ELEM(img.data, ymax, xmax-1);

                if (std::abs(vp + vn - 2.0*vmax) > eps)
                {
                    p.x -= 0.5 * (vp - vn) / (vp + vn - 2.0*vmax);
                }
            }

            if (xmax > 0 && xmax < w-1)
            {
                const T vp = DIRECT_A2D_ELEM(img.data, ymax+1, xmax);
                const T vn = DIRECT_A2D_ELEM(img.data, ymax-1, xmax);

                if (std::abs(vp + vn - 2.0*vmax) > eps)
                {
                    p.y -= 0.5 * (vp - vn) / (vp + vn - 2.0*vmax);
                }
            }

            return p;
        }

        template<typename T>
        static gravis::d2Vector quadraticMaxWrapXY(
				const Image<T>& img, double eps = 1e-25,
				int wMax = -1, int hMax = -1)
        {
            const int w = img.data.xdim;
            const int h = img.data.ydim;
			
			if (wMax < 0) wMax = w;
			if (hMax < 0) hMax = h;

            int xmax = -1, ymax = -1;
            double vmax = 0.0;

            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
				if ((x > wMax && w - x > wMax)
				  || y > hMax && h - x > hMax)
				{
					continue;
				}
					
                T v = DIRECT_A2D_ELEM(img.data, y, x);

                if (xmax < 0 || v > vmax)
                {
                    vmax = v;
                    xmax = x;
                    ymax = y;
                }
            }

            gravis::d2Vector p(xmax, ymax);

            {
                const T vp = DIRECT_A2D_ELEM(img.data, ymax, (xmax+1)%w);
                const T vn = DIRECT_A2D_ELEM(img.data, ymax, (xmax-1+w)%w);

                if (std::abs(vp + vn - 2.0*vmax) > eps)
                {
                    p.x -= 0.5 * (vp - vn) / (vp + vn - 2.0*vmax);
                }
            }

            {
                const T vp = DIRECT_A2D_ELEM(img.data, (ymax+1)%w, xmax);
                const T vn = DIRECT_A2D_ELEM(img.data, (ymax-1+w)%w, xmax);

                if (std::abs(vp + vn - 2.0*vmax) > eps)
                {
                    p.y -= 0.5 * (vp - vn) / (vp + vn - 2.0*vmax);
                }
            }

            return p;
        }

        static double cubic1D(double y0, double y1, double y2, double y3, double t);

        static inline void cubicPosWeights(double t, double w[4])
        {
            const double t2 = t * t;
            const double t3 = t2 * t;
            w[0] = -0.5 * t3 + t2 - 0.5 * t;
            w[1] =  1.5 * t3 - 2.5 * t2 + 1.0;
            w[2] = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
            w[3] =  0.5 * t3 - 0.5 * t2;
        }

        static inline void cubicWeights(double t, double w[4], double wd[4])
        {
            const double t2 = t * t;
            const double t3 = t2 * t;
            w[0] = -0.5 * t3 + t2 - 0.5 * t;
            w[1] =  1.5 * t3 - 2.5 * t2 + 1.0;
            w[2] = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
            w[3] =  0.5 * t3 - 0.5 * t2;
            wd[0] = -1.5 * t2 + 2.0 * t - 0.5;
            wd[1] =  4.5 * t2 - 5.0 * t;
            wd[2] = -4.5 * t2 + 4.0 * t + 0.5;
            wd[3] =  1.5 * t2 - t;
        }

        static inline void cubicIndices(int xi, int yi, int w, int h, bool wrap, int ix[4], int iy[4])
        {
            if (wrap)
            {
                if (w >= 4)
                {
                    const int ix0 = INTERPOL_WRAP(xi - 1, w);
                    ix[0] = ix0;
                    ix[1] = (ix0 + 1 < w) ? ix0 + 1 : ix0 + 1 - w;
                    ix[2] = (ix0 + 2 < w) ? ix0 + 2 : ix0 + 2 - w;
                    ix[3] = (ix0 + 3 < w) ? ix0 + 3 : ix0 + 3 - w;
                }
                else
                {
                    for (int k = 0; k < 4; k++) ix[k] = INTERPOL_WRAP(xi - 1 + k, w);
                }

                if (h >= 4)
                {
                    const int iy0 = INTERPOL_WRAP(yi - 1, h);
                    iy[0] = iy0;
                    iy[1] = (iy0 + 1 < h) ? iy0 + 1 : iy0 + 1 - h;
                    iy[2] = (iy0 + 2 < h) ? iy0 + 2 : iy0 + 2 - h;
                    iy[3] = (iy0 + 3 < h) ? iy0 + 3 : iy0 + 3 - h;
                }
                else
                {
                    for (int k = 0; k < 4; k++) iy[k] = INTERPOL_WRAP(yi - 1 + k, h);
                }
            }
            else
            {
                const int wm1 = w - 1;
                const int hm1 = h - 1;
                for (int k = 0; k < 4; k++)
                {
                    const int xx = xi - 1 + k;
                    const int yy = yi - 1 + k;
                    ix[k] = xx < 0 ? 0 : (xx > wm1 ? wm1 : xx);
                    iy[k] = yy < 0 ? 0 : (yy > hm1 ? hm1 : yy);
                }
            }
        }

        template<typename T>
        static void cubicXYValAndGrad(
                const Image<T>& img, double x, double y,
                double& val, gravis::d2Vector& grad,
                int z = 0, int n = 0, bool wrap = false)
        {
            const int xi = (int)std::floor(x);
            const int yi = (int)std::floor(y);
            int ix[4], iy[4];
            cubicIndices(xi, yi, img.data.xdim, img.data.ydim, wrap, ix, iy);

            double wx[4], wxd[4], wy[4], wyd[4];
            cubicWeights(x - xi, wx, wxd);
            cubicWeights(y - yi, wy, wyd);

            const int ix0 = ix[0], ix1 = ix[1], ix2 = ix[2], ix3 = ix[3];
            double v = 0.0, gx = 0.0, gy = 0.0;
            for (int c = 0; c < 4; c++)
            {
                const T* row = &DIRECT_NZYX_ELEM(img.data, n, z, iy[c], 0);
                const double f0 = (double)row[ix0];
                const double f1 = (double)row[ix1];
                const double f2 = (double)row[ix2];
                const double f3 = (double)row[ix3];
                const double rc  = wx[0]*f0  + wx[1]*f1  + wx[2]*f2  + wx[3]*f3;
                const double rdc = wxd[0]*f0 + wxd[1]*f1 + wxd[2]*f2 + wxd[3]*f3;
                v  += wy[c]  * rc;
                gx += wy[c]  * rdc;
                gy += wyd[c] * rc;
            }
            val = v;
            grad.x = gx;
            grad.y = gy;
        }

        template<typename T>
        static T cubicXY(const Image<T>& img, double x, double y, int z = 0, int n = 0, bool wrap = false)
        {
            const int xi = (int)std::floor(x);
            const int yi = (int)std::floor(y);
            int ix[4], iy[4];
            cubicIndices(xi, yi, img.data.xdim, img.data.ydim, wrap, ix, iy);

            double wx[4], wy[4];
            cubicPosWeights(x - xi, wx);
            cubicPosWeights(y - yi, wy);

            const int ix0 = ix[0], ix1 = ix[1], ix2 = ix[2], ix3 = ix[3];
            double v = 0.0;
            for (int c = 0; c < 4; c++)
            {
                const T* row = &DIRECT_NZYX_ELEM(img.data, n, z, iy[c], 0);
                const double rc = wx[0]*(double)row[ix0] + wx[1]*(double)row[ix1]
                                + wx[2]*(double)row[ix2] + wx[3]*(double)row[ix3];
                v += wy[c] * rc;
            }
            return (T)v;
        }

        template<typename T>
        static gravis::t2Vector<T> cubicXYgrad(const Image<T>& img, double x, double y, int z = 0, int n = 0, bool wrap = false)
        {
            double dummy;
            gravis::d2Vector g;
            cubicXYValAndGrad(img, x, y, dummy, g, z, n, wrap);
            return gravis::t2Vector<T>((T)g.x, (T)g.y);
        }

        static void test2D();
};

#endif
