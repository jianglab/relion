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

#ifndef ALIGNMENT_SET_H
#define ALIGNMENT_SET_H

#include <src/image.h>
#include <src/metadata_table.h>
#include <src/jaz/gravis/t2Vector.h>
#include <src/jaz/gravis/t3Vector.h>
#include <vector>
#include <omp.h>


template<class T>
class AlignmentSet
{
    public:

        AlignmentSet();
        AlignmentSet(const std::vector<MetaDataTable>& mdts,
                     int fc, int s, int k0, int k1);

            int mc, fc, s, sh, k0, k1, accPix;

            // micrograph < particle < frame <pixels> > >
            std::vector<std::vector<std::vector< Image<T> >>> CCs;
            // micrograph < particle < frame <pixels> > >
            std::vector<std::vector<std::vector< std::vector<gravis::t2Vector<T>> >>> obs;
            // micrograph < particle <pixels> >
            std::vector<std::vector< std::vector<gravis::t2Vector<T>> >> pred;
            // frame <pixels>
            std::vector< std::vector<double> > damage;

            std::vector<std::vector<gravis::d2Vector>> positions;
            std::vector<std::vector<std::vector<gravis::d2Vector>>> initialTracks;
            std::vector<std::vector<gravis::d2Vector>> globComp;

            std::vector<gravis::t2Vector<int>> accCoords;
            std::vector<int> rowStart, accX;
            int maxAccX;
            bool tscPrepared;
            std::vector<double> constW1, constW2;
            std::vector<std::vector<std::vector< std::vector<gravis::t2Vector<T>> >>> cross;

        void prepareTsc(int threads = 1);


		template<class T2>
        void copyCC(int m, int p, int f, const Image<T2>& src);
		
        void accelerate(const Image<Complex>& img, std::vector<gravis::t2Vector<T>>& dest);
        void accelerate(const Image<RFLOAT>& img, std::vector<double>& dest);

        gravis::d3Vector updateTsc(
            const std::vector<std::vector<gravis::d2Vector>>& tracks,
            int mg, int threads);
};

template<class T>
AlignmentSet<T>::AlignmentSet()
:   mc(0), fc(0), s(0), sh(0), k0(0), k1(0), accPix(0), maxAccX(0), tscPrepared(false)
{
}

template<class T>
AlignmentSet<T>::AlignmentSet(
        const std::vector<MetaDataTable> &mdts,
        int fc, int s, int k0, int k1)
:   mc(mdts.size()),
    fc(fc),
    s(s),
    sh(s/2+1),
    k0(k0),
    k1(k1),
    maxAccX(0),
    tscPrepared(false)
{
    accCoords.reserve(sh*s);
    accX.reserve(sh*s);
    rowStart.assign(s + 1, 0);

    int num = 0;

    for (int y = 0; y < s; y++)
    {
        rowStart[y] = num;
        for (int x = 0; x < sh; x++)
        {
            const double xx = x;
            const double yy = y < sh? y : y - s;

            int r = ROUND(sqrt(xx*xx + yy*yy));

            if (r >= k0 && r < k1)
            {
                accCoords.push_back(gravis::t2Vector<int>(x,y));
                accX.push_back(x);
                if (x > maxAccX) maxAccX = x;
                num++;
            }
        }
    }
    rowStart[s] = num;

    accPix = num;

    CCs.resize(mc);
    obs.resize(mc);
    pred.resize(mc);

    positions.resize(mc);
    initialTracks.resize(mc);
    globComp.resize(mc);

    for (int m = 0; m < mc; m++)
    {
        const int pc = mdts[m].numberOfObjects();

        positions[m].resize(pc);
        globComp[m].resize(fc);

        initialTracks[m].resize(pc);
        CCs[m].resize(pc);
        obs[m].resize(pc);
        pred[m].resize(pc);

        for (int p = 0; p < pc; p++)
        {
            initialTracks[m][p].resize(fc);
            pred[m][p].resize(accPix);

            CCs[m][p].resize(fc);
            obs[m][p].resize(fc);

            for (int f = 0; f < fc; f++)
            {
                obs[m][p][f].resize(accPix);
            }
        }
    }

    damage.resize(fc);

    for (int f = 0; f < fc; f++)
    {
        damage[f].resize(accPix);
    }
}

template<class T>
template<class T2>
void AlignmentSet<T>::copyCC(int m, int p, int f, const Image<T2> &src)
{
    if (m < 0 || m >= mc ||
        p < 0 || p >= CCs[m].size() ||
        f < 0 || f >= fc)
    {
        REPORT_ERROR_STR("AlignmentSet::copyCC: bad CC-index: "
            << m << ", " << p << ", " << f << " for "
            << mc << ", " << ((m >= 0 && m < mc)? CCs[m].size() : 0) << ", " << fc << ".");
    }

	CCs[m][p][f] = Image<T>(src.data.xdim, src.data.ydim);
	
    for (int y = 0; y < src.data.ydim; y++)
    for (int x = 0; x < src.data.xdim; x++)
    {
        CCs[m][p][f](y,x) = (T)src(y,x);
    }
}

template<class T>
void AlignmentSet<T>::accelerate(
        const Image<Complex> &img,
        std::vector<gravis::t2Vector<T>>& dest)
{
    for (int i = 0; i < accPix; i++)
    {
        gravis::t2Vector<int> c = accCoords[i];

        Complex z = img(c.y, c.x);

        dest[i] = gravis::t2Vector<T>(z.real, z.imag);
    }
}

template<class T>
void AlignmentSet<T>::accelerate(const Image<RFLOAT> &img, std::vector<double>& dest)
{
    for (int i = 0; i < accPix; i++)
    {
        gravis::t2Vector<int> c = accCoords[i];
        dest[i] = img(c.y, c.x);
    }
}

template<class T>
void AlignmentSet<T>::prepareTsc(int threads)
{
    constW1.assign(mc, 0.0);
    constW2.assign(mc, 0.0);
    cross.resize(mc);

    for (int mg = 0; mg < mc; mg++)
    {
        const int pc = pred[mg].size();
        cross[mg].resize(pc);
        std::vector<double> w1_p(pc, 0.0), w2_p(pc, 0.0);

        #pragma omp parallel for num_threads(threads) if(threads > 1 && pc > 1)
        for (int p = 0; p < pc; p++)
        {
            cross[mg][p].resize(fc);
            const std::vector<gravis::t2Vector<T>>& pred_p = pred[mg][p];
            double w1 = 0.0, w2 = 0.0;

            for (int f = 0; f < fc; f++)
            {
                cross[mg][p][f].resize(accPix);
                const std::vector<gravis::t2Vector<T>>& obs_pf = obs[mg][p][f];
                std::vector<gravis::t2Vector<T>>& cross_pf = cross[mg][p][f];
                const std::vector<double>& dmg_f = damage[f];

                for (int i = 0; i < accPix; i++)
                {
                    const double dmg = dmg_f[i];
                    const double c = (double) obs_pf[i].x;
                    const double d = (double) obs_pf[i].y;
                    const double pr = (double) pred_p[i].x;
                    const double pi = (double) pred_p[i].y;

                    w1 += dmg * (c * c + d * d);
                    w2 += dmg * (pr * pr + pi * pi);
                    cross_pf[i] = gravis::t2Vector<T>((T)(dmg * (pr * c + pi * d)),
                                                     (T)(dmg * (pi * c - pr * d)));
                }
            }
            w1_p[p] = w1;
            w2_p[p] = w2;
        }

        double s1 = 0.0, s2 = 0.0;
        for (int p = 0; p < pc; p++)
        {
            s1 += w1_p[p];
            s2 += w2_p[p];
        }
        constW1[mg] = s1;
        constW2[mg] = s2;

        std::vector<std::vector<std::vector<gravis::t2Vector<T>>>>().swap(obs[mg]);
        std::vector<std::vector<gravis::t2Vector<T>>>().swap(pred[mg]);
    }

    tscPrepared = true;
}

template<class T>
gravis::d3Vector AlignmentSet<T>::updateTsc(
    const std::vector<std::vector<gravis::d2Vector>>& tracks,
    int mg, int threads)
{
    const int pc = tracks.size();

    if (tscPrepared)
    {
        std::vector<double> cc_p(pc, 0.0);
        const int x_dim = maxAccX + 1;

        #pragma omp parallel num_threads(threads) if(threads > 1 && pc > 1)
        {
            std::vector<double> cx(x_dim), sx(x_dim);

            #pragma omp for schedule(static)
            for (int p = 0; p < pc; p++)
            {
                double p_sum = 0.0;

                for (int f = 0; f < fc; f++)
                {
                    const gravis::d2Vector shift = tracks[p][f] / s;
                    const double two_pi_xs = 2.0 * PI * shift.x;
                    const double two_pi_ys = 2.0 * PI * shift.y;

                    for (int x = 0; x < x_dim; x++)
                    {
                        SINCOS(two_pi_xs * x, &sx[x], &cx[x]);
                    }

                    const gravis::t2Vector<T>* __restrict cr = cross[mg][p][f].data();

                    for (int y = 0; y < s; y++)
                    {
                        const int r0 = rowStart[y];
                        const int r1 = rowStart[y + 1];
                        if (r0 == r1) continue;

                        const double yy = y < sh ? y : y - s;
                        double cy, sy;
                        SINCOS(two_pi_ys * yy, &sy, &cy);

                        double row_re = 0.0, row_im = 0.0;
                        for (int idx = r0; idx < r1; idx++)
                        {
                            const int x = accX[idx];
                            const double wr = (double)cr[idx].x;
                            const double wi = (double)cr[idx].y;
                            row_re += cx[x] * wr + sx[x] * wi;
                            row_im += cx[x] * wi - sx[x] * wr;
                        }
                        p_sum += cy * row_re + sy * row_im;
                    }
                }
                cc_p[p] = p_sum;
            }
        }

        double cc_tot = 0.0;
        for (int p = 0; p < pc; p++)
        {
            cc_tot += cc_p[p];
        }

        return gravis::d3Vector(cc_tot, constW1[mg], constW2[mg]);
    }

    std::vector<gravis::d3Vector> outT(pc, gravis::d3Vector(0.0, 0.0, 0.0));

    #pragma omp parallel for num_threads(threads) if(threads > 1 && pc > 1)
    for (int p = 0; p < pc; p++)
    {
        gravis::d3Vector acc(0.0, 0.0, 0.0);
        for (int f = 0; f < fc; f++)
        {
            const gravis::d2Vector shift = tracks[p][f] / s;

            for (int i = 0; i < accPix; i++)
            {
                gravis::t2Vector<int> accCoord = accCoords[i];

                double x = accCoord.x;
                double y = accCoord.y < sh? accCoord.y : accCoord.y - s;

                const double dotp = 2 * PI * (x * shift.x + y * shift.y);

                double a, b;

                SINCOS(dotp, &b, &a);

                const gravis::t2Vector<T> z_obs_t2 = obs[mg][p][f][i];
                const double c = (double) z_obs_t2.x;
                const double d = (double) z_obs_t2.y;

                const double ac = a * c;
                const double bd = b * d;

                const double ab_cd = (a + b) * (c + d);

                const dComplex z_obs(ac - bd, ab_cd - ac - bd);

                const gravis::t2Vector<T> z_pred_t2 = pred[mg][p][i];
                const dComplex z_pred((double) z_pred_t2.x, (double) z_pred_t2.y);

                const double dmg = damage[f][i];

                acc[0] += dmg * (z_pred.real * z_obs.real + z_pred.imag * z_obs.imag);
                acc[1] += dmg * z_obs.norm();
                acc[2] += dmg * z_pred.norm();
            }
        }
        outT[p] = acc;
    }

    gravis::d3Vector out(0.0, 0.0, 0.0);

    for (int p = 0; p < pc; p++)
    {
        out += outT[p];
    }

    return out;
}



#endif
