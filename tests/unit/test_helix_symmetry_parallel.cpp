/*
 * tests/unit/test_helix_symmetry_parallel.cpp
 *
 * imposeHelicalSymmetryInRealSpace() runs over threads; its result must stay bitwise identical to
 * the original single-threaded loop, kept here verbatim (renamed) as the reference.
 * BackProjector::applyHelicalSymmetry() (Fourier space) likewise must not depend on the thread count.
 */

#include <catch2/catch.hpp>

#include "src/helix.h"
#include "src/backprojector.h"
#include "src/macros.h"
#include "src/gpu_reconstruct.h"

#include <omp.h>
#include <random>

namespace
{
// The implementation before the change, verbatim except for its name
static void referenceImposeHelicalSymmetry(
		MultidimArray<RFLOAT>& v,
		RFLOAT pixel_size_A,
		RFLOAT sphere_radius_A,
		RFLOAT cyl_inner_radius_A,
		RFLOAT cyl_outer_radius_A,
		RFLOAT z_percentage,
		RFLOAT rise_A,
		RFLOAT twist_deg,
		RFLOAT cosine_width_pix)
{
	bool ignore_helical_symmetry = false;
	long int Xdim, Ydim, Zdim, Ndim, box_len;
	RFLOAT rise_pix, sphere_radius_pix, cyl_inner_radius_pix, cyl_outer_radius_pix, r_min, r_max, d_min, d_max, D_min, D_max, z_min, z_max;

	int rec_len;
	std::vector<RFLOAT> sin_rec, cos_rec;
	MultidimArray<RFLOAT> vout;

	if (v.getDim() != 3)
		REPORT_ERROR("helix.cpp::imposeHelicalSymmetryInRealSpace(): Input helical reference is not 3D! (vol.getDim() = " + integerToString(v.getDim()) + ")");
	v.getDimensions(Xdim, Ydim, Zdim, Ndim);
	box_len = (Xdim < Ydim) ? Xdim : Ydim;
	box_len = (box_len < Zdim) ? box_len : Zdim;

	// Check helical parameters
	checkParametersFor3DHelicalReconstruction(
			false,
			false,
			1,
			rise_A,
			rise_A,
			rise_A,
			twist_deg,
			twist_deg,
			twist_deg,
			box_len,
			pixel_size_A,
			z_percentage,
			sphere_radius_A * 2.,
			cyl_inner_radius_A * 2.,
			cyl_outer_radius_A * 2.);

	// Parameters of mask
	if (cosine_width_pix < 1.)
		cosine_width_pix = 1.;  // Avoid 'divided by 0' error
	rise_pix = fabs(rise_A / pixel_size_A);  // Keep helical rise as a positive number
	sphere_radius_pix = sphere_radius_A / pixel_size_A;
	cyl_inner_radius_pix = cyl_inner_radius_A / pixel_size_A;
	cyl_outer_radius_pix = cyl_outer_radius_A / pixel_size_A;
	r_min = sphere_radius_pix;
	r_max = sphere_radius_pix + cosine_width_pix;
	d_min = cyl_inner_radius_pix - cosine_width_pix;
	d_max = cyl_inner_radius_pix;
	D_min = cyl_outer_radius_pix;
	D_max = cyl_outer_radius_pix + cosine_width_pix;

	// Crop the central slices
	v.setXmippOrigin();
	z_max = (((RFLOAT)(Zdim)) * z_percentage) * 0.5;
	if (z_max > (((RFLOAT)(FINISHINGZ(v))) - 1.))
		z_max = (((RFLOAT)(FINISHINGZ(v))) - 1.);
	z_min = -z_max;
	if (z_min < (((RFLOAT)(STARTINGZ(v))) + 1.))
		z_min = (((RFLOAT)(STARTINGZ(v))) + 1.);

	// Init volumes
	v.setXmippOrigin();
	vout.clear();
	vout.resize(v);
	vout.setXmippOrigin();

	// Calculate tabulated sine and cosine values
	rec_len = 2 + (CEIL((RFLOAT(Zdim) + 2.) / rise_pix));
	sin_rec.clear();
	cos_rec.clear();
	sin_rec.resize(rec_len);
	cos_rec.resize(rec_len);
	for (int id = 0; id < rec_len; id++)
#ifdef RELION_SINGLE_PRECISION
		SINCOSF(DEG2RAD(((RFLOAT)(id)) * twist_deg), &sin_rec[id], &cos_rec[id]);
#else
		SINCOS(DEG2RAD(((RFLOAT)(id)) * twist_deg), &sin_rec[id], &cos_rec[id]);
#endif

	FOR_ALL_ELEMENTS_IN_ARRAY3D(v)
	{
		// Out of the mask
		RFLOAT dd = (RFLOAT)(i * i + j * j);
		RFLOAT rr = dd + (RFLOAT)(k * k);
		RFLOAT d = sqrt(dd);
		RFLOAT r = sqrt(rr);
		if ( (r > r_max) || (d < d_min) || (d > D_max) )
		{
			A3D_ELEM(v, k, i, j) = 0.;
			continue;
		}

		// How many voxels should be used to calculate the average?
		RFLOAT zi = (RFLOAT)(k);
		RFLOAT yi = (RFLOAT)(i);
		RFLOAT xi = (RFLOAT)(j);
		int rot_max = -(CEIL((zi - z_max) / rise_pix));
		int rot_min = -(FLOOR((zi - z_min) / rise_pix));
		if (rot_max < rot_min)
			REPORT_ERROR("helix.cpp::makeHelicalReferenceInRealSpace(): ERROR in imposing symmetry!");

		// Do the average
		RFLOAT pix_sum, pix_weight;
		pix_sum = pix_weight = 0.;
		for (int id = rot_min; id <= rot_max; id++)
		{
			// Get the sine and cosine value
			RFLOAT sin_val, cos_val;
			if (id >= 0)
			{
				sin_val = sin_rec[id];
				cos_val = cos_rec[id];
			}
			else
			{
				sin_val = (-1.) * sin_rec[-id];
				cos_val = cos_rec[-id];
			}

			// Get the voxel coordinates
			RFLOAT zp = zi + ((RFLOAT)(id)) * rise_pix;
			RFLOAT yp = xi * sin_val + yi * cos_val;
			RFLOAT xp = xi * cos_val - yi * sin_val;

			// Trilinear interpolation (with physical coords)
			// Subtract STARTINGY and STARTINGZ to accelerate access to data (STARTINGX=0)
			// In that way use DIRECT_A3D_ELEM, rather than A3D_ELEM
			int x0, y0, z0, x1, y1, z1;
			RFLOAT fx, fy, fz;
			x0 = FLOOR(xp); fx = xp - x0; x0 -= STARTINGX(v); x1 = x0 + 1;
			y0 = FLOOR(yp); fy = yp - y0; y0 -= STARTINGY(v); y1 = y0 + 1;
			z0 = FLOOR(zp); fz = zp - z0; z0 -= STARTINGZ(v); z1 = z0 + 1;

			RFLOAT d000, d001, d010, d011, d100, d101, d110, d111;
			d000 = DIRECT_A3D_ELEM(v, z0, y0, x0);
			d001 = DIRECT_A3D_ELEM(v, z0, y0, x1);
			d010 = DIRECT_A3D_ELEM(v, z0, y1, x0);
			d011 = DIRECT_A3D_ELEM(v, z0, y1, x1);
			d100 = DIRECT_A3D_ELEM(v, z1, y0, x0);
			d101 = DIRECT_A3D_ELEM(v, z1, y0, x1);
			d110 = DIRECT_A3D_ELEM(v, z1, y1, x0);
			d111 = DIRECT_A3D_ELEM(v, z1, y1, x1);

			RFLOAT dx00, dx01, dx10, dx11;
			dx00 = LIN_INTERP(fx, d000, d001);
			dx01 = LIN_INTERP(fx, d100, d101);
			dx10 = LIN_INTERP(fx, d010, d011);
			dx11 = LIN_INTERP(fx, d110, d111);

			RFLOAT dxy0, dxy1;
			dxy0 = LIN_INTERP(fy, dx00, dx10);
			dxy1 = LIN_INTERP(fy, dx01, dx11);

			pix_sum += LIN_INTERP(fz, dxy0, dxy1);
			pix_weight += 1.;
		}

		if (pix_weight > 0.9)
		{
			A3D_ELEM(vout, k, i, j) = pix_sum / pix_weight;

			if ( (d > d_max) && (d < D_min) && (r < r_min) )
			{}
			else // The pixel is within cosine edge(s)
			{
				pix_weight = 1.;
				if (d < d_max)  // d_min < d < d_max : w=(0~1)
					pix_weight = 0.5 + (0.5 * cos(PI * ((d_max - d) / cosine_width_pix)));
				else if (d > D_min) // D_min < d < D_max : w=(1~0)
					pix_weight = 0.5 + (0.5 * cos(PI * ((d - D_min) / cosine_width_pix)));
				if (r > r_min) // r_min < r < r_max
				{
					pix_sum = 0.5 + (0.5 * cos(PI * ((r - r_min) / cosine_width_pix)));
					pix_weight = (pix_sum < pix_weight) ? (pix_sum) : (pix_weight);
				}
				A3D_ELEM(vout, k, i, j) *= pix_weight;
			}
		}
		else
			A3D_ELEM(vout, k, i, j) = 0.;
    }

	// Copy and exit
	v = vout;
	sin_rec.clear();
	cos_rec.clear();
	vout.clear();

	return;
};


// The Fourier-space implementation before the per-plane cos/sin change, verbatim except for its name
struct ReferenceBackProjector : public BackProjector
{
	using BackProjector::BackProjector;
	void referenceApplyHelicalSymmetry(int nr_helical_asu, RFLOAT helical_twist, RFLOAT helical_rise)
	{
		if (rect)
		{
			REPORT_ERROR("rect not supported in reference");
			return;
		}
		if ( (nr_helical_asu < 2) || (ref_dim != 3) )
			return;

		int rmax2 = ROUND(r_max * padding_factor) * ROUND(r_max * padding_factor);

		Matrix2D<RFLOAT> R(4, 4); // A matrix from the list
		MultidimArray<RFLOAT> sum_weight;
		MultidimArray<Complex > sum_data;
		RFLOAT x, y, z, fx, fy, fz, xp, yp, zp, r2;
		bool is_neg_x;
		int x0, x1, y0, y1, z0, z1;
		Complex d000, d001, d010, d011, d100, d101, d110, d111;
		Complex dx00, dx01, dx10, dx11, dxy0, dxy1, ddd;
		RFLOAT dd000, dd001, dd010, dd011, dd100, dd101, dd110, dd111;
		RFLOAT ddx00, ddx01, ddx10, ddx11, ddxy0, ddxy1;

		// First symmetry operator (not stored in SL) is the identity matrix
		sum_weight = weight;
		sum_data = data;
		int h_min = -nr_helical_asu/2;
		int h_max = -h_min + nr_helical_asu%2;
		for (int hh = h_min; hh < h_max; hh++)
		{
			if (hh != 0) // h==0 is done before the for loop (where sum_data = data)
			{
				RFLOAT rot_ang = hh * (-helical_twist);
				rotation3DMatrix(rot_ang, 'Z', R);
				R.setSmallValuesToZero(); // TODO: invert rotation matrix?

				// Loop over all points in the output (i.e. rotated, or summed) array
				FOR_ALL_ELEMENTS_IN_ARRAY3D(sum_weight)
				{
					x = (RFLOAT)j; // STARTINGX(sum_weight) is zero!
					y = (RFLOAT)i;
					z = (RFLOAT)k;
					r2 = x*x + y*y + z*z;
					if (r2 <= rmax2)
					{
						// coords_output(x,y) = A * coords_input (xp,yp)
						xp = x * R(0, 0) + y * R(0, 1) + z * R(0, 2);
						yp = x * R(1, 0) + y * R(1, 1) + z * R(1, 2);
						zp = x * R(2, 0) + y * R(2, 1) + z * R(2, 2);

						// Only asymmetric half is stored
						if (xp < 0)
						{
							// Get complex conjugated hermitian symmetry pair
							xp = -xp;
							yp = -yp;
							zp = -zp;
							is_neg_x = true;
						}
						else
						{
							is_neg_x = false;
						}

						// Trilinear interpolation (with physical coords)
						// Subtract STARTINGY and STARTINGZ to accelerate access to data (STARTINGX=0)
						// In that way use DIRECT_A3D_ELEM, rather than A3D_ELEM
						x0 = FLOOR(xp);
						fx = xp - x0;
						x1 = x0 + 1;

						y0 = FLOOR(yp);
						fy = yp - y0;
						y0 -=  STARTINGY(data);
						y1 = y0 + 1;

						z0 = FLOOR(zp);
						fz = zp - z0;
						z0 -= STARTINGZ(data);
						z1 = z0 + 1;

	#ifdef CHECK_SIZE
						if (x0 < 0 || y0 < 0 || z0 < 0 ||
							x1 < 0 || y1 < 0 || z1 < 0 ||
							x0 >= XSIZE(data) || y0  >= YSIZE(data) || z0 >= ZSIZE(data) ||
							x1 >= XSIZE(data) || y1  >= YSIZE(data)  || z1 >= ZSIZE(data) 	)
						{
							std::cerr << " x0= " << x0 << " y0= " << y0 << " z0= " << z0 << std::endl;
							std::cerr << " x1= " << x1 << " y1= " << y1 << " z1= " << z1 << std::endl;
							data.printShape();
							REPORT_ERROR("BackProjector::applyPointGroupSymmetry: checksize!!!");
						}
	#endif
						// First interpolate (complex) data
						d000 = DIRECT_A3D_ELEM(data, z0, y0, x0);
						d001 = DIRECT_A3D_ELEM(data, z0, y0, x1);
						d010 = DIRECT_A3D_ELEM(data, z0, y1, x0);
						d011 = DIRECT_A3D_ELEM(data, z0, y1, x1);
						d100 = DIRECT_A3D_ELEM(data, z1, y0, x0);
						d101 = DIRECT_A3D_ELEM(data, z1, y0, x1);
						d110 = DIRECT_A3D_ELEM(data, z1, y1, x0);
						d111 = DIRECT_A3D_ELEM(data, z1, y1, x1);

						dx00 = LIN_INTERP(fx, d000, d001);
						dx01 = LIN_INTERP(fx, d100, d101);
						dx10 = LIN_INTERP(fx, d010, d011);
						dx11 = LIN_INTERP(fx, d110, d111);
						dxy0 = LIN_INTERP(fy, dx00, dx10);
						dxy1 = LIN_INTERP(fy, dx01, dx11);

						// Take complex conjugated for half with negative x
						ddd = LIN_INTERP(fz, dxy0, dxy1);

						if (is_neg_x)
							ddd = conj(ddd);

						// Also apply a phase shift for helical translation along Z
						if (ABS(helical_rise) > 0.)
						{
							RFLOAT zshift = hh * helical_rise;
							zshift /= - ori_size * (RFLOAT)padding_factor;
							RFLOAT dotp = 2 * PI * (z * zshift);
							RFLOAT a = cos(dotp);
							RFLOAT b = sin(dotp);
							RFLOAT c = ddd.real;
							RFLOAT d = ddd.imag;
							RFLOAT ac = a * c;
							RFLOAT bd = b * d;
							RFLOAT ab_cd = (a + b) * (c + d);
							ddd = Complex(ac - bd, ab_cd - ac - bd);
						}
						// Accumulated sum of the data term
						A3D_ELEM(sum_data, k, i, j) += ddd;

						// Then interpolate (real) weight
						dd000 = DIRECT_A3D_ELEM(weight, z0, y0, x0);
						dd001 = DIRECT_A3D_ELEM(weight, z0, y0, x1);
						dd010 = DIRECT_A3D_ELEM(weight, z0, y1, x0);
						dd011 = DIRECT_A3D_ELEM(weight, z0, y1, x1);
						dd100 = DIRECT_A3D_ELEM(weight, z1, y0, x0);
						dd101 = DIRECT_A3D_ELEM(weight, z1, y0, x1);
						dd110 = DIRECT_A3D_ELEM(weight, z1, y1, x0);
						dd111 = DIRECT_A3D_ELEM(weight, z1, y1, x1);

						ddx00 = LIN_INTERP(fx, dd000, dd001);
						ddx01 = LIN_INTERP(fx, dd100, dd101);
						ddx10 = LIN_INTERP(fx, dd010, dd011);
						ddx11 = LIN_INTERP(fx, dd110, dd111);
						ddxy0 = LIN_INTERP(fy, ddx00, ddx10);
						ddxy1 = LIN_INTERP(fy, ddx01, ddx11);

						A3D_ELEM(sum_weight, k, i, j) +=  LIN_INTERP(fz, ddxy0, ddxy1);

					} // end if r2 <= rmax2
				 } // end loop over all elements of sum_weight
			} // end if hh!=0
		} // end loop over hh

		data = sum_data;
		weight = sum_weight;
	}
};

MultidimArray<RFLOAT> randomVolume(int n, unsigned seed)
{
	MultidimArray<RFLOAT> v(n, n, n);
	std::mt19937 rng(seed);
	std::normal_distribution<RFLOAT> g(0., 1.);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(v) DIRECT_MULTIDIM_ELEM(v, n) = g(rng);
	return v;
}
}

TEST_CASE("helical symmetry in real space: the same at any thread count as the original loop", "[helix_symmetry]")
{
	const int n = 48;
	const RFLOAT angpix = 2.;
	struct Case { RFLOAT sphere_A, inner_A, outer_A, z_frac, rise_A, twist; };
	for (const Case &c : {Case{44., -1., 30., 0.3, 4.75, -1.2}, Case{40., 8., 26., 0.3, 4., 30.},
	                      Case{46., -1., 20., 0.5, 2.38, 179.4}})
		for (int threads : {1, 3, 8})
		{
			INFO("twist " << c.twist << " rise " << c.rise_A << " threads " << threads);
			MultidimArray<RFLOAT> a = randomVolume(n, 11), b = a;
			imposeHelicalSymmetryInRealSpace(a, angpix, c.sphere_A, c.inner_A, c.outer_A, c.z_frac, c.rise_A, c.twist, 3., threads);
			referenceImposeHelicalSymmetry(b, angpix, c.sphere_A, c.inner_A, c.outer_A, c.z_frac, c.rise_A, c.twist, 3.);
			REQUIRE(a.sameShape(b));
			bool same = true;
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a)
				if (DIRECT_MULTIDIM_ELEM(a, n) != DIRECT_MULTIDIM_ELEM(b, n)) { same = false; break; }
			CHECK(same);
		}
}

TEST_CASE("helical symmetry in Fourier space: the same at any thread count", "[helix_symmetry]")
{
	const int ori = 40;
	auto run = [&](int threads)
	{
		BackProjector bp(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
		bp.initZeros(ori / 2);
		std::mt19937 rng(5);
		std::normal_distribution<RFLOAT> g(0., 1.);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bp.data)
			DIRECT_MULTIDIM_ELEM(bp.data, n) = Complex(g(rng), g(rng));
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bp.weight)
			DIRECT_MULTIDIM_ELEM(bp.weight, n) = 1. + 0.1 * g(rng);
		bp.applyHelicalSymmetry(7, -1.2, 4.75 / 2., threads);
		return bp;
	};
	const BackProjector a = run(1);
	for (int threads : {3, 8})
	{
		INFO("threads " << threads);
		const BackProjector b = run(threads);
		REQUIRE(a.data.sameShape(b.data));
		bool same = true;
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a.data)
			if (DIRECT_MULTIDIM_ELEM(a.data, n).real != DIRECT_MULTIDIM_ELEM(b.data, n).real ||
			    DIRECT_MULTIDIM_ELEM(a.data, n).imag != DIRECT_MULTIDIM_ELEM(b.data, n).imag ||
			    DIRECT_MULTIDIM_ELEM(a.weight, n) != DIRECT_MULTIDIM_ELEM(b.weight, n))
				{ same = false; break; }
		CHECK(same);
	}
}

TEST_CASE("helical symmetry in Fourier space: identical to the version with cos/sin per voxel", "[helix_symmetry]")
{
	const int ori = 40;
	for (RFLOAT rise : {4.75 / 2., 0.})
		for (int threads : {1, 4})
		{
			INFO("rise " << rise << " threads " << threads);
			auto fill = [&](BackProjector &bp)
			{
				bp.initZeros(ori / 2);
				std::mt19937 rng(5);
				std::normal_distribution<RFLOAT> g(0., 1.);
				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bp.data)
					DIRECT_MULTIDIM_ELEM(bp.data, n) = Complex(g(rng), g(rng));
				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bp.weight)
					DIRECT_MULTIDIM_ELEM(bp.weight, n) = 1. + 0.1 * g(rng);
			};
			ReferenceBackProjector ref(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
			BackProjector bp(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
			fill(ref);
			fill(bp);
			ref.referenceApplyHelicalSymmetry(7, -1.2, rise);
			bp.applyHelicalSymmetry(7, -1.2, rise, threads);
			REQUIRE(ref.data.sameShape(bp.data));
			bool same = true;
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(ref.data)
				if (DIRECT_MULTIDIM_ELEM(ref.data, n).real != DIRECT_MULTIDIM_ELEM(bp.data, n).real ||
				    DIRECT_MULTIDIM_ELEM(ref.data, n).imag != DIRECT_MULTIDIM_ELEM(bp.data, n).imag ||
				    DIRECT_MULTIDIM_ELEM(ref.weight, n) != DIRECT_MULTIDIM_ELEM(bp.weight, n))
					{ same = false; break; }
			CHECK(same);
		}
}

TEST_CASE("helical symmetry in Fourier space on the GPU: the same as the CPU", "[helix_symmetry][gpu_symmetrise]")
{
	if (!gpurec::deviceAvailable())
	{
		WARN("no GPU here: test skipped");
		return;
	}
	const int ori = 48;
	const double tol = sizeof(RFLOAT) == 8 ? 1e-12 : 1e-4;
	gpurec::setGpuRequested(true);
	setenv("RELION_GPU_SYMMETRISE", "1", 1);
	for (RFLOAT rise : {4.75 / 2., 0.})
		for (const char *planes : {"0", "5"}) // all planes at once, and groups of 5 planes
		{
			INFO("rise " << rise << " planes per group " << planes);
			setenv("RELION_GPU_SYMMETRISE_PLANES", planes, 1);
			auto fill = [&](BackProjector &bp)
			{
				bp.initZeros(ori / 2);
				std::mt19937 rng(11);
				std::normal_distribution<RFLOAT> g(0., 1.);
				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bp.data)
					DIRECT_MULTIDIM_ELEM(bp.data, n) = Complex(g(rng), g(rng));
				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bp.weight)
					DIRECT_MULTIDIM_ELEM(bp.weight, n) = 1. + 0.1 * g(rng);
			};
			ReferenceBackProjector ref(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
			BackProjector bp(ori, 3, "c1", TRILINEAR, 2, 10, 0, 1.9, 15, 2, false);
			fill(ref);
			fill(bp);
			ref.referenceApplyHelicalSymmetry(7, -1.2, rise);
			bp.applyHelicalSymmetry(7, -1.2, rise, 1);
			REQUIRE(ref.data.sameShape(bp.data));
			double max_diff = 0., max_val = 0.;
			long int exact = 0;
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(ref.data)
			{
				const Complex a = DIRECT_MULTIDIM_ELEM(ref.data, n), b = DIRECT_MULTIDIM_ELEM(bp.data, n);
				max_diff = std::max(max_diff, std::max(std::abs(a.real - b.real), std::abs(a.imag - b.imag)));
				max_diff = std::max(max_diff, std::abs(DIRECT_MULTIDIM_ELEM(ref.weight, n) - DIRECT_MULTIDIM_ELEM(bp.weight, n)));
				max_val = std::max(max_val, (double)std::max(std::abs(a.real), std::abs(a.imag)));
				if (a.real == b.real && a.imag == b.imag) exact++;
			}
			INFO("max difference " << max_diff << ", largest value " << max_val << ", " << exact << " of "
			     << MULTIDIM_SIZE(ref.data) << " voxels bit-identical");
			CHECK(max_diff <= tol * max_val);
		}
	unsetenv("RELION_GPU_SYMMETRISE");
	unsetenv("RELION_GPU_SYMMETRISE_PLANES");
	gpurec::setGpuRequested(false);
}
