/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

// Rectangular-image helpers of MlOptimiser; see documentation/rectangular_particles.md

#include "src/ml_optimiser.h"

void MlOptimiser::windowFT(MultidimArray<Complex> &in, MultidimArray<Complex> &out, int nominal_size) const
{
	if (!mymodel.isRect())
		windowFourierTransform(in, out, nominal_size);
	else
		windowFourierTransformRect(in, out, rectSizeX(nominal_size), rectSizeY(nominal_size));
}

void MlOptimiser::windowFT(MultidimArray<RFLOAT> &in, MultidimArray<RFLOAT> &out, int nominal_size) const
{
	if (!mymodel.isRect())
	{
		windowFourierTransform(in, out, nominal_size);
		return;
	}
	// Real FFTW-layout arrays (such as CTFs): window them as the real part of a complex array
	MultidimArray<Complex> cin, cout;
	cin.resize(in);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(in)
		DIRECT_MULTIDIM_ELEM(cin, n) = Complex(DIRECT_MULTIDIM_ELEM(in, n), 0.);
	windowFourierTransformRect(cin, cout, rectSizeX(nominal_size), rectSizeY(nominal_size));
	out.resize(cout);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(cout)
		DIRECT_MULTIDIM_ELEM(out, n) = DIRECT_MULTIDIM_ELEM(cout, n).real;
}

void MlOptimiser::shiftFT(MultidimArray<Complex> &in, MultidimArray<Complex> &out, RFLOAT nominal_size,
                          RFLOAT xshift, RFLOAT yshift, RFLOAT zshift) const
{
	if (!mymodel.isRect())
		shiftImageInFourierTransform(in, out, nominal_size, xshift, yshift, zshift);
	else
		shiftImageInFourierTransformRect(in, out, mymodel.imgX(), mymodel.imgY(), 1., xshift, yshift, 0.);
}

void MlOptimiser::shiftTabFT(MultidimArray<Complex> &in, MultidimArray<Complex> &out, RFLOAT nominal_size, long int new_nominal_size,
                             TabSine &tabsin, TabCosine &tabcos, RFLOAT xshift, RFLOAT yshift, RFLOAT zshift) const
{
	if (!mymodel.isRect())
		shiftImageInFourierTransformWithTabSincos(in, out, nominal_size, new_nominal_size, tabsin, tabcos, xshift, yshift, zshift);
	else
		shiftImageInFourierTransformWithTabSincosRect(in, out, mymodel.imgX(), mymodel.imgY(),
		                                              rectSizeX(new_nominal_size), rectSizeY(new_nominal_size), tabsin, tabcos, xshift, yshift);
}

void MlOptimiser::checkRectangularSupport() const
{
	if (!mymodel.isRect() && !mydata.obsModel.anyRectBox())
		return;

	const char *prefix = "ERROR: rectangular images are not supported together with ";
	if (mymodel.data_dim == 3)
		REPORT_ERROR(std::string(prefix) + "3D (subtomogram) data.");
	if (do_grad || gradient_refine)
		REPORT_ERROR(std::string(prefix) + "gradient-driven (VDAM) optimisation. Run without --grad (in the GUI: untick \"Use VDAM algorithm\").");
	if (do_gpu || do_sycl || do_cpu)
		REPORT_ERROR(std::string(prefix) + "--gpu, --sycl or --cpu. Run without them.");
	if (do_initialise_bodies || mymodel.nr_bodies > 1)
		REPORT_ERROR(std::string(prefix) + "multi-body refinement.");
	if (!do_parallel_disc_io)
		REPORT_ERROR(std::string(prefix) + "--no_parallel_disc_io (the images are always read by each rank).");
	if (do_phase_random_fsc)
		REPORT_ERROR(std::string(prefix) + "--solvent_correct_fsc (the phase-randomised FSC uses cubic shells). Run without it.");
	if (do_ewald)
		REPORT_ERROR(std::string(prefix) + "the Ewald sphere correction.");
	if (fn_local_symmetry != "None")
		REPORT_ERROR(std::string(prefix) + "local symmetry.");
	if (!mydata.obsModel.allBoxSizesIdentical() || !mydata.obsModel.allPixelSizesIdentical())
		REPORT_ERROR("ERROR: with rectangular images all optics groups must have the same box size and pixel size.");

	if (verb > 0 && mymodel.ref_dim == 3 && mymodel.isRect() && mymodel.padding_factor > 0.)
	{
		const double pad = mymodel.padding_factor;
		const double voxels = (pad * mymodel.boxZ()) * (pad * mymodel.boxY()) * (std::floor(pad * mymodel.boxX() / 2.) + 1.);
		const double gb_proj = voxels * 2. * sizeof(RFLOAT) / 1e9;
		const double gb_back = voxels * 3. * sizeof(RFLOAT) / 1e9;
		const double per_rank = (gb_proj + gb_back) * mymodel.nr_classes;
		std::cout << " + Rectangular reference " << mymodel.boxX() << " x " << mymodel.boxY() << " x " << mymodel.boxZ()
		          << " voxels, padding " << pad << ": about " << per_rank << " GB per MPI process for the padded references and back-projectors." << std::endl;
		if (per_rank > 16.)
			std::cout << " + WARNING: this is large. Use --pad 1, or a reference map with a smaller z size, to reduce it." << std::endl;
	}
}

int MlOptimiser::rectMaskLength() const
{
	int longest = XMIPP_MAX(mymodel.imgX(), mymodel.imgY());
	if (mymodel.ref_dim == 3)
	{
		const int ref_longest = XMIPP_MAX(XMIPP_MAX(mymodel.boxX(), mymodel.boxY()), mymodel.boxZ());
		longest = XMIPP_MIN(longest, ref_longest);
	}
	return longest;
}

void MlOptimiser::checkRectangularMask() const
{
	if (!rectMaskAlongLength())
		return;

	// As for square images, a larger mask than the box just masks less (only helical masks have limits)
	const bool is_helical_segment = (do_helical_refine) || ((mymodel.ref_dim == 2) && (helical_tube_outer_diameter > 0.));
	if (!is_helical_segment || helical_tube_outer_diameter <= 0.)
		return;

	const RFLOAT angpix = mymodel.pixel_size;
	const int longest = rectMaskLength();
	const int half_longest = longest / 2 - ((longest + 1) % 2); // as in softMaskOutsideMapForHelix()
	if (particle_diameter / (2. * angpix) > half_longest)
		REPORT_ERROR("ERROR: with rectangular images of helical segments the mask diameter is the long axis of an elliptical mask, so it must fit the long side of the box: at most " + floatToString(2 * half_longest * angpix) + " A; it is " + floatToString(particle_diameter) + " A.");

	// The tube has to fit across the ellipse of the images and, in 3D, across the ellipsoid of the reference
	struct Box { int across, longest; std::string what; };
	std::vector<Box> boxes;
	boxes.push_back({XMIPP_MIN(mymodel.imgX(), mymodel.imgY()), XMIPP_MAX(mymodel.imgX(), mymodel.imgY()),
	                 "the " + integerToString(mymodel.imgX()) + " x " + integerToString(mymodel.imgY()) + " images"});
	if (mymodel.ref_dim == 3)
		boxes.push_back({XMIPP_MIN(mymodel.boxX(), mymodel.boxY()), XMIPP_MAX(XMIPP_MAX(mymodel.boxX(), mymodel.boxY()), mymodel.boxZ()),
		                 "the " + integerToString(mymodel.boxX()) + " x " + integerToString(mymodel.boxY()) + " x " + integerToString(mymodel.boxZ()) + " reference"});
	for (const Box &b : boxes)
	{
		const RFLOAT short_axis = particle_diameter * b.across / b.longest;
		if (short_axis < helical_tube_outer_diameter)
			REPORT_ERROR("ERROR: with rectangular images the mask diameter (" + floatToString(particle_diameter) + " A) is the long axis of an elliptical mask, whose short axis shrinks in proportion to the box. For "
			             + b.what + " the short axis is " + floatToString(short_axis) + " A, smaller than the helical tube diameter (" + floatToString(helical_tube_outer_diameter)
			             + " A). Use a mask diameter of at least " + floatToString(helical_tube_outer_diameter * b.longest / b.across) + " A (at most " + floatToString(b.longest * angpix) + " A), or a smaller tube diameter.");
	}
}

void MlOptimiser::warnRectangularWrapAround() const
{
	if (!mymodel.isRect() || verb <= 0)
		return;

	// Reference projections and rotated 2D references are sampled on the Fourier grid of the
	// image, so in real space they repeat with the period of the box: whatever reaches beyond
	// the box comes back in on the other side. The reference is long along the box length.
	// Turning it out of the image plane (tilt) only shortens its projection, but turning it in
	// the plane (psi) makes it reach across the box width.
	const bool is_helical_segment = (do_helical_refine) || ((mymodel.ref_dim == 2) && (helical_tube_outer_diameter > 0.));
	// Only masked references have a known extent
	const bool known_extent = is_helical_segment ?
			(mymodel.ref_dim == 2 ? do_solvent : !ignore_helical_symmetry) :
			(do_solvent && fn_mask == "None");
	if (!known_extent || particle_diameter <= 0.)
		return;

	const RFLOAT angpix = mymodel.pixel_size;
	const RFLOAT longest = XMIPP_MAX(mymodel.imgX(), mymodel.imgY());
	const RFLOAT shortest = XMIPP_MIN(mymodel.imgX(), mymodel.imgY());
	const RFLOAT radius = particle_diameter / (2. * angpix);

	// Semi-axes of the masked reference along and across the box
	RFLOAT along, across;
	if (rectMaskAlongLength())
	{
		along = radius;
		across = radius * shortest / longest;
	}
	else if (is_helical_segment)
		along = across = radius; // circle / sphere
	else
	{
		across = radius;
		along = radius * longest / shortest;
	}
	const RFLOAT tube = is_helical_segment ? helical_tube_outer_diameter / (2. * angpix) : -1.;

	const RFLOAT angle_ok = inPlaneAngleBeforeWrapAround(along, across, tube, longest, shortest);
	if (angle_ok >= 90.)
		return;

	// The largest in-plane angle from the box length that the searches reach
	RFLOAT max_start = 0.;
	for (long int n = 0; n < mydata.MDimg.numberOfObjects(); n++)
	{
		RFLOAT psi = 0.;
		if (!mydata.MDimg.getValue(EMDL_ORIENT_PSI_PRIOR, psi, n))
			mydata.MDimg.getValue(EMDL_ORIENT_PSI, psi, n);
		RFLOAT a = fmod(fabs(psi), 180.);
		a = XMIPP_MIN(a, 180. - a);
		max_start = XMIPP_MAX(max_start, a);
	}
	const bool no_search = do_skip_align || do_skip_rotate;
	const bool local_psi = mymodel.sigma2_psi > 0.;
	const RFLOAT search = no_search ? 0. : (local_psi ? 3. * sqrt(mymodel.sigma2_psi) : 90.);
	const RFLOAT reach = XMIPP_MIN(90., max_start + search);
	if (reach <= angle_ok)
		return;

	std::cout << " + WARNING: rectangular images: the masked reference (" << floatToString(2. * along * angpix) << " A along the box, "
	          << floatToString(2. * (tube > 0. ? XMIPP_MIN(across, tube) : across) * angpix) << " A across) does not fit across the "
	          << floatToString(shortest * angpix) << " A box width ";
	if (angle_ok < 0.)
		std::cout << "even when it lies along the box";
	else
		std::cout << "once it turns more than " << floatToString(angle_ok) << " degrees in the image plane (psi) from the box length";
	std::cout << ". Its projections are sampled on the image grid, so they then wrap around to the other side of the box, "
	          << "and those orientations are compared with wrong references. The in-plane search here reaches "
	          << floatToString(reach) << " degrees (";
	if (no_search)
		std::cout << "no search, angles up to " << floatToString(max_start) << " degrees";
	else if (local_psi)
		std::cout << "angles or priors up to " << floatToString(max_start) << " degrees, plus 3 x --sigma_psi";
	else
		std::cout << "a global in-plane search";
	std::cout << "). Out-of-plane (tilt) changes do not cause this. Narrow the psi search (--sigma_psi; for helical segments "
	          << "extract them with --rotate_to_horizontal), use a smaller mask diameter, or extract wider boxes." << std::endl;
}
