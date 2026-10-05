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
