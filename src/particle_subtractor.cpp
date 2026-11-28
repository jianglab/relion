/***************************************************************************
 *
 * Author: "Sjors H.W. Scheres"
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

#include <src/rect_refusal.h>
#include "src/particle_subtractor.h"
#include "src/async_mrc_stack.h"
#include "src/prefetch.h"
#include "src/resource_report.h"
#include "src/relion_finufft.h"
#include <cstdlib>
#include <exception>
#include <memory>
#ifdef _OPENMP
#include <omp.h>
#endif

void ParticleSubtractor::read(int argc, char **argv)
{
	parser.setCommandLine(argc, argv);
	int gen_section = parser.addSection("General options");
	fn_opt = parser.getOption("--i", "Name of optimiser.star file from refinement/classification to use for subtraction", "");
	fn_out = parser.getOption("--o", "Output directory name", "Subtract/");
	fn_msk = parser.getOption("--mask", "Name of the 3D mask with all density that should be kept, i.e. not subtracted", "");
	fn_sel = parser.getOption("--data", "Name of particle STAR file, in case not all particles from optimiser are to be used", "");
	ignore_class = parser.checkOption("--ignore_class", "Ignore the rlnClassNumber column in the particle STAR file.");
	fn_revert = parser.getOption("--revert", "Name of particle STAR file to revert. When this is provided, all other options are ignored.", "");
	do_ssnr = parser.checkOption("--ssnr", "Don't subtract, only calculate average spectral SNR in the images");
	write_float16  = parser.checkOption("--float16", "Write in half-precision 16 bit floating point numbers (MRC mode 12), instead of 32 bit (MRC mode 0).");
	nr_threads = textToInteger(parser.getOption("--j", "Number of threads per process (-1 = automatic)", "-1"));

	int center_section = parser.addSection("Centering options");
	do_recenter_on_mask = parser.checkOption("--recenter_on_mask", "Use this flag to center the subtracted particles on projections of the centre-of-mass of the input mask");
	new_center.resize(3);
	XX(new_center) = textToInteger(parser.getOption("--center_x", "X-coordinate of 3D coordinate, which will be projected to center the subtracted particles.", "9999"));
	YY(new_center) = textToInteger(parser.getOption("--center_y", "Y-coordinate of 3D coordinate, which will be projected to center the subtracted particles.", "9999"));
	ZZ(new_center) = textToInteger(parser.getOption("--center_z", "Z-coordinate of 3D coordinate, which will be projected to center the subtracted particles.", "9999"));
	boxsize = textToInteger(parser.getOption("--new_box", "Output size of the subtracted particles", "-1"));

	// The reference is projected with NUFFT central slices by default;
	// RELION_INTERPOLATION=linear selects the classic trilinear interpolation.
	Projector::applyFinufftEnvDefaults();
	Projector::finufft_mode_crop = textToFloat(parser.getOption("--finufft_mode_crop", "Size of the NUFFT mode array as a multiple of the box size", Projector::finufftModeCropAsString()));
	Projector::finufft_tol = textToDouble(parser.getOption("--finufft_tol", "Requested relative accuracy of the NUFFT central-slice transforms", Projector::finufftTolAsString()));

	verb = 1;
	// Check for errors in the command-line option
	if (parser.checkForErrors())
		REPORT_ERROR("Errors encountered on the command line (see above), exiting...");

	if ((fn_opt == "" && fn_revert == "") || (fn_opt != "" && fn_revert != ""))
		REPORT_ERROR("Please specify only one of --i OR --revert");
}

void ParticleSubtractor::usage()
{
	parser.writeUsage(std::cout);
}

void ParticleSubtractor::divideLabour(int _rank, int _size, long int &my_first, long int &my_last)
{
	if (opt.do_split_random_halves)
	{
		if (size < 2) REPORT_ERROR("You need to run this program with at least 2 MPI processes for subtraction of 3D auto-refine jobs");
		opt.mydata.divideParticlesInRandomHalves(0, opt.do_helical_refine);

		int my_halfset = (_rank % 2 == 1) ? 1 : 2;
		int mysize = (my_halfset == 1) ? _size / 2 : _size / 2 + _size % 2;
		divide_equally(opt.mydata.numberOfParticles(my_halfset), mysize, _rank / 2, my_first, my_last);
		if (my_halfset == 2)
		{
    		my_first += opt.mydata.numberOfParticles(1);
    		my_last += opt.mydata.numberOfParticles(1);
		}
	}
	else
	{
		divide_equally(opt.mydata.numberOfParticles(), _size, _rank, my_first, my_last);
	}
	//std::cerr << " rank= " << _rank << "size= " << _size << " my_first= " << my_first << " my_last= " << my_last << std::endl;
}

// Initialise some stuff after reading
void ParticleSubtractor::initialise(int _rank, int _size)
{
	rank = _rank;
	size = _size;

	if (rank > 0) verb = 0;

	if (nr_threads <= 0)
	{
		if (const char *omp_env = getenv("OMP_NUM_THREADS"))
		{
			if (atoi(omp_env) > 0) nr_threads = atoi(omp_env);
		}
		if (nr_threads <= 0)
		{
			int local_ranks = size;
			if (const char *e = getenv("OMPI_COMM_WORLD_LOCAL_SIZE"))
				local_ranks = std::max(1, atoi(e));
			else if (const char *e2 = getenv("MPI_LOCALNRANKS"))
				local_ranks = std::max(1, atoi(e2));
			const int n_cores = std::max<int>(1, (int)resrep::allowedCores().size());
			nr_threads = std::max(1, std::min(16, n_cores / std::max(1, local_ranks)));
		}
	}
#ifdef _OPENMP
	if (omp_in_parallel()) nr_threads = 1;
#endif

	if (fn_revert != "")
		return;

	// Make directory for output particles
	if (fn_out[fn_out.length()-1] != '/') fn_out += "/";
	if (verb > 0)
	{
		mktree(fn_out + "Particles");
	}

	opt.read(fn_opt, rank, true); // true means: prevent prereading all particle images
	nr_particles_in_optics_group.resize(opt.mydata.obsModel.opticsMdt.numberOfObjects(), 0);

	// Overwrite the particles STAR file with a smaller subset
	if (fn_sel != "")
	{
		opt.mydata.clear();
		bool is_helical_segment = (opt.do_helical_refine) || ((opt.mymodel.ref_dim == 2) && (opt.helical_tube_outer_diameter > 0.));
		opt.mydata.read(fn_sel, "", "", false, false, false, is_helical_segment);
	}

	if (opt.mymodel.isRect() || opt.mydata.obsModel.anyRectBox())
		refuseRectangularImages("relion_particle_subtract", "the refinement or the optics groups describe non-square particles");

	divideLabour(rank, size, my_first_part_id, my_last_part_id);

	Image<RFLOAT> Imask;
	if (fn_msk != "" && !do_ssnr)
	{
		if (verb > 0) std::cout << " + Reading in mask ... " << std::endl;
		// Mask stuff
		Imask.read(fn_msk);
		Imask().setXmippOrigin();

		RFLOAT minval, maxval;
		Imask().computeDoubleMinMax(minval, maxval);
		if (minval < 0. || maxval > 1.)
		{
			REPORT_ERROR("ERROR: the keep_inside mask has values outside the range [0,1]");
		}
	}
	else
	{
		Imask().initZeros(opt.mymodel.Iref[0]);
	}

	if (do_ssnr)
	{
		sum_S2.initZeros(opt.mymodel.ori_size/2+1);
		sum_N2.initZeros(opt.mymodel.ori_size/2+1);
		sum_count.initZeros(opt.mymodel.ori_size/2+1);
	}

	if (do_recenter_on_mask)
	{
		Imask().centerOfMass(new_center);
		do_center = true;
	}
	else if ((int)XX(new_center) != 9999 && (int)YY(new_center) != 9999 && (int)ZZ(new_center) != 9999)
	{
		do_center = true;
	}
	else
	{
		new_center.initZeros();
		do_center = false;
	}

	if (verb > 0 && (do_center || opt.fn_body_masks != "None"))
	{
		std::cout << " + The subtracted particles will be re-centred on projections of 3D-coordinate: ("
		          << new_center(0) << " , " << new_center(1) << " , " << new_center(2) << ")" << std::endl;
	}

	if (opt.fn_body_masks != "None")
	{
		if (verb > 0)
		{
			std::cout << " + Initialising multi-body subtraction ..." << std::endl;
		}

		// This creates a rotation matrix for (rot,tilt,psi) = (0,90,0)
		// It will be used to make all Abody orientation matrices relative to (0,90,0) instead of the more logical (0,0,0)
		// This is useful, as psi-priors are ill-defined around tilt=0, as rot becomes the same as -psi!!
        //SHWS 2feb2024: found bug in rotation around Y: reverse direction!
        //rotation3DMatrix(-90., 'Y', A_rot90, false);
		rotation3DMatrix(90., 'Y', A_rot90, false);
		A_rot90T = A_rot90.transpose();

		// Find out which body has the biggest overlap with the keepmask, use these orientations
		RFLOAT best_overlap = 0.;
		subtract_body = -1;
		for (int ibody = 0; ibody < opt.mymodel.nr_bodies; ibody++)
		{
			if (!Imask().sameShape(opt.mymodel.masks_bodies[ibody]))
			{
				Imask().printShape();
				opt.mymodel.masks_bodies[ibody].printShape();
				REPORT_ERROR("ERROR: input mask is not of same shape as body masks.");
			}

			RFLOAT overlap = 0.;
			const long int n_elems = MULTIDIM_SIZE(Imask());
			const RFLOAT *m_body = MULTIDIM_ARRAY(opt.mymodel.masks_bodies[ibody]);
			const RFLOAT *m_keep = MULTIDIM_ARRAY(Imask());
			#pragma omp parallel for reduction(+:overlap) schedule(static) num_threads(nr_threads) if(nr_threads > 1)
			for (long int n = 0; n < n_elems; n++)
				overlap += m_body[n] * m_keep[n];

			if (overlap > best_overlap)
			{
				best_overlap = overlap;
				subtract_body = ibody;
			}
		}

		if (subtract_body < 0) REPORT_ERROR("ERROR: input mask does not overlap with any of the bodies....");

		// Apply the inverse of the keepmask to all the mask_bodies
		for (int obody = 0; obody < opt.mymodel.nr_bodies; obody++)
		{
			int ii = DIRECT_A2D_ELEM(opt.mymodel.pointer_body_overlap, subtract_body, obody);
			const long int n_elems = MULTIDIM_SIZE(Imask());
			RFLOAT *m_body = MULTIDIM_ARRAY(opt.mymodel.masks_bodies[ii]);
			const RFLOAT *m_keep = MULTIDIM_ARRAY(Imask());
			#pragma omp parallel for schedule(static) num_threads(nr_threads) if(nr_threads > 1)
			for (long int n = 0; n < n_elems; n++)
				m_body[n] *= (1. - m_keep[n]);
		}
	}
	else
	{
		// For normal refinement/classification: just apply the inverse of keepmask to the references
		for (int iclass = 0; iclass < opt.mymodel.nr_classes; iclass++)
		{
			if (!Imask().sameShape(opt.mymodel.Iref[iclass]))
			{
				Imask().printShape();
				opt.mymodel.Iref[iclass].printShape();
				REPORT_ERROR("ERROR: input mask is not of same shape as reference inside the optimiser.");
			}

			const long int n_elems = MULTIDIM_SIZE(Imask());
			RFLOAT *m_ref = MULTIDIM_ARRAY(opt.mymodel.Iref[iclass]);
			const RFLOAT *m_keep = MULTIDIM_ARRAY(Imask());
			#pragma omp parallel for schedule(static) num_threads(nr_threads) if(nr_threads > 1)
			for (long int n = 0; n < n_elems; n++)
				m_ref[n] *= (1. - m_keep[n]);
		}
	}

	if (verb > 0)
	{
		std::cout << " + Calculating Fourier transforms of the maps ..." << std::endl;
	}

	// Now set up the Projectors inside the model
	relionFftwInitThreads();
	relionFftwPlanThreads(nr_threads);
	projector_interpolator = Projector::resolveForwardInterpolator(false, verb);
	opt.mymodel.projector_interpolator = (projector_interpolator == FINUFFT) ? FINUFFT : opt.mymodel.interpolator;
	opt.mymodel.setFourierTransformMaps(false, nr_threads); // false means ignore tau2_class

	// ensure even boxsize of subtracted images
	if (boxsize > 0)
		boxsize -= boxsize%2;
}

void ParticleSubtractor::revert()
{
	ObservationModel obsModel;
	MetaDataTable MD;

	ObservationModel::loadSafely(fn_revert, obsModel, MD);

	if (!MD.containsLabel(EMDL_IMAGE_ORI_NAME))
		REPORT_ERROR("The input STAR file does not contain the rlnImageOriginalName column.");

	if (!MD.containsLabel(EMDL_IMAGE_NAME))
		REPORT_ERROR("The input STAR file does not contain the rlnImageName column");

	// Swap image names
	FOR_ALL_OBJECTS_IN_METADATA_TABLE(MD)
	{
		FileName f1, f2;
		MD.getValue(EMDL_IMAGE_ORI_NAME, f1);
		MD.getValue(EMDL_IMAGE_NAME, f2);
		MD.setValue(EMDL_IMAGE_ORI_NAME, f2);
		MD.setValue(EMDL_IMAGE_NAME, f1);
	}

	// Fix box size
	std::vector<bool> fixed_box_size(obsModel.numberOfOpticsGroups(), false);
	FOR_ALL_OBJECTS_IN_METADATA_TABLE(MD)
	{
		const int og = obsModel.getOpticsGroup(MD);
		if (fixed_box_size[og])
			continue;

		FileName img_name, fn_img;
		long int dummy;
		MD.getValue(EMDL_IMAGE_NAME, img_name);
		img_name.decompose(dummy, fn_img);

		if (!exists(fn_img))
			REPORT_ERROR("Failed to read " + fn_img + " to determine the box size.");
		Image<RFLOAT> Ihead;
		Ihead.read(img_name, false, -1, false, true);
		if (XSIZE(Ihead()) != YSIZE(Ihead()))
			refuseRectangularImages("relion_particle_subtract --revert", "particle " + std::string(img_name) + " is not square");
		obsModel.setBoxSize(og, XSIZE(Ihead()));
		obsModel.opticsMdt.setValue(EMDL_IMAGE_SIZE, XSIZE(Ihead()), og);

		fixed_box_size[og] = true;
	}

	for (int i = 0; i < obsModel.numberOfOpticsGroups(); i++)
	{
		if (!fixed_box_size[i])
			std::cerr << "WARNING: could not determine the box size of optics group " << obsModel.getGroupName(i) << std::endl;
		else
			std::cout << "The box size of the optics group " << obsModel.getGroupName(i) << " was updated to " << obsModel.getBoxSize(i) << " px" << std::endl;
	}

	obsModel.save(MD, fn_out + "original.star");
	std::cout << "Writen " << (fn_out + "original.star") << std::endl;
}

void ParticleSubtractor::run()
{

	long int nr_parts = my_last_part_id - my_first_part_id + 1;
	long int barstep = XMIPP_MAX(1, nr_parts/120);
	if (verb > 0)
	{
		if (do_ssnr) std::cout << " + Calculating SNR for all particles ..." << std::endl;
		else std::cout << " + Subtracting all particles ..." << std::endl;
		time_config();
		init_progress_bar(nr_parts);
	}

	MDimg_out.clear();
	if (nr_parts <= 0)
	{
		if (verb > 0) progress_bar(nr_parts);
		return;
	}

	const bool has_class_col = (!ignore_class && opt.mydata.MDimg.containsLabel(EMDL_PARTICLE_CLASS));
	const bool is_multibody = (opt.fn_body_masks != "None");
	const bool can_batch_finufft = (!is_multibody && projector_interpolator == FINUFFT);
	const int my_subset = (rank % 2 == 1) ? 1 : 2;
	const long int chunk_max = std::min<long int>(256, std::max<long int>(64, (long int)nr_threads * 16));
	const DataType datatype = write_float16 ? Float16 : Float;

	struct SubBodyMeta
	{
		RFLOAT rot = 0., tilt = 0., psi = 0.;
		Matrix1D<RFLOAT> offset;
	};
	struct SubItem
	{
		long int part_id = 0;
		long int counter = 0;
		int optics_group = 0;
		int myclass = 0;
		int group_id = 0;
		bool ctf_premultiplied = false;
		RFLOAT my_pixel_size = 1.;
		RFLOAT mynorm = 1.;
		RFLOAT rot = 0., tilt = 0., psi = 0.;
		FileName part_name, fn_ctf;
		CTF ctf;
		Matrix1D<RFLOAT> my_old_offset, my_residual_offset, my_projected_com, my_refined_ibody_offset;
		Matrix2D<RFLOAT> Aori, A3D_pure_rot, A3D;
		std::vector<SubBodyMeta> body_meta;
		Image<RFLOAT> img;
		MultidimArray<Complex> Fimg, Fsubtract;
		MultidimArray<RFLOAT> Fctf;
	};
	struct SubWorkerWs
	{
		OpenStacks open_stacks;
		FourierTransformer transformer;
		MultidimArray<Complex> FTo, Faux;
		Image<RFLOAT> Ictf;
		MultidimArray<RFLOAT> sum_S2, sum_N2, sum_count;
		explicit SubWorkerWs(size_t max_open, bool do_ssnr, long int ssnr_len)
			: open_stacks(max_open)
		{
			if (do_ssnr)
			{
				sum_S2.initZeros(ssnr_len);
				sum_N2.initZeros(ssnr_len);
				sum_count.initZeros(ssnr_len);
			}
		}
	};

	const int n_ws = std::max(1, nr_threads);
	const size_t max_open_per_ws = std::max<size_t>(8, 128 / (size_t)n_ws);
	const long int ssnr_len = do_ssnr ? (long int)MULTIDIM_SIZE(sum_S2) : 0;
	std::vector<std::unique_ptr<SubWorkerWs> > ws_pool(n_ws);
	for (int t = 0; t < n_ws; t++)
		ws_pool[t].reset(new SubWorkerWs(max_open_per_ws, do_ssnr, ssnr_len));

	relionFftwInitThreads();

	std::vector<SubItem> chunk(chunk_max);
	for (long int cc_base = 0; cc_base < nr_parts; cc_base += chunk_max)
	{
		const long int chunk_n = std::min(chunk_max, nr_parts - cc_base);

		// Sequential pre-pass: validate subset, read metadata and precompute matrices cleanly
		for (long int k = 0; k < chunk_n; k++)
		{
			const long int cc = cc_base + k;
			if (cc % barstep == 0 && pipeline_control_check_abort_job())
				exit(RELION_EXIT_ABORTED);

			const long int part_id = opt.mydata.sorted_idx[my_first_part_id + cc];
			SubItem &it = chunk[k];
			it.part_id = part_id;
			it.counter = cc;
			it.optics_group = opt.mydata.getOpticsGroup(part_id);
			it.part_name = opt.mydata.particles[part_id].name;

			if (opt.do_split_random_halves && my_subset != opt.mydata.getRandomSubset(part_id))
			{
				std::cerr << " rank= " << rank << " part_id= " << part_id << " opt.mydata.getRandomSubset(part_id)= " << opt.mydata.getRandomSubset(part_id) << std::endl;
				REPORT_ERROR("BUG:: gold-standard separation of halves is broken!");
			}

			it.my_pixel_size = opt.mydata.getImagePixelSize(part_id);
			it.ctf_premultiplied = opt.mydata.obsModel.getCtfPremultiplied(it.optics_group);
			it.group_id = opt.do_scale_correction ? opt.mydata.getGroupId(part_id) : 0;
			it.myclass = 0;
			if (has_class_col)
			{
				opt.mydata.MDimg.getValue(EMDL_PARTICLE_CLASS, it.myclass, part_id);
				if (it.myclass > opt.mymodel.nr_classes)
				{
					std::cerr << "A particle belongs to class " << it.myclass << " while the number of classes in the optimiser.star is only " << opt.mymodel.nr_classes << "." << std::endl;
					REPORT_ERROR("Tried to subtract a non-existing class from a particle. If you have performed non-alignment Class3D after Refine3D and want to subtract a map from the Refine3D job, use the --ignore_class option.");
				}
				it.myclass--;
			}

			opt.mydata.MDimg.getValue(EMDL_ORIENT_ROT, it.rot, part_id);
			opt.mydata.MDimg.getValue(EMDL_ORIENT_TILT, it.tilt, part_id);
			opt.mydata.MDimg.getValue(EMDL_ORIENT_PSI, it.psi, part_id);
			it.my_old_offset.initZeros(3);
			opt.mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, XX(it.my_old_offset), part_id);
			opt.mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, YY(it.my_old_offset), part_id);
			if (opt.mymodel.data_dim == 3)
				opt.mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_Z_ANGSTROM, ZZ(it.my_old_offset), part_id);
			it.my_old_offset /= it.my_pixel_size;

			if (!opt.mydata.MDimg.getValue(EMDL_IMAGE_NORM_CORRECTION, it.mynorm, part_id))
				it.mynorm = 1.;

			if (opt.do_ctf_correction)
			{
				if (opt.mymodel.data_dim == 3)
				{
					if (!it.ctf_premultiplied)
						REPORT_ERROR("3D data must be ctf_premultiplied for CTF correction.");
					opt.mydata.MDimg.getValue(EMDL_CTF_IMAGE, it.fn_ctf, part_id);
				}
				else
				{
					it.ctf.readByGroup(opt.mydata.MDimg, &opt.mydata.obsModel, part_id);
				}
			}

			if (is_multibody)
			{
				Euler_angles2matrix(it.rot, it.tilt, it.psi, it.Aori, false);
				it.my_projected_com = it.Aori * opt.mymodel.com_bodies[subtract_body];
				it.body_meta.resize(opt.mymodel.nr_bodies);
				for (int obody = 0; obody < opt.mymodel.nr_bodies; obody++)
				{
					SubBodyMeta &bm = it.body_meta[obody];
					bm.offset.initZeros(3);
					opt.mydata.MDbodies[obody].getValue(EMDL_ORIENT_ROT, bm.rot, part_id);
					opt.mydata.MDbodies[obody].getValue(EMDL_ORIENT_TILT, bm.tilt, part_id);
					opt.mydata.MDbodies[obody].getValue(EMDL_ORIENT_PSI, bm.psi, part_id);
					opt.mydata.MDbodies[obody].getValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, XX(bm.offset), part_id);
					opt.mydata.MDbodies[obody].getValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, YY(bm.offset), part_id);
					if (opt.mymodel.data_dim == 3)
						opt.mydata.MDbodies[obody].getValue(EMDL_ORIENT_ORIGIN_Z_ANGSTROM, ZZ(bm.offset), part_id);
					bm.offset /= it.my_pixel_size;
				}
				it.my_refined_ibody_offset = it.body_meta[subtract_body].offset;
			}
			else
			{
				Euler_angles2matrix(it.rot, it.tilt, it.psi, it.A3D_pure_rot);
				it.A3D = opt.mydata.obsModel.applyAnisoMag(it.A3D_pure_rot, it.optics_group);
				it.A3D = opt.mydata.obsModel.applyScaleDifference(it.A3D, it.optics_group, opt.mymodel.ori_size, opt.mymodel.pixel_size);
			}
		}

		// Stage 1: Parallel image read (using persistent OpenStacks), multi-body shift, FFT, and CTF
		std::exception_ptr err;
		#pragma omp parallel num_threads(n_ws)
		{
			relionFftwSetThreadPlanThreads(1);
#ifdef _OPENMP
			const int tid = omp_get_thread_num();
#else
			const int tid = 0;
#endif
			SubWorkerWs &ws = *ws_pool[tid];
			#pragma omp for schedule(dynamic, 1)
			for (long int k = 0; k < chunk_n; k++)
			{
				try
				{
					SubItem &it = chunk[k];
					if (opt.mymodel.data_dim == 2 && it.part_name.contains("@"))
					{
						long int img_no;
						FileName fn_stack;
						it.part_name.decompose(img_no, fn_stack);
						if (img_no > 0 && fn_stack.getExtension() != "vstack")
							it.img.readFromOpenFile(it.part_name, ws.open_stacks.get(fn_stack), -1, false);
						else
							it.img.read(it.part_name);
					}
					else
					{
						it.img.read(it.part_name);
					}
					it.img().setXmippOrigin();

					if (opt.do_norm_correction)
						it.img() *= opt.mymodel.avg_norm_correction / it.mynorm;

					if (is_multibody)
					{
						it.my_old_offset -= it.my_projected_com;
						it.my_residual_offset = it.my_old_offset;
						it.my_old_offset.selfROUND();
						selfTranslate(it.img(), it.my_old_offset, WRAP);
						it.my_residual_offset -= it.my_old_offset;
					}

					ws.transformer.FourierTransform(it.img(), it.Fimg);
					CenterFFTbySign(it.Fimg);
					it.Fctf.resize(it.Fimg);

					if (opt.do_ctf_correction)
					{
						if (opt.mymodel.data_dim == 3)
						{
							ws.Ictf.read(it.fn_ctf);
							if (XSIZE(ws.Ictf()) == YSIZE(ws.Ictf()))
							{
								ws.Ictf().setXmippOrigin();
								FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(it.Fctf)
									DIRECT_A3D_ELEM(it.Fctf, k, i, j) = A3D_ELEM(ws.Ictf(), -kp, -ip, -jp);
							}
							else if (XSIZE(ws.Ictf()) == YSIZE(ws.Ictf()) / 2 + 1)
							{
								if (ZSIZE(ws.Ictf()) == YSIZE(ws.Ictf()))
									windowFourierTransform(ws.Ictf(), it.Fctf, YSIZE(it.Fctf));
								else if (ZSIZE(ws.Ictf()) == YSIZE(ws.Ictf()) * 2)
								{
									MultidimArray<RFLOAT> &Mctf = ws.Ictf();
									long int max_r2 = (XSIZE(Mctf) - 1) * (XSIZE(Mctf) - 1);
									FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(it.Fctf)
									{
										if (kp * kp + ip * ip + jp * jp <= max_r2)
											FFTW_ELEM(it.Fctf, kp, ip, jp) = DIRECT_A3D_ELEM(Mctf, ((kp < 0) ? (kp + YSIZE(Mctf)) : kp),
												((ip < 0) ? (ip + YSIZE(Mctf)) : ip), jp);
									}
								}
								else
									REPORT_ERROR("3D CTF volume in FFTW format must cointain CTF or CTF and MULTI concatenated along Z !");
							}
							else
								REPORT_ERROR("3D CTF volume must be either cubical or adhere to FFTW format!");
						}
						else
						{
							it.ctf.getFftwImage(it.Fctf, XSIZE(it.img()), YSIZE(it.img()), it.my_pixel_size,
							                    opt.ctf_phase_flipped, false, opt.intact_ctf_first_peak, true);
							if (it.ctf_premultiplied)
							{
								const long int nelem = MULTIDIM_SIZE(it.Fctf);
								RFLOAT *fctf_p = MULTIDIM_ARRAY(it.Fctf);
								for (long int n = 0; n < nelem; n++)
									fctf_p[n] *= fctf_p[n];
							}
						}
					}
					else
					{
						it.Fctf.initConstant(1.);
					}

					it.Fsubtract.initZeros(it.Fimg);
				}
				catch (...)
				{
#pragma omp critical
					if (!err) err = std::current_exception();
				}
			}
			relionFftwSetThreadPlanThreads(0);
		}
		if (err) std::rethrow_exception(err);

		// Stage 2: Batched FINUFFT central-slice projections (or parallel trilinear / multi-body)
		if (can_batch_finufft)
		{
			relionFftwPlanThreads(nr_threads);
			setFinufftThreadCount(nr_threads);
			std::vector<MultidimArray<Complex>*> ptrs;
			std::vector<Matrix2D<RFLOAT> > mats;
			for (int c = 0; c < opt.mymodel.nr_classes; c++)
			{
				ptrs.clear();
				mats.clear();
				for (long int k = 0; k < chunk_n; k++)
				{
					if (chunk[k].myclass == c)
					{
						ptrs.push_back(&chunk[k].Fsubtract);
						mats.push_back(chunk[k].A3D);
					}
				}
				if (!ptrs.empty())
					opt.mymodel.PPref[c].get2DFourierTransformMany(ptrs, mats);
			}
		}
		else
		{
			if (projector_interpolator == FINUFFT)
				setFinufftThreadCount(1);
			#pragma omp parallel num_threads(n_ws)
			{
				relionFftwSetThreadPlanThreads(1);
#ifdef _OPENMP
				const int tid = omp_get_thread_num();
#else
				const int tid = 0;
#endif
				SubWorkerWs &ws = *ws_pool[tid];
				#pragma omp for schedule(dynamic, 1)
				for (long int k = 0; k < chunk_n; k++)
				{
					try
					{
						SubItem &it = chunk[k];
						if (is_multibody)
						{
							Matrix2D<RFLOAT> Aresi_subtract;
							for (int obody = 0; obody < opt.mymodel.nr_bodies; obody++)
							{
								const SubBodyMeta &bm = it.body_meta[obody];
								Matrix2D<RFLOAT> Aresi, Abody;
								Euler_angles2matrix(bm.rot, bm.tilt, bm.psi, Aresi, false);
								if (obody == subtract_body) Aresi_subtract = Aresi;
								Abody = it.Aori * (opt.mymodel.orient_bodies[obody]).transpose() * A_rot90 * Aresi * opt.mymodel.orient_bodies[obody];
								Abody = opt.mydata.obsModel.applyAnisoMag(Abody, it.optics_group);
								Abody = opt.mydata.obsModel.applyScaleDifference(Abody, it.optics_group, opt.mymodel.ori_size, opt.mymodel.pixel_size);

								ws.FTo.initZeros(it.Fimg);
								int oobody = DIRECT_A2D_ELEM(opt.mymodel.pointer_body_overlap, subtract_body, obody);
								opt.mymodel.PPref[oobody].get2DFourierTransform(ws.FTo, Abody);

								Matrix1D<RFLOAT> other_projected_com = it.Aori * (opt.mymodel.com_bodies[obody]);
								other_projected_com -= bm.offset;
								other_projected_com -= it.my_projected_com;

								shiftImageInFourierTransform(ws.FTo, ws.Faux, (RFLOAT)XSIZE(it.img()),
									XX(other_projected_com), YY(other_projected_com), ZZ(other_projected_com));
								it.Fsubtract += ws.Faux;
							}

							Matrix2D<RFLOAT> Abody = it.Aori * (opt.mymodel.orient_bodies[subtract_body]).transpose() * A_rot90 * Aresi_subtract * opt.mymodel.orient_bodies[subtract_body];
							Euler_matrix2angles(Abody, it.rot, it.tilt, it.psi);
							it.my_residual_offset += it.my_refined_ibody_offset;
							it.my_residual_offset += Abody * (opt.mymodel.com_bodies[subtract_body] - new_center * opt.mymodel.pixel_size / it.my_pixel_size);
						}
						else
						{
							opt.mymodel.PPref[it.myclass].get2DFourierTransform(it.Fsubtract, it.A3D);
						}
					}
					catch (...)
					{
#pragma omp critical
						if (!err) err = std::current_exception();
					}
				}
				relionFftwSetThreadPlanThreads(0);
			}
			if (projector_interpolator == FINUFFT)
				setFinufftThreadCount(nr_threads);
			if (err) std::rethrow_exception(err);
		}

		// Stage 3: Parallel Fourier shift, CTF/optics/scale modulation, subtraction, iFFT, recenter & crop
		#pragma omp parallel num_threads(n_ws)
		{
			relionFftwSetThreadPlanThreads(1);
#ifdef _OPENMP
			const int tid = omp_get_thread_num();
#else
			const int tid = 0;
#endif
			SubWorkerWs &ws = *ws_pool[tid];
			#pragma omp for schedule(dynamic, 1)
			for (long int k = 0; k < chunk_n; k++)
			{
				try
				{
					SubItem &it = chunk[k];
					if (!is_multibody)
					{
						shiftImageInFourierTransform(it.Fsubtract, it.Fsubtract, (RFLOAT)XSIZE(it.img()),
							-XX(it.my_old_offset), -YY(it.my_old_offset), -ZZ(it.my_old_offset));
						if (do_center)
							it.my_residual_offset = it.my_old_offset - it.A3D_pure_rot * (new_center * opt.mymodel.pixel_size / it.my_pixel_size);
					}

					const long int nelem = MULTIDIM_SIZE(it.Fsubtract);
					if (opt.do_ctf_correction)
					{
						Complex *fsub_p = MULTIDIM_ARRAY(it.Fsubtract);
						const RFLOAT *fctf_p = MULTIDIM_ARRAY(it.Fctf);
						for (long int n = 0; n < nelem; n++)
							fsub_p[n] *= fctf_p[n];
						opt.mydata.obsModel.demodulatePhase(it.optics_group, it.Fsubtract, true);
						opt.mydata.obsModel.divideByMtf(it.optics_group, it.Fsubtract, true);
					}

					if (opt.do_scale_correction)
					{
						const RFLOAT myscale = opt.mymodel.scale_correction[it.group_id];
						Complex *fsub_p = MULTIDIM_ARRAY(it.Fsubtract);
						for (long int n = 0; n < nelem; n++)
							fsub_p[n] *= myscale;
					}

					{
						Complex *fimg_p = MULTIDIM_ARRAY(it.Fimg);
						const Complex *fsub_p = MULTIDIM_ARRAY(it.Fsubtract);
						for (long int n = 0; n < nelem; n++)
							fimg_p[n] -= fsub_p[n];
					}

					if (do_ssnr)
					{
						const RFLOAT remap_image_sizes = (opt.mymodel.ori_size * opt.mymodel.pixel_size) / (XSIZE(it.img()) * it.my_pixel_size);
						const int max_shell = opt.mymodel.ori_size / 2 + 1;
						FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(it.Fimg)
						{
							long int idx = ROUND(sqrt(kp*kp + ip*ip + jp*jp));
							int idx_remapped = ROUND(remap_image_sizes * idx);
							if (idx_remapped < max_shell)
							{
								RFLOAT S2 = norm(dAkij(it.Fsubtract, k, i, j));
								RFLOAT N2 = norm(dAkij(it.Fimg, k, i, j));
								ws.sum_S2(idx_remapped) += S2 / 2.;
								ws.sum_N2(idx_remapped) += N2 / 2.;
								ws.sum_count(idx_remapped) += 1.;
							}
						}
					}
					else
					{
						CenterFFTbySign(it.Fimg);
						ws.transformer.inverseFourierTransform(it.Fimg, it.img());

						if (do_center || is_multibody)
						{
							Matrix1D<RFLOAT> centering_offset = it.my_residual_offset;
							centering_offset.selfROUND();
							it.my_residual_offset -= centering_offset;
							selfTranslate(it.img(), centering_offset, WRAP);
						}

						if (boxsize > 0)
						{
							if (it.img().getDim() == 2)
							{
								it.img().window(FIRST_XMIPP_INDEX(boxsize), FIRST_XMIPP_INDEX(boxsize),
								                LAST_XMIPP_INDEX(boxsize), LAST_XMIPP_INDEX(boxsize));
							}
							else if (it.img().getDim() == 3)
							{
								it.img().window(FIRST_XMIPP_INDEX(boxsize), FIRST_XMIPP_INDEX(boxsize), FIRST_XMIPP_INDEX(boxsize),
								                LAST_XMIPP_INDEX(boxsize), LAST_XMIPP_INDEX(boxsize), LAST_XMIPP_INDEX(boxsize));
							}
						}
						it.img.setSamplingRateInHeader(it.my_pixel_size);
					}
				}
				catch (...)
				{
#pragma omp critical
					if (!err) err = std::current_exception();
				}
			}
			relionFftwSetThreadPlanThreads(0);
		}
		if (err) std::rethrow_exception(err);

		// Stage 4: Sequential metadata update & async stack write in deterministic order
		for (long int k = 0; k < chunk_n; k++)
		{
			SubItem &it = chunk[k];
			if (!do_ssnr)
			{
				if (is_multibody)
				{
					opt.mydata.MDimg.setValue(EMDL_ORIENT_ROT, it.rot, it.part_id);
					opt.mydata.MDimg.setValue(EMDL_ORIENT_TILT, it.tilt, it.part_id);
					opt.mydata.MDimg.setValue(EMDL_ORIENT_PSI, it.psi, it.part_id);
				}
				if (do_center || is_multibody)
				{
					opt.mydata.MDimg.setValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, it.my_pixel_size * XX(it.my_residual_offset), it.part_id);
					opt.mydata.MDimg.setValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, it.my_pixel_size * YY(it.my_residual_offset), it.part_id);
					if (opt.mymodel.data_dim == 3)
						opt.mydata.MDimg.setValue(EMDL_ORIENT_ORIGIN_Z_ANGSTROM, it.my_pixel_size * ZZ(it.my_residual_offset), it.part_id);
				}

				FileName fn_img = getParticleName(it.counter, rank, it.optics_group);
				opt.mydata.MDimg.setValue(EMDL_IMAGE_NAME, fn_img, it.part_id);
				opt.mydata.MDimg.setValue(EMDL_IMAGE_ORI_NAME, opt.mydata.particles[it.part_id].name, it.part_id);
				opt.mydata.MDimg.setValue(EMDL_IMAGE_ID, it.part_id, it.part_id);
				MDimg_out.addObject();
				MDimg_out.setObject(opt.mydata.MDimg.getObject(it.part_id));

				if (opt.mymodel.data_dim == 3)
				{
					std::shared_ptr<Image<RFLOAT> > mine(new Image<RFLOAT>(it.img));
					AsyncFileWriter::putFile(fn_img, MULTIDIM_SIZE(it.img()) * sizeof(RFLOAT), [mine, datatype](const std::string &tmp)
					{
						mine->write(tmp, -1, false, WRITE_OVERWRITE, datatype);
					});
				}
				else
				{
					long int dummy;
					FileName fn_stack;
					fn_img.decompose(dummy, fn_stack);
					std::shared_ptr<AsyncMrcStack> &stack = stacks[fn_stack];
					if (!stack) stack.reset(new AsyncMrcStack(fn_stack, datatype));
					stack->add(it.img);
				}
			}
			it.img().clear();
			it.Fimg.clear();
			it.Fsubtract.clear();
			it.Fctf.clear();

			if (it.counter % barstep == 0 && verb > 0)
				progress_bar(it.counter);
		}
	}

	if (do_ssnr)
	{
		for (int t = 0; t < n_ws; t++)
		{
			sum_S2 += ws_pool[t]->sum_S2;
			sum_N2 += ws_pool[t]->sum_N2;
			sum_count += ws_pool[t]->sum_count;
		}
	}

	relionFftwPlanThreads(nr_threads);
	setFinufftThreadCount(nr_threads);

	if (verb > 0) progress_bar(nr_parts);
}

void ParticleSubtractor::saveStarFile(int myrank)
{
	// All images must be on disc (an error stops the job) before the STAR files that list them are written
	for (auto &s : stacks) s.second->finish();
	stacks.clear();
	AsyncFileWriter::instance().waitOrThrow();

	if (do_ssnr)
	{

		// Only leader writes out the STAR file
		if (myrank==0)
		{
			MetaDataTable MD;
			for (int ires = 0; ires < XSIZE(sum_S2); ires++)
			{
				MD.addObject();
				MD.setValue(EMDL_SPECTRAL_IDX, ires);
				MD.setValue(EMDL_RESOLUTION, opt.mymodel.getResolution(ires));
				MD.setValue(EMDL_RESOLUTION_ANGSTROM, opt.mymodel.getResolutionAngstrom(ires));
				MD.setValue(EMDL_MLMODEL_SSNR_REF, sum_S2(ires)/sum_N2(ires));
				MD.setValue(EMDL_MLMODEL_TAU2_REF, sum_S2(ires) / sum_count(ires) );
				MD.setValue(EMDL_MLMODEL_SIGMA2_NOISE, sum_N2(ires) / sum_count(ires) );
			}
			std::cout << " Writing out STAR file with spectral SNR in: " << fn_out << "spectral_snr.star" << " ..." << std::endl;
			MD.write(fn_out+"spectral_snr.star");
		}

	}
	else
	{

		// Reset image size in optics table, if the images were rewindowed in a different box
		if (boxsize > 0)
		{
			FOR_ALL_OBJECTS_IN_METADATA_TABLE(opt.mydata.obsModel.opticsMdt)
			{
				opt.mydata.obsModel.opticsMdt.setValue(EMDL_IMAGE_SIZE, boxsize);
			}
		}

		// Remove origin prior columns if present, as we have re-centered.
		if (do_center || opt.fn_body_masks != "None")
		{
			if (MDimg_out.containsLabel(EMDL_ORIENT_ORIGIN_X_PRIOR_ANGSTROM))
			{
				MDimg_out.deactivateLabel(EMDL_ORIENT_ORIGIN_X_PRIOR_ANGSTROM);
			}
			if (MDimg_out.containsLabel(EMDL_ORIENT_ORIGIN_Y_PRIOR_ANGSTROM))
			{
				MDimg_out.deactivateLabel(EMDL_ORIENT_ORIGIN_Y_PRIOR_ANGSTROM);
			}
		}

		FileName fn_star;
		if (size == 0)
		{
			fn_star = fn_out + "particles_subtracted.star";
			MDimg_out.deactivateLabel(EMDL_IMAGE_ID);
		}
		else
		{
			fn_star = fn_out + "Particles/subtracted_rank" + integerToString(myrank) + "star";
		}
		opt.mydata.obsModel.save(MDimg_out, fn_star);

	}

#ifdef DEBUG
	std::cout << "myrank = " << myrank << " size = " << size << std::endl;
#endif
}

void ParticleSubtractor::combineStarFile(int myrank)
{

	if (do_ssnr) return;

	if (myrank != 0) REPORT_ERROR("BUG: this function should only be called by leader!");

	MetaDataTable MD;
	for (int i = 1; i < size; i++)
	{
		FileName fn_star = fn_out + "Particles/subtracted_rank" + integerToString(i) + "star";
		MD.read(fn_star, "particles");
		MDimg_out.append(MD);
	}

	MDimg_out.sort(EMDL_IMAGE_ID);
	MDimg_out.deactivateLabel(EMDL_IMAGE_ID);
	opt.mydata.obsModel.save(MDimg_out, fn_out + "particles_subtracted.star");

	std::cout << " + Saved STAR file with " << MDimg_out.numberOfObjects()
	          << " subtracted particles in " << fn_out << "particles_subtracted.star" << std::endl;

}

FileName ParticleSubtractor::getParticleName(long int imgno, int myrank, int optics_group)
{
	if (imgno_to_filename.find(imgno) != imgno_to_filename.end())
		return imgno_to_filename[imgno];

	if (optics_group == -1)
	{
		std::cerr << "rank = " << rank << " imgno = " << imgno << std::endl;
		REPORT_ERROR("Logic error: optics group must be specified to register a new entry");
	}

	nr_particles_in_optics_group[optics_group]++;

	// Now write out the image
	FileName fn_img;
	FileName fn_stack = fn_out + "Particles/subtracted";
	if (size > 1)
		fn_stack += "_rank" + integerToString(myrank + 1);

	if (opt.mymodel.data_dim == 3)
	{
		fn_img.compose(fn_stack, imgno + 1, "mrc");
	}
	else
	{

		fn_img.compose(nr_particles_in_optics_group[optics_group], fn_stack + "_opticsgroup" + integerToString(optics_group + 1) + ".mrcs");
	}

	imgno_to_filename[imgno] = fn_img;
#ifdef DEBUG
	std::cout << "rank = " << rank << " imgno = " << imgno << " fn_img = " << fn_img << std::endl;
#endif
	return fn_img;
}

void ParticleSubtractor::subtractOneParticle(long int part_id, long int imgno, long int counter)
{
	// Read the particle image
	Image<RFLOAT> img;
	int optics_group = opt.mydata.getOpticsGroup(part_id);
	img.read(opt.mydata.particles[part_id].name);
	img().setXmippOrigin();

	// Make sure gold-standard is adhered to!
	int my_subset = (rank % 2 == 1) ? 1 : 2;
	if (opt.do_split_random_halves && my_subset != opt.mydata.getRandomSubset(part_id))
	{
		std::cerr << " rank= " << rank << " part_id= " << part_id << " opt.mydata.getRandomSubset(part_id)= " << opt.mydata.getRandomSubset(part_id) << std::endl;
		REPORT_ERROR("BUG:: gold-standard separation of halves is broken!");
	}

	// Get the consensus class, orientational parameters and norm (if present)
	RFLOAT my_pixel_size = opt.mydata.getImagePixelSize(part_id);
	RFLOAT remap_image_sizes = (opt.mymodel.ori_size * opt.mymodel.pixel_size) / (XSIZE(img()) * my_pixel_size);
	Matrix1D<RFLOAT> my_old_offset(3), my_residual_offset(3), centering_offset(3);
	Matrix2D<RFLOAT> Aori;
	RFLOAT rot, tilt, psi, xoff, yoff, zoff, mynorm, scale;
	int myclass = 0;
	if (!ignore_class && opt.mydata.MDimg.containsLabel(EMDL_PARTICLE_CLASS))
	{
		opt.mydata.MDimg.getValue(EMDL_PARTICLE_CLASS, myclass, part_id);
		if (myclass > opt.mymodel.nr_classes)
		{
			std::cerr << "A particle belongs to class " << myclass << " while the number of classes in the optimiser.star is only " << opt.mymodel.nr_classes << "." << std::endl;
			REPORT_ERROR("Tried to subtract a non-existing class from a particle. If you have performed non-alignment Class3D after Refine3D and want to subtract a map from the Refine3D job, use the --ignore_class option.");
		}
		myclass--; // start counting at zero instead of one
	}
	opt.mydata.MDimg.getValue(EMDL_ORIENT_ROT, rot, part_id);
	opt.mydata.MDimg.getValue(EMDL_ORIENT_TILT, tilt, part_id);
	opt.mydata.MDimg.getValue(EMDL_ORIENT_PSI, psi, part_id);
	opt.mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, XX(my_old_offset), part_id);
	opt.mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, YY(my_old_offset), part_id);
	if (opt.mymodel.data_dim == 3) opt.mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_Z_ANGSTROM, ZZ(my_old_offset), part_id);
	// As of v3.1, offsets are in Angstrom: convert back to pixels!
	my_old_offset /= my_pixel_size;

	// Apply the norm_correction term
	if (!opt.mydata.MDimg.getValue(EMDL_IMAGE_NORM_CORRECTION, mynorm, part_id)) mynorm = 1.;
	if (opt.do_norm_correction) img() *= opt.mymodel.avg_norm_correction / mynorm;

	Matrix1D<RFLOAT> my_projected_com(3), my_refined_ibody_offset(3);
	if (opt.fn_body_masks != "None")
	{
		// 17May2017: Shift image to the projected COM for this body!
		// Aori is the original transformation matrix of the consensus refinement
		Euler_angles2matrix(rot, tilt, psi, Aori, false);
		my_projected_com = Aori * opt.mymodel.com_bodies[subtract_body];

		// Subtract the projected COM offset, to position this body in the center
		my_old_offset -= my_projected_com;
		my_residual_offset = my_old_offset;
		// Apply the old_offset (rounded to avoid interpolation errors)
		my_old_offset.selfROUND();
		selfTranslate(img(), my_old_offset, WRAP);
		// keep track of the differences between the rounded and the original offsets
		my_residual_offset -= my_old_offset;

	}

	// Now that the particle is centered (for multibody), get the FourierTransform of the particle
	MultidimArray<Complex> Faux, Fimg;
	MultidimArray<RFLOAT> Fctf;
	FourierTransformer transformer;
	transformer.FourierTransform(img(), Fimg);
	CenterFFTbySign(Fimg);
	Fctf.resize(Fimg);
	bool ctf_premultiplied = opt.mydata.obsModel.getCtfPremultiplied(optics_group);

	if (opt.do_ctf_correction)
	{
		if (opt.mymodel.data_dim == 3)
		{
			if (!ctf_premultiplied)
				REPORT_ERROR("3D data must be ctf_premultiplied for CTF correction.");

			Image<RFLOAT> Ictf;
			FileName fn_ctf;
			opt.mydata.MDimg.getValue(EMDL_CTF_IMAGE, fn_ctf, part_id);
			Ictf.read(fn_ctf);

			// If there is a redundant half, get rid of it
			if (XSIZE(Ictf()) == YSIZE(Ictf()))
			{
				Ictf().setXmippOrigin();
				FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fctf)
				{
					// Use negative kp,ip and jp indices, because the origin in the ctf_img lies half a pixel to the right of the actual center....
					DIRECT_A3D_ELEM(Fctf, k, i, j) = A3D_ELEM(Ictf(), -kp, -ip, -jp);
				}
			}
			// otherwise, check if the file also contains Multiplicity
			else if (XSIZE(Ictf()) == YSIZE(Ictf()) / 2 + 1)
			{
				//CTF Only: Just window the CTF to the current resolution
				if (ZSIZE(Ictf()) == YSIZE(Ictf()))
				{
					windowFourierTransform(Ictf(), Fctf, YSIZE(Fctf));
				}
				// Subtomo Multiplicity weights included. Read solo CTF
				else if (ZSIZE(Ictf()) == YSIZE(Ictf())*2)
				{
					MultidimArray<RFLOAT> &Mctf = Ictf();
					long int max_r2 = (XSIZE(Mctf) - 1) * (XSIZE(Mctf) - 1);

					FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fctf)
					{
						// Make sure windowed FT has nothing in the corners, otherwise we end up with an asymmetric FT!
						if (kp * kp + ip * ip + jp * jp <= max_r2)
						{
							FFTW_ELEM(Fctf, kp, ip, jp) = DIRECT_A3D_ELEM(Mctf, ((kp < 0) ? (kp + YSIZE(Mctf))
																						  : (kp)), \
								((ip < 0) ? (ip + YSIZE(Mctf)) : (ip)), jp);
						}
					}
				}
					// if Z dimension is neither containing CTF or CTF+MULTI, stop
				else
				{
					REPORT_ERROR("3D CTF volume in FFTW format must cointain CTF or CTF and MULTI concatenated along Z !");
				}
			}
			// if dimensions are neither cubical nor FFTW, stop
			else
			{
				REPORT_ERROR("3D CTF volume must be either cubical or adhere to FFTW format!");
			}
		}
		else
		{
			// TODO! This will not yet work for 2D stacks in STA!!!

            CTF ctf;
			ctf.readByGroup(opt.mydata.MDimg, &opt.mydata.obsModel, part_id);
			ctf.getFftwImage(Fctf, XSIZE(img()), YSIZE(img()), my_pixel_size,
					opt.ctf_phase_flipped, false, opt.intact_ctf_first_peak, true);

			if (ctf_premultiplied)
				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fctf)
					(DIRECT_MULTIDIM_ELEM(Fctf, n) *= DIRECT_MULTIDIM_ELEM(Fctf, n));
		}
	}
	else
	{
		Fctf.initConstant(1.);
	}

	MultidimArray<Complex> Fsubtract;
	Fsubtract.initZeros(Fimg);

	if (opt.fn_body_masks != "None")
	{
		// For multi-body refinement
		Matrix2D<RFLOAT> Aresi_subtract;
		for (int obody = 0; obody < opt.mymodel.nr_bodies; obody++)
		{
			// Unlike getFourierTransformsAndCtfs, no check for ibody==obody: also subtract rest of subtract_body!

			Matrix1D<RFLOAT> body_offset(3);
			RFLOAT body_rot, body_tilt, body_psi;
			opt.mydata.MDbodies[obody].getValue(EMDL_ORIENT_ROT, body_rot, part_id);
			opt.mydata.MDbodies[obody].getValue(EMDL_ORIENT_TILT, body_tilt, part_id);
			opt.mydata.MDbodies[obody].getValue(EMDL_ORIENT_PSI,  body_psi, part_id);
			opt.mydata.MDbodies[obody].getValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, XX(body_offset), part_id);
			opt.mydata.MDbodies[obody].getValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, YY(body_offset), part_id);
			if (opt.mymodel.data_dim == 3)
			{
				opt.mydata.MDbodies[obody].getValue(EMDL_ORIENT_ORIGIN_Z_ANGSTROM, ZZ(body_offset), part_id);
			}

			// As of v3.1, offsets are in Angstrom: convert back to pixels!
			body_offset /= my_pixel_size;

			Matrix2D<RFLOAT> Aresi,  Abody;
			// Aresi is the residual orientation for this obody
			Euler_angles2matrix(body_rot, body_tilt, body_psi, Aresi, false);
			if (obody == subtract_body) Aresi_subtract = Aresi;
			// The real orientation to be applied is the obody transformation applied and the original one
			Abody = Aori * (opt.mymodel.orient_bodies[obody]).transpose() * A_rot90 * Aresi * opt.mymodel.orient_bodies[obody];
			// Apply anisotropic mag and scaling
			Abody = opt.mydata.obsModel.applyAnisoMag(Abody, optics_group);
			Abody = opt.mydata.obsModel.applyScaleDifference(Abody, optics_group, opt.mymodel.ori_size, opt.mymodel.pixel_size);

			// Get the FT of the projection in the right direction
			MultidimArray<Complex> FTo;
			FTo.initZeros(Fimg);
			// The following line gets the correct pointer to account for overlap in the bodies
			int oobody = DIRECT_A2D_ELEM(opt.mymodel.pointer_body_overlap, subtract_body, obody);
			opt.mymodel.PPref[oobody].get2DFourierTransform(FTo, Abody);

			// Body is centered at its own COM: move it back to its place in the original particle image

			// Projected COM for this body (using Aori, just like above for ibody and my_projected_com!!!)
			Matrix1D<RFLOAT> other_projected_com(3);
			other_projected_com = Aori * (opt.mymodel.com_bodies[obody]);

			// Subtract refined obody-displacement
			other_projected_com -= body_offset;

			// Subtract the projected COM already applied to this image for ibody
			other_projected_com -= my_projected_com;

			shiftImageInFourierTransform(FTo, Faux, (RFLOAT)XSIZE(img()),
					XX(other_projected_com), YY(other_projected_com), ZZ(other_projected_com));

			// Sum the Fourier transforms of all the obodies
			Fsubtract += Faux;
		} // end for obody

		// Set orientations back into the original RELION system of coordinates
		Matrix2D<RFLOAT> Abody;

		// Write out the rot,tilt,psi as the combination of Aori and Aresi!! So get rid of the rotations around the tilt=90 axes,
		Abody = Aori * (opt.mymodel.orient_bodies[subtract_body]).transpose() * A_rot90 * Aresi_subtract * opt.mymodel.orient_bodies[subtract_body];
		Euler_matrix2angles(Abody, rot, tilt, psi);

		// Store the optimal orientations in the MDimg table
		opt.mydata.MDimg.setValue(EMDL_ORIENT_ROT, rot, part_id);
		opt.mydata.MDimg.setValue(EMDL_ORIENT_TILT, tilt, part_id);
		opt.mydata.MDimg.setValue(EMDL_ORIENT_PSI, psi, part_id);

		// Also get refined offset for this body
		opt.mydata.MDbodies[subtract_body].getValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, XX(my_refined_ibody_offset), part_id);
		opt.mydata.MDbodies[subtract_body].getValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, YY(my_refined_ibody_offset), part_id);
		if (opt.mymodel.data_dim == 3)
		{
			opt.mydata.MDbodies[subtract_body].getValue(EMDL_ORIENT_ORIGIN_Z_ANGSTROM, ZZ(my_refined_ibody_offset), part_id);
		}
		// As of v3.1, offsets are in Angstrom: convert back to pixels!
		my_refined_ibody_offset /= my_pixel_size;

		// re-center to new_center
		my_residual_offset += my_refined_ibody_offset;
		my_residual_offset += Abody * (opt.mymodel.com_bodies[subtract_body] - new_center * opt.mymodel.pixel_size / my_pixel_size);
	}
	else
	{
		// Normal 3D classification/refinement: get the projection in rot,tilt,psi for the corresponding class
		Matrix2D<RFLOAT> A3D_pure_rot, A3D;
		Euler_angles2matrix(rot, tilt, psi, A3D_pure_rot);

		// Apply anisotropic mag and scaling
		A3D = opt.mydata.obsModel.applyAnisoMag(A3D_pure_rot, optics_group);
		A3D = opt.mydata.obsModel.applyScaleDifference(A3D, optics_group, opt.mymodel.ori_size, opt.mymodel.pixel_size);
		opt.mymodel.PPref[myclass].get2DFourierTransform(Fsubtract, A3D);

		// Shift in opposite direction as offsets in the STAR file
		shiftImageInFourierTransform(Fsubtract, Fsubtract, (RFLOAT)XSIZE(img()),
				-XX(my_old_offset), -YY(my_old_offset), -ZZ(my_old_offset));

		if (do_center)
		{
			// Re-center the output particle to a new centre...
			my_residual_offset = my_old_offset - A3D_pure_rot * (new_center * opt.mymodel.pixel_size / my_pixel_size);
		}
	}


	// Apply the CTF to the to-be-subtracted projection
	if (opt.do_ctf_correction)
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fsubtract)
		{
			DIRECT_MULTIDIM_ELEM(Fsubtract, n) *= DIRECT_MULTIDIM_ELEM(Fctf, n);
		}

		// Also do phase modulation, for beam tilt correction and other asymmetric aberrations
		opt.mydata.obsModel.demodulatePhase(optics_group, Fsubtract, true); // true means do_modulate_instead
		opt.mydata.obsModel.divideByMtf(optics_group, Fsubtract, true); // true means do_multiply_instead
	}


	if (opt.do_scale_correction)
	{
		int group_id = opt.mydata.getGroupId(part_id);
		RFLOAT myscale = opt.mymodel.scale_correction[group_id];
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fsubtract)
		{
			DIRECT_MULTIDIM_ELEM(Fsubtract, n) *= myscale;
		}
	}

	// Do the actual subtraction
	Fimg -= Fsubtract;

	if (do_ssnr)
	{
		// Don't write out subtracted image,
		// only accumulate power of the signal (in Fsubtract) divided by the power of the noise (now in Fimg)
		FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fimg)
		{
			long int idx = ROUND(sqrt(kp*kp + ip*ip + jp*jp));
			int idx_remapped = ROUND(remap_image_sizes * idx);
			if (idx_remapped < opt.mymodel.ori_size/2 + 1)
			{
				RFLOAT S2 = norm( dAkij(Fsubtract, k, i, j) );
				RFLOAT N2 = norm( dAkij(Fimg, k, i, j) );
				// division by two keeps the numbers similar to tau2 and sigma2_noise,
				// which are per real/imaginary component
				sum_S2(idx_remapped) += S2 / 2.;
				sum_N2(idx_remapped) += N2 / 2.;
				sum_count(idx_remapped) += 1.;
			}
		}
	}
	else
	{
		// And go finally back to real-space
		CenterFFTbySign(Fimg);
		transformer.inverseFourierTransform(Fimg, img());

		if (do_center || opt.fn_body_masks != "None")
		{
			// Recenter the particles
			centering_offset = my_residual_offset;
			centering_offset.selfROUND();
			my_residual_offset -= centering_offset;
			selfTranslate(img(), centering_offset, WRAP);

			// Set the non-integer difference between the rounded centering offset and the actual offsets in the STAR file
			opt.mydata.MDimg.setValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, my_pixel_size * XX(my_residual_offset), part_id);
			opt.mydata.MDimg.setValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, my_pixel_size * YY(my_residual_offset), part_id);
			if (opt.mymodel.data_dim == 3)
			{
				opt.mydata.MDimg.setValue(EMDL_ORIENT_ORIGIN_Z_ANGSTROM, my_pixel_size * ZZ(my_residual_offset), part_id);
			}
		}

		// Rebox the image
		if (boxsize > 0)
		{
			if (img().getDim() == 2)
			{
				img().window(FIRST_XMIPP_INDEX(boxsize), FIRST_XMIPP_INDEX(boxsize),
						   LAST_XMIPP_INDEX(boxsize),  LAST_XMIPP_INDEX(boxsize));
			}
			else if (img().getDim() == 3)
			{
				img().window(FIRST_XMIPP_INDEX(boxsize), FIRST_XMIPP_INDEX(boxsize), FIRST_XMIPP_INDEX(boxsize),
						   LAST_XMIPP_INDEX(boxsize),  LAST_XMIPP_INDEX(boxsize),  LAST_XMIPP_INDEX(boxsize));
			}
		}

		// Now write out the image & set filenames in output metadatatable
		FileName fn_img = getParticleName(counter, rank, optics_group);
		opt.mydata.MDimg.setValue(EMDL_IMAGE_NAME, fn_img, part_id);
		opt.mydata.MDimg.setValue(EMDL_IMAGE_ORI_NAME, opt.mydata.particles[part_id].name, part_id);
		//Also set the original order in the input STAR file for later combination
		opt.mydata.MDimg.setValue(EMDL_IMAGE_ID, part_id, part_id);
		MDimg_out.addObject();
		MDimg_out.setObject(opt.mydata.MDimg.getObject(part_id));

		//printf("Writing: fn_orig = %s counter = %ld rank = %d optics_group = %d fn_img = %s SIZE = %d nr_particles_in_optics_group[optics_group] = %d\n", fn_orig.c_str(), counter, rank, optics_group+1, fn_img.c_str(), XSIZE(img()), nr_particles_in_optics_group[optics_group]);
		img.setSamplingRateInHeader(my_pixel_size);
		const DataType datatype = write_float16 ? Float16: Float;
		if (opt.mymodel.data_dim == 3)
		{
			std::shared_ptr<Image<RFLOAT> > mine(new Image<RFLOAT>(img));
			AsyncFileWriter::putFile(fn_img, MULTIDIM_SIZE(img()) * sizeof(RFLOAT), [mine, datatype](const std::string &tmp)
			{
				mine->write(tmp, -1, false, WRITE_OVERWRITE, datatype);
			});
		}
		else
		{
			// The particles of a stack go through the background writer, which writes it in one pass
			long int dummy;
			FileName fn_stack;
			fn_img.decompose(dummy, fn_stack);
			std::shared_ptr<AsyncMrcStack> &stack = stacks[fn_stack];
			if (!stack) stack.reset(new AsyncMrcStack(fn_stack, datatype));
			stack->add(img);
		}
	}
}
