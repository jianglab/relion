/***************************************************************************
 *
 * Regenerate a motion-corrected micrograph from its movie and the motion
 * record (MotionCorr/jobNNN/.../<movie>.star) written by RELION's own motion
 * correction with RELION_VIRTUAL_MOVIE_AVERAGES=1. The result is bit for bit
 * the micrograph MotionCorr wrote.
 *
 ***************************************************************************/

#include <src/args.h>
#include <src/image.h>
#include <src/motioncorr_runner.h>

int main(int argc, char *argv[])
{
	try
	{
		IOParser parser;
		parser.setCommandLine(argc, argv);
		parser.addSection("Options");
		const FileName fn_star = parser.getOption("--i", "Motion STAR file written next to the micrograph by MotionCorr");
		const FileName fn_out = parser.getOption("--o", "Output micrograph (.mrc)");
		const int n_threads = textToInteger(parser.getOption("--j", "Number of threads", "1"));
		const bool float16 = parser.checkOption("--float16", "Write in float16, as MotionCorr --float16 does");
		const FileName fn_compare = parser.getOption("--compare", "Report whether the result is identical to this micrograph", "");
		if (parser.checkForErrors())
			REPORT_ERROR("Errors encountered on the command line (see above), exiting...");

		Image<float> Isum;
		MotioncorrRunner::regenerateMicrograph(fn_star, Isum, n_threads);
		Isum.write(fn_out, -1, false, WRITE_OVERWRITE, float16 ? Float16 : Float);

		if (fn_compare != "")
		{
			// Compare what was written (after any float16 rounding) with the stored file
			Image<float> A, B;
			A.read(fn_out);
			B.read(fn_compare);
			if (XSIZE(A()) != XSIZE(B()) || YSIZE(A()) != YSIZE(B()))
			{
				std::cout << "DIFFERENT: size " << XSIZE(A()) << "x" << YSIZE(A()) << " vs " << XSIZE(B()) << "x" << YSIZE(B()) << std::endl;
				return RELION_EXIT_FAILURE;
			}
			long n_diff = 0;
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(A())
				if (DIRECT_MULTIDIM_ELEM(A(), n) != DIRECT_MULTIDIM_ELEM(B(), n)) n_diff++;
			if (n_diff == 0)
				std::cout << "IDENTICAL: " << fn_out << " == " << fn_compare << std::endl;
			else
			{
				std::cout << "DIFFERENT: " << n_diff << " of " << MULTIDIM_SIZE(A()) << " pixels" << std::endl;
				return RELION_EXIT_FAILURE;
			}
		}
	}
	catch (RelionError XE)
	{
		std::cerr << XE;
		return RELION_EXIT_FAILURE;
	}
	return RELION_EXIT_SUCCESS;
}
