/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

// relion_helix_symmetry_report --i run_data.star --model run_model.star [--o root]
// The read-only report that helical Refine3D and Class3D write at their end (src/helix_report.h), for a finished job.

#include <fstream>
#include <iostream>
#include <map>

#include "src/args.h"
#include "src/helix_report.h"
#include "src/metadata_table.h"

int main(int argc, char *argv[])
{
	IOParser parser;
	parser.setCommandLine(argc, argv);
	int general = parser.addSection("General options");
	FileName fn_data = parser.getOption("--i", "The particle star file of the run (run_data.star)");
	FileName fn_model = parser.getOption("--model", "The model star file of the run (for the helical twist and rise)", "");
	RFLOAT twist = textToFloat(parser.getOption("--twist", "Helical twist in degrees (instead of --model)", "0"));
	RFLOAT rise = textToFloat(parser.getOption("--rise", "Helical rise in Angstrom (instead of --model)", "0"));
	FileName fn_out = parser.getOption("--o", "Root of the report files (<root>_helical_symmetry_report.txt and _filaments.csv); empty: print only", "");
	if (parser.checkForErrors())
		return 1;
	try
	{
		if (fn_model != "")
		{
			MetaDataTable mdm;
			mdm.read(fn_model, "model_classes");
			mdm.getValue(EMDL_MLMODEL_HELICAL_TWIST, twist, 0);
			mdm.getValue(EMDL_MLMODEL_HELICAL_RISE, rise, 0);
		}
		MetaDataTable md, optics;
		md.read(fn_data, "particles");
		optics.read(fn_data, "optics");
		std::map<int, double> angpix;
		int g = 1;
		FOR_ALL_OBJECTS_IN_METADATA_TABLE(optics)
		{
			RFLOAT a = 1.;
			int og = g++;
			optics.getValue(EMDL_IMAGE_OPTICS_GROUP, og);
			optics.getValue(EMDL_IMAGE_PIXEL_SIZE, a);
			angpix[og] = a;
		}
		helixreport::Result r = helixreport::analyse(md, [&](int og) { return angpix.count(og) ? angpix[og] : 1.; }, twist, rise);
		if (r.text.empty())
		{
			std::cerr << "The particle table lacks helical tube ids, angles, track lengths, coordinates or origin shifts." << std::endl;
			return 1;
		}
		std::cout << r.text;
		if (fn_out != "")
		{
			std::ofstream(fn_out + "_helical_symmetry_report.txt") << r.text;
			std::ofstream(fn_out + "_helical_symmetry_filaments.csv") << helixreport::csv(r);
		}
	}
	catch (RelionError XE)
	{
		std::cerr << XE;
		return RELION_EXIT_FAILURE;
	}
	return RELION_EXIT_SUCCESS;
}
