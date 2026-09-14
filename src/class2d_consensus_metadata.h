#ifndef SRC_CLASS2D_CONSENSUS_METADATA_H_
#define SRC_CLASS2D_CONSENSUS_METADATA_H_

#include "src/metadata_table.h"

// Keep preparation-specific metadata handling separate from categorical fitting.
// Priors remain intact for RELION's normal helical initialisation/prior searches.
inline void resetClass2DConsensusAlignments(MetaDataTable &particles)
{
	const EMDLabel angles_and_offsets[] = {
		EMDL_ORIENT_ROT, EMDL_ORIENT_TILT, EMDL_ORIENT_PSI,
		EMDL_ORIENT_ORIGIN_X_ANGSTROM, EMDL_ORIENT_ORIGIN_Y_ANGSTROM
	};
	const bool have_zoff = particles.containsLabel(EMDL_ORIENT_ORIGIN_Z_ANGSTROM);
	for (long int row = 0; row < particles.numberOfObjects(); ++row)
	{
		for (EMDLabel label : angles_and_offsets)
			particles.setValue(label, 0., row);
		if (have_zoff) particles.setValue(EMDL_ORIENT_ORIGIN_Z_ANGSTROM, 0., row);
	}
}

#endif
