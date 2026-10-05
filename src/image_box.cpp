/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/image_box.h"

bool Box::fromTable(const MetaDataTable &mdt, long int row, int dim, Box &out)
{
	int sx, sy, sz = 1;
	if (mdt.containsLabel(EMDL_IMAGE_SIZE_X) && mdt.containsLabel(EMDL_IMAGE_SIZE_Y))
	{
		mdt.getValue(EMDL_IMAGE_SIZE_X, sx, row);
		mdt.getValue(EMDL_IMAGE_SIZE_Y, sy, row);
		if (dim == 3)
		{
			if (!mdt.containsLabel(EMDL_IMAGE_SIZE_Z))
				REPORT_ERROR("Box::fromTable: a 3D box needs rlnImageSizeZ as well as rlnImageSizeX and rlnImageSizeY.");
			mdt.getValue(EMDL_IMAGE_SIZE_Z, sz, row);
		}
		out = Box(sx, sy, sz, out.pixel_size);
		return true;
	}
	if (mdt.containsLabel(EMDL_IMAGE_SIZE))
	{
		mdt.getValue(EMDL_IMAGE_SIZE, sx, row);
		out = Box::fromSize(sx, dim, out.pixel_size);
		return true;
	}
	return false;
}

void Box::toTable(MetaDataTable &mdt, long int row) const
{
	if (isSquare())
	{
		mdt.setValue(EMDL_IMAGE_SIZE, nx, row);
		return;
	}
	mdt.setValue(EMDL_IMAGE_SIZE_X, nx, row);
	mdt.setValue(EMDL_IMAGE_SIZE_Y, ny, row);
	if (nz > 1) mdt.setValue(EMDL_IMAGE_SIZE_Z, nz, row);
}
