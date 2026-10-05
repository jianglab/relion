/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef IMAGE_BOX_H
#define IMAGE_BOX_H

/* Box: the size and pixel size of a 2D image or 3D volume that need not be
 * square or cubic, with every frequency expressed in physical units (1/Angstrom)
 * rather than in Fourier-pixel indices.
 *
 * A Box with nx == ny (== nz for volumes) behaves exactly like the single
 * integer "ori_size" RELION has always used: isSquare() is true and size()
 * returns that integer.
 *
 * Fourier indices follow RELION's half-transform layout: kx in [0, nx/2],
 * ky in [-ny/2, ny/2] (wrapped), kz likewise.
 */

#include <cmath>
#include <string>

#include "src/macros.h"
#include "src/metadata_table.h"

struct Box
{
	int nx, ny, nz;
	double pixel_size; ///< Angstrom per pixel; <= 0 means unknown

	Box() : nx(0), ny(0), nz(1), pixel_size(1.0) {}
	Box(int nx_, int ny_, int nz_ = 1, double pixel_size_ = 1.0)
		: nx(nx_), ny(ny_), nz(nz_), pixel_size(pixel_size_) {}

	/// A square image (dim 2) or cubic volume (dim 3) of the old integer size.
	static Box fromSize(int size, int dim = 2, double pixel_size = 1.0)
	{
		return Box(size, size, dim == 3 ? size : 1, pixel_size);
	}

	int dim() const { return nz > 1 ? 3 : 2; }

	bool isSquare() const
	{
		return nx == ny && (nz == 1 || nz == nx);
	}

	/// The old integer size; an error for a box that is not square/cubic.
	int size() const
	{
		if (!isSquare())
			REPORT_ERROR("Box::size(): the box " + str() + " is not square; this code path supports square boxes only.");
		return nx;
	}

	/// Number of Fourier columns of the half transform.
	int halfWidth() const { return nx / 2 + 1; }

	double extentX() const { return nx * pixel_size; }
	double extentY() const { return ny * pixel_size; }
	double extentZ() const { return nz * pixel_size; }

	/// Physical spatial frequency (1/Angstrom) of Fourier index k along each axis.
	double freqX(int kx) const { return kx / extentX(); }
	double freqY(int ky) const { return ky / extentY(); }
	double freqZ(int kz) const { return kz / extentZ(); }

	/// Radial physical frequency (1/Angstrom) of a Fourier index.
	double radialFreq(int kx, int ky, int kz = 0) const
	{
		const double fx = freqX(kx), fy = freqY(ky), fz = (nz > 1) ? freqZ(kz) : 0.0;
		return std::sqrt(fx * fx + fy * fy + fz * fz);
	}

	/// Nyquist frequency (1/Angstrom) along the axis with the finest sampling.
	double nyquist() const { return 0.5 / pixel_size; }

	/// Number of Fourier pixels from the origin to the physical frequency s_max
	/// (1/Angstrom) along each axis, rounded up and limited to Nyquist.
	int limitX(double s_max) const { return limit(s_max, extentX(), nx); }
	int limitY(double s_max) const { return limit(s_max, extentY(), ny); }
	int limitZ(double s_max) const { return limit(s_max, extentZ(), nz); }

	bool operator==(const Box &o) const
	{
		return nx == o.nx && ny == o.ny && nz == o.nz && pixel_size == o.pixel_size;
	}

	std::string str() const
	{
		std::string s = integerToString(nx) + "x" + integerToString(ny);
		if (nz > 1) s += "x" + integerToString(nz);
		return s;
	}

	/// Reads the size from a STAR row: rlnImageSizeX/Y/Z if present, otherwise
	/// rlnImageSize as a square/cube, with dim 2 or 3. Returns false if neither is there.
	static bool fromTable(const MetaDataTable &mdt, long int row, int dim, Box &out);

	/// Writes the size into a STAR table: only rlnImageSize for a square/cubic
	/// box (so square projects are written exactly as before), rlnImageSizeX/Y(/Z) otherwise.
	void toTable(MetaDataTable &mdt, long int row) const;

private:
	static int limit(double s_max, double extent, int n)
	{
		int k = (int)std::ceil(s_max * extent);
		if (k > n / 2) k = n / 2;
		return k;
	}
};

#endif
