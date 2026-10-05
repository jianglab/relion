#ifndef RECT_REFUSAL_H_
#define RECT_REFUSAL_H_

#include <string>
#include <src/error.h>
#include "src/multidim_array.h"

// Programs whose Fourier-shell code assumes a square or cubic box call these to stop with a clear message.
inline void refuseRectangularImages(const std::string &program, const std::string &detail = "")
{
	REPORT_ERROR("ERROR: rectangular particle images are not supported by " + program
		+ (detail.empty() ? std::string(".") : " (" + detail + ").")
		+ " It needs square particles; use relion_refine, relion_reconstruct or relion_preprocess for rectangular boxes.");
}

template <typename T>
inline void refuseNonCubicMap(const MultidimArray<T> &map, const std::string &program, const std::string &what = "map")
{
	if (ZSIZE(map) > 1 && (XSIZE(map) != YSIZE(map) || XSIZE(map) != ZSIZE(map)))
		refuseRectangularImages(program, "the " + what + " is " + std::to_string(XSIZE(map)) + " x "
			+ std::to_string(YSIZE(map)) + " x " + std::to_string(ZSIZE(map)) + " voxels, not cubic");
}

template <typename T>
inline void refuseNonSquareImage(const MultidimArray<T> &img, const std::string &program, const std::string &what = "image")
{
	if (ZSIZE(img) == 1 && XSIZE(img) != YSIZE(img))
		refuseRectangularImages(program, "the " + what + " is " + std::to_string(XSIZE(img)) + " x "
			+ std::to_string(YSIZE(img)) + " pixels, not square");
}

#endif
