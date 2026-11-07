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
/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#include "src/multidim_array.h"
#include <atomic>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <sstream>

namespace
{
	std::atomic<bool> huge_enabled(false);
	std::mutex huge_mutex;
	long huge_used = 0, huge_skipped = 0;
	double huge_used_gb = 0.0, huge_skipped_gb = 0.0;

	// Free 2 MB blocks the kernel has without compacting memory (orders 9 and up in /proc/buddyinfo)
	long freeHugeBlocks()
	{
		std::ifstream in("/proc/buddyinfo");
		std::string line;
		long n = 0;

		while (std::getline(in, line))
		{
			const size_t z = line.find("zone");
			if (z == std::string::npos) continue;

			std::istringstream counts(line.substr(z + 4));
			std::string zone;
			counts >> zone;

			long c;
			for (int order = 0; counts >> c; order++)
			{
				if (order >= 9) n += c << (order - 9);
			}
		}

		return n;
	}
}

void hugepages::enable()
{
	huge_enabled = true;
}

void hugepages::advise(void *data, size_t bytes)
{
#ifdef MADV_HUGEPAGE
	const size_t huge = (size_t)2 << 20;
	if (data == NULL || bytes < 32 * huge) return;

	static const int mode = []{ const char *e = getenv("RELION_HUGE_PAGES"); return e ? (atoi(e) > 0 ? 1 : 0) : -1; }();
	if (mode == 0 || (mode < 0 && !huge_enabled)) return;

	const uintptr_t b = ((uintptr_t)data + huge - 1) & ~(uintptr_t)(huge - 1);
	const uintptr_t e = ((uintptr_t)data + bytes) & ~(uintptr_t)(huge - 1);

	std::lock_guard<std::mutex> lock(huge_mutex);
	const bool used = freeHugeBlocks() >= 2 * (long)((e - b) / huge) && madvise((void*)b, e - b, MADV_HUGEPAGE) == 0;

	(used? huge_used : huge_skipped)++;
	(used? huge_used_gb : huge_skipped_gb) += bytes / 1e9;
#endif
}

std::string hugepages::report()
{
	std::lock_guard<std::mutex> lock(huge_mutex);
	if (huge_used + huge_skipped == 0) return "";

	char line[256];
	snprintf(line, sizeof(line), "huge memory pages for %ld large arrays (%.1f GB); not for %ld (%.1f GB), "
	         "for lack of free 2 MB blocks. RELION_HUGE_PAGES=0 turns this off.",
	         huge_used, huge_used_gb, huge_skipped, huge_skipped_gb);
	return line;
}


// Show a complex array ---------------------------------------------------
std::ostream& operator<<(std::ostream& ostrm,
    const MultidimArray< Complex >& v)
{
    if (v.xdim == 0)
        ostrm << "NULL MultidimArray\n";
    else
        ostrm << std::endl;

    for (int l = 0; l < NSIZE(v); l++)
    {
        if (NSIZE(v)>1) ostrm << "Image No. " << l << std::endl;
        for (int k = STARTINGZ(v); k <= FINISHINGZ(v); k++)
        {
            if (ZSIZE(v)>1) ostrm << "Slice No. " << k << std::endl;
            for (int i = STARTINGY(v); i <= FINISHINGY(v); i++)
            {
                for (int j = STARTINGX(v); j <= FINISHINGX(v); j++)
                    ostrm << "(" << A3D_ELEM(v, k, i, j).real << "," << A3D_ELEM(v, k, i, j).imag <<")" << ' ';
                ostrm << std::endl;
            }
        }
    }

    return ostrm;
}
