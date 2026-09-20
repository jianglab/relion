#ifndef CRYOSPARC_CS_H
#define CRYOSPARC_CS_H

/* Typed, by-name access to the columns of a CryoSPARC .cs file, plus the two
 * other things that need CryoSPARC's own readers: the .cs -> STAR conversion
 * and reading a .npy trajectory.
 *
 * This is a declaration-only facade on purpose. The vendored npy.hpp defines
 * explicit template specializations at namespace scope, so it can be included
 * in exactly one translation unit of a library without the linker reporting
 * multiple definitions. cryosparc_cs.cpp is that translation unit; everything
 * else in the importer goes through this header.
 */

#include <string>
#include <vector>
#include <memory>

namespace cryosparc {

/// One CryoSPARC .cs file (a NumPy structured array), addressed by column name.
/// Columns may be scalar ("micrograph_blob/psize_A") or fixed-size vectors
/// ("movie_blob/shape" is (3,)), which is what `component` selects.
class CsTable {
public:
	explicit CsTable(const std::string& filename);
	~CsTable();

	size_t rows() const;
	const std::string& filename() const;

	bool has(const std::string& field) const;
	std::vector<std::string> fieldNames() const;
	int components(const std::string& field) const;

	double      getDouble(size_t row, const std::string& field, int component = 0) const;
	long        getInt   (size_t row, const std::string& field, int component = 0) const;
	std::string getString(size_t row, const std::string& field) const;

private:
	struct Impl;
	std::shared_ptr<Impl> impl_;
};

/// Convert a .cs table (merging passthrough files on uid) to a RELION STAR file.
void convertCsToStar(const std::string& cs_filename,
                     const std::string& star_filename,
                     const std::string& optics_group_name,
                     double pixel_size, double kV, double Cs, double Q0,
                     const std::string& passthrough_filenames);

/// Read a .npy array of doubles. Returns false (with a warning) if unreadable.
bool loadNpyDouble(const std::string& path,
                   std::vector<unsigned long>& shape,
                   std::vector<double>& data);

} // namespace cryosparc

#endif // CRYOSPARC_CS_H
