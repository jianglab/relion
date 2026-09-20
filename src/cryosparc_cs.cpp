#include "src/cryosparc_cs.h"

// The one library translation unit that may include these; see cryosparc_cs.h.
#include "src/cryosparc_import.h"
#include "src/npy.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace cryosparc {

struct CsTable::Impl {
	std::string filename;
	CsHeader header;
	std::vector<char> data;

	const char* rowPtr(size_t row) const
	{
		if (row >= header.num_rows)
			throw std::runtime_error("row out of range in " + filename);
		return &data[row * header.total_itemsize];
	}

	const CsField& field(const std::string& name) const
	{
		const int idx = find_field(header.fields, name);
		if (idx < 0)
			throw std::runtime_error("no field '" + name + "' in " + filename);
		return header.fields[idx];
	}
};

CsTable::CsTable(const std::string& filename) : impl_(new Impl())
{
	impl_->filename = filename;
	impl_->header = read_header(filename);
	impl_->data = read_data(filename, impl_->header);
}

CsTable::~CsTable() {}

size_t CsTable::rows() const { return impl_->header.num_rows; }
const std::string& CsTable::filename() const { return impl_->filename; }

bool CsTable::has(const std::string& field) const
{
	return find_field(impl_->header.fields, field) >= 0;
}

std::vector<std::string> CsTable::fieldNames() const
{
	std::vector<std::string> out;
	for (size_t i = 0; i < impl_->header.fields.size(); i++)
		out.push_back(impl_->header.fields[i].name);
	return out;
}

int CsTable::components(const std::string& field) const
{
	const CsField& f = impl_->field(field);
	int n = 1;
	for (size_t i = 0; i < f.subshape.size(); i++) n *= f.subshape[i];
	return n;
}

double CsTable::getDouble(size_t row, const std::string& field, int component) const
{
	const CsField& f = impl_->field(field);
	return read_double(impl_->rowPtr(row), f.offset + component * f.itemsize, f.kind, f.itemsize);
}

long CsTable::getInt(size_t row, const std::string& field, int component) const
{
	const CsField& f = impl_->field(field);
	return read_int(impl_->rowPtr(row), f.offset + component * f.itemsize, f.kind, f.itemsize);
}

std::string CsTable::getString(size_t row, const std::string& field) const
{
	const CsField& f = impl_->field(field);
	return read_string(impl_->rowPtr(row), f.offset, f.itemsize);
}

void convertCsToStar(const std::string& cs_filename,
                     const std::string& star_filename,
                     const std::string& optics_group_name,
                     double pixel_size, double kV, double Cs, double Q0,
                     const std::string& passthrough_filenames)
{
	convert(cs_filename, star_filename, optics_group_name,
	        (RFLOAT)pixel_size, (RFLOAT)kV, (RFLOAT)Cs, (RFLOAT)Q0,
	        passthrough_filenames);
}

bool loadNpyDouble(const std::string& path,
                   std::vector<unsigned long>& shape,
                   std::vector<double>& data)
{
	// npy.hpp's 3-argument LoadArrayFromNumpy calls a 4-argument overload that
	// is declared after it, so unqualified lookup cannot find it; use the
	// 4-argument form directly.
	bool fortran_order = false;
	try { npy::LoadArrayFromNumpy<double>(path, shape, fortran_order, data); }
	catch (const std::exception& e)
	{
		std::cerr << " WARNING: cannot read " << path << ": " << e.what() << std::endl;
		return false;
	}
	return true;
}

} // namespace cryosparc
