#ifndef CRYOSPARC_IMPORT_H
#define CRYOSPARC_IMPORT_H

#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <sys/stat.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <unistd.h>

#include "src/metadata_table.h"
#include "src/npy.hpp"
#include "src/Eigen/Dense"

namespace cryosparc {

struct CsField {
    std::string name;
    char kind;
    int itemsize;
    int offset;
    std::vector<int> subshape;
};

struct CsHeader {
    std::vector<CsField> fields;
    size_t num_rows;
    int total_itemsize;
    bool fortran_order;
};

// Parse a Python literal string (trim quotes)
static std::string py_unstring(const std::string& s) {
    std::string t = s;
    t.erase(0, t.find_first_not_of(" \t\r\n"));
    t.erase(t.find_last_not_of(" \t\r\n") + 1);
    if (t.size() >= 2 && t.front() == '\'' && t.back() == '\'')
        return t.substr(1, t.size() - 2);
    if (t.size() >= 2 && t.front() == '"' && t.back() == '"')
        return t.substr(1, t.size() - 2);
    return t;
}

// Split a comma-separated list respecting parentheses/brackets
static std::vector<std::string> split_top_level(const std::string& s, char delim) {
    std::vector<std::string> parts;
    int depth = 0;
    size_t start = 0;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '(' || s[i] == '[' || s[i] == '{') depth++;
        else if (s[i] == ')' || s[i] == ']' || s[i] == '}') depth--;
        else if (s[i] == delim && depth == 0) {
            parts.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    if (start < s.size())
        parts.push_back(s.substr(start));
    return parts;
}

// Parse a tuple like ('name', 'typestr') or ('name', 'typestr', (shape,))
static void parse_field_tuple(const std::string& t, std::string& name,
                               std::string& typestr, std::vector<int>& shape) {
    std::string s = t;
    s.erase(0, s.find_first_not_of(" \t\r\n"));
    s.erase(s.find_last_not_of(" \t\r\n") + 1);
    if (s.size() < 2 || s.front() != '(' || s.back() != ')')
        throw std::runtime_error("Expected tuple in dtype list, got: " + t);
    s = s.substr(1, s.size() - 2);

    auto parts = split_top_level(s, ',');
    if (parts.size() < 2)
        throw std::runtime_error("Expected at least 2 elements in dtype tuple");

    name = py_unstring(parts[0]);
    typestr = py_unstring(parts[1]);

    shape.clear();
    if (parts.size() >= 3) {
        // Third element is a shape tuple like (2,) or (3,4)
        std::string shape_str = parts[2];
        shape_str.erase(0, shape_str.find_first_not_of(" \t\r\n"));
        shape_str.erase(shape_str.find_last_not_of(" \t\r\n") + 1);
        if (shape_str.front() == '(' && shape_str.back() == ')') {
            shape_str = shape_str.substr(1, shape_str.size() - 2);
            auto dims = split_top_level(shape_str, ',');
            for (auto& d : dims) {
                d.erase(0, d.find_first_not_of(" \t\r\n"));
                if (!d.empty())
                    shape.push_back(std::stoi(d));
            }
        }
    }
}

// Parse dtype string like '<f8', '<i4', '|S256'
static void parse_typestr(const std::string& typestr, char& kind, int& itemsize) {
    if (typestr.size() < 3)
        throw std::runtime_error("Invalid typestring: " + typestr);
    kind = typestr[1];
    itemsize = std::stoi(typestr.substr(2));
}

static CsHeader read_header(const std::string& filename) {
    std::ifstream stream(filename, std::ifstream::binary);
    if (!stream)
        throw std::runtime_error("Cannot open file: " + filename);

    std::string header_s = npy::read_header(stream);

    // Extract descr value
    size_t descr_pos = header_s.find("'descr': ");
    if (descr_pos == std::string::npos)
        throw std::runtime_error("Cannot find 'descr' in npy header");
    descr_pos += 9; // skip "'descr': "
    descr_pos = header_s.find_first_not_of(" \t", descr_pos);
    if (descr_pos == std::string::npos || header_s[descr_pos] != '[')
        throw std::runtime_error("Expected structured dtype (list) in .cs file");

    size_t end_pos = header_s.find(", 'fortran_order'", descr_pos);
    if (end_pos == std::string::npos)
        throw std::runtime_error("Cannot find end of descr list");
    std::string list_str = header_s.substr(descr_pos, end_pos - descr_pos);

    // Parse the list
    if (list_str.size() < 2 || list_str.front() != '[' || list_str.back() != ']')
        throw std::runtime_error("Invalid dtype list format");
    list_str = list_str.substr(1, list_str.size() - 2);

    auto field_strs = split_top_level(list_str, ',');
    std::vector<CsField> fields;
    int offset = 0;

    for (auto& fs : field_strs) {
        fs.erase(0, fs.find_first_not_of(" \t\r\n"));
        fs.erase(fs.find_last_not_of(" \t\r\n") + 1);
        if (fs.empty()) continue;

        std::string name, typestr;
        std::vector<int> subshape;
        parse_field_tuple(fs, name, typestr, subshape);

        char kind;
        int itemsize;
        parse_typestr(typestr, kind, itemsize);

        int total_size = itemsize;
        for (int d : subshape)
            total_size *= d;

        CsField f;
        f.name = name;
        f.kind = kind;
        f.itemsize = itemsize;
        f.offset = offset;
        f.subshape = subshape;
        fields.push_back(f);

        offset += total_size;
    }

    // Extract shape
    size_t shape_pos = header_s.find("'shape': ");
    if (shape_pos == std::string::npos)
        throw std::runtime_error("Cannot find 'shape' in npy header");
    shape_pos += 9;
    shape_pos = header_s.find_first_not_of(" \t", shape_pos);
    if (shape_pos == std::string::npos || header_s[shape_pos] != '(')
        throw std::runtime_error("Expected shape tuple");
    size_t shape_end = header_s.find(")", shape_pos);
    if (shape_end == std::string::npos)
        throw std::runtime_error("Invalid shape tuple");
    std::string shape_str = header_s.substr(shape_pos + 1, shape_end - shape_pos - 1);
    auto dims = split_top_level(shape_str, ',');
    size_t num_rows = 1;
    for (auto& d : dims) {
        d.erase(0, d.find_first_not_of(" \t\r\n"));
        if (!d.empty())
            num_rows *= std::stoul(d);
    }

    // Extract fortran_order
    bool fortran_order = false;
    size_t fo_pos = header_s.find("'fortran_order': ");
    if (fo_pos != std::string::npos) {
        fo_pos += 17;
        std::string fo_val = header_s.substr(fo_pos);
        fo_val.erase(0, fo_val.find_first_not_of(" \t"));
        if (fo_val.compare(0, 4, "True") == 0)
            fortran_order = true;
    }

    CsHeader hdr;
    hdr.fields = fields;
    hdr.num_rows = num_rows;
    hdr.total_itemsize = offset;
    hdr.fortran_order = fortran_order;
    return hdr;
}

static std::vector<char> read_data(const std::string& filename, const CsHeader& hdr) {
    std::ifstream stream(filename, std::ifstream::binary);
    if (!stream)
        throw std::runtime_error("Cannot open file: " + filename);

    // Skip header manually instead of using npy::read_header, which uses a
    // reserve/data() pattern that can leave the stream in a bad state.
    npy::version_t ver = npy::read_magic(stream);
    uint32_t header_len;
    if (ver.first == 1) {
        unsigned char buf[2];
        stream.read(reinterpret_cast<char*>(buf), 2);
        header_len = (uint32_t)buf[0] | ((uint32_t)buf[1] << 8);
    } else {
        unsigned char buf[4];
        stream.read(reinterpret_cast<char*>(buf), 4);
        header_len = (uint32_t)buf[0] | ((uint32_t)buf[1] << 8) |
                     ((uint32_t)buf[2] << 16) | ((uint32_t)buf[3] << 24);
    }
    stream.seekg(header_len, std::ios::cur);

    size_t data_size = hdr.num_rows * hdr.total_itemsize;
    std::vector<char> data(data_size);
    stream.read(data.data(), data_size);
    if (!stream)
        throw std::runtime_error("Failed to read data from: " + filename);
    return data;
}

// Extract one double value from raw data at given offset
static long read_int(const char* base, int offset, char kind, int itemsize);
static double read_double(const char* base, int offset, char kind, int itemsize) {
    if (kind == 'f') {
        if (itemsize == 4) {
            float v;
            std::memcpy(&v, base + offset, 4);
            return v;
        } else if (itemsize == 8) {
            double v;
            std::memcpy(&v, base + offset, 8);
            return v;
        }
    } else {
        return (double)read_int(base, offset, kind, itemsize);
    }
    return 0.0;
}

// Extract one integer value from raw data
static long read_int(const char* base, int offset, char kind, int itemsize) {
    (void)kind;
    if (itemsize == 1) {
        int8_t v; std::memcpy(&v, base + offset, 1); return v;
    } else if (itemsize == 2) {
        int16_t v; std::memcpy(&v, base + offset, 2); return v;
    } else if (itemsize == 4) {
        int32_t v; std::memcpy(&v, base + offset, 4); return v;
    } else if (itemsize == 8) {
        int64_t v; std::memcpy(&v, base + offset, 8); return v;
    }
    return 0;
}

// Extract string from fixed-length byte string field
static std::string read_string(const char* base, int offset, int itemsize) {
    const char* s = base + offset;
    // Find null terminator or take up to itemsize
    int len = 0;
    while (len < itemsize && s[len] != '\0') len++;
    return std::string(s, len);
}

static std::string normalize_path(const std::string& path) {
    std::vector<std::string> parts;
    size_t start = (path[0] == '/') ? 1 : 0;
    while (start < path.size()) {
        size_t end = path.find('/', start);
        std::string part = (end == std::string::npos)
            ? path.substr(start)
            : path.substr(start, end - start);
        if (part == "..") {
            if (!parts.empty())
                parts.pop_back();
        } else if (part != "." && !part.empty()) {
            parts.push_back(part);
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    std::string result;
    if (path[0] == '/') result = "/";
    for (size_t i = 0; i < parts.size(); i++) {
        if (i > 0) result += "/";
        result += parts[i];
    }
    return result.empty() ? (path[0] == '/' ? "/" : ".") : result;
}

// Resolve a (possibly relative) path against a base directory.
// If the path is already absolute, return it unchanged.
// Some CryoSPARC .cs files store absolute Unix paths without the leading '/';
// this is detected by checking whether the first path component matches the
// first component of base_dir when the relative result stays relative.
static std::string resolve_path(const std::string& path, const std::string& base_dir) {
    if (path.empty() || path[0] == '/')
        return path;
    if (base_dir.empty())
        return path;

    std::string full = base_dir + "/" + path;
    std::string result = normalize_path(full);

    // If the result is still a relative path (base_dir was also relative),
    // check whether the raw path starts with the same first component as
    // base_dir. If so, the .cs file likely stored an absolute path without
    // the leading '/', and we should use the absolute form.
    if (!result.empty() && result[0] != '/') {
        size_t path_slash = path.find('/');
        size_t base_slash = base_dir.find('/');
        std::string path_first = path.substr(0, path_slash);
        std::string base_first = base_dir.substr(0, base_slash);
        if (path_first == base_first) {
            std::string abs_result = normalize_path("/" + path);
            if (!abs_result.empty() && abs_result[0] == '/')
                result = abs_result;
        }
    }

    return result;
}

// Verify a file path exists; warn once per unique path, up to SAMPLE_LIMIT paths per label.
static void verify_path(const std::string& path, const std::string& label) {
    static const int SAMPLE_LIMIT = 10;
    static std::set<std::string> warned;
    static std::map<std::string, int> verified;
    if (warned.count(path)) return;
    if (verified[label] >= SAMPLE_LIMIT) return;
    verified[label]++;
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        warned.insert(path);
        std::cerr << " WARNING: " << label << " path does not exist: " << path << std::endl;
    }
}

// Check if a field exists
static int find_field(const std::vector<CsField>& fields, const std::string& name) {
    for (int i = 0; i < (int)fields.size(); i++)
        if (fields[i].name == name) return i;
    return -1;
}

// Store a value in MetaDataTable
static void set_value(MetaDataTable& MD, const std::string& rln_name, double val) {
    EMDLabel label = EMDL::str2Label(rln_name);
    if (label != EMDL_UNDEFINED)
        MD.setValue(label, val);
}

static void set_value(MetaDataTable& MD, const std::string& rln_name, long val) {
    EMDLabel label = EMDL::str2Label(rln_name);
    if (label != EMDL_UNDEFINED)
        MD.setValue(label, val);
}

static void set_value(MetaDataTable& MD, const std::string& rln_name, const std::string& val) {
    EMDLabel label = EMDL::str2Label(rln_name);
    if (label != EMDL_UNDEFINED)
        MD.setValue(label, val);
}

static void convert(const std::string& cs_filename,
                     const std::string& star_filename,
                     const std::string& optics_group_name,
                     RFLOAT pixel_size, RFLOAT kV, RFLOAT Cs, RFLOAT Q0,
                     const std::string& passthrough_filename = "")
{
    CsHeader hdr = read_header(cs_filename);
    std::vector<char> data = read_data(cs_filename, hdr);

    // Determine CryoSPARC project root from .cs file location:
    // project_root = dirname(cs_filename)/..
    std::string project_dir;
    {
        size_t slash = cs_filename.rfind('/');
        std::string cs_dir = (slash != std::string::npos) ? cs_filename.substr(0, slash) : ".";
        // Resolve cs_dir to absolute if it is relative, so that project_dir
        // and all subsequently resolved blob paths are absolute.
        if (!cs_dir.empty() && cs_dir[0] != '/') {
            char cwd[4096];
            if (getcwd(cwd, sizeof(cwd))) {
                cs_dir = normalize_path(std::string(cwd) + "/" + cs_dir);
            }
        }
        project_dir = resolve_path("..", cs_dir);
    }

    // Read passthrough file if provided
    CsHeader hdr_pt;
    std::vector<char> data_pt;
    std::map<long, size_t> pt_uid_to_row;
    int pt_fi_uid = -1;
    bool have_passthrough = !passthrough_filename.empty();
    if (have_passthrough) {
        hdr_pt = read_header(passthrough_filename);
        data_pt = read_data(passthrough_filename, hdr_pt);
        pt_fi_uid = find_field(hdr_pt.fields, "uid");
        if (pt_fi_uid >= 0) {
            for (size_t i = 0; i < hdr_pt.num_rows; i++) {
                const char* row = data_pt.data() + i * hdr_pt.total_itemsize;
                long uid = read_int(row + hdr_pt.fields[pt_fi_uid].offset, 0,
                                    hdr_pt.fields[pt_fi_uid].kind,
                                    hdr_pt.fields[pt_fi_uid].itemsize);
                pt_uid_to_row[uid] = i;
            }
        }
    }

    // Find key fields (main file)
    int fi_blob_path   = find_field(hdr.fields, "blob/path");
    int fi_blob_idx    = find_field(hdr.fields, "blob/idx");
    int fi_blob_psize  = find_field(hdr.fields, "blob/psize_A");
    int fi_blob_shape  = find_field(hdr.fields, "blob/shape");
    int fi_mic_path    = find_field(hdr.fields, "micrograph_blob/path");
    int fi_mic_psize   = find_field(hdr.fields, "micrograph_blob/psize_A");
    int fi_loc_path    = find_field(hdr.fields, "location/micrograph_path");
    int fi_loc_cx      = find_field(hdr.fields, "location/center_x_frac");
    int fi_loc_cy      = find_field(hdr.fields, "location/center_y_frac");
    int fi_loc_shape   = find_field(hdr.fields, "location/micrograph_shape");
    int fi_mic_shape   = find_field(hdr.fields, "micrograph_blob/shape");
    int fi_movie_path  = find_field(hdr.fields, "movie_blob/path");
    int fi_ctf_kv      = find_field(hdr.fields, "ctf/accel_kv");
    int fi_ctf_cs      = find_field(hdr.fields, "ctf/cs_mm");
    int fi_ctf_q0      = find_field(hdr.fields, "ctf/amp_contrast");
    int fi_ctf_df1     = find_field(hdr.fields, "ctf/df1_A");
    int fi_ctf_df2     = find_field(hdr.fields, "ctf/df2_A");
    int fi_ctf_dfang   = find_field(hdr.fields, "ctf/df_angle_rad");
    int fi_ctf_pshift  = find_field(hdr.fields, "ctf/phase_shift_rad");
    int fi_ctf_bfac    = find_field(hdr.fields, "ctf/bfactor");
    int fi_ctf_scale   = find_field(hdr.fields, "ctf/scale");
    int fi_ctf_fit     = find_field(hdr.fields, "ctf/ctf_fit_to_A");
    int fi_ctf_tilt    = find_field(hdr.fields, "ctf/tilt_A");
    int fi_ctf_anisomag= find_field(hdr.fields, "ctf/anisomag");
    int fi_align2d_cls = find_field(hdr.fields, "alignments2D/class");
    int fi_align2d_sh  = find_field(hdr.fields, "alignments2D/shift");
    int fi_align2d_ps  = find_field(hdr.fields, "alignments2D/pose");
    int fi_align3d_cls = find_field(hdr.fields, "alignments3D/class");
    int fi_align3d_cc  = find_field(hdr.fields, "alignments3D/cross_cor");
    int fi_align3d_ps  = find_field(hdr.fields, "alignments3D/pose");
    int fi_align3d_sh  = find_field(hdr.fields, "alignments3D/shift");
    int fi_align3d_sp  = find_field(hdr.fields, "alignments3D/split");
    int fi_fil_uid     = find_field(hdr.fields, "filament/filament_uid");
    int fi_fil_pose    = find_field(hdr.fields, "filament/filament_pose");
    int fi_fil_posA    = find_field(hdr.fields, "filament/position_A");

    // Also find uid in main file (for passthrough lookup)
    int fi_uid = find_field(hdr.fields, "uid");

    // Pre-compute passthrough field indices for fallback
    int pt_fi_mic_path    = have_passthrough ? find_field(hdr_pt.fields, "micrograph_blob/path") : -1;
    int pt_fi_mic_psize   = have_passthrough ? find_field(hdr_pt.fields, "micrograph_blob/psize_A") : -1;
    int pt_fi_loc_path    = have_passthrough ? find_field(hdr_pt.fields, "location/micrograph_path") : -1;
    int pt_fi_loc_cx      = have_passthrough ? find_field(hdr_pt.fields, "location/center_x_frac") : -1;
    int pt_fi_loc_cy      = have_passthrough ? find_field(hdr_pt.fields, "location/center_y_frac") : -1;
    int pt_fi_loc_shape   = have_passthrough ? find_field(hdr_pt.fields, "location/micrograph_shape") : -1;
    int pt_fi_mic_shape   = have_passthrough ? find_field(hdr_pt.fields, "micrograph_blob/shape") : -1;
    int pt_fi_movie_path  = have_passthrough ? find_field(hdr_pt.fields, "movie_blob/path") : -1;
    int pt_fi_ctf_kv      = have_passthrough ? find_field(hdr_pt.fields, "ctf/accel_kv") : -1;
    int pt_fi_ctf_cs      = have_passthrough ? find_field(hdr_pt.fields, "ctf/cs_mm") : -1;
    int pt_fi_ctf_q0      = have_passthrough ? find_field(hdr_pt.fields, "ctf/amp_contrast") : -1;
    int pt_fi_ctf_df1     = have_passthrough ? find_field(hdr_pt.fields, "ctf/df1_A") : -1;
    int pt_fi_ctf_df2     = have_passthrough ? find_field(hdr_pt.fields, "ctf/df2_A") : -1;
    int pt_fi_ctf_dfang   = have_passthrough ? find_field(hdr_pt.fields, "ctf/df_angle_rad") : -1;
    int pt_fi_ctf_pshift  = have_passthrough ? find_field(hdr_pt.fields, "ctf/phase_shift_rad") : -1;
    int pt_fi_ctf_bfac    = have_passthrough ? find_field(hdr_pt.fields, "ctf/bfactor") : -1;
    int pt_fi_ctf_scale   = have_passthrough ? find_field(hdr_pt.fields, "ctf/scale") : -1;
    int pt_fi_ctf_fit     = have_passthrough ? find_field(hdr_pt.fields, "ctf/ctf_fit_to_A") : -1;
    int pt_fi_ctf_tilt    = have_passthrough ? find_field(hdr_pt.fields, "ctf/tilt_A") : -1;
    int pt_fi_ctf_anisomag = have_passthrough ? find_field(hdr_pt.fields, "ctf/anisomag") : -1;
    int pt_fi_blob_psize  = have_passthrough ? find_field(hdr_pt.fields, "blob/psize_A") : -1;
    int pt_fi_blob_shape  = have_passthrough ? find_field(hdr_pt.fields, "blob/shape") : -1;

    bool has_blob_info = (fi_blob_path >= 0 && fi_blob_idx >= 0);
    bool have_micrographs = (fi_mic_path >= 0 || fi_loc_path >= 0);
    bool have_particles = has_blob_info;

    // Detect image size once
    // blob/shape stores integer pixel dimensions, often as uint16.
    // read_int handles all integer widths; read_double only handles 4/8 byte floats.
    int image_size = 0;
    if (fi_blob_shape >= 0 && hdr.num_rows > 0) {
        const auto& f = hdr.fields[fi_blob_shape];
        const char* row = data.data();
        image_size = (int)read_int(row + f.offset, 0, f.kind, f.itemsize);
    } else if (have_passthrough && pt_fi_blob_shape >= 0 && hdr_pt.num_rows > 0) {
        const auto& f = hdr_pt.fields[pt_fi_blob_shape];
        const char* row = data_pt.data();
        image_size = (int)read_int(row + f.offset, 0, f.kind, f.itemsize);
    }

    const double DEG_PER_RAD = 180.0 / M_PI;

    // Pre-pass: detect unique optics groups via ctf/exp_group_id.
    // If present, it maps 1:1 to RELION rlnOpticsGroup.
    // Optics params (kV, Cs, Q0, psize, tilt, anisomag) are extracted from
    // the first occurrence of each group ID.
    int fi_expgroup = find_field(hdr.fields, "ctf/exp_group_id");
    int pt_fi_expgroup = have_passthrough ? find_field(hdr_pt.fields, "ctf/exp_group_id") : -1;
    bool has_expgroup = (fi_expgroup >= 0 || pt_fi_expgroup >= 0);

    struct OptGroup {
        double voltage, spherical_aberration, amplitude_contrast, image_pixel_size;
        double micrograph_pixel_size;
        double beam_tilt_x, beam_tilt_y;
        double mag00, mag01, mag10, mag11;
        int image_size;
    };
    std::map<long, int> optgroup_map;  // exp_group_id -> RELION group number (1-based)
    std::vector<OptGroup> optgroup_params;
    std::vector<long> row_expgroup(hdr.num_rows, 0);
    std::vector<int> row_optgroup(hdr.num_rows, 1);

    // Helper: get passthrough row for a given main row index
    auto get_pt_row = [&](size_t i) -> const char* {
        if (!have_passthrough || fi_uid < 0 || pt_fi_uid < 0) return nullptr;
        const char* row = data.data() + i * hdr.total_itemsize;
        long uid = read_int(row + hdr.fields[fi_uid].offset, 0,
                            hdr.fields[fi_uid].kind,
                            hdr.fields[fi_uid].itemsize);
        auto it = pt_uid_to_row.find(uid);
        return it != pt_uid_to_row.end() ? data_pt.data() + it->second * hdr_pt.total_itemsize : nullptr;
    };

    auto rd_v = [&](const char* row, const char* pt_row, int fi, int pt_fi, double def) -> double {
        if (fi >= 0) return read_double(row + hdr.fields[fi].offset, 0, hdr.fields[fi].kind, hdr.fields[fi].itemsize);
        if (pt_row && pt_fi >= 0) return read_double(pt_row + hdr_pt.fields[pt_fi].offset, 0, hdr_pt.fields[pt_fi].kind, hdr_pt.fields[pt_fi].itemsize);
        return def;
    };

    struct ExpGroupData {
        long exp_id;       // CryoSPARC group ID (0-based)
        int relion_id;     // RELION group ID (1-based)
    };

    for (size_t i = 0; i < hdr.num_rows; i++) {
        const char* row = data.data() + i * hdr.total_itemsize;
        const char* pt_row = get_pt_row(i);

        long exp_id;
        if (fi_expgroup >= 0)
            exp_id = read_int(row + hdr.fields[fi_expgroup].offset, 0,
                              hdr.fields[fi_expgroup].kind,
                              hdr.fields[fi_expgroup].itemsize);
        else if (pt_row && pt_fi_expgroup >= 0)
            exp_id = read_int(pt_row + hdr_pt.fields[pt_fi_expgroup].offset, 0,
                              hdr_pt.fields[pt_fi_expgroup].kind,
                              hdr_pt.fields[pt_fi_expgroup].itemsize);
        else
            exp_id = 0;

        row_expgroup[i] = exp_id;

        auto it = optgroup_map.find(exp_id);
        if (it == optgroup_map.end()) {
            int relion_id = optgroup_map.size() + 1;
            optgroup_map[exp_id] = relion_id;

            OptGroup og;
            og.voltage = rd_v(row, pt_row, fi_ctf_kv, pt_fi_ctf_kv, kV);
            og.spherical_aberration = rd_v(row, pt_row, fi_ctf_cs, pt_fi_ctf_cs, Cs);
            og.amplitude_contrast = rd_v(row, pt_row, fi_ctf_q0, pt_fi_ctf_q0, Q0);
            og.image_pixel_size = rd_v(row, pt_row, fi_blob_psize, pt_fi_blob_psize, pixel_size);
            og.micrograph_pixel_size = rd_v(row, pt_row, fi_mic_psize, pt_fi_mic_psize, pixel_size);

            og.beam_tilt_x = 0.0; og.beam_tilt_y = 0.0;
            int tilt_fi = (fi_ctf_tilt >= 0) ? fi_ctf_tilt : -1;
            int tilt_pt = (pt_fi_ctf_tilt >= 0) ? pt_fi_ctf_tilt : -1;
            if (tilt_fi >= 0 || (pt_row && tilt_pt >= 0)) {
                bool use_pt = (tilt_fi < 0);
                const auto& f = use_pt ? hdr_pt.fields[tilt_pt] : hdr.fields[tilt_fi];
                const char* src = use_pt ? pt_row : row;
                double tilt_x = read_double(src + f.offset, 0, f.kind, f.itemsize);
                double tilt_y = (f.subshape.size() > 0 && f.subshape[0] > 1)
                    ? read_double(src + f.offset, 1 * f.itemsize, f.kind, f.itemsize) : 0.0;
                // Convert from CryoSPARC units (tilt in Angstrom at specimen) to mrad.
                // Formula from pyem (cryosparc_2_cs_ctf_parameters):
                //   beam_tilt_mrad = arcsin(tilt_A / cs_mm * 1e-7) * 1000
                if (og.spherical_aberration != 0.0) {
                    double arg_x = tilt_x / og.spherical_aberration * 1e-7;
                    double arg_y = tilt_y / og.spherical_aberration * 1e-7;
                    if (arg_x >= -1.0 && arg_x <= 1.0)
                        og.beam_tilt_x = std::asin(arg_x) * 1000.0;
                    if (arg_y >= -1.0 && arg_y <= 1.0)
                        og.beam_tilt_y = std::asin(arg_y) * 1000.0;
                }
            }

            og.mag00 = 0.0; og.mag01 = 0.0; og.mag10 = 0.0; og.mag11 = 0.0;
            int aniso_fi = (fi_ctf_anisomag >= 0) ? fi_ctf_anisomag : -1;
            int aniso_pt = (pt_fi_ctf_anisomag >= 0) ? pt_fi_ctf_anisomag : -1;
            if (aniso_fi >= 0 || (pt_row && aniso_pt >= 0)) {
                bool use_pt = (aniso_fi < 0);
                const auto& f = use_pt ? hdr_pt.fields[aniso_pt] : hdr.fields[aniso_fi];
                const char* src = use_pt ? pt_row : row;
                og.mag00 = read_double(src + f.offset, 0, f.kind, f.itemsize);
                og.mag01 = (f.subshape.size() > 0 && f.subshape[0] > 1) ? read_double(src + f.offset, 1*f.itemsize, f.kind, f.itemsize) : 0.0;
                og.mag10 = (f.subshape.size() > 0 && f.subshape[0] > 2) ? read_double(src + f.offset, 2*f.itemsize, f.kind, f.itemsize) : 0.0;
                og.mag11 = (f.subshape.size() > 0 && f.subshape[0] > 3) ? read_double(src + f.offset, 3*f.itemsize, f.kind, f.itemsize) : 0.0;
            }

            og.image_size = image_size;
            optgroup_params.push_back(og);
            row_optgroup[i] = relion_id;
        } else {
            row_optgroup[i] = it->second;
        }
    }

    // Build optics table
    std::string og_name_base = optics_group_name;
    while (og_name_base.size() > 0 && std::isdigit(og_name_base.back()))
        og_name_base.pop_back();
    MetaDataTable MDopt;
    MDopt.setName("optics");
    for (int g = 0; g < (int)optgroup_params.size(); g++) {
        MDopt.addObject();
        const OptGroup& og = optgroup_params[g];
        int gid = g + 1;
        MDopt.setValue(EMDL_IMAGE_OPTICS_GROUP_NAME,
                       og_name_base + std::to_string(gid));
        MDopt.setValue(EMDL_IMAGE_OPTICS_GROUP, gid);
        MDopt.setValue(EMDL_MICROGRAPH_ORIGINAL_PIXEL_SIZE, pixel_size);
        if (have_micrographs)
            MDopt.setValue(EMDL_MICROGRAPH_PIXEL_SIZE, og.micrograph_pixel_size);
        MDopt.setValue(EMDL_IMAGE_PIXEL_SIZE, og.image_pixel_size);
        MDopt.setValue(EMDL_IMAGE_DIMENSIONALITY, 2);
        if (og.image_size > 0)
            MDopt.setValue(EMDL_IMAGE_SIZE, og.image_size);
        MDopt.setValue(EMDL_CTF_VOLTAGE, og.voltage);
        MDopt.setValue(EMDL_CTF_CS, og.spherical_aberration);
        MDopt.setValue(EMDL_CTF_Q0, og.amplitude_contrast);
    }

    // Build data table
    MetaDataTable MD;
    std::string tablename = have_particles ? "particles" : (have_micrographs ? "micrographs" : "images");
    MD.setName(tablename);

    // Helper to get micrograph shape for a row
    auto get_mic_shape = [&](const char* src_row, int fi, int pt_fi, const char* pt_row) -> std::vector<double> {
        const auto* f = (fi >= 0) ? &hdr.fields[fi] : ((pt_row && pt_fi >= 0) ? &hdr_pt.fields[pt_fi] : nullptr);
        if (!f) return {};
        const char* ptr = (fi >= 0) ? src_row + f->offset : pt_row + f->offset;
        int count = (f->subshape.empty() ? 1 : f->subshape[0]);
        std::vector<double> shape;
        for (int j = 0; j < count; j++)
            shape.push_back(read_double(ptr, j * f->itemsize, f->kind, f->itemsize));
        return shape;
    };

    for (size_t i = 0; i < hdr.num_rows; i++) {
        const char* row = data.data() + i * hdr.total_itemsize;

        // Look up passthrough row by uid
        const char* pt_row = nullptr;
        if (have_passthrough && fi_uid >= 0 && pt_fi_uid >= 0) {
            long uid = read_int(row + hdr.fields[fi_uid].offset, 0,
                                hdr.fields[fi_uid].kind,
                                hdr.fields[fi_uid].itemsize);
            auto it = pt_uid_to_row.find(uid);
            if (it != pt_uid_to_row.end())
                pt_row = data_pt.data() + it->second * hdr_pt.total_itemsize;
        }

        // Helper: read double from main or fallback to passthrough
        auto rd = [&](int fi, int pt_fi) -> double {
            if (fi >= 0) {
                const auto& f = hdr.fields[fi];
                return read_double(row + f.offset, 0, f.kind, f.itemsize);
            }
            if (pt_row && pt_fi >= 0) {
                const auto& f = hdr_pt.fields[pt_fi];
                return read_double(pt_row + f.offset, 0, f.kind, f.itemsize);
            }
            return 0.0;
        };

        MD.addObject();

        // --- Image name ---
        if (has_blob_info) {
            const CsField& f_path = hdr.fields[fi_blob_path];
            const CsField& f_idx  = hdr.fields[fi_blob_idx];
            std::string path = resolve_path(read_string(row + f_path.offset, 0, f_path.itemsize), project_dir);
            verify_path(path, "Particle stack");
            long idx = read_int(row + f_idx.offset, 0, f_idx.kind, f_idx.itemsize);
            std::string idx_str = std::to_string(idx + 1);
            if (idx_str.size() < 6)
                idx_str = std::string(6 - idx_str.size(), '0') + idx_str;
            set_value(MD, "rlnImageName", idx_str + "@" + path);
        }

        // --- Micrograph name ---
        auto read_mic_name = [&](int fi, int pt_fi) -> std::string {
            std::string p;
            if (fi >= 0) {
                const auto& f = hdr.fields[fi];
                p = read_string(row + f.offset, 0, f.itemsize);
            } else if (pt_row && pt_fi >= 0) {
                const auto& f = hdr_pt.fields[pt_fi];
                p = read_string(pt_row + f.offset, 0, f.itemsize);
            }
            return p.empty() ? p : resolve_path(p, project_dir);
        };
        std::string mic_name = read_mic_name(fi_mic_path, pt_fi_mic_path);
        if (mic_name.empty())
            mic_name = read_mic_name(fi_loc_path, pt_fi_loc_path);
        if (!mic_name.empty()) {
            verify_path(mic_name, "Micrograph");
            set_value(MD, "rlnMicrographName", mic_name);
        }

        // --- Movie name ---
        {
            std::string val;
            if (fi_movie_path >= 0) {
                const auto& f = hdr.fields[fi_movie_path];
                val = read_string(row + f.offset, 0, f.itemsize);
            } else if (pt_row && pt_fi_movie_path >= 0) {
                const auto& f = hdr_pt.fields[pt_fi_movie_path];
                val = read_string(pt_row + f.offset, 0, f.itemsize);
            }
            if (!val.empty()) {
                std::string movie_path = resolve_path(val, project_dir);
                verify_path(movie_path, "Movie");
                set_value(MD, "rlnMicrographMovieName", movie_path);
            }
        }

        // --- CTF parameters ---
        auto set_ctf = [&](int fi, int pt_fi, const char* rln, double mul = 1.0) {
            double val;
            if (fi >= 0)
                val = read_double(row + hdr.fields[fi].offset, 0, hdr.fields[fi].kind, hdr.fields[fi].itemsize);
            else if (pt_row && pt_fi >= 0)
                val = read_double(pt_row + hdr_pt.fields[pt_fi].offset, 0, hdr_pt.fields[pt_fi].kind, hdr_pt.fields[pt_fi].itemsize);
            else
                return;
            set_value(MD, rln, val * mul);
        };
        set_ctf(fi_ctf_df1, pt_fi_ctf_df1, "rlnDefocusU");
        set_ctf(fi_ctf_df2, pt_fi_ctf_df2, "rlnDefocusV");
        set_ctf(fi_ctf_dfang, pt_fi_ctf_dfang, "rlnDefocusAngle", DEG_PER_RAD);
        set_ctf(fi_ctf_pshift, pt_fi_ctf_pshift, "rlnPhaseShift");
        set_ctf(fi_ctf_bfac, pt_fi_ctf_bfac, "rlnCtfBfactor");
        set_ctf(fi_ctf_scale, pt_fi_ctf_scale, "rlnCtfScalefactor");
        set_ctf(fi_ctf_fit, pt_fi_ctf_fit, "rlnCtfMaxResolution");

        // --- Pixel sizes ---
        auto set_psize = [&](int fi, int pt_fi, const char* rln) {
            if (fi >= 0)
                set_value(MD, rln, read_double(row + hdr.fields[fi].offset, 0, hdr.fields[fi].kind, hdr.fields[fi].itemsize));
            else if (pt_row && pt_fi >= 0)
                set_value(MD, rln, read_double(pt_row + hdr_pt.fields[pt_fi].offset, 0, hdr_pt.fields[pt_fi].kind, hdr_pt.fields[pt_fi].itemsize));
        };
        set_psize(fi_blob_psize, pt_fi_blob_psize, "rlnImagePixelSize");
        set_psize(fi_mic_psize, pt_fi_mic_psize, "rlnMicrographPixelSize");

        // --- 2D alignments ---
        if (fi_align2d_cls >= 0) {
            long val = read_int(row + hdr.fields[fi_align2d_cls].offset, 0,
                                hdr.fields[fi_align2d_cls].kind,
                                hdr.fields[fi_align2d_cls].itemsize);
            set_value(MD, "rlnClassNumber", val + 1);
        }

        if (fi_align2d_sh >= 0) {
            const auto& f = hdr.fields[fi_align2d_sh];
            double sx = read_double(row + f.offset, 0, f.kind, f.itemsize);
            double sy = (f.subshape.size() > 0 && f.subshape[0] > 1)
                ? read_double(row + f.offset, 1 * f.itemsize, f.kind, f.itemsize) : 0.0;
            // Use pixel size to convert to Angstrom
            double apix = pixel_size;
            if (fi_blob_psize >= 0)
                apix = read_double(row + hdr.fields[fi_blob_psize].offset, 0,
                                   hdr.fields[fi_blob_psize].kind,
                                   hdr.fields[fi_blob_psize].itemsize);
            set_value(MD, "rlnOriginXAngst", (-sx) * apix);
            set_value(MD, "rlnOriginYAngst", (-sy) * apix);
        }

        if (fi_align2d_ps >= 0) {
            double psi = read_double(row + hdr.fields[fi_align2d_ps].offset, 0,
                                     hdr.fields[fi_align2d_ps].kind,
                                     hdr.fields[fi_align2d_ps].itemsize);
            set_value(MD, "rlnAnglePsi", -psi * DEG_PER_RAD);
        }

        // --- 3D alignments ---
        if (fi_align3d_cls >= 0) {
            long val = read_int(row + hdr.fields[fi_align3d_cls].offset, 0,
                                hdr.fields[fi_align3d_cls].kind,
                                hdr.fields[fi_align3d_cls].itemsize);
            set_value(MD, "rlnClassNumber", val + 1);
        }

        if (fi_align3d_cc >= 0) {
            double val = read_double(row + hdr.fields[fi_align3d_cc].offset, 0,
                                     hdr.fields[fi_align3d_cc].kind,
                                     hdr.fields[fi_align3d_cc].itemsize);
            set_value(MD, "rlnLogLikeliContribution", val);
        }

        if (fi_align3d_sp >= 0) {
            long val = read_int(row + hdr.fields[fi_align3d_sp].offset, 0,
                                hdr.fields[fi_align3d_sp].kind,
                                hdr.fields[fi_align3d_sp].itemsize);
            set_value(MD, "rlnRandomSubset", val + 1);
        }

        if (fi_align3d_ps >= 0) {
            // Rotation vector → ZYZ Euler angles
            const auto& f = hdr.fields[fi_align3d_ps];
            double rx = read_double(row + f.offset, 0, f.kind, f.itemsize);
            double ry = (f.subshape.size() > 0 && f.subshape[0] > 1)
                ? read_double(row + f.offset, 1 * f.itemsize, f.kind, f.itemsize) : 0.0;
            double rz = (f.subshape.size() > 0 && f.subshape[0] > 2)
                ? read_double(row + f.offset, 2 * f.itemsize, f.kind, f.itemsize) : 0.0;

            Eigen::Vector3d rotvec(rx, ry, rz);
            double angle = rotvec.norm();
            if (angle > 1e-15) {
                Eigen::Vector3d axis = rotvec / angle;
                Eigen::AngleAxisd aa(angle, axis);
                Eigen::Matrix3d rotmat = aa.toRotationMatrix();
                Eigen::Vector3d euler = rotmat.eulerAngles(2, 1, 2); // ZYZ
                double rot = euler[0] * DEG_PER_RAD;
                double tilt = euler[1] * DEG_PER_RAD;
                double psi = euler[2] * DEG_PER_RAD;
                // Force positive tilt to match RELION convention
                if (tilt < 0) {
                    rot += 180.0;
                    tilt = -tilt;
                    psi -= 180.0;
                }
                // Normalize to [0, 360)
                auto norm = [](double &x) { x = fmod(x, 360.0); if (x < 0) x += 360.0; };
                norm(rot); norm(psi);
                set_value(MD, "rlnAngleRot", rot);
                set_value(MD, "rlnAngleTilt", tilt);
                set_value(MD, "rlnAnglePsi", psi);
            }
        }

        if (fi_align3d_sh >= 0) {
            const auto& f = hdr.fields[fi_align3d_sh];
            double sx = read_double(row + f.offset, 0, f.kind, f.itemsize);
            double sy = (f.subshape.size() > 0 && f.subshape[0] > 1)
                ? read_double(row + f.offset, 1 * f.itemsize, f.kind, f.itemsize) : 0.0;
            double apix = pixel_size;
            if (fi_blob_psize >= 0)
                apix = read_double(row + hdr.fields[fi_blob_psize].offset, 0,
                                   hdr.fields[fi_blob_psize].kind,
                                   hdr.fields[fi_blob_psize].itemsize);
            set_value(MD, "rlnOriginXAngst", sx * apix);
            set_value(MD, "rlnOriginYAngst", sy * apix);
        }

        // --- Coordinates ---
        auto set_coord = [&](int fi_cx, int pt_fi_cx, int fi_cy, int pt_fi_cy) {
            if (fi_cx >= 0 || (pt_row && pt_fi_cx >= 0)) {
                double cx = (fi_cx >= 0) ? read_double(row + hdr.fields[fi_cx].offset, 0, hdr.fields[fi_cx].kind, hdr.fields[fi_cx].itemsize)
                                         : read_double(pt_row + hdr_pt.fields[pt_fi_cx].offset, 0, hdr_pt.fields[pt_fi_cx].kind, hdr_pt.fields[pt_fi_cx].itemsize);
                double cy = (fi_cy >= 0) ? read_double(row + hdr.fields[fi_cy].offset, 0, hdr.fields[fi_cy].kind, hdr.fields[fi_cy].itemsize)
                                         : read_double(pt_row + hdr_pt.fields[pt_fi_cy].offset, 0, hdr_pt.fields[pt_fi_cy].kind, hdr_pt.fields[pt_fi_cy].itemsize);
                std::vector<double> shape = get_mic_shape(row, fi_loc_shape, pt_fi_loc_shape, pt_row);
                if (shape.empty())
                    shape = get_mic_shape(row, fi_mic_shape, pt_fi_mic_shape, pt_row);
                if (shape.size() >= 2) {
                    set_value(MD, "rlnCoordinateX", cx * shape[0]);
                    set_value(MD, "rlnCoordinateY", cy * shape[1]);
                }
            }
        };
        set_coord(fi_loc_cx, pt_fi_loc_cx, fi_loc_cy, pt_fi_loc_cy);

        // --- Optics group ---
        set_value(MD, "rlnOpticsGroup", (long)row_optgroup[i]);
    }

    // Write STAR file
    std::ofstream fh(star_filename.c_str());
    if (!fh)
        throw std::runtime_error("Cannot write file: " + star_filename);

    MDopt.write(fh);
    MD.write(fh);
    fh.close();

    std::cout << " Written " << star_filename << " with " << hdr.num_rows << " particles from CryoSPARC .cs file" << std::endl;
}

// Check whether the .cs file's particles originated from a RELION STAR file
// that was imported into CryoSPARC. Returns true if the original STAR file
// and its matching imported_particles.cs are found.
static bool detect_original_star(
        const std::string& cs_filename,
        std::string& import_star_path_out,
        std::string& import_cs_path_out)
{
    try {
        CsHeader hdr = read_header(cs_filename);
        std::vector<char> data = read_data(cs_filename, hdr);

        int fi_path = find_field(hdr.fields, "blob/path");
        if (fi_path < 0 || hdr.num_rows == 0)
            return false;

        const char* row0 = data.data();
        const CsField& fp = hdr.fields[fi_path];
        std::string first_path = read_string(row0 + fp.offset, 0, fp.itemsize);

        size_t first_slash = first_path.find('/');
        if (first_slash == std::string::npos)
            return false;
        std::string import_job = first_path.substr(0, first_slash);

        // Compute project_dir from cs_filename
        std::string project_dir;
        {
            size_t slash = cs_filename.rfind('/');
            std::string cs_dir = (slash != std::string::npos)
                ? cs_filename.substr(0, slash) : ".";
            if (!cs_dir.empty() && cs_dir[0] != '/') {
                char cwd[4096];
                if (getcwd(cwd, sizeof(cwd))) {
                    cs_dir = normalize_path(std::string(cwd) + "/" + cs_dir);
                }
            }
            project_dir = resolve_path("..", cs_dir);
        }

        import_star_path_out = project_dir + "/" + import_job + "/particles.star";
        import_cs_path_out   = project_dir + "/" + import_job + "/imported_particles.cs";

        return exists(import_star_path_out) && exists(import_cs_path_out);
    } catch (...) {
        return false;
    }
}

// Convert a CryoSPARC .cs file by tracing back to the original RELION STAR file
// that was imported into CryoSPARC. Uses the original STAR file as the data
// source (preserving all original RELION fields) and the .cs file as a subset
// selector via uid matching. Also overlays fields from the .cs file (class,
// alignments, CTF) on the selected particles.
//
// The import job is determined from the blob/path in the .cs file:
//   e.g. "J1/imported/..." → import job is J1
// The import job directory must contain:
//   - imported_particles.cs  (uid->row mapping, same order as particles.star)
//   - particles.star         (the original imported RELION STAR file)
static void convert_with_original_star(
        const std::string& cs_filename,
        const std::string& star_filename,
        const std::string& optics_group_name,
        RFLOAT pixel_size, RFLOAT kV, RFLOAT Cs, RFLOAT Q0)
{
    // 1. Read target .cs file (subset selector)
    CsHeader hdr = read_header(cs_filename);
    std::vector<char> data = read_data(cs_filename, hdr);

    // 2. Compute project_dir (same as convert())
    std::string project_dir;
    {
        size_t slash = cs_filename.rfind('/');
        std::string cs_dir = (slash != std::string::npos) ? cs_filename.substr(0, slash) : ".";
        if (!cs_dir.empty() && cs_dir[0] != '/') {
            char cwd[4096];
            if (getcwd(cwd, sizeof(cwd))) {
                cs_dir = normalize_path(std::string(cwd) + "/" + cs_dir);
            }
        }
        project_dir = resolve_path("..", cs_dir);
    }

    // 3. Find key fields in target .cs
    int fi_path = find_field(hdr.fields, "blob/path");
    int fi_uid  = find_field(hdr.fields, "uid");

    if (fi_path < 0 || fi_uid < 0)
        REPORT_ERROR("convert_with_original_star: .cs file missing blob/path or uid");

    // 4. Determine import job name from first blob/path (e.g. "J1" from "J1/imported/...")
    const CsField& fp = hdr.fields[fi_path];
    const char* row0 = data.data();
    std::string first_path = read_string(row0 + fp.offset, 0, fp.itemsize);
    size_t first_slash = first_path.find('/');
    if (first_slash == std::string::npos)
        REPORT_ERROR("convert_with_original_star: cannot parse import job from blob/path: " + first_path);
    std::string import_job = first_path.substr(0, first_slash);

    std::string import_cs_path   = project_dir + "/" + import_job + "/imported_particles.cs";
    std::string import_star_path = project_dir + "/" + import_job + "/particles.star";

    // 5. Read import job's imported_particles.cs to build uid→position mapping
    CsHeader hdr_imp;
    std::vector<char> data_imp;
    try {
        hdr_imp = read_header(import_cs_path);
        data_imp = read_data(import_cs_path, hdr_imp);
    } catch (...) {
        REPORT_ERROR("convert_with_original_star: cannot read " + import_cs_path);
    }
    int imp_fi_uid = find_field(hdr_imp.fields, "uid");
    if (imp_fi_uid < 0)
        REPORT_ERROR("convert_with_original_star: import .cs has no uid field");

    std::vector<long> import_uids(hdr_imp.num_rows);
    for (size_t i = 0; i < hdr_imp.num_rows; i++) {
        const char* row = data_imp.data() + i * hdr_imp.total_itemsize;
        import_uids[i] = read_int(row + hdr_imp.fields[imp_fi_uid].offset, 0,
                                  hdr_imp.fields[imp_fi_uid].kind,
                                  hdr_imp.fields[imp_fi_uid].itemsize);
    }

    // 6. Build selected_uids set and uid→cs_row map from target .cs
    std::unordered_set<long> selected_uids;
    std::unordered_map<long, size_t> uid_to_csrow;
    {
        const CsField& fu = hdr.fields[fi_uid];
        for (size_t i = 0; i < hdr.num_rows; i++) {
            const char* row = data.data() + i * hdr.total_itemsize;
            long uid = read_int(row + fu.offset, 0, fu.kind, fu.itemsize);
            selected_uids.insert(uid);
            uid_to_csrow[uid] = i;
        }
    }

    // 7. Read original STAR file
    if (!exists(import_star_path))
        REPORT_ERROR("convert_with_original_star: cannot read " + import_star_path);
    std::cout << " Using original STAR file: " << import_star_path << std::endl;
    std::cout << " Total particles in original: " << hdr_imp.num_rows
              << ", selected: " << selected_uids.size() << std::endl;

    MetaDataTable MDopt, MDparts;
    MDopt.read(import_star_path, "optics");
    MDparts.read(import_star_path, "particles");

    if (MDparts.numberOfObjects() != (long)hdr_imp.num_rows)
        REPORT_ERROR("convert_with_original_star: particle count mismatch between " +
                     import_star_path + " and " + import_cs_path);

    // 8. Find overlay fields in target .cs
    struct CsFieldRef {
        int fi;
        EMDLabel label;
        double mul;
    };
    std::vector<CsFieldRef> overlay_scalar;
    std::vector<int> overlay_2d;
    int fi_cs_ctf_df1   = find_field(hdr.fields, "ctf/df1_A");
    int fi_cs_ctf_df2   = find_field(hdr.fields, "ctf/df2_A");
    int fi_cs_ctf_dfang = find_field(hdr.fields, "ctf/df_angle_rad");
    int fi_cs_ctf_pshift= find_field(hdr.fields, "ctf/phase_shift_rad");
    int fi_cs_ctf_bfac  = find_field(hdr.fields, "ctf/bfactor");
    int fi_cs_ctf_scale = find_field(hdr.fields, "ctf/scale");
    int fi_cs_align2d_cls = find_field(hdr.fields, "alignments2D/class");
    int fi_cs_align2d_sh  = find_field(hdr.fields, "alignments2D/shift");
    int fi_cs_align2d_ps  = find_field(hdr.fields, "alignments2D/pose");
    int fi_cs_blob_psize  = find_field(hdr.fields, "blob/psize_A");

    const double DEG_PER_RAD = 180.0 / M_PI;

    // 9. Overlay .cs fields on selected particles, then remove non-selected
    //    Iterate once: overlay selected particles, track which to keep.
    std::vector<bool> keep(MDparts.numberOfObjects(), false);
    for (long i = 0; i < MDparts.numberOfObjects(); i++) {
        long uid = import_uids[i];
        auto it = selected_uids.find(uid);
        if (it == selected_uids.end())
            continue;

        keep[i] = true;
        size_t cs_row = uid_to_csrow[uid];
        const char* csr = data.data() + cs_row * hdr.total_itemsize;

        // Overlay 2D class
        if (fi_cs_align2d_cls >= 0) {
            const auto& f = hdr.fields[fi_cs_align2d_cls];
            long val = read_int(csr + f.offset, 0, f.kind, f.itemsize);
            MDparts.setValue(EMDL_PARTICLE_CLASS, val + 1, i);
        }

        // Overlay 2D shift → origin
        if (fi_cs_align2d_sh >= 0) {
            const auto& f = hdr.fields[fi_cs_align2d_sh];
            double sx = read_double(csr + f.offset, 0, f.kind, f.itemsize);
            double sy = (f.subshape.size() > 0 && f.subshape[0] > 1)
                ? read_double(csr + f.offset, 1 * f.itemsize, f.kind, f.itemsize) : 0.0;
            double apix = pixel_size;
            if (fi_cs_blob_psize >= 0)
                apix = read_double(csr + hdr.fields[fi_cs_blob_psize].offset, 0,
                                  hdr.fields[fi_cs_blob_psize].kind,
                                  hdr.fields[fi_cs_blob_psize].itemsize);
            MDparts.setValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, (-sx) * apix, i);
            MDparts.setValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, (-sy) * apix, i);
        }

        // Overlay 2D psi
        if (fi_cs_align2d_ps >= 0) {
            const auto& f = hdr.fields[fi_cs_align2d_ps];
            double psi = read_double(csr + f.offset, 0, f.kind, f.itemsize);
            MDparts.setValue(EMDL_ORIENT_PSI, -psi * DEG_PER_RAD, i);
        }

        // Overlay CTF params (scalar fields with optional multiplier)
        auto set_ctf = [&](int fi, EMDLabel label, double mul) {
            if (fi < 0) return;
            double val = read_double(csr + hdr.fields[fi].offset, 0,
                                    hdr.fields[fi].kind, hdr.fields[fi].itemsize);
            MDparts.setValue(label, val * mul, i);
        };
        set_ctf(fi_cs_ctf_df1,   EMDL_CTF_DEFOCUSU,     1.0);
        set_ctf(fi_cs_ctf_df2,   EMDL_CTF_DEFOCUSV,     1.0);
        set_ctf(fi_cs_ctf_dfang, EMDL_CTF_DEFOCUS_ANGLE, DEG_PER_RAD);
        set_ctf(fi_cs_ctf_pshift,EMDL_CTF_PHASESHIFT,  1.0);
        set_ctf(fi_cs_ctf_bfac,  EMDL_CTF_BFACTOR,      1.0);
        set_ctf(fi_cs_ctf_scale, EMDL_CTF_SCALEFACTOR,  1.0);
    }

    // 10. Remove non-selected particles (iterate backwards to preserve indices)
    for (long i = MDparts.numberOfObjects() - 1; i >= 0; i--) {
        if (!keep[i])
            MDparts.removeObject(i);
    }

    // 11. Write output STAR file
    std::ofstream fh(star_filename.c_str());
    if (!fh)
        REPORT_ERROR("convert_with_original_star: cannot write " + star_filename);

    MDopt.write(fh);
    MDparts.write(fh);
    fh.close();

    std::cout << " Written " << star_filename << " with " << MDparts.numberOfObjects()
              << " particles (subset from original STAR file via uid matching)"
              << std::endl;
}

} // namespace cryosparc

#endif // CRYOSPARC_IMPORT_H
