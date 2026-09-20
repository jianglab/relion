#include "src/cryosparc_motioncorr.h"
#include "src/cryosparc_cs.h"
#include "src/cryosparc_spline.h"
#include "src/cryosparc_motion_model.h"
#include "src/image.h"
#include "src/filename.h"

#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace cryosparc {

bool isMotionCorrectionJob(const std::string& t)
{
	return t == "patch_motion_correction_multi"
	    || t == "motion_correction_multi"
	    || t == "motion_correction"
	    || t == "rigid_motion_correction";
}

namespace {

void makeDirs(const std::string& path)
{
	std::string cur;
	for (size_t i = 0; i < path.size(); i++)
	{
		cur += path[i];
		if (path[i] == '/' || i + 1 == path.size())
			mkdir(cur.c_str(), 0755);
	}
}

std::string baseName(const std::string& p)
{
	const size_t s = p.find_last_of('/');
	return (s == std::string::npos) ? p : p.substr(s + 1);
}

/// Strip a movie extension, matching cs2relion's r"\.(tiff?|eer|mrc)$"
std::string stripMovieExt(const std::string& name)
{
	static const char* exts[] = {".tiff", ".tif", ".eer", ".mrc", ".mrcs", NULL};
	for (int i = 0; exts[i]; i++)
	{
		const std::string e = exts[i];
		if (name.size() > e.size())
		{
			std::string tail = name.substr(name.size() - e.size());
			std::string lower;
			for (size_t k = 0; k < tail.size(); k++) lower += tolower(tail[k]);
			if (lower == e) return name.substr(0, name.size() - e.size());
		}
	}
	return name;
}

/// Path of `abs` relative to `root`, for writing into a STAR file.
std::string relativeTo(const std::string& abs, const std::string& root)
{
	std::string r = root;
	if (!r.empty() && r[r.size() - 1] != '/') r += "/";
	if (abs.compare(0, r.size(), r) == 0) return abs.substr(r.size());
	return abs;
}

void symlinkOnce(const std::string& target, const std::string& link)
{
	struct stat st;
	if (lstat(link.c_str(), &st) == 0) return;   // already there
	if (symlink(target.c_str(), link.c_str()) != 0 && errno != EEXIST)
		std::cerr << " WARNING: could not symlink " << link << " -> " << target << std::endl;
}

/* Copy CryoSPARC's aligned average into the RELION job, Y-flipped and as
 * float32. See the header for why this cannot be a symlink. */
void writeAlignedAverage(const std::string& src_abs, const std::string& dest)
{
	struct stat st;
	if (stat(dest.c_str(), &st) == 0) return;   // already converted

	// Note: RELION's own half-float reader flushes float16 subnormals to zero,
	// so the result is not bit-identical to CryoSPARC's float16 source. Measured
	// on a real micrograph: 65 pixels out of 14.2 million, all with |v| below
	// 6.104e-05 (the smallest normal float16), in data spanning +/-75. That is
	// ~1e-6 of the data range on 0.0005% of pixels, and is RELION reader
	// behaviour rather than anything this importer does.
	Image<float> img;
	img.read(src_abs);

	const long nx = XSIZE(img());
	const long ny = YSIZE(img());

	Image<float> out(nx, ny);
	for (long y = 0; y < ny; y++)
	for (long x = 0; x < nx; x++)
		DIRECT_A2D_ELEM(out(), y, x) = DIRECT_A2D_ELEM(img(), ny - 1 - y, x);

	// Image::write produces MRC mode 2 (float32), which is what CTFFIND4 needs
	out.setSamplingRateInHeader(img.samplingRateX(), img.samplingRateY());
	out.write(dest);
}

/// Load a (1, n_frames, 2) or (n_frames, 2) rigid trajectory.
bool loadRigidTrajectory(const std::string& path, int n_frames,
                         std::vector<double>& x, std::vector<double>& y)
{
	std::vector<unsigned long> shape;
	std::vector<double> data;
	if (!loadNpyDouble(path, shape, data)) return false;

	// (1, n, 2) as written by patch motion, or (n, 2)
	size_t n = 0, stride = 0, base = 0;
	if (shape.size() == 3 && shape[2] == 2) { n = shape[1]; stride = 2; base = 0; }
	else if (shape.size() == 2 && shape[1] == 2) { n = shape[0]; stride = 2; base = 0; }
	else
	{
		std::cerr << " WARNING: unexpected rigid trajectory shape in " << path << std::endl;
		return false;
	}
	if ((int)n != n_frames)
		n = std::min((size_t)n_frames, n);

	x.assign(n_frames, 0.);
	y.assign(n_frames, 0.);
	for (size_t i = 0; i < n; i++)
	{
		x[i] = data[base + i * stride + 0];
		y[i] = data[base + i * stride + 1];
	}
	return true;
}

void writeMovieStar(const std::string& path,
                    long width, long height, long n_frames,
                    const std::string& movie_name, const std::string& gain_name,
                    double pixel_size_A, double dose_per_frame_A2, double pre_exposure_A2,
                    double voltage, int first_frame,
                    const std::vector<double>& gx, const std::vector<double>& gy,
                    const std::vector<double>& coeff_x, const std::vector<double>& coeff_y,
                    const std::vector<double>& lf, const std::vector<double>& lx,
                    const std::vector<double>& ly, const std::vector<double>& lsx,
                    const std::vector<double>& lsy)
{
	std::ofstream f(path.c_str());
	if (!f) { std::cerr << " WARNING: cannot write " << path << std::endl; return; }

	f << "\n# version 30001\n\ndata_general\n\n";
	f << "_rlnImageSizeX " << width << "\n";
	f << "_rlnImageSizeY " << height << "\n";
	f << "_rlnImageSizeZ " << n_frames << "\n";
	f << "_rlnMicrographMovieName " << movie_name << "\n";
	if (!gain_name.empty()) f << "_rlnMicrographGainName " << gain_name << "\n";
	f << "_rlnMicrographBinning 1.000000\n";
	char buf[256];
	snprintf(buf, sizeof(buf), "_rlnMicrographOriginalPixelSize %.6f\n", pixel_size_A); f << buf;
	snprintf(buf, sizeof(buf), "_rlnMicrographDoseRate %.6f\n", dose_per_frame_A2);     f << buf;
	snprintf(buf, sizeof(buf), "_rlnMicrographPreExposure %.6f\n", pre_exposure_A2);    f << buf;
	snprintf(buf, sizeof(buf), "_rlnVoltage %.6f\n", voltage);                          f << buf;
	f << "_rlnMicrographStartFrame " << first_frame << "\n";
	f << "_rlnMotionModelVersion 1\n \n";

	f << "\n# version 30001\n\ndata_global_shift\n\nloop_ \n";
	f << "_rlnMicrographFrameNumber #1 \n";
	f << "_rlnMicrographShiftX #2 \n";
	f << "_rlnMicrographShiftY #3 \n";
	for (size_t i = 0; i < gx.size(); i++)
	{
		snprintf(buf, sizeof(buf), "%6d %12.6f %12.6f\n", (int)(i + first_frame), gx[i], gy[i]);
		f << buf;
	}
	f << " \n";

	f << "\n# version 30001\n\ndata_local_motion_model\n\nloop_ \n";
	f << "_rlnMotionModelCoeffsIdx #1 \n";
	f << "_rlnMotionModelCoeff #2 \n";
	for (size_t i = 0; i < coeff_x.size(); i++)
	{
		snprintf(buf, sizeof(buf), "%6d %14.6g\n", (int)i, coeff_x[i]);
		f << buf;
	}
	for (size_t i = 0; i < coeff_y.size(); i++)
	{
		snprintf(buf, sizeof(buf), "%6d %14.6g\n", (int)(i + NUM_MOTION_COEFFS_PER_DIM), coeff_y[i]);
		f << buf;
	}
	f << " \n";

	// RELION does not reload this table for correction (only the fitted model
	// above is applied), but its own MotionCorr writes it and it is what makes
	// the fit inspectable.
	f << "\n# version 30001\n\ndata_local_shift\n\nloop_ \n";
	f << "_rlnMicrographFrameNumber #1 \n";
	f << "_rlnCoordinateX #2 \n";
	f << "_rlnCoordinateY #3 \n";
	f << "_rlnMicrographShiftX #4 \n";
	f << "_rlnMicrographShiftY #5 \n";
	for (size_t i = 0; i < lf.size(); i++)
	{
		snprintf(buf, sizeof(buf), "%6d %12.6f %12.6f %12.6f %12.6f\n",
		         (int)lf[i], lx[i], ly[i], lsx[i], lsy[i]);
		f << buf;
	}
	f << " \n";
}

struct MicRow {
	std::string name, metadata;
	double total, early, late;
};

void writeCorrectedMicrographsStar(const std::string& path,
                                   double pixel_size_A, double voltage, double cs_mm,
                                   double amplitude_contrast,
                                   const std::vector<MicRow>& rows)
{
	std::ofstream f(path.c_str());
	if (!f) { std::cerr << " WARNING: cannot write " << path << std::endl; return; }

	char buf[512];
	f << "\n# version 30001\n\ndata_optics\n\nloop_ \n";
	f << "_rlnOpticsGroupName #1 \n_rlnOpticsGroup #2 \n_rlnMicrographOriginalPixelSize #3 \n"
	     "_rlnVoltage #4 \n_rlnSphericalAberration #5 \n_rlnAmplitudeContrast #6 \n"
	     "_rlnMicrographPixelSize #7 \n";
	snprintf(buf, sizeof(buf), "opticsGroup1 1 %.6f %.6f %.6f %.6f %.6f \n \n",
	         pixel_size_A, voltage, cs_mm, amplitude_contrast, pixel_size_A);
	f << buf;

	f << "\n# version 30001\n\ndata_micrographs\n\nloop_ \n";
	f << "_rlnMicrographName #1 \n_rlnMicrographMetadata #2 \n_rlnOpticsGroup #3 \n"
	     "_rlnAccumMotionTotal #4 \n_rlnAccumMotionEarly #5 \n_rlnAccumMotionLate #6 \n";
	for (size_t i = 0; i < rows.size(); i++)
	{
		snprintf(buf, sizeof(buf), "%s %s 1 %.6f %.6f %.6f \n",
		         rows[i].name.c_str(), rows[i].metadata.c_str(),
		         rows[i].total, rows[i].early, rows[i].late);
		f << buf;
	}
	f << " \n";
}

} // anonymous namespace

long writeMotionCorrJob(const Project& project,
                        const Job& cs_job,
                        const std::string& out_project,
                        const std::string& out_job_dir,
                        const Project::Acquisition& acq,
                        const MotionCorrOptions& opts)
{
	// The micrographs group carries both trajectories and the blob paths.
	const OutputGroup* grp = cs_job.group("micrographs");
	if (grp == NULL) grp = cs_job.groupOfType("exposure");
	if (grp == NULL)
		throw std::runtime_error(cs_job.uid + ": no micrographs output group to import");

	const std::string primary = grp->primaryMetafile();
	if (primary.empty())
		throw std::runtime_error(cs_job.uid + ": micrographs group has no metadata file");

	CsTable cs(project.resolve(primary));

	if (!cs.has("rigid_motion/path") || !cs.has("spline_motion/path"))
		throw std::runtime_error(cs_job.uid + ": " + baseName(primary) +
		                         " has no rigid_motion/spline_motion fields; "
		                         "is this a patch motion correction job?");

	if (acq.voltage <= 0. || acq.pixel_size_A <= 0.)
		throw std::runtime_error(cs_job.uid + ": could not determine voltage and pixel size "
		                         "from an import job in this project");

	const std::string movies_dir = out_job_dir + "/Movies";
	if (opts.write_images) makeDirs(movies_dir);

	// Gain reference: untouched, so a symlink is fine
	std::string gain_rel;
	if (!acq.gainref_path.empty())
	{
		const std::string gain_abs = project.resolve(acq.gainref_path);
		const std::string link = movies_dir + "/" + baseName(gain_abs);
		if (opts.write_images) symlinkOnce(gain_abs, link);
		gain_rel = relativeTo(link, out_project);
	}

	std::vector<MicRow> mic_rows;

	for (size_t i = 0; i < cs.rows(); i++)
	{
		const std::string movie_path = cs.getString(i, "movie_blob/path");
		const std::string mic_path   = cs.getString(i, "micrograph_blob/path");
		if (movie_path.empty() || mic_path.empty()) continue;

		const long n_frames = cs.getInt(i, "movie_blob/shape", 0);
		const long height   = cs.getInt(i, "movie_blob/shape", 1);
		const long width    = cs.getInt(i, "movie_blob/shape", 2);
		if (n_frames <= 0 || width <= 0 || height <= 0) continue;

		double dose = opts.dose_per_frame_A2;
		if (dose <= 0. && acq.total_dose_e_per_A2 > 0.) dose = acq.total_dose_e_per_A2 / n_frames;
		if (dose <= 0.)
			throw std::runtime_error(cs_job.uid + ": no dose rate; the import job has no "
			                         "total dose, so pass one explicitly");

		const std::string movie_name = baseName(movie_path);
		const std::string stem = stripMovieExt(movie_name);

		// Short in-project paths: RELION mirrors out-of-project absolute paths
		// under every downstream job directory, and a deep CryoSPARC path can
		// then overflow external tools' filename buffers.
		const std::string movie_link = movies_dir + "/" + movie_name;
		if (opts.write_images) symlinkOnce(project.resolve(movie_path), movie_link);

		const std::string avg_dest = movies_dir + "/" + baseName(mic_path);
		if (opts.write_images) writeAlignedAverage(project.resolve(mic_path), avg_dest);

		// --- global trajectory ---
		std::vector<double> rx, ry;
		if (!loadRigidTrajectory(project.resolve(cs.getString(i, "rigid_motion/path")),
		                         (int)n_frames, rx, ry))
			continue;

		// (negate x, keep y) into RELION's convention, then gauge-fixed to zero
		// at frame 1 the way RELION writes it.
		std::vector<double> gx(n_frames), gy(n_frames);
		for (long f = 0; f < n_frames; f++) { gx[f] = -rx[f]; gy[f] = ry[f]; }
		const double gx0 = gx[0], gy0 = gy[0];
		for (long f = 0; f < n_frames; f++) { gx[f] -= gx0; gy[f] -= gy0; }

		// --- local field, sampled on a RELION patch grid ---
		std::vector<unsigned long> sshape;
		std::vector<double> sdata;
		std::vector<double> coeff_x(NUM_MOTION_COEFFS_PER_DIM, 0.);
		std::vector<double> coeff_y(NUM_MOTION_COEFFS_PER_DIM, 0.);
		std::vector<double> lf, lx, ly, lsx, lsy;

		bool have_local =
			loadNpyDouble(project.resolve(cs.getString(i, "spline_motion/path")), sshape, sdata)
			&& sshape.size() == 4 && sshape[0] == 2;

		if (have_local)
		{
			const int KZ = (int)sshape[1], KY = (int)sshape[2], KX = (int)sshape[3];
			const size_t per = (size_t)KZ * KY * KX;

			std::vector<std::pair<double, double> > grid =
				motionPatchGrid((double)width, (double)height, opts.patches_x, opts.patches_y);

			// CryoSPARC's spline lives in its own pre-flip row frame, so query it
			// at the Y-mirrored position to describe the same physical patch.
			std::vector<double> qx, qy;
			for (size_t p = 0; p < grid.size(); p++)
			{
				qx.push_back(grid[p].first);
				qy.push_back((height - 1) - grid[p].second);
			}

			std::vector<double> rawx, rawy;
			splineInterpTraj((int)n_frames, (int)height, (int)width,
			                 &sdata[0], &sdata[per], KZ, KY, KX, qx, qy, rawx, rawy);

			for (size_t p = 0; p < grid.size(); p++)
			for (long f = 0; f < n_frames; f++)
			{
				const size_t k = p * n_frames + f;
				lf.push_back((double)(f + 1));
				lx.push_back(grid[p].first);
				ly.push_back(grid[p].second);
				lsx.push_back(-rawx[k]);   // same (negate x, keep y) calibration
				lsy.push_back(rawy[k]);
			}

			fitMotionModel((double)width, (double)height, 1, lf, lx, ly, lsx, lsy, coeff_x, coeff_y);
		}

		double total, early, late;
		accumulatedMotion(gx, gy, acq.pixel_size_A, dose, opts.pre_exposure_A2, 4.0,
		                  total, early, late);

		const std::string star_path = movies_dir + "/" + stem + ".star";
		if (opts.write_images)
			writeMovieStar(star_path, width, height, n_frames,
			               relativeTo(movie_link, out_project), gain_rel,
			               acq.pixel_size_A, dose, opts.pre_exposure_A2, acq.voltage, 1,
			               gx, gy, coeff_x, coeff_y, lf, lx, ly, lsx, lsy);

		MicRow r;
		r.name = relativeTo(avg_dest, out_project);
		r.metadata = relativeTo(star_path, out_project);
		r.total = total; r.early = early; r.late = late;
		mic_rows.push_back(r);

		if (opts.verb > 1)
			std::cout << "   " << stem << ": accum motion total=" << total
			          << " early=" << early << " late=" << late << " A" << std::endl;
	}

	if (opts.write_images)
		writeCorrectedMicrographsStar(out_job_dir + "/corrected_micrographs.star",
		                              acq.pixel_size_A, acq.voltage,
		                              acq.spherical_aberration, opts.amplitude_contrast,
		                              mic_rows);

	return (long)mic_rows.size();
}

} // namespace cryosparc
