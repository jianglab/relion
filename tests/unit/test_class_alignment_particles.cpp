#include <catch2/catch.hpp>

#include "src/ml_optimiser.h"

namespace
{

void addParticle(MetaDataTable &md, int cls, RFLOAT rot, RFLOAT tilt, RFLOAT psi, RFLOAT dx, RFLOAT dy)
{
	md.addObject();
	md.setValue(EMDL_PARTICLE_CLASS, cls);
	md.setValue(EMDL_ORIENT_ROT, rot);
	md.setValue(EMDL_ORIENT_TILT, tilt);
	md.setValue(EMDL_ORIENT_PSI, psi);
	md.setValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, dx);
	md.setValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, dy);
	md.setValue(EMDL_ORIENT_ORIGIN_X_PRIOR_ANGSTROM, dx);
	md.setValue(EMDL_ORIENT_ORIGIN_Y_PRIOR_ANGSTROM, dy);
}

} // namespace

TEST_CASE("MlOptimiser: class alignment degrees of freedom follow the symmetry", "[align_classes]")
{
	MlOptimiser opt;
	opt.do_helical_refine = false;
	opt.sampling.fn_sym = "C1";
	REQUIRE(opt.classAlignmentFreedom() == 6);
	opt.sampling.fn_sym = "C3";
	REQUIRE(opt.classAlignmentFreedom() == 2);
	opt.sampling.fn_sym = "D2";
	REQUIRE(opt.classAlignmentFreedom() == 0);
	opt.sampling.fn_sym = "C1";
	opt.do_helical_refine = true;
	opt.ignore_helical_symmetry = false;
	REQUIRE(opt.classAlignmentFreedom() == 2);
}

TEST_CASE("MlOptimiser: class alignment is applied to the particles of that class only", "[align_classes]")
{
	MlOptimiser opt;
	opt.mymodel.data_dim = 2;
	addParticle(opt.mydata.MDimg, 1, 0., 0., 0., 1., 2.);
	addParticle(opt.mydata.MDimg, 2, 0., 0., 0., 1., 2.);   // looking down the map's z axis
	addParticle(opt.mydata.MDimg, 2, 0., 90., 0., -3., 4.); // looking along the map's x axis

	// The map of class 2 was moved by t = (0.5, -1.5, 2) A, with no rotation. The image-plane offset
	// of a particle is the first two components of A * t, A being the particle's orientation.
	opt.applyClassAlignmentToParticles(1, 6, 0., 0., 0., 0.5, -1.5, 2.);

	RFLOAT dx, dy, px, py, rot;
	opt.mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, dx, 0);
	opt.mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, dy, 0);
	REQUIRE(dx == Approx(1.)); // class 1: untouched
	REQUIRE(dy == Approx(2.));

	opt.mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, dx, 1);
	opt.mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, dy, 1);
	opt.mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_X_PRIOR_ANGSTROM, px, 1);
	opt.mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_Y_PRIOR_ANGSTROM, py, 1);
	opt.mydata.MDimg.getValue(EMDL_ORIENT_ROT, rot, 1);
	REQUIRE(dx == Approx(1.5)); // identity orientation: A * t = (tx, ty)
	REQUIRE(dy == Approx(0.5));
	REQUIRE(px == Approx(1.5)); // the offset prior moves with the offset
	REQUIRE(py == Approx(0.5));
	REQUIRE(rot == Approx(0.)); // identity rotation leaves the angles alone

	// tilt 90: the image x axis is the map's -z axis, the image y axis is the map's y axis
	opt.mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, dx, 2);
	opt.mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, dy, 2);
	REQUIRE(dx == Approx(-5.));
	REQUIRE(dy == Approx(2.5));
}

TEST_CASE("MlOptimiser: Cn class alignment changes only rot", "[align_classes]")
{
	MlOptimiser opt;
	opt.mymodel.data_dim = 2;
	addParticle(opt.mydata.MDimg, 2, 10., 20., 30., 1., 2.);
	opt.applyClassAlignmentToParticles(1, 2, 4., 7., 9., 0., 0., 0.);
	RFLOAT rot, tilt, psi;
	opt.mydata.MDimg.getValue(EMDL_ORIENT_ROT, rot, 0);
	opt.mydata.MDimg.getValue(EMDL_ORIENT_TILT, tilt, 0);
	opt.mydata.MDimg.getValue(EMDL_ORIENT_PSI, psi, 0);
	REQUIRE(rot == Approx(6.));
	REQUIRE(tilt == Approx(20.));
	REQUIRE(psi == Approx(30.));
}

TEST_CASE("MlOptimiser: one particle of a half is adjusted like a class, offset prior included", "[align_classes]")
{
	MlOptimiser opt;
	opt.mymodel.data_dim = 2;
	addParticle(opt.mydata.MDimg, 1, 0., 90., 0., -3., 4.);
	addParticle(opt.mydata.MDimg, 2, 0., 90., 0., -3., 4.);

	// tilt 90: image x is the map's -z axis, so a shift along z moves the particle along x
	opt.applyMapAlignmentToParticle(1, 6, 0., 0., 0., 0., 0., 2.);

	RFLOAT dx, dy, px, py;
	opt.mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, dx, 0);
	REQUIRE(dx == Approx(-3.)); // the other particle is untouched
	opt.mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, dx, 1);
	opt.mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, dy, 1);
	opt.mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_X_PRIOR_ANGSTROM, px, 1);
	opt.mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_Y_PRIOR_ANGSTROM, py, 1);
	REQUIRE(dx == Approx(-5.));
	REQUIRE(dy == Approx(4.));
	REQUIRE(px == Approx(-5.));
	REQUIRE(py == Approx(4.));
}
