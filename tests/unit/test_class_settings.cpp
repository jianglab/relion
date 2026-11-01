/*
 * tests/unit/test_class_settings.cpp
 *
 * Per-class settings of a --ref STAR file (MlModel::ClassSettings): the columns
 * that are present are read for every class, written to model.star as table
 * model_class_settings and read back for --continue.
 */

#include <catch2/catch.hpp>
#include <cstdio>
#include <fstream>

#include "src/metadata_table.h"
#include "src/ml_model.h"

static MetaDataTable makeRefs()
{
    MetaDataTable MD;
    const char *maps[] = {"a.mrc", "b.mrc", "c.mrc"};
    const RFLOAT twist[] = {-1.13, -0.89, -1.14}, outer[] = {150., -1., 170.};
    const char *sym[] = {"C3", "C1", "C2"}, *mask[] = {"auto", "None", "c_mask.mrc"};
    for (int i = 0; i < 3; i++)
    {
        MD.addObject();
        MD.setValue(EMDL_MLMODEL_REF_IMAGE, std::string(maps[i]), i);
        MD.setValue(EMDL_MLMODEL_HELICAL_TWIST, twist[i], i);
        MD.setValue(EMDL_OPTIMISER_HELICAL_TUBE_OUTER_DIAMETER, outer[i], i);
        MD.setValue(EMDL_SAMPLING_SYMMETRY, std::string(sym[i]), i);
        MD.setValue(EMDL_OPTIMISER_SOLVENT_MASK_NAME, std::string(mask[i]), i);
    }
    return MD;
}

TEST_CASE("class settings: present columns are read, absent ones stay empty", "[class_settings]")
{
    MetaDataTable MD = makeRefs();
    MlModel::ClassSettings cs;
    cs.read(MD);
    REQUIRE(cs.twist.size() == 3);
    CHECK(cs.twist[1] == Approx(-0.89));
    REQUIRE(cs.outer_diameter.size() == 3);
    CHECK(cs.outer_diameter[1] < 0.);
    CHECK(cs.sym[0] == "C3");
    CHECK(cs.mask[0] == "auto");
    CHECK(cs.mask[2] == "c_mask.mrc");
    CHECK(cs.rise.empty());
    CHECK(cs.twist_min.empty());
    CHECK(cs.inner_diameter.empty());
    CHECK(!cs.empty());
}

TEST_CASE("class settings: round trip through model_class_settings", "[class_settings]")
{
    MetaDataTable MD = makeRefs();
    MlModel::ClassSettings cs, back;
    cs.read(MD);
    const std::string fn = "relion_class_settings_test.star";
    {
        std::ofstream fo(fn.c_str());
        fo << "data_model_classes\n\nloop_\n_rlnReferenceImage #1\nx.mrc\n\n"; // another table before it
        cs.write(fo);
    }
    MetaDataTable MDin;
    std::ifstream in(fn.c_str());
    MDin.readStar(in, "model_class_settings");
    back.read(MDin);
    in.close();
    std::remove(fn.c_str());
    CHECK(back.twist == cs.twist);
    CHECK(back.outer_diameter == cs.outer_diameter);
    CHECK(back.sym == cs.sym);
    CHECK(back.mask == cs.mask);
    CHECK(back.rise.empty());
}

TEST_CASE("class settings: nothing written without settings, half a range is refused", "[class_settings]")
{
    MlModel::ClassSettings none;
    std::ostringstream os;
    none.write(os);
    CHECK(os.str().empty());

    MetaDataTable MD;
    MD.addObject();
    MD.setValue(EMDL_MLMODEL_HELICAL_TWIST_MIN, (RFLOAT)-1.2, 0);
    MlModel::ClassSettings cs;
    REQUIRE_THROWS_AS(cs.read(MD), RelionError);
}
