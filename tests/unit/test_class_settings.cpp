/*
 * tests/unit/test_class_settings.cpp
 *
 * Per-class settings of a --ref STAR file (MlModel::ClassSettings): the columns
 * that are present are read for every class, written to model.star as table
 * model_class_settings and read back for --continue. Also the class order of
 * --sort_classes.
 */

#include <catch2/catch.hpp>
#include <cstdio>
#include <fstream>
#include <limits>

#include "src/metadata_table.h"
#include "src/ml_model.h"
#include <unistd.h>

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
    const std::string fn = "relion_class_settings_test_" + std::to_string(getpid()) + ".star";
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
    CHECK(back.twist_step.empty());
}

TEST_CASE("class settings: per-class search steps are read and kept", "[class_settings]")
{
    MetaDataTable MD = makeRefs();
    const RFLOAT twist_step[] = {0.1, 0., 0.05}, rise_step[] = {0.02, 0.02, 0.};
    for (int i = 0; i < 3; i++)
    {
        MD.setValue(EMDL_MLMODEL_HELICAL_TWIST_INITIAL_STEP, twist_step[i], i);
        MD.setValue(EMDL_MLMODEL_HELICAL_RISE_INITIAL_STEP, rise_step[i], i);
    }
    MlModel::ClassSettings cs, back;
    cs.read(MD);
    REQUIRE(cs.twist_step.size() == 3);
    CHECK(cs.twist_step[2] == Approx(0.05));
    REQUIRE(cs.rise_step.size() == 3);
    CHECK(cs.rise_step[0] == Approx(0.02));
    const std::string fn = "relion_class_steps_test_" + std::to_string(getpid()) + ".star";
    {
        std::ofstream fo(fn.c_str());
        cs.write(fo);
    }
    MetaDataTable MDin;
    std::ifstream in(fn.c_str());
    MDin.readStar(in, "model_class_settings");
    back.read(MDin);
    in.close();
    std::remove(fn.c_str());
    CHECK(back.twist_step == cs.twist_step);
    CHECK(back.rise_step == cs.rise_step);
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

// --sort_classes: the order (MlModel::sortedClassOrder) and the renumbering (MlModel::reorderClasses)
TEST_CASE("class order: by helical type, then size, junk last", "[class_order]")
{
    const RFLOAT inf = std::numeric_limits<RFLOAT>::infinity();
    const std::vector<RFLOAT> pdf = {0.1, 0.3, 0.0, 0.2, 0.25, 0.15}, res = {8., 7., inf, 9., 30., 8.};
    std::string rule;
    // median 8 A: 30 A is junk at 1.5 x, as is the empty class 3; type 2 has more particles than type 1
    CHECK(MlModel::sortedClassOrder(pdf, res, {2, 1, 0, 2, 1, 2}, 1.5, rule) == std::vector<int>({3, 5, 0, 1, 4, 2}));
    CHECK(rule.find("by helical type") != std::string::npos);
    CHECK(rule.find("12 A") != std::string::npos);
    // no types: by size, junk last; junk factor 0: only empty classes are junk
    CHECK(MlModel::sortedClassOrder(pdf, res, {}, 1.5, rule) == std::vector<int>({1, 3, 5, 0, 4, 2}));
    CHECK(MlModel::sortedClassOrder(pdf, res, {}, 0., rule) == std::vector<int>({1, 4, 3, 5, 0, 2}));
    // types renumbered in the order of their classes (type 2 first); type 3 has no class: last
    CHECK(MlModel::typeNumbersFromClassOrder({3, 5, 0, 1, 4, 2}, {2, 1, 0, 2, 1, 2}, 3) == std::vector<int>({0, 2, 1, 3}));
}

TEST_CASE("class order: reorderClasses moves every per-class value together", "[class_order]")
{
    MlModel m;
    m.nr_classes = 3;
    m.nr_bodies = 1;
    m.pdf_class = {0.2, 0.5, 0.3};
    m.estimated_resolution = {9., 7., 8.};
    m.helical_type_class = {1, 2, 1};
    m.Iref.resize(3);
    for (int k = 0; k < 3; k++)
    {
        m.Iref[k].initZeros(4, 4);
        m.Iref[k].initConstant(k + 1.);
    }
    m.class_settings.sym = {"C1", "C2", "C3"};
    m.reorderClasses({1, 2, 0});
    CHECK(m.pdf_class == std::vector<RFLOAT>({0.5, 0.3, 0.2}));
    CHECK(m.estimated_resolution == std::vector<RFLOAT>({7., 8., 9.}));
    CHECK(m.helical_type_class == std::vector<int>({2, 1, 1}));
    CHECK(DIRECT_MULTIDIM_ELEM(m.Iref[0], 0) == 2.);
    CHECK(DIRECT_MULTIDIM_ELEM(m.Iref[2], 5) == 1.);
    CHECK(m.class_settings.sym == std::vector<std::string>({"C2", "C3", "C1"}));
    CHECK(m.class_settings.twist.empty());  // absent settings stay absent
}
