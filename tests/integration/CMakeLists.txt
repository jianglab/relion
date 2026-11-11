# Integration tests for s² spatial frequency mode workflows
#
# These tests spawn actual RELION programs with synthetic data and validate outputs.
# Uses pytest (Python) for end-to-end testing with external programs.
#
# Run with: pytest tests/integration/ -v
# Run specific test: pytest tests/integration/test_reconstruction.py -v
#
# Note: earlier Catch2/C++ integration test prototypes were removed once the
# Python-based suite became the supported implementation.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

message(STATUS "Integration tests configured for pytest (Python)")
message(STATUS "  Run with: pytest tests/integration/ -v")
message(STATUS "  See tests/integration/README.md for documentation")

# Per-category test groups (only tags with existing test_${tag}.py files)
foreach(tag reconstruction ewald reconstruct_crash cryosparc_import cryosparc_project_import mpi_prefetch filament_class3d filament_context_class3d class_settings class2d_consensus virtual_particles virtual_movie_averages keep_iterations fused_reconstruct rect_program_audit bp_pack gpu_keep_pool gpu_small_speedups coarse_cache skipalign_block cc_coarse coarse_early group_shared mpi_align_classes mpi_align_halves async_serial async_write auto_threads async_star_write bp_accum_mpi continue_skip_align gpu_select_fused ini_high_reference_free lazy_cache map_fsc prep_ahead rectangular_reconstruct helical_end_weight write_unmasked_maps)
    add_test(
        NAME integration_${tag}
        COMMAND ${Python3_EXECUTABLE} -m pytest -v ${CMAKE_SOURCE_DIR}/tests/integration/test_${tag}.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    set_property(TEST integration_${tag} PROPERTY LABELS integration integration_${tag})
    # PYTEST_DISABLE_PLUGIN_AUTOLOAD: these tests drive RELION binaries through
    # subprocess and need no third-party pytest plugin, but an unrelated broken
    # plugin in the environment takes the whole suite down during plugin loading,
    # before any test is even collected.  (Seen with napari's plugin, which uses
    # pydantic v1's ModelMetaclass, against pydantic v2.)  Add plugins explicitly
    # with -p here if the suite ever needs one.
    set_property(TEST integration_${tag} PROPERTY ENVIRONMENT
        "RELION_BIN_DIR=${CMAKE_BINARY_DIR}/bin;PYTEST_DISABLE_PLUGIN_AUTOLOAD=1")
endforeach()
# Six small 3D classifications take about an hour on 8 cores, more than ctest's default 1500 s
set_property(TEST integration_mpi_align_classes PROPERTY TIMEOUT 7200)

# Aggregate test to run all integration tests
add_test(
    NAME integration_all
    COMMAND ${Python3_EXECUTABLE} -m pytest -v ${CMAKE_SOURCE_DIR}/tests/integration
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST integration_all PROPERTY LABELS integration)
# The whole suite takes about 22 min on one RTX A4000 (2 h with the three slow MPI class-alignment runs and the
# sequential refinements that used to be repeated). The timeout stays generous for slower hosts.
set_property(TEST integration_all PROPERTY TIMEOUT 14400)
set_property(TEST integration_all PROPERTY ENVIRONMENT
    "RELION_BIN_DIR=${CMAKE_BINARY_DIR}/bin;PYTEST_DISABLE_PLUGIN_AUTOLOAD=1")

message(STATUS "  → Integration test framework ready")
message(STATUS "  → Run: ctest -L integration [--verbose]")
message(STATUS "  → Python fixtures: tests/integration/fixtures.py")
message(STATUS "  → Validation: tests/integration/validate.py")
