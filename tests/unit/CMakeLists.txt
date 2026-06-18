# tests/unit/CMakeLists.txt
#
# Self-contained unit tests for RELION.  All tests use synthetic / analytical
# ground-truth data and do NOT require access to external datasets.
#
# Build and run:
#   cmake --build <build-dir> --target relion_unit_tests
#   cd <build-dir> && ctest -R unit --output-on-failure
#
# Or run directly:
#   <build-dir>/bin/relion_unit_tests [--reporter compact] [<tag>]

set(RELION_UNIT_TEST_SOURCES
 main.cpp
 test_ctf.cpp
 test_spatial_frequency_grid.cpp
 test_fft_nufft_consistency.cpp
 test_fftw.cpp
 test_metadata_table.cpp
 test_multidim_array.cpp
 test_matrix2d.cpp
 test_zernike.cpp
 test_anisomag_ctf.cpp
 test_aberrations.cpp
 test_ewald.cpp
 test_backprojector.cpp
 test_ewald_ctf_weight.cpp
 test_relion_finufft.cpp
 test_backprojector_nonuniform.cpp
 test_s2_vs_smode_consistency.cpp
)

add_executable(relion_unit_tests ${RELION_UNIT_TEST_SOURCES})

# Inherit the same compile definitions and include paths as relion_lib
target_include_directories(relion_unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/tests
    ${FFTW_INCLUDES}
)

# When CUDA is enabled, test sources transitively include CUDA headers through
# obs_model.h -> projector.h -> acc/cuda/cuda_mem_utils.h -> cuda_runtime.h
if (CUDA_FOUND)
    target_include_directories(relion_unit_tests PRIVATE ${CUDA_TOOLKIT_INCLUDE})
endif()

target_link_libraries(relion_unit_tests
    relion_lib
    ${FFTW_LIBRARIES}
    ${TIFF_LIBRARIES}
)

add_dependencies(relion_unit_tests relion_lib)

# Register with CTest
enable_testing()
add_test(
    NAME unit_tests
    COMMAND relion_unit_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Also register per-tag test groups so individual suites can be filtered
foreach(tag ctf s2grid s2hybrid fftw nufft metadata mda matrix2d zernike anisomag aberrations ewald backprojector smode ewald_ctf_weight relion_finufft unon s2smode)
    add_test(
        NAME unit_${tag}
        COMMAND relion_unit_tests [${tag}]
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endforeach()
