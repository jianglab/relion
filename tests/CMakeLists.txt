add_executable(tests EXCLUDE_FROM_ALL
		tests.cpp
	)

add_executable(cache_tests
		cache.cpp
	)

add_dependencies(tests relion_lib)
add_dependencies(cache_tests relion_lib)

target_link_libraries(tests relion_lib)
target_link_libraries(tests ${FFTW_LIBRARIES})
target_link_libraries(tests ${TIFF_LIBRARIES})

target_link_libraries(cache_tests relion_lib)
target_link_libraries(cache_tests ${FFTW_LIBRARIES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/tests)
include_directories(tests PRIVATE ${FFTW_INCLUDES})

# When CUDA is enabled, test targets transitively include CUDA headers
# through exp_model.h -> obs_model.h -> projector.h -> cuda_mem_utils.h
if (CUDA_FOUND)
    target_include_directories(cache_tests PRIVATE ${CUDA_TOOLKIT_INCLUDE})
endif()

add_test(NAME cache_unit_tests COMMAND cache_tests)

find_package(OpenMP)
if (OPENMP_FOUND)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Self-contained unit tests with analytical ground truth (no external data)
add_subdirectory(unit)

# Integration tests (optional, longer runtime)
add_subdirectory(integration)
