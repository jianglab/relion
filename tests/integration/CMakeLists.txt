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
foreach(tag reconstruction ewald reconstruct_crash)
    add_test(
        NAME integration_${tag}
        COMMAND ${Python3_EXECUTABLE} -m pytest -v ${CMAKE_SOURCE_DIR}/tests/integration/test_${tag}.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    set_property(TEST integration_${tag} PROPERTY LABELS integration integration_${tag})
    set_property(TEST integration_${tag} PROPERTY ENVIRONMENT "RELION_BIN_DIR=${CMAKE_BINARY_DIR}/bin")
endforeach()

# Aggregate test to run all integration tests
add_test(
    NAME integration_all
    COMMAND ${Python3_EXECUTABLE} -m pytest -v ${CMAKE_SOURCE_DIR}/tests/integration
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST integration_all PROPERTY LABELS integration)
set_property(TEST integration_all PROPERTY ENVIRONMENT "RELION_BIN_DIR=${CMAKE_BINARY_DIR}/bin")

message(STATUS "  → Integration test framework ready")
message(STATUS "  → Run: ctest -L integration [--verbose]")
message(STATUS "  → Python fixtures: tests/integration/fixtures.py")
message(STATUS "  → Validation: tests/integration/validate.py")
