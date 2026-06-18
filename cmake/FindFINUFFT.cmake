find_path(FINUFFT_INCLUDE_DIR
    NAMES finufft.h
)

find_library(FINUFFT_LIBRARY
    NAMES finufft libfinufft
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FINUFFT DEFAULT_MSG FINUFFT_LIBRARY FINUFFT_INCLUDE_DIR)

if(FINUFFT_FOUND)
    set(FINUFFT_LIBRARIES ${FINUFFT_LIBRARY})
    set(FINUFFT_INCLUDE_DIRS ${FINUFFT_INCLUDE_DIR})
endif()

mark_as_advanced(FINUFFT_INCLUDE_DIR FINUFFT_LIBRARY)