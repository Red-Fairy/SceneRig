
include(FetchContent)
FetchContent_Declare(
    zlib
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG        cacf7f1d4e3d44d871b605da3b647f07d718623f
    OVERRIDE_FIND_PACKAGE
)

FetchContent_MakeAvailable(zlib)

set_target_properties(zlibstatic PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
add_library(ZLIB::ZLIB ALIAS zlibstatic)
set(ZLIB_LIBRARY ZLIB::ZLIB)
set(ZLIB_INCLUDE_DIR ${zlib_SOURCE_DIR})
set(ZLIB_FOUND TRUE)
target_include_directories(zlibstatic PUBLIC ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
