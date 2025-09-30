cmake_minimum_required(VERSION 3.30)
Include(FetchContent)

project(PANNA CXX)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)
set(CMAKE_BUILD_PARALLEL_LEVEL 8)
#set(CMAKE_CXX_COMPILER "/nix/store/gj9lra51hwhxnhz05jqk5lh03wipamv0-gcc-wrapper-14-20241116/bin/gcc")
# set(CMAKE_CXX_CLANG_TIDY 
#     clang-tidy;
#     -format-style='file';
#     -header-filter=${CMAKE_CURRENT_SOURCE_DIR}
#     )


# Add warnings
add_compile_options(-Wall -Wextra -Wno-noexcept-type -Wno-implicit-fallthrough -Wno-unused-function)
# Always optimize as much as we can
add_compile_options(-march=native -O3 -g)

find_package(OpenMP)
if (OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories("include")
include_directories("external")

FetchContent_Declare(dbg_macro GIT_REPOSITORY https://github.com/sharkdp/dbg-macro)
FetchContent_MakeAvailable(dbg_macro)

if (NOT SKBUILD)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.8.0 # or a later release
  )
  FetchContent_MakeAvailable(Catch2)


  # include_directories("test/include")
  # add_executable(tests test/tests.cpp)
  # target_link_libraries(tests PRIVATE Catch2::Catch2WithMain dbg_macro)

  find_package(HighFive)
  add_executable(emst test/emst_test.cpp)
  target_link_libraries(emst PRIVATE dbg_macro HighFive)

  add_executable(recall test/recall_test.cpp)
  target_link_libraries(recall PRIVATE dbg_macro HighFive)

  add_executable(scalability test/scalability_test.cpp)
  target_link_libraries(scalability PRIVATE dbg_macro HighFive)

  add_executable(mare test/mare_test.cpp)
  target_link_libraries(mare PRIVATE HighFive)

  add_executable(dtb test/DTB_test.cpp)
  target_link_libraries(dtb PRIVATE dbg_macro HighFive)

  # if (HighFive_FOUND)
  #   add_executable(glove examples/glove.cpp)
  #   target_link_libraries(glove HighFive dbg_macro)

  #   add_executable(fashion examples/fashion.cpp)
  #   target_link_libraries(fashion HighFive dbg_macro)
  # endif()
endif()

###############################################################################
# Python Bindings
# 
# This cmake code is adapted from
# - https://nanobind.readthedocs.io/en/latest/building.html
# - https://nanobind.readthedocs.io/en/latest/packaging.html

find_package(Python 3.12
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_panna_impl NOMINSIZE NB_STATIC python/wrapper.cpp)
target_link_libraries(_panna_impl PRIVATE dbg_macro)

install(TARGETS _panna_impl LIBRARY DESTINATION panna)
