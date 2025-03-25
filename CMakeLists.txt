cmake_minimum_required(VERSION 3.30)
Include(FetchContent)

project(PANNA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-noexcept-type -Wno-implicit-fallthrough -Wno-unused-function -march=native -g -O3")

find_package(OpenMP)
if (OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories("include")
include_directories("external")

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.0 # or a later release
)

FetchContent_MakeAvailable(Catch2)

FetchContent_Declare(dbg_macro GIT_REPOSITORY https://github.com/sharkdp/dbg-macro)
FetchContent_MakeAvailable(dbg_macro)

include_directories("test/include")
add_executable(tests test/tests.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain dbg_macro)


find_package(HighFive)
if (HighFive_FOUND)
  add_executable(glove examples/glove.cpp)
  target_link_libraries(glove HighFive dbg_macro)
endif()

