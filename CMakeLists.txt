cmake_minimum_required(VERSION 3.5...3.28)
project(wendy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE WENDY_SOURCES src/*.cpp)

add_library(wendy STATIC ${WENDY_SOURCES})
find_package(OpenMP REQUIRED)

target_link_directories( wendy PRIVATE /opt/homebrew/lib)

target_include_directories(wendy PUBLIC
        /opt/homebrew/include
        /opt/homebrew/include/eigen3
        /opt/homebrew/include/coin-or
        ${CMAKE_CURRENT_SOURCE_DIR}/../../external/xtl/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../external/xtensor/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../external/xtensor-blas/include
)

target_link_directories(wendy PUBLIC /opt/homebrew/lib)

target_link_libraries(wendy PUBLIC
        OpenMP::OpenMP_CXX
        symengine
        fftw3
        ceres
        flint
        gmp
        mpfr
        lapack
        ipopt
)

target_compile_options(wendy PRIVATE
        -O3
        -march=native
        -mtune=native
        -fno-math-errno -fno-signed-zeros -fno-trapping-math -freciprocal-math
        -funroll-loops
        -flto
        -fstrict-aliasing
        -DNDEBUG
)

option(BUILD_EXAMPLES "Build example programs" OFF)
option(BUILD_TESTS "Build tests" OFF)

if(BUILD_EXAMPLES)
    add_subdirectory(example)
endif()
if(BUILD_TESTS)
    add_subdirectory(test)
endif()
