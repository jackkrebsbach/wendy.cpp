# test/CMakeLists.txt

cmake_minimum_required(VERSION 3.15)

project(wendy_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE WENDY_SOURCES ../src/*.cpp)

file(GLOB TEST_SOURCES *.cpp)

add_executable(wendy_tests ${WENDY_SOURCES} ${TEST_SOURCES})

target_include_directories(wendy_tests
        PRIVATE
        ../src
        ${CMAKE_CURRENT_SOURCE_DIR} # for doctest.h
        /opt/homebrew/Caskroom/miniforge/base/include
)

target_link_directories(wendy_tests PRIVATE /opt/homebrew/lib)

find_package(spdlog REQUIRED)
find_package(xtensor REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP REQUIRED)

# For BLAS and LAPACK
add_definitions(-DHAVE_CBLAS=1)

if(OpenMP_CXX_FOUND)
    target_link_libraries(wendy_tests PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(wendy_tests
        PRIVATE
        symengine
        flint
        mpfr
        gmp
        spdlog::spdlog
        fmt::fmt
        fftw3
        xtensor
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
)

# Optionally, add test to CTest
enable_testing()
add_test(NAME wendy_tests COMMAND wendy_tests)
