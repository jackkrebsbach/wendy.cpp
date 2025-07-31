# test/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(wendy_tests LANGUAGES CXX)

file(GLOB_RECURSE WENDY_SOURCES ../src/*.cpp)

file(GLOB TEST_SOURCES *.cpp)

add_executable(wendy_tests ${WENDY_SOURCES} ${TEST_SOURCES})

target_include_directories(wendy_tests PRIVATE /opt/homebrew/include
        /opt/homebrew/Caskroom/miniforge/base/include)
target_include_directories(wendy_tests PRIVATE /opt/homebrew/include
        /opt/homebrew/Caskroom/miniforge/base/include)

target_link_directories(wendy_tests PRIVATE /opt/homebrew/lib)


find_library(FLINT_LIBRARY flint REQUIRED)
find_library(MPFR_LIBRARY mpfr REQUIRED)
find_library(GMP_LIBRARY gmp REQUIRED)
find_library(FFTW3_LIBRARY fftw3 REQUIRED)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(xtensor REQUIRED)
find_package(symengine REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)


target_link_libraries(wendy_tests
        PRIVATE
        symengine
        xtensor
        xtensor-blas
        Eigen3::Eigen
        CppNumericalSolvers
        fftw3
        Ceres::ceres
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${FLINT_LIBRARY}
        ${MPFR_LIBRARY}
        ${GMP_LIBRARY}
        ${FFTW3_LIBRARY}
)

enable_testing()
add_test(NAME wendy_tests COMMAND wendy_tests)
