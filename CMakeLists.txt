cmake_minimum_required(VERSION 3.5...3.28)
project(wendy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_library(FLINT_LIBRARY flint REQUIRED)
find_library(MPFR_LIBRARY mpfr REQUIRED)
find_library(GMP_LIBRARY gmp REQUIRED)

if(EXISTS /opt/homebrew/lib/libfftw3.a)
    set(FFTW3_LIBRARY /opt/homebrew/lib/libfftw3.a)
else()
    find_library(FFTW3_LIBRARY fftw3 REQUIRED)
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(IPOPT REQUIRED IMPORTED_TARGET ipopt)

if (APPLE)
  message(STATUS "Manually linking SymEngine on macOS")
  include_directories(/opt/homebrew/include)
  link_directories(/opt/homebrew/lib)
  set(SYMENGINE_LIB symengine)
else()
  find_package(SymEngine REQUIRED)
  set(SYMENGINE_LIB SymEngine::symengine)
endif()

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

file(GLOB_RECURSE WENDY_SOURCES src/*.cpp)
file(GLOB_RECURSE WENDY_HEADERS include/wendy/*.h)

add_library(wendy STATIC ${WENDY_SOURCES} ${WENDY_HEADERS})

target_include_directories(wendy PUBLIC
        /opt/homebrew/include
        /opt/homebrew/include/coin-or
        /opt/homebrew/Caskroom/miniforge/base/include
        $<TARGET_PROPERTY:Eigen3::Eigen,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:xtensor,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:xtensor-blas,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:symengine,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:Ceres::ceres,INTERFACE_INCLUDE_DIRECTORIES>
        ${CMAKE_CURRENT_SOURCE_DIR}/include/wendy
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${IPOPT_INCLUDE_DIRS}
        ${FFTW3_INCLUDE_DIR}
)

#target_link_directories(wendy PRIVATE /opt/homebrew/lib)

target_link_libraries(wendy PUBLIC
        symengine
        xtensor
        xtensor-blas
        ipopt
        Eigen3::Eigen
        SymEngine::symengine
        Ceres::ceres
        OpenMP::OpenMP_CXX
        ${IPOPT_LIBRARIES}
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${FLINT_LIBRARY}
        ${MPFR_LIBRARY}
        ${GMP_LIBRARY}
        ${FFTW3_LIBRARY}
)

target_compile_options(wendy PRIVATE
        -O3
        -march=native
        -mtune=native
        -ffast-math
        -funroll-loops
        -flto
        -fstrict-aliasing
        -DNDEBUG
)

option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build tests" OFF)

if(BUILD_EXAMPLES)
    add_subdirectory(example)
endif()
if(BUILD_TESTS)
    add_subdirectory(test)
endif()

#add_subdirectory(example)
#add_subdirectory(test)
