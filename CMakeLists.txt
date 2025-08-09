cmake_minimum_required(VERSION 3.5...3.28)
project(wendy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_library(GMP_LIBRARY NAMES gmp REQUIRED)
find_library(GMPXX_LIBRARY NAMES gmpxx REQUIRED)
find_package(Symengine REQUIRED)
find_library(FFTW3_LIBRARY NAMES fftw3 REQUIRED)

file(GLOB_RECURSE WENDY_SOURCES src/*.cpp)
file(GLOB_RECURSE WENDY_HEADERS include/wendy/*.h)

message(STATUS "WENDY_SOURCES: ${WENDY_SOURCES}")
message(STATUS "WENDY_HEADERS: ${WENDY_HEADERS}")


add_library(wendy STATIC ${WENDY_SOURCES} ${WENDY_HEADERS})

target_include_directories(wendy PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../external/xtl/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../external/xtensor/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../external/xtensor-blas/include
        ${FFTW3_INCLUDE_DIR}
)

target_link_libraries(wendy PUBLIC
        symengine
        Eigen3::Eigen
        Ceres::ceres
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${FFTW3_LIBRARY}
        ${GMP_LIBRARY}
        ${GMPXX_LIBRARY}

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
