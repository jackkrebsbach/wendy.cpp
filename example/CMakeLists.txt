cmake_minimum_required(VERSION 3.15)
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

file(GLOB_RECURSE WENDY_CORE_SOURCES ../src/*.cpp)
file(GLOB_RECURSE WENDY_HEADERS ../include/wendy/*.h)

function(add_wendy_example EXAMPLE_NAME EXAMPLE_SOURCE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE} ${WENDY_CORE_SOURCES} ${WENDY_HEADERS})
    target_include_directories(${EXAMPLE_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../external/xtl/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../external/xtensor/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../external/xtensor-blas/include
    )
    target_link_libraries(${EXAMPLE_NAME} PRIVATE
            symengine
            Eigen3::Eigen
            Ceres::ceres
            ${BLAS_LIBRARIES}
            ${LAPACK_LIBRARIES}
            ${FFTW3_LIBRARY}
            ${GMP_LIBRARY}
            ${GMPXX_LIBRARY}

    )
    target_compile_options(${EXAMPLE_NAME} PRIVATE
            -O3
            -march=native
            -mtune=native
            -fno-math-errno -fno-signed-zeros -fno-trapping-math -freciprocal-math
            -funroll-loops
            -fstrict-aliasing
            -DNDEBUG
    )
endfunction()

add_wendy_example(example_lorenz example_lorenz.cpp)
add_wendy_example(example_logistic example_logistic.cpp)
add_wendy_example(example_goodwin_2d example_goodwin_2d.cpp)
add_wendy_example(example_goodwin_3d example_goodwin_3d.cpp)
