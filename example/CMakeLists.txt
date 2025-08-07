cmake_minimum_required(VERSION 3.15)
project(wendy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

find_library(GMP_LIB gmp REQUIRED)
find_library(SYMENGINE_LIB symengine PATHS /usr/local/lib REQUIRED)
find_path(SYMENGINE_INCLUDE_DIR symengine/lambda_double.h PATHS /usr/local/include REQUIRED)



find_library(FFTW3_LIBRARY NAMES fftw3)
if(NOT FFTW3_LIBRARY)
    if(EXISTS /opt/homebrew/lib/libfftw3.a)
        set(FFTW3_LIBRARY /opt/homebrew/lib/libfftw3.a)
    else()
        message(FATAL_ERROR "FFTW3 library not found")
    endif()
endif()

pkg_check_modules(IPOPT REQUIRED IMPORTED_TARGET ipopt)

file(GLOB_RECURSE WENDY_SOURCES ../src/*.cpp)
file(GLOB_RECURSE WENDY_HEADERS ../include/wendy/*.h)

function(add_wendy_example EXAMPLE_NAME EXAMPLE_SOURCE)
    file(GLOB_RECURSE EXAMPLE_CORE ../src/*.cpp)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE} ${EXAMPLE_CORE})

    target_include_directories(${EXAMPLE_NAME} PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/../include/wendy
            ${CMAKE_CURRENT_SOURCE_DIR}/../include
            ${SYMENGINE_INCLUDE_DIR}
            ${IPOPT_INCLUDE_DIRS}
            /usr/local/include/

    )

    target_link_libraries(${EXAMPLE_NAME} PUBLIC
            xtensor
            xtensor-blas
            Eigen3::Eigen
            Ceres::ceres
            PkgConfig::IPOPT
            ${SYMENGINE_LIB}
            ${GMP_LIB}
            ${BLAS_LIBRARIES}
            ${LAPACK_LIBRARIES}
            ${FFTW3_LIBRARY}
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

