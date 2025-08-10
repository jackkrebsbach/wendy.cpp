cmake_minimum_required(VERSION 3.15)
project(wendy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


file(GLOB_RECURSE WENDY_CORE_SOURCES ../src/*.cpp)
find_package(OpenMP REQUIRED)

function(add_wendy_example EXAMPLE_NAME EXAMPLE_SOURCE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE} ${WENDY_CORE_SOURCES} ${WENDY_HEADERS})

    target_link_directories( ${EXAMPLE_NAME} PRIVATE /opt/homebrew/lib)

    target_include_directories(${EXAMPLE_NAME} PRIVATE
            /opt/homebrew/include
            /opt/homebrew/include/eigen3
            /opt/homebrew/include/coin-or
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../external/xtl/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../external/xtensor/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../external/xtensor-blas/include
    )
    target_link_libraries(${EXAMPLE_NAME} PRIVATE
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
