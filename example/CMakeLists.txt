cmake_minimum_required(VERSION 3.15)
project(wendy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

file(GLOB_RECURSE WENDY_CORE_SOURCES ../src/*.cpp)

if (WIN32)
    find_package(OpenBLAS REQUIRED)
    set(BLAS_LIBRARIES ${CMAKE_INSTALL_PREFIX}${OpenBLAS_LIBRARIES})
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()

message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})

function(add_wendy_example EXAMPLE_NAME EXAMPLE_SOURCE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE} ${WENDY_CORE_SOURCES} ${WENDY_HEADERS})

    target_link_directories( ${EXAMPLE_NAME} PRIVATE /opt/homebrew/lib)

    target_include_directories(${EXAMPLE_NAME} PRIVATE
            /opt/homebrew/include
            /opt/homebrew/include/eigen3
            /opt/homebrew/include/coin-or
            ${CMAKE_CURRENT_SOURCE_DIR}/../external/xtl/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../external/xtensor/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../external/xtensor-blas/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../external/spdlog/include
    )

    target_link_libraries(${EXAMPLE_NAME} PRIVATE
            spdlog
            symengine
            fftw3
            ceres
            flint
            gmp
            mpfr
            ipopt
            fmt
            ${BLAS_LIBRARIES}
            ${LAPACK_LIBRARIES}
    )
    target_compile_options(${EXAMPLE_NAME} PRIVATE -O2 -DNDEBUG -funroll-loops)

endfunction()

add_wendy_example(example_lorenz example_lorenz.cpp)
add_wendy_example(example_logistic example_logistic.cpp)
add_wendy_example(example_goodwin_2d example_goodwin_2d.cpp)
add_wendy_example(example_goodwin_3d example_goodwin_3d.cpp)
add_wendy_example(example_hindmarsh_rose example_hindmarsh_rose.cpp)
