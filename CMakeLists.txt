cmake_minimum_required(VERSION 3.5...3.28)
project(wendy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DHAVE_CBLAS=1)

file(GLOB_RECURSE WENDY_SOURCES src/*.cpp)

add_library(wendy STATIC ${WENDY_SOURCES})

if (WIN32)
    find_package(OpenBLAS REQUIRED)
    set(BLAS_LIBRARIES ${CMAKE_INSTALL_PREFIX}${OpenBLAS_LIBRARIES})
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()

message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})

target_link_directories( wendy PRIVATE /opt/homebrew/lib)

target_include_directories(wendy PUBLIC
        /opt/homebrew/include
        /opt/homebrew/include/eigen3
        /opt/homebrew/include/coin-or
        ${CMAKE_CURRENT_SOURCE_DIR}/external/xtl/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/xtensor/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/xtensor-blas/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/spdlog/include
)

target_link_directories(wendy PUBLIC /opt/homebrew/lib)

target_link_libraries(wendy PUBLIC
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

#target_compile_options(wendy PRIVATE
#        -O3
#        -march=native
#        -mtune=native
#        -O3 -DNDEBUG -march=native -mtune=native -funroll-loops
#        -fno-math-errno
#        -funsafe-math-optimizations
#        -fno-trapping-math
#        -fno-signed-zeros
#        -fcx-limited-range
#        -fexcess-precision=fast
#        -flto -fstrict-aliasing
#        -funroll-loops
#        -flto
#        -fstrict-aliasing
#        -DNDEBUG
#)

option(BUILD_EXAMPLES "Build example programs" OFF)
option(BUILD_TESTS "Build tests" OFF)

if(BUILD_EXAMPLES)
    add_subdirectory(example)
endif()
if(BUILD_TESTS)
    add_subdirectory(test)
endif()
