#include "./doctest.h"
#include "../src/utils.h"


TEST_CASE("get_corner_index basic test with known corner") {
    xt::xtensor<double, 1> yy = {0.0, 1.0, 4.0, 9.0, 16.0}; // y = x^2, max curvature at beginning or end
    size_t result = get_corner_index(yy, nullptr);
    CHECK(result == 2); // midpoint in parabola
}

TEST_CASE("get_corner_index with descending curve") {
    xt::xtensor<double, 1> yy = {16.0, 9.0, 4.0, 1.0, 0.0};
    size_t result = get_corner_index(yy, nullptr);
    CHECK(result == 2);
}

TEST_CASE("get_corner_index with custom x spacing") {
    xt::xtensor<double, 1> xx = {0.0, 1.0, 2.0, 5.0, 10.0};
    xt::xtensor<double, 1> yy = {0.0, 1.0, 4.0, 25.0, 100.0};
    size_t result = get_corner_index(yy, &xx);
    CHECK(result == 2);
}
