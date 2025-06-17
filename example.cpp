#include <iostream>
#include <vector>
#include <string>
#include <xtensor/containers/xarray.hpp>
#include "src/wendy.h"

int main() {
    std::vector<std::string> system_eqs = {
        "p0 * u0 + 2*p0 + 32",
        "p1 * u1"
    };

    xt::xarray<double> U = {
        {1.0, 2.0},
        {1.1, 2.1},
        {1.2, 2.2},
        {1.3, 2.3}
    };

    std::vector<float> p0 = {0.5f, 1.5f};

    try {
        const Wendy wendy(system_eqs, U, p0);
        wendy.log_details();
    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
