#include "./doctest.h"
#include "../src/utils.h"
#include "../src/fft.h"


TEST_CASE("calculate_fft basic test") {
    // Create a test matrix: 2 rows of constant values
    xt::xarray<double> test_data = {{1.0, 1.0, 1.0, 1.0},
                                    {2.0, 2.0, 2.0, 2.0}};

    auto result = calculate_fft(test_data);

    CHECK(result.shape()[0] == 2); // 2 rows
    CHECK(result.shape()[1] == 3); // nfreq = 4/2 + 1 = 3

    for (size_t row = 0; row < 2; ++row) {
        CHECK(std::abs(result(row, 0).real() - 4.0 * test_data(row, 0)) < 1e-10); // DC component
        for (size_t freq = 1; freq < 3; ++freq) {
            CHECK(std::abs(result(row, freq)) < 1e-10); // Should be zero for constant input
        }
    }
}
