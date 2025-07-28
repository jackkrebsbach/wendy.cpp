#ifndef UTILS_H
#define UTILS_H

#include "weak_residual.h"

#include <xtensor/misc/xsort.hpp>
#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <limits.h>

inline void print_cwd() {
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd))) {
        std::cout << "Working dir: " << cwd << std::endl;
    }
}

inline xt::xtensor<double, 2> read_csv(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> rows;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(ss, cell, ',')) {
            cell.erase(std::remove_if(cell.begin(), cell.end(), ::isspace), cell.end());
            if (!cell.empty()) {
                try {
                    row.push_back(std::stod(cell));
                } catch (...) {
                    std::cerr << "Warning: Non-numeric cell: " << cell << '\n';
                    row.push_back(0.0);
                }
            }
        }

        if (!row.empty()) {
            rows.push_back(std::move(row));
        }
    }

    size_t num_rows = rows.size();
    size_t num_cols = rows[0].size();
    std::vector<double> flat_data;
    flat_data.reserve(num_rows * num_cols);

    for (const auto& row : rows) {
        flat_data.insert(flat_data.end(), row.begin(), row.end());
    }
    return xt::adapt(flat_data, std::vector<std::size_t>{num_rows, num_cols});

}



template<typename T, typename Predicate>
int find_last(const xt::xtensor<T, 1> &arr, Predicate pred) {
    for (int i = arr.size() - 1; i >= 0; --i) {
        if (pred(arr.flat(i))) return i;
    }
    return -1;
}

inline bool is_symmetric(const std::vector<std::vector<double> > &H, double tol = 1e-10) {
    const size_t n = H.size();
    for (size_t i = 0; i < n; ++i) {
        if (H[i].size() != n) return false; // Not square
        for (size_t j = 0; j < n; ++j) {
            if (std::abs(H[i][j] - H[j][i]) > tol) {
                std::cout << "Non-symmetric at (" << i << "," << j << "): "
                        << H[i][j] << " vs " << H[j][i] << std::endl;
                return false;
            }
        }
    }
    return true;
}

inline void print_matrix(const std::vector<std::vector<double> > &mat, const int precision = 1) {
    const size_t n = mat.size();
    for (size_t i = 0; i < n; ++i) {
        for (const double j: mat[i]) {
            std::cout << std::setw(precision + 6) << std::setprecision(precision) << std::fixed << j << " ";
        }
        std::cout << std::endl;
    }
}

inline void print_vector(const std::vector<double>& vec, const int precision = 1) {
    for (const double val : vec) {
        std::cout << std::setw(precision + 6)
                  << std::setprecision(precision)
                  << std::fixed
                  << val << " ";
    }
    std::cout << std::endl;
}

xt::xarray<double> solve_cholesky(const xt::xarray<double>& L, const xt::xarray<double>& B);

std::function<double(double)> make_scalar_function(const SymEngine::Expression &expr,
                                                   const SymEngine::RCP<const SymEngine::Symbol> &var);

f_functor build_f(const std::vector<SymEngine::Expression> &f_symbolic, size_t D, size_t J);

J_f_functor build_J_f(const std::vector<std::vector<SymEngine::Expression> > &J_f_symbolic, size_t D, size_t J);

H_f_functor build_H_f(const std::vector<std::vector<std::vector<SymEngine::Expression> > > &H_f_symbolic, size_t D,
                      size_t J);

T_f_functor build_T_f(const std::vector<std::vector<std::vector<std::vector<SymEngine::Expression> > > > &T_f_symbolic,
                      size_t D, size_t J);

size_t get_corner_index(const xt::xtensor<double, 1> &yy, const xt::xtensor<double, 1> *xx_in = nullptr);


#endif //UTILS_H
