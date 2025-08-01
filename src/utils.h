#ifndef UTILS_H
#define UTILS_H

#include "weak_residual.h"

#include <xtensor/misc/xsort.hpp>
#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>


template<typename T>
void print_xtensor2d(const T &tensor) {
    auto shape = tensor.shape();
    if (shape.size() != 2) {
        std::cerr << "Tensor is not 2D!" << std::endl;
        return;
    }
    for (std::size_t i = 0; i < shape[0]; ++i) {
        for (std::size_t j = 0; j < shape[1]; ++j) {
            std::cout << tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

inline void print_system(const std::vector<SymEngine::Expression> &system) {
    std::cout << "\n==================== ODE system ====================\n";
    for (size_t i = 0; i < system.size(); ++i) {
        std::cout << "f[" << i << "] = " << SymEngine::str(*system[i].get_basic())
            << std::endl;
    }
    std::cout << "===================================================\n"
        << std::endl;
}

inline void print_jacobian(
  const std::vector<std::vector<SymEngine::Expression> > &jac) {
    std::cout << "\n==================== Jacobian =====================\n";
    for (size_t i = 0; i < jac.size(); ++i) {
        for (size_t j = 0; j < jac[i].size(); ++j) {
            std::cout << SymEngine::str(*jac[i][j].get_basic()) << "  ";
        }
        std::cout << std::endl;
    }
    std::cout << "===================================================\n"
        << std::endl;
}



inline std::vector<double> gradient_4th_order(
    const std::function<double(const std::vector<double>&)>& f,
    const std::vector<double>& x,
    double h = 1e-5
) {
    const size_t n = x.size();
    std::vector<double> grad(n);

    for (size_t i = 0; i < n; ++i) {
        std::vector<double> xp2h = x, xph = x, xmh = x, xm2h = x;
        xp2h[i] += 2*h;
        xph[i]  += h;
        xmh[i]  -= h;
        xm2h[i] -= 2*h;

        grad[i] = (-f(xp2h) + 8*f(xph) - 8*f(xmh) + f(xm2h)) / (12*h);
    }
    return grad;
}

inline std::vector<std::vector<double>> hessian_3rd_order(
    const std::function<double(const std::vector<double>&)>& f,
    const std::vector<double>& x,
    double h = 1e-6
) {
    const size_t n = x.size();
    std::vector<std::vector<double>> H(n, std::vector<double>(n, 0.0));

    // Diagonal: 5-point stencil (4th order)
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> xpp = x, xp = x, xm = x, xmm = x;
        xpp[i] += 2*h;
        xp[i]  += h;
        xm[i]  -= h;
        xmm[i] -= 2*h;
        H[i][i] = (-f(xpp) + 16*f(xp) - 30*f(x) + 16*f(xm) - f(xmm)) / (12 * h * h);
    }

    // Off-diagonal: 3rd-order central difference for mixed partials
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            std::vector<double> xpp = x, xpm = x, xmp = x, xmm = x;

            xpp[i] += h; xpp[j] += h;
            xpm[i] += h; xpm[j] -= h;
            xmp[i] -= h; xmp[j] += h;
            xmm[i] -= h; xmm[j] -= h;

            H[i][j] = H[j][i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h * h);
        }
    }

    return H;
}


inline void print_cwd() {
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd))) {
        std::cout << "Working dir: " << cwd << std::endl;
    }
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

inline void print_vector(const std::vector<double>& vec, const int precision = 3) {
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

size_t get_corner_index(const xt::xtensor<double, 1> &y, const xt::xtensor<double, 1> *xx_in = nullptr);

#endif //UTILS_H
