#ifndef UTILS_H
#define UTILS_H

#include "weak_residual.h"

#include <xtensor/misc/xsort.hpp>
#include <symengine/expression.h>
#include <symengine/lambda_double.h>



template<typename T, typename Predicate>
int find_last(const xt::xtensor<T, 1> &arr, Predicate pred) {
    for (int i = arr.size() - 1; i >= 0; --i) {
        if (pred(arr.flat(i))) return i;
    }
    return -1;
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
