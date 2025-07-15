
#ifndef MLE_H
#define MLE_H

#include "weak_residual_covariance.h"
#include "weak_residual.h"

// S(p)^-1(g(p) - b) is a function of the parameters p⃗
struct S_inv_g_minus_b {
    CovarianceFactor L;
    g_functor g;
    xt::xtensor<double,1> b;
    S_inv_g_minus_b(
        const CovarianceFactor &L_,
        const g_functor &g_,
        const xt::xtensor<double,1> &b_
        ): L(L_), g(g_) , b(b_){
    }

    // TODO: Investigate in using solve_triangular to take advantage of structure of L
    xt::xtensor<double,1> operator()(const std::vector<double> &p) const {
        const auto Lp = L(p);
        const auto y = xt::linalg::solve(Lp, g(p)-b); // Solve Ly = g(p) - b;
        const auto x = xt::linalg::solve(xt::transpose(Lp), y); // Solve L^T x = y
        return(x);
    }

};

#endif //MLE_H
