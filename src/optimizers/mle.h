#ifndef MLE_H
#define MLE_H

#include "weak_residual_covariance.h"
#include "weak_residual.h"

// Weak negative log-likelihood
// The first term is constant so it is neglected in optimization
struct wnll {
    const CovarianceFactor &L;
    const g_functor &g;
    const xt::xtensor<double,1> &b;

    wnll(
    const CovarianceFactor &L_,
    const g_functor &g_,
    const xt::xtensor<double,1> &b_
    ): L(L_), g(g_) , b(b_) {}

    double operator()(const std::vector<double> &p) const {
        const auto Lp = L(p);
        const auto diff = g(p)-b;
        const auto S = xt::linalg::dot(Lp, xt::transpose(Lp));
        const auto y = xt::linalg::solve(Lp, diff); // Solve Ly = g(p) - b;
        const auto x = xt::linalg::solve(xt::transpose(Lp), y); // Solve L^T x = y
        const auto logdetS = std::log(xt::linalg::det(S));
        const auto quad = xt::linalg::dot(diff, x)();
        const auto wnll = logdetS + quad;
        return(wnll);
    }

};


// ∇ wnll(p) first derivative of the weak negative log likelihood (Jacobian <=> ∇ᵀ)
struct J_wnll {
    const xt::xtensor<double,2> &U;
    const xt::xtensor<double,1> &tt;
    const xt::xtensor<double,2> &V_prime;
    const CovarianceFactor &L;
    const g_functor &g;
    const xt::xtensor<double,1> &b;
    const J_g_functor &JU_g;
    const J_f_functor &Jp_f;
    const S_inv_g_minus_b_functor S_inv_g_minus_b;
   J_wnll(
       const xt::xtensor<double,2> &U_,
       const xt::xtensor<double,1> &tt_,
       const xt::xtensor<double,2> &V_prime_,
       const CovarianceFactor &L_,
       const g_functor &g_,
       const xt::xtensor<double,1> &b_,
       const J_g_functor &JU_g_,
       const J_f_functor &Jp_f_
       ):U(U_),tt(tt_),V_prime(V_prime_), L(L_), g(g_) , b(b_), JU_g(JU_g_),Jp_f(Jp_f_), S_inv_g_minus_b(S_inv_g_minus_b_functor({L,g, b})) {}

    xt::xtensor<double,2> operator()(const std::vector<double> &p) const {
       xt::xtensor<double, 1> J_wnn_eval = xt::zeros<double>({p.size()});
        // Precomputions
        const auto Lp = L(p);
        const auto S = xt::linalg::dot(Lp, xt::transpose(Lp));
        const auto S_inv_g_minus_bp = S_inv_g_minus_b(p);
        const auto diff = g(p) - b;
        // Precomputed Partial information w.r.t p
        const auto Jp_F3p = Jp_F3(Jp_f,U, tt, p);

       for (int i =0; i < p.size(); ++i) {
        // Shared values in blocks
           const auto partial_p_Fp = xt::view(Jp_F3p, xt::all(), xt::all(), i);
           const auto partial_p_gp = xt::ravel<xt::layout_type::column_major>(xt::linalg::dot(xt::eval(partial_p_Fp), V_prime));
        // Blocks
        //1

        //2

        //3
       }

   }
};

#endif //MLE_H

