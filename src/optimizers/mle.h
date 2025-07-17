#ifndef MLE_H
#define MLE_H

#include "../weak_residual_covariance.h"
#include "../weak_residual.h"

// Weak negative log-likelihood
struct wnll {
    const CovarianceFactor &L;
    const g_functor &g;
    const xt::xtensor<double, 1> &b;

    wnll(
        const CovarianceFactor &L_,
        const g_functor &g_,
        const xt::xtensor<double, 1> &b_
    ): L(L_), g(g_), b(b_) {
    }

    double operator()(const std::vector<double> &p) const {
        const auto Lp = L(p);
        const auto r = g(p) - b;
        const auto S = xt::linalg::dot(Lp, xt::transpose(Lp));
        const auto y = xt::linalg::solve(Lp, r); // Solve Ly = g(p) - b;
        const auto x = xt::linalg::solve(xt::transpose(Lp), y); // Solve L^T x = y
        const auto logdetS = std::log(xt::linalg::det(S));
        const auto quad = xt::linalg::dot(r, x)();
        const auto wnll = logdetS + quad;
        return (wnll);
    }
};


// ∇ wnll(p) first derivative of the weak negative log likelihood (Jacobian <=> ∇ᵀ)
struct J_wnll {
    const xt::xtensor<double, 2> &U;
    const xt::xtensor<double, 1> &tt;
    const xt::xtensor<double, 2> &V;
    const xt::xtensor<double, 2> &V_prime;
    const CovarianceFactor &L;
    const g_functor &g;
    const xt::xtensor<double, 1> &b;
    const J_g_functor &JU_g;
    const J_g_functor &Jp_g;
    const S_inv_r_functor S_inv_r;
    size_t K;
    size_t mp1;
    size_t D;

    J_wnll(
        const xt::xtensor<double, 2> &U_,
        const xt::xtensor<double, 1> &tt_,
        const xt::xtensor<double, 2> &V_,
        const xt::xtensor<double, 2> &V_prime_,
        const CovarianceFactor &L_,
        const g_functor &g_,
        const xt::xtensor<double, 1> &b_,
        const J_f_functor &Ju_f_,
        const J_f_functor &Jp_f_
    ): U(U_), tt(tt_), V(V_), V_prime(V_prime_), L(L_), g(g_), b(b_),
       JU_g(J_g_functor(U, tt, V, Ju_f_)), Jp_g(J_g_functor(U, tt, V, Jp_f_)),
       S_inv_r(S_inv_r_functor({L, g, b})), K(V_.shape()[0]), mp1(U.shape()[0]), D(U.shape()[1])  {
    }

    xt::xtensor<double, 2> operator()(const std::vector<double> &p) const {
        xt::xtensor<double, 1> J_wnn_eval = xt::zeros<double>({p.size()});
        // Precomputions
        const auto Lp = L(p);
        const auto S = xt::linalg::dot(Lp, xt::transpose(Lp));
        const auto S_inv_rp = S_inv_r(p);
        const auto r = g(p) - b;

        // Precomputed partial information w.r.t p⃗ and U⃗
        const auto  JU_gp = xt::reshape_view(JU_g(p), {K*D, D*mp1}); // ∇ᵤg(p) ∈ ℝ^(K*D x D*mp1)
        const auto Jp_gp = xt::reshape_view( xt::sum(Jp_g(p),{3}),{K*D,D}); // ∇ₚg(p) ∈ ℝ^(K*D x D)

        // Precomputed mixed partials w.r.t p⃗ and U⃗

        for (int i = 0; i < p.size(); ++i) {
            // 1

            // 2

            // 3
        }

        return(V_prime);
    }
};

#endif //MLE_H

