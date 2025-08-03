#include "mle.h"
#include "../utils.h"

#include <numbers>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>


constexpr double DIAG_REG = 1e-9;

MLE::MLE(
    const xt::xtensor<double, 2> &U_,
    const xt::xtensor<double, 1> &tt_,
    const xt::xtensor<double, 2> &V_,
    const xt::xtensor<double, 2> &V_prime_,
    const CovarianceFactor &L_,
    const g_functor &g_,
    const xt::xtensor<double, 1> &b_,
    const J_f_functor &Jp_f_,
    const J_g_functor &Ju_g_,
    const J_g_functor &Jp_g_,
    const H_g_functor &Jp_Ju_g_,
    const H_g_functor &Jp_Jp_g_,
    const T_g_functor &Jp_Jp_Ju_g_

): L(L_), U(U_), tt(tt_), V(V_), V_prime(V_prime_), b(b_), g(g_),
   Jp_f(Jp_f_), Ju_g(Ju_g_),Jp_g(Jp_g_), Jp_Ju_g(Jp_Ju_g_), Jp_Jp_g(Jp_Jp_g_), Jp_Jp_Ju_g(Jp_Jp_Ju_g_),
    K(V_.shape()[0]), mp1(U.shape()[0]), D(U.shape()[1]) {
    J = Jp_Ju_g.grad2_len;
    constant_term = K * D * std::log(2 * std::numbers::pi);
}

void MLE::S_inv(xt::xarray<double> &out_, const xt::xtensor<double,2> &S, const xt::xarray<double> &R) {
    try {
        const auto C = xt::linalg::cholesky(S);
        out_ = solve_cholesky(C, R);
    } catch (const std::exception &e) {
        out_ = xt::linalg::solve(S,R);
    }
}

double MLE::operator()(const std::vector<double> &p) const {
    const auto S = L.Cov(p);
    const auto r = g(p) - b;
    double logDetS;
    try {
        // const auto C = xt::linalg::cholesky(S);
        // const auto diagC_eval = xt::eval(xt::diag(C));
        // const auto filtered_diag = xt::filter(diagC_eval, xt::abs(diagC_eval) > 1e-14);
        // logDetS = 2.0 * xt::sum(xt::log(filtered_diag))();
        const auto [U, s, Vt] = xt::linalg::svd(S, false);
        logDetS = xt::sum(xt::log(s))();
    } catch (const std::exception &e) {
        logDetS = std::log(xt::linalg::det(S));
    }

    xt::xarray<double> quad_; S_inv(quad_, S, r);
    const auto quad = xt::linalg::dot(r, quad_)();

    const auto wnll = 0.5 * (logDetS + quad + constant_term);

    return (wnll);
}

xt::xtensor<double,2> MLE::Jp_r(const std::vector<double> &p) const {

    xt::xtensor<double, 3> Jp_F({mp1, D, J});

    for (size_t i = 0; i < mp1; ++i) {
        const double &t = tt[i];
        const xt::xtensor<double,1> &u = xt::row(U, i);
        xt::view(Jp_F, i, xt::all(), xt::all()) = Jp_f(p, u, t);
    }

    xt::xtensor<double,4> Jp_r_ = xt::zeros<double>({K, mp1, D, J});

    for (int k = 0; k < K; ++k)
        for (int m = 0; m < mp1; ++m)
            for (int d1 = 0; d1 < D; ++d1)
                    for (int j = 0; j < J; ++j)
                        Jp_r_(k, m, d1, j) = V(k, m) * Jp_F(m, d1, j) ;

    const auto Jp_r = xt::reshape_view(xt::sum(Jp_r_, {1}), {K * D, J}); // ∇ₚr(p) ∈ ℝ^(K*D x J)

    return Jp_r;
}


std::vector<double> MLE::Jacobian(const std::vector<double> &p) const {
    const auto S = L.Cov(p);

    const auto Lp = L(p);
    const auto Jp_Lp = L.Jacobian(p);

    const auto r = g(p) - b;

    const auto Jp_rp = Jp_r(p);

    xt::xarray<double> S_inv_rp; S_inv(S_inv_rp, S, r);

    std::vector<double> J_wnn(p.size());

    for (int j = 0; j < p.size(); ++j) {

        const xt::xtensor<double, 2> Jp_L_j = xt::view(Jp_Lp, xt::all(), xt::all(), j);

        const auto Jp_ = xt::linalg::dot(Jp_L_j, xt::transpose(Lp));
        const auto Jp_S = Jp_ + xt::transpose(Jp_);

        const auto Jp_r_j = xt::view(Jp_rp, xt::all(), j);

        xt::xarray<double> _; S_inv(_, S, Jp_S);

        const double prt1 = xt::linalg::trace(_)();
        const double prt2 = 2.0 * xt::linalg::dot(xt::eval(xt::transpose(Jp_r_j)), S_inv_rp)();
        const double prt3 = -1.0 * xt::linalg::dot(xt::linalg::dot(xt::transpose(S_inv_rp), Jp_S), S_inv_rp)();

        J_wnn[j] = 0.5 * (prt1 + prt2 + prt3);
    }

    return (J_wnn);
}

std::vector<std::vector<double> > MLE::Hessian(const std::vector<double> &p) const {
    // Precomputations
    const auto Lp = L(p); // L(p)
    const auto Sp = L.Cov(p);
    const auto Jp_Lp = L.Jacobian(p); // ∇ₚL(p)
    const auto Hp_Lp = L.Hessian(p); // ∇ₚ∇ₚL(p)

    const auto r = g(p) - b; // r(p) = g(p) - b
    xt::xarray<double> S_inv_rp = xt::zeros<double>({K*D}); S_inv(S_inv_rp, Sp, r);

    // Precomputed partial information of g(p) w.r.t p⃗ and U⃗
    const auto Ju_gp = xt::reshape_view(Ju_g(p), {K * D, D * mp1}); // ∇ᵤg(p) ∈ ℝ^(K*D x D*mp1)
    const auto Jp_gp = xt::reshape_view(xt::sum(Jp_g(p), {2}), {K * D, J}); // ∇ₚg(p) ∈ ℝ^(K*D x J)

    // Precomputed Hessian information of ∇p∇pg(p) w.r.t p⃗
    const auto Hp_gp = xt::reshape_view(xt::sum(Jp_Jp_g(p), {2}), {K * D, J, J});

    std::vector<std::vector<double> > H_wnn(p.size(), std::vector<double>(p.size()));
    for (int j = 0; j < p.size(); ++j) {
        // Extract partial information for each p_i from the gradients
        const auto Jp_LLT_j = xt::linalg::dot(xt::eval(xt::view(Jp_Lp, xt::all(), xt::all(), j)),
                                              xt::eval(xt::transpose(Lp)));
        const auto Jp_Sp_j = Jp_LLT_j + xt::transpose(Jp_LLT_j);

        const auto Jp_gp_j = xt::view(Jp_gp, xt::all(), j);
        const auto Ju_gp_j = xt::view(Ju_gp, xt::all(), j);

        // Compute  S(p)^(-1)𝜕ⱼS(p)
        const auto X_j = xt::linalg::solve(Sp, Jp_Sp_j);
        // Compute  𝜕ⱼS(p)^(-1) = -S(p)^-1𝜕ⱼS(p)S(p)^-1
        const auto Jp_Sp_inv_j = -1.0 * xt::linalg::solve(Sp, X_j);

        const auto Jp_Lp_j = xt::view(Jp_Lp, xt::all(), xt::all(), j);

        for (int i = j; i < p.size(); ++i) {
            // 𝜕ᵢS(p) (Jacobian information)
            const auto Jp_LLT_i = xt::linalg::dot(xt::eval(xt::view(Jp_Lp, xt::all(), xt::all(), i)),
                                                  xt::eval(xt::transpose(Lp)));
            const auto Jp_Sp_i = Jp_LLT_i + xt::transpose(Jp_LLT_i);

            // Compute  S(p)^(-1)𝜕ᵢS(p)
            const auto X_i = xt::linalg::solve(Sp, Jp_Sp_i);

            // Compute  𝜕ᵢS(p)^(-1) = -S(p)^-1𝜕ᵢS(p)S(p)^-1
            const auto Jp_Sp_inv_i = -1.0 * xt::linalg::solve(Sp, X_i);

            // 𝜕ᵢ g(p) (Jacobian information)
            const auto Jp_gp_i = xt::view(Jp_gp, xt::all(), i);

            // 𝜕ᵢ L(p) (Jacobian information)
            auto Jp_Lp_i = xt::view(Jp_Lp, xt::all(), xt::all(), i);

            // 𝜕ᵢ𝜕ⱼS(p) (Hessian information)
            const auto Hp_ijL_LT = xt::linalg::dot(xt::eval(xt::view(Hp_Lp, xt::all(), xt::all(), j, i)),
                                                   xt::transpose(Lp));
            const auto Jp_Lp_j_Jp_Lp_iT = xt::linalg::dot(xt::eval(Jp_Lp_j), xt::transpose(xt::eval(Jp_Lp_i)));

            const auto H_ij = Hp_ijL_LT + Jp_Lp_j_Jp_Lp_iT; //𝜕ᵢ𝜕ⱼLLᵀ + 𝜕ⱼL𝜕ᵢLᵀ
            const auto Hp_S_ij = H_ij + xt::transpose(xt::eval(H_ij)); // 𝜕ᵢ𝜕ⱼS(p)
            // 𝜕ᵢ𝜕ⱼ g(p) (Hessian information)
            const auto Hp_gp_ij = xt::view(Hp_gp, xt::all(), xt::all(), j, i);
            //𝜕ᵢ𝜕ⱼS(p)^-1
            //prt1
            const auto prt1_inv = xt::linalg::solve(Sp, xt::linalg::dot(-1.0 * Jp_Sp_inv_i, Jp_Sp_j));
            //prt2 S^(-1)𝜕ᵢ𝜕ⱼS(p)S^(-1)
            const auto prt2_inv = -1.0 * xt::linalg::solve(Sp, xt::linalg::solve(Sp, Hp_S_ij));
            // prt3
            const auto prt3_inv = xt::linalg::dot(Sp, xt::linalg::dot(Jp_Sp_j, -1.0 * Jp_Sp_inv_i));
            // adding terms together
            const auto Jp_Jp_S_inv_ij = prt1_inv + prt2_inv + prt3_inv;

            // 𝜕ᵢ𝜕ⱼ l(p) Weak negative log likelihood
            const auto x1 = xt::linalg::dot(Jp_Sp_inv_i, Jp_Sp_j);
            const auto y1 = xt::linalg::solve(Sp, Hp_S_ij);
            const auto prt1 = 0.5 * (xt::linalg::trace(x1)() + xt::linalg::trace(y1)());
            const auto prt2 = xt::linalg::dot(xt::transpose(Hp_gp_ij), S_inv_rp)();
            const auto prt3 = xt::linalg::dot(xt::transpose(Jp_gp_j), xt::linalg::dot(Jp_Sp_inv_i, r))();
            const auto prt4 = xt::linalg::dot(xt::transpose(Jp_gp_j), xt::linalg::solve(Sp, Jp_gp_i))();
            const auto prt5 = xt::linalg::dot(xt::linalg::dot(xt::transpose(Jp_gp_i), Jp_Sp_inv_j), r)();
            const auto prt6 = 0.5 * (xt::linalg::dot(xt::linalg::dot(xt::transpose(r), Jp_Jp_S_inv_ij), r)());

            const double Hij = prt1 + prt2 + prt3 + prt4 + prt5 + prt6;

            H_wnn[j][i] = Hij;

            if (i != j) H_wnn[i][j] = Hij; // symmetric fill
        }
    }
    return H_wnn;
}

