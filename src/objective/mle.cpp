#include "mle.h"
#include "../utils.h"

#include <numbers>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>


constexpr double DIAG_REG = 1e-12;

MLE::MLE(
    const xt::xtensor<double, 2> &U_,
    const xt::xtensor<double, 1> &tt_,
    const xt::xtensor<double, 2> &V_,
    const xt::xtensor<double, 2> &V_prime_,
    const CovarianceFactor &L_,
    const g_functor &g_,
    const xt::xtensor<double, 1> &b_,
    const J_g_functor &Ju_g_,
    const J_g_functor &Jp_g_,
    const H_g_functor &Jp_Ju_g_,
    const H_g_functor &Jp_Jp_g_,
    const T_g_functor &Jp_Jp_Ju_g_,
    const S_inv_r_functor &S_inv_r_

): L(L_), U(U_), tt(tt_), V(V_), V_prime(V_prime_), b(b_), g(g_),
   Ju_g(Ju_g_), Jp_g(Jp_g_), Jp_Ju_g(Jp_Ju_g_), Jp_Jp_g(Jp_Jp_g_), Jp_Jp_Ju_g(Jp_Jp_Ju_g_),
   S_inv_r(S_inv_r_), K(V_.shape()[0]), mp1(U.shape()[0]), D(U.shape()[1]) {
    J = Jp_Ju_g.grad2_len;
    constant_term = static_cast<double>(K) * static_cast<double>(D) * std::log(2 * std::numbers::pi);
}

double MLE::operator()(const std::vector<double> &p) const {
    const auto Lp = L(p);
    auto S_ = xt::eval(xt::linalg::dot(Lp, xt::transpose(Lp)));
    constexpr double weight = 1.0 - DIAG_REG;
    const std::size_t n = S_.shape()[0];

    xt::xtensor<double, 2> S = xt::eval(weight * S_) + DIAG_REG * xt::eye<double>(n);
    const auto r = g(p) - b;

    double logDetS;
    xt::xtensor<double, 1> quad_;

    try {
        const auto C = xt::linalg::cholesky(S);
        const auto diagC = xt::diag(C);
        const auto filtered_diag = xt::filter(diagC, xt::not_equal(diagC, 0.0)); // optional safeguard
        logDetS = 2.0 * xt::sum(xt::log(filtered_diag))();
    } catch (const std::exception &e) {
        logDetS = std::log(xt::linalg::det(S));
    }

    quad_ = xt::linalg::solve(S, r);

    const auto quad = xt::linalg::dot(r, quad_)();
    const auto wnll = 0.5 * (logDetS + quad + constant_term);

    return (wnll);
}

std::vector<double> MLE::Jacobian(const std::vector<double> &p) const {
    const auto Lp = L(p); // L(p)
    const auto Jp_Lp = L.Jacobian(p); // ∇ₚL(p)

    auto S_ = xt::eval(xt::linalg::dot(Lp, xt::transpose(Lp)));

    constexpr double weight = 1.0 - DIAG_REG;
    const std::size_t n = S_.shape()[0];

    xt::xtensor<double, 2> S = xt::eval(weight * S_) + DIAG_REG * xt::eye<double>(n);

    const auto r = g(p) - b; // r(p) = g(p) - b
    const auto S_inv_rp = xt::linalg::solve(S, r);
    const auto Jp_gp = xt::reshape_view(xt::sum(Jp_g(p), {2}), {K * D, J}); // ∇ₚg(p) ∈ ℝ^(K*D x J)

    std::vector<double> J_wnn(p.size()); // Output
    for (int j = 0; j < p.size(); ++j) {
        // Extract partial information for each p_i from the gradients
        const auto Jp_LLT_j = xt::linalg::dot(xt::eval(xt::view(Jp_Lp, xt::all(), xt::all(), j)),
                                              xt::eval(xt::transpose(Lp)));
        const auto Jp_Sp_j = Jp_LLT_j + xt::transpose(Jp_LLT_j);
        const auto Jp_gp_j = xt::view(Jp_gp, xt::all(), j);

        const double prt1 = xt::linalg::trace(xt::linalg::solve(S, Jp_Sp_j))();
        const double prt2 = 2.0 * xt::linalg::dot(xt::eval(xt::transpose(Jp_gp_j)), S_inv_rp)();
        const double prt3 = -1.0 * xt::linalg::dot(xt::linalg::dot(xt::transpose(S_inv_rp), Jp_Sp_j), S_inv_rp)();

        J_wnn[j] = 0.5 * (prt1 + prt2 + prt3);
    }

    return (J_wnn);
}

std::vector<std::vector<double> > MLE::Hessian(const std::vector<double> &p) const {
    // Precomputations
    const auto Lp = L(p); // L(p)
    const auto Jp_Lp = L.Jacobian(p); // ∇ₚL(p)
    const auto Hp_Lp = L.Hessian(p); // ∇ₚ∇ₚL(p)

    const auto r = g(p) - b; // r(p) = g(p) - b
    const auto S_inv_rp = S_inv_r(p); // S^(-1)r(p)

    // Precomputed partial information of g(p) w.r.t p⃗ and U⃗
    const auto Ju_gp = xt::reshape_view(Ju_g(p), {K * D, D * mp1}); // ∇ᵤg(p) ∈ ℝ^(K*D x D*mp1)
    const auto Jp_gp = xt::reshape_view(xt::sum(Jp_g(p), {2}), {K * D, J}); // ∇ₚg(p) ∈ ℝ^(K*D x J)

    // Precomputed Hessian information of ∇p∇pg(p) w.r.t p⃗
    const auto Hp_gp = xt::reshape_view(xt::sum(Jp_Jp_g(p), {2}), {K * D, J, J});

    // Precompute S(p) = LLᵀ
    auto S_ = xt::eval(xt::linalg::dot(Lp, xt::transpose(Lp)));
    constexpr double diagReg = 1e-10;
    constexpr double weight = 1.0 - diagReg;
    const std::size_t n = S_.shape()[0];

    xt::xtensor<double, 2> Sp = xt::eval(weight * S_) + diagReg * xt::eye<double>(n);

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
