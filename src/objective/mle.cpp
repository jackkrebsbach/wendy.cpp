#include "mle.h"
#include "../utils.h"

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>


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
    const H_g_functor &Jp_JU_g_,
    const H_g_functor &Jp_Jp_g_,
    const T_g_functor &Jp_Jp_JU_g_,
    const S_inv_r_functor &S_inv_r_

): L(L_), U(U_), tt(tt_), V(V_), V_prime(V_prime_), b(b_), g(g_),
   JU_g(Ju_g_), Jp_g(Jp_g_), Jp_JU_g(Jp_JU_g_), Jp_Jp_g(Jp_Jp_g_), Jp_Jp_JU_g(Jp_Jp_JU_g_),
   S_inv_r(S_inv_r_), K(V_.shape()[0]), mp1(U.shape()[0]), D(U.shape()[1]) {
    J = Jp_JU_g.grad2_len;
}

double MLE::operator()(const std::vector<double> &p) const {
    const auto Lp = L(p);
    const auto r = g(p) - b;
    const auto S = xt::linalg::dot(Lp, xt::transpose(Lp));
    const auto x = xt::linalg::solve(S, r); // Solve Sy = g(p) - b;
    const auto logdetS = std::log(xt::linalg::det(S));
    const auto quad = xt::linalg::dot(r, x)();
    const auto wnll = 0.5*(logdetS + quad);
    return (wnll);
}

std::vector<double> MLE::Jacobian(const std::vector<double> &p) const {
    // Precomputions
    const auto Lp = L(p); // L(p)
    const auto Jp_Lp = L.Jacobian(p); // ∇ₚL(p)
    const auto S = xt::linalg::dot(Lp, xt::transpose(Lp)); // S(p) (covariance)
    const auto F = xt::linalg::cholesky(S); // Cholesky factor of covariance

    const auto r = g(p) - b; // r(p) = g(p) - b
    const auto S_inv_rp = S_inv_r(p); // S^(-1)r(p)

    // Precomputed partial information of g(p) w.r.t p⃗ and U⃗
    const auto JU_gp = xt::reshape_view(JU_g(p), {K * D, D * mp1}); // ∇ᵤg(p) ∈ ℝ^(K*D x D*mp1)
    const auto Jp_gp = xt::reshape_view(xt::sum(Jp_g(p), {3}), {K * D, J}); // ∇ₚg(p) ∈ ℝ^(K*D x J)

    // Precomputed ∇ₚS(p) gradient of the covariance matrix, 3D Tensor
    const auto Jp_LLT = xt::transpose(xt::linalg::tensordot(Jp_Lp, xt::transpose(Lp), {1}, {0}), {0, 2, 1}); //∇ₚLLᵀ
    const auto Jp_Sp = Jp_LLT + xt::transpose(Jp_LLT, {1, 0, 2}); // ∇ₚS(p) = ∇ₚLLᵀ + (∇ₚLLᵀ)ᵀ 3D tensor

    // Output
    std::vector<double> J_wnn(p.size());
    for (int j = 0; j < p.size(); ++j) {
        // Extract partial information for each p_i from the gradients
        const auto Jp_Sp_j = xt::view(Jp_Sp, xt::all(), xt::all(), j);
        const auto Jp_gp_j = xt::view(Jp_gp, xt::all(), j);
        const auto JU_gp_j = xt::view(JU_gp, xt::all(), j);

        // Compute  S(p)^(-1)𝜕ⱼS(p)
        const auto X = solve_cholesky(F,Jp_Sp_j );
        // Compute  𝜕ⱼS(p)^(-1) = -S(p)^-1𝜕ⱼS(p)S(p)^-1 from L and S(p)^-1𝜕ⱼS(p)
        const auto Jp_Sp_inv= -1*xt::transpose(solve_cholesky(F, xt::transpose(X)));

        const double prt1 = 0.5 * xt::linalg::trace(X)();
        const double prt2 = xt::linalg::dot(xt::eval(xt::transpose(Jp_gp_j)), S_inv_rp)();
        const double prt3 = 0.5*xt::linalg::dot(xt::linalg::dot(xt::transpose(r), Jp_Sp_inv), r)();

        J_wnn[j] = prt1 + prt2 + prt3;
    }

    return (J_wnn);
}

std::vector<std::vector<double> > MLE::Hessian(const std::vector<double> &p) const {
    // Precomputions
    const auto Lp = L(p); // L(p)
    const auto Jp_Lp = L.Jacobian(p); // ∇ₚL(p)
    const auto Hp_Lp = L.Hessian(p); // ∇ₚ∇ₚL(p)

    const auto r = g(p) - b; // r(p) = g(p) - b
    const auto S_inv_rp = S_inv_r(p); // S^(-1)r(p)

    // Precomputed partial information of g(p) w.r.t p⃗ and U⃗
    const auto JU_gp = xt::reshape_view(JU_g(p), {K * D, D * mp1}); // ∇ᵤg(p) ∈ ℝ^(K*D x D*mp1)
    const auto Jp_gp = xt::reshape_view(xt::sum(Jp_g(p), {3}), {K * D, J}); // ∇ₚg(p) ∈ ℝ^(K*D x J)

    // Precomputed Hessian information of ∇p∇pg(p) w.r.t p⃗
    const auto Hp_gp = xt::reshape_view(xt::sum(Jp_Jp_g(p), {3}), {K * D, J, J});

    // Precompute S(p) = LLᵀ
    const auto Sp = xt::linalg::dot(Lp, xt::transpose(Lp)); // S(p) (Covariance)
    const auto F = xt::linalg::cholesky(Sp);

    // Precomputed ∇ₚS(p) gradient of the covariance matrix, 3D Tensor
    const auto Jp_LLT = xt::transpose(xt::linalg::tensordot(Jp_Lp, xt::transpose(Lp), {1}, {0}), {0, 2, 1}); //∇ₚLLᵀ
    const auto Jp_Sp = Jp_LLT + xt::transpose(Jp_LLT, {1, 0, 2}); // ∇ₚS(p) = ∇ₚLLᵀ + (∇ₚLLᵀ)ᵀ 3D tensor

    const auto Hp_LLT = xt::transpose(xt::linalg::tensordot(Hp_Lp, xt::transpose(Lp), {1}, {0}), {0, 3, 1, 2});

    // Output
    std::vector<std::vector<double> > H_wnn(p.size(), std::vector<double>(p.size()));
    for (int j = 0; j < p.size(); ++j) {
        // Extract partial information for each p_i from the gradients
        const auto Jp_Sp_j = xt::view(Jp_Sp, xt::all(), xt::all(), j);
        const auto Jp_gp_j = xt::view(Jp_gp, xt::all(), j);
        const auto JU_gp_j = xt::view(JU_gp, xt::all(), j);

        // Compute  S(p)^(-1)𝜕ⱼS(p)
        const auto X_j = solve_cholesky(F,Jp_Sp_j );

        // Compute  𝜕ⱼS(p)^(-1) = -S(p)^-1𝜕ⱼS(p)S(p)^-1 from L and S(p)^-1𝜕ⱼS(p)
        const auto Jp_Sp_inv_j= -1*xt::transpose(solve_cholesky(F, xt::transpose(X_j)));

        const auto Jp_Lp_j = xt::view(Jp_Lp, xt::all(), xt::all(), j);

        for (int i = 0; i < p.size(); ++i) {
            // 𝜕ⱼS(p) (Jacobian information)
            const auto Jp_Sp_i = xt::view(Jp_Sp, xt::all(), xt::all(), i);

            // Compute  S(p)^(-1)𝜕ᵢS(p)
            const auto X_i = solve_cholesky(F,Jp_Sp_i );

            // Compute  𝜕ᵢS(p)^(-1) = -S(p)^-1𝜕ᵢS(p)S(p)^-1 from L and S(p)^-1𝜕ᵢS(p)
            const auto Jp_Sp_inv_i= -1*xt::transpose(solve_cholesky(F, xt::transpose(X_i)));

            // 𝜕ᵢ g(p) (Jacobian information)
            const auto Jp_gp_i = xt::view(Jp_gp, xt::all(), i);

            // 𝜕ᵢ L(p) (Jacobian information)
            auto Jp_Lp_i = xt::view(Jp_Lp, xt::all(), xt::all(), i);

            // 𝜕ₚ𝜕ₚS(p) (Hessian information)
            const auto Hp_LLT_ij = xt::eval(xt::view(Hp_LLT, xt::all(), xt::all(), j, i));
            const auto Jp_Lp_Jp_LpT_ij = xt::linalg::dot(xt::eval(Jp_Lp_j), xt::transpose(xt::eval(Jp_Lp_i)));

            const auto H_ij = Hp_LLT_ij + Jp_Lp_Jp_LpT_ij; //𝜕ₚ𝜕ₚLLᵀ + 𝜕ₚL𝜕ₚLᵀ
            const auto Hp_S_ij = H_ij + xt::transpose(xt::eval(H_ij)); // 𝜕ₚ𝜕ₚS(p)

            // 𝜕ₚ𝜕ₚ g(p) (Hessian information)
            const auto Hp_gp_ij = xt::view(Hp_gp, xt::all(), xt::all(), j, i);

            //𝜕ₚ𝜕ₚS(p)^-1
            //prt1
            const auto prt1_inv = solve_cholesky(F,xt::linalg::dot( Jp_Sp_i, -1 * Jp_Sp_inv_j));
            //prt2 S^(-1)𝜕ᵢ𝜕ⱼS(p)S^(-1)
            const auto prt2_inv = -1*solve_cholesky(F, xt::transpose(solve_cholesky(F, xt::transpose(Hp_S_ij))));
            // prt3
            const auto prt3_inv = xt::linalg::dot(xt::linalg::dot(Sp, Jp_Sp_j), -1 * Jp_Sp_inv_i);
            // adding terms together
            const auto Jp_Jp_S_inv_ij = prt1_inv + prt2_inv + prt3_inv;

            // 𝜕ᵢ𝜕ⱼ l(p) Weak negative log likelihood
            const auto x1 = xt::linalg::dot(Jp_Sp_inv_i, Jp_Sp_j);
            const auto y1 = solve_cholesky(F, Hp_S_ij);
            const auto prt1 = 0.5 * (xt::linalg::trace(x1 + y1)());
            const auto prt2 = xt::linalg::dot(xt::transpose(Hp_gp_ij), S_inv_rp)();
            const auto prt3 = xt::linalg::dot(xt::transpose(Jp_gp_j), xt::linalg::dot(Jp_Sp_inv_i, r))();
            const auto prt4 = xt::linalg::dot(xt::transpose(Jp_gp_j), xt::linalg::solve(Sp, Jp_gp_i))();
            const auto prt5 = xt::linalg::dot(xt::linalg::dot(xt::transpose(Jp_gp_i), Jp_Sp_inv_j), r)();
            const auto prt6 = 0.5 * (xt::linalg::dot(xt::linalg::dot(xt::transpose(r), Jp_Jp_S_inv_ij), r)());

            H_wnn[i][j] = prt1 + prt2 + prt3 + prt4 + prt5 + prt6;
        }
    }
    return H_wnn;
}
