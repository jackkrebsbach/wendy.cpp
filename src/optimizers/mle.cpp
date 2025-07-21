#include "mle.h"

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
    const auto wnll = logdetS + quad;
    return (wnll);
}

xt::xtensor<double, 1> MLE::Jacobian(const std::vector<double> &p) const {
    // Precomputions
    const auto Lp = L(p); // L(p)
    const auto Jp_Lp = L.Jacobian(p); // ∇ₚL(p)
    const auto S = xt::linalg::dot(Lp, xt::transpose(Lp)); // S(p) (Covariance)

    const auto r = g(p) - b; // r(p) = g(p) - b
    const auto S_inv_rp = S_inv_r(p); // S^(-1)r(p)

    // Precomputed partial information of g(p) w.r.t p⃗ and U⃗
    const auto JU_gp = xt::reshape_view(JU_g(p), {K * D, D * mp1}); // ∇ᵤg(p) ∈ ℝ^(K*D x D*mp1)
    const auto Jp_gp = xt::reshape_view(Jp_g(p), {K * D, D}); // ∇ₚg(p) ∈ ℝ^(K*D x D)

    // Precomputed ∇ₚS(p) gradient of the covariance matrix, 3D Tensor
    const auto Jp_LLT = xt::transpose(xt::linalg::tensordot(Jp_Lp, xt::transpose(Lp), {1},{0}),{0,2,1}); //∇ₚLLᵀ
    const auto Jp_Sp = Jp_LLT + xt::transpose(Jp_LLT, {1, 0, 2}); // ∇ₚS(p) = ∇ₚLLᵀ + (∇ₚLLᵀ)ᵀ 3D tensor


    // Output
    xt::xtensor<double, 1> J_wnn = xt::zeros<double>({p.size()});
    for (int i = 0; i < p.size(); ++i) {
        // Extract partial information for each p_i from the gradients
        const auto Jp_Sp_i = xt::view(Jp_Sp, xt::all(), xt::all(), i);
        const auto Jp_gp_i = xt::view(Jp_gp, xt::all(), i);
        const auto JU_gp_i = xt::view(JU_gp, xt::all(), i);

        // Compute  ∇ₚS(p)^(-1) = -S(p)^(-1)∇ₚS(p)S(p)^(-1) from L and  ∇ₚS(p)
        const auto Y = xt::linalg::solve(S, Jp_Sp_i); // S^(-1)𝜕pᵢS(p)
        const auto Xt = xt::linalg::solve(xt::transpose(S), xt::transpose(Y));
        const auto Jp_Sp_inv = -1 * xt::transpose(Xt);

        const double prt1 = 0.5 * xt::linalg::trace(Y)();
        const double prt2 = 2 * xt::linalg::dot(xt::eval(xt::transpose(Jp_gp_i)), S_inv_rp)();
        const double prt3 = xt::linalg::dot(xt::linalg::dot(xt::transpose(r), Jp_Sp_inv), r)();

        J_wnn[i] = prt1 + prt2 + prt3;
    }

    return (J_wnn);
}

xt::xtensor<double, 2> MLE::Hessian(const std::vector<double> &p) const {
    // Precomputions
    const auto Lp = L(p); // L(p)
    const auto Jp_Lp = L.Jacobian(p); // ∇ₚL(p)
    const auto Hp_Lp = L.Hessian(p); // ∇ₚ∇ₚL(p)

    const auto r = g(p) - b; // r(p) = g(p) - b
    const auto S_inv_rp = S_inv_r(p); // S^(-1)r(p)

    // Precomputed partial information of g(p) w.r.t p⃗ and U⃗
    const auto JU_gp = xt::reshape_view(JU_g(p), {K * D, D * mp1}); // ∇ᵤg(p) ∈ ℝ^(K*D x D*mp1)
    const auto Jp_gp = xt::reshape_view(xt::sum(Jp_g(p), {3}), {K * D, D}); // ∇ₚg(p) ∈ ℝ^(K*D x D)

    // Precomputed Hessian information of ∇p∇pg(p) w.r.t p⃗ and U⃗
    const auto Hp_gp = xt::reshape_view( xt::sum(Jp_Jp_g(p), {3}), {K*D, D, J});

    // Precompute S(p) = LLᵀ
    const auto Sp = xt::linalg::dot(Lp, xt::transpose(Lp)); // S(p) (Covariance)

    // Precomputed ∇ₚS(p) gradient of the covariance matrix, 3D Tensor
    const auto Jp_LLT = xt::transpose(xt::linalg::tensordot(Jp_Lp, xt::transpose(Lp), {1},{0}),{0,2,1}); //∇ₚLLᵀ
    const auto Jp_Sp = Jp_LLT + xt::transpose(Jp_LLT, {1, 0, 2}); // ∇ₚS(p) = ∇ₚLLᵀ + (∇ₚLLᵀ)ᵀ 3D tensor


    // Precomputed first part of HₚS(p) Hessian of the covariance matrix, 4D Tensor
    // ∇ₚ∇ₚS(p) = ∇ₚ∇ₚLLᵀ + ∇ₚL∇ₚLᵀ + (∇ₚ∇ₚLLᵀ + ∇ₚL∇ₚLᵀ)ᵀ 4D tensor where Lᵀ is broadcasted
    const auto Hp_LLT = xt::transpose(xt::linalg::tensordot(Hp_Lp, xt::transpose(Lp), {1},{0}),{0,3,1,2}); //∇ₚ∇ₚLLᵀ
    // ∇ₚL∇ₚLᵀ
    auto Jp_LpT = xt::transpose(Jp_Lp, {1, 0, 2});
    auto Jp_Lp_Jp_LpT_ = xt::linalg::tensordot(Jp_Lp, Jp_LpT, {1}, {0});
    auto Jp_Lp_Jp_LpT = xt::transpose(Jp_Lp_Jp_LpT_, {0, 2, 1, 3});

    // Output
    xt::xtensor<double, 2> H_wnn = xt::zeros<double>({p.size(), p.size()});
    for (int i = 0; i < p.size(); ++i) {
        // Extract partial information for each p_i from the gradients
        const auto Jp_Sp_i = xt::view(Jp_Sp, xt::all(), xt::all(), i);
        const auto Jp_gp_i = xt::view(Jp_gp, xt::all(), i);
        const auto JU_gp_i = xt::view(JU_gp, xt::all(), i);

        // Compute  ∇ₚS(p)^(-1) = -S(p)^(-1)∇ₚS(p)S(p)^(-1) from L and  ∇ₚS(p)
        const auto Y = xt::linalg::solve(Sp, Jp_Sp_i); // S^(-1)𝜕pᵢS(p)
        const auto Xt = xt::linalg::solve(xt::transpose(Sp), xt::transpose(Y));
        const auto Jp_Sp_inv_i = -1 * xt::transpose(Xt);

        const auto Jp_Lp_i = xt::view(Jp_Lp, xt::all(), xt::all(), i);
        for (int j = 0; j < p.size(); ++j) {
            // 𝜕ⱼS(p) (Jacobian information)
            const auto Jp_Sp_j = xt::view(Jp_Sp, xt::all(), xt::all(), j);
            // 𝜕ⱼ g(p) (Jacobian information)
            const auto Jp_gp_j = xt::view(Jp_gp, xt::all(), j);

            // 𝜕ₚ𝜕ₚS(p) (Hessian information)
            const auto Hp_LLT_ji = xt::view(Hp_LLT, xt::all(), xt::all(), i, j);
            const auto Jp_Lp_Jp_LpT_ji  = xt::view(Jp_Lp_Jp_LpT,xt::all(), xt::all(), i, j);
            const auto H_ji =  Hp_LLT_ji + Jp_Lp_Jp_LpT_ji;            //𝜕ₚ𝜕ₚLLᵀ + 𝜕ₚL𝜕ₚLᵀ
            const auto Hp_S_ji = H_ji  + xt::transpose(H_ji);       // 𝜕ₚ𝜕ₚS(p)

            // 𝜕ₚ𝜕ₚ g(p) (Hessian information)
            const auto Hp_gp_ji = xt::view(Hp_gp, xt::all(), xt::all(), i, j);

            //𝜕ₚ𝜕ₚS(p)^-1
            //prt1
            const auto S_inv_Jp_Spj = xt::linalg::solve(Sp,Jp_Sp_j);
            const auto prt1_inv = xt::linalg::dot(S_inv_Jp_Spj,-1*Jp_Sp_inv_i);
            //prt2
            const auto Y2 = xt::linalg::solve(Sp, Hp_S_ji); // S^(-1)𝜕pᵢS(p)
            const auto Xt2 = xt::linalg::solve(Sp, xt::transpose(Y2));
            const auto prt2_inv = -1 * xt::transpose(Xt2);
            // prt3
            const auto x = xt::transpose( xt::linalg::solve(Sp, xt::transpose(S_inv_Jp_Spj)));
            const auto  prt3_inv = xt::linalg::dot(Sp, xt::linalg::dot(Jp_Sp_i, x));
            const auto Jp_Jp_S_inv_ji = prt1_inv + prt2_inv +prt3_inv;


            // 𝜕ₚ𝜕ₚ l(p) Weak negative log likelihood
            //prt1
            const auto x1 = xt::linalg::solve(Jp_Sp_j, Jp_Sp_i);
            const auto y1 = xt::linalg::solve(Sp, Hp_S_ji);
            const auto prt1 = 0.5*(xt::linalg::trace(x1 + y1)());
            //prt2
            const auto prt2 = xt::linalg::dot(xt::transpose(Hp_gp_ji),S_inv_rp)();
            //prt3
            const auto prt3 =  xt::linalg::dot(xt::transpose(Jp_gp_i), xt::linalg::solve(Jp_Sp_j, r))();
            //prt4
            const auto prt4 = xt::linalg::dot( xt::transpose(Jp_gp_i), xt::linalg::solve(Sp, Jp_gp_j))();
            //prt5
            const auto prt5  = xt::linalg::dot(xt::linalg::dot( xt::transpose(Jp_gp_j), Jp_Sp_inv_i),r)();
            //prt6
            const auto prt6 = 0.5*(xt::linalg::dot(xt::linalg::dot(xt::transpose(r), Jp_Jp_S_inv_ji), r)());

            H_wnn(i,j)= prt1 + prt2 + prt3 + prt4 + prt5 + prt6;

        }
    }
    return H_wnn;
}
