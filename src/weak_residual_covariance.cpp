#include "utils.h"
#include "logger.h"
#include "weak_residual_covariance.h"
#include "weak_residual.h"
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>


/**
 * Calculate the covariancefactor L for S(p,U,t) = (∇g + ϕ'∘I)(Σ∘I)(∇gᵀ + ϕ'ᵀ∘I) = LL^T
 * We can factor Σ∘I =(Σ∘I)^1/2(Σ∘I)^1/2 because it is symmetric positive definite (it is also diagonal).
 * callable covariance of L where LLᵀ=S(p) : the data, U and t, are constant.
 **/

CovarianceFactor::CovarianceFactor(
    const xt::xtensor<double, 2> &U_,
    const xt::xtensor<double, 1> &tt_,
    const xt::xtensor<double, 2> &V_,
    const xt::xtensor<double, 2> &V_prime_,
    const xt::xtensor<double, 2> &Sigma_,
    const J_f_functor &Ju_f_,
    const H_f_functor &Jp_Ju_f_,
    const T_f_functor &Jp_Jp_Ju_f_
)
    : U(U_), tt(tt_), V(V_), V_prime(V_prime_),
      Sigma(Sigma_), JU_g(J_g_functor(U_, tt_, V_, Ju_f_)), Jp_JU_g(H_g_functor(U_, tt_, V_, Jp_Ju_f_)), Jp_Jp_JU_g(T_g_functor(U_, tt_, V_, Jp_Jp_Ju_f_)) {
    mp1 = U.shape()[0];
    D = U.shape()[1];
    K = V.shape()[0];
    J = Jp_JU_g.grad2_len;
    const auto sqrt_Sigma = xt::linalg::cholesky(Sigma); // Precompute square root of Sigma (Sigma is diagonal)
    sqrt_Sigma_I_mp1 = xt::linalg::kron(sqrt_Sigma, xt::eye(mp1));
    // (ΣxI)^1/2 same as Cholesky factorization of Σ∘I because Σ is diagonal
    phi_prime_I_D = xt::linalg::kron(V_prime, xt::eye(D)); // ϕ'x I_d
}

// L(p) where Covariance = S(p) = L(p)L(p)ᵀ
xt::xtensor<double, 2> CovarianceFactor::operator()(
    const std::vector<double> &p
) const {
    const auto JU_gp = xt::reshape_view(JU_g(p), {D * K, D * mp1 }); // (K*D, D*Mp1)

    assert(JU_gp.shape() == phi_prime_I_D.shape() && JU_gp.shape()[1] == sqrt_Sigma_I_mp1.shape()[0]);

    const auto L = xt::linalg::dot((JU_gp + phi_prime_I_D), sqrt_Sigma_I_mp1);
    return (L);
};

// ∇ₚL(p) gradient of the Covariance factor where ∇ₚS(p) = ∇ₚLLᵀ + (∇ₚLLᵀ)ᵀ
xt::xtensor<double, 2> CovarianceFactor::Jacobian(const std::vector<double> &p) const {
    const auto Jp_JU_gp = xt::reshape_view(Jp_JU_g(p), {D * K, D * mp1, J});
    const auto A = Jp_JU_gp + phi_prime_I_D; // shape: (D*K, D*mp1, J)
    const auto& B = sqrt_Sigma_I_mp1;        // shape: (D*mp1 D*mp1)
    const auto Jp_L_ = xt::linalg::tensordot(A, B, {1}, {0}); // shape (D*K, J, D*mp1)
    const auto Jp_L = xt::transpose(Jp_L_, {0, 2, 1}); // shape: (D*K, D*mp1, J)
    return (Jp_L);
};


// ∇ₚ∇ₚL(p) Hessain of the Covariance factor where
xt::xtensor<double, 3> CovarianceFactor::Hessian(const std::vector<double> &p) const {
    const auto Jp_Jp_JU_gp = xt::reshape_view(Jp_Jp_JU_g(p), {D * K, D * mp1, J, J});
    const auto A = Jp_Jp_JU_gp + phi_prime_I_D; // shape: (D*K, D*mp1, J, J)
    const auto& B = sqrt_Sigma_I_mp1;        // shape: (D*mp1 D*mp1)
    const auto Jp_H_ = xt::linalg::tensordot(A, B, {1}, {0}); // shape: (D*K,J, J, D*mp1)
    const auto Jp_H = xt::transpose(Jp_H_, {0, 3, 1, 2}); // shape: (D*K, D*mp1, J, J)
    return(Jp_H);
};


