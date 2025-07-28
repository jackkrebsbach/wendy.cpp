#include "utils.h"
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
    const J_g_functor &Ju_g_,
    const H_g_functor &Jp_Ju_g_,
    const T_g_functor &Jp_Jp_Ju_g_
)
    : U(U_), tt(tt_), V(V_), V_prime(V_prime_),
      Sigma(Sigma_), JU_g(Ju_g_), Jp_JU_g(Jp_Ju_g_), Jp_Jp_JU_g(Jp_Jp_Ju_g_) {
    mp1 = U.shape()[0];
    D = U.shape()[1];
    K = V.shape()[0];
    J = Jp_JU_g.grad2_len;

    Sigma_I_mp1 = xt::linalg::kron(Sigma, xt::eye(mp1));

    phi_prime_I_D = xt::linalg::kron( xt::eye(D), V_prime); // I_d x ϕ'
}

// L(p) where Covariance = S(p) = L(p)L(p)ᵀ
xt::xtensor<double, 2> CovarianceFactor::operator()(
    const std::vector<double> &p
) const {
    const auto JU_gp = xt::reshape_view(JU_g(p), {K * D, D * mp1});

    assert((JU_gp.shape() == phi_prime_I_D.shape()) && JU_gp.shape()[1] == Sigma_I_mp1.shape()[0]);

    const auto sum = xt::eval(JU_gp + phi_prime_I_D);

    const auto L = xt::linalg::dot(sum, Sigma_I_mp1);
    return (L);
};

// ∇ₚL(p) gradient of the Covariance factor where ∇ₚS(p) = ∇ₚLLᵀ + (∇ₚLLᵀ)ᵀ
xt::xtensor<double, 3> CovarianceFactor::Jacobian(const std::vector<double> &p) const {

    const auto Jp_JU_gp = xt::reshape_view(Jp_JU_g(p), {D * K, mp1 * D, J});
    const auto Jp_L_ = xt::linalg::tensordot(Jp_JU_gp, Sigma_I_mp1, {1}, {0}); // shape (D*K, J, D*mp1)
    const auto Jp_L = xt::transpose(Jp_L_, {0, 2, 1}); // shape: (D*K, D*mp1, J)

    return (Jp_L);
};


// ∇ₚ∇ₚL(p) Hessian of the Covariance factor where ∇ₚ∇ₚS(p) = ∇ₚ∇ₚLLᵀ + ∇ₚL∇ₚLᵀ + (∇ₚ∇ₚLLᵀ + ∇ₚL∇ₚLᵀ)ᵀ
xt::xtensor<double, 4> CovarianceFactor::Hessian(const std::vector<double> &p) const {
    const auto Jp_Jp_JU_gp = xt::reshape_view(Jp_Jp_JU_g(p), {D * K, D * mp1, J, J});
    const auto Jp_H_ = xt::linalg::tensordot(Jp_Jp_JU_gp, Sigma_I_mp1, {1}, {0}); // shape: (D*K,J, J, D*mp1)
    const auto Jp_H = xt::transpose(Jp_H_, {0, 3, 1, 2}); // shape: (D*K, D*mp1, J, J)
    return (Jp_H);
};


