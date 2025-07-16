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
    const J_f_functor &Ju_f_
)
    : U(U_), tt(tt_), V(V_), V_prime(V_prime_),
      Sigma(Sigma_), JU_g(J_g_functor(U_, tt_, V_, Ju_f_)) {


    const auto mp1 = U.shape()[0];
    const auto D = U.shape()[1];
    const auto sqrt_Sigma = xt::linalg::cholesky(Sigma);// Precompute square root of Sigma (Sigma is diagonal)
    sqrt_Sigma_I_D = xt::linalg::kron(sqrt_Sigma, xt::eye(mp1));     // (ΣxI)^1/2 same as Cholesky factorization of Σ∘I because Σ is diagonal
    phi_prime_I_D = xt::linalg::kron(V_prime, xt::eye(D));      // ϕ'x I_d
}

xt::xtensor<double, 2> CovarianceFactor::operator()(
    const std::vector<double> &p
) const {
    const auto JU_gp = JU_g(p); // gradient information for a given set of parameters p
    assert(JU_gp.shape() == phi_prime_I_D.shape() && JU_gp.shape()[1] == sqrt_Sigma_I_D.shape()[0]);
    auto L = xt::linalg::dot((JU_gp + phi_prime_I_D), sqrt_Sigma_I_D);
    return (L);
};
