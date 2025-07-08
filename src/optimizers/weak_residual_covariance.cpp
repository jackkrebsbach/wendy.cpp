#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <symengine/lambda_double.h>

// Calculate the covariance S(p,U,t) = (∇g + ϕ'∘I)(Σ∘I)(∇g + ϕ'∘I)^T = LL^T
// We can factor Σ∘I =(Σ∘I)^1/2(Σ∘I)^1/2 because it is diagonal and symmetric positive definite.
// returns covariance S(p,U,t)
xt::xtensor<double, 2> covariance(
    std::vector<double>& p, // parameters of the system
    xt::xtensor<double, 2>& U, // state data
    xt::xtensor<double, 1>& tt, // time stamps
    xt::xtensor<double, 2>& V, // test function matrix
    std::vector<std::vector<SymEngine::LambdaRealDoubleVisitor>>& J_f_u // Jacobian of f w.r.t the state variables
    ) {
    //Build L where LL^T = S
    // ∇ᵤg gradient of g with respect to the state

    // ϕ'∘I

    // (Σ∘I)^1/2

    return xt::zeros<xt::xtensor<double, 2>>(dx);
}