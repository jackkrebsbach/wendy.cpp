#ifndef WEAK_RESIDUAL_COVARIANCE_H
#define WEAK_RESIDUAL_COVARIANCE_H

#include <xtensor/containers/xtensor.hpp>

template <typename F>
xt::xtensor<double, 2> covariance(
    std::vector<double>& p, // parameters of the system
    xt::xtensor<double, 2>& U, // state data
    xt::xtensor<double, 1>& tt, // equispaced time stamps
    xt::xtensor<double, 2>& V, // test function matrix
     F& Ju_f // Jacobian of f w.r.t the state variables: Jᵤf(p,u,t)
    );


#endif //WEAK_RESIDUAL_COVARIANCE_H
