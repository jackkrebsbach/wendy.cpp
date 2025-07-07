#ifndef WEAK_RESIDUAL_EQUATIONS_H
#define WEAK_RESIDUAL_EQUATIONS_H
#include <xtensor/containers/xarray.hpp>


// Since we are passing the data with pointers abstracting into functions will not hurt performance

// g(p) = vec[Phi F(p,U,t)]

// ∇g_u(p) gradient of g w.r.t the vectorized data

// S^-1(p)

// S^-1(p)(g(p)-b) = S^-1(p)r(p)

// r(p) = g(p) - b

#endif //WEAK_RESIDUAL_EQUATIONS_H

