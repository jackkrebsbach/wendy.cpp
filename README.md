# wendy.cpp

>[!WARNING]
> This package is not ready for usage and is still under development.


Weak Form Estimation of Nonlinear Dynamics (WENDy) is an algorithm to estimate parameters of a system of Ordinary
Differential Equations (ODE).

# Development

## Dependencies

- [Xtensor](https://xtensor.readthedocs.io/en/latest/getting_started.html) tensor manipulation
- [Xtensor-blas](https://xtensor-blas.readthedocs.io/en/stable/index.html) linear algebra extension for xtensor (SVD, Cholesky etc)
- [SymEngine](https://github.com/symengine/symengine) for symbolic differentiation.
- [Spdlog](https://github.com/gabime/spdlog) for logging 

# TODO

- [x] Create symbolic representations of the right hand side of ODE system
- [x] Create function to solve for the min radius needed for test functions
- [ ] Build the weak log-likelihood of the residual, the gradient, and hessian w.r.t parameters using symbolic
  information
- [ ] Define all inputs needed to solve the main problem (estimate $`p^\star`$)
- [ ] Find trust region solver c++ library to use

## Future

- [ ] Build the G matrix efficiently when the RHS is Linear in Parameters.
- [ ] Lognormal noise
- [ ] Implement other solvers

