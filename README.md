# wendy.cpp

Weak Form Estimation of Nonlinear Dynamics (WENDy) is an algorithm to estimate parameters of a system of Ordinary
Differential Equations (ODE).

# Development

## Dependencies

- [Xtensor](https://xtensor.readthedocs.io/en/latest/getting_started.html) tensor manipulation
- [SymEngine](https://github.com/symengine/symengine) for symbolic differentiation.
- [Spdlog](https://github.com/gabime/spdlog) for logging (future)

# TODO

- [x] Create symbolic representations of the right hand side of ODE system
- [ ] Create function to solve for the min radius needed for test functions
- [ ] Build the weak log-likelihood of the residual, the gradient, and hessian w.r.t parameters using symbolic
  information
- [ ] Define all inputs needed to solve the main problem (estimate $`p^\star`$)
- [ ] Find trust region solver c++ library to use
- [ ] Bundle symengine c++ library with Wendy instead of using a system level dependency.
- [ ] Probably want the c++ core code in a seperate repo and to use gitsubmodules.

## Future

- [ ] Build the G matrix efficiently when the RHS is Linear in Parameters.
- [ ] Lognormal noise
- [ ] Implement other solvers

