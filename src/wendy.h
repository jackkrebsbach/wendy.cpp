#ifndef WENDY_H
#define WENDY_H

#include <Rcpp.h>
#include <symengine/expression.h>
#include <symengine/lambda_double.h>

using namespace Rcpp;

/**
 * @brief Weak form estimation of nonlinear dynamics
 */
class Wendy {
public:
  /**
   * @brief Number of state variables (u1, ..., uD) aka dimension of the system
   */
  int D;

  /**
   * @brief Number of parameters (p1, ..., pJ).
   */
  int J;

  /**
   * @brief Minimum radius for the test functions (diagnostic & implementation
   * usage).
   */
  double min_radius;

  /**
   * @brief Symbolic system expressions (one per dimension of f the RHS of the
   * system).
   */
  std::vector<SymEngine::Expression> sym_system;

  /**
   * @brief Symbolic Jacobian of the system
   */
  std::vector<std::vector<SymEngine::Expression>> sym_system_jac;

  /**
   * @brief Constructor for Wendy.
   * @param f Character vector of system equations as strings (will be passed
   * from R)
   * @param U Numeric matrix of state values (noisy data).
   * @param p0 Numeric vector of initial parameter guess (p0)
   */
  Wendy(CharacterVector f, NumericMatrix U, NumericVector p0);

  /**
   *
   * @brief Logs all public member details to std::cout
   */
  void log_details() const;
};

#endif // WENDY_H
