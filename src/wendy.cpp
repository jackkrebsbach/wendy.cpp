#include "wendy.h"
#include "symbolic_utils.h"
#include <Rcpp.h>
#include <symengine/expression.h>

Wendy::Wendy(Rcpp::CharacterVector f_vec, NumericMatrix U, NumericVector p0) {
  J = p0.length(); // Number of parameters in the system
  D = U.cols();    // Dimension of the system

  // First need to convert the Rcpp::CharacterVector to the std vector
  std::vector<std::string> f(f_vec.size());
  for (int i = 0; i < f_vec.size(); ++i) {
    f[i] = Rcpp::as<std::string>(f_vec[i]);
  }
  sym_system = create_symbolic_system(f);

  auto p_symbols = create_symbolic_vars("p", J);
  auto u_symbols = create_symbolic_vars("u", D);

  auto grad_p_f = compute_jacobian(sym_system, p_symbols);

  sym_system_jac = grad_p_f;
}

void Wendy::log_details() const {
  std::cout << "Wendy class details:" << std::endl;
  std::cout << "  D (Number of state variables): " << D << std::endl;
  std::cout << "  J (Number of parameters): " << J << std::endl;
  std::cout << "  min_radius: " << min_radius << std::endl;

  std::cout << "  sym_system (Symbolic system expressions):" << std::endl;
  std::cout << "    Size: " << sym_system.size() << std::endl;
  for (size_t i = 0; i < sym_system.size(); ++i) {
    std::cout << "      [" << i << "]: " << sym_system[i] << std::endl;
  }

  std::cout << "  sym_system_jac (Symbolic Jacobian):" << std::endl;
  std::cout << "    Size: " << sym_system_jac.size() << std::endl;
  for (size_t i = 0; i < sym_system_jac.size(); ++i) {
    std::cout << "      Row " << i << " (size " << sym_system_jac[i].size()
              << "): ";
    for (size_t j = 0; j < sym_system_jac[i].size(); ++j) {
      std::cout << sym_system_jac[i][j];
      if (j < sym_system_jac[i].size() - 1)
        std::cout << ", ";
    }
    std::cout << std::endl;
  }
}
