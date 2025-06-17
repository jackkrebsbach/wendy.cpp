#include "wendy.h"
#include "symbolic_utils.h"
#include <symengine/expression.h>
#include <xtensor/containers/xarray.hpp>

Wendy::Wendy(std::vector<std::string> f, xt::xarray<double> U, std::vector<float> p0) {
  if (U.dimension() != 2) {
    throw std::invalid_argument("U must be 2-dimensional");
  }

  J = p0.size(); // Number of parameters in the system
  D = U.shape()[1]; // Dimension of the system

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
