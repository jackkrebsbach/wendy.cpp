#include "wendy.h"
#include "logger.h"
#include "symbolic_utils.h"
#include <symengine/expression.h>
#include <xtensor/containers/xarray.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <fmt/ranges.h>


Wendy::Wendy(const std::vector<std::string> &f, const xt::xarray<double> &U, const std::vector<float> &p0) {
    if (U.dimension() != 2) {
        throw std::invalid_argument("U must be 2-dimensional");
    }

    J = p0.size(); // Number of parameters in the system
    D = U.shape()[1]; // Dimension of the system

    sym_system = create_symbolic_system(f);

    auto p_symbols = create_symbolic_vars("p", J);

    std::vector<std::string> p_symbol_strs;
    p_symbol_strs.reserve(p_symbols.size());
    for (const auto &sym: p_symbols) {
        p_symbol_strs.push_back(str(sym));
    }

    logger->debug("p_symbols: [{}]", fmt::join(p_symbol_strs, ", "));

    const auto grad_p_f = compute_jacobian(sym_system, p_symbols);

    sym_system_jac = grad_p_f;
}

void Wendy::log_details() const {

    logger->info("Wendy class details:");
    logger->info("  D (Number of state variables): {}", D);
    logger->info("  J (Number of parameters): {}", J);
    logger->info("  min_radius: {}", min_radius);

    logger->info("  sym_system (Symbolic system expressions):");
    logger->info("    Size: {}", sym_system.size());
    for (size_t i = 0; i < sym_system.size(); ++i) {
        logger->info("      [{}]: {}", i, str(sym_system[i]));
    }

    logger->info("  sym_system_jac (Symbolic Jacobian):");
    logger->info("    Size: {}", sym_system_jac.size());
    for (size_t i = 0; i < sym_system_jac.size(); ++i) {
        std::string row;
        for (size_t j = 0; j < sym_system_jac[i].size(); ++j) {
            row += str(sym_system_jac[i][j]);
            if (j < sym_system_jac[i].size() - 1)
                row += ", ";
        }
        logger->info("      Row {} (size {}): {}", i, sym_system_jac[i].size(), row);
    }
}
