#ifndef WENDY_H
#define WENDY_H

#include <xtensor/containers/xarray.hpp>
#include <symengine/expression.h>
#include <symengine/lambda_double.h>

/**
 * @brief Weak form estimation of nonlinear dynamics
 */
class Wendy {
public:
    size_t D;
    size_t J;
    double min_radius = 2;
    std::vector<SymEngine::Expression> sym_system;
    std::vector<std::vector<SymEngine::Expression> > sym_system_jac;

    Wendy(const std::vector<std::string> &f, const xt::xarray<double> &U, const std::vector<float> &p0);

    void log_details() const;
};

#endif // WENDY_H
