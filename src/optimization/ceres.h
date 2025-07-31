#pragma once
#include "../objective/mle.h"
#include <ceres/ceres.h>

class MleCeresCostFunction final : public ceres::CostFunction {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MleCeresCostFunction(const MLE& mle_, const std::vector<double>& p0)
        : parameter_dim(static_cast<int>(p0.size())), mle(mle_) {
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(parameter_dim);
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {

        const double* x_ptr = parameters[0];
        const std::vector p(x_ptr, x_ptr + parameter_dim);

        residuals[0] = mle(p);

        if (jacobians && jacobians[0]) {
            // const std::vector<double> g = gradient_4th_order(mle, p);
            const std::vector<double> g = mle.Jacobian(p);

            for (int i = 0; i < parameter_dim; ++i) {
                jacobians[0][i] = g[i];
            }
        }

        return true;
    }

    int parameter_dim;
    const MLE mle;
};
