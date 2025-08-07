#pragma once
#include <wendy/wnll.h>
#include <ceres/ceres.h>

class FirstOrderCostFunction final : public ceres::FirstOrderFunction {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    bool opt_sigma;

    explicit FirstOrderCostFunction(const WNLL &cost_, const bool opt_sigma = false)
        : opt_sigma(opt_sigma), cost(cost_) {
    }

    int NumParameters() const override {
        return static_cast<int>(cost.J + (opt_sigma ? cost.D : 0));
    }

    bool Evaluate(const double *x, double *f, double *grad) const override {
        const std::vector<double> p(x, x + cost.J);
        const xt::xtensor<double, 1> sig = xt::adapt(
            x + cost.J,
            NumParameters() - cost.J, // size of the buffer
            xt::no_ownership(),
            std::vector<std::size_t>{NumParameters() - cost.J} // shape
        );

        *f = cost(p, sig);

        if (grad) {
            const std::vector<double> g = cost.Jacobian(p, sig);
            for (int i = 0; i < NumParameters(); ++i) {
                grad[i] = g[i];
            }
        }
        return true;
    }

private:
    const WNLL &cost;
};


struct WNLLCostFunction : public ceres::CostFunction {
    WNLL const &cost;

    WNLLCostFunction(WNLL const &cost_) : cost(cost_) {
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(cost.J + cost.D);
    }

    bool Evaluate(double const *const*parameters, double *residuals, double **jacobians) const override {
        const double *x = parameters[0];
        const std::vector<double> p(x, x + cost.J);
        const xt::xtensor<double, 1> sig = xt::adapt(
            x + cost.J,
            static_cast<std::size_t>(cost.D), // buffer size
            xt::no_ownership(),
            std::array<std::size_t, 1>{cost.D} // proper shape
        );


        residuals[0] = cost(p, sig);

        if (jacobians && jacobians[0]) {
            std::vector<double> grad = cost.Jacobian(p, sig);
            std::copy(grad.begin(), grad.end(), jacobians[0]);
        }

        return true;
    }
};
