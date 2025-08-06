#pragma once
#include <wendy/wnll.h>
#include <ceres/ceres.h>

class FirstOrderCostFunction final : public ceres::FirstOrderFunction {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit FirstOrderCostFunction(const WNLL& cost_)
        : cost(cost_) {}

    int NumParameters() const override {
        return static_cast<int>(cost.J);
    }

    bool Evaluate(const double* x, double* f, double* grad) const override {
        const std::vector<double> p(x, x + NumParameters());
        *f = cost(p);

        if (grad) {
            const std::vector<double> g = cost.Jacobian(p);
            for (int i = 0; i < NumParameters(); ++i) {
                grad[i] = g[i];
            }
        }

        return true;
    }

private:
    const WNLL& cost;
};