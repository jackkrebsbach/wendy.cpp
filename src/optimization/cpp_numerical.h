#pragma once
#include "../objective/mle.h"
#include <cppoptlib/function.h>
#include <cppoptlib/solver/lbfgs.h>
#include <cppoptlib/solver/newton_descent.h>

class MleProblem final : public cppoptlib::function::FunctionCRTP<MleProblem, double, cppoptlib::function::DifferentiabilityMode::Second> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    const MLE mle;

    explicit MleProblem(
        const MLE &mle_
    ) : mle(mle_) {}

    ScalarType operator()(const VectorType &x) const {
        return this->operator()(x, nullptr);
    }
    ScalarType operator()(const VectorType &x, VectorType *grad) const {
        return this->operator()(x, grad, nullptr);
    }
    ScalarType operator()(const VectorType &x, VectorType *grad, MatrixType *hessian) const {
        const std::vector<double> p(x.data(), x.data() + x.size());

        if (grad) {
            grad->resize(x.size());
            // const auto g = mle.Jacobian(p);
            const auto g = gradient_4th_order(mle, p);

            for (Eigen::Index i = 0; i < x.size(); ++i){
                (*grad)(i) = g[i];
            }
        }
        if (hessian) {
            hessian->resize(x.size(), x.size());
            // const auto H = mle.Hessian(p);
            const auto H = hessian_3rd_order(mle, p);

            for (Eigen::Index i = 0; i < x.size(); ++i) {
                for (Eigen::Index j = 0; j < x.size(); ++j) {
                    (*hessian)(i, j) = H[i][j];
                }
            }
        }

        std::cout << "Objective at x = " << x.transpose() << " is " << mle(p) << std::endl;

        return mle(p);
    }
};
