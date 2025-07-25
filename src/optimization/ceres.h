#pragma once
#include "../objective/mle.h"
#include <cppoptlib/function.h>
#include <cppoptlib/solver/lbfgs.h>
#include <cppoptlib/solver/newton_descent.h>

class MyMLEProblem final : public cppoptlib::function::FunctionCRTP<MyMLEProblem, double, cppoptlib::function::DifferentiabilityMode::First> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    const MLE &mle;

    explicit MyMLEProblem(
        const MLE &mle_
    ) : mle(mle_) {}

    ScalarType operator()(const VectorType &x) const {
        return this->operator()(x, nullptr);
    }
    // ScalarType operator()(const VectorType &x, VectorType *grad) const {
    //     return this->operator()(x, grad, nullptr);
    // }
    ScalarType operator()(const VectorType &x, VectorType *grad) const {
        const std::vector<double> p(x.data(), x.data() + x.size());

        if (grad) {
            grad->resize(x.size());
            const auto g = mle.Jacobian(p);
            for (Eigen::Index i = 0; i < x.size(); ++i){
                (*grad)(i) = g[i];
            }
        }

        // if (hessian) {
        //     hessian->resize(x.size(), x.size());
        //     const auto H = mle.Hessian(p);
        //     for (Eigen::Index i = 0; i < H.size(); ++i) {
        //         for (Eigen::Index j = 0; j < x.size(); ++j) {
        //             (*hessian)(i, j) = H[i][j];
        //         }
        //     }
        // }

        return mle(p);
    }
};
