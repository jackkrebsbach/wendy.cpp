#ifndef IPOPT_H
#define IPOPT_H

#pragma once
#include <wendy/wnll.h>
#include <IpTNLP.hpp>
#include <Eigen/src/Core/util/Memory.h>

class IpoptCostFunction final : public Ipopt::TNLP {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit IpoptCostFunction(const WNLL &cost_)
        : cost(cost_), J(static_cast<Ipopt::Index>(cost_.J)) {
    }

    int NumParameters() const {
        return static_cast<int>(J);
    }

    std::vector<double> solution;

    bool get_nlp_info(Ipopt::Index &n, Ipopt::Index &m,
                      Ipopt::Index &nnz_jac_g, Ipopt::Index &nnz_h_lag,
                      Ipopt::TNLP::IndexStyleEnum &index_style) override {
        n = J; // Number of parameters
        m = 0; // No constraints
        nnz_jac_g = 0;
        nnz_h_lag = (J * (J + 1)) / 2;
        index_style = Ipopt::TNLP::C_STYLE;
        return true;
    }

    bool get_bounds_info(Ipopt::Index n, Ipopt::Number *x_l,
                         Ipopt::Number *x_u, Ipopt::Index /*m*/,
                         Ipopt::Number * /*g_l*/, Ipopt::Number * /*g_u*/) override {
        for (Ipopt::Index i = 0; i < n; ++i) {
            x_l[i] = -2e19;
            x_u[i] = +2e19;
        }
        return true;
    }

    bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number *x,
                            bool /*init_z*/, Ipopt::Number * /*z_L*/, Ipopt::Number * /*z_U*/,
                            Ipopt::Index /*m*/, bool /*init_lambda*/, Ipopt::Number * /*lambda*/) override {
        if (init_x) {
            for (Ipopt::Index i = 0; i < n; ++i) {
                x[i] = 1.0; // You can customize this with a better guess
            }
        }
        return true;
    }

    bool eval_f(Ipopt::Index n, const Ipopt::Number *x, bool /*new_x*/, Ipopt::Number &obj_value) override {
        std::vector<double> p(x, x + n);
        obj_value = cost(p);
        return true;
    }

    bool eval_grad_f(Ipopt::Index n, const Ipopt::Number *x, bool /*new_x*/, Ipopt::Number *grad_f) override {
        std::vector<double> p(x, x + n);
        const auto g = cost.Jacobian(p);
        for (Ipopt::Index i = 0; i < n; ++i) {
            grad_f[i] = g[i];
        }
        return true;
    }

    bool eval_h(Ipopt::Index n, const Ipopt::Number* x, bool /*new_x*/,
            Ipopt::Number obj_factor,
            Ipopt::Index /*m*/, const Ipopt::Number* /*lambda*/,
            bool /*new_lambda*/,
            Ipopt::Index /*nele_hess*/,
            Ipopt::Index* iRow, Ipopt::Index* jCol,
            Ipopt::Number* values) override
    {
        if (values == nullptr) {
            // Return sparsity structure: lower triangle of dense symmetric matrix
            Ipopt::Index idx = 0;
            for (Ipopt::Index i = 0; i < n; ++i) {
                for (Ipopt::Index j = 0; j <= i; ++j) {
                    iRow[idx] = i;
                    jCol[idx] = j;
                    ++idx;
                }
            }
        } else {
            // Return actual Hessian values (lower triangle only)
            std::vector<double> p(x, x + n);
            const auto H = cost.Hessian(p);  // std::vector<std::vector<double>> or equivalent

            Ipopt::Index idx = 0;
            for (Ipopt::Index i = 0; i < n; ++i) {
                for (Ipopt::Index j = 0; j <= i; ++j) {
                    values[idx++] = obj_factor * H[i][j];
                }
            }
        }

        return true;
    }

    bool eval_g(Ipopt::Index n, const Ipopt::Number *x, bool new_x,
                Ipopt::Index m, Ipopt::Number *g) override {
        // No constraints, so do nothing
        return true;
    }

    bool eval_jac_g(Ipopt::Index n, const Ipopt::Number *x, bool new_x,
                    Ipopt::Index m, Ipopt::Index nele_jac,
                    Ipopt::Index *iRow, Ipopt::Index *jCol,
                    Ipopt::Number *values) override {
        // No constraint Jacobian, so do nothing
        return true;
    }

    void finalize_solution(Ipopt::SolverReturn status,
                           Ipopt::Index n, const Ipopt::Number *x,
                           const Ipopt::Number * /*z_L*/, const Ipopt::Number * /*z_U*/,
                           Ipopt::Index /*m*/, const Ipopt::Number * /*g*/,
                           const Ipopt::Number * /*lambda*/,
                           Ipopt::Number obj_value,
                           const Ipopt::IpoptData * /*ip_data*/,
                           Ipopt::IpoptCalculatedQuantities * /*ip_cq*/) override {
        std::cout << "Ipopt finished with objective value: " << obj_value << "\n";
        for (Ipopt::Index i = 0; i < n; ++i) {
            std::cout << "x[" << i << "] = " << x[i] << "\n";
        }
    }

private:
    const WNLL &cost;
    const Ipopt::Index J;
};


#endif //IPOPT_H
