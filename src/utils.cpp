#include "utils.h"
#include "symbolic_utils.h"
#include <wendy/weak_residual.h>

#include <cmath>
#include <xtensor/misc/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <symengine/expression.h>
#include <symengine/lambda_double.h>

// The solve_cholesky and solve_triangular from xt is busted as of now
// https://github.com/xtensor-stack/xtensor-blas/issues/242
xt::xarray<double> solve_cholesky(const xt::xarray<double> &L, const xt::xarray<double> &B) {
    using namespace cxxlapack;

    xt::xarray<double, xt::layout_type::column_major> A = xt::eval(L);
    xt::xarray<double, xt::layout_type::column_major> X = B;

    const int n = static_cast<int>(L.shape()[0]);

    // Ensure B is 2D: reshape if it's 1D
    xt::xarray<double> B_2D = (B.dimension() == 1)
                                  ? xt::reshape_view(B, {n, 1})
                                  : B;

    const int nrhs = static_cast<int>(B_2D.shape()[1]);

    const int info = cxxlapack::potrs(
        'L', // Lower triangle
        n, // Matrix size
        nrhs, // Number of RHS columns
        A.data(), // Cholesky factor
        n, // Leading dimension of A
        X.data(), // RHS, overwritten with solution
        n // Leading dimension of B/X
    );

    if (info != 0) {
        throw std::runtime_error("Cholesky solve failed with info = " + std::to_string(info));
    }

    return X;
}

xt::xarray<double> cholesky_factor(const xt::xarray<double> &S) {
    using namespace cxxlapack;

    // Ensure input is square
    if (S.dimension() != 2 || S.shape()[0] != S.shape()[1]) {
        throw std::invalid_argument("Matrix must be square for Cholesky factorization.");
    }

    const int n = static_cast<int>(S.shape()[0]);

    // Work on a mutable copy with column-major layout
    xt::xarray<double, xt::layout_type::column_major> A = xt::eval(S);

    // Compute Cholesky factor in-place in A
    const int info = cxxlapack::potrf(
        'L', // Lower triangle
        n,   // Size of matrix
        A.data(), // Data pointer
        n    // Leading dimension
    );

    if (info != 0) {
        throw std::runtime_error("Cholesky factorization failed with info = " + std::to_string(info));
    }

    // Zero out upper triangle for clarity
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            A(i, j) = 0.0;

    return A;
}



QRFactor qr_factor(const xt::xarray<double> &A_in) {
    using namespace cxxlapack;

    auto A = xt::xarray<double, xt::layout_type::column_major>(A_in);
    int m = A.shape()[0], n = A.shape()[1];

    int lda = std::max(1, m);
    std::vector<double> tau(std::min(m, n));

    // Query optimal workspace
    double work_query;
    int lwork = -1;
    int info = geqrf(m, n, A.data(), lda, tau.data(), &work_query, lwork);
    if (info != 0) throw std::runtime_error("geqrf workspace query failed");

    lwork = static_cast<int>(work_query);
    std::vector<double> work(lwork);

    // Actual QR factorization
    info = geqrf(m, n, A.data(), lda, tau.data(), work.data(), lwork);
    if (info != 0) throw std::runtime_error("QR factorization failed");

    return {std::move(A), std::move(tau), m, n};
}

xt::xarray<double> solve_qr(const QRFactor &F, const xt::xarray<double> &B_in) {
    using namespace cxxlapack;

    auto B = xt::xarray<double, xt::layout_type::column_major>(B_in);
    bool was_vector = (B.dimension() == 1);
    if (was_vector)
        B = xt::reshape_view(B, {F.m, 1});

    int nrhs = B.shape()[1];
    int ldb = std::max(1, F.m);

    // Apply Qᵀ to B: B := Qᵀ * B
    double work_query;
    int lwork = -1;

    int info = ormqr(
        'L', 'T',
        F.m, nrhs, std::min(F.m, F.n),
        const_cast<double *>(F.A_fact.data()), F.m,
        F.tau.data(),
        B.data(), ldb,
        &work_query, lwork
    );

    if (info != 0) throw std::runtime_error("ormqr workspace query failed");

    lwork = static_cast<int>(work_query);
    std::vector<double> work(lwork);

    info = ormqr(
        'L', 'T',
        F.m, nrhs, std::min(F.m, F.n),
        const_cast<double *>(F.A_fact.data()), F.m,
        F.tau.data(),
        B.data(), ldb,
        work.data(), lwork
    );

    if (info != 0) throw std::runtime_error("ormqr failed");

    // Solve R * X = Qᵀ * B
    info = trtrs('U', 'N', 'N', F.n, nrhs, F.A_fact.data(), F.m, B.data(), ldb);
    if (info != 0) throw std::runtime_error("trtrs failed");

    auto X = xt::view(B, xt::range(0, F.n), xt::all());

    if (was_vector)
        return xt::reshape_view(X, {F.n});
    return X;
}

std::function<double(double)>
make_scalar_function(const SymEngine::Expression &expr, const SymEngine::RCP<const SymEngine::Symbol> &var) {
    return [expr, var](const double x) {
        SymEngine::map_basic_basic subs;
        subs[var] = SymEngine::real_double(x);
        const SymEngine::Expression substituted = expr.subs(subs);
        return SymEngine::eval_double(*substituted.get_basic());
    };
}

f_functor build_f(const std::vector<SymEngine::Expression> &f_symbolic, const size_t D, const size_t J) {
    const auto dx = build_f_visitors(f_symbolic, D, J); // Symengine object to call numerical input
    return {dx, D};
}

J_f_functor build_J_f(const std::vector<std::vector<SymEngine::Expression> > &J_f_symbolic, const size_t D,
                      const size_t J) {
    const auto dx = build_jacobian_visitors(J_f_symbolic, D, J); // Symengine object to call numerical input
    return J_f_functor(dx);
}

H_f_functor build_H_f(const std::vector<std::vector<std::vector<SymEngine::Expression> > > &H_f_symbolic,
                      const size_t D,
                      const size_t J) {
    const auto dx = build_jacobian_visitors(H_f_symbolic, D, J); // Symengine object to call numerical input
    return H_f_functor(dx);
}

T_f_functor build_T_f(const std::vector<std::vector<std::vector<std::vector<SymEngine::Expression> > > > &T_f_symbolic,
                      const size_t D,
                      const size_t J) {
    const auto dx = build_jacobian_visitors(T_f_symbolic, D, J); // Symengine object to call numerical input
    return T_f_functor(dx);
}

size_t get_corner_index(const xt::xtensor<double, 1> &y, const xt::xtensor<double, 1> *xx_in) {
    const size_t N = y.size();
    const size_t M = N - 1;

    xt::xtensor<double, 1> E = xt::zeros<float>({N});

    xt::xtensor<double, 1> x;
    if (xx_in == nullptr) {
        x = xt::arange<double>(1, N + 1);
    } else {
        x = *xx_in;
    }

    for (size_t k = 1; k <= M - 1; ++k) {
        double x0 = x[0], xk = x[k], xM = x[M];
        double y0 = y[0], yk = y[k], yM = y[M];

        double slope1 = (yk - y0) / (xk - x0);
        double slope2 = (yM - yk) / (xM - xk);

        auto L1 = [&](double x_val) {
            return slope1 * (x_val - x0) + y0;
        };
        auto L2 = [&](double x_val) {
            return slope2 * (x_val - xk) + yk;
        };

        double sum1 = 0.0;
        for (size_t m = 0; m <= k; ++m) {
            const double err = (L1(x[m]) - y[m]) / y[m];
            sum1 += err * err;
        }

        double sum2 = 0.0;
        for (size_t m = k; m <= M; ++m) {
            const double err = (L2(x[m]) - y[m]) / y[m];
            sum2 += err * err;
        }

        E[k] = std::sqrt(sum1 + sum2);
    }

    constexpr double INF_APPROX = 1e300;
    E[0] = INF_APPROX;
    E[E.size() - 1] = INF_APPROX;

    return xt::argmin(E)();
}
