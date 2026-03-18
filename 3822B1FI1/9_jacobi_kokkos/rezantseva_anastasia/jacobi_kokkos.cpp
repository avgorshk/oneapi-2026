#include "jacobi_kokkos.h"

#include <cmath>

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {
    using ExecSpace = Kokkos::SYCL;
    using MemSpace = Kokkos::SYCLDeviceUSMSpace;

    const int n = static_cast<int>(b.size());

    Kokkos::View<float**, Kokkos::LayoutLeft, MemSpace> d_a("A", n, n);
    Kokkos::View<float*, MemSpace> d_b("b", n);
    Kokkos::View<float*, MemSpace> inv_diag("inv_diag", n);
    Kokkos::View<float*, MemSpace> x_curr("x_curr", n);
    Kokkos::View<float*, MemSpace> x_next("x_next", n);

    auto h_a = Kokkos::create_mirror_view(d_a);
    auto h_b = Kokkos::create_mirror_view(d_b);

    for (int i = 0; i < n; ++i) {
        h_b(i) = b[i];
        for (int j = 0; j < n; ++j) {
            h_a(i, j) = a[i * n + j];
        }
    }

    Kokkos::deep_copy(d_a, h_a);
    Kokkos::deep_copy(d_b, h_b);

    Kokkos::parallel_for(
            "JacobiInit",
            Kokkos::RangePolicy<ExecSpace>(0, n),
            KOKKOS_LAMBDA(const int i) {
                inv_diag(i) = 1.0f / d_a(i, i);
                x_curr(i) = 0.0f;
            });

    bool converged = false;
    const int check_interval = 8;

    for (int iter = 0; iter < ITERATIONS && !converged; ++iter) {
        Kokkos::parallel_for(
                "JacobiStep",
                Kokkos::RangePolicy<ExecSpace>(0, n),
                KOKKOS_LAMBDA(const int i) {
                    float sigma = 0.0f;
                    for (int j = 0; j < n; ++j) {
                        if (j != i) {
                            sigma += d_a(i, j) * x_curr(j);
                        }
                    }
                    x_next(i) = (d_b(i) - sigma) * inv_diag(i);
                });

        if ((iter + 1) % check_interval == 0) {
            float max_diff = 0.0f;

            Kokkos::parallel_reduce(
                    "JacobiCheck",
                    Kokkos::RangePolicy<ExecSpace>(0, n),
                    KOKKOS_LAMBDA(const int i, float& local_max) {
                        const float diff = Kokkos::fabs(x_next(i) - x_curr(i));
                        if (diff > local_max) {
                            local_max = diff;
                        }
                    },
                    Kokkos::Max<float>(max_diff));

            if (max_diff < accuracy) {
                converged = true;
                break;
            }
        }

        Kokkos::kokkos_swap(x_curr, x_next);
    }

    auto final_x = converged ? x_next : x_curr;
    auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), final_x);

    std::vector<float> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = h_x(i);
    }

    return result;
}