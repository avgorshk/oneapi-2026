#include "jacobi_kokkos.h"

#include <vector>

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {
    using ExecSpace = Kokkos::SYCL;
    using MemSpace  = Kokkos::SYCLDeviceUSMSpace;

    const int n = static_cast<int>(b.size());

    if (n == 0) {
        return {};
    }

    if (a.size() != static_cast<std::size_t>(n) * static_cast<std::size_t>(n)) {
        return {};
    }

    if (accuracy <= 0.0f) {
        accuracy = 1e-6f;
    }

    // Храним A в 1D виде по строкам, как в условии
    Kokkos::View<float*, MemSpace> d_a("A", a.size());
    Kokkos::View<float*, MemSpace> d_b("b", n);
    Kokkos::View<float*, MemSpace> x_old("x_old", n);
    Kokkos::View<float*, MemSpace> x_new("x_new", n);
    Kokkos::View<float*, MemSpace> inv_diag("inv_diag", n);

    auto h_a = Kokkos::create_mirror_view(d_a);
    auto h_b = Kokkos::create_mirror_view(d_b);

    for (std::size_t i = 0; i < a.size(); ++i) {
        h_a(i) = a[i];
    }
    for (int i = 0; i < n; ++i) {
        h_b(i) = b[i];
    }

    Kokkos::deep_copy(d_a, h_a);
    Kokkos::deep_copy(d_b, h_b);

    Kokkos::parallel_for(
        "JacobiInit",
        Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const int i) {
            inv_diag(i) = 1.0f / d_a(static_cast<std::size_t>(i) * n + i);
            x_old(i) = 0.0f;
        });

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        float max_diff = 0.0f;

        Kokkos::parallel_reduce(
            "JacobiIter",
            Kokkos::RangePolicy<ExecSpace>(0, n),
            KOKKOS_LAMBDA(const int i, float& local_max) {
                float sigma = 0.0f;
                const std::size_t row = static_cast<std::size_t>(i) * n;

                for (int j = 0; j < n; ++j) {
                    if (j != i) {
                        sigma += d_a(row + j) * x_old(j);
                    }
                }

                const float xi = (d_b(i) - sigma) * inv_diag(i);
                x_new(i) = xi;

                const float diff = Kokkos::fabs(xi - x_old(i));
                if (diff > local_max) {
                    local_max = diff;
                }
            },
            Kokkos::Max<float>(max_diff));

        Kokkos::kokkos_swap(x_old, x_new);

        if (max_diff < accuracy) {
            break;
        }
    }

    auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_old);

    std::vector<float> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = h_x(i);
    }

    return result;
}