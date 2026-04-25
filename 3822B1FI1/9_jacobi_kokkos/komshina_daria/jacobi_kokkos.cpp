#include "jacobi_kokkos.h"

#include <utility>

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {
    const size_t n = b.size();

    if (n == 0 || a.size() != n * n) {
        return {};
    }

    using Exec = Kokkos::SYCL;
    using Mem = Kokkos::SYCLDeviceUSMSpace;
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<size_t>>;

    Kokkos::View<float*, Mem> A("A", a.size());
    Kokkos::View<float*, Mem> B("B", n);
    Kokkos::View<float*, Mem> old_x("old_x", n);
    Kokkos::View<float*, Mem> new_x("new_x", n);

    Kokkos::View<const float*, Kokkos::HostSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_A(a.data(), a.size());

    Kokkos::View<const float*, Kokkos::HostSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_B(b.data(), n);

    Kokkos::deep_copy(A, host_A);
    Kokkos::deep_copy(B, host_B);
    Kokkos::deep_copy(old_x, 0.0f);
    Kokkos::deep_copy(new_x, 0.0f);

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        float max_change = 0.0f;

        Kokkos::parallel_reduce(
            "JacobiKokkosIteration",
            Policy(0, n),
            KOKKOS_LAMBDA(const size_t row, float& local_max) {
                float sum = 0.0f;

                for (size_t col = 0; col < n; ++col) {
                    if (col != row) {
                        sum += A(row * n + col) * old_x(col);
                    }
                }

                const float value = (B(row) - sum) / A(row * n + row);
                new_x(row) = value;

                float diff = value - old_x(row);

                if (diff < 0.0f) {
                    diff = -diff;
                }

                if (diff > local_max) {
                    local_max = diff;
                }
            },
            Kokkos::Max<float>(max_change)
        );

        Kokkos::fence();

        std::swap(old_x, new_x);

        if (max_change < accuracy) {
            break;
        }
    }

    std::vector<float> result(n);

    Kokkos::View<float*, Kokkos::HostSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_result(result.data(), n);

    Kokkos::deep_copy(host_result, old_x);

    return result;
}
