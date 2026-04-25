#include "jacobi_kokkos.h"
#include <cmath>

std::vector<float> JacobiKokkos(
    const std::vector<float> &A_host_data,
    const std::vector<float> &B_host_data,
    float eps)
{

    using ExecutionSpace = Kokkos::SYCL;
    using MemorySpace = Kokkos::SYCLDeviceUSMSpace;

    const int N = static_cast<int>(B_host_data.size());
    if (N <= 0)
        return {};

    Kokkos::View<float **, Kokkos::LayoutLeft, MemorySpace> A_dev("A_dev", N, N);
    Kokkos::View<float *, MemorySpace> B_dev("B_dev", N);
    Kokkos::View<float *, MemorySpace> diag_inv("diag_inv", N);
    Kokkos::View<float *, MemorySpace> x_old("x_old", N);
    Kokkos::View<float *, MemorySpace> x_new("x_new", N);

    auto A_mirror = Kokkos::create_mirror_view(A_dev);
    auto B_mirror = Kokkos::create_mirror_view(B_dev);

    for (int i = 0; i < N; ++i)
    {
        B_mirror(i) = B_host_data[i];
        for (int j = 0; j < N; ++j)
        {
            A_mirror(i, j) = A_host_data[i * N + j];
        }
    }

    Kokkos::deep_copy(A_dev, A_mirror);
    Kokkos::deep_copy(B_dev, B_mirror);

    Kokkos::parallel_for(
        "InitDiagAndVectors",
        Kokkos::RangePolicy<ExecutionSpace>(0, N),
        KOKKOS_LAMBDA(const int idx) {
            diag_inv(idx) = 1.0f / A_dev(idx, idx);
            x_old(idx) = 0.0f;
            x_new(idx) = 0.0f;
        });

    constexpr int CHECK_FREQ = 4;
    bool converged = false;
    int iteration = 0;

    for (; iteration < ITERATIONS && !converged; ++iteration)
    {
        Kokkos::parallel_for(
            "JacobiIteration",
            Kokkos::RangePolicy<ExecutionSpace>(0, N),
            KOKKOS_LAMBDA(const int row) {
                float sum = 0.0f;
                for (int col = 0; col < N; ++col)
                {
                    if (col != row)
                    {
                        sum += A_dev(row, col) * x_old(col);
                    }
                }
                x_new(row) = (B_dev(row) - sum) * diag_inv(row);
            });

        if ((iteration + 1) % CHECK_FREQ == 0)
        {
            float max_change = 0.0f;
            Kokkos::parallel_reduce(
                "CheckConvergence",
                Kokkos::RangePolicy<ExecutionSpace>(0, N),
                KOKKOS_LAMBDA(const int i, float &local_max) {
                    float diff = Kokkos::fabs(x_new(i) - x_old(i));
                    if (diff > local_max)
                        local_max = diff;
                },
                Kokkos::Max<float>(max_change));
            if (max_change < eps)
                converged = true;
        }

        Kokkos::kokkos_swap(x_old, x_new);
    }

    auto result_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_old);

    std::vector<float> solution(N);
    for (int i = 0; i < N; ++i)
        solution[i] = result_mirror(i);
    return solution;
}