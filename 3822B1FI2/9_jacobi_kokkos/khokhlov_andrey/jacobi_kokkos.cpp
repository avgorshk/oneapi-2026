#include "jacobi_kokkos.h"
#include <cmath>

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {

    using ExecutionSpace = Kokkos::SYCL;
    using MemorySpace = Kokkos::SYCLDeviceUSMSpace;

    const int matrix_dimension = b.size();

    Kokkos::View<float**, Kokkos::LayoutLeft, MemorySpace> matrix_a("matrix_a", matrix_dimension, matrix_dimension);
    Kokkos::View<float*, MemorySpace> vector_b("vector_b", matrix_dimension);
    Kokkos::View<float*, MemorySpace> previous_solution("previous_solution", matrix_dimension);
    Kokkos::View<float*, MemorySpace> current_solution("current_solution", matrix_dimension);

    auto host_matrix_a = Kokkos::create_mirror_view(matrix_a);
    auto host_vector_b = Kokkos::create_mirror_view(vector_b);

    for (int row = 0; row < matrix_dimension; ++row) {
        host_vector_b(row) = b[row];
        for (int col = 0; col < matrix_dimension; ++col) {
            host_matrix_a(row, col) = a[row * matrix_dimension + col];
        }
    }

    Kokkos::deep_copy(matrix_a, host_matrix_a);
    Kokkos::deep_copy(vector_b, host_vector_b);
    Kokkos::deep_copy(previous_solution, 0.0f);
    Kokkos::deep_copy(current_solution, 0.0f);

    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        Kokkos::parallel_for(
            "JacobiIteration",
            Kokkos::RangePolicy<ExecutionSpace>(0, matrix_dimension),
            KOKKOS_LAMBDA(int index) {
                float sum = vector_b(index);
                
                for (int j = 0; j < matrix_dimension; ++j) {
                    if (index != j) {
                        sum -= matrix_a(index, j) * previous_solution(j);
                    }
                }
                
                current_solution(index) = sum / matrix_a(index, index);
            }
        );

        float max_deviation = 0.0f;
        Kokkos::parallel_reduce(
            "ComputeError",
            Kokkos::RangePolicy<ExecutionSpace>(0, matrix_dimension),
            KOKKOS_LAMBDA(int index, float& local_max) {
                float difference = Kokkos::fabs(current_solution(index) - previous_solution(index));
                if (difference > local_max) {
                    local_max = difference;
                }
            },
            Kokkos::Max<float>(max_deviation)
        );

        Kokkos::deep_copy(previous_solution, current_solution);

        if (max_deviation < accuracy) {
            break;
        }
    }

    auto final_solution_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), previous_solution);

    std::vector<float> result(matrix_dimension);
    for (int i = 0; i < matrix_dimension; ++i) {
        result[i] = final_solution_host(i);
    }

    return result;
}