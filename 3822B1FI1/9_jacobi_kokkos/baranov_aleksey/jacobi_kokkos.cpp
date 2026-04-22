#include "jacobi_kokkos.h"

std::vector<float> JacobiKokkos(const std::vector<float>& a,
                                const std::vector<float>& b,
                                float accuracy)
{
    int size = static_cast<int>(b.size());

    Kokkos::View<float**> matrixA("matrixA", size, size);
    Kokkos::View<float*>  rhsVector("rhsVector", size);
    Kokkos::View<float*>  currX("currX", size);
    Kokkos::View<float*>  nextX("nextX", size);
    Kokkos::View<float*>  invDiag("invDiag", size);

    auto hostMatrixA = Kokkos::create_mirror_view(matrixA);
    auto hostRhs = Kokkos::create_mirror_view(rhsVector);

    for (int i = 0; i < size; ++i) {
        hostRhs(i) = b[i];
        for (int j = 0; j < size; ++j) {
            hostMatrixA(i, j) = a[i * size + j];
        }
    }

    Kokkos::deep_copy(matrixA, hostMatrixA);
    Kokkos::deep_copy(rhsVector, hostRhs);
    Kokkos::deep_copy(currX, 0.0f);

    Kokkos::parallel_for("InitInvDiag", size,
        KOKKOS_LAMBDA(const int i) {
            invDiag(i) = 1.0f / matrixA(i, i);
        });

    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        Kokkos::parallel_for("JacobiStep", size,
            KOKKOS_LAMBDA(const int i) {
                float rowSum = 0.0f;

                for (int j = 0; j < size; ++j) {
                    if (i != j)
                        rowSum += matrixA(i, j) * currX(j);
                }

                nextX(i) = (rhsVector(i) - rowSum) * invDiag(i);
            });

        float maxDiff = 0.0f;

        Kokkos::parallel_reduce("JacobiCheck", size,
            KOKKOS_LAMBDA(const int i, float& localMax) {
                float delta = Kokkos::abs(nextX(i) - currX(i));
                if (delta > localMax)
                    localMax = delta;
            },
            Kokkos::Max<float>(maxDiff));

        Kokkos::kokkos_swap(currX, nextX);

        if (maxDiff < accuracy)
            break;
    }

    std::vector<float> result(size);
    auto hostCurrX = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), currX);

    for (int i = 0; i < size; ++i)
        result[i] = hostCurrX(i);

    return result;
}