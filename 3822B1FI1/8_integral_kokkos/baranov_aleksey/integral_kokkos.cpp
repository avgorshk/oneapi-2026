#include "integral_kokkos.h"

float IntegralKokkos(float start, float end, int count) {
    float step = (end - start) / count;
    float accumulatedSum = 0.0f;

    auto rangePolicy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {count, count});

    Kokkos::parallel_reduce("DoubleIntegral", rangePolicy, 
        KOKKOS_LAMBDA (const int idx, const int idy, float& partialSum) {
            float x = start + (idx + 0.5f) * step;
            float y = start + (idy + 0.5f) * step;
            partialSum += Kokkos::sin(x) * Kokkos::cos(y);
        }, accumulatedSum);

    return accumulatedSum * step * step;
}