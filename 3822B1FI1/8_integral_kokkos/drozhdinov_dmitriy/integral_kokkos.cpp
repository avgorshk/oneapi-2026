#include "integral_kokkos.h"
#include <cmath>

float IntegralKokkos(float start, float end, int count) {
    float result = 0.0f;
    const float h = (end - start) / count;
	
	Kokkos::parallel_reduce(
        "DoubleIntegral",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {count, count}),
        KOKKOS_LAMBDA(const int i, const int j, float& local_sum) {
            float x = start + (i + 0.5f) * h;
            float y = start + (j + 0.5f) * h;

            local_sum += sinf(x) * cosf(y);
        },
        result
    );
	
	result *= h * h;
    return result;
}