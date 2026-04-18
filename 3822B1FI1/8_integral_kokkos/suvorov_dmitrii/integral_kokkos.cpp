#include "integral_kokkos.h"

struct SinCosSums {
    float sin_sum;
    float cos_sum;
};

struct IntegralFunctor {
    float start;
    float step;

    using value_type = SinCosSums;

    KOKKOS_INLINE_FUNCTION
    void init(value_type& v) const {
        v.sin_sum = 0.0f;
        v.cos_sum = 0.0f;
    }

    KOKKOS_INLINE_FUNCTION
    void join(value_type& dst, const value_type& src) const {
        dst.sin_sum += src.sin_sum;
        dst.cos_sum += src.cos_sum;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, value_type& v) const {
        const float mid = start + (static_cast<float>(i) + 0.5f) * step;
        v.sin_sum += Kokkos::sin(mid);
        v.cos_sum += Kokkos::cos(mid);
    }
};

float IntegralKokkos(float start, float end, int count) {
    using ExecSpace = Kokkos::SYCL;

    if (count <= 0) {
        return 0.0f;
    }

    const float step = (end - start) / static_cast<float>(count);

    SinCosSums sums{0.0f, 0.0f};

    Kokkos::parallel_reduce(
        "IntegralSinCos",
        Kokkos::RangePolicy<ExecSpace>(0, count),
        IntegralFunctor{start, step},
        sums
    );

    Kokkos::fence();

    return sums.sin_sum * sums.cos_sum * step * step;
}