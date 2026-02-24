#include <cmath>
#include <sycl/sycl.hpp>

#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device)
{
    const float dx = (end - start) / static_cast<float>(count);

    sycl::queue q(device);

    float sum = 0.0f;

    sycl::buffer<float> sum_buf(&sum, 1);

    q.submit([&](sycl::handler& cgh) {
        auto reduction = sycl::reduction(sum_buf, cgh, sycl::plus<>());

        cgh.parallel_for(
            sycl::range<2>(count, count),
            reduction,
            [=](sycl::id<2> id, auto& sum) {
                const int i = static_cast<int>(id.get(0));
                const int j = static_cast<int>(id.get(1));

                const float x = start + dx * (i + 0.5f);
                const float y = start + dx * (j + 0.5f);

                sum += std::sin(x) * std::cos(y);
            }
        );
    }).wait();

    return sum * dx * dx;

}
