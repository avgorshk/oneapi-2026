#include "integral_oneapi.h"
#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    const float step = (end - start) / static_cast<float>(count);

    float result = 0.0f;

    sycl::queue queue(device);

    {
        sycl::buffer<float> result_buf(&result, 1);

        queue.submit([&](sycl::handler& h) {
            auto red = sycl::reduction(result_buf, h, sycl::plus<float>());

            h.parallel_for(sycl::range<2>(count, count), red,
                [=](sycl::id<2> idx, auto& sum) {
                    const int i = idx[0];
                    const int j = idx[1];

                    const float x = start + (i + 0.5f) * step;
                    const float y = start + (j + 0.5f) * step;

                    sum += sycl::sin(x) * sycl::cos(y);
                });
            }).wait();
    }

    return result * step * step;
}