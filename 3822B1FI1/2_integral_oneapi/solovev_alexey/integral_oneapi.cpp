#include "integral_oneapi.h"
#include <cmath>
#include <vector>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    sycl::queue queue(device);

    float step = (end - start) / static_cast<float>(count);
    size_t total = static_cast<size_t>(count) * static_cast<size_t>(count);

    std::vector<float> partial(total, 0.0f);

    {
        sycl::buffer<float> buffer(partial.data(), sycl::range<1>(total));

        queue.submit([&](sycl::handler& h) {
            auto acc = buffer.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
                int i = idx[0] % count;
                int j = idx[0] / count;

                float x_mid = start + (i + 0.5f) * step;
                float y_mid = start + (j + 0.5f) * step;

                float value = sycl::sin(x_mid) * sycl::cos(y_mid);
                acc[idx] = value * step * step;
                });
            });

        queue.wait_and_throw();
    }

    float result = 0.0f;
    for (float v : partial) {
        result += v;
    }

    return result;
}