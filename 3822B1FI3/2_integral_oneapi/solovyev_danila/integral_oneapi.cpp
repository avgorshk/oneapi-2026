#include "integral_oneapi.h"
#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    float delta = (end - start) / static_cast<float>(count);
    float totalSum = 0.0f;

    sycl::queue q(device);
    {
    sycl::buffer<float, 1> sumBuf(&totalSum, sycl::range<1>(1));

    q.submit([&](sycl::handler& h) {

        auto reductionOp = sycl::reduction(sumBuf, h, sycl::plus<float>());

        h.parallel_for(sycl::range<2>(count, count), reductionOp,
            [=](sycl::id<2> idx, auto& sumAcc) {
                int i = idx[0];
                int j = idx[1];

                float x = start + (i + 0.5f) * delta;
                float y = start + (j + 0.5f) * delta;

                sumAcc += sycl::sin(x) * sycl::cos(y);
            });
    }).wait();
    }
    return totalSum * (delta * delta);
}