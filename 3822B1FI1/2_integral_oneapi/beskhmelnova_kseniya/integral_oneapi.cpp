#include "integral_oneapi.h"
#include <vector>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    const float step = (end - start) / static_cast<float>(count);
    const float area = step * step;
    sycl::queue q(device, {sycl::property::queue::in_order{}});

    float sum_sin = 0.0f, sum_cos = 0.0f;
    {
        sycl::buffer<float> sum_sin_buf(&sum_sin, 1);
        sycl::buffer<float> sum_cos_buf(&sum_cos, 1);
        
        q.submit([&](sycl::handler& cgh) {
            auto sin_acc = sum_sin_buf.get_access<sycl::access::mode::atomic>(cgh);
            auto cos_acc = sum_cos_buf.get_access<sycl::access::mode::atomic>(cgh);
            
            cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
                size_t i = idx[0];
                float x = start + step * (i + 0.5f);
                sin_acc[0].fetch_add(sycl::sin(x));
                cos_acc[0].fetch_add(sycl::cos(x));
            });
        }).wait();
    }
    return (sum_sin * sum_cos) * area;
}
