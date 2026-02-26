#include "acc_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>

std::vector<float> JacobiAccONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        float accuracy, sycl::device device) {
    const size_t size = static_cast<size_t>(std::sqrt(a.size()));
    const float accuracy_sq = accuracy * accuracy;

    std::vector<float> inverse_diagonal(size);
    for (size_t i = 0; i < size; i++) {
        inverse_diagonal[i] = 1.0f / a[i * size + i];
    }

    sycl::queue q(device, {sycl::property::queue::in_order{}});

    sycl::buffer<float, 1> a_buf(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float, 1> b_buf(b.data(), sycl::range<1>(b.size()));
    sycl::buffer<float, 1> inverse_diag_buffer(inverse_diagonal.data(), sycl::range<1>(size));

    sycl::buffer<float, 1> x_curr_buf{sycl::range<1>(size)};
    sycl::buffer<float, 1> x_next_buf{sycl::range<1>(size)};
    sycl::buffer<float, 1> norm_buf{sycl::range<1>(1)};

    q.submit([&](sycl::handler& cgh) {
        auto x_acc = x_curr_buf.get_access<sycl::access::mode::write>(cgh);
        cgh.fill(x_acc, 0.0f);
    });

    for (int iterator = 0; iterator < ITERATIONS; iterator++) {
        q.submit([&](sycl::handler& cgh) {
            auto a_acc = a_buf.get_access<sycl::access::mode::read>(cgh);
            auto b_acc = b_buf.get_access<sycl::access::mode::read>(cgh);

            auto inverse_diag_access  = inverse_diag_buffer.get_access<sycl::access::mode::read>(cgh);
            auto current_x_access  = x_curr_buf.get_access<sycl::access::mode::read>(cgh);
            auto next_x_access  = x_next_buf.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
                size_t i = idx[0];
                float sum = 0.0f;
                size_t row_start = i * size;

                #pragma unroll(4)
                for (size_t j = 0; j < size; ++j) {
                    if (j != i) {
                        sum += a_acc[row_start + j] * current_x_access [j];
                    }
                }

                next_x_access [i] = inverse_diag_access [i] * (b_acc[i] - sum);
            });
        });

      q.submit([&](sycl::handler& cgh) {
          auto reduction = sycl::reduction(norm_buf, cgh, sycl::plus<float>());
          
          auto current_x_access  = x_curr_buf.get_access<sycl::access::mode::read>(cgh);
          auto next_x_access  = x_next_buf.get_access<sycl::access::mode::read>(cgh);

          cgh.parallel_for(sycl::range<1>(size), reduction,
              [=](sycl::id<1> idx, auto& norm_red) {
                  size_t i = idx[0];
                  float diff = next_x_access [i] - current_x_access [i];
                  norm_red += diff * diff;
              });
      }).wait();

        float current_norm_sq = norm_buf.get_host_access()[0];
        if (current_norm_sq < accuracy_sq) {
            break;
        }

        std::swap(x_curr_buf, x_next_buf);
    }

    std::vector<float> result(size);
    q.submit([&](sycl::handler& cgh) {
        auto x_acc = x_curr_buf.get_access<sycl::access::mode::read>(cgh);
        cgh.copy(x_acc, result.data());
    }).wait();

    return result;
}