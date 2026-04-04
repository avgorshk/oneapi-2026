#include "shared_jacobi_oneapi.h"
#include <algorithm>

std::vector<float> JacobiSharedONEAPI(
    const std::vector<float>& mat_a, const std::vector<float>& vec_b,
    float target_acc, sycl::device dev) {
    
    const size_t n_size = vec_b.size();
    std::vector<float> final_res(n_size);
    sycl::queue q(dev);

    float* s_a = sycl::malloc_shared<float>(n_size * n_size, q);
    float* s_b = sycl::malloc_shared<float>(n_size, q);
    float* s_x = sycl::malloc_shared<float>(n_size, q);
    float* s_x_next = sycl::malloc_shared<float>(n_size, q);

    if (!s_a || !s_b || !s_x || !s_x_next) return {};

    std::copy(mat_a.begin(), mat_a.end(), s_a);
    std::copy(vec_b.begin(), vec_b.end(), s_b);
    std::fill(s_x, s_x + n_size, 0.0f);

    int iter = 0;
    float current_diff = 0.0f;

    do {
        q.parallel_for(sycl::range<1>(n_size), [=](sycl::id<1> idx) {
            size_t row = idx[0];
            float sum = 0.0f;
            
            for (size_t col = 0; col < n_size; ++col) {
                if (row != col) {
                    sum += s_a[row * n_size + col] * s_x[col];
                }
            }
            s_x_next[row] = (s_b[row] - sum) / s_a[row * n_size + row];
        }).wait();

        current_diff = 0.0f;
        for (size_t i = 0; i < n_size; ++i) {
            float d = std::abs(s_x_next[i] - s_x[i]);
            if (d > current_diff) current_diff = d;
        }

        for (size_t i = 0; i < n_size; ++i) {
            s_x[i] = s_x_next[i];
        }

        iter++;
    } while (iter < ITERATIONS && current_diff >= target_acc);

    std::copy(s_x, s_x + n_size, final_res.begin());

    sycl::free(s_a, q);
    sycl::free(s_b, q);
    sycl::free(s_x, q);
    sycl::free(s_x_next, q);

    return final_res;
}