#include "dev_jacobi_oneapi.h"

std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {
    const size_t n = b.size();

    if (n == 0 || a.size() != n * n) {
        return {};
    }

    sycl::queue q(device, sycl::property::queue::in_order{});

    const size_t matrix_size = a.size();
    const size_t vector_size = n;

    float* d_a = sycl::malloc_device<float>(matrix_size, q);
    float* d_b = sycl::malloc_device<float>(vector_size, q);
    float* d_old = sycl::malloc_device<float>(vector_size, q);
    float* d_new = sycl::malloc_device<float>(vector_size, q);
    float* d_diff = sycl::malloc_device<float>(1, q);

    if (d_a == nullptr || d_b == nullptr || d_old == nullptr ||
        d_new == nullptr || d_diff == nullptr) {
        sycl::free(d_a, q);
        sycl::free(d_b, q);
        sycl::free(d_old, q);
        sycl::free(d_new, q);
        sycl::free(d_diff, q);
        return {};
    }

    q.memcpy(d_a, a.data(), matrix_size * sizeof(float));
    q.memcpy(d_b, b.data(), vector_size * sizeof(float));
    q.fill(d_old, 0.0f, vector_size);
    q.fill(d_new, 0.0f, vector_size);
    q.wait_and_throw();

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        float max_difference = 0.0f;

        q.fill(d_diff, 0.0f, 1).wait_and_throw();

        q.submit([&](sycl::handler& h) {
            auto max_reduction = sycl::reduction(
                d_diff,
                0.0f,
                sycl::maximum<float>()
            );

            h.parallel_for(
                sycl::range<1>(n),
                max_reduction,
                [=](sycl::id<1> id, auto& max_value) {
                    const size_t row = id[0];

                    float sum = 0.0f;

                    for (size_t col = 0; col < n; ++col) {
                        if (col != row) {
                            sum += d_a[row * n + col] * d_old[col];
                        }
                    }

                    const float next_value =
                        (d_b[row] - sum) / d_a[row * n + row];

                    d_new[row] = next_value;

                    const float difference =
                        sycl::fabs(next_value - d_old[row]);

                    max_value.combine(difference);
                }
            );
        }).wait_and_throw();

        q.memcpy(&max_difference, d_diff, sizeof(float)).wait_and_throw();

        float* temp = d_old;
        d_old = d_new;
        d_new = temp;

        if (max_difference < accuracy) {
            break;
        }
    }

    std::vector<float> result(n);
    q.memcpy(result.data(), d_old, vector_size * sizeof(float)).wait_and_throw();

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_old, q);
    sycl::free(d_new, q);
    sycl::free(d_diff, q);

    return result;
}
