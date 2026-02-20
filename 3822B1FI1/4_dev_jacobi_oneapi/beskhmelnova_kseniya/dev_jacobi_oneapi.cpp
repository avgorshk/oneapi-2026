#include "dev_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>

std::vector<float> JacobiDevONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        float accuracy, sycl::device device) {
    
    const size_t n = b.size();
    const float accuracy_sq = accuracy * accuracy;

    std::vector<float> inv_diag(n);
    for (size_t i = 0; i < n; i++) {
        inv_diag[i] = 1.0f / a[i * n + i];
    }

    sycl::queue q(device, {sycl::property::queue::in_order{}});

    float* d_a = sycl::malloc_device<float>(n * n, q);
    float* d_b = sycl::malloc_device<float>(n, q);
    float* d_inv_diag = sycl::malloc_device<float>(n, q);
    float* d_x_curr = sycl::malloc_device<float>(n, q);
    float* d_x_next = sycl::malloc_device<float>(n, q);
    float* d_norm_sq = sycl::malloc_device<float>(1, q);

    q.memcpy(d_a, a.data(), n * n * sizeof(float));
    q.memcpy(d_b, b.data(), n * sizeof(float));
    q.memcpy(d_inv_diag, inv_diag.data(), n * sizeof(float));
    q.fill(d_x_curr, 0.0f, n).wait();

    size_t wg_size = device.is_gpu() ? 128 : 256;
    wg_size = std::min(wg_size, device.get_info<sycl::info::device::max_work_group_size>());

    bool converged = false;
    for (int iter = 0; iter < ITERATIONS && !converged; iter++) {
        q.memset(d_norm_sq, 0, sizeof(float));

        sycl::buffer<float, 1> norm_buf(d_norm_sq, sycl::range<1>(1));
        
        q.submit([&](sycl::handler& cgh) {
            auto a_acc = sycl::accessor(d_a, cgh, sycl::read_only);
            auto b_acc = sycl::accessor(d_b, cgh, sycl::read_only);
            auto inv_diag_acc = sycl::accessor(d_inv_diag, cgh, sycl::read_only);
            auto x_curr_acc = sycl::accessor(d_x_curr, cgh, sycl::read_only);
            auto x_next_acc = sycl::accessor(d_x_next, cgh, sycl::write_only);
            auto reduction = sycl::reduction(norm_buf, cgh, sycl::plus<float>());

            cgh.parallel_for(sycl::nd_range<1>(
                sycl::range<1>(((n + wg_size - 1) / wg_size) * wg_size),
                sycl::range<1>(wg_size)
            ), reduction, [=](sycl::nd_item<1> item, auto& norm_red) {
                size_t gid = item.get_global_id(0);
                if (gid >= n) return;

                float sum = 0.0f;
                size_t row_start = gid * n;
                #pragma unroll(4)
                for (size_t j = 0; j < n; j++) {
                    if (j != gid) {
                        sum += a_acc[row_start + j] * x_curr_acc[j];
                    }
                }
                float x_new = inv_diag_acc[gid] * (b_acc[gid] - sum);
                x_next_acc[gid] = x_new;
                float diff = x_new - x_curr_acc[gid];
                norm_red += diff * diff;
            });
        }).wait();

        float norm_host;
        q.memcpy(&norm_host, d_norm_sq, sizeof(float)).wait();
        if (norm_host < accuracy_sq) {
            converged = true;
        }

        std::swap(d_x_curr, d_x_next);
    }

    std::vector<float> result(n);
    q.memcpy(result.data(), d_x_curr, n * sizeof(float)).wait();

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_inv_diag, q);
    sycl::free(d_x_curr, q);
    sycl::free(d_x_next, q);
    sycl::free(d_norm_sq, q);

    return result;
}
