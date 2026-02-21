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

        q.parallel_for(sycl::nd_range<1>(
            sycl::range<1>(((n + wg_size - 1) / wg_size) * wg_size),
            sycl::range<1>(wg_size)
        ), [=](sycl::nd_item<1> item) {
            size_t gid = item.get_global_id(0);
            size_t lid = item.get_local_id(0);
            size_t group_size = item.get_local_range()[0];

            sycl::group_local_memory_for_overwrite<float[256]> local_mem(item.get_group());
            float* local_sum = local_mem.get_ptr();

            float local_val = 0.0f;
            if (gid < n) {
                float sum = 0.0f;
                size_t row_start = gid * n;
                #pragma unroll(4)
                for (size_t j = 0; j < n; j++) {
                    if (j != gid) {
                        sum += d_a[row_start + j] * d_x_curr[j];
                    }
                }
                float x_new = d_inv_diag[gid] * (d_b[gid] - sum);
                d_x_next[gid] = x_new;

                float diff = x_new - d_x_curr[gid];
                local_val = diff * diff;
            }

            local_sum[lid] = local_val;
            item.barrier(sycl::access::fence_space::local_space);

            for (size_t stride = group_size / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_sum[lid] += local_sum[lid + stride];
                }
                item.barrier(sycl::access::fence_space::local_space);
            }

            if (lid == 0) {
                sycl::atomic_ref<float, sycl::memory_order::relaxed, 
                                sycl::memory_scope::device> 
                    atomic_norm(d_norm_sq);
                atomic_norm.fetch_add(local_sum[0]);
            }
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
