#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(
    const std::vector<float>& mat_a, const std::vector<float>& mat_b,
    size_t dim, sycl::device dev) {
    
    std::vector<float> mat_c(dim * dim);
    sycl::queue q(dev);

    constexpr size_t TILE = 16;

    // Выделяем память на устройстве
    float* d_a = sycl::malloc_device<float>(dim * dim, q);
    float* d_b = sycl::malloc_device<float>(dim * dim, q);
    float* d_c = sycl::malloc_device<float>(dim * dim, q);

    if (!d_a || !d_b || !d_c) return {};

    q.memcpy(d_a, mat_a.data(), sizeof(float) * dim * dim);
    q.memcpy(d_b, mat_b.data(), sizeof(float) * dim * dim);
    q.memset(d_c, 0, sizeof(float) * dim * dim).wait();

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> tile_a(sycl::range<1>(TILE * TILE), h);
        sycl::local_accessor<float, 1> tile_b(sycl::range<1>(TILE * TILE), h);

        h.parallel_for(sycl::nd_range<2>{
            sycl::range<2>{dim, dim}, // Общий объем работ
            sycl::range<2>{TILE, TILE} // Размер рабочей группы
        }, [=](sycl::nd_item<2> it) {
            
            const int row = it.get_global_id(0);
            const int col = it.get_global_id(1);
            
            const int loc_row = it.get_local_id(0);
            const int loc_col = it.get_local_id(1);

            float acc = 0.0f;
            const int num_tiles = dim / TILE;

            for (int t = 0; t < num_tiles; ++t) {
                tile_a[loc_row * TILE + loc_col] = d_a[row * dim + (t * TILE + loc_col)];
                tile_b[loc_row * TILE + loc_col] = d_b[(t * TILE + loc_row) * dim + col];

                it.barrier(sycl::access::fence_space::local_space);

                for (int k = 0; k < TILE; ++k) {
                    acc += tile_a[loc_row * TILE + k] * tile_b[k * TILE + loc_col];
                }

                it.barrier(sycl::access::fence_space::local_space);
            }

            if (row < dim && col < dim) {
                d_c[row * dim + col] = acc;
            }
        });
    }).wait();

    q.memcpy(mat_c.data(), d_c, sizeof(float) * dim * dim).wait();

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);

    return mat_c;
}