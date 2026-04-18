#include "block_gemm_oneapi.h"

#include <vector>

namespace {
    constexpr std::size_t TILE = 16;
}

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    if (size == 0) {
        return {};
    }

    if (a.size() != size * size || b.size() != size * size) {
        return {};
    }

    try {
        sycl::queue q(device);

        std::vector<float> c(size * size, 0.0f);

        sycl::buffer<float, 2> a_buf(a.data(), sycl::range<2>(size, size));
        sycl::buffer<float, 2> b_buf(b.data(), sycl::range<2>(size, size));
        sycl::buffer<float, 2> c_buf(c.data(), sycl::range<2>(size, size));

        const std::size_t global_rows =
            ((size + TILE - 1) / TILE) * TILE;
        const std::size_t global_cols =
            ((size + TILE - 1) / TILE) * TILE;

        q.submit([&](sycl::handler& h) {
            auto a_acc = a_buf.get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.get_access<sycl::access::mode::read>(h);
            auto c_acc = c_buf.get_access<sycl::access::mode::write>(h);

            sycl::local_accessor<float, 2> tile_a(sycl::range<2>(TILE, TILE), h);
            sycl::local_accessor<float, 2> tile_b(sycl::range<2>(TILE, TILE), h);

            h.parallel_for(
                sycl::nd_range<2>(
                    sycl::range<2>(global_rows, global_cols),
                    sycl::range<2>(TILE, TILE)),
                [=](sycl::nd_item<2> item) {
                    const std::size_t global_row = item.get_global_id(0);
                    const std::size_t global_col = item.get_global_id(1);

                    const std::size_t local_row = item.get_local_id(0);
                    const std::size_t local_col = item.get_local_id(1);

                    float sum = 0.0f;

                    for (std::size_t block = 0; block < size; block += TILE) {
                        const std::size_t a_col = block + local_col;
                        const std::size_t b_row = block + local_row;

                        if (global_row < size && a_col < size) {
                            tile_a[local_row][local_col] = a_acc[global_row][a_col];
                        } else {
                            tile_a[local_row][local_col] = 0.0f;
                        }

                        if (b_row < size && global_col < size) {
                            tile_b[local_row][local_col] = b_acc[b_row][global_col];
                        } else {
                            tile_b[local_row][local_col] = 0.0f;
                        }

                        item.barrier(sycl::access::fence_space::local_space);

                        for (std::size_t k = 0; k < TILE; ++k) {
                            sum += tile_a[local_row][k] * tile_b[k][local_col];
                        }

                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    if (global_row < size && global_col < size) {
                        c_acc[global_row][global_col] = sum;
                    }
                });
        });

        q.wait();
        return c;
    } catch (const sycl::exception&) {
        return {};
    }
}