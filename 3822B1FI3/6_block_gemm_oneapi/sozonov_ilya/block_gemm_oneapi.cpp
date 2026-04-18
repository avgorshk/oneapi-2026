#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(const std::vector<float>& a,
                                   const std::vector<float>& b, size_t size,
                                   sycl::device device) {
  sycl::queue compute_queue(device);

  const size_t n = size;
  const size_t matrix_elements = n * n;

  float* device_a = sycl::malloc_device<float>(matrix_elements, compute_queue);
  float* device_b = sycl::malloc_device<float>(matrix_elements, compute_queue);
  float* device_c = sycl::malloc_device<float>(matrix_elements, compute_queue);

  compute_queue.memcpy(device_a, a.data(), sizeof(float) * matrix_elements);
  compute_queue.memcpy(device_b, b.data(), sizeof(float) * matrix_elements);
  compute_queue.memset(device_c, 0, sizeof(float) * matrix_elements).wait();

  constexpr size_t TILE_SIZE = 16;
  constexpr size_t SHARED_PAD = TILE_SIZE + 1;

  sycl::range<2> global_grid(n, n);
  sycl::range<2> local_block(TILE_SIZE, TILE_SIZE);

  compute_queue
      .submit([&](sycl::handler& command_group) {
        sycl::local_accessor<float, 1> shared_tile_a(TILE_SIZE * SHARED_PAD,
                                                     command_group);
        sycl::local_accessor<float, 1> shared_tile_b(TILE_SIZE * SHARED_PAD,
                                                     command_group);

        command_group.parallel_for(
            sycl::nd_range<2>(global_grid, local_block),
            [=](sycl::nd_item<2> work_item) {
              const size_t global_row = work_item.get_global_id(0);
              const size_t global_col = work_item.get_global_id(1);

              const size_t local_row = work_item.get_local_id(0);
              const size_t local_col = work_item.get_local_id(1);

              float accumulation_register = 0.0f;

              for (size_t tile_offset = 0; tile_offset < n;
                   tile_offset += TILE_SIZE) {
                const size_t a_column = tile_offset + local_col;

                shared_tile_a[local_row * SHARED_PAD + local_col] =
                    (global_row < n && a_column < n)
                        ? device_a[global_row * n + a_column]
                        : 0.0f;

                const size_t b_row = tile_offset + local_row;

                shared_tile_b[local_row * SHARED_PAD + local_col] =
                    (b_row < n && global_col < n)
                        ? device_b[b_row * n + global_col]
                        : 0.0f;

                work_item.barrier(sycl::access::fence_space::local_space);

#pragma unroll 8
                for (size_t inner_k = 0; inner_k < TILE_SIZE; ++inner_k) {
                  const float a_element =
                      shared_tile_a[local_row * SHARED_PAD + inner_k];

                  const float b_element =
                      shared_tile_b[inner_k * SHARED_PAD + local_col];

                  accumulation_register += a_element * b_element;
                }

                work_item.barrier(sycl::access::fence_space::local_space);
              }

              if (global_row < n && global_col < n) {
                device_c[global_row * n + global_col] = accumulation_register;
              }
            });
      })
      .wait();

  std::vector<float> result_matrix(matrix_elements);
  compute_queue
      .memcpy(result_matrix.data(), device_c, sizeof(float) * matrix_elements)
      .wait();

  sycl::free(device_a, compute_queue);
  sycl::free(device_b, compute_queue);
  sycl::free(device_c, compute_queue);

  return result_matrix;
}