#include "acc_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>

std::vector<float> JacobiAccONEAPI(const std::vector<float> a,
                                   const std::vector<float> b,
                                   float accuracy,
                                   sycl::device device) {
  int size = b.size();
  std::vector<float> current_solution(size, 0.0f);
  std::vector<float> previous_solution(size, 0.0f);
  float max_diff = 0.0f;
  int iteration = 0;

  {
    sycl::buffer<float> matrix_buffer(a.data(), a.size());
    sycl::buffer<float> vector_b_buffer(b.data(), b.size());
    sycl::buffer<float> buf_curr(current_solution.data(), current_solution.size());
    sycl::buffer<float> previous_solution_buffer(previous_solution.data(), previous_solution.size());
    sycl::buffer<float> max_diff_buffer(&max_diff, 1);
    sycl::queue queue(device);

    while (iteration++ < ITERATIONS) {
      queue.submit([&](sycl::handler &cgh) {
        auto a_access = matrix_buffer.get_access<sycl::access::mode::read>(cgh);
        auto b_access = vector_b_buffer.get_access<sycl::access::mode::read>(cgh);
        auto prev_access = previous_solution_buffer.get_access<sycl::access::mode::read>(cgh);
        auto curr_access = buf_curr.get_access<sycl::access::mode::write>(cgh);
        
        auto reduction = sycl::reduction(max_diff_buffer, cgh, sycl::maximum<float>());

        cgh.parallel_for(sycl::range<1>(size), reduction,
                         [=](sycl::id<1> idx, auto &max_diff_reduction) {
          int i = idx[0];
          float sum = b_access[i];
          
          for (int j = 0; j < size; ++j) {
            if (i != j) {
              sum -= a_access[i * size + j] * prev_access[j];
            }
          }
          
          float new_value = sum / a_access[i * size + i];
          curr_access[i] = new_value;
          
          float diff = sycl::fabs(new_value - prev_access[i]);
          max_diff_reduction.combine(diff);
        });
      });

      queue.wait();

      float current_max_diff = max_diff_buffer.get_host_access()[0];
      if (current_max_diff < accuracy) {
        break;
      }

      queue.submit([&](sycl::handler &cgh) {
        auto curr_access = buf_curr.get_access<sycl::access::mode::read>(cgh);
        auto prev_access = previous_solution_buffer.get_access<sycl::access::mode::write>(cgh);
        
        cgh.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
          prev_access[idx] = curr_access[idx];
        });
      });

      queue.wait();
    }
  }

  return current_solution;
}