#include "jacobi_acc_oneapi.h"
#include <algorithm>
#include <buffer.hpp>
#include <handler.hpp>
#include <range.hpp>
#include <reduction.hpp>
#include <vector>

std::vector<float> JacobiAccONEAPI(const std::vector<float> a,
                                   const std::vector<float> b,
                                   float accuracy,
                                   sycl::device device) {
  int size = b.size();
  std::vector<float> curr_ans(size, 0.0f);
  std::vector<float> prev_ans(size, 0.0f);
  float max_diff = 0.0f;
  int iteration = 0;

  {
    sycl::buffer<float> buf_a(a.data(), a.size());
    sycl::buffer<float> buf_b(b.data(), b.size());
    sycl::buffer<float> buf_curr(curr_ans.data(), curr_ans.size());
    sycl::buffer<float> buf_prev(prev_ans.data(), prev_ans.size());
    sycl::buffer<float> buf_max_diff(&max_diff, 1);
    sycl::queue queue(device);

    while (iteration++ < ITERATIONS) {
      queue.submit([&](sycl::handler &cgh) {
        auto a_access = buf_a.get_access<sycl::access::mode::read>(cgh);
        auto b_access = buf_b.get_access<sycl::access::mode::read>(cgh);
        auto prev_access = buf_prev.get_access<sycl::access::mode::read>(cgh);
        auto curr_access = buf_curr.get_access<sycl::access::mode::write>(cgh);
        
        auto reduction = sycl::reduction(buf_max_diff, cgh, sycl::maximum<float>());

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

      float current_max_diff = buf_max_diff.get_host_access()[0];
      if (current_max_diff < accuracy) {
        break;
      }

      queue.submit([&](sycl::handler &cgh) {
        auto curr_access = buf_curr.get_access<sycl::access::mode::read>(cgh);
        auto prev_access = buf_prev.get_access<sycl::access::mode::write>(cgh);
        
        cgh.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
          prev_access[idx] = curr_access[idx];
        });
      });

      queue.wait();
    }
  }

  return curr_ans;
}