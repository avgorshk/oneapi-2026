#include "acc_jacobi_oneapi.h"
#include <algorithm>
#include <cmath>

using BufferType = sycl::buffer<float>;

std::vector<float> JacobiAccONEAPI(const std::vector<float> &a,
                                   const std::vector<float> &b, float accuracy,
                                   sycl::device device) {
  const size_t n = b.size();
  std::vector<float> prevSolution(n, 0.0f);
  std::vector<float> currSolution(n, 0.0f);

  sycl::queue computeQueue(device);

  BufferType matrixBuffer(a.data(), a.size());
  BufferType rhsBuffer(b.data(), b.size());
  BufferType prevBuffer(prevSolution.data(), prevSolution.size());
  BufferType currBuffer(currSolution.data(), currSolution.size());

  for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
    computeQueue
        .submit([&](sycl::handler &cgh) {
          auto matrixAcc = matrixBuffer.get_access<sycl::access::mode::read>(cgh);
          auto rhsAcc    = rhsBuffer.get_access<sycl::access::mode::read>(cgh);
          auto prevAcc   = prevBuffer.get_access<sycl::access::mode::read>(cgh);
          auto currAcc   = currBuffer.get_access<sycl::access::mode::write>(cgh);

          cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            const int row = id.get(0);
            float newValue = 0.0f;

            for (int col = 0; col < static_cast<int>(n); ++col) {
              if (row == col) {
                newValue += rhsAcc[col];
              } else {
                newValue -= matrixAcc[row * n + col] * prevAcc[col];
              }
            }
            newValue /= matrixAcc[row * n + row];
            currAcc[row] = newValue;
          });
        })
        .wait();

    auto prevHost = prevBuffer.get_host_access(sycl::read_write);
    auto currHost = currBuffer.get_host_access(sycl::read_only);

    float maxDiff = 0.0f;
    for (size_t i = 0; i < n; ++i) {
      const float diff = std::fabs(currHost[i] - prevHost[i]);
      if (diff > maxDiff) {
        maxDiff = diff;
      }
      prevHost[i] = currHost[i];
    }

    if (maxDiff < accuracy) {
      break;
    }
  }

  auto finalHost = prevBuffer.get_host_access(sycl::read_only);
  std::vector<float> result(n);
  for (size_t i = 0; i < n; ++i) {
    result[i] = finalHost[i];
  }

  return result;
}