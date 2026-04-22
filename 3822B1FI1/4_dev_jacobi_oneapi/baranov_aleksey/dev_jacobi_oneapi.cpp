#include "dev_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>

std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {
    
    const int n = static_cast<int>(b.size());
    const float squaredTolerance = accuracy * accuracy;
    sycl::queue computeQueue(device, sycl::property::queue::in_order{});

    // Обратные диагональные элементы
    std::vector<float> inverseDiagonal(n);
    for (int i = 0; i < n; i++) {
        inverseDiagonal[i] = 1.0f / a[i * n + i];
    }

    // Выделение памяти на устройстве
    float* devMatrix   = sycl::malloc_device<float>(n * n, computeQueue);
    float* devRhs      = sycl::malloc_device<float>(n, computeQueue);
    float* devInvDiag  = sycl::malloc_device<float>(n, computeQueue);
    float* devXcurrent = sycl::malloc_device<float>(n, computeQueue);
    float* devXnext    = sycl::malloc_device<float>(n, computeQueue);

    computeQueue.memcpy(devMatrix, a.data(), sizeof(float) * n * n);
    computeQueue.memcpy(devRhs,    b.data(), sizeof(float) * n);
    computeQueue.memcpy(devInvDiag, inverseDiagonal.data(), sizeof(float) * n);
    computeQueue.fill(devXcurrent, 0.0f, n);

    const size_t workGroupSize = 64;
    const size_t globalRange = ((n + workGroupSize - 1) / workGroupSize) * workGroupSize;
    
    bool hasConverged = false;
    const int convergenceCheckFreq = 8;
    
    std::vector<float> hostSolution(n);
    std::vector<float> prevHostSolution(n, 0.0f);

    for (int iteration = 0; iteration < ITERATIONS && !hasConverged; ++iteration) {
        computeQueue.parallel_for(sycl::nd_range<1>(globalRange, workGroupSize),
            [=](sycl::nd_item<1> item) {
                size_t i = item.get_global_id(0);
                if (i >= static_cast<size_t>(n)) return;

                float rowSum = 0.0f;
                const size_t rowOffset = i * n;

                #pragma unroll(4)
                for (int j = 0; j < n; ++j) {
                    if (j != static_cast<int>(i)) {
                        rowSum += devMatrix[rowOffset + j] * devXcurrent[j];
                    }
                }
                devXnext[i] = devInvDiag[i] * (devRhs[i] - rowSum);
            });

        if ((iteration + 1) % convergenceCheckFreq == 0) {
            computeQueue.memcpy(hostSolution.data(), devXnext, sizeof(float) * n).wait();
            
            float squaredError = 0.0f;
            for (int i = 0; i < n; ++i) {
                float delta = hostSolution[i] - prevHostSolution[i];
                squaredError += delta * delta;
            }
            
            if (squaredError < squaredTolerance) {
                hasConverged = true;
                break;
            }
            
            prevHostSolution = hostSolution;
        }

        std::swap(devXcurrent, devXnext);
    }

    computeQueue.memcpy(hostSolution.data(), devXcurrent, sizeof(float) * n).wait();

    sycl::free(devMatrix,   computeQueue);
    sycl::free(devRhs,      computeQueue);
    sycl::free(devInvDiag,  computeQueue);
    sycl::free(devXcurrent, computeQueue);
    sycl::free(devXnext,    computeQueue);

    return hostSolution;
}