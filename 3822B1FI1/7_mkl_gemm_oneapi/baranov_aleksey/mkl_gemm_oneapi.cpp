#include "mkl_gemm_oneapi.h"

#include <cstdint>
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        size_t size,
        sycl::device device) {
    sycl::queue computeQueue(device, sycl::property::queue::in_order{});

    const size_t totalElements = size * size;
    const std::int64_t matrixSize = static_cast<std::int64_t>(size);
    const std::int64_t leadingDim = static_cast<std::int64_t>(size);
    const float one = 1.0f;
    const float zero = 0.0f;

    std::vector<float> result(totalElements);

    float* devMatrixA = sycl::aligned_alloc_device<float>(64, totalElements, computeQueue);
    float* devMatrixB = sycl::aligned_alloc_device<float>(64, totalElements, computeQueue);
    float* devMatrixC = sycl::aligned_alloc_device<float>(64, totalElements, computeQueue);

    computeQueue.memcpy(devMatrixA, a.data(), totalElements * sizeof(float));
    computeQueue.memcpy(devMatrixB, b.data(), totalElements * sizeof(float));

    oneapi::mkl::blas::row_major::gemm(
            computeQueue,
            oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans,
            matrixSize,
            matrixSize,
            matrixSize,
            one,
            devMatrixA,
            leadingDim,
            devMatrixB,
            leadingDim,
            zero,
            devMatrixC,
            leadingDim);

    computeQueue.memcpy(result.data(), devMatrixC, totalElements * sizeof(float)).wait();

    sycl::free(devMatrixA, computeQueue);
    sycl::free(devMatrixB, computeQueue);
    sycl::free(devMatrixC, computeQueue);

    return result;
}