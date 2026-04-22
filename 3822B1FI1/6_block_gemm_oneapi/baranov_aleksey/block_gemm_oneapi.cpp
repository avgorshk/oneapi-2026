#include "block_gemm_oneapi.h"
#include <algorithm>

constexpr size_t BLOCK_SIZE = 16;
constexpr size_t PADDED_BLOCK = 17;

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        size_t size,
        sycl::device device) {
    
    sycl::queue computeQueue(device, sycl::property::queue::in_order{});

    float* devMatrixA = sycl::aligned_alloc_device<float>(64, size * size, computeQueue);
    float* devMatrixB = sycl::aligned_alloc_device<float>(64, size * size, computeQueue);
    float* devMatrixC = sycl::aligned_alloc_device<float>(64, size * size, computeQueue);

    computeQueue.memcpy(devMatrixA, a.data(), size * size * sizeof(float));
    computeQueue.memcpy(devMatrixB, b.data(), size * size * sizeof(float));
    computeQueue.memset(devMatrixC, 0, size * size * sizeof(float));
    computeQueue.wait();

    const size_t numBlocks = size / BLOCK_SIZE;

    computeQueue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 2> tileA(
            sycl::range<2>(BLOCK_SIZE, PADDED_BLOCK), cgh);
        sycl::local_accessor<float, 2> tileB(
            sycl::range<2>(PADDED_BLOCK, BLOCK_SIZE), cgh);

        cgh.parallel_for(sycl::nd_range<2>(
            sycl::range<2>(numBlocks * BLOCK_SIZE, numBlocks * BLOCK_SIZE),
            sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE)
        ), [=](sycl::nd_item<2> item) {
            const size_t blockRow = item.get_group(0);
            const size_t blockCol = item.get_group(1);
            const size_t threadRow = item.get_local_id(0);
            const size_t threadCol = item.get_local_id(1);

            const size_t globalRowBase = blockRow * BLOCK_SIZE;
            const size_t globalColBase = blockCol * BLOCK_SIZE;

            float accumulator = 0.0f;

            for (size_t blockK = 0; blockK < numBlocks; ++blockK) {
                const size_t aRow = globalRowBase + threadRow;
                const size_t aCol = blockK * BLOCK_SIZE + threadCol;
                tileA[threadRow][threadCol] = devMatrixA[aRow * size + aCol];

                const size_t bRow = blockK * BLOCK_SIZE + threadRow;
                const size_t bCol = globalColBase + threadCol;
                tileB[threadRow][threadCol] = devMatrixB[bRow * size + bCol];

                item.barrier(sycl::access::fence_space::local_space);

                #pragma unroll
                for (size_t k = 0; k < BLOCK_SIZE; ++k) {
                    accumulator += tileA[threadRow][k] * tileB[k][threadCol];
                }

                item.barrier(sycl::access::fence_space::local_space);
            }

            const size_t globalRow = globalRowBase + threadRow;
            const size_t globalCol = globalColBase + threadCol;
            devMatrixC[globalRow * size + globalCol] = accumulator;
        });
    }).wait();

    std::vector<float> result(size * size);
    computeQueue.memcpy(result.data(), devMatrixC, size * size * sizeof(float)).wait();

    sycl::free(devMatrixA, computeQueue);
    sycl::free(devMatrixB, computeQueue);
    sycl::free(devMatrixC, computeQueue);

    return result;
}