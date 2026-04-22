#include "shared_jacobi_oneapi.h"
#include <algorithm>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    
    const int n = static_cast<int>(b.size());
    sycl::queue computeQueue(device);

    // Выделение общей (shared) памяти USM
    float* sharedMatrix = sycl::malloc_shared<float>(a.size(), computeQueue);
    float* sharedRhs    = sycl::malloc_shared<float>(n, computeQueue);
    float* sharedX      = sycl::malloc_shared<float>(n, computeQueue);
    float* sharedXnext  = sycl::malloc_shared<float>(n, computeQueue);
    float* sharedMaxDiff = sycl::malloc_shared<float>(1, computeQueue);

    std::copy(a.begin(), a.end(), sharedMatrix);
    std::copy(b.begin(), b.end(), sharedRhs);
    std::fill(sharedX, sharedX + n, 0.0f);

    // Оптимизация: подбор размера рабочей группы для эффективного использования SLM
    const size_t workGroupSize = 256;
    const size_t numGroups = (n + workGroupSize - 1) / workGroupSize;
    const size_t globalSize = numGroups * workGroupSize;

    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        *sharedMaxDiff = 0.0f;

        computeQueue.submit([&](sycl::handler& cgh) {
            // Локальная память для кэширования части вектора X (размер равен workGroupSize)
            sycl::local_accessor<float, 1> localX(workGroupSize, cgh);

            auto maxReducer = sycl::reduction(sharedMaxDiff, sycl::maximum<float>());

            cgh.parallel_for(
                sycl::nd_range<1>(globalSize, workGroupSize),
                maxReducer,
                [=](sycl::nd_item<1> item, auto& diff) {
                    const int i = item.get_global_id(0);
                    const int localId = item.get_local_id(0);
                    const int groupSize = item.get_local_range(0);

                    // Инициализация разности нулём, если поток за границами
                    float localDiff = 0.0f;

                    if (i < n) {
                        float rowSum = 0.0f;

                        // Цикл по блокам столбцов с использованием локальной памяти
                        for (int colBlock = 0; colBlock < n; colBlock += groupSize) {
                            // Кооперативная загрузка блока вектора X в SLM
                            const int col = colBlock + localId;
                            localX[localId] = (col < n) ? sharedX[col] : 0.0f;
                            item.barrier(sycl::access::fence_space::local_space);

                            // Вычисление частичного скалярного произведения для текущего блока
                            const int endCol = sycl::min(colBlock + groupSize, n);
                            #pragma unroll(4)
                            for (int j = colBlock; j < endCol; ++j) {
                                if (i != j) {
                                    rowSum += sharedMatrix[i * n + j] * localX[j - colBlock];
                                }
                            }
                            item.barrier(sycl::access::fence_space::local_space);
                        }

                        // Вычисление нового значения x_i
                        const float newX = (sharedRhs[i] - rowSum) / sharedMatrix[i * n + i];
                        sharedXnext[i] = newX;

                        // Локальная разность для редукции
                        localDiff = sycl::fabs(newX - sharedX[i]);
                    }

                    // Редукция (SYCL сам выполнит комбинацию через max)
                    diff.combine(localDiff);
                });
        }).wait();

        // Проверка сходимости каждую итерацию (без изменений)
        if (*sharedMaxDiff < accuracy) {
            std::swap(sharedX, sharedXnext);
            break;
        }

        std::swap(sharedX, sharedXnext);
    }

    std::vector<float> result(sharedX, sharedX + n);

    sycl::free(sharedMatrix, computeQueue);
    sycl::free(sharedRhs, computeQueue);
    sycl::free(sharedX, computeQueue);
    sycl::free(sharedXnext, computeQueue);
    sycl::free(sharedMaxDiff, computeQueue);

    return result;
}