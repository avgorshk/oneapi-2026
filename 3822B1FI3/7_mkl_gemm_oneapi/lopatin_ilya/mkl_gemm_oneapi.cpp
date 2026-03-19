#include <mkl_gemm_oneapi.h>
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    size_t size, sycl::device device) {
    sycl::queue execution_queue(device);
    std::vector<float> result_matrix(size * size);
    {
        sycl::buffer<float> left_matrix_buffer(a.data(), a.size());
        sycl::buffer<float> right_matrix_buffer(b.data(), b.size());
        sycl::buffer<float> result_matrix_buffer(result_matrix.data(),
                                                 result_matrix.size());

        using oneapi::mkl::blas::row_major::gemm;
        using oneapi::mkl::transpose;

        gemm(execution_queue,
             transpose::nontrans,
             transpose::nontrans,
             size,
             size,
             size,
             1,
             left_matrix_buffer,
             size,
             right_matrix_buffer,
             size,
             0,
             result_matrix_buffer,
             size);
    }
    return result_matrix;
}
