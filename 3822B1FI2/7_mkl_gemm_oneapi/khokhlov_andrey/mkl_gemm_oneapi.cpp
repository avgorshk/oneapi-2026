#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>
#include <vector>

std::vector<float> GemmMklONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        size_t size,
        sycl::device device) {
    
    sycl::queue compute_queue(device);
    
    const size_t matrix_dim = size;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    const size_t total_entries = matrix_dim * matrix_dim;
    std::vector<float> result(total_entries, 0.0f);
    
    sycl::buffer<float, 1> matrix_a_buffer(a.data(), sycl::range<1>(total_entries));
    sycl::buffer<float, 1> matrix_b_buffer(b.data(), sycl::range<1>(total_entries));
    sycl::buffer<float, 1> matrix_c_buffer(result.data(), sycl::range<1>(total_entries));
    
    oneapi::mkl::blas::row_major::gemm(
        compute_queue,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        matrix_dim,
        matrix_dim,
        matrix_dim,
        alpha,
        matrix_a_buffer,
        matrix_dim,
        matrix_b_buffer,
        matrix_dim,
        beta,
        matrix_c_buffer,
        matrix_dim
    );
    
    compute_queue.wait_and_throw();
    
    return result;
}