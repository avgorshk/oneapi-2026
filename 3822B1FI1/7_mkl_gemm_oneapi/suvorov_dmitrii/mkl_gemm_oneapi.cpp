#include "mkl_gemm_oneapi.h"

#include <oneapi/mkl.hpp>
#include <vector>

std::vector<float> GemmMklONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    if (size == 0) {
        return {};
    }

    if (a.size() != size * size || b.size() != size * size) {
        return {};
    }

    try {
        sycl::queue q(device, sycl::property::queue::in_order{});
        std::vector<float> c(size * size);

        {
            sycl::buffer<float, 1> a_buf(a.data(), sycl::range<1>(size * size));
            sycl::buffer<float, 1> b_buf(b.data(), sycl::range<1>(size * size));
            sycl::buffer<float, 1> c_buf(c.data(), sycl::range<1>(size * size));

            constexpr float alpha = 1.0f;
            constexpr float beta = 0.0f;

            oneapi::mkl::blas::row_major::gemm(
                q,
                oneapi::mkl::transpose::nontrans,
                oneapi::mkl::transpose::nontrans,
                static_cast<std::int64_t>(size),
                static_cast<std::int64_t>(size),
                static_cast<std::int64_t>(size),
                alpha,
                a_buf,
                static_cast<std::int64_t>(size),
                b_buf,
                static_cast<std::int64_t>(size),
                beta,
                c_buf,
                static_cast<std::int64_t>(size)
            );
        }

        return c;
    } catch (const oneapi::mkl::exception&) {
        return {};
    } catch (const sycl::exception&) {
        return {};
    }
}