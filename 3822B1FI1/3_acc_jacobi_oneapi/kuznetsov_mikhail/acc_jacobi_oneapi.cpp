#include "acc_jacobi_oneapi.h"

std::vector<float> JacobiAccONEAPI(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float accuracy,
    sycl::device device)
{
    const size_t n = b.size();

    std::vector<float> x_init(n, 0.0f);
    std::vector<float> result(n, 0.0f);

    sycl::queue q(device);

    sycl::buffer<float, 1> mat(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float, 1> vec_b(b.data(), sycl::range<1>(n));

    sycl::buffer<float, 1> x_old(x_init.data(), sycl::range<1>(n));
    sycl::buffer<float, 1> x_new(sycl::range<1>(n));

    auto* prev = &x_old;
    auto* curr = &x_new;

    for (int it = 0; it < ITERATIONS; ++it)
    {
        sycl::buffer<float, 1> diff(sycl::range<1>(1));

        q.submit([&](sycl::handler& h)
        {
            auto A = mat.get_access<sycl::access::mode::read>(h);
            auto B = vec_b.get_access<sycl::access::mode::read>(h);

            auto Xp = prev->get_access<sycl::access::mode::read>(h);
            auto Xc = curr->get_access<sycl::access::mode::write>(h);

            auto red = sycl::reduction(diff, h, sycl::maximum<float>());

            h.parallel_for(sycl::range<1>(n), red,
                [=](sycl::id<1> idx, auto& maxv)
                {
                    size_t i = idx[0];

                    float s = 0.0f;

                    for (size_t j = 0; j < n; ++j)
                        if (j != i)
                            s += A[i * n + j] * Xp[j];

                    float d = A[i * n + i];

                    float x =
                        (sycl::fabs(d) < 1e-12f)
                            ? Xp[i]
                            : (B[i] - s) / d;

                    Xc[i] = x;

                    maxv.combine(sycl::fabs(x - Xp[i]));
                });
        });

        q.wait();

        float max_diff = sycl::host_accessor(diff, sycl::read_only)[0];

        std::swap(prev, curr);

        if (max_diff < accuracy)
            break;
    }

    sycl::host_accessor out(*prev, sycl::read_only);

    for (size_t i = 0; i < n; ++i)
        result[i] = out[i];

    return result;
}