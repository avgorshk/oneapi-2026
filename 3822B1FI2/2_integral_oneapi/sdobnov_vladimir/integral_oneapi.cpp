#include "integral_oneapi.h"

#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
	sycl::queue q(device);

	const float step = (end - start) / count;
	const int total = count * count;

	float result = 0.0f;

	{
		sycl::buffer<float> result_buf(&result, sycl::range<1>(1));

		q.submit([&](sycl::handler& h) {
			auto reduction = sycl::reduction(
				result_buf, h, 0.0f, std::plus<>()
			);

			h.parallel_for(
				sycl::range<1>(total),
				reduction,
				[=](sycl::id<1> id, auto& sum) {
					int idx = id[0];

					int i = idx / count;
					int j = idx % count;

					float x = start + (i + 0.5f) * step;
					float y = start + (j + 0.5f) * step;

					sum += sycl::sin(x) * sycl::cos(y) * step * step;
				}
			);
			});
	}

	return result;
}