#include <ATen/ATen.h>
#include <torch/library.h>

#include "cpu_helpers.h"

namespace rasa {
    namespace ops {
        namespace {
            template<typename scalar_t>
            void relative_attend_query2d_forward_kernel_impl(
                    const at::TensorAccessor<scalar_t, 3> query,
                    const at::TensorAccessor<scalar_t, 3> affinity,
                    const int64_t height,
                    const int64_t width,
                    const int64_t batch_sz,
                    const int64_t channels,
                    const int64_t out_h,
                    const int64_t out_w,
                    at::TensorAccessor<scalar_t, 3> output
            ) {
                CPU_1D_KERNEL_LOOP(index, batch_sz * out_h * out_w) {
                    int64_t j = index % out_w;
                    int64_t i = (index / out_w) % out_h;
                    int64_t b = index / (out_w * out_h);

                    int64_t aw = j % width - i % width + width - 1;
                    int64_t ah = j / width - i / width + height - 1;

                    scalar_t val = 0;
                    for (int64_t k = 0; k < channels; k++)
                        val += query[b][k][i] * affinity[k][ah][aw];
                    output[b][i][j] = val;
                }
            }

            template<typename scalar_t>
            void relative_attend_query2d_backward_kernel_impl(
                    const at::TensorAccessor<scalar_t, 3> grad_output,
                    const at::TensorAccessor<scalar_t, 3> query,
                    const at::TensorAccessor<scalar_t, 3> affinity,
                    const int64_t height,
                    const int64_t width,
                    const int64_t batch_sz,
                    const int64_t channels,
                    const int64_t out_h,
                    const int64_t out_w,
                    at::TensorAccessor<scalar_t, 3> grad_query,
                    at::TensorAccessor<scalar_t, 3> grad_affinity
            ) {
                CPU_1D_KERNEL_LOOP(index, batch_sz * out_h * out_w) {
                    int64_t j = index % out_w;
                    int64_t i = (index / out_w) % out_h;
                    int64_t b = index / (out_w * out_h);

                    int64_t aw = j % width - i % width + width - 1;
                    int64_t ah = j / width - i / width + height - 1;

                    scalar_t grad_val = grad_output[b][i][j];
                    for (int64_t k = 0; k < channels; k++) {
                        grad_query[b][k][i] += grad_val * affinity[k][ah][aw];
                        grad_affinity[k][ah][aw] += grad_val * query[b][k][i];
                    }
                }
            }

            at::Tensor relative_attend_query2d_forward_kernel(
                    const at::Tensor &query,
                    const at::Tensor &affinity,
                    int64_t height,
                    int64_t width
            ) {
                int64_t batch_sz = query.size(0);
                int64_t channels = query.size(1);
                int64_t out_h = height * width;
                int64_t out_w = height * width;

                auto output = at::empty({batch_sz,
                                         out_h,
                                         out_w},
                                        query.options());

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        query.scalar_type(), "relative_attend_query2d_forward", ([&] {
                    auto output_accessor = output.accessor<scalar_t, 3>();
                    relative_attend_query2d_forward_kernel_impl(
                            query.accessor<scalar_t, 3>(),
                            affinity.accessor<scalar_t, 3>(),
                            height,
                            width,
                            batch_sz,
                            channels,
                            out_h,
                            out_w,
                            output_accessor
                    );
                }));
                return output;
            }

            std::tuple<at::Tensor, at::Tensor> relative_attend_query2d_backward_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &query,
                    const at::Tensor &affinity,
                    int64_t height,
                    int64_t width
            ) {
                int64_t batch_sz = query.size(0);
                int64_t channels = query.size(1);
                int64_t out_h = height * width;
                int64_t out_w = height * width;

                auto grad_query = at::zeros_like(query);
                auto grad_affinity = at::zeros_like(affinity);

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        query.scalar_type(), "relative_attend_query2d_backward", ([&] {
                    auto grad_query_accessor = grad_query.accessor<scalar_t, 3>();
                    auto grad_affinity_accessor = grad_affinity.accessor<scalar_t, 3>();
                    relative_attend_query2d_backward_kernel_impl(
                            grad_output.accessor<scalar_t, 3>(),
                            query.accessor<scalar_t, 3>(),
                            affinity.accessor<scalar_t, 3>(),
                            height,
                            width,
                            batch_sz,
                            channels,
                            out_h,
                            out_w,
                            grad_query_accessor,
                            grad_affinity_accessor
                    );
                }));
                return std::make_tuple(grad_query, grad_affinity);
            }
        }

        TORCH_LIBRARY_IMPL(rasa, CPU, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("rasa::relative_attend_query2d"),
                    TORCH_FN(relative_attend_query2d_forward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("rasa::_relative_attend_query2d_backward"),
                    TORCH_FN(relative_attend_query2d_backward_kernel));
        }
    }
}
