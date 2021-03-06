#include <ATen/ATen.h>
#include <torch/library.h>

#include "cuda_helpers.h"

namespace rpa_sa {
    namespace ops {
        namespace {
            template<typename scalar_t>
            __global__ void relative_attend_weight2d_forward_kernel_impl(
                    const at::GenericPackedTensorAccessor<scalar_t, 3> weight,
                    const at::GenericPackedTensorAccessor<scalar_t, 3> affinity,
                    const int64_t height,
                    const int64_t width,
                    const int64_t batch_sz,
                    const int64_t channels,
                    const int64_t weight_h,
                    const int64_t weight_w,
                    at::GenericPackedTensorAccessor<scalar_t, 3> output
            ) {
                CUDA_1D_KERNEL_LOOP(index, batch_sz * channels * weight_h) {
                    int64_t i = index % weight_h;
                    int64_t k = (index / weight_h) % channels;
                    int64_t b = index / (weight_h * channels);

                    scalar_t val = 0;
                    for (int64_t j = 0; j < weight_w; j++) {
                        int64_t aw = j % width - i % width + width - 1;
                        int64_t ah = j / width - i / width + height - 1;
                        val += weight[b][i][j] * affinity[k][ah][aw];
                    }
                    output[b][k][i] = val;
                }
            }

            template<typename scalar_t>
            __global__ void relative_attend_weight2d_backward_kernel_impl(
                    const at::GenericPackedTensorAccessor<scalar_t, 3> grad_output,
                    const at::GenericPackedTensorAccessor<scalar_t, 3> weight,
                    const at::GenericPackedTensorAccessor<scalar_t, 3> affinity,
                    const int64_t height,
                    const int64_t width,
                    const int64_t batch_sz,
                    const int64_t channels,
                    const int64_t weight_h,
                    const int64_t weight_w,
                    at::GenericPackedTensorAccessor<scalar_t, 3> grad_weight,
                    at::GenericPackedTensorAccessor<scalar_t, 3> grad_affinity
            ) {
                CUDA_1D_KERNEL_LOOP(index, batch_sz * channels * weight_h) {
                    int64_t i = index % weight_h;
                    int64_t k = (index / weight_h) % channels;
                    int64_t b = index / (weight_h * channels);

                    scalar_t grad_val = grad_output[b][k][i];
                    for (int64_t j = 0; j < weight_w; j++) {
                        int64_t aw = j % width - i % width + width - 1;
                        int64_t ah = j / width - i / width + height - 1;
                        gpuAtomicAdd(&grad_weight[b][i][j], grad_val * affinity[k][ah][aw]);
                        gpuAtomicAdd(&grad_affinity[k][ah][aw], grad_val * weight[b][i][j]);
                    }
                }
            }

            at::Tensor relative_attend_weight2d_forward_kernel(
                    const at::Tensor &weight,
                    const at::Tensor &affinity,
                    int64_t height,
                    int64_t width
            ) {
                int64_t batch_sz = weight.size(0);
                int64_t channels = affinity.size(0);
                int64_t weight_h = weight.size(1);
                int64_t weight_w = weight.size(2);

                auto output = at::empty({batch_sz,
                                         channels,
                                         weight_h},
                                        weight.options());

                const unsigned int threads = GET_THREADS();
                const unsigned int blocks = GET_BLOCKS(
                        threads,
                        batch_sz * channels * weight_h);
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        weight.scalar_type(), "relative_attend_weight2d_forward", ([&] {
                            auto output_accessor = output.generic_packed_accessor<scalar_t, 3>();
                            relative_attend_weight2d_forward_kernel_impl<<<blocks, threads>>>(
                                    weight.generic_packed_accessor<scalar_t, 3>(),
                                    affinity.generic_packed_accessor<scalar_t, 3>(),
                                    height,
                                    width,
                                    batch_sz,
                                    channels,
                                    weight_h,
                                    weight_w,
                                    output_accessor
                            );
                        }));
                return output;
            }

            std::tuple<at::Tensor, at::Tensor> relative_attend_weight2d_backward_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &weight,
                    const at::Tensor &affinity,
                    int64_t height,
                    int64_t width
            ) {
                int64_t batch_sz = weight.size(0);
                int64_t channels = affinity.size(0);
                int64_t weight_h = weight.size(1);
                int64_t weight_w = weight.size(2);

                auto grad_weight = at::zeros_like(weight);
                auto grad_affinity = at::zeros_like(affinity);

                const unsigned int threads = GET_THREADS();
                const unsigned int blocks = GET_BLOCKS(
                        threads,
                        batch_sz * channels * weight_h);
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        weight.scalar_type(), "relative_attend_weight2d_backward", ([&] {
                            auto grad_weight_accessor = grad_weight.generic_packed_accessor<scalar_t, 3>();
                            auto grad_affinity_accessor = grad_affinity.generic_packed_accessor<scalar_t, 3>();
                            relative_attend_weight2d_backward_kernel_impl<<<blocks, threads>>>(
                                    grad_output.generic_packed_accessor<scalar_t, 3>(),
                                    weight.generic_packed_accessor<scalar_t, 3>(),
                                    affinity.generic_packed_accessor<scalar_t, 3>(),
                                    height,
                                    width,
                                    batch_sz,
                                    channels,
                                    weight_h,
                                    weight_w,
                                    grad_weight_accessor,
                                    grad_affinity_accessor
                            );
                        }));
                return std::make_tuple(grad_weight, grad_affinity);
            }
        }

        TORCH_LIBRARY_IMPL(rpa_sa, CUDA, m) {
        m.impl(
                TORCH_SELECTIVE_NAME("rpa_sa::relative_attend_weight2d"),
                TORCH_FN(relative_attend_weight2d_forward_kernel));
        m.impl(
                TORCH_SELECTIVE_NAME("rpa_sa::_relative_attend_weight2d_backward"),
                TORCH_FN(relative_attend_weight2d_backward_kernel));
    }
}
}
