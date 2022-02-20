#include <ATen/ATen.h>
#include <torch/library.h>

#include "cuda_helpers.h"
#include "../common/rotative_pad_interpolation_method.h"

namespace rpa_sa {
    namespace ops {
        namespace {
            template<typename scalar_t, RotativePadInterpolationMethod interp_method>
            __global__ void rotative_pad2d_forward_kernel_impl(
                    const at::GenericPackedTensorAccessor<scalar_t, 3> input,
                    const scalar_t center_y,
                    const scalar_t center_x,
                    const int64_t batch_sz,
                    const int64_t channels,
                    const int64_t width,
                    const int64_t out_h,
                    const int64_t out_w,
                    at::GenericPackedTensorAccessor<scalar_t, 4> output
            ) {
                CUDA_1D_KERNEL_LOOP(index, batch_sz * channels * out_h * out_w) {
                    int64_t j = index % out_w;
                    int64_t i = (index / out_w) % out_h;
                    int64_t c = (index / (out_w * out_h)) % channels;
                    int64_t b = index / (out_w * out_h * channels);

                    scalar_t dy = i - center_y;
                    scalar_t dx = j - center_x;
                    scalar_t d = sqrt(dy * dy + dx * dx);

                    if (interp_method == RotativePadInterpolationMethod::lerp) {
                        int64_t ind_l = min(static_cast<int64_t>(floor(d)), width - 1);
                        int64_t ind_h = ind_l + 1;
                        scalar_t d_l = d - ind_l;
                        scalar_t d_h = 1 - d_l;

                        output[b][c][i][j] = (ind_l < width - 1) ?
                                             d_h * input[b][c][ind_l] + d_l * input[b][c][ind_h]
                                                                 : input[b][c][width - 1];
                    } else {
                        int64_t ind_n = min(static_cast<int64_t>(round(d)), width - 1);
                        output[b][c][i][j] = input[b][c][ind_n];
                    }
                }
            }

            template<typename scalar_t, RotativePadInterpolationMethod interp_method>
            __global__ void rotative_pad2d_backward_kernel_impl(
                    const at::GenericPackedTensorAccessor<scalar_t, 4> grad_output,
                    const scalar_t center_y,
                    const scalar_t center_x,
                    const int64_t batch_sz,
                    const int64_t channels,
                    const int64_t width,
                    const int64_t out_h,
                    const int64_t out_w,
                    at::GenericPackedTensorAccessor<scalar_t, 3> grad_input
            ) {
                CUDA_1D_KERNEL_LOOP(index, batch_sz * channels * out_h * out_w) {
                    int64_t j = index % out_w;
                    int64_t i = (index / out_w) % out_h;
                    int64_t c = (index / (out_w * out_h)) % channels;
                    int64_t b = index / (out_w * out_h * channels);

                    scalar_t dy = i - center_y;
                    scalar_t dx = j - center_x;
                    scalar_t d = sqrt(dy * dy + dx * dx);

                    scalar_t grad_val = grad_output[b][c][i][j];
                    if (interp_method == RotativePadInterpolationMethod::lerp) {
                        int64_t ind_l = min(static_cast<int64_t>(floor(d)), width - 1);
                        int64_t ind_h = ind_l + 1;
                        if (ind_l < width - 1) {
                            scalar_t d_l = d - ind_l;
                            scalar_t d_h = 1 - d_l;
                            gpuAtomicAdd(&grad_input[b][c][ind_l], d_h * grad_val);
                            gpuAtomicAdd(&grad_input[b][c][ind_h], d_l * grad_val);
                        } else {
                            gpuAtomicAdd(&grad_input[b][c][width - 1], grad_val);
                        }
                    } else {
                        int64_t ind_n = min(static_cast<int64_t>(round(d)), width - 1);
                        gpuAtomicAdd(&grad_input[b][c][ind_n], grad_val);
                    }
                }
            }

            at::Tensor rotative_pad2d_forward_kernel(
                    const at::Tensor &input,
                    at::IntArrayRef pad,
                    const std::string &interpolation
            ) {
                at::DeviceGuard guard(input.device());

                int64_t batch_sz = input.size(0);
                int64_t channels = input.size(1);
                int64_t width = input.size(2);

                int64_t pad_l = pad.at(0);
                int64_t pad_r = pad.at(1);
                int64_t pad_u = pad.at(2);
                int64_t pad_d = pad.at(3);

                int64_t out_h = 1 + pad_u + pad_d;
                int64_t out_w = 1 + pad_l + pad_r;

                auto output = at::empty({batch_sz,
                                         channels,
                                         out_h,
                                         out_w},
                                        input.options());

                const unsigned int threads = GET_THREADS();
                const unsigned int blocks = GET_BLOCKS(
                        threads,
                        batch_sz * channels * out_h * out_w);
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        input.scalar_type(), "rotative_pad2d_forward", ([&] {
                            scalar_t center_y = pad_u;
                            scalar_t center_x = pad_l;
                            auto output_accessor = output.generic_packed_accessor<scalar_t, 4>();
                            RPA_SA_ROTATIVE_PAD_INTERP_METHOD_OPTION(
                                    get_interpolation_method(interpolation), ([&] {
                                        rotative_pad2d_forward_kernel_impl<scalar_t, interp_method><<<blocks, threads>>>(
                                                input.generic_packed_accessor<scalar_t, 3>(),
                                                center_y,
                                                center_x,
                                                batch_sz,
                                                channels,
                                                width,
                                                out_h,
                                                out_w,
                                                output_accessor
                                        );
                                    }));
                        }));
                AT_CUDA_CHECK(cudaGetLastError());
                return output;
            }

            at::Tensor rotative_pad2d_backward_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &input,
                    at::IntArrayRef pad,
                    const std::string &interpolation
            ) {
                at::DeviceGuard guard(input.device());

                int64_t batch_sz = input.size(0);
                int64_t channels = input.size(1);
                int64_t width = input.size(2);

                int64_t pad_l = pad.at(0);
                int64_t pad_r = pad.at(1);
                int64_t pad_u = pad.at(2);
                int64_t pad_d = pad.at(3);

                int64_t out_h = 1 + pad_u + pad_d;
                int64_t out_w = 1 + pad_l + pad_r;

                auto grad_input = at::zeros({batch_sz,
                                             channels,
                                             width},
                                            input.options());

                const unsigned int threads = GET_THREADS();
                const unsigned int blocks = GET_BLOCKS(
                        threads,
                        batch_sz * channels * out_h * out_w);
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        input.scalar_type(), "rotative_pad2d_backward", ([&] {
                            scalar_t center_y = pad_u;
                            scalar_t center_x = pad_l;
                            auto grad_input_accessor = grad_input.generic_packed_accessor<scalar_t, 3>();
                            RPA_SA_ROTATIVE_PAD_INTERP_METHOD_OPTION(
                                    get_interpolation_method(interpolation), ([&] {
                                        rotative_pad2d_backward_kernel_impl<scalar_t, interp_method><<<blocks, threads>>>(
                                                grad_output.generic_packed_accessor<scalar_t, 4>(),
                                                center_y,
                                                center_x,
                                                batch_sz,
                                                channels,
                                                width,
                                                out_h,
                                                out_w,
                                                grad_input_accessor
                                        );
                                    }));
                        }));
                AT_CUDA_CHECK(cudaGetLastError());
                return grad_input;
            }
        }

        TORCH_LIBRARY_IMPL(rpa_sa, CUDA, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("rpa_sa::rotative_pad2d"),
                    TORCH_FN(rotative_pad2d_forward_kernel)
            );
            m.impl(
                    TORCH_SELECTIVE_NAME("rpa_sa::_rotative_pad2d_backward"),
                    TORCH_FN(rotative_pad2d_backward_kernel)
            );
        }
    }
}
