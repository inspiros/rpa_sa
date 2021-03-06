#include <ATen/ATen.h>
#include <torch/library.h>

#include "cpu_helpers.h"
#include "ibst.h"
#include "../common/spherical_pad_interpolation_method.h"

namespace rpa_sa {
    namespace ops {
        namespace {
            std::tuple<at::Tensor, at::Tensor> get_border_inds_and_thetas2d(
                    const at::Tensor &input,
                    const bool circular = true
            ) {
                int64_t width = input.size(-1);
                int64_t height = input.size(-2);

                at::Tensor border_mask = at::ones({height, width}, at::TensorOptions(at::kBool));
                border_mask.slice(0, 1, -1).slice(1, 1, -1).logical_not_();
                at::Tensor border_inds = at::stack(at::where(border_mask)).t();

                at::Tensor border_coords = border_inds.to(input.dtype());
                border_coords.slice(1, 0, 1).sub_(static_cast<double_t>(height - 1) / 2.0);
                border_coords.slice(1, 1, 2).sub_(static_cast<double_t>(width - 1) / 2.0);
                at::Tensor border_thetas = at::atan2(
                        border_coords.slice(1, 0, 1),
                        border_coords.slice(1, 1, 2)
                ).view(-1);

                // sort
                auto reverse_sorted_inds = border_thetas.argsort().argsort();
                border_thetas.scatter_(0, reverse_sorted_inds, border_thetas.clone());
                border_inds.scatter_(0, reverse_sorted_inds.repeat({2, 1}).t(), border_inds.clone());

                if (circular) {
                    border_inds = at::constant_pad_nd(border_inds, {0, 0, 1, 1});
                    border_inds[0] = border_inds[-2];
                    border_inds[-1] = border_inds[1];

                    border_thetas = at::constant_pad_nd(border_thetas, {1, 1});
                    border_thetas[0] = border_thetas[-2] - 2 * M_PI;
                    border_thetas[-1] = border_thetas[1] + 2 * M_PI;
                }
                return std::make_tuple(border_inds, border_thetas);
            }

            template<typename scalar_t, SphericalPadInterpolationMethod interp_method>
            void spherical_pad2d_forward_kernel_impl(
                    const at::TensorAccessor<scalar_t, 4> input,
                    const at::TensorAccessor<int64_t, 2> border_inds,
                    const at::TensorAccessor<scalar_t, 1> border_thetas,
                    const IBST<scalar_t> &ibst,
                    const scalar_t center_y,
                    const scalar_t center_x,
                    const int64_t batch_sz,
                    const int64_t channels,
                    const int64_t height,
                    const int64_t width,
                    const int64_t pad_u,
                    const int64_t pad_l,
                    const int64_t out_h,
                    const int64_t out_w,
                    at::TensorAccessor<scalar_t, 4> output
            ) {
                CPU_1D_KERNEL_LOOP(index, batch_sz * channels * out_h * out_w) {
                    int64_t j = index % out_w;
                    int64_t i = (index / out_w) % out_h;
                    int64_t c = (index / (out_w * out_h)) % channels;
                    int64_t b = index / (out_w * out_h * channels);

                    if (i < pad_u || i >= height + pad_u || j < pad_l || j >= width + pad_l) {
                        scalar_t dy = i - center_y;
                        scalar_t dx = j - center_x;
                        scalar_t theta = atan2(dy, dx);

                        int64_t theta_l_ind = ibst.get_index(theta);
                        int64_t theta_h_ind = theta_l_ind + 1;
                        scalar_t theta_l = border_thetas[theta_l_ind];
                        scalar_t theta_h = border_thetas[theta_h_ind];
                        scalar_t delta_theta_l = theta - theta_l;
                        scalar_t delta_theta_h = theta_h - theta;
                        scalar_t delta_theta_lh = theta_h - theta_l;

                        if (interp_method == SphericalPadInterpolationMethod::alerp) {
                            int64_t i_in_l = border_inds[theta_l_ind][0];
                            int64_t j_in_l = border_inds[theta_l_ind][1];
                            int64_t i_in_h = border_inds[theta_h_ind][0];
                            int64_t j_in_h = border_inds[theta_h_ind][1];
                            scalar_t val = delta_theta_h * input[b][c][i_in_l][j_in_l] +
                                           delta_theta_l * input[b][c][i_in_h][j_in_h];
                            output[b][c][i][j] = val / delta_theta_lh;
                        } else if (interp_method == SphericalPadInterpolationMethod::slerp) {
                            int64_t i_in_l = border_inds[theta_l_ind][0];
                            int64_t j_in_l = border_inds[theta_l_ind][1];
                            int64_t i_in_h = border_inds[theta_h_ind][0];
                            int64_t j_in_h = border_inds[theta_h_ind][1];
                            scalar_t val = sin(delta_theta_h) * input[b][c][i_in_l][j_in_l] +
                                           sin(delta_theta_l) * input[b][c][i_in_h][j_in_h];
                            output[b][c][i][j] = val / sin(delta_theta_lh);
                        } else {
                            int64_t ind_n = (delta_theta_l <= delta_theta_h) ? theta_l_ind : theta_h_ind;
                            int64_t i_in_n = border_inds[ind_n][0];
                            int64_t j_in_n = border_inds[ind_n][1];
                            output[b][c][i][j] = input[b][c][i_in_n][j_in_n];
                        }
                    } else {
                        output[b][c][i][j] = input[b][c][i - pad_u][j - pad_l];
                    }
                }
            }

            template<typename scalar_t, SphericalPadInterpolationMethod interp_method>
            void spherical_pad2d_backward_kernel_impl(
                    const at::TensorAccessor<scalar_t, 4> grad_output,
                    const at::TensorAccessor<int64_t, 2> border_inds,
                    const at::TensorAccessor<scalar_t, 1> border_thetas,
                    const IBST<scalar_t> &ibst,
                    const scalar_t center_y,
                    const scalar_t center_x,
                    const int64_t batch_sz,
                    const int64_t channels,
                    const int64_t height,
                    const int64_t width,
                    const int64_t pad_u,
                    const int64_t pad_l,
                    const int64_t out_h,
                    const int64_t out_w,
                    at::TensorAccessor<scalar_t, 4> grad_input
            ) {
                CPU_1D_KERNEL_LOOP(index, batch_sz * channels * out_h * out_w) {
                    int64_t j = index % out_w;
                    int64_t i = (index / out_w) % out_h;
                    int64_t c = (index / (out_w * out_h)) % channels;
                    int64_t b = index / (out_w * out_h * channels);

                    scalar_t grad_val = grad_output[b][c][i][j];
                    if (i < pad_u || i >= height + pad_u || j < pad_l || j >= width + pad_l) {
                        scalar_t dy = i - center_y;
                        scalar_t dx = j - center_x;
                        scalar_t theta = atan2(dy, dx);

                        int64_t theta_l_ind = ibst.get_index(theta);
                        int64_t theta_h_ind = theta_l_ind + 1;
                        scalar_t theta_l = border_thetas[theta_l_ind];
                        scalar_t theta_h = border_thetas[theta_h_ind];
                        scalar_t delta_theta_l = theta - theta_l;
                        scalar_t delta_theta_h = theta_h - theta;
                        scalar_t delta_theta_lh = theta_h - theta_l;

                        if (interp_method == SphericalPadInterpolationMethod::alerp) {
                            int64_t i_in_l = border_inds[theta_l_ind][0];
                            int64_t j_in_l = border_inds[theta_l_ind][1];
                            int64_t i_in_h = border_inds[theta_h_ind][0];
                            int64_t j_in_h = border_inds[theta_h_ind][1];
                            grad_input[b][c][i_in_l][j_in_l] +=
                                    grad_val * delta_theta_h / delta_theta_lh;
                            grad_input[b][c][i_in_h][j_in_h] +=
                                    grad_val * delta_theta_l / delta_theta_lh;
                        } else if (interp_method == SphericalPadInterpolationMethod::slerp) {
                            int64_t i_in_l = border_inds[theta_l_ind][0];
                            int64_t j_in_l = border_inds[theta_l_ind][1];
                            int64_t i_in_h = border_inds[theta_h_ind][0];
                            int64_t j_in_h = border_inds[theta_h_ind][1];
                            grad_input[b][c][i_in_l][j_in_l] +=
                                    grad_val * sin(delta_theta_h) / sin(delta_theta_lh);
                            grad_input[b][c][i_in_h][j_in_h] +=
                                    grad_val * sin(delta_theta_l) / sin(delta_theta_lh);
                        } else {
                            int64_t ind_n = (delta_theta_l <= delta_theta_h) ? theta_l_ind : theta_h_ind;
                            int64_t i_in_n = border_inds[ind_n][0];
                            int64_t j_in_n = border_inds[ind_n][1];
                            grad_input[b][c][i_in_n][j_in_n] += grad_val;
                        }
                    } else {
                        grad_input[b][c][i - pad_u][j - pad_l] += grad_val;
                    }
                }
            }

            at::Tensor spherical_pad2d_forward_kernel(
                    const at::Tensor &input,
                    at::IntArrayRef pad,
                    const std::string &interpolation
            ) {
                int64_t batch_sz = input.size(0);
                int64_t channels = input.size(1);
                int64_t height = input.size(2);
                int64_t width = input.size(3);

                int64_t pad_l = pad.at(0);
                int64_t pad_r = pad.at(1);
                int64_t pad_u = pad.at(2);
                int64_t pad_d = pad.at(3);

                int64_t out_h = height + pad_u + pad_d;
                int64_t out_w = width + pad_l + pad_r;

                auto output = at::empty({batch_sz,
                                         channels,
                                         out_h,
                                         out_w},
                                        input.options());

                auto border_inds_and_thetas = get_border_inds_and_thetas2d(input);
                auto border_inds = std::get<0>(border_inds_and_thetas);
                auto border_thetas = std::get<1>(border_inds_and_thetas);

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        input.scalar_type(), "spherical_pad2d_forward", ([&] {
                    scalar_t center_y = pad_u + static_cast<scalar_t>(height - 1) / 2.0;
                    scalar_t center_x = pad_l + static_cast<scalar_t>(width - 1) / 2.0;
                    auto output_accessor = output.accessor<scalar_t, 4>();
                    auto ibst = IBST<scalar_t>(border_thetas);
                    RPA_SA_SPHERICAL_PAD_INTERP_METHOD_OPTION(
                            get_interpolation_method(interpolation), ([&] {
                                spherical_pad2d_forward_kernel_impl<scalar_t, interp_method>(
                                        input.accessor<scalar_t, 4>(),
                                        border_inds.accessor<int64_t, 2>(),
                                        border_thetas.accessor<scalar_t, 1>(),
                                        ibst,
                                        center_y,
                                        center_x,
                                        batch_sz,
                                        channels,
                                        height,
                                        width,
                                        pad_u,
                                        pad_l,
                                        out_h,
                                        out_w,
                                        output_accessor
                                );
                            }));
                }));
                return output;
            }

            at::Tensor spherical_pad2d_backward_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &input,
                    at::IntArrayRef pad,
                    const std::string &interpolation
            ) {
                int64_t batch_sz = input.size(0);
                int64_t channels = input.size(1);
                int64_t height = input.size(2);
                int64_t width = input.size(3);

                int64_t pad_l = pad.at(0);
                int64_t pad_r = pad.at(1);
                int64_t pad_u = pad.at(2);
                int64_t pad_d = pad.at(3);

                int64_t out_h = height + pad_u + pad_d;
                int64_t out_w = width + pad_l + pad_r;

                auto grad_input = at::zeros({batch_sz,
                                             channels,
                                             height,
                                             width},
                                            input.options());

                auto border_inds_and_thetas = get_border_inds_and_thetas2d(input);
                auto border_inds = std::get<0>(border_inds_and_thetas);
                auto border_thetas = std::get<1>(border_inds_and_thetas);

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        input.scalar_type(), "spherical_pad2d_backward", ([&] {
                    scalar_t center_y = pad_u + static_cast<scalar_t>(height - 1) / 2.0;
                    scalar_t center_x = pad_l + static_cast<scalar_t>(width - 1) / 2.0;
                    auto grad_input_accessor = grad_input.accessor<scalar_t, 4>();
                    auto ibst = IBST<scalar_t>(border_thetas);
                    RPA_SA_SPHERICAL_PAD_INTERP_METHOD_OPTION(
                            get_interpolation_method(interpolation), ([&] {
                                spherical_pad2d_backward_kernel_impl<scalar_t, interp_method>(
                                        grad_output.accessor<scalar_t, 4>(),
                                        border_inds.accessor<int64_t, 2>(),
                                        border_thetas.accessor<scalar_t, 1>(),
                                        ibst,
                                        center_y,
                                        center_x,
                                        batch_sz,
                                        channels,
                                        height,
                                        width,
                                        pad_u,
                                        pad_l,
                                        out_h,
                                        out_w,
                                        grad_input_accessor
                                );
                            }));
                }));
                return grad_input;
            }
        }

        TORCH_LIBRARY_IMPL(rpa_sa, CPU, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("rpa_sa::spherical_pad2d"),
                    TORCH_FN(spherical_pad2d_forward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("rpa_sa::_spherical_pad2d_backward"),
                    TORCH_FN(spherical_pad2d_backward_kernel));
        }
    }
}
