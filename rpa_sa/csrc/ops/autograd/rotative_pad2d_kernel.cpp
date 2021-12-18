#include "../rotative_pad2d.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace rpa_sa {
    namespace ops {
        namespace {
            class RotativePad2dFunction
                    : public torch::autograd::Function<RotativePad2dFunction> {
            public:
                static torch::autograd::variable_list forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &input,
                        const at::IntArrayRef &pad,
                        const std::string &interpolation) {
                    at::AutoDispatchBelowADInplaceOrView g;
                    auto output = rotative_pad2d(
                            input,
                            pad,
                            interpolation);

                    ctx->save_for_backward({input});

//                    ctx->saved_data["pad"] = pad;
                    ctx->saved_data["pad_l"] = pad.at(0);
                    ctx->saved_data["pad_r"] = pad.at(1);
                    ctx->saved_data["pad_u"] = pad.at(2);
                    ctx->saved_data["pad_d"] = pad.at(3);
                    ctx->saved_data["interpolation"] = interpolation;

                    return {
                            output,
                    };
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {

                    auto saved = ctx->get_saved_variables();
                    auto input = saved[0];

//                    at::IntArrayRef pad = ctx->saved_data["pad"].toIntVector();
                    std::vector<int64_t> pad = {ctx->saved_data["pad_l"].toInt(),
                                                ctx->saved_data["pad_r"].toInt(),
                                                ctx->saved_data["pad_u"].toInt(),
                                                ctx->saved_data["pad_d"].toInt()};
                    auto interpolation = ctx->saved_data["interpolation"].toString()->string();

                    auto grad_input = detail::_rotative_pad2d_backward(
                            grad_output[0],
                            input,
                            pad,
                            interpolation);

                    return {
                            grad_input,
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                    };
                }
            };

            class RotativePad2dBackwardFunction
                    : public torch::autograd::Function<RotativePad2dBackwardFunction> {
            public:
                static torch::autograd::variable_list forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &grad_output,
                        const torch::autograd::Variable &input,
                        const at::IntArrayRef &pad,
                        const std::string &interpolation) {
                    at::AutoDispatchBelowADInplaceOrView g;
                    auto grad_input = detail::_rotative_pad2d_backward(
                            grad_output,
                            input,
                            pad,
                            interpolation);

                    ctx->save_for_backward({input});

//                    ctx->saved_data["pad"] = pad;
                    ctx->saved_data["pad_l"] = pad.at(0);
                    ctx->saved_data["pad_r"] = pad.at(1);
                    ctx->saved_data["pad_u"] = pad.at(2);
                    ctx->saved_data["pad_d"] = pad.at(3);
                    ctx->saved_data["interpolation"] = interpolation;

                    return {
                            grad_input,
                    };
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {
                    TORCH_CHECK(0, "double backwards on rotative_pad2d not supported");
                }
            };
        } // namespace

        at::Tensor rotative_pad2d_autograd(
                const at::Tensor &input,
                at::IntArrayRef pad,
                const std::string &interpolation
        ) {
            return RotativePad2dFunction::apply(
                    input,
                    pad,
                    interpolation
            )[0];
        }

        at::Tensor rotative_pad2d_backward_autograd(
                const at::Tensor &grad,
                const at::Tensor &input,
                at::IntArrayRef pad,
                const std::string &interpolation
        ) {
            return RotativePad2dBackwardFunction::apply(
                    grad,
                    input,
                    pad,
                    interpolation
            )[0];
        }

        TORCH_LIBRARY_IMPL(rpa_sa, Autograd, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("rpa_sa::rotative_pad2d"),
                    TORCH_FN(rotative_pad2d_autograd));
            m.impl(
                    TORCH_SELECTIVE_NAME("rpa_sa::_rotative_pad2d_backward"),
                    TORCH_FN(rotative_pad2d_backward_autograd));
        }
    }
}
