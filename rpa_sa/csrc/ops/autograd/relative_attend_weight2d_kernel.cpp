#include "../relative_attend_weight2d.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace rpa_sa {
    namespace ops {
        namespace {
            class RelativeAttendWeight2dFunction
                    : public torch::autograd::Function<RelativeAttendWeight2dFunction> {
            public:
                static torch::autograd::variable_list forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &weight,
                        const torch::autograd::Variable &affinity,
                        int64_t height,
                        int64_t width) {
                    at::AutoDispatchBelowADInplaceOrView g;
                    auto output = relative_attend_weight2d(
                            weight,
                            affinity,
                            height,
                            width);

                    ctx->save_for_backward({weight, affinity});
                    ctx->saved_data["height"] = height;
                    ctx->saved_data["width"] = width;

                    return {
                            output,
                    };
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {

                    auto saved = ctx->get_saved_variables();
                    auto weight = saved[0];
                    auto affinity = saved[1];
                    auto height = ctx->saved_data["height"].toInt();
                    auto width = ctx->saved_data["width"].toInt();

                    auto result = detail::_relative_attend_weight2d_backward(
                            grad_output[0],
                            weight,
                            affinity,
                            height,
                            width);

                    auto grad_weight = std::get<0>(result);
                    auto grad_affinity = std::get<1>(result);

                    return {
                            grad_weight,
                            grad_affinity,
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                    };
                }
            };

            class RelativeAttendWeight2dBackwardFunction
                    : public torch::autograd::Function<RelativeAttendWeight2dBackwardFunction> {
            public:
                static torch::autograd::variable_list forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &grad_output,
                        const torch::autograd::Variable &weight,
                        const torch::autograd::Variable &affinity,
                        int64_t height,
                        int64_t width) {
                    at::AutoDispatchBelowADInplaceOrView g;
                    auto result = detail::_relative_attend_weight2d_backward(
                            grad_output,
                            weight,
                            affinity,
                            height,
                            width);

                    auto grad_weight = std::get<0>(result);
                    auto grad_affinity = std::get<1>(result);

                    ctx->save_for_backward({weight, affinity});
                    ctx->saved_data["height"] = height;
                    ctx->saved_data["width"] = width;

                    return {
                            grad_weight,
                            grad_affinity,
                    };
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {
                    TORCH_CHECK(0, "double backwards on relative_attend_weight2d not supported");
                }
            };
        } // namespace

        at::Tensor relative_attend_weight2d_autograd(
                const at::Tensor &weight,
                const at::Tensor &affinity,
                int64_t height,
                int64_t width
        ) {
            return RelativeAttendWeight2dFunction::apply(
                    weight,
                    affinity,
                    height,
                    width
            )[0];
        }

        std::tuple<at::Tensor, at::Tensor> relative_attend_weight2d_backward_autograd(
                const at::Tensor &grad,
                const at::Tensor &weight,
                const at::Tensor &affinity,
                int64_t height,
                int64_t width
        ) {
            auto result = RelativeAttendWeight2dBackwardFunction::apply(
                    grad,
                    weight,
                    affinity,
                    height,
                    width
            );

            return std::make_tuple(result[0], result[1]);
        }

        TORCH_LIBRARY_IMPL(rpa_sa, Autograd, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("rpa_sa::relative_attend_weight2d"),
                    TORCH_FN(relative_attend_weight2d_autograd));
            m.impl(
                    TORCH_SELECTIVE_NAME("rpa_sa::_relative_attend_weight2d_backward"),
                    TORCH_FN(relative_attend_weight2d_backward_autograd));
        }
    }
}
