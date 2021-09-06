#include "../relative_attend_query2d.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace rasa {
    namespace ops {
        namespace {
            class RelativeAttend2dFunction
                    : public torch::autograd::Function<RelativeAttend2dFunction> {
            public:
                static torch::autograd::variable_list forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &query,
                        const torch::autograd::Variable &affinity,
                        int64_t height,
                        int64_t width) {
                    at::AutoNonVariableTypeMode g;
                    auto output = relative_attend_query2d(
                            query,
                            affinity,
                            height,
                            width);

                    ctx->save_for_backward({query, affinity});
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
                    auto query = saved[0];
                    auto affinity = saved[1];
                    auto height = ctx->saved_data["height"].toInt();
                    auto width = ctx->saved_data["width"].toInt();

                    auto result = detail::_relative_attend_query2d_backward(
                            grad_output[0],
                            query,
                            affinity,
                            height,
                            width);

                    auto grad_query = std::get<0>(result);
                    auto grad_affinity = std::get<1>(result);

                    return {
                            grad_query,
                            grad_affinity,
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                    };
                }
            };

            class RelativeAttend2dBackwardFunction
                    : public torch::autograd::Function<RelativeAttend2dBackwardFunction> {
            public:
                static torch::autograd::variable_list forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &grad_output,
                        const torch::autograd::Variable &query,
                        const torch::autograd::Variable &affinity,
                        int64_t height,
                        int64_t width) {
                    at::AutoNonVariableTypeMode g;
                    auto result = detail::_relative_attend_query2d_backward(
                            grad_output,
                            query,
                            affinity,
                            height,
                            width);

                    auto grad_query = std::get<0>(result);
                    auto grad_affinity = std::get<1>(result);

                    ctx->save_for_backward({query, affinity});
                    ctx->saved_data["height"] = height;
                    ctx->saved_data["width"] = width;

                    return {
                            grad_query,
                            grad_affinity,
                    };
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {
                    TORCH_CHECK(0, "double backwards on relative_attend_query2d not supported");
                }
            };
        } // namespace

        at::Tensor relative_attend_query2d_autograd(
                const at::Tensor &query,
                const at::Tensor &affinity,
                int64_t height,
                int64_t width
        ) {
            return RelativeAttend2dFunction::apply(
                    query,
                    affinity,
                    height,
                    width
            )[0];
        }

        std::tuple<at::Tensor, at::Tensor> relative_attend_query2d_backward_autograd(
                const at::Tensor &grad,
                const at::Tensor &query,
                const at::Tensor &affinity,
                int64_t height,
                int64_t width
        ) {
            auto result = RelativeAttend2dBackwardFunction::apply(
                    grad,
                    query,
                    affinity,
                    height,
                    width
            );

            return std::make_tuple(result[0], result[1]);
        }

        TORCH_LIBRARY_IMPL(rasa, Autograd, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("rasa::relative_attend_query2d"),
                    TORCH_FN(relative_attend_query2d_autograd));
            m.impl(
                    TORCH_SELECTIVE_NAME("rasa::_relative_attend_query2d_backward"),
                    TORCH_FN(relative_attend_query2d_backward_autograd));
        }
    }
}
