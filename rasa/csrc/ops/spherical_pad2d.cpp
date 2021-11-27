#include "spherical_pad2d.h"

#include <torch/types.h>

namespace rasa {
    namespace ops {
        at::Tensor spherical_pad2d(
                const at::Tensor &input,
                at::IntArrayRef pad,
                const std::string &interpolation
        ) {
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("rasa::spherical_pad2d", "")
                    .typed<decltype(spherical_pad2d)>();
            return op.call(
                    input,
                    pad,
                    interpolation);
        }

        namespace detail {
            at::Tensor _spherical_pad2d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &input,
                    at::IntArrayRef pad,
                    const std::string &interpolation
            ) {
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("rasa::_spherical_pad2d_backward", "")
                                .typed<decltype(_spherical_pad2d_backward)>();
                return op.call(
                        grad,
                        input,
                        pad,
                        interpolation);
            }
        } // namespace detail

        TORCH_LIBRARY_FRAGMENT(rasa, m) {
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "rasa::spherical_pad2d(Tensor input, int[] pad, str interpolation) -> Tensor")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "rasa::_spherical_pad2d_backward(Tensor grad, Tensor input, int[] pad, str interpolation) -> Tensor")
            );
        }
    } // namespace ops
} // namespace rasa
