#include "rotative_pad2d.h"

#include <torch/types.h>

namespace rpa_sa {
    namespace ops {
        at::Tensor rotative_pad2d(
                const at::Tensor &input,
                at::IntArrayRef pad,
                const std::string &interpolation
        ) {
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("rpa_sa::rotative_pad2d", "")
                    .typed<decltype(rotative_pad2d)>();
            return op.call(
                    input,
                    pad,
                    interpolation);
        }

        namespace detail {
            at::Tensor _rotative_pad2d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &input,
                    at::IntArrayRef pad,
                    const std::string &interpolation
            ) {
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("rpa_sa::_rotative_pad2d_backward", "")
                                .typed<decltype(_rotative_pad2d_backward)>();
                return op.call(
                        grad,
                        input,
                        pad,
                        interpolation);
            }
        } // namespace detail

        TORCH_LIBRARY_FRAGMENT(rpa_sa, m) {
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "rpa_sa::rotative_pad2d(Tensor input, int[] pad, str interpolation) -> Tensor")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "rpa_sa::_rotative_pad2d_backward(Tensor grad, Tensor input, int[] pad, str interpolation) -> Tensor")
            );
        }
    } // namespace ops
} // namespace rpa_sa
