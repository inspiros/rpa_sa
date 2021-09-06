#include "rotative_pad2d.h"

#include <torch/types.h>

namespace rasa {
    namespace ops {
        at::Tensor rotative_pad2d(
                const at::Tensor &input,
                int64_t pad_l,
                int64_t pad_r,
                int64_t pad_u,
                int64_t pad_d,
                const std::string &interpolation
        ) {
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("rasa::rotative_pad2d", "")
                    .typed<decltype(rotative_pad2d)>();
            return op.call(
                    input,
                    pad_l,
                    pad_r,
                    pad_u,
                    pad_d,
                    interpolation);
        }

        namespace detail {
            at::Tensor _rotative_pad2d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &input,
                    int64_t pad_l,
                    int64_t pad_r,
                    int64_t pad_u,
                    int64_t pad_d,
                    const std::string &interpolation
            ) {
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("rasa::_rotative_pad2d_backward", "")
                                .typed<decltype(_rotative_pad2d_backward)>();
                return op.call(
                        grad,
                        input,
                        pad_l,
                        pad_r,
                        pad_u,
                        pad_d,
                        interpolation);
            }
        } // namespace detail

        TORCH_LIBRARY_FRAGMENT(rasa, m) {
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "rasa::rotative_pad2d(Tensor input, int pad_l, int pad_r, int pad_u, int pad_d, str interpolation) -> Tensor")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "rasa::_rotative_pad2d_backward(Tensor grad, Tensor input, int pad_l, int pad_r, int pad_u, int pad_d, str interpolation) -> Tensor")
            );
        }
    } // namespace ops
} // namespace rasa
