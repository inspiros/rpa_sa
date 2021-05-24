#include "relative_attend_query2d.h"

#include <torch/types.h>

namespace rpa {
    namespace ops {
        at::Tensor relative_attend_query2d(
                const at::Tensor &query,
                const at::Tensor &affinity,
                int64_t height,
                int64_t width
        ) {
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("rpa::relative_attend_query2d", "")
                    .typed<decltype(relative_attend_query2d)>();
            return op.call(
                    query,
                    affinity,
                    height,
                    width);
        }

        namespace detail {
            std::tuple<at::Tensor, at::Tensor> _relative_attend_query2d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &query,
                    const at::Tensor &affinity,
                    int64_t height,
                    int64_t width
            ) {
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("rpa::_relative_attend_query2d_backward", "")
                                .typed<decltype(_relative_attend_query2d_backward)>();
                return op.call(
                        grad,
                        query,
                        affinity,
                        height,
                        width);
            }
        } // namespace detail

        TORCH_LIBRARY_FRAGMENT(rpa, m) {
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "rpa::relative_attend_query2d(Tensor query, Tensor affinity, int height, int width) -> Tensor")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "rpa::_relative_attend_query2d_backward(Tensor grad, Tensor query, Tensor affinity, int height, int width) -> (Tensor, Tensor)")
            );
        }
    } // namespace ops
} // namespace rpa
