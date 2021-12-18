#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace rpa_sa {
    namespace ops {
        VISION_API at::Tensor rotative_pad2d(
                const at::Tensor &input,
                at::IntArrayRef pad,
                const std::string &interpolation = "nearest"
        );

        namespace detail {
            at::Tensor _rotative_pad2d_backward(
                    const at::Tensor &grad_output,
                    const at::Tensor &input,
                    at::IntArrayRef pad,
                    const std::string &interpolation = "nearest"
            );
        }
    }
}
