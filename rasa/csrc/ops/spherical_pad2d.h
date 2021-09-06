#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace rasa {
    namespace ops {
        VISION_API at::Tensor spherical_pad2d(
                const at::Tensor &input,
                int64_t pad_l,
                int64_t pad_r,
                int64_t pad_u,
                int64_t pad_d,
                const std::string &interpolation = "nearest"
        );

        namespace detail {
            at::Tensor _spherical_pad2d_backward(
                    const at::Tensor &grad_output,
                    const at::Tensor &input,
                    int64_t pad_l,
                    int64_t pad_r,
                    int64_t pad_u,
                    int64_t pad_d,
                    const std::string &interpolation = "nearest"
            );
        }
    }
}
