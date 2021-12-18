#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace rpa_sa {
    namespace ops {
        VISION_API at::Tensor relative_attend_weight2d(
                const at::Tensor &weight,
                const at::Tensor &affinity,
                int64_t height,
                int64_t width
        );

        namespace detail {
            std::tuple<at::Tensor, at::Tensor> _relative_attend_weight2d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &weight,
                    const at::Tensor &affinity,
                    int64_t height,
                    int64_t width
            );
        }
    }
}
