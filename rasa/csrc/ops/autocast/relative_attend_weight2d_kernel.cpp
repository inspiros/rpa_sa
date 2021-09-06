#include "../relative_attend_weight2d.h"

#include <ATen/autocast_mode.h>
#include <torch/types.h>

namespace rasa {
    namespace ops {
        namespace {
            at::Tensor relative_attend_weight2d_autocast(
                    const at::Tensor &weight,
                    const at::Tensor &affinity,
                    int64_t height,
                    int64_t width) {
                c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
                return relative_attend_weight2d(
                        at::autocast::cached_cast(at::kFloat, weight),
                        at::autocast::cached_cast(at::kFloat, affinity),
                        height,
                        width)
                        .to(weight.scalar_type());
            }
        } // namespace

        TORCH_LIBRARY_IMPL(rasa, Autocast, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("rasa::relative_attend_weight2d"),
                    TORCH_FN(relative_attend_weight2d_autocast));
        }
    }
}
