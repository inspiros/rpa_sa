#include "../spherical_pad2d.h"

#include <ATen/autocast_mode.h>
#include <torch/types.h>

namespace rasa {
    namespace ops {
        namespace {
            at::Tensor spherical_pad2d_autocast(
                    const at::Tensor &input,
                    int64_t pad_l,
                    int64_t pad_r,
                    int64_t pad_u,
                    int64_t pad_d,
                    const std::string &interpolation) {
                c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
                return spherical_pad2d(
                        at::autocast::cached_cast(at::kFloat, input),
                        pad_l,
                        pad_r,
                        pad_u,
                        pad_d,
                        interpolation)
                        .to(input.scalar_type());
            }
        } // namespace

        TORCH_LIBRARY_IMPL(rasa, Autocast, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("rasa::spherical_pad2d"),
                    TORCH_FN(spherical_pad2d_autocast));
        }
    }
}
