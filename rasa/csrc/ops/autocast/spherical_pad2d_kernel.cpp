#include "../spherical_pad2d.h"

#include <ATen/autocast_mode.h>
#include <torch/types.h>

namespace rasa {
    namespace ops {
        namespace {
            at::Tensor spherical_pad2d_autocast(
                    const at::Tensor &input,
                    at::IntArrayRef pad,
                    const std::string &interpolation) {
                c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
                return spherical_pad2d(
                        at::autocast::cached_cast(at::kFloat, input),
                        pad,
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
