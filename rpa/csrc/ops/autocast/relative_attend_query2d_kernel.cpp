#include "../relative_attend_query2d.h"

#include <ATen/autocast_mode.h>
#include <torch/types.h>

namespace rpa {
    namespace ops {
        namespace {
            at::Tensor relative_attend_query2d_autocast(
                    const at::Tensor &query,
                    const at::Tensor &affinity,
                    int64_t height,
                    int64_t width) {
                c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
                return relative_attend_query2d(
                        at::autocast::cached_cast(at::kFloat, query),
                        at::autocast::cached_cast(at::kFloat, affinity),
                        height,
                        width)
                        .to(query.scalar_type());
            }
        } // namespace

        TORCH_LIBRARY_IMPL(rpa, Autocast, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("rpa::relative_attend_query2d"),
                    TORCH_FN(relative_attend_query2d_autocast));
        }
    }
}
