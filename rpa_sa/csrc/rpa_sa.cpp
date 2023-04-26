#ifdef USE_PYTHON
#include <Python.h>
#endif // USE_PYTHON

#include <torch/library.h>
#include "ops/ops.h"

#ifdef WITH_CUDA
#include <cuda.h>
#endif

// If we are in a Windows environment, we need to define
// initialization functions for the _C extension
#ifdef _WIN32
#ifdef USE_PYTHON
PyMODINIT_FUNC PyInit__C(void) {
    // No need to do anything.
    // extension.py will run on load
    return nullptr;
}
#endif // USE_PYTHON
#endif // _WIN32

namespace rpa_sa {
int64_t cuda_version() {
#ifdef WITH_CUDA
    return CUDA_VERSION;
#else
    return -1;
#endif
}

TORCH_LIBRARY_FRAGMENT(rpa_sa, m) {
    m.def("_cuda_version", &cuda_version);
}
} // namespace rpa_sa
