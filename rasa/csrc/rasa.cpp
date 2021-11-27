#ifndef MOBILE
#include <Python.h>
#endif

#include <torch/library.h>
#include "ops/ops.h"

// If we are in a Windows environment, we need to define
// initialization functions for the _C extension
#ifdef _WIN32
PyMODINIT_FUNC PyInit__C(void) {
    // No need to do anything.
    // extension.py will run on load
    return nullptr;
}
#endif
