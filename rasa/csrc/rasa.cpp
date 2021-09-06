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

namespace rasa {
    at::Tensor sum(
            const at::Tensor &a,
            const at::Tensor &b) {
        return a + b;
    }

    at::Tensor rand_op(
            const at::Tensor &a,
            const at::Tensor &b,
            at::string &op) {
        auto c = at::randn_like(a);
        auto out = at::empty_like(a);
        if (op == "+") {
            out = a + b + c;
        } else if (op == "*") {
            out = a * b * c;
        } else {
            TORCH_CHECK(0, "unsupported op, expected + or *")
        }
        return out;
    }

    TORCH_LIBRARY_FRAGMENT(rasa, m) {
        m.def("rasa::sum", sum);
        m.def("rasa::rand_op", rasa::rand_op);
    }
}
