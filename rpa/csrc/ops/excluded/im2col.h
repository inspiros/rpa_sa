#pragma once

#include <ATen/Tensor.h>

at::Tensor im2col_(
        const at::Tensor &input,
        const at::IntArrayRef &stride,
        const at::IntArrayRef &padding,
        const at::IntArrayRef &dilation
);

at::Tensor col2im_(
        const at::Tensor &grad,
        const at::Tensor &input,
        const at::IntArrayRef &stride,
        const at::IntArrayRef &padding,
        const at::IntArrayRef &dilation
);

using namespace at;
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class Im2ColFunction : public torch::autograd::Function<Im2ColFunction> {
public:
    static variable_list forward(
            AutogradContext *ctx,
            Variable input,
            at::IntArrayRef &kernel_size,
            at::IntArrayRef &stride,
            at::IntArrayRef &padding,
            at::IntArrayRef &dilation) {
        auto output = at::im2col(
                input,
                kernel_size,
                dilation,
                padding,
                stride);

        at::IntArrayRef input_size = std::vector<int64_t>({input.size(-2), input.size(-1)});
        ctx->saved_data["input_size"] = input_size;
        ctx->saved_data["kernel_size"] = kernel_size;
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["padding"] = padding;
        ctx->saved_data["dilation"] = dilation;

        return {
                output,
        };
    }

    static variable_list backward(
            AutogradContext *ctx,
            variable_list grad_output) {
        at::IntArrayRef input_size = ctx->saved_data["input_size"].toIntVector();
        at::IntArrayRef kernel_size = ctx->saved_data["kernel_size"].toIntVector();
        at::IntArrayRef stride = ctx->saved_data["stride"].toIntVector();
        at::IntArrayRef padding = ctx->saved_data["padding"].toIntVector();
        at::IntArrayRef dilation = ctx->saved_data["dilation"].toIntVector();

        auto grad_input = at::im2col_backward(
                grad_output[0],
                input_size,
                kernel_size,
                dilation,
                padding,
                stride);

        return {
                grad_input,
                Variable(),
                Variable(),
                Variable(),
                Variable(),
        };
    }
};

namespace rpa {
    at::Tensor im2col(
            const at::Tensor &input,
            at::IntArrayRef &kernel_size,
            at::IntArrayRef &stride,
            at::IntArrayRef &padding,
            at::IntArrayRef &dilation) {
        auto result = Im2ColFunction::apply(
                input,
                kernel_size,
                stride,
                padding,
                dilation);
        return result[0];
    }
}
