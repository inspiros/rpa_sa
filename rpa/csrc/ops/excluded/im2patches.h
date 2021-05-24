#pragma once

#include <ATen/ATen.h>

at::Tensor im2patches(
        const at::Tensor &input,
        at::IntArrayRef kernel_size,
        at::IntArrayRef stride,
        at::IntArrayRef padding,
        at::IntArrayRef dilation);
