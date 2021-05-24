#include <ATen/ATen.h>

at::Tensor im2patches(
        const at::Tensor &input,
        const at::IntArrayRef kernel_size,
        const at::IntArrayRef stride,
        const at::IntArrayRef padding,
        const at::IntArrayRef dilation) {
    int64_t batch_sz = input.size(0);
    int64_t channels = input.size(1);
    int64_t height = input.size(2);
    int64_t width = input.size(3);

    int64_t window_h = kernel_size[0];
    int64_t window_w = kernel_size[1];

    int64_t stride_h = stride[0];
    int64_t stride_w = stride[1];

    int64_t pad_h = padding[0];
    int64_t pad_w = padding[1];

    int64_t dilation_h = dilation[0];
    int64_t dilation_w = dilation[1];

    auto options = input.options();

    auto output = at::empty({batch_sz, channels, window_h * window_w}, options);
    return output;
}
