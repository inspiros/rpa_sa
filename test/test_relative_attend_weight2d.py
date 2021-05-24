import os

import time
import torch


def main():
    torch.ops.load_library(os.path.abspath("../rpa/_C.cp38-win_amd64.pyd"))

    batch_sz = 1
    channels = 2
    height, width = (8, 8)
    dtype = torch.float64
    device = 'cuda'
    requires_grad = True

    w = torch.rand(batch_sz, height * width, height * width,
                   dtype=dtype, device=device, requires_grad=requires_grad)
    a = torch.rand(channels, 2 * height - 1, 2 * width - 1,
                   dtype=dtype, device=device, requires_grad=requires_grad)

    print('W', w.shape)
    print('A', a.shape)

    forward_start = time.time()
    out = torch.ops.rpa.relative_attend_weight2d(w, a,
                                                 height=height,
                                                 width=width)
    forward_end = time.time()
    print('out', out.shape)
    print('forward elapsed_time', forward_end - forward_start)

    backward_start = time.time()
    out.backward(torch.ones_like(out))
    backward_end = time.time()
    print(w.grad)
    print(a.grad)
    print('backward elapsed_time', backward_end - backward_start)

    # grad_check
    true_grad = torch.autograd.gradcheck(lambda W, A: torch.ops.rpa.relative_attend_weight2d(W, A, height, width),
                                         inputs=(w, a), nondet_tol=1e-5)
    print('grad_check', true_grad)


if __name__ == '__main__':
    main()
