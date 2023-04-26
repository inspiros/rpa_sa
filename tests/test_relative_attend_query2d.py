import os

import time
import torch


def main():
    torch.ops.load_library(os.path.abspath("../rpa_sa/_C.cp38-win_amd64.pyd"))

    batch_sz = 1
    channels = 8
    height, width = (8, 8)
    dtype = torch.float64
    device = 'cuda'
    requires_grad = True

    q = torch.rand(batch_sz, channels, height * width,
                   dtype=dtype, device=device, requires_grad=requires_grad)
    k = torch.rand(batch_sz, channels, height * width,
                   dtype=dtype, device=device, requires_grad=requires_grad)
    a = torch.rand(channels, 2 * height - 1, 2 * width - 1,
                   dtype=dtype, device=device, requires_grad=requires_grad)
    a.data.fill_(0)

    print('Q', q.shape)
    print('K', k.shape)
    print('A', a.shape)

    forward_start = time.time()
    out = torch.ops.rpa_sa.relative_attend_query2d(q, a,
                                                   height=height,
                                                   width=width)
    forward_end = time.time()
    print('out', out)
    print('forward elapsed_time', forward_end - forward_start)

    backward_start = time.time()
    out.backward(torch.ones_like(out))
    backward_end = time.time()
    print(q.grad)
    print(a.grad)
    print('backward elapsed_time', backward_end - backward_start)

    # grad_check
    true_grad = torch.autograd.gradcheck(lambda Q, A: torch.ops.rpa_sa.relative_attend_query2d(Q, A, height, width),
                                         inputs=(q, a), nondet_tol=1e-5)
    print('grad_check', true_grad)


if __name__ == '__main__':
    main()
