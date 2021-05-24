import os

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import NoNorm


def main():
    torch.ops.load_library(os.path.abspath("../rpa/_C.cp38-win_amd64.pyd"))

    # print(torch.ops.loaded_libraries)
    # print(torch.ops.rpa.sum(torch.rand(3, 3), torch.rand(3, 3)))
    # print(torch.ops.rpa.rand_op(torch.rand(3, 3), torch.rand(3, 3), "*"))
    # print(torch.ops.rpa.rotative_pad2d(torch.rand(1, 32, 6),
    #                                    5, 5, 5, 5, interpolation='lerp').shape)

    batck_sz = 1
    height, width = (16, 16)
    channels = 32
    k = 5
    device = 'cuda'
    requires_grad = True

    pad_u = pad_d = height - 1
    pad_l = pad_r = width - 1

    x = torch.rand(batck_sz, channels, k + 1,
                   dtype=torch.float64, device=device, requires_grad=requires_grad)
    out = torch.ops.rpa.rotative_pad2d(x,
                                       pad_l, pad_r, pad_u, pad_d,
                                       interpolation='lerp')
    out.backward(torch.ones_like(out))
    print(x.grad)

    true_grad = torch.autograd.gradcheck(lambda _: torch.ops.rpa.rotative_pad2d(x,
                                                                                pad_l, pad_r, pad_u, pad_d,
                                                                                interpolation='lerp'),
                                         inputs=(x,), nondet_tol=1e-5)
    print('grad_check', true_grad)
    exit()

    A_ref = torch.rand(channels, k + 1, dtype=torch.float64, device='cuda', requires_grad=False)
    A_padded = torch.ops.rpa.rotative_pad2d(A_ref.unsqueeze(0),
                                            pad_l, pad_r, pad_u, pad_d,
                                            interpolation='nearest').squeeze()
    print(A_ref)
    print(A_padded)

    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.imshow(A_ref[0].unsqueeze(0), cmap='Spectral', norm=NoNorm())
    plt.title('A_ref')

    plt.subplot(122)
    plt.imshow(A_padded[0].cpu(), cmap='Spectral', norm=NoNorm())
    plt.title('A_padded')
    plt.show()


if __name__ == '__main__':
    main()
