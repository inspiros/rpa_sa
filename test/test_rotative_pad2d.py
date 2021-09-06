import os

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import NoNorm


def main():
    torch.ops.load_library(os.path.abspath("../rasa/_C.cp38-win_amd64.pyd"))
    print(torch.ops.loaded_libraries)

    batck_sz = 1
    height, width = (10, 10)
    channels = 8
    affinity_w = 6
    device = torch.device('cuda')
    requires_grad = True

    pad_u = pad_d = height - 1
    pad_l = pad_r = width - 1

    # forward backward check
    x = torch.rand(batck_sz, channels, affinity_w,
                   dtype=torch.float64, device=device, requires_grad=requires_grad)
    out = torch.ops.rasa.rotative_pad2d(x,
                                        pad_l, pad_r, pad_u, pad_d,
                                        interpolation='lerp')
    print(out.shape)
    out.backward(torch.ones_like(out))
    print(x.grad)

    # grad check
    true_grad = torch.autograd.gradcheck(lambda _: torch.ops.rasa.rotative_pad2d(x,
                                                                                 pad_l, pad_r, pad_u, pad_d,
                                                                                 interpolation='lerp'),
                                         inputs=(x,), nondet_tol=1e-5)
    print('grad_check', true_grad)

    # visualize
    plt.figure(figsize=(8, 4))

    A_ref = torch.rand(channels, affinity_w, dtype=torch.float64, device=device, requires_grad=False)
    plt.subplot(131)
    plt.imshow(A_ref[0].unsqueeze(0).cpu(), cmap='Spectral', norm=NoNorm())
    plt.title('A_ref')

    A_padded = torch.ops.rasa.rotative_pad2d(A_ref.unsqueeze(0),
                                             pad_l, pad_r, pad_u, pad_d,
                                             interpolation='nearest').squeeze()
    plt.subplot(132)
    plt.imshow(A_padded[0].cpu(), cmap='Spectral', norm=NoNorm())
    plt.title('nearest')

    A_padded = torch.ops.rasa.rotative_pad2d(A_ref.unsqueeze(0),
                                             pad_l, pad_r, pad_u, pad_d,
                                             interpolation='lerp').squeeze()
    plt.subplot(133)
    plt.imshow(A_padded[0].cpu(), cmap='Spectral', norm=NoNorm())
    plt.title('lerp')
    plt.show()


if __name__ == '__main__':
    main()
