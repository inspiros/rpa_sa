import os

import math
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import NoNorm
from matplotlib.patches import Rectangle


def main():
    torch.ops.load_library(os.path.abspath("../rpa/_C.cp38-win_amd64.pyd"))
    print(torch.ops.loaded_libraries)

    batck_sz = 1
    height, width = (16, 16)
    channels = 8
    affinity_h = 5
    affinity_w = 5
    device = torch.device('cuda')
    requires_grad = True

    padded_shape = (height - affinity_h // 2 - 1, width - affinity_w // 2 - 1)
    pad_u = math.ceil(padded_shape[0])
    pad_d = math.floor(padded_shape[0])
    pad_l = math.ceil(padded_shape[1])
    pad_r = math.floor(padded_shape[1])
    print(pad_u, pad_d, pad_l, pad_r)

    # forward backward check
    x = torch.rand(batck_sz, channels, affinity_h, affinity_w,
                   dtype=torch.float64, device=device, requires_grad=requires_grad)
    out = torch.ops.rpa.spherical_pad2d(x,
                                        (pad_l, pad_r, pad_u, pad_d),
                                        interpolation='nearest')
    print(out)
    out.backward(torch.ones_like(out))
    print(x.grad)

    # grad check
    true_grad = torch.autograd.gradcheck(lambda _: torch.ops.rpa.spherical_pad2d(x,
                                                                                 (pad_l, pad_r, pad_u, pad_d),
                                                                                 interpolation='slerp'),
                                         inputs=(x,), nondet_tol=1e-5)
    print('grad_check', true_grad)

    # visualize
    fig = plt.figure(figsize=(8, 4))

    plt.subplot(141)
    A_ref = torch.rand(channels, affinity_h, affinity_w, dtype=torch.float64, device=device, requires_grad=False)
    plt.gca().imshow(A_ref[0].cpu(), cmap='Spectral', norm=NoNorm())
    plt.gca().set_title('A_ref')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    plt.subplot(142)
    A_padded = torch.ops.rpa.spherical_pad2d(A_ref.unsqueeze(0),
                                             (pad_l, pad_r, pad_u, pad_d),
                                             interpolation='nearest')[0]
    plt.gca().imshow(A_padded[0].cpu(), cmap='Spectral', norm=NoNorm())
    plt.gca().set_title('nearest')
    plt.gca().add_patch(
        Rectangle((pad_l - 0.5, pad_u - 0.5),
                  width=A_ref.size(-1), height=A_ref.size(-2),
                  linewidth=2, edgecolor='k', facecolor='none'))
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    plt.subplot(143)
    A_padded = torch.ops.rpa.spherical_pad2d(A_ref.unsqueeze(0),
                                             (pad_l, pad_r, pad_u, pad_d),
                                             interpolation='alerp')[0]
    plt.gca().imshow(A_padded[0].cpu(), cmap='Spectral', norm=NoNorm())
    plt.gca().set_title('alerp')
    plt.gca().add_patch(
        Rectangle((pad_l - 0.5, pad_u - 0.5),
                  width=A_ref.size(-1), height=A_ref.size(-2),
                  linewidth=2, edgecolor='k', facecolor='none'))
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    plt.subplot(144)
    A_padded = torch.ops.rpa.spherical_pad2d(A_ref.unsqueeze(0),
                                             (pad_l, pad_r, pad_u, pad_d),
                                             interpolation='slerp')[0]
    plt.gca().imshow(A_padded[0].cpu(), cmap='Spectral', norm=NoNorm())
    plt.gca().set_title('slerp')
    plt.gca().add_patch(
        Rectangle((pad_l - 0.5, pad_u - 0.5),
                  width=A_ref.size(-1), height=A_ref.size(-2),
                  linewidth=2, edgecolor='k', facecolor='none'))
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    # plt.savefig('C:/Users/inspi/Desktop/spherical_pad.png', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
