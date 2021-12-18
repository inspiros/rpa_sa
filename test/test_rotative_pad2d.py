import os

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import NoNorm
from matplotlib.patches import Rectangle


def main():
    torch.ops.load_library(os.path.abspath("../rpa_sa/_C.cp38-win_amd64.pyd"))
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
    out = torch.ops.rpa_sa.rotative_pad2d(x,
                                          (pad_l, pad_r, pad_u, pad_d),
                                          interpolation='lerp')
    print(out.shape)
    out.backward(torch.ones_like(out))
    print(x.grad)

    # grad check
    true_grad = torch.autograd.gradcheck(lambda _: torch.ops.rpa_sa.rotative_pad2d(x,
                                                                                   (pad_l, pad_r, pad_u, pad_d),
                                                                                   interpolation='nearest'),
                                         inputs=(x,), nondet_tol=1e-5)
    print('grad_check', true_grad)

    # visualize
    plt.figure(figsize=(8, 4))

    A_ref = torch.rand(channels, affinity_w, dtype=torch.float64, device=device, requires_grad=False)
    plt.subplot(131)
    plt.gca().imshow(A_ref[0].unsqueeze(0).cpu(), cmap='Spectral', norm=NoNorm())
    plt.gca().set_title('A_ref')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    A_padded = torch.ops.rpa_sa.rotative_pad2d(A_ref.unsqueeze(0),
                                               (pad_l, pad_r, pad_u, pad_d),
                                               interpolation='nearest').squeeze()
    plt.subplot(132)
    plt.gca().imshow(A_padded[0].cpu(), cmap='Spectral', norm=NoNorm())
    plt.gca().set_title('nearest')
    plt.gca().add_patch(
        Rectangle((pad_l - 0.5, pad_u - 0.5),
                  width=A_ref.size(-1), height=1,
                  linewidth=2, edgecolor='k', facecolor='none'))
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    A_padded = torch.ops.rpa_sa.rotative_pad2d(A_ref.unsqueeze(0),
                                               (pad_l, pad_r, pad_u, pad_d),
                                               interpolation='lerp').squeeze()
    plt.subplot(133)
    plt.gca().imshow(A_padded[0].cpu(), cmap='Spectral', norm=NoNorm())
    plt.gca().set_title('lerp')
    plt.gca().add_patch(
        Rectangle((pad_l - 0.5, pad_u - 0.5),
                  width=A_ref.size(-1), height=1,
                  linewidth=2, edgecolor='k', facecolor='none'))
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    # plt.savefig('C:/Users/inspi/Desktop/rotative_pad.png', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
