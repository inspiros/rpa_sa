import torch
import rpa_nl


def main():
    x = torch.rand(1, 3, 6, 6)
    kernel_size = (4, 4)
