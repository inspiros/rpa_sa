Relative Position-Aware Self-Attention
======

PyTorch's implementation of the Relative Distance-Aware and Relative Position-Aware Self-Attention presented in our paper
"[Spatial and Temporal Hand-Raising Recognition from Classroom Videos using Locality Relative Position-Aware Non-local Networks and Hand Tracking](https://www.researchgate.net/publication/362963001_Spatial_and_Temporal_Hand_Raising_Recognition_from_Classroom_Videos_using_Locality_Relative_Position_Aware_Non-local_Networks_and_Hand_Tracking)".

## Requirements
- `torch>=1.8`

#### For compiling from source:
- C++ compiler (gcc, MSVC)
- CUDA compiler (nvcc)

## Installation

Clone this repo and run:
```terminal
python setup.py build_ext -i
```

## Usage

Currently, only ``MultiheadNonlocal2d`` is supported.
The module is an extension of ``Nonlocal`` block with some special arguments:
- ``rpa_mode``: Relative position representation mode, can be either:
  - If ``None`` or ``False``, don't use relative position representation.
  - ``'relative_distance'``: Use relative distance representation.
  - ``'relative_position'``: Use relative position representation with polar coordinates.
- ``rpa_kernel_size``: Maximum distance for absolute position of relative distance/position representation.
- ``rpa_interpolation``: Interpolation method.
  - ``relative_distance``: ``nearest`` or ``lerp``.
  - ``relative_position``: ``nearest`` or ``lerp`` or ``slerp``.
- ``rpa_zero_init``: If ``True``, initialize learnable relative distance/position representation's parameters with
  zeros.

### Example:

```python
import torch

from rpa_sa import MultiheadNonlocal2d

model = MultiheadNonlocal2d(in_channels=8,
                            rpa_mode='relative_position',
                            rpa_kernel_size=5,
                            rpa_interpolation='slerp')

x = torch.rand(1, 8, 12, 12)
out = model(x)
print(out)
```

Some simple examples and the code for generating the figures in the paper can be found in [``tests``](tests) folder.

## Known Bugs
The step of computation of reference affinity matrix in both ``relative_distance`` and ``relative_position`` modes has
a severe problem that makes them run **extremely slow** after a number of inital iterations on CUDA
(of course they are much slower on CPU but at constant and expected speed).

I still am unable to trace the causes and am giving up because the CUDA codes gave me too much headache, sorry.
This repo is just meant to be a demonstration of how to implement the idea in the paper.
Don't use it for any serious purpose.

## Citation
```latex
@article{article,
    author = {Le, Thu-Hien and Tran, Hoang-Nhat and Phuong, Dung and Hong Quan, Nguyen and Nguyen, Thuybinh and Tran, Thanh-Hai and Hai, Vu and Tran, Thi-Thao and Le, Thi},
    year = {2022},
    month = {08},
    pages = {},
    title = {Spatial and Temporal Hand Raising Recognition from Classroom Videos using Locality, Relative Position Aware Non-local Networks and Hand Tracking},
    journal = {Vietnam Journal of Computer Science},
    doi = {10.1142/S2196888822500397}
}
```

## License

The code is released under the MIT license. See [`LICENSE.txt`](LICENSE.txt) for details.
