from typing import Tuple, Union, Optional

import itertools
import math
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _ntuple

__all__ = [
    'MultiheadNonlocal1d',
    'MultiheadNonlocal2d',
    'MultiheadNonlocal3d',
]


def _get_conv_module(dimension, transpose=False):
    if dimension == 1:
        return nn.Conv1d if not transpose else nn.ConvTranspose1d
    elif dimension == 2:
        return nn.Conv2d if not transpose else nn.ConvTranspose2d
    elif dimension == 3:
        return nn.Conv3d if not transpose else nn.ConvTranspose3d
    raise ValueError(f'Only supports 1, 2, and 3-D; got dimension={dimension}.')


def _get_rasa_funcs(dimension):
    if dimension == 2:
        return (torch.ops.rasa.relative_attend_query2d,
                torch.ops.rasa.relative_attend_weight2d,
                torch.ops.rasa.rotative_pad2d,
                torch.ops.rasa.spherical_pad2d)
    raise ValueError(f'Only supports 2-D at the moment; got dimension={dimension}.')


def _compute_conv_output_shape(input_shape: Tuple[int, ...],
                               kernel_size: Tuple[int, ...],
                               stride: Tuple[int, ...],
                               padding: Tuple[int, ...],
                               dilation: Tuple[int, ...],
                               ) -> Tuple[int, ...]:
    return tuple((i - 2 * p - (d * (k - 1) + 1)) // s + 1 for i, k, s, p, d in
                 zip(input_shape, kernel_size, stride, padding, dilation))


def _compute_conv_transpose_output_shape(input_shape: Tuple[int, ...],
                                         kernel_size: Tuple[int, ...],
                                         stride: Tuple[int, ...],
                                         padding: Tuple[int, ...],
                                         dilation: Tuple[int, ...],
                                         output_padding: Tuple[int, ...],
                                         ) -> Tuple[int, ...]:
    return tuple((i - 1) * s - 2 * p + d * (k - 1) + 1 + op for i, k, s, p, d, op in
                 zip(input_shape, kernel_size, stride, padding, dilation, output_padding))


def _compute_output_padding(input_shape: Tuple[int, ...],
                            output_shape: Tuple[int, ...],
                            kernel_size: Tuple[int, ...],
                            stride: Tuple[int, ...],
                            padding: Tuple[int, ...],
                            dilation: Tuple[int, ...],
                            ) -> Tuple[int, ...]:
    return tuple(o - (i - 1) * s + 2 * p - d * (k - 1) - 1 for i, o, k, s, p, d in
                 zip(input_shape, output_shape, kernel_size, stride, padding, dilation))


# noinspection PyInitNewSignature
class _MultiheadNonlocalNd(nn.Module):

    def __init__(self,
                 dimension: int,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 num_heads: int = 1,
                 kernel_size: Union[int, Tuple[int, ...]] = 1,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 use_scale: bool = True,
                 residual: bool = True,
                 rasa_mode: Union[str, bool] = None,
                 rasa_kernel_size: Union[int, Tuple[int, ...]] = None,
                 rasa_interpolation: str = 'nearest',
                 rasa_zero_init: bool = True):
        super(_MultiheadNonlocalNd, self).__init__()
        _to_tuple = _ntuple(dimension)
        _embed_conv_module = _get_conv_module(dimension)
        _output_conv_module = _get_conv_module(dimension, transpose=True)
        (self._attend_query,
         self._attend_weight,
         self._relative_distance_pad,
         self._relative_position_pad) = _get_rasa_funcs(dimension)

        if rasa_mode in [None, False]:
            rasa_mode = None
        elif rasa_mode == 'relative_distance':
            rasa_kernel_size = 5 if rasa_kernel_size is None else rasa_kernel_size
            rasa_kernel_size = _ntuple(1)(rasa_kernel_size)
            if len(rasa_kernel_size) > 1:
                raise ValueError('rasa_kernel_size must be 1 dimension for '
                                 f'relative_distance mode; got {rasa_kernel_size}.')
        elif rasa_mode == 'relative_position':
            rasa_kernel_size = 5 if rasa_kernel_size is None else rasa_kernel_size
            rasa_kernel_size = _to_tuple(rasa_kernel_size)
        else:
            raise ValueError(f'rasa_mode must be either relative_distance or '
                             f'relative_position; got {rasa_mode}.')

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else in_channels
        self.num_heads = num_heads
        self.residual = residual
        self.scale = self.hidden_channels ** -0.5 if use_scale else 1

        self.rasa_mode = rasa_mode
        self.rasa_kernel_size = rasa_kernel_size
        if self.rasa_mode == 'relative_distance':
            self._compute_reference_affinity = self._relative_distance_reference_affinity
        elif self.rasa_mode == 'relative_position':
            self._compute_reference_affinity = self._relative_position_reference_affinity
        self.rasa_interpolation = rasa_interpolation
        self.rasa_zero_init = rasa_zero_init

        self.kernel_size = _to_tuple(kernel_size)
        self.stride = _to_tuple(stride)
        self.padding = _to_tuple(padding)
        self.dilation = _to_tuple(dilation)

        self.embed_conv = nn.ModuleList(
            [_embed_conv_module(in_channels,
                                self.hidden_channels * 3,
                                kernel_size=self.kernel_size,
                                stride=self.stride,
                                padding=self.padding,
                                dilation=self.dilation)
             for _ in range(self.num_heads)]
        )
        self.output_conv = _output_conv_module(self.hidden_channels * self.num_heads,
                                               in_channels,
                                               kernel_size=self.kernel_size,
                                               stride=self.stride,
                                               padding=self.padding,
                                               dilation=self.dilation)
        if self.rasa_mode is not None:
            self.a_k = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.zeros(self.hidden_channels, *self.rasa_kernel_size))
                 for _ in range(self.num_heads)]
            )
            self.a_v = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.zeros(self.hidden_channels, *self.rasa_kernel_size))
                 for _ in range(self.num_heads)]
            )
        else:
            self.register_parameter('a_k', None)
            self.register_parameter('a_v', None)

        # init weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, _ConvNd):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if self.rasa_mode is not None:
            if self.rasa_zero_init:
                for head_id in range(self.num_heads):
                    nn.init.constant_(self.a_k[head_id], 0)
                    nn.init.constant_(self.a_v[head_id], 0)
            else:
                for head_id in range(self.num_heads):
                    nn.init.uniform_(self.a_k[head_id])
                    nn.init.uniform_(self.a_v[head_id])

    def _relative_distance_reference_affinity(self, a, input_shape):
        padding = tuple(itertools.chain(*[
            _ for _ in ([i - 1, i - 1] for i in reversed(input_shape))
        ]))
        return self._relative_distance_pad(
            a.unsqueeze(0),
            *padding,
            interpolation=self.rasa_interpolation).squeeze(0)

    def _relative_position_reference_affinity(self, a, input_shape):
        padded_shape = tuple(i - k // 2 - 1 for i, k in zip(input_shape, a.size()[1:]))
        padding = tuple(itertools.chain(*[
            _ for _ in ([math.floor(p), math.ceil(p)] for p in padded_shape)
        ]))
        return self._relative_position_pad(
            a.unsqueeze(0),
            *padding,
            interpolation=self.rasa_interpolation).squeeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_sz = x.size(0)
        in_dims = x.size()[2:]
        hidden_dims = _compute_conv_output_shape(input_shape=in_dims,
                                                 kernel_size=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=self.padding,
                                                 dilation=self.dilation)

        ys = []
        for head_id in range(self.num_heads):
            q, k, v = self.embed_conv[head_id](x).flatten(2).chunk(3, dim=1)

            attn = torch.einsum('bci,bcj->bij', q, k)
            if self.a_k is not None:
                A_k_ref = self._compute_reference_affinity(self.a_k[head_id], hidden_dims)
                attn += self._attend_query(q, A_k_ref, *hidden_dims)
            attn = attn.mul(self.scale).softmax(dim=-1)

            y = torch.einsum('bij,bcj->bci', attn, v)
            if self.a_v is not None:
                A_v_ref = self._compute_reference_affinity(self.a_v[head_id], hidden_dims)
                y += self._attend_weight(attn, A_v_ref, *hidden_dims)
            ys.append(y)
        ys = torch.cat(ys, dim=1).view(batch_sz,
                                       self.hidden_channels * self.num_heads,
                                       *hidden_dims)

        self.output_conv.output_padding = _compute_output_padding(input_shape=hidden_dims,
                                                                  output_shape=in_dims,
                                                                  kernel_size=self.kernel_size,
                                                                  stride=self.stride,
                                                                  padding=self.padding,
                                                                  dilation=self.dilation)
        out = self.output_conv(ys)
        if self.residual:
            out = out + x
        return out


# noinspection PyInitNewSignature
class MultiheadNonlocal1d(_MultiheadNonlocalNd):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 num_heads: int = 1,
                 kernel_size: Union[int, Tuple[int, ...]] = 1,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 use_scale: bool = True,
                 residual: bool = True,
                 rasa_mode: Union[str, bool] = None,
                 rasa_kernel_size: Union[int, Tuple[int, ...]] = None,
                 rasa_interpolation: Optional[str] = 'nearest',
                 rasa_zero_init: bool = True):
        super(MultiheadNonlocal1d, self).__init__(1,
                                                  in_channels,
                                                  hidden_channels,
                                                  num_heads,
                                                  kernel_size,
                                                  stride,
                                                  padding,
                                                  dilation,
                                                  use_scale,
                                                  residual,
                                                  rasa_mode,
                                                  rasa_kernel_size,
                                                  rasa_interpolation,
                                                  rasa_zero_init)


# noinspection PyInitNewSignature
class MultiheadNonlocal2d(_MultiheadNonlocalNd):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 num_heads: int = 1,
                 kernel_size: Union[int, Tuple[int, ...]] = 1,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 use_scale: bool = True,
                 residual: bool = True,
                 rasa_mode: Union[str, bool] = None,
                 rasa_kernel_size: Union[int, Tuple[int, ...]] = None,
                 rasa_interpolation: Optional[str] = 'nearest',
                 rasa_zero_init: bool = True):
        super(MultiheadNonlocal2d, self).__init__(2,
                                                  in_channels,
                                                  hidden_channels,
                                                  num_heads,
                                                  kernel_size,
                                                  stride,
                                                  padding,
                                                  dilation,
                                                  use_scale,
                                                  residual,
                                                  rasa_mode,
                                                  rasa_kernel_size,
                                                  rasa_interpolation,
                                                  rasa_zero_init)


# noinspection PyInitNewSignature
class MultiheadNonlocal3d(_MultiheadNonlocalNd):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 num_heads: int = 1,
                 kernel_size: Union[int, Tuple[int, ...]] = 1,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 use_scale: bool = True,
                 residual: bool = True,
                 rasa_mode: Union[str, bool] = None,
                 rasa_kernel_size: Union[int, Tuple[int, ...]] = None,
                 rasa_interpolation: Optional[str] = 'nearest',
                 rasa_zero_init: bool = True):
        super(MultiheadNonlocal3d, self).__init__(3,
                                                  in_channels,
                                                  hidden_channels,
                                                  num_heads,
                                                  kernel_size,
                                                  stride,
                                                  padding,
                                                  dilation,
                                                  use_scale,
                                                  residual,
                                                  rasa_mode,
                                                  rasa_kernel_size,
                                                  rasa_interpolation,
                                                  rasa_zero_init)
