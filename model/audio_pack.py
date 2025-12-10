#Modified from https://github.com/Omni-Avatar/OmniAvatar/blob/main/OmniAvatar/models/audio_pack.py
import torch
from typing import Tuple, Union
import torch
from einops import rearrange
from torch import nn


def make_triple(value: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    value = (value,) * 3 if isinstance(value, int) else value
    assert len(value) == 3
    return value


class AudioPack(nn.Module):
    def __init__(
            self,
            in_channels: int,
            patch_size: Union[int, Tuple[int, int, int]],
            dim: int,
            layernorm=False,
    ):
        super().__init__()
        t, h, w = make_triple(patch_size)
        self.patch_size = t, h, w
        self.proj = nn.Linear(in_channels * t * h * w, dim)
        if layernorm:
            self.norm_out = nn.LayerNorm(dim)
        else:
            self.norm_out = None

    def forward(
            self,
            vid: torch.Tensor,
    ) -> torch.Tensor:
        t, h, w = self.patch_size
        vid = rearrange(vid, "b c (T t) (H h) (W w) -> b T H W (t h w c)", t=t, h=h, w=w)
        vid = self.proj(vid)     
        if self.norm_out is not None:
            vid = self.norm_out(vid)
        return vid