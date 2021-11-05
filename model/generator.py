import torch
import torch.nn as nn

from model.layers import ConvLayer, StyledResBlock

class Generator(nn.Module):
    def __init__(self, channel, structure_channel=8, texture_channel=2048,
                 blur_kernel=(1, 3, 3, 1)):
        super().__init__()

        ch_multiplier = (4, 8, 12, 16, 16, 16, 8, 4)
        upsample = (False, False, False, False, True, True, True, True)

        self.layers = nn.ModuleList()
        in_ch = structure_channel
        for ch_mul, up in zip(ch_multiplier, upsample):
            self.layers.append(StyledResBlock(in_ch, channel * ch_mul,
                                              texture_channel, up, blur_kernel))
            in_ch = channel * ch_mul

        self.to_rgb = ConvLayer(in_ch, 3, 1, activate=False)

    def forward(self, structure, texture, noises=None):
        if noises is None:
            noises = [None] * len(self.layers)

        out = structure
        for layer, noise in zip(self.layers, noises):
            out = layer(out, texture, noise)

        out = self.to_rgb(out)

        return out