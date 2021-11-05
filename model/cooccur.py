import torch
import torch.nn as nn

from model.layers import ConvLayer, ResBlock
from stylegan2.layers import EqualLinear

class CooccurDiscriminator(nn.Module):
    def __init__(self, channel, size=256):
        super().__init__()

        encoder = [ConvLayer(3, channel, 1)]

        ch_multiplier = (2, 4, 8, 12, 12, 24)
        downsample = (True, True, True, True, True, False)
        in_ch = channel
        for ch_mul, down in zip(ch_multiplier, downsample):
            encoder.append(ResBlock(in_ch, channel * ch_mul, down))
            in_ch = channel * ch_mul

        if size > 511:
            k_size = 3
            feat_size = 2 * 2

        else:
            k_size = 2
            feat_size = 1 * 1

        encoder.append(ConvLayer(in_ch, channel * 12, k_size, padding="valid"))

        self.encoder = nn.Sequential(*encoder)

        self.linear = nn.Sequential(
            EqualLinear(channel * 12 * 2 * feat_size, channel * 32, activation="fused_lrelu"),
            EqualLinear(channel * 32, channel * 32, activation="fused_lrelu"),
            EqualLinear(channel * 32, channel * 16, activation="fused_lrelu"),
            EqualLinear(channel * 16, 1),
        )

    def forward(self, input, reference=None, ref_batch=None, ref_input=None):
        
        out_input = self.encoder(input)
        
        if ref_input is None:
            ref_input = self.encoder(reference)
            _, channel, height, width = ref_input.shape
            ref_input = ref_input.view(-1, ref_batch, channel, height, width)
            ref_input = ref_input.mean(1)

        out = torch.cat((out_input, ref_input), 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)

        return out, ref_input