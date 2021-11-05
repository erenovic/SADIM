
import torch.nn as nn

import math

from model.layers import ConvLayer, ResBlock
from stylegan2.layers import EqualLinear

class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=1, blur_kernel=(1, 3, 3, 1)):
        super().__init__()
        
        # I added divided by 4 to match dimensions !!!!! BE CAREFUL!!!!
        channels = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256 * channel_multiplier,
                    128: 128 * channel_multiplier, 256: 64 * channel_multiplier,
                    512: 32 * channel_multiplier, 1024: 16 * channel_multiplier}

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, downsample=True))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)
        out = self.final_conv(out)

        out = out.view(out.shape[0], -1)
        out = self.final_linear(out)

        return out