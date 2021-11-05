import torch
import torch.nn as nn

from model.layers import ConvLayer, ResBlock

class Encoder(nn.Module):
    def __init__(self, channel, structure_channel=8, texture_channel=2048,
                 blur_kernel=(1, 3, 3, 1)):
        super().__init__()

        stem = [ConvLayer(3, channel, 1)]

        in_channel = channel
        for i in range(1, 5):
            ch = channel * (2 ** i)
            stem.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch

        self.stem = nn.Sequential(*stem)

        self.structure = nn.Sequential(ConvLayer(ch, ch, 1),
                                       ConvLayer(ch, structure_channel, 1))

        self.texture = nn.Sequential(ConvLayer(ch, ch * 2, 3, downsample=True, padding="valid"),
                                     ConvLayer(ch * 2, ch * 4, 3, downsample=True, padding="valid"),
                                     nn.AdaptiveAvgPool2d(1),
                                     ConvLayer(ch * 4, ch * 4, 1))

    def forward(self, input):
        out = self.stem(input)

        structure = self.structure(out)

        texture = torch.flatten(self.texture(out), 1)
        
        return structure, texture