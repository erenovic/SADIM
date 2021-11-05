
########################## REFERENCES #############################

# https://github.com/rosinality/swapping-autoencoder-pytorch

###################################################################

import torch
import torch.nn as nn
from torch.nn import functional as F

import math

from stylegan2.op.fused_act import FusedLeakyReLU
from stylegan2.layers import Blur, ModulatedConv2d, NoiseInjection

class ScaledLeakyReLU(nn.Module):
    """Leaky ReLU with scaling and speficied negative slope"""

    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class EqualConv2d(nn.Module):
    """
    from stylegan2\model.py
    Convolutional layer with a scaling factor for weights
    where weights have Gaussian initialization
    """

    def __init__(self, in_channel, out_channel, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel,
                                               kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(input, self.weight * self.scale, bias=self.bias,
                       stride=self.stride, padding=self.padding)
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualConvTranspose2d(nn.Module):
    """
    from model.py
    Transposed Convolutional layer with a scaling factor for weights
    where weights have Gaussian initialization, the symmetric of
    convolutional layer in StyleGAN2 architecture
    """

    def __init__(self, in_channel, out_channel, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(in_channel, out_channel,
                                               kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv_transpose2d(input, self.weight * self.scale, bias=self.bias,
                                 stride=self.stride, padding=self.padding)
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class ConvLayer(nn.Sequential):
    """
    from model.py
    Convolutional Layer which:

    upsamples if upsample=True
    downsamples if downsample=True
    doesn't change dimensions if both are False

    The activation fucntion is FusedLeakyReLU (?) if bias is True,
    otherwise it is a scaled version of LeakyReLU with negative slope of 0.2.
    """

    def __init__(self, in_channel, out_channel, kernel_size, upsample=False,
                 downsample=False, blur_kernel=(1, 3, 3, 1), bias=True,
                 activate=True, padding="zero"):
        layers = []

        self.padding = 0
        stride = 1

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            # To downsample, we have stride = 2
            stride = 2

        if upsample:
            # Upsamples using a Transposed Convolution with stride = 2
            layers.append(EqualConvTranspose2d(in_channel, out_channel,
                                               kernel_size, padding=0, stride=2,
                                               bias=bias and not activate))

            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

        else:
            if not downsample:
                if padding == "zero":
                    self.padding = (kernel_size - 1) // 2

                elif padding == "reflect":
                    padding = (kernel_size - 1) // 2
                    if padding > 0:
                        layers.append(nn.ReflectionPad2d(padding))
                    self.padding = 0

                elif padding != "valid":
                    raise ValueError('Padding should be "zero", "reflect", or "valid"')

            # Stride is specified above, that changes the downsampling behaviour
            # Padding is also specified above
            # If both bias and activate, then we will apply FusedLeakyReLU, in that case
            # we don't use bias (But why ?????????????????????????????????????????????????????????)
            layers.append(EqualConv2d(in_channel, out_channel, kernel_size,
                                      padding=self.padding, stride=stride,
                                      bias=bias and not activate))
            # ????????????????????????????????????????????????????????????????????????????????????

        if activate:
            if bias:

                # ?????????????????????????????????????????????????????????????????????????????
                # I don't understand what FusedLeakyReLU is ??
                layers.append(FusedLeakyReLU(out_channel))
                # ?????????????????????????????????????????????????????????????????????????????

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class StyledConv(nn.Module):
    """
    Styled Convolution Layer which imposes style coming from texture code,
    also changes small details through noise injection from StyleGAN2
    ModulatedConv2d layer adds the style info into the input structure code
    """
    def __init__(self, in_channel, out_channel, kernel_size, style_dim,
                 upsample=False, blur_kernel=[1, 3, 3, 1], demodulate=True):
        super().__init__()

        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size,
                                    style_dim, upsample=upsample,
                                    blur_kernel=blur_kernel,
                                    demodulate=demodulate)

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ResBlock(nn.Module):
    """Residual Block for the Encoder architecture"""
    def __init__(self, in_channel, out_channel, downsample,
                 padding="zero", blur_kernel=(1, 3, 3, 1)):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, out_channel, 3, padding=padding)

        self.conv2 = ConvLayer(out_channel, out_channel, 3, downsample=downsample,
                               padding=padding, blur_kernel=blur_kernel)

        # If dimensions are not matching at the ending addition, we apply a ConvLayer
        # also on the skip connection
        if downsample or in_channel != out_channel:
            self.skip = ConvLayer(in_channel, out_channel, 1, downsample=downsample,
                                  blur_kernel=blur_kernel, bias=False, activate=False)

        else:
            self.skip = None

    def forward(self, input):

        out = self.conv1(input)
        out = self.conv2(out)
        
        if self.skip is not None:
            skip = self.skip(input)

        else:
            skip = input

        # We apply some normalization, not sure why
        return (out + skip) / math.sqrt(2)


class StyledResBlock(nn.Module):
    """
    Residual Block architecture used in Generator network,
    The style is imposed through texture code given through
    StyledConv convolutional units
    """

    def __init__(self, in_channel, out_channel, style_dim,
                 upsample, blur_kernel=(1, 3, 3, 1)):
        super().__init__()

        self.conv1 = StyledConv(in_channel, out_channel, 3, style_dim,
                                upsample=upsample, blur_kernel=blur_kernel)

        self.conv2 = StyledConv(out_channel, out_channel, 3, style_dim)

        if upsample or in_channel != out_channel:
            self.skip = ConvLayer(in_channel, out_channel, 1, upsample=upsample,
                                  blur_kernel=blur_kernel, bias=False, activate=False)
        else:
            self.skip = None

    def forward(self, input, style, noise=None):
        out = self.conv1(input, style, noise)
        out = self.conv2(out, style, noise)

        if self.skip is not None:
            skip = self.skip(input)
        else:
            skip = input

        return (out + skip) / math.sqrt(2)